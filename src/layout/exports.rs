//! The global half of the output symbol table's rows, and the copy slots that
//! back an imported data object.
//!
//! An [`Export`] is one defined global the writer emits; a [`CopySlot`] is the
//! `.bss` storage an executable owns on a dependency's behalf, named by every
//! [`CopyName`] the dependency gives that object. Collecting the exports is
//! the last thing layout does, once every address is known.
//!
//! The locals that precede them in `.symtab` are not collected here: a local
//! is file-scoped, so nothing interns it and layout never resolves it. The
//! writer reads those rows straight out of each input's own symbol table (see
//! `crate::writer::symtab`).

use rustc_hash::FxHashMap;

use super::{Layout, Sect, addr::definition_addr};
use crate::{
    elf::constants::{
        SHN_ABS, SHN_UNDEF, STT_FUNC, STT_GNU_IFUNC, STT_NOTYPE, STT_OBJECT,
        STT_TLS,
    },
    linker::{Context, DepAddr},
    symbol::{DefSource, Definition, Symbol, SymbolId, SymbolKind},
    tls::TlsBlock,
};

/// A symbol selected for the output symbol table.
///
/// The name is not copied here: it stays interned in
/// [`crate::symbol::SymbolTable`], which outlives the layout, and the writer
/// resolves it through [`Self::id`] when it builds the string table.
#[derive(Clone, Copy, Debug)]
pub struct Export {
    /// The resolved symbol this export names.
    pub id: SymbolId,
    pub addr: u64,
    pub size: u64,
    pub shndx: u16,
    pub sym_type: u8,
    /// The merged visibility (`STV_*`) written back out as `st_other`. It also
    /// decides whether the export reaches `.dynsym`: see
    /// [`crate::symbol::exports_to_dynsym`].
    pub visibility: u8,
    pub weak: bool,
    /// The winning definition was `STB_GNU_UNIQUE`; the binding is written
    /// back out, since the loader keys its one-copy search on it.
    pub unique: bool,
}

/// A data symbol imported from a shared dependency that the executable copies
/// into its own `.bss`.
///
/// Emitted via a single `R_X86_64_COPY` relocation naming [`Self::id`].
/// `align` is the placement alignment; `addr` is filled once `.bss` is placed.
///
/// A dependency may spell one object several ways (glibc exports `environ`,
/// `__environ` and `_environ` for the same eight bytes). Every such name is in
/// [`Self::names`] and every one of them is defined at this one slot: giving a
/// name its own slot would leave the program with two copies of an object the
/// dependency has one of, so a write through one name would be invisible
/// through the other and the dependency's own writes would reach neither.
#[derive(Clone, Debug)]
pub struct CopySlot {
    /// The symbol whose reference selected the slot; the `R_X86_64_COPY`
    /// names it.
    pub id: SymbolId,
    pub size: u64,
    pub align: u64,
    pub addr: u64,
    /// Which object of which dependency the slot copies. Two references that
    /// reach this object share the slot, whichever of its names they used.
    pub origin: DepAddr,
    /// Every name the dependency gives [`Self::origin`], in its symbol table
    /// order, including the one that selected the slot. Each becomes exactly
    /// one defined `.dynsym` entry at [`Self::addr`], so the loader binds
    /// references from other images to the copy too.
    pub names: Vec<CopyName>,
}

/// One name a [`CopySlot`] defines.
#[derive(Clone, Debug)]
pub struct CopyName {
    /// The name as the dependency spells it.
    pub name: Vec<u8>,
    /// The `st_size` the dependency gives this name; the `.dynsym` entry
    /// carries it verbatim, as lld does.
    pub size: u64,
    /// The resolved symbol, when an input references the name. `None` for an
    /// alias nothing referenced: it is still defined at the slot, for other
    /// images to bind to, but resolves nothing in this link.
    pub id: Option<SymbolId>,
}

/// Collects defined globals for the output symbol table. A TLS symbol's
/// `st_value` is its offset within the TLS block (matching lld's
/// `sym.getVA` for TLS), not its TPOFF or vaddr.
///
/// `defsym_shndx` carries the section each linker-defined bound is anchored
/// to; see [`crate::defsym`] for why those cannot be published `SHN_ABS`.
pub(super) fn collect_exports(
    ctx: &Context<'_>,
    block: Option<TlsBlock>,
    global_addr: &[u64],
    defsym_shndx: &FxHashMap<SymbolId, u16>,
    layout: &mut Layout,
) {
    use rayon::prelude::*;
    // Each row is a function of one symbol and the finished layout, so the
    // rows are derived in parallel and appended in id order -- the same list
    // the serial walk built, without half a million serial map probes on the
    // layout's critical path.
    let ids: Vec<SymbolId> = ctx.symbols.ids().collect();
    let rows: Vec<Option<Export>> = ids
        .par_iter()
        .map(|&id| {
            export_for(ctx, block, global_addr, defsym_shndx, layout, id)
        })
        .collect();
    // At most one export per resolved symbol; sizing once avoids regrowing a
    // list that reaches tens of megabytes on a large link.
    layout.exports.reserve(ctx.symbols.len());
    layout.exports.extend(rows.into_iter().flatten());
}

/// The output symbol-table entry for `id`, or `None` if it contributes none.
fn export_for(
    ctx: &Context<'_>,
    block: Option<TlsBlock>,
    global_addr: &[u64],
    defsym_shndx: &FxHashMap<SymbolId, u16>,
    layout: &Layout,
    id: SymbolId,
) -> Option<Export> {
    let sym = ctx.symbols.symbol(id)?;
    let SymbolKind::Defined(def) = &sym.kind else {
        // A common is a definition the linker allocated: it is placed in
        // `.bss` and given an address, and publishing nothing for it meant
        // `nm` and every debugger showed no symbol at all for a `-fcommon`
        // global. lld emits a row at the allocated address.
        if let SymbolKind::Undefined { .. } = sym.kind {
            // A reference this link has no definition for still gets its row,
            // `SHN_UNDEF` with a zero value: that is how `nm` prints the `U`
            // entries and how a debugger lists what the image imports. lld
            // writes the same row, `STT_NOTYPE` with the reference's binding,
            // for every symbol a surviving relocation names -- a reference
            // each lowering consumed (the relaxed `__tls_get_addr` call)
            // reaches nothing and earns no row.
            return undefined_export(id, sym, layout);
        }
        return common_export(global_addr, layout, id, sym);
    };
    // A definition whose backing input section was garbage-collected has no
    // address to publish: skip it, matching lld's removal of GC'd symbols from
    // the symtab. Guarded on `ctx.gc` so the default link is byte-identical to
    // the pre-GC behaviour (every allocated section is placed, so the check
    // would otherwise be a no-op anyway, but the guard makes that explicit).
    if ctx.gc
        && let DefSource::Section { index, .. } = def.source
        && layout.section_vaddr(def.file, index).is_none()
    {
        return None;
    }
    // For a TLS symbol, emit the offset within the TLS block; otherwise emit
    // the resolved runtime address.
    let addr = if def.sym_type == STT_TLS {
        let base = block.map_or(0, |b| b.vaddr);
        definition_addr(&ctx.merge, layout, def).wrapping_sub(base)
    } else {
        global_addr.get(id.0).copied().unwrap_or_default()
    };
    let mut export = Export {
        id,
        addr,
        size: def.size,
        shndx: export_shndx(layout, defsym_shndx, id, def),
        sym_type: def.sym_type,
        visibility: sym.visibility,
        weak: def.weak,
        unique: def.unique,
    };
    canonicalize_ifunc(layout, &mut export);
    Some(export)
}

/// The `.symtab` row for a common symbol, or `None` if `id` is not one.
///
/// The address is the one address resolution already computed for it, which
/// is `.bss` plus its offset within the commons block.
fn common_export(
    global_addr: &[u64],
    layout: &Layout,
    id: SymbolId,
    sym: &Symbol,
) -> Option<Export> {
    let SymbolKind::Common { weak, size, .. } = sym.kind else {
        return None;
    };
    Some(Export {
        id,
        addr: global_addr.get(id.0).copied().unwrap_or_default(),
        size,
        shndx: layout.shndx(Sect::Bss),
        sym_type: STT_OBJECT,
        visibility: sym.visibility,
        weak,
        unique: false,
    })
}

/// Publishes an unresolved reference as an `SHN_UNDEF` row, when a surviving
/// relocation still names it.
fn undefined_export(
    id: SymbolId,
    sym: &Symbol,
    layout: &Layout,
) -> Option<Export> {
    let SymbolKind::Undefined { weak } = sym.kind else {
        return None;
    };
    if !layout.referenced.contains(&id) {
        return None;
    }
    Some(Export {
        id,
        addr: 0,
        size: 0,
        shndx: SHN_UNDEF,
        sym_type: STT_NOTYPE,
        visibility: sym.visibility,
        weak,
        unique: false,
    })
}

/// Publishes an indirect function at its PLT stub, as a plain function.
///
/// Every reference in the image already resolves to the stub: the address the
/// symbol itself names is a resolver, which nothing may reach. The symbol
/// tables carry that same address, in the section the stub lives in, so a
/// pointer taken through the tables compares equal to one taken through a
/// reference.
///
/// The type has to change with it. A loader binding another image to an
/// `STT_GNU_IFUNC` definition calls the definition's value and takes what
/// comes back for the function's address; against a stub that would be
/// whatever the implementation returned when called with no arguments. lld
/// canonicalizes an indirect function the same way, and for the same reason.
fn canonicalize_ifunc(layout: &Layout, export: &mut Export) {
    if export.sym_type != STT_GNU_IFUNC {
        return;
    }
    let Some(stub) = layout.plt_entry_addr(export.id) else {
        return;
    };
    export.addr = stub;
    export.shndx = layout.shndx(Sect::Plt);
    export.sym_type = STT_FUNC;
    // The stub is not the resolver's bytes, so the resolver's size does not
    // describe it.
    export.size = 0;
}

/// The output section of an exported definition.
///
/// A bound the linker defined itself is recorded as absolute -- the layout owns
/// its value, not any input section -- yet it names a place in the image, and
/// publishing it `SHN_ABS` would tell a loader not to add the load base to it.
/// `defsym_shndx` supplies the section each one is anchored to; see
/// [`crate::defsym`]. Only a genuine `SHN_ABS` from an input, whose value
/// really is a constant, is published as one.
fn export_shndx(
    layout: &Layout,
    defsym_shndx: &FxHashMap<SymbolId, u16>,
    id: SymbolId,
    def: &Definition,
) -> u16 {
    match def.source {
        DefSource::Section { index, .. } => layout
            .sec_shndx
            .get(def.file)
            .and_then(|f| f.get(usize::from(index)).copied())
            .unwrap_or_default(),
        DefSource::Absolute { .. } => {
            defsym_shndx.get(&id).copied().unwrap_or(SHN_ABS)
        }
    }
}
