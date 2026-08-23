//! `.rela.dyn` construction: the dynamic relocations a position-independent
//! image needs, plus the sizing scan that mirrors it.
//!
//! The loader relocates three kinds of slots: slots whose target address is
//! known at link time (covered by `R_*_RELATIVE`), slots naming a symbol only
//! the loader can resolve (covered by `R_*_GLOB_DAT` in the GOT and by the
//! target's absolute reloc elsewhere), and GOT slots holding the
//! thread-pointer offset of an imported thread-local (covered by
//! `R_*_TPOFF64`). Absolute data references (`R_*_64`) in
//! allocated sections add one entry each, and copy-relocated data symbols add a
//! trailing `R_*_COPY`. [`build_rela_dyn`] emits the table with every
//! `RELATIVE` first so the loader can short-circuit that prefix via
//! `DT_RELACOUNT`; [`count_data_relocs`] is the sizing-only mirror used by the
//! probe pass.
//!
//! A GOT slot and a pointer slot in a data section pose the loader the same
//! question, so [`classify_slot`] answers it once for both, and the sizing
//! pass reads the same answer. The absolute data references are what the two
//! passes must also agree on in *which sections they read*, so both drive
//! [`for_each_data_reloc_section`].
//!
//! The two halves are not symmetric across link flavours: an image with a
//! fixed base emits the symbol-based half and not the `RELATIVE` one, whose
//! load base there is zero. That asymmetry is stated once, in
//! [`relative_applies`], and every pass that can produce a `RELATIVE` reads it
//! from there -- the GOT emitter directly, the absolute-data emitter through
//! [`DataRelocScope::emitted`], and both sizing walks alongside them.

use rayon::prelude::*;
use rustc_hash::FxHashSet;

use super::{LinkMode, copy_id_set, dynsym::DynSym, is_runtime_import};
use crate::{
    elf::{
        Rela64, Sym64, SymbolTable,
        constants::{SHF_ALLOC, SHF_WRITE, SHT_NOBITS, STB_LOCAL},
    },
    endian::{I64, U64},
    error::{Error, Result},
    input::InputFile,
    layout::{FileResolver, GotKind, Layout, Sect},
    linker::{Context, in_input_order},
    reloc::{DynRelocs, RelExpr, Target},
    symbol::{DefSource, Definition, SymbolId, SymbolKind, version_stem},
    tls::{TlsBlock, block_offset},
    util::show,
};

/// Bytes in one GOT entry.
const GOT_ENTRY: u64 = 8;

/// Builds `.rela.dyn`. Every `R_*_RELATIVE` comes first (the loader
/// optimises that prefix via `DT_RELACOUNT`), followed by the symbol-based
/// entries (`R_*_GLOB_DAT` for GOT slots of globals, the target's absolute
/// reloc for absolute data references to globals, `R_*_COPY` for
/// copy-relocated data symbols). `r_offset` is the slot's image offset, which
/// equals its vaddr since the shared object's load base is zero.
pub(super) fn build_rela_dyn(
    ctx: &Context<'_>,
    layout: &Layout,
    target: Target,
    names: &[&[u8]],
    table: &DynSym,
) -> Result<Vec<Rela64>> {
    let relocs = target.dyn_relocs();
    let mut relative = Vec::new();
    let mut sym_based = Vec::new();
    let copy_ids = copy_id_set(layout);
    let fixed = FixedImports::new(layout, &copy_ids);
    let mut out = RelaSink {
        relative: &mut relative,
        sym_based: &mut sym_based,
    };
    push_got_entries(ctx, layout, relocs, names, table, fixed, &mut out);
    // Absolute data references (the target's `abs64` type) in allocated
    // sections: a target whose address this link fixes becomes RELATIVE, and
    // one the loader must resolve by name becomes the target's absolute reloc
    // against its dynsym entry. See [`classify_data_reloc`], which also drops
    // the half a fixed-base image has no use for. PC-relative and GOT
    // relocations are resolved at link time and produce no dynamic entry.
    let dr = DataRelocCtx {
        scope: DataRelocScope::new(ctx, layout, fixed),
        layout,
        table,
        abs64: relocs.abs64,
        relative_type: relocs.relative,
    };
    scan_data_relocs(dr, &mut out)?;
    // Copy relocations come last: they reference the import by dynsym index
    // and tell the loader to copy the dependency's initial bytes into the
    // `.bss` slot. They are not `RELATIVE`, so they sit outside the prefix
    // `DT_RELACOUNT` counts.
    for slot in &layout.copy_slots {
        let Some(name) = names.get(slot.id.0) else {
            continue;
        };
        let Some(sym_idx) = table.index_of(name) else {
            continue;
        };
        out.sym_based
            .push(make_rela(slot.addr, sym_idx, relocs.copy, 0));
    }
    relative.append(&mut sym_based);
    Ok(relative)
}

/// Appends the entries every GOT slot needs.
///
/// An address slot is classified by [`classify_slot`], exactly as an absolute
/// data reference is: the loader fills a preemptible symbol's slot by name
/// (`GLOB_DAT`), this link fixes an address the loader shifts by the load base
/// (`RELATIVE`), and a slot already holding a link-time constant needs nothing.
/// A file-local slot always holds an address. The TLS kinds are covered by
/// [`tls_entry`]; a shared object's own thread-pointer offsets are chosen by
/// the loader, so its slots need a relocation whether or not the thread-local
/// comes from elsewhere.
///
/// Both `RELATIVE` verdicts are filtered through [`relative_applies`], which
/// is where the answer for an image with no load base is stated;
/// [`super::sizes::count_got_plt_imports`] mirrors this walk and reads the
/// same predicate, so the reserved region and the emitted table agree.
///
/// Slots are counted rather than indexed by position, because a
/// general-dynamic entry takes two.
fn push_got_entries(
    ctx: &Context<'_>,
    layout: &Layout,
    relocs: DynRelocs,
    names: &[&[u8]],
    table: &DynSym,
    fixed: FixedImports<'_>,
    out: &mut RelaSink<'_>,
) {
    let shared = layout.mode() == LinkMode::Shared;
    let tls = TlsRelocs {
        relocs,
        loader_owns_tp: shared,
        ctx,
        target: layout.target,
        block: layout.tls_block(),
    };
    let mut index = 0u64;
    for key in &layout.got_keys {
        let slot = layout
            .region(Sect::Got)
            .vaddr
            .wrapping_add(index * GOT_ENTRY);
        let addend = layout
            .got_values
            .get(usize::try_from(index).unwrap_or(0))
            .copied()
            .unwrap_or_default();
        index = index.saturating_add(key.kind.slots());
        let global = key.global_id();
        let dyn_idx = global
            .and_then(|id| names.get(id.0))
            .and_then(|name| table.index_of(name));
        if key.kind != GotKind::Addr {
            out.sym_based.extend(tls_entry(
                &tls, key.kind, global, dyn_idx, slot, addend,
            ));
            continue;
        }
        // A file-local slot holds an address this link fixed; anything else is
        // the classifier's to answer.
        let kind = global
            .map_or(Some(SlotReloc::Relative), |id| {
                classify_slot(ctx, shared, fixed, id)
            })
            .filter(|k| {
                !matches!(k, SlotReloc::Relative)
                    || relative_applies(layout.pie)
            });
        match kind {
            // A target with no `.dynsym` entry has no name to look up; see
            // [`emit_by_name`] for when that is expected.
            Some(SlotReloc::ByName(_)) => {
                if let Some(sym_idx) = dyn_idx {
                    out.sym_based.push(make_rela(
                        slot,
                        sym_idx,
                        relocs.glob_dat,
                        0,
                    ));
                }
            }
            Some(SlotReloc::Relative) => {
                out.relative
                    .push(make_rela(slot, 0, relocs.relative, addend));
            }
            None => {}
        }
    }
}

/// What the TLS entries of one image need from the loader.
struct TlsRelocs<'a, 'b> {
    relocs: DynRelocs,
    /// Whether the loader, rather than this link, fixes the offset from the
    /// thread pointer -- which is the case for every shared object.
    loader_owns_tp: bool,
    ctx: &'a Context<'b>,
    /// The target and the static TLS block, which together turn a filled
    /// thread-pointer offset back into an offset within the block. That is
    /// the addend a symbol-less `TPOFF64` carries; see [`tls_entry`].
    target: Target,
    block: Option<TlsBlock>,
}

impl TlsRelocs<'_, '_> {
    /// The offset within this image's TLS block of a thread-local whose GOT
    /// slot was filled with the thread-pointer offset `filled`.
    ///
    /// `None` when the image has no TLS block, in which case there is no
    /// thread-local of this image's to describe.
    fn block_offset(&self, filled: u64) -> Option<u64> {
        self.block
            .as_ref()
            .map(|b| block_offset(self.target, filled, b))
    }
}

/// The dynamic relocations one TLS GOT entry needs.
///
/// A `TlsOffset` slot needs the loader only when a shared object owns the
/// thread-local: this image fixes the offset for one it places itself, and
/// that value is already in the slot. A `TlsModule` pair always needs its
/// module id filled in, since only the loader knows which module a block
/// belongs to; the offset beside it is left to the loader too when the
/// thread-local can be preempted, and is otherwise already in the slot.
///
/// A shared object's slot is the loader's either way, and it has two
/// spellings. One naming a thread-local the object exports carries the symbol
/// index, and the loader looks the offset up. One naming a thread-local the
/// object keeps *hidden* has no `.dynsym` row to point at -- and needs none,
/// since a name no other image can bind cannot be preempted -- so it carries
/// `r_sym` 0 and states the offset within this image's block as its addend.
/// Without that second form the slot keeps the executable-convention constant
/// `fill_got` put there, which is measured from a thread pointer only an
/// executable has, and the program reads the wrong thread-local memory.
/// `filled` is that constant, and converting it back is how the block offset
/// is recovered without a second address lookup. lld reaches the same shape
/// through `addAddendOnlyRelocIfNonPreemptible`.
fn tls_entry(
    tls: &TlsRelocs<'_, '_>,
    kind: GotKind,
    global: Option<SymbolId>,
    dyn_idx: Option<u32>,
    slot: u64,
    filled: u64,
) -> Vec<Rela64> {
    let relocs = tls.relocs;
    let mut out = Vec::new();
    match kind {
        GotKind::Addr => {}
        GotKind::TlsOffset => {
            let wanted = global.is_some_and(|id| {
                tls.loader_owns_tp || is_runtime_import(tls.ctx, id)
            });
            match (wanted, dyn_idx) {
                (true, Some(sym_idx)) => {
                    out.push(make_rela(slot, sym_idx, relocs.tpoff, 0));
                }
                (true, None) => {
                    // Nameless, so this image places the thread-local itself
                    // and only the block offset is left to say.
                    if let Some(off) = tls.block_offset(filled) {
                        out.push(make_rela(slot, 0, relocs.tpoff, off));
                    }
                }
                (false, _) => {}
            }
        }
        // A pair naming no symbol is this image's own module entry: the
        // loader supplies the id, and the offset beside it stays zero because
        // each access adds its own.
        GotKind::TlsModule => {
            out.push(make_rela(slot, dyn_idx.unwrap_or(0), relocs.dtpmod, 0));
            if let Some(sym_idx) = dyn_idx {
                out.push(make_rela(
                    slot.wrapping_add(GOT_ENTRY),
                    sym_idx,
                    relocs.dtpoff,
                    0,
                ));
            }
        }
    }
    out
}

/// One input section that can contribute dynamic relocations, as handed to
/// both passes by [`for_each_data_reloc_section`].
struct DataRelocSection<'a> {
    /// The input file the section belongs to.
    file: usize,
    /// The section's index within that file.
    section: u16,
    /// Whether the section carries `SHF_WRITE`, and so whether the loader can
    /// store into a slot inside it at all; see [`classify_data_reloc`].
    writable: bool,
    /// The file's symbol table, which each relocation's symbol index selects
    /// from.
    symtab: &'a SymbolTable<'a>,
    /// The section's relocation entries, of every type.
    entries: &'a [Rela64],
}

impl DataRelocSection<'_> {
    /// The pointer-width absolute relocations
    /// (`R_X86_64_64`/`R_AARCH64_ABS64`/`R_RISCV_64`), the only kind that
    /// becomes a dynamic relocation.
    fn absolute(&self, abs64: u32) -> impl Iterator<Item = &Rela64> {
        self.entries.iter().filter(move |r| r.r_type() == abs64)
    }

    /// The input symbol `r` names, or `None` when its index is out of range.
    fn symbol(&self, r: &Rela64) -> Option<&Sym64> {
        self.symtab.syms.get(r.sym() as usize)
    }
}

/// Visits the sections of one input that contribute dynamic relocations -- an
/// allocated, file-backed section carrying an `SHT_RELA` table -- in section
/// order. [`map_data_reloc_files`] drives this across the inputs.
///
/// Both [`survey_data_relocs`] and [`scan_data_relocs`] drive this walk, so
/// the section set `.rela.dyn` is sized from, the set its entries are emitted
/// from, and the set the `.dynsym` rows are reserved for cannot drift apart.
/// Two kinds of section are excluded on purpose:
///
/// - One without `SHF_ALLOC` holds no runtime address, so a relocation in it --
///   the `R_X86_64_64` entries of `.rela.debug_addr` in a `-g` build, or the
///   `.annobin.notes` a hardened build carries -- must produce no dynamic
///   relocation at all.
/// - One folded onto another by `--icf` or by the merge pass contributes no
///   bytes: the representative carries the surviving copy and its own
///   relocations describe it. Layout stamps the folded section with the
///   representative's address, so it is [`crate::icf::FoldMap::is_folded`]
///   rather than an address lookup that tells them apart -- without this, every
///   folded copy added a duplicate entry aimed at the representative's slot.
///
///   That duplication is no longer reachable through `--icf`, which folds only
///   executable sections without `SHF_WRITE`: a dynamic relocation in one of
///   those is now refused outright by [`classify_data_reloc`], and the refusal
///   fires on the representative, which carries the same relocations that made
///   the copy foldable. The exclusion is kept as the walk's own invariant --
///   an input section that contributes no bytes contributes no relocations
///   either -- and stands ready for a fold source where a slot the loader can
///   write is duplicated.
fn for_each_data_reloc_section_of(
    ctx: &Context<'_>,
    file: usize,
    input: &InputFile<'_>,
    candidates: Option<&[u16]>,
    mut visit: impl FnMut(&DataRelocSection<'_>) -> Result<()>,
) -> Result<()> {
    let obj = input.object()?;
    let Some(symtab) = input.symbol_table()? else {
        return Ok(());
    };
    // Either every section of the file, or the candidate list the layout
    // scan collected -- the sections with at least one pointer-width
    // absolute relocation, in the same ascending order this walk visits
    // them in. A candidate still passes every filter below, so the two
    // spellings answer identically; the list just skips the sections that
    // could not contribute a row.
    // `map_or_else` cannot spell this: the fallback borrows a scratch list
    // that has to outlive the closure.
    let mut all = Vec::new();
    if candidates.is_none() {
        all.extend(
            (0..obj.sections().len()).filter_map(|i| u16::try_from(i).ok()),
        );
    }
    let sections: &[u16] = candidates.unwrap_or(&all);
    for &section in sections {
        let Some(shdr) = obj.sections().get(usize::from(section)) else {
            continue;
        };
        let flags = shdr.sh_flags.get();
        if flags & SHF_ALLOC == 0 || shdr.sh_type.get() == SHT_NOBITS {
            continue;
        }
        if ctx.folding.is_folded((file, section)) {
            continue;
        }
        let Some(entries) = input.relocations(section)? else {
            continue;
        };
        visit(&DataRelocSection {
            file,
            section,
            writable: flags & SHF_WRITE != 0,
            symtab: &symtab,
            entries,
        })?;
    }
    Ok(())
}

/// Runs the walk over every input at once, one accumulator per file, and
/// returns them in input order.
///
/// A file's sections are read-only here and nothing one file yields depends on
/// another, so the walk itself parallelises; what must not vary with the
/// schedule is the result. Keeping one accumulator per file and returning the
/// row vector in file-index order is what fixes that: every consumer either
/// sums the rows (an order-free reduction) or concatenates them in the order
/// the inputs were named, which is the order the serial walk produced.
///
/// The first refusal in input order is the one reported, for the reason
/// [`crate::linker::in_input_order`] gives: a diagnostic that changes between
/// identical runs is the same defect as an image that does.
fn map_data_reloc_files<T>(
    ctx: &Context<'_>,
    candidates: Option<&[Vec<u16>]>,
    visit: impl Fn(&DataRelocSection<'_>, &mut T) -> Result<()> + Sync,
) -> Result<Vec<T>>
where
    T: Default + Send,
{
    let mut rows: Vec<Result<T>> = Vec::new();
    ctx.files
        .par_iter()
        .enumerate()
        .map(|(file, input)| {
            let mut acc = T::default();
            let list = candidates.and_then(|c| c.get(file)).map(Vec::as_slice);
            for_each_data_reloc_section_of(ctx, file, input, list, |sec| {
                visit(sec, &mut acc)
            })?;
            Ok(acc)
        })
        .collect_into_vec(&mut rows);
    in_input_order(rows)
}

/// Which dynamic relocation a slot holding a symbol's address needs.
///
/// A GOT slot and a pointer slot in an ordinary data section pose the loader
/// the same question, so both are classified through this and both read the
/// answer off [`classify_global`].
#[derive(Clone, Copy)]
pub(super) enum SlotReloc {
    /// This link fixes the target address, so the loader only has to add the
    /// load base: an `R_*_RELATIVE` carrying the resolved address as its
    /// addend.
    Relative,
    /// The loader resolves the target by name, so the entry names it in
    /// `.dynsym`.
    ByName(SymbolId),
}

/// Whether an `R_*_RELATIVE` entry does anything in an image whose
/// position-independence is `pie`.
///
/// The entry asks the loader to add the load base to an address this link
/// already resolved. An image with a fixed base (`ET_EXEC`) has no load base
/// to add: the loader maps it at the addresses it names, so the sum is the
/// value the writer already stored and the entry is a no-op -- one that still
/// costs a `.rela.dyn` row and a `DT_RELACOUNT` the loader walks at startup.
/// lld emits nothing for the same slot: `addGotEntry` stores a constant rather
/// than adding a relative relocation once `!ctx.arg.isPic`.
///
/// Three walks ask this, and all three must agree or the region reserved for
/// `.rela.dyn` will not match what is written into it: the GOT emitter
/// ([`push_got_entries`]), the absolute-data emitter
/// ([`DataRelocScope::emitted`]), and the sizing pass that mirrors both
/// ([`super::sizes::count_got_plt_imports`] and
/// [`count_data_relocs`]).
pub(super) const fn relative_applies(pie: bool) -> bool {
    pie
}

/// The imports whose address this link fixes rather than leaving to the
/// loader.
///
/// Two mechanisms put an import at an address chosen here, and both stop the
/// name from describing where the storage lives: a `.bss` copy slot the loader
/// fills from the dependency, and a canonical PLT stub that stands in for a
/// shared function. A slot holding either takes `RELATIVE` and never a
/// symbol-based entry: the loader would otherwise write the dependency's
/// address over the one this link chose, which for a canonical PLT leaves two
/// values for one function pointer in one image.
#[derive(Clone, Copy)]
pub(super) struct FixedImports<'a> {
    /// The symbols a `.bss` copy slot defines, aliases included.
    copy: &'a FxHashSet<SymbolId>,
    /// The symbols a canonical PLT stub stands in for.
    canonical: &'a FxHashSet<SymbolId>,
}

impl<'a> FixedImports<'a> {
    /// The imports `layout` fixes the address of, over a copy-slot id set the
    /// caller already built.
    pub(super) fn new(
        layout: &'a Layout,
        copy: &'a FxHashSet<SymbolId>,
    ) -> Self {
        Self {
            copy,
            canonical: &layout.canonical_plt,
        }
    }

    /// Whether this link fixes `id`'s address.
    fn contains(self, id: SymbolId) -> bool {
        self.copy.contains(&id) || self.canonical.contains(&id)
    }

    /// Whether a `.bss` copy slot defines `id`.
    ///
    /// A copy slot supplies its own *defined* `.dynsym` row, so the passes
    /// that hand out import rows skip these; a canonical PLT symbol keeps its
    /// undefined row and is not skipped with them.
    pub(super) fn is_copy(self, id: SymbolId) -> bool {
        self.copy.contains(&id)
    }
}

/// What deciding a [`SlotReloc`] depends on, which is everything the sizing
/// pass can already see: the resolved symbols, the imports this link fixes and
/// the link flavour. No address is involved, so the probe pass classifies a
/// reference exactly as the emitter will once placement is known.
#[derive(Clone, Copy)]
struct DataRelocScope<'a, 'data> {
    ctx: &'a Context<'data>,
    /// The imports whose address this link fixes.
    fixed: FixedImports<'a>,
    /// Whether the image is a shared object, whose own definitions the
    /// executable that loads it may interpose.
    shared: bool,
    /// Whether the loader picks the load base (`ET_DYN`), which is what gives
    /// the `RELATIVE` half of the scan anything to do; see
    /// [`relative_applies`].
    pie: bool,
}

impl<'a, 'data> DataRelocScope<'a, 'data> {
    /// The scope `layout` classifies a data reference in.
    fn new(
        ctx: &'a Context<'data>,
        layout: &'a Layout,
        fixed: FixedImports<'a>,
    ) -> Self {
        Self {
            ctx,
            fixed,
            shared: layout.mode() == LinkMode::Shared,
            pie: layout.pie,
        }
    }

    /// What verdict `kind` reduces to for this image, or `None` when the value
    /// the writer stored is already final.
    ///
    /// The two halves of the scan are not symmetric across link flavours, and
    /// the reason they differ is the point of this function. A symbol-based
    /// entry names something only the loader can resolve, which is as true of
    /// a fixed-base (`ET_EXEC`) image as of a position-independent one:
    /// `ld.lld` gives a non-PIE executable `R_X86_64_64 environ` for
    /// `char ***p = &environ;` exactly as it gives a PIE one, and without it
    /// the slot keeps the zero the writer left and the program dereferences
    /// null. A `RELATIVE` entry, by contrast, asks the loader to add the load
    /// base to an address this link already resolved; a fixed-base image has
    /// no load base to add, so the writer's value is final and the entry would
    /// be a no-op at best.
    ///
    /// The symbol-based half has one exception, [`Self::resolves_to_zero`]:
    /// a name the loader could only ever resolve to zero needs no entry
    /// either. `writable` is the slot's section, which is half of that
    /// question.
    fn emitted(self, kind: SlotReloc, writable: bool) -> Option<SlotReloc> {
        match kind {
            SlotReloc::ByName(id) => (!self.resolves_to_zero(id, writable))
                .then_some(SlotReloc::ByName(id)),
            SlotReloc::Relative => {
                relative_applies(self.pie).then_some(SlotReloc::Relative)
            }
        }
    }

    /// Whether a slot naming `id` is a link-time constant of zero, so the zero
    /// the writer already stored is final and no entry describes anything.
    ///
    /// A weak undefined nothing defines is the ABI's own "absent" answer: the
    /// feature-probe idiom, `extern void f(void) __attribute__((weak));
    /// if (f) f();`, compiles to a pointer-width absolute reference whose value
    /// is zero precisely because no definition exists. This link supplies none
    /// and no `DT_NEEDED` dependency exports the name, so the only address the
    /// loader could ever store is the zero already there. Without this the
    /// reference is classified symbol-based and [`classify_data_reloc`] then
    /// refuses the link outright whenever the probe sits in a read-only
    /// section, which is where a `-fno-pie` compiler puts it.
    ///
    /// This is lld's rule for an in-place reference, the second clause of
    /// `RelocScan::isStaticLinkTimeConstant`
    /// (`lld/ELF/Relocations.cpp`), whose preemptible arm returns
    /// `sym.isUndefined() && !ctx.arg.isPic`.
    ///
    /// `isUndefined` is the first half here. It is false for a name a
    /// dependency defines -- lld resolves that to a `SharedSymbol` -- which
    /// xold records as a `dep_exports` row beside a still-undefined symbol, so
    /// a reference the loader really can bind keeps its entry.
    ///
    /// `!isPic` is the second half, and it does not translate directly. lld
    /// reads it off the command line; xold has no such flag and derives the
    /// answer instead, so the same input can land either way and gating on
    /// [`Self::pie`] alone would leave the rule hostage to whether the program
    /// happened to contain an unrelated absolute reference. What the clause
    /// really asks is whether the loader could still put something else in the
    /// slot, and there are two ways it cannot: the image has a fixed base, or
    /// the slot is read-only. A writable slot in a position-independent image
    /// is neither, and keeps its entry.
    ///
    /// A *strong* undefined is deliberately not included, though lld treats the
    /// two alike. lld can, because it has already refused the link at the
    /// reference (`maybeReportUndefined`); xold leaves a strong import for the
    /// loader to resolve, and the entry is what makes the loader report it.
    /// Dropping it would turn a startup failure into a silent null pointer.
    fn resolves_to_zero(self, id: SymbolId, writable: bool) -> bool {
        if self.pie && writable {
            return false;
        }
        let weak_undef = self.ctx.symbols.symbol(id).is_some_and(|s| {
            matches!(s.kind, SymbolKind::Undefined { weak: true })
        });
        weak_undef
            && self
                .ctx
                .dep_exports
                .get(self.ctx.symbols.name(id))
                .is_none()
    }
}

/// What the absolute-data-relocation emitter reads, bundled so the per-section
/// and per-relocation helpers stay small.
#[derive(Clone, Copy)]
struct DataRelocCtx<'a, 'data> {
    scope: DataRelocScope<'a, 'data>,
    layout: &'a Layout,
    /// The dynamic symbol table, which names each target the loader resolves.
    table: &'a DynSym,
    /// The target's pointer-width absolute relocation type.
    abs64: u32,
    /// The target's `R_*_RELATIVE` type.
    relative_type: u32,
}

/// Classifies one absolute data reference by what its target resolves to, and
/// refuses the reference outright when the entry it needs cannot be applied.
///
/// Every pass calls this, so the bucket the sizing pass counts is the bucket
/// the emitter fills, down to the link flavour ([`DataRelocScope::emitted`])
/// and the refusal below. `Ok(None)` means the reference needs no dynamic
/// relocation at all: the value the link wrote into the slot is already final.
///
/// A file-local or section symbol is an address this link fixes; everything
/// else is [`classify_slot`]'s to answer.
///
/// A section without `SHF_WRITE` is one the loader cannot store into. It could
/// be asked to make the mapping writable first, which is what `DT_TEXTREL`
/// means, but that tag is deprecated and xold emits none -- so an entry
/// pointing into such a section describes a fixup no loader will perform, and
/// the image loads with the slot unrelocated. Refusing is lld's answer too,
/// reached at the end of `RelocationScanner::processAux`
/// (`lld/ELF/Relocations.cpp`: "relocation ... cannot be used
/// against ...; recompile with -fPIC"); GNU ld and mold instead accept it and
/// emit `DT_TEXTREL`. No compiler output reaches the refusal: `-fPIC` and
/// `-fPIE` put relocatable data in `.data.rel.ro`, which is writable, and a
/// `-fno-pic` reference to the image's own symbols is `Relative`, which a
/// fixed-base image drops before it gets here. So does a weak undefined
/// nothing can define, whose slot needs no entry at all
/// ([`DataRelocScope::resolves_to_zero`]) -- the feature-probe idiom is
/// exactly a read-only pointer to a symbol this link cannot pin down, and
/// refusing it is refusing the answer "absent". What reaches the refusal is
/// the rest of that shape: assembly with a pointer
/// inside an `SHF_EXECINSTR` section, an import no copy slot and no canonical
/// PLT stub stands in for (an export that is neither an object nor a function,
/// or one with no storage to copy), and either of those in a shared object,
/// whose own definitions are preemptible and which has no fixed base to fall
/// back on. An executable's read-only pointer to an import it *can* pin down
/// does not: [`crate::layout::scan`] counts that among the references forcing
/// a fixed base, which makes it `Relative` and [`DataRelocScope::emitted`]
/// drops it.
fn classify_data_reloc(
    scope: DataRelocScope<'_, '_>,
    sec: &DataRelocSection<'_>,
    r: &Rela64,
) -> Result<Option<SlotReloc>> {
    let Some(sym) = sec.symbol(r) else {
        return Ok(None);
    };
    let verdict = if sym.bind() == STB_LOCAL {
        Some(SlotReloc::Relative)
    } else {
        // The resolved id by index rather than by name: this walk visits every
        // absolute relocation of every allocated section, and hashing the name
        // and comparing its bytes at each one was the single hottest thing a
        // dynamic link did. See [`Context::global_id`].
        scope.ctx.global_id(sec.file, r.sym()).and_then(|id| {
            classify_slot(scope.ctx, scope.shared, scope.fixed, id)
        })
    };
    let Some(kind) = verdict.and_then(|k| scope.emitted(k, sec.writable))
    else {
        return Ok(None);
    };
    if !sec.writable {
        return Err(text_reloc(sec.symtab.name(sym)));
    }
    Ok(Some(kind))
}

/// The diagnostic for a dynamic relocation that would land in a section the
/// loader cannot write. The target is defined -- preemptible, or bound to an
/// address the loader owns -- so this is not the undefined-reference error;
/// [`Error::TextRelocation`] names the symbol, and the remedy, the way lld's
/// does.
fn text_reloc(name: &[u8]) -> Error {
    Error::TextRelocation(show(name))
}

/// Which dynamic relocation a slot holding the address of resolved global `id`
/// needs, for an image that fixes the address of the imports in `fixed`.
///
/// This is the one predicate every slot is classified through: the GOT slots
/// here, the absolute data references beside them, and the sizing pass that
/// reserves room for both. Splitting the decision was what let the emitter and
/// the sizing pass disagree about a copy-relocated symbol.
///
/// An import this link fixes the address of takes `RELATIVE`, not a
/// symbol-based entry against a name that no longer describes where the value
/// lives. That covers both mechanisms in [`FixedImports`]:
///
/// - A copy-relocated symbol is defined by the executable. The loader copies
///   the dependency's initial bytes into a `.bss` slot this link placed, and
///   every reference resolves to that slot, so nothing can preempt it.
/// - A canonical PLT symbol stands for its stub, and every reference in the
///   image resolves to that one address. `.dynsym` carries the stub's address
///   in the import's `st_value` so other images bind the name to it too
///   (`lld/ELF/Relocations.cpp`), which is what keeps a function pointer
///   comparing equal wherever it was taken from.
///
/// lld reaches the same answer for both by replacing the shared symbol with a
/// definition -- `.bss`-backed for a copy, stub-backed for a canonical PLT --
/// and recomputing preemptibility from it.
pub(super) fn classify_slot(
    ctx: &Context<'_>,
    shared: bool,
    fixed: FixedImports<'_>,
    id: SymbolId,
) -> Option<SlotReloc> {
    if fixed.contains(id) {
        return Some(SlotReloc::Relative);
    }
    classify_global(ctx, shared, id)
}

/// Which dynamic relocation a slot holding the address of resolved global `id`
/// needs, or `None` when the value this link wrote is already final.
///
/// Mirrors lld's `addGotEntry`, and holds for any such slot rather than just a
/// GOT one. A preemptible symbol is one another image may define: an undefined
/// reference the loader satisfies, or a shared object's own default-visibility
/// definition, which the executable that loads it may interpose. Neither is
/// this link's to fix, so the slot stays symbol-based. What is left resolves
/// here, and takes `RELATIVE` when it is an address that moves with the load
/// base and nothing when it is not: an `SHN_ABS` definition is a constant, and
/// a reference nothing can preempt and nothing defines resolves to zero.
/// Shifting either by the load base would corrupt it.
pub(super) fn classify_global(
    ctx: &Context<'_>,
    shared: bool,
    id: SymbolId,
) -> Option<SlotReloc> {
    let sym = ctx.symbols.symbol(id)?;
    if sym.is_preemptible(shared) {
        return Some(SlotReloc::ByName(id));
    }
    match &sym.kind {
        SymbolKind::Defined(def) => {
            definition_is_address(ctx, id, def).then_some(SlotReloc::Relative)
        }
        SymbolKind::Common { .. } => Some(SlotReloc::Relative),
        SymbolKind::Undefined { .. } => None,
    }
}

/// Whether a resolved definition names a place in the image rather than a
/// link-time constant.
///
/// A section-backed definition always does. An absolute one usually does not
/// (`SHN_ABS` in the input), with one exception: the bounds the linker defines
/// itself (`_end`, `__init_array_start`, ...) are recorded as absolute yet name
/// addresses the layout chose, so they move with the load base.
fn definition_is_address(
    ctx: &Context<'_>,
    id: SymbolId,
    def: &Definition,
) -> bool {
    matches!(def.source, DefSource::Section { .. })
        || ctx.defsyms.iter().any(|&(defsym, _)| defsym == id)
}

/// The two halves of `.rela.dyn` under construction: the `R_*_RELATIVE` prefix
/// `DT_RELACOUNT` covers, and the symbol-based entries that follow it.
struct RelaSink<'a> {
    relative: &'a mut Vec<Rela64>,
    sym_based: &'a mut Vec<Rela64>,
}

impl RelaSink<'_> {
    /// Appends an `R_*_RELATIVE` entry for a slot the loader fills with
    /// `value` shifted by the load base.
    fn push_relative(&mut self, r_type: u32, slot: u64, value: u64) {
        self.relative.push(make_rela(slot, 0, r_type, value));
    }
}

/// Appends one dynamic relocation per pointer-width absolute relocation of
/// every section [`for_each_data_reloc_section_of`] yields. The slot address is
/// the section's placed address plus the relocation's in-section offset.
///
/// The per-file rows are appended in input order, so the emitted table is the
/// one the serial walk produced whatever order the files were classified in.
fn scan_data_relocs(
    dr: DataRelocCtx<'_, '_>,
    out: &mut RelaSink<'_>,
) -> Result<()> {
    let rows = map_data_reloc_files(
        dr.scope.ctx,
        Some(&dr.layout.abs_candidates),
        |sec, acc: &mut RelaRow| {
            // A section `--gc-sections` dropped keeps no placed address: its
            // bytes are not in the image, so nothing needs fixing
            // up at runtime. The sizing pass counts it all the same
            // (placement is not known that early), which
            // over-reserves the region rather than overrunning it.
            let Some(vaddr) = dr.layout.section_vaddr(sec.file, sec.section)
            else {
                return Ok(());
            };
            let resolver = dr.layout.resolver(sec.file);
            let mut sink = RelaSink {
                relative: &mut acc.relative,
                sym_based: &mut acc.sym_based,
            };
            for r in sec.absolute(dr.abs64) {
                let slot = vaddr.wrapping_add(r.r_offset.get());
                emit_data_reloc(dr, sec, &resolver, r, slot, &mut sink)?;
            }
            Ok(())
        },
    )?;
    for mut row in rows {
        out.relative.append(&mut row.relative);
        out.sym_based.append(&mut row.sym_based);
    }
    Ok(())
}

/// One input file's contribution to the two halves of `.rela.dyn`, before the
/// rows are appended in input order.
#[derive(Default)]
struct RelaRow {
    relative: Vec<Rela64>,
    sym_based: Vec<Rela64>,
}

/// Appends the dynamic relocation one absolute data reference needs, following
/// the [`classify_data_reloc`] verdict the sizing pass counted.
fn emit_data_reloc(
    dr: DataRelocCtx<'_, '_>,
    sec: &DataRelocSection<'_>,
    resolver: &FileResolver<'_>,
    r: &Rela64,
    slot: u64,
    out: &mut RelaSink<'_>,
) -> Result<()> {
    match classify_data_reloc(dr.scope, sec, r)? {
        None => Ok(()),
        Some(SlotReloc::Relative) => {
            // The addend the loader applies is the value the writer would have
            // stored had the base been known: `S + A`, evaluated through the
            // very expression and per-file resolver the in-place apply uses,
            // over an addend rewritten by the one merge rule. A second
            // spelling of `S + A` here is how the two paths drifted apart
            // before, leaving a reference into a deduplicated pool at its
            // pre-merge offset.
            let sym = (r.sym() != 0)
                .then(|| SymbolId(usize::try_from(r.sym()).unwrap_or(0)));
            let addend = dr.scope.ctx.merge.addend(sec.symtab, sec.file, r);
            let value = RelExpr::Abs.compute(sym, addend, 0, resolver)?;
            out.push_relative(dr.relative_type, slot, value);
            Ok(())
        }
        Some(SlotReloc::ByName(id)) => emit_by_name(dr, sec, r, slot, id, out),
    }
}

/// Appends the symbol-based entry a target the loader resolves needs.
///
/// A target with no `.dynsym` entry has no name the loader can look up.
/// [`data_import_ids`] gives a row to every import an absolute data reference
/// resolves by name, so one case is left: a hidden or internal weak undefined,
/// which no loader could ever satisfy and which therefore resolves to zero by
/// definition. Its slot already holds that zero and stays as it is; a
/// `RELATIVE` would shift it by the load base. In an executable naming a
/// strong symbol no dependency exports, nothing can ever fill the slot, so the
/// reference is reported as the undefined reference it is rather than dropped.
fn emit_by_name(
    dr: DataRelocCtx<'_, '_>,
    sec: &DataRelocSection<'_>,
    r: &Rela64,
    slot: u64,
    id: SymbolId,
    out: &mut RelaSink<'_>,
) -> Result<()> {
    let Some(sym) = sec.symbol(r) else {
        return Ok(());
    };
    // The reference may arrive in its versioned spelling (`bar@VER`); the
    // dynsym row it asks for is filed under the stem.
    let name = version_stem(sec.symtab.name(sym));
    if let Some(sym_idx) = dr.table.index_of(name) {
        out.sym_based.push(make_rela(
            slot,
            sym_idx,
            dr.abs64,
            r.r_addend.get().cast_unsigned(),
        ));
        return Ok(());
    }
    let strong_import = dr.scope.ctx.symbols.symbol(id).is_some_and(|s| {
        matches!(s.kind, SymbolKind::Undefined { weak: false })
    });
    if !strong_import
        || dr.scope.shared
        || dr.scope.ctx.dep_exports.get(name).is_some()
    {
        return Ok(());
    }
    Err(Error::UndefinedReference(show(name)))
}

/// What one pass over the absolute data references tells the sizing pass: how
/// many entries of each half they add to `.rela.dyn`, and which globals they
/// leave for the loader to resolve by name.
///
/// One walk answers both, and the plan carries its result to all three
/// consumers: the two sizing halves and the `.dynsym` rows they reserve.
#[derive(Default)]
pub(super) struct DataRelocSurvey {
    /// `R_*_RELATIVE` entries the absolute data references add.
    pub(super) relative: u64,
    /// Symbol-based entries they add.
    pub(super) glob: u64,
    /// The globals they leave for the loader, in input order, each once.
    pub(super) imports: Vec<SymbolId>,
}

/// One input file's half of a [`DataRelocSurvey`], before the rows are joined.
#[derive(Default)]
struct SurveyRow {
    relative: u64,
    glob: u64,
    imports: Vec<SymbolId>,
}

/// Surveys the target's pointer-width absolute relocations across the sections
/// [`for_each_data_reloc_section_of`] yields, classifying each by the same
/// [`classify_data_reloc`] the emitter follows and without resolving any
/// address.
///
/// The tally is an upper bound on what the emitter writes, never a lower one:
/// the two walk the same sections and classify alike, but the emitter skips a
/// section placement left out of the image, and a symbol-based target that
/// reached no `.dynsym` entry. The surplus stays zeroed in the region, which
/// the loader reads as `R_*_NONE`.
///
/// The import list is the third source of `.dynsym` rows, beside the copy slots
/// and the GOT/PLT keys. A reference such as `long *p = &imported;` allocates
/// neither a GOT slot nor a PLT entry, so nothing else asks for a row; without
/// one, [`emit_by_name`] has no index to name and the entry this reserved stays
/// zeroed, which the loader reads as `R_*_NONE` and which leaves a null pointer
/// behind. Only an undefined reference is listed: a shared object's own
/// definition already has an export row, and a copy-relocated symbol is
/// classified `Relative` and not resolved by name at all. It is a list rather
/// than a set because the order reaches the image -- `.dynsym` keeps its
/// undefined rows in insertion order -- so `seen` decides membership only,
/// over rows joined in input order.
///
/// This runs in the sizing probe, before any region has an address, so it is
/// also where a reference [`classify_data_reloc`] refuses fails the link:
/// nothing has been placed, let alone written.
pub(super) fn survey_data_relocs(
    ctx: &Context<'_>,
    layout: &Layout,
    fixed: FixedImports<'_>,
    abs64: u32,
) -> Result<DataRelocSurvey> {
    let scope = DataRelocScope::new(ctx, layout, fixed);
    let rows = map_data_reloc_files(ctx, None, |sec, acc: &mut SurveyRow| {
        for r in sec.absolute(abs64) {
            match classify_data_reloc(scope, sec, r)? {
                Some(SlotReloc::Relative) => {
                    acc.relative = acc.relative.saturating_add(1);
                }
                Some(SlotReloc::ByName(id)) => {
                    acc.glob = acc.glob.saturating_add(1);
                    if is_runtime_import(ctx, id)
                        && !ctx.symbols.name(id).is_empty()
                    {
                        acc.imports.push(id);
                    }
                }
                None => {}
            }
        }
        Ok(())
    })?;
    let mut out = DataRelocSurvey::default();
    let mut seen: FxHashSet<SymbolId> = FxHashSet::default();
    for row in rows {
        out.relative = out.relative.saturating_add(row.relative);
        out.glob = out.glob.saturating_add(row.glob);
        out.imports
            .extend(row.imports.into_iter().filter(|&id| seen.insert(id)));
    }
    Ok(out)
}

/// Constructs one packed `Rela64`.
#[allow(clippy::cast_possible_truncation)]
fn make_rela(offset: u64, sym: u32, r_type: u32, addend: u64) -> Rela64 {
    Rela64 {
        r_offset: U64::new(offset),
        r_info: U64::new((u64::from(sym) << 32) | u64::from(r_type)),
        r_addend: I64::new(addend.cast_signed()),
    }
}

/// Counts the leading `R_*_RELATIVE` entries (the loader prefix).
pub(super) fn count_relative(rela: &[Rela64], relative_type: u32) -> u32 {
    let mut n = 0u32;
    for r in rela {
        if (r.r_info.get() & 0xffff_ffff) as u32 == relative_type {
            n = n.saturating_add(1);
        } else {
            break;
        }
    }
    n
}
