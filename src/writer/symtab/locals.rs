//! The file-scoped symbols copied from the inputs into `.symtab`.
//!
//! Symbol resolution interns only global and weak names: a local is
//! file-scoped, so nothing outside its own object can reference it and there
//! is nothing to merge. That leaves the output symbol table with no row for
//! any `static` function or object, and a program whose every private symbol
//! is nameless to `gdb`, `perf`, `addr2line` and every backtrace symbolizer.
//! The rows are therefore read back out of each input's own symbol table here,
//! at write time, rather than carried through the link.
//!
//! ELF requires every local to precede every global in `.symtab`, and
//! `sh_info` to hold the index of the first global, so these rows are emitted
//! as one block directly after the null entry (see
//! [`super::build_symtab`]).

use rayon::prelude::*;

use crate::{
    elf::{
        Shdr64, Sym64,
        constants::{
            SHF_ALLOC, SHF_MERGE, SHN_ABS, SHN_LORESERVE, SHN_UNDEF, STB_LOCAL,
            STT_SECTION, STT_TLS,
        },
    },
    error::Result,
    input::InputFile,
    layout::Layout,
    linker::Context,
};

/// One input local symbol selected for the output symbol table.
///
/// The name is borrowed from the input's string table, which outlives the
/// writer, so the row costs no allocation until it is interned.
pub(super) struct LocalSym<'a> {
    pub(super) name: &'a [u8],
    /// The resolved `st_value`: an address in the image, an offset within the
    /// TLS block for a thread-local, or the input's own value for an absolute
    /// symbol.
    pub(super) value: u64,
    pub(super) size: u64,
    /// `st_info`, copied verbatim: the binding is already `STB_LOCAL` and the
    /// type is the input's.
    pub(super) info: u8,
    pub(super) other: u8,
    pub(super) shndx: u16,
}

/// Collects the local symbols of every input, in file order and then in input
/// symbol-table order.
///
/// That order is the whole determinism story for this block: it derives from
/// the command line and from each object's own table, never from a hash
/// container or from which thread finished first. The per-file walks are
/// independent and read nothing shared, so they run in parallel and are
/// concatenated in file order.
///
/// This is lld's `demoteAndCopyLocalSymbols` (`lld/ELF/Writer.cpp`), which
/// walks the same two loops in the same order.
pub(super) fn collect<'a>(
    ctx: &'a Context<'_>,
    layout: &Layout,
) -> Result<Vec<LocalSym<'a>>> {
    let per_file: Vec<Vec<LocalSym<'a>>> = ctx
        .files
        .par_iter()
        .enumerate()
        .map(|(file, input)| collect_file(ctx, layout, file, input))
        .collect::<Result<Vec<_>>>()?;
    Ok(per_file.into_iter().flatten().collect())
}

/// Collects one input's local symbols.
fn collect_file<'a>(
    ctx: &'a Context<'_>,
    layout: &Layout,
    file: usize,
    input: &'a InputFile<'_>,
) -> Result<Vec<LocalSym<'a>>> {
    let Some(symtab) = input.symbol_table()? else {
        return Ok(Vec::new());
    };
    let sections = input.object()?.sections();
    // A thread-local's `st_value` is its offset within the TLS block, which is
    // what layout already publishes for a global one (`exports::export_for`).
    let tls_base = layout.tls_block().map_or(0, |b| b.vaddr);
    let mut out = Vec::new();
    for sym in symtab.syms {
        // Only the locals, and never a section symbol: lld's
        // `shouldKeepInSymtab` opens with `if (sym.isSection()) return false`,
        // and emits its own `STT_SECTION` rows instead -- but only under
        // `--emit-relocs`/`-r` (`lld/ELF/Writer.cpp` calls
        // `addSectionSymbols` behind `if (ctx.arg.copyRelocs)`). xold
        // implements neither, so an ordinary link needs no section symbol at
        // all, and one copied from an input would name an input section index
        // that no longer exists.
        if sym.bind() != STB_LOCAL || sym.type_() == STT_SECTION {
            continue;
        }
        let name = symtab.name(sym);
        let Some((value, shndx)) =
            place(ctx, layout, file, sections, sym, name, tls_base)
        else {
            continue;
        };
        out.push(LocalSym {
            name,
            value,
            size: sym.st_size.get(),
            info: sym.st_info,
            other: sym.st_other,
            shndx,
        });
    }
    Ok(out)
}

/// Where one local symbol lands: its `st_value` and output `st_shndx`, or
/// `None` when it contributes no row.
///
/// Four kinds are turned away, three of them matching lld:
///
/// - An undefined or common local, which defines nothing. lld keeps only what
///   `dyn_cast<Defined>` accepts.
/// - A local defined in a section that is not `SHF_ALLOC`. lld keeps such a
///   row, pointing it at the non-allocated output section the contribution
///   landed in; xold does not, and the omission is deliberate rather than a
///   consequence of the placement query below. The only non-allocated input
///   sections xold carries into the image are the aggregated `.debug_*` ones,
///   and a debugger reads those through the DWARF headers rather than through
///   `.symtab`. No compiler output surveyed defines a local symbol in a
///   `.debug_*` section at all, and emitting the rows would mean giving
///   `Layout` a second numbering scheme for the debug headers and a second
///   meaning for an offset of zero.
/// - A local whose section did not reach the image. That covers every way an
///   allocated section can be dropped -- collected by `--gc-sections`, or
///   discarded with a duplicate COMDAT group -- because
///   [`Layout::section_shndx`] answers only for a section that was placed. lld
///   reaches the same set through `!dr->section->isLive()`. A section ICF
///   folded onto a representative is *not* in this set: layout stamps it with
///   the representative's address, so the local resolves to the identical bytes
///   that survived, exactly as a global in the same section does.
/// - An assembler temporary in a mergeable section. `shouldKeepInSymtab` drops
///   `.L*` when `sym.section && (sym.section->flags & SHF_MERGE)`, because a
///   `.L` name is only there to anchor a string the assembler could not name
///   otherwise, and the pool it lands in has one copy per distinct content
///   rather than one per name.
fn place(
    ctx: &Context<'_>,
    layout: &Layout,
    file: usize,
    sections: &[Shdr64],
    sym: &Sym64,
    name: &[u8],
    tls_base: u64,
) -> Option<(u64, u16)> {
    let shndx = sym.st_shndx.get();
    // An absolute local names no section: its value is already final. This is
    // the `STT_FILE` row every object opens its locals with, which lld keeps
    // and which is what makes `nm` group the rest by translation unit.
    if shndx == SHN_ABS {
        return Some((sym.st_value.get(), SHN_ABS));
    }
    if shndx == SHN_UNDEF || shndx >= SHN_LORESERVE {
        return None;
    }
    let flags = sections.get(usize::from(shndx))?.sh_flags.get();
    // Stated here rather than left to `section_shndx` answering `None`: the
    // two questions are different, and a change that taught the placement
    // tables about the debug sections would otherwise start emitting rows
    // naming them without anything having decided to.
    if flags & SHF_ALLOC == 0 {
        return None;
    }
    let out_shndx = layout.section_shndx(file, shndx)?;
    let base = layout.section_vaddr(file, shndx)?;
    if flags & SHF_MERGE != 0 && name.starts_with(b".L") {
        return None;
    }
    // A symbol in a merged section names content that moved when it was
    // deduplicated: `base` is the pool's, so the offset has to be the one the
    // merge pass assigned. This is `layout::addr::definition_addr`'s rule, and
    // it must be spelled the same way here -- the relocation path resolves
    // such a symbol to the pool base and folds the offset into the addend
    // instead, which is right for a relocation and wrong for a symbol table.
    let offset = sym.st_value.get();
    let moved = ctx.merge.remap(file, shndx, offset).unwrap_or(offset);
    let addr = base.wrapping_add(moved);
    let value = if sym.type_() == STT_TLS {
        addr.wrapping_sub(tls_base)
    } else {
        addr
    };
    Some((value, out_shndx))
}
