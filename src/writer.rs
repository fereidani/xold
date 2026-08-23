//! The ELF image writer: serialises a [`crate::layout::Layout`] into an
//! `ET_EXEC` byte image.
//!
//! The writer is format glue only. It copies each allocated input section's
//! bytes into the image at the offset [`crate::layout`] chose, asks the
//! arch-neutral [`crate::reloc::apply`] driver to patch every relocation
//! through the per-file resolver, fills the GOT, and appends a symbol table,
//! string tables and section headers for inspection. No address arithmetic
//! lives here; the layout already fixed every offset.
//!
//! The work is split across focused submodules under `writer/`: ELF header
//! ([`ehdr`]), program headers ([`phdrs`]), section headers ([`shdrs`]),
//! section copy and relocation ([`sections`]), GOT fill ([`got`]), symbol and
//! string tables ([`symtab`]), DWARF debug sections ([`dwarf`]) and the
//! synthetic PLT/dynamic byte emission ([`bytes`]). This module is the thin
//! orchestrator that drives them in order and owns the shared constants and
//! image-buffer helpers they all reach through `super::`.

mod bytes;
mod dwarf;
mod ehdr;
mod got;
mod phdrs;
mod sections;
mod shdrs;
mod symtab;

use std::path::Path;

use bytes::{write_dynamic_bytes, write_plt_bytes};
use dwarf::{DebugPlan, debug_end, plan_debug, write_debug_bytes};
use ehdr::write_ehdr;
use got::fill_got;
use sections::copy_sections;
use shdrs::{TableOffsets, build_shdrs, materialise};
use symtab::{StrTab, SymtabPlan, build_symtab};

use crate::{
    dynamic::{DynConfig, DynamicPlan, LinkMode, Strip},
    elf::{Shdr64, Sym64},
    error::{Error, Result},
    layout::{Layout, Region, Sect, sect},
    linker::Context,
    mmap_file::OutputFile,
    plt::PltPlan,
    reloc::Target,
    util::align_up,
};

/// On-disk size of one section header.
const SHDR_SIZE: u64 = 64;
/// On-disk size of the ELF64 header.
const EHDR_SIZE: u64 = 64;
/// On-disk size of one program header.
const PHDR_SIZE: u64 = 56;
/// On-disk size of one symbol-table entry.
const SYM_SIZE: u64 = 24;
const DYNAMIC_ENTRY_SIZE: u64 = 16;
/// ELF version (`EV_CURRENT`).
const EV_CURRENT: u32 = 1;

/// The file offsets of everything that follows the loaded image, and the
/// tables themselves.
///
/// The writer computes this first so the output file can be sized and mapped
/// once. Nothing here depends on the image bytes: the symbol table comes from
/// the layout's exports and the inputs' own local symbols, and the section
/// headers from the section table, all fixed by the time the writer runs.
struct ImagePlan {
    /// Placement of the non-allocated `.debug_*` sections.
    debug: DebugPlan,
    /// `.symtab` entries and the `.strtab` bytes their names live in. The
    /// index of the first non-local entry rides in [`TableOffsets`], which is
    /// what the section header is built from.
    syms: Vec<Sym64>,
    strtab: StrTab,
    /// How many section headers the image carries, and the `.shstrtab`
    /// bytes that name them. The header rows themselves are materialised
    /// late, after the synthetic regions are tightened; the count and the
    /// names are fixed here.
    shdr_count: usize,
    shstrtab: StrTab,
    /// Offsets of the trailing tables.
    off: TableOffsets,
    shdr_off: u64,
    /// Total file size, and so the size the output mapping is created at.
    total: u64,
}

/// Links `layout` into `output`.
///
/// `target` selects the relocation table used to patch each relocation in
/// place. When `relax` is set, the writer asks each relaxable site's arch hook
/// to rewrite it into a cheaper form (for example x86-64 `GOTPCRELX` `mov` ->
/// `lea`) before falling back to the spec's normal apply path; otherwise the
/// apply loop runs the unchanged GOT/PLT value computation. A shared layout
/// additionally serialises the dynamic synthetic sections and skips the
/// link-time GOT fill (the loader fills those slots via `.rela.dyn`).
///
/// Planning every offset first is what makes the image writable through a
/// mapping: the file can be sized exactly once, up front, instead of being
/// grown as each table is appended. The writer then serialises straight into
/// the page cache rather than into a heap buffer that has to be copied out.
/// See [`OutputFile`] for why the image is built under a temporary name.
pub fn write(
    ctx: &Context<'_>,
    target: Target,
    layout: &mut Layout,
    mode: LinkMode,
    config: &DynConfig<'_>,
    output: &Path,
) -> Result<()> {
    let plan = plan_image(ctx, layout, config.strip)?;
    let mut file = OutputFile::create(output, plan.total)?;
    fill_image(ctx, target, layout, mode, config, &plan, file.bytes())?;
    file.finish()
}

/// Computes every file offset and builds the trailing tables.
fn plan_image(
    ctx: &Context<'_>,
    layout: &Layout,
    strip: Strip,
) -> Result<ImagePlan> {
    // The debug sections follow the loaded image; the symbol and string
    // tables follow them. What `--strip-debug` and `--strip-all` drop is
    // dropped here, at the plan: a table that is not planned is not sized,
    // not written and given no section header, so the image simply does not
    // have it rather than carrying an empty one.
    let loaded = loaded_extent(layout);
    let debug = if strip == Strip::None {
        plan_debug(ctx, loaded)
    } else {
        DebugPlan::new()
    };
    let mut strtab = StrTab::new();
    let symtab = if strip == Strip::All {
        SymtabPlan::default()
    } else {
        build_symtab(ctx, layout, &mut strtab)?
    };
    let syms = symtab.syms;
    let symtab_size =
        u64::try_from(syms.len()).unwrap_or(0xffff_ffff) * SYM_SIZE;
    let strtab_size = u64::try_from(strtab.bytes.len()).unwrap_or(0);

    let symtab_off = align_up(debug_end(&debug, loaded), 8);
    let strtab_off = symtab_off.saturating_add(symtab_size);
    let shstrtab_off = strtab_off.saturating_add(strtab_size);

    let off = TableOffsets {
        symtab_off,
        symtab_size,
        symtab_first_global: symtab.first_global,
        strtab_off,
        strtab_size,
        shstrtab_off,
    };
    let mut shstrtab = StrTab::new();
    let shdrs = build_shdrs(
        layout,
        &mut shstrtab,
        &off,
        &debug,
        debug_sections(ctx, strip),
        strip != Strip::All,
    );

    let shstrtab_size = u64::try_from(shstrtab.bytes.len()).unwrap_or(0);
    let shdr_off = align_up(shstrtab_off.saturating_add(shstrtab_size), 8);
    let total = shdr_off
        .saturating_add(u64::try_from(shdrs.len()).unwrap_or(0) * SHDR_SIZE);
    Ok(ImagePlan {
        debug,
        syms,
        strtab,
        shdr_count: shdrs.len(),
        shstrtab,
        off,
        shdr_off,
        total,
    })
}

/// Writes the `NT_GNU_BUILD_ID` note, when the link asked for one.
///
/// The header first, then the digest over the whole image with the
/// descriptor still zeroed, which is what makes the note a function of the
/// image alone and so the same on every run.
fn write_build_id(config: &DynConfig<'_>, layout: &Layout, image: &mut [u8]) {
    let Some(build_id) = config.build_id else {
        return;
    };
    let at = layout.region(Sect::BuildId).offset;
    if layout.region(Sect::BuildId).size == 0 {
        return;
    }
    build_id.write_header(image, at);
    build_id.fill(image, at);
}

/// The debug sections the headers describe: none of them once
/// `--strip-debug` or `--strip-all` has dropped their content, since a header
/// naming bytes that were not written points a reader at whatever follows.
fn debug_sections<'a>(
    ctx: &'a Context<'_>,
    strip: Strip,
) -> &'a [crate::debug::DebugSection] {
    if strip == Strip::None {
        &ctx.debug.sections
    } else {
        &[]
    }
}

/// One past the last file byte of the loaded image: the ELF and program
/// headers plus every file-backed region. The `SHT_NOBITS` pair takes memory
/// but no file bytes and so does not extend it.
fn loaded_extent(layout: &Layout) -> u64 {
    let headers = EHDR_SIZE + u64::from(layout.phdr_count()) * PHDR_SIZE;
    sect::TABLE
        .iter()
        .filter(|(s, _)| sect::file_backed(*s))
        .map(|(s, _)| layout.region(*s).end())
        .fold(headers, u64::max)
}

/// Serialises the whole image into `image`, which the plan sized exactly.
fn fill_image(
    ctx: &Context<'_>,
    target: Target,
    layout: &mut Layout,
    mode: LinkMode,
    config: &DynConfig<'_>,
    plan: &ImagePlan,
    image: &mut [u8],
) -> Result<()> {
    let base = layout.base();
    let relax = config.relax;
    // The full dynamic and PLT plans read only the finished layout, never
    // the image, and the copy pass never reads them: the two run side by
    // side. The probe's placed regions keep answering `Layout::region`
    // for the copy while the full plan is built; it is swapped in below.
    let shared: &Layout = layout;
    let (copied, plans) = rayon::join(
        || write_loaded(ctx, target, shared, base, relax, image),
        || build_plans(ctx, shared, target, mode, config),
    );
    copied?;
    let (dyn_plan, plt_plan) = plans?;
    if let Some(p) = dyn_plan {
        layout.dynamic = Some(p);
    }
    if let Some(p) = plt_plan {
        layout.plt_plan = Some(p);
    }
    // The section headers are materialised only now: the plans above
    // tighten the synthetic regions to the serialised lengths, and
    // `sh_size` must describe the bytes actually written. The names are
    // the same either way, so the string table planned earlier still
    // matches; only the header rows are rebuilt.
    let mut shstrtab = StrTab::new();
    let shdrs = build_shdrs(
        layout,
        &mut shstrtab,
        &plan.off,
        &plan.debug,
        debug_sections(ctx, config.strip),
        config.strip != Strip::All,
    );
    debug_assert_eq!(shstrtab.bytes, plan.shstrtab.bytes);
    debug_assert_eq!(shdrs.len(), plan.shdr_count);
    if let Some(dyn_plan) = layout.dynamic.as_ref() {
        write_dynamic_bytes(dyn_plan, image)?;
    }
    if let Some(plt_plan) = layout.plt_plan.as_ref() {
        write_plt_bytes(plt_plan, layout, image)?;
    }
    // DWARF `.debug_*` sections: copy each member's bytes and apply its
    // `.rela.debug_*` relocations. Non-allocated: file-only, no vaddr. The
    // plan is empty when the debug sections were stripped, and the copy then
    // has nothing to do.
    write_debug_bytes(ctx, target, layout, &plan.debug, image)?;

    // `Sym64` and `Shdr64` are `Pod` and both tables are contiguous, so each
    // is one copy rather than a store per entry. The symbol and string
    // tables are tens of megabytes on a large link and land on the critical
    // path, so those two copy in parallel chunks.
    write_slice_at_par(
        image,
        plan.off.symtab_off,
        bytemuck::cast_slice(&plan.syms),
    );
    write_slice_at_par(image, plan.off.strtab_off, &plan.strtab.bytes);
    write_slice_at(image, plan.off.shstrtab_off, &plan.shstrtab.bytes);
    let table: Vec<Shdr64> = shdrs.iter().map(materialise).collect();
    write_slice_at(image, plan.shdr_off, bytemuck::cast_slice(&table));

    // `.shstrtab` is the last header in the table.
    let shstrtab_shndx =
        u16::try_from(shdrs.len().saturating_sub(1)).unwrap_or(0);
    write_ehdr(
        image,
        target,
        layout,
        plan.shdr_off,
        shdrs.len(),
        shstrtab_shndx,
    )?;
    phdrs::write_phdrs(image, layout, base);
    // The build-id note is the last thing written, and in two steps: its
    // header goes in before the digest is taken, and the digest covers the
    // finished image with the descriptor still zero. Anything written after
    // it would not be covered by the id that claims to name the image.
    write_build_id(config, layout, image);
    // Nothing may be published that a serialiser could not reach: the image is
    // sized from the layout, so a write past its end is a sizing bug.
    crate::util::check_truncated_write()?;
    Ok(())
}

/// Builds the full dynamic and PLT plans off the finished layout.
///
/// Runs beside the copy pass: nothing here reads the image, and the copy
/// reads neither plan. A static image builds no dynamic plan; a PLT plan is
/// built for any link with a PLT key, since a static image still needs
/// stubs for its own indirect functions.
fn build_plans(
    ctx: &Context<'_>,
    layout: &Layout,
    target: Target,
    mode: LinkMode,
    config: &DynConfig<'_>,
) -> Result<(Option<DynamicPlan>, Option<PltPlan>)> {
    let dyn_plan = if mode.is_dynamic() {
        Some(crate::dynamic::build_plan_from(
            ctx, layout, target, mode, config,
        )?)
    } else {
        None
    };
    let plt_plan = if layout.plt_keys.is_empty() {
        None
    } else {
        Some(crate::plt::build_plan(layout, dyn_plan.as_ref())?)
    };
    Ok((dyn_plan, plt_plan))
}

/// Writes the loaded content (sections copied and relocated, GOT filled) into
/// `image`. In a shared link the GOT is left zeroed so the loader can fill it
/// from `.rela.dyn`.
fn write_loaded(
    ctx: &Context<'_>,
    target: Target,
    layout: &Layout,
    base: u64,
    relax: bool,
    image: &mut [u8],
) -> Result<()> {
    let end = usize::try_from(loaded_extent(layout)).unwrap_or(usize::MAX);
    let loaded = image.get_mut(..end).ok_or(Error::OutOfRange("image"))?;
    copy_sections(ctx, target, layout, base, relax, loaded)?;
    // Every mode fills the GOT. A dynamic image has a relocation for most of
    // its slots, and the loader overwrites what it finds there, but not for
    // all of them: a slot holding a thread-local's offset within its own
    // module is fixed by this link and is the only copy of that value.
    fill_got(layout, loaded);
    // `.interp` is a plain byte sequence the loader reads via `PT_INTERP`.
    if layout.has_interp() {
        write_region(
            image,
            layout.region(Sect::Interp),
            &layout.interp,
            ".interp",
        )?;
    }
    // `.eh_frame_hdr` indexes the now-relocated `.eh_frame` bytes, so it must
    // be serialised after the copy+relocation pass patched every FDE's
    // initial_location slot. Its region was sized from an FDE count taken
    // before the table was built, so it is checked like the other two-pass
    // sections; the table may be shorter (a row covering no placed code is
    // dropped), never longer.
    if layout.region(Sect::EhFrameHdr).size != 0 {
        let hdr = crate::ehframe::build_hdr(image, layout)?;
        write_region(
            image,
            layout.region(Sect::EhFrameHdr),
            &hdr,
            ".eh_frame_hdr",
        )?;
    }
    Ok(())
}

/// Writes `bytes` at `offset` without a region to check them against.
///
/// Reserved for the trailing tables (`.symtab`, `.strtab`, `.shstrtab` and the
/// section headers), whose offsets [`plan_image`] derived from the very byte
/// lengths written here: those cannot disagree with what the file was sized
/// for. Everything placed as a [`Region`] goes through [`write_region`]
/// instead, because its size and its content come from two different passes.
fn write_slice_at(image: &mut [u8], offset: u64, bytes: &[u8]) {
    let start = usize::try_from(offset).unwrap_or(usize::MAX);
    let end = start.saturating_add(bytes.len());
    if let Some(slot) = image.get_mut(start..end) {
        slot.copy_from_slice(bytes);
    }
}

/// [`write_slice_at`], copying in parallel chunks: same bytes, same place,
/// but a multi-megabyte table is spread across the workers (and so is the
/// kernel's first-touch zeroing of the fresh output pages under it).
fn write_slice_at_par(image: &mut [u8], offset: u64, bytes: &[u8]) {
    use rayon::prelude::*;
    /// One chunk per copy task: big enough to amortise the fork, small
    /// enough to spread a table across every worker.
    const CHUNK: usize = 1 << 20;
    let start = usize::try_from(offset).unwrap_or(usize::MAX);
    let end = start.saturating_add(bytes.len());
    if let Some(slot) = image.get_mut(start..end) {
        slot.par_chunks_mut(CHUNK)
            .zip(bytes.par_chunks(CHUNK))
            .for_each(|(dst, src)| dst.copy_from_slice(src));
    }
}

/// Writes `bytes` into `region`, failing when they do not fit.
///
/// The synthetic sections are sized by a counting pass and serialised by a
/// separate one. Should the two ever disagree, an unchecked write would run
/// past the region and silently corrupt whatever the layout placed next (for
/// `.rela.dyn` in an executable, that is `.rela.plt`, and the image would fault
/// before `main`). `what` names the region in the error.
fn write_region(
    image: &mut [u8],
    region: Region,
    bytes: &[u8],
    what: &'static str,
) -> Result<()> {
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > region.size {
        return Err(Error::OutOfRange(what));
    }
    let start =
        usize::try_from(region.offset).map_err(|_| Error::OutOfRange(what))?;
    let end = start
        .checked_add(bytes.len())
        .ok_or(Error::OutOfRange(what))?;
    let slot = image.get_mut(start..end).ok_or(Error::OutOfRange(what))?;
    slot.copy_from_slice(bytes);
    Ok(())
}
