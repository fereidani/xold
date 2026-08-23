//! Output-section measurement and region placement: the pass that turns the
//! grouped input sections into placed regions with file offsets and virtual
//! addresses, including the synthetic PLT, GOT.PLT, `.rela.plt` and `.interp`
//! regions a dynamic executable adds.
//!
//! The image uses an identity map (a byte's virtual address is the load base
//! plus its file offset), so every region's `vaddr = base + offset`.

// Private helpers defined in the parent module (visible to child modules).
use super::{EHDR_SIZE, GOT_ENTRY, PAGE, PHDR_SIZE, max_page_size};
use crate::{
    dynamic::{DYNAMIC_ALIGN, DynConfig, LinkMode},
    layout::{Context, Layout, OutKind, Region, Sect, sect},
    startstop::Bounds,
    symbol::SymbolKind,
    util::align_up,
};

/// Stamps each debug member's offset within its aggregated output section
/// into `sec_vaddr`, so a `.rela.debug_*` relocation against the member's
/// section symbol resolves to the contribution's base. DWARF offsets are
/// section-relative, so the section symbol of a debug section resolves to
/// the start of this file's contribution inside the output section (not to
/// a loaded address: debug sections are non-allocated and have no vaddr).
/// Runs after the allocated sections are placed and before address
/// resolution reads any section.
pub(super) fn stamp_debug_offsets(ctx: &Context<'_>, layout: &mut Layout) {
    for sec in &ctx.debug.sections {
        for m in &sec.members {
            // A merged section's content was deduplicated into a pool, so its
            // symbol resolves to the pool base and the writer's rewritten
            // addend supplies the offset within it. Stamping the member's own
            // offset here would add it twice.
            if ctx.merge.is_merged(m.file, m.section) {
                continue;
            }
            if let Some(f) = layout.sec_vaddr.get_mut(m.file)
                && let Some(slot) = f.get_mut(usize::from(m.section))
            {
                *slot = m.out_offset;
            }
        }
    }
}

/// The owned `.interp` bytes (a NUL-terminated path) for a dynamic executable.
/// Empty unless the mode is [`LinkMode::DynExec`] and a path was supplied.
pub(super) fn interp_bytes(config: &DynConfig<'_>, mode: LinkMode) -> Vec<u8> {
    if mode != LinkMode::DynExec {
        return Vec::new();
    }
    let Some(path) = config.interpreter else {
        return Vec::new();
    };
    let mut bytes = Vec::with_capacity(path.len().saturating_add(1));
    bytes.extend_from_slice(path);
    bytes.push(0);
    bytes
}

/// The `.plt` section size: the target's resolver trampoline (`PLT[0]`,
/// 16 bytes on x86-64 and 32 on AArch64/RISC-V) plus one 16-byte stub per
/// imported symbol.
fn plt_section_size(layout: &Layout) -> u64 {
    let spec = layout.target.plt_spec();
    let entries = u64::try_from(layout.plt_keys.len()).unwrap_or(0);
    spec.header_size
        .wrapping_add(entries.wrapping_mul(spec.entry_size))
}

/// The `.got.plt` size: the target's reserved header slots (three for the
/// `SysV` ABI used by x86-64/`AArch64`, two for RISC-V) plus one slot per
/// PLT entry.
fn got_plt_section_size(layout: &Layout) -> u64 {
    let spec = layout.target.plt_spec();
    let entries = u64::try_from(layout.plt_keys.len()).unwrap_or(0);
    spec.got_plt_reserved
        .wrapping_mul(GOT_ENTRY)
        .wrapping_add(entries.wrapping_mul(GOT_ENTRY))
}

/// The `.rela.plt` size: one 24-byte `R_*_JUMP_SLOT` entry per PLT entry.
fn rela_plt_section_size(layout: &Layout) -> u64 {
    let entries = u64::try_from(layout.plt_keys.len()).unwrap_or(0);
    entries.wrapping_mul(24)
}

/// The measured extent of every output section, plus the relative placement of
/// every member within its section.
pub(super) struct Relative {
    /// Per file, per input section: the member's offset within its output
    /// section. Indexed rather than hashed -- there is one entry per input
    /// section in the link, and the rows are already sized by the caller.
    member: Vec<Vec<u64>>,
    /// Aligned total size of each output section, indexed by [`OutKind`].
    size: [u64; OutKind::COUNT],
    /// Alignment of each output section, indexed by [`OutKind`].
    align: [u64; OutKind::COUNT],
}

impl Relative {
    /// The offset of one member within its output section, or zero if the
    /// section is not a placed member.
    fn member_offset(&self, file: usize, section: u16) -> u64 {
        self.member
            .get(file)
            .and_then(|f| f.get(usize::from(section)))
            .copied()
            .unwrap_or_default()
    }

    /// The aligned byte size of one output section.
    pub(super) fn size(&self, kind: OutKind) -> u64 {
        self.size.get(kind.index()).copied().unwrap_or_default()
    }

    /// The alignment of one output section (at least 1).
    pub(super) fn align(&self, kind: OutKind) -> u64 {
        self.align.get(kind.index()).copied().unwrap_or(1)
    }

    /// Records a measured extent.
    fn set(&mut self, kind: OutKind, size: u64, align: u64) {
        if let Some(slot) = self.size.get_mut(kind.index()) {
            *slot = size;
        }
        if let Some(slot) = self.align.get_mut(kind.index()) {
            *slot = align;
        }
    }
}

/// Walks the output sections once, measuring sizes (with alignment) and the
/// relative offset of every member. Common symbols are appended to `.bss`.
///
/// `.eh_frame` is the one section whose members are packed rather than
/// individually aligned. Its content is a sequence of records, each introduced
/// by a length word that is zero only at the end of the section, so four zero
/// bytes of padding between two members would read as the end of the unwind
/// data. [`Context::split_eh_frame`] has already re-emitted every member as
/// whole records for exactly this reason; the section as a whole still starts
/// on its `out.align` boundary.
pub(super) fn measure_sections(
    ctx: &Context<'_>,
    member: Vec<Vec<u64>>,
) -> (Relative, Vec<u64>) {
    let mut rel = Relative {
        member,
        size: [0; OutKind::COUNT],
        align: [1; OutKind::COUNT],
    };
    for out in ctx.outputs.iter() {
        let mut cursor = 0u64;
        let align = out.align.max(1);
        let packed = out.kind == OutKind::EhFrame;
        for m in &out.members {
            if !packed {
                cursor = align_up(cursor, m.align);
            }
            if let Some(f) = rel.member.get_mut(m.file)
                && let Some(slot) = f.get_mut(usize::from(m.section))
            {
                *slot = cursor;
            }
            cursor = cursor.saturating_add(m.size);
        }
        rel.set(out.kind, align_up(cursor, align), align);
    }
    let common_off = measure_commons(ctx, &mut rel);
    (rel, common_off)
}

/// Assigns each common symbol an offset within `.bss`, after its members, and
/// raises `.bss`'s own alignment to cover them.
///
/// The offsets are measured from the section start, so they only describe an
/// aligned address if the section start is at least as aligned as the strictest
/// common asks for. `.bss`'s members do not bound that: a `-fcommon`
/// translation unit declaring a 64-byte-aligned array in an image whose `.bss`
/// members ask for 16 would put it at `base + 16 (mod 64)`, and the aligned
/// SIMD or atomic access it was declared for faults. lld reaches the same
/// place by converting commons into input sections, whose alignment the output
/// section absorbs like any other member's.
fn measure_commons(ctx: &Context<'_>, rel: &mut Relative) -> Vec<u64> {
    let mut off = vec![0u64; ctx.symbols.len()];
    let mut cursor = rel.size(OutKind::Bss);
    let mut align = rel.align(OutKind::Bss);
    for id in ctx.symbols.ids() {
        let Some(sym) = ctx.symbols.symbol(id) else {
            continue;
        };
        if let SymbolKind::Common { size, align: a, .. } = &sym.kind {
            let a = (*a).max(1);
            align = align.max(a);
            cursor = align_up(cursor, a);
            off[id.0] = cursor;
            cursor = cursor.saturating_add(*size);
        }
    }
    rel.set(OutKind::Bss, align_up(cursor, 16), align);
    off
}

/// Assigns file offsets and virtual addresses to every region. In a dynamic
/// link the synthetic sections are placed alongside the static ones: the
/// read-only tables (`.hash`, `.dynsym`, `.dynstr`, `.rela.dyn`, `.rela.plt`)
/// extend the read-execute segment after `.rodata`, `.plt` sits between
/// `.text` and `.rodata`, and `.dynamic` opens the read-write segment at the
/// head of the RELRO run (see [`RW_ORDER`]).
/// The sizes of the synthetic sections the caller computed before placement:
/// neither is measured from an input member, and both have to be reserved
/// before an address exists.
#[derive(Clone, Copy)]
pub(super) struct Sizes {
    /// `.eh_frame_hdr`, sized from the FDE count.
    pub eh_frame_hdr: u64,
    /// `.note.gnu.build-id`, sized from the digest `--build-id` asks for.
    pub build_id: u64,
}

pub(super) fn place_regions(
    ctx: &Context<'_>,
    rel: &Relative,
    mode: LinkMode,
    sizes: Sizes,
    layout: &mut Layout,
) {
    let eh_frame_hdr_size = sizes.eh_frame_hdr;
    let dynamic = mode.is_dynamic();
    let has_interp = !layout.interp.is_empty();
    let has_plt = !layout.plt_keys.is_empty();
    // `.eh_frame_hdr` exists iff some input contributed `.eh_frame` bytes; the
    // caller sized it from the FDE count. A non-zero size is the signal that
    // both the section and its `PT_GNU_EH_FRAME` segment must be emitted.
    let has_eh_frame_hdr = eh_frame_hdr_size != 0;
    // A `PT_NOTE` exists iff the image carries a note: one the inputs
    // contributed, the build-id note this link synthesises, or both. The two
    // regions are placed next to each other so a single segment covers them.
    let has_note = rel.size(OutKind::Note) != 0 || sizes.build_id != 0;
    // Both the read-write segment and the RELRO run inside it exist when any
    // of their regions does. The answers come off the same ordered list the
    // placement below walks, so the header count cannot disagree with the
    // headers `crate::writer::phdrs` goes on to emit. A dynamic link always
    // has both, at minimum for `.dynamic`.
    let rw = RW_ORDER
        .iter()
        .any(|&(_, src)| rw_present(rel, layout, src));
    let relro = RW_ORDER
        .iter()
        .any(|&(s, src)| sect::relro(s) && rw_present(rel, layout, src));
    let has_tls = rel.size(OutKind::Tdata) != 0 || rel.size(OutKind::Tbss) != 0;
    // A position-independent dynamic executable is `ET_DYN` with a
    // runtime-chosen load base, so it emits `PT_PHDR`; a copy-relocating
    // executable uses a fixed base (`ET_EXEC`) and does not need it. A shared
    // object loaded via `dlopen` does not need it either.
    let has_phdr = mode == LinkMode::DynExec && layout.pie;
    layout.phdr_count = 1
        + u16::from(has_phdr)
        + u16::from(has_interp)
        + u16::from(rw)
        + 1
        + u16::from(has_tls)
        + u16::from(dynamic)
        + u16::from(relro)
        + u16::from(has_eh_frame_hdr)
        + u16::from(has_note);
    let headers = EHDR_SIZE + u64::from(layout.phdr_count) * PHDR_SIZE;

    let flags = RxFlags {
        has_interp,
        has_plt,
        has_eh_frame_hdr,
        dynamic,
    };
    let rx_end = place_rx_segment(ctx, rel, flags, sizes, headers, layout);
    if rw {
        place_rw_segment(ctx, rel, rx_end, layout);
    }
}

/// The synthetic-region presence flags shared by [`place_regions`], bundled
/// so the placement helpers stay under the argument and boolean-parameter
/// limits.
#[derive(Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
struct RxFlags {
    has_interp: bool,
    has_plt: bool,
    has_eh_frame_hdr: bool,
    dynamic: bool,
}

/// Places the read-execute segment: the ELF header, `.interp`, `.text`,
/// `.plt`, `.rodata`, `.eh_frame` (+ `.eh_frame_hdr`), the read-only dynamic
/// tables and `.rela.plt`. Returns one past the segment's final byte.
fn place_rx_segment(
    ctx: &Context<'_>,
    rel: &Relative,
    flags: RxFlags,
    sizes: Sizes,
    headers: u64,
    layout: &mut Layout,
) -> u64 {
    let eh_frame_hdr_size = sizes.eh_frame_hdr;
    // `.interp` leads the read-execute segment, right after the headers.
    let mut cursor = headers;
    if flags.has_interp {
        let size = u64::try_from(layout.interp.len()).unwrap_or(0);
        layout.interp_region = region_at(layout, cursor, size);
        cursor = layout.interp_region.end();
    }
    // Notes lead the image, right after `.interp`: they are read-only,
    // allocated, and the loader reads some of them (`.note.gnu.property`)
    // before anything else. lld places them in the same spot. The build-id
    // note leads the notes so that a reader looking for it finds it in the
    // first page, and so that one `PT_NOTE` covers both regions.
    if sizes.build_id != 0 {
        let off = align_up(cursor, 4);
        layout.build_id = region_at(layout, off, sizes.build_id);
        cursor = layout.build_id.end();
    }
    cursor = place_output_after(ctx, rel, OutKind::Note, cursor, layout);
    cursor = place_output_after(ctx, rel, OutKind::Init, cursor, layout);
    cursor = place_output_after(ctx, rel, OutKind::Text, cursor, layout);
    cursor = place_output_after(ctx, rel, OutKind::Fini, cursor, layout);
    // `.plt` follows `.text`: both are executable and share the R+X segment.
    let plt_start = align_up(cursor, 16);
    if flags.has_plt {
        layout.plt = region_at(layout, plt_start, plt_section_size(layout));
    }
    // The cursor already sits past every region placed above: an unplaced
    // region reports offset zero, so reading its end alone would hand the
    // next region the ELF header's own bytes.
    cursor = place_output_after(
        ctx,
        rel,
        OutKind::Rodata,
        layout.plt.end().max(cursor),
        layout,
    );
    // `.eh_frame` follows `.rodata` (both are read-only, allocated). Keeping
    // it as its own placed region gives `.eh_frame_hdr` a contiguous address
    // range to index. `cursor` is the running high-water mark of everything
    // placed above -- `place_output_after` never returns less than it was
    // given -- and it starts at the end of the headers. Anchoring on it
    // rather than on the ends of `.rodata`, `.plt`, `.fini` and `.text`
    // matters for a link that contributes only unwind data: those four
    // regions are then all unplaced and report offset zero, which would put
    // `.eh_frame` on top of the ELF header.
    let mut rx_end =
        place_output_after(ctx, rel, OutKind::EhFrame, cursor, layout);
    // `.eh_frame_hdr` sits right after `.eh_frame` so the binary-search table
    // is close to the data it indexes; both ride in the same R+X segment.
    if flags.has_eh_frame_hdr {
        let off = align_up(rx_end, 4);
        layout.eh_frame_hdr = region_at(layout, off, eh_frame_hdr_size);
        rx_end = layout.eh_frame_hdr.end();
    }
    // Read-only dynamic tables extend the R+X segment after the unwind data.
    if flags.dynamic {
        rx_end = crate::dynamic::place_ro(layout, rx_end);
    }
    // `.rela.plt` sits with the read-only relocation tables.
    if flags.has_plt {
        let off = align_up(rx_end, 8);
        layout.rela_plt = region_at(layout, off, rela_plt_section_size(layout));
        rx_end = layout.rela_plt.end();
    }
    rx_end
}

/// A region at `offset`, whose virtual address is the load base plus that
/// offset.
fn region_at(layout: &Layout, offset: u64, size: u64) -> Region {
    Region {
        offset,
        vaddr: layout.base.wrapping_add(offset),
        size,
    }
}

/// Places one output section after `cursor`, at its own alignment, and
/// returns the offset the next region starts from: the section's end, or its
/// aligned start when the section is empty. Keeping the start in the answer
/// is what preserves the padding an empty section still consumes.
fn place_output_after(
    ctx: &Context<'_>,
    rel: &Relative,
    kind: OutKind,
    cursor: u64,
    layout: &mut Layout,
) -> u64 {
    let align = rel.align(kind);
    let start = align_up(cursor, align);
    place_output(ctx, rel, kind, start, align, layout);
    layout.region(region_sect(kind)).end().max(start)
}

/// Where a read-write region's bytes come from.
#[derive(Copy, Clone, Eq, PartialEq)]
enum RwSource {
    /// The `.dynamic` table, which the dynamic emitter owns and places.
    Dynamic,
    /// The GOT proper.
    Got,
    /// The PLT's GOT.
    GotPlt,
    /// An output section aggregated from input members.
    Out(OutKind),
}

/// The read-write segment in placement order, paired with the section-table
/// entry each region is emitted as.
///
/// One ordered list drives the whole pass, the way [`sect::TABLE`] drives the
/// section headers. That is what lets an absent region be skipped outright:
/// the alternative -- a hand-written chain of
/// `align_up(a.end().max(b.end()).max(cursor), align)` -- has to remember that
/// an unplaced region reports offset zero, and forgetting it once puts a
/// section at the ELF header's address.
///
/// The RELRO run comes first ([`sect::relro`]) because `PT_GNU_RELRO`
/// describes a single range; the regions the loader must keep writable follow
/// it. The list covers the same `Sect`s, in the same order, as the writable
/// half of [`sect::TABLE`]; the assertion in [`place_rw_segment`] pins the two
/// together.
const RW_ORDER: [(Sect, RwSource); 11] = [
    (Sect::Dynamic, RwSource::Dynamic),
    (Sect::Got, RwSource::Got),
    (Sect::DataRelRo, RwSource::Out(OutKind::DataRelRo)),
    (Sect::PreinitArray, RwSource::Out(OutKind::PreinitArray)),
    (Sect::InitArray, RwSource::Out(OutKind::InitArray)),
    (Sect::FiniArray, RwSource::Out(OutKind::FiniArray)),
    (Sect::GotPlt, RwSource::GotPlt),
    (Sect::Data, RwSource::Out(OutKind::Data)),
    (Sect::Tdata, RwSource::Out(OutKind::Tdata)),
    (Sect::Tbss, RwSource::Out(OutKind::Tbss)),
    (Sect::Bss, RwSource::Out(OutKind::Bss)),
];

/// Places the read-write segment by walking [`RW_ORDER`] with a running
/// cursor, then appends the copy-relocation slots to `.bss`.
///
/// The one thing the walk adds to a plain concatenation is the page boundary
/// that closes the RELRO run, and it is what makes the protection real.
/// glibc's `_dl_protect_relro` rounds *both* ends of the range down to a page
/// and mprotects what is left:
///
/// ```text
/// start = ALIGN_DOWN(l_addr + l_relro_addr, pagesize);
/// end   = ALIGN_DOWN(l_addr + l_relro_addr + l_relro_size, pagesize);
/// if (start != end && __mprotect(...) < 0) ...
/// ```
///
/// The run already starts on a page boundary (the read-write segment does), so
/// rounding the start down changes nothing. The end is the whole question: an
/// unrounded run shorter than a page collapses to `start == end` and mprotect
/// is skipped outright, and a longer one loses its final partial page. lld
/// buys the same boundary with a `.relro_padding` section
/// (ELF/SyntheticSections.cpp:2778) rounded up to `commonPageSize`
/// (ELF/LinkerScript.cpp:1285-1288). xold maps the image identically
/// (`vaddr == base + offset`), so the padding is real file bytes: up to
/// `PAGE - 1` of them, and none at all when the run has nothing after it.
fn place_rw_segment(
    ctx: &Context<'_>,
    rel: &Relative,
    rx_end: u64,
    layout: &mut Layout,
) {
    debug_assert!(
        sect::TABLE
            .iter()
            .filter(|(_, spec)| sect::writable(spec))
            .map(|(s, _)| *s)
            .eq(RW_ORDER.iter().map(|&(s, _)| s)),
        "read-write placement order matches the section table"
    );
    // The boundary between the two `PT_LOAD`s is aligned to the largest page
    // the target may be running under, not the smallest: a system page shared
    // by both segments is mapped twice and the writable mapping's permissions
    // win for the whole of it, which strips execute from the tail of `.text`.
    let mut cursor = align_up(rx_end, max_page_size(layout.target));
    let mut in_relro = false;
    // Recorded before the walk, because the walk reads it back through
    // `rw_align` to place `.tdata` -- the block's first byte, which `PT_TLS`
    // says is aligned to this.
    layout.tls_align = tls_align(rel);
    for (sect, source) in RW_ORDER {
        if !rw_present(rel, layout, source) {
            continue;
        }
        if in_relro && !sect::relro(sect) {
            cursor = align_up(cursor, PAGE);
        }
        in_relro = sect::relro(sect);
        let align = rw_align(rel, source);
        let start = align_up(cursor, align);
        cursor = place_rw_region(ctx, rel, source, start, align, layout);
    }
    // Copy-relocation slots extend `.bss`: each is a fixed-size, aligned
    // destination the loader fills from a shared dependency at load time.
    place_copy_slots(layout);
}

/// Whether a read-write region takes part in the placement.
///
/// A region with no bytes is skipped, so an empty `.init_array` neither pads
/// the cursor nor leaves `.fini_array` anywhere unexpected. `.bss` is the one
/// exception: [`place_copy_slots`] appends to it after the walk, so it needs an
/// address even when no input contributed a byte to it.
fn rw_present(rel: &Relative, layout: &Layout, source: RwSource) -> bool {
    if source == RwSource::Out(OutKind::Bss) && !layout.copy_slots.is_empty() {
        return true;
    }
    rw_size(rel, layout, source) != 0
}

/// The byte size a read-write region will take. Answerable before the walk
/// runs, which is what lets the program header count be decided from the same
/// list that later places the regions.
fn rw_size(rel: &Relative, layout: &Layout, source: RwSource) -> u64 {
    match source {
        RwSource::Dynamic => {
            layout.dynamic.as_ref().map_or(0, |p| p.sizes.dynamic)
        }
        RwSource::Got => u64::try_from(layout.got_values.len())
            .unwrap_or(0)
            .saturating_mul(GOT_ENTRY),
        RwSource::GotPlt => {
            if layout.plt_keys.is_empty() {
                0
            } else {
                got_plt_section_size(layout)
            }
        }
        RwSource::Out(kind) => rel.size(kind),
    }
}

/// The alignment `PT_TLS` declares: the pair's, not either section's.
///
/// A loader lays the thread block down by this number, so it is also the
/// alignment the block's first byte has to sit at.
fn tls_align(rel: &Relative) -> u64 {
    rel.align(OutKind::Tdata).max(rel.align(OutKind::Tbss))
}

/// The alignment a read-write region is placed at.
///
/// The synthetic tables hold 8-byte words. An output section takes the largest
/// alignment across its members, with two exceptions.
///
/// `.bss` is floored at 16. Its recorded alignment already covers the common
/// symbols, which [`measure_commons`] folds in for the reason given there.
///
/// `.tdata` and `.tbss` both take the pair's alignment, because `PT_TLS`
/// declares one for both and its `p_vaddr` has to be congruent to it. `.tdata`
/// at align 1 beside `.tbss` at align 64 left `p_vaddr % 64 != 0`, and a
/// loader that places the block by `roundup(memsz, align)` without
/// compensating for the first byte then reads every thread-local at the wrong
/// offset and breaks the declared alignment besides. lld aligns the segment
/// start for this case in `fixSectionAlignments`, citing glibc PR/24606,
/// FreeBSD's rtld and musl before 1.1.23.
fn rw_align(rel: &Relative, source: RwSource) -> u64 {
    match source {
        RwSource::Dynamic => DYNAMIC_ALIGN,
        RwSource::Got | RwSource::GotPlt => GOT_ENTRY,
        RwSource::Out(OutKind::Bss) => rel.align(OutKind::Bss).max(16),
        RwSource::Out(OutKind::Tdata | OutKind::Tbss) => tls_align(rel),
        RwSource::Out(kind) => rel.align(kind),
    }
}

/// Places one read-write region at `start`, returning one past its last byte.
fn place_rw_region(
    ctx: &Context<'_>,
    rel: &Relative,
    source: RwSource,
    start: u64,
    align: u64,
    layout: &mut Layout,
) -> u64 {
    match source {
        RwSource::Dynamic => crate::dynamic::place_rw(layout, start),
        RwSource::Got => {
            place_got(start, layout);
            layout.got.end()
        }
        RwSource::GotPlt => {
            let size = got_plt_section_size(layout);
            layout.got_plt = region_at(layout, start, size);
            layout.got_plt.end()
        }
        RwSource::Out(OutKind::Bss) => {
            place_bss(ctx, rel, start, align, layout);
            layout.bss.end()
        }
        // `.tbss` is a template, not storage: the loader copies it into each
        // thread's block and no address in the image is ever read through it.
        // Advancing the cursor past it pushed `.bss`, `_end` and the
        // read-write `p_memsz` up by its whole size, so a
        // `__thread char big[1 << 20]` cost a megabyte of address space
        // nothing occupies. Its members still get addresses -- `PT_TLS` needs
        // the extent, and `tls_block` reads it from the assigned range -- but
        // whatever follows starts where `.tbss` did, which is what lld, mold
        // and GNU ld all do.
        RwSource::Out(OutKind::Tbss) => {
            place_output(ctx, rel, OutKind::Tbss, start, align, layout);
            start
        }
        RwSource::Out(kind) => {
            place_output(ctx, rel, kind, start, align, layout);
            start.wrapping_add(rel.size(kind))
        }
    }
}

/// Places one `SHF_ALLOC` output section (text, rodata, or data).
///
/// `align` is the alignment the caller placed `start` at, recorded so the
/// section header reports the alignment the cursor honoured rather than a
/// constant that may be stricter than it.
fn place_output(
    ctx: &Context<'_>,
    rel: &Relative,
    kind: OutKind,
    start: u64,
    align: u64,
    layout: &mut Layout,
) {
    let Some(out) = ctx.outputs.section(kind) else {
        return;
    };
    let base_vaddr = layout.base.wrapping_add(start);
    for m in &out.members {
        let vaddr =
            base_vaddr.wrapping_add(rel.member_offset(m.file, m.section));
        set_section(layout, m.file, m.section, vaddr);
    }
    set_region(
        layout,
        kind,
        align,
        Region {
            offset: start,
            vaddr: base_vaddr,
            size: rel.size(kind),
        },
    );
}

/// Records a region into the matching field of the layout, with the alignment
/// it was placed at.
fn set_region(layout: &mut Layout, kind: OutKind, align: u64, region: Region) {
    layout.set_measured_align(region_sect(kind), align);
    match kind {
        OutKind::Text => layout.text = region,
        OutKind::Rodata => layout.rodata = region,
        OutKind::Note => layout.note = region,
        OutKind::EhFrame => layout.eh_frame = region,
        OutKind::DataRelRo => layout.data_rel_ro = region,
        OutKind::Data => layout.data = region,
        OutKind::Bss => layout.bss = region,
        OutKind::Tdata => layout.tdata = region,
        OutKind::Tbss => layout.tbss = region,
        OutKind::PreinitArray => layout.preinit_array = region,
        OutKind::InitArray => layout.init_array = region,
        OutKind::FiniArray => layout.fini_array = region,
        OutKind::Init => layout.init = region,
        OutKind::Fini => layout.fini = region,
    }
}

/// The section-table entry an aggregated output section is emitted as.
///
/// The two enumerations are separate on purpose -- [`OutKind`] names the
/// buckets input sections are gathered into, [`Sect`] names every region of the
/// image, synthetic ones included -- and this is the one place they meet.
const fn region_sect(kind: OutKind) -> Sect {
    match kind {
        OutKind::Text => Sect::Text,
        OutKind::Rodata => Sect::Rodata,
        OutKind::Note => Sect::Note,
        OutKind::EhFrame => Sect::EhFrame,
        OutKind::DataRelRo => Sect::DataRelRo,
        OutKind::Data => Sect::Data,
        OutKind::Bss => Sect::Bss,
        OutKind::Tdata => Sect::Tdata,
        OutKind::Tbss => Sect::Tbss,
        OutKind::PreinitArray => Sect::PreinitArray,
        OutKind::InitArray => Sect::InitArray,
        OutKind::FiniArray => Sect::FiniArray,
        OutKind::Init => Sect::Init,
        OutKind::Fini => Sect::Fini,
    }
}

/// Places the GOT region at `start`.
fn place_got(start: u64, layout: &mut Layout) {
    let size = u64::try_from(layout.got_values.len())
        .unwrap_or(0)
        .saturating_mul(GOT_ENTRY);
    layout.got = region_at(layout, start, size);
}

/// Places `.bss` (members plus common storage) at `start`; it occupies no file
/// space, only memory.
fn place_bss(
    ctx: &Context<'_>,
    rel: &Relative,
    start: u64,
    align: u64,
    layout: &mut Layout,
) {
    let base_vaddr = layout.base.wrapping_add(start);
    if let Some(out) = ctx.outputs.section(OutKind::Bss) {
        for m in &out.members {
            let vaddr =
                base_vaddr.wrapping_add(rel.member_offset(m.file, m.section));
            set_section(layout, m.file, m.section, vaddr);
        }
    }
    layout.set_measured_align(Sect::Bss, align);
    layout.bss = region_at(layout, start, rel.size(OutKind::Bss));
}

/// Appends one `.bss` slot per copy-relocation candidate after the section's
/// existing content, recording each slot's address on the candidate. The
/// slots are `SHT_NOBITS` storage (no file bytes): the loader writes the
/// copied initial values, so only the in-memory size grows.
fn place_copy_slots(layout: &mut Layout) {
    if layout.copy_slots.is_empty() {
        return;
    }
    let start = layout.bss.end();
    let mut cursor = start;
    let mut max_align = 1u64;
    for slot in &mut layout.copy_slots {
        let align = slot.align.max(1);
        cursor = align_up(cursor, align);
        slot.addr = layout.base.wrapping_add(cursor);
        cursor = cursor.saturating_add(slot.size);
        if align > max_align {
            max_align = align;
        }
    }
    let grown = align_up(cursor, max_align.max(16));
    layout.bss.size = grown.wrapping_sub(layout.bss.offset);
}

/// Records the placed address and (later) shndx of an input section.
fn set_section(layout: &mut Layout, file: usize, section: u16, vaddr: u64) {
    if let Some(f) = layout.sec_vaddr.get_mut(file)
        && let Some(slot) = f.get_mut(usize::from(section))
    {
        *slot = vaddr;
    }
}

/// Measures the placed bounds of every `__start_`/`__stop_` run, so
/// [`crate::defsym`] can give the encapsulation symbols an address.
///
/// The members of a run were gathered into one contiguous block within their
/// output section ([`Context::group_start_stop`]), so the bounds are the first
/// member's address and one past the last member's. Should the same name reach
/// two different output sections -- possible only when two inputs give it
/// different flags -- the run is bounded within the section its first member
/// landed in, rather than being stretched over the unrelated content between
/// the two.
pub(super) fn measure_runs(ctx: &Context<'_>, layout: &mut Layout) {
    let count = ctx.start_stop.len();
    if count == 0 {
        return;
    }
    let mut bounds = vec![Bounds::default(); count];
    let mut home: Vec<Option<OutKind>> = vec![None; count];
    for out in ctx.outputs.iter() {
        for m in &out.members {
            let Some(run) = ctx.start_stop.run_of_section(m.file, m.section)
            else {
                continue;
            };
            let (Some(slot), Some(seen)) =
                (bounds.get_mut(run), home.get_mut(run))
            else {
                continue;
            };
            let Some(vaddr) = layout.section_vaddr(m.file, m.section) else {
                continue;
            };
            let end = vaddr.wrapping_add(m.size);
            match *seen {
                None => {
                    *seen = Some(out.kind);
                    slot.start = vaddr;
                    slot.stop = end;
                }
                Some(kind) if kind == out.kind => {
                    slot.stop = slot.stop.max(end);
                }
                Some(_) => {}
            }
        }
    }
    layout.start_stop = bounds;
}

/// Numbers the output section headers, skipping empty regions, so a region's
/// index matches the position the writer emits its header at. Both passes walk
/// [`sect::TABLE`], so the two orders cannot drift apart.
pub(super) fn assign_shndx(ctx: &Context<'_>, layout: &mut Layout) {
    let mut next = 1u16;
    for (sect, _spec) in sect::TABLE {
        if layout.region(sect).size == 0 {
            continue;
        }
        layout.set_shndx(sect, next);
        next = next.saturating_add(1);
    }
    // Stamp each input section with the shndx of the region that holds it.
    stamp_shndx(ctx, layout);
}

/// Writes the assigned region shndx back to every placed input section.
///
/// The answer comes from the output section the member belongs to, which is
/// the fact placement already recorded. Deriving it from the member's address
/// instead asked which region *covers* that address, over a half-open range --
/// and a zero-size member placed at its region's exact end is covered by
/// nothing. Its defined globals were then published `SHN_UNDEF`, which is the
/// shape an assembler end-marker label takes: `_data_end:` at the tail of
/// `.data` with nothing after it.
fn stamp_shndx(ctx: &Context<'_>, layout: &mut Layout) {
    for out in ctx.outputs.iter() {
        let shndx = layout.shndx(region_sect(out.kind));
        for m in &out.members {
            if let Some(f) = layout.sec_shndx.get_mut(m.file)
                && let Some(slot) = f.get_mut(usize::from(m.section))
            {
                *slot = shndx;
            }
        }
    }
}
