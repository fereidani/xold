//! Program-header (Phdr) emission, split out of the main writer.
//!
//! The segment layout mirrors [`crate::layout::place_regions`]: a read+execute
//! `PT_LOAD` (extended with `.interp`, `.plt`, the read-only dynamic tables
//! and `.rela.plt` in a dynamic link), an optional `PT_INTERP`, an optional
//! read+write `PT_LOAD` (opened by `.dynamic`, holding the RELRO run and then
//! `.got.plt`/`.data`), an optional `PT_TLS`, the dynamic-link `PT_DYNAMIC`,
//! an optional `PT_GNU_RELRO` over the run at the head of the read-write
//! segment, and the `PT_GNU_STACK` marker.
//!
//! Every optional segment here is counted a second time, before placement, by
//! `place_regions`: the count sizes the header block every region is then
//! placed after, so the two have to agree. Both ask the same question of the
//! same ordered list ([`sect::TABLE`], and [`sect::relro`] over it), which is
//! what keeps them from drifting into a truncated or over-long table.

// Private helpers and sizes owned by the parent writer module: visible
// here because a child module can reach its parent's private items.
use super::{DYNAMIC_ENTRY_SIZE, EHDR_SIZE, PHDR_SIZE};
// Re-exported ELF segment/flag constants for brevity.
use crate::elf::constants::{
    PF_R, PF_W, PF_X, PT_DYNAMIC, PT_GNU_EH_FRAME, PT_GNU_RELRO, PT_GNU_STACK,
    PT_INTERP, PT_LOAD, PT_NOTE, PT_PHDR, PT_TLS,
};
use crate::{
    dynamic::LinkMode,
    elf::Phdr64,
    endian::{U32, U64},
    layout::{Layout, Sect, max_page_size, sect},
    util::{PAGE, align_up, write_pod},
};

/// Writes the program headers right after the ELF header. See the module docs
/// for the segment plan.
pub(super) fn write_phdrs(image: &mut [u8], layout: &Layout, base: u64) {
    let mut phdrs: Vec<Phdr64> =
        Vec::with_capacity(usize::from(layout.phdr_count()));
    // `PT_PHDR` then `PT_INTERP`, both before any `PT_LOAD`: the gABI
    // requires the interpreter entry to precede every loadable one, and
    // requires `PT_PHDR` to precede any entry it is present alongside. Linux
    // scans the whole table so the old order loaded anyway, but `eu-elflint`
    // and stricter loaders read the requirement as written, and conforming
    // costs nothing -- the entries carry their own offsets, so only their
    // position in the table changes.
    phdrs.extend(phdr_phdr(layout, base));
    phdrs.extend(interp_phdr(layout));
    let page = max_page_size(layout.target);
    phdrs.push(rx_load(rx_segment_end(layout), base, page));
    phdrs.extend(note_phdr(layout));
    phdrs.extend(rw_load(layout, base, page));
    phdrs.extend(tls_phdr(layout));
    phdrs.extend(dynamic_phdr(layout, base));
    phdrs.extend(relro_phdr(layout, base));
    phdrs.extend(eh_frame_hdr_phdr(layout, base));
    phdrs.push(gnu_stack(layout.stack_size));
    let mut off = EHDR_SIZE;
    for p in &phdrs {
        write_pod(image, off, p);
        off += PHDR_SIZE;
    }
}

/// Builds one program header, using the same value for physical and virtual
/// addresses as every header in this image does.
fn new_phdr(
    p_type: u32,
    flags: u32,
    offset: u64,
    vaddr: u64,
    filesz: u64,
    memsz: u64,
    align: u64,
) -> Phdr64 {
    Phdr64 {
        p_type: U32::new(p_type),
        p_flags: U32::new(flags),
        p_offset: U64::new(offset),
        p_vaddr: U64::new(vaddr),
        p_paddr: U64::new(vaddr),
        p_filesz: U64::new(filesz),
        p_memsz: U64::new(memsz),
        p_align: U64::new(align),
    }
}

/// The file offset that opens the read-write segment: the first writable
/// region that holds bytes, in placement order. That is `.dynamic` in a
/// dynamic link and `.got` in most static ones, but naming either here would
/// be a second copy of the placement order; reading it off [`sect::TABLE`] is
/// how the answer stays right when the order changes. An absent region
/// defaults to offset zero, so only regions that hold bytes are considered.
fn rw_segment_start(layout: &Layout) -> u64 {
    writable_sects()
        .map(|s| layout.region(s))
        .find(|r| r.size != 0)
        .map_or(0, |r| r.offset)
}

/// Every writable region, in placement order.
fn writable_sects() -> impl Iterator<Item = Sect> {
    sect::TABLE
        .iter()
        .filter(|(_, spec)| sect::writable(spec))
        .map(|(s, _)| *s)
}

/// One past the end of the read+execute `PT_LOAD` segment: the headers plus
/// every region the section table marks read-only or executable (`.interp`,
/// `.text`, `.plt`, `.rodata`, the unwind pair, the read-only dynamic tables
/// and `.rela.plt`).
fn rx_segment_end(layout: &Layout) -> u64 {
    let headers = EHDR_SIZE + u64::from(layout.phdr_count()) * PHDR_SIZE;
    sect::TABLE
        .iter()
        .filter(|(_, spec)| !sect::writable(spec))
        .map(|(s, _)| layout.region(*s).end())
        .fold(headers, u64::max)
}

/// `PT_PHDR` for a position-independent dynamic executable, so the loader can
/// recover its load base. A fixed-base (`ET_EXEC`) executable does not need it.
fn phdr_phdr(layout: &Layout, base: u64) -> Option<Phdr64> {
    if layout.mode() != LinkMode::DynExec || !layout.is_pie() {
        return None;
    }
    let size = u64::from(layout.phdr_count()) * PHDR_SIZE;
    Some(new_phdr(
        PT_PHDR,
        PF_R,
        EHDR_SIZE,
        base.wrapping_add(EHDR_SIZE),
        size,
        size,
        8,
    ))
}

/// The read+execute `PT_LOAD`, covering offset zero through `rx_end`.
fn rx_load(rx_end: u64, base: u64, page: u64) -> Phdr64 {
    new_phdr(PT_LOAD, PF_R | PF_X, 0, base, rx_end, rx_end, page)
}

/// `PT_INTERP` for a dynamic executable with an interpreter path.
fn interp_phdr(layout: &Layout) -> Option<Phdr64> {
    let interp = layout.region(Sect::Interp);
    layout.has_interp().then(|| {
        new_phdr(
            PT_INTERP,
            PF_R,
            interp.offset,
            interp.vaddr,
            interp.size,
            interp.size,
            1,
        )
    })
}

/// `PT_NOTE` over the image's allocated notes.
///
/// A note is found by segment, not by section name: a reader walks
/// `PT_NOTE`'s extent and decodes each self-describing record. Without the
/// segment the bytes are unreachable, whatever the section they sit in --
/// which is why the notes get their own `SHT_NOTE` region rather than folding
/// into `.rodata`. lld emits one such header per contiguous run of notes;
/// there is one run here, because the notes are gathered into one region.
fn note_phdr(layout: &Layout) -> Option<Phdr64> {
    // One segment covers both note regions. They are placed adjacently, the
    // synthesised build-id note first, so the span between them holds nothing
    // but the alignment padding a note reader skips anyway.
    let build_id = layout.region(Sect::BuildId);
    let note = layout.region(Sect::Note);
    let start = [build_id, note]
        .into_iter()
        .filter(|r| r.size != 0)
        .map(|r| r.offset)
        .min()?;
    let end = build_id.end().max(note.end());
    let size = end.saturating_sub(start);
    (size != 0).then(|| {
        new_phdr(
            PT_NOTE,
            PF_R,
            start,
            layout.base().wrapping_add(start),
            size,
            size,
            4,
        )
    })
}

/// The read+write `PT_LOAD`, opened by `.dynamic` (or `.got`), covering the
/// RELRO run, `.got.plt`, `.data` and the TLS/bss tail.
fn rw_load(layout: &Layout, base: u64, page: u64) -> Option<Phdr64> {
    if !writable_sects().any(|s| layout.region(s).size != 0) {
        return None;
    }
    let rw_off = rw_segment_start(layout);
    // The file extent stops at the last file-backed region; `.bss`/`.tbss`
    // extend the memory extent past it.
    //
    // Both folds start at the segment's own offset rather than at zero. A
    // segment whose only content is `SHT_NOBITS` -- a link with a `.bss` and
    // no `.data`, no GOT and no dynamic table -- has no file-backed region to
    // measure, and folding from zero would then make `p_filesz` the
    // underflowed `0 - rw_off`. Starting from `rw_off` reports the empty file
    // extent such a segment actually has.
    let file_end = writable_sects()
        .filter(|s| sect::file_backed(*s))
        .map(|s| layout.region(s).end())
        .fold(rw_off, u64::max);
    // `.tbss` is excluded: it is the template a loader copies into each
    // thread's block, and no address in the image is read through it, so its
    // extent is not memory this segment has to reserve. The other writable
    // regions overlap it, which is what makes its size free rather than
    // merely unadvanced. lld, mold and GNU ld all leave it out.
    let mem_end = writable_sects()
        .filter(|s| *s != Sect::Tbss)
        .map(|s| layout.region(s).end())
        .fold(file_end, u64::max);
    // The RELRO header rounds its end up to a page, because that is what
    // glibc's `_dl_protect_relro` needs to protect the run's final partial
    // page at all. Those bytes have to be inside a `PT_LOAD` or they describe
    // memory nothing maps: with no writable region after the run -- code plus
    // `.dynamic` and `.got` and nothing else -- RELRO otherwise ran up to a
    // page past the segment covering it. Placement already pads to a page when
    // a non-RELRO region follows, so this max only bites when the run is the
    // segment's tail, and it grows `p_memsz` alone: the extra is zero memory,
    // exactly like `.bss`. lld reaches the same extent with a
    // `.relro_padding` section inside both segments.
    let mem_end = relro_extent(layout)
        .map_or(mem_end, |(_, end)| mem_end.max(align_up(end, PAGE)));
    Some(new_phdr(
        PT_LOAD,
        PF_R | PF_W,
        rw_off,
        base.wrapping_add(rw_off),
        file_end.wrapping_sub(rw_off),
        mem_end.wrapping_sub(rw_off),
        page,
    ))
}

/// `PT_TLS` when a static TLS block exists.
fn tls_phdr(layout: &Layout) -> Option<Phdr64> {
    let block = layout.tls_block()?;
    Some(new_phdr(
        PT_TLS,
        PF_R,
        block.offset,
        block.vaddr,
        block.filesz,
        block.memsz,
        block.align.max(1),
    ))
}

/// `PT_DYNAMIC` for a dynamic link.
fn dynamic_phdr(layout: &Layout, base: u64) -> Option<Phdr64> {
    let plan = layout.dynamic.as_ref()?;
    let dyn_off = plan.regions.dynamic.offset;
    let dyn_size =
        u64::try_from(plan.dynamic.len()).unwrap_or(0) * DYNAMIC_ENTRY_SIZE;
    Some(new_phdr(
        PT_DYNAMIC,
        PF_R | PF_W,
        dyn_off,
        base.wrapping_add(dyn_off),
        dyn_size,
        dyn_size,
        8,
    ))
}

/// `PT_GNU_RELRO` over the run at the head of the read-write segment: the
/// regions [`sect::relro`] names, which the loader may re-protect read-only
/// once it has applied this image's relocations.
///
/// `p_memsz` runs to the page boundary above the run's last byte, because
/// glibc reads `p_memsz` into `l_relro_size` and `_dl_protect_relro` mprotects
/// only whole pages: an unrounded size would leave the run's final page
/// writable. `p_filesz` stops at the run's real content, as lld's does -- its
/// `.relro_padding` is `SHT_NOBITS`, so the padding counts towards `p_memsz`
/// alone. `p_align` is 1, matching lld's `relRo->p_align = 1`
/// (ELF/Writer.cpp:2372): the segment is not loaded, it only names a range.
fn relro_phdr(layout: &Layout, base: u64) -> Option<Phdr64> {
    let (start, end) = relro_extent(layout)?;
    Some(new_phdr(
        PT_GNU_RELRO,
        PF_R,
        start,
        base.wrapping_add(start),
        end.wrapping_sub(start),
        align_up(end, PAGE).wrapping_sub(start),
        1,
    ))
}

/// The RELRO run's file offsets: the first protected region's start and one
/// past the last protected region's end, or `None` when the image has no
/// protected region. Placement keeps the run contiguous, so the pair is the
/// whole range.
fn relro_extent(layout: &Layout) -> Option<(u64, u64)> {
    let mut extent: Option<(u64, u64)> = None;
    for s in writable_sects().filter(|s| sect::relro(*s)) {
        let r = layout.region(s);
        if r.size == 0 {
            continue;
        }
        extent = Some(match extent {
            None => (r.offset, r.end()),
            Some((start, end)) => (start.min(r.offset), end.max(r.end())),
        });
    }
    extent
}

/// `PT_GNU_EH_FRAME` covering `.eh_frame_hdr`, the binary-search table the
/// runtime unwinder (libgcc/glibc) walks to locate an FDE for a thrown PC.
/// Emitted only when `.eh_frame_hdr` exists.
fn eh_frame_hdr_phdr(layout: &Layout, base: u64) -> Option<Phdr64> {
    let r = layout.region(Sect::EhFrameHdr);
    if r.size == 0 {
        return None;
    }
    Some(new_phdr(
        PT_GNU_EH_FRAME,
        PF_R,
        r.offset,
        base.wrapping_add(r.offset),
        r.size,
        r.size,
        4,
    ))
}

/// `PT_GNU_STACK` marker: a non-executable stack of `size` bytes.
///
/// The segment describes no bytes; the kernel reads its flags alone, and only
/// `PF_X` changes what it does -- with the bit set the stack is mapped
/// executable, without it it is not. The other two are still part of the
/// answer, because the segment stands for the stack's permissions and a stack
/// is readable as well as writable. lld spells all three out in its
/// `PT_GNU_STACK` block and GNU ld emits `RW` too; claiming write-without-read
/// would describe a stack no loader produces.
///
/// `p_align` is left at `0x10`, which is what GNU ld emits here; lld emits 0
/// (its `PhdrEntry` constructor aligns only `PT_LOAD`). The field has no
/// meaning for a segment with no content and no address, both linkers are
/// self-consistent, and readers ignore it -- so this stays as it is rather
/// than churning every output for a field nothing reads.
/// `p_memsz` carries what `-z stack-size=N` asked for. Zero, the value every
/// other link emits, leaves the size to the system.
fn gnu_stack(size: u64) -> Phdr64 {
    new_phdr(PT_GNU_STACK, PF_R | PF_W, 0, 0, 0, size, 0x10)
}
