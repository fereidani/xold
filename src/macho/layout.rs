//! Mach-O segment and section placement for a static executable.
//!
//! Sections are grouped into `LC_SEGMENT_64` segments by their segment name
//! (`__TEXT`, `__DATA`); `__LINKEDIT` holds the symbol and string tables. The
//! `__TEXT` segment is mapped at the conventional base (`0x100000000` for both
//! `x86_64` and `arm64`) starting at file offset zero, so the header and load
//! commands lead `__TEXT` and the first real section follows them. Every
//! segment boundary is page-aligned (`0x4000`), keeping `vmaddr % page` equal
//! to `fileoff % page` so each segment is mmap-able.
//!
//! Code, read-only data, data/bss and exception payloads are linked.
//! `__LD,__compact_unwind` is consumed to synthesize `__TEXT,__unwind_info`;
//! `__eh_frame` and `__gcc_except_tab` remain in `__TEXT`. DWARF debug and
//! input dynamic-linker sections are deferred and dropped here.

use crate::{
    error::{Error, Result},
    macho::{
        LinkOptions, MachOFile, MachSection,
        constants::{
            ARM_THREAD_STATE64_COUNT, S_ATTR_PURE_INSTRUCTIONS,
            S_ATTR_SOME_INSTRUCTIONS, S_NON_LAZY_SYMBOL_POINTERS,
            S_SYMBOL_STUBS, S_THREAD_LOCAL_ZEROFILL, S_ZEROFILL, SECTION_TYPE,
            VM_PROT_EXECUTE, VM_PROT_READ, X86_THREAD_STATE64_COUNT,
        },
        live::LiveSections,
        reloc::MachoTarget,
        symtab::CommonSym,
    },
    util::{align_up, pad_name, trim_nul},
};

/// Conventional base of the `__TEXT` segment for both supported architectures.
pub const TEXT_BASE: u64 = 0x1_0000_0000;
/// Segment file/virtual alignment (`0x4000` page, used by arm64 and modern
/// `x86_64` darwin).
pub const FILE_PAGE: u64 = 0x4000;
/// On-disk size of `mach_header_64`.
pub const HEADER_SIZE: u64 = 32;
/// Free bytes between the load commands and the first section. `codesign`
/// inserts `LC_CODE_SIGNATURE` after the link, growing `sizeofcmds` by 16;
/// without header padding that command overwrites the first instructions.
const HEADER_PAD: u64 = 32;

/// One member input section selected for linking.
#[derive(Clone, Copy)]
pub struct Member {
    pub file: usize,
    /// 1-based section ordinal in the input (matches `n_sect`).
    pub section: u32,
    pub size: u64,
    pub align: u32,
    /// Byte offset of this member within its output section (honouring
    /// per-member alignment), filled by [`layout_members`].
    pub offset: u64,
}

/// An output section aggregating members, with its placement.
pub struct OutSection {
    pub segname: [u8; 16],
    pub sectname: [u8; 16],
    pub flags: u32,
    pub align: u32,
    pub members: Vec<Member>,
    pub addr: u64,
    pub offset: u64,
    pub size: u64,
    /// `S_ZEROFILL`: occupies virtual space but no file bytes.
    pub zerofill: bool,
    /// Indirect-symbol-table start index for stubs and pointer sections.
    pub reserved1: u32,
    /// Fixed stub size for `S_SYMBOL_STUBS`.
    pub reserved2: u32,
}

/// A placed output segment and its sections (file-backed first, zero-fill
/// last).
pub struct OutSegment {
    pub name: [u8; 16],
    pub vmaddr: u64,
    pub vmsize: u64,
    pub fileoff: u64,
    pub filesize: u64,
    pub maxprot: u32,
    pub initprot: u32,
    pub sections: Vec<OutSection>,
}

/// The placed `__got` section: its virtual address and file offset, and the
/// slot count (one per external symbol). `None` when no symbol needs a GOT
/// entry, in which case no `__got` section is emitted.
#[derive(Clone, Copy)]
pub struct GotLayout {
    /// Virtual address of the first slot.
    pub addr: u64,
    /// File offset of the first slot.
    pub offset: u64,
    /// Number of 8-byte slots.
    pub count: u32,
}

/// The placed synthetic arm64 `__stubs` section.
#[derive(Clone, Copy)]
pub struct StubLayout {
    pub addr: u64,
    pub offset: u64,
    pub count: u32,
}

/// The laid-out Mach-O image: segments, the `__LINKEDIT` extent, the load
/// command table size, and per-input-section virtual addresses.
pub struct MachLayout {
    pub text: OutSegment,
    pub data: Option<OutSegment>,
    pub linkedit_vmaddr: u64,
    pub linkedit_fileoff: u64,
    pub linkedit_filesize: u64,
    pub sizeofcmds: u32,
    pub entry_off: u64,
    /// Per file, per 1-based input section ordinal: the placed virtual address
    /// (0 if the section was not linked).
    pub sec_vaddr: Vec<Vec<u64>>,
    /// The placed `__got` section, if any GOT entries were allocated.
    pub got: Option<GotLayout>,
    pub stubs: Option<StubLayout>,
    /// The virtual address of each merged common symbol, parallel to the
    /// `commons` list handed to [`build`].
    pub common_addr: Vec<u64>,
}

/// The content the layout synthesises rather than copies from an input.
///
/// The four travel together because each one decides whether a section
/// exists at all, and the layout has to know all of them before it places
/// anything.
#[derive(Clone, Copy, Default)]
pub struct Synthetic<'a> {
    /// `__got` slots to allocate, for external symbols referenced through a
    /// GOT relocation. Zero emits no `__got` section.
    pub got: u32,
    /// `__stubs` entries, one per imported function called directly.
    pub stubs: u32,
    /// `__unwind_info` entries. Zero emits no compact-unwind index.
    pub unwind: u32,
    /// The merged tentative definitions, which a non-empty list turns into a
    /// synthetic `__common` zero-fill section.
    pub commons: &'a [CommonSym<'a>],
}

/// Builds the layout for `inputs`.
///
/// `linkedit_size` is the `__LINKEDIT` content size (symtab + strtab),
/// precomputed. `entry` names the entry symbol.
pub fn build(
    inputs: &[MachOFile<'_>],
    linkedit_size: u64,
    entry: &[u8],
    synthetic: &Synthetic<'_>,
    target: MachoTarget,
    options: &LinkOptions<'_>,
    live: &LiveSections,
) -> Result<MachLayout> {
    let mut text_secs = Vec::new();
    let mut data_secs = Vec::new();
    collect_sections(inputs, live, &mut text_secs, &mut data_secs);
    if synthetic.unwind > 0 {
        inject_unwind_info(&mut text_secs, synthetic.unwind);
    }
    if synthetic.stubs > 0 {
        inject_stubs(&mut text_secs, synthetic.stubs);
    }
    if synthetic.got > 0 {
        inject_got_section(&mut data_secs, synthetic.got, synthetic.stubs);
    }
    if !synthetic.commons.is_empty() {
        inject_common_section(&mut data_secs, synthetic.commons);
    }

    let nsects_text = u32::try_from(text_secs.len())
        .map_err(|_| Error::OutOfRange("text section count"))?;
    let nsects_data = u32::try_from(data_secs.len())
        .map_err(|_| Error::OutOfRange("data section count"))?;
    let sizeofcmds = commands_size(nsects_text, nsects_data, target, options);

    let text = place_text_segment(text_secs, sizeofcmds);
    let data = place_data_segment(data_secs, &text);
    let (linkedit_vmaddr, linkedit_fileoff) =
        linkedit_extents(&text, data.as_ref());

    let mut sec_vaddr = empty_section_map(inputs);
    stamp(&text, &mut sec_vaddr);
    if let Some(d) = data.as_ref() {
        stamp(d, &mut sec_vaddr);
    }

    let entry_off = if options.dylib {
        0
    } else {
        entry_offset(inputs, &sec_vaddr, entry)?
    };
    let got = find_got(data.as_ref());
    let stubs = find_stubs(&text);
    let common_addr = common_addresses(data.as_ref(), synthetic.commons);
    Ok(MachLayout {
        text,
        data,
        linkedit_vmaddr,
        linkedit_fileoff,
        linkedit_filesize: linkedit_size,
        sizeofcmds,
        entry_off,
        sec_vaddr,
        got,
        stubs,
        common_addr,
    })
}

/// Inserts a regular-page compact-unwind index immediately before
/// `__eh_frame`. The writer fills it after section relocation, once final
/// function and FDE offsets are known.
fn inject_unwind_info(text_secs: &mut Vec<OutSection>, entries: u32) {
    let size = crate::macho::unwind::section_size(entries);
    let section = OutSection {
        segname: pad_name(b"__TEXT"),
        sectname: pad_name(b"__unwind_info"),
        flags: 0,
        align: 2,
        members: Vec::new(),
        addr: 0,
        offset: 0,
        size,
        zerofill: false,
        reserved1: 0,
        reserved2: 0,
    };
    let at = text_secs
        .iter()
        .position(|candidate| trim_nul(&candidate.sectname) == b"__eh_frame")
        .unwrap_or(text_secs.len());
    text_secs.insert(at, section);
}

/// Prepends the synthetic `__got` section to `__DATA`. It is a non-lazy symbol
/// pointer table (`S_NON_LAZY_SYMBOL_POINTERS`), 8-byte aligned, one slot per
/// referenced symbol. It carries no input members: the writer fills its bytes
/// directly from the resolved symbol addresses.
fn inject_got_section(
    data_secs: &mut Vec<OutSection>,
    got_count: u32,
    stub_count: u32,
) {
    let size = u64::from(got_count).checked_mul(8).unwrap_or(0);
    data_secs.insert(
        0,
        OutSection {
            segname: pad_name(b"__DATA"),
            sectname: pad_name(b"__got"),
            flags: S_NON_LAZY_SYMBOL_POINTERS,
            align: 3,
            members: Vec::new(),
            addr: 0,
            offset: 0,
            size,
            zerofill: false,
            reserved1: stub_count,
            reserved2: 0,
        },
    );
}

/// Appends one 12-byte arm64 stub per imported function.
fn inject_stubs(text_secs: &mut Vec<OutSection>, stub_count: u32) {
    text_secs.push(OutSection {
        segname: pad_name(b"__TEXT"),
        sectname: pad_name(b"__stubs"),
        flags: S_SYMBOL_STUBS
            | S_ATTR_PURE_INSTRUCTIONS
            | S_ATTR_SOME_INSTRUCTIONS,
        align: 2,
        members: Vec::new(),
        addr: 0,
        offset: 0,
        size: u64::from(stub_count).saturating_mul(12),
        zerofill: false,
        reserved1: 0,
        reserved2: 12,
    });
}

/// Locates the placed `__got` section in `__DATA` and reports its address,
/// offset and slot count, or `None` when no `__got` was injected.
fn find_got(data: Option<&OutSegment>) -> Option<GotLayout> {
    let seg = data?;
    let got = seg
        .sections
        .iter()
        .find(|s| trim_nul(&s.sectname) == b"__got")?;
    let count = u32::try_from(got.size / 8).unwrap_or(0);
    Some(GotLayout {
        addr: got.addr,
        offset: got.offset,
        count,
    })
}

fn find_stubs(text: &OutSegment) -> Option<StubLayout> {
    let stubs = text
        .sections
        .iter()
        .find(|s| trim_nul(&s.sectname) == b"__stubs")?;
    Some(StubLayout {
        addr: stubs.addr,
        offset: stubs.offset,
        count: u32::try_from(stubs.size / 12).unwrap_or(0),
    })
}

/// Appends the synthetic `__common` section to `__DATA`: zero-fill storage
/// for merged tentative definitions, aligned to the strictest common's
/// alignment. Like `__got` it carries no input members; the addresses of
/// the individual commons are handed out by [`common_addresses`].
fn inject_common_section(
    data_secs: &mut Vec<OutSection>,
    commons: &[CommonSym<'_>],
) {
    let size = commons
        .iter()
        .try_fold(0u64, |cursor, c| {
            let a = u64::from(1u32.checked_shl(c.align).unwrap_or(1));
            align_up(cursor, a).checked_add(c.size)
        })
        .unwrap_or(0);
    let align = commons.iter().map(|c| c.align).max().unwrap_or(0);
    data_secs.push(OutSection {
        segname: pad_name(b"__DATA"),
        sectname: pad_name(b"__common"),
        flags: S_ZEROFILL,
        align,
        members: Vec::new(),
        addr: 0,
        offset: 0,
        size,
        zerofill: true,
        reserved1: 0,
        reserved2: 0,
    });
}

/// The virtual address of each common, walking the placed `__common` section
/// with each common's own alignment.
fn common_addresses(
    data: Option<&OutSegment>,
    commons: &[CommonSym<'_>],
) -> Vec<u64> {
    let section = data
        .and_then(|seg| {
            seg.sections
                .iter()
                .find(|s| trim_nul(&s.sectname) == b"__common")
        })
        .filter(|_| !commons.is_empty());
    let Some(section) = section else {
        return Vec::new();
    };
    let mut cursor = section.addr;
    commons
        .iter()
        .map(|c| {
            let a = u64::from(1u32.checked_shl(c.align).unwrap_or(1));
            cursor = align_up(cursor, a);
            let at = cursor;
            cursor = cursor.saturating_add(c.size);
            at
        })
        .collect()
}

/// Total byte size of the load-command table. The segment commands carry their
/// embedded section headers; the trailing commands are a fixed size.
fn commands_size(
    nsects_text: u32,
    nsects_data: u32,
    target: MachoTarget,
    options: &LinkOptions<'_>,
) -> u32 {
    let seg_text = 72 + 80 * nsects_text;
    let seg_data = if nsects_data > 0 {
        72 + 80 * nsects_data
    } else {
        0
    };
    // __LINKEDIT, LC_SYMTAB, LC_DYSYMTAB and LC_UUID are common. Executables
    // also carry __PAGEZERO; dylibs instead identify themselves.
    let base = 72 + 24 + 80 + 24 + 72 * u32::from(!options.dylib);
    let entry = if options.dylib {
        // LC_DYLD_INFO_ONLY + LC_ID_DYLIB.
        48 + dylib_command_size(options.install_name.unwrap_or(""))
    } else if options.dynamic {
        // LC_DYLD_INFO_ONLY + LC_LOAD_DYLINKER + LC_MAIN.
        48 + dylinker_command_size() + 24
    } else {
        thread_command_size(target)
    };
    let platform = 24 * u32::from(options.platform.is_some());
    let dylibs = options.dylibs.iter().fold(0u32, |sum, name| {
        sum.saturating_add(dylib_command_size(name.install_name))
    });
    let trailing = base + entry + platform + dylibs;
    seg_text + seg_data + trailing
}

/// Size of `LC_LOAD_DYLINKER` with `/usr/lib/dyld`, 8-byte aligned.
pub const fn dylinker_command_size() -> u32 {
    align_command(12 + 14)
}

/// Size of one `LC_LOAD_DYLIB` including its install name.
pub fn dylib_command_size(name: &str) -> u32 {
    let name_len = u32::try_from(name.len()).unwrap_or(u32::MAX);
    align_command(24u32.saturating_add(name_len).saturating_add(1))
}

/// Mach-O load commands in a 64-bit image are padded to eight bytes.
const fn align_command(size: u32) -> u32 {
    size.saturating_add(7) & !7
}

/// The byte size of the `LC_UNIXTHREAD` command for `target`: the four header
/// words plus the register state it carries.
pub const fn thread_command_size(target: MachoTarget) -> u32 {
    let words = match target {
        MachoTarget::X86_64 => X86_THREAD_STATE64_COUNT,
        MachoTarget::Arm64 => ARM_THREAD_STATE64_COUNT,
    };
    16 + 4 * words
}

/// Places the `__TEXT` segment: the first section follows the header and load
/// commands; vmsize equals filesize (no zero-fill in `__TEXT`).
fn place_text_segment(
    mut sections: Vec<OutSection>,
    sizeofcmds: u32,
) -> OutSegment {
    order_sections(&mut sections);
    let first_section = HEADER_SIZE + u64::from(sizeofcmds) + HEADER_PAD;
    let mut cursor = first_section;
    let mut vcursor = TEXT_BASE + first_section;
    for s in &mut sections {
        let a = u64::from(1u32.checked_shl(s.align).unwrap_or(1));
        cursor = align_up(cursor, a);
        vcursor = align_up(vcursor, a);
        s.offset = cursor;
        s.addr = vcursor;
        cursor += s.size;
        vcursor += s.size;
    }
    // Darwin maps executable/data segments in whole pages. A short segment
    // command that ended at the last section was structurally readable but
    // rejected as a bad executable by dyld.
    let filesize = align_up(cursor, FILE_PAGE);
    OutSegment {
        name: pad_name(b"__TEXT"),
        vmaddr: TEXT_BASE,
        vmsize: filesize,
        fileoff: 0,
        filesize,
        maxprot: VM_PROT_READ | VM_PROT_EXECUTE,
        initprot: VM_PROT_READ | VM_PROT_EXECUTE,
        sections,
    }
}

/// Places the `__DATA` segment right after `__TEXT`, page-aligned. Zero-fill
/// sections extend `vmsize` past `filesize`; they are ordered last so the file
/// extent closes at the last file-backed section.
fn place_data_segment(
    mut sections: Vec<OutSection>,
    text: &OutSegment,
) -> Option<OutSegment> {
    if sections.is_empty() {
        return None;
    }
    order_sections(&mut sections);
    let vmaddr = align_up(text.vmaddr + text.vmsize, FILE_PAGE);
    let fileoff = align_up(text.fileoff + text.filesize, FILE_PAGE);
    let mut cursor = fileoff;
    let mut vcursor = vmaddr;
    let mut last_file_end = fileoff;
    let mut last_vm_end = vmaddr;
    for s in &mut sections {
        let a = u64::from(1u32.checked_shl(s.align).unwrap_or(1));
        cursor = align_up(cursor, a);
        vcursor = align_up(vcursor, a);
        s.addr = vcursor;
        if s.zerofill {
            // Zero-fill occupies virtual space but no file bytes.
            s.offset = 0;
        } else {
            s.offset = cursor;
            cursor += s.size;
            last_file_end = cursor;
        }
        // Every section advances the virtual cursor, so distinct sections get
        // distinct addresses (zero-fill sections extend `vmsize` past
        // `filesize`).
        vcursor = vcursor.wrapping_add(s.size);
        last_vm_end = vcursor;
    }
    Some(OutSegment {
        name: pad_name(b"__DATA"),
        vmaddr,
        vmsize: align_up(last_vm_end.wrapping_sub(vmaddr), FILE_PAGE),
        fileoff,
        filesize: align_up(last_file_end.wrapping_sub(fileoff), FILE_PAGE),
        maxprot: VM_PROT_READ | crate::macho::constants::VM_PROT_WRITE,
        initprot: VM_PROT_READ | crate::macho::constants::VM_PROT_WRITE,
        sections,
    })
}

/// Computes the `__LINKEDIT` vmaddr and fileoff right after `__DATA` (or
/// `__TEXT` when there is no `__DATA`).
fn linkedit_extents(
    text: &OutSegment,
    data: Option<&OutSegment>,
) -> (u64, u64) {
    let (last_vmaddr, last_vmsize, last_fileoff, last_filesize) = data.map_or(
        (text.vmaddr, text.vmsize, text.fileoff, text.filesize),
        |d| (d.vmaddr, d.vmsize, d.fileoff, d.filesize),
    );
    let vmaddr = align_up(last_vmaddr + last_vmsize, FILE_PAGE);
    let fileoff = align_up(last_fileoff + last_filesize, FILE_PAGE);
    (vmaddr, fileoff)
}

/// Orders a segment's sections file-backed first, zero-fill last, preserving
/// insertion order within each group. A real darwin link orders by section
/// type; this grouping is enough to keep the file extent contiguous.
fn order_sections(sections: &mut [OutSection]) {
    sections.sort_by_key(|s| {
        let name = trim_nul(&s.sectname);
        if s.zerofill {
            3
        } else if name == b"__thread_vars" {
            1
        } else if name == b"__thread_data" {
            2
        } else {
            0
        }
    });
}

/// Records every placed member's base virtual address in the per-input map:
/// the output section address plus the member's offset within it. A symbol
/// whose `n_value` is normalized against the member's input-frame address
/// resolves to `base + offset`.
fn stamp(segment: &OutSegment, sec_vaddr: &mut [Vec<u64>]) {
    for s in &segment.sections {
        for m in &s.members {
            let zero_based =
                usize::try_from(m.section).unwrap_or(0).saturating_sub(1);
            let base = s.addr.wrapping_add(m.offset);
            if let Some(file) = sec_vaddr.get_mut(m.file)
                && let Some(slot) = file.get_mut(zero_based)
            {
                *slot = base;
            }
        }
    }
}

/// The entry symbol's file offset within `__TEXT` (`main_vaddr - TEXT_BASE`).
/// `__TEXT` starts at file offset zero, so the file offset equals the virtual
/// address minus the base.
///
/// An entry that is not defined is an error. Returning zero made the header
/// bytes the entry point, so an image missing `_main` linked and then executed
/// its own magic number. lld hard-errors.
fn entry_offset(
    inputs: &[MachOFile<'_>],
    sec_vaddr: &[Vec<u64>],
    entry: &[u8],
) -> Result<u64> {
    for (file, input) in inputs.iter().enumerate() {
        let syms = input.symbols();
        for sym in &syms {
            if sym.name != entry {
                continue;
            }
            let zero_based = usize::from(sym.n_sect).saturating_sub(1);
            let Some(base) =
                sec_vaddr.get(file).and_then(|f| f.get(zero_based)).copied()
            else {
                continue;
            };
            let Some(frame) = input.section_addr(sym.n_sect) else {
                continue;
            };
            let addr = base.wrapping_add(sym.n_value.wrapping_sub(frame));
            return Ok(addr.wrapping_sub(TEXT_BASE));
        }
    }
    Err(Error::UndefinedEntry(
        String::from_utf8_lossy(entry).into_owned(),
    ))
}

/// Collects linkable input sections into output sections keyed by
/// `(segname, sectname)`. `__TEXT`-segment sections go to `text_secs`,
/// `__DATA`-segment sections to `data_secs`.
fn collect_sections(
    inputs: &[MachOFile<'_>],
    live: &LiveSections,
    text_secs: &mut Vec<OutSection>,
    data_secs: &mut Vec<OutSection>,
) {
    for (file, input) in inputs.iter().enumerate() {
        for section in input.sections() {
            if !is_linkable(&section) || !live.section(file, section.index) {
                continue;
            }
            if section.segname == b"__TEXT" {
                add_member(text_secs, file, &section);
            } else {
                add_member(data_secs, file, &section);
            }
        }
    }
    layout_members(text_secs);
    layout_members(data_secs);
}

/// Adds `section` to the matching output section in `bucket`, creating it on
/// first sight. Alignment is the max across members; flags come from the first
/// member (members of one output section share their section type). Member
/// offsets and the section size are finalised by [`layout_members`].
fn add_member(
    bucket: &mut Vec<OutSection>,
    file: usize,
    section: &MachSection<'_>,
) {
    let key = (section.segname, section.sectname);
    let out = bucket
        .iter_mut()
        .find(|s| (s.segname_for_cmp(), s.sectname_for_cmp()) == key);
    let section_align = if section.sectname == b"__thread_vars" {
        section.align.max(3)
    } else {
        section.align
    };
    let member = Member {
        file,
        section: section.index,
        size: section.size,
        align: section_align,
        offset: 0,
    };
    if let Some(out) = out {
        out.align = out.align.max(section_align);
        out.members.push(member);
        return;
    }
    bucket.push(OutSection {
        segname: pad_name(section.segname),
        sectname: pad_name(section.sectname),
        flags: section.flags,
        align: section_align,
        members: vec![member],
        addr: 0,
        offset: 0,
        size: 0,
        zerofill: matches!(
            section.flags & SECTION_TYPE,
            S_ZEROFILL | S_THREAD_LOCAL_ZEROFILL
        ),
        reserved1: 0,
        reserved2: 0,
    });
}

/// Assigns each member its byte offset within its output section (aligned to
/// the member's own alignment) and sets the section's total size. Members keep
/// their insertion order, so the layout is deterministic.
fn layout_members(sections: &mut [OutSection]) {
    for s in sections {
        let mut cursor = 0u64;
        for m in &mut s.members {
            let a = u64::from(1u32.checked_shl(m.align).unwrap_or(1));
            cursor = align_up(cursor, a);
            m.offset = cursor;
            cursor += m.size;
        }
        s.size = cursor;
    }
}

impl OutSection {
    /// The segment name as a trimmed slice, for matching against input names.
    fn segname_for_cmp(&self) -> &[u8] {
        trim_nul(&self.segname)
    }
    /// The section name as a trimmed slice, for matching against input names.
    fn sectname_for_cmp(&self) -> &[u8] {
        trim_nul(&self.sectname)
    }
}

/// Whether an input section is linked into the output.
///
/// Code, read-only data and `__DATA` content are linked, including
/// final-image DWARF unwind data; debug and input dynamic-linker
/// symbol-pointer / stub sections are deferred.
pub fn is_linkable(section: &MachSection<'_>) -> bool {
    if !is_content_segment(section.segname) {
        return false;
    }
    !is_deferred(section.sectname)
}

/// Whether a segment holds content the image loads, as opposed to metadata
/// the link consumes (`__DWARF`, `__LD`, `__LLVM`).
///
/// `__DATA_CONST` and `__DATA_DIRTY` are ordinary data segments that clang
/// emits routinely -- const globals, vtables and relocated pointers live in
/// the first of them -- so accepting only `__TEXT` and `__DATA` dropped real
/// data with no diagnostic, leaving every reference to it resolving through a
/// section that was never placed. lld gives each its own output segment
/// (`lld/MachO/OutputSegment.h`); this places their content in `__DATA`,
/// which costs the separate const page protection and keeps the bytes and
/// their addresses.
fn is_content_segment(segname: &[u8]) -> bool {
    matches!(
        segname,
        b"__TEXT" | b"__DATA" | b"__DATA_CONST" | b"__DATA_DIRTY"
    )
}

/// Sections consumed or synthesized elsewhere, debug tables, and input
/// dynamic-linker metadata that must not be copied verbatim.
///
/// Everything else in `__TEXT` and `__DATA` is linked. A whitelist of eleven
/// names dropped anything a user named themselves --
/// `__attribute__((section("__DATA,__mine")))`, `__mod_init_func`,
/// `__objc_*` -- with no diagnostic, and a reference into a dropped section
/// resolved against a base of zero.
fn is_deferred(name: &[u8]) -> bool {
    if name.starts_with(b"__debug") {
        return true;
    }
    matches!(
        name,
        b"__unwind_info"
            | b"__compact_unwind"
            | b"__llvm_addrsig"
            | b"__got"
            | b"__la_symbol_ptr"
            | b"__nl_symbol_ptr"
            | b"__stubs"
            | b"__stub_helper"
            | b"__objc_imageinfo"
    )
}

/// Per-file scratch storage for section virtual addresses, indexed by the
/// zero-based form of the input's 1-based section ordinal.
fn empty_section_map(inputs: &[MachOFile<'_>]) -> Vec<Vec<u64>> {
    let mut out = Vec::with_capacity(inputs.len());
    for input in inputs {
        out.push(vec![0u64; input.sections().len()]);
    }
    out
}
