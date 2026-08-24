//! Synthesis of the final-image `__TEXT,__unwind_info` index.
//!
//! Darwin objects carry one 32-byte `__LD,__compact_unwind` row per function
//! and ordinary DWARF CIE/FDE records in `__TEXT,__eh_frame`.  `__LD` is
//! linker input and must not be copied to the image.  Dyld/libunwind instead
//! discovers frames through the sorted `__unwind_info` index produced here.
//!
//! The first implementation deliberately uses the simple regular second-level
//! page format and points every function with an FDE at that DWARF record.
//! This is larger than ld64's compressed pages but preserves the authoritative
//! personality and LSDA encodings already emitted by the compiler.

use std::collections::BTreeMap;

use crate::{
    error::{Error, Result},
    macho::{
        GotPlan, MachOFile, MachReloc,
        layout::{MachLayout, Member, OutSection, TEXT_BASE},
        live::LiveSections,
        reloc::MachoTarget,
    },
    reloc::{
        macho_arm64::{ARM64_RELOC_SUBTRACTOR, ARM64_RELOC_UNSIGNED},
        macho_x86_64::{X86_64_RELOC_SUBTRACTOR, X86_64_RELOC_UNSIGNED},
    },
    util::trim_nul,
};

const COMPACT_ENTRY_SIZE: usize = 32;
const UNWIND_INFO_VERSION: u32 = 1;
const REGULAR_SECOND_LEVEL: u32 = 2;
const DWARF_MODE: u32 = 0x0300_0000;
const MODE_MASK: u32 = 0x0f00_0000;
const DWARF_OFFSET_MASK: u32 = 0x00ff_ffff;
const HEADER_SIZE: u32 = 28;
const INDEX_SIZE: u32 = 12;
const PAGE_HEADER_SIZE: u16 = 8;
/// A regular second-level page is at most 4 KiB: its eight-byte header plus
/// 511 eight-byte rows.
const ENTRIES_PER_PAGE: usize = 511;

#[derive(Clone, Copy)]
struct Entry {
    function: u32,
    length: u32,
    encoding: u32,
    personality: Option<u32>,
    lsda: Option<u32>,
}

/// Upper bound on the number of rows the output index will need.
///
/// A retained object can contain compact rows for code discarded within the
/// same section group. The writer filters those after final addresses exist;
/// reserving their eight-byte rows keeps layout independent of resolution.
pub fn entry_capacity(inputs: &[MachOFile<'_>], live: &LiveSections) -> u32 {
    let mut count = 0u32;
    for (file, input) in inputs.iter().enumerate() {
        let has_live_code = input.sections().into_iter().any(|section| {
            section.segname == b"__TEXT"
                && section.sectname != b"__eh_frame"
                && live.section(file, section.index)
        });
        if !has_live_code {
            continue;
        }
        for section in input.sections() {
            if section.segname == b"__LD"
                && section.sectname == b"__compact_unwind"
            {
                count = count.saturating_add(
                    u32::try_from(section.data.len() / COMPACT_ENTRY_SIZE)
                        .unwrap_or(u32::MAX),
                );
            }
        }
    }
    count
}

/// Reserves the indirect pointer required by each retained compact-unwind
/// personality. Final-image personality-array rows point at these GOT slots.
pub fn reserve_personality_got<'d>(
    inputs: &'d [MachOFile<'d>],
    live: &LiveSections,
    got: &mut GotPlan<'d>,
) {
    for (file, input) in inputs.iter().enumerate() {
        let symbols: Vec<_> = input.symbols().iter().collect();
        for section in input.sections() {
            if section.segname != b"__LD"
                || section.sectname != b"__compact_unwind"
            {
                continue;
            }
            for reloc in section.relocations {
                if reloc.r_extern
                    && reloc.r_address % COMPACT_ENTRY_SIZE as u32 == 16
                    && live.symbol(file, reloc.r_symbolnum as usize)
                    && let Some(sym) = symbols.get(reloc.r_symbolnum as usize)
                    && !sym.name.is_empty()
                {
                    got.ensure_global(sym.name);
                }
            }
        }
    }
}

/// File size reserved for a regular-page index with at most `entries` rows.
pub fn section_size(entries: u32) -> u64 {
    if entries == 0 {
        return 0;
    }
    let entries = u64::from(entries);
    let pages = entries.saturating_add(ENTRIES_PER_PAGE as u64 - 1)
        / ENTRIES_PER_PAGE as u64;
    // Header + room for all three personality rows + one index per page +
    // sentinel + a worst-case LSDA row per function + page headers + rows.
    52u64
        .saturating_add(pages.saturating_mul(20))
        .saturating_add(entries.saturating_mul(16))
}

/// Writes the synthetic section using final function addresses and relocated
/// `__eh_frame` member offsets.
pub fn write(
    image: &mut [u8],
    inputs: &[MachOFile<'_>],
    target: MachoTarget,
    layout: &MachLayout,
    sym_addr: &[Vec<u64>],
    got_addr: &[Vec<u64>],
) -> Result<()> {
    let Some(output) = find_output(&layout.text.sections, b"__unwind_info")
    else {
        return Ok(());
    };
    let fdes = fde_offsets(inputs, target, layout, sym_addr)?;
    let mut entries =
        compact_entries(inputs, layout, sym_addr, got_addr, &fdes)?;
    if entries.is_empty() {
        return Err(Error::Format(
            "Mach-O unwind inputs produced an empty __unwind_info index",
        ));
    }
    let mut personalities: Vec<u32> = entries
        .iter()
        .filter_map(|entry| entry.personality)
        .collect();
    personalities.sort_unstable();
    personalities.dedup();
    if personalities.len() > 3 {
        return Err(Error::OutOfRange("Mach-O unwind personalities"));
    }
    for entry in &mut entries {
        if let Some(personality) = entry.personality {
            let index = personalities
                .iter()
                .position(|candidate| *candidate == personality)
                .ok_or(Error::Format("missing Mach-O unwind personality"))?;
            entry.encoding |= u32::try_from(index + 1).unwrap_or(3) << 28;
        }
        if entry.lsda.is_some() {
            entry.encoding |= 0x4000_0000;
        }
    }
    let lsdas: Vec<_> = entries
        .iter()
        .filter_map(|entry| entry.lsda.map(|lsda| (entry.function, lsda)))
        .collect();
    let page_count = entries.len().div_ceil(ENTRIES_PER_PAGE);
    let index_count = u32::try_from(page_count.saturating_add(1))
        .map_err(|_| Error::OutOfRange("Mach-O unwind index count"))?;
    let needed = usize::try_from(section_size(
        u32::try_from(entries.len())
            .map_err(|_| Error::OutOfRange("Mach-O unwind entry count"))?,
    ))
    .map_err(|_| Error::OutOfRange("Mach-O unwind info size"))?;
    let start = usize::try_from(output.offset)
        .map_err(|_| Error::OutOfRange("Mach-O unwind info offset"))?;
    let size = usize::try_from(output.size)
        .map_err(|_| Error::OutOfRange("Mach-O unwind info size"))?;
    if needed > size {
        return Err(Error::OutOfRange("Mach-O unwind info capacity"));
    }
    let out = image
        .get_mut(start..start.saturating_add(size))
        .ok_or(Error::OutOfRange("Mach-O unwind info bytes"))?;

    let personality_offset = HEADER_SIZE;
    let index_offset = personality_offset
        .saturating_add(u32::try_from(personalities.len()).unwrap_or(3) * 4);
    let lsda_offset = index_offset.saturating_add(index_count * INDEX_SIZE);
    let pages_offset = lsda_offset.saturating_add(
        u32::try_from(lsdas.len())
            .unwrap_or(u32::MAX)
            .saturating_mul(8),
    );
    put_u32(out, 0, UNWIND_INFO_VERSION)?;
    put_u32(out, 4, HEADER_SIZE)?;
    put_u32(out, 8, 0)?;
    put_u32(out, 12, personality_offset)?;
    put_u32(out, 16, u32::try_from(personalities.len()).unwrap_or(3))?;
    put_u32(out, 20, index_offset)?;
    put_u32(out, 24, index_count)?;
    for (i, personality) in personalities.iter().enumerate() {
        put_u32(
            out,
            28usize.saturating_add(i.saturating_mul(4)),
            *personality,
        )?;
    }
    let lsda_start = usize::try_from(lsda_offset)
        .map_err(|_| Error::OutOfRange("Mach-O LSDA index offset"))?;
    for (i, (function, lsda)) in lsdas.iter().enumerate() {
        let at = lsda_start.saturating_add(i.saturating_mul(8));
        put_u32(out, at, *function)?;
        put_u32(out, at + 4, *lsda)?;
    }

    let first = entries[0].function;
    let sentinel = entries.iter().fold(first.saturating_add(1), |end, row| {
        end.max(row.function.saturating_add(row.length.max(1)))
    });
    let mut page_at = usize::try_from(pages_offset)
        .map_err(|_| Error::OutOfRange("Mach-O unwind page offset"))?;
    for (page, chunk) in entries.chunks(ENTRIES_PER_PAGE).enumerate() {
        let index_at = usize::try_from(index_offset)
            .unwrap_or(usize::MAX)
            .saturating_add(page.saturating_mul(12));
        let page_lsda = lsda_offset.saturating_add(
            u32::try_from(
                lsdas
                    .iter()
                    .take_while(|(function, _)| *function < chunk[0].function)
                    .count(),
            )
            .unwrap_or(u32::MAX)
            .saturating_mul(8),
        );
        put_index(
            out,
            index_at,
            chunk[0].function,
            u32::try_from(page_at)
                .map_err(|_| Error::OutOfRange("Mach-O unwind page offset"))?,
            page_lsda,
        )?;
        put_u32(out, page_at, REGULAR_SECOND_LEVEL)?;
        put_u16(out, page_at + 4, PAGE_HEADER_SIZE)?;
        put_u16(
            out,
            page_at + 6,
            u16::try_from(chunk.len())
                .map_err(|_| Error::OutOfRange("Mach-O unwind page entries"))?,
        )?;
        for (i, row) in chunk.iter().enumerate() {
            let at = page_at
                .saturating_add(8)
                .saturating_add(i.saturating_mul(8));
            put_u32(out, at, row.function)?;
            put_u32(out, at + 4, row.encoding)?;
        }
        page_at = page_at
            .saturating_add(8)
            .saturating_add(chunk.len().saturating_mul(8));
    }
    let sentinel_at = usize::try_from(index_offset)
        .unwrap_or(usize::MAX)
        .saturating_add(page_count.saturating_mul(12));
    put_index(out, sentinel_at, sentinel, 0, pages_offset)?;
    Ok(())
}

fn compact_entries(
    inputs: &[MachOFile<'_>],
    layout: &MachLayout,
    sym_addr: &[Vec<u64>],
    got_addr: &[Vec<u64>],
    fdes: &BTreeMap<u64, u32>,
) -> Result<Vec<Entry>> {
    let mut rows = BTreeMap::<u32, Entry>::new();
    for (file, input) in inputs.iter().enumerate() {
        for section in input.sections() {
            if section.segname != b"__LD"
                || section.sectname != b"__compact_unwind"
            {
                continue;
            }
            for (index, raw) in
                section.data.chunks_exact(COMPACT_ENTRY_SIZE).enumerate()
            {
                let record = index.saturating_mul(COMPACT_ENTRY_SIZE);
                let address = u32::try_from(record).map_err(|_| {
                    Error::OutOfRange("compact unwind relocation")
                })?;
                let Some(reloc) = section
                    .relocations
                    .iter()
                    .find(|reloc| reloc.r_address == address)
                else {
                    continue;
                };
                let start = resolve_target(
                    inputs,
                    layout,
                    sym_addr,
                    file,
                    reloc,
                    read_u64(raw, 0)?,
                );
                let Some(function) = start
                    .checked_sub(TEXT_BASE)
                    .and_then(|offset| u32::try_from(offset).ok())
                else {
                    continue;
                };
                let length = read_u32(raw, 8)?;
                if length == 0 {
                    continue;
                }
                let input_encoding = read_u32(raw, 12)? & !0x3000_0000;
                let encoding = if let Some(&fde) = fdes.get(&start) {
                    if fde > DWARF_OFFSET_MASK {
                        return Err(Error::OutOfRange(
                            "Mach-O DWARF FDE offset",
                        ));
                    }
                    DWARF_MODE | fde
                } else if input_encoding & MODE_MASK == DWARF_MODE {
                    // A DWARF-mode row without its FDE would send libunwind to
                    // an arbitrary record. Omit it instead of publishing a
                    // corrupt index.
                    continue;
                } else {
                    input_encoding
                };
                let personality = section
                    .relocations
                    .iter()
                    .find(|candidate| {
                        candidate.r_address == address.saturating_add(16)
                            && candidate.r_extern
                    })
                    .and_then(|candidate| {
                        got_addr
                            .get(file)
                            .and_then(|row| {
                                row.get(candidate.r_symbolnum as usize)
                            })
                            .copied()
                    })
                    .and_then(relative_text_offset);
                let lsda = section
                    .relocations
                    .iter()
                    .find(|candidate| {
                        candidate.r_address == address.saturating_add(24)
                    })
                    .and_then(|candidate| {
                        relative_text_offset(resolve_target(
                            inputs,
                            layout,
                            sym_addr,
                            file,
                            candidate,
                            read_u64(raw, 24).ok()?,
                        ))
                    });
                rows.entry(function).or_insert(Entry {
                    function,
                    length,
                    encoding,
                    personality,
                    lsda,
                });
            }
        }
    }
    Ok(rows.into_values().collect())
}

/// Maps each relocated FDE's function start to its offset within the merged
/// output `__eh_frame`. The initial-location field is always eight bytes into
/// an FDE and is expressed by the compiler's `SUBTRACTOR, UNSIGNED` pair.
fn fde_offsets(
    inputs: &[MachOFile<'_>],
    target: MachoTarget,
    layout: &MachLayout,
    sym_addr: &[Vec<u64>],
) -> Result<BTreeMap<u64, u32>> {
    let mut out = BTreeMap::new();
    let Some(eh) = find_output(&layout.text.sections, b"__eh_frame") else {
        return Ok(out);
    };
    for member in &eh.members {
        let input = member_section(inputs, member)?;
        let data = input.data;
        let mut at = 0usize;
        while at.saturating_add(8) <= data.len() {
            let length = usize::try_from(read_u32(data, at)?)
                .map_err(|_| Error::OutOfRange("Mach-O FDE length"))?;
            if length == 0
                || length == usize::try_from(u32::MAX).unwrap_or(usize::MAX)
            {
                break;
            }
            let end = at
                .checked_add(4)
                .and_then(|value| value.checked_add(length))
                .ok_or(Error::OutOfRange("Mach-O FDE record"))?;
            if end > data.len() {
                return Err(Error::Format(
                    "truncated Mach-O __eh_frame record",
                ));
            }
            if read_u32(data, at + 4)? != 0 {
                let field = u32::try_from(at.saturating_add(8))
                    .map_err(|_| Error::OutOfRange("Mach-O FDE relocation"))?;
                if let Some(unsigned) =
                    paired_unsigned(&input.relocations, field, target)
                {
                    let function = resolve_target(
                        inputs,
                        layout,
                        sym_addr,
                        member.file,
                        unsigned,
                        0,
                    );
                    if function != 0 {
                        let offset = member.offset.saturating_add(at as u64);
                        let offset = u32::try_from(offset).map_err(|_| {
                            Error::OutOfRange("Mach-O FDE offset")
                        })?;
                        out.entry(function).or_insert(offset);
                    }
                }
            }
            at = end;
        }
    }
    Ok(out)
}

fn paired_unsigned(
    relocs: &[MachReloc],
    address: u32,
    target: MachoTarget,
) -> Option<&MachReloc> {
    relocs.windows(2).find_map(|pair| {
        let subtractor = &pair[0];
        let unsigned = &pair[1];
        (subtractor.r_address == address
            && unsigned.r_address == address
            && is_subtractor(target, subtractor)
            && is_unsigned(target, unsigned))
        .then_some(unsigned)
    })
}

fn resolve_target(
    inputs: &[MachOFile<'_>],
    layout: &MachLayout,
    sym_addr: &[Vec<u64>],
    file: usize,
    reloc: &MachReloc,
    raw: u64,
) -> u64 {
    if reloc.r_extern {
        return sym_addr
            .get(file)
            .and_then(|row| row.get(reloc.r_symbolnum as usize))
            .copied()
            .unwrap_or(0)
            .wrapping_add(raw);
    }
    let index = usize::try_from(reloc.r_symbolnum)
        .unwrap_or(0)
        .saturating_sub(1);
    let output = layout
        .sec_vaddr
        .get(file)
        .and_then(|row| row.get(index))
        .copied()
        .unwrap_or(0);
    let input = inputs
        .get(file)
        .and_then(|object| {
            u8::try_from(reloc.r_symbolnum)
                .ok()
                .and_then(|ordinal| object.section_addr(ordinal))
        })
        .unwrap_or(0);
    output.wrapping_add(raw.wrapping_sub(input))
}

fn relative_text_offset(address: u64) -> Option<u32> {
    address
        .checked_sub(TEXT_BASE)
        .and_then(|offset| u32::try_from(offset).ok())
}

fn member_section<'a>(
    inputs: &'a [MachOFile<'a>],
    member: &Member,
) -> Result<crate::macho::MachSection<'a>> {
    inputs
        .get(member.file)
        .and_then(|input| {
            input
                .sections()
                .into_iter()
                .find(|section| section.index == member.section)
        })
        .ok_or(Error::OutOfRange("Mach-O unwind member"))
}

fn find_output<'a>(
    sections: &'a [OutSection],
    name: &[u8],
) -> Option<&'a OutSection> {
    sections
        .iter()
        .find(|section| trim_nul(&section.sectname) == name)
}

fn is_subtractor(target: MachoTarget, reloc: &MachReloc) -> bool {
    u32::from(reloc.r_type)
        == match target {
            MachoTarget::X86_64 => X86_64_RELOC_SUBTRACTOR,
            MachoTarget::Arm64 => ARM64_RELOC_SUBTRACTOR,
        }
}

fn is_unsigned(target: MachoTarget, reloc: &MachReloc) -> bool {
    u32::from(reloc.r_type)
        == match target {
            MachoTarget::X86_64 => X86_64_RELOC_UNSIGNED,
            MachoTarget::Arm64 => ARM64_RELOC_UNSIGNED,
        }
}

fn put_index(
    out: &mut [u8],
    at: usize,
    function: u32,
    page: u32,
    lsda: u32,
) -> Result<()> {
    put_u32(out, at, function)?;
    put_u32(out, at + 4, page)?;
    put_u32(out, at + 8, lsda)
}

fn put_u16(out: &mut [u8], at: usize, value: u16) -> Result<()> {
    let slot = out
        .get_mut(at..at.saturating_add(2))
        .ok_or(Error::OutOfRange("Mach-O unwind u16"))?;
    slot.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn put_u32(out: &mut [u8], at: usize, value: u32) -> Result<()> {
    let slot = out
        .get_mut(at..at.saturating_add(4))
        .ok_or(Error::OutOfRange("Mach-O unwind u32"))?;
    slot.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn read_u32(data: &[u8], at: usize) -> Result<u32> {
    let bytes: [u8; 4] = data
        .get(at..at.saturating_add(4))
        .and_then(|slot| slot.try_into().ok())
        .ok_or(Error::Format("truncated Mach-O unwind u32"))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(data: &[u8], at: usize) -> Result<u64> {
    let bytes: [u8; 8] = data
        .get(at..at.saturating_add(8))
        .and_then(|slot| slot.try_into().ok())
        .ok_or(Error::Format("truncated Mach-O unwind u64"))?;
    Ok(u64::from_le_bytes(bytes))
}
