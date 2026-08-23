//! Section byte copy and relocation application for the Mach-O writer.
//!
//! For each output section the writer concatenates its members' bytes into the
//! image at the section's file offset, then patches every input relocation in
//! place through the arch-neutral [`crate::reloc::apply`] driver. The per-file
//! resolver reports each input symbol's resolved output address; the
//! Mach-O-specific addend (the REL-style bytes plus the `x86_64` PC-relative
//! field correction, or the `arm64` `ADDEND` hint) is folded in here.
//!
//! Zero-fill sections (`S_ZEROFILL`, e.g. `__bss`) occupy virtual space but no
//! file bytes, so they are skipped here; their addresses are already recorded
//! for symbol resolution by the layout.

use crate::{
    error::{Error, Result},
    macho::{
        MachOFile, MachReloc, MachSection,
        layout::{MachLayout, Member, OutSection, OutSegment},
        reloc::{
            MachResolver, MachoTarget, SectionResolver, addend_for,
            apply_reloc, arm64_embedded, arm64_pending_addend, is_arm64_addend,
            table_type,
        },
    },
    reloc::Resolver,
    symbol::SymbolId,
};

/// The per-file address tables the writer threads through section copy.
///
/// Each input symbol's resolved address, each symbol's `__got` slot address
/// (zero where it has no slot), and each input section's laid-out virtual
/// address. Bundled so the copy helpers stay under the argument limit.
pub struct AddrTables<'a> {
    pub sym_addr: &'a [Vec<u64>],
    pub got_addr: &'a [Vec<u64>],
    pub sec_vaddr: &'a [Vec<u64>],
    /// Per file, per 1-based input section ordinal: the address the input
    /// file gave that section, which section-relative relocations are stored
    /// against.
    pub in_addr: &'a [Vec<u64>],
}

/// Writes the file-backed section bytes for every segment, applying each
/// member's relocations in place.
pub fn write(
    image: &mut [u8],
    inputs: &[MachOFile<'_>],
    target: MachoTarget,
    layout: &MachLayout,
    tables: &AddrTables<'_>,
) -> Result<()> {
    for segment in [Some(&layout.text), layout.data.as_ref()]
        .into_iter()
        .flatten()
    {
        write_segment(image, inputs, target, segment, tables)?;
    }
    Ok(())
}

/// Writes one segment's file-backed sections and applies their relocations.
fn write_segment(
    image: &mut [u8],
    inputs: &[MachOFile<'_>],
    target: MachoTarget,
    segment: &OutSegment,
    tables: &AddrTables<'_>,
) -> Result<()> {
    for section in &segment.sections {
        if section.zerofill {
            continue;
        }
        for member in &section.members {
            copy_member(image, inputs, target, section, member, tables)?;
        }
    }
    Ok(())
}

/// Copies one member's bytes into its slot within `section` and patches the
/// member's relocations. The slot lives at file offset
/// `section.offset + member.offset`.
fn copy_member(
    image: &mut [u8],
    inputs: &[MachOFile<'_>],
    target: MachoTarget,
    section: &OutSection,
    member: &Member,
    tables: &AddrTables<'_>,
) -> Result<()> {
    let input = inputs
        .get(member.file)
        .ok_or(Error::OutOfRange("member file index"))?;
    let sections = input.sections();
    let zero_based = usize::try_from(member.section)
        .unwrap_or(0)
        .saturating_sub(1);
    let src = sections
        .get(zero_based)
        .ok_or(Error::OutOfRange("member section ordinal"))?;
    let data = src.data;

    let dst_off =
        usize::try_from(section.offset + member.offset).unwrap_or(usize::MAX);
    let end = dst_off
        .checked_add(data.len())
        .ok_or(Error::OutOfRange("section copy"))?;
    let slot = image
        .get_mut(dst_off..end)
        .ok_or(Error::OutOfRange("section image slot"))?;
    slot.copy_from_slice(data);

    let member_vaddr = section.addr.wrapping_add(member.offset);
    let resolver = MachResolver::new(
        file_table(tables.sym_addr, member.file),
        file_table(tables.got_addr, member.file),
    );
    let ctx = RelocCtx {
        target,
        file: member.file,
        sec_vaddr: tables.sec_vaddr,
        in_addr: file_table(tables.in_addr, member.file),
        site_in_addr: src.addr,
    };
    apply_relocations(image, dst_off, member_vaddr, src, &ctx, &resolver)
}

/// Returns one file's row from a per-file table, or an empty table.
fn file_table(tables: &[Vec<u64>], file: usize) -> &[u64] {
    tables.get(file).map(Vec::as_slice).unwrap_or_default()
}

/// Per-member relocation context: the architecture, the owning file and the
/// section-address map (for section-relative fixups). Bundled so the apply
/// helpers stay under the argument limit.
struct RelocCtx<'a> {
    target: MachoTarget,
    file: usize,
    sec_vaddr: &'a [Vec<u64>],
    /// Input section addresses of the file being copied, by 1-based ordinal.
    ///
    /// A section-relative relocation names its referent by ordinal and encodes
    /// the target's address *in the input file*. Turning that into an offset
    /// within the referent needs the address the input gave that section.
    in_addr: &'a [u64],
    /// The input address of the section being copied, for the PC-relative
    /// form, whose stored value is relative to the site's input address.
    site_in_addr: u64,
}

/// Applies every relocation of one input section. `image_off` is the file
/// offset where the section's bytes begin; `place_base` is its virtual address.
fn apply_relocations<R: Resolver>(
    image: &mut [u8],
    image_off: usize,
    place_base: u64,
    section: &MachSection<'_>,
    ctx: &RelocCtx<'_>,
    resolver: &R,
) -> Result<()> {
    let relocs = &section.relocations;
    let mut i = 0;
    while i < relocs.len() {
        let reloc = &relocs[i];
        // An arm64 ADDEND hint annotates the following entry; it has no fixup
        // of its own, so consume it and carry its addend forward.
        if ctx.target == MachoTarget::Arm64 && is_arm64_addend(reloc) {
            let pending = arm64_pending_addend(reloc);
            i = i.saturating_add(1);
            if i >= relocs.len() {
                break;
            }
            let next = &relocs[i];
            let slot_off = slot_offset(image_off, next.r_address);
            apply_one(
                image, slot_off, place_base, next, pending, ctx, resolver,
            )?;
        } else {
            let slot_off = slot_offset(image_off, reloc.r_address);
            apply_one(image, slot_off, place_base, reloc, 0, ctx, resolver)?;
        }
        i = i.saturating_add(1);
    }
    Ok(())
}

/// File offset of a fixup site within a copied section.
fn slot_offset(image_off: usize, r_address: u32) -> usize {
    image_off.wrapping_add(usize::try_from(r_address).unwrap_or(usize::MAX))
}

/// Resolves and writes one relocation. The addend is the Mach-O normalised
/// value (REL bytes + `x86_64` correction, or the `arm64` pending hint).
fn apply_one<R: Resolver>(
    image: &mut [u8],
    slot_off: usize,
    place_base: u64,
    reloc: &MachReloc,
    pending: i64,
    ctx: &RelocCtx<'_>,
    resolver: &R,
) -> Result<()> {
    let width = reloc.width();
    let end = slot_off
        .checked_add(width)
        .ok_or(Error::OutOfRange("reloc slot"))?;
    let slot = image
        .get_mut(slot_off..end)
        .ok_or(Error::OutOfRange("reloc image slot"))?;
    let place = place_base.wrapping_add(u64::from(reloc.r_address));
    let addend = if reloc.r_extern {
        addend_for(ctx.target, reloc, slot, pending)
    } else {
        section_addend(ctx, reloc, slot, pending)
    };
    let r_type = table_type(ctx.target, reloc);
    if reloc.r_extern {
        let sym = SymbolId(usize::try_from(reloc.r_symbolnum).unwrap_or(0));
        apply_reloc(
            ctx.target,
            r_type,
            Some(sym),
            addend,
            place,
            resolver,
            slot,
        )
    } else {
        let vaddr = section_vaddr(ctx.sec_vaddr, ctx.file, reloc.r_symbolnum);
        let sec_resolver = SectionResolver { vaddr };
        apply_reloc(
            ctx.target,
            r_type,
            Some(SymbolId(0)),
            addend,
            place,
            &sec_resolver,
            slot,
        )
    }
}

/// The signed value stored at the fixup site, with no correction applied.
fn raw_addend(reloc: &MachReloc, slot: &[u8]) -> i64 {
    let n = reloc.width().min(slot.len()).min(8);
    let mut wide = [0u8; 8];
    let Some(take) = slot.get(..n) else {
        return 0;
    };
    wide[..n].copy_from_slice(take);
    let value = i64::from_le_bytes(wide);
    let bits = 64 - 8 * u32::try_from(n).unwrap_or(8);
    if bits == 0 || bits >= 64 {
        return value;
    }
    (value << bits) >> bits
}

/// The addend of a section-relative relocation (`r_extern == 0`).
///
/// The stored bytes name the target by its address *in the input file*, and
/// the resolver answers with the referent section's *output* address, so the
/// difference between those two frames has to come out of the addend. lld
/// states the same two formulas (`MachO/InputFiles.cpp:588-603`): a non-PC-
/// relative entry stores the target's absolute input address, and a
/// PC-relative one stores it relative to the end of its own field, so the
/// site's input address goes back in.
fn section_addend(
    ctx: &RelocCtx<'_>,
    reloc: &MachReloc,
    slot: &[u8],
    pending: i64,
) -> i64 {
    let referent = section_in_addr(ctx.in_addr, reloc.r_symbolnum);
    let raw = match ctx.target {
        MachoTarget::X86_64 => raw_addend(reloc, slot),
        // The arm64 section form is an `UNSIGNED` pointer, whose addend lives
        // in the stored bytes (`InputFiles.cpp` reads it the same way).
        MachoTarget::Arm64 => arm64_embedded(reloc, slot) + pending,
    };
    let site = if reloc.r_pcrel {
        ctx.site_in_addr
            .wrapping_add(u64::from(reloc.r_address))
            .cast_signed()
    } else {
        0
    };
    site.wrapping_add(raw).wrapping_sub(referent.cast_signed())
}

/// The input-file address of a section, by 1-based ordinal.
fn section_in_addr(in_addr: &[u64], ordinal: u32) -> u64 {
    let zero_based = usize::try_from(ordinal).unwrap_or(0).saturating_sub(1);
    in_addr.get(zero_based).copied().unwrap_or(0)
}

/// The output virtual address of an input section referenced by a
/// section-relative relocation (`r_extern == 0`), looked up by its 1-based
/// input section ordinal.
fn section_vaddr(sec_vaddr: &[Vec<u64>], file: usize, ordinal: u32) -> u64 {
    let zero_based = usize::try_from(ordinal).unwrap_or(0).saturating_sub(1);
    sec_vaddr
        .get(file)
        .and_then(|f| f.get(zero_based))
        .copied()
        .unwrap_or(0)
}
