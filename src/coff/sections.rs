//! Section byte copy and relocation application for the PE writer.
//!
//! For each output section the writer concatenates its members' bytes into the
//! image at the section's raw-data offset, then patches every input relocation
//! in place through the arch-neutral [`crate::reloc::apply`] driver via the
//! COFF [`CoffResolver`]. The COFF addend (REL-style target bytes plus the
//! per-type PC-relative correction) is folded in here, mirroring the Mach-O
//! writer's section copy pass.
//!
//! `.bss` sections occupy virtual space but no file bytes, so they are skipped.
//!
//! The section-relative escape relocations (`IMAGE_REL_AMD64_SECREL`,
//! `IMAGE_REL_AMD64_ADDR32NB`) need the image base and the symbol's owning
//! output section, which the portable [`Resolver`] trait does not carry. They
//! are intercepted here, before the generic driver, and resolved from the
//! [`CoffResolver`]'s section-RVA table.

use crate::{
    coff::{
        CoffFile, CoffSection, CoffSymbolTable,
        layout::{Member, OutSection, PeLayout},
        reloc::{
            CoffResolver, CoffTarget, addend_for, apply_reloc, reloc_width,
        },
    },
    error::{Error, Result},
    reloc::{
        Resolver,
        coff_x86_64::{
            IMAGE_REL_AMD64_ABSOLUTE, IMAGE_REL_AMD64_ADDR32NB,
            IMAGE_REL_AMD64_SECREL,
        },
    },
    symbol::SymbolId,
};

/// Writes the file-backed output sections, applying each member's relocations
/// in place.
///
/// `sym_addr` is the per-file, per-symbol resolved virtual address;
/// `sym_sec_rva` is the per-file, per-symbol owning output section RVA.
pub fn write(
    image: &mut [u8],
    inputs: &[CoffFile<'_>],
    target: CoffTarget,
    layout: &PeLayout,
    image_base: u64,
    sym_addr: &[Vec<u64>],
    sym_sec_rva: &[Vec<u32>],
) -> Result<()> {
    for section in &layout.sections {
        if !section.has_file_data() {
            continue;
        }
        for member in &section.members {
            copy_member(
                image,
                inputs,
                target,
                section,
                member,
                image_base,
                sym_addr,
                sym_sec_rva,
            )?;
        }
    }
    Ok(())
}

/// Copies one member's bytes into its slot within `section` and patches the
/// member's relocations. The slot lives at file offset
/// `section.pointer_to_raw_data + member.offset`.
#[allow(clippy::too_many_arguments)]
fn copy_member(
    image: &mut [u8],
    inputs: &[CoffFile<'_>],
    target: CoffTarget,
    section: &OutSection,
    member: &Member,
    image_base: u64,
    sym_addr: &[Vec<u64>],
    sym_sec_rva: &[Vec<u32>],
) -> Result<()> {
    let input = inputs
        .get(member.file)
        .ok_or(Error::OutOfRange("member file index"))?;
    let src = input
        .section_at(member.section)
        .ok_or(Error::OutOfRange("member section ordinal"))?;
    let data = src.data;

    let dst_off = usize::try_from(
        section
            .pointer_to_raw_data
            .wrapping_add(u32::try_from(member.offset).unwrap_or(u32::MAX)),
    )
    .unwrap_or(usize::MAX);
    let end = dst_off
        .checked_add(data.len())
        .ok_or(Error::OutOfRange("section copy"))?;
    let slot = image
        .get_mut(dst_off..end)
        .ok_or(Error::OutOfRange("section image slot"))?;
    slot.copy_from_slice(data);

    let member_rva = section
        .virtual_address
        .wrapping_add(u32::try_from(member.offset).unwrap_or(u32::MAX));
    let resolver = CoffResolver::new(
        sym_addr
            .get(member.file)
            .map(Vec::as_slice)
            .unwrap_or_default(),
        sym_sec_rva
            .get(member.file)
            .map(Vec::as_slice)
            .unwrap_or_default(),
        image_base,
    );
    let symbols = input.symbols();
    apply_relocations(
        image, dst_off, member_rva, image_base, src, target, &resolver, symbols,
    )
}

/// Applies every relocation of one input section. `image_off` is the file
/// offset where the section's bytes begin; `place_rva` is its RVA. The
/// section-relative escape types are resolved from the resolver's section-RVA
/// table before the generic driver runs.
#[allow(clippy::too_many_arguments)]
fn apply_relocations(
    image: &mut [u8],
    image_off: usize,
    place_rva: u32,
    image_base: u64,
    section: &CoffSection<'_>,
    target: CoffTarget,
    resolver: &CoffResolver<'_>,
    symbols: &CoffSymbolTable<'_>,
) -> Result<()> {
    for reloc in &section.relocations {
        let slot_off = image_off
            .checked_add(
                usize::try_from(reloc.virtual_address).unwrap_or(usize::MAX),
            )
            .ok_or(Error::OutOfRange("reloc slot offset"))?;
        let width = reloc_width(reloc);
        let end = slot_off
            .checked_add(width)
            .ok_or(Error::OutOfRange("reloc slot"))?;
        let slot = image
            .get_mut(slot_off..end)
            .ok_or(Error::OutOfRange("reloc image slot"))?;
        let sym =
            SymbolId(usize::try_from(reloc.symbol_table_index).unwrap_or(0));
        let r_type = u32::from(reloc.typ);
        // A PE image never loads at address 0, so a relocation whose symbol
        // resolved to address 0 targets an unresolved symbol: its section was
        // dropped (for example the deferred `.pdata`/`.xdata`) or it is an
        // undefined external the linker did not synthesise. The value would
        // then overflow with an opaque `relocation overflow (type N)`; report
        // the symbol by name instead so the cause is actionable. Absolute
        // constants (`IMAGE_SYM_ABSOLUTE`, value 0) and the null padding
        // relocation (`ABSOLUTE`) legitimately carry a zero, so they are
        // excluded.
        assert_symbol_resolved(symbols, sym, r_type, resolver)?;
        if try_section_relative(r_type, sym, slot, resolver, image_base)? {
            continue;
        }
        let place = image_base
            .wrapping_add(u64::from(place_rva))
            .wrapping_add(u64::from(reloc.virtual_address));
        let addend = addend_for(reloc, slot);
        apply_reloc(target, r_type, Some(sym), addend, place, resolver, slot)?;
    }
    Ok(())
}

/// Reports an unresolved symbol when a relocation references one whose address
/// resolved to zero. A PE image loads at a non-zero image base, so the only
/// legitimate zeros are an absolute constant (`IMAGE_SYM_ABSOLUTE`), the null
/// `ABSOLUTE` padding relocation that carries no symbol value, and a weak
/// external that found no definition: binding to zero is what a weak
/// declaration asks for (clang gives it an absolute-zero default; lld does the
/// same). The name lookup runs only on the cold zero-address path.
fn assert_symbol_resolved(
    symbols: &CoffSymbolTable<'_>,
    sym: SymbolId,
    r_type: u32,
    resolver: &CoffResolver<'_>,
) -> Result<()> {
    if r_type == IMAGE_REL_AMD64_ABSOLUTE || resolver.symbol_addr(sym) != 0 {
        return Ok(());
    }
    let Some(found) = symbols.iter().find(|s| s.index as usize == sym.0) else {
        return Err(Error::UndefinedReference(format!(
            "<symbol #{idx}> (resolves to address 0; unresolved)",
            idx = sym.0
        )));
    };
    if found.is_absolute() || found.is_weak() {
        return Ok(());
    }
    let name = String::from_utf8_lossy(found.name);
    let display = if name.is_empty() {
        format!("<symbol #{}>", sym.0)
    } else {
        name.into_owned()
    };
    Err(Error::UndefinedReference(format!(
        "{display} (resolves to address 0; \
         its section was dropped or it is undefined)"
    )))
}

/// Resolves a section-relative escape relocation, returning `true` when the
/// type is one this path handles (`SECREL`, `ADDR32NB`). `SECREL` stores the
/// symbol's offset within its owning output section; `ADDR32NB` stores the
/// symbol's RVA. Both fold in the REL addend already in the target bytes.
fn try_section_relative(
    r_type: u32,
    sym: SymbolId,
    slot: &mut [u8],
    resolver: &CoffResolver<'_>,
    image_base: u64,
) -> Result<bool> {
    // Sign-extended and folded as a signed quantity, the way every other
    // relocation here reads its inline addend. Read unsigned, a negative
    // label difference -- which is what `a - b` arithmetic leaves in the slot
    // -- became a value near 2^32, so the sum either wrapped past the section
    // or failed the range check below with a number nobody wrote.
    let addend = i64::from(read_u32_le(slot).cast_signed());
    let value: u32 = match r_type {
        IMAGE_REL_AMD64_SECREL => {
            let sym_addr = resolver.symbol_addr(sym);
            let sec_rva = resolver.section_rva(sym);
            sym_addr
                .wrapping_sub(image_base)
                .wrapping_sub(u64::from(sec_rva))
                .wrapping_add_signed(addend)
                .try_into()
                .map_err(|_| Error::RelocOverflow(r_type))?
        }
        IMAGE_REL_AMD64_ADDR32NB => resolver
            .symbol_addr(sym)
            .wrapping_sub(image_base)
            .wrapping_add_signed(addend)
            .try_into()
            .map_err(|_| Error::RelocOverflow(r_type))?,
        _ => return Ok(false),
    };
    slot.copy_from_slice(&value.to_le_bytes());
    Ok(true)
}

/// Reads the first four bytes of `slot` as a little-endian `u32`, or zero when
/// the slot is shorter.
fn read_u32_le(slot: &[u8]) -> u32 {
    let bytes = slot.get(..4).unwrap_or_default();
    let mut buf = [0u8; 4];
    let n = bytes.len().min(4);
    buf[..n].copy_from_slice(&bytes[..n]);
    u32::from_le_bytes(buf)
}
