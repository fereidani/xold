//! The darwin arm64 (`AArch64`) relocation table: each `ARM64_RELOC_*` type
//! mapped to a [`Spec`].
//!
//! Mach-O arm64 shares the `AArch64` instruction encoding with ELF, so the
//! instruction-fixup types map directly onto the same
//! [`Field`](crate::reloc::Field) layouts and the
//! same split-immediate [`Escape`] path used in [`super::aarch64`]:
//!
//! * `BRANCH26` is the contiguous imm26 field of a `B`/`BL`, reused verbatim.
//! * `PAGEOFF12` is the 12-bit ADD/LDR displacement at bits [10..21].
//! * `PAGE21` and the GOT/TLS page forms are the split 21-bit ADR/ADRP pair
//!   (immlo at bits [29..30], immhi at bits [5..23]); a single
//!   [`Field`](crate::reloc::Field) cannot describe two masks, so they route
//!   through [`Escape`] together with [`MachoArm64::escape`], which computes
//!   the value reusing the portable [`RelExpr`] arithmetic and writes the
//!   scattered encoding itself.
//!
//! `SUBTRACTOR` is the first half of a paired `S1 - S2` fixup (Escape,
//! resolved by walking the reloc pair at the driver layer). `ADDEND` carries no
//! fixup of its own: it annotates the *following* reloc with a signed 24-bit
//! addend, folded in when the reloc array is walked. At the spec level it is
//! [`RelExpr::None`] since it emits nothing at its own site.
//!
//! TLS (`TLVP`) is recognised but routed to [`Escape`] and rejected until the
//! TLS models land.
//!
//! [`Escape`]: crate::reloc::RelExpr::Escape

use super::{branch26, insn_cell, of, page};
use crate::{
    error::{Error, Result},
    reloc::{Arch, Needs, RelExpr, Resolver, Spec, Write, WriteKind},
    symbol::SymbolId,
};

// --- raw relocation type numbers (darwin arm64) --------------------------

/// Absolute address: `S + A`, 8 bytes.
pub const ARM64_RELOC_UNSIGNED: u32 = 0;
/// The 4-byte form of `UNSIGNED` (`r_length = 2`), as a synthetic type. See
/// [`crate::reloc::macho_x86_64::X86_64_RELOC_UNSIGNED_4`].
pub const ARM64_RELOC_UNSIGNED_4: u32 = 0x100;
/// First half of an `S1 - S2` pair; must be followed by `UNSIGNED` (Escape).
pub const ARM64_RELOC_SUBTRACTOR: u32 = 1;
/// Unconditional `B`/`BL` imm26 target: `S + A - P`.
pub const ARM64_RELOC_BRANCH26: u32 = 2;
/// `ADRP` page21: `page(S + A) - page(P)` (split immediate, Escape).
pub const ARM64_RELOC_PAGE21: u32 = 3;
/// `ADD`/`LDR` pageoff12: `(S + A) & 0xfff`.
pub const ARM64_RELOC_PAGEOFF12: u32 = 4;
/// `ADRP` page21 of a GOT entry (split immediate, Escape; needs GOT).
pub const ARM64_RELOC_GOT_LOAD_PAGE21: u32 = 5;
/// `LDR` pageoff12 of a GOT entry: `GOT[sym] + A` (scaled by 8).
pub const ARM64_RELOC_GOT_LOAD_PAGEOFF12: u32 = 6;
/// 32-bit PC-relative pointer to a GOT entry: `GOT[sym] + A - P`.
pub const ARM64_RELOC_POINTER_TO_GOT: u32 = 7;
/// `ADRP` page21 of a TLV (TLS, Escape).
pub const ARM64_RELOC_TLVP_LOAD_PAGE21: u32 = 8;
/// `LDR` pageoff12 of a TLV (TLS, Escape).
pub const ARM64_RELOC_TLVP_LOAD_PAGEOFF12: u32 = 9;
/// Signed 24-bit addend hint for the following reloc; emits nothing here.
pub const ARM64_RELOC_ADDEND: u32 = 10;

/// The darwin arm64 architecture, for use as the `A` type parameter of the
/// drivers.
pub struct MachoArm64;

impl Arch for MachoArm64 {
    fn spec(r_type: u32) -> Result<Spec> {
        Ok(match r_type {
            ARM64_RELOC_UNSIGNED => {
                of(RelExpr::Abs, Write::Bytes(WriteKind::W64))
            }
            ARM64_RELOC_UNSIGNED_4 => {
                of(RelExpr::Abs, Write::Bytes(WriteKind::W32SU))
            }
            ARM64_RELOC_BRANCH26 => {
                of(RelExpr::PltPc, Write::Field(branch26()))
            }

            ARM64_RELOC_POINTER_TO_GOT => {
                of(RelExpr::GotPc, Write::Bytes(WriteKind::W32S))
            }
            // Split 21-bit immediates, paired fixups and TLS route to escape.
            // The placeholder field only sizes a 4-byte instruction slot; the
            // value and scattered write are handled by `escape`.
            //
            // Both page-offset forms escape for a second reason: the shift
            // depends on the instruction being patched, which a field cannot
            // read. A field shifts the whole value, so encoding the GOT form
            // that way let address bits 12 to 14 reach the immediate.
            ARM64_RELOC_PAGE21
            | ARM64_RELOC_PAGEOFF12
            | ARM64_RELOC_GOT_LOAD_PAGE21
            | ARM64_RELOC_GOT_LOAD_PAGEOFF12
            | ARM64_RELOC_TLVP_LOAD_PAGE21
            | ARM64_RELOC_TLVP_LOAD_PAGEOFF12
            | ARM64_RELOC_SUBTRACTOR => {
                of(RelExpr::Escape, Write::Field(insn_cell(4)))
            }
            // Annotates the following reloc with an addend; no fixup at its own
            // site, so the driver produces no output for it.
            ARM64_RELOC_ADDEND => {
                of(RelExpr::None, Write::Bytes(WriteKind::W64))
            }
            other => return Err(Error::UnsupportedReloc(other)),
        })
    }

    fn scan_needs(r_type: u32) -> Result<Needs> {
        let base = Self::spec(r_type)?.expr.needs();
        // `GOT_LOAD_PAGE21` is Escape for writing but references a GOT entry,
        // so the scan must allocate one.
        Ok(
            if matches!(
                r_type,
                ARM64_RELOC_GOT_LOAD_PAGE21 | ARM64_RELOC_GOT_LOAD_PAGEOFF12
            ) {
                Needs { got: true, ..base }
            } else {
                base
            },
        )
    }

    fn escape<R: Resolver>(
        r_type: u32,
        sym: Option<SymbolId>,
        addend: i64,
        place: u64,
        resolver: &R,
        out: &mut [u8],
    ) -> Result<()> {
        match r_type {
            ARM64_RELOC_PAGE21 => {
                // page(S + A) - page(P), stored as the page count.
                let s_a = RelExpr::Abs.compute(sym, addend, 0, resolver)?;
                write_adr(out, page_delta(s_a, place))
            }
            ARM64_RELOC_GOT_LOAD_PAGE21 => {
                // page(GOT[sym] + A) - page(P), stored as the page count.
                let got = RelExpr::Got.compute(sym, addend, 0, resolver)?;
                write_adr(out, page_delta(got, place))
            }
            ARM64_RELOC_PAGEOFF12 => {
                let s_a = RelExpr::Abs.compute(sym, addend, 0, resolver)?;
                write_pageoff12(out, s_a)
            }
            ARM64_RELOC_GOT_LOAD_PAGEOFF12 => {
                let got = RelExpr::Got.compute(sym, addend, 0, resolver)?;
                write_pageoff12(out, got)
            }
            // SUBTRACTOR pairs and TLS relocations are recognised but not
            // reduced in this phase.
            _ => Err(Error::UnsupportedReloc(r_type)),
        }
    }
}

/// The signed byte distance from the page holding `place` to the page holding
/// `target`.
fn page_delta(target: u64, place: u64) -> i64 {
    page(target)
        .cast_signed()
        .wrapping_sub(page(place).cast_signed())
}

/// Writes the in-page displacement of `va` into an ADD or load/store
/// instruction, shifted by the access size the opcode implies.
///
/// A `Field` shifts the whole value, so it cannot express "mask to the page,
/// then scale": encoding `GOT_LOAD_PAGEOFF12` that way let address bits 12 to
/// 14 reach the immediate, and every folded `adrp + ldr` global access read
/// eight times its offset. lld reads the opcode for the scale and encodes only
/// the in-page bits (`Arch/ARM64Common.h:88-104`).
fn write_pageoff12(out: &mut [u8], va: u64) -> Result<()> {
    if out.len() != 4 {
        return Err(Error::OutOfRange("PAGEOFF12 relocation slot"));
    }
    let base = u32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    let scale = ldst_scale(base);
    let size = 1u64 << scale;
    if va & (size - 1) != 0 {
        return Err(Error::OutOfRange("unaligned PAGEOFF12 access"));
    }
    #[allow(clippy::cast_possible_truncation)]
    let imm = ((va & 0xfff) >> scale) as u32;
    let patched = (base & !(0xfff << 10)) | ((imm & 0xfff) << 10);
    out.copy_from_slice(&patched.to_le_bytes());
    Ok(())
}

/// The access-size shift an instruction implies: zero for `ADD`, and `size`
/// for a load or store, with the 128-bit variant reported as 4.
fn ldst_scale(base: u32) -> u32 {
    if base & 0x3b00_0000 != 0x3900_0000 {
        return 0;
    }
    let scale = base >> 30;
    if scale == 0 && base & 0x0480_0000 == 0x0480_0000 {
        return 4;
    }
    scale
}

/// Writes the split 21-bit ADR/ADRP immediate into a 4-byte instruction cell:
/// immlo (the low 2 bits) at cell bits [29..30] and immhi (the high 19 bits) at
/// cell bits [5..23]. Opcode bits outside both ranges are preserved.
fn write_adr(out: &mut [u8], delta: i64) -> Result<()> {
    if out.len() != 4 {
        return Err(Error::OutOfRange("ADR relocation slot"));
    }
    // The instruction holds 21 signed bits of page count, so the byte
    // distance it can name is +/- 4 GiB. Truncating past that silently
    // pointed the ADRP at a page 4 GiB from the one asked for.
    if !(-(1i64 << 32)..(1i64 << 32)).contains(&delta) {
        return Err(Error::OutOfRange("ADRP page distance"));
    }
    let cell = u32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let imm = (delta >> 12) as u32;
    let imm_lo = (imm & 0x3) << 29;
    let imm_hi = (imm & 0x001F_FFFC) << 3;
    let mask: u32 = (0x3 << 29) | (0x001F_FFFC << 3);
    let patched = (cell & !mask) | imm_lo | imm_hi;
    out.copy_from_slice(&patched.to_le_bytes());
    Ok(())
}
