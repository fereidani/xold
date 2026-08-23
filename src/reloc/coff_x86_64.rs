//! The COFF `x86_64` relocation table: each `IMAGE_REL_AMD64_*` type mapped to
//! a [`Spec`].
//!
//! COFF `x86_64` shares the instruction encoding with ELF and Mach-O but uses
//! its own relocation numbering. The standard address and PC-relative types
//! map directly onto the portable [`RelExpr`] arithmetic, so the scan and
//! apply drivers are shared with the other formats. The type fixes the write
//! width; COFF relocations are REL-encoded (the addend lives in the target
//! bytes), so the addend is extracted at the driver-layer normalisation step,
//! not by the spec.
//!
//! # Addend discipline for REL32 / `REL32_1..5`
//!
//! A COFF PC-relative fixup stores the displacement from the end of the
//! instruction, so the implicit addend is `-4` for `REL32`. The `_1` through
//! `_5` variants carry an extra `-1`..`-5` byte correction (for the trailing
//! opcode bytes after the 4-byte displacement, as in `MOV`/`CALL` encodings
//! with extra immediates). The total addend (`-4`, `-5`, ..., `-9`) is applied
//! when the COFF reloc is normalised into a [`crate::reloc::Reloc`] record at
//! the driver layer; the declarative table only fixes the expression (`S + A -
//! P`) and the write kind (`W32S`), exactly as the Mach-O table handles
//! `SIGNED`/`SIGNED_1/2/4`. An undefined symbol referenced through an
//! `__imp_` alias resolves to the IAT slot; for a statically resolved symbol
//! the value is `S + A - P`.
//!
//! # Section-relative and RVA types
//!
//! `ADDR32NB` (an RVA, `S + A - ImageBase`), `SECTION` (section index),
//! `SECREL` (offset within the section) and `SECREL7` (the low 7 bits of that
//! offset) need information the portable [`Resolver`] does not supply (the
//! image base or a section base). They are routed through [`RelExpr::Escape`]:
//! the COFF writer, which owns the image base and the laid-out section bases,
//! resolves them. The same seam the `AArch64` split-immediate and the Mach-O
//! `SUBTRACTOR` relocations use.
//!
//! Note: the Microsoft PE/COFF spec defines no `GOTPCREL32` or `GOTREL32` for
//! AMD64 (those are ELF constructs; `R_X86_64_GOTPCREL`). Windows resolves
//! imported symbols through the IAT and `__imp_` aliases instead of a GOT, so
//! the standard `REL32`/`ADDR32` types cover those references.

use super::bytes_of as of;
use crate::{
    error::Result,
    reloc::{Arch, Check, Field, RelExpr, Spec, Write, WriteKind},
};

// --- raw relocation type numbers (Microsoft PE/COFF, AMD64) ----------------

/// The relocation is ignored (padding / sentinel).
pub const IMAGE_REL_AMD64_ABSOLUTE: u32 = 0x0000;
/// 64-bit absolute address: `S + A`.
pub const IMAGE_REL_AMD64_ADDR64: u32 = 0x0001;
/// 32-bit absolute address: `S + A`.
pub const IMAGE_REL_AMD64_ADDR32: u32 = 0x0002;
/// 32-bit RVA (address relative to the image base): `S + A - ImageBase`.
pub const IMAGE_REL_AMD64_ADDR32NB: u32 = 0x0003;
/// 32-bit PC-relative: `S + A - P` (addend `-4`).
pub const IMAGE_REL_AMD64_REL32: u32 = 0x0004;
/// 32-bit PC-relative, extra `-1` byte adjustment (addend `-5`).
pub const IMAGE_REL_AMD64_REL32_1: u32 = 0x0005;
/// 32-bit PC-relative, extra `-2` byte adjustment (addend `-6`).
pub const IMAGE_REL_AMD64_REL32_2: u32 = 0x0006;
/// 32-bit PC-relative, extra `-3` byte adjustment (addend `-7`).
pub const IMAGE_REL_AMD64_REL32_3: u32 = 0x0007;
/// 32-bit PC-relative, extra `-4` byte adjustment (addend `-8`).
pub const IMAGE_REL_AMD64_REL32_4: u32 = 0x0008;
/// 32-bit PC-relative, extra `-5` byte adjustment (addend `-9`).
pub const IMAGE_REL_AMD64_REL32_5: u32 = 0x0009;
/// 16-bit section index of the symbol. Section-relative (Escape).
pub const IMAGE_REL_AMD64_SECTION: u32 = 0x000A;
/// 32-bit offset of the symbol within its section. Section-relative (Escape).
pub const IMAGE_REL_AMD64_SECREL: u32 = 0x000B;
/// 7-bit offset of the symbol within its section. Section-relative (Escape).
pub const IMAGE_REL_AMD64_SECREL7: u32 = 0x000C;
/// CLR metadata token (Escape).
pub const IMAGE_REL_AMD64_TOKEN: u32 = 0x000D;
/// 32-bit signed span-dependent value, paired with PAIR (Escape).
pub const IMAGE_REL_AMD64_SREL32: u32 = 0x000E;
/// High half of an SREL32 pair (Escape).
pub const IMAGE_REL_AMD64_PAIR: u32 = 0x000F;
/// 32-bit signed span-dependent value (Escape).
pub const IMAGE_REL_AMD64_SSPAN32: u32 = 0x0010;

/// The COFF `x86_64` architecture, for use as the `A` type parameter of the
/// drivers.
pub struct CoffX86_64;

impl Arch for CoffX86_64 {
    fn spec(r_type: u32) -> Result<Spec> {
        Ok(match r_type {
            IMAGE_REL_AMD64_ABSOLUTE => of(RelExpr::None, WriteKind::W64),
            IMAGE_REL_AMD64_ADDR64 => of(RelExpr::Abs, WriteKind::W64),
            IMAGE_REL_AMD64_ADDR32 => of(RelExpr::Abs, WriteKind::W32),
            // REL32 and REL32_1..5 share the PC-relative 32-bit signed store;
            // the extra `-1..-5` byte correction each `_N` form carries is
            // applied at reloc normalisation, not by the spec.
            IMAGE_REL_AMD64_REL32
            | IMAGE_REL_AMD64_REL32_1
            | IMAGE_REL_AMD64_REL32_2
            | IMAGE_REL_AMD64_REL32_3
            | IMAGE_REL_AMD64_REL32_4
            | IMAGE_REL_AMD64_REL32_5 => of(RelExpr::Pc, WriteKind::W32S),
            // RVA and section-relative types need the image/section base,
            // which is not in the portable Resolver. The COFF writer resolves
            // them through the escape path.
            IMAGE_REL_AMD64_ADDR32NB
            | IMAGE_REL_AMD64_SECREL
            | IMAGE_REL_AMD64_TOKEN
            | IMAGE_REL_AMD64_SREL32
            | IMAGE_REL_AMD64_PAIR
            | IMAGE_REL_AMD64_SSPAN32 => of(RelExpr::Escape, WriteKind::W32),
            IMAGE_REL_AMD64_SECTION => of(RelExpr::Escape, WriteKind::W16),
            // SECREL7 patches the low 7 bits of a byte: a 1-byte cell with a
            // 7-bit field, no range check (the value is truncated).
            IMAGE_REL_AMD64_SECREL7 => Spec {
                expr: RelExpr::Escape,
                write: Write::Field(Field::new(1, 0, 0, 7, Check::None)),
            },
            other => return Err(crate::error::Error::UnsupportedReloc(other)),
        })
    }
}
