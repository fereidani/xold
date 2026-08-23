//! The COFF i386 relocation table: each `IMAGE_REL_I386_*` type mapped to a
//! [`Spec`].
//!
//! The i386 numbering differs from AMD64 but the semantics are the same shape:
//! direct (`DIR*`) types compute `S + A`, `REL*` types compute `S + A - P`,
//! and the `DIR32NB`/`SECTION`/`SECREL`/`SECREL7` forms are section-relative or
//! RVA and route through [`RelExpr::Escape`] for the same reasons as the AMD64
//! table. COFF i386 relocations are REL-encoded; the implicit addend lives in
//! the target bytes and is extracted at the driver-layer normalisation step,
//! including the `-4` byte correction for `REL32`. See
//! [`super::coff_x86_64`] for the addend and escape rationale.

use super::bytes_of as of;
use crate::{
    error::Result,
    reloc::{Arch, Check, Field, RelExpr, Spec, Write, WriteKind},
};

// --- raw relocation type numbers (Microsoft PE/COFF, i386) -----------------

/// The relocation is ignored (padding / sentinel).
pub const IMAGE_REL_I386_ABSOLUTE: u32 = 0x0000;
/// Direct 16-bit virtual address: `S + A`.
pub const IMAGE_REL_I386_DIR16: u32 = 0x0001;
/// PC-relative 16-bit: `S + A - P`.
pub const IMAGE_REL_I386_REL16: u32 = 0x0002;
/// Direct 32-bit virtual address: `S + A`.
pub const IMAGE_REL_I386_DIR32: u32 = 0x0006;
/// 32-bit RVA (address relative to the image base). Escape.
pub const IMAGE_REL_I386_DIR32NB: u32 = 0x0007;
/// 16-bit section index of the symbol. Section-relative (Escape).
pub const IMAGE_REL_I386_SECTION: u32 = 0x000A;
/// 32-bit offset of the symbol within its section. Section-relative (Escape).
pub const IMAGE_REL_I386_SECREL: u32 = 0x000B;
/// CLR metadata token (Escape).
pub const IMAGE_REL_I386_TOKEN: u32 = 0x000C;
/// 7-bit offset of the symbol within its section. Section-relative (Escape).
pub const IMAGE_REL_I386_SECREL7: u32 = 0x000D;
/// PC-relative 32-bit: `S + A - P` (addend `-4`).
pub const IMAGE_REL_I386_REL32: u32 = 0x0014;

/// The COFF i386 architecture, for use as the `A` type parameter of the
/// drivers.
pub struct CoffI386;

impl Arch for CoffI386 {
    fn spec(r_type: u32) -> Result<Spec> {
        Ok(match r_type {
            IMAGE_REL_I386_ABSOLUTE => of(RelExpr::None, WriteKind::W32),
            IMAGE_REL_I386_DIR16 => of(RelExpr::Abs, WriteKind::W16),
            IMAGE_REL_I386_REL16 => of(RelExpr::Pc, WriteKind::W16),
            IMAGE_REL_I386_DIR32 => of(RelExpr::Abs, WriteKind::W32),
            IMAGE_REL_I386_REL32 => of(RelExpr::Pc, WriteKind::W32S),
            // RVA and section-relative types route through Escape; see the
            // AMD64 table for the rationale.
            IMAGE_REL_I386_DIR32NB
            | IMAGE_REL_I386_SECREL
            | IMAGE_REL_I386_TOKEN => of(RelExpr::Escape, WriteKind::W32),
            IMAGE_REL_I386_SECTION => of(RelExpr::Escape, WriteKind::W16),
            IMAGE_REL_I386_SECREL7 => Spec {
                expr: RelExpr::Escape,
                write: Write::Field(Field::new(1, 0, 0, 7, Check::None)),
            },
            other => return Err(crate::error::Error::UnsupportedReloc(other)),
        })
    }
}
