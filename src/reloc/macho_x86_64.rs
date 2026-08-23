//! The darwin `x86_64` relocation table: each `X86_64_RELOC_*` type mapped to a
//! [`Spec`].
//!
//! Mach-O relocs share the `x86_64` instruction encoding with ELF but use a
//! different type numbering and carry the fixup width in a separate `r_length`
//! field rather than in the type. The width a normal darwin object emits is
//! fixed per type -- `UNSIGNED` is 8 bytes (`r_length = 3`), the PC-relative
//! and branch forms are 4 bytes (`r_length = 2`) -- so each type maps to a
//! single [`WriteKind`]. A width derived from `r_length` by the driver would
//! override the default at apply time; the classification here covers the
//! widths clang actually produces.
//!
//! The addend discipline also differs: Mach-O PC-relative relocs store the
//! displacement from the *end* of the instruction, so the implicit addend is
//! `-(1 << r_length)` plus a small correction for the `_SIGNED_1/2/4` forms.
//! That adjustment is applied when the Mach-O reloc is normalised into a
//! [`crate::reloc::Reloc`] record (a driver-layer concern); the declarative
//! table only fixes the expression and the write kind, so the scan and apply
//! drivers are shared with ELF.
//!
//! `SUBTRACTOR` and `TLV` cannot be reduced to a single portable expression
//! and are routed through [`RelExpr::Escape`]: `SUBTRACTOR` is the first half
//! of a paired `S1 - S2` fixup (a read-modify-write resolved by walking the
//! reloc pair at the driver layer), and `TLV` is a thread-local reference
//! deferred until the TLS models land.

use super::bytes_of as of;
use crate::{
    error::Result,
    reloc::{Arch, RelExpr, Spec, WriteKind},
};

// --- raw relocation type numbers (darwin x86_64) -------------------------

/// Absolute address: `S + A`. Usually 8 bytes (`r_length = 3`); a 4-byte form
/// exists for low 32-bit absolute references.
pub const X86_64_RELOC_UNSIGNED: u32 = 0;
/// PC-relative 32-bit displacement: `S + A - P`.
pub const X86_64_RELOC_SIGNED: u32 = 1;
/// `CALL`/`JMP` with a 32-bit displacement: `PLT[sym] + A - P` (collapses to
/// `S + A - P` in a static link).
pub const X86_64_RELOC_BRANCH: u32 = 2;
/// `MOVQ` load of a GOT entry: `GOT[sym] + A - P`.
pub const X86_64_RELOC_GOT_LOAD: u32 = 3;
/// Other GOT references: `GOT[sym] + A - P`.
pub const X86_64_RELOC_GOT: u32 = 4;
/// First half of an `S1 - S2` pair; must be followed by `UNSIGNED`. Routed to
/// [`RelExpr::Escape`].
pub const X86_64_RELOC_SUBTRACTOR: u32 = 5;
/// PC-relative 32-bit displacement with a `-1` addend correction.
pub const X86_64_RELOC_SIGNED_1: u32 = 6;
/// PC-relative 32-bit displacement with a `-2` addend correction.
pub const X86_64_RELOC_SIGNED_2: u32 = 7;
/// PC-relative 32-bit displacement with a `-4` addend correction.
pub const X86_64_RELOC_SIGNED_4: u32 = 8;
/// Thread-local variable reference; deferred (Escape).
pub const X86_64_RELOC_TLV: u32 = 9;

/// The 4-byte form of `UNSIGNED` (`r_length = 2`), as a synthetic type.
///
/// Mach-O sizes an absolute fixup through `r_length` rather than through the
/// type, and the declarative table is keyed on the type alone. The reader
/// normalises a 4-byte `UNSIGNED` to this so the table can state the narrow
/// store without the driver growing a width override.
pub const X86_64_RELOC_UNSIGNED_4: u32 = 0x100;

/// The darwin `x86_64` architecture, for use as the `A` type parameter of the
/// drivers.
pub struct MachoX86_64;

impl Arch for MachoX86_64 {
    fn spec(r_type: u32) -> Result<Spec> {
        Ok(match r_type {
            // `UNSIGNED` is 8 bytes for the pointer-width absolute references
            // a darwin object emits. `.long _sym` is the legal 4-byte form,
            // which the reader normalises to its own type: pinning both to a
            // 64-bit store made the narrow one die on a slot half its size.
            X86_64_RELOC_UNSIGNED => of(RelExpr::Abs, WriteKind::W64),
            X86_64_RELOC_UNSIGNED_4 => of(RelExpr::Abs, WriteKind::W32SU),
            X86_64_RELOC_BRANCH => of(RelExpr::PltPc, WriteKind::W32S),
            X86_64_RELOC_GOT_LOAD | X86_64_RELOC_GOT => {
                of(RelExpr::GotPc, WriteKind::W32S)
            }
            // `SIGNED` and the `_SIGNED_1/2/4` forms share the PC-relative
            // 32-bit store; the extra `-1/-2/-4` addend correction each carries
            // is applied at reloc normalisation, not by the spec.
            X86_64_RELOC_SIGNED
            | X86_64_RELOC_SIGNED_1
            | X86_64_RELOC_SIGNED_2
            | X86_64_RELOC_SIGNED_4 => of(RelExpr::Pc, WriteKind::W32S),
            // `SUBTRACTOR` is a paired fixup; `TLV` is TLS. Both need the
            // escape path the drivers reject for now.
            X86_64_RELOC_SUBTRACTOR | X86_64_RELOC_TLV => {
                of(RelExpr::Escape, WriteKind::W64)
            }
            other => return Err(crate::error::Error::UnsupportedReloc(other)),
        })
    }
}
