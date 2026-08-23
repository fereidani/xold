//! The `AArch64` relocation table: each raw `R_AARCH64_*` type mapped to a
//! [`Spec`].
//!
//! This is the declarative table consumed by the arch-neutral drivers in the
//! parent module. The table splits cleanly along the storage model:
//!
//! * Whole-byte ABS/PREL widths reuse [`Write::Bytes`], identical to x86-64.
//! * Contiguous instruction immediates (the 26-bit branch offset, the 19/14-bit
//!   conditional/test branch targets, the 12-bit ADD/LDST displacement, the
//!   16-bit MOVW slices) reuse [`Write::Field`]; the shared driver reads the
//!   32-bit instruction cell, replaces the field bits, and writes the cell
//!   back.
//! * The split 21-bit ADR/ADRP immediate (immlo at bits [29..30], immhi at bits
//!   [5..23]) cannot be captured by a single mask, so those relocations go
//!   through [`RelExpr::Escape`] together with [`Arch::escape`], which computes
//!   the value reusing the portable [`RelExpr`] arithmetic and writes the
//!   scattered encoding itself. That path bypasses the shared driver, so it
//!   also carries the range and alignment checks the driver would otherwise
//!   have applied, expressed as the [`Field`] whose rule they are.
//!
//! TLS relocations: the local-exec pair `TLSLE_ADD_TPREL_HI12` /
//! `TLSLE_ADD_TPREL_LO12_NC` write contiguous slices of the 12-bit ADD
//! immediate (the hi12 and lo12 of the thread-pointer-relative offset), so
//! they reduce to [`RelExpr::Abs`] over a [`Write::Field`] like the other
//! ADD/LDST immediates. The shift-by-12 bit of the `HI12` ADD is already set
//! by the assembler in the instruction cell and is outside the field mask, so
//! it is preserved untouched.
//!
//! The `TLSLE_MOVW_TPREL_*` slices, which a compiler reaches for when the
//! offset overflows that pair, write the same 16 bits as `MOVW_UABS_*` and are
//! not the same relocation: the slice is *signed*, and its sign picks the
//! instruction rather than a bit in the immediate, so they route through
//! [`RelExpr::Escape`] as well. The dynamic TLS models (`TLSGD`, `TLSIE`,
//! `TLSDESC`, ...) stay there too and are rejected for a static link.
//!
//! [`Write::Bytes`]: crate::reloc::Write::Bytes
//! [`Write::Field`]: crate::reloc::Write::Field
//! [`RelExpr`]: crate::reloc::RelExpr
//! [`Field`]: crate::reloc::Field

use super::{branch26, insn_cell, of, page};
use crate::{
    error::{Error, Result},
    reloc::{
        Arch, Check, Field, Needs, RelExpr, Resolver, Spec, Write, WriteKind,
        pc_value,
    },
    symbol::SymbolId,
};

// --- raw relocation type numbers (AArch64 psABI) -------------------------

/// No relocation; the entry is skipped.
pub const R_AARCH64_NONE: u32 = 0x000;
/// Direct 64-bit: `S + A`.
pub const R_AARCH64_ABS64: u32 = 0x101;
/// Direct 32-bit: `S + A`.
pub const R_AARCH64_ABS32: u32 = 0x102;
/// Direct 16-bit: `S + A`.
pub const R_AARCH64_ABS16: u32 = 0x103;
/// PC-relative 64-bit: `S + A - P`.
pub const R_AARCH64_PREL64: u32 = 0x104;
/// PC-relative 32-bit: `S + A - P`.
pub const R_AARCH64_PREL32: u32 = 0x105;
/// PC-relative 16-bit: `S + A - P`.
pub const R_AARCH64_PREL16: u32 = 0x106;
/// MOVZ/MOVK unsigned 16-bit slice, bits [0..16].
pub const R_AARCH64_MOVW_UABS_G0: u32 = 0x107;
/// MOVZ/MOVK unsigned 16-bit slice, bits [0..16], no check.
pub const R_AARCH64_MOVW_UABS_G0_NC: u32 = 0x108;
/// MOVZ/MOVK unsigned 16-bit slice, bits [16..32].
pub const R_AARCH64_MOVW_UABS_G1: u32 = 0x109;
/// MOVZ/MOVK unsigned 16-bit slice, bits [16..32], no check.
pub const R_AARCH64_MOVW_UABS_G1_NC: u32 = 0x10a;
/// MOVZ/MOVK unsigned 16-bit slice, bits [32..48].
pub const R_AARCH64_MOVW_UABS_G2: u32 = 0x10b;
/// MOVZ/MOVK unsigned 16-bit slice, bits [32..48], no check.
pub const R_AARCH64_MOVW_UABS_G2_NC: u32 = 0x10c;
/// MOVZ/MOVK unsigned 16-bit slice, bits [48..64].
pub const R_AARCH64_MOVW_UABS_G3: u32 = 0x10d;
/// Signed absolute MOVW slices: `S + A`, sliced 16 bits at a time.
///
/// The sign is carried by `movz` versus `movn` rather than by a bit in the
/// field, so the checked forms accept 17, 33 and 49 signed bits -- one wider
/// than the field each time, exactly as the thread-pointer slices do.
pub const R_AARCH64_MOVW_SABS_G0: u32 = 0x10e;
pub const R_AARCH64_MOVW_SABS_G1: u32 = 0x10f;
pub const R_AARCH64_MOVW_SABS_G2: u32 = 0x110;
/// A literal-pool load's 19-bit PC-relative offset: `ldr xN, label`.
pub const R_AARCH64_LD_PREL_LO19: u32 = 0x111;
/// PC-relative ADR low 21 bits: `S + A - P` (split immediate).
pub const R_AARCH64_ADR_PREL_LO21: u32 = 0x112;
/// Page-relative ADRP high 21 bits: `page(S+A) - page(P)` (split immediate).
pub const R_AARCH64_ADR_PREL_PG_HI21: u32 = 0x113;
/// Page-relative ADRP high 21 bits, no check (split immediate).
pub const R_AARCH64_ADR_PREL_PG_HI21_NC: u32 = 0x114;
/// ADD/LDR low 12 bits: `S + A`.
pub const R_AARCH64_ADD_ABS_LO12_NC: u32 = 0x115;
/// LDST unsigned byte offset, low 12 bits: `S + A`.
pub const R_AARCH64_LDST8_ABS_LO12_NC: u32 = 0x116;
/// TBZ/TBNZ 14-bit test branch target: `S + A - P`.
pub const R_AARCH64_TSTBR14: u32 = 0x117;
/// B.cond 19-bit conditional branch target: `S + A - P`.
pub const R_AARCH64_CONDBR19: u32 = 0x118;
/// PC-relative MOVW slices: `S + A - P`, the `:prel_g0:` family hand-written
/// assembly reaches for. Same slicing and same widths as the absolute set.
pub const R_AARCH64_MOVW_PREL_G0: u32 = 0x11f;
pub const R_AARCH64_MOVW_PREL_G0_NC: u32 = 0x120;
pub const R_AARCH64_MOVW_PREL_G1: u32 = 0x121;
pub const R_AARCH64_MOVW_PREL_G1_NC: u32 = 0x122;
pub const R_AARCH64_MOVW_PREL_G2: u32 = 0x123;
pub const R_AARCH64_MOVW_PREL_G2_NC: u32 = 0x124;
pub const R_AARCH64_MOVW_PREL_G3: u32 = 0x125;
/// Unconditional branch (B) 26-bit target: `S + A - P`.
pub const R_AARCH64_JUMP26: u32 = 0x11a;
/// Call (BL) 26-bit target: `S + A - P`.
pub const R_AARCH64_CALL26: u32 = 0x11b;
/// LDST unsigned halfword offset, low 12 bits: `S + A`.
pub const R_AARCH64_LDST16_ABS_LO12_NC: u32 = 0x11c;
/// LDST unsigned word offset, low 12 bits: `S + A`.
pub const R_AARCH64_LDST32_ABS_LO12_NC: u32 = 0x11d;
/// LDST unsigned doubleword offset, low 12 bits: `S + A`.
pub const R_AARCH64_LDST64_ABS_LO12_NC: u32 = 0x11e;
/// LDST unsigned quadword offset, low 12 bits: `S + A`.
pub const R_AARCH64_LDST128_ABS_LO12_NC: u32 = 0x12b;
/// Page-relative GOT high 21 bits: `page(GOT[sym]+A) - page(P)`.
/// A GOT slot reached by a literal-pool load's 19-bit PC-relative offset.
pub const R_AARCH64_GOT_LD_PREL19: u32 = 0x135;
pub const R_AARCH64_ADR_GOT_PAGE: u32 = 0x137;
/// PC-relative 32-bit to the symbol or its PLT entry: `PLT[sym] + A - P`.
pub const R_AARCH64_PLT32: u32 = 0x13a;
/// GOT low 12 bits: `GOT[sym] + A`.
pub const R_AARCH64_LD64_GOT_LO12_NC: u32 = 0x138;
/// PC-relative GOT 32-bit: `GOT[sym] + A - P`.
pub const R_AARCH64_GOTPCREL32: u32 = 0x13b;

// --- dynamic relocation types (emitted into `.rela.dyn`/`.rela.plt`) ------

/// Loader copies a defined data symbol's bytes from a shared object into a
/// `.bss` slot the executable owns. Emitted only for executables that import
/// a data symbol.
pub const R_AARCH64_COPY: u32 = 0x400;
/// Loader fills a GOT slot with the resolved address of a dynamic symbol.
/// Loader fills a GOT slot with the id of the module defining a thread-local.
pub const R_AARCH64_TLS_DTPMOD64: u32 = 1028;
/// Loader fills a GOT slot with a thread-local's offset within its module.
pub const R_AARCH64_TLS_DTPREL64: u32 = 1029;
/// Loader fills a GOT slot with a symbol's offset from the thread pointer.
pub const R_AARCH64_TLS_TPREL64: u32 = 1030;
pub const R_AARCH64_GLOB_DAT: u32 = 0x401;
/// Lazy-binding PLT entry; the loader patches the PLT slot on first call.
pub const R_AARCH64_JUMP_SLOT: u32 = 0x402;
/// Loader computes `B + A` (load base plus addend); used for internal
/// pointer slots so they follow the image at runtime.
pub const R_AARCH64_RELATIVE: u32 = 0x403;
/// Loader resolves an `STT_GNU_IFUNC` indirect function through its resolver.
pub const R_AARCH64_IRELATIVE: u32 = 0x408;

// --- TLS placeholders (recognised but not reduced; routed to Escape) ------

/// General-dynamic TLS, ADR page21.
pub const R_AARCH64_TLSGD_ADR_PAGE21: u32 = 0x201;
/// Local-dynamic TLS, ADR page21.
pub const R_AARCH64_TLSLD_ADR_PAGE21: u32 = 0x206;
/// Initial-exec TLS, ADR GOT page21.
pub const R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21: u32 = 0x21d;
/// Initial-exec TLS, GOT lo12.
pub const R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC: u32 = 0x21e;
/// Local-exec TLS, MOVZ/MOVK TPREL slice [32..48].
pub const R_AARCH64_TLSLE_MOVW_TPREL_G2: u32 = 0x220;
/// Local-exec TLS, MOVZ/MOVK TPREL slice [16..32].
pub const R_AARCH64_TLSLE_MOVW_TPREL_G1: u32 = 0x221;
/// Local-exec TLS, MOVZ/MOVK TPREL slice [16..32], no check.
pub const R_AARCH64_TLSLE_MOVW_TPREL_G1_NC: u32 = 0x222;
/// Local-exec TLS, MOVZ/MOVK TPREL slice [0..16].
pub const R_AARCH64_TLSLE_MOVW_TPREL_G0: u32 = 0x223;
/// Local-exec TLS, MOVZ/MOVK TPREL slice [0..16], no check.
pub const R_AARCH64_TLSLE_MOVW_TPREL_G0_NC: u32 = 0x224;
/// Local-exec TLS, ADD TPREL hi12.
pub const R_AARCH64_TLSLE_ADD_TPREL_HI12: u32 = 0x225;
/// Local-exec TLS, ADD TPREL lo12.
pub const R_AARCH64_TLSLE_ADD_TPREL_LO12_NC: u32 = 0x227;
/// TLS descriptor, ADR page21.
pub const R_AARCH64_TLSDESC_ADR_PAGE21: u32 = 0x232;
/// The rest of a TLS descriptor sequence.
///
/// The descriptor load, the `add` that follows it, and the indirect call
/// through the resolver the descriptor holds. gcc's default TLS model on this
/// target emits all four, so a refusal that names only the first is a refusal
/// a user meets four times.
pub const R_AARCH64_TLSDESC_LD64_LO12: u32 = 0x233;
pub const R_AARCH64_TLSDESC_ADD_LO12: u32 = 0x234;
pub const R_AARCH64_TLSDESC_CALL: u32 = 0x239;
/// TLS descriptor, lo12.
pub const R_AARCH64_TLSDESC: u32 = 0x407;

/// The `AArch64` architecture, for use as the `A` type parameter of the
/// drivers.
pub struct AArch64;

impl Arch for AArch64 {
    fn spec(r_type: u32) -> Result<Spec> {
        Ok(match r_type {
            R_AARCH64_NONE => of(RelExpr::None, Write::Bytes(WriteKind::W64)),
            R_AARCH64_ABS64 => of(RelExpr::Abs, Write::Bytes(WriteKind::W64)),
            // The narrow ABS/PREL widths take either signedness: lld checks
            // them with `checkIntUInt`, so `-1` and `0xffffffff` are both
            // legal in a 32-bit slot. Pinning one of them to a single
            // signedness would reject input the psABI allows.
            R_AARCH64_ABS32 => of(RelExpr::Abs, Write::Bytes(WriteKind::W32SU)),
            R_AARCH64_ABS16 => of(RelExpr::Abs, Write::Bytes(WriteKind::W16SU)),
            R_AARCH64_PREL64 => of(RelExpr::Pc, Write::Bytes(WriteKind::W64)),
            R_AARCH64_PREL32 => of(RelExpr::Pc, Write::Bytes(WriteKind::W32SU)),
            R_AARCH64_PREL16 => of(RelExpr::Pc, Write::Bytes(WriteKind::W16SU)),
            // `ADD_ABS_LO12_NC` and local-exec `TLSLE_ADD_TPREL_LO12_NC`
            // share the spec: both store `S + A` into the 12-bit ADD
            // immediate. For the TLSLE form the resolver supplies the
            // thread-pointer-relative offset as `S`.
            R_AARCH64_ADD_ABS_LO12_NC | R_AARCH64_TLSLE_ADD_TPREL_LO12_NC => {
                of(RelExpr::Abs, Write::Field(imm12()))
            }
            // 16-bit MOVZ/MOVK slices of an absolute value, checked as
            // unsigned: the value is an address, so lld range-checks each
            // slice with `checkUInt` against the bits at or below it. The
            // thread-pointer-relative `TLSLE_MOVW_TPREL_*` set writes the same
            // field but is not the same relocation; it goes through escape
            // below.
            R_AARCH64_MOVW_UABS_G0 => of(RelExpr::Abs, Write::Field(movw(0))),
            R_AARCH64_MOVW_UABS_G0_NC => {
                of(RelExpr::Abs, Write::Field(movw_nc(0)))
            }
            R_AARCH64_MOVW_UABS_G1 => of(RelExpr::Abs, Write::Field(movw(16))),
            R_AARCH64_MOVW_UABS_G1_NC => {
                of(RelExpr::Abs, Write::Field(movw_nc(16)))
            }
            R_AARCH64_MOVW_UABS_G2 => of(RelExpr::Abs, Write::Field(movw(32))),
            R_AARCH64_MOVW_UABS_G2_NC => {
                of(RelExpr::Abs, Write::Field(movw_nc(32)))
            }
            R_AARCH64_MOVW_UABS_G3 => {
                of(RelExpr::Abs, Write::Field(movw_nc(48)))
            }
            R_AARCH64_CALL26 | R_AARCH64_JUMP26 => {
                of(RelExpr::PltPc, Write::Field(branch26()))
            }
            R_AARCH64_CONDBR19 => of(RelExpr::PltPc, Write::Field(branch19())),
            // A literal-pool load reaches its datum, and a GOT one reaches
            // its slot, through the same 19-bit field a conditional branch
            // uses. lld writes all three with one case
            // (`lld/ELF/Arch/AArch64.cpp`).
            R_AARCH64_LD_PREL_LO19 => of(RelExpr::Pc, Write::Field(branch19())),
            R_AARCH64_GOT_LD_PREL19 => {
                of(RelExpr::GotPc, Write::Field(branch19()))
            }
            // A 32-bit PC-relative reference that routes through the PLT when
            // the symbol has one, which is what makes it usable for a
            // relative vtable or a jump table over imported functions.
            R_AARCH64_PLT32 => {
                of(RelExpr::PltPc, Write::Bytes(WriteKind::W32S))
            }
            // The signed and PC-relative MOVW slices go through the same
            // escape as the thread-pointer ones: a slice has no sign bit of
            // its own, so the sign decides between `movz` and `movn` and the
            // value has to reach `write_smovw` whole.
            R_AARCH64_MOVW_SABS_G0
            | R_AARCH64_MOVW_SABS_G1
            | R_AARCH64_MOVW_SABS_G2
            | R_AARCH64_MOVW_PREL_G0
            | R_AARCH64_MOVW_PREL_G0_NC
            | R_AARCH64_MOVW_PREL_G1
            | R_AARCH64_MOVW_PREL_G1_NC
            | R_AARCH64_MOVW_PREL_G2
            | R_AARCH64_MOVW_PREL_G2_NC
            | R_AARCH64_MOVW_PREL_G3 => {
                of(RelExpr::Escape, Write::Field(movw(0)))
            }
            R_AARCH64_TSTBR14 => of(RelExpr::PltPc, Write::Field(branch14())),
            R_AARCH64_GOTPCREL32 => {
                of(RelExpr::GotPc, Write::Bytes(WriteKind::W32S))
            }
            // Local-exec TLS HI12: `S` is the thread-pointer-relative offset
            // (TPOFF). Writes the high 12 bits (val >> 12) into the ADD
            // immediate and range-checks the offset against 24 unsigned bits.
            // The ADD shift bit lives outside the field mask and is preserved.
            // (`TLSLE_ADD_TPREL_LO12_NC` shares `ADD_ABS_LO12_NC`'s spec.)
            R_AARCH64_TLSLE_ADD_TPREL_HI12 => {
                of(RelExpr::Abs, Write::Field(add_tprel_hi12()))
            }
            // These relocations route through escape: their value or encoding
            // cannot be reduced to a portable expression over a contiguous
            // bitfield. The placeholder field only sizes a 4-byte instruction
            // slot; the value and the scattered/masked write are handled by
            // `Arch::escape`, which dispatches on the raw type.
            //   LDST lo12: the ABI masks the address to the low 12 bits FIRST
            //     and then scales by the access size `((value & 0xfff) >>
            //     scale)`. The plain `Field` shifts the full value before
            //     masking, which lets high address bits bleed into the
            //     displacement (an `0x401000` address reached by a word load
            //     would encode offset `0x1000` instead of `0`). `ADD` lo12
            //     stays a `Field` above: its scale is 0, so the orderings
            //     coincide.
            //   ADR/ADRP: the 21-bit immediate is split across immlo [29..30]
            //     and immhi [5..23], which one mask cannot describe.
            //   TLSLE MOVW: the slice is signed, and its sign selects the
            //     instruction. See [`write_smovw`].
            //   TLS: the dynamic sequences are not reduced for a static link.
            R_AARCH64_TLSLE_MOVW_TPREL_G0
            | R_AARCH64_TLSLE_MOVW_TPREL_G0_NC
            | R_AARCH64_TLSLE_MOVW_TPREL_G1
            | R_AARCH64_TLSLE_MOVW_TPREL_G1_NC
            | R_AARCH64_TLSLE_MOVW_TPREL_G2
            | R_AARCH64_LDST8_ABS_LO12_NC
            | R_AARCH64_LDST16_ABS_LO12_NC
            | R_AARCH64_LDST32_ABS_LO12_NC
            | R_AARCH64_LDST64_ABS_LO12_NC
            | R_AARCH64_LDST128_ABS_LO12_NC
            | R_AARCH64_LD64_GOT_LO12_NC
            | R_AARCH64_ADR_PREL_PG_HI21
            | R_AARCH64_ADR_PREL_PG_HI21_NC
            | R_AARCH64_ADR_PREL_LO21
            | R_AARCH64_ADR_GOT_PAGE
            | R_AARCH64_TLSGD_ADR_PAGE21
            | R_AARCH64_TLSLD_ADR_PAGE21
            | R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21
            | R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC
            | R_AARCH64_TLSDESC_ADR_PAGE21
            | R_AARCH64_TLSDESC_LD64_LO12
            | R_AARCH64_TLSDESC_ADD_LO12
            | R_AARCH64_TLSDESC_CALL
            | R_AARCH64_TLSDESC => {
                of(RelExpr::Escape, Write::Field(insn_cell(4)))
            }
            other => return Err(Error::UnsupportedReloc(other)),
        })
    }

    /// The dynamic-TLS types have no lowering here, and saying so through the
    /// rewrite path is what makes the refusal legible.
    ///
    /// Declining a mandatory rewrite is reported by the writer as an
    /// unresolvable thread-local, naming the variable. Reaching the escape
    /// instead reports an unsupported relocation number, which is true and
    /// useless: `unsupported relocation 562` names neither the symbol nor the
    /// fact that it is a thread-local. gcc defaults to TLSDESC on this
    /// target, so that number is the first thing a user with any TLS at all
    /// sees.
    ///
    /// x86-64 already routes its dynamic-TLS types this way, for the sequences
    /// it lowers; here nothing is lowered, so the rewrite always declines and
    /// the diagnostic always fires.
    fn relax_covers(r_type: u32) -> bool {
        Self::relax_lead(r_type) != 0 || is_dynamic_tls(r_type)
    }

    fn relax_required<R: Resolver>(
        r_type: u32,
        _sym: Option<SymbolId>,
        _resolver: &R,
    ) -> bool {
        is_dynamic_tls(r_type)
    }

    fn scan_needs(r_type: u32) -> Result<Needs> {
        let base = Self::spec(r_type)?.expr.needs();
        // `R_AARCH64_ADR_GOT_PAGE` and `R_AARCH64_LD64_GOT_LO12_NC` are
        // classified as `Escape` (the former for its split 21-bit immediate,
        // the latter because it shares the mask-first LDST lo12 writer), but
        // both reference a GOT entry, so the scan must allocate one. (The TLS
        // GOT/TLSDESC relocations would join this set when TLS is reduced.)
        Ok(
            if r_type == R_AARCH64_ADR_GOT_PAGE
                || r_type == R_AARCH64_LD64_GOT_LO12_NC
            {
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
            R_AARCH64_ADR_PREL_PG_HI21 | R_AARCH64_ADR_PREL_PG_HI21_NC => {
                // page(S + A) - page(P), stored as the page count (>> 12).
                let s_a = RelExpr::Abs.compute(sym, addend, 0, resolver)?;
                write_page21(r_type, out, s_a, place)
            }
            R_AARCH64_ADR_PREL_LO21 => {
                let val =
                    pc_value::<Self, R>(r_type, sym, addend, place, resolver)?;
                check_range(r_type, adr_lo21(), val)?;
                write_adr(out, val)
            }
            R_AARCH64_ADR_GOT_PAGE => {
                // page(GOT[sym] + A) - page(P), stored as the page count.
                let got = RelExpr::Got.compute(sym, addend, 0, resolver)?;
                write_page21(r_type, out, got, place)
            }
            // LDST low-12 displacements: mask the value to the low 12 bits
            // first, then scale by the access size (see the spec-table note).
            // The ABS variants use `S + A`; the GOT lo12 variant uses
            // `GOT[sym] + A` and shares the 64-bit access scale.
            R_AARCH64_LDST8_ABS_LO12_NC
            | R_AARCH64_LDST16_ABS_LO12_NC
            | R_AARCH64_LDST32_ABS_LO12_NC
            | R_AARCH64_LDST64_ABS_LO12_NC
            | R_AARCH64_LDST128_ABS_LO12_NC => {
                let val = RelExpr::Abs.compute(sym, addend, 0, resolver)?;
                write_ldst12(r_type, out, val, ldst_scale(r_type))
            }
            R_AARCH64_LD64_GOT_LO12_NC => {
                let val = RelExpr::Got.compute(sym, addend, 0, resolver)?;
                write_ldst12(r_type, out, val, 3)
            }
            // Local-exec TLS MOVW slices: `S` is the thread-pointer-relative
            // offset, which the sign of each slice is read off.
            R_AARCH64_TLSLE_MOVW_TPREL_G0
            | R_AARCH64_TLSLE_MOVW_TPREL_G0_NC
            | R_AARCH64_TLSLE_MOVW_TPREL_G1
            | R_AARCH64_TLSLE_MOVW_TPREL_G1_NC
            | R_AARCH64_TLSLE_MOVW_TPREL_G2 => {
                let val = RelExpr::Abs.compute(sym, addend, 0, resolver)?;
                write_movw(r_type, out, val)
            }
            R_AARCH64_MOVW_SABS_G0
            | R_AARCH64_MOVW_SABS_G1
            | R_AARCH64_MOVW_SABS_G2 => {
                let val = RelExpr::Abs.compute(sym, addend, place, resolver)?;
                write_movw(r_type, out, val)
            }
            R_AARCH64_MOVW_PREL_G0
            | R_AARCH64_MOVW_PREL_G0_NC
            | R_AARCH64_MOVW_PREL_G1
            | R_AARCH64_MOVW_PREL_G1_NC
            | R_AARCH64_MOVW_PREL_G2
            | R_AARCH64_MOVW_PREL_G2_NC
            | R_AARCH64_MOVW_PREL_G3 => {
                let val =
                    pc_value::<Self, R>(r_type, sym, addend, place, resolver)?;
                write_movw(r_type, out, val)
            }
            // TLS relocations are recognised but not reduced yet.
            _ => Err(Error::UnsupportedReloc(r_type)),
        }
    }

    /// A branch to an undefined weak goes to the next instruction; anything
    /// else PC-relative takes the place's own address. lld's list, and its
    /// comment, are `getAArch64UndefinedRelativeWeakVA`
    /// (`lld/ELF/InputSection.cpp`).
    fn undef_weak_pc(r_type: u32, addend: i64, _place: u64) -> u64 {
        let a = addend.cast_unsigned();
        match r_type {
            R_AARCH64_CALL26 | R_AARCH64_CONDBR19 | R_AARCH64_JUMP26
            | R_AARCH64_TSTBR14 => a.wrapping_add(4),
            _ => a,
        }
    }
}

/// Whether `r_type` is part of a general-dynamic, local-dynamic or descriptor
/// TLS sequence -- the models this linker does not lower.
///
/// Naming them lets the refusal go through the rewrite path, which reports an
/// unresolvable thread-local by name. Reaching the escape instead reports an
/// unsupported relocation number: true, and useless, since it names neither
/// the symbol nor the fact that it is a thread-local. gcc defaults to TLSDESC
/// on this target, so that number is the first thing a user with any TLS at
/// all sees.
const fn is_dynamic_tls(r_type: u32) -> bool {
    matches!(
        r_type,
        R_AARCH64_TLSGD_ADR_PAGE21
            | R_AARCH64_TLSLD_ADR_PAGE21
            | R_AARCH64_TLSDESC_ADR_PAGE21
            | R_AARCH64_TLSDESC_LD64_LO12
            | R_AARCH64_TLSDESC_ADD_LO12
            | R_AARCH64_TLSDESC_CALL
            | R_AARCH64_TLSDESC
    )
}

/// The shared 12-bit displacement field of an ADD or LDST instruction, sitting
/// at instruction bits [10..21] with no range check (the `_NC` variants).
const fn imm12() -> Field {
    Field::new(4, 0, 10, 12, Check::None)
}

/// The high 12 bits of a local-exec TPREL ADD immediate, occupying the same
/// bits [10..21] as [`imm12`] but selecting `val >> 12` and range-checking the
/// offset against 24 unsigned bits (matching lld's `checkUInt(loc, val, 24)`).
const fn add_tprel_hi12() -> Field {
    Field::new(4, 12, 10, 12, Check::Unsigned)
}

/// Maps an `R_AARCH64_LDST*_ABS_LO12_NC` type to its access-size scale: the
/// number of low value bits dropped after masking to the page offset. LDST8 has
/// scale 0, LDST16 scale 1, and so on through LDST128 scale 4.
const fn ldst_scale(r_type: u32) -> u8 {
    match r_type {
        R_AARCH64_LDST16_ABS_LO12_NC => 1,
        R_AARCH64_LDST32_ABS_LO12_NC => 2,
        R_AARCH64_LDST64_ABS_LO12_NC => 3,
        R_AARCH64_LDST128_ABS_LO12_NC => 4,
        // LDST8 and any non-LDST type (the caller only passes LDST types) have
        // scale 0: the raw page offset goes straight into the immediate.
        _ => 0,
    }
}

/// A 16-bit MOVZ/MOVK slice at instruction bits [5..20]. `base` is the value's
/// bit offset of the slice's lowest bit (0, 16, 32, 48); the encoded bits drop
/// those `base` low values, so `right_shift = base`.
const fn movw(base: u8) -> Field {
    Field::new(4, base, 5, 16, Check::Unsigned)
}

/// A 16-bit MOVZ/MOVK slice with no overflow check (the `_NC` variants).
const fn movw_nc(base: u8) -> Field {
    Field::new(4, base, 5, 16, Check::None)
}

/// The shift that brings a signed, PC-relative or thread-pointer MOVW slice
/// down, and the signed width the checked forms accept, or `None` for an
/// `_NC` form.
///
/// The widths are lld's (`lld/ELF/Arch/AArch64.cpp`) and are
/// one bit wider than the field: a slice is stored as a signed quantity whose
/// sign is carried by the choice of `movz` or `movn` rather than by a bit in
/// the immediate. The `_NC` forms and `G3` take whatever is there. See
/// [`write_smovw`].
const fn movw_slice(r_type: u32) -> Option<(u8, Option<u32>)> {
    match r_type {
        R_AARCH64_TLSLE_MOVW_TPREL_G0
        | R_AARCH64_MOVW_SABS_G0
        | R_AARCH64_MOVW_PREL_G0 => Some((0, Some(17))),
        R_AARCH64_TLSLE_MOVW_TPREL_G0_NC | R_AARCH64_MOVW_PREL_G0_NC => {
            Some((0, None))
        }
        R_AARCH64_TLSLE_MOVW_TPREL_G1
        | R_AARCH64_MOVW_SABS_G1
        | R_AARCH64_MOVW_PREL_G1 => Some((16, Some(33))),
        R_AARCH64_TLSLE_MOVW_TPREL_G1_NC | R_AARCH64_MOVW_PREL_G1_NC => {
            Some((16, None))
        }
        R_AARCH64_TLSLE_MOVW_TPREL_G2
        | R_AARCH64_MOVW_SABS_G2
        | R_AARCH64_MOVW_PREL_G2 => Some((32, Some(49))),
        R_AARCH64_MOVW_PREL_G2_NC => Some((32, None)),
        R_AARCH64_MOVW_PREL_G3 => Some((48, None)),
        _ => None,
    }
}

/// A 19-bit B.cond target at bits [5..23], scaled and signed.
const fn branch19() -> Field {
    Field::scaled(4, 2, 5, 19, Check::Signed)
}

/// A 14-bit TBZ/TBNZ target at bits [5..18], scaled and signed.
const fn branch14() -> Field {
    Field::scaled(4, 2, 5, 14, Check::Signed)
}

/// The 21-bit ADRP page-count immediate, used for its range rule alone: the
/// value checked against it is the *unshifted* page delta, so the encoding's
/// 21 stored bits sit above the 12 the page shift drops. [`Field::fits`] checks
/// `width + right_shift` bits, which makes this lld's `checkInt(val, 33)`. The
/// split store is [`write_adr`]'s job, so no in-cell position is described.
const fn adr_page21() -> Field {
    Field::new(4, 12, 0, 21, Check::Signed)
}

/// The 21-bit ADR immediate, likewise used for its range rule alone: the value
/// is stored as it stands, so `right_shift` is 0 and the check is lld's
/// `checkInt(val, 21)`.
const fn adr_lo21() -> Field {
    Field::new(4, 0, 0, 21, Check::Signed)
}

/// The 12-bit LDST displacement of an access `1 << scale` bytes wide, used for
/// its alignment rule alone: the immediate counts units of the access size, so
/// lld pairs every scaled form with `checkAlignment(val, 1 << scale)`. LDST8
/// has scale 0 and constrains nothing.
const fn ldst_lo12(scale: u8) -> Field {
    Field::scaled(4, scale, 10, 12, Check::None)
}

/// Applies a field's range check to a value the escape path stores itself.
///
/// A [`RelExpr::Escape`] relocation bypasses the shared [`Write::Field`]
/// driver, so it has to carry the check the driver would otherwise have
/// applied; without it a checked relocation and its `_NC` variant behave
/// identically and an out-of-range value is silently truncated.
fn check_range(r_type: u32, field: Field, value: u64) -> Result<()> {
    if field.fits(value) {
        Ok(())
    } else {
        Err(Error::RelocOverflow(r_type))
    }
}

/// Writes the 12-bit unsigned LDST displacement into a 4-byte instruction cell
/// at bits [10..21]. The `AArch64` ABI masks the value to the low 12 bits FIRST
/// and then scales it right by the access size: `((value & 0xfff) >> scale)`.
/// Shifting the full address before masking would let high address bits bleed
/// down into the displacement (for example a `0x401000` address reached by a
/// word load would encode offset `0x1000` rather than `0`). Opcode bits outside
/// the field are preserved.
///
/// The scale is what forces the alignment check: an address that is not a
/// multiple of the access size cannot be encoded and would be stored as a
/// nearby one instead.
fn write_ldst12(
    r_type: u32,
    out: &mut [u8],
    value: u64,
    scale: u8,
) -> Result<()> {
    if out.len() != 4 {
        return Err(Error::OutOfRange("LDST lo12 relocation slot"));
    }
    if !ldst_lo12(scale).aligned(value) {
        return Err(Error::RelocMisaligned(r_type));
    }
    let scaled = (value & 0xFFF) >> scale;
    let cell = u32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    #[allow(clippy::cast_possible_truncation)]
    let field = (scaled as u32) << 10;
    let mask: u32 = 0xFFF << 10;
    let patched = (cell & !mask) | field;
    out.copy_from_slice(&patched.to_le_bytes());
    Ok(())
}

/// Writes one signed, PC-relative or thread-pointer MOVW slice,
/// range-checking the checked forms first.
fn write_movw(r_type: u32, out: &mut [u8], value: u64) -> Result<()> {
    let Some((shift, check)) = movw_slice(r_type) else {
        return Err(Error::UnsupportedReloc(r_type));
    };
    if let Some(bits) = check
        && !super::fits_signed(value, bits)
    {
        return Err(Error::RelocOverflow(r_type));
    }
    write_smovw(out, value >> shift)
}

/// Writes a 16-bit MOVZ/MOVK/MOVN immediate at instruction bits [5..20],
/// selecting the instruction from the sign of the slice. Mirrors lld's
/// `writeSMovWImm` (`lld/ELF/Arch/AArch64.cpp`).
///
/// The immediate field holds no sign bit, so a negative slice is encoded by
/// `movn`, which loads the bitwise complement of what it carries. Bits 30 and
/// 29 of the instruction select between the three: `10` is `movz`, `00` is
/// `movn`, `11` is `movk`. A `movk` merges its slice into a register the
/// earlier instructions already built, so it has no sign of its own and is left
/// as the assembler wrote it; the other two are rewritten to match the slice.
///
/// `imm` is the value already shifted down to its slice, so bit 16 is the sign:
/// the checked forms accept a signed 17-bit quantity, and the `_NC` forms
/// truncate whatever is above.
fn write_smovw(out: &mut [u8], imm: u64) -> Result<()> {
    if out.len() != 4 {
        return Err(Error::OutOfRange("MOVW relocation slot"));
    }
    let mut cell = u32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    #[allow(clippy::cast_possible_truncation)]
    let mut imm = imm as u32;
    if cell & (1 << 29) == 0 {
        if imm & 0x1_0000 == 0 {
            cell |= 1 << 30;
        } else {
            imm ^= 0xFFFF;
            cell &= !(1 << 30);
        }
    }
    let mask: u32 = 0xFFFF << 5;
    let patched = (cell & !mask) | ((imm & 0xFFFF) << 5);
    out.copy_from_slice(&patched.to_le_bytes());
    Ok(())
}

/// Writes an ADRP page-count immediate: the page delta `page(value) -
/// page(place)`, shifted down by 12 and stored in the split imm21.
///
/// Every form but `ADR_PREL_PG_HI21_NC` range-checks the delta *before* the
/// shift, against 33 signed bits, which is what lld does: the checked cases
/// call `checkInt(ctx, loc, val, 33, rel)` and then fall through into the `_NC`
/// case that performs `write32AArch64Addr(loc, val >> 12)`. Checking the
/// shifted page count against 33 bits instead would accept deltas far beyond
/// ADRP's +/- 4 GiB reach and truncate them into a nearby page.
fn write_page21(
    r_type: u32,
    out: &mut [u8],
    value: u64,
    place: u64,
) -> Result<()> {
    let delta = page(value).wrapping_sub(page(place));
    if r_type != R_AARCH64_ADR_PREL_PG_HI21_NC {
        check_range(r_type, adr_page21(), delta)?;
    }
    write_adr(out, delta >> 12)
}

/// Writes the split 21-bit ADR/ADRP immediate into a 4-byte instruction cell:
/// immlo (the low 2 bits) at cell bits [29..30] and immhi (the high 19 bits) at
/// cell bits [5..23]. Opcode bits outside both ranges are preserved.
fn write_adr(out: &mut [u8], imm: u64) -> Result<()> {
    if out.len() != 4 {
        return Err(Error::OutOfRange("ADR relocation slot"));
    }
    let cell = u32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    #[allow(clippy::cast_possible_truncation)]
    let imm = imm as u32;
    let imm_lo = (imm & 0x3) << 29;
    let imm_hi = (imm & 0x001F_FFFC) << 3;
    let mask: u32 = (0x3 << 29) | (0x001F_FFFC << 3);
    let patched = (cell & !mask) | imm_lo | imm_hi;
    out.copy_from_slice(&patched.to_le_bytes());
    Ok(())
}
