//! The RISC-V (rv64) relocation table: each raw `R_RISCV_*` type mapped to a
//! [`Spec`].
//!
//! This is the declarative table consumed by the arch-neutral drivers in the
//! parent module. The storage model splits three ways, mirroring how lld and
//! mold classify RISC-V immediates:
//!
//! * Whole-byte ABS widths reuse [`Write::Bytes`], identical to x86-64.
//! * The I-type 12-bit immediate of `R_RISCV_LO12_I` is contiguous at
//!   instruction bits [31..20] with no value adjustment, so it reuses
//!   [`Write::Field`]; the shared driver patches the cell directly.
//! * Every other RISC-V immediate routes through [`RelExpr::Escape`]:
//!   - The U-type hi20 (`HI20`, `PCREL_HI20`, and the auipc half of `CALL`) is
//!     contiguous at bits [31..12] but needs a `+0x800` sign-compensation
//!     before the high 20 bits are taken, which a plain bitfield cannot
//!     express.
//!   - The S-type lo12 (`LO12_S`), B-type branch (`BRANCH`), and J-type jump
//!     (`JAL`) immediates are scattered across non-adjacent bit ranges.
//!   - `CALL`/`CALL_PLT` cover a paired `auipc`+`jalr` sequence (8 bytes),
//!     wider than one instruction cell.
//!
//! `escape` reuses the portable [`RelExpr`] arithmetic to compute the value,
//! then writes the (possibly scattered) encoding itself.
//!
//! The `.eh_frame` paired relocations (`ADD8/16/32/64`, `SUB8/16/32/64`,
//! `SET6/8/16/32`, `SUB6`, `32_PCREL`) are also routed through `escape`:
//! ADD/SUB and SET6/SUB6 are read-modify-write over a fixed width, which no
//! portable `RelExpr` can express. `ADD<n>` reads the slot, adds `S + A`, and
//! writes it back; `SUB<n>` subtracts `S + A`. The two halves of a pair share
//! one offset, so applying the section's entries in order yields the intended
//! `original + S_add - S_sub`. `SET<n>` overwrites the slot's low `8*n` bits
//! with `S + A` (a plain store, not read-modify-write). `SET6` and `SUB6`
//! touch only the low 6 bits of the byte, preserving the high 2 bits.
//! `32_PCREL` is the plain `S + A - P` signed 32-bit store used for the
//! `.eh_frame` `PC_begin` field.
//!
//! The PCREL hi/lo pair (`PCREL_LO12_I`/`_S`) points its symbol at the
//! instruction carrying the matching `PCREL_HI20`, not at the real target, so
//! resolving it needs a section-level "find the paired HI20" pass. That pass
//! ([`rewrite_pcrel_pairs`]) runs once per RISC-V section, in the writer's
//! per-section copy task, before the parallel apply: it rewrites each LO12
//! entry to carry its paired HI20's target symbol with an addend adjusted so
//! the `Pc` expression evaluated at the LO12's own place yields the same
//! `S + A - P_hi` the HI20 used. [`Riscv64::escape`] then stores the low 12
//! bits of that value, reconstructing the address with the auipc.
//!
//! Linker relaxation (`RELAX` collapsing `auipc`+`jalr` to `jal`, `ALIGN`
//! padding hints) is deferred: both are classified [`RelExpr::None`] so the
//! drivers skip them, leaving the assembler's bytes untouched.
//!
//! TLS: the local-exec set (`TPREL_HI20`, `TPREL_LO12_I`, `TPREL_LO12_S`,
//! `TPREL_ADD`) is reduced. `TPREL_HI20` and `TPREL_LO12_S` reuse the absolute
//! hi20 / S-type lo12 escape writers, since the only difference from
//! `HI20`/`LO12_S` is that the resolver returns the thread-pointer-relative
//! offset for an `STT_TLS` symbol. `TPREL_LO12_I` reduces to the same
//! contiguous I-type field as `LO12_I`. `TPREL_ADD` is an annotation on the
//! `add rd, rd, tp` instruction and writes nothing, so it is treated as
//! [`RelExpr::None`]. The dynamic TLS models (`TLS_GD_HI20`, `TLS_GOT_HI20`,
//! `TLSDESC_*`) stay routed through [`RelExpr::Escape`] and are rejected for a
//! static link.
//!
//! [`Write::Bytes`]: crate::reloc::Write::Bytes
//! [`Write::Field`]: crate::reloc::Write::Field
//! [`RelExpr`]: crate::reloc::RelExpr

use rustc_hash::FxHashMap;

use super::{insn_cell, of, read_le, write_le};
use crate::{
    elf::Rela64,
    endian::{I64, U64},
    error::{Error, Result},
    reloc::{
        Arch, Check, Field, Needs, RelExpr, Resolver, Spec, Write, WriteKind,
        pc_value, plt_pc_value,
    },
    symbol::SymbolId,
};

// --- raw relocation type numbers (RISC-V psABI) --------------------------

/// No relocation; the entry is skipped.
pub const R_RISCV_NONE: u32 = 0;
/// Direct 32-bit: `S + A`.
pub const R_RISCV_32: u32 = 1;
/// Direct 64-bit: `S + A`.
pub const R_RISCV_64: u32 = 2;
/// PC-relative branch (B-type): `S + A - P`.
pub const R_RISCV_BRANCH: u32 = 16;
/// PC-relative jump (J-type): `S + A - P`.
pub const R_RISCV_JAL: u32 = 17;
/// Paired `auipc`+`jalr` (PC-relative): `PLT[sym] + A - P`.
pub const R_RISCV_CALL: u32 = 18;
/// Paired `auipc`+`jalr` via the PLT; equals `R_RISCV_CALL` in a static link.
pub const R_RISCV_CALL_PLT: u32 = 19;
/// The `auipc` of a GOT-indirect access (U-type): `GOT[sym] + A - P`.
///
/// Paired with a `PCREL_LO12_I` on the load that follows, exactly as
/// `PCREL_HI20` is. This is what `-fPIE`/`-fPIC` code emits for every
/// reference to extern data, spelled `%got_pcrel_hi` in assembly.
pub const R_RISCV_GOT_HI20: u32 = 20;
/// PC-relative hi20 (U-type): `S + A - P`.
pub const R_RISCV_PCREL_HI20: u32 = 23;
/// Low 12 of a PC-relative value, I-type; paired to a `PCREL_HI20`.
pub const R_RISCV_PCREL_LO12_I: u32 = 24;
/// Low 12 of a PC-relative value, S-type; paired to a `PCREL_HI20`.
pub const R_RISCV_PCREL_LO12_S: u32 = 25;
/// Absolute hi20 (U-type): `S + A`.
pub const R_RISCV_HI20: u32 = 26;
/// Absolute low 12 (I-type): `S + A`.
pub const R_RISCV_LO12_I: u32 = 27;
/// Absolute low 12 (S-type): `S + A`.
pub const R_RISCV_LO12_S: u32 = 28;
/// Read-modify-write: add `S + A` to an 8-bit slot.
pub const R_RISCV_ADD8: u32 = 33;
/// Read-modify-write: add `S + A` to a 16-bit slot.
pub const R_RISCV_ADD16: u32 = 34;
/// Read-modify-write: add `S + A` to a 32-bit slot.
pub const R_RISCV_ADD32: u32 = 35;
/// Read-modify-write: add `S + A` to a 64-bit slot.
pub const R_RISCV_ADD64: u32 = 36;
/// Read-modify-write: subtract `S + A` from an 8-bit slot.
pub const R_RISCV_SUB8: u32 = 37;
/// Read-modify-write: subtract `S + A` from a 16-bit slot.
pub const R_RISCV_SUB16: u32 = 38;
/// Read-modify-write: subtract `S + A` from a 32-bit slot.
pub const R_RISCV_SUB32: u32 = 39;
/// Read-modify-write: subtract `S + A` from a 64-bit slot.
pub const R_RISCV_SUB64: u32 = 40;
/// A 32-bit PC-relative reference to a symbol's GOT slot: `GOT[sym] + A - P`.
/// Used where the value has to be a plain word rather than an instruction
/// immediate.
pub const R_RISCV_GOT32_PCREL: u32 = 41;
/// Alignment hint emitted with relaxation; deferred (treated as `None`).
pub const R_RISCV_ALIGN: u32 = 43;
/// PC-relative compressed branch (CB-type, `c.beqz`/`c.bnez`): `S + A - P`.
pub const R_RISCV_RVC_BRANCH: u32 = 44;
/// PC-relative compressed jump (CJ-type, `c.j`): `S + A - P`.
pub const R_RISCV_RVC_JUMP: u32 = 45;
/// Relaxation hint paired with another reloc; deferred (treated as `None`).
pub const R_RISCV_RELAX: u32 = 51;
/// Sets a ULEB128 field to `S + A`, always paired with [`R_RISCV_SUB_ULEB128`]
/// at the same offset: the two together encode `a - b`.
///
/// GAS emits the pair for `.uleb128 a - b` whenever relaxation is enabled,
/// which is the default, and a `.gcc_except_table` call-site table is exactly
/// that. So every C++ object with exception handling carries them.
pub const R_RISCV_SET_ULEB128: u32 = 60;
/// The subtrahend of the pair above.
pub const R_RISCV_SUB_ULEB128: u32 = 61;
/// Read-modify-write: subtract `S + A` from the low 6 bits of a byte.
pub const R_RISCV_SUB6: u32 = 52;
/// Overwrite the low 6 bits of a byte with `S + A`, preserving high 2 bits.
pub const R_RISCV_SET6: u32 = 53;
/// Overwrite a byte with `S + A`.
pub const R_RISCV_SET8: u32 = 54;
/// Overwrite a halfword with `S + A`.
pub const R_RISCV_SET16: u32 = 55;
/// Overwrite a word with `S + A`.
pub const R_RISCV_SET32: u32 = 56;
/// PC-relative 32-bit: `S + A - P` (`.eh_frame` `PC_begin`).
pub const R_RISCV_32_PCREL: u32 = 57;
/// A 32-bit PC-relative reference through the PLT: `PLT[sym] + A - P`.
/// Collapses to `S + A - P` in a static link, as `CALL_PLT` does.
pub const R_RISCV_PLT32: u32 = 59;

// --- ELF header flags (`e_flags`) ----------------------------------------

/// The object holds compressed (RVC) instructions.
const EF_RISCV_RVC: u32 = 0x1;
/// The floating-point calling convention: soft, single, double or quad.
const EF_RISCV_FLOAT_ABI: u32 = 0x6;
/// The object was built for the reduced (RV32E/RV64E) register set.
const EF_RISCV_RVE: u32 = 0x8;

/// Folds one input's `e_flags` into the value the output header carries.
///
/// `so_far` is the merged value of the inputs read before this one, or `None`
/// for the first, which sets the baseline. This is lld's `RISCV::calcEFlags`
/// (`lld/ELF/Arch/RISCV.cpp`): the compressed-instruction bit is the
/// union over the inputs, because an image holding one RVC object holds RVC
/// instructions, while the floating-point ABI and the register set have to
/// agree outright -- they describe how functions pass arguments, so mixing them
/// produces calls that disagree about where the arguments are.
///
/// The flags matter beyond bookkeeping: glibc's loader and the kernel read the
/// float ABI back off the image, and an executable that reports the soft-float
/// ABI is refused against hard-float libraries.
///
/// A disagreement is reported as the name of what disagreed, for the caller to
/// pair with the input that carried it: which file it was is the caller's to
/// know, not this table's.
pub(super) fn merge_eflags(
    so_far: Option<u32>,
    incoming: u32,
) -> core::result::Result<u32, &'static str> {
    let Some(target) = so_far else {
        return Ok(incoming);
    };
    if incoming & EF_RISCV_FLOAT_ABI != target & EF_RISCV_FLOAT_ABI {
        return Err("floating-point ABI");
    }
    if incoming & EF_RISCV_RVE != target & EF_RISCV_RVE {
        return Err("register set");
    }
    Ok(target | (incoming & EF_RISCV_RVC))
}

// --- dynamic relocation types (emitted into `.rela.dyn`/`.rela.plt`) ------

/// Loader computes `B + A` (load base plus addend); used for internal
/// pointer slots so they follow the image at runtime.
/// Loader fills a GOT slot with the id of the module defining a thread-local.
pub const R_RISCV_TLS_DTPMOD64: u32 = 7;
/// Loader fills a GOT slot with a thread-local's offset within its module.
pub const R_RISCV_TLS_DTPREL64: u32 = 9;
/// Loader fills a GOT slot with a symbol's offset from the thread pointer.
pub const R_RISCV_TLS_TPREL64: u32 = 11;
pub const R_RISCV_RELATIVE: u32 = 3;
/// Loader copies a defined data symbol's bytes from a shared object into a
/// `.bss` slot the executable owns. Emitted only for executables that import
/// a data symbol.
pub const R_RISCV_COPY: u32 = 4;
/// Lazy-binding PLT entry; the loader patches the PLT slot on first call.
pub const R_RISCV_JUMP_SLOT: u32 = 5;
/// Loader resolves an `STT_GNU_IFUNC` indirect function through its resolver.
pub const R_RISCV_IRELATIVE: u32 = 58;

// --- TLS placeholders (recognised but not reduced; routed to Escape) ------

/// General-dynamic TLS, hi20.
pub const R_RISCV_TLS_GD_HI20: u32 = 22;
/// Initial-exec TLS, GOT hi20.
pub const R_RISCV_TLS_GOT_HI20: u32 = 21;
/// Local-exec TLS, hi20.
pub const R_RISCV_TPREL_HI20: u32 = 29;
/// Local-exec TLS, ADD annotation.
pub const R_RISCV_TPREL_ADD: u32 = 32;
/// Local-exec TLS, I-type lo12.
pub const R_RISCV_TPREL_LO12_I: u32 = 30;
/// Local-exec TLS, S-type lo12.
pub const R_RISCV_TPREL_LO12_S: u32 = 31;
/// TLS descriptor, hi20.
pub const R_RISCV_TLSDESC_HI20: u32 = 62;
/// TLS descriptor, load lo12.
pub const R_RISCV_TLSDESC_LOAD_LO12: u32 = 63;

/// The RISC-V 64 architecture, for use as the `A` type parameter of the
/// drivers.
pub struct Riscv64;

impl Arch for Riscv64 {
    fn spec(r_type: u32) -> Result<Spec> {
        Ok(match r_type {
            // The pair is folded into one entry by `rewrite_pcrel_pairs`
            // before the writer sees it: the difference lands in the `SET`
            // entry's addend and the `SUB` entry becomes `R_RISCV_NONE`. What
            // remains is a mandatory rewrite, because the field is a ULEB128
            // whose length the existing bytes fix and no fixed-width store
            // can express. An unpaired `SET` never reaches here -- the
            // pre-pass refuses it -- and an unpaired `SUB` lands on this
            // escape and is refused for want of an arm.
            R_RISCV_SET_ULEB128 | R_RISCV_SUB_ULEB128 => {
                of(RelExpr::Escape, Write::Bytes(WriteKind::W8))
            }
            R_RISCV_NONE | R_RISCV_RELAX | R_RISCV_ALIGN => {
                of(RelExpr::None, Write::Bytes(WriteKind::W64))
            }
            // Signed or unsigned. The psABI gives `R_RISCV_32` no
            // signedness and lld writes it as a plain `word32` with no check
            // at all, so a `.word sym - bigger_sym` difference -- which is
            // negative and perfectly ordinary -- was refused here and
            // accepted everywhere else. `W32SU` is the same reading
            // `AArch64`'s ABS32 already uses, and it still bounds the field.
            R_RISCV_32 => of(RelExpr::Abs, Write::Bytes(WriteKind::W32SU)),
            R_RISCV_64 => of(RelExpr::Abs, Write::Bytes(WriteKind::W64)),
            // PC-relative 32-bit signed store (eh_frame PC_begin): `S + A - P`.
            R_RISCV_32_PCREL => of(RelExpr::Pc, Write::Bytes(WriteKind::W32S)),
            // The word-sized PC-relative pair of the two above, through the
            // PLT and through the GOT. lld range-checks both as signed 32
            // (`Arch/RISCV.cpp:545`), the same check `32_PCREL` gets.
            R_RISCV_PLT32 => of(RelExpr::PltPc, Write::Bytes(WriteKind::W32S)),
            R_RISCV_GOT32_PCREL => {
                of(RelExpr::GotPc, Write::Bytes(WriteKind::W32S))
            }
            // I-type lo12: contiguous 12-bit field at [31..20], no adjustment.
            R_RISCV_LO12_I | R_RISCV_TPREL_LO12_I => {
                of(RelExpr::Abs, Write::Field(itype_lo12()))
            }
            // Local-exec TPREL_ADD annotates the `add rd, rd, tp` instruction
            // and contributes no value; the instruction is already correct.
            R_RISCV_TPREL_ADD => of(RelExpr::None, Write::Field(insn_cell(4))),
            // U-type hi20 needs the +0x800 compensation; scattered and paired
            // immediates and the 8-byte CALL sequence go through escape. The
            // placeholder field only sizes the instruction slot; the value and
            // encoding are handled by `Arch::escape`.
            R_RISCV_HI20
            | R_RISCV_LO12_S
            | R_RISCV_GOT_HI20
            | R_RISCV_PCREL_HI20
            | R_RISCV_PCREL_LO12_I
            | R_RISCV_PCREL_LO12_S
            | R_RISCV_JAL
            | R_RISCV_BRANCH
            | R_RISCV_TLS_GD_HI20
            | R_RISCV_TLS_GOT_HI20
            | R_RISCV_TPREL_HI20
            | R_RISCV_TPREL_LO12_S
            | R_RISCV_TLSDESC_HI20
            | R_RISCV_TLSDESC_LOAD_LO12 => {
                of(RelExpr::Escape, Write::Field(insn_cell(4)))
            }
            // Paired auipc+jalr spans two instruction cells (8 bytes).
            R_RISCV_CALL | R_RISCV_CALL_PLT => {
                of(RelExpr::Escape, Write::Field(insn_cell(8)))
            }
            // Compressed (16-bit) branch and jump: scattered immediates in a
            // 2-byte cell, PC-relative.
            R_RISCV_RVC_BRANCH | R_RISCV_RVC_JUMP => {
                of(RelExpr::Escape, Write::Field(insn_cell(2)))
            }
            // Read-modify-write paired relocations used in `.eh_frame` and
            // relaxation-safe data. The `Bytes` placeholder only sizes the
            // slot; the value and store are handled by `Arch::escape`, which
            // reads the current bytes, adds or subtracts `S + A` (or patches
            // the low 6 bits), and writes them back.
            R_RISCV_ADD8 | R_RISCV_SUB8 | R_RISCV_SET6 | R_RISCV_SUB6
            | R_RISCV_SET8 => of(RelExpr::Escape, Write::Bytes(WriteKind::W8)),
            R_RISCV_ADD16 | R_RISCV_SUB16 | R_RISCV_SET16 => {
                of(RelExpr::Escape, Write::Bytes(WriteKind::W16))
            }
            R_RISCV_ADD32 | R_RISCV_SUB32 | R_RISCV_SET32 => {
                of(RelExpr::Escape, Write::Bytes(WriteKind::W32))
            }
            R_RISCV_ADD64 | R_RISCV_SUB64 => {
                of(RelExpr::Escape, Write::Bytes(WriteKind::W64))
            }
            other => return Err(Error::UnsupportedReloc(other)),
        })
    }

    fn scan_needs(r_type: u32) -> Result<Needs> {
        let base = Self::spec(r_type)?.expr.needs();
        // `CALL`/`CALL_PLT` are `Escape` for writing but semantically
        // `PLT[sym] + A - P`, so the scan must request a PLT entry like the
        // `PltPc` expression would. (In a static link the PLT collapses to the
        // symbol, so this need has no allocation effect; recording it keeps the
        // table honest about the reference kind.)
        // `GOT_HI20` is `Escape` for the same reason -- the U-type encoding
        // needs the `+0x800` compensation -- but semantically it is
        // `GOT[sym] + A - P`, so the scan must allocate the slot the
        // `GotPc` expression would.
        Ok(match r_type {
            R_RISCV_CALL | R_RISCV_CALL_PLT => Needs { plt: true, ..base },
            R_RISCV_GOT_HI20 => Needs { got: true, ..base },
            _ => base,
        })
    }

    /// The dynamic-TLS types have no lowering here, and saying so through the
    /// rewrite path is what makes the refusal legible.
    ///
    /// Declining a mandatory rewrite is reported by the writer as an
    /// unresolvable thread-local, naming the variable. Reaching the escape
    /// instead reports an unsupported relocation number, which is true and
    /// useless: `unsupported relocation 562` names neither the symbol nor the
    /// fact that it is a thread-local. A general-dynamic sequence is what any
    /// shared-library thread-local compiles to, so the number is the first
    /// thing a user with TLS sees.
    ///
    /// x86-64 already routes its dynamic-TLS types this way, for the sequences
    /// it lowers; here nothing is lowered, so the rewrite always declines and
    /// the diagnostic always fires.
    /// The ULEB128 rewrite declares no lead -- it works entirely within its
    /// own field -- so the default answer, which reads the lead, would be no.
    fn relax_covers(r_type: u32) -> bool {
        r_type == R_RISCV_SET_ULEB128
            || matches!(
                r_type,
                R_RISCV_TLS_GD_HI20
                    | R_RISCV_TLSDESC_HI20
                    | R_RISCV_TLSDESC_LOAD_LO12
            )
    }

    /// A folded `SET_ULEB128` must be rewritten: the field is a ULEB128 whose
    /// length the bytes already there fix, so there is no fixed-width store to
    /// fall back to. Everything else is left alone.
    fn relax_required<R: Resolver>(
        r_type: u32,
        _sym: Option<SymbolId>,
        _resolver: &R,
    ) -> bool {
        r_type == R_RISCV_SET_ULEB128
    }

    /// The rest of the ULEB128 encoding: every byte after the first that the
    /// one before it continued into. The window the rewrite gets is then the
    /// whole field, which is what lets the value be written back at exactly
    /// the length it was stored at.
    fn relax_trail(r_type: u32, slot: &[u8], after: &[u8]) -> usize {
        if r_type != R_RISCV_SET_ULEB128 {
            return 0;
        }
        // The field's length is written into the field: every byte but the
        // last carries a continuation bit. The slot is its first byte, so the
        // walk starts there and stops at the first byte that does not
        // continue -- which is why the slot has to be in hand and not just
        // what follows it. Ten bytes hold any 64-bit value, so nothing longer
        // is a ULEB128 this field could have held.
        if slot.first().is_none_or(|b| b & 0x80 == 0) {
            return 0;
        }
        let mut n = 0;
        for &byte in after.iter().take(ULEB_MAX - 1) {
            n += 1;
            if byte & 0x80 == 0 {
                break;
            }
        }
        n
    }

    /// Writes the folded difference back over the field, at the length the
    /// field already had.
    fn relax<R: Resolver>(
        r_type: u32,
        _sym: Option<SymbolId>,
        addend: i64,
        _place: u64,
        _resolver: &R,
        window: &mut [u8],
    ) -> Result<bool> {
        if r_type != R_RISCV_SET_ULEB128 {
            return Ok(false);
        }
        write_uleb_in_place(window, addend.cast_unsigned())?;
        Ok(true)
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
            R_RISCV_HI20 | R_RISCV_TPREL_HI20 => {
                let val = RelExpr::Abs.compute(sym, addend, 0, resolver)?;
                write_hi20(out, val, r_type)
            }
            R_RISCV_LO12_S | R_RISCV_TPREL_LO12_S => {
                let val = RelExpr::Abs.compute(sym, addend, 0, resolver)?;
                write_lo12_s(out, val)
            }
            R_RISCV_PCREL_HI20 => {
                let val =
                    pc_value::<Self, R>(r_type, sym, addend, place, resolver)?;
                write_hi20(out, val, r_type)
            }
            // The GOT-indirect `auipc`. Only the expression differs from the
            // ordinary PC-relative one -- it names the symbol's GOT slot
            // rather than the symbol -- so the encoding, the `+0x800`
            // compensation and the 20-bit check are the same writer. lld
            // groups it with `PCREL_HI20` in the same `case` for this reason.
            R_RISCV_GOT_HI20 => {
                let val =
                    RelExpr::GotPc.compute(sym, addend, place, resolver)?;
                write_hi20(out, val, r_type)
            }
            R_RISCV_JAL => {
                let val =
                    pc_value::<Self, R>(r_type, sym, addend, place, resolver)?;
                write_jal(out, val, r_type)
            }
            R_RISCV_BRANCH => {
                let val =
                    pc_value::<Self, R>(r_type, sym, addend, place, resolver)?;
                write_branch(out, val, r_type)
            }
            R_RISCV_RVC_BRANCH => {
                let val =
                    pc_value::<Self, R>(r_type, sym, addend, place, resolver)?;
                write_cbtype(out, val, r_type)
            }
            R_RISCV_RVC_JUMP => {
                let val =
                    pc_value::<Self, R>(r_type, sym, addend, place, resolver)?;
                write_cjtype(out, val, r_type)
            }
            R_RISCV_CALL | R_RISCV_CALL_PLT => {
                let val = plt_pc_value::<Self, R>(
                    r_type, sym, addend, place, resolver,
                )?;
                write_call(out, val, r_type)
            }
            // Read-modify-write paired relocations: ADD adds `S + A`, SUB
            // subtracts it, SET6 overwrites and SUB6 subtracts from the low 6
            // bits. The value is `S + A` (null symbol contributes zero), so
            // the absolute expression computes it for all of them.
            R_RISCV_ADD8 | R_RISCV_ADD16 | R_RISCV_ADD32 | R_RISCV_ADD64
            | R_RISCV_SUB8 | R_RISCV_SUB16 | R_RISCV_SUB32 | R_RISCV_SUB64
            | R_RISCV_SET6 | R_RISCV_SUB6 | R_RISCV_SET8 | R_RISCV_SET16
            | R_RISCV_SET32 => {
                let val = RelExpr::Abs.compute(sym, addend, 0, resolver)?;
                apply_rmw(r_type, out, val)
            }
            // The LO12 half of a PCREL pair. The writer's pre-pass
            // ([`rewrite_pcrel_pairs`]) has rewritten this entry to carry its
            // paired HI20's target symbol and an addend adjusted so the `Pc`
            // value here equals `S + A - P_hi` (the value the auipc encoded).
            // Its low 12 bits go into the same field as the absolute LO12.
            R_RISCV_PCREL_LO12_I => {
                let val =
                    pc_value::<Self, R>(r_type, sym, addend, place, resolver)?;
                write_lo12_i(out, val)
            }
            R_RISCV_PCREL_LO12_S => {
                let val =
                    pc_value::<Self, R>(r_type, sym, addend, place, resolver)?;
                write_lo12_s(out, val)
            }
            // TLS relocations are recognised but not reduced yet.
            _ => Err(Error::UnsupportedReloc(r_type)),
        }
    }

    /// A branch to an undefined weak goes to itself: the deliberate infinite
    /// loop, which makes the missing definition visible instead of jumping
    /// somewhere arbitrary. Everything else keeps the plain `A - P`. lld's
    /// list is `getRISCVUndefinedRelativeWeakVA`
    /// (`lld/ELF/InputSection.cpp`).
    fn undef_weak_pc(r_type: u32, addend: i64, place: u64) -> u64 {
        let a = addend.cast_unsigned();
        match r_type {
            R_RISCV_BRANCH | R_RISCV_JAL | R_RISCV_CALL | R_RISCV_CALL_PLT
            | R_RISCV_RVC_BRANCH | R_RISCV_RVC_JUMP | R_RISCV_PLT32 => a,
            _ => a.wrapping_sub(place),
        }
    }
}

/// The I-type 12-bit immediate field of `LO12_I`, at instruction bits
/// [31..20] with no range check: the low 12 bits of any value fit.
const fn itype_lo12() -> Field {
    Field::new(4, 0, 20, 12, Check::None)
}

/// Whether the signed interpretation of `value` fits in `bits` bits.
fn fits_signed(value: i64, bits: u8) -> bool {
    super::fits_signed(value.cast_unsigned(), u32::from(bits))
}

/// Writes the U-type hi20 immediate into a 4-byte instruction cell, applying
/// the `+0x800` sign-compensation that pairs the hi20 with a sign-extended
/// lo12. Bits [11..0] (opcode + rd) are preserved. The adjusted value
/// `(val + 0x800) >> 12` must fit 20 signed bits.
fn write_hi20(out: &mut [u8], val: u64, r_type: u32) -> Result<()> {
    if out.len() != 4 {
        return Err(Error::OutOfRange("RISC-V hi20 relocation slot"));
    }
    let adjusted = val.wrapping_add(0x800);
    let hi = adjusted.cast_signed() >> 12;
    if !fits_signed(hi, 20) {
        return Err(Error::RelocOverflow(r_type));
    }
    #[allow(clippy::cast_possible_truncation)]
    let imm = (adjusted & 0xFFFF_F000) as u32;
    patch32(out, 0x0FFF, imm)
}

/// Writes the S-type lo12 immediate into a 4-byte store instruction cell. The
/// 12-bit value is split into imm[11..5] at cell bits [31..25] and imm[4..0]
/// at cell bits [11..7]; the rest (opcode, rs1, rs2, funct3) is preserved.
fn write_lo12_s(out: &mut [u8], val: u64) -> Result<()> {
    if out.len() != 4 {
        return Err(Error::OutOfRange("RISC-V lo12_s relocation slot"));
    }
    let imm = val & 0xFFF;
    let field = ((imm >> 5) << 25) | ((imm & 0x1F) << 7);
    #[allow(clippy::cast_possible_truncation)]
    patch32(out, 0x01FF_F07F, field as u32)
}

/// Writes the I-type lo12 immediate into a 4-byte instruction cell (the
/// `addi`/`jalr` half of a PCREL pair). The low 12 bits of `val` occupy cell
/// bits [31..20]; the low 20 bits (opcode, rd, funct3, rs1) are preserved.
fn write_lo12_i(out: &mut [u8], val: u64) -> Result<()> {
    if out.len() != 4 {
        return Err(Error::OutOfRange("RISC-V lo12_i relocation slot"));
    }
    let field = (val & 0xFFF) << 20;
    #[allow(clippy::cast_possible_truncation)]
    patch32(out, 0x000F_FFFF, field as u32)
}

/// Whether a branch or jump displacement can be encoded at all.
///
/// Every RISC-V branch field drops bit 0, so an odd displacement cannot be
/// represented: the encoders would silently store `val - 1` and the branch
/// would land one byte short of its target. Instructions are 2-byte aligned,
/// so an odd value means the input asked for something impossible -- an odd
/// addend, or a symbol placed at an odd address by hand-written assembly.
/// lld rejects the same values with `checkAlignment(.., 2, ..)`
/// (`lld/ELF/Arch/RISCV.cpp`).
const fn check_even(val: u64, r_type: u32) -> Result<()> {
    if val & 1 != 0 {
        return Err(Error::RelocMisaligned(r_type));
    }
    Ok(())
}

/// Writes the J-type immediate of `JAL` into a 4-byte instruction cell. The
/// 21-bit signed byte offset is scattered as imm[20] at [31], imm[10..1] at
/// [30..21], imm[11] at [20], imm[19..12] at [19..12]; bit 0 is dropped (2-byte
/// alignment). Bits [11..0] (rd + opcode) are preserved.
fn write_jal(out: &mut [u8], val: u64, r_type: u32) -> Result<()> {
    if out.len() != 4 {
        return Err(Error::OutOfRange("RISC-V jal relocation slot"));
    }
    if !fits_signed(val.cast_signed(), 21) {
        return Err(Error::RelocOverflow(r_type));
    }
    check_even(val, r_type)?;
    let field = (((val >> 20) & 0x1) << 31)
        | (((val >> 1) & 0x3FF) << 21)
        | (((val >> 11) & 0x1) << 20)
        | (((val >> 12) & 0xFF) << 12);
    #[allow(clippy::cast_possible_truncation)]
    patch32(out, 0x0FFF, field as u32)
}

/// Writes the B-type immediate of a conditional branch into a 4-byte
/// instruction cell. The 13-bit signed byte offset is scattered as imm[12] at
/// [31], imm[10..5] at [30..25], imm[4..1] at [11..8], imm[11] at [7]; bit 0 is
/// dropped. Bits [6..0] (opcode), [11..7] outside imm[4..1]/imm[11], rs1, rs2
/// and funct3 are preserved.
fn write_branch(out: &mut [u8], val: u64, r_type: u32) -> Result<()> {
    if out.len() != 4 {
        return Err(Error::OutOfRange("RISC-V branch relocation slot"));
    }
    if !fits_signed(val.cast_signed(), 13) {
        return Err(Error::RelocOverflow(r_type));
    }
    check_even(val, r_type)?;
    let field = (((val >> 12) & 0x1) << 31)
        | (((val >> 5) & 0x3F) << 25)
        | (((val >> 1) & 0xF) << 8)
        | (((val >> 11) & 0x1) << 7);
    #[allow(clippy::cast_possible_truncation)]
    patch32(out, 0x01FF_F07F, field as u32)
}

/// Writes the 9-bit immediate of a compressed conditional branch (`c.beqz`/
/// `c.bnez`, CB-type) into a 2-byte instruction cell. The signed byte offset is
/// scattered as imm[8] at [12], imm[4..3] at [11..10], imm[7..6] at [6..5],
/// imm[2..1] at [4..3], imm[5] at [2]; bit 0 is dropped (2-byte alignment).
/// Bits [15..13] and [1..0] (opcode and funct3 parts) are preserved.
fn write_cbtype(out: &mut [u8], val: u64, r_type: u32) -> Result<()> {
    if out.len() != 2 {
        return Err(Error::OutOfRange("RISC-V c.b relocation slot"));
    }
    if !fits_signed(val.cast_signed(), 9) {
        return Err(Error::RelocOverflow(r_type));
    }
    check_even(val, r_type)?;
    let insn = u16::from_le_bytes([out[0], out[1]]);
    let field = (((val >> 8) & 0x1) << 12)
        | (((val >> 3) & 0x3) << 10)
        | (((val >> 6) & 0x3) << 5)
        | (((val >> 1) & 0x3) << 3)
        | (((val >> 5) & 0x1) << 2);
    #[allow(clippy::cast_possible_truncation)]
    let patched = (insn & 0xE383) | (field as u16);
    out.copy_from_slice(&patched.to_le_bytes());
    Ok(())
}

/// Writes the 11-bit immediate of a compressed jump (`c.j`, CJ-type) into a
/// 2-byte instruction cell. The signed byte offset is scattered as imm[11] at
/// [12], imm[4] at [11], imm[9..8] at [10..9], imm[10] at [8], imm[6] at [7],
/// imm[7] at [6], imm[3..1] at [5..3], imm[5] at [2]; bit 0 is dropped.
/// Bits [15..13] and [1..0] (opcode) are preserved.
fn write_cjtype(out: &mut [u8], val: u64, r_type: u32) -> Result<()> {
    if out.len() != 2 {
        return Err(Error::OutOfRange("RISC-V c.j relocation slot"));
    }
    if !fits_signed(val.cast_signed(), 12) {
        return Err(Error::RelocOverflow(r_type));
    }
    check_even(val, r_type)?;
    let insn = u16::from_le_bytes([out[0], out[1]]);
    let field = (((val >> 11) & 0x1) << 12)
        | (((val >> 4) & 0x1) << 11)
        | (((val >> 8) & 0x3) << 9)
        | (((val >> 10) & 0x1) << 8)
        | (((val >> 6) & 0x1) << 7)
        | (((val >> 7) & 0x1) << 6)
        | (((val >> 1) & 0x7) << 3)
        | (((val >> 5) & 0x1) << 2);
    #[allow(clippy::cast_possible_truncation)]
    let patched = (insn & 0xE003) | (field as u16);
    out.copy_from_slice(&patched.to_le_bytes());
    Ok(())
}

/// Writes the paired `auipc`+`jalr` sequence of `CALL`/`CALL_PLT` into an
/// 8-byte slot. The auipc (bytes [0..4]) takes the U-type hi20 of `val` with
/// the `+0x800` compensation; the jalr (bytes [4..8]) takes the I-type lo12
/// (`val & 0xfff`). The high half must fit signed 20 bits.
fn write_call(out: &mut [u8], val: u64, r_type: u32) -> Result<()> {
    if out.len() != 8 {
        return Err(Error::OutOfRange("RISC-V call relocation slot"));
    }
    write_hi20(&mut out[..4], val, r_type)?;
    write_lo12_i(&mut out[4..], val)
}

/// Read-modify-write dispatcher for the ADD/SUB and SET6/SUB6 relocations.
/// `val` is `S + A`. The slot width and the operation are fixed by `r_type`;
/// see [`add_sub_le`] and [`set_sub6`] for the per-width semantics.
fn apply_rmw(r_type: u32, out: &mut [u8], val: u64) -> Result<()> {
    match r_type {
        R_RISCV_ADD8 => add_sub_le(out, 1, val, false),
        R_RISCV_ADD16 => add_sub_le(out, 2, val, false),
        R_RISCV_ADD32 => add_sub_le(out, 4, val, false),
        R_RISCV_ADD64 => add_sub_le(out, 8, val, false),
        R_RISCV_SUB8 => add_sub_le(out, 1, val, true),
        R_RISCV_SUB16 => add_sub_le(out, 2, val, true),
        R_RISCV_SUB32 => add_sub_le(out, 4, val, true),
        R_RISCV_SUB64 => add_sub_le(out, 8, val, true),
        R_RISCV_SET6 => set_sub6(out, val, false),
        R_RISCV_SUB6 => set_sub6(out, val, true),
        R_RISCV_SET8 => set_le(out, 1, val),
        R_RISCV_SET16 => set_le(out, 2, val),
        R_RISCV_SET32 => {
            // The only SET that can overflow its slot. SET6, SET8 and SET16
            // write a label difference the assembler already sized -- lld
            // range-checks none of them -- but SET32 shares its width with
            // `32_PCREL` and `PLT32`, and lld checks all three together
            // (`lld/ELF/Arch/RISCV.cpp`). Without the check a value
            // past 2^31 is stored truncated, so the difference the section
            // records is not the difference between the two labels.
            if !fits_signed(val.cast_signed(), 32) {
                return Err(Error::RelocOverflow(r_type));
            }
            set_le(out, 4, val)
        }
        other => Err(Error::UnsupportedReloc(other)),
    }
}

/// Adds (`sub == false`, `R_RISCV_ADD*`) or subtracts (`sub == true`,
/// `R_RISCV_SUB*`) `val` to the little-endian `n`-byte value at `out`, in
/// place, modulo 2^(8n). The slot must be exactly `n` bytes wide.
fn add_sub_le(out: &mut [u8], n: usize, val: u64, sub: bool) -> Result<()> {
    if out.len() != n {
        return Err(Error::OutOfRange("RISC-V add/sub relocation slot"));
    }
    let cur = read_le(out, n);
    let next = if sub {
        cur.wrapping_sub(val)
    } else {
        cur.wrapping_add(val)
    };
    write_le(out, n, next);
    Ok(())
}

/// Overwrites the little-endian `n`-byte slot at `out` with the low `8*n` bits
/// of `val` (`R_RISCV_SET8/16/32`). Unlike `ADD`/`SUB` this is a plain store,
/// not read-modify-write: the slot's previous contents are discarded, matching
/// lld and mold (`*loc = S + A`). The slot must be exactly `n` bytes wide.
fn set_le(out: &mut [u8], n: usize, val: u64) -> Result<()> {
    if out.len() != n {
        return Err(Error::OutOfRange("RISC-V set relocation slot"));
    }
    write_le(out, n, val);
    Ok(())
}

/// Sets (`sub == false`, `R_RISCV_SET6`) or subtracts-from (`sub == true`,
/// `R_RISCV_SUB6`) the low 6 bits of the byte at `out[0]`, leaving the high 2
/// bits untouched. `val` is `S + A`; SET6 writes `val & 0x3f`, SUB6 writes
/// `((byte & 0x3f) - val) & 0x3f`, matching lld.
fn set_sub6(out: &mut [u8], val: u64, sub: bool) -> Result<()> {
    if out.len() != 1 {
        return Err(Error::OutOfRange("RISC-V set6/sub6 relocation slot"));
    }
    let byte = u64::from(out[0]);
    let low = if sub {
        (byte & 0x3f).wrapping_sub(val) & 0x3f
    } else {
        val & 0x3f
    };
    // The masked value is at most 0xc0 | 0x3f = 0xff, so truncation is sound.
    #[allow(clippy::cast_possible_truncation)]
    let merged = ((byte & 0xc0) | low) as u8;
    out[0] = merged;
    Ok(())
}

// --- PCREL_HI20 / PCREL_LO12 paired resolution ---------------------------

/// Rewrites each `R_RISCV_PCREL_LO12_I`/`_S` entry so it carries its paired
/// `R_RISCV_PCREL_HI20`'s target symbol, letting the arch-neutral apply path
/// resolve it without access to the relocation list.
///
/// The LO12 half of a PCREL pair points its symbol at the `auipc` instruction
/// (a local label), not at the real target. That label's value, relative to
/// the section base, is the HI20 site. The pass indexes every HI20 site, then
/// for each LO12 rewrites the symbol to its paired HI20's symbol and shifts
/// the addend by `(lo_offset - hi_offset)`: the `Pc` expression evaluated at
/// the LO12's own place `P_lo = base + lo_offset` then yields
/// `S_hi + A_hi - P_hi` (the value the auipc encoded), whose low 12 bits the
/// LO12 must store.
///
/// Returns `Ok(None)` when the section has no paired LO12, so the caller keeps
/// the borrowed slice in the common case and avoids cloning. A LO12 with no
/// matching HI20 is malformed input and yields an error rather than risk
/// writing wrong bytes.
/// Whether `r_type` is an `auipc` a `PCREL_LO12_*` can be paired with.
///
/// The LO12 half names the instruction rather than the target, so the pairing
/// pass has to recognise every flavour of the instruction: the plain
/// PC-relative one, the GOT-indirect one `-fPIC` code uses for extern data,
/// and the two dynamic-TLS ones. lld's `RISCVPCRel::isHiReloc` lists the same
/// four (`InputSection.cpp:668`).
const fn is_hi20(r_type: u32) -> bool {
    matches!(
        r_type,
        R_RISCV_PCREL_HI20
            | R_RISCV_GOT_HI20
            | R_RISCV_TLS_GD_HI20
            | R_RISCV_TLS_GOT_HI20
    )
}

pub fn rewrite_pcrel_pairs<R: Resolver>(
    entries: &[Rela64],
    section_base: u64,
    resolver: &R,
) -> Result<Option<Vec<Rela64>>> {
    // Cheap pre-scan: skip sections with no PCREL relocation at all, so the
    // common case pays neither the map nor the clone. A section with a LO12
    // but no HI20 still falls through to the validation below and errors.
    let has_pcrel = entries.iter().any(|r| {
        let t = r.r_type();
        is_hi20(t)
            || t == R_RISCV_PCREL_LO12_I
            || t == R_RISCV_PCREL_LO12_S
            || t == R_RISCV_SET_ULEB128
            || t == R_RISCV_SUB_ULEB128
    });
    if !has_pcrel {
        return Ok(None);
    }
    // Pass 0: fold each `SET_ULEB128`/`SUB_ULEB128` pair into its `SET`
    // entry. The two share an offset and encode one difference, which no
    // single relocation expression spells, so the difference is computed here
    // -- where both are in hand -- and left in the `SET` entry's addend
    // against no symbol. The `SUB` entry becomes `R_RISCV_NONE`, which the
    // writer skips. lld folds the pair at the same point, in its own apply
    // loop (`lld/ELF/Arch/RISCV.cpp`).
    let mut out = Vec::from(entries);
    let mut rewrote = fold_uleb_pairs(&mut out, resolver)?;
    // Pass 1: index HI20 sites by their offset within the section. Every type
    // a LO12 can point at belongs here -- lld keeps the same four in
    // `RISCVPCRel::isHiReloc` -- because the LO12's symbol names the
    // instruction, not the target, whichever flavour of `auipc` it is.
    //
    // Borrowed rather than copied: the index is built from `out` and finished
    // with before pass 2 takes it mutably, so a second copy of every
    // relocation in the section buys nothing.
    let mut hi20: FxHashMap<u64, (u32, i64)> = FxHashMap::default();
    for r in &out {
        if is_hi20(r.r_type()) {
            hi20.insert(r.r_offset.get(), (r.sym(), r.r_addend.get()));
        }
    }
    // Pass 2: rewrite each LO12 to its paired HI20's target.
    for r in &mut out {
        let is_lo12 = r.r_type() == R_RISCV_PCREL_LO12_I
            || r.r_type() == R_RISCV_PCREL_LO12_S;
        if !is_lo12 {
            continue;
        }
        let sym = r.sym();
        if sym == 0 {
            return Err(Error::Format(
                "R_RISCV_PCREL_LO12_* has no paired HI20 (null symbol)",
            ));
        }
        let sym_id = SymbolId(usize::try_from(sym).unwrap_or(0));
        // The LO12 names the `auipc` it pairs with, and it names it the way
        // any other reference spells a location: symbol plus addend. A tool
        // that drops the local label the assembler emitted -- `objcopy -x`,
        // `ld -r` -- rewrites the reference against the section symbol and
        // moves the whole offset into the addend, so leaving the addend out
        // looks the pair up at the section's start. lld folds it in the same
        // place (`lld/ELF/InputSection.cpp`, `d->value + addend`).
        let hi_off = resolver
            .symbol_addr(sym_id)
            .wrapping_add_signed(r.r_addend.get())
            .wrapping_sub(section_base);
        let Some(&(hi_sym, hi_addend)) = hi20.get(&hi_off) else {
            return Err(Error::Format(
                "R_RISCV_PCREL_LO12_* has no paired HI20 at its label",
            ));
        };
        let adjusted = hi_addend
            .wrapping_add(r.r_offset.get().cast_signed())
            .wrapping_sub(hi_off.cast_signed());
        r.r_info = U64::new(pack_info(hi_sym, r.r_type()));
        r.r_addend = I64::new(adjusted);
        rewrote = true;
    }
    Ok(rewrote.then_some(out))
}

/// The most bytes a ULEB128 needs for a 64-bit value.
const ULEB_MAX: usize = 10;

/// Rewrites the ULEB128 at the start of `field` to `value`, keeping the length
/// it already has.
///
/// The length is not the linker's to choose: the field sits inside a table
/// whose other entries follow it, so growing or shrinking it would move them.
/// Every byte but the last keeps its continuation bit and takes seven bits of
/// the value; the last takes what is left, and a value with more bits than fit
/// there is one this field cannot hold. lld does the same and reports the same
/// overflow (`overwriteULEB128`, checked against `0x80` at
/// `lld/ELF/Arch/RISCV.cpp`).
fn write_uleb_in_place(field: &mut [u8], value: u64) -> Result<()> {
    let Some((last, rest)) = field.split_last_mut() else {
        return Err(Error::OutOfRange("RISC-V ULEB128 field"));
    };
    let mut left = value;
    for byte in rest {
        #[allow(clippy::cast_possible_truncation)]
        let seven = (left & 0x7f) as u8;
        *byte = 0x80 | seven;
        left >>= 7;
    }
    if left >= 0x80 {
        return Err(Error::RelocOverflow(R_RISCV_SET_ULEB128));
    }
    #[allow(clippy::cast_possible_truncation)]
    {
        *last = left as u8;
    }
    Ok(())
}

/// Folds every `SET_ULEB128`/`SUB_ULEB128` pair in `entries` into its `SET`
/// entry, and reports whether any was folded.
///
/// The pair is matched by offset, as lld matches it: the two describe one
/// field, and the `SUB` is required to follow the `SET` there. An unpaired
/// `SET` is a malformed input rather than something to guess at -- the field
/// would be written with a subtrahend of nothing -- so it ends the link.
fn fold_uleb_pairs<R: Resolver>(
    entries: &mut [Rela64],
    resolver: &R,
) -> Result<bool> {
    let mut folded = false;
    for i in 0..entries.len() {
        let Some(set) = entries.get(i) else { break };
        if set.r_type() != R_RISCV_SET_ULEB128 {
            continue;
        }
        let at = set.r_offset.get();
        let minuend = uleb_term(set, resolver);
        let Some(j) = entries.iter().position(|r| {
            r.r_type() == R_RISCV_SUB_ULEB128 && r.r_offset.get() == at
        }) else {
            return Err(Error::Format(
                "R_RISCV_SET_ULEB128 without a paired R_RISCV_SUB_ULEB128",
            ));
        };
        let Some(sub) = entries.get(j) else { break };
        let value = minuend.wrapping_sub(uleb_term(sub, resolver));
        if let Some(slot) = entries.get_mut(i) {
            slot.r_info = U64::new(pack_info(0, R_RISCV_SET_ULEB128));
            slot.r_addend = I64::new(value.cast_signed());
        }
        if let Some(slot) = entries.get_mut(j) {
            slot.r_info = U64::new(pack_info(0, R_RISCV_NONE));
            slot.r_addend = I64::new(0);
        }
        folded = true;
    }
    Ok(folded)
}

/// One side of a ULEB128 pair: `S + A`.
fn uleb_term<R: Resolver>(r: &Rela64, resolver: &R) -> u64 {
    let sym = SymbolId(usize::try_from(r.sym()).unwrap_or(0));
    resolver
        .symbol_addr(sym)
        .wrapping_add_signed(r.r_addend.get())
}

/// Packs `(sym, r_type)` into an ELF64 `r_info` word.
#[allow(clippy::cast_possible_truncation)]
fn pack_info(sym: u32, r_type: u32) -> u64 {
    (u64::from(sym) << 32) | u64::from(r_type)
}

/// Reads a 4-byte little-endian cell from `cell` (a 4-byte slice), replaces the
/// bits selected by `keep` (1 = preserve, 0 = overwrite) with `field`, and
/// writes the cell back.
fn patch32(cell: &mut [u8], keep: u32, field: u32) -> Result<()> {
    let buf = cell
        .get(..4)
        .ok_or(Error::OutOfRange("RISC-V instruction cell"))?;
    let insn = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let patched = (insn & keep) | field;
    cell[..4].copy_from_slice(&patched.to_le_bytes());
    Ok(())
}
