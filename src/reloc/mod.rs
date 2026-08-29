//! Architecture-neutral relocation handling, driven by a declarative table.
//!
//! Each raw relocation type is mapped, once per architecture, to a [`Spec`]: a
//! semantic [`RelExpr`] that fixes how the value is computed, paired with a
//! [`Write`] that fixes how it is stored. The scan and apply phases are then
//! written once, here, and shared by every architecture. This single seam is
//! what keeps the three hand-synchronised per-arch switches of lld and mold
//! from being copied: the table is data, the drivers are code.
//!
//! Only the value machinery lives here. Address resolution and GOT/PLT
//! allocation are supplied by a [`Resolver`]: the writer provides one over the
//! laid-out output, and tests supply fixed values.
//!
//! # Storage model
//!
//! [`Write`] comes in two flavours. [`Write::Bytes`] is a plain little-endian
//! whole-byte store (x86-64, and the ABS/PREL width relocations on every arch).
//! [`Write::Field`] patches a *contiguous bitfield* inside a little-endian
//! instruction cell: the driver reads the cell, replaces the field bits with
//! the computed value, and writes the cell back, leaving opcode bits untouched.
//! This is how `AArch64` stores immediates such as the 26-bit branch offset,
//! the 19-bit condition code target, or the 12-bit load/store displacement.
//!
//! A few `AArch64` immediates are *split* across two non-adjacent bit ranges
//! (the 21-bit ADR/ADRP pair, scattered as immlo at bits [29..30] and immhi at
//! bits [5..23]). One [`Write::Field`] cannot describe two masks, so those
//! relocations are routed through [`RelExpr::Escape`] together with
//! [`Arch::escape`], which computes the value reusing the [`RelExpr`]
//! arithmetic and writes the split encoding itself. That is exactly what the
//! escape seam exists for.

pub mod aarch64;
pub mod coff_i386;
pub mod coff_x86_64;
pub mod macho_arm64;
pub mod macho_x86_64;
pub mod riscv;
pub mod x86_64;

use std::ops::Range;

use crate::{
    elf::{
        Rela64,
        constants::{EM_AARCH64, EM_RISCV, EM_X86_64},
    },
    error::{Error, Result},
    symbol::SymbolId,
};

/// The semantic meaning of a relocation, after lld's `RelExpr`.
///
/// A variant fixes the arithmetic used to compute the stored value. `S` is the
/// symbol address, `A` the addend, `P` the place being patched, `GOT[sym]` the
/// address of the symbol's GOT entry, and `GOT` the GOT base address.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelExpr {
    /// No effect on the output (e.g. `R_X86_64_NONE`).
    None,
    /// `S + A`.
    Abs,
    /// `S + A - P`.
    Pc,
    /// `GOT[sym] + A`.
    Got,
    /// `GOT[sym] + A - GOT`: the slot's offset from the table's base, `G + A`
    /// in the psABI. The number a register holding
    /// `_GLOBAL_OFFSET_TABLE_` is added to, which is how the large code
    /// model addresses the table (`R_X86_64_GOT32`/`GOT64`).
    GotOffset,
    /// `GOT[sym] + A - P`.
    GotPc,
    /// `TLSGOT[sym] + A - P`: PC-relative reference to the GOT slot holding
    /// the symbol's offset from the thread pointer, which is the initial-exec
    /// TLS access.
    TlsGotPc,
    /// `TLSINDEX + A - P`: PC-relative reference to this image's own module
    /// GOT entry, which the local-dynamic model hands to `__tls_get_addr`.
    TlsIndexPc,
    /// `S - DTP + A`: the symbol's offset from the base a local-dynamic
    /// sequence loaded.
    DtpOff,
    /// `S + A - GOT`.
    GotOff,
    /// `GOT + A - P`.
    GotBase,
    /// `PLT[sym] + A`. In a static link the PLT collapses to the symbol.
    Plt,
    /// `PLT[sym] + A - P`. Collapses to `S + A - P` statically.
    PltPc,
    /// `Z + A`: the size of the symbol's definition, not its address. Used by
    /// `R_X86_64_SIZE32`/`SIZE64`, which a compiler emits for `sizeof` of an
    /// object it cannot see the definition of.
    Size,
    /// No portable value: the architecture handles it via [`Arch::escape`],
    /// used for irreducible cases such as split bitfields or multi-instruction
    /// TLS sequences.
    Escape,
}

impl RelExpr {
    /// Computes the modular 64-bit value of this expression. The result is
    /// range-checked against the write kind by [`apply`]. The null symbol
    /// (`sym == None`) contributes address zero, so a pure-addend absolute
    /// reference yields just the addend.
    pub fn compute<R: Resolver>(
        self,
        sym: Option<SymbolId>,
        addend: i64,
        place: u64,
        resolver: &R,
    ) -> Result<u64> {
        // The addend is folded in modular 64-bit arithmetic, matching
        // pointer-width wraparound on the target.
        let a = addend.cast_unsigned();
        // Each lookup lives in the arm that uses it. Loading the GOT, PLT and
        // GOT-base addresses ahead of the match asked three questions of the
        // resolver for every plain `Abs` and `Pc` site, which is the hot path:
        // whether the optimiser sank them was left to inlining rather than
        // stated.
        let sym_addr = || sym.map_or(0, |id| resolver.symbol_addr(id));
        let got_addr = || sym.map_or(0, |id| resolver.got_addr(id));
        let plt_addr = || sym.map_or(0, |id| resolver.plt_addr(id));
        let value = match self {
            Self::Abs => sym_addr().wrapping_add(a),
            Self::Pc => sym_addr().wrapping_add(a).wrapping_sub(place),
            Self::Got => got_addr().wrapping_add(a),
            Self::GotOffset => {
                got_addr().wrapping_sub(resolver.got_base()).wrapping_add(a)
            }
            Self::GotPc => got_addr().wrapping_add(a).wrapping_sub(place),
            Self::TlsGotPc => sym
                .map_or(0, |id| resolver.tls_got_addr(id))
                .wrapping_add(a)
                .wrapping_sub(place),
            // No module entry means the link expected this reference to have
            // been rewritten. Refuse rather than store an offset from zero.
            Self::TlsIndexPc => match resolver.tls_index_addr() {
                0 => return Err(Error::UnsupportedReloc(0)),
                index => index.wrapping_add(a).wrapping_sub(place),
            },
            Self::DtpOff => {
                sym.map_or(0, |id| resolver.tls_dtp_off(id)).wrapping_add(a)
            }
            Self::GotOff => {
                sym_addr().wrapping_add(a).wrapping_sub(resolver.got_base())
            }
            Self::GotBase => {
                resolver.got_base().wrapping_add(a).wrapping_sub(place)
            }
            Self::Plt => plt_addr().wrapping_add(a),
            Self::PltPc => plt_addr().wrapping_add(a).wrapping_sub(place),
            Self::Size => {
                sym.map_or(0, |id| resolver.sym_size(id)).wrapping_add(a)
            }
            Self::None => 0,
            Self::Escape => return Err(Error::UnsupportedReloc(0)),
        };
        Ok(value)
    }

    /// Which entries this expression forces the scan to allocate for its
    /// symbol.
    pub const fn needs(self) -> Needs {
        match self {
            Self::Got | Self::GotOffset | Self::GotPc => Needs {
                got: true,
                ..Needs::NONE
            },
            Self::TlsGotPc => Needs {
                tls_got: true,
                ..Needs::NONE
            },
            Self::Plt | Self::PltPc => Needs {
                plt: true,
                ..Needs::NONE
            },
            Self::Size => Needs {
                size: true,
                ..Needs::NONE
            },
            _ => Needs::NONE,
        }
    }
}

/// How a computed value is stored into the output bytes.
///
/// `Bytes` covers whole-byte little-endian stores (x86-64, the ABS/PREL
/// widths). `Field` patches a contiguous bitfield of a fixed-width instruction
/// cell (`AArch64` immediates); the driver reads the cell, applies the field,
/// and writes the cell back so opcode bits outside the field survive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Write {
    /// A whole-byte little-endian store.
    Bytes(WriteKind),
    /// A contiguous bitfield written into a little-endian instruction cell.
    Field(Field),
}

impl Write {
    /// The number of output bytes this store touches: the byte width, or the
    /// instruction cell width.
    pub fn width(self) -> usize {
        match self {
            Self::Bytes(k) => k.width(),
            Self::Field(f) => usize::from(f.cell_bytes),
        }
    }
}

/// A whole-byte little-endian store kind.
///
/// The width fixes the number of bytes; the variant fixes the range check. All
/// encodings are little-endian, matching x86-64, `AArch64` and RISC-V.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WriteKind {
    /// One byte, unsigned.
    W8,
    /// One byte, signed.
    W8S,
    /// One byte, signed or unsigned: the value is accepted if it fits either
    /// 8-bit range, after lld's `checkIntUInt`.
    W8SU,
    /// Two bytes, unsigned.
    W16,
    /// Two bytes, signed.
    W16S,
    /// Two bytes, signed or unsigned: the value is accepted if it fits either
    /// 16-bit range, after lld's `checkIntUInt`.
    W16SU,
    /// Four bytes, unsigned.
    W32,
    /// Four bytes, signed.
    W32S,
    /// Four bytes, signed or unsigned: the value is accepted if it fits either
    /// 32-bit range, after lld's `checkIntUInt`. This is the check the
    /// `AArch64` ABS/PREL widths call for, where a negative offset and a large
    /// unsigned address are both legal in the same slot.
    W32SU,
    /// Eight bytes, full 64-bit.
    W64,
}

impl WriteKind {
    /// The number of output bytes this kind occupies.
    pub const fn width(self) -> usize {
        match self {
            Self::W8 | Self::W8S | Self::W8SU => 1,
            Self::W16 | Self::W16S | Self::W16SU => 2,
            Self::W32 | Self::W32S | Self::W32SU => 4,
            Self::W64 => 8,
        }
    }

    /// Whether the modular value fits this kind's range.
    const fn fits(self, value: u64) -> bool {
        match self {
            Self::W64 => true,
            Self::W32 => value <= u32::MAX as u64,
            Self::W16 => value <= u16::MAX as u64,
            Self::W8 => value <= u8::MAX as u64,
            Self::W32S => fits_signed(value, 32),
            Self::W16S => fits_signed(value, 16),
            Self::W8S => fits_signed(value, 8),
            Self::W32SU => value <= u32::MAX as u64 || fits_signed(value, 32),
            Self::W16SU => value <= u16::MAX as u64 || fits_signed(value, 16),
            Self::W8SU => value <= u8::MAX as u64 || fits_signed(value, 8),
        }
    }

    /// Writes `value` into `out`, which must be exactly [`width`](Self::width)
    /// bytes long. The caller must have checked [`fits`](Self::fits).
    fn store(self, value: u64, out: &mut [u8]) {
        match self {
            // Truncation is sound: `fits` has bounded the value to this width.
            #[allow(clippy::cast_possible_truncation)]
            Self::W8 | Self::W8S | Self::W8SU => out[0] = value as u8,
            #[allow(clippy::cast_possible_truncation)]
            Self::W16 | Self::W16S | Self::W16SU => {
                out.copy_from_slice(&(value as u16).to_le_bytes());
            }
            #[allow(clippy::cast_possible_truncation)]
            Self::W32 | Self::W32SU => {
                out.copy_from_slice(&(value as u32).to_le_bytes());
            }
            #[allow(clippy::cast_possible_truncation)]
            Self::W32S => out.copy_from_slice(&(value as i32).to_le_bytes()),
            Self::W64 => out.copy_from_slice(&value.to_le_bytes()),
        }
    }
}

/// Whether the modular `value` is a sign-extended `bits`-bit signed integer,
/// after lld's `checkInt`. `bits` must be in `1..64`.
const fn fits_signed(value: u64, bits: u32) -> bool {
    // `1 << (bits - 1)` is in range for `bits` below 64.
    let bound = 1i64 << (bits - 1);
    let signed = value.cast_signed();
    signed >= -bound && signed < bound
}

/// The range check applied to a [`Field`] store before the value is encoded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Check {
    /// No check: the value is truncated to the field width. Used for `_NC`
    /// (no-check) relocations that by definition cannot overflow.
    None,
    /// The value must fit the field as an unsigned `width + right_shift` bit
    /// integer.
    Unsigned,
    /// The value must fit the field as a signed `width + right_shift` bit
    /// integer.
    Signed,
}

/// A declarative description of a contiguous bitfield inside a little-endian
/// instruction cell, after lld's `writeMaskedBits32le`.
///
/// The cell is read as a little-endian unsigned integer, the field bits are
/// replaced with the encoded value, and the cell is written back. The encoded
/// bits are `((value >> right_shift) & ((1 << width) - 1)) << left_shift`, and
/// the in-cell mask is `((1 << width) - 1) << left_shift`; opcode bits outside
/// the mask are preserved.
///
/// `right_shift` is the bits the encoding drops off the bottom of the value.
/// It means one of two things, which is why the alignment requirement is
/// carried separately in `align_bits`:
///
/// - A scale, when the field counts units larger than a byte: 2 for a branch,
///   which counts 4-byte instructions, or the access size of a scaled
///   load/store displacement. The dropped bits must be zero, so the field is
///   built with [`Field::scaled`].
/// - A slice offset, when the field holds one window of a wider value: the
///   `MOVW` 16-bit slices and the `TLSLE_ADD_TPREL_HI12` high half. The dropped
///   bits belong to another instruction, so the value is under no alignment
///   constraint and the field is built with [`Field::new`].
///
/// `left_shift` is the bit position of the field's lowest bit inside the cell.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Field {
    /// Width of the instruction cell in bytes (4 for an `AArch64`
    /// instruction).
    pub cell_bytes: u8,
    /// Low bits dropped from the value before encoding.
    pub right_shift: u8,
    /// Bit position of the field's lowest bit within the cell.
    pub left_shift: u8,
    /// Field width in value bits.
    pub width: u8,
    /// Range check applied to the value before encoding.
    pub check: Check,
    /// The value must be a multiple of `1 << align_bits`. Zero for a field
    /// that imposes no alignment.
    pub align_bits: u8,
}

impl Field {
    /// Builds a field whose `right_shift` selects a slice of a wider value and
    /// so constrains nothing about the bits below it.
    pub const fn new(
        cell_bytes: u8,
        right_shift: u8,
        left_shift: u8,
        width: u8,
        check: Check,
    ) -> Self {
        Self {
            cell_bytes,
            right_shift,
            left_shift,
            width,
            check,
            align_bits: 0,
        }
    }

    /// Builds a field whose `scale` is the unit the encoding counts in, so the
    /// value must be a multiple of `1 << scale`.
    pub const fn scaled(
        cell_bytes: u8,
        scale: u8,
        left_shift: u8,
        width: u8,
        check: Check,
    ) -> Self {
        Self {
            cell_bytes,
            right_shift: scale,
            left_shift,
            width,
            check,
            align_bits: scale,
        }
    }

    /// Whether `value` is a multiple of the unit this field counts in.
    ///
    /// [`store`](Self::store) shifts the value right by `right_shift` before
    /// encoding it, so a value with those bits set would be silently truncated
    /// to a different target. Mirrors lld's `checkAlignment`, which the
    /// `AArch64` branch and scaled load/store relocations call alongside their
    /// range check. A field that imposes no alignment (`align_bits == 0`)
    /// accepts everything, so an unscaled field is never rejected.
    pub fn aligned(self, value: u64) -> bool {
        if self.align_bits == 0 {
            return true;
        }
        // Widths in use are all < 64, so the mask does not overflow.
        value & ((1u64 << self.align_bits) - 1) == 0
    }

    /// Whether `value` satisfies this field's range check. The check is
    /// applied to the unscaled value against `width + right_shift` bits, so a
    /// branch field of `width = 26, right_shift = 2, check = Signed` accepts
    /// exactly the signed 28-bit byte offsets that lld does.
    pub fn fits(self, value: u64) -> bool {
        let total =
            u32::from(self.width).saturating_add(u32::from(self.right_shift));
        match self.check {
            Check::None => true,
            Check::Unsigned => {
                if total >= 64 {
                    return true;
                }
                value >> total == 0
            }
            Check::Signed => total >= 64 || fits_signed(value, total),
        }
    }

    /// Writes `value` into `out`, which must be exactly `cell_bytes` long. The
    /// caller must have checked [`fits`](Self::fits). Opcode bits outside the
    /// field are preserved.
    pub fn store(self, value: u64, out: &mut [u8]) {
        let n = usize::from(self.cell_bytes);
        let cell = read_le(out, n);
        // Signed fields hold PC-relative offsets stored as two's-complement
        // instruction counts, so the scale shift must be arithmetic.
        let scaled: u64 = match self.check {
            Check::Signed => {
                (value.cast_signed() >> self.right_shift).cast_unsigned()
            }
            Check::None | Check::Unsigned => value >> self.right_shift,
        };
        // Widths in use are all < 64, so the mask does not overflow.
        let width_mask: u64 = (1u64 << self.width) - 1;
        let mask = width_mask << self.left_shift;
        let field = (scaled & width_mask) << self.left_shift;
        let patched = (cell & !mask) | field;
        write_le(out, n, patched);
    }
}

/// Reads the first `n` bytes of `buf` as a little-endian unsigned integer.
fn read_le(buf: &[u8], n: usize) -> u64 {
    let mut wide = [0u8; 8];
    wide[..n].copy_from_slice(&buf[..n]);
    u64::from_le_bytes(wide)
}

/// Writes `value` to the first `n` bytes of `buf` as a little-endian integer.
fn write_le(buf: &mut [u8], n: usize, value: u64) {
    buf[..n].copy_from_slice(&value.to_le_bytes()[..n]);
}

/// One architecture's classification of a raw relocation type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Spec {
    /// How the value is computed.
    pub expr: RelExpr,
    /// How the value is stored.
    pub write: Write,
}

/// Builds a spec from its expression and write.
const fn of(expr: RelExpr, write: Write) -> Spec {
    Spec { expr, write }
}

/// Builds a whole-byte spec from its expression and width.
const fn bytes_of(expr: RelExpr, write: WriteKind) -> Spec {
    of(expr, Write::Bytes(write))
}

/// The placeholder field for an escape relocation: the driver routes
/// [`RelExpr::Escape`] to [`Arch::escape`] before the field is stored, so
/// only `cell_bytes` (the slot width) is observed, by the writer when it
/// sizes the relocation slot.
const fn insn_cell(cell_bytes: u8) -> Field {
    Field::new(cell_bytes, 0, 0, 0, Check::None)
}

/// A 26-bit branch target at instruction bits [0..26]; the byte offset is
/// divided by 4 (the instruction scale) and range-checked as signed.
const fn branch26() -> Field {
    Field::scaled(4, 2, 0, 26, Check::Signed)
}

/// The `AArch64` page of `addr`, defined as `addr & !0xfff` per the psABI
/// (a 4 KiB page regardless of the runtime page size).
const fn page(addr: u64) -> u64 {
    addr & !0xFFF
}

/// Allocation an expression forces for its symbol, used by the scan pass.
///
/// Four independent yes-or-no answers about one relocation, read once each by
/// the scan. Named fields are what makes the call sites legible; a flag word
/// would trade that for nothing, since none of them is state and none is
/// carried anywhere.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct Needs {
    /// The symbol needs a GOT entry.
    pub got: bool,
    /// The symbol needs a PLT entry.
    pub plt: bool,
    /// The symbol needs a GOT slot holding its offset from the thread
    /// pointer.
    pub tls_got: bool,
    /// The relocation reads the symbol's `st_size` rather than its address,
    /// so the layout has to build the per-symbol size rows. Carried here
    /// rather than re-derived from the spec, so the scan pays no second table
    /// lookup per relocation.
    pub size: bool,
}

impl Needs {
    /// No allocation required.
    pub const NONE: Self = Self {
        got: false,
        plt: false,
        tls_got: false,
        size: false,
    };
}

/// Supplies the addresses the arch-neutral value computation needs.
///
/// In the link pipeline this is backed by the output layout; a test supplies
/// fixed values.
pub trait Resolver {
    /// `S`: the runtime address of the symbol.
    fn symbol_addr(&self, sym: SymbolId) -> u64;
    /// `GOT[sym]`: the address of the symbol's GOT entry.
    fn got_addr(&self, sym: SymbolId) -> u64;
    /// `GOT`: the base address of the global offset table.
    fn got_base(&self) -> u64;
    /// `PLT[sym]`: the address of the symbol's PLT entry.
    fn plt_addr(&self, sym: SymbolId) -> u64;

    /// The address of this image's own module GOT entry, or 0 if it has
    /// none. The local-dynamic model reads that entry to find the module's
    /// thread-local block.
    fn tls_index_addr(&self) -> u64 {
        0
    }

    /// The symbol's offset from the base a local-dynamic sequence loads: the
    /// module's own block in a shared object, and the thread pointer in an
    /// executable, where that sequence is lowered to read it directly.
    fn tls_dtp_off(&self, sym: SymbolId) -> u64 {
        let _ = sym;
        0
    }

    /// The size of the symbol's definition (`st_size`), or 0 when the link
    /// has no size relocation and so never asked for one.
    fn sym_size(&self, sym: SymbolId) -> u64 {
        let _ = sym;
        0
    }

    /// The address of the GOT slot holding the symbol's offset from the
    /// thread pointer, or 0 if it has none.
    ///
    /// The scan gives a thread-local a slot only when a shared object defines
    /// it, so a non-zero answer is exactly the case where the offset is the
    /// loader's to choose and a general-dynamic sequence must be rewritten to
    /// load it rather than fold it in. The default is 0, for a resolver with
    /// no dynamic TLS.
    fn tls_got_addr(&self, sym: SymbolId) -> u64 {
        let _ = sym;
        0
    }

    /// Whether the image being written is an executable.
    ///
    /// A thread-local of the main module lives in the static TLS block, so its
    /// offset from the thread pointer is fixed at link time; that is what lets
    /// a general-dynamic TLS sequence be rewritten into the local-exec form. A
    /// shared object has no such guarantee, since it may be dlopened. The
    /// default is the conservative answer, so a resolver that does not track
    /// the link mode never triggers a TLS rewrite.
    fn is_exec(&self) -> bool {
        false
    }

    /// Whether the symbol's definition may be replaced at load time by one of
    /// the same name from another image.
    ///
    /// Folding a GOT-indirect access into a direct reference resolves it here
    /// and for good, so it is only sound for a symbol nothing can preempt. The
    /// default is the conservative answer, so a resolver that does not track
    /// preemptibility never relaxes a GOT access.
    fn is_preemptible(&self, sym: SymbolId) -> bool {
        let _ = sym;
        true
    }

    /// Whether the symbol's value is a link-time constant rather than a place
    /// in the image.
    ///
    /// An `SHN_ABS` definition, a reference nothing defines, and a thread-local
    /// (whose value is an offset from the thread pointer) are all constants:
    /// they do not move with the load base, where a PC-relative reference to
    /// them would. Folding a GOT load of one into a direct reference therefore
    /// changes the value, so the rewrite is refused. Mirrors the
    /// `isAbsoluteValue` guard on lld's `adjustGotPcExpr` call. The default is
    /// the conservative answer, so a resolver that does not track this never
    /// relaxes a GOT access.
    fn is_absolute(&self, sym: SymbolId) -> bool {
        let _ = sym;
        true
    }

    /// Whether the scan allocated a GOT slot for the symbol.
    ///
    /// The scan drops every reference to the runtime TLS helper an executable
    /// lowers away, and allocates a slot for each of the GOT-indirect
    /// references that remain. A `false` answer therefore names a reference
    /// nothing was reserved for, and applying it would store a displacement
    /// measured from address zero. The default is the conservative answer, so
    /// a resolver that does not track slots never triggers that mandatory
    /// rewrite.
    fn has_got_slot(&self, sym: SymbolId) -> bool {
        let _ = sym;
        true
    }

    /// Whether the symbol is a weak reference that nothing defines.
    ///
    /// Such a reference is the feature-probe idiom, and what a PC-relative
    /// site against it stores is the architecture's call, not a plain
    /// `S + A - P` with `S = 0`: see [`Arch::undef_weak_pc`]. The default is
    /// the conservative answer, so a resolver that does not track bindings
    /// never takes that path.
    fn is_undef_weak(&self, sym: SymbolId) -> bool {
        let _ = sym;
        false
    }
}

/// One relocation site, as [`Arch::relax_span`] is asked about it.
///
/// The fields are the same ones [`Arch::relax`] receives, plus the section
/// bytes and the slot's offset within them: the span seam runs before any
/// relocation has been applied, so it carves its own window rather than being
/// handed one.
#[derive(Clone, Copy)]
pub struct RelaxSite<'a> {
    /// Raw relocation type (`R_X86_64_*` and similar).
    pub r_type: u32,
    /// The symbol the relocation names, or `None` for the null symbol.
    pub sym: Option<SymbolId>,
    /// The addend to resolve the site with.
    pub addend: i64,
    /// `P`: the runtime address of the slot.
    pub place: u64,
    /// The bytes of the section being patched, as they will be applied over.
    pub data: &'a [u8],
    /// The slot's offset within `data`.
    pub slot: u64,
}

/// The per-architecture relocation table and escape hatch.
///
/// [`Arch::spec`] is the declarative table: the single place that maps a raw
/// type to a [`Spec`], shared by both [`scan`] and [`apply`]. [`Arch::escape`]
/// is the seam for relocations that cannot be reduced to a portable expression
/// (split bitfields such as the ADR/ADRP imm21 pair, multi-instruction TLS
/// sequences, branch-range thunks); its default rejects them, and an
/// architecture overrides it when those relaxations land.
///
/// [`Arch::relax_lead`] and [`Arch::relax`] form the optional relaxation seam:
/// an architecture that can rewrite a relocation site into a cheaper form
/// (for example x86-64 `GOTPCRELX` `mov` -> `lea`) overrides them. Relaxation
/// runs after layout, in the writer's apply loop, and only when the link
/// requests it; with relaxation off the default link is byte-identical.
///
/// [`Arch::relax_span`] is the seam beside it for a rewrite that spans more
/// than one relocation: it reports the bytes the rewrite consumes, so the
/// writer can drop the relocations that land inside them.
pub trait Arch {
    /// Maps a raw relocation type to its spec.
    fn spec(r_type: u32) -> Result<Spec>;

    /// The allocation a relocation type forces for its symbol (the scan phase).
    /// The default derives it from the spec's expression, so most relocations
    /// need no special handling; an architecture overrides this for escape
    /// relocations whose allocation cannot be inferred from [`RelExpr`] alone,
    /// such as a split-immediate GOT reference that needs a GOT entry but is
    /// classified as [`RelExpr::Escape`] for writing.
    fn scan_needs(r_type: u32) -> Result<Needs> {
        Ok(Self::spec(r_type)?.expr.needs())
    }

    /// Handles a relocation whose spec is [`RelExpr::Escape`]. The default
    /// rejects it as unsupported.
    fn escape<R: Resolver>(
        r_type: u32,
        _sym: Option<SymbolId>,
        _addend: i64,
        _place: u64,
        _resolver: &R,
        _out: &mut [u8],
    ) -> Result<()> {
        Err(Error::UnsupportedReloc(r_type))
    }

    /// The value a PC-relative relocation of this type stores when its symbol
    /// is a weak reference nothing defines.
    ///
    /// `S = 0` makes the plain expression `A - P`, which is x86-64's answer
    /// and this default. `AArch64` and RISC-V answer differently, because a
    /// branch field has to land somewhere executable: on `AArch64` a branch
    /// to an undefined weak goes to the next instruction (`4 + A`) and a data
    /// reference takes the place's own address (`A`); on RISC-V a branch goes
    /// to itself (`A`), the deliberate infinite loop that makes the miss
    /// visible. lld spells these three rules out beside its `R_PC` case
    /// (`lld/ELF/InputSection.cpp`, the
    /// `get*UndefinedRelativeWeakVA` helpers above it). `place` is `P`, for
    /// the architectures whose non-branch answer still measures from it.
    fn undef_weak_pc(r_type: u32, addend: i64, place: u64) -> u64 {
        let _ = r_type;
        addend.cast_unsigned().wrapping_sub(place)
    }

    /// The number of leading opcode bytes the relax hook needs immediately
    /// before the value slot, or `0` when `r_type` is not relaxable. The
    /// writer carves a window of `relax_lead + spec.write.width()` bytes
    /// ending at the slot's end and hands it to [`Self::relax`]. The default
    /// returns `0`, so an architecture that overrides [`Self::relax`] must
    /// also override this to expose the types it can rewrite.
    fn relax_lead(r_type: u32) -> usize {
        let _ = r_type;
        0
    }

    /// Whether any rewrite claims this relocation type.
    ///
    /// The window a rewrite is handed spans [`Self::relax_lead`] bytes, the
    /// slot, and [`Self::relax_trail`] bytes, so a type rewritten entirely
    /// within its own slot declares no lead and no trail and is otherwise
    /// indistinguishable from a type nothing rewrites. x86-64 has one --
    /// the `call` a TLS descriptor sequence closes with, whose two bytes
    /// become a `nop` in place -- so the question is asked separately from the
    /// window's size. The default derives the answer from the lead, which is
    /// right for every rewrite that needs the opcode bytes ahead of its slot.
    fn relax_covers(r_type: u32) -> bool {
        Self::relax_lead(r_type) != 0
    }

    /// Whether the scan may decline to allocate a GOT slot for this site,
    /// because relaxation will certainly rewrite it into a form that reads no
    /// slot.
    ///
    /// `lead` is the bytes immediately before the fixup, the same window
    /// [`Self::relax`] rewrites, so an architecture can read the instruction
    /// it would be replacing. Answering `true` is a promise that
    /// [`Self::relax`] will accept the site, and the caller must have settled
    /// every fact the answer depends on that this signature does not carry --
    /// the symbol's preemptibility and whether its value is an address.
    ///
    /// The default is `false`: keep the slot.
    fn relax_drops_got(_r_type: u32, _addend: i64, _lead: &[u8]) -> bool {
        false
    }

    /// Whether a rewrite must happen for the link to be correct, rather than
    /// being an optimisation the caller may decline.
    ///
    /// Most relaxations are optional: leaving a GOT-indirect access alone
    /// produces a working image, just a slower one, so they run only when the
    /// link asks for them. A general-dynamic TLS pair in an executable is
    /// different -- there is no runtime `__tls_get_addr` to fall back on, so
    /// lowering it is the only way the site can be resolved at all, and it runs
    /// whether or not relaxation was requested. So does the reference to the
    /// helper the pair no longer calls, which is why the symbol and resolver
    /// are in scope: the scan drops that reference, and a site whose symbol it
    /// did not allocate for is one the writer must not fill in. The default is
    /// `false`.
    ///
    /// An architecture answering `true` promises there is no other way to
    /// resolve the site, and the writer takes it at its word: a mandatory
    /// rewrite that then declines ends the link with a diagnostic rather than
    /// falling through to the spec's value computation, which for these sites
    /// would measure a displacement from storage nothing allocated. Answer
    /// `false` for anything the value computation can still resolve, however
    /// much better the rewritten form would be.
    fn relax_required<R: Resolver>(
        r_type: u32,
        sym: Option<SymbolId>,
        resolver: &R,
    ) -> bool {
        let _ = (r_type, sym, resolver);
        false
    }

    /// The number of bytes *after* the value slot the relax hook rewrites, or
    /// `0` when the rewrite ends at the slot. A multi-instruction sequence
    /// reaches past its own slot -- the general-dynamic TLS pair ends in a
    /// `call` that the local-exec form replaces -- and declares the extra
    /// bytes here, which the writer adds to the window. The default returns
    /// `0`.
    ///
    /// `after` is the section's bytes from the end of the slot onwards, or as
    /// much of them as there is. One sequence needs them: the x86-64
    /// local-dynamic form closes with a direct `call` under the default ABI
    /// and with a GOT-indirect one under `-fno-plt`, which is a byte longer,
    /// and only the opcode says which. Nothing may be claimed on the strength
    /// of a byte that is not there, so an architecture reading `after` must
    /// treat a short slice as "not that form" -- claiming an extra byte would
    /// suppress a relocation on an instruction the rewrite never touches.
    fn relax_trail(r_type: u32, slot: &[u8], after: &[u8]) -> usize {
        let _ = (r_type, slot, after);
        0
    }

    /// The bytes lowering this site would consume, as a range within the
    /// section being patched, or `None` when the type has no lowering or the
    /// site is not one this image lowers.
    ///
    /// A multi-instruction sequence carries one relocation per instruction and
    /// is rewritten as a unit, so every relocation but the first has nothing
    /// left to patch. The answer belongs to the sequence, not to the
    /// instructions it swallows: an x86-64 general-dynamic pair opens with a
    /// `lea` carrying `TLSGD` and closes with an ordinary `call rel32`, and
    /// nothing about that call's own type, symbol or bytes says it was
    /// consumed -- only the pair beside it does. So the writer asks this of
    /// every relocation before it applies any, and drops the ones whose slot
    /// falls inside a reported range. That is order-independent: it does not
    /// matter which of the two the input lists first.
    ///
    /// A range is reported only when the rewrite will actually happen, so the
    /// answer has to agree with [`Self::relax`] exactly. An architecture
    /// implements the two against one predicate rather than two spellings of
    /// it: a disagreement means either a relocation applied over a lowered
    /// sequence or a live instruction left unpatched. The default reports
    /// nothing, as [`Self::relax_lead`] and [`Self::relax_trail`] default to
    /// zero.
    fn relax_span<R: Resolver>(
        site: RelaxSite<'_>,
        resolver: &R,
    ) -> Option<Range<u64>> {
        let _ = (site, resolver);
        None
    }

    /// Whether a lowering could span a relocation of this type at all.
    ///
    /// [`Self::relax_span`] opens with this and reports nothing when it is
    /// false, so it is the same predicate rather than a second spelling of one
    /// -- widening it can only cost a wasted `relax_span` call, and narrowing
    /// it below what `relax_span` accepts is the disagreement that hook's
    /// contract rules out.
    ///
    /// It is asked separately because building a [`RelaxSite`] is not free:
    /// the writer's pre-pass runs over every relocation of every member, and
    /// each site costs a merge-plan probe for the addend. Answering from the
    /// type alone keeps that off the relocations no lowering could touch,
    /// which is all but a handful. The default spans nothing.
    fn spans_type(r_type: u32) -> bool {
        let _ = r_type;
        false
    }

    /// Attempts to rewrite the relocation site into a cheaper form. `window`
    /// covers [`Self::relax_lead`] opcode bytes followed by the value slot
    /// (whose width comes from the spec); the slot begins at offset
    /// `Self::relax_lead(r_type)` within `window`. Returns `Ok(true)` if the
    /// site was rewritten (the caller skips the normal apply), `Ok(false)` to
    /// fall back to the spec's value computation. The default never relaxes.
    fn relax<R: Resolver>(
        r_type: u32,
        _sym: Option<SymbolId>,
        _addend: i64,
        _place: u64,
        _resolver: &R,
        _window: &mut [u8],
    ) -> Result<bool> {
        let _ = r_type;
        Ok(false)
    }
}

/// A normalised relocation record, independent of the REL/RELA encoding.
///
/// This is the unit the drivers consume. REL entries (whose addend lives in
/// the target bytes) are not produced: x86-64, `AArch64` and RISC-V are all
/// RELA-only psABIs, so only the RELA constructor exists.
#[derive(Clone, Copy, Debug)]
pub struct Reloc {
    /// Raw relocation type (`R_X86_64_*` and similar).
    pub r_type: u32,
    /// Object-local symbol index, or zero for the null symbol.
    pub sym: u32,
    /// Inline addend.
    pub addend: i64,
    /// Offset within the relocation's target section.
    pub offset: u64,
}

impl Reloc {
    /// Builds the record from a RELA entry.
    pub const fn from_rela(r: &Rela64) -> Self {
        Self {
            r_type: r.r_type(),
            sym: r.sym(),
            addend: r.r_addend.get(),
            offset: r.r_offset.get(),
        }
    }
}

/// Resolves and writes a relocation: classify, compute, range-check, store.
///
/// `out` is the output slice at the relocation site and must be exactly the
/// width the spec's write kind requires. For [`RelExpr::None`] the output is
/// left untouched; for [`RelExpr::Escape`] the work is delegated to
/// [`Arch::escape`].
pub fn apply<A: Arch, R: Resolver>(
    r_type: u32,
    sym: Option<SymbolId>,
    addend: i64,
    place: u64,
    resolver: &R,
    out: &mut [u8],
) -> Result<()> {
    let spec = A::spec(r_type)?;
    if spec.expr == RelExpr::Escape {
        return A::escape(r_type, sym, addend, place, resolver, out);
    }
    if spec.expr == RelExpr::None {
        return Ok(());
    }
    let write = spec.write;
    if out.len() != write.width() {
        return Err(Error::OutOfRange("relocation output slot"));
    }
    let value = match spec.expr {
        // A PC-relative site against an undefined weak leaves the plain
        // arithmetic and takes the architecture's answer instead.
        RelExpr::Pc => pc_value::<A, R>(r_type, sym, addend, place, resolver)?,
        RelExpr::PltPc => {
            plt_pc_value::<A, R>(r_type, sym, addend, place, resolver)?
        }
        _ => spec.expr.compute(sym, addend, place, resolver)?,
    };
    match write {
        Write::Bytes(k) => {
            if !k.fits(value) {
                return Err(Error::RelocOverflow(r_type));
            }
            k.store(value, out);
        }
        Write::Field(f) => {
            // Alignment first, as lld checks it: a misaligned target and an
            // out-of-range one are different mistakes, and reporting the
            // former as an overflow would send the reader after the wrong
            // one.
            if !f.aligned(value) {
                return Err(Error::RelocMisaligned(r_type));
            }
            if !f.fits(value) {
                return Err(Error::RelocOverflow(r_type));
            }
            f.store(value, out);
        }
    }
    Ok(())
}

/// The allocation a relocation type forces for its symbol (the scan phase).
pub fn scan<A: Arch>(r_type: u32) -> Result<Needs> {
    A::scan_needs(r_type)
}

/// The [`RelExpr::Pc`] value for one site: the ordinary computation, or the
/// architecture's undefined-weak answer when the symbol is one.
pub(crate) fn pc_value<A: Arch, R: Resolver>(
    r_type: u32,
    sym: Option<SymbolId>,
    addend: i64,
    place: u64,
    resolver: &R,
) -> Result<u64> {
    if sym.is_some_and(|id| resolver.is_undef_weak(id)) {
        return Ok(A::undef_weak_pc(r_type, addend, place));
    }
    RelExpr::Pc.compute(sym, addend, place, resolver)
}

/// The [`RelExpr::PltPc`] counterpart of [`pc_value`].
///
/// A PLT entry routes through the stub however weak the reference is: the
/// loader binds it at run time, and lld's `R_PLT_PC` never asks whether the
/// symbol is an undefined weak for exactly that reason
/// (`lld/ELF/InputSection.cpp`). Only the entry-less site
/// reduces to the plain question -- the case lld answers by demoting the
/// reloc to `R_PC` -- and only there does the architecture's answer apply.
pub(crate) fn plt_pc_value<A: Arch, R: Resolver>(
    r_type: u32,
    sym: Option<SymbolId>,
    addend: i64,
    place: u64,
    resolver: &R,
) -> Result<u64> {
    let entry = sym.map_or(0, |id| resolver.plt_addr(id));
    if entry == 0 && sym.is_some_and(|id| resolver.is_undef_weak(id)) {
        return Ok(A::undef_weak_pc(r_type, addend, place));
    }
    RelExpr::PltPc.compute(sym, addend, place, resolver)
}

// --- multi-architecture dispatch ----------------------------------------

/// The architecture a link targets, derived from the inputs' `e_machine`.
///
/// The drivers [`scan`] and [`apply`] are generic over an [`Arch`]; the linker
/// owns a single concrete [`Target`] and routes through the matching arch via
/// [`scan_target`]/[`apply_target`]. The match lives in one place here, so the
/// rest of the linker stays unaware of the architecture set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Target {
    /// x86-64 (`EM_X86_64`).
    X86_64,
    /// `AArch64` (`EM_AARCH64`).
    AArch64,
    /// RISC-V 64 (`EM_RISCV`).
    Riscv64,
}

impl Target {
    /// Maps an `e_machine` value to a target, rejecting unsupported machines.
    pub fn from_machine(e_machine: u16) -> Result<Self> {
        match e_machine {
            EM_X86_64 => Ok(Self::X86_64),
            EM_AARCH64 => Ok(Self::AArch64),
            EM_RISCV => Ok(Self::Riscv64),
            _ => Err(Error::Format("unsupported e_machine")),
        }
    }

    /// The `e_machine` value written into the output ELF header.
    pub const fn machine(self) -> u16 {
        match self {
            Self::X86_64 => EM_X86_64,
            Self::AArch64 => EM_AARCH64,
            Self::Riscv64 => EM_RISCV,
        }
    }

    /// The dynamic relocation types this target emits into `.rela.dyn` and
    /// `.rela.plt`. The dynamic layer is otherwise target-neutral: every
    /// hardcoded `R_X86_64_*` is replaced by a slot from this table so the
    /// loader sees the relocation type it expects for the architecture.
    pub const fn dyn_relocs(self) -> DynRelocs {
        match self {
            Self::X86_64 => DynRelocs {
                relative: x86_64::R_X86_64_RELATIVE,
                glob_dat: x86_64::R_X86_64_GLOB_DAT,
                jump_slot: x86_64::R_X86_64_JUMP_SLOT,
                copy: x86_64::R_X86_64_COPY,
                irelative: x86_64::R_X86_64_IRELATIVE,
                dtpmod: x86_64::R_X86_64_DTPMOD64,
                dtpoff: x86_64::R_X86_64_DTPOFF64,
                tpoff: x86_64::R_X86_64_TPOFF64,
                abs64: x86_64::R_X86_64_64,
            },
            Self::AArch64 => DynRelocs {
                relative: aarch64::R_AARCH64_RELATIVE,
                glob_dat: aarch64::R_AARCH64_GLOB_DAT,
                jump_slot: aarch64::R_AARCH64_JUMP_SLOT,
                copy: aarch64::R_AARCH64_COPY,
                irelative: aarch64::R_AARCH64_IRELATIVE,
                dtpmod: aarch64::R_AARCH64_TLS_DTPMOD64,
                dtpoff: aarch64::R_AARCH64_TLS_DTPREL64,
                tpoff: aarch64::R_AARCH64_TLS_TPREL64,
                abs64: aarch64::R_AARCH64_ABS64,
            },
            // RISC-V has no dedicated `R_RISCV_GLOB_DAT`; lld sets
            // `gotRel = symbolicRel = R_RISCV_64`, so a symbol-based GOT
            // entry uses the absolute 64-bit relocation type.
            Self::Riscv64 => DynRelocs {
                relative: riscv::R_RISCV_RELATIVE,
                glob_dat: riscv::R_RISCV_64,
                jump_slot: riscv::R_RISCV_JUMP_SLOT,
                copy: riscv::R_RISCV_COPY,
                irelative: riscv::R_RISCV_IRELATIVE,
                dtpmod: riscv::R_RISCV_TLS_DTPMOD64,
                dtpoff: riscv::R_RISCV_TLS_DTPREL64,
                tpoff: riscv::R_RISCV_TLS_TPREL64,
                abs64: riscv::R_RISCV_64,
            },
        }
    }

    /// Folds one input's `e_flags` into the value the output ELF header
    /// carries, or reports what the input disagreed with the earlier ones
    /// about.
    ///
    /// `so_far` is the merged value of the inputs read before this one, or
    /// `None` for the first. x86-64 and `AArch64` define no header flags --
    /// both psABIs leave the word reserved, and lld's `calcEFlags` is the
    /// base implementation returning zero for them -- so the answer there
    /// is zero whatever the inputs carry. RISC-V is the target that uses
    /// it, and the value is part of the image's ABI rather than
    /// bookkeeping.
    pub fn merge_eflags(
        self,
        so_far: Option<u32>,
        incoming: u32,
    ) -> core::result::Result<u32, &'static str> {
        match self {
            Self::X86_64 | Self::AArch64 => Ok(0),
            Self::Riscv64 => riscv::merge_eflags(so_far, incoming),
        }
    }

    /// The PLT section geometry for this target: the resolver trampoline
    /// size (`PLT[0]`), the per-entry size, and the number of reserved
    /// `.got.plt` header slots. The dynamic layer reads this so it can size
    /// the synthetic sections and compute per-symbol PLT addresses without
    /// the layout having to know the arch's stub layout.
    pub const fn plt_spec(self) -> PltSpec {
        match self {
            Self::X86_64 => PltSpec {
                header_size: 16,
                entry_size: 16,
                got_plt_reserved: 3,
            },
            // AArch64's lazy resolver is 32 bytes; like x86-64 it reserves
            // three `.got.plt` header slots (`_DYNAMIC`, link_map, resolver).
            Self::AArch64 => PltSpec {
                header_size: 32,
                entry_size: 16,
                got_plt_reserved: 3,
            },
            // RISC-V's lazy resolver is 32 bytes and `.got.plt` keeps only
            // two loader-owned header slots (no `_DYNAMIC` pointer).
            Self::Riscv64 => PltSpec {
                header_size: 32,
                entry_size: 16,
                got_plt_reserved: 2,
            },
        }
    }
}

/// The per-target dynamic relocation types, keyed by the role the dynamic
/// layer plays.
///
/// One constant per role the emitter uses; threading this struct through
/// `dynamic`/`plt` is what makes the output loader-correct for every
/// architecture.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DynRelocs {
    /// `R_*_RELATIVE`: `B + A`, the loader prefix counted by `DT_RELACOUNT`.
    pub relative: u32,
    /// `R_*_GLOB_DAT`: loader fills a GOT slot with the symbol's address.
    pub glob_dat: u32,
    /// `R_*_JUMP_SLOT`: lazy-binding PLT entry, patched on first call.
    pub jump_slot: u32,
    /// `R_*_COPY`: loader copies a shared object's data symbol into `.bss`.
    pub copy: u32,
    /// `R_*_IRELATIVE`: loader calls an `STT_GNU_IFUNC` resolver.
    pub irelative: u32,
    /// The target's pointer-width absolute relocation (`R_X86_64_64`,
    /// `R_AARCH64_ABS64`, `R_RISCV_64`). Used as the "symbol-based absolute
    /// data reference" type emitted into `.rela.dyn` for a global that is
    /// neither a copy slot nor a RELATIVE, and as the scan's detector for
    /// which input relocations contribute a dynamic entry.
    pub abs64: u32,
    /// `R_*_TPOFF64`/`TLS_TPREL64`: loader fills a GOT slot with a symbol's
    /// offset from the thread pointer.
    pub tpoff: u32,
    /// `R_*_DTPMOD64`: loader fills a GOT slot with the id of the module that
    /// defines a thread-local. The first half of a general-dynamic pair.
    pub dtpmod: u32,
    /// `R_*_DTPOFF64`: loader fills a GOT slot with a thread-local's offset
    /// within its own module's block. The second half of that pair.
    pub dtpoff: u32,
}

/// The per-target PLT section geometry.
///
/// The bytes themselves are emitted by the arch's PLT builder; this struct
/// carries only the sizes the layout needs to reserve space for `.plt` and
/// `.got.plt` and to compute a symbol's PLT entry address from its index.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PltSpec {
    /// `.plt` bytes of the resolver trampoline (`PLT[0]`).
    pub header_size: u64,
    /// `.plt` bytes of one user entry.
    pub entry_size: u64,
    /// Reserved 8-byte entries at the front of `.got.plt`: three for the
    /// `SysV` ABI used by x86-64/`AArch64` (the `_DYNAMIC` pointer plus two
    /// loader slots), two for RISC-V (loader-only).
    pub got_plt_reserved: u64,
}

/// Classifies a relocation for a given target (the scan dispatch shim).
pub fn scan_target(target: Target, r_type: u32) -> Result<Needs> {
    match target {
        Target::X86_64 => scan::<x86_64::X86_64>(r_type),
        Target::AArch64 => scan::<aarch64::AArch64>(r_type),
        Target::Riscv64 => scan::<riscv::Riscv64>(r_type),
    }
}

/// Looks up the spec for a given target. Used by the writer to size a
/// relocation slot before delegating the actual store to [`apply_target`].
pub fn spec_target(target: Target, r_type: u32) -> Result<Spec> {
    match target {
        Target::X86_64 => x86_64::X86_64::spec(r_type),
        Target::AArch64 => aarch64::AArch64::spec(r_type),
        Target::Riscv64 => riscv::Riscv64::spec(r_type),
    }
}

/// Resolves and writes a relocation for a given target (the apply dispatch
/// shim). Monomorphisation is preserved: each arm instantiates the generic
/// driver for one concrete [`Arch`].
pub fn apply_target<R: Resolver>(
    target: Target,
    r_type: u32,
    sym: Option<SymbolId>,
    addend: i64,
    place: u64,
    resolver: &R,
    out: &mut [u8],
) -> Result<()> {
    match target {
        Target::X86_64 => apply::<x86_64::X86_64, R>(
            r_type, sym, addend, place, resolver, out,
        ),
        Target::AArch64 => apply::<aarch64::AArch64, R>(
            r_type, sym, addend, place, resolver, out,
        ),
        Target::Riscv64 => apply::<riscv::Riscv64, R>(
            r_type, sym, addend, place, resolver, out,
        ),
    }
}

/// The relax-lead for a given target (the relax dispatch shim). Returns `0`
/// when the type is not relaxable, so the writer can cheaply skip the
/// relaxation attempt for the common case.
pub fn relax_lead_target(target: Target, r_type: u32) -> usize {
    match target {
        Target::X86_64 => x86_64::X86_64::relax_lead(r_type),
        Target::AArch64 => aarch64::AArch64::relax_lead(r_type),
        Target::Riscv64 => riscv::Riscv64::relax_lead(r_type),
    }
}

/// Whether any rewrite claims a type, for a given target (the dispatch shim).
///
/// This, not a non-zero lead, is what says the relaxation attempt is worth
/// making: a rewrite confined to its own slot declares no lead at all.
/// Whether the scan may skip this site's GOT slot; see
/// [`Arch::relax_drops_got`].
pub fn relax_drops_got_target(
    target: Target,
    r_type: u32,
    addend: i64,
    lead: &[u8],
) -> bool {
    match target {
        Target::X86_64 => x86_64::X86_64::relax_drops_got(r_type, addend, lead),
        Target::AArch64 => {
            aarch64::AArch64::relax_drops_got(r_type, addend, lead)
        }
        Target::Riscv64 => {
            riscv::Riscv64::relax_drops_got(r_type, addend, lead)
        }
    }
}

pub fn relax_covers_target(target: Target, r_type: u32) -> bool {
    match target {
        Target::X86_64 => x86_64::X86_64::relax_covers(r_type),
        Target::AArch64 => aarch64::AArch64::relax_covers(r_type),
        Target::Riscv64 => riscv::Riscv64::relax_covers(r_type),
    }
}

/// Whether a rewrite is mandatory for a given target (the dispatch shim).
pub fn relax_required_target<R: Resolver>(
    target: Target,
    r_type: u32,
    sym: Option<SymbolId>,
    resolver: &R,
) -> bool {
    match target {
        Target::X86_64 => x86_64::X86_64::relax_required(r_type, sym, resolver),
        Target::AArch64 => {
            aarch64::AArch64::relax_required(r_type, sym, resolver)
        }
        Target::Riscv64 => {
            riscv::Riscv64::relax_required(r_type, sym, resolver)
        }
    }
}

/// The relax-trail for a given target (the relax dispatch shim). Returns `0`
/// when the rewrite does not reach past the value slot.
pub fn relax_trail_target(
    target: Target,
    r_type: u32,
    slot: &[u8],
    after: &[u8],
) -> usize {
    match target {
        Target::X86_64 => x86_64::X86_64::relax_trail(r_type, slot, after),
        Target::AArch64 => aarch64::AArch64::relax_trail(r_type, slot, after),
        Target::Riscv64 => riscv::Riscv64::relax_trail(r_type, slot, after),
    }
}

/// The bytes a lowering consumes for a given target (the relax dispatch shim).
///
/// Returns `None` for every type no lowering spans, which is all but a handful,
/// so the writer's pre-pass costs one type check per relocation.
pub fn relax_span_target<R: Resolver>(
    target: Target,
    site: RelaxSite<'_>,
    resolver: &R,
) -> Option<Range<u64>> {
    match target {
        Target::X86_64 => x86_64::X86_64::relax_span(site, resolver),
        Target::AArch64 => aarch64::AArch64::relax_span(site, resolver),
        Target::Riscv64 => riscv::Riscv64::relax_span(site, resolver),
    }
}

/// Whether any lowering for `target` can span a relocation of this type (the
/// [`Arch::spans_type`] dispatch shim).
///
/// The writer's pre-pass asks this before it builds a [`RelaxSite`], which is
/// what keeps the per-site cost off relocations no lowering could touch.
pub fn relax_spans_type(target: Target, r_type: u32) -> bool {
    match target {
        Target::X86_64 => x86_64::X86_64::spans_type(r_type),
        Target::AArch64 => aarch64::AArch64::spans_type(r_type),
        Target::Riscv64 => riscv::Riscv64::spans_type(r_type),
    }
}

/// Attempts to relax a relocation for a given target (the relax dispatch
/// shim).
///
/// Returns `Ok(true)` if the site was rewritten, `Ok(false)` to fall back to
/// [`apply_target`]. The driver is monomorphised per target, matching
/// [`apply_target`].
pub fn relax_target<R: Resolver>(
    target: Target,
    r_type: u32,
    sym: Option<SymbolId>,
    addend: i64,
    place: u64,
    resolver: &R,
    window: &mut [u8],
) -> Result<bool> {
    match target {
        Target::X86_64 => {
            x86_64::X86_64::relax(r_type, sym, addend, place, resolver, window)
        }
        Target::AArch64 => aarch64::AArch64::relax(
            r_type, sym, addend, place, resolver, window,
        ),
        Target::Riscv64 => {
            riscv::Riscv64::relax(r_type, sym, addend, place, resolver, window)
        }
    }
}
