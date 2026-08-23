//! Mach-O relocation normalisation and apply dispatch.
//!
//! The arch-neutral [`crate::reloc::apply`] driver is reused: each darwin
//! relocation type maps to a [`Spec`] in [`crate::reloc::macho_x86_64`] /
//! [`crate::reloc::macho_arm64`], and the value arithmetic is shared with ELF.
//! What is specific to Mach-O is the *addend*: Mach-O uses REL-style entries
//! (the addend lives in the target bytes, not in the reloc), and `x86_64`
//! PC-relative fixups address the end of the field, so the implicit addend is
//! `-(1 << r_length)` plus a small per-type correction. `arm64` instead encodes
//! the fixup in scattered instruction fields that `apply` overwrites, so its
//! addend is zero (plus the explicit `ARM64_RELOC_ADDEND` hint when present).
//!
//! Section-relative relocations (`r_extern == 0`) reference a section ordinal
//! rather than a symbol; they are resolved through a fixed-address resolver
//! that reports the section's laid-out virtual address.

use crate::{
    error::{Error, Result},
    macho::MachReloc,
    reloc::{
        Needs, Resolver, apply,
        macho_arm64::{
            ARM64_RELOC_ADDEND, ARM64_RELOC_UNSIGNED, ARM64_RELOC_UNSIGNED_4,
            MachoArm64,
        },
        macho_x86_64::{
            MachoX86_64, X86_64_RELOC_UNSIGNED, X86_64_RELOC_UNSIGNED_4,
        },
        scan,
    },
    symbol::SymbolId,
};

/// The darwin architecture being linked, derived from the inputs' `cputype`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MachoTarget {
    X86_64,
    Arm64,
}

impl MachoTarget {
    /// Maps a Mach-O `cputype` to a target, rejecting unsupported cpus.
    pub fn from_cpu(cpu: u32) -> Result<Self> {
        match cpu {
            crate::macho::constants::CPU_TYPE_X86_64 => Ok(Self::X86_64),
            crate::macho::constants::CPU_TYPE_ARM64 => Ok(Self::Arm64),
            _ => Err(Error::Format("unsupported Mach-O cputype")),
        }
    }
}

/// Resolves and writes one relocation through the arch-neutral driver.
///
/// `sym` is the symbol index for `r_extern` entries; for section-relative
/// entries the caller passes a fixed-address resolver and `sym =
/// Some(SymbolId(0))`.
pub fn apply_reloc<R: Resolver>(
    target: MachoTarget,
    r_type: u32,
    sym: Option<SymbolId>,
    addend: i64,
    place: u64,
    resolver: &R,
    out: &mut [u8],
) -> Result<()> {
    match target {
        MachoTarget::X86_64 => {
            apply::<MachoX86_64, R>(r_type, sym, addend, place, resolver, out)
        }
        MachoTarget::Arm64 => {
            apply::<MachoArm64, R>(r_type, sym, addend, place, resolver, out)
        }
    }
}

/// The allocation a relocation type forces for its symbol (the scan dispatch).
///
/// Mirrors [`apply_reloc`]: one arm per architecture, reusing the arch-neutral
/// [`scan`] driver so the GOT/PLT sizing is shared with ELF.
pub fn scan_needs(target: MachoTarget, r_type: u32) -> Result<Needs> {
    match target {
        MachoTarget::X86_64 => scan::<MachoX86_64>(r_type),
        MachoTarget::Arm64 => scan::<MachoArm64>(r_type),
    }
}

/// The relocation type to look up in the architecture table.
///
/// Mach-O sizes an absolute fixup through `r_length`, not through the type, so
/// a 4-byte `UNSIGNED` and an 8-byte one arrive as the same number. The narrow
/// form gets its own synthetic type here, which keeps the width where every
/// other architecture states it: in the table.
pub fn table_type(target: MachoTarget, reloc: &MachReloc) -> u32 {
    let r_type = u32::from(reloc.r_type);
    let unsigned = match target {
        MachoTarget::X86_64 => X86_64_RELOC_UNSIGNED,
        MachoTarget::Arm64 => ARM64_RELOC_UNSIGNED,
    };
    if r_type != unsigned || reloc.r_length == 3 {
        return r_type;
    }
    match target {
        MachoTarget::X86_64 => X86_64_RELOC_UNSIGNED_4,
        MachoTarget::Arm64 => ARM64_RELOC_UNSIGNED_4,
    }
}

/// The addend for one Mach-O relocation.
///
/// `slot` is the unmodified target bytes at the fixup site (REL-style: the
/// addend is encoded there for `x86_64`). `pending` is the explicit addend
/// carried by a preceding `ARM64_RELOC_ADDEND` entry on `arm64`, or zero.
pub fn addend_for(
    target: MachoTarget,
    reloc: &MachReloc,
    slot: &[u8],
    pending: i64,
) -> i64 {
    match target {
        MachoTarget::X86_64 => x86_addend(reloc, slot),
        MachoTarget::Arm64 => arm64_addend(reloc, slot, pending),
    }
}

/// The `arm64` embedded addend of an `UNSIGNED` fixup: the value stored in
/// the field itself.
///
/// Every other arm64 type overwrites scattered instruction bits, so its
/// embedded addend is zero by construction
/// (`ARM64Common::getEmbeddedAddend` reads the slot for `UNSIGNED` alone).
pub fn arm64_embedded(reloc: &MachReloc, slot: &[u8]) -> i64 {
    if u32::from(reloc.r_type) != ARM64_RELOC_UNSIGNED {
        return 0;
    }
    read_signed_le(slot, reloc.width())
}

/// `arm64` addend: the explicit `ADDEND` hint plus, for `UNSIGNED`, the value
/// stored in the field. The two never carry the same fixup
/// (`InputFiles.cpp` asserts `!(embeddedAddend && pairedAddend)`), so adding
/// them matches lld's `totalAddend` either way.
fn arm64_addend(reloc: &MachReloc, slot: &[u8], pending: i64) -> i64 {
    arm64_embedded(reloc, slot) + pending
}

/// `x86_64` addend: the signed little-endian value in `slot`, plus the
/// PC-relative field-width correction (`-(1 << r_length)` when `r_pcrel`).
///
/// `SIGNED_1/2/4` name how many bytes of instruction follow the field, and
/// used to subtract that count a second time here. lld adds it to the addend
/// and adds it again to the program counter it subtracts
/// (`Arch/X86_64.cpp:101,109`), so the two cancel and the net value is
/// `S + raw - (1 << length) - P`. Subtracting it once left every
/// `movl $1, _g(%rip)` writing to `_g - 4`.
fn x86_addend(reloc: &MachReloc, slot: &[u8]) -> i64 {
    let raw = read_signed_le(slot, reloc.width());
    if reloc.r_pcrel {
        return raw - (1i64 << reloc.r_length);
    }
    raw
}

/// Reads `len` bytes from `slot` as a little-endian signed integer, sign
/// extended to `i64`. Reading the bytes straight into the matching signed type
/// avoids an intermediate unsigned cast; `try_from` keeps the copy panic-free.
fn read_signed_le(slot: &[u8], len: usize) -> i64 {
    let n = len.min(slot.len()).min(8);
    let buf = slot.get(..n).unwrap_or_default();
    match n {
        1 => <&[u8; 1]>::try_from(buf)
            .map_or(0, |a| i64::from(i8::from_le_bytes(*a))),
        2 => <&[u8; 2]>::try_from(buf)
            .map_or(0, |a| i64::from(i16::from_le_bytes(*a))),
        4 => <&[u8; 4]>::try_from(buf)
            .map_or(0, |a| i64::from(i32::from_le_bytes(*a))),
        _ => {
            let mut wide = [0u8; 8];
            if let Some(take) = buf.get(..n.min(8)) {
                wide[..n.min(8)].copy_from_slice(take);
            }
            i64::from_le_bytes(wide)
        }
    }
}

/// Whether `reloc` is an `ARM64_RELOC_ADDEND` hint (it annotates the following
/// reloc and produces no fixup at its own site).
pub fn is_arm64_addend(reloc: &MachReloc) -> bool {
    u32::from(reloc.r_type) == ARM64_RELOC_ADDEND
}

/// The explicit addend an `ARM64_RELOC_ADDEND` entry carries for the next
/// reloc, read from its `r_symbolnum` field.
///
/// The field is 24 bits and signed: lld reads it with `SignExtend64<24>`.
/// Reading it unsigned turned every negative addend into a value just under
/// 16 MB.
pub fn arm64_pending_addend(reloc: &MachReloc) -> i64 {
    let raw = i64::from(reloc.r_symbolnum & 0x00ff_ffff);
    (raw << 40) >> 40
}

/// A resolver over one input file's resolved symbol addresses and GOT slot
/// addresses.
///
/// The GOT slice is parallel to `addr`: entry `i` is the address of the `__got`
/// slot for the file's `i`-th symbol, or zero if that symbol has no slot. The
/// PLT collapses: a static link allocates none, so `plt_addr` falls back to the
/// symbol address (direct call).
pub struct MachResolver<'a> {
    addr: &'a [u64],
    got: &'a [u64],
}

impl<'a> MachResolver<'a> {
    pub fn new(addr: &'a [u64], got: &'a [u64]) -> Self {
        Self { addr, got }
    }
}

impl Resolver for MachResolver<'_> {
    fn symbol_addr(&self, sym: SymbolId) -> u64 {
        self.addr.get(sym.0).copied().unwrap_or_default()
    }
    fn got_addr(&self, sym: SymbolId) -> u64 {
        self.got.get(sym.0).copied().unwrap_or_default()
    }
    fn got_base(&self) -> u64 {
        0
    }
    fn plt_addr(&self, sym: SymbolId) -> u64 {
        self.symbol_addr(sym)
    }
}

/// A resolver that pins every symbol to one section's virtual address, used
/// for section-relative relocations (`r_extern == 0`).
pub struct SectionResolver {
    pub vaddr: u64,
}

impl Resolver for SectionResolver {
    fn symbol_addr(&self, _sym: SymbolId) -> u64 {
        self.vaddr
    }
    fn got_addr(&self, _sym: SymbolId) -> u64 {
        self.vaddr
    }
    fn got_base(&self) -> u64 {
        0
    }
    fn plt_addr(&self, _sym: SymbolId) -> u64 {
        self.vaddr
    }
}
