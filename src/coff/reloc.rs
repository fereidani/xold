//! COFF relocation normalisation and apply dispatch.
//!
//! The arch-neutral [`crate::reloc::apply`] driver is reused: each COFF
//! relocation type maps to a [`Spec`] in [`crate::reloc::coff_x86_64`] /
//! [`crate::reloc::coff_i386`], and the value arithmetic is shared with ELF
//! and Mach-O. COFF relocations are REL-encoded: the addend lives in the target
//! bytes, plus a small per-type PC-relative correction. This mirrors how the
//! Mach-O writer folds its `x86_64` REL-style addend at the boundary.
//!
//! A COFF relocation references a symbol by its index in the owning file's
//! symbol table. [`CoffResolver`] maps that index to the symbol's resolved
//! virtual address (image base plus the laid-out RVA), which the portable
//! [`Resolver`] trait then hands to the driver.

use crate::{
    coff::{
        CoffFile, CoffReloc,
        constants::{
            IMAGE_FILE_MACHINE_AMD64, IMAGE_FILE_MACHINE_I386,
            IMAGE_SYM_CLASS_EXTERNAL,
        },
    },
    error::{Error, Result},
    reloc::{
        Resolver, apply,
        coff_i386::CoffI386,
        coff_x86_64::{
            CoffX86_64, IMAGE_REL_AMD64_ADDR64, IMAGE_REL_AMD64_REL32,
            IMAGE_REL_AMD64_REL32_1, IMAGE_REL_AMD64_REL32_2,
            IMAGE_REL_AMD64_REL32_3, IMAGE_REL_AMD64_REL32_4,
            IMAGE_REL_AMD64_REL32_5,
        },
    },
    symbol::SymbolId,
};

/// The COFF target architecture being linked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoffTarget {
    X86_64,
    I386,
}

impl CoffTarget {
    /// Derives the target from the object's machine type, rejecting the rest.
    ///
    /// I386 is recognised here because the reader needs to name what it is
    /// looking at; whether an image can be produced for it is
    /// [`Self::is_writable`]'s question.
    pub fn from_machine(machine: u16) -> Result<Self> {
        match machine {
            IMAGE_FILE_MACHINE_AMD64 => Ok(Self::X86_64),
            IMAGE_FILE_MACHINE_I386 => Ok(Self::I386),
            _ => Err(Error::Format("unsupported COFF machine type")),
        }
    }

    /// Whether this linker can write an image for the target.
    ///
    /// Only x86-64. An I386 object was accepted and then written as a
    /// PE32+ image -- 64-bit optional header, x86-64 image base, 8-byte
    /// import thunks and an x86-64 entry stub -- under `machine = 0x014c`,
    /// and the relocation table compared i386 type numbers against AMD64
    /// constants besides. Windows refuses the result, which is the right
    /// answer arriving from the wrong place: a linker that cannot write a
    /// PE32 image should say so rather than write a PE32+ one and label it
    /// i386.
    pub const fn is_writable(self) -> bool {
        matches!(self, Self::X86_64)
    }
}

/// Resolves and writes one relocation through the arch-neutral driver.
pub fn apply_reloc<R: Resolver>(
    target: CoffTarget,
    r_type: u32,
    sym: Option<SymbolId>,
    addend: i64,
    place: u64,
    resolver: &R,
    out: &mut [u8],
) -> Result<()> {
    match target {
        CoffTarget::X86_64 => {
            apply::<CoffX86_64, R>(r_type, sym, addend, place, resolver, out)
        }
        CoffTarget::I386 => {
            apply::<CoffI386, R>(r_type, sym, addend, place, resolver, out)
        }
    }
}

/// The number of target bytes a relocation patches (its field width).
pub fn reloc_width(reloc: &CoffReloc) -> usize {
    match u32::from(reloc.typ) {
        IMAGE_REL_AMD64_ADDR64 => 8,
        _ => 4,
    }
}

/// The REL-style addend of one relocation: the signed little-endian value in
/// `slot` plus the per-type PC-relative correction.
///
/// COFF `x86_64` PC-relative fixups address the end of the 4-byte field, so the
/// implicit correction is `-4` for `REL32` and `-5`..`-9` for the `_1`..`_5`
/// variants (the extra trailing opcode bytes). Absolute types take the raw
/// value verbatim. This matches the addend discipline documented with the
/// [`crate::reloc::coff_x86_64`] table.
pub fn addend_for(reloc: &CoffReloc, slot: &[u8]) -> i64 {
    let raw = read_signed_le(slot, reloc_width(reloc));
    match u32::from(reloc.typ) {
        IMAGE_REL_AMD64_REL32 => raw - 4,
        IMAGE_REL_AMD64_REL32_1 => raw - 5,
        IMAGE_REL_AMD64_REL32_2 => raw - 6,
        IMAGE_REL_AMD64_REL32_3 => raw - 7,
        IMAGE_REL_AMD64_REL32_4 => raw - 8,
        IMAGE_REL_AMD64_REL32_5 => raw - 9,
        _ => raw,
    }
}

/// Reads `len` bytes from `slot` as a little-endian signed integer, sign
/// extended to `i64`.
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

/// A resolver over one input file's resolved symbol addresses.
///
/// Each symbol's absolute virtual address (image base plus its laid-out RVA)
/// is precomputed by the layout and indexed by symbol-table index. Undefined
/// references that resolve to a synthesised linker symbol (the `__imp_`
/// import aliases, the entry point) are reported via the same index space the
/// caller assigns; the GOT and PLT collapse in a static PE link, so `got_*`
/// and `plt_addr` fall back to the symbol address.
///
/// `sec_rva` carries, for each symbol index, the RVA of the output section
/// the symbol was laid out in. The section-relative escape relocations
/// (`IMAGE_REL_AMD64_SECREL`, `IMAGE_REL_AMD64_ADDR32NB`) read it through
/// [`Self::section_rva`] to compute a section offset or an image RVA.
pub struct CoffResolver<'a> {
    addr: &'a [u64],
    sec_rva: &'a [u32],
    image_base: u64,
}

impl<'a> CoffResolver<'a> {
    pub fn new(addr: &'a [u64], sec_rva: &'a [u32], image_base: u64) -> Self {
        Self {
            addr,
            sec_rva,
            image_base,
        }
    }

    /// The RVA of the output section the symbol was laid out in, or zero when
    /// the symbol is undefined or absolute.
    pub fn section_rva(&self, sym: SymbolId) -> u32 {
        self.sec_rva.get(sym.0).copied().unwrap_or_default()
    }

    /// The image base the absolute addresses were computed against.
    pub const fn image_base(&self) -> u64 {
        self.image_base
    }
}

impl Resolver for CoffResolver<'_> {
    fn symbol_addr(&self, sym: SymbolId) -> u64 {
        self.addr.get(sym.0).copied().unwrap_or_default()
    }
    fn got_addr(&self, sym: SymbolId) -> u64 {
        self.symbol_addr(sym)
    }
    fn got_base(&self) -> u64 {
        0
    }
    fn plt_addr(&self, sym: SymbolId) -> u64 {
        self.symbol_addr(sym)
    }
}

/// Resolves the address of the symbol named `name` across `inputs`.
///
/// Used for the linker-generated entry stub (which calls the user entry and
/// the `ExitProcess` IAT slot). `sym_addr` is indexed by raw symbol-table
/// index. Returns the absolute virtual address.
pub fn global_symbol_addr(
    inputs: &[CoffFile<'_>],
    sym_addr: &[Vec<u64>],
    name: &[u8],
) -> Option<u64> {
    // Only a defined external answers to a global name. The match was on the
    // name alone, so a `static main` in an earlier file won over the real
    // external -- and so would any file-scoped symbol that happened to share
    // a name with an import or an export. A negative or zero section number
    // is not a definition either: it is an undefined reference, a common, or
    // an absolute.
    for (file, input) in inputs.iter().enumerate() {
        for sym in input.symbols().iter() {
            if sym.name != name
                || sym.storage_class != IMAGE_SYM_CLASS_EXTERNAL
                || sym.section_number <= 0
            {
                continue;
            }
            return sym_addr
                .get(file)
                .and_then(|f| f.get(sym.index as usize))
                .copied();
        }
    }
    None
}
