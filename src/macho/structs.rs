//! Plain, endian-resolved view records for Mach-O data.
//!
//! The `object` crate exposes Mach-O structures as endian-typed types
//! (`Section64<Endian>`, `Nlist64<Endian>`, `Relocation<Endian>`), where every
//! integer field is wrapped in a `U32`/`U64` that must be resolved against an
//! [`Endianness`]. The reader resolves endianness once at the boundary and
//! yields the plain records defined here, so the rest of xold reads native
//! `u64`/`u32` fields with no dependency on `object`'s endian machinery. This
//! mirrors how the ELF reader re-views object's bytes as plain [`Shdr64`]
//! structs.
//!
//! Each record borrows `&'data` from the original byte mapping (section names
//! and payload); scalar fields are copied. Relocations carry no borrowed data,
//! so [`MachReloc`] is `Copy`.

use crate::macho::constants::{
    N_ABS, N_EXT, N_PEXT, N_SECT, N_STAB, N_TYPE, N_UNDF,
};

/// A Mach-O section, endian-resolved. Section names are NUL-trimmed views into
/// the original bytes; `data` is the section payload (empty for zero-fill).
#[derive(Clone)]
pub struct MachSection<'data> {
    /// 1-based section ordinal, matching `n_sect` values in the symbol table.
    pub index: u32,
    /// `sectname`, trimmed at the first NUL.
    pub sectname: &'data [u8],
    /// `segname`, trimmed at the first NUL.
    pub segname: &'data [u8],
    /// Virtual address (`addr`).
    pub addr: u64,
    /// Size in bytes (`size`).
    pub size: u64,
    /// Power-of-two alignment (`align`), as stored.
    pub align: u32,
    /// Raw `flags` (section type in the low byte, attributes above).
    pub flags: u32,
    /// Payload bytes; empty for `S_ZEROFILL` and other no-file-space types.
    pub data: &'data [u8],
    /// Decoded relocation entries for this section.
    pub relocations: Vec<MachReloc>,
}

/// A decoded Mach-O relocation entry.
///
/// `r_address` is an offset within the owning section. `r_symbolnum` is a
/// symbol index when `r_extern` is set, otherwise a 1-based section ordinal.
/// `r_length` encodes the fixup width as `1 << r_length` bytes. `r_type` is the
/// architecture-specific type number (see `reloc::macho_x86_64` and
/// `reloc::macho_arm64`).
#[derive(Clone, Copy, Debug)]
pub struct MachReloc {
    /// Offset within the section to the fixup site.
    pub r_address: u32,
    /// Symbol index (`r_extern` set) or section ordinal (`r_extern` clear).
    pub r_symbolnum: u32,
    /// PC-relative fixup.
    pub r_pcrel: bool,
    /// `log2` of the fixup width (0=byte, 1=word, 2=long, 3=quad).
    pub r_length: u8,
    /// References an external symbol rather than a section.
    pub r_extern: bool,
    /// Architecture-specific relocation type.
    pub r_type: u8,
    /// Scattered (legacy 32-bit) relocation encoding.
    pub r_scattered: bool,
}

impl MachReloc {
    /// The fixup width in bytes: `1 << r_length`.
    pub const fn width(self) -> usize {
        1 << self.r_length
    }
}

/// A Mach-O symbol (`nlist_64`), endian-resolved. `name` is NUL-terminated and
/// borrows the string table; an out-of-range `n_strx` yields an empty slice.
#[derive(Clone, Copy)]
pub struct MachSymbol<'data> {
    /// NUL-terminated name bytes from the string table.
    pub name: &'data [u8],
    /// String-table index of the name.
    pub n_strx: u32,
    /// Raw `n_type` (`N_STAB` / `N_PEXT` / `N_TYPE` / `N_EXT` bits).
    pub n_type: u8,
    /// Section ordinal (1-based), or 0 if not section-relative.
    pub n_sect: u8,
    /// Symbol description (flags such as `N_WEAK_DEF`, reference method,
    /// etc.).
    pub n_desc: u16,
    /// Value (address for defined symbols, size for common symbols).
    pub n_value: u64,
}

impl MachSymbol<'_> {
    /// Whether this is a debug (`N_STAB`) entry, which carries no link
    /// semantics.
    pub const fn is_stab(&self) -> bool {
        self.n_type & N_STAB != 0
    }

    /// Whether this symbol is exported across object boundaries (`N_EXT`).
    pub const fn is_external(&self) -> bool {
        !self.is_stab() && self.n_type & N_EXT != 0
    }

    /// Whether an external symbol is private (`N_PEXT`): visible to the linker
    /// for resolution but not exported in the final image.
    pub const fn is_private_external(&self) -> bool {
        self.is_external() && self.n_type & N_PEXT != 0
    }

    /// Undefined reference: resolves to `N_TYPE == N_UNDF` with no section.
    pub const fn is_undefined(&self) -> bool {
        !self.is_stab() && self.n_type & N_TYPE == N_UNDF
    }

    /// Absolute symbol, not associated with any section (`N_TYPE == N_ABS`).
    pub const fn is_absolute(&self) -> bool {
        !self.is_stab() && self.n_type & N_TYPE == N_ABS
    }

    /// Section-relative symbol (`N_TYPE == N_SECT`); `n_sect` selects which.
    pub const fn is_section(&self) -> bool {
        !self.is_stab() && self.n_type & N_TYPE == N_SECT
    }
}
