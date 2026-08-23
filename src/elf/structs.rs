//! On-disk ELF64 little-endian structures.
//!
//! Every type is `#[repr(C)]` and derives `bytemuck::Pod` so it can be viewed
//! directly over mapped bytes with no `unsafe` at the call site. Field order
//! matches the ELF64 specification; sizes are asserted in the test suite.
//!
//! Integer fields are [`crate::endian`] little-endian types rather than plain
//! primitives. `bytemuck` reinterprets bytes and never reorders them, so plain
//! primitives would read a file's little-endian integers in host order --
//! right on a little-endian host, silently wrong on a big-endian one. The
//! types are `#[repr(transparent)]`, so the layout and the casts are
//! unchanged, and on a little-endian host the conversion compiles away.

use bytemuck::{Pod, Zeroable};

use crate::{
    elf::constants::STV_MASK,
    endian::{I64, U16, U32, U64},
};

/// ELF header (Ehdr), 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Ehdr64 {
    /// File identification (`ELFMAG...`).
    pub e_ident: [u8; 16],
    pub e_type: U16,
    pub e_machine: U16,
    pub e_version: U32,
    pub e_entry: U64,
    pub e_phoff: U64,
    pub e_shoff: U64,
    pub e_flags: U32,
    pub e_ehsize: U16,
    pub e_phentsize: U16,
    pub e_phnum: U16,
    pub e_shentsize: U16,
    pub e_shnum: U16,
    pub e_shstrndx: U16,
}

/// Section header (Shdr), 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Shdr64 {
    pub sh_name: U32,
    pub sh_type: U32,
    pub sh_flags: U64,
    pub sh_addr: U64,
    pub sh_offset: U64,
    pub sh_size: U64,
    pub sh_link: U32,
    pub sh_info: U32,
    pub sh_addralign: U64,
    pub sh_entsize: U64,
}

/// Program header (Phdr), 56 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Phdr64 {
    pub p_type: U32,
    pub p_flags: U32,
    pub p_offset: U64,
    pub p_vaddr: U64,
    pub p_paddr: U64,
    pub p_filesz: U64,
    pub p_memsz: U64,
    pub p_align: U64,
}

/// Symbol table entry (Sym), 24 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Sym64 {
    pub st_name: U32,
    pub st_info: u8,
    pub st_other: u8,
    pub st_shndx: U16,
    pub st_value: U64,
    pub st_size: U64,
}

/// Relocation with explicit addend (Rela), 24 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Rela64 {
    pub r_offset: U64,
    pub r_info: U64,
    pub r_addend: I64,
}

/// Dynamic table entry (Dyn), 16 bytes. An array of these in `.dynamic`
/// hands the runtime loader every table it needs (hash, symtab, strtab,
/// relocations). `d_tag` selects the meaning of `d_un`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Dyn64 {
    /// Selects the meaning of `d_un` (`DT_*`).
    pub d_tag: I64,
    /// A value or an address, interpreted per `d_tag`.
    pub d_un: U64,
}

impl Rela64 {
    /// Symbol index encoded in `r_info` (ELF64 layout).
    #[allow(clippy::cast_possible_truncation)]
    pub const fn sym(&self) -> u32 {
        // The shift leaves at most 32 significant bits, so truncation to u32
        // is lossless.
        (self.r_info.get() >> 32) as u32
    }

    /// Relocation type encoded in `r_info` (ELF64 layout).
    #[allow(clippy::cast_possible_truncation)]
    pub const fn r_type(&self) -> u32 {
        // The mask keeps only the low 32 bits, so truncation is lossless.
        (self.r_info.get() & 0xffff_ffff) as u32
    }
}

impl Sym64 {
    /// Symbol binding: the upper nibble of `st_info`.
    pub const fn bind(&self) -> u8 {
        self.st_info >> 4
    }

    /// Symbol type: the lower nibble of `st_info`.
    pub const fn type_(&self) -> u8 {
        self.st_info & 0x0f
    }

    /// Symbol visibility (`STV_*`): the low two bits of `st_other`.
    pub const fn visibility(&self) -> u8 {
        self.st_other & STV_MASK
    }
}

/// Version need record (`Elf_Verneed`), 16 bytes. One per versioned dependency
/// soname in `.gnu.version_r`.
///
/// The first follows at the section start; `vn_next` gives the byte offset of
/// the next, zero on the last entry. Each is followed by `vn_cnt` of
/// [`Vernaux64`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Verneed64 {
    /// Version of this structure (always 1).
    pub vn_version: U16,
    /// Number of [`Vernaux64`] entries immediately following this record.
    pub vn_cnt: U16,
    /// Offset of the dependency soname string within `.dynstr`.
    pub vn_file: U32,
    /// Byte offset to the first [`Vernaux64`] from this record's start.
    pub vn_aux: U32,
    /// Byte offset to the next [`Verneed64`] from this record's start, or 0.
    pub vn_next: U32,
}

/// One version requirement within a [`Verneed64`] (`Elf_Vernaux`), 16 bytes.
///
/// Records one version name (e.g. `GLIBC_2.2.5`) the dependency must supply,
/// plus the index the loader matches against `.gnu.version` for symbols that
/// reference it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vernaux64 {
    /// ELF hash of the version name string.
    pub vna_hash: U32,
    /// `VER_FLG_*` flags (e.g. `VER_FLG_WEAK`).
    pub vna_flags: U16,
    /// Version index this entry assigns; `.gnu.version` references this value.
    pub vna_other: U16,
    /// Offset of the version name string within `.dynstr`.
    pub vna_name: U32,
    /// Byte offset to the next [`Vernaux64`] from this entry's start, or 0.
    pub vna_next: U32,
}

/// Version definition record (`Elf_Verdef`), 20 bytes. One per version an
/// object exports (in `.gnu.version_d`): the base version (the soname) and
/// each exported version name (e.g. `GLIBC_2.2.5`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Verdef64 {
    /// Version of this structure (always 1).
    pub vd_version: U16,
    /// `VER_FLG_*` flags (`VER_FLG_BASE` for the soname entry).
    pub vd_flags: U16,
    /// Version index this definition assigns; matches `.gnu.version`.
    pub vd_ndx: U16,
    /// Number of [`Verdaux64`] entries (version names; >= 1).
    pub vd_cnt: U16,
    /// ELF hash of the primary version name.
    pub vd_hash: U32,
    /// Byte offset to the first [`Verdaux64`] from this record's start.
    pub vd_aux: U32,
    /// Byte offset to the next [`Verdef64`] from this record's start, or 0.
    pub vd_next: U32,
}

/// One auxiliary name within a [`Verdef64`] (`Elf_Verdaux`), 8 bytes: the
/// version name itself, plus (for the base entry) the parent version name it
/// inherits from.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Verdaux64 {
    /// Offset of the name string within `.dynstr`.
    pub vda_name: U32,
    /// Byte offset to the next [`Verdaux64`] from this entry's start, or 0.
    pub vda_next: U32,
}
