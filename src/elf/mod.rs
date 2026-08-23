//! ELF binary format support: on-disk structures, format constants, and a
//! zero-copy reader for relocatable object files.
//!
//! Only 64-bit little-endian ELF is implemented for now; this covers the
//! initial `x86-64`, `AArch64` and `RISC-V` targets. Word size and endianness
//! will be generalised when a 32-bit or big-endian target is added.

pub mod constants;
pub mod read;
pub mod structs;

pub use read::{
    Group, ObjectFile, Relocs, SymbolTable, VersionDaux, VersionDef,
    VersionNaux, VersionNeed, VersionTable,
};
pub use structs::{
    Dyn64, Ehdr64, Phdr64, Rela64, Shdr64, Sym64, Verdaux64, Verdef64,
    Vernaux64, Verneed64,
};
