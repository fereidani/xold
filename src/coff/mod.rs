//! COFF/PE binary format support: constants, records, a reader, and a writer.
//!
//! Provides on-disk constants, plain view records, a zero-copy reader for bare
//! COFF relocatable objects and PE images, and a writer plus static link driver
//! that produces a PE32+ x86-64 executable.
//!
//! Bare COFF objects (`.obj`/`.o`) cover the initial `x86_64` MSVC/GNU and
//! `i386` MSVC targets; PE images (`.exe`/`.dll`) wrap a COFF object in a DOS
//! stub plus an optional header. The reader parses both: [`CoffFile`] for bare
//! objects (the linker's input) and [`PeImage`] for PE images (used to
//! round-trip and verify the writer's output). The reader is the foundation
//! for the COFF relocation tables in [`crate::reloc`]; the writer lives behind
//! [`pe`], [`layout`], [`imports`], [`sections`] and [`writer`], a path fully
//! separate from the ELF and Mach-O linkers. [`dllmap`] supplies the
//! well-known function-to-DLL mapping that drives the import table; [`exports`]
//! and [`basereloc`] build the export and base-relocation directories a DLL
//! image carries.

pub mod basereloc;
pub mod comdat;
pub mod commons;
pub mod constants;
pub mod dllmap;
pub mod exports;
pub mod imports;
pub mod layout;
pub mod pe;
pub mod read;
pub mod reloc;
pub mod sections;
pub mod structs;
pub mod tls;
pub mod writer;

pub use read::{CoffFile, CoffSymbolTable, PeImage, PeSection};
pub use structs::{
    CoffReloc, CoffSection, CoffSymbol, FunctionAux, SectionAux, SymbolAux,
    WeakAux,
};
pub use writer::link_coff;
