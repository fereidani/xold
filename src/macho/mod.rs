//! Mach-O binary format support: on-disk constants, plain view records, a
//! zero-copy reader for 64-bit relocatable objects, and a writer + static link
//! driver that produces a 64-bit `MH_EXECUTE` image.
//!
//! Only 64-bit Mach-O is implemented, covering the `x86_64` and `arm64`
//! darwin targets. 32-bit and fat containers are rejected at parse time. The
//! reader backs the Mach-O relocation tables in [`crate::reloc`]; the writer is
//! a path fully separate from the ELF linker.

pub mod constants;
pub mod got;
pub mod layout;
pub mod lc;
pub mod read;
pub mod reloc;
pub mod sections;
pub mod structs;
pub mod symtab;
pub mod writer;

pub use got::GotPlan;
pub use read::{MachOFile, MachSymbolIter, MachSymbolTable};
pub use structs::{MachReloc, MachSection, MachSymbol};
pub use writer::link_macho;
