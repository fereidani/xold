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
pub mod imports;
pub mod layout;
pub mod lc;
pub mod live;
pub mod read;
pub mod reloc;
pub mod sections;
pub mod structs;
pub mod symtab;
pub mod tapi;
pub mod unwind;
pub mod writer;

pub use got::GotPlan;
pub use read::{MachOFile, MachSymbolIter, MachSymbolTable};
pub use reloc::MachoTarget;
pub use structs::{MachReloc, MachSection, MachSymbol};
pub use writer::{link_macho, link_macho_with_options};

/// The version tuple carried by `LC_BUILD_VERSION`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlatformVersion {
    /// Mach platform number (`PLATFORM_MACOS` is 1).
    pub platform: u32,
    /// Minimum supported OS, packed as `major << 16 | minor << 8 | patch`.
    pub min_os: u32,
    /// SDK version, in the same packed form as [`Self::min_os`].
    pub sdk: u32,
}

/// One dynamic dependency and the names its TAPI stub exports.
pub struct Dylib<'a> {
    pub install_name: &'a str,
    pub exports: &'a [String],
}

/// Mach-O-specific settings understood by the image writer.
#[derive(Default)]
pub struct LinkOptions<'a> {
    /// Architecture the driver selected. The input objects remain the source
    /// of truth; a disagreement is a command-line error.
    pub arch: Option<MachoTarget>,
    /// Whether dyld, rather than an `LC_UNIXTHREAD`, starts the executable.
    pub dynamic: bool,
    /// Whether the output is a dynamic library rather than an executable.
    pub dylib: bool,
    /// The dylib identity written as `LC_ID_DYLIB`. Required when `dylib` is
    /// true and absent for executables.
    pub install_name: Option<&'a str>,
    /// An explicit export allow-list. `None` publishes every public external
    /// definition in a dylib; `Some` publishes only these exact names.
    pub exported_symbols: Option<&'a [String]>,
    /// Optional deployment target and SDK load command.
    pub platform: Option<PlatformVersion>,
    /// Whether ld64-style section reachability and dead stripping is enabled.
    pub dead_strip: bool,
    /// Install names to record as `LC_LOAD_DYLIB` commands.
    pub dylibs: &'a [Dylib<'a>],
}
