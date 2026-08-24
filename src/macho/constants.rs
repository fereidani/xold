//! Mach-O format constants.
//!
//! Only the values the reader and the relocation tables reference are defined
//! here, mirroring how `elf::constants` is organised. Values are kept as plain
//! `u32`/`u8` literals matching the on-disk encoding so they interoperate
//! directly with the structures returned by the `object` crate, without a
//! wrapper enum. Relocation type numbers live with their arch tables
//! (`reloc::macho_x86_64`, `reloc::macho_arm64`), matching the ELF convention.

// --- magic numbers (first four bytes of the file) ------------------------

// Each pair is one value and its byte swap. `MAGIC` is what the first four
// bytes read as when the file's byte order matches the reader's, and `CIGAM`
// ("magic" reversed, the traditional spelling) is what they read as when it
// does not. The reader is little-endian, so the `MAGIC` spellings are the
// ones a little-endian file on disk produces.

/// 64-bit Mach-O, little-endian byte order on disk. This is the magic a
/// darwin `x86_64` / `arm64` object carries.
pub const MH_MAGIC_64: u32 = 0xFEED_FACF;
/// 64-bit Mach-O, big-endian byte order on disk (the swapped form of
/// [`MH_MAGIC_64`]).
pub const MH_CIGAM_64: u32 = 0xCFFA_EDFE;
/// 32-bit Mach-O, little-endian.
pub const MH_MAGIC: u32 = 0xFEED_FACE;
/// 32-bit Mach-O, big-endian.
pub const MH_CIGAM: u32 = 0xCEFA_EDFE;
/// Multi-architecture ("fat") container, 32-bit offsets.
///
/// A fat header is defined to be big-endian on disk whatever it contains, so
/// a real container reads as [`FAT_CIGAM`] here and never as this. The value
/// is kept because it is the spelling the headers give and the one a
/// byte-swapped reader would see.
pub const FAT_MAGIC: u32 = 0xCAFE_BABE;
/// Multi-architecture container, 64-bit offsets. See [`FAT_MAGIC`].
pub const FAT_MAGIC_64: u32 = 0xCAFE_BABF;
/// The swapped form of [`FAT_MAGIC`], and so what an actual fat container's
/// first four bytes read as on a little-endian host.
pub const FAT_CIGAM: u32 = 0xBEBA_FECA;
/// The swapped form of [`FAT_MAGIC_64`]. See [`FAT_CIGAM`].
pub const FAT_CIGAM_64: u32 = 0xBFBA_FECA;

// --- filetype ------------------------------------------------------------

/// Relocatable object file (compiler output, the linker's input).
pub const MH_OBJECT: u32 = 1;
/// Demand-paged executable file (the linker's output for a program).
pub const MH_EXECUTE: u32 = 2;
/// Dynamically linked shared library.
pub const MH_DYLIB: u32 = 6;

// --- cpu types and subtypes ----------------------------------------------

/// The `ABI64` flag, `OR`-ed into a 32-bit cpu type to mark a 64-bit
/// architecture.
pub const CPU_ARCH_ABI64: u32 = 0x0100_0000;
/// The base 32-bit x86 cpu type.
pub const CPU_TYPE_X86: u32 = 0x0000_0007;
/// The base 32-bit ARM cpu type.
pub const CPU_TYPE_ARM: u32 = 0x0000_000c;
/// `x86_64`. All darwin `x86_64` objects report this `cputype`.
pub const CPU_TYPE_X86_64: u32 = CPU_TYPE_X86 | CPU_ARCH_ABI64;
/// `ARM64` (`AArch64`). All darwin `arm64` objects report this `cputype`.
pub const CPU_TYPE_ARM64: u32 = CPU_TYPE_ARM | CPU_ARCH_ABI64;

/// All `x86_64` cpus share this subtype for relocatable objects.
pub const CPU_SUBTYPE_X86_64_ALL: u32 = 3;
/// All `arm64` cpus share this subtype for relocatable objects.
pub const CPU_SUBTYPE_ARM64_ALL: u32 = 0;

// --- load commands -------------------------------------------------------

/// 64-bit segment load command; carries the section array for `MH_OBJECT`.
pub const LC_SEGMENT_64: u32 = 0x19;
/// Symbol table load command; locates the `nlist` array and strings.
pub const LC_SYMTAB: u32 = 0x02;
/// Dynamic symbol table load command; describes local/external/undefined
/// symbol ranges and the indirect-symbol table.
pub const LC_DYSYMTAB: u32 = 0x0b;
/// Load the dynamic linker named by the command's string payload.
pub const LC_LOAD_DYLINKER: u32 = 0x0e;
/// Load one dynamic library dependency.
pub const LC_LOAD_DYLIB: u32 = 0x0c;
/// Identity of a dynamic library, using the same payload as `LC_LOAD_DYLIB`.
pub const LC_ID_DYLIB: u32 = 0x0d;
/// A 16-byte unique image identifier.
pub const LC_UUID: u32 = 0x1b;
/// Main entry point (`entryoff` into `__TEXT`). Modern darwin executables use
/// this in preference to `LC_UNIXTHREAD`.
pub const LC_MAIN: u32 = 0x28 | LC_REQ_DYLD;
/// Deployment platform, minimum OS and SDK version.
pub const LC_BUILD_VERSION: u32 = 0x32;
/// Classic dyld rebase/bind/export byte streams.
pub const LC_DYLD_INFO_ONLY: u32 = 0x22 | LC_REQ_DYLD;
/// Flag bit OR-ed into a load command `cmd` when dyld must understand the
/// command to launch the image. `LC_MAIN` sets it because a `LC_MAIN`-only
/// image has no thread state to boot directly.
pub const LC_REQ_DYLD: u32 = 0x8000_0000;

/// `LC_UNIXTHREAD`: the entry state for an image that launches without a
/// dynamic linker.
///
/// `LC_MAIN` sets [`LC_REQ_DYLD`], and XNU's `load_main` demands a
/// `LC_LOAD_DYLINKER` alongside it, so a true-static image carrying `LC_MAIN`
/// can never exec. lld pairs `LC_MAIN` with a dylinker only.
pub const LC_UNIXTHREAD: u32 = 0x05;

/// `x86_THREAD_STATE64`, and its size in 32-bit words.
pub const X86_THREAD_STATE64: u32 = 4;
/// 21 64-bit registers.
pub const X86_THREAD_STATE64_COUNT: u32 = 42;
/// Index of `rip` among the 21 registers, counting from `rax`.
pub const X86_THREAD_STATE64_RIP: usize = 16;

/// `ARM_THREAD_STATE64`, and its size in 32-bit words.
pub const ARM_THREAD_STATE64: u32 = 6;
/// 29 general registers, then fp, lr, sp, pc, then cpsr and its padding.
pub const ARM_THREAD_STATE64_COUNT: u32 = 68;
/// Index of `pc` among the 34 64-bit slots.
pub const ARM_THREAD_STATE64_PC: usize = 32;

/// The `__PAGEZERO` segment name and extent.
///
/// ld64 and lld emit it unconditionally for an executable: it covers the whole
/// low 4 GiB with no protection, so a null dereference faults instead of
/// reaching a mappable page.
pub const PAGEZERO_SIZE: u64 = 0x1_0000_0000;

// --- section flags -------------------------------------------------------

/// Mask selecting the section type from `flags`.
pub const SECTION_TYPE: u32 = 0x0000_00ff;
/// Regular section (code or data with no special handling).
pub const S_REGULAR: u32 = 0x0;
/// Zero-fill section (occupies no file space, like `.bss`).
pub const S_ZEROFILL: u32 = 0x1;
/// Variable-length string literals.
pub const S_CSTRING_LITERALS: u32 = 0x2;
/// 4-byte literal constants.
pub const S_4BYTE_LITERALS: u32 = 0x3;
/// 8-byte literal constants.
pub const S_8BYTE_LITERALS: u32 = 0x4;
/// Non-lazy symbol pointer table (the GOT on darwin): one 8-byte slot per
/// external data symbol, resolved at link time for a static executable.
pub const S_NON_LAZY_SYMBOL_POINTERS: u32 = 0x6;
/// Fixed-size symbol stubs; `reserved2` is the size of one stub.
pub const S_SYMBOL_STUBS: u32 = 0x8;
/// Array of function pointers run when the image is loaded.
pub const S_MOD_INIT_FUNC_POINTERS: u32 = 0x9;
/// Array of function pointers run when the image is unloaded.
pub const S_MOD_TERM_FUNC_POINTERS: u32 = 0x0a;
/// Thread-local zero-fill data.
pub const S_THREAD_LOCAL_ZEROFILL: u32 = 0x12;

/// Mask selecting the section attributes (above the type byte).
pub const SECTION_ATTRIBUTES: u32 = 0xffff_ff00;
/// Section holds machine instructions (`S_ATTR_SOME_INSTRUCTIONS`).
pub const S_ATTR_SOME_INSTRUCTIONS: u32 = 0x0000_0400;
/// Section is pure instructions (`S_ATTR_PURE_INSTRUCTIONS`).
pub const S_ATTR_PURE_INSTRUCTIONS: u32 = 0x8000_0000;
/// The section is a root for ld64 dead stripping.
pub const S_ATTR_NO_DEAD_STRIP: u32 = 0x1000_0000;

// --- symbol `n_type` masks and values ------------------------------------

/// Mask selecting the debug (`N_STAB`) bits of `n_type`.
pub const N_STAB: u8 = 0xe0;
/// Private external symbol (visible to the linker, not exported).
pub const N_PEXT: u8 = 0x10;
/// Mask selecting the symbol type bits of `n_type`.
pub const N_TYPE: u8 = 0x0e;
/// External symbol (visible across object boundaries).
pub const N_EXT: u8 = 0x01;
/// Undefined symbol (a reference to be resolved by the linker).
pub const N_UNDF: u8 = 0x00;
/// Absolute symbol (not tied to any section).
pub const N_ABS: u8 = 0x02;
/// Indirect symbol (alias for another symbol).
pub const N_INDR: u8 = 0x0a;
/// Section symbol: `n_sect` selects which section (1-based ordinal).
pub const N_SECT: u8 = 0x0e;

// --- symbol `n_desc` flags --------------------------------------------------

/// `n_desc` flag: this definition is weak (a strong definition overrides it).
pub const N_WEAK_DEF: u16 = 0x0080;
/// `n_desc` flag: this reference is weak (binds to zero when undefined).
pub const N_WEAK_REF: u16 = 0x0040;

// --- indirect symbol table entries ---------------------------------------

/// Indirect-symbol entry marking a slot bound to no symbol (an absolute).
pub const INDIRECT_SYMBOL_ABS: u32 = 0x4000_0000;
/// Indirect-symbol entry marking a slot bound to a symbol with no table
/// row.
pub const INDIRECT_SYMBOL_LOCAL: u32 = 0x8000_0000;

// --- relocation bitfield encoding ----------------------------------------

/// The bit in `r_word0` that marks a scattered relocation entry. Scattered
/// relocations are a legacy 32-bit encoding; `x86_64` never uses them.
pub const R_SCATTERED: u32 = 0x8000_0000;

// --- virtual-memory protection bits (segment maxprot / initprot) ---------

/// No access.
pub const VM_PROT_NONE: u32 = 0x0;
/// Read access.
pub const VM_PROT_READ: u32 = 0x1;
/// Write access.
pub const VM_PROT_WRITE: u32 = 0x2;
/// Execute access.
pub const VM_PROT_EXECUTE: u32 = 0x4;

// --- mach_header flags ---------------------------------------------------

/// The image has no undefined symbols (every reference resolves at link time).
/// Set on a fully-static executable so dyld skips the undefined check.
pub const MH_NOUNDEFS: u32 = 0x0000_0001;
/// Image is an input to the dynamic linker.
pub const MH_DYLDLINK: u32 = 0x0000_0004;
/// Undefined symbols carry two-level library ordinals.
pub const MH_TWOLEVEL: u32 = 0x0000_0080;
/// Executable may be slid from its preferred address.
pub const MH_PIE: u32 = 0x0020_0000;
/// Image contains `S_THREAD_LOCAL_VARIABLES` descriptors for dyld to register.
pub const MH_HAS_TLV_DESCRIPTORS: u32 = 0x0080_0000;
