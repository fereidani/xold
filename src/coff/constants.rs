//! COFF/PE format constants.
//!
//! Only the values the reader and the relocation tables reference are defined
//! here, mirroring how `macho::constants` is organised. Values are kept as
//! plain literals matching the on-disk little-endian encoding so they
//! interoperate directly with the structures returned by the `object` crate.
//! Relocation type numbers live with their arch tables
//! (`reloc::coff_x86_64`, `reloc::coff_i386`), matching the ELF and Mach-O
//! convention.

// --- machine types (ImageFileHeader.machine) ------------------------------

/// The `x86_64` machine type. Both the MSVC and GNU `x86_64` COFF objects
/// report this.
pub const IMAGE_FILE_MACHINE_AMD64: u16 = 0x8664;
/// The i386 machine type.
pub const IMAGE_FILE_MACHINE_I386: u16 = 0x014c;
/// The ARM64 (`AArch64`) machine type.
pub const IMAGE_FILE_MACHINE_ARM64: u16 = 0xAA64;
/// The ARM Thumb-2 machine type.
pub const IMAGE_FILE_MACHINE_ARMNT: u16 = 0x01c4;
/// Used by the bigobj header sentinel, not a real machine.
pub const IMAGE_FILE_MACHINE_UNKNOWN: u16 = 0x0000;

// --- section characteristics ----------------------------------------------

/// Section contains code.
pub const IMAGE_SCN_CNT_CODE: u32 = 0x0000_0020;
/// Section contains initialised data.
pub const IMAGE_SCN_CNT_INITIALIZED_DATA: u32 = 0x0000_0040;
/// Section contains uninitialised data (`.bss`, no file space).
pub const IMAGE_SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;
/// Section contents are COMDAT (comdat group member).
pub const IMAGE_SCN_LNK_COMDAT: u32 = 0x0000_1000;
/// Section is discardable (may be dropped from the final image).
pub const IMAGE_SCN_MEM_DISCARDABLE: u32 = 0x0200_0000;
/// Section is executable.
pub const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
/// Section is readable.
pub const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;
/// Section is writeable.
pub const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;
/// Mask selecting the alignment field of `characteristics`.
pub const IMAGE_SCN_ALIGN_MASK: u32 = 0x00F0_0000;
/// 16-byte alignment, the COFF default when no alignment is stated.
pub const IMAGE_SCN_ALIGN_16BYTES: u32 = 0x0050_0000;

// --- symbol storage class --------------------------------------------------

/// External symbol, visible across object boundaries.
pub const IMAGE_SYM_CLASS_EXTERNAL: u8 = 0x02;
/// Static (file-scoped) symbol; section symbols use this too.
pub const IMAGE_SYM_CLASS_STATIC: u8 = 0x03;
/// Label symbol.
pub const IMAGE_SYM_CLASS_LABEL: u8 = 0x06;
/// Section symbol; carries an auxiliary section record.
pub const IMAGE_SYM_CLASS_SECTION: u8 = 0x68;
/// File-name symbol; its auxiliary records hold the source name.
pub const IMAGE_SYM_CLASS_FILE: u8 = 0x67;
/// Weak external; resolves to a default if no definition is found.
pub const IMAGE_SYM_CLASS_WEAK_EXTERNAL: u8 = 0x69;

// --- special section numbers (ImageSymbol.section_number) ------------------

/// Symbol is undefined or common (the value is the common size).
pub const IMAGE_SYM_UNDEFINED: i32 = 0;
/// Symbol is an absolute value, not tied to any section.
pub const IMAGE_SYM_ABSOLUTE: i32 = -1;
/// Symbol is a special debug item.
pub const IMAGE_SYM_DEBUG: i32 = -2;

// --- COMDAT selection (ImageAuxSymbolSection.selection) --------------------

/// Cannot be combined with other definitions.
pub const IMAGE_COMDAT_SELECT_NODUPLICATES: u8 = 1;
/// Pick any one definition.
pub const IMAGE_COMDAT_SELECT_ANY: u8 = 2;
/// Pick the definition with the largest size.
pub const IMAGE_COMDAT_SELECT_SAME_SIZE: u8 = 3;
/// Pick the definition with exact byte match.
pub const IMAGE_COMDAT_SELECT_EXACT_MATCH: u8 = 4;
/// This section is associated with another COMDAT section.
pub const IMAGE_COMDAT_SELECT_ASSOCIATIVE: u8 = 5;
/// Pick the definition with the largest size (alias).
pub const IMAGE_COMDAT_SELECT_LARGEST: u8 = 6;

// --- symbol type derived-component decode ----------------------------------

/// Mask selecting the derived type of `ImageSymbol.typ`.
pub const N_TMASK: u16 = 0x0030;
/// Shift of the derived-type nibble.
pub const N_BTSHFT: u32 = 4;
/// The derived-type value marking a function (`IMAGE_SYM_DTYPE_FUNCTION`).
pub const IMAGE_SYM_DTYPE_FUNCTION: u16 = 2;

// --- PE image constants (writer) -------------------------------------------

/// `MZ`: the DOS stub magic that opens every PE image.
pub const IMAGE_DOS_SIGNATURE: u16 = 0x5A4D;
/// `PE\0\0`: the PE header signature that follows the DOS stub.
pub const IMAGE_NT_SIGNATURE: u32 = 0x0000_4550;
/// Optional-header magic for PE32+ (64-bit) images.
pub const IMAGE_NT_OPTIONAL_HDR64_MAGIC: u16 = 0x020B;
/// File is a runnable image (no unresolved externals).
pub const IMAGE_FILE_EXECUTABLE_IMAGE: u16 = 0x0002;
/// The image is a DLL, not an executable: it cannot run standalone but is
/// loaded by the PE loader through `LoadLibrary` / the import table.
pub const IMAGE_FILE_DLL: u16 = 0x2000;
/// Application may use addresses above 2 GiB.
pub const IMAGE_FILE_LARGE_ADDRESS_AWARE: u16 = 0x0020;
/// Line numbers stripped (set by every modern linker).
pub const IMAGE_FILE_LINE_NUMS_STRIPPED: u16 = 0x0004;
/// Local symbols stripped from the image.
pub const IMAGE_FILE_LOCAL_SYMS_STRIPPED: u16 = 0x0008;

/// Number of entries in the optional-header data directory array.
pub const IMAGE_NUMBEROF_DIRECTORY_ENTRIES: usize = 16;

/// Data directory index: export table.
pub const IMAGE_DIRECTORY_ENTRY_EXPORT: usize = 0;
/// Data directory index: import table.
pub const IMAGE_DIRECTORY_ENTRY_IMPORT: usize = 1;
/// Data directory index: the resource directory tree (`.rsrc`).
pub const IMAGE_DIRECTORY_ENTRY_RESOURCE: usize = 2;
/// Data directory index: base relocation table.
pub const IMAGE_DIRECTORY_ENTRY_BASERELOC: usize = 5;
/// Data directory index: the TLS directory (`ImageTlsDirectory64`).
pub const IMAGE_DIRECTORY_ENTRY_TLS: usize = 9;
/// Data directory index: import address table (IAT).
pub const IMAGE_DIRECTORY_ENTRY_IAT: usize = 12;

/// Windows subsystem: character-mode (console) application.
pub const IMAGE_SUBSYSTEM_WINDOWS_CUI: u16 = 3;

/// The image may be relocated at load time (ASLR). Left clear for a minimal
/// fixed-image-base executable so no base-relocation table is required.
pub const IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE: u16 = 0x0040;
/// The image supports addresses above 2 GiB.
pub const IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA: u16 = 0x0020;
/// Data execution prevention: the stack is non-executable.
pub const IMAGE_DLLCHARACTERISTICS_NX_COMPAT: u16 = 0x0100;

/// On-disk size of the DOS header (`ImageDosHeader`).
pub const IMAGE_SIZEOF_DOS_HEADER: usize = 64;
/// On-disk size of an `ImageSectionHeader`.
pub const IMAGE_SIZEOF_SECTION_HEADER: usize = 40;
/// On-disk size of the COFF `ImageFileHeader`.
pub const IMAGE_SIZEOF_FILE_HEADER_PE: usize = 20;
/// On-disk size of `ImageOptionalHeader64` excluding the data-directory array.
pub const IMAGE_SIZEOF_OPTIONAL_HEADER64: usize = 112;
/// On-disk size of an `ImageImportDescriptor`.
pub const IMAGE_SIZEOF_IMPORT_DESCRIPTOR: usize = 20;
/// On-disk size of `ImageTlsDirectory64` (4 VA fields + 2 dword fields).
pub const IMAGE_SIZEOF_TLS_DIRECTORY64: usize = 40;

/// Conventional load address for an x86-64 executable image.
pub const IMAGE_BASE_X86_64: u64 = 0x0000_0001_4000_0000;
/// Conventional preferred load address for an x86-64 DLL image.
///
/// Distinct from the executable base so a process can hold both; the loader
/// relocates the DLL elsewhere via the base-relocation table when this address
/// is unavailable.
pub const IMAGE_BASE_DLL_X86_64: u64 = 0x0000_0001_8000_0000;
/// Virtual-address alignment of every section (one page).
pub const SECTION_ALIGNMENT: u32 = 0x1000;
/// File-offset alignment of every section's raw data.
pub const FILE_ALIGNMENT: u32 = 0x0200;

// --- base-relocation types (ImageBaseRelocation entry, high nibble) --------

/// A no-op base-relocation entry: padding that fills a block out to a 4-byte
/// boundary, or marks the end of a page block.
pub const IMAGE_REL_BASED_ABSOLUTE: u16 = 0;
/// A 64-bit base relocation: the loader adds the relocation delta to the 8-byte
/// value at the entry's offset. The only kind a PE32+ x86-64 image needs.
pub const IMAGE_REL_BASED_DIR64: u16 = 10;

/// The byte size of an `ImageExportDirectory`.
pub const IMAGE_SIZEOF_EXPORT_DIRECTORY: usize = 40;
