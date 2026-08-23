//! On-disk PE header structures, as plain `Pod` records written through
//! `bytemuck`.
//!
//! These mirror `object::pe` (`ImageDosHeader`, `ImageNtHeaders64`,
//! `ImageOptionalHeader64`, `ImageSectionHeader`, `ImageDataDirectory`,
//! `ImageImportDescriptor`, `ImageImportByName`) but are plain `Pod` records
//! the writer fills and serialises directly, without the `object` endian
//! machinery at every field. This matches how [`crate::macho::lc`] defines its
//! load-command records.
//!
//! Integer fields are [`crate::endian`] little-endian types, not primitives.
//! PE is little-endian on disk and `bytemuck` never reorders bytes, so plain
//! primitives would emit host-order integers -- correct on a little-endian
//! host and invalid on any other. The types are `#[repr(transparent)]`, so the
//! layout is unchanged and the conversion compiles away where it is not
//! needed.
//!
//! The optional header is split from its trailing data-directory array: the
//! array is written separately so the directory entries can be computed after
//! the section layout is known. `size_of_optional_header` in the file header
//! still reports the full extent (struct plus 16 data directories).

use bytemuck::{Pod, Zeroable};

use crate::endian::{U16, U32, U64};

/// The DOS stub that opens every PE image. `e_lfanew` locates the PE header;
/// every other field is zero for a minimal stub (no DOS relocations, no
/// message).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct DosHeader {
    pub e_magic: U16,
    pub e_cblp: U16,
    pub e_cp: U16,
    pub e_crlc: U16,
    pub e_cparhdr: U16,
    pub e_minalloc: U16,
    pub e_maxalloc: U16,
    pub e_ss: U16,
    pub e_sp: U16,
    pub e_csum: U16,
    pub e_ip: U16,
    pub e_cs: U16,
    pub e_lfarlc: U16,
    pub e_ovno: U16,
    pub e_res: [u16; 4],
    pub e_oemid: U16,
    pub e_oeminfo: U16,
    pub e_res2: [u16; 10],
    pub e_lfanew: U32,
}

/// The PE signature plus the COFF file header. The optional header follows in
/// memory but is written separately, so this record stops at
/// `size_of_optional_header`.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct NtFileHeader {
    pub signature: U32,
    pub machine: U16,
    pub number_of_sections: U16,
    pub time_date_stamp: U32,
    pub pointer_to_symbol_table: U32,
    pub number_of_symbols: U32,
    pub size_of_optional_header: U16,
    pub characteristics: U16,
}

/// `ImageOptionalHeader64` without the trailing data-directory array.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct OptionalHeader64 {
    pub magic: U16,
    pub major_linker_version: u8,
    pub minor_linker_version: u8,
    pub size_of_code: U32,
    pub size_of_initialized_data: U32,
    pub size_of_uninitialized_data: U32,
    pub address_of_entry_point: U32,
    pub base_of_code: U32,
    pub image_base: U64,
    pub section_alignment: U32,
    pub file_alignment: U32,
    pub major_operating_system_version: U16,
    pub minor_operating_system_version: U16,
    pub major_image_version: U16,
    pub minor_image_version: U16,
    pub major_subsystem_version: U16,
    pub minor_subsystem_version: U16,
    pub win32_version_value: U32,
    pub size_of_image: U32,
    pub size_of_headers: U32,
    pub check_sum: U32,
    pub subsystem: U16,
    pub dll_characteristics: U16,
    pub size_of_stack_reserve: U64,
    pub size_of_stack_commit: U64,
    pub size_of_heap_reserve: U64,
    pub size_of_heap_commit: U64,
    pub loader_flags: U32,
    pub number_of_rva_and_sizes: U32,
}

/// One data-directory entry: an RVA plus size.
#[derive(Clone, Copy, Pod, Zeroable, Default)]
#[repr(C)]
pub struct DataDirectory {
    pub virtual_address: U32,
    pub size: U32,
}

/// A PE output section header (`ImageSectionHeader`).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct SectionHeader {
    pub name: [u8; 8],
    pub virtual_size: U32,
    pub virtual_address: U32,
    pub size_of_raw_data: U32,
    pub pointer_to_raw_data: U32,
    pub pointer_to_relocations: U32,
    pub pointer_to_linenumbers: U32,
    pub number_of_relocations: U16,
    pub number_of_linenumbers: U16,
    pub characteristics: U32,
}

/// One import descriptor (`ImageImportDescriptor`). The import directory is an
/// array of these terminated by an all-zero entry.
#[derive(Clone, Copy, Pod, Zeroable, Default)]
#[repr(C)]
pub struct ImportDescriptor {
    /// RVA of the import lookup table (pre-binding name pointers).
    pub original_first_thunk: U32,
    pub time_date_stamp: U32,
    pub forwarder_chain: U32,
    /// RVA of the DLL name string (NUL-terminated).
    pub name: U32,
    /// RVA of the import address table (overwritten with addresses at load).
    pub first_thunk: U32,
}

/// The export directory header (`ImageExportDirectory`).
///
/// One of these opens the export table; it is followed by the function-RVA,
/// name-RVA and name-ordinal arrays it points at. The DLL name and the export
/// name strings sit after those arrays. All pointer fields are absolute RVAs.
#[derive(Clone, Copy, Pod, Zeroable, Default)]
#[repr(C)]
pub struct ExportDirectory {
    pub characteristics: U32,
    pub time_date_stamp: U32,
    pub major_version: U16,
    pub minor_version: U16,
    /// RVA of the NUL-terminated DLL name string.
    pub name: U32,
    /// The smallest ordinal exported; ordinals are `base + table index`.
    pub base: U32,
    /// Length of the `AddressOfFunctions` array.
    pub number_of_functions: U32,
    /// Length of both `AddressOfNames` and `AddressOfNameOrdinals`.
    pub number_of_names: U32,
    /// RVA of the function-RVA table (`number_of_functions` entries).
    pub address_of_functions: U32,
    /// RVA of the name-pointer table (`number_of_names` entries, sorted).
    pub address_of_names: U32,
    /// RVA of the name-ordinal table (`number_of_names` WORD entries).
    pub address_of_name_ordinals: U32,
}

/// One base-relocation block header (`ImageBaseRelocation`).
///
/// The block covers one 4 KiB page; the `(size_of_block - 8) / 2` entries that
/// follow are WORDs packing a 4-bit type in the high nibble and a 12-bit page
/// offset in the rest.
#[derive(Clone, Copy, Pod, Zeroable, Default)]
#[repr(C)]
pub struct BaseRelocation {
    /// RVA of the page this block patches.
    pub virtual_address: U32,
    /// Total block size in bytes, including this header and every entry.
    pub size_of_block: U32,
}

/// The TLS directory (`ImageTlsDirectory64`).
///
/// Pointed at by `IMAGE_DIRECTORY_ENTRY_TLS`. The loader allocates a per-thread
/// block of `EndAddressOfRawData - StartAddressOfRawData + SizeOfZeroFill`
/// bytes, copies the template raw data into it, fills `*AddressOfIndex` with
/// this module's TLS slot index, and walks the null-terminated callback array
/// at `AddressOfCallBacks`. Code accesses a `__declspec(thread)` variable
/// through `gs:[ThreadLocalStoragePointer + index*8]` plus the variable's
/// offset within the template (resolved by an `IMAGE_REL_AMD64_SECREL` fixup).
#[derive(Clone, Copy, Pod, Zeroable, Default)]
#[repr(C)]
pub struct TlsDirectory64 {
    /// VA of the first byte of the TLS template raw data.
    pub start_address_of_raw_data: U64,
    /// VA one past the last byte of the template raw data.
    pub end_address_of_raw_data: U64,
    /// VA of the 4-byte `_tls_index` slot the loader fills.
    pub address_of_index: U64,
    /// VA of a null-terminated array of TLS callback pointers.
    pub address_of_call_backs: U64,
    /// Bytes of zero-fill the loader appends after the raw template.
    pub size_of_zero_fill: U32,
    /// Alignment of the template, as an `IMAGE_SCN_ALIGN_*` characteristics
    /// value.
    pub characteristics: U32,
}

impl DosHeader {
    /// A minimal DOS header: `MZ` magic, the stub message offset, and
    /// `e_lfanew` pointing at `pe_offset`. All other fields are zero.
    pub fn minimal(pe_offset: u32) -> Self {
        Self {
            e_magic: U16::new(crate::coff::constants::IMAGE_DOS_SIGNATURE),
            // DOS headers measure their own size in 16-byte paragraphs; the
            // standard value for a one-paragraph header.
            e_cparhdr: U16::new(4),
            e_minalloc: U16::new(0xFFFF),
            e_maxalloc: U16::new(0xFFFF),
            e_sp: U16::new(0xB8),
            e_lfarlc: U16::new(0x40),
            e_lfanew: U32::new(pe_offset),
            ..Self::zeroed()
        }
    }
}
