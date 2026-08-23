//! On-disk Mach-O 64-bit structures for writing.
//!
//! Every type is `#[repr(C)]` and derives `bytemuck::Pod`, so the writer casts
//! it to bytes directly. Field order and sizes match the 64-bit Mach-O layout;
//! this mirrors how the ELF writer serialises `Ehdr64` / `Shdr64`. Only the
//! load commands and records the writer emits are defined here; the read side
//! reuses the `object` crate and the plain view records in [`super::structs`].
//!
//! Every target xold writes is little-endian (`x86_64`, `arm64`), so the
//! on-disk order is fixed. The *host* is not, and `bytemuck` reinterprets
//! bytes without reordering them, so integer fields are [`crate::endian`]
//! little-endian types rather than primitives: plain `u32`/`u64` would emit
//! host-order integers, correct on a little-endian host and invalid on any
//! other. The types are `#[repr(transparent)]`, so the layout is unchanged and
//! the conversion compiles away where it is not needed.

use bytemuck::{Pod, Zeroable};

use crate::endian::{U16, U32, U64};

/// `mach_header_64`, 32 bytes. The first 32 bytes of every 64-bit Mach-O.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MachHeader64 {
    /// `MH_MAGIC_64` / `MH_CIGAM_64`.
    pub magic: U32,
    /// `cputype` (`CPU_TYPE_X86_64`, `CPU_TYPE_ARM64`, ...).
    pub cputype: U32,
    /// `cpusubtype`.
    pub cpusubtype: U32,
    /// `filetype` (`MH_OBJECT`, `MH_EXECUTE`, ...).
    pub filetype: U32,
    /// Number of load commands that follow the header.
    pub ncmds: U32,
    /// Total byte size of the load-command table.
    pub sizeofcmds: U32,
    /// `MH_*` flags.
    pub flags: U32,
    /// Reserved (`reserved` in `mach_header_64`).
    pub reserved: U32,
}

/// `segment_command_64`, 72 bytes. Describes a segment and precedes its
/// `section_64` headers (when `nsects > 0`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SegmentCommand64 {
    /// `LC_SEGMENT_64`.
    pub cmd: U32,
    /// Total size: the command header plus `nsects` section headers.
    pub cmdsize: U32,
    /// 16-byte segment name, NUL-padded.
    pub segname: [u8; 16],
    /// Starting virtual address.
    pub vmaddr: U64,
    /// Virtual size (`__bss` extends this past `filesize`).
    pub vmsize: U64,
    /// Starting file offset.
    pub fileoff: U64,
    /// File size (zero for a pure `__bss` segment).
    pub filesize: U64,
    /// Maximum permitted protection.
    pub maxprot: U32,
    /// Initial protection.
    pub initprot: U32,
    /// Number of `section_64` headers immediately following.
    pub nsects: U32,
    /// Segment flags.
    pub flags: U32,
}

/// `section_64`, 80 bytes. One section header, embedded in its segment command.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SectionHeader64 {
    /// 16-byte section name, NUL-padded.
    pub sectname: [u8; 16],
    /// 16-byte segment name, NUL-padded.
    pub segname: [u8; 16],
    /// Virtual address (`addr`).
    pub addr: U64,
    /// Size in bytes (`size`).
    pub size: U64,
    /// File offset (`offset`); 0 for `S_ZEROFILL`.
    pub offset: U32,
    /// Power-of-two alignment, as stored (`align`).
    pub align: U32,
    /// File offset of the relocation entries (`reloff`).
    pub reloff: U32,
    /// Number of relocation entries (`nreloc`).
    pub nreloc: U32,
    /// Section flags (type in the low byte, attributes above).
    pub flags: U32,
    /// Reserved (`reserved1`; indirect-symbol index for symbol-stub sections).
    pub reserved1: U32,
    /// Reserved (`reserved2`; stub size for symbol-stub sections).
    pub reserved2: U32,
    /// Reserved (`reserved3`; arm64 stub index, otherwise 0).
    pub reserved3: U32,
}

/// `symtab_command`, 24 bytes. `LC_SYMTAB`: locates the `nlist_64` array and
/// the string table.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SymtabCommand {
    /// `LC_SYMTAB`.
    pub cmd: U32,
    pub cmdsize: U32,
    /// File offset of the first `nlist_64`.
    pub symoff: U32,
    /// Number of `nlist_64` entries.
    pub nsyms: U32,
    /// File offset of the string table.
    pub stroff: U32,
    /// Byte size of the string table.
    pub strsize: U32,
}

/// `dysymtab_command`, 80 bytes. `LC_DYSYMTAB`: indexes into the `nlist` array
/// partitioning it into local, external-defined and undefined ranges.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DysymtabCommand {
    /// `LC_DYSYMTAB`.
    pub cmd: U32,
    pub cmdsize: U32,
    /// Index of the first local symbol.
    pub ilocalsym: U32,
    /// Number of local symbols.
    pub nlocalsym: U32,
    /// Index of the first external-defined symbol.
    pub iextdefsym: U32,
    /// Number of external-defined symbols.
    pub nextdefsym: U32,
    /// Index of the first undefined symbol.
    pub iundefsym: U32,
    /// Number of undefined symbols.
    pub nundefsym: U32,
    /// File offset of the table-of-contents (0 when none).
    pub tocoff: U32,
    /// Number of table-of-contents entries (0 when none).
    pub ntoc: U32,
    /// File offset of the module table (0 when none).
    pub modtaboff: U32,
    /// Number of module-table entries (0 when none).
    pub nmodtab: U32,
    /// File offset of the external reference table (0 when none).
    pub extrefsymoff: U32,
    /// Number of external reference entries (0 when none).
    pub nextrefsyms: U32,
    /// File offset of the indirect symbol table (0 when none).
    pub indirectsymoff: U32,
    /// Number of indirect symbol entries (0 when none).
    pub nindirectsyms: U32,
    /// File offset of the external relocation table (0 when none).
    pub extreloff: U32,
    /// Number of external relocation entries (0 when none).
    pub nextrel: U32,
    /// File offset of the local relocation table (0 when none).
    pub locreloff: U32,
    /// Number of local relocation entries (0 when none).
    pub nlocrel: U32,
}

/// `entry_point_command`, 24 bytes. `LC_MAIN`: the file offset of `main()`
/// within `__TEXT`, plus the initial stack size to request from dyld.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct EntryPointCommand {
    /// `LC_MAIN`.
    pub cmd: U32,
    pub cmdsize: U32,
    /// File offset of `main()` from the start of `__TEXT`.
    pub entryoff: U64,
    /// Initial stack size in bytes (64 KiB if zero).
    pub stacksize: U64,
}

/// `thread_command` header, 16 bytes, followed by `count` 32-bit words of
/// register state.
///
/// `LC_UNIXTHREAD` says where a statically linked image begins: the kernel
/// loads this register file and jumps to its program counter, with no dynamic
/// linker in the picture.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ThreadCommand {
    /// `LC_UNIXTHREAD`.
    pub cmd: U32,
    pub cmdsize: U32,
    /// `x86_THREAD_STATE64` or `ARM_THREAD_STATE64`.
    pub flavor: U32,
    /// Size of the state that follows, in 32-bit words.
    pub count: U32,
}

/// `uuid_command`, 24 bytes. `LC_UUID`: a 16-byte image identifier.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct UuidCommand {
    /// `LC_UUID`.
    pub cmd: U32,
    pub cmdsize: U32,
    /// The 16-byte UUID.
    pub uuid: [u8; 16],
}

/// `nlist_64`, 16 bytes. One symbol-table entry.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Nlist64 {
    /// String-table index of the name (0 for the null symbol).
    pub n_strx: U32,
    /// Type byte (`n_type`: `N_STAB` / `N_TYPE` / `N_EXT`).
    pub n_type: u8,
    /// Section ordinal (`n_sect`); 0 if not section-relative.
    pub n_sect: u8,
    /// Description (`n_desc`: flags, reference method, ...).
    pub n_desc: U16,
    /// Value (address for defined symbols, size for common symbols).
    pub n_value: U64,
}

impl Nlist64 {
    /// An all-zero entry: the conventional symbol-table slot 0.
    pub const fn zeroed() -> Self {
        Self {
            n_strx: U32::new(0),
            n_type: 0,
            n_sect: 0,
            n_desc: U16::new(0),
            n_value: U64::new(0),
        }
    }
}

/// `relocation_info`, 8 bytes. One relocation entry.
///
/// The writer emits no output relocations for a fully-resolved static link, but
/// the layout matches the reader so a future dynamic link can serialise them.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RelocEntry {
    /// `r_word0`: address within the section plus the symbol/section number.
    pub r_word0: U32,
    /// `r_word1`: flags (pcrel, length, extern, type).
    pub r_word1: U32,
}
