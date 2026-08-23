//! ELF format constants.
//!
//! Only the values needed so far are defined; the set grows as the linker
//! handles more section types, relocations and machines. Constants are kept as
//! `u32`/`u16` literals matching the specification rather than a wrapper enum
//! so they interoperate directly with the on-disk structure fields.

// --- e_ident indices and magic -------------------------------------------

/// Start of the ELF magic bytes within `e_ident`.
pub const ELFMAG: [u8; 4] = *b"\x7fELF";
/// Index of the file class byte (32- vs 64-bit) within `e_ident`.
pub const EI_CLASS: usize = 4;
/// Index of the data encoding byte (endianness) within `e_ident`.
pub const EI_DATA: usize = 5;
/// 64-bit objects.
pub const ELFCLASS64: u8 = 2;
/// Little-endian data encoding.
pub const ELFDATA2LSB: u8 = 1;

// --- e_type: object file type --------------------------------------------

/// Relocatable object file.
pub const ET_REL: u16 = 1;
/// Executable file.
pub const ET_EXEC: u16 = 2;
/// Shared object / position-independent executable.
pub const ET_DYN: u16 = 3;

// --- e_machine: architectures --------------------------------------------

pub const EM_X86_64: u16 = 62;
pub const EM_AARCH64: u16 = 183;
pub const EM_RISCV: u16 = 243;

// --- p_type: segment types -----------------------------------------------

/// A loadable program segment.
pub const PT_LOAD: u32 = 1;
/// The dynamic segment: locates the `.dynamic` table for the runtime loader.
pub const PT_DYNAMIC: u32 = 2;
/// The program-header-table segment: lets a position-independent executable
/// recover its load base by comparing the on-disk `p_vaddr` with the runtime
/// address of the program headers.
pub const PT_PHDR: u32 = 6;
/// The program interpreter (`ld.so`): locates `.interp`, the NUL-terminated
/// path of the runtime linker the kernel should exec to load this image.
pub const PT_INTERP: u32 = 3;
/// Auxiliary note information: the only way a reader finds the image's notes,
/// since a note is located by segment rather than by section name.
pub const PT_NOTE: u32 = 4;
/// Thread-local storage segment; describes the static TLS block
/// (`.tdata`/`.tbss`) so the runtime can size and place the per-thread copy.
pub const PT_TLS: u32 = 7;
/// GNU extension: records the stack's permissions (an absent entry means an
/// executable stack). Emitting it with no execute bit marks the stack NX.
pub const PT_GNU_STACK: u32 = 0x6474_e551;
/// GNU extension: locates `.eh_frame_hdr`, the binary-search table the runtime
/// unwinder (libgcc/glibc) uses to find an FDE for a thrown exception's PC.
pub const PT_GNU_EH_FRAME: u32 = 0x6474_e550;
/// GNU extension: the run of writable bytes the loader re-protects read-only
/// once it has applied the image's relocations.
///
/// glibc reads `p_memsz` into `l_relro_size` and mprotects the page-aligned
/// interior of the range, so the run has to end on a page boundary to be
/// protected at all.
pub const PT_GNU_RELRO: u32 = 0x6474_e552;

// --- p_flags: segment permissions ----------------------------------------

/// The segment is executable.
pub const PF_X: u32 = 1;
/// The segment is writable.
pub const PF_W: u32 = 2;
/// The segment is readable.
pub const PF_R: u32 = 4;

// --- sh_type: section types ----------------------------------------------

/// Inactive / unused section header.
pub const SHT_NULL: u32 = 0;
/// Program-defined contents.
pub const SHT_PROGBITS: u32 = 1;
/// Symbol table.
pub const SHT_SYMTAB: u32 = 2;
/// String table.
pub const SHT_STRTAB: u32 = 3;
/// Relocation entries with explicit addends.
pub const SHT_RELA: u32 = 4;
/// `SysV` symbol hash table; speeds up `.dynsym` lookups by the loader.
pub const SHT_HASH: u32 = 5;
/// GNU symbol hash table (`.gnu.hash`): a bloom filter plus bucket/chain
/// layout the loader prefers for O(1)-ish name lookup over `.hash`.
pub const SHT_GNU_HASH: u32 = 0x6fff_fff6;
/// Dynamic linking information.
pub const SHT_DYNAMIC: u32 = 6;
/// Note section.
pub const SHT_NOTE: u32 = 7;
/// Uninitialised data (occupies no file space).
pub const SHT_NOBITS: u32 = 8;
/// Relocation entries without addends.
pub const SHT_REL: u32 = 9;
/// Dynamic symbol table.
pub const SHT_DYNSYM: u32 = 11;
/// Array of constructor function pointers (`.init_array`); the runtime calls
/// each entry at startup, in array order.
pub const SHT_INIT_ARRAY: u32 = 14;
/// Array of destructor function pointers (`.fini_array`); the runtime calls
/// each entry at exit, in reverse array order.
pub const SHT_FINI_ARRAY: u32 = 15;
/// Array of pre-initialisation function pointers (`.preinit_array`); the
/// runtime calls each entry before any `.init_array` constructor.
pub const SHT_PREINIT_ARRAY: u32 = 16;
/// Section group.
pub const SHT_GROUP: u32 = 17;

// --- group flags (first word of an SHT_GROUP section's payload) -----------

/// COMDAT group: only one group with this signature is kept; later
/// duplicates are discarded.
///
/// The signature is the name of the group's signature symbol, or for an
/// `STT_SECTION` signature, the section's name.
pub const GRP_COMDAT: u32 = 1;

// --- sh_flags ------------------------------------------------------------

/// Section is writable.
pub const SHF_WRITE: u64 = 0x1;
/// Section occupies memory at runtime.
pub const SHF_ALLOC: u64 = 0x2;
/// Section is executable.
pub const SHF_EXECINSTR: u64 = 0x4;
/// Section holds mergeable data (deduplicated by the linker).
pub const SHF_MERGE: u64 = 0x10;
/// Section holds NUL-terminated strings (implies `SHF_MERGE` semantics).
pub const SHF_STRINGS: u64 = 0x20;
/// `sh_info` holds a section header index rather than a plain number, so a
/// tool that renumbers sections knows to rewrite it. Carried by `.rela.plt`,
/// whose `sh_info` names the section its entries patch.
pub const SHF_INFO_LINK: u64 = 0x40;
/// `sh_link` names a section this one is ordered against: the two are kept or
/// dropped as a unit, and metadata sections such as
/// `__patchable_function_entries` use it to point at the code they describe.
pub const SHF_LINK_ORDER: u64 = 0x80;
/// Section holds thread-local storage (`.tdata`/`.tbss`); its vaddr is the
/// load-time template address, and TLS relocations resolve to thread-pointer
/// offsets rather than to the vaddr.
pub const SHF_TLS: u64 = 0x400;
/// GNU extension (`SHF_GNU_RETAIN`): the section is pinned in the output and
/// never garbage-collected, however unreachable it appears.
///
/// Inhibits `--gc-sections` removal of a single, otherwise-unused section.
pub const SHF_GNU_RETAIN: u64 = 0x200_000;
/// The producer asks that the section not survive the final link.
///
/// Set on `.text.unlikely` under a profile and on the `.discard` families.
/// A final link drops the section; only a relocatable one keeps it for the
/// next link to drop.
pub const SHF_EXCLUDE: u64 = 0x8000_0000;

/// The section's contents are compressed, behind an `Elf64_Chdr` header that
/// gives the uncompressed size and algorithm.
pub const SHF_COMPRESSED: u64 = 0x800;

// --- special section indices ---------------------------------------------

/// Undefined / absent section index.
pub const SHN_UNDEF: u16 = 0;
/// Lower bound of reserved indices.
pub const SHN_LORESERVE: u16 = 0xff00;
/// Common symbol (tentative definition).
pub const SHN_COMMON: u16 = 0xfff2;
/// Absolute value, not associated with a section.
pub const SHN_ABS: u16 = 0xfff1;
/// The real section index lives in the `SHT_SYMTAB_SHNDX` table, because it
/// does not fit the 16-bit `st_shndx` field.
pub const SHN_XINDEX: u16 = 0xffff;

// --- symbol binding (high nibble of st_info) -----------------------------

pub const STB_LOCAL: u8 = 0;
pub const STB_GLOBAL: u8 = 1;
pub const STB_WEAK: u8 = 2;
/// `STB_GNU_UNIQUE`: a definition the loader must keep one copy of
/// process-wide. Ranks like a weak definition at link time; see
/// [`crate::symbol`]'s merge rules.
pub const STB_GNU_UNIQUE: u8 = 10;

// --- symbol visibility (low two bits of st_other) ------------------------

/// Mask selecting the visibility field of `st_other`. The remaining bits are
/// processor-specific and are not interpreted here.
pub const STV_MASK: u8 = 3;
/// Default visibility: the binding alone decides who sees the symbol, and a
/// definition in a shared object may be preempted by one the executable
/// supplies.
pub const STV_DEFAULT: u8 = 0;
/// Processor-specific visibility, stricter than `STV_HIDDEN` on the targets
/// that define it. Treated as hidden everywhere else, which is what the psABIs
/// that leave it unspecified expect.
pub const STV_INTERNAL: u8 = 1;
/// Not visible outside the image being built: the name is not part of its ABI,
/// so it is kept out of `.dynsym` and can never be preempted.
pub const STV_HIDDEN: u8 = 2;
/// Visible outside the image but not preemptible: other images may reference
/// the definition, and every reference from within this one resolves to it.
pub const STV_PROTECTED: u8 = 3;

// --- symbol type (low nibble of st_info) ---------------------------------

pub const STT_NOTYPE: u8 = 0;
pub const STT_OBJECT: u8 = 1;
pub const STT_FUNC: u8 = 2;
pub const STT_SECTION: u8 = 3;
pub const STT_FILE: u8 = 4;
pub const STT_COMMON: u8 = 5;
pub const STT_TLS: u8 = 6;
/// An indirect function: the symbol's value is a resolver the runtime calls
/// once, whose return value is the address every reference should use.
pub const STT_GNU_IFUNC: u8 = 10;

// --- d_tag: dynamic table entries ----------------------------------------

/// End of the `_DYNAMIC` array.
pub const DT_NULL: i64 = 0;
/// String table offset naming a needed shared object (`DT_NEEDED`).
pub const DT_NEEDED: i64 = 1;
/// Address of the `.hash` `SysV` symbol hash table.
pub const DT_HASH: i64 = 4;
/// Address of the `.gnu.hash` GNU symbol hash table; the loader prefers it
/// over `DT_HASH` when both are present.
pub const DT_GNU_HASH: i64 = 0x6fff_fef5;
/// Address of the `.dynstr` string table.
pub const DT_STRTAB: i64 = 5;
/// Address of the `.dynsym` symbol table.
pub const DT_SYMTAB: i64 = 6;
/// Address of the `.rela.dyn` relocation table.
pub const DT_RELA: i64 = 7;
/// Byte size of the `.rela.dyn` table.
pub const DT_RELASZ: i64 = 8;
/// Size of one `.rela.dyn` entry (`sizeof Rela64`).
pub const DT_RELAENT: i64 = 9;
/// Byte size of the `.dynstr` string table.
pub const DT_STRSZ: i64 = 10;
/// Size of one `.dynsym` entry (`sizeof Sym64`).
pub const DT_SYMENT: i64 = 11;
/// Address of the initialisation function the runtime calls before it walks
/// `.init_array`.
pub const DT_INIT: i64 = 12;
/// Address of the termination function the runtime calls after it walks
/// `.fini_array`.
pub const DT_FINI: i64 = 13;
/// Address of the `.init_array` section; the runtime calls each function
/// pointer it holds at startup.
pub const DT_INIT_ARRAY: i64 = 25;
/// Address of the `.fini_array` section; the runtime calls each function
/// pointer it holds at exit.
pub const DT_FINI_ARRAY: i64 = 26;
/// Byte size of the `.init_array` section.
pub const DT_INIT_ARRAYSZ: i64 = 27;
/// Byte size of the `.fini_array` section.
pub const DT_FINI_ARRAYSZ: i64 = 28;
/// Address of the `.preinit_array` section; the runtime calls each function
/// pointer it holds before any `.init_array` entry.
pub const DT_PREINIT_ARRAY: i64 = 32;
/// Byte size of the `.preinit_array` section.
pub const DT_PREINIT_ARRAYSZ: i64 = 33;
/// Address of `.rela.plt` (the lazy-binding relocation table).
pub const DT_JMPREL: i64 = 23;
/// Count of `R_*_RELATIVE` entries at the start of `.rela.dyn` (loader hint).
///
/// One of the `DT_VALRNGHI` extensions, which is why the number is far above
/// the low tags rather than beside `DT_RELA`. Tag 24 is `DT_BIND_NOW`, and
/// using it here would tell the loader to resolve every relocation eagerly.
pub const DT_RELACOUNT: i64 = 0x6fff_fff9;
/// Address of `.got.plt` (`PLT_GOT`), the GOT used by the PLT.
pub const DT_PLTGOT: i64 = 3;
/// Byte size of the `.rela.plt` table.
pub const DT_PLTRELSZ: i64 = 2;
/// Type of the `.rela.plt` relocations: `DT_RELA` (7) or `DT_REL` (17).
pub const DT_PLTREL: i64 = 20;
/// String table offset naming the shared object (`DT_SONAME`).
pub const DT_SONAME: i64 = 14;
/// Debugger hook: the loader fills this with the address of its `r_debug`
/// structure. Emitted as zero by the linker; an executable carries one so the
/// loader has a slot to patch.
pub const DT_DEBUG: i64 = 21;
/// Flag word the loader reads before relocating the image.
pub const DT_FLAGS: i64 = 30;
/// GNU extension: a second flag word, for flags the gABI has no room for.
pub const DT_FLAGS_1: i64 = 0x6fff_fffb;

// --- DT_FLAGS bits -------------------------------------------------------

/// The image uses initial-exec thread-local storage.
///
/// Such an object's offsets are fixed at load time, so glibc reserves a slot
/// in the static TLS area when it opens one and refuses the `dlopen` cleanly
/// if it cannot. Without the flag the failure comes later, when a relocation
/// cannot be applied.
pub const DF_STATIC_TLS: u64 = 0x10;
/// The runtime search path, searched after `LD_LIBRARY_PATH` and not
/// inherited by the image's own dependencies. The tag `-rpath` records.
pub const DT_RUNPATH: i64 = 29;
/// Every relocation is resolved before control reaches the image (`-z now`).
///
/// The loader binds the whole PLT at load time instead of on first call. The
/// image still carries lazy stubs; the flag is what makes them unused.
pub const DF_BIND_NOW: u64 = 0x08;
/// The image may use `$ORIGIN` in its runtime paths (`-z origin`).
pub const DF_ORIGIN: u64 = 0x01;

// --- DT_FLAGS_1 bits -----------------------------------------------------

/// The image is a position-independent executable.
///
/// glibc refuses to `dlopen` one, which is what keeps a program from being
/// loaded as a library and running its startup a second time.
pub const DF_1_PIE: u64 = 0x0800_0000;
/// `-z now`, in the word loaders read in preference to `DT_FLAGS`.
pub const DF_1_NOW: u64 = 0x0000_0001;
/// `-z global`: the object's symbols join the global search scope.
pub const DF_1_GLOBAL: u64 = 0x0000_0002;
/// `-z nodelete`: the object stays mapped once loaded.
pub const DF_1_NODELETE: u64 = 0x0000_0008;
/// `-z initfirst`: the object's initialisers run before any other's.
pub const DF_1_INITFIRST: u64 = 0x0000_0020;
/// `-z nodlopen`: the object may not be opened with `dlopen`.
pub const DF_1_NOOPEN: u64 = 0x0000_0040;
/// `-z origin`, in the word loaders read in preference to `DT_FLAGS`.
pub const DF_1_ORIGIN: u64 = 0x0000_0080;
/// `-z interpose`: the object's definitions take precedence over every
/// other's.
pub const DF_1_INTERPOSE: u64 = 0x0000_0400;
/// The `DT_PLTREL` value selecting `DT_RELA` relocations for `.rela.plt`.
pub const DT_RELA_VAL: u64 = 7;
/// Address of the `.gnu.version` section: one `u16` per `.dynsym` entry giving
/// its version index (0 = local, 1 = global, >= 2 indexes `.gnu.version_r` or
/// `.gnu.version_d`).
pub const DT_VERSYM: i64 = 0x6fff_fff0;
/// Address of the `.gnu.version_r` (VERNEED) section: one `Elf_Verneed` per
/// versioned dependency soname, each listing the version names required from
/// it. Paired with `DT_VERNEEDNUM` for the count.
pub const DT_VERNEED: i64 = 0x6fff_fffe;
/// Number of `Elf_Verneed` entries at the address named by `DT_VERNEED`.
pub const DT_VERNEEDNUM: i64 = 0x6fff_ffff;
/// Address of the `.gnu.version_d` (VERDEF) section: one `Elf_Verdef` per
/// version this object exports. Paired with `DT_VERDEFNUM`.
pub const DT_VERDEF: i64 = 0x6fff_fffd;
/// Number of `Elf_Verdef` entries at the address named by `DT_VERDEF`.
pub const DT_VERDEFNUM: i64 = 0x6fff_fffc;

/// `SHT_GNU_versym`: the `.gnu.version` section, an array of `u16` version
/// indices parallel to `.dynsym`.
pub const SHT_GNU_VERSYM: u32 = 0x6fff_ffff;
/// `SHT_GNU_verneed`: the `.gnu.version_r` section, the VERNEED table.
pub const SHT_GNU_VERNEED: u32 = 0x6fff_fffe;
/// `SHT_GNU_verdef`: the `.gnu.version_d` section, the VERDEF table.
pub const SHT_GNU_VERDEF: u32 = 0x6fff_fffd;

// --- version index (`Elf_Versym`) values ---------------------------------

/// Symbol is local and not visible for versioned lookup.
pub const VER_NDX_LOCAL: u16 = 0;
/// Symbol is global; the default for an unversioned export.
pub const VER_NDX_GLOBAL: u16 = 1;
/// Lower bound of reserved indices (hidden / version-specific).
pub const VER_NDX_LORESERVE: u16 = 0xff00;
/// High bit of a version index: when set the symbol is hidden from the
/// default (unversioned) lookup, so only an explicit `name@version` reference
/// resolves it.
pub const VER_NDX_HIDDEN: u16 = 0x8000;

// --- VERNEED / VERDEF flags ----------------------------------------------

/// `VER_FLG_BASE`: the version is the file's base version (the soname itself).
/// Used in VERDEF entries.
pub const VER_FLG_BASE: u16 = 1;
/// `VER_FLG_WEAK`: weak version requirement. Valid in Vernaux flags.
pub const VER_FLG_WEAK: u16 = 2;
