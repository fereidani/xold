//! PE section placement: virtual addresses, file offsets and image extents.
//!
//! Output sections follow the PE convention: `.text` (code) first, then
//! read-only data (`.rdata`), then read-write data (`.data`), then `.bss`
//! (uninitialised, no file bytes); the resource tree (`.rsrc`) and the
//! writer-synthesised directories follow: `.idata`, `.edata`, `.reloc`. Each
//! section's virtual address is aligned to
//! [`SECTION_ALIGNMENT`] (`0x1000`) and starts at `0x1000`; each section's raw
//! data is aligned to [`FILE_ALIGNMENT`] (`0x200`) and starts after the
//! headers. Executables load at a fixed [`IMAGE_BASE_X86_64`] with ASLR
//! disabled (no base-relocation table); DLLs prefer [`IMAGE_BASE_DLL_X86_64`]
//! and carry a `.reloc` table so the loader can relocate them.
//!
//! The entry stub is a small piece of linker-generated code placed at the very
//! start of `.text`, so an executable's `AddressOfEntryPoint` is the `.text`
//! RVA. A DLL has no entry stub and its entry point is zero. The user's `.text`
//! contributions follow the stub; their internal relocation offsets are
//! preserved because the stub shifts the symbol address and the relocation
//! place by the same amount.

use crate::{
    coff::{
        CoffFile, CoffSection,
        comdat::ComdatPlan,
        constants::{
            FILE_ALIGNMENT, IMAGE_SCN_ALIGN_MASK, IMAGE_SCN_CNT_CODE,
            IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_CNT_UNINITIALIZED_DATA,
            IMAGE_SCN_MEM_DISCARDABLE, IMAGE_SCN_MEM_EXECUTE,
            IMAGE_SCN_MEM_READ, IMAGE_SCN_MEM_WRITE, SECTION_ALIGNMENT,
        },
    },
    error::{Error, Result},
    util::{align_up_u32, pad_name, trim_nul},
};

/// On-disk size of the `.tlsdir` trailer: the `ImageTlsDirectory64` (40 bytes),
/// the null callbacks slot (8 bytes) and the `_tls_index` dword (4 bytes).
pub const TLS_DIR_SIZE: u32 = 52;

/// On-disk size of the generated entry stub in bytes (see [`ENTRY_STUB`]).
pub const ENTRY_STUB_SIZE: u32 = 19;

/// What the common block is aligned to within `.bss`: the largest alignment
/// any single common can ask for, so every slot inside it lands where its own
/// alignment says.
const COMMON_BLOCK_ALIGN: u32 = 32;

/// What the entry stub is rounded up to, so the first real `.text` member
/// starts somewhere its own alignment can be honoured from.
const STUB_ALIGN: u64 = 16;

/// A member input section selected for linking.
#[derive(Clone, Copy)]
pub struct Member {
    pub file: usize,
    /// 1-based section ordinal in the input file.
    pub section: u32,
    pub size: u64,
    /// Byte offset of this member within its output section.
    pub offset: u64,
    /// What the input declared through `IMAGE_SCN_ALIGN_*`.
    ///
    /// Members were packed end to end and this was read nowhere, so a
    /// 16-byte-aligned `.rdata` COMDAT -- which is where clang puts its
    /// `__xmm@` constants -- landed wherever the member before it ended.
    /// Behind an odd-length string that is a misaligned constant and a
    /// `movaps` fault. lld aligns every chunk (`COFF/Writer.cpp:1792`).
    pub align: u64,
}

/// An output PE section: the merged members, its RVA, file offset and extents.
pub struct OutSection {
    pub name: [u8; 8],
    pub members: Vec<Member>,
    pub characteristics: u32,
    pub virtual_address: u32,
    pub virtual_size: u32,
    pub pointer_to_raw_data: u32,
    pub size_of_raw_data: u32,
    /// `.bss`: occupies virtual space but no file bytes.
    pub zerofill: bool,
}

impl OutSection {
    /// Whether this section has file-backed data.
    pub const fn has_file_data(&self) -> bool {
        !self.zerofill
    }
}

/// The laid-out PE image: output sections, header size, image size, entry RVA,
/// the image base, and the entry symbol name to resolve.
pub struct PeLayout {
    pub sections: Vec<OutSection>,
    pub size_of_headers: u32,
    pub size_of_image: u32,
    pub address_of_entry_point: u32,
    pub base_of_code: u32,
    pub image_base: u64,
    pub entry_stub_size: u32,
    /// Byte offset of the common-symbol block within `.bss`.
    ///
    /// Commons are tentative definitions the linker allocates itself; they
    /// sit past every `.bss` member so member offsets are unaffected.
    pub commons_off: u32,
    /// Whether the image is a DLL (versus an executable). Selects
    /// `IMAGE_FILE_DLL`, the DLL image base, ASLR and a zero entry point.
    pub is_dll: bool,
}

/// Builds the section layout for `inputs`.
///
/// Places linkable sections into output sections and reserves
/// `entry_stub_size` bytes at the start of `.text`. The three synthetic
/// sections are reserved when their size is non-zero: `import_size` reserves
/// `.idata`, `export_size` reserves `.edata` and `reloc_size` reserves
/// `.reloc`. `tls_meta_size` reserves `.tlsdir` (the TLS directory, the null
/// callbacks slot and the `_tls_index` dword) when the inputs declare any
/// `__declspec(thread)` storage. `commons_size` reserves storage for the
/// tentative definitions past the end of `.bss`. `comdats` names the COMDAT
/// copies that lost their group, which are not placed. `image_base` is the
/// preferred load address
/// (the conventional EXE or DLL base). When `is_dll` is set the image has no
/// entry stub and its entry point is zero; otherwise the entry point is the
/// `.text` RVA (the entry stub).
#[allow(clippy::too_many_arguments)]
pub fn build(
    inputs: &[CoffFile<'_>],
    entry_stub_size: u32,
    import_size: u32,
    export_size: u32,
    reloc_size: u32,
    tls_meta_size: u32,
    commons_size: u32,
    comdats: &ComdatPlan,
    image_base: u64,
    is_dll: bool,
) -> Result<PeLayout> {
    let mut groups: Vec<OutSection> = Vec::new();
    gather_input_sections(inputs, comdats, &mut groups)?;
    sort_members_by_name(inputs, &mut groups);
    add_synthetic_sections(
        &mut groups,
        import_size,
        export_size,
        reloc_size,
        tls_meta_size,
    );
    // Order: code, rdata, data, bss, tls, tlsdir, idata, edata, rsrc, reloc.
    groups.sort_by_key(section_rank);
    layout_members(&mut groups, entry_stub_size);
    let commons_off = reserve_commons(&mut groups, commons_size);
    place(&mut groups)?;

    let text_rva = groups
        .iter()
        .find(|s| s.name == *b".text\0\0\0")
        .map(|s| s.virtual_address)
        .ok_or(Error::Format("no .text section in inputs"))?;
    let size_of_headers = align_up_u32(
        headers_extent(u32::try_from(groups.len()).unwrap_or(0)),
        FILE_ALIGNMENT,
    );
    let last = groups
        .last()
        .ok_or(Error::Format("no sections to lay out"))?;
    let image_end = align_up_u32(
        last.virtual_address.wrapping_add(last.virtual_size),
        SECTION_ALIGNMENT,
    );
    Ok(PeLayout {
        sections: groups,
        size_of_headers,
        size_of_image: image_end,
        // A DLL enters through the loader's DllMain dispatch, not a fixed
        // entry stub; zero disables the entry point.
        address_of_entry_point: if is_dll { 0 } else { text_rva },
        base_of_code: text_rva,
        image_base,
        entry_stub_size,
        commons_off,
        is_dll,
    })
}

/// Places every linkable input section as a member of its output section.
///
/// A COMDAT copy that lost its group is not placed, nor is a section that
/// does not classify or is empty.
fn gather_input_sections(
    inputs: &[CoffFile<'_>],
    comdats: &ComdatPlan,
    groups: &mut Vec<OutSection>,
) -> Result<()> {
    // The one input whose `.rsrc` tree the link places. A second tree from
    // another object cannot be concatenated behind it: each tree's internal
    // offsets are relative to its own base, so merging means walking and
    // rebuilding the tree (what lld's `.rsrc` merger does,
    // `lld/COFF/Driver.cpp` and `lld/COFF/Writer.cpp`), not
    // appending bytes. Refusing keeps the corruption loud.
    let mut rsrc_file: Option<usize> = None;
    for (file, input) in inputs.iter().enumerate() {
        for section in input.sections() {
            // A COMDAT that lost its group is not placed: its references go to
            // the copy that won.
            if comdats.is_dropped(file, section.index) {
                continue;
            }
            let Some(kind) = classify(section) else {
                continue;
            };
            // A `.tls$` section that declares thread storage but carries no
            // raw data is BSS-style TLS. Its bytes would have to be counted
            // into the TLS directory's `SizeOfZeroFill` rather than into the
            // template, which this linker does not build; dropping it as an
            // empty member instead would leave the variable with no
            // per-thread storage at all while its SECREL fixups still
            // resolved into the template. Refuse it rather than miscompile
            // it. clang does not emit this shape -- it zero-fills into the
            // initialised `.tls$` -- so this is reached only from MSVC or
            // hand-written objects.
            if matches!(kind, Kind::Tls)
                && member_size(kind, section) == 0
                && section.virtual_size > 0
            {
                return Err(Error::Format(
                    "a .tls section with no raw data (BSS-style thread                      storage) is not supported",
                ));
            }
            if member_size(kind, section) == 0 {
                continue;
            }
            if matches!(kind, Kind::Rsrc)
                && rsrc_file.is_some_and(|first| first != file)
            {
                return Err(Error::Format(
                    "multiple objects carry .rsrc resource trees; merge them \
                     into one before linking",
                ));
            }
            if matches!(kind, Kind::Rsrc) {
                rsrc_file = Some(file);
            }
            add_member(groups, file, section, kind);
        }
    }
    Ok(())
}

/// Reserves each synthetic (linker-generated) section whose size is non-zero.
fn add_synthetic_sections(
    groups: &mut Vec<OutSection>,
    import_size: u32,
    export_size: u32,
    reloc_size: u32,
    tls_meta_size: u32,
) {
    const INIT_READ: u32 = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ;
    const READ_WRITE: u32 = INIT_READ | IMAGE_SCN_MEM_WRITE;
    const DISCARDABLE: u32 = INIT_READ | IMAGE_SCN_MEM_DISCARDABLE;
    let reserved: [(&[u8], u32, u32); 4] = [
        (b".idata", import_size, INIT_READ),
        (b".tlsdir", tls_meta_size, READ_WRITE),
        (b".edata", export_size, INIT_READ),
        (b".reloc", reloc_size, DISCARDABLE),
    ];
    for (name, size, characteristics) in reserved {
        if size > 0 {
            groups.push(synthetic(name, size, characteristics));
        }
    }
}

/// Appends `size` bytes of common storage to `.bss`, creating the section when
/// no input contributed one, and returns the block's offset within it.
///
/// The block goes past every member, so no member offset moves and the only
/// section whose extent changes is the one the loader zeroes anyway.
fn reserve_commons(groups: &mut Vec<OutSection>, size: u32) -> u32 {
    if size == 0 {
        return 0;
    }
    if groups.iter().all(|s| s.name != *b".bss\0\0\0\0") {
        let mut bss = synthetic(b".bss", 0, output_chars(Kind::Bss));
        bss.zerofill = true;
        groups.push(bss);
        groups.sort_by_key(section_rank);
    }
    let Some(bss) = groups.iter_mut().find(|s| s.name == *b".bss\0\0\0\0")
    else {
        return 0;
    };
    let off = align_up_u32(bss.virtual_size, COMMON_BLOCK_ALIGN);
    bss.virtual_size = off.saturating_add(size);
    off
}

/// Builds a synthetic (linker-generated) output section of `size` bytes.
fn synthetic(name: &[u8], size: u32, characteristics: u32) -> OutSection {
    OutSection {
        name: pad_name(name),
        members: Vec::new(),
        characteristics,
        virtual_address: 0,
        virtual_size: size,
        pointer_to_raw_data: 0,
        size_of_raw_data: 0,
        zerofill: false,
    }
}

/// Output-section kind derived from an input section's characteristics.
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
enum Kind {
    Code,
    RData,
    Data,
    Bss,
    /// `__declspec(thread)` storage: a `.tls$*` input section whose raw bytes
    /// seed the per-thread TLS template.
    Tls,
    /// A compiled resource tree: a `.rsrc$*` input section. cvtres emits the
    /// tree as `.rsrc$01` and its strings and data as `.rsrc$02`.
    Rsrc,
}

/// Maps an input section to an output kind, dropping sections the minimal
/// link defers (debug `.debug_*`, unwind `.pdata`/`.xdata`, COMDAT oddities
/// return their primary kind). `.tls$*` sections are classified as
/// [`Kind::Tls`] so they aggregate into the output `.tls` template. `None`
/// drops the section.
fn classify(section: &CoffSection<'_>) -> Option<Kind> {
    let ch = section.characteristics;
    if ch & IMAGE_SCN_CNT_CODE != 0 && ch & IMAGE_SCN_MEM_EXECUTE != 0 {
        return Some(Kind::Code);
    }
    if is_tls_section(section.name) {
        return Some(Kind::Tls);
    }
    if is_rsrc_section(section.name) {
        return Some(Kind::Rsrc);
    }
    if is_deferred(section.name) {
        return None;
    }
    if ch & IMAGE_SCN_CNT_UNINITIALIZED_DATA != 0 {
        return Some(Kind::Bss);
    }
    let writable = ch & IMAGE_SCN_MEM_WRITE != 0;
    if ch & IMAGE_SCN_CNT_INITIALIZED_DATA != 0 {
        return Some(if writable { Kind::Data } else { Kind::RData });
    }
    None
}

/// Whether `name` identifies a TLS template input section. clang-msvc emits
/// each `__declspec(thread)` variable's initial bytes into a `.tls$` section
/// (grouped later as `.tls$aaa`, `.tls$zzz` by the CRT); any name beginning
/// with `.tls$` contributes to the output `.tls` template.
fn is_tls_section(name: &[u8]) -> bool {
    trim_nul(name).starts_with(b".tls$")
}

/// Whether `name` identifies a compiled resource section. cvtres emits the
/// resource directory as `.rsrc$01` and its strings and data as `.rsrc$02`;
/// any name beginning with `.rsrc` contributes to the output `.rsrc`.
fn is_rsrc_section(name: &[u8]) -> bool {
    trim_nul(name).starts_with(b".rsrc")
}

/// Whether the named section is dropped by the minimal link. Debug sections
/// (`.debug_*`), the exception tables (`.pdata`, `.xdata`) need machinery
/// beyond this phase (SEH unwind registration, debug directory) and are
/// deferred.
///
/// The `$` forms are deferred with the bare names. A compiler emitting one
/// section per function writes `.pdata$foo` and `.xdata$foo`, and matching
/// only the exact names let those through to be placed as ordinary read-only
/// data: unwind records nothing points at, since no `.pdata` output section
/// or exception data directory is built. Whether the input was compiled with
/// one section per function is not something the treatment of its unwind
/// tables should turn on.
fn is_deferred(name: &[u8]) -> bool {
    let trim = trim_nul(name);
    trim.starts_with(b".debug")
        || is_unwind_section(trim)
        || matches!(trim, b".llvm_addrsig" | b".drectve")
}

/// Whether `name` is an exception-table section, in either the bare or the
/// per-function `$` form.
fn is_unwind_section(name: &[u8]) -> bool {
    for base in [b".pdata".as_slice(), b".xdata".as_slice()] {
        if name == base
            || name
                .strip_prefix(base)
                .is_some_and(|rest| rest.starts_with(b"$"))
        {
            return true;
        }
    }
    false
}

/// The sort key for an output section: code, rdata, data, bss, tls, tlsdir,
/// idata, edata, rsrc, then reloc. `.idata` and any unexpected section name
/// share rank 6, so an unknown section sorts after the code and data sections
/// rather than ahead of them. `.reloc` stays last: the writer reads every other
/// section's final RVA off the provisional layout before reserving it.
fn section_rank(s: &OutSection) -> u8 {
    match &s.name {
        b".text\0\0\0" => 0,
        b".rdata\0\0" => 1,
        b".data\0\0\0" => 2,
        b".bss\0\0\0\0" => 3,
        b".tls\0\0\0\0" => 4,
        b".tlsdir\0" => 5,
        b".edata\0\0" => 7,
        b".rsrc\0\0\0" => 8,
        b".reloc\0\0" => 9,
        _ => 6,
    }
}

/// The byte size a member contributes: file data for initialised sections,
/// the virtual (uninitialised) size for `.bss`.
///
/// A `.tls$` section with no raw data answers zero here.
/// [`gather_input_sections`] refuses that shape outright rather than let it
/// pass as an empty member, because BSS-style thread storage needs the TLS
/// directory's `SizeOfZeroFill` and not the template.
fn member_size(kind: Kind, section: &CoffSection<'_>) -> u64 {
    match kind {
        // `.bss` has no file bytes; its contribution is purely virtual.
        Kind::Bss => u64::from(section.virtual_size),
        // Code, read-only/read-write data and the TLS template copy their raw
        // bytes; `data.len()` guards against a stale `size_of_raw_data`.
        _ => u64::from(section.size_of_raw_data).max(section.data.len() as u64),
    }
}

/// The alignment an input section declares, as a byte count.
///
/// The field is a four-bit exponent biased by one: 1 means 1 byte, 2 means 2,
/// up to 14 for 8192. Zero means the input said nothing, and the PE
/// specification's default for that is 16 -- which is also the strictest
/// alignment an x86-64 constant pool asks for, so it is the safe reading of
/// silence.
fn member_align(section: &CoffSection<'_>) -> u64 {
    const DEFAULT: u64 = 16;
    let field = (section.characteristics & IMAGE_SCN_ALIGN_MASK) >> 20;
    if field == 0 {
        return DEFAULT;
    }
    1u64 << (field - 1).min(13)
}

/// Orders every output section's members by their full input names.
///
/// `$` and what follows it in an input name decides nothing about which
/// output section the member belongs to, and everything about where inside
/// it the member lands: the CRT's `.CRT$XCA` .. `.CRT$XZ` run is one table
/// of pointers split across objects, and the suffixes spell the order that
/// table needs. Ordering by name once every member is collected gives each
/// such family its order whatever order the objects arrived in.
///
/// Members of one name keep their input order, which the C++ initialiser
/// tables rely on within a single object: two initialisers in one
/// translation unit must run in declaration order, and they share a section
/// name. lld reaches the same arrangement by collecting inputs into a map
/// keyed by full name and walking it in key order
/// (`lld/COFF/Writer.cpp`), then making the within-name
/// order explicit for `.CRT`
/// (`sortCRTSectionChunks`, `lld/COFF/Writer.cpp`).
fn sort_members_by_name(inputs: &[CoffFile<'_>], groups: &mut [OutSection]) {
    for s in groups.iter_mut() {
        s.members.sort_by_key(|m| member_name(inputs, m));
    }
}

/// The NUL-trimmed name of the input section `m` places, empty when the
/// member's file or section has fallen out of view.
fn member_name<'a>(inputs: &[CoffFile<'a>], m: &Member) -> &'a [u8] {
    inputs
        .get(m.file)
        .and_then(|f| f.section_at(m.section))
        .map_or(&[][..], |s| trim_nul(s.name))
}

/// Adds `section` to the matching output section, creating it on first sight.
fn add_member(
    groups: &mut Vec<OutSection>,
    file: usize,
    section: &CoffSection<'_>,
    kind: Kind,
) {
    let name: &[u8] = match kind {
        Kind::Code => b".text",
        Kind::RData => b".rdata",
        Kind::Data => b".data",
        Kind::Bss => b".bss",
        Kind::Tls => b".tls",
        Kind::Rsrc => b".rsrc",
    };
    let out_name: [u8; 8] = pad_name(name);
    let member = Member {
        file,
        section: section.index,
        size: member_size(kind, section),
        offset: 0,
        align: member_align(section),
    };
    if let Some(out) = groups.iter_mut().find(|s| s.name == out_name) {
        out.members.push(member);
        out.characteristics |= output_chars(kind);
        return;
    }
    groups.push(OutSection {
        name: out_name,
        members: vec![member],
        characteristics: output_chars(kind),
        virtual_address: 0,
        virtual_size: 0,
        pointer_to_raw_data: 0,
        size_of_raw_data: 0,
        // `.bss` is uninitialised by definition: the loader zeroes it. Writing
        // it as file bytes made a 100 MB array 100 MB of zeros on disk, and
        // counted them as initialised data besides. `place` already honours
        // this flag; nothing ever set it.
        zerofill: matches!(kind, Kind::Bss),
    });
}

/// The output section characteristics for `kind`.
const fn output_chars(kind: Kind) -> u32 {
    match kind {
        Kind::Code => {
            IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ
        }
        // The resource tree is read-only data like `.rdata`: `FindResource`
        // walks it, nothing writes it.
        Kind::RData | Kind::Rsrc => {
            IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ
        }
        // The TLS template is read/write initialised data, like `.data`: the
        // loader copies it per thread and user code mutates its copy.
        Kind::Data | Kind::Tls => {
            IMAGE_SCN_CNT_INITIALIZED_DATA
                | IMAGE_SCN_MEM_READ
                | IMAGE_SCN_MEM_WRITE
        }
        Kind::Bss => {
            IMAGE_SCN_CNT_UNINITIALIZED_DATA
                | IMAGE_SCN_MEM_READ
                | IMAGE_SCN_MEM_WRITE
        }
    }
}

/// Assigns each member its offset within its output section. `.text` reserves
/// `entry_stub_size` bytes at offset zero for the linker-generated entry stub.
/// Synthetic sections without members (the import `.idata`) keep their preset
/// virtual size.
fn layout_members(groups: &mut [OutSection], entry_stub_size: u32) {
    for s in groups.iter_mut() {
        let base = u64::from(s.virtual_size);
        let mut cursor = if s.name == *b".text\0\0\0" {
            // The stub is 19 bytes, so every `.text` member after it started
            // at an odd offset. Rounding it here is what makes the members'
            // own alignments mean anything.
            base.max(u64::from(entry_stub_size))
                .next_multiple_of(STUB_ALIGN)
        } else {
            base
        };
        for m in &mut s.members {
            cursor = cursor.next_multiple_of(m.align.max(1));
            m.offset = cursor;
            cursor = cursor.wrapping_add(m.size);
        }
        s.virtual_size = u32::try_from(cursor).unwrap_or(u32::MAX);
    }
}

/// Assigns RVAs (aligned to [`SECTION_ALIGNMENT`]) and file offsets (aligned to
/// [`FILE_ALIGNMENT`]) to each section. The first section RVA is `0x1000`;
/// `.bss` extends the image but consumes no file bytes.
fn place(groups: &mut [OutSection]) -> Result<()> {
    let mut rva = SECTION_ALIGNMENT;
    let mut file_off = align_up_u32(
        headers_extent(u32::try_from(groups.len()).unwrap_or(0)),
        FILE_ALIGNMENT,
    );
    for s in groups.iter_mut() {
        rva = align_up_u32(rva, SECTION_ALIGNMENT);
        s.virtual_address = rva;
        // `VirtualSize` is the section's real extent, and rounding it to the
        // section alignment before deriving `SizeOfRawData` padded every
        // section's file bytes to a page rather than to `FileAlignment` --
        // up to 3.5 KiB of zeros each, and an extent no tool could read. The
        // rounding belongs to the RVA cursor, which advances below.
        if s.zerofill {
            s.pointer_to_raw_data = 0;
            s.size_of_raw_data = 0;
        } else {
            s.pointer_to_raw_data = file_off;
            let raw = align_up_u32(s.virtual_size, FILE_ALIGNMENT);
            s.size_of_raw_data = raw;
            file_off = file_off
                .checked_add(raw)
                .ok_or(Error::OutOfRange("section file offset"))?;
        }
        rva = rva
            .checked_add(align_up_u32(s.virtual_size, SECTION_ALIGNMENT))
            .ok_or(Error::OutOfRange("section virtual address"))?;
    }
    Ok(())
}

/// The byte size of the DOS + PE headers + section table, before alignment.
fn headers_extent(num_sections: u32) -> u32 {
    use crate::coff::constants::{
        IMAGE_NUMBEROF_DIRECTORY_ENTRIES, IMAGE_SIZEOF_DOS_HEADER,
        IMAGE_SIZEOF_FILE_HEADER_PE, IMAGE_SIZEOF_OPTIONAL_HEADER64,
        IMAGE_SIZEOF_SECTION_HEADER,
    };
    let data_dirs =
        u32::try_from(IMAGE_NUMBEROF_DIRECTORY_ENTRIES).unwrap_or(0) * 8;
    // PE signature (4) + file header (20) + optional header + data dirs.
    let nt = 4
        + u32::try_from(IMAGE_SIZEOF_FILE_HEADER_PE).unwrap_or(0)
        + u32::try_from(IMAGE_SIZEOF_OPTIONAL_HEADER64).unwrap_or(0)
        + data_dirs;
    u32::try_from(IMAGE_SIZEOF_DOS_HEADER).unwrap_or(0)
        + nt
        + num_sections * u32::try_from(IMAGE_SIZEOF_SECTION_HEADER).unwrap_or(0)
}

/// The RVA range `[va, va+size)` of an output section by name, for the writer
/// and the import plan.
pub fn section_rva(layout: &PeLayout, name: &[u8]) -> Option<(u32, u32)> {
    layout
        .sections
        .iter()
        .find(|s| trim_nul(&s.name) == name)
        .map(|s| (s.virtual_address, s.virtual_size))
}
