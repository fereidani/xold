//! Input files and the per-symbol view consumed by resolution.
//!
//! An [`Input`] names one thing to link, either as a path xold opens itself or
//! as bytes the caller already holds. [`Input::open`] turns it into an
//! [`InputBytes`], which is what the parsers read; from there the two are
//! indistinguishable.
//!
//! Direct inputs are read zero-copy over their memory mapping. Archive members
//! cannot be viewed in place: they sit at arbitrary, often misaligned offsets
//! inside the archive, so [`InputFile`] holds an owned copy of a member and
//! re-parses a view on demand. Both kinds expose the same interface, so the
//! core never distinguishes them.

use std::path::{Path, PathBuf};

use crate::{
    elf::{
        Group, ObjectFile, Rela64, Relocs, Shdr64,
        constants::{
            SHN_ABS, SHN_COMMON, SHN_LORESERVE, SHN_UNDEF, SHN_XINDEX,
            STB_LOCAL,
        },
    },
    endian::U64,
    error::{Error, Result},
    mmap_file::{MappedFile, OpenedFile},
};

/// One thing to link, in command-line order.
///
/// Order is meaning rather than presentation: an archive or shared object only
/// satisfies references made by the inputs named before it, so a caller that
/// mixes the two kinds must be free to put either at any position.
///
/// # The name
///
/// Both variants carry a name, and the name is used the same way for both: in
/// diagnostics, and as the `DT_NEEDED` fallback for a shared object that
/// carries no `DT_SONAME` of its own. A [`Self::Memory`] input is therefore
/// byte-for-byte equivalent to writing those bytes to a file of that name and
/// passing it as [`Self::Path`] -- the linker has no other way to tell them
/// apart, which is what keeps the output a pure function of the inputs.
#[derive(Clone, Copy)]
pub enum Input<'data> {
    /// A file xold opens and maps itself, named as it was written.
    Path(&'data Path),
    /// The same, but found by searching for `-lNAME` rather than named
    /// directly.
    ///
    /// The distinction survives this far for one reason: a shared object with
    /// no `DT_SONAME` takes its `DT_NEEDED` from how it was named, and only a
    /// `-l` library takes the basename. An explicitly named
    /// `/opt/vendor/libpriv.so` keeps the path as written, because that is the
    /// only spelling that lets the produced image find it again. lld draws the
    /// line in the same place: `withLOption ? path::filename(path) : path`
    /// (`lld/ELF/Driver.cpp`).
    Library(&'data Path),
    /// A library an `AS_NEEDED` list of a linker script named.
    ///
    /// It takes part in the link exactly like the other two: its exports
    /// resolve references and its versions are read. The difference is only in
    /// `DT_NEEDED`, which records it just when the image actually binds
    /// something to it -- every glibc `libc.so` names the loader this way, and
    /// a program that calls nothing the loader defines has no business
    /// depending on it.
    ///
    /// `from_l` carries the spelling the script used, since an `AS_NEEDED`
    /// list may hold either a path or a `-lNAME`, and only the second takes
    /// its `DT_NEEDED` fallback from the basename.
    AsNeeded { path: &'data Path, from_l: bool },
    /// Bytes the caller already holds. The file is never opened; `name` only
    /// stands in for the path, so it need not exist on disk.
    Memory {
        name: &'data Path,
        bytes: &'data [u8],
    },
}

impl<'data> Input<'data> {
    /// Views a path list as link inputs, preserving order.
    ///
    /// The one allocation here is per link, not per input file, and it lets
    /// the path-taking entry points stay one-line wrappers over the general
    /// one.
    pub fn from_paths(paths: &'data [PathBuf]) -> Vec<Self> {
        paths.iter().map(|p| Self::Path(p)).collect()
    }

    /// The name this input is known by.
    pub const fn name(&self) -> &'data Path {
        match *self {
            Self::Path(path)
            | Self::Library(path)
            | Self::AsNeeded { path, .. } => path,
            Self::Memory { name, .. } => name,
        }
    }

    /// Whether this input was found by a `-l` search rather than named
    /// directly. See [`Self::Library`].
    pub const fn is_library_search(&self) -> bool {
        matches!(
            *self,
            Self::Library(_) | Self::AsNeeded { from_l: true, .. }
        )
    }

    /// Whether an `AS_NEEDED` list named this input. See [`Self::AsNeeded`].
    pub const fn is_as_needed(&self) -> bool {
        matches!(*self, Self::AsNeeded { .. })
    }

    /// Makes the input's bytes readable, mapping a file if there is one.
    ///
    /// Caller-supplied bytes are borrowed in place when they are already
    /// aligned for the on-disk structures, and copied into an aligned buffer
    /// when they are not. A `Vec<u8>` promises alignment 1 and gets more only
    /// by the grace of the allocator, so the copy is what makes this total:
    /// the alternative is a parse that fails on bytes nothing is wrong with.
    pub fn open(&self) -> Result<InputBytes<'data>> {
        self.open_with(self.open_file()?)
    }

    /// Opens the input's file without mapping it, or answers `None` for
    /// caller-supplied bytes.
    ///
    /// The open half of [`Self::open`], split out so a large input list can
    /// fan the opens out across threads and keep the mappings -- which
    /// serialise on the process's address-space lock -- on one. See
    /// [`crate::util::open_all`].
    pub fn open_file(&self) -> Result<Option<OpenedFile>> {
        match *self {
            Self::Path(path)
            | Self::Library(path)
            | Self::AsNeeded { path, .. } => {
                MappedFile::open_file(path).map(Some)
            }
            Self::Memory { .. } => Ok(None),
        }
    }

    /// Finishes [`Self::open`] over a file [`Self::open_file`] produced.
    pub fn open_with(
        &self,
        file: Option<OpenedFile>,
    ) -> Result<InputBytes<'data>> {
        match (*self, file) {
            (
                Self::Path(path)
                | Self::Library(path)
                | Self::AsNeeded { path, .. },
                Some(file),
            ) => Ok(InputBytes::Mapped(MappedFile::map(path, &file)?)),
            (Self::Memory { bytes, .. }, _) if is_parse_aligned(bytes) => {
                Ok(InputBytes::Borrowed(bytes))
            }
            (Self::Memory { bytes, .. }, _) => {
                Ok(InputBytes::Owned(AlignedBytes::new(bytes)))
            }
            // A path input whose file is absent: the two halves were paired
            // up wrongly, which no caller can recover from mid-open.
            (_, None) => Err(Error::Format("input opened without its file")),
        }
    }
}

/// The bytes of one opened [`Input`], however they were obtained.
pub enum InputBytes<'data> {
    /// A memory mapping xold made and owns.
    Mapped(MappedFile),
    /// Caller bytes, borrowed in place.
    Borrowed(&'data [u8]),
    /// Caller bytes that needed realigning, copied.
    Owned(AlignedBytes),
}

impl InputBytes<'_> {
    /// The bytes, borrowed for as long as this value lives.
    pub fn bytes(&self) -> &[u8] {
        match self {
            Self::Mapped(map) => map.bytes(),
            Self::Borrowed(bytes) => bytes,
            Self::Owned(owned) => owned.bytes(),
        }
    }
}

/// An owned copy of input bytes that every on-disk structure can be viewed
/// over.
///
/// The storage is a `Vec<u64>` rather than a `Vec<u8>` because that is the
/// only part of the request the type system honours: `Vec<u8>` asks the
/// allocator for alignment 1, and a real allocator happening to return more is
/// not a guarantee to build a zero-copy parser on.
pub struct AlignedBytes {
    words: Vec<u64>,
    /// Length of the original slice. The final word is zero-padded.
    len: usize,
}

impl AlignedBytes {
    /// Copies `bytes` into aligned storage.
    pub fn new(bytes: &[u8]) -> Self {
        let mut words = vec![0u64; bytes.len().div_ceil(WORD)];
        let raw: &mut [u8] = bytemuck::cast_slice_mut(&mut words);
        if let Some(slot) = raw.get_mut(..bytes.len()) {
            slot.copy_from_slice(bytes);
        }
        Self {
            words,
            len: bytes.len(),
        }
    }

    /// The copied bytes, without the padding of the final word.
    pub fn bytes(&self) -> &[u8] {
        let raw: &[u8] = bytemuck::cast_slice(&self.words);
        raw.get(..self.len).unwrap_or(&[])
    }
}

/// Bytes per word of [`AlignedBytes`] storage.
const WORD: usize = core::mem::size_of::<u64>();

/// The alignment a zero-copy structure view needs. Every on-disk struct the
/// readers cast to is `#[repr(C)]` over the little-endian integer newtypes, so
/// the widest of those decides it.
const PARSE_ALIGN: usize = core::mem::align_of::<U64>();

/// Whether `bytes` starts where a structure view may begin. A slice that does
/// not is not malformed input; it is a buffer the caller allocated without
/// knowing what would be read out of it.
fn is_parse_aligned(bytes: &[u8]) -> bool {
    bytes.as_ptr().addr().is_multiple_of(PARSE_ALIGN)
}

/// The binary format of an input object, identified by its leading magic
/// bytes.
///
/// This is the dispatch seam between the per-format readers: the driver routes
/// an input to the ELF, Mach-O or COFF link path based on [`Format::detect`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Format {
    /// ELF (`\x7fELF`).
    Elf,
    /// Mach-O, including fat containers (a 64-bit single slice is read
    /// directly; 32-bit and fat are rejected by the reader).
    MachO,
    /// Bare COFF object (`.obj`/`.o`), identified by a known `Machine` field
    /// in the first two bytes.
    Coff,
    /// PE image (`.exe`/`.dll`), identified by the `MZ` DOS stub. Wraps a
    /// COFF object; produced by the COFF writer, never taken as link input.
    Pe,
    /// LLVM IR bitcode, in either the bare (`BC\xc0\xde`) or the wrapped
    /// spelling a compiler emits for `-flto`.
    ///
    /// Not a format any of the writers consume. It is identified so that the
    /// link can hand it to the LTO plugin, and so that a link without one can
    /// say what the file is rather than that it is unrecognised.
    Bitcode,
}

impl Format {
    /// Identifies the format of `bytes` from its magic, or returns `None` for
    /// an unknown or truncated prefix.
    pub fn detect(bytes: &[u8]) -> Option<Self> {
        // PE is checked first: its `MZ` DOS stub precedes any COFF machine
        // field, and a COFF object never begins with `MZ`.
        if bytes.starts_with(MZ.as_slice()) {
            return Some(Self::Pe);
        }
        let magic: [u8; 4] = bytes.get(..4)?.try_into().ok()?;
        match magic {
            ELFMAGIC => Some(Self::Elf),
            BITCODE_MAGIC | BITCODE_WRAPPER_MAGIC => Some(Self::Bitcode),
            MH_MAGIC_BE_64 | MH_MAGIC_LE_64 | MH_MAGIC_BE_32
            | MH_MAGIC_LE_32 | FAT_MAGIC_BE | FAT_MAGIC_BE_64 => {
                Some(Self::MachO)
            }
            _ if is_coff_machine(bytes) => Some(Self::Coff),
            _ => None,
        }
    }
}

// ELF magic: `\x7fELF`.
const ELFMAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
// LLVM IR bitcode, as `clang -flto -c` writes it.
const BITCODE_MAGIC: [u8; 4] = *b"BC\xc0\xde";
// The bitcode wrapper header, `0x0b17c0de` little-endian. Darwin toolchains
// emit it; the payload it points at is ordinary bitcode.
const BITCODE_WRAPPER_MAGIC: [u8; 4] = [0xde, 0xc0, 0x17, 0x0b];
// Mach-O 64-bit, big-endian and little-endian on-disk byte orders.
const MH_MAGIC_BE_64: [u8; 4] = [0xfe, 0xed, 0xfa, 0xcf];
const MH_MAGIC_LE_64: [u8; 4] = [0xcf, 0xfa, 0xed, 0xfe];
// Mach-O 32-bit, big-endian and little-endian on-disk byte orders.
const MH_MAGIC_BE_32: [u8; 4] = [0xfe, 0xed, 0xfa, 0xce];
const MH_MAGIC_LE_32: [u8; 4] = [0xce, 0xfa, 0xed, 0xfe];
// Fat ("universal") Mach-O containers, 32-bit and 64-bit offsets.
const FAT_MAGIC_BE: [u8; 4] = [0xca, 0xfe, 0xba, 0xbe];
const FAT_MAGIC_BE_64: [u8; 4] = [0xca, 0xfe, 0xba, 0xbf];
// PE DOS-stub magic.
const MZ: [u8; 2] = *b"MZ";

/// Whether the first two bytes are a known COFF `Machine` value. COFF objects
/// carry no signature of their own; the little-endian machine field is the only
/// discriminator, matching the set `object::read::FileKind` recognises.
fn is_coff_machine(bytes: &[u8]) -> bool {
    let Some(head) = bytes.get(..2) else {
        return false;
    };
    let machine = u16::from_le_bytes([head[0], head[1]]);
    matches!(
        machine,
        crate::coff::constants::IMAGE_FILE_MACHINE_AMD64
            | crate::coff::constants::IMAGE_FILE_MACHINE_I386
            | crate::coff::constants::IMAGE_FILE_MACHINE_ARM64
            | crate::coff::constants::IMAGE_FILE_MACHINE_ARMNT
    )
}

/// An object taking part in the link.
pub enum InputFile<'data> {
    /// A zero-copy view over a memory-mapped object.
    Borrowed(BorrowedFile<'data>),
    /// An owned copy of an archive member's bytes.
    Owned(OwnedFile),
}

/// A zero-copy object view over mapped bytes.
pub struct BorrowedFile<'data> {
    path: PathBuf,
    obj: ObjectFile<'data>,
    /// Precomputed `target section index -> relocations` table, built once at
    /// open so the per-section scan/apply loops avoid re-walking the whole
    /// section table for every lookup. [`Relocs::None`] for sections with no
    /// relocation section of their own.
    reloc_by_target: Vec<Relocs<'data>>,
}

/// An owned archive member.
pub struct OwnedFile {
    path: PathBuf,
    bytes: AlignedBytes,
    /// `target section index -> where its relocations live`, built once when
    /// the member is recorded.
    ///
    /// [`BorrowedFile`] holds slices; a member owns its bytes, so a slice into
    /// them cannot be stored beside them. Offsets can, and they answer the
    /// same question: the per-section lookup used to re-parse the member and
    /// re-walk its whole section table, once per query, for every pass that
    /// asks. A static `libc.a` link touches hundreds of members.
    reloc_by_target: Vec<RelocSpan>,
}

/// Where one target section's relocations live within a member's bytes.
#[derive(Clone, Copy)]
enum RelocSpan {
    /// No relocation section applies.
    None,
    /// `SHT_REL`, which this linker does not apply.
    Rel,
    /// `SHT_RELA`: a byte range holding whole [`Rela64`] entries.
    Rela { off: usize, len: usize },
}

impl<'data> InputFile<'data> {
    /// Parses `bytes` (a mapped object) in place.
    pub fn open(path: &Path, bytes: &'data [u8]) -> Result<Self> {
        let obj = ObjectFile::parse(bytes)?;
        let reloc_by_target = obj.reloc_index()?;
        Ok(Self::Borrowed(BorrowedFile {
            path: path.to_path_buf(),
            obj,
            reloc_by_target,
        }))
    }

    /// Parses `bytes` and extracts its global symbols in one pass, returning
    /// both the file view and its symbols. Both borrow the same `'data` byte
    /// mapping (not each other), so the caller can hold them side by side with
    /// no self-referential borrow. Used by the parallel parse pass so
    /// extraction runs alongside parsing without shared mutable state.
    pub fn open_with_symbols(
        path: &Path,
        bytes: &'data [u8],
    ) -> Result<(Self, Vec<InputSymbol<'data>>, Vec<Group<'data>>)> {
        let obj = ObjectFile::parse(bytes)?;
        // Only a relocatable object can be linked into an image. A finished
        // executable or shared object reaching here is not one: its symbols
        // carry absolute addresses, which this path reads as offsets within
        // their sections, so the definitions come out as nonsense or as
        // duplicate-symbol errors against the real inputs. A shared object is
        // recognised before this point and taken as a dependency; anything
        // else the caller named is a mistake worth saying out loud, which is
        // what lld's "unknown file type" does.
        if !obj.is_relocatable() {
            return Err(Error::Format(
                "not a relocatable object: only ET_REL inputs can be linked \
                 into an image",
            ));
        }
        let symbols = extract_globals(&obj)?;
        // The section groups ride along: this pass is the one place every
        // direct object is already parsed in parallel, and the COMDAT scan
        // wants them all up front. Parsing them here spares a second
        // parallel walk over every file's section table.
        let mut groups = Vec::new();
        obj.groups(&mut groups)?;
        let reloc_by_target = obj.reloc_index()?;
        Ok((
            Self::Borrowed(BorrowedFile {
                path: path.to_path_buf(),
                obj,
                reloc_by_target,
            }),
            symbols,
            groups,
        ))
    }

    /// Records an archive member by copying its bytes into a stable home.
    ///
    /// Members sit at arbitrary offsets inside the archive, so the copy is the
    /// only way to view one; [`AlignedBytes`] makes it one a structure view may
    /// start at, which the source offset was under no obligation to be.
    pub fn from_member(path: &Path, bytes: &[u8]) -> Result<Self> {
        let owned = AlignedBytes::new(bytes);
        let reloc_by_target = reloc_spans(owned.bytes())?;
        Ok(Self::Owned(OwnedFile {
            path: path.to_path_buf(),
            bytes: owned,
            reloc_by_target,
        }))
    }

    /// The path this input was loaded from (a synthetic name for members).
    pub fn path(&self) -> &Path {
        match self {
            Self::Borrowed(b) => &b.path,
            Self::Owned(o) => &o.path,
        }
    }

    /// The final path component, used in diagnostics.
    pub fn basename(&self) -> &str {
        self.path()
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
    }

    /// A view of the object. For borrowed inputs this is a cheap copy; for
    /// owned members the bytes are re-parsed each call.
    pub fn object(&self) -> Result<ObjectFile<'_>> {
        match self {
            Self::Borrowed(b) => Ok(b.obj),
            Self::Owned(o) => ObjectFile::parse(o.bytes.bytes()),
        }
    }

    /// The `SHT_RELA` entries that apply to `target_shndx`, if any. For
    /// borrowed inputs this reads a precomputed O(1) lookup table; for owned
    /// archive members it falls back to a fresh scan (members are rare).
    ///
    /// Fails for a section relocated by a form xold cannot apply; see
    /// [`Relocs`].
    pub fn relocations(&self, target_shndx: u16) -> Result<Option<&[Rela64]>> {
        match self {
            Self::Borrowed(b) => b
                .reloc_by_target
                .get(usize::from(target_shndx))
                .copied()
                .unwrap_or(Relocs::None)
                .entries(),
            Self::Owned(o) => o.relocations(target_shndx),
        }
    }

    /// One section's bytes, borrowed from this input rather than copied.
    ///
    /// The slice borrows the input, not the temporary view used to find it,
    /// so a caller can keep it for as long as it keeps the input. Folding
    /// copied every candidate's bytes into a `Vec` for want of this.
    pub fn section_bytes(&self, shdr: &Shdr64) -> Result<&[u8]> {
        match self {
            Self::Borrowed(b) => b.obj.section_data(shdr),
            Self::Owned(o) => {
                ObjectFile::parse(o.bytes.bytes())?.section_data(shdr)
            }
        }
    }

    /// The full input symbol table (local and global symbols), needed to
    /// resolve relocations that reference local and section symbols.
    pub fn symbol_table(&self) -> Result<Option<crate::elf::SymbolTable<'_>>> {
        self.object()?.symbol_table()
    }

    /// Every global or weak symbol this input contributes. Local symbols are
    /// file-scoped and are never merged across files, so they are skipped.
    pub fn global_symbols(&self) -> Result<Vec<InputSymbol<'_>>> {
        let obj = self.object()?;
        extract_globals(&obj)
    }

    /// [`Self::global_symbols`] with the names tied to the input mapping's
    /// own lifetime, which only a direct (borrowed) input can offer.
    ///
    /// The bulk symbol fold stores borrowed names, so it needs `'data`
    /// slices; an archive member's bytes are owned by this very value and
    /// cannot outlive it. Members never appear in the bulk fold -- they are
    /// pulled only after it -- so asking this of one is a driver bug, and
    /// it is reported rather than worked around.
    pub fn global_symbols_direct(&self) -> Result<Vec<InputSymbol<'data>>> {
        match self {
            Self::Borrowed(file) => extract_globals(&file.obj),
            Self::Owned(_) => Err(Error::Format(
                "archive member reached the direct-input symbol fold",
            )),
        }
    }
}

impl OwnedFile {
    /// The relocations applying to `target_shndx`, read through the cached
    /// span rather than by re-parsing the member.
    fn relocations(&self, target_shndx: u16) -> Result<Option<&[Rela64]>> {
        let span = self
            .reloc_by_target
            .get(usize::from(target_shndx))
            .copied()
            .unwrap_or(RelocSpan::None);
        let RelocSpan::Rela { off, len } = span else {
            return match span {
                RelocSpan::Rel => Err(Error::Format(
                    "SHT_REL relocations (only SHT_RELA is supported)",
                )),
                _ => Ok(None),
            };
        };
        let bytes = self
            .bytes
            .bytes()
            .get(off..off.saturating_add(len))
            .ok_or(Error::OutOfRange("member relocation span"))?;
        bytemuck::try_cast_slice(bytes)
            .map(Some)
            .map_err(|_| Error::Format("misaligned member relocations"))
    }
}

/// Builds the per-target relocation spans of one member.
///
/// The member is parsed once, here, and the result answers every later query.
fn reloc_spans(bytes: &[u8]) -> Result<Vec<RelocSpan>> {
    use crate::elf::constants::{SHT_REL, SHT_RELA};
    let obj = ObjectFile::parse(bytes)?;
    let sections = obj.sections();
    let mut out = vec![RelocSpan::None; sections.len()];
    for shdr in sections {
        let kind = shdr.sh_type.get();
        if kind != SHT_RELA && kind != SHT_REL {
            continue;
        }
        let Ok(target) = usize::try_from(shdr.sh_info.get()) else {
            continue;
        };
        let Some(slot) = out.get_mut(target) else {
            continue;
        };
        if !matches!(slot, RelocSpan::None) {
            continue;
        }
        *slot = if kind == SHT_REL {
            RelocSpan::Rel
        } else {
            RelocSpan::Rela {
                off: usize::try_from(shdr.sh_offset.get()).unwrap_or(0),
                len: usize::try_from(shdr.sh_size.get()).unwrap_or(0),
            }
        };
    }
    Ok(out)
}

/// Extracts every global or weak symbol from `obj`, borrowing the symbol names
/// from the same `'data` mapping as the object. Shared between the direct-input
/// parallel parse ([`InputFile::open_with_symbols`]) and the on-demand
/// extraction path ([`InputFile::global_symbols`]).
///
/// Each record also carries its position (`sym_idx`) inside the input's symbol
/// table, so the resolution pass can record `(file, sym_idx) -> SymbolId`
/// directly at intern time instead of re-probing the global name table later.
fn extract_globals<'d>(obj: &ObjectFile<'d>) -> Result<Vec<InputSymbol<'d>>> {
    let Some(symtab) = obj.symbol_table()? else {
        return Ok(Vec::new());
    };
    // Globals are a fraction of a symbol table, but sizing to the whole of it
    // costs one allocation instead of a dozen regrowths, and the excess is
    // released when the caller drops the list.
    let mut out = Vec::with_capacity(symtab.syms.len());
    for (idx, sym) in symtab.iter().enumerate() {
        // Every symbol is checked, local ones included: a local resolves
        // through the same raw `st_shndx` and lands in the same wrong place.
        // This walk already visits the whole table, so the check is one
        // comparison per symbol and no extra pass.
        check_shndx(sym.st_shndx.get())?;
        if sym.bind() == STB_LOCAL {
            continue;
        }
        let name = symtab.name(sym);
        // A version suffix is a spelling of a name, not part of it: the
        // stem is what every spelling resolves as.
        let name = crate::symbol::version_stem(name);
        if name.is_empty() {
            continue;
        }
        let sym_idx = u32::try_from(idx)
            .map_err(|_| Error::OutOfRange("symbol index"))?;
        // A common carries its alignment in `st_value`, and every placement
        // cursor uses it as a mask -- the same contract `sh_addralign` has,
        // checked at the same distance from the file. A non-power-of-two
        // would silently round to the power below; a huge one is an
        // arithmetic hazard, not a request.
        if sym.st_shndx.get() == SHN_COMMON {
            crate::util::check_align(
                sym.st_value.get(),
                "common symbol alignment is not a usable power of two",
            )?;
        }
        out.push(InputSymbol {
            name,
            name_hash: crate::symbol::name_hash(name),
            binding: sym.bind(),
            type_: sym.type_(),
            visibility: sym.visibility(),
            shndx: sym.st_shndx.get(),
            value: sym.st_value.get(),
            size: sym.st_size.get(),
            sym_idx,
        });
    }
    Ok(out)
}

/// Rejects an `st_shndx` this linker would resolve to the wrong place.
///
/// Three reserved values carry their own meaning and are handled everywhere
/// `st_shndx` is read: `SHN_UNDEF`, `SHN_ABS` and `SHN_COMMON`. The rest are
/// section indices, and a reserved value that is none of the three is not one.
///
/// `SHN_XINDEX` is the case that occurs in practice. An object with 0xff00 or
/// more sections -- a large C++ translation unit built with
/// `-ffunction-sections` reaches that -- spells the real index in a
/// `SHT_SYMTAB_SHNDX` table alongside the symbol table. Nothing here reads
/// that table, and taking the field raw treats the symbol as defined in
/// section 0xffff, which is never placed: the address lookup misses, falls
/// back to zero, and the symbol resolves to `st_value` bytes from the start of
/// the image. Every reference to it is then silently wrong.
///
/// Refusing is what a linker owes an input it cannot represent. lld reads the
/// table instead, which is the better answer and the one to implement when
/// such objects need to link; until then the failure is loud.
fn check_shndx(shndx: u16) -> Result<()> {
    if shndx < SHN_LORESERVE
        || matches!(shndx, SHN_UNDEF | SHN_ABS | SHN_COMMON)
    {
        return Ok(());
    }
    if shndx == SHN_XINDEX {
        return Err(Error::Format(
            "symbol uses SHN_XINDEX: SHT_SYMTAB_SHNDX is not implemented",
        ));
    }
    Err(Error::Format(
        "symbol has a reserved st_shndx this linker \
                       does not implement",
    ))
}

/// A single symbol from one input: the unit of cross-file resolution.
#[derive(Clone, Copy)]
pub struct InputSymbol<'a> {
    /// NUL-delimited name bytes borrowed from the input string table.
    pub name: &'a [u8],
    /// `st_info` binding nibble (`STB_*`).
    pub binding: u8,
    /// `st_info` type nibble (`STT_*`).
    pub type_: u8,
    /// `st_other` visibility field (`STV_*`), already masked. Carried per
    /// occurrence rather than per definition: an undefined reference's
    /// visibility constrains the resolved symbol just as a definition's does.
    pub visibility: u8,
    /// Raw `st_shndx` (a section index or a `SHN_*` reserved value).
    pub shndx: u16,
    /// `st_value`; for common symbols this carries the alignment.
    pub value: u64,
    /// `st_size`.
    pub size: u64,
    /// Position of this symbol inside its input's symbol table. Carried
    /// through extraction so the resolution pass can record the `(file,
    /// sym_idx) -> SymbolId` mapping at intern time without a later name
    /// probe.
    pub sym_idx: u32,
    /// Hash of [`Self::name`] under the symbol table's hasher.
    ///
    /// Computed here because extraction runs per input on the parallel parse
    /// pass, whereas the fold that consumes it is serial: hashing a million
    /// names is work the serial pass should not be doing.
    pub name_hash: u64,
}

impl InputSymbol<'_> {
    /// An undefined reference (`SHN_UNDEF`).
    pub const fn is_undefined(&self) -> bool {
        self.shndx == SHN_UNDEF
    }

    /// A tentative definition (`SHN_COMMON`).
    pub const fn is_common(&self) -> bool {
        self.shndx == SHN_COMMON
    }

    /// An absolute symbol not tied to a section (`SHN_ABS`).
    pub const fn is_absolute(&self) -> bool {
        self.shndx == SHN_ABS
    }
}
