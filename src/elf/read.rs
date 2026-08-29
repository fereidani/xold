//! Zero-copy reader for ELF64 little-endian objects.
//!
//! The parse path delegates validation and the section-table walk to the
//! `object` crate (gimli-rs/object): it checks the magic, class, endianness
//! and version, resolves the extended section-numbering scheme (`SHN_XINDEX`)
//! for both the section count and the section-name string-table index, and
//! bounds-checks every table it returns. The bytes object locates are then
//! re-viewed as xold's plain [`Shdr64`] structures, so the writer and every
//! call site keep working with native `u64` fields and [`ObjectFile`] stays
//! `Copy` (it borrows only `&'data` views; the parsed `object` value is
//! dropped at the end of [`ObjectFile::parse`]).
//!
//! All structures are viewed directly over the mapped bytes. The only fallible
//! primitive is [`view_slice`]/[`view_one`], which delegate to `bytemuck` so
//! that a misaligned or wrongly sized table becomes a [`crate::Error`] rather
//! than undefined behaviour. Nothing here panics.

use bytemuck::Pod;
use object::{Endianness, elf::FileHeader64, pod, read::elf::FileHeader};

use crate::{
    elf::{
        constants::{
            ET_REL, SHT_DYNSYM, SHT_GNU_VERDEF, SHT_GNU_VERNEED,
            SHT_GNU_VERSYM, SHT_GROUP, SHT_NOBITS, SHT_REL, SHT_RELA,
            SHT_SYMTAB, STT_SECTION, VER_NDX_GLOBAL,
        },
        structs::{
            Ehdr64, Rela64, Shdr64, Sym64, Verdaux64, Verdef64, Vernaux64,
            Verneed64,
        },
    },
    error::{Error, Result},
    util::cstr_at,
};

/// On-disk size of the ELF64 header, used for bounds checks.
const EHDR_SIZE: usize = core::mem::size_of::<Ehdr64>();

/// A parsed relocatable ELF object, borrowing the mapped input bytes.
///
/// The file owns no copy of its data: every accessor returns a reference into
/// the original mapping. It is cheap to copy (a handful of references and
/// indices), which lets an owned-copy variant hand out a fresh view without a
/// self-referential borrow. The `SHT_SYMTAB`/`SHT_DYNSYM` positions are
/// resolved once at parse time so the per-section hot loops pay an O(1) lookup
/// instead of re-scanning the whole section table on every call.
#[derive(Clone, Copy)]
pub struct ObjectFile<'data> {
    bytes: &'data [u8],
    header: &'data Ehdr64,
    sections: &'data [Shdr64],
    shstrtab: &'data [u8],
    /// The single `SHT_SYMTAB` and its string table, if any.
    symtab: Option<SymbolTable<'data>>,
    /// The single `SHT_DYNSYM` and its string table, if any.
    dynsym: Option<SymbolTable<'data>>,
}

/// What relocates one input section.
///
/// xold applies `SHT_RELA` only. `SHT_REL` keeps its addend in the target
/// bytes rather than in the entry, so every relocation type needs a second
/// implementation that reads the field before writing it, and none of xold's
/// value computations has one. An `SHT_REL` section is therefore recorded as
/// present-but-unreadable rather than dropped: reading it as "no relocations"
/// would link an image whose call sites were never patched, which is a running
/// binary rather than a diagnostic.
///
/// The distinction is recorded here and collapsed in [`Self::entries`], which
/// is the one way to reach the entries, so no pass can reintroduce the silent
/// answer. It is not refused at parse time: a `.rel.*` section applying to a
/// section this link never places -- a discarded COMDAT member, a `.comment`
/// -- costs the link nothing, and refusing it would turn a linkable input away
/// over bytes nobody reads. Asking for it is what fails.
///
/// All three 64-bit targets xold supports mandate RELA in their psABI, so this
/// is a guard against malformed or hand-written input rather than a gap in
/// coverage. Both reference linkers implement REL because both also target
/// architectures that use it.
#[derive(Clone, Copy)]
pub enum Relocs<'data> {
    /// No relocation section applies.
    None,
    /// `SHT_RELA`: addends stored inline with each entry.
    Rela(&'data [Rela64]),
    /// `SHT_REL`: addends stored in the target bytes.
    Rel,
}

impl<'data> Relocs<'data> {
    /// The entries a pass may apply, or an error for a form xold cannot.
    pub fn entries(self) -> Result<Option<&'data [Rela64]>> {
        match self {
            Self::None => Ok(None),
            Self::Rela(entries) => Ok(Some(entries)),
            Self::Rel => Err(Error::Format(
                "SHT_REL relocations (only SHT_RELA is supported)",
            )),
        }
    }
}

/// A symbol table together with its string table.
#[derive(Clone, Copy)]
pub struct SymbolTable<'data> {
    pub syms: &'data [Sym64],
    strtab: &'data [u8],
    /// `sh_info`: the index of the first non-local symbol, clamped to the
    /// table length. See [`Self::global_count`].
    first_global: usize,
}

/// One `SHT_GROUP` (COMDAT) section: the signature that identifies the group,
/// the flag word, and the member section indices in this file.
///
/// The signature is borrowed from the file's string table (symbol name) or the
/// section-name table (for an `STT_SECTION` signature). The members slice
/// excludes the leading flag word.
#[derive(Clone, Copy)]
pub struct Group<'data> {
    /// The group's signature bytes (a symbol or section name).
    pub signature: &'data [u8],
    /// [`crate::symbol::name_hash`] of the signature, computed where groups
    /// parse in parallel so the order-dependent dedup walk that follows only
    /// probes with it.
    pub signature_hash: u64,
    /// The first word of the group payload (`GRP_COMDAT`, ...).
    pub flags: u32,
    /// Section header indices in this file that belong to the group.
    pub members: &'data [u32],
}

impl<'data> ObjectFile<'data> {
    /// Parses `bytes` as an ELF64 little-endian object.
    ///
    /// `object` performs every structural check (magic, class, endianness,
    /// version, extended numbering, table bounds and alignment). The section
    /// bytes it locates are then re-viewed as plain [`Shdr64`] so the rest of
    /// the linker reads native `u64` fields. xold handles little-endian ELF
    /// only; a big-endian object is rejected so the plain views are
    /// interpretation-correct.
    pub fn parse(bytes: &'data [u8]) -> Result<Self> {
        let header = FileHeader64::<Endianness>::parse(bytes)
            .map_err(|_| Error::Format("invalid ELF header"))?;
        if !header.is_little_endian() {
            return Err(Error::Format("expected little-endian ELF"));
        }
        let endian = Endianness::Little;
        let table = header
            .sections(endian, bytes)
            .map_err(|_| Error::Format("invalid ELF section table"))?;

        // `object` validated the 64-byte header; re-view it as `Ehdr64`.
        let ehdr = view_one::<Ehdr64>(bytes, 0, EHDR_SIZE, "ELF header")?;
        // `object` located and bounds-checked the section headers; re-view the
        // same bytes as the plain `Shdr64` the writer and readers expect. The
        // two layouts are `#[repr(C)]`-identical, and re-viewing the verified
        // bytes avoids any reinterpretation of object's endian-typed types.
        let sections = cast_section_table(table.iter().as_slice())?;
        // `object` resolved `e_shstrndx`, including the `SHN_XINDEX` case.
        // The accessor refuses the reserved zero before it ever learns the
        // file is table-free, so the empty case never reaches it.
        let shstrtab = if sections.is_empty() {
            &[][..]
        } else {
            let shstrndx = header
                .shstrndx(endian, bytes)
                .map_err(|_| Error::Format("invalid ELF section name index"))?;
            name_table(bytes, sections, shstrndx)?
        };
        // A section-name offset past the table would read as an empty name,
        // dropping the section out of every name-based classification
        // without a word said. Zero stays legal: it is the unnamed symbol
        // every section table leads with.
        names_in_range(
            shstrtab,
            sections.iter().map(|s| s.sh_name.get()),
            "section name offset",
        )?;
        // Build both symbol tables once here, so the hot accessors below
        // hand back a parsed table instead of re-locating and re-validating
        // it on each call. A duplicate of either type is rejected exactly as
        // the scanning variant did.
        //
        // Validating the name offsets is what makes this worth doing: the
        // check is linear in the symbol count, and the accessors are reached
        // from every pass that reads a symbol -- resolution, gc, icf, layout,
        // relocation scanning and the writer -- so leaving it there re-scans
        // each table more than a dozen times over per link.
        let symtab = read_symbol_table(
            bytes,
            sections,
            find_unique_section(sections, SHT_SYMTAB)?,
            "symbol table entries",
            "symbol string table",
            "symbol name offset",
        )?;
        let dynsym = read_symbol_table(
            bytes,
            sections,
            find_unique_section(sections, SHT_DYNSYM)?,
            "dynamic symbol table entries",
            "dynamic string table",
            "dynamic symbol name offset",
        )?;

        Ok(Self {
            bytes,
            header: ehdr,
            sections,
            shstrtab,
            symtab,
            dynsym,
        })
    }

    /// The raw ELF header.
    pub fn header(&self) -> &Ehdr64 {
        self.header
    }

    /// The machine this object was compiled for (`EM_*`).
    pub fn machine(&self) -> u16 {
        self.header.e_machine.get()
    }

    /// Whether this is a relocatable object (`ET_REL`).
    pub fn is_relocatable(&self) -> bool {
        self.header.e_type.get() == ET_REL
    }

    /// All section headers, in file order.
    pub fn sections(&self) -> &'data [Shdr64] {
        self.sections
    }

    /// The NUL-terminated name of `shdr`, borrowed from the section-name table.
    pub fn section_name(&self, shdr: &Shdr64) -> &'data [u8] {
        cstr_at(self.shstrtab, shdr.sh_name.get())
    }

    /// Whether a section's name starts with `prefix`, without scanning to the
    /// terminator.
    ///
    /// Classification asks only a couple of fixed questions of each name, and
    /// a link reaches hundreds of thousands of sections; finding the end of
    /// every name just to compare its first few bytes is most of the cost.
    pub fn section_name_starts_with(
        &self,
        shdr: &Shdr64,
        prefix: &[u8],
    ) -> bool {
        let at = shdr.sh_name.get() as usize;
        self.shstrtab
            .get(at..at.saturating_add(prefix.len()))
            .is_some_and(|head| head == prefix)
    }

    /// Whether a section's name is exactly `want`, without scanning to the
    /// terminator: the bytes must match and be followed by the NUL.
    pub fn section_name_is(&self, shdr: &Shdr64, want: &[u8]) -> bool {
        let at = shdr.sh_name.get() as usize;
        let end = at.saturating_add(want.len());
        self.shstrtab.get(at..end).is_some_and(|head| head == want)
            && self.shstrtab.get(end) == Some(&0)
    }

    /// The payload bytes of `shdr`. `SHT_NOBITS` sections return an empty
    /// slice since they occupy no file space.
    pub fn section_data(&self, shdr: &Shdr64) -> Result<&'data [u8]> {
        if shdr.sh_type.get() == SHT_NOBITS {
            return Ok(&[]);
        }
        section_payload(self.bytes, shdr, "section data")
    }

    /// The primary symbol table, if present.
    ///
    /// Parsed and validated once in [`Self::parse`]; this hands that back.
    pub fn symbol_table(&self) -> Result<Option<SymbolTable<'data>>> {
        Ok(self.symtab)
    }

    /// The dynamic symbol table (`.dynsym`), if present. A shared object's
    /// exports live here; the loader resolves imports against it.
    ///
    /// Parsed and validated once in [`Self::parse`]; this hands that back.
    pub fn dynamic_symbols(&self) -> Result<Option<SymbolTable<'data>>> {
        Ok(self.dynsym)
    }

    /// The `SHT_RELA` entries that apply to `target_shndx`, if any.
    ///
    /// Fails for a section an `SHT_REL` section applies to; see [`Relocs`].
    pub fn relocations(
        &self,
        target_shndx: u16,
    ) -> Result<Option<&'data [Rela64]>> {
        for shdr in self.sections {
            let applies =
                shdr.sh_type.get() == SHT_RELA || shdr.sh_type.get() == SHT_REL;
            if applies && shdr.sh_info.get() == u32::from(target_shndx) {
                return relocs_of(self.bytes, shdr)?.entries();
            }
        }
        Ok(None)
    }

    /// Precomputes a dense `target section index -> relocations` table so the
    /// per-section hot loops (scan, apply, gc, icf) look up a section's
    /// relocations in O(1) instead of re-scanning the whole section table on
    /// every call. Each target keeps the first matching relocation section in
    /// file order, matching [`Self::relocations`].
    ///
    /// The table records what applies rather than what can be read, so an
    /// unsupported form is reported by the lookup and not by the parse; see
    /// [`Relocs`].
    pub fn reloc_index(&self) -> Result<Vec<Relocs<'data>>> {
        let len = self.sections.len();
        let mut out = vec![Relocs::None; len];
        for shdr in self.sections {
            let is_reloc =
                shdr.sh_type.get() == SHT_RELA || shdr.sh_type.get() == SHT_REL;
            if !is_reloc {
                continue;
            }
            let Ok(target) = usize::try_from(shdr.sh_info.get()) else {
                continue;
            };
            if target < len && matches!(out[target], Relocs::None) {
                out[target] = relocs_of(self.bytes, shdr)?;
            }
        }
        Ok(out)
    }

    /// Every `SHT_GROUP` (COMDAT) section in this file, with its signature
    /// resolved and its member section indices. Used by the COMDAT dedup pass
    /// to keep one group per signature and discard the rest.
    pub fn groups(&self, out: &mut Vec<Group<'data>>) -> Result<()> {
        out.clear();
        // Every group in an object names the same symbol table, so the view is
        // resolved once and reused. A C++ translation unit emits one group per
        // template instantiation and per inline function, and re-deriving the
        // table for each cost a bounds-checked slice cast per group.
        let mut syms = GroupSymbols::default();
        for shdr in self.sections {
            if shdr.sh_type.get() != SHT_GROUP {
                continue;
            }
            let data = self.section_data(shdr)?;
            let words: &[u32] = bytemuck::try_cast_slice(data)
                .map_err(|_| Error::Format("group section words"))?;
            if words.is_empty() {
                continue;
            }
            let signature = self.group_signature(shdr, &mut syms)?;
            out.push(Group {
                signature,
                signature_hash: crate::symbol::name_hash(signature),
                flags: words[0],
                members: &words[1..],
            });
        }
        Ok(())
    }

    /// Parses the GNU symbol-versioning sections (`.gnu.version`,
    /// `.gnu.version_r`, `.gnu.version_d`) into a borrowed [`VersionTable`].
    /// Returns an empty table when the object carries none. The `sh_link` of
    /// each version section names the string table its offsets resolve
    /// against (`.dynstr` for a shared object, sometimes `.strtab` for a
    /// relocatable object that emits version info).
    pub fn version_table(&self) -> Result<VersionTable<'data>> {
        let mut versym: &[u16] = &[];
        let mut needs = Vec::new();
        let mut defs = Vec::new();
        for shdr in self.sections {
            match shdr.sh_type.get() {
                SHT_GNU_VERSYM => {
                    let data = self.section_data(shdr)?;
                    versym = bytemuck::try_cast_slice(data)
                        .map_err(|_| Error::Format(".gnu.version words"))?;
                    self.check_versym_len(shdr, versym.len())?;
                }
                SHT_GNU_VERNEED => {
                    let data = self.section_data(shdr)?;
                    let strtab = self.strtab_for(shdr)?;
                    needs = parse_verneed(data, strtab)?;
                }
                SHT_GNU_VERDEF => {
                    let data = self.section_data(shdr)?;
                    let strtab = self.strtab_for(shdr)?;
                    defs = parse_verdef(data, strtab)?;
                }
                _ => {}
            }
        }
        Ok(VersionTable {
            versym,
            needs,
            defs,
        })
    }

    /// Rejects a `.gnu.version` array shorter than the symbol table its
    /// `sh_link` names.
    ///
    /// The array is defined to be parallel to that table, one `u16` per
    /// symbol. A short one is malformed, and answering the missing rows with
    /// `VER_NDX_GLOBAL` would silently turn a hidden or version-bound symbol
    /// into a default-version one. A longer array is left alone: the extra
    /// words are unreachable and some producers pad.
    ///
    /// A `sh_link` that names no section, or one that is not a symbol table,
    /// carries nothing to check against and is passed through.
    fn check_versym_len(&self, shdr: &Shdr64, len: usize) -> Result<()> {
        let Some(linked) = self.sections.get(shdr.sh_link.get() as usize)
        else {
            return Ok(());
        };
        let ty = linked.sh_type.get();
        if ty != SHT_SYMTAB && ty != SHT_DYNSYM {
            return Ok(());
        }
        let entsize = core::mem::size_of::<Sym64>() as u64;
        let count = usize::try_from(linked.sh_size.get() / entsize)
            .map_err(|_| Error::OutOfRange("symbol table entries"))?;
        if len < count {
            return Err(Error::Format(
                ".gnu.version is shorter than its symbol table",
            ));
        }
        Ok(())
    }

    /// Resolves the string table a versioning section's `sh_link` names. The
    /// dynamic linker resolves `.gnu.version_r`/`.gnu.version_d` offsets
    /// against `.dynstr`; some relocatable objects point at `.strtab` instead.
    fn strtab_for(&self, shdr: &Shdr64) -> Result<&'data [u8]> {
        let linked = self
            .sections
            .get(shdr.sh_link.get() as usize)
            .ok_or(Error::OutOfRange("version string table index"))?;
        self.section_data(linked)
    }

    /// Resolves the signature of one group section. `sh_info` selects the
    /// signature symbol in the symbol table named by `sh_link`. An
    /// `STT_SECTION` signature contributes its section's name (the common
    /// compiler-emitted form); any other symbol contributes its own name.
    ///
    /// `syms` caches the table `sh_link` named for the previous group, which
    /// in practice is the same one; see [`Self::groups`].
    fn group_signature(
        &self,
        group: &Shdr64,
        syms: &mut GroupSymbols<'data>,
    ) -> Result<&'data [u8]> {
        let link = group.sh_link.get();
        if syms.link != Some(link) {
            *syms = self.group_symbols(link)?;
        }
        let sym = syms
            .syms
            .get(group.sh_info.get() as usize)
            .ok_or(Error::OutOfRange("group signature symbol"))?;
        if sym.type_() == STT_SECTION {
            let sec = self
                .sections
                .get(sym.st_shndx.get() as usize)
                .ok_or(Error::OutOfRange("signature section"))?;
            Ok(self.section_name(sec))
        } else {
            Ok(cstr_at(syms.strtab, sym.st_name.get()))
        }
    }

    /// The symbol table a group's `sh_link` names, with the string table its
    /// own `sh_link` names.
    fn group_symbols(&self, link: u32) -> Result<GroupSymbols<'data>> {
        let symtab = self
            .sections
            .get(link as usize)
            .ok_or(Error::OutOfRange("group symbol table section"))?;
        let syms: &[Sym64] = view_slice(
            self.bytes,
            symtab,
            core::mem::size_of::<Sym64>(),
            "group symbol table",
        )?;
        let strtab = self
            .sections
            .get(symtab.sh_link.get() as usize)
            .ok_or(Error::OutOfRange("group string table"))?;
        let strtab = section_payload(self.bytes, strtab, "group string table")?;
        Ok(GroupSymbols {
            link: Some(link),
            syms,
            strtab,
        })
    }
}

/// The symbol and string tables one object's groups resolve their signatures
/// through, held across [`ObjectFile::groups`] so the views are derived once.
#[derive(Default)]
struct GroupSymbols<'data> {
    /// The `sh_link` these views came from, or `None` before the first group.
    link: Option<u32>,
    syms: &'data [Sym64],
    strtab: &'data [u8],
}

/// Finds the index of the single section of `sh_type`, reporting an error if
/// there is more than one. Runs once at parse time so the per-call accessors
/// avoid re-scanning the section table.
fn find_unique_section(
    sections: &[Shdr64],
    sh_type: u32,
) -> Result<Option<usize>> {
    let mut found: Option<usize> = None;
    for (i, shdr) in sections.iter().enumerate() {
        if shdr.sh_type.get() == sh_type {
            if found.is_some() {
                return Err(Error::Format("duplicate symbol table section"));
            }
            found = Some(i);
        }
    }
    Ok(found)
}

impl<'data> SymbolTable<'data> {
    /// The NUL-terminated name of `sym`, borrowed from the string table.
    pub fn name(&self, sym: &Sym64) -> &'data [u8] {
        cstr_at(self.strtab, sym.st_name.get())
    }

    /// Iterates the symbol entries.
    pub fn iter(&self) -> core::slice::Iter<'_, Sym64> {
        self.syms.iter()
    }

    /// How many symbols can be non-local, from the table's own `sh_info`.
    ///
    /// ELF orders every local symbol before every global one, so this is an
    /// exact upper bound on what a globals-only pass keeps -- the right size
    /// for its output, rather than the whole table. Roughly half of a
    /// compiler's symbol table is local, so sizing to the table over-reserves
    /// by about that much on every input at once.
    pub fn global_count(&self) -> usize {
        self.syms.len().saturating_sub(self.first_global)
    }

    /// Rejects a table whose `st_name` columns point past the string table.
    ///
    /// A corrupt offset would read as an empty name: two broken symbols
    /// would merge under "" and a broken one would shadow the null symbol
    /// every table leads with, all silently. Zero stays legal; it is the
    /// unnamed slot.
    fn check_names(&self, what: &'static str) -> Result<()> {
        names_in_range(
            self.strtab,
            self.syms.iter().map(|s| s.st_name.get()),
            what,
        )
    }
}

impl<'data> IntoIterator for &'data SymbolTable<'data> {
    type Item = &'data Sym64;
    type IntoIter = core::slice::Iter<'data, Sym64>;

    fn into_iter(self) -> Self::IntoIter {
        self.syms.iter()
    }
}

// --- parsing helpers ------------------------------------------------------

/// Locates, views and validates one symbol table and its string table.
///
/// `idx` is the section index [`find_unique_section`] resolved, so a `None`
/// simply means the file carries no table of that type.
fn read_symbol_table<'data>(
    bytes: &'data [u8],
    sections: &'data [Shdr64],
    idx: Option<usize>,
    entries_what: &'static str,
    strtab_what: &'static str,
    names_what: &'static str,
) -> Result<Option<SymbolTable<'data>>> {
    let Some(idx) = idx else {
        return Ok(None);
    };
    let symtab = sections
        .get(idx)
        .ok_or(Error::OutOfRange("symbol table index"))?;
    let strtab = sections
        .get(symtab.sh_link.get() as usize)
        .ok_or(Error::OutOfRange("symbol string table index"))?;
    let syms: &'data [Sym64] =
        view_slice(bytes, symtab, core::mem::size_of::<Sym64>(), entries_what)?;
    // ELF requires the local symbols to come first and `sh_info` to hold how
    // many there are. It is a hint here, never a bound: it only sizes an
    // allocation, so a file that misstates it costs a regrowth or some
    // slack rather than a wrong answer. Clamping keeps it in range.
    let first_global = (symtab.sh_info.get() as usize).min(syms.len());
    let table = SymbolTable {
        syms,
        strtab: section_payload(bytes, strtab, strtab_what)?,
        first_global,
    };
    table.check_names(names_what)?;
    Ok(Some(table))
}

/// Whether every name offset lands inside `table`.
///
/// Offsets are checked once, where the table is parsed, so the hot
/// name reads stay unchecked borrows. Zero is exempt: it names nothing,
/// which is a state every table legitimately uses.
fn names_in_range(
    table: &[u8],
    offsets: impl Iterator<Item = u32>,
    what: &'static str,
) -> Result<()> {
    for off in offsets {
        let start = usize::try_from(off).unwrap_or(usize::MAX);
        if off != 0 && start >= table.len() {
            return Err(Error::OutOfRange(what));
        }
    }
    Ok(())
}

/// Re-views the section-header slice `object` located as plain [`Shdr64`].
/// `object::elf::SectionHeader64` and [`Shdr64`] are `#[repr(C)]`-identical;
/// borrowing the underlying bytes through `object`'s own `Pod` view keeps the
/// cast within `bytemuck`'s safe, bounds- and alignment-checked path.
fn cast_section_table(
    sections: &[object::elf::SectionHeader64<Endianness>],
) -> Result<&[Shdr64]> {
    if sections.is_empty() {
        // A table-free file (`e_shoff`/`e_shnum`/`e_shstrndx` all zero)
        // borrows nothing. The cast below would still test the pointer it
        // has no byte behind, rejecting a well-formed file over the
        // caller's buffer alignment instead of its content.
        return Ok(&[]);
    }
    let raw = pod::bytes_of_slice(sections);
    bytemuck::try_cast_slice::<u8, Shdr64>(raw).map_err(Into::into)
}

/// Resolves the section-name string table from the header at the index
/// `object` reported, borrowing the payload bytes directly.
fn name_table<'a>(
    bytes: &'a [u8],
    sections: &'a [Shdr64],
    shstrndx: u32,
) -> Result<&'a [u8]> {
    let index = usize::try_from(shstrndx)
        .map_err(|_| Error::OutOfRange("section name table index"))?;
    let shdr = sections
        .get(index)
        .ok_or(Error::OutOfRange("section name table index"))?;
    section_payload(bytes, shdr, "section name string table")
}

fn section_payload<'a>(
    bytes: &'a [u8],
    shdr: &Shdr64,
    what: &'static str,
) -> Result<&'a [u8]> {
    let offset = usize_from(shdr.sh_offset.get(), what)?;
    let size = usize_from(shdr.sh_size.get(), what)?;
    bytes
        .get(offset..end_of(offset, size, what)?)
        .ok_or(Error::OutOfRange(what))
}

/// One past the last byte of a `size`-byte span at `offset`.
///
/// Both come from the file, so their sum is the file's to choose and not a
/// number this module may assume fits. Adding them plainly overflows under a
/// profile with overflow checks -- which is a panic, in a module whose own
/// header says nothing here panics -- and wraps to a small, in-bounds range
/// without them, which is worse: the view would succeed over the wrong bytes.
fn end_of(offset: usize, size: usize, what: &'static str) -> Result<usize> {
    offset.checked_add(size).ok_or(Error::OutOfRange(what))
}

/// Classifies one relocation section, viewing its entries when xold can read
/// them. See [`Relocs`] for why an `SHT_REL` section is recorded rather than
/// refused here.
fn relocs_of<'a>(bytes: &'a [u8], shdr: &Shdr64) -> Result<Relocs<'a>> {
    match shdr.sh_type.get() {
        SHT_RELA => Ok(Relocs::Rela(view_slice(
            bytes,
            shdr,
            core::mem::size_of::<Rela64>(),
            "RELA entries",
        )?)),
        SHT_REL => Ok(Relocs::Rel),
        _ => Err(Error::Format("not a relocation section")),
    }
}

/// Views `count` entries of `entsize` bytes starting at `offset` as a slice of
/// `T`, validating both the element size and alignment.
fn view_slice<'a, T: Pod>(
    bytes: &'a [u8],
    shdr: &Shdr64,
    entsize: usize,
    what: &'static str,
) -> Result<&'a [T]> {
    if shdr.sh_entsize.get() != 0
        && usize_from(shdr.sh_entsize.get(), what)? != entsize
    {
        return Err(Error::Format(what));
    }
    let offset = usize_from(shdr.sh_offset.get(), what)?;
    let count = usize_from(shdr.sh_size.get(), what)? / entsize.max(1);
    view_slice_raw(bytes, offset, count, entsize, what)
}

/// Views a single `T` of `size` bytes at `offset`.
fn view_one<'a, T: Pod>(
    bytes: &'a [u8],
    offset: usize,
    size: usize,
    what: &'static str,
) -> Result<&'a T> {
    let raw = bytes
        .get(offset..end_of(offset, size, what)?)
        .ok_or(Error::OutOfRange(what))?;
    bytemuck::try_from_bytes(raw).map_err(Into::into)
}

/// Views `count` elements of `entsize` bytes at `offset` as `&[T]`.
fn view_slice_raw<'a, T: Pod>(
    bytes: &'a [u8],
    offset: usize,
    count: usize,
    entsize: usize,
    what: &'static str,
) -> Result<&'a [T]> {
    let span = count.checked_mul(entsize).ok_or(Error::OutOfRange(what))?;
    let raw = bytes
        .get(offset..end_of(offset, span, what)?)
        .ok_or(Error::OutOfRange(what))?;
    bytemuck::try_cast_slice::<u8, T>(raw).map_err(Into::into)
}

/// Converts a file offset or count to `usize`, rejecting implausibly large
/// values rather than truncating.
fn usize_from<T: TryInto<usize>>(
    value: T,
    what: &'static str,
) -> Result<usize> {
    value.try_into().map_err(|_| Error::OutOfRange(what))
}

// --- symbol versioning -----------------------------------------------------

/// One `Elf_Verneed` and its `Elf_Vernaux` children, parsed from
/// `.gnu.version_r`. Owned (the auxiliary vec is small) so callers can iterate
/// it without holding the section bytes.
#[derive(Clone)]
pub struct VersionNeed<'data> {
    /// The dependency soname (e.g. `libc.so.6`), borrowed from `.dynstr`.
    pub soname: &'data [u8],
    /// The version requirements inside this dependency.
    pub aux: Vec<VersionNaux<'data>>,
}

/// One `Elf_Vernaux`: a version name (e.g. `GLIBC_2.2.5`), its ELF hash and
/// flags, plus the version index that symbols referencing it carry in
/// `.gnu.version`.
#[derive(Clone, Copy)]
pub struct VersionNaux<'data> {
    /// Version name borrowed from `.dynstr`.
    pub name: &'data [u8],
    /// ELF hash of `name` (the GNU/SysV `elf_hash` of the version string).
    pub hash: u32,
    /// `VER_FLG_*` flags.
    pub flags: u16,
    /// The version index assigned to this entry (matches `.gnu.version`).
    pub index: u16,
}

/// One `Elf_Verdef` parsed from `.gnu.version_d`. The first `names` entry is
/// the version name itself; later entries are parent versions it inherits.
#[derive(Clone)]
pub struct VersionDef<'data> {
    /// `VER_FLG_*` flags (`VER_FLG_BASE` for the soname entry).
    pub flags: u16,
    /// The version index this definition assigns (matches `.gnu.version`).
    pub index: u16,
    /// ELF hash of the primary version name.
    pub hash: u32,
    /// The version name (entry 0) and any parent names.
    pub names: Vec<&'data [u8]>,
}

/// One `Elf_Verdaux` entry within a [`VersionDef`].
#[derive(Clone, Copy)]
pub struct VersionDaux<'data> {
    pub name: &'data [u8],
}

/// The parsed contents of `.gnu.version`, `.gnu.version_r` and
/// `.gnu.version_d`, borrowed from the object's mapped bytes. Cheap to copy
/// (the vectors are usually small or empty).
///
/// `versym` is parallel to `.dynsym`: entry `i` is the version index of dynsym
/// entry `i`. Index 0 means local; 1 means global (default version); >= 2 is
/// a real version reference resolved via the needs or defs tables.
#[derive(Clone, Default)]
pub struct VersionTable<'data> {
    /// One `u16` per `.dynsym` entry, in the same order. Empty when the
    /// object carries no `.gnu.version`.
    pub versym: &'data [u16],
    /// VERNEED entries from `.gnu.version_r`. Empty when absent.
    pub needs: Vec<VersionNeed<'data>>,
    /// VERDEF entries from `.gnu.version_d`. Empty when absent.
    pub defs: Vec<VersionDef<'data>>,
}

impl VersionTable<'_> {
    /// The version index of dynsym entry `i`, or `VER_NDX_GLOBAL` when the
    /// object has no `.gnu.version` (the loader treats unversioned symbols as
    /// the dependency's default version).
    pub fn version_of(&self, i: usize) -> u16 {
        self.versym.get(i).copied().unwrap_or(VER_NDX_GLOBAL)
    }

    /// Whether the object carries any version information at all.
    pub const fn is_empty(&self) -> bool {
        self.versym.is_empty() && self.needs.is_empty() && self.defs.is_empty()
    }
}

/// Parses a `.gnu.version_r` payload (a sequence of `Elf_Verneed` records)
/// into [`VersionNeed`] entries. Each record is followed by `vn_cnt`
/// `Elf_Vernaux` entries; `vn_next`/`vna_next` give byte offsets to the next
/// sibling (zero terminates the chain). String offsets resolve against
/// `dynstr`. Returns an empty vector when `payload` is empty.
///
/// Every record a chain link promises must be present: a cut-off payload or
/// a dangling offset is a malformed object, reported rather than silently
/// dropped, because a partial table would resolve some version references
/// and quietly mislabel the rest.
fn parse_verneed<'data>(
    payload: &'data [u8],
    dynstr: &'data [u8],
) -> Result<Vec<VersionNeed<'data>>> {
    let mut out = Vec::new();
    let mut off = 0usize;
    for _ in 0..=u16::MAX {
        let Some(rec) = view_one_at::<Verneed64>(payload, off) else {
            if payload.is_empty() {
                return Ok(out);
            }
            return Err(Error::Format("truncated .gnu.version_r record"));
        };
        let soname = cstr_at(dynstr, rec.vn_file.get());
        let mut aux = Vec::new();
        let mut aoff = off + usize::try_from(rec.vn_aux.get()).unwrap_or(0);
        for _ in 0..=u16::MAX {
            let Some(a) = view_one_at::<Vernaux64>(payload, aoff) else {
                return Err(Error::Format(
                    "truncated .gnu.version_r auxiliary record",
                ));
            };
            aux.push(VersionNaux {
                name: cstr_at(dynstr, a.vna_name.get()),
                hash: a.vna_hash.get(),
                flags: a.vna_flags.get(),
                index: a.vna_other.get(),
            });
            if a.vna_next.get() == 0 {
                break;
            }
            aoff = advance_offset(aoff, a.vna_next.get());
        }
        out.push(VersionNeed { soname, aux });
        if rec.vn_next.get() == 0 {
            return Ok(out);
        }
        off = advance_offset(off, rec.vn_next.get());
    }
    Err(Error::Format("unterminated .gnu.version_r chain"))
}

/// Parses a `.gnu.version_d` payload (a sequence of `Elf_Verdef` records)
/// into [`VersionDef`] entries. Each record is followed by `vd_cnt`
/// `Elf_Verdaux` entries (the first is the version name; later ones name
/// parents). As [`parse_verneed`]: a promised record that is absent is a
/// malformed object, not an empty table.
fn parse_verdef<'data>(
    payload: &'data [u8],
    dynstr: &'data [u8],
) -> Result<Vec<VersionDef<'data>>> {
    let mut out = Vec::new();
    let mut off = 0usize;
    for _ in 0..=u16::MAX {
        let Some(rec) = view_one_at::<Verdef64>(payload, off) else {
            if payload.is_empty() {
                return Ok(out);
            }
            return Err(Error::Format("truncated .gnu.version_d record"));
        };
        let mut names = Vec::new();
        let mut aoff = off + usize::try_from(rec.vd_aux.get()).unwrap_or(0);
        for _ in 0..=u16::MAX {
            let Some(a) = view_one_at::<Verdaux64>(payload, aoff) else {
                return Err(Error::Format(
                    "truncated .gnu.version_d auxiliary record",
                ));
            };
            names.push(cstr_at(dynstr, a.vda_name.get()));
            if a.vda_next.get() == 0 {
                break;
            }
            aoff = advance_offset(aoff, a.vda_next.get());
        }
        out.push(VersionDef {
            flags: rec.vd_flags.get(),
            index: rec.vd_ndx.get(),
            hash: rec.vd_hash.get(),
            names,
        });
        if rec.vd_next.get() == 0 {
            return Ok(out);
        }
        off = advance_offset(off, rec.vd_next.get());
    }
    Err(Error::Format("unterminated .gnu.version_d chain"))
}

/// Adds a record-relative byte offset, saturating rather than wrapping.
fn advance_offset(off: usize, delta: u32) -> usize {
    off.saturating_add(usize::try_from(delta).unwrap_or(0))
}

/// Views one `T` at `offset` within `bytes`, bounds- and alignment-checked.
fn view_one_at<T: Pod>(bytes: &[u8], offset: usize) -> Option<&T> {
    let size = core::mem::size_of::<T>();
    let raw = bytes.get(offset..offset.checked_add(size)?)?;
    bytemuck::try_from_bytes(raw).ok()
}
