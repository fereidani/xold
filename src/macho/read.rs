//! Zero-copy reader for 64-bit Mach-O objects.
//!
//! Parsing and structural validation are delegated to the `object` crate
//! (gimli-rs/object): it checks the magic, endianness, load-command table and
//! every table offset it returns. The endian-typed structures `object` exposes
//! (`Section64<Endian>`, `Nlist64<Endian>`, `Relocation<Endian>`) are then
//! resolved once at this boundary into the plain records in [`super::structs`],
//! so downstream code reads native `u64`/`u32` fields with no dependency on
//! `object`'s endian machinery. This mirrors how the ELF reader re-views
//! `object`'s bytes as plain `Shdr64` structs.
//!
//! The reader covers the 64-bit little-endian Mach-O objects produced by
//! `clang --target=x86_64-apple-darwin` and `--target=arm64-apple-darwin`.
//! 32-bit and fat containers are rejected; the bytes are otherwise viewed
//! directly, so nothing here copies the input (section views hold `&'data`
//! references into the mapping).

use object::{
    Endianness, macho,
    read::{
        Object, ObjectSection, StringTable,
        macho::{MachHeader, MachOFile64, Nlist, Section},
    },
};

use crate::{
    error::{Error, Result},
    macho::{
        constants::{
            CPU_TYPE_ARM64, CPU_TYPE_X86_64, FAT_CIGAM, FAT_CIGAM_64,
            FAT_MAGIC, FAT_MAGIC_64, MH_CIGAM, MH_CIGAM_64, MH_MAGIC,
            MH_MAGIC_64, MH_OBJECT,
        },
        structs::{MachReloc, MachSection, MachSymbol},
    },
    util::trim_nul,
};

/// A parsed 64-bit Mach-O object, borrowing the mapped input bytes.
///
/// The file owns no copy of its data: every accessor returns references into
/// the original mapping. It wraps `object`'s validated [`MachOFile64`], which
/// holds the parsed load-command table and section/symbol locations; those in
/// turn point into the `&'data` byte mapping.
pub struct MachOFile<'data> {
    inner: MachOFile64<'data>,
}

/// A symbol table view: the `nlist_64` array plus its string table.
///
/// Both pieces borrow `&'data` directly from the mapping (not from the
/// [`MachOFile`]), so a cloned view outlives the borrow of the file that
/// produced it. This mirrors the ELF reader's `SymbolTable<'data>`.
#[derive(Clone, Copy)]
pub struct MachSymbolTable<'data> {
    syms: &'data [macho::Nlist64<Endianness>],
    strings: StringTable<'data, &'data [u8]>,
    endian: Endianness,
}

impl<'data> MachOFile<'data> {
    /// Parses `bytes` as a 64-bit Mach-O object.
    ///
    /// The fat and 32-bit Mach-O magics are rejected up front so the error is
    /// precise; `object` then validates the header, load commands and every
    /// table bounds check for the 64-bit case.
    pub fn parse(bytes: &'data [u8]) -> Result<Self> {
        reject_non_object(bytes)?;
        let inner = MachOFile64::parse(bytes)
            .map_err(|_| Error::Format("invalid Mach-O header"))?;
        Ok(Self { inner })
    }

    /// The cpu type (`cputype`): `CPU_TYPE_X86_64` or `CPU_TYPE_ARM64` for the
    /// objects this reader targets.
    pub fn cpu_type(&self) -> u32 {
        self.inner.macho_header().cputype(self.inner.endian())
    }

    /// The cpu subtype (`cpusubtype`).
    pub fn cpu_subtype(&self) -> u32 {
        self.inner.macho_header().cpusubtype(self.inner.endian())
    }

    /// The `filetype` field (`MH_OBJECT` for a relocatable object).
    pub fn filetype(&self) -> u32 {
        self.inner.macho_header().filetype(self.inner.endian())
    }

    /// Whether this is a relocatable object (`MH_OBJECT`).
    pub fn is_object(&self) -> bool {
        self.filetype() == MH_OBJECT
    }

    /// Whether the object targets `x86_64`.
    pub fn is_x86_64(&self) -> bool {
        self.cpu_type() == CPU_TYPE_X86_64
    }

    /// Whether the object targets `arm64`.
    pub fn is_arm64(&self) -> bool {
        self.cpu_type() == CPU_TYPE_ARM64
    }

    /// The entry point. For an executable this is the `LC_MAIN` `entryoff` (a
    /// file offset into `__TEXT`) or the `LC_UNIXTHREAD` entry register value;
    /// for a relocatable object it is zero.
    pub fn entry(&self) -> u64 {
        self.inner.entry()
    }

    /// The input-frame address of the section with 1-based ordinal `n_sect`.
    ///
    /// An object lays its sections out sequentially in one frame, and a
    /// symbol's `n_value` is an address in that frame, so placing the symbol
    /// means subtracting this base first (`MachO/InputFiles.cpp` in lld does
    /// the same normalization).
    pub fn section_addr(&self, n_sect: u8) -> Option<u64> {
        let endian = self.inner.endian();
        let zero_based = usize::from(n_sect).checked_sub(1)?;
        self.inner
            .sections()
            .nth(zero_based)
            .map(|s| s.macho_section().addr(endian))
    }

    /// Every section in file order, carrying its decoded relocations.
    ///
    /// Sections arrive in `LC_SEGMENT_64` order; the 1-based `index` matches
    /// the `n_sect` ordinals used by the symbol table. `S_ZEROFILL` sections
    /// yield an empty `data` slice.
    pub fn sections(&self) -> Vec<MachSection<'data>> {
        let endian = self.inner.endian();
        let cputype = self.cpu_type();
        let mut out = Vec::new();
        for (i, section) in self.inner.sections().enumerate() {
            let raw = section.macho_section();
            let relocs = section.macho_relocations().map_or_else(
                |_| Vec::new(),
                |e| decode_relocations(endian, cputype, e),
            );
            let data = section.data().unwrap_or(&[]);
            out.push(MachSection {
                index: u32::try_from(i + 1).unwrap_or(u32::MAX),
                sectname: trim_nul(raw.name()),
                segname: trim_nul(raw.segment_name()),
                addr: raw.addr(endian),
                size: raw.size(endian),
                align: raw.align(endian),
                flags: raw.flags(endian),
                data,
                relocations: relocs,
            });
        }
        out
    }

    /// The symbol table, or an empty view if the object has no `LC_SYMTAB`.
    pub fn symbols(&self) -> MachSymbolTable<'data> {
        let table = self.inner.macho_symbol_table();
        MachSymbolTable {
            syms: table.iter().as_slice(),
            strings: table.strings(),
            endian: self.inner.endian(),
        }
    }
}

impl<'data> MachSymbolTable<'data> {
    /// The number of `nlist` entries (including debug stab entries).
    pub fn len(&self) -> usize {
        self.syms.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.syms.is_empty()
    }

    /// The symbol at `index`, decoded on the spot.
    ///
    /// Resolving one symbol used to walk the iterator into a `Vec` and index
    /// that, which allocated the whole table per lookup and made a file's
    /// resolution quadratic in its symbol count. The entries are fixed width,
    /// so the one asked for is a slice index away.
    pub fn nth(&self, index: usize) -> Option<MachSymbol<'data>> {
        let n = self.syms.get(index)?;
        Some(decode_symbol(n, self.strings, self.endian))
    }

    /// Iterates the resolved symbols. The yielded [`MachSymbol`] values borrow
    /// `&'data` (the byte mapping), so they outlive the borrow of this table,
    /// mirroring the ELF reader's `SymbolTable::iter`.
    pub fn iter(&self) -> MachSymbolIter<'data> {
        MachSymbolIter {
            syms: self.syms.iter(),
            strings: self.strings,
            endian: self.endian,
        }
    }
}

/// Iterator over [`MachSymbol`], borrowing the symbol table view.
pub struct MachSymbolIter<'a> {
    syms: core::slice::Iter<'a, macho::Nlist64<Endianness>>,
    strings: StringTable<'a, &'a [u8]>,
    endian: Endianness,
}

impl<'a> Iterator for MachSymbolIter<'a> {
    type Item = MachSymbol<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.syms.next()?;
        Some(decode_symbol(n, self.strings, self.endian))
    }
}

/// Decodes one `nlist_64` entry against the table's string table and endian.
fn decode_symbol<'data>(
    n: &'data macho::Nlist64<Endianness>,
    strings: StringTable<'data, &'data [u8]>,
    endian: Endianness,
) -> MachSymbol<'data> {
    MachSymbol {
        name: n.name(endian, strings).unwrap_or(&[]),
        n_strx: n.n_strx(endian),
        n_type: n.n_type(),
        n_sect: n.n_sect(),
        n_desc: n.n_desc(endian),
        n_value: n.n_value(endian),
    }
}

impl<'data> IntoIterator for &'data MachSymbolTable<'data> {
    type Item = MachSymbol<'data>;
    type IntoIter = MachSymbolIter<'data>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// --- parsing helpers ------------------------------------------------------

/// Rejects fat containers and 32-bit Mach-O up front, so a non-64-bit object
/// yields a precise error rather than a generic header failure. A 64-bit magic
/// (either byte order) is allowed through to `object`.
///
/// Both spellings of every magic are listed. That matters most for the fat
/// container: its header is big-endian on disk by definition, so it always
/// arrives here byte-swapped, and matching only `FAT_MAGIC` would report the
/// one shape this function names first as "not a Mach-O object".
fn reject_non_object(bytes: &[u8]) -> Result<()> {
    match magic_u32(bytes) {
        MH_MAGIC_64 | MH_CIGAM_64 => Ok(()),
        MH_MAGIC | MH_CIGAM => {
            Err(Error::Format("32-bit Mach-O is not supported"))
        }
        FAT_MAGIC | FAT_MAGIC_64 | FAT_CIGAM | FAT_CIGAM_64 => {
            Err(Error::Format("fat Mach-O is not supported"))
        }
        _ => Err(Error::Format("not a Mach-O object")),
    }
}

/// Reads the first four bytes of `bytes` as a little-endian `u32`. Mach-O magic
/// values are recognised in either byte order, so the read order is not
/// significant here.
fn magic_u32(bytes: &[u8]) -> u32 {
    let mut wide = [0u8; 4];
    let n = wide.len().min(bytes.len());
    wide[..n].copy_from_slice(&bytes[..n]);
    u32::from_le_bytes(wide)
}

/// Decodes the raw `Relocation` entries for a section into [`MachReloc`]
/// records, honouring the scattered-encoding flag for legacy 32-bit sections.
fn decode_relocations(
    endian: Endianness,
    cputype: u32,
    entries: &[macho::Relocation<Endianness>],
) -> Vec<MachReloc> {
    let mut out = Vec::with_capacity(entries.len());
    for reloc in entries {
        if reloc.r_scattered(endian, cputype) {
            let info = reloc.scattered_info(endian);
            out.push(MachReloc {
                r_address: info.r_address,
                r_symbolnum: info.r_value,
                r_pcrel: info.r_pcrel,
                r_length: info.r_length,
                r_extern: false,
                r_type: info.r_type,
                r_scattered: true,
            });
        } else {
            let info = reloc.info(endian);
            out.push(MachReloc {
                r_address: info.r_address,
                r_symbolnum: info.r_symbolnum,
                r_pcrel: info.r_pcrel,
                r_length: info.r_length,
                r_extern: info.r_extern,
                r_type: info.r_type,
                r_scattered: false,
            });
        }
    }
    out
}
