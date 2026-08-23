//! Zero-copy reader for COFF objects and PE images.
//!
//! Parsing and structural validation are delegated to the `object` crate
//! (gimli-rs/object): it checks the machine magic, locates the section table,
//! the symbol table and the string table, and bounds-checks every offset it
//! returns. The endian-typed structures `object` exposes
//! (`ImageSectionHeader`, `ImageSymbol`, `ImageRelocation`) are then resolved
//! once at this boundary into the plain records in [`super::structs`], so
//! downstream code reads native `u32`/`u16` fields with no dependency on
//! `object`'s endian machinery. This mirrors how the Mach-O reader resolves
//! `Section64<Endian>` into [`crate::macho::MachSection`] at the boundary.
//!
//! [`CoffFile`] covers the bare COFF objects produced by
//! `clang --target=x86_64-pc-windows-msvc`, `-gnu`, and `i686-pc-windows-msvc`
//! (the linker's input). [`PeImage`] covers PE images (`.exe`/`.dll`): a COFF
//! object wrapped in a DOS stub plus an optional header. It is the
//! writer-facing counterpart used to round-trip and verify the PE output.
//! Nothing here copies the input: section and symbol views hold `&'data`
//! references into the mapping; the only allocation is the `Vec` of plain
//! records returned at the boundary.

use std::sync::OnceLock;

use object::{
    LittleEndian, pe,
    read::{
        Object,
        coff::{
            CoffFile as ObjCoffFile, CoffHeader, ImageSymbol,
            SymbolTable as ObjSymbolTable,
        },
        pe::{ImageNtHeaders, ImageOptionalHeader, PeFile64},
    },
};

use crate::{
    coff::{
        constants::{
            IMAGE_FILE_MACHINE_AMD64, IMAGE_FILE_MACHINE_ARM64,
            IMAGE_FILE_MACHINE_I386, IMAGE_SYM_CLASS_WEAK_EXTERNAL,
        },
        structs::{
            CoffReloc, CoffSection, CoffSymbol, FunctionAux, SectionAux,
            SymbolAux, WeakAux,
        },
    },
    error::{Error, Result},
    util::trim_nul,
};

/// The little-endian byte order COFF always uses. The `pe` on-disk structs
/// are endian-typed against `LittleEndian` (not the `Endianness` enum the
/// Mach-O reader uses), so this constant matches their `get` calls.
const LE: LittleEndian = LittleEndian;

/// A parsed COFF object, borrowing the mapped input bytes.
///
/// The file owns no copy of its data: every accessor returns references into
/// the original mapping. It wraps `object`'s validated [`ObjCoffFile`], which
/// holds the parsed header, section table and symbol/string-table locations;
/// `bytes` is retained so the section-data and relocation accessors (which
/// take the file data) re-borrow the same mapping the caller opened.
pub struct CoffFile<'data> {
    inner: ObjCoffFile<'data>,
    bytes: &'data [u8],
    sections: OnceLock<Vec<CoffSection<'data>>>,
    symbols: OnceLock<CoffSymbolTable<'data>>,
}

/// A symbol table view: the resolved plain records.
///
/// Each [`CoffSymbol`] borrows `&'data` (the byte mapping for names), so a
/// cloned record outlives the borrow of the file that produced it. The table
/// is built eagerly at the boundary: COFF symbol tables are modest (hundreds
/// to thousands of entries), and the plain records let downstream code read
/// native fields with no `object` dependency.
#[derive(Clone)]
pub struct CoffSymbolTable<'data> {
    syms: Vec<CoffSymbol<'data>>,
}

impl<'data> CoffFile<'data> {
    /// Parses `bytes` as a bare COFF object.
    ///
    /// `object` validates the header, section table, symbol table and string
    /// table, and bounds-checks every offset. PE images (`MZ`-prefixed) are
    /// rejected here with a precise error; the COFF writer that consumes a PE
    /// image lands later.
    pub fn parse(bytes: &'data [u8]) -> Result<Self> {
        reject_pe_image(bytes)?;
        let inner = ObjCoffFile::parse(bytes)
            .map_err(|_| Error::Format("invalid COFF header"))?;
        Ok(Self {
            inner,
            bytes,
            sections: OnceLock::new(),
            symbols: OnceLock::new(),
        })
    }

    /// The machine type (`IMAGE_FILE_MACHINE_*`).
    pub fn machine(&self) -> u16 {
        self.inner.coff_header().machine()
    }

    /// Whether the object targets `x86_64`.
    pub fn is_x86_64(&self) -> bool {
        self.machine() == IMAGE_FILE_MACHINE_AMD64
    }

    /// Whether the object targets i386.
    pub fn is_i386(&self) -> bool {
        self.machine() == IMAGE_FILE_MACHINE_I386
    }

    /// Whether the object targets ARM64.
    pub fn is_arm64(&self) -> bool {
        self.machine() == IMAGE_FILE_MACHINE_ARM64
    }

    /// The raw COFF file header characteristics (`IMAGE_FILE_*` flags).
    pub fn characteristics(&self) -> u16 {
        self.inner.coff_header().characteristics()
    }

    /// The number of sections in the section table.
    pub fn number_of_sections(&self) -> u32 {
        self.inner.coff_header().number_of_sections()
    }

    /// The section at `ordinal`, numbered from one the way the symbol table
    /// spells section numbers.
    pub fn section_at(&self, ordinal: u32) -> Option<&CoffSection<'data>> {
        let at = usize::try_from(ordinal).ok()?.checked_sub(1)?;
        self.sections().get(at)
    }

    /// Every section in file order, carrying its decoded relocations.
    ///
    /// Sections arrive in table order; the 1-based `index` matches the section
    /// numbers used by the symbol table. `.bss`-style sections yield an empty
    /// `data` slice.
    /// Decoded once and reused: the placement, relocation and TLS passes each
    /// walk every section of every input, and re-decoding per call made the
    /// cost quadratic in passes.
    pub fn sections(&self) -> &[CoffSection<'data>] {
        self.sections.get_or_init(|| self.decode_sections())
    }

    fn decode_sections(&self) -> Vec<CoffSection<'data>> {
        let table = self.inner.coff_section_table();
        let strings = self.inner.coff_symbol_table().strings();
        let mut out = Vec::with_capacity(table.len());
        for (i, raw) in table.iter().enumerate() {
            let name = raw.name(strings).unwrap_or_else(|_| raw.raw_name());
            let data = raw.coff_data(self.bytes).unwrap_or(&[]);
            let relocations = raw
                .coff_relocations(self.bytes)
                .map(decode_relocations)
                .unwrap_or_default();
            out.push(CoffSection {
                index: u32::try_from(i + 1).unwrap_or(u32::MAX),
                name,
                virtual_size: raw.virtual_size.get(LE),
                virtual_address: raw.virtual_address.get(LE),
                size_of_raw_data: raw.size_of_raw_data.get(LE),
                pointer_to_raw_data: raw.pointer_to_raw_data.get(LE),
                number_of_relocations: u32::from(
                    raw.number_of_relocations.get(LE),
                ),
                characteristics: raw.characteristics.get(LE),
                data,
                relocations,
            });
        }
        out
    }

    /// The decoded symbol table, with auxiliary records resolved for the entry
    /// kinds the linker uses (section and function definitions).
    /// Decoded once and reused, for the same reason as [`Self::sections`].
    pub fn symbols(&self) -> &CoffSymbolTable<'data> {
        self.symbols.get_or_init(|| self.decode_symbols())
    }

    fn decode_symbols(&self) -> CoffSymbolTable<'data> {
        let symtab = self.inner.coff_symbol_table();
        let strings = symtab.strings();
        let mut out = Vec::with_capacity(symtab.len());
        for (idx, sym) in symtab.iter() {
            let name = sym.name(strings).unwrap_or(&[]);
            out.push(CoffSymbol {
                index: u32::try_from(idx.0).unwrap_or(u32::MAX),
                name,
                value: sym.value(),
                section_number: sym.section_number(),
                typ: sym.typ(),
                storage_class: sym.storage_class(),
                number_of_aux_symbols: sym.number_of_aux_symbols(),
                aux: decode_aux(symtab, idx.0, sym),
            });
        }
        CoffSymbolTable { syms: out }
    }
}

impl<'data> CoffSymbolTable<'data> {
    /// The number of symbol entries (auxiliary records excluded).
    pub fn len(&self) -> usize {
        self.syms.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.syms.is_empty()
    }

    /// Iterates the resolved symbols.
    pub fn iter(&self) -> core::slice::Iter<'_, CoffSymbol<'data>> {
        self.syms.iter()
    }
}

impl<'data> IntoIterator for &'data CoffSymbolTable<'data> {
    type Item = &'data CoffSymbol<'data>;
    type IntoIter = core::slice::Iter<'data, CoffSymbol<'data>>;

    fn into_iter(self) -> Self::IntoIter {
        self.syms.iter()
    }
}

// --- parsing helpers ------------------------------------------------------

/// Rejects PE images up front so the error is precise. A PE image begins with
/// the `MZ` DOS stub; the COFF object reader does not unwrap the DOS + optional
/// header layer (that is the writer's job, later).
fn reject_pe_image(bytes: &[u8]) -> Result<()> {
    if bytes.len() >= 2 && bytes[0] == b'M' && bytes[1] == b'Z' {
        return Err(Error::Format(
            "PE image not supported by the COFF object reader",
        ));
    }
    Ok(())
}

/// Decodes the raw `ImageRelocation` entries for a section into [`CoffReloc`]
/// records.
fn decode_relocations(entries: &[pe::ImageRelocation]) -> Vec<CoffReloc> {
    entries
        .iter()
        .map(|r| CoffReloc {
            virtual_address: r.virtual_address.get(LE),
            symbol_table_index: r.symbol_table_index.get(LE),
            typ: r.typ.get(LE),
        })
        .collect()
}

/// Resolves the auxiliary record for `sym`, if it carries one the linker uses.
///
/// Section-definition records (auxiliary kind 5) carry COMDAT metadata;
/// function-definition records (auxiliary kind 4) carry the code size; a weak
/// external's record names the definition the reference falls back to. Other
/// auxiliary kinds are dropped at the boundary; the primary entry still
/// reports `number_of_aux_symbols` so callers can account for them.
fn decode_aux<'data>(
    symtab: &ObjSymbolTable<'data, &'data [u8], pe::ImageFileHeader>,
    index: usize,
    sym: &pe::ImageSymbol,
) -> SymbolAux {
    if sym.has_aux_section()
        && let Ok(aux) = symtab.aux_section(object::SymbolIndex(index))
    {
        let number = u32::from(aux.number.get(LE))
            | (u32::from(aux.high_number.get(LE)) << 16);
        return SymbolAux::Section(SectionAux {
            length: aux.length.get(LE),
            selection: aux.selection,
            number,
        });
    }
    if sym.storage_class() == IMAGE_SYM_CLASS_WEAK_EXTERNAL
        && sym.number_of_aux_symbols() > 0
        && let Ok(aux) = symtab.aux_weak_external(object::SymbolIndex(index))
    {
        return SymbolAux::Weak(WeakAux {
            tag_index: aux.weak_default_sym_index.get(LE),
            search_type: aux.weak_search_type.get(LE),
        });
    }
    if sym.has_aux_function()
        && let Ok(aux) = symtab.aux_function(object::SymbolIndex(index))
    {
        return SymbolAux::Function(FunctionAux {
            total_size: aux.total_size.get(LE),
        });
    }
    SymbolAux::None
}

// --- PE image reader ------------------------------------------------------

/// A parsed PE32+ image, borrowing the mapped input bytes.
///
/// Wraps `object`'s validated [`PeFile64`], which locates the DOS header, the
/// NT headers, the data directories, the section table and the (deprecated)
/// COFF symbol table. Used to round-trip the PE writer's output: it exposes
/// the machine, the optional-header fields (image base, entry, alignment),
/// the section table and the import table.
pub struct PeImage<'data> {
    inner: PeFile64<'data>,
    bytes: &'data [u8],
}

/// A PE section header resolved to plain native fields.
///
/// Carries the section's raw-data payload sliced from the image. The name is
/// trimmed at the first NUL of the 8-byte short-name cell (long `/N` names are
/// not resolved here).
#[derive(Clone)]
pub struct PeSection<'data> {
    pub name: &'data [u8],
    pub virtual_size: u32,
    pub virtual_address: u32,
    pub size_of_raw_data: u32,
    pub pointer_to_raw_data: u32,
    pub characteristics: u32,
    pub data: &'data [u8],
}

impl<'data> PeImage<'data> {
    /// Parses `bytes` as a PE32+ image.
    pub fn parse(bytes: &'data [u8]) -> Result<Self> {
        let inner = PeFile64::parse(bytes)
            .map_err(|_| Error::Format("invalid PE32+ image"))?;
        Ok(Self { inner, bytes })
    }

    /// The machine type (`IMAGE_FILE_MACHINE_*`).
    pub fn machine(&self) -> u16 {
        self.inner.nt_headers().file_header().machine.get(LE)
    }

    /// Whether the image targets `x86_64`.
    pub fn is_x86_64(&self) -> bool {
        self.machine() == IMAGE_FILE_MACHINE_AMD64
    }

    /// The raw file-header characteristics.
    pub fn characteristics(&self) -> u16 {
        self.inner
            .nt_headers()
            .file_header()
            .characteristics
            .get(LE)
    }

    /// The optional-header magic (`0x20b` for PE32+).
    pub fn magic(&self) -> u16 {
        self.inner.nt_headers().optional_header().magic()
    }

    /// The preferred load address of the image.
    pub fn image_base(&self) -> u64 {
        self.inner.nt_headers().optional_header().image_base()
    }

    /// The RVA of the entry point.
    pub fn address_of_entry_point(&self) -> u32 {
        self.inner
            .nt_headers()
            .optional_header()
            .address_of_entry_point()
    }

    /// The virtual-address alignment of sections.
    pub fn section_alignment(&self) -> u32 {
        self.inner
            .nt_headers()
            .optional_header()
            .section_alignment()
    }

    /// The file-offset alignment of section raw data.
    pub fn file_alignment(&self) -> u32 {
        self.inner.nt_headers().optional_header().file_alignment()
    }

    /// The total virtual size of the image (sections aligned up).
    pub fn size_of_image(&self) -> u32 {
        self.inner.nt_headers().optional_header().size_of_image()
    }

    /// The combined size of the DOS + PE headers and the section table, rounded
    /// up to `file_alignment`.
    pub fn size_of_headers(&self) -> u32 {
        self.inner.nt_headers().optional_header().size_of_headers()
    }

    /// The Windows subsystem (`IMAGE_SUBSYSTEM_*`).
    pub fn subsystem(&self) -> u16 {
        self.inner.nt_headers().optional_header().subsystem()
    }

    /// One data-directory entry `(rva, size)` by index, if present.
    pub fn data_directory(&self, id: usize) -> Option<(u32, u32)> {
        self.inner
            .data_directory(id)
            .map(|d| (d.virtual_address.get(LE), d.size.get(LE)))
    }

    /// Every section in table order with its raw-data payload.
    pub fn sections(&self) -> Vec<PeSection<'data>> {
        let table = self.inner.section_table();
        let mut out = Vec::with_capacity(table.len());
        for raw in table.iter() {
            let name = trim_nul(&raw.name);
            let raw_size = raw.size_of_raw_data.get(LE);
            let raw_ptr = raw.pointer_to_raw_data.get(LE);
            let start = usize::try_from(raw_ptr).unwrap_or(usize::MAX);
            let end = start
                .checked_add(usize::try_from(raw_size).unwrap_or(0))
                .unwrap_or(self.bytes.len());
            let data = self.bytes.get(start..end).unwrap_or(&[]);
            out.push(PeSection {
                name,
                virtual_size: raw.virtual_size.get(LE),
                virtual_address: raw.virtual_address.get(LE),
                size_of_raw_data: raw_size,
                pointer_to_raw_data: raw_ptr,
                characteristics: raw.characteristics.get(LE),
                data,
            });
        }
        out
    }

    /// The imported `(dll, function-name)` pairs, walked via the `Object`
    /// trait's import table. Empty for a no-import image.
    pub fn imports(&self) -> Vec<(Vec<u8>, Vec<u8>)> {
        Object::imports(&self.inner)
            .unwrap_or_default()
            .into_iter()
            .map(|imp| (imp.library().to_vec(), imp.name().to_vec()))
            .collect()
    }
}
