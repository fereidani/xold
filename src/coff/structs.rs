//! Plain, endian-resolved view records for COFF data.
//!
//! The `object` crate exposes COFF structures as endian-typed types
//! (`ImageSectionHeader`, `ImageSymbol`, `ImageRelocation`), where every
//! integer field is wrapped in a `U16`/`U32` that must be resolved against an
//! [`Endianness`]. COFF is always little-endian, so the reader resolves the
//! fields once at the boundary and yields the plain records defined here, so
//! the rest of xold reads native `u32`/`u16` fields with no dependency on
//! `object`'s endian machinery. This mirrors how the Mach-O reader resolves
//! `Section64<Endian>` into [`crate::macho::MachSection`] at the boundary.
//!
//! Each record borrows `&'data` from the original byte mapping (section names,
//! payloads and symbol names); scalar fields are copied. Relocations carry no
//! borrowed data, so [`CoffReloc`] is `Copy`.

use crate::coff::constants::{
    IMAGE_SYM_ABSOLUTE, IMAGE_SYM_CLASS_EXTERNAL,
    IMAGE_SYM_CLASS_WEAK_EXTERNAL, IMAGE_SYM_UNDEFINED,
};

/// A COFF section, endian-resolved.
///
/// Names longer than eight characters are spelt as `/N` offsets into the
/// symbol string table; the reader resolves those, so `name` is the final name
/// view. `data` is the raw payload (empty for `.bss`).
#[derive(Clone)]
pub struct CoffSection<'data> {
    /// 1-based section ordinal, matching section numbers in the symbol table.
    pub index: u32,
    /// Resolved section name (inline short name or string-table name).
    pub name: &'data [u8],
    /// Virtual size (the size in the image; for an object this is the section
    /// size).
    pub virtual_size: u32,
    /// Virtual address (RVA base; for an object, section-relative zero).
    pub virtual_address: u32,
    /// Size of the raw data on disk.
    pub size_of_raw_data: u32,
    /// File offset of the raw data.
    pub pointer_to_raw_data: u32,
    /// Number of relocation entries.
    pub number_of_relocations: u32,
    /// Raw `characteristics` (`IMAGE_SCN_*` flags).
    pub characteristics: u32,
    /// Payload bytes; empty for `.bss` and other no-file-space sections.
    pub data: &'data [u8],
    /// Decoded relocation entries for this section.
    pub relocations: Vec<CoffReloc>,
}

/// A decoded COFF relocation entry.
///
/// `virtual_address` is the offset of the fixup site within the owning
/// section. `symbol_table_index` selects a symbol in the file's symbol table.
/// `typ` is the architecture-specific type number (see `reloc::coff_x86_64`
/// and `reloc::coff_i386`).
#[derive(Clone, Copy, Debug)]
pub struct CoffReloc {
    /// Offset within the section to the fixup site.
    pub virtual_address: u32,
    /// Index of the referenced symbol in the symbol table.
    pub symbol_table_index: u32,
    /// Architecture-specific relocation type (`IMAGE_REL_AMD64_*` /
    /// `IMAGE_REL_I386_*`).
    pub typ: u16,
}

/// A COFF symbol (`ImageSymbol`), endian-resolved.
///
/// `name` is NUL-terminated and borrows the string table (or the inline short
/// name). `section_number` uses the COFF encoding: positive for a real
/// section, and [`IMAGE_SYM_UNDEFINED`]/[`IMAGE_SYM_ABSOLUTE`]/
/// [`IMAGE_SYM_DEBUG`] for the reserved values.
#[derive(Clone, Copy)]
pub struct CoffSymbol<'data> {
    /// Index in the symbol table (skips auxiliary records).
    pub index: u32,
    /// NUL-terminated name bytes (inline short name or string-table view).
    pub name: &'data [u8],
    /// `value` (offset within the section, or the common size for commons).
    pub value: u32,
    /// Section number; negative for the reserved `IMAGE_SYM_*` values.
    pub section_number: i32,
    /// Raw `typ` (base type and derived type, packed).
    pub typ: u16,
    /// Storage class (`IMAGE_SYM_CLASS_*`).
    pub storage_class: u8,
    /// Count of auxiliary records that follow this entry.
    pub number_of_aux_symbols: u8,
    /// Auxiliary record, when this entry carries one the reader exposes.
    pub aux: SymbolAux,
}

/// The auxiliary record attached to a symbol, if any, decoded to a plain form.
///
/// Only the two aux kinds the linker uses are exposed: section definitions
/// (COMDAT) and function definitions. Other aux records are dropped at the
/// boundary; the primary entry still carries `number_of_aux_symbols` so the
/// caller can skip them.
#[derive(Clone, Copy, Debug, Default)]
pub enum SymbolAux {
    /// No auxiliary record of interest.
    #[default]
    None,
    /// Section-definition record (auxiliary record 5). Carries COMDAT
    /// metadata.
    Section(SectionAux),
    /// Function-definition record (auxiliary record 4). Carries the code size.
    Function(FunctionAux),
    /// Weak-external record. Carries the index of the default definition.
    Weak(WeakAux),
}

/// Decoded `ImageAuxSymbolWeak`: the fallback definition and how to find it.
#[derive(Clone, Copy, Debug, Default)]
pub struct WeakAux {
    /// Raw symbol-table index of the definition the reference falls back to
    /// when nothing defines the name for real.
    pub tag_index: u32,
    /// `IMAGE_WEAK_EXTERN_SEARCH_*`: how far the search for a real definition
    /// is meant to reach.
    pub search_type: u32,
}

/// Decoded `ImageAuxSymbolSection`: COMDAT selection and length.
#[derive(Clone, Copy, Debug, Default)]
pub struct SectionAux {
    /// Section length (bytes), from the auxiliary record.
    pub length: u32,
    /// COMDAT selection type (0 if not COMDAT).
    pub selection: u8,
    /// Associative-section number (0 if not associative).
    pub number: u32,
}

/// Decoded `ImageAuxSymbolFunction`: the function's total code size.
#[derive(Clone, Copy, Debug, Default)]
pub struct FunctionAux {
    /// Total size of the function body, in bytes.
    pub total_size: u32,
}

impl CoffSymbol<'_> {
    /// Whether this is an external symbol visible across object boundaries.
    pub const fn is_external(&self) -> bool {
        self.storage_class == IMAGE_SYM_CLASS_EXTERNAL
    }

    /// Whether this is a weak external.
    pub const fn is_weak(&self) -> bool {
        self.storage_class == IMAGE_SYM_CLASS_WEAK_EXTERNAL
    }

    /// Undefined reference: section number is `IMAGE_SYM_UNDEFINED`.
    pub const fn is_undefined(&self) -> bool {
        self.is_undefined_section() && self.value == 0
    }

    /// Common (tentative) definition: undefined with a non-zero value that
    /// gives the byte size.
    pub const fn is_common(&self) -> bool {
        self.is_undefined_section() && self.value != 0
    }

    /// Absolute symbol, not tied to any section.
    pub const fn is_absolute(&self) -> bool {
        self.section_number == IMAGE_SYM_ABSOLUTE
    }

    /// Undefined, so either a plain reference or a tentative definition.
    const fn is_undefined_section(&self) -> bool {
        self.is_external() && self.section_number == IMAGE_SYM_UNDEFINED
    }

    /// The index of a weak external's default definition, if this is one.
    pub const fn weak_tag_index(&self) -> Option<u32> {
        match self.aux {
            SymbolAux::Weak(w) => Some(w.tag_index),
            _ => None,
        }
    }
}
