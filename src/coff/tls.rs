//! PE static thread-local storage: the `.tls` template and the
//! `ImageTlsDirectory64` that lets the loader allocate one per-thread copy of
//! every `__declspec(thread)` variable.
//!
//! clang `-msvc` emits each thread-local variable's initial bytes into a
//! `.tls$*` section. The linker aggregates those into the output `.tls`
//! template; the loader copies that template (plus `SizeOfZeroFill` zero
//! bytes) into a fresh per-thread block at thread start and records the
//! module's slot index in `_tls_index`. Code loads the thread pointer from
//! `gs:0x58` (`ThreadLocalStoragePointer`), indexes it by `_tls_index` to find
//! this module's block, and adds the variable's offset within the template
//! (resolved by an `IMAGE_REL_AMD64_SECREL` fixup) to read or write its slot.
//!
//! [`TlsPlan`] lays out the `.tlsdir` trailer that holds the
//! `ImageTlsDirectory64`, the null callback array and the `_tls_index` dword.
//! Offsets within the section are fixed at construction; `finalize` stamps the
//! absolute VAs once the layout places `.tls` and `.tlsdir`. The TLS
//! data-directory entry is published for the writer; the index slot VA is
//! exposed so the symbol resolver can bind the undefined `_tls_index` symbol
//! the compiler emits.

use crate::{
    coff::{
        CoffFile, CoffSection,
        constants::{
            IMAGE_SCN_ALIGN_16BYTES, IMAGE_SCN_ALIGN_MASK,
            IMAGE_SIZEOF_TLS_DIRECTORY64,
        },
        layout::TLS_DIR_SIZE,
        pe::{DataDirectory, TlsDirectory64},
    },
    endian::{U32, U64},
    util::trim_nul,
};

/// The offset of the `_tls_index` dword within `.tlsdir` (it follows the
/// 40-byte directory header and the 8-byte null callback pointer).
const INDEX_OFF: u32 = TLS_STRUCT_SIZE + 8;
/// The offset of the null-terminated callback array within `.tlsdir`.
const CALLBACKS_OFF: u32 = TLS_STRUCT_SIZE;

/// `ImageTlsDirectory64` extent as a `u32`, used for in-section offsets.
/// The source constant is a compile-time `40`, so the truncation cast is
/// total.
#[allow(clippy::cast_possible_truncation)]
pub const TLS_STRUCT_SIZE: u32 = IMAGE_SIZEOF_TLS_DIRECTORY64 as u32;

/// The laid-out TLS directory and its supporting slots.
///
/// `template_size`/`zero_fill`/`align` describe the `.tls` template the
/// loader copies per thread; `template_size` is the section's laid-out
/// virtual size, gaps included, not a sum of raw member sizes. The directory
/// fields that depend on the final section RVAs are stamped by
/// [`Self::finalize`].
pub struct TlsPlan {
    template_size: u32,
    zero_fill: u32,
    align_chars: u32,
    bytes: Vec<u8>,
    directory: DataDirectory,
    template_rva: u32,
    index_rva: u32,
}

/// Whether any input declares TLS storage (a `.tls$*` section). The writer
/// uses this to decide whether to reserve `.tlsdir` and resolve `_tls_index`.
pub fn has_tls(inputs: &[CoffFile<'_>]) -> bool {
    inputs
        .iter()
        .any(|f| f.sections().iter().any(is_tls_section))
}

/// Whether a section contributes to the `.tls` template.
fn is_tls_section(section: &CoffSection<'_>) -> bool {
    trim_nul(section.name).starts_with(b".tls$")
}

impl TlsPlan {
    /// Builds the plan from the aggregated `.tls` inputs: the template's raw
    /// size, the bytes of `.bss`-style TLS that follow it as zero-fill, and
    /// the template alignment as a section-characteristics value. The trailer
    /// bytes (directory, null callbacks, `_tls_index`) are zero-initialised;
    /// `finalize` stamps them.
    pub fn new(
        inputs: &[CoffFile<'_>],
        template_size: u32,
        zero_fill: u32,
    ) -> Self {
        let align_chars = alignment_of(inputs);
        let bytes = vec![0u8; usize::try_from(TLS_DIR_SIZE).unwrap_or(0)];
        Self {
            template_size,
            zero_fill,
            align_chars,
            bytes,
            directory: DataDirectory::default(),
            template_rva: 0,
            index_rva: 0,
        }
    }

    /// The on-disk byte size of the `.tlsdir` trailer, for the layout to
    /// reserve space.
    pub const fn size(&self) -> u32 {
        TLS_DIR_SIZE
    }

    /// Stamps the directory's absolute VAs and the TLS data-directory entry.
    /// `template_rva` is the `.tls` section RVA; `meta_rva` is the `.tlsdir`
    /// section RVA; `image_base` is the preferred load address.
    pub fn finalize(
        &mut self,
        template_rva: u32,
        meta_rva: u32,
        image_base: u64,
    ) {
        self.template_rva = template_rva;
        self.index_rva = meta_rva.wrapping_add(INDEX_OFF);
        let start = image_base.wrapping_add(u64::from(template_rva));
        let end = start.wrapping_add(u64::from(self.template_size));
        let index_va = image_base.wrapping_add(u64::from(self.index_rva));
        let callbacks_va = image_base
            .wrapping_add(u64::from(meta_rva.wrapping_add(CALLBACKS_OFF)));
        let dir = TlsDirectory64 {
            start_address_of_raw_data: U64::new(start),
            end_address_of_raw_data: U64::new(end),
            address_of_index: U64::new(index_va),
            address_of_call_backs: U64::new(callbacks_va),
            size_of_zero_fill: U32::new(self.zero_fill),
            characteristics: U32::new(self.align_chars),
        };
        let src = bytemuck::bytes_of(&dir);
        if let Some(slot) = self.bytes.get_mut(..src.len()) {
            slot.copy_from_slice(src);
        }
        self.directory = DataDirectory {
            virtual_address: U32::new(meta_rva),
            // The directory entry covers just the `ImageTlsDirectory64`
            // struct (40 bytes); the trailing callbacks slot and `_tls_index`
            // dword live in the same `.tlsdir` section but are pointed to by
            // the struct, not part of the directory proper.
            size: U32::new(TLS_STRUCT_SIZE),
        };
    }

    /// The absolute VA of the `_tls_index` dword, for the symbol resolver.
    pub fn index_va(&self, image_base: u64) -> u64 {
        image_base.wrapping_add(u64::from(self.index_rva))
    }

    /// The TLS data-directory entry (set by `finalize`).
    pub fn directory(&self) -> DataDirectory {
        self.directory
    }

    /// The serialised trailer bytes (directory + callbacks + `_tls_index`).
    /// Written verbatim into `.tlsdir`.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The `.tls` template RVA (set by `finalize`).
    pub const fn template_rva(&self) -> u32 {
        self.template_rva
    }
}

/// The `.tlsdir` field RVAs that hold an absolute VA, given the section's own
/// RVA.
///
/// The four pointers in `ImageTlsDirectory64` are stamped from the preferred
/// base, so a rebased image reads them at the wrong address unless each one
/// carries a `DIR64` fixup. In lld the directory is the CRT's `_tls_used`
/// chunk and its own `ADDR64` relocations flow through the ordinary pass;
/// here it is synthesized, so the sites are stated.
pub fn directory_fixup_rvas(meta_rva: u32) -> [u32; 4] {
    [
        meta_rva,
        meta_rva.wrapping_add(8),
        meta_rva.wrapping_add(16),
        meta_rva.wrapping_add(24),
    ]
}

/// The maximum `IMAGE_SCN_ALIGN_*` value across the `.tls$*` inputs, clamped
/// to at least 16-byte alignment so the loader's per-thread copy of the
/// template is naturally aligned. Defaults to 16-byte when no input states an
/// alignment.
fn alignment_of(inputs: &[CoffFile<'_>]) -> u32 {
    inputs
        .iter()
        .flat_map(CoffFile::sections)
        .filter(|section| is_tls_section(section))
        .map(|section| section.characteristics & IMAGE_SCN_ALIGN_MASK)
        .max()
        .unwrap_or(IMAGE_SCN_ALIGN_16BYTES)
}
