//! PE base-relocation table (`.reloc`): the fixups the loader applies when it
//! cannot load the image at its preferred base.
//!
//! A DLL is subject to ASLR: the loader may map it away from
//! [`IMAGE_BASE_DLL_X86_64`], so every absolute virtual address embedded in the
//! image must carry a base-relocation entry. On PE32+ x86-64 the only kind that
//! occurs is [`IMAGE_REL_BASED_DIR64`] (an 8-byte absolute fixup produced by an
//! `IMAGE_REL_AMD64_ADDR64` input relocation); the collector walks every linked
//! input section's relocations and records each `ADDR64` site's final image
//! RVA.
//!
//! The table is a sequence of `ImageBaseRelocation` blocks, one per 4 KiB page
//! that contains at least one fixup, each followed by WORD entries packing a
//! 4-bit type in the high nibble and a 12-bit page offset in the low bits. A
//! block is padded to a 4-byte boundary with [`IMAGE_REL_BASED_ABSOLUTE`]
//! entries, which the loader ignores. The data-directory entry points at the
//! whole table; its size is the sum of every block.

use crate::{
    coff::{
        constants::{
            IMAGE_REL_BASED_ABSOLUTE, IMAGE_REL_BASED_DIR64, SECTION_ALIGNMENT,
        },
        pe::{BaseRelocation, DataDirectory},
    },
    endian::U32,
};

/// The base-relocation table bytes, built from the final image RVAs of every
/// absolute fixup site.
///
/// The bytes do not depend on the `.reloc` section's own placement (entry
/// offsets are page-relative), so no finalisation pass is needed: the writer
/// passes the `.reloc` RVA to [`Self::directory`].
pub struct RelocPlan {
    bytes: Vec<u8>,
}

impl RelocPlan {
    /// Builds the table for `site_rvas` (the absolute image RVAs of the fixup
    /// sites, in any order). RVAs are grouped by their 4 KiB page; each page
    /// becomes one block whose entries are sorted by offset for a reproducible
    /// layout. The result is empty when there are no sites (a position-
    /// independent image with no absolute fixups).
    pub fn new(site_rvas: &[u32]) -> Self {
        let pages = group_pages(site_rvas);
        let mut bytes = Vec::new();
        for (page_base, mut offsets) in pages {
            offsets.sort_unstable();
            offsets.dedup();
            write_block(&mut bytes, page_base, &offsets);
        }
        Self { bytes }
    }

    /// The on-disk byte size, for the layout to reserve `.reloc` space.
    pub fn size(&self) -> u32 {
        u32::try_from(self.bytes.len()).unwrap_or(u32::MAX)
    }

    /// The BASERELOC data-directory entry, given the `.reloc` section RVA.
    pub fn directory(&self, base_rva: u32) -> DataDirectory {
        DataDirectory {
            virtual_address: U32::new(base_rva),
            size: U32::new(u32::try_from(self.bytes.len()).unwrap_or(0)),
        }
    }

    /// The serialised bytes, written verbatim into `.reloc`.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Groups `rvas` by their 4 KiB page, returning `(page_base, offsets)` pairs in
/// ascending page order with each page's offsets unsorted (sorted by the
/// caller).
fn group_pages(rvas: &[u32]) -> Vec<(u32, Vec<u16>)> {
    let mut pages: Vec<(u32, Vec<u16>)> = Vec::new();
    for &rva in rvas {
        let page_base = rva & !(SECTION_ALIGNMENT - 1);
        let offset = u16::try_from(rva & (SECTION_ALIGNMENT - 1)).unwrap_or(0);
        match pages.last_mut() {
            Some((base, offs)) if *base == page_base => offs.push(offset),
            _ => pages.push((page_base, vec![offset])),
        }
    }
    pages
}

/// Writes one relocation block for `page_base` covering `offsets` (the
/// page-relative offsets of the fixup sites). The block is padded to a 4-byte
/// boundary with `IMAGE_REL_BASED_ABSOLUTE` entries.
fn write_block(bytes: &mut Vec<u8>, page_base: u32, offsets: &[u16]) {
    let entry_count = offsets.len();
    let pad = pad_to_dword(entry_count);
    let size = u32::try_from(8 + (entry_count + pad) * 2).unwrap_or(0);
    let header = BaseRelocation {
        virtual_address: U32::new(page_base),
        size_of_block: U32::new(size),
    };
    bytes.extend_from_slice(bytemuck::bytes_of(&header));
    for &off in offsets {
        let entry = IMAGE_REL_BASED_DIR64 << 12 | (off & 0x0FFF);
        bytes.extend_from_slice(&entry.to_le_bytes());
    }
    for _ in 0..pad {
        let entry = IMAGE_REL_BASED_ABSOLUTE << 12;
        bytes.extend_from_slice(&entry.to_le_bytes());
    }
}

/// The number of padding entries needed so a block with `entry_count` real
/// entries lands on a 4-byte boundary (the header is 8 bytes, entries are
/// 2 bytes, so the whole block must be a multiple of 4).
fn pad_to_dword(entry_count: usize) -> usize {
    entry_count % 2
}
