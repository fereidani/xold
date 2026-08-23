//! Aggregates non-allocated DWARF `.debug_*` sections for the writer.
//!
//! A compiler emitting `-g` produces `.debug_info`, `.debug_line`,
//! `.debug_abbrev`, `.debug_str`, ...: `SHT_PROGBITS` sections without
//! `SHF_ALLOC`. The runtime loader ignores them, but a debugger reads them
//! from the file to map addresses back to source.
//! [`crate::linker::Context::assign_sections`] places only `SHF_ALLOC`
//! sections, so this module owns the collection path for the debug ones.
//!
//! Each output debug section is the byte concatenation of its input members
//! in input order. The member's offset within the aggregated section is
//! recorded on the member so the layout can stamp it into `sec_vaddr`,
//! which lets the arch-neutral relocation driver resolve a `.rela.debug_*`
//! section-symbol reference to the member's base (DWARF offsets are
//! section-relative, so the section symbol resolves to the contribution's
//! start within the output section).
//!
//! `.debug_str` and `.debug_line_str` carry `SHF_MERGE|SHF_STRINGS` and are
//! deduplication candidates; simple concatenation is correct (just larger)
//! and dedup is deferred.

use crate::{elf::Shdr64, util::align_up};

/// True for DWARF debug section names that xold preserves for debuggers.
///
/// Matches any name beginning with `.debug`: `.debug_info`, `.debug_line`,
/// `.debug_abbrev`, `.debug_str`, `.debug_str_offsets`, `.debug_addr`,
/// `.debug_line_str`, `.debug_ranges`, `.debug_loc`, `.debug_frame`,
/// `.debug_aranges`, `.debug_macro`, and the rest. GNU compressed
/// `.zdebug_*` sections are not matched; see [`ZDEBUG_PREFIX`] for why
/// they refuse the link instead of vanishing.
pub fn is_debug_section(name: &[u8]) -> bool {
    name.starts_with(DEBUG_PREFIX)
}

/// The name prefix that marks a DWARF section.
pub const DEBUG_PREFIX: &[u8] = b".debug";

/// The legacy prefix that marks a zlib-compressed DWARF section: the same
/// deflate stream the `SHF_COMPRESSED` spelling carries, signalled by the
/// name alone and with no `Elf64_Chdr` in front of it.
pub const ZDEBUG_PREFIX: &[u8] = b".zdebug";

/// One input contribution to an aggregated debug output section.
#[derive(Copy, Clone, Debug)]
pub struct DebugMember {
    pub file: usize,
    pub section: u16,
    pub size: u64,
    /// This member's byte offset within the aggregated output section. The
    /// layout stamps this into `sec_vaddr` so a relocation against the
    /// member's section symbol resolves to the contribution's base.
    pub out_offset: u64,
}

/// One aggregated debug output section (for example `.debug_info`), holding
/// same-named input sections concatenated in input order.
#[derive(Debug)]
pub struct DebugSection {
    pub name: Vec<u8>,
    pub sh_type: u32,
    /// Raw `sh_flags` from the first contributor. `SHF_ALLOC` is never set
    /// (the collector rejects allocated sections); `SHF_MERGE|SHF_STRINGS`
    /// is preserved on `.debug_str` / `.debug_line_str`, whose content the
    /// merge pass deduplicates.
    pub sh_flags: u64,
    /// Raw `sh_entsize` from the first contributor, carried out with the flags
    /// it belongs to.
    ///
    /// `SHF_MERGE` says the content is a run of fixed-size entries and
    /// `sh_entsize` is how large they are; for `SHF_STRINGS` it is the width
    /// of a character, which is one. The two are one statement, so
    /// dropping the size while keeping the flags leaves a section
    /// describing entries of no length -- which is what `.debug_str` was
    /// emitted as, where lld writes 1.
    pub entsize: u64,
    pub align: u64,
    pub size: u64,
    pub members: Vec<DebugMember>,
}

/// All aggregated debug output sections, in first-seen order.
#[derive(Default, Debug)]
pub struct DebugSections {
    pub sections: Vec<DebugSection>,
}

impl DebugSections {
    /// An empty collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether no debug sections were collected.
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    /// Appends one input debug section to its output section, allocating the
    /// member's offset within the aggregated output and growing the output
    /// size. Same-named inputs concatenate in input order; alignment between
    /// members follows each member's `sh_addralign`.
    pub fn add(
        &mut self,
        file: usize,
        section: u16,
        shdr: &Shdr64,
        name: &[u8],
    ) {
        let align = shdr.sh_addralign.get().max(1);
        let idx = if let Some(i) =
            self.sections.iter().position(|s| s.name == name)
        {
            i
        } else {
            self.sections.push(DebugSection {
                name: name.to_vec(),
                sh_type: shdr.sh_type.get(),
                sh_flags: shdr.sh_flags.get(),
                entsize: shdr.sh_entsize.get(),
                align: 1,
                size: 0,
                members: Vec::new(),
            });
            self.sections.len() - 1
        };
        let out = &mut self.sections[idx];
        let cursor = align_up(out.size, align);
        out.members.push(DebugMember {
            file,
            section,
            size: shdr.sh_size.get(),
            out_offset: cursor,
        });
        out.size = cursor.saturating_add(shdr.sh_size.get());
        if align > out.align {
            out.align = align;
        }
    }

    /// Re-sizes members and lays the sections out again.
    ///
    /// `size_of` reports a member's new size, or `None` to leave it as it is.
    /// Merging `.debug_str` collapses every contributor but one to nothing and
    /// grows that one to the deduplicated pool, which moves every following
    /// member and shrinks the section.
    pub fn relayout(&mut self, size_of: impl Fn(usize, u16) -> Option<u64>) {
        for out in &mut self.sections {
            let mut cursor = 0u64;
            for m in &mut out.members {
                if let Some(size) = size_of(m.file, m.section) {
                    m.size = size;
                }
                m.out_offset = cursor;
                cursor = cursor.saturating_add(m.size);
            }
            out.size = cursor;
        }
    }
}
