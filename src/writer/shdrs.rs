//! Section-header (Shdr) table construction: builds the planned output section
//! headers and materialises them into their on-disk `Shdr64` form.
//!
//! The allocated headers come straight from [`sect::TABLE`], the same ordered
//! list layout numbered the sections from, so a symbol's stamped `shndx`
//! always selects the header emitted at that position. The non-allocated
//! `.debug_*` headers follow (file-only regions a debugger reads), and the
//! trailing `.symtab`, `.strtab` and `.shstrtab` headers close the table;
//! their indices are computed from the final length, so the symtab's `sh_link`
//! and the ELF `e_shstrndx` stay correct however many dynamic or debug
//! sections exist.

use super::{SYM_SIZE, dwarf::DebugPlan, symtab::StrTab};
use crate::{
    debug::DebugSection,
    elf::{
        Shdr64,
        constants::{SHT_STRTAB, SHT_SYMTAB},
    },
    endian::{U32, U64},
    layout::{
        Layout,
        sect::{self, Info, Link, Sect, Spec},
    },
};

/// One planned output section header, before string-table offsets are final.
#[derive(Default)]
pub(super) struct PlanShdr {
    pub(super) name_off: u32,
    pub(super) type_: u32,
    pub(super) flags: u64,
    pub(super) addr: u64,
    pub(super) offset: u64,
    pub(super) size: u64,
    pub(super) align: u64,
    pub(super) link: u32,
    pub(super) info: u32,
    pub(super) entsize: u64,
}

/// File offsets and sizes of the non-allocated trailing sections.
pub(super) struct TableOffsets {
    pub(super) symtab_off: u64,
    pub(super) symtab_size: u64,
    /// `.symtab`'s `sh_info`: the index of its first non-local entry.
    pub(super) symtab_first_global: u32,
    pub(super) strtab_off: u64,
    pub(super) strtab_size: u64,
    pub(super) shstrtab_off: u64,
}

/// Builds the section-header table. See the module docs for the order.
pub(super) fn build_shdrs(
    layout: &Layout,
    shstrtab: &mut StrTab,
    off: &TableOffsets,
    debug_plan: &DebugPlan,
    debug_sections: &[DebugSection],
    symbols: bool,
) -> Vec<PlanShdr> {
    let mut out = Vec::with_capacity(sect::TABLE.len().saturating_add(12));
    out.push(PlanShdr::default());
    for (sect, spec) in &sect::TABLE {
        // The row exists wherever placement stamped an index, not wherever
        // the region still holds bytes. A synthetic region can tighten to
        // nothing after placement -- the sizing probe counts the data
        // relocations of sections `--gc-sections` later drops -- and its
        // header row was still counted when the indices were stamped, by
        // `assign_shndx` here and by the image plan's own walk. Gating on
        // the size would delete the row and shift every later index under
        // one that names a different section.
        if layout.shndx(*sect) == 0 {
            continue;
        }
        let region = layout.region(*sect);
        let align = layout.sect_align(*sect, spec);
        // The ELF spec requires `sh_addr` to be congruent to zero modulo
        // `sh_addralign`. Every region is placed at the alignment its header
        // reports, so this holds by construction; the assertion is what keeps
        // a later placement change from quietly parting the two.
        debug_assert!(
            region.vaddr.is_multiple_of(align),
            "a region's address is a multiple of the alignment it reports"
        );
        out.push(PlanShdr {
            name_off: shstrtab.intern(spec.name),
            type_: spec.sh_type,
            flags: spec.flags,
            addr: region.vaddr,
            offset: region.offset,
            size: region.size,
            align,
            link: sh_link(layout, spec),
            info: sh_info(layout, spec),
            entsize: spec.entsize,
        });
    }
    push_debug_shdrs(debug_sections, debug_plan, shstrtab, &mut out);
    push_trailing_shdrs(&mut out, shstrtab, off, symbols);
    out
}

/// Resolves a header's `sh_link` to the section index it cross-references.
fn sh_link(layout: &Layout, spec: &Spec) -> u32 {
    match spec.link {
        Link::None => 0,
        Link::Dynsym => u32::from(layout.shndx(Sect::Dynsym)),
        Link::Dynstr => u32::from(layout.shndx(Sect::Dynstr)),
    }
}

/// Resolves a header's `sh_info`. `.dynsym` reports the index of its first
/// non-local symbol (it carries only the null local, so 1); `.gnu.version_r`
/// reports its `Elf_Verneed` record count; `.rela.plt` names the section its
/// entries patch.
fn sh_info(layout: &Layout, spec: &Spec) -> u32 {
    match spec.info {
        Info::Zero => 0,
        Info::FirstGlobal => 1,
        Info::VerneedCount => {
            layout.dynamic.as_ref().map_or(0, |p| p.verneed_count)
        }
        Info::GotPlt => u32::from(layout.shndx(Sect::GotPlt)),
    }
}

/// Pushes the trailing `.symtab`, `.strtab` and `.shstrtab` headers. The
/// symtab's `sh_link` is the strtab index, which sits at `out.len() + 1`;
/// `shstrtab` is the final header.
///
/// The symtab's `sh_info` is the index of its first non-local entry, which the
/// table's builder counted. Every reader splits the table there -- `readelf`
/// and `nm` to label the bindings, and a linker reading the object back to
/// find the globals -- so it has to be the real count and not a constant.
fn push_trailing_shdrs(
    out: &mut Vec<PlanShdr>,
    shstrtab: &mut StrTab,
    off: &TableOffsets,
    symbols: bool,
) {
    if symbols {
        let strtab_idx = u32::try_from(out.len() + 1).unwrap_or(0);
        out.push(PlanShdr {
            name_off: shstrtab.intern(b".symtab"),
            type_: SHT_SYMTAB,
            offset: off.symtab_off,
            size: off.symtab_size,
            align: 8,
            link: strtab_idx,
            info: off.symtab_first_global,
            entsize: SYM_SIZE,
            ..PlanShdr::default()
        });
        let strtab_name = shstrtab.intern(b".strtab");
        out.push(strtab_shdr(strtab_name, off.strtab_off, off.strtab_size));
    }
    // Interning `.shstrtab` is the last thing to touch the table, so its size
    // is final only after the call.
    let shstrtab_name = shstrtab.intern(b".shstrtab");
    let size = u64::try_from(shstrtab.bytes.len()).unwrap_or(0);
    out.push(strtab_shdr(shstrtab_name, off.shstrtab_off, size));
}

/// A non-allocated string-table header (`.strtab`, `.shstrtab`).
fn strtab_shdr(name_off: u32, offset: u64, size: u64) -> PlanShdr {
    PlanShdr {
        name_off,
        type_: SHT_STRTAB,
        offset,
        size,
        align: 1,
        ..PlanShdr::default()
    }
}

/// Pushes section headers for the aggregated `.debug_*` output sections.
/// Non-allocated (`sh_addr = 0`): file-only regions a debugger reads via
/// `sh_offset`. Each header preserves the input `sh_type`, `sh_flags` and
/// `sh_entsize` (so `.debug_str` keeps `SHF_MERGE|SHF_STRINGS` *and* the entry
/// width those flags describe), minus `SHF_ALLOC` (the collector rejects
/// allocated sections).
fn push_debug_shdrs(
    debug_sections: &[DebugSection],
    plan: &DebugPlan,
    shstrtab: &mut StrTab,
    out: &mut Vec<PlanShdr>,
) {
    for (i, sec) in debug_sections.iter().enumerate() {
        let Some(region) = plan.get(i) else {
            continue;
        };
        out.push(PlanShdr {
            name_off: shstrtab.intern(&sec.name),
            type_: sec.sh_type,
            flags: sec.sh_flags,
            offset: region.offset,
            size: region.size,
            align: sec.align.max(1),
            entsize: sec.entsize,
            ..PlanShdr::default()
        });
    }
}

/// Materialises a planned section header into its on-disk form.
pub(super) fn materialise(p: &PlanShdr) -> Shdr64 {
    Shdr64 {
        sh_name: U32::new(p.name_off),
        sh_type: U32::new(p.type_),
        sh_flags: U64::new(p.flags),
        sh_addr: U64::new(p.addr),
        sh_offset: U64::new(p.offset),
        sh_size: U64::new(p.size),
        sh_link: U32::new(p.link),
        sh_info: U32::new(p.info),
        sh_addralign: U64::new(p.align),
        sh_entsize: U64::new(p.entsize),
    }
}
