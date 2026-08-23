//! Output image layout: the bridge between symbol resolution and the writer.
//!
//! Layout turns the grouped input sections ([`crate::output::OutputSections`])
//! into placed regions with file offsets and virtual addresses, allocates the
//! GOT and common storage that the relocation scan asked for, and resolves
//! every relocation symbol to a concrete address. The result is a
//! [`FileResolver`] per input that implements [`crate::reloc::Resolver`], so
//! the arch-neutral [`crate::reloc::apply`] driver can patch bytes without
//! knowing where any address came from.
//!
//! The image uses an identity map: a byte's virtual address is the load base
//! plus its file offset. This keeps every program header congruent
//! (`p_vaddr - p_offset` is the page-aligned base) without per-segment padding
//! arithmetic. A static link needs no PLT: `PLT[sym]` collapses to the symbol
//! address, so no PLT section is allocated.
//!
//! [`build`] runs the passes in order and owns the [`Layout`] they fill; each
//! pass lives in its own submodule. `sect` is the ordered section table,
//! `scan` classifies the relocations, `place` assigns offsets and addresses,
//! `addr` resolves every symbol against them, `got` holds the GOT key types
//! and slot lookups, `exports` collects the output symbol table, and
//! `resolver` is what the writer applies relocations through.

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use self::{
    addr::{definition_addr, resolve_addresses},
    exports::collect_exports,
    got::{fill_got, got_slot, plt_slot},
    resolver::{SymFlags, SymRows},
};
use crate::{
    buildid::BuildId,
    dynamic::{DynConfig, DynamicPlan, LinkMode},
    elf::constants::{SHN_UNDEF, STB_WEAK},
    error::{Error, Result},
    input::InputFile,
    linker::Context,
    output::OutKind,
    reloc::Target,
    startstop::Bounds,
    symbol::{SymbolId, SymbolKind},
    tls::TlsBlock,
    util::PAGE,
};

mod addr;
mod exports;
mod got;
mod place;
mod resolver;
mod scan;
pub mod sect;

pub use exports::{CopyName, CopySlot, Export};
pub use got::{GotKey, GotKind, GotOwner};
pub use resolver::FileResolver;
pub use sect::Sect;

/// Default non-PIE load base for an `ET_EXEC`, matching the historical System V
/// layout used by `ld`.
const BASE: u64 = 0x0040_0000;
/// Load base for a position-independent `ET_DYN` shared object.
const SHARED_BASE: u64 = 0;

/// The largest page size the image must remain loadable under.
///
/// Two `PT_LOAD` segments with different permissions may not share a system
/// page, or the kernel maps one page twice and the second mapping's
/// permissions win for the whole of it. The boundary between them, and the
/// `p_align` that declares it, therefore has to be the largest page the
/// target's kernels are configured with -- not the smallest.
///
/// `AArch64` is the target where the two differ. Linux builds for it with 4K,
/// 16K (Asahi, much of Android) and 64K (RHEL and Fedora on ARM) pages, so an
/// image aligned to 4K loses execute permission on the tail of its `.text`
/// the moment the read-write segment lands in the same 16K or 64K page. A
/// small binary can lose it on the whole of `.text` and die at the entry
/// point. lld sets `defaultMaxPageSize = 65536` for exactly this target
/// (`Arch/AArch64.cpp:131`) and leaves the others at 4096.
pub(crate) const fn max_page_size(target: Target) -> u64 {
    match target {
        Target::AArch64 => 0x1_0000,
        Target::X86_64 | Target::Riscv64 => PAGE,
    }
}
/// On-disk size of the ELF64 header.
const EHDR_SIZE: u64 = 64;
/// On-disk size of one program header.
const PHDR_SIZE: u64 = 56;
/// Bytes in one GOT entry.
const GOT_ENTRY: u64 = 8;

/// A placed region of the output image: file offset, virtual address, size.
#[derive(Copy, Clone, Debug, Default)]
pub struct Region {
    /// Byte offset within the output file.
    pub offset: u64,
    /// Virtual address (`BASE + offset` for loaded content).
    pub vaddr: u64,
    /// Region size in bytes.
    pub size: u64,
}

impl Region {
    /// One past the last byte of the region in the file.
    pub const fn end(self) -> u64 {
        self.offset + self.size
    }
}

/// The fully laid-out image, ready for the writer.
pub struct Layout {
    entry: u64,
    phdr_count: u16,
    /// The link target; stored on the layout so the PLT/GOT.PLT emitters and
    /// the per-symbol PLT-address helpers can branch on architecture without
    /// a second parameter.
    pub target: Target,
    /// The ELF header's `e_flags`: which variant of the architecture the image
    /// is built for, folded from the inputs by
    /// [`crate::linker::derive_eflags`].
    pub e_flags: u32,
    /// Static vs shared: chooses `ET_EXEC`/prefilled GOT vs `ET_DYN`/dynamic.
    pub mode: LinkMode,
    /// Whether the image is position independent (`ET_DYN`). `Static` is never
    /// PIE; `Shared` always is; `DynExec` is PIE unless it copy-relocates an
    /// absolute data import or stores a link-time address in a slot no load
    /// base can shift -- a reference to one of its own symbols, or to an
    /// import a copy slot or a canonical PLT stub stands in for -- in which
    /// case a fixed base (`ET_EXEC`) is required so the absolute address
    /// resolves correctly at runtime.
    pub pie: bool,
    /// `-z stack-size=N`: what `PT_GNU_STACK` asks the loader to reserve.
    /// Zero means the system default, which is what every link that does not
    /// name the option gets.
    pub stack_size: u64,
    /// Load base: `BASE` for static or a fixed-base executable, zero for a
    /// position-independent shared object.
    base: u64,
    /// `.init` and `.fini`: one function apiece, assembled from the C
    /// runtime's fragments, which must stay contiguous and in link order.
    init: Region,
    fini: Region,
    text: Region,
    rodata: Region,
    /// The `.eh_frame` region: file-backed unwind data, allocated read-only.
    /// Kept as its own section so `.eh_frame_hdr` can index a contiguous range
    /// and `PT_GNU_EH_FRAME` can locate it.
    eh_frame: Region,
    /// The `.eh_frame_hdr` region: the binary-search table the runtime
    /// unwinder uses to find an FDE for a thrown PC. Covered by
    /// `PT_GNU_EH_FRAME`.
    eh_frame_hdr: Region,
    got: Region,
    /// The `.data.rel.ro` region: `const` objects whose initialisers needed a
    /// relocation. Placed inside the RELRO run, so the loader can make them
    /// read-only again once it has applied those relocations.
    data_rel_ro: Region,
    data: Region,
    bss: Region,
    /// The `.tdata` (initialised TLS) region; file-backed.
    tdata: Region,
    /// The `.tbss` (zero-init TLS) region; memory-only. Its `offset` is the
    /// notional position right after `.tdata`; it occupies no file bytes.
    tbss: Region,
    /// The `.preinit_array` region: file-backed slots holding the function
    /// pointers the runtime calls before any constructor. `DT_PREINIT_ARRAY`
    /// points here when non-empty.
    preinit_array: Region,
    /// The `.init_array` region: file-backed slots holding constructor
    /// function pointers. `DT_INIT_ARRAY` points here when non-empty.
    init_array: Region,
    /// The `.fini_array` region: file-backed slots holding destructor
    /// function pointers. `DT_FINI_ARRAY` points here when non-empty.
    fini_array: Region,
    /// Maximum alignment across `.tdata`/`.tbss` (`PT_TLS` `p_align`).
    pub tls_align: u64,
    /// The `.interp` bytes (NUL-terminated interpreter path), owned so the
    /// writer can emit them without borrowing the driver's config. Empty in a
    /// non-executable link.
    pub interp: Vec<u8>,
    /// The `.interp` placement. Only meaningful when `interp` is non-empty.
    interp_region: Region,
    /// The aggregated allocated notes, covered by `PT_NOTE`.
    pub note: Region,
    /// The synthesised `NT_GNU_BUILD_ID` note, empty when `--build-id` was
    /// not asked for.
    pub build_id: Region,
    /// The `.plt` region (lazy-binding stubs). Empty unless PLT entries were
    /// allocated for imported symbols.
    plt: Region,
    /// The `.got.plt` region (the GOT used by the PLT: three reserved entries
    /// plus one slot per PLT entry).
    got_plt: Region,
    /// The `.rela.plt` region (one `R_X86_64_JUMP_SLOT` per PLT entry).
    rela_plt: Region,
    /// Section-header index of every emitted region, indexed by [`Sect`]. Zero
    /// means the region was empty and got no header; index zero is the null
    /// header, so the value doubles as "absent".
    shndx: [u16; sect::TABLE.len()],
    /// The alignment placement put each measured region at, indexed by
    /// [`Sect`]. Meaningful only where [`sect::Spec::align`] is zero; the
    /// synthetic tables state theirs there instead.
    measured_align: [u64; sect::TABLE.len()],
    /// The placed bounds of every `__start_`/`__stop_` run, indexed by run.
    /// Empty unless the link bounds a C-identifier section name.
    start_stop: Vec<Bounds>,
    /// Per file, per input section: the placed virtual address (0 if the
    /// section was not allocated).
    sec_vaddr: Vec<Vec<u64>>,
    /// Per file, per input section: the output section header index.
    sec_shndx: Vec<Vec<u16>>,
    /// Per file: the resolved per-symbol values the relocation resolver hands
    /// out. One entry per input file, each owning disjoint rows, so the
    /// resolution pass fills them in parallel.
    sym: Vec<SymRows>,
    /// GOT entry contents, in allocation order.
    pub got_values: Vec<u64>,
    /// The address of this image's own module GOT entry, or 0 when it has
    /// none. Resolved once the GOT is placed, because the resolver each
    /// section copy builds would otherwise probe for it.
    tls_index: u64,
    /// Indirect functions this image defines and something references, in
    /// first-seen order. Each has a stub in `.plt` and a slot the runtime
    /// fills by calling the resolver the symbol names.
    pub ifunc: Vec<SymbolId>,
    /// The resolver address of each entry in `ifunc`, captured before those
    /// symbols were pointed at their stubs.
    ifunc_resolvers: Vec<u64>,
    /// `ifunc` paired with `ifunc_resolvers`, so the PLT emitter can ask both
    /// "is this an indirect function" and "what was its resolver" without a
    /// scan per entry. Built by [`Layout::index_ifuncs`] once both lists are
    /// final; it decides identity only and is never iterated.
    ifunc_index: FxHashMap<SymbolId, u64>,
    /// GOT keys in allocation order, parallel to `got_values`. Exposed so the
    /// shared-link emitter can describe each slot with a dynamic relocation.
    pub got_keys: Vec<GotKey>,
    /// Whether any relocation reads a symbol's `st_size`. Set by the scan, so
    /// the per-symbol size rows are only allocated for a link that needs them.
    pub needs_sym_sizes: bool,
    /// Per file, the sections carrying at least one pointer-width absolute
    /// relocation, in section-index order. Collected by the scan, which walks
    /// every relocation anyway, so the `.rela.dyn` emitter visits exactly
    /// these instead of re-walking every section of every file.
    pub abs_candidates: Vec<Vec<u16>>,
    /// Resolved symbols for the output symbol table.
    pub exports: Vec<Export>,
    /// The undefined symbols a surviving relocation still names. These alone
    /// earn `SHN_UNDEF` rows: a reference every lowering consumed (the
    /// relaxed `__tls_get_addr` call) names nothing the program still
    /// reaches, matching lld's `used` gate on `.symtab` inclusion.
    pub referenced: FxHashSet<SymbolId>,
    /// GOT keys in allocation order, used to rebuild per-file GOT addresses.
    got_index: FxHashMap<GotKey, u32>,
    /// PLT keys (one per imported symbol referenced through a call), in
    /// allocation order. Parallel to the PLT user entries `PLT[1..]`.
    pub plt_keys: Vec<GotKey>,
    /// Symbol identity -> PLT entry index (0-based into `plt_keys`).
    plt_index: FxHashMap<GotKey, u32>,
    /// Data symbols selected for an `R_X86_64_COPY` relocation, in scan order.
    /// `addr` is zero until `.bss` placement assigns the copy slot.
    pub copy_slots: Vec<CopySlot>,
    /// Function symbols selected for a canonical PLT entry: an executable's
    /// absolute reference to a shared-library function (for example the
    /// `__gxx_personality_v0` pointer in a `.eh_frame` CIE) cannot be resolved
    /// at link time and cannot be left for the loader in a read-only section.
    /// Instead the linker allocates a PLT entry at a fixed address and
    /// resolves every reference to it; a paired `JUMP_SLOT` lets the
    /// loader bind the real function at runtime. Mirrors lld's `NEEDS_PLT`
    /// path for an absolute reloc against a shared `STT_FUNC` in an
    /// executable.
    pub canonical_plt: FxHashSet<SymbolId>,
    /// Whether any allocated section contains a non-PIC absolute reference
    /// (`R_X86_64_64`/`32`/`32S`/`16`) whose stored value this link fixes: a
    /// reference to a symbol the executable defines itself, or one to an
    /// import bound to a copy slot or a canonical PLT stub. Such a reference
    /// stores a full virtual address that is wrong under ASLR and cannot be
    /// loader-relocated in a 32-bit slot, so the link drops to a fixed base
    /// (`ET_EXEC`) and resolves it at link time.
    pub has_absolute_refs: bool,
    /// The synthetic dynamic sections. `None` in a static link.
    pub dynamic: Option<DynamicPlan>,
    /// The PLT synthetic sections (`.plt`, `.got.plt`, `.rela.plt`). `None`
    /// when no PLT entries were allocated.
    pub plt_plan: Option<crate::plt::PltPlan>,
}

impl Layout {
    /// The entry-point virtual address.
    pub const fn entry(&self) -> u64 {
        self.entry
    }

    /// The number of program headers the image emits.
    pub const fn phdr_count(&self) -> u16 {
        self.phdr_count
    }

    /// The static vs shared link flavour.
    pub const fn mode(&self) -> LinkMode {
        self.mode
    }

    /// Whether the output is position independent (`ET_DYN`). A dynamic
    /// executable drops to a fixed base (`ET_EXEC`) when it copy-relocates an
    /// absolute data import, or when a non-PIC absolute reference stores an
    /// address this link chose -- to one of its own symbols, or to an import a
    /// copy slot or a canonical PLT stub stands in for -- so that the address
    /// resolves correctly at runtime.
    pub const fn is_pie(&self) -> bool {
        self.pie
    }

    /// The load base shared by every segment.
    pub const fn base(&self) -> u64 {
        self.base
    }

    /// Whether a `PT_INTERP` segment must be emitted (dynamic executable with
    /// a non-empty interpreter path).
    pub fn has_interp(&self) -> bool {
        !self.interp.is_empty()
    }

    /// Every resolved symbol a copy slot defines, paired with the slot address
    /// it resolves to.
    ///
    /// A slot defines its aliases as well as the symbol that selected it, so
    /// this is the whole set that must resolve to the copy rather than to a
    /// runtime import. Address resolution and the dynamic emitters both walk
    /// it, so neither can disagree about which symbols the copy owns.
    pub fn copy_bindings(&self) -> impl Iterator<Item = (SymbolId, u64)> {
        self.copy_slots.iter().flat_map(|slot| {
            slot.names
                .iter()
                .filter_map(move |n| Some((n.id?, slot.addr)))
        })
    }

    /// The placement of one region of the output image.
    ///
    /// The synthetic dynamic sections live on the [`DynamicPlan`], so this
    /// hands out a zeroed region for them in a static link.
    pub fn region(&self, sect: Sect) -> Region {
        match sect {
            Sect::Interp => self.interp_region,
            Sect::Note => self.note,
            Sect::BuildId => self.build_id,
            Sect::Init => self.init,
            Sect::Fini => self.fini,
            Sect::Text => self.text,
            Sect::Plt => self.plt,
            Sect::Rodata => self.rodata,
            Sect::EhFrame => self.eh_frame,
            Sect::EhFrameHdr => self.eh_frame_hdr,
            Sect::RelaPlt => self.rela_plt,
            Sect::GotPlt => self.got_plt,
            Sect::Got => self.got,
            Sect::DataRelRo => self.data_rel_ro,
            Sect::Data => self.data,
            Sect::PreinitArray => self.preinit_array,
            Sect::InitArray => self.init_array,
            Sect::FiniArray => self.fini_array,
            Sect::Tdata => self.tdata,
            Sect::Tbss => self.tbss,
            Sect::Bss => self.bss,
            dynamic => self
                .dynamic
                .as_ref()
                .map_or_else(Region::default, |p| p.regions.get(dynamic)),
        }
    }

    /// The section-header index assigned to a region, or zero if the region is
    /// empty and so has no header.
    pub fn shndx(&self, sect: Sect) -> u16 {
        self.shndx.get(sect as usize).copied().unwrap_or_default()
    }

    /// The placed bounds of one `__start_`/`__stop_` run.
    ///
    /// A run nothing contributed to reports an empty range at address zero,
    /// which is what every absent bound in [`crate::defsym`] reports.
    pub fn run_bounds(&self, run: usize) -> Bounds {
        self.start_stop.get(run).copied().unwrap_or_default()
    }

    /// Records the section-header index of a region. Called once per region by
    /// [`place::assign_shndx`], in [`sect::TABLE`] order.
    pub(crate) fn set_shndx(&mut self, sect: Sect, index: u16) {
        if let Some(slot) = self.shndx.get_mut(sect as usize) {
            *slot = index;
        }
    }

    /// The `sh_addralign` of a region.
    ///
    /// A synthetic table holds fixed-width words, so its alignment is a
    /// constant and [`sect::TABLE`] states it. A region aggregated from input
    /// members takes the largest alignment its members ask for, which only
    /// measurement knows, and its table entry is zero to say so.
    ///
    /// The two must describe the same placement: the ELF spec requires
    /// `sh_addr` to be congruent to zero modulo `sh_addralign`, so a header
    /// claiming an alignment stricter than the one the placement cursor
    /// honoured is malformed. [`crate::writer`] asserts the congruence over
    /// every region it emits.
    pub fn sect_align(&self, sect: Sect, spec: &sect::Spec) -> u64 {
        if spec.align != 0 {
            return spec.align;
        }
        self.measured_align
            .get(sect as usize)
            .copied()
            .unwrap_or(1)
            .max(1)
    }

    /// Records the alignment a measured region was placed at. Called once per
    /// such region by the placement pass, with the value its cursor honoured.
    pub(crate) fn set_measured_align(&mut self, sect: Sect, align: u64) {
        if let Some(slot) = self.measured_align.get_mut(sect as usize) {
            *slot = align.max(1);
        }
    }

    /// The PLT entry address for `sym`, or `None` if `sym` has no PLT entry.
    /// The first user entry is `PLT[1]`; `PLT[0]` is the resolver trampoline
    /// whose size is target-specific (16 bytes on x86-64, 32 on `AArch64`
    /// and RISC-V).
    pub fn plt_entry_addr(&self, sym: SymbolId) -> Option<u64> {
        let key = GotKey::addr(sym);
        let &idx = self.plt_index.get(&key)?;
        let spec = self.target.plt_spec();
        let entry_off = spec
            .header_size
            .wrapping_add(u64::from(idx) * spec.entry_size);
        Some(self.plt.vaddr.wrapping_add(entry_off))
    }

    /// The virtual address of an input section, if it was placed.
    pub fn section_vaddr(&self, file: usize, section: u16) -> Option<u64> {
        let v = self
            .sec_vaddr
            .get(file)
            .and_then(|f| f.get(usize::from(section)))?;
        (*v != UNPLACED).then_some(*v)
    }

    /// The output section header index of an input section, if placed.
    pub fn section_shndx(&self, file: usize, section: u16) -> Option<u16> {
        self.sec_shndx
            .get(file)
            .and_then(|f| f.get(usize::from(section)).copied())
            .filter(|&s| s != 0)
    }

    /// A resolver over one input file's resolved symbol addresses.
    ///
    /// The one way to read a resolved address: the writer patches slots
    /// through it and the dynamic emitter computes `RELATIVE` addends through
    /// it, so neither can compute a value the other would not.
    pub fn resolver(&self, file: usize) -> FileResolver<'_> {
        let row = self.sym.get(file);
        FileResolver {
            addr: row.map_or(&[][..], |r| &r.addr),
            got: row.map_or(&[][..], |r| &r.got),
            plt: row.map_or(&[][..], |r| &r.plt),
            tls_got: row.map_or(&[][..], |r| &r.tls_got),
            flags: row.map_or(&[][..], |r| &r.flags),
            size: row.map_or(&[][..], |r| &r.size),
            tls_index: self.tls_index,
            tls: self.tls_block().map(|b| (self.target, b)),
            got_base: self.got_base(),
            exec: self.mode != LinkMode::Shared,
        }
    }

    /// The address every GOT-relative expression measures from, and the value
    /// of `_GLOBAL_OFFSET_TABLE_`.
    ///
    /// One accessor rather than two readings of `self.got.vaddr`, so the
    /// symbol [`crate::defsym`] publishes and the base
    /// [`crate::reloc::RelExpr::GotBase`] subtracts cannot drift apart.
    pub fn got_base(&self) -> u64 {
        // An unplaced `.got` has no address, and zero is not one: a
        // GOT-relative expression would measure from the bottom of the
        // address space and `_GLOBAL_OFFSET_TABLE_` would be published there.
        // The load base is where an empty table would begin, and it is what
        // every other empty bound in `crate::defsym` reports.
        if self.got.size == 0 {
            return self.base();
        }
        self.got.vaddr
    }

    /// Whether `id` is an indirect function this image defines, and so takes
    /// an `R_*_IRELATIVE` rather than a `JUMP_SLOT`.
    pub fn is_ifunc(&self, id: SymbolId) -> bool {
        self.ifunc_index.contains_key(&id)
    }

    /// The address of the resolver of one indirect function, which is the
    /// value its symbol named before every reference was pointed at its stub.
    pub fn ifunc_resolver(&self, id: SymbolId) -> u64 {
        self.ifunc_index.get(&id).copied().unwrap_or_default()
    }

    /// Pairs each entry of `ifunc` with its resolver address. Called once,
    /// after address resolution has captured the resolvers and before the PLT
    /// emitter asks about either.
    fn index_ifuncs(&mut self) {
        self.ifunc_index.clear();
        self.ifunc_index.reserve(self.ifunc.len());
        for (&id, &resolver) in self.ifunc.iter().zip(&self.ifunc_resolvers) {
            self.ifunc_index.insert(id, resolver);
        }
    }

    /// The regions that hold bytes, in placement order.
    ///
    /// Placement walks [`sect::TABLE`] with a cursor that only ever moves
    /// forward, so this order is also address order: the first entry starts the
    /// image and the last one ends it.
    fn placed(&self) -> impl Iterator<Item = Sect> {
        sect::TABLE
            .iter()
            .map(|(s, _)| *s)
            .filter(|s| self.region(*s).size != 0)
    }

    /// The first region that holds bytes, or `None` for an image with none.
    pub fn first_placed(&self) -> Option<Sect> {
        self.placed().next()
    }

    /// The last region that holds bytes, or `None` for an image with none.
    /// Its end is one past the last byte of the image in memory.
    pub fn last_placed(&self) -> Option<Sect> {
        self.placed().last()
    }

    /// The last region that holds bytes *in the file*, or `None` when the image
    /// has none. Its end is one past the last byte of initialised data, which
    /// is where `_edata` sits.
    pub fn last_file_backed(&self) -> Option<Sect> {
        self.placed().filter(|s| sect::file_backed(*s)).last()
    }

    /// The last region that holds executable bytes, or `None` when the image
    /// has none. Its end is one past the last byte of program text, which is
    /// where `etext` sits.
    ///
    /// Executability comes from the flags in [`sect::TABLE`] rather than a
    /// second list, so a new executable region joins by construction.
    pub fn last_executable(&self) -> Option<Sect> {
        sect::TABLE
            .iter()
            .filter(|(_, spec)| sect::executable(spec))
            .map(|(s, _)| *s)
            .rfind(|s| self.region(*s).size != 0)
    }

    /// The section-header index of the region covering `vaddr`, or zero when
    /// the address falls outside every region that holds content.
    ///
    /// Only the regions that hold input-section content are considered; a
    /// synthetic table never contains one.
    pub fn shndx_at(&self, vaddr: u64) -> u16 {
        for sect in sect::CONTENT {
            let r = self.region(sect);
            if vaddr >= r.vaddr && vaddr < r.vaddr.wrapping_add(r.size) {
                return self.shndx(sect);
            }
        }
        0
    }

    /// The laid-out static TLS block, if any TLS content exists. Used by the
    /// writer to emit `PT_TLS` and by resolution to compute TPOFFs.
    pub fn tls_block(&self) -> Option<TlsBlock> {
        let (vaddr, filesz) = if self.tdata.size != 0 {
            (self.tdata.vaddr, self.tdata.size)
        } else if self.tbss.size != 0 {
            (self.tbss.vaddr, 0)
        } else {
            return None;
        };
        // `.tbss` (if any) is placed immediately after `.tdata`, so the block's
        // memory extent runs from `.tdata` start to `.tbss` end.
        let mem_end = self
            .tbss
            .vaddr
            .wrapping_add(self.tbss.size)
            .max(self.tdata.vaddr.wrapping_add(self.tdata.size));
        let offset = if self.tdata.size != 0 {
            self.tdata.offset
        } else {
            self.tbss.offset
        };
        Some(TlsBlock {
            offset,
            vaddr,
            filesz,
            memsz: mem_end.wrapping_sub(vaddr),
            // `PT_TLS`'s alignment is the pair's, not either section's, so it
            // stays its own quantity rather than being read off one header.
            align: self.tls_align.max(1),
        })
    }
}

/// Builds the layout for `ctx`.
///
/// `entry` names the entry-point symbol (resolved to its address, or left at
/// zero if absent). `target` selects the relocation table used by the scan
/// pass. `mode` selects a static `ET_EXEC`, a shared `ET_DYN`, or a dynamic
/// `ET_DYN` executable; in a dynamic mode the GOT is left for the loader to
/// fill and the dynamic tables are laid out alongside the static sections.
/// `config` carries the dynamic-link inputs (soname, interpreter, needed).
pub fn build(
    ctx: &Context<'_>,
    target: Target,
    mode: LinkMode,
    entry: &[u8],
    config: &DynConfig<'_>,
) -> Result<Layout> {
    let mut layout = empty_layout(ctx, target, mode, config)?;
    let (rel, common_off) =
        place::measure_sections(ctx, scratch_map(ctx, 0, section_count)?);
    let eh_frame_hdr_size = hdr_size_for(ctx, rel.size(OutKind::EhFrame));
    // The per-file sym_idx -> SymbolId map was recorded during the intern pass
    // (`Context::resolve_file`), so the scan and address-resolution passes
    // read it directly by index with zero name probes.
    let sym_id = &ctx.sym_id;
    scan::scan_relocations(
        ctx,
        target,
        mode,
        sym_id,
        config.relax,
        &mut layout,
    )?;
    // Now that the scan has classified every relocation, size the per-file
    // GOT/PLT symbol tables only when something reads them. The Region
    // `.size` is not final until placement, so the signal is the index the
    // scan just built. A link with no GOT/PLT skips this entirely.
    // The size rows are on the same footing: `needs_sym_sizes` says a
    // relocation reads `st_size`, which a link can do with no GOT at all.
    //
    // Relaxation reads them too. The rows carry each symbol's preemptibility
    // and whether its value is an address, which is what a rewrite is gated
    // on, and a missing row reads as "preemptible, absolute" -- decline. A
    // link whose every GOT reference relaxes away has an empty GOT index and
    // still needs the rows, or the very sites the scan predicted would be
    // refused for want of the facts that predicted them.
    // A weak undefined reference needs its flag row wherever it appears,
    // including the link that has none of the above: a bare program probing
    // one optional symbol allocates nothing else.
    let any_weak_undef = ctx.files.iter().any(has_weak_undef);
    let size_rows = !layout.got_index.is_empty()
        || !layout.plt_index.is_empty()
        || layout.needs_sym_sizes
        || config.relax
        || any_weak_undef;
    fix_base(mode, config, &mut layout)?;
    // The row sizing touches only the per-file symbol rows, and the sizing
    // probe -- whose survey walks every allocated section's relocations --
    // reads everything else the scan produced. The rows come out for the
    // duration so the two run side by side. The probe runs after
    // `fix_base_if_needed` because its RELATIVE tally reads the final PIE
    // answer.
    let mut rows = std::mem::take(&mut layout.sym);
    let (sized, probe) = rayon::join(
        || {
            if size_rows {
                size_sym_slot_tables(ctx, config.relax, &layout, &mut rows)
            } else {
                Ok(())
            }
        },
        || {
            if mode.is_dynamic() {
                // Size the dynamic plan without building it; addresses are
                // placeholders here (exports and GOT placement are not
                // resolved yet). The sizes depend only on the symbol set
                // and the GOT slot count, which are stable across address
                // resolution; the full tables are built once the addresses
                // exist.
                DynamicPlan::probe(ctx, &layout, target, mode, config).map(Some)
            } else {
                Ok(None)
            }
        },
    );
    layout.sym = rows;
    sized?;
    layout.dynamic = probe?;
    place::place_regions(
        ctx,
        &rel,
        mode,
        place::Sizes {
            eh_frame_hdr: eh_frame_hdr_size,
            build_id: config.build_id.map_or(0, BuildId::note_size),
        },
        &mut layout,
    );
    place::measure_runs(ctx, &mut layout);
    place::assign_shndx(ctx, &mut layout);
    apply_folding(ctx, &mut layout);
    // Stamp debug-section contribution offsets into `sec_vaddr` so the
    // address-resolution pass below resolves a `.rela.debug_*` section
    // symbol to the member's base within its output debug section. Allocated
    // sections were stamped by `place_regions`; this fills the debug rows.
    place::stamp_debug_offsets(ctx, &mut layout);
    resolve_addresses(ctx, target, &common_off, sym_id, &mut layout)?;
    // Both indirect-function lists are final here, so pair them up for the
    // PLT emitter, which otherwise scans them once per entry it writes.
    layout.index_ifuncs();
    layout.entry = entry_addr(ctx, &layout, entry, mode)?;
    // The full dynamic and PLT plans are no longer built here: they depend
    // only on this finished layout, not on the image bytes, so the writer
    // builds them concurrently with its copy pass and swaps them in. Until
    // then the probe's placed regions keep answering `Layout::region`.
    Ok(layout)
}

/// The layout before any measurement: every region empty, the per-file tables
/// sized but zeroed, and the flavour-dependent base and PIE flag chosen.
fn empty_layout(
    ctx: &Context<'_>,
    target: Target,
    mode: LinkMode,
    config: &DynConfig<'_>,
) -> Result<Layout> {
    let sec_vaddr: Vec<Vec<u64>> = scratch_map(ctx, UNPLACED, section_count)?;
    let sec_shndx: Vec<Vec<u16>> = scratch_map(ctx, 0, section_count)?;
    // The `got`/`plt` rows are sized lazily once the scan knows whether a GOT
    // or PLT exists. A static non-PIC link has neither, so they stay empty and
    // the resolver hands out empty slices -- the writer only indexes them for
    // GOT/PLT relocations, which do not exist in that case.
    let sym: Vec<SymRows> = ctx
        .files
        .par_iter()
        .map(|input| {
            Ok(SymRows {
                addr: vec![0u64; symbol_count(input)?],
                got: Vec::new(),
                plt: Vec::new(),
                tls_got: Vec::new(),
                size: Vec::new(),
                flags: Vec::new(),
            })
        })
        .collect::<Result<_>>()?;
    let interp = place::interp_bytes(config, mode);
    Ok(Layout {
        needs_sym_sizes: false,
        abs_candidates: Vec::new(),
        entry: 0,
        phdr_count: 0,
        target,
        e_flags: crate::linker::derive_eflags(ctx, target)?,
        mode,
        // `Static` loads at a fixed base and is never PIE; `Shared`/`DynExec`
        // default to PIE at base 0, until the scan discovers a copy candidate
        // or a non-PIC absolute reference that forces a fixed base.
        pie: mode != LinkMode::Static,
        stack_size: config.z.stack_size.unwrap_or(0),
        base: if mode == LinkMode::Static {
            BASE
        } else {
            SHARED_BASE
        },
        init: Region::default(),
        fini: Region::default(),
        text: Region::default(),
        rodata: Region::default(),
        eh_frame: Region::default(),
        eh_frame_hdr: Region::default(),
        got: Region::default(),
        data_rel_ro: Region::default(),
        data: Region::default(),
        bss: Region::default(),
        tdata: Region::default(),
        tbss: Region::default(),
        preinit_array: Region::default(),
        init_array: Region::default(),
        fini_array: Region::default(),
        tls_align: 1,
        interp,
        interp_region: Region::default(),
        note: Region::default(),
        build_id: Region::default(),
        shndx: [0; sect::TABLE.len()],
        measured_align: [1; sect::TABLE.len()],
        start_stop: Vec::new(),
        sec_vaddr,
        sec_shndx,
        sym,
        got_values: Vec::new(),
        ifunc: Vec::new(),
        ifunc_resolvers: Vec::new(),
        ifunc_index: FxHashMap::default(),
        tls_index: 0,
        got_keys: Vec::new(),
        exports: Vec::new(),
        referenced: FxHashSet::default(),
        got_index: FxHashMap::default(),
        plt_keys: Vec::new(),
        plt_index: FxHashMap::default(),
        copy_slots: Vec::new(),
        canonical_plt: FxHashSet::default(),
        has_absolute_refs: false,
        plt: Region::default(),
        got_plt: Region::default(),
        rela_plt: Region::default(),
        dynamic: None,
        plt_plan: None,
    })
}

/// The `.eh_frame_hdr` byte size to reserve, one table entry per FDE the
/// output will carry, so placement can reserve the region before any address
/// is known. `eh_frame_size` is the measured `.eh_frame` output size: a link
/// that contributes none skips the unwind infrastructure entirely.
///
/// The count comes from the split plan alone. [`Context::split_eh_frame`] runs
/// on every link and covers every member of the `.eh_frame` output section, so
/// a non-zero `eh_frame_size` means a non-empty plan; counting the input
/// records again here would be the same walk with a second answer.
fn hdr_size_for(ctx: &Context<'_>, eh_frame_size: u64) -> u64 {
    if eh_frame_size == 0 {
        return 0;
    }
    crate::ehframe::hdr_size(ctx.eh.live_fdes())
}

/// Drops a dynamic executable to a fixed load base (`ET_EXEC`) when it
/// contains a non-PIC absolute reference whose link-time value must remain
/// valid at runtime.
///
/// One such reference decides it for the whole image, which costs the program
/// ASLR: that is what `clang` does for non-PIC input, and xold has no `-pie`
/// flag with which a caller could demand a position-independent image instead,
/// so there is nothing to diagnose against. lld, which does have the flag,
/// rejects the same input with "can not be used when making a PIE object"
/// unless `-no-pie` was passed. Add that error alongside a `-pie` flag, not
/// before it.
///
/// Two cases: a copy-relocated data import (the absolute reference is patched
/// to the copy slot's address) and a reference whose stored value this link
/// fixes -- to a symbol the executable defines itself (a local/section symbol
/// or defined global), or to an import a copy slot or a canonical PLT stub
/// stands in for. A position-independent image (`ET_DYN`) gets a random base,
/// so such a link-time absolute address would miss its target; and a 32-bit
/// slot (`R_X86_64_32S`/`32`/`16`) cannot hold a 64-bit load base for the
/// loader to relocate. Resolving at a fixed base matches `clang` (which
/// defaults to `ET_EXEC` for non-PIC input) and `clang -no-pie`.
///
/// The canonical PLT half arrives through [`Layout::has_absolute_refs`], set
/// at the reference, rather than as a `canonical_plt` clause here: the stub
/// moves with the image, so a PC-relative reference to it -- the only kind a
/// position-independent compiler emits -- stays correct at a random base, and
/// both `ld.lld -pie` and GNU `ld -pie` keep such an image `ET_DYN`. It is
/// the read-only or narrow slot holding the stub's address that no load base
/// can shift, and that is what the scan flags.
fn fix_base_if_needed(mode: LinkMode, layout: &mut Layout) {
    if mode == LinkMode::DynExec
        && (!layout.copy_slots.is_empty() || layout.has_absolute_refs)
    {
        layout.pie = false;
        layout.base = BASE;
    }
}

/// Settles whether the executable is position-independent, honouring `-pie`
/// and `-no-pie` when one of them was written.
///
/// Without either, [`fix_base_if_needed`] decides from the inputs. With one,
/// the caller has said which image they want, and the two answers can
/// conflict: `-pie` over an input carrying a non-PIC absolute reference asks
/// for an image whose stored addresses would be wrong at a random load base.
/// lld refuses that command line rather than producing it, and so does this.
/// `-no-pie` never conflicts -- a fixed base is always a valid answer for an
/// executable -- so it simply applies.
fn fix_base(
    mode: LinkMode,
    config: &DynConfig<'_>,
    layout: &mut Layout,
) -> Result<()> {
    let forced = mode == LinkMode::DynExec
        && (!layout.copy_slots.is_empty() || layout.has_absolute_refs);
    match config.pie {
        Some(true) if forced => {
            return Err(Error::CommandLine(
                "-pie was given, but an input holds a non-PIC absolute \
                 reference whose value this link has to fix, so the image \
                 cannot be loaded at a random base"
                    .into(),
            ));
        }
        Some(true) => {
            layout.pie = true;
            layout.base = SHARED_BASE;
        }
        Some(false) => {
            layout.pie = false;
            layout.base = BASE;
        }
        None => fix_base_if_needed(mode, layout),
    }
    Ok(())
}

/// The `sec_vaddr` entry of an input section that was never placed.
///
/// Zero cannot say it. A debug section's entry holds its contribution's offset
/// within the aggregated output section, and the first contribution's offset
/// is zero; reading that as "discarded" would tombstone every reference into
/// the first file's `.debug_str`. The sentinel is a value no section can be
/// placed at, so "unplaced" and "at offset zero" stay distinct facts.
/// Placement stamps the ones it keeps, so whatever is still the sentinel
/// afterwards was discarded -- by a COMDAT group losing, by `--gc-sections`,
/// or by never being a kind this linker emits.
pub(crate) const UNPLACED: u64 = u64::MAX;

/// Per-file scratch storage, one row per input file with every entry `fill`,
/// sized by `row_len`. Used for the per-section tables the layout passes index
/// by `(file, section)`.
fn scratch_map<T: Copy + Send + Sync>(
    ctx: &Context<'_>,
    fill: T,
    row_len: impl Fn(&InputFile<'_>) -> Result<usize> + Sync,
) -> Result<Vec<Vec<T>>> {
    // Per-file rows over thousands of files and millions of entries: build
    // them in parallel so the zeroing is not a serial stall. Order is the
    // file order either way; the rows carry no cross-file state.
    ctx.files
        .par_iter()
        .map(|input| Ok(vec![fill; row_len(input)?]))
        .collect()
}

/// The number of sections in an input file.
fn section_count(input: &InputFile<'_>) -> Result<usize> {
    Ok(input.object()?.sections().len())
}

/// The number of symbols in an input file's symbol table.
fn symbol_count(input: &InputFile<'_>) -> Result<usize> {
    Ok(input.symbol_table()?.map_or(0, |st| st.syms.len()))
}

/// Whether the input declares a weak reference it does not define, the entry
/// the flag row exists to name.
fn has_weak_undef(input: &InputFile<'_>) -> bool {
    input.symbol_table().is_ok_and(|table| {
        table.is_some_and(|st| {
            st.syms
                .iter()
                .any(|s| s.bind() == STB_WEAK && s.st_shndx.get() == SHN_UNDEF)
        })
    })
}

/// Sizes the per-file GOT and PLT symbol tables to each input's symbol count
/// when the corresponding section exists. Rows start empty; this zeroes them
/// to the right length just before resolution fills them.
fn size_sym_slot_tables(
    ctx: &Context<'_>,
    relax: bool,
    layout: &Layout,
    rows: &mut [SymRows],
) -> Result<()> {
    let need_got = !layout.got_index.is_empty();
    let need_plt = !layout.plt_index.is_empty();
    let need_tls = layout.got_keys.iter().any(|key| key.kind != GotKind::Addr);
    // A size relocation reads `st_size` rather than an address, and almost no
    // object has one. Sizing the row only when the scan saw one keeps a `u64`
    // per symbol per file off every other link.
    let need_size = layout.needs_sym_sizes;
    // Each file owns its row, so the sizing -- which is mostly the kernel
    // zeroing fresh pages -- fans out.
    rows.par_iter_mut().zip(ctx.files.par_iter()).try_for_each(
        |(row, input)| {
            let n = symbol_count(input)?;
            if need_size {
                row.size.resize(n, 0);
                if let Ok(Some(symtab)) = input.symbol_table() {
                    for (slot, sym) in row.size.iter_mut().zip(symtab.syms) {
                        *slot = sym.st_size.get();
                    }
                }
            }
            if need_got {
                row.got.resize(n, 0);
            }
            // The flag rows are what a rewrite is gated on, and a missing row
            // reads as "preemptible, absolute" -- decline. A link whose every
            // GOT reference relaxed away has no GOT index and still needs
            // them. So does a file carrying a weak undefined reference: its
            // PC-relative sites read `undef_weak` off this row, and a link of
            // nothing but such references has no GOT or PLT for the other
            // conditions to notice.
            if need_got || relax || has_weak_undef(input) {
                row.flags.resize(n, SymFlags::default());
            }
            if need_plt {
                row.plt.resize(n, 0);
            }
            if need_tls {
                row.tls_got.resize(n, 0);
            }
            Ok(())
        },
    )
}

/// Aliases every ICF-folded section onto its representative: copies the
/// representative's placed virtual address and section-header index into the
/// folded section's slots, so symbol resolution (`definition_addr`,
/// `local_addr`) and the output symbol table resolve folded symbols to the
/// representative's address. Runs after placement stamped the representatives
/// and before address resolution reads any section.
fn apply_folding(ctx: &Context<'_>, layout: &mut Layout) {
    for ((f_file, f_sec), (r_file, r_sec)) in ctx.folding.iter() {
        let Some(&rep_vaddr) = layout
            .sec_vaddr
            .get(r_file)
            .and_then(|f| f.get(usize::from(r_sec)))
        else {
            continue;
        };
        let rep_shndx = layout
            .sec_shndx
            .get(r_file)
            .and_then(|f| f.get(usize::from(r_sec)).copied())
            .unwrap_or(0);
        if let Some(f) = layout.sec_vaddr.get_mut(f_file)
            && let Some(slot) = f.get_mut(usize::from(f_sec))
        {
            *slot = rep_vaddr;
        }
        if let Some(f) = layout.sec_shndx.get_mut(f_file)
            && let Some(slot) = f.get_mut(usize::from(f_sec))
        {
            *slot = rep_shndx;
        }
    }
}

/// The address of the entry symbol.
///
/// An executable whose entry symbol is not defined has nowhere to start.
/// Answering zero links clean and dies on `exec` with nothing to say why. lld
/// warns here and GNU ld warns and falls back to the start of `.text`; xold
/// stops, because neither fallback produces an image the caller asked for.
///
/// A shared object legitimately has none: it is entered through its symbols,
/// and `e_entry` stays zero unless `--entry` named something.
fn entry_addr(
    ctx: &Context<'_>,
    layout: &Layout,
    entry: &[u8],
    mode: LinkMode,
) -> Result<u64> {
    let addr = ctx
        .symbols
        .find(entry)
        .and_then(|id| ctx.symbols.symbol(id))
        .and_then(|sym| match &sym.kind {
            SymbolKind::Defined(def) => {
                Some(definition_addr(&ctx.merge, layout, def))
            }
            _ => None,
        });
    match addr {
        Some(addr) => Ok(addr),
        None if mode == LinkMode::Shared || entry.is_empty() => Ok(0),
        None => Err(Error::UndefinedEntry(
            String::from_utf8_lossy(entry).into_owned(),
        )),
    }
}
