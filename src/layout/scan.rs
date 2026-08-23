//! The relocation scan pass: classifies every allocated section's relocations
//! to size the GOT and PLT before addresses are known.
//!
//! The pass is parallelised per input section (each section's relocations are
//! independent) and deterministic: the flat work list is built serially and the
//! per-item key lists are merged in a fixed serial order, so the GOT/PLT
//! allocation order, and thus every output byte, is independent of how the work
//! is scheduled across threads.

use rayon::prelude::*;
use rustc_hash::FxHashSet;

use self::absolute::classify_absolute;
use crate::{
    dynamic::LinkMode,
    elf::{
        Shdr64,
        constants::{SHF_ALLOC, SHF_WRITE, STT_GNU_IFUNC},
    },
    error::{Error, Result},
    layout::{CopySlot, GotKey, GotKind, GotOwner, Layout},
    linker::{Context, DepAddr},
    output::OutKind,
    reloc::{
        Target, relax_drops_got_target, relax_lead_target, relax_trail_target,
        scan_target, spec_target,
        x86_64::{
            R_X86_64_GOTPC32_TLSDESC, R_X86_64_TLSDESC_CALL, R_X86_64_TLSGD,
            R_X86_64_TLSLD,
        },
    },
    symbol::{SymbolId, SymbolKind, exports_to_dynsym},
};

mod absolute;

/// Runs the scan over `ctx`, populating `layout.got_keys`/`plt_keys` and their
/// index maps in first-seen order, recording the data symbols selected for an
/// `R_X86_64_COPY` relocation in `layout.copy_slots`, recording the function
/// symbols selected for a canonical PLT entry in `layout.canonical_plt`, and
/// flagging in `layout.has_absolute_refs` any non-PIC absolute reference whose
/// slot no load base can shift: one to a symbol this image defines, and one to
/// an import the scan bound to a copy slot or a canonical PLT stub.
pub(super) fn scan_relocations(
    ctx: &Context<'_>,
    target: Target,
    mode: LinkMode,
    sym_id: &[Vec<Option<SymbolId>>],
    relax: bool,
    layout: &mut Layout,
) -> Result<()> {
    let items = scan_work_items(ctx);
    let undef = UndefFlags::build(ctx, mode);
    let per_item: Vec<ScanNeeds> = items
        .par_iter()
        .map(|item| {
            scan_work_item(ctx, target, mode, sym_id, &undef, relax, item)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut totals = ScanTotals::default();
    for s in per_item {
        totals.merge(s);
    }
    totals.freeze(layout);
    Ok(())
}

/// The cross-item merge: every key the scan collected, in the first-seen order
/// of a serial walk over the per-item results.
///
/// Each list is paired with a set that answers "already seen" in constant
/// time. The set decides identity only; the list decides order, and that order
/// is what the GOT and PLT are laid out from, so it must stay a function of
/// the walk and of nothing else.
#[derive(Default)]
struct ScanTotals {
    got: Vec<GotKey>,
    seen_got: FxHashSet<GotKey>,
    plt: Vec<GotKey>,
    seen_plt: FxHashSet<GotKey>,
    copy_slots: Vec<CopySlot>,
    /// The objects a copy slot is already reserved for. Slots are
    /// deduplicated by the object they copy, not by the symbol that asked for
    /// it: two sections may reach one object through two of its names, and a
    /// second slot would give the program two copies of it.
    seen_copy: FxHashSet<DepAddr>,
    /// A set rather than a list: nothing reads a canonical-PLT candidate in
    /// order, only whether a symbol is one.
    canonical_plt: FxHashSet<SymbolId>,
    ifunc: Vec<SymbolId>,
    seen_ifunc: FxHashSet<SymbolId>,
    /// Undefined symbols a surviving relocation still names. See
    /// [`ScanNeeds::referenced`].
    referenced: FxHashSet<SymbolId>,
    has_absolute_refs: bool,
    needs_sym_sizes: bool,
    /// Per file, the sections carrying pointer-width absolute relocations,
    /// appended in item order (one item per file, in file order).
    abs_candidates: Vec<Vec<u16>>,
}

impl ScanTotals {
    /// Folds one work item's needs in, keeping the first occurrence of each
    /// key and dropping every later one.
    fn merge(&mut self, s: ScanNeeds) {
        for key in s.got {
            if self.seen_got.insert(key) {
                self.got.push(key);
            }
        }
        for key in s.plt {
            if self.seen_plt.insert(key) {
                self.plt.push(key);
            }
        }
        for slot in s.copy {
            if self.seen_copy.insert(slot.origin) {
                self.copy_slots.push(slot);
            }
        }
        self.canonical_plt.extend(s.canonical_plt);
        for id in s.ifunc {
            if self.seen_ifunc.insert(id) {
                self.ifunc.push(id);
            }
        }
        self.referenced.extend(s.referenced);
        self.has_absolute_refs |= s.has_abs;
        self.needs_sym_sizes |= s.needs_sizes;
        self.abs_candidates.push(s.abs_sections);
    }

    /// Freezes the key order onto `layout` so GOT and PLT contents are
    /// deterministic. GOT slots are handed out by a running total rather than
    /// by position, because a general-dynamic TLS entry takes two of them.
    fn freeze(self, layout: &mut Layout) {
        let mut slot = 0u64;
        for key in &self.got {
            layout
                .got_index
                .insert(*key, u32::try_from(slot).unwrap_or(u32::MAX));
            slot = slot.saturating_add(key.kind.slots());
        }
        layout.got_values = vec![0u64; usize::try_from(slot).unwrap_or(0)];
        layout.got_keys = self.got;
        for (i, key) in self.plt.iter().enumerate() {
            layout
                .plt_index
                .insert(*key, u32::try_from(i).unwrap_or(u32::MAX));
        }
        layout.plt_keys = self.plt;
        layout.copy_slots = self.copy_slots;
        layout.canonical_plt = self.canonical_plt;
        layout.ifunc = self.ifunc;
        layout.referenced = self.referenced;
        layout.has_absolute_refs = self.has_absolute_refs;
        layout.needs_sym_sizes = self.needs_sym_sizes;
        layout.abs_candidates = self.abs_candidates;
        // The emitter visits a file's candidates in section-index order, the
        // order the old whole-file walk visited them in; the items pushed
        // them in output-section-member order, so each row is sorted here.
        for row in &mut layout.abs_candidates {
            row.sort_unstable();
        }
    }
}

/// Per resolved symbol: whether it is an undefined reference, and whether that
/// reference is weak. Carries the id of the runtime TLS helper alongside,
/// because the scan asks about it as often as it asks about the flags.
///
/// The scan asks this of nearly every relocation. Reading it off a flat byte
/// per symbol keeps the question inside the cache, where probing the symbol
/// arena for a resolved kind would be a miss per relocation.
struct UndefFlags {
    /// One byte per [`SymbolId`]: [`Self::UNDEF`] and [`Self::WEAK`].
    bits: Vec<u8>,
    /// The id of `__tls_get_addr`, if the link resolved one. Comparing ids is
    /// what keeps the per-relocation test off the string table.
    tls_helper: Option<SymbolId>,
}

impl UndefFlags {
    const UNDEF: u8 = 1;
    const WEAK: u8 = 2;
    /// The name is private to this image, so no loader can supply it.
    const PRIVATE: u8 = 4;
    /// A definition another image may replace at load time.
    const PREEMPTIBLE: u8 = 8;
    /// A `DT_NEEDED` dependency exports this name, so the loader has somewhere
    /// to bind the reference.
    const IMPORTABLE: u8 = 16;

    /// Builds the table with one linear pass over the resolved symbols.
    ///
    /// The dependency probe is per undefined symbol, not per relocation: a
    /// reference is classified by the bit this leaves behind.
    fn build(ctx: &Context<'_>, mode: LinkMode) -> Self {
        let shared = mode == LinkMode::Shared;
        let mut bits = vec![0u8; ctx.symbols.len()];
        for (slot, id) in bits.iter_mut().zip(ctx.symbols.ids()) {
            let Some(sym) = ctx.symbols.symbol(id) else {
                continue;
            };
            if sym.is_preemptible(shared) {
                *slot |= Self::PREEMPTIBLE;
            }
            if let SymbolKind::Undefined { weak } = sym.kind {
                *slot |= Self::UNDEF;
                if weak {
                    *slot |= Self::WEAK;
                }
                if !exports_to_dynsym(sym.visibility) {
                    *slot |= Self::PRIVATE;
                }
                if ctx.dep_exports.contains(ctx.symbols.name(id)) {
                    *slot |= Self::IMPORTABLE;
                }
            }
        }
        Self {
            bits,
            tls_helper: ctx.symbols.find(TLS_GET_ADDR),
        }
    }

    /// Whether `id` is the runtime helper a lowered TLS sequence drops.
    fn is_tls_helper(&self, id: Option<SymbolId>) -> bool {
        id.is_some() && id == self.tls_helper
    }

    /// Whether `id` is an undefined reference (an import satisfied at runtime
    /// by a shared object, or an error in a static link).
    fn is_undefined(&self, id: Option<SymbolId>) -> bool {
        self.flags(id) & Self::UNDEF != 0
    }

    /// Whether `id` is a non-weak undefined reference, which a static link
    /// cannot satisfy.
    fn is_strong_undefined(&self, id: Option<SymbolId>) -> bool {
        self.flags(id) & (Self::UNDEF | Self::WEAK) == Self::UNDEF
    }

    /// Whether a definition of `id` could be replaced at load time by one
    /// from another image. Undefined references are preemptible too -- which
    /// image satisfies them is not fixed here -- so this is not the
    /// complement of [`Self::is_undefined`].
    fn is_preemptible(&self, id: Option<SymbolId>) -> bool {
        self.flags(id) & Self::PREEMPTIBLE != 0
    }

    /// Whether `id` is an undefined reference no loader can bind, because the
    /// name is private to this image. Such a reference resolves to zero here
    /// or not at all, so it claims no PLT stub and no `.dynsym` row.
    fn is_private(&self, id: Option<SymbolId>) -> bool {
        self.flags(id) & Self::PRIVATE != 0
    }

    /// Whether a reference to `id` can never be satisfied.
    ///
    /// Three ways a strong undefined reference has nowhere to go:
    ///
    /// - A static link has no loader to ask.
    /// - A hidden or internal name is private to this image, so no loader could
    ///   resolve it whatever the image kind.
    /// - An executable's reference to a default-visibility name that no
    ///   `DT_NEEDED` dependency exports. The loader looks the name up in the
    ///   dependencies this image lists, and if none of them offers it the
    ///   lookup fails at startup -- so the link, not the program's first run,
    ///   is where the missing library is reported. Naming a library on the
    ///   command line is what puts the name within reach; that this image *has*
    ///   a loader says nothing about whether the loader can find it.
    ///
    /// A shared object is exempt from the third: its undefined references are
    /// resolved against whatever loads it, which is not known here. That is
    /// lld's rule as well -- `-shared` sets `unresolvedSymbols` to
    /// `ignore-all`, and the policy is consulted only for a name that
    /// `canBeExternal`, so a hidden one is still reported.
    ///
    /// This is lld's `maybeReportUndefined`, whose `canBeExternal` test is the
    /// visibility one. A weak reference is exempt in both linkers: it is a
    /// probe, and resolving to zero is the answer the program tests for.
    fn is_unresolvable(&self, mode: LinkMode, id: Option<SymbolId>) -> bool {
        if !self.is_strong_undefined(id) {
            return false;
        }
        match mode {
            LinkMode::Static => true,
            LinkMode::Shared => self.is_private(id),
            LinkMode::DynExec => self.is_private(id) || !self.is_importable(id),
        }
    }

    /// Whether a `DT_NEEDED` dependency exports `id`, so the loader has
    /// somewhere to bind a reference to it.
    fn is_importable(&self, id: Option<SymbolId>) -> bool {
        self.flags(id) & Self::IMPORTABLE != 0
    }

    fn flags(&self, id: Option<SymbolId>) -> u8 {
        id.and_then(|i| self.bits.get(i.0)).copied().unwrap_or(0)
    }
}

/// One unit of scan work: every placed section of a single input file.
///
/// The unit is a file rather than a section so the file's object header and
/// symbol table are derived once per file instead of once per section, which
/// on a large link is the difference between a few hundred and a few hundred
/// thousand derivations.
struct ScanItem {
    file: usize,
    sections: Vec<u16>,
}

/// The GOT and PLT keys one input section's relocations need, in first-seen
/// order (before cross-section de-duplication), plus the data symbols it
/// selects for a copy relocation, the function symbols it selects for a
/// canonical PLT entry, and whether it contains a non-PIC absolute reference
/// whose slot no load base can shift -- to a symbol this image defines, or to
/// an import bound to a copy slot or a canonical PLT stub.
struct ScanNeeds {
    got: Vec<GotKey>,
    plt: Vec<GotKey>,
    copy: Vec<CopySlot>,
    canonical_plt: Vec<SymbolId>,
    /// Indirect functions this image defines and something references, in
    /// first-seen order. Each takes a stub whose slot the runtime fills from
    /// the resolver.
    ifunc: Vec<SymbolId>,
    /// Undefined symbols a surviving relocation still names, in first-seen
    /// order. Mirrors lld's `used` bit: a reference every lowering consumed
    /// (the relaxed `__tls_get_addr` call) is not part of the program's
    /// interface, so it earns no `.symtab` row.
    referenced: Vec<SymbolId>,
    seen_referenced: FxHashSet<SymbolId>,
    has_abs: bool,
    /// Whether any relocation here reads a symbol's `st_size`.
    needs_sizes: bool,
    /// The sections of this file carrying at least one pointer-width absolute
    /// relocation. The `.rela.dyn` emitter walks exactly these instead of
    /// re-scanning every section of every file; the sizing survey keeps its
    /// own walk, whose over-count for dropped sections is part of the placed
    /// region sizes.
    abs_sections: Vec<u16>,
}

/// Builds the flat, deterministic list of `(file, section)` pairs whose
/// relocations the scan must classify. The order fixes the GOT/PLT allocation
/// order, so it is built serially before the parallel scan.
///
/// Members are bucketed by input file in one pass, then concatenated in file
/// order. That yields exactly the file-major, output-section-order,
/// member-order sequence a nested scan over every file would, without walking
/// every member once per input file.
fn scan_work_items(ctx: &Context<'_>) -> Vec<ScanItem> {
    let mut by_file: Vec<Vec<u16>> = vec![Vec::new(); ctx.files.len()];
    for section in ctx.outputs.iter() {
        if section.kind == OutKind::Bss {
            continue;
        }
        for m in &section.members {
            if let Some(bucket) = by_file.get_mut(m.file) {
                bucket.push(m.section);
            }
        }
    }
    by_file
        .into_iter()
        .enumerate()
        .map(|(file, sections)| ScanItem { file, sections })
        .collect()
}

/// Classifies every relocation of one input section, collecting the GOT and
/// PLT keys it needs in first-seen order. Stateless and shareable across
/// threads: the work item borrows `ctx` and the per-file symbol table by
/// shared reference.
fn scan_work_item(
    ctx: &Context<'_>,
    target: Target,
    mode: LinkMode,
    sym_id: &[Vec<Option<SymbolId>>],
    undef: &UndefFlags,
    relax: bool,
    item: &ScanItem,
) -> Result<ScanNeeds> {
    let mut needs = ScanNeeds {
        got: Vec::new(),
        plt: Vec::new(),
        copy: Vec::new(),
        canonical_plt: Vec::new(),
        ifunc: Vec::new(),
        referenced: Vec::new(),
        seen_referenced: FxHashSet::default(),
        has_abs: false,
        needs_sizes: false,
        abs_sections: Vec::new(),
    };
    let Some(input) = ctx.files.get(item.file) else {
        return Ok(needs);
    };
    if item.sections.is_empty() {
        return Ok(needs);
    }
    let obj = input.object()?;
    let Some(symtab) = input.symbol_table()? else {
        return Ok(needs);
    };
    let sections = obj.sections();
    let abs64 = target.dyn_relocs().abs64;
    let sym_id_row =
        sym_id.get(item.file).map(Vec::as_slice).unwrap_or_default();
    let mut seen_got: FxHashSet<GotKey> = FxHashSet::default();
    let mut seen_plt: FxHashSet<GotKey> = FxHashSet::default();
    let mut seen_copy: FxHashSet<SymbolId> = FxHashSet::default();
    let mut seen_canonical: FxHashSet<SymbolId> = FxHashSet::default();
    let mut seen_ifunc: FxHashSet<SymbolId> = FxHashSet::default();
    for &section in &item.sections {
        let Some(entries) = input.relocations(section)? else {
            continue;
        };
        // Whether this section's bytes live in a writable segment: a 64-bit
        // absolute pointer stored there becomes a `RELATIVE` dynamic
        // relocation the loader applies, so it need not force a fixed-base
        // `ET_EXEC` link.
        let shdr = sections.get(usize::from(section));
        let flags = shdr.map_or(0, |s| s.sh_flags.get());
        let writable = flags & SHF_ALLOC != 0 && flags & SHF_WRITE != 0;
        let site = RelocSite {
            symtab: &symtab,
            sections,
            sym_id_row,
            writable,
            relax,
            // The section's own bytes, so a site can be asked what
            // instruction it sits in. Only the GOT-reclaim question reads
            // them, and only for the types that carry a relaxable form.
            data: shdr
                .and_then(|s| obj.section_data(s).ok())
                .unwrap_or_default(),
        };
        let mut work = ScanWork {
            file: item.file,
            got: &mut needs.got,
            plt: &mut needs.plt,
            copy: &mut needs.copy,
            canonical_plt: &mut needs.canonical_plt,
            seen_got: &mut seen_got,
            seen_plt: &mut seen_plt,
            seen_copy: &mut seen_copy,
            seen_canonical: &mut seen_canonical,
            ifunc: &mut needs.ifunc,
            seen_ifunc: &mut seen_ifunc,
            referenced: &mut needs.referenced,
            seen_referenced: &mut needs.seen_referenced,
            has_abs: &mut needs.has_abs,
            needs_sizes: &mut needs.needs_sizes,
        };
        let mut has_abs64 = false;
        for (i, r) in entries.iter().enumerate() {
            has_abs64 |= r.r_type() == abs64;
            // The entry before this one, which is what says whether a
            // reference to the TLS helper is the tail of a sequence this
            // image lowers. See [`is_lowered_tls_call`].
            let pair = RelocPair {
                r,
                prev: i.checked_sub(1).and_then(|k| entries.get(k)),
            };
            classify_reloc(ctx, target, mode, &site, undef, pair, &mut work)?;
        }
        if has_abs64 {
            needs.abs_sections.push(section);
        }
    }
    Ok(needs)
}

/// Whether relaxation will certainly rewrite this site into a form that reads
/// no GOT slot, so the scan need not reserve one.
///
/// A relaxed `GOTPCRELX` becomes a direct reference: the slot it would have
/// read is filled and never looked at, and it sits in the RELRO GOT, so every
/// image paid for storage nothing used. mold predicts the same rewrite at
/// scan time for the same reason.
///
/// The prediction has to be exact in one direction only. Reserving a slot the
/// writer then relaxes away wastes eight bytes; declining one the writer then
/// needs is a site with nothing to measure against, so every condition the
/// writer will test is tested here -- except the relaxed displacement's range,
/// which needs addresses the layout has not assigned. An image whose
/// displacement does not fit is one whose ordinary PC-relative references do
/// not fit either.
///
/// Only a global definition is considered. A file-local symbol is named by an
/// index rather than an id, and answering "not preemptible, not absolute" for
/// one asks a different question than the writer's resolver will; keeping its
/// slot costs eight bytes and no correctness.
fn relaxes_away_got(
    ctx: &Context<'_>,
    target: Target,
    mode: LinkMode,
    site: &RelocSite<'_>,
    id_opt: Option<SymbolId>,
    r: &crate::elf::Rela64,
) -> bool {
    if !site.relax {
        return false;
    }
    let Some(id) = id_opt else {
        return false;
    };
    let Some(sym) = ctx.symbols.symbol(id) else {
        return false;
    };
    // The rewrite binds the site to this image's definition for good, and
    // computes an address from the program counter. Neither is right for a
    // symbol the loader may resolve elsewhere or whose value is a constant.
    if sym.is_preemptible(mode == LinkMode::Shared) || sym.is_absolute_value() {
        return false;
    }
    let lead = relax_lead_target(target, r.r_type());
    let at = usize::try_from(r.r_offset.get()).unwrap_or(usize::MAX);
    let Some(start) = at.checked_sub(lead) else {
        return false;
    };
    let Some(window) = site.data.get(start..at) else {
        return false;
    };
    relax_drops_got_target(target, r.r_type(), r.r_addend.get(), window)
}

/// Mutable accumulators the per-reloc classifier writes into, bundled so the
/// helper stays under the argument limit.
struct ScanWork<'a> {
    file: usize,
    got: &'a mut Vec<GotKey>,
    plt: &'a mut Vec<GotKey>,
    copy: &'a mut Vec<CopySlot>,
    canonical_plt: &'a mut Vec<SymbolId>,
    seen_got: &'a mut FxHashSet<GotKey>,
    seen_plt: &'a mut FxHashSet<GotKey>,
    seen_copy: &'a mut FxHashSet<SymbolId>,
    seen_canonical: &'a mut FxHashSet<SymbolId>,
    ifunc: &'a mut Vec<SymbolId>,
    seen_ifunc: &'a mut FxHashSet<SymbolId>,
    referenced: &'a mut Vec<SymbolId>,
    seen_referenced: &'a mut FxHashSet<SymbolId>,
    has_abs: &'a mut bool,
    /// Set when a relocation reads a symbol's `st_size`, so the layout knows
    /// to build the per-symbol size rows the resolver reads.
    needs_sizes: &'a mut bool,
}

/// The read-only view of the section being scanned that the per-relocation
/// classifier needs: its file's symbol table and section headers, the cached
/// global ids, and whether the section's bytes land in a writable segment.
struct RelocSite<'a> {
    symtab: &'a crate::elf::SymbolTable<'a>,
    sections: &'a [Shdr64],
    sym_id_row: &'a [Option<SymbolId>],
    writable: bool,
    /// Whether relaxation is on, which decides whether a site the rewrite
    /// covers still needs the GOT slot it asks for.
    relax: bool,
    /// The scanned section's bytes.
    data: &'a [u8],
}

/// One relocation as the classifier sees it: the entry itself, and the entry
/// before it in the same section.
///
/// The predecessor is what says whether a reference to the TLS helper is the
/// tail of a sequence this image lowers, which the name alone cannot --
/// see [`is_lowered_tls_call`].
#[derive(Copy, Clone)]
struct RelocPair<'a> {
    r: &'a crate::elf::Rela64,
    prev: Option<&'a crate::elf::Rela64>,
}

/// Classifies one relocation: records any GOT/PLT key it needs, COPY and
/// canonical-PLT candidates, and whether it forces a fixed-base `ET_EXEC`.
fn classify_reloc(
    ctx: &Context<'_>,
    target: Target,
    mode: LinkMode,
    site: &RelocSite<'_>,
    undef: &UndefFlags,
    pair: RelocPair<'_>,
    work: &mut ScanWork<'_>,
) -> Result<()> {
    let RelocPair { r, prev } = pair;
    let symtab = site.symtab;
    let sym_id_row = site.sym_id_row;
    let needs = scan_target(target, r.r_type())?;
    *work.needs_sizes |= needs.size;
    let sym_idx = r.sym();
    // The per-file sym_id cache gives the global id (or None for locals
    // and section symbols) without a `by_name` hashmap probe, so every
    // per-relocation classification below reads the cache instead of
    // calling `ctx.symbols.find`.
    let id_opt = cached_global_id(sym_id_row, sym_idx);
    // A general-dynamic or local-dynamic sequence calls a runtime helper that
    // an executable never reaches: the sequence is lowered to a form that
    // consumes the call. The reference is dropped before any allocation, so it
    // claims neither a PLT entry nor a GOT slot, and neither demands a
    // definition -- which is why dropping it here does not leave a strong
    // undefined `__tls_get_addr` unreported. A sequence the writer cannot lower
    // is rejected there, by `writer::sections`, which treats a mandatory
    // rewrite that declined as an error rather than falling through to a value
    // computation. So a link that reaches an unlowerable sequence fails at the
    // TLS reference itself, naming the thread-local, rather than at the helper
    // the lowering was going to consume.
    if is_lowered_tls_call(target, mode, undef, id_opt, needs, r, prev) {
        return Ok(());
    }
    // A reference a surviving relocation names is part of the program's
    // interface; see [`ScanTotals::referenced`].
    if let Some(id) = id_opt
        && work.seen_referenced.insert(id)
    {
        work.referenced.push(id);
    }
    let owner = GotOwner::of(id_opt, work.file, sym_idx);
    if needs.got && !relaxes_away_got(ctx, target, mode, site, id_opt, r) {
        push_got(work, GotKey::new(GotKind::Addr, owner));
    }
    // An initial-exec reference loads the offset from a GOT slot. This image
    // fixes that offset only for a thread-local it places itself, and only a
    // loader can fill in one a shared object owns, so the two cases it cannot
    // resolve are rejected rather than filled with a link-time guess.
    if needs.tls_got {
        // A shared object's offsets are the loader's to choose, so the slot
        // is filled by a relocation naming the thread-local. One the object
        // does not export cannot be named, and so cannot be reached this way
        // at all: the compiler uses the local-dynamic model for those.
        if mode == LinkMode::Shared && id_opt.is_none() {
            return Err(unresolvable_tls(
                symtab,
                sym_idx,
                "an initial-exec reference to a thread-local a shared object \
                 does not export has no name for the loader to resolve",
            ));
        }
        // A weak reference is exempt: nothing defines the thread-local, and
        // the program said so, so the offset is zero and the caller's test
        // for it fails. A C library probes its own optional pieces this way
        // (`weak_extern` on an initial-exec thread-local), and a strong
        // reference is still what a missing input looks like.
        if mode == LinkMode::Static && undef.is_strong_undefined(id_opt) {
            return Err(unresolvable_tls(symtab, sym_idx, NO_LOADER));
        }
        push_got(work, GotKey::new(GotKind::TlsOffset, owner));
    }
    // What a general-dynamic reference needs depends on the image. A shared
    // object keeps the model whole: it calls `__tls_get_addr` with a pair of
    // GOT slots naming the module and the offset within it, both of which
    // only the loader knows. An executable has no such helper, so the writer
    // lowers the pair: to the local-exec form when it places the thread-local
    // itself, and to the initial-exec form -- one slot holding the offset from
    // the thread pointer -- when a shared object places it. A static link can
    // do neither for a thread-local it does not define, and lowering against
    // an address of zero would read the wrong storage silently, so it is
    // rejected.
    // Local-dynamic: one entry names this module, and every thread-local in
    // it is reached by a fixed offset from the block that entry describes. An
    // executable does not need the entry at all: its block sits at a known
    // offset from the thread pointer, so the writer rewrites the sequence to
    // read that directly.
    if is_local_dynamic(target, r.r_type()) {
        if mode == LinkMode::Shared {
            push_got(work, GotKey::MODULE);
        }
        return Ok(());
    }
    // A TLS descriptor asks for the same thing a general-dynamic reference
    // does and gets it a different way: the loader fills a descriptor with a
    // resolver function and its argument, and the sequence calls through it.
    // An executable has neither, so the writer lowers both instructions -- to
    // the local-exec form when it places the thread-local, and to the
    // initial-exec form, one slot holding the offset from the thread pointer,
    // when a shared object does. A shared object cannot keep the model here:
    // filling a descriptor takes an `R_X86_64_TLSDESC` dynamic relocation this
    // linker does not emit, and lowering is not open to it either, so the only
    // honest answer is to say so and name the thread-local.
    if is_tls_desc(target, r.r_type()) {
        if mode == LinkMode::Shared {
            return Err(unresolvable_tls(symtab, sym_idx, NO_TLSDESC));
        }
        // Only the reference itself names storage; the call it closes with
        // becomes a `nop` and allocates nothing.
        if r.r_type() == R_X86_64_GOTPC32_TLSDESC && undef.is_undefined(id_opt)
        {
            // As for a general-dynamic reference: only a strong reference to a
            // thread-local nothing defines is a static link's error, and a
            // weak one lowers against an offset of zero.
            if mode == LinkMode::Static && undef.is_strong_undefined(id_opt) {
                return Err(unresolvable_tls(symtab, sym_idx, NO_LOADER));
            }
            push_got(work, GotKey::new(GotKind::TlsOffset, owner));
        }
        return Ok(());
    }
    if is_general_dynamic(target, r.r_type()) {
        if mode == LinkMode::Shared {
            push_got(work, GotKey::new(GotKind::TlsModule, owner));
        } else if undef.is_undefined(id_opt) {
            // As above: only a strong reference to a thread-local nothing
            // defines is a static link's error. A weak one lowers against an
            // offset of zero, which is the answer the program tests for.
            if mode == LinkMode::Static && undef.is_strong_undefined(id_opt) {
                return Err(unresolvable_tls(symtab, sym_idx, NO_LOADER));
            }
            push_got(work, GotKey::new(GotKind::TlsOffset, owner));
        }
        return Ok(());
    }
    // An indirect function is not at the address its symbol names: that is a
    // resolver the runtime calls once, and every reference must reach whatever
    // it returns. The reference therefore goes through a stub whose slot the
    // runtime fills, exactly as an import does, and the value the symbol
    // resolves to is that stub. This holds in every mode: a static image
    // applies these itself before `main`.
    if let Some(id) = id_opt.filter(|&id| is_ifunc(ctx, id)) {
        push_plt(work, GotKey::addr(id));
        if work.seen_ifunc.insert(id) {
            work.ifunc.push(id);
        }
        return Ok(());
    }
    if needs.plt
        && undef.is_preemptible(id_opt)
        && !undef.is_private(id_opt)
        && mode != LinkMode::Static
    {
        // A call needs a stub exactly when this link cannot say what it will
        // reach: an import, whose definition the loader finds; and, in a
        // shared object, an exported definition of its own, which anything
        // loaded earlier may replace. Calling the latter directly is
        // `-Bsymbolic-functions` semantics nobody asked for -- LD_PRELOAD
        // would interpose the function for every other image and not for this
        // one, while its data accesses still went through GLOB_DAT and did
        // see the replacement. lld gives every preemptible symbol NEEDS_PLT
        // in `processAux` for the same reason.
        //
        // A static image has no loader to bind anything: the only undefined
        // reference that survives its scan is a weak one, which resolves to
        // zero for the caller to test. So does a weak reference to a name no
        // loader could look up, which is why a private one takes no stub
        // either: lld folds that call to a direct one for the same reason.
        push_plt(work, GotKey::new(GotKind::Addr, owner));
    }
    if mode == LinkMode::DynExec && !needs.got && !needs.plt {
        classify_absolute(ctx, target, site, undef, r, work)?;
    }
    if undef.is_unresolvable(mode, id_opt) {
        return Err(undefined_reference(symtab, sym_idx));
    }
    Ok(())
}

/// Whether a resolved symbol is an indirect function this image defines.
fn is_ifunc(ctx: &Context<'_>, id: SymbolId) -> bool {
    ctx.symbols.symbol(id).is_some_and(|sym| {
        matches!(&sym.kind, SymbolKind::Defined(def)
            if def.sym_type == STT_GNU_IFUNC)
    })
}

/// Whether a relocation is a general-dynamic TLS reference.
fn is_general_dynamic(target: Target, r_type: u32) -> bool {
    target == Target::X86_64 && r_type == R_X86_64_TLSGD
}

/// Whether a relocation is part of a TLS descriptor sequence: the reference to
/// the descriptor, or the call through the resolver it holds.
fn is_tls_desc(target: Target, r_type: u32) -> bool {
    target == Target::X86_64
        && matches!(r_type, R_X86_64_GOTPC32_TLSDESC | R_X86_64_TLSDESC_CALL)
}

/// Whether a relocation asks for this module's own thread-local block, which
/// is the local-dynamic model's one indirection.
fn is_local_dynamic(target: Target, r_type: u32) -> bool {
    target == Target::X86_64 && r_type == R_X86_64_TLSLD
}

/// Records a GOT entry the first time it is asked for.
fn push_got(work: &mut ScanWork<'_>, key: GotKey) {
    if work.seen_got.insert(key) {
        work.got.push(key);
    }
}

/// Records a PLT entry the first time it is asked for.
fn push_plt(work: &mut ScanWork<'_>, key: GotKey) {
    if work.seen_plt.insert(key) {
        work.plt.push(key);
    }
}

/// The name of relocation symbol `sym_idx`, empty when the symbol is absent.
fn symbol_name<'data>(
    symtab: &crate::elf::SymbolTable<'data>,
    sym_idx: u32,
) -> &'data [u8] {
    symtab
        .syms
        .get(sym_idx as usize)
        .map_or(&[] as &[u8], |sym| symtab.name(sym))
}

/// The error for a thread-local reference this image cannot resolve.
fn unresolvable_tls(
    symtab: &crate::elf::SymbolTable<'_>,
    sym_idx: u32,
    reason: &'static str,
) -> Error {
    let name = symbol_name(symtab, sym_idx);
    Error::unresolvable_tls(String::from_utf8_lossy(name).into_owned(), reason)
}

/// Why a static link cannot resolve a thread-local it does not define.
const NO_LOADER: &str = "it is defined by a shared object, and a static link has no loader to \
     place it";

/// Why a shared object cannot keep the TLS descriptor model.
const NO_TLSDESC: &str = "a shared object's TLS descriptors are filled by the loader through \
     R_X86_64_TLSDESC relocations this linker does not emit; rebuild the \
     input with -mtls-dialect=gnu";

/// The runtime helper a general-dynamic TLS pair calls.
const TLS_GET_ADDR: &[u8] = b"__tls_get_addr";

/// Whether this relocation is the call a lowered dynamic TLS sequence leaves
/// behind. Such a call is reached through the PLT in the usual encoding and
/// through the GOT under `-fno-plt`; either way the lowering consumes it, so
/// it claims neither entry.
///
/// Only an executable lowers the sequence -- a shared object reaches its own
/// thread-locals through the runtime, at an offset not known at link time --
/// so in a `-shared` link the call stands and keeps its PLT entry.
///
/// The name alone does not settle it. A reference to the helper that is *not*
/// the tail of a lowerable sequence -- hand-written assembly, a static libc
/// that both defines and calls it, code taking its address -- is an ordinary
/// reference and needs its ordinary entry. Dropping those left the input's
/// placeholder bytes in the image: a `PLT32` call to address zero, or a
/// `GOTPCREL` displacement measured from storage nothing allocated. So the
/// question is asked of the *sequence*, as [`crate::reloc::Arch::relax_span`]
/// asks it in the writer: this relocation must sit inside the bytes the entry
/// before it would consume. lld draws the same line, consuming exactly the
/// relocation that follows a relaxed `TLSGD`/`TLSLD` (`getTlsGdRelaxSkip`)
/// and giving a standalone helper reference normal treatment.
///
/// The trail is asked for with no bytes in hand, which is the conservative
/// answer: the local-dynamic sequence has a longer form under `-fno-plt`, and
/// the shorter form's reach still covers the closing call in both.
fn is_lowered_tls_call(
    target: Target,
    mode: LinkMode,
    undef: &UndefFlags,
    id: Option<SymbolId>,
    needs: crate::reloc::Needs,
    r: &crate::elf::Rela64,
    prev: Option<&crate::elf::Rela64>,
) -> bool {
    if mode == LinkMode::Shared
        || !(needs.plt || needs.got)
        || !undef.is_tls_helper(id)
    {
        return false;
    }
    prev.is_some_and(|p| consumes(target, p, r.r_offset.get()))
}

/// Whether lowering the sequence `p` opens would swallow the slot at `at`.
///
/// The lowered window runs from the opening opcode bytes through the
/// instruction that closes the sequence; only the part past `p`'s own slot can
/// hold another relocation, so that is the range tested. A type whose rewrite
/// ends at its own slot reaches nothing and answers `false`.
fn consumes(target: Target, p: &crate::elf::Rela64, at: u64) -> bool {
    let trail = relax_trail_target(target, p.r_type(), &[], &[]);
    if trail == 0 {
        return false;
    }
    let Ok(spec) = spec_target(target, p.r_type()) else {
        return false;
    };
    let slot_end = p.r_offset.get().wrapping_add(spec.write.width() as u64);
    (slot_end..slot_end.wrapping_add(trail as u64)).contains(&at)
}

/// The error for a relocation against a symbol that stayed strong-undefined.
/// `symtab`/`sym_idx` supply the name for the diagnostic only.
fn undefined_reference(
    symtab: &crate::elf::SymbolTable<'_>,
    sym_idx: u32,
) -> Error {
    let name = symbol_name(symtab, sym_idx);
    Error::UndefinedReference(String::from_utf8_lossy(name).into_owned())
}

/// The cached global id for `sym_idx`, read from the per-file `sym_id` map.
/// Returns None for locals, section symbols and out-of-range indices -- the
/// same population the `by_name` probe would skip or miss.
fn cached_global_id(
    sym_id_row: &[Option<SymbolId>],
    sym_idx: u32,
) -> Option<SymbolId> {
    sym_id_row.get(sym_idx as usize).copied().flatten()
}
