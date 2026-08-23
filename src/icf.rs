//! Identical code folding (`--icf=all` / `--icf=safe`).
//!
//! Merges allocated input sections whose contents AND relocation signatures
//! are identical, aliasing every reference to a folded section onto one
//! surviving representative. Run after [`crate::gc`] (which first drops dead
//! sections), before layout: the folded members are removed from
//! [`crate::output::OutputSections`] so they contribute no bytes, and a
//! [`FoldMap`] is recorded so layout resolves their symbols and relocations to
//! the representative's address.
//!
//! # Algorithm
//!
//! The classic partition-then-refine scheme lld's `ICF.cpp` and mold's
//! `icf.cc` share, after the "optimistic algorithm" in the Safe ICF paper:
//!
//! 1. Collect eligible code sections (allocated, executable, non-writable, and
//!    none of the kinds [`Eligibility`] turns away). Assign each an equivalence
//!    class keyed by its content hash; every other allocated section gets a
//!    unique class.
//! 2. Propagate relocation-target classes into each section's class for two
//!    rounds, so sections whose targets already differ split early.
//! 3. Sort by class and segregate by the *constant* part: equal `sh_flags`,
//!    equal bytes, equal reloc count, and per-reloc equal `(offset, type,
//!    target offset)`.
//! 4. Iterate the *variable* segregation -- per-reloc target sections must be
//!    in the same class -- to a fixpoint. Splitting one pair may enable
//!    splitting another whose targets previously agreed, hence the loop.
//! 5. Each surviving class with more than one member folds onto its
//!    first-in-collection representative, which inherits the strictest
//!    alignment in the group.
//!
//! Folding is conservative on hashes: the final segregation compares sections
//! exactly, so a hash collision can never merge sections that are not truly
//! identical. `--icf=safe` additionally refuses to fold a section whose address
//! is observable (taken by an absolute or GOT relocation rather than a call).
//!
//! Two sections can be identical in every byte this compares and still not be
//! interchangeable, because what separates them is not in the section at all.
//! A C++ function's catch clauses live in `.gcc_except_table`, named by the
//! FDE that describes the function; folding two handlers that catch different
//! types gives both the surviving one's table. Such sections are kept out of
//! the partition rather than compared more carefully -- there is nothing in
//! them to compare. See [`crate::ehframe::sections_with_lsda`].

use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

use crate::{
    dynamic::LinkMode,
    elf::{Shdr64, constants::SHF_ALLOC},
    error::Result,
    gc::real_section,
    linker::Context,
    output::OutKind,
    reloc::{RelExpr, Target, spec_target},
    symbol::{DefSource, SymbolId, SymbolKind},
    util::is_c_identifier,
};

/// Whether, and how aggressively, identical sections are folded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IcfMode {
    /// Folding disabled (the default): output is byte-identical to a link
    /// with no ICF pass.
    None,
    /// Fold every pair of genuinely identical eligible sections.
    All,
    /// Fold only sections whose address is never observed: referenced solely
    /// through call (PC-relative branch) relocations, never through an absolute
    /// or GOT relocation that would reveal the address.
    Safe,
}

impl IcfMode {
    /// Whether folding is active at all.
    pub const fn is_active(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Maps a folded `(file, section)` to its surviving representative.
///
/// Only a section that resolves to a *different* one is recorded, so sections
/// absent from the map (representatives and never-folded sections) resolve to
/// themselves. Consulted by layout to alias a folded section's placed address
/// and section index onto the representative's, and by
/// [`FoldMap::is_folded`] for the passes that have to skip a folded section
/// outright.
#[derive(Default)]
pub struct FoldMap {
    by: FxHashMap<(usize, u16), (usize, u16)>,
    /// For each representative, the sections recorded as resolving to it: the
    /// reverse of [`Self::by`], and the only thing [`Self::collapse_onto`]
    /// needs to visit.
    ///
    /// Kept as a superset -- a key is pushed on every insert and never removed
    /// when its value is overwritten -- so a stale row costs one probe and
    /// nothing else. `collapse_onto` re-reads `by` before it repoints
    /// anything, which is what makes the superset safe.
    rev: FxHashMap<(usize, u16), Vec<(usize, u16)>>,
}

impl FoldMap {
    /// Records that `folded` resolves to `representative`.
    ///
    /// Used by ICF for identical code and by the merge pass for the input
    /// sections whose content was absorbed into another section's pool: both
    /// need every reference redirected to the surviving copy.
    pub fn alias(
        &mut self,
        folded: (usize, u16),
        representative: (usize, u16),
    ) {
        let target = self.resolve(representative);
        self.record(folded, target);
        self.collapse_onto(folded);
    }

    /// Points `folded` at `target`, in both directions.
    fn record(&mut self, folded: (usize, u16), target: (usize, u16)) {
        self.by.insert(folded, target);
        self.rev.entry(target).or_default().push(folded);
    }

    /// Follows `section` to the end of its alias chain.
    ///
    /// Bounded by the number of entries: each hop moves to a different key,
    /// and `alias` keeps the map free of cycles by resolving before it
    /// inserts. The bound is the loop's termination proof, not a guess.
    fn resolve(&self, section: (usize, u16)) -> (usize, u16) {
        let mut at = section;
        for _ in 0..self.by.len() {
            match self.by.get(&at) {
                Some(&next) if next != at => at = next,
                _ => break,
            }
        }
        at
    }

    /// Repoints every entry that resolved to `folded` at what `folded` now
    /// resolves to.
    ///
    /// Two passes record aliases -- identical code folding and the merge
    /// pool -- and their eligibility sets overlap, so an X to R to C chain is
    /// buildable. Resolving it a hop at a time leaves the answer depending on
    /// which entry the reader reached first, and `apply_folding` reads the map
    /// in hash order. Flattening on insert makes every entry name a final
    /// representative, so the walk has no order to depend on.
    ///
    /// The entries that need repointing are read off [`Self::rev`] rather than
    /// found by scanning: the merge pass aliases every absorbed section, and a
    /// scan per alias made that quadratic in the number of merged sections --
    /// which for a C++ link is every `.rodata.str` and `.debug_str` in it.
    fn collapse_onto(&mut self, folded: (usize, u16)) {
        let Some(&target) = self.by.get(&folded) else {
            return;
        };
        let Some(sources) = self.rev.remove(&folded) else {
            return;
        };
        for source in sources {
            // `rev` is a superset, so a row may name a section that has since
            // been pointed elsewhere. Only one still resolving to `folded` is
            // this collapse's to move.
            if self.by.get(&source) == Some(&folded) {
                self.record(source, target);
            }
        }
    }

    /// Whether `section` was folded onto a different section, and so
    /// contributes no bytes of its own to the image.
    ///
    /// A pass that walks *input* sections rather than output members needs
    /// this: layout stamps a folded section with the representative's virtual
    /// address, so an address lookup cannot tell the two apart, and the folded
    /// copy would contribute a second helping of whatever the representative
    /// already contributes. This is how the dynamic relocation passes exclude
    /// them.
    pub fn is_folded(&self, section: (usize, u16)) -> bool {
        self.by.contains_key(&section)
    }

    /// An iterator over `(folded, representative)` pairs.
    pub fn iter(
        &self,
    ) -> impl Iterator<Item = ((usize, u16), (usize, u16))> + '_ {
        self.by.iter().map(|(&k, &v)| (k, v))
    }
}

/// Runs ICF over `ctx.outputs`, folding duplicate code sections.
///
/// Partitions eligible sections, folds the duplicates, records the alias map
/// in [`Context::folding`] and drops the folded members from
/// [`crate::output::OutputSections`] so layout places and the writer copies
/// only the representatives. A no-op for [`IcfMode::None`].
///
/// `link` selects the image kind, which decides which of this link's own
/// definitions another image may replace; see [`resolve_target`].
pub fn run(ctx: &mut Context<'_>, mode: IcfMode, link: LinkMode) -> Result<()> {
    if !mode.is_active() {
        return Ok(());
    }
    let folds = compute(ctx, mode, link)?;
    if folds.is_empty() {
        return Ok(());
    }
    // The representative inherits the strictest alignment in its group before
    // the rest are dropped: after the drop their requirement is unrecorded.
    ctx.outputs.absorb_alignments(|file, section| {
        folds.get(&(file, section)).copied()
    });
    let folded: FxHashSet<(usize, u16)> = folds.keys().copied().collect();
    ctx.outputs.drop_members(&folded);
    // Through `alias` rather than by assignment, so the entries are
    // flattened the same way the merge pass's are. ICF's own map holds no
    // chains -- every member of a class names its representative directly --
    // but the two passes share one map, and the property has to hold of the
    // map rather than of whichever pass filled it.
    //
    // The insertion order is the class walk's, which is derived from input
    // order; flattening makes the result independent of it either way.
    for (&folded, &rep) in &folds {
        ctx.folding.alias(folded, rep);
    }
    Ok(())
}

// --- entry points --------------------------------------------------------

/// A signature extracted from one eligible section: the byte content plus a
/// normalised relocation list. Owned so the algorithm can outlive the immutable
/// borrow that collected it, before the caller mutates the context.
struct Sig<'a> {
    file: usize,
    section: u16,
    /// The output section the candidate lands in.
    ///
    /// Two sections that end up in different output sections are not the same
    /// thing however identical their bytes: folding them would move one of
    /// them out of the region its references measure against. Today every
    /// eligible candidate routes to `.text`, so this is never the field that
    /// decides -- but nothing enforces that, and a future split (`.text.hot`
    /// beside `.text`) would silently permit a cross-region fold. lld opens
    /// its comparison with the same question, `a->getParent() !=
    /// b->getParent()`.
    out: OutKind,
    /// `sh_flags`, compared in full. Eligibility already pins the three bits
    /// that decide which output section a candidate lands in, so what is left
    /// is everything else an input can declare about the section -- group
    /// membership, mergeability, retention -- and lld compares the whole word
    /// too (`ICF::equalsConstant`, `lld/ELF/ICF.cpp`: `a->flags !=
    /// b->flags`). Two sections that disagree about any of it were not
    /// declared to be the same thing.
    flags: u64,
    content: &'a [u8],
    relocs: Vec<RelocSig>,
}

/// One normalised relocation. The `kind` fixes how the target is compared; the
/// constant `val` is the part of the target that does not change as folding
/// proceeds (a section offset or an absolute value).
struct RelocSig {
    offset: u64,
    r_type: u32,
    kind: TargetKind,
}

/// The fold-aware category of a relocation target.
///
/// Every variant carries the whole constant part of the reference, so two
/// relocations compare equal only when they resolve to the same thing at the
/// same offset. A category that dropped any of it would report two references
/// as interchangeable on the strength of what they have in common, which is
/// how a fold happens that should not have.
#[derive(Clone, Copy, Eq, PartialEq)]
enum TargetKind {
    /// A section-backed definition. `(file, section)` selects its equivalence
    /// class; `val` is `st_value + addend` (the offset within that section).
    Sec { target: (usize, u16), val: u64 },
    /// An absolute symbol. `val` is its value plus addend.
    Abs(u64),
    /// A global symbol whose definition this link does not pin down: an
    /// import, a tentative definition, or one another image may replace. `id`
    /// is its global identity and `val` its addend.
    Ext { id: SymbolId, val: u64 },
    /// A file-local symbol that names neither a section nor an absolute value,
    /// so nothing about it can be compared across files. Identified by the
    /// position it holds in its own file's symbol table, which no other symbol
    /// shares; `val` is its addend.
    ///
    /// The identity is the pair itself rather than a number mixed out of it: a
    /// hash would answer "the same symbol" for two that merely collided, and
    /// the answer decides whether two sections fold.
    LocalExt { at: (usize, u32), val: u64 },
}

/// Computes the `(folded -> representative)` map for `ctx` under `mode`.
fn compute(
    ctx: &Context<'_>,
    mode: IcfMode,
    link: LinkMode,
) -> Result<FxHashMap<(usize, u16), (usize, u16)>> {
    let target = crate::linker::derive_target(ctx)?;
    let shared = link == LinkMode::Shared;
    let rule = Eligibility {
        mode,
        lsda: crate::ehframe::sections_with_lsda(ctx)?,
        address_taken: if mode == IcfMode::Safe {
            scan_address_taken(ctx, target, link)?
        } else {
            FxHashSet::default()
        },
    };
    let mut class: FxHashMap<(usize, u16), u64> = FxHashMap::default();
    let mut sigs = collect_sigs(ctx, shared, &rule, &mut class)?;
    if sigs.len() < 2 {
        return Ok(FxHashMap::default());
    }
    propagate_classes(&sigs, &mut class);
    segregate_to_fixpoint(&mut sigs, &mut class);
    Ok(build_folds(&sigs, &class))
}

// --- collection ----------------------------------------------------------

/// Collects signatures for every eligible section and seeds the class map for
/// every allocated member (eligible sections get their content hash; the rest
/// get unique ids so relocations into them never match unless they target the
/// very same section).
fn collect_sigs<'a>(
    ctx: &'a Context<'_>,
    shared: bool,
    rule: &Eligibility,
    class: &mut FxHashMap<(usize, u16), u64>,
) -> Result<Vec<Sig<'a>>> {
    let mut next_unique = 1u64;
    let mut sigs = Vec::new();
    for out in ctx.outputs.iter() {
        for m in &out.members {
            let key = (m.file, m.section);
            let Some(shdr) = section_shdr(ctx, m.file, m.section)? else {
                set_unique(class, key, &mut next_unique);
                continue;
            };
            let name = section_name(ctx, m.file, m.section);
            if rule.admits(&shdr, &name, key) {
                if let Some(sig) =
                    make_sig(ctx, shared, out.kind, m.file, m.section, &shdr)?
                {
                    class.insert(key, content_hash(&sig));
                    sigs.push(sig);
                } else {
                    set_unique(class, key, &mut next_unique);
                }
            } else {
                set_unique(class, key, &mut next_unique);
            }
        }
    }
    Ok(sigs)
}

/// Records a fresh unique class id for `key`.
fn set_unique(
    class: &mut FxHashMap<(usize, u16), u64>,
    key: (usize, u16),
    next: &mut u64,
) {
    let id = *next;
    *next = next.wrapping_add(1);
    class.insert(key, id);
}

/// The sections this link declines to fold, whatever their contents.
///
/// Held together because the three questions are asked of every candidate and
/// two of them are set membership: gathering each set once keeps the
/// eligibility test off the per-section scan that would otherwise rebuild them.
struct Eligibility {
    mode: IcfMode,
    /// Sections described by an FDE whose CIE declares an LSDA. Excluded in
    /// every mode; see [`crate::ehframe::sections_with_lsda`].
    lsda: FxHashSet<(usize, u16)>,
    /// Sections whose address is observed by a non-call relocation. Excluded
    /// under [`IcfMode::Safe`] only, and empty otherwise; see
    /// [`scan_address_taken`].
    address_taken: FxHashSet<(usize, u16)>,
}

impl Eligibility {
    /// Whether `shdr` is foldable. Eligible sections are allocated,
    /// executable, non-writable `SHT_PROGBITS` code sections of non-zero size,
    /// excluding `.init`/`.fini` (which the runtime must execute distinctly),
    /// sections whose name is a valid C identifier (a user may reach those
    /// through `__start_*`/`__stop_*`), sections another section is ordered
    /// against, and sections whose unwind data carries an LSDA. Under
    /// [`IcfMode::Safe`], a section whose address is taken is excluded too.
    fn admits(&self, shdr: &Shdr64, name: &[u8], key: (usize, u16)) -> bool {
        use crate::elf::constants::{
            SHF_EXECINSTR, SHF_LINK_ORDER, SHF_WRITE, SHT_PROGBITS,
        };
        let flags = shdr.sh_flags.get();
        if flags & SHF_ALLOC == 0
            || flags & SHF_EXECINSTR == 0
            || flags & SHF_WRITE != 0
            || shdr.sh_type.get() != SHT_PROGBITS
            || shdr.sh_size.get() == 0
        {
            return false;
        }
        // An `SHF_LINK_ORDER` section is ordered against the one its `sh_link`
        // names and belongs with it; folding either of the pair on its own
        // would leave the other describing bytes that moved. lld declines for
        // the same reason (`lld/ELF/ICF.cpp`: "SHF_LINK_ORDER
        // sections are ICF'd as a unit with their dependent sections, so we
        // don't consider them for ICF individually").
        if flags & SHF_LINK_ORDER != 0 {
            return false;
        }
        if name == b".init" || name == b".fini" || is_c_identifier(name) {
            return false;
        }
        if self.lsda.contains(&key) {
            return false;
        }
        if self.mode == IcfMode::Safe && self.address_taken.contains(&key) {
            return false;
        }
        true
    }
}

/// Builds the signature for one section: its bytes plus a normalised reloc
/// list. `None` if the bytes or relocations cannot be read (the section is
/// then treated as ineligible).
fn make_sig<'a>(
    ctx: &'a Context<'_>,
    shared: bool,
    out: OutKind,
    file: usize,
    section: u16,
    shdr: &Shdr64,
) -> Result<Option<Sig<'a>>> {
    let Some(input) = ctx.files.get(file) else {
        return Ok(None);
    };
    let content = input.section_bytes(shdr)?;
    let relocs = reloc_signatures(ctx, shared, file, section)?;
    Ok(Some(Sig {
        file,
        section,
        out,
        flags: shdr.sh_flags.get(),
        content,
        relocs,
    }))
}

/// The normalised relocations of one section.
fn reloc_signatures(
    ctx: &Context<'_>,
    shared: bool,
    file: usize,
    section: u16,
) -> Result<Vec<RelocSig>> {
    let Some(input) = ctx.files.get(file) else {
        return Ok(Vec::new());
    };
    let Some(entries) = input.relocations(section)? else {
        return Ok(Vec::new());
    };
    let Some(symtab) = input.symbol_table()? else {
        return Ok(entries
            .iter()
            .map(|r| RelocSig {
                offset: r.r_offset.get(),
                r_type: r.r_type(),
                kind: TargetKind::Abs(r.r_addend.get().cast_unsigned()),
            })
            .collect());
    };
    let mut out = Vec::with_capacity(entries.len());
    for r in entries {
        let site = RelocRef {
            file,
            sym_idx: r.sym(),
            addend: r.r_addend.get(),
        };
        let kind = resolve_target(ctx, shared, &symtab, site);
        out.push(RelocSig {
            offset: r.r_offset.get(),
            r_type: r.r_type(),
            kind,
        });
    }
    Ok(out)
}

/// One relocation reduced to what target resolution needs: which file it came
/// from, which of that file's symbols it names, and its raw RELA addend.
#[derive(Clone, Copy)]
struct RelocRef {
    file: usize,
    sym_idx: u32,
    addend: i64,
}

/// A `Sec` target read through the merge plan.
///
/// ICF runs after the merge pass, and a relocation into a `SHF_MERGE`
/// section names content the pool has already deduplicated: every
/// translation unit carries its own copy of a string literal, so two
/// identical functions in different objects reference the same string
/// through two different input sections. Compared by that raw `(file,
/// section)` pair the classes stay apart and the functions never fold,
/// however alike they are.
///
/// Naming the pool's carrier as the target and the pool offset as the
/// value is lld's comparison -- relocations into mergeable sections are
/// equal when their offsets in the parent output section are
/// (`lld/ELF/ICF.cpp`) -- and the pool is where the copies
/// meet. `inside` is the part of the reference that is an offset into the
/// input section and `bias` the part that is not, split the same way
/// [`MergePlan::addend`] splits it: a section symbol carries the whole
/// offset in its addend, a named symbol in its `st_value` with the addend
/// a bias on the site. An offset outside the merged content keeps the
/// input section it came from, which can only refuse the fold, never
/// grant one.
fn merged_sec(
    ctx: &Context<'_>,
    target: (usize, u16),
    inside: u64,
    bias: i64,
) -> TargetKind {
    let raw = inside.wrapping_add(bias.cast_unsigned());
    if !ctx.merge.file_has_merges(target.0) {
        return TargetKind::Sec { target, val: raw };
    }
    let moved = ctx.merge.remap(target.0, target.1, inside);
    let carrier = ctx.merge.carrier_of(target.0, target.1);
    match (moved, carrier) {
        (Some(moved), Some(carrier)) => TargetKind::Sec {
            target: carrier,
            val: moved.wrapping_add(bias.cast_unsigned()),
        },
        _ => TargetKind::Sec { target, val: raw },
    }
}

/// Resolves a relocation symbol to a fold-aware target kind. The constant `val`
/// folds the addend into the target offset so two relocations agree only when
/// their stored value agrees.
///
/// A symbol another image may replace is reduced to its own identity rather
/// than to the section it happens to be defined in. Two sections that differ
/// only by which preemptible symbol they name are identical in this image and
/// not after interposition, and one folded pair cannot call two different
/// things; lld's `constantEq` refuses such a pair for the same reason.
fn resolve_target(
    ctx: &Context<'_>,
    shared: bool,
    symtab: &crate::elf::SymbolTable<'_>,
    site: RelocRef,
) -> TargetKind {
    use crate::elf::constants::SHN_ABS;
    let RelocRef {
        file,
        sym_idx,
        addend,
    } = site;
    let add = addend.cast_unsigned();
    let Some(sym) = symtab.syms.get(sym_idx as usize) else {
        return TargetKind::LocalExt {
            at: (file, sym_idx),
            val: add,
        };
    };
    let local = TargetKind::LocalExt {
        at: (file, sym_idx),
        val: add,
    };
    if sym.bind() == crate::elf::constants::STB_LOCAL {
        return match sym.st_shndx.get() {
            SHN_ABS => TargetKind::Abs(sym.st_value.get().wrapping_add(add)),
            // Undefined, tentative, and every other reserved index name no
            // section this partition classifies.
            shndx if real_section(shndx).is_none() => local,
            shndx => {
                // A section symbol carries the whole in-section offset in
                // its addend; a named local holds it in `st_value` and the
                // addend only biases the site. The same split
                // [`MergePlan::addend`] makes.
                let section_symbol =
                    sym.type_() == crate::elf::constants::STT_SECTION;
                let (inside, bias) = if section_symbol {
                    (sym.st_value.get().wrapping_add(add), 0)
                } else {
                    (sym.st_value.get(), addend)
                };
                merged_sec(ctx, (file, shndx), inside, bias)
            }
        };
    }
    // A global is resolved through the table so the true definition's home
    // section supplies the equivalence class. The table keys stems, so a
    // versioned spelling is stemmed first; a miss here only suppresses
    // folding, but a suppressed fold is still a missed fold.
    let name = crate::symbol::version_stem(symtab.name(sym));
    let Some(id) = ctx.symbols.find(name) else {
        return local;
    };
    let Some(resolved) = ctx.symbols.symbol(id) else {
        return local;
    };
    let external = TargetKind::Ext { id, val: add };
    if resolved.is_preemptible(shared) {
        return external;
    }
    match &resolved.kind {
        SymbolKind::Defined(def) => match def.source {
            DefSource::Section { index, offset } => {
                merged_sec(ctx, (def.file, index), offset, addend)
            }
            DefSource::Absolute { value } => {
                TargetKind::Abs(value.wrapping_add(add))
            }
        },
        SymbolKind::Common { .. } | SymbolKind::Undefined { .. } => external,
    }
}

// --- hash propagation ----------------------------------------------------

/// Rounds of relocation-hash propagation. Two is the empirical value lld uses:
/// enough to shrink the average class before the quadratic segregation runs,
/// cheap enough to be worth it.
const PROPAGATION_ROUNDS: usize = 2;

/// Propagates relocation-target classes into each section's class for
/// [`PROPAGATION_ROUNDS`] rounds, so groups that cannot agree on their targets
/// split before the exact segregation runs.
///
/// Each round reads the classes of the previous round only, and publishes the
/// new ones after the whole round is over. Updating in place would break the
/// invariant that all possibly-identical sections share an equivalence class at
/// any moment: a section hashed before its relocation target sees the target's
/// old class while one hashed after sees the new one, so two identical sections
/// end up in different classes purely because of where their common target sits
/// in the iteration order, and the segregation never even compares them.
fn propagate_classes(
    sigs: &[Sig<'_>],
    class: &mut FxHashMap<(usize, u16), u64>,
) {
    let mut next = Vec::with_capacity(sigs.len());
    for _ in 0..PROPAGATION_ROUNDS {
        next.clear();
        for s in sigs {
            let base = class.get(&(s.file, s.section)).copied().unwrap_or(0);
            let mut h = base.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            for r in &s.relocs {
                if let TargetKind::Sec { target, .. } = r.kind {
                    let tc = class.get(&target).copied().unwrap_or(0);
                    h = h
                        .wrapping_mul(31)
                        .wrapping_add(tc.wrapping_mul(0x100_0000_01b3));
                }
            }
            next.push(h | (1u64 << 62));
        }
        debug_assert_eq!(next.len(), sigs.len(), "one new class per section");
        for (s, &h) in sigs.iter().zip(next.iter()) {
            class.insert((s.file, s.section), h);
        }
    }
}

/// The content hash seeding a section's initial class.
fn content_hash(sig: &Sig<'_>) -> u64 {
    let mut h = FxHasher::default();
    sig.content.hash(&mut h);
    h.finish() | (1u64 << 62)
}

// --- segregation ---------------------------------------------------------

/// Sorts sections by class, splits each class by the constant part, then
/// iterates the variable split to a fixpoint. Exact comparisons throughout, so
/// hash collisions never cause a false fold.
fn segregate_to_fixpoint(
    sigs: &mut [Sig],
    class: &mut FxHashMap<(usize, u16), u64>,
) {
    sigs.sort_by_key(|s| class.get(&(s.file, s.section)).copied().unwrap_or(0));
    let mut base = next_base(class);
    let mut scratch: FxHashMap<(usize, u16), u64> = FxHashMap::default();
    segregate_pass(sigs, class, &mut scratch, base, true);
    // A pass that splits anything raises the class count by at least one and
    // the count can never exceed the number of sections, so this many passes
    // always reach the fixpoint; the loop leaves early on the first pass that
    // splits nothing, which is what happens after a handful of passes in
    // practice. A tighter bound would be wrong rather than slow: a class that
    // one more pass would have split holds sections that are not identical,
    // and they would fold.
    for _ in 0..sigs.len() {
        base = base.wrapping_add(sigs.len() as u64 + 1);
        if !segregate_pass(sigs, class, &mut scratch, base, false) {
            break;
        }
    }
}

/// Splits each same-class range of `sigs` into sub-classes. When `constant`
/// holds, two sections match on bytes and the constant part of each reloc;
/// otherwise they match on the current class of each reloc's target section.
/// Returns whether any range split. Reassigns classes as `base + first_index`
/// of the subgroup, which keeps subgroups distinct within and across ranges.
fn segregate_pass(
    sigs: &mut [Sig<'_>],
    class: &mut FxHashMap<(usize, u16), u64>,
    scratch: &mut FxHashMap<(usize, u16), u64>,
    base: u64,
    constant: bool,
) -> bool {
    // The entering classes, copied into a buffer the caller reuses. Cloning
    // the map allocated once per refinement pass, and a link with many
    // candidates runs many passes.
    if constant {
        scratch.clear();
    } else {
        scratch.clone_from(class);
    }
    let snapshot = (!constant).then_some(&*scratch);
    // The pass reads the classes it entered with and writes the new ones into
    // `class`, so a range split early on cannot disturb a range visited later.
    let lookup =
        |key: (usize, u16)| -> Option<u64> { snapshot?.get(&key).copied() };
    let mut changed = false;
    let mut i = 0;
    while i < sigs.len() {
        let cur = class
            .get(&(sigs[i].file, sigs[i].section))
            .copied()
            .unwrap_or(0);
        let mut end = i + 1;
        while end < sigs.len()
            && class
                .get(&(sigs[end].file, sigs[end].section))
                .copied()
                .unwrap_or(0)
                == cur
        {
            end += 1;
        }
        let mut subgroups = 0u64;
        for k in i..end {
            let rep = first_equal(sigs, i, k, constant, &lookup);
            let id = base.wrapping_add(rep as u64);
            class.insert((sigs[k].file, sigs[k].section), id);
            if rep == k {
                subgroups += 1;
            }
        }
        if subgroups > 1 {
            changed = true;
        }
        i = end;
    }
    sigs.sort_by_key(|s| class.get(&(s.file, s.section)).copied().unwrap_or(0));
    changed
}

/// The first index `m` in `[begin, k]` whose section equals `sigs[k]`, or `k`
/// itself. Defines subgroup membership within a class range.
fn first_equal(
    sigs: &[Sig<'_>],
    begin: usize,
    k: usize,
    constant: bool,
    lookup: &impl Fn((usize, u16)) -> Option<u64>,
) -> usize {
    for m in begin..=k {
        if m == k || sections_equal(&sigs[m], &sigs[k], constant, lookup) {
            return m;
        }
    }
    k
}

/// Exact equality of two sections. `constant` compares bytes and the
/// non-moving reloc parts; the variable pass compares reloc target classes.
fn sections_equal(
    a: &Sig<'_>,
    b: &Sig<'_>,
    constant: bool,
    lookup: &impl Fn((usize, u16)) -> Option<u64>,
) -> bool {
    if a.out != b.out || a.flags != b.flags || a.relocs.len() != b.relocs.len()
    {
        return false;
    }
    // The variable pass only ever visits pairs the constant pass already
    // found equal -- classes are refined from there on, never merged -- so
    // their bytes are known to match and re-comparing them is a memcmp over
    // every candidate, per pass, for nothing.
    if constant && a.content != b.content {
        return false;
    }
    for (ra, rb) in a.relocs.iter().zip(b.relocs.iter()) {
        if ra.offset != rb.offset || ra.r_type != rb.r_type {
            return false;
        }
        if !targets_equal(ra.kind, rb.kind, constant, lookup) {
            return false;
        }
    }
    true
}

/// Equality of two relocation targets.
///
/// The constant pass compares everything that does not move as folding
/// proceeds: an absolute target's value, an external target's identity and
/// addend, and a section-backed target's offset. The variable pass only ever
/// visits pairs the constant pass has already found equal -- classes are
/// refined from there on, never merged -- so it re-examines the one part that
/// does move, the equivalence class of a section-backed target, and takes the
/// rest on the constant pass's word. Rejecting an absolute or external target
/// here would split pairs that are known to be identical.
fn targets_equal(
    a: TargetKind,
    b: TargetKind,
    constant: bool,
    lookup: &impl Fn((usize, u16)) -> Option<u64>,
) -> bool {
    match (a, b) {
        (
            TargetKind::Sec {
                target: ta,
                val: va,
            },
            TargetKind::Sec {
                target: tb,
                val: vb,
            },
        ) => {
            if constant {
                return va == vb;
            }
            // A target ICF never classified (a section outside the output, so
            // outside the partition) is equal to nothing, not even to another
            // unclassified target: nothing says the two are the same section.
            // This is lld's reserved equivalence class 0.
            match (lookup(ta), lookup(tb)) {
                (Some(ca), Some(cb)) => ca == cb,
                _ => false,
            }
        }
        (TargetKind::Abs(va), TargetKind::Abs(vb)) => !constant || va == vb,
        (
            TargetKind::Ext { id: ia, val: va },
            TargetKind::Ext { id: ib, val: vb },
        ) => !constant || (ia == ib && va == vb),
        (
            TargetKind::LocalExt { at: aa, val: va },
            TargetKind::LocalExt { at: ab, val: vb },
        ) => !constant || (aa == ab && va == vb),
        _ => false,
    }
}

// --- fold selection ------------------------------------------------------

/// Picks one representative per final class with more than one member and maps
/// the rest onto it. The representative is the lowest-`(file, section)` member
/// of its class, so the choice is deterministic and independent of the sort.
fn build_folds(
    sigs: &[Sig],
    class: &FxHashMap<(usize, u16), u64>,
) -> FxHashMap<(usize, u16), (usize, u16)> {
    let mut groups: FxHashMap<u64, Vec<(usize, u16)>> = FxHashMap::default();
    for s in sigs {
        let c = class.get(&(s.file, s.section)).copied().unwrap_or(0);
        groups.entry(c).or_default().push((s.file, s.section));
    }
    let mut folds = FxHashMap::default();
    for (_c, mut members) in groups {
        if members.len() < 2 {
            continue;
        }
        members.sort_unstable();
        let rep = members[0];
        for m in members.into_iter().skip(1) {
            folds.insert(m, rep);
        }
    }
    folds
}

/// A class id strictly above every current one, used as the base for a fresh
/// segregation pass.
fn next_base(class: &FxHashMap<(usize, u16), u64>) -> u64 {
    let max = class.values().copied().max().unwrap_or(0);
    max.wrapping_add(1)
}

// --- --icf=safe ----------------------------------------------------------

/// Collects the sections whose address is observable under [`IcfMode::Safe`],
/// which is what that mode promises to preserve: `&f == &g` must stay false
/// for two functions the program can compare.
///
/// Two things make an address observable. A relocation that computes the
/// address rather than transferring control to it -- see [`is_call_reloc`] --
/// and an exported definition, whose address another image in the process can
/// take without this link ever seeing the reference. lld marks the second set
/// for the same reason, over `sym->isExported` in `findKeepUniqueSections`.
///
/// Two section kinds are deliberately not scanned, both because they describe
/// a function rather than use it. `.rela.debug_*` is not scanned because
/// honouring it would make `--icf=safe` fold nothing in a `-g` build; the
/// consequence is the one every folding linker has, lld included, that a
/// debugger attributes one address to whichever of the folded functions it
/// looks up first. `.eh_frame` is not scanned for the same reason and with
/// more force: an FDE names every function in the image, so treating its
/// `PC_begin` as address-taking would leave nothing foldable at all. lld
/// never consults either, taking its answer from the address-significance
/// table and the dynamic symbols instead.
///
/// What is *not* implemented is `SHT_LLVM_ADDRSIG`. Without it this stays
/// conservative rather than wrong: a compiler that emits the table is telling
/// the linker about addresses it could not otherwise see, and ignoring the
/// table only means fewer folds.
fn scan_address_taken(
    ctx: &Context<'_>,
    target: Target,
    link: LinkMode,
) -> Result<FxHashSet<(usize, u16)>> {
    let mut taken: FxHashSet<(usize, u16)> = FxHashSet::default();
    for out in ctx.outputs.iter() {
        if out.kind == OutKind::EhFrame {
            continue;
        }
        for m in &out.members {
            mark_address_taken(ctx, target, m.file, m.section, &mut taken)?;
        }
    }
    mark_exported(ctx, link, &mut taken);
    Ok(taken)
}

/// Marks the section behind every exported definition.
///
/// An exported function's address is reachable from outside this link, so
/// nothing here can prove it is never taken. A static link exports nothing and
/// this walk finds nothing; the predicate is
/// [`crate::dynamic::exports_definition`], the same one the `.dynsym` rows are
/// built from, so the two cannot disagree about what "exported" means.
fn mark_exported(
    ctx: &Context<'_>,
    link: LinkMode,
    taken: &mut FxHashSet<(usize, u16)>,
) {
    if link == LinkMode::Static {
        return;
    }
    for (name, sym) in ctx.symbols.entries() {
        let SymbolKind::Defined(def) = &sym.kind else {
            continue;
        };
        let DefSource::Section { index, .. } = def.source else {
            continue;
        };
        if crate::dynamic::exports_definition(ctx, link, sym.visibility, name) {
            taken.insert((def.file, index));
        }
    }
}

/// Marks every section-backed target of a non-call relocation in one section.
fn mark_address_taken(
    ctx: &Context<'_>,
    target: Target,
    file: usize,
    section: u16,
    taken: &mut FxHashSet<(usize, u16)>,
) -> Result<()> {
    let Some(input) = ctx.files.get(file) else {
        return Ok(());
    };
    let Some(entries) = input.relocations(section)? else {
        return Ok(());
    };
    let Some(symtab) = input.symbol_table()? else {
        return Ok(());
    };
    for r in entries {
        if is_call_reloc(target, r.r_type())? {
            continue;
        }
        let Some(sym) = symtab.syms.get(r.sym() as usize) else {
            continue;
        };
        let key = reloc_target_section(ctx, file, &symtab, sym);
        if let Some(k) = key {
            taken.insert(k);
        }
    }
    Ok(())
}

/// Whether a relocation transfers control to the symbol rather than computing
/// its address. The inverse of "address-taking".
///
/// Only the PLT forms qualify. `RelExpr::Pc` does not, however much it looks
/// like a call: on x86-64 `R_X86_64_PC32` is what `lea f(%rip), %rdi` uses to
/// take a function's address, while a direct call to a global is
/// `R_X86_64_PLT32` even in non-PIC code. Counting `Pc` as a call left an
/// address-taken function foldable, so `&f == &g` became observable -- the
/// exact thing `--icf=safe` exists to prevent.
///
/// The cost of the stricter reading is folds, not correctness: a PC-relative
/// jump between two sections marks the target address-taken when it was only
/// branched to. Erring that way is what "safe" means.
fn is_call_reloc(target: Target, r_type: u32) -> Result<bool> {
    let expr = spec_target(target, r_type)?.expr;
    Ok(matches!(expr, RelExpr::Plt | RelExpr::PltPc))
}

/// The `(file, section)` a relocation symbol is defined in, if it is
/// section-backed. A local symbol targets a section in `file`; a global is
/// resolved through the table so the definition's true home is used.
fn reloc_target_section(
    ctx: &Context<'_>,
    file: usize,
    symtab: &crate::elf::SymbolTable<'_>,
    sym: &crate::elf::Sym64,
) -> Option<(usize, u16)> {
    if sym.bind() == crate::elf::constants::STB_LOCAL {
        // `real_section` and not a bare `< SHN_LORESERVE` test: the latter
        // admits `SHN_UNDEF`, which would report section zero as the target
        // and mark it address-taken. Section zero is the null entry, never an
        // output member, so nothing downstream could act on it.
        return real_section(sym.st_shndx.get()).map(|shndx| (file, shndx));
    }
    // The table keys stems, so a versioned spelling must be stemmed here or
    // the reference misses its resolved symbol entirely.
    let name = crate::symbol::version_stem(symtab.name(sym));
    let id = ctx.symbols.find(name)?;
    let resolved = ctx.symbols.symbol(id)?;
    if let SymbolKind::Defined(def) = &resolved.kind
        && let DefSource::Section { index, .. } = def.source
    {
        Some((def.file, index))
    } else {
        None
    }
}

// --- helpers -------------------------------------------------------------

/// The `Shdr64` of `(file, section)`, if readable.
fn section_shdr(
    ctx: &Context<'_>,
    file: usize,
    section: u16,
) -> Result<Option<Shdr64>> {
    let Some(input) = ctx.files.get(file) else {
        return Ok(None);
    };
    let obj = input.object()?;
    Ok(obj.sections().get(usize::from(section)).copied())
}

/// The name of `(file, section)`, borrowed from the object's section-name
/// table. Owned as an empty `Vec` when unreadable.
fn section_name(ctx: &Context<'_>, file: usize, section: u16) -> Vec<u8> {
    let Some(input) = ctx.files.get(file) else {
        return Vec::new();
    };
    input.object().map_or_else(
        |_| Vec::new(),
        |obj| {
            obj.sections()
                .get(usize::from(section))
                .map_or_else(Vec::new, |s| obj.section_name(s).to_vec())
        },
    )
}
