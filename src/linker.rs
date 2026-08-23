//! The link driver: owns the inputs and runs the resolution and layout passes.
//!
//! The passes mirror the seam lld and mold share between symbol resolution and
//! section layout. [`Context`] borrows its input bytes for its whole lifetime,
//! so the driver keeps the [`InputBytes`] values alive in [`link_image`] for as
//! long as the context.
//!
//! An input is named by [`Input`], which is either a path xold maps itself or
//! bytes the caller already holds; the two are indistinguishable once open, so
//! a caller that generates objects in process need not write them out first.

mod assemble;
mod deps;

use std::path::{Path, PathBuf};

use rayon::prelude::*;
use rustc_hash::FxHashSet;

use self::assemble::{
    assemble_full, check_dep_machines, derive_target as derive_target_inner,
    needed_sonames,
};
pub use self::deps::{
    DepAddr, DepExports, DepUndefs, DepVersion, DepVersions, DynExport,
};
use crate::{
    archive::Archive,
    buildid::BuildId,
    comdat::GroupDedup,
    debug::DebugSections,
    dynamic::{DynConfig, HashStyle, LinkMode, Strip, ZOptions},
    ehframe::EhFramePlan,
    elf::{
        Group, ObjectFile, Shdr64,
        constants::{
            ET_DYN, SHF_ALLOC, SHF_COMPRESSED, SHF_EXCLUDE, SHF_EXECINSTR,
            SHF_MERGE, SHF_TLS, SHF_WRITE, SHN_UNDEF, SHT_NOBITS, SHT_NOTE,
            SHT_NULL, SHT_PROGBITS, STB_GLOBAL, STT_NOTYPE, STV_DEFAULT,
        },
    },
    error::{Error, Result},
    icf::{FoldMap, IcfMode},
    input::{Input, InputBytes, InputFile, InputSymbol},
    merge::MergePlan,
    output::{
        Member, NO_PRIORITY, Named, OutKind, OutputSections, init_priority,
    },
    pool,
    startstop::StartStop,
    symbol::{
        DefSource, Definition, Symbol, SymbolId, SymbolKind, SymbolTable,
    },
    util::{open_all, show},
    versionscript::{VersionScript, Visibility as VersionVisibility},
};

/// Everything the linker knows during one run, borrowing the mapped inputs.
pub struct Context<'data> {
    /// Parsed inputs, in command-line order.
    pub files: Vec<InputFile<'data>>,
    /// Global symbols pre-extracted during the parallel parse pass, parallel
    /// to [`Self::files`]. [`Self::resolve_symbols`] consumes these (serial
    /// merge) for direct objects; [`Option::None`] for archive members added
    /// later by [`Self::pull_archives`], which extract on demand.
    extracted: Vec<Option<Vec<InputSymbol<'data>>>>,
    /// Section groups pre-parsed during the same pass, parallel to
    /// [`Self::files`] until [`Self::resolve_symbols`] consumes the lot for
    /// the batch COMDAT scan. Archive members added later parse on demand.
    group_lists: Vec<Vec<Group<'data>>>,
    /// Each file's shard partition of [`Self::extracted`], parallel to it
    /// and consumed with it. Computed where the file parsed.
    bins: Vec<Vec<(u32, u32)>>,
    /// The global resolved symbol table.
    pub symbols: SymbolTable<'data>,
    /// Per file, per input symbol: the resolved global `SymbolId`, or `None`
    /// for locals and section symbols. Built once during
    /// [`Self::resolve_file`] at intern time (the id is the value `intern`
    /// just returned), so the layout scan and address-resolution passes
    /// read it by index with zero name probes. Parallel to
    /// [`Self::files`]; grown whenever a file is added.
    pub sym_id: Vec<Vec<Option<SymbolId>>>,
    /// The output-section layout.
    pub outputs: OutputSections,
    /// Symbols the linker defines itself, paired with the bound each names.
    /// Filled by [`crate::defsym::define`] once no more inputs can arrive;
    /// their addresses are only known after the layout places those bounds.
    pub defsyms: Vec<(SymbolId, crate::defsym::DefSym)>,
    /// Symbols exported by `DT_NEEDED` shared-object dependencies, keyed by
    /// name. A dynamic executable reads a data symbol's size and alignment
    /// from here to size a copy relocation; empty for a static or `-shared`
    /// link (no dependencies).
    pub dep_exports: DepExports<'data>,
    /// Each dependency's position among the direct inputs, indexed by
    /// dependency index (the `dep` of a [`DepAddr`]). The archive pass reads
    /// it to ask whether a shared library that exports a name was reached
    /// before a given archive: only a dependency already in hand stops the
    /// member's extraction (see [`Self::extracts_member`]).
    pub dep_order: Vec<u32>,
    /// Versioned exports of the shared-object dependencies, keyed by symbol
    /// name: each entry pairs the dependency soname with the version name and
    /// hash the symbol carries in the dependency's `.gnu.version`. Used by the
    /// dynamic plan to emit `.gnu.version_r` (VERNEED) and the per-symbol
    /// `.gnu.version` indices.
    pub dep_versions: DepVersions<'data>,
    /// Names the `DT_NEEDED` dependencies reference but do not define. A
    /// dynamic executable exports a definition of its own exactly when one of
    /// its dependencies names it here, so the loader can bind the dependency's
    /// reference to the program's definition. Empty for a static or `-shared`
    /// link, and for any link with no shared-object input.
    pub dep_undefs: DepUndefs<'data>,
    /// Whether `--gc-sections` dead-section garbage collection is enabled. Off
    /// by default; when on, [`Context::assign_sections`] hands the grouped
    /// sections to [`crate::gc::run`], which drops unreferenced `SHF_ALLOC`
    /// members before layout measures and places them.
    pub gc: bool,
    /// Whether `--export-dynamic` puts every default-visibility definition of
    /// an executable into `.dynsym`.
    ///
    /// Without it an executable exports only the names a dependency asks for.
    /// A program that loads a plugin and expects the plugin to bind back to
    /// its own symbols needs the whole set, which is what this asks for.
    pub export_dynamic: bool,
    /// The identical-code-folding mode (`--icf=all` / `--icf=safe`). Off by
    /// default; when active, [`Context::icf_sections`] runs after
    /// `--gc-sections` and folds duplicate code sections onto one
    /// representative, recording the aliases here so layout and the writer
    /// resolve folded symbols and relocations to the representative.
    pub icf: IcfMode,
    /// The `(folded -> representative)` map produced by the last ICF pass.
    /// Empty unless [`Context::icf`] is active and at least one pair of
    /// identical sections was found.
    pub folding: FoldMap,
    /// COMDAT group dedup state: signatures already kept and members of
    /// discarded duplicate groups. Scanned incrementally during symbol
    /// resolution so the surviving group's definitions win and discarded
    /// member sections are excluded from layout.
    pub groups: GroupDedup,
    /// The `.eh_frame` record map [`Context::split_eh_frame`] builds: every
    /// member's records with the output offset each was assigned, so the
    /// writer emits them packed and rebases their relocations. Empty only when
    /// no input carried an `.eh_frame`.
    pub eh: EhFramePlan,
    /// Whether any input carries a mergeable section, noticed by
    /// [`Self::assign_sections`] while it is already reading every section
    /// header. Without it the merge pass would walk every input section in the
    /// link just to discover there is nothing to do.
    has_mergeable: bool,
    /// Deduplicated `SHF_MERGE` content. Empty unless
    /// [`Self::merge_sections`] found a mergeable input section, in which case
    /// each pool is carried by one contributing section and the rest are
    /// aliased onto it.
    pub merge: MergePlan,
    /// Aggregated non-allocated `.debug_*` sections, collected during
    /// [`Self::assign_sections`]. The writer concatenates each output
    /// section's members after the loaded image and applies their
    /// `.rela.debug_*` relocations so a debugger can decode the result.
    pub debug: DebugSections,
    /// The C-identifier section names this link bounds with
    /// `__start_NAME`/`__stop_NAME`. Filled by [`crate::defsym::define`] from
    /// the names the inputs mention, then by [`Self::assign_sections`] with
    /// the sections that contribute to each. Empty for input that uses no
    /// custom section, which is every ordinary C and C++ translation unit.
    pub start_stop: StartStop,
}

impl Default for Context<'_> {
    fn default() -> Self {
        Self {
            files: Vec::new(),
            extracted: Vec::new(),
            group_lists: Vec::new(),
            bins: Vec::new(),
            symbols: SymbolTable::new(),
            sym_id: Vec::new(),
            outputs: OutputSections::new(),
            defsyms: Vec::new(),
            dep_exports: DepExports::default(),
            dep_order: Vec::new(),
            dep_versions: DepVersions::default(),
            dep_undefs: DepUndefs::default(),
            gc: false,
            export_dynamic: false,
            icf: IcfMode::None,
            folding: FoldMap::default(),
            groups: GroupDedup::default(),
            eh: EhFramePlan::default(),
            has_mergeable: false,
            merge: MergePlan::default(),
            debug: DebugSections::new(),
            start_stop: StartStop::default(),
        }
    }
}

impl<'data> Context<'data> {
    /// An empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parses and records one input object.
    pub fn add_file(&mut self, path: &Path, bytes: &'data [u8]) -> Result<()> {
        self.files.push(InputFile::open(path, bytes)?);
        self.extracted.push(None);
        self.group_lists.push(Vec::new());
        self.bins.push(Vec::new());
        self.sym_id.push(Vec::new());
        Ok(())
    }

    /// The resolved global an input symbol index names, read off the map
    /// [`Self::sym_id`] records at intern time.
    ///
    /// The id is the one `intern` returned for that very symbol, so this
    /// answers exactly what `symbols.find(name)` would -- without hashing the
    /// name or comparing its bytes. A local symbol, a section symbol and an
    /// index past the last global all read `None`, as they do through the
    /// table: neither is interned by name.
    pub fn global_id(&self, file: usize, sym_idx: u32) -> Option<SymbolId> {
        *self.sym_id.get(file)?.get(sym_idx as usize)?
    }

    /// Pass 1: fold the global symbols of every direct input into the table.
    pub fn resolve_symbols(&mut self) -> Result<()> {
        let Self {
            files,
            extracted,
            group_lists,
            bins,
            symbols: table,
            groups,
            sym_id,
            ..
        } = self;
        // Section groups first. Only the dedup decision is order-dependent --
        // the first group per signature wins -- and the groups themselves
        // were already read where every direct object parsed in parallel, so
        // the batch scan starts from the stored lists.
        let parsed = std::mem::take(group_lists);
        groups.scan_batch(&parsed)?;
        drop(parsed);
        let symbols: Vec<Result<Vec<InputSymbol<'_>>>> = extracted
            .par_iter_mut()
            .zip(bins.par_iter_mut())
            .enumerate()
            .map(|(file, (slot, bin))| {
                let mut syms = if let Some(syms) = slot.take() {
                    syms
                } else {
                    // A file added outside the parse pass has no stored
                    // partition; extract and partition it here, in the
                    // same parallel walk.
                    let mut syms = files
                        .get(file)
                        .map(InputFile::global_symbols_direct)
                        .transpose()?
                        .unwrap_or_default();
                    *bin = crate::symbol::partition_file(&mut syms);
                    syms
                };
                // The rewrite mutates in place and moves nothing, so the
                // stored shard partition stays valid.
                for sym in &mut syms {
                    if groups.is_discarded(file, sym.shndx) {
                        sym.shndx = SHN_UNDEF;
                    }
                }
                Ok(syms)
            })
            .collect::<Vec<Result<_>>>();
        let symbols = in_input_order(symbols)?;
        let taken_bins = std::mem::take(bins);
        *sym_id = table.intern_all(&symbols, &taken_bins).map_err(|err| {
            // The clash is found where only file indices are in hand; this is
            // the first place that knows what the caller named each input.
            name_duplicate_files(err, files)
        })?;
        Ok(())
    }

    /// Folds one input's global symbols into the table and records the
    /// `(file, sym_idx) -> SymbolId` mapping at intern time. Uses the symbols
    /// pre-extracted by the parallel parse pass when available; otherwise
    /// extracts on demand (archive members pulled after that pass).
    ///
    /// Section groups are scanned first so that definitions sitting in a
    /// discarded duplicate group are folded as undefined: the symbol is still
    /// interned by name (so relocations referencing it map to the right global
    /// id) but it cannot outrank the surviving group's definition.
    fn resolve_file(&mut self, file: usize) -> Result<()> {
        self.scan_groups(file)?;
        let symbols = match self.extracted.get_mut(file).and_then(Option::take)
        {
            Some(syms) => syms,
            None => self
                .files
                .get(file)
                .map(InputFile::global_symbols)
                .transpose()?
                .unwrap_or_default(),
        };
        // Build the per-file `sym_idx -> SymbolId` row in parallel with the
        // intern loop. The id each `intern` call returns is exactly what a
        // later `find(name)` probe would yield, so recording it here lets the
        // layout scan and address-resolution passes read it by index with zero
        // name probes. The row is sized to one past the last non-local symbol's
        // index; trailing locals and any out-of-range lookup map to `None`
        // (via `.get`), matching the old `build_sym_id_cache` row.
        let len = symbols
            .iter()
            .map(|s| s.sym_idx as usize)
            .max()
            .map_or(0, |m| m.saturating_add(1));
        let mut row = vec![None; len];
        for sym in symbols {
            // A definition whose backing section is a member of a discarded
            // group contributes no real definition: drop it so the surviving
            // group's copy wins resolution. The symbol is still interned by
            // name, so any relocation that references it resolves to the kept
            // copy through the global table.
            let sym = if self.groups.is_discarded(file, sym.shndx) {
                InputSymbol {
                    shndx: SHN_UNDEF,
                    ..sym
                }
            } else {
                sym
            };
            let id = self.symbols.intern(&sym, file)?;
            if let Some(slot) = row.get_mut(sym.sym_idx as usize) {
                *slot = Some(id);
            }
        }
        if let Some(target) = self.sym_id.get_mut(file) {
            *target = row;
        }
        Ok(())
    }

    /// Scans one file's `SHT_GROUP` sections into the dedup state, keeping the
    /// first COMDAT group per signature and recording later duplicates'
    /// members as discarded. A group without `GRP_COMDAT` asked for no
    /// deduplication and keeps every copy. Runs before this file's symbols are
    /// folded so discarded definitions can be suppressed.
    fn scan_groups(&mut self, file: usize) -> Result<()> {
        let Self { files, groups, .. } = self;
        let Some(input) = files.get(file) else {
            return Ok(());
        };
        let mut parsed = Vec::new();
        input.object()?.groups(&mut parsed)?;
        groups.scan(file, &parsed)
    }

    /// Pass 1b: lazily extract archive members that satisfy still-undefined
    /// references, iterating until no new member is pulled. Each pulled member
    /// is parsed and resolved, which may surface further undefined references.
    pub fn pull_archives(&mut self, archives: &[Archive<'data>]) -> Result<()> {
        // No archive, no member to pull: skip the walk over every resolved
        // symbol (and the per-name allocation) that only exists to probe
        // archive indexes.
        if archives.is_empty() {
            return Ok(());
        }
        let mut pulled: FxHashSet<(usize, u64)> = FxHashSet::default();
        loop {
            // Candidates for this round: the names still strong-undefined.
            // Which archive each name may come from is decided per archive
            // below, because a dependency's position only preempts the
            // archives it precedes.
            let undefined: Vec<Vec<u8>> = self
                .symbols
                .entries()
                .filter(|(_, s)| strong_undefined(&s.kind))
                .map(|(n, _)| n.to_vec())
                .collect();
            let mut progressed = false;
            for name in &undefined {
                for (index, archive) in archives.iter().enumerate() {
                    // The list was taken before this round began, and a member
                    // pulled since may already have defined this name, or the
                    // reference may never have been this archive's to answer.
                    if !self.still_extractable(name, archive.pos()) {
                        continue;
                    }
                    let Some(offset) = archive.lookup(name) else {
                        continue;
                    };
                    // A member is pulled at most once across the whole run.
                    if !pulled.insert((index, offset)) {
                        continue;
                    }
                    // A thin member arrives as an owned copy of the file its
                    // header named; a fat one as a borrow of the archive.
                    let member = archive.member(offset)?;
                    let label = archive
                        .member_name(offset)
                        .unwrap_or_else(|_| format!("0x{offset:x}"));
                    let path = PathBuf::from(format!("({label})"));
                    let file = self.files.len();
                    self.files
                        .push(InputFile::from_member(&path, &member[..])?);
                    self.extracted.push(None);
                    self.group_lists.push(Vec::new());
                    self.bins.push(Vec::new());
                    self.sym_id.push(Vec::new());
                    self.resolve_file(file)?;
                    progressed = true;
                    break;
                }
            }
            if !progressed {
                break;
            }
        }
        Ok(())
    }

    /// Whether an unresolved reference to `name` extracts a member of the
    /// archive at input position `archive_pos`.
    ///
    /// A strong undefined reference does, unless a shared dependency the
    /// command line reached *before this archive* already defines the name.
    /// Such a dependency is a definition, so the reference is not unresolved
    /// at all: pulling a member over it links a second copy of the same code
    /// statically while still recording `DT_NEEDED` for the library, and the
    /// program runs with two copies of whatever state that code owns.
    /// `xold main.o libgcc_s.so.1 libgcc.a` -- the gcc driver's standard
    /// pairing -- is the case that matters, and it ends with two unwinders.
    ///
    /// A dependency named after the archive is too late to stop the
    /// extraction: the member is in the link from the moment the archive is
    /// reached, which is how lld's symbol lattice answers too -- an undefined
    /// reference meeting a lazy archive symbol extracts it
    /// (`resolve(const LazySymbol &)`), and the shared symbol that arrives
    /// later cannot un-extract what is now a defined name. Command-line
    /// order, not merely membership, decides which spelling of a library
    /// supplies a name.
    ///
    /// Non-default visibility is the exception, and lld spells it out: "an
    /// undefined symbol with non default visibility must be satisfied in the
    /// same DSO". Such a reference is private to this image, no loader will
    /// ever bind it to the dependency, so the archive member is the only
    /// thing that can satisfy it and it is extracted as before.
    fn extracts_member(
        &self,
        name: &[u8],
        sym: &Symbol,
        archive_pos: u32,
    ) -> bool {
        strong_undefined(&sym.kind)
            && !(sym.visibility == STV_DEFAULT
                && self.dep_preempts(name, archive_pos))
    }

    /// Whether a shared dependency that exports `name` was reached before
    /// the input position `archive_pos`, and so holds the definition the
    /// reference must not extract a member over.
    fn dep_preempts(&self, name: &[u8], archive_pos: u32) -> bool {
        let Some(exp) = self.dep_exports.get(name) else {
            return false;
        };
        self.dep_order
            .get(exp.at.dep as usize)
            .is_some_and(|&dep_pos| dep_pos < archive_pos)
    }

    /// Whether `name` is still a reference that extracts an archive member
    /// positioned before every exporting dependency. `archive_pos` is the
    /// position of the archive the caller is about to ask.
    fn still_extractable(&self, name: &[u8], archive_pos: u32) -> bool {
        self.symbols.find(name).is_some_and(|id| {
            self.symbols
                .symbol(id)
                .is_some_and(|s| self.extracts_member(name, s, archive_pos))
        })
    }

    /// Pass 2: assign every allocated input section to an output section,
    /// and collect every non-allocated `.debug_*` section for the writer to
    /// append after the loaded image. Members of discarded COMDAT groups are
    /// skipped: they contribute no bytes, no relocations and no symbols to
    /// the output.
    pub fn assign_sections(&mut self) -> Result<()> {
        // Classification is per file and reads nothing shared, so it runs in
        // parallel; the results are folded in file order so the member lists,
        // and therefore every output offset, match the serial walk exactly.
        let groups = &self.groups;
        let start_stop = &self.start_stop;
        let classified: Vec<Result<Vec<Assigned<'_>>>> = self
            .files
            .par_iter()
            .enumerate()
            .map(|(file, input)| {
                classify_sections(file, input, groups, start_stop)
            })
            .collect::<Vec<Result<_>>>();
        for row in in_input_order(classified)? {
            for item in row {
                self.has_mergeable |= item.mergeable;
                if let Some(run) = item.run {
                    self.start_stop.record(item.file, item.section, run);
                }
                match item.kind {
                    Some(kind) => self.outputs.add(
                        kind,
                        Member {
                            file: item.file,
                            section: item.section,
                            size: item.size,
                            align: item.align,
                            priority: item.priority,
                        },
                    ),
                    None => self.debug.add(
                        item.file,
                        item.section,
                        item.shdr,
                        item.name,
                    ),
                }
            }
        }
        // The initialisation arrays are ordered by priority rather than by
        // input order, which is the one place a member's own name outranks the
        // command line. Done once here, over the finished member lists, rather
        // than per insertion: the later passes only ever remove members
        // (`--gc-sections`, ICF), and removal preserves relative order, so
        // sorting before them gives the same layout as sorting after.
        self.outputs.sort_init_fini();
        Ok(())
    }

    /// Pass 2b: when `--gc-sections` is set, drop every `SHF_ALLOC` input
    /// section not reachable from the link roots. Roots and reachability
    /// follow lld's `MarkLive` and mold's `gc-sections.cc`: the entry symbol's
    /// section, `.init_array`/`.fini_array`, `SHT_NOTE`, `SHF_GNU_RETAIN`, and
    /// the defining section of every exported definition seed a worklist; each
    /// live section's relocations then mark their defining sections to
    /// fixpoint. Unmarked members are removed from [`self.outputs`], so the
    /// downstream scan, layout and writer see a smaller image with no further
    /// changes.
    pub fn gc_sections(
        &mut self,
        entry: &[u8],
        undefined: &[&[u8]],
        mode: LinkMode,
    ) -> Result<()> {
        if self.gc {
            crate::gc::run(self, entry, undefined, mode)?;
        }
        Ok(())
    }

    /// Pass 2c: when ICF is active, fold duplicate code sections onto one
    /// representative. Runs after [`Self::gc_sections`]: garbage collection
    /// first drops the dead sections, then ICF folds the live identical ones.
    /// Folded members are removed from [`self.outputs`] and their aliases are
    /// recorded in [`self.folding`], so layout resolves their symbols and
    /// relocations to the representative's address without placing their bytes.
    pub fn icf_sections(&mut self, mode: LinkMode) -> Result<()> {
        if self.icf.is_active() {
            crate::icf::run(self, self.icf, mode)?;
        }
        Ok(())
    }

    /// Pass 2d: split every `.eh_frame` member into its CIE/FDE records, so
    /// the output section is a packed sequence of records rather than a
    /// concatenation of input sections.
    ///
    /// A zero `length` word ends an `.eh_frame`, and four zero bytes of
    /// inter-member alignment padding are exactly that word: concatenating the
    /// inputs hides every record behind the first member whose size is not a
    /// multiple of its alignment, from `.eh_frame_hdr` and from the runtime
    /// unwinder both. Packing the records removes the padding, and dropping
    /// each input's own terminator leaves one, at the end. lld and mold rebuild
    /// the section the same way and for the same reason.
    ///
    /// Under `--gc-sections` the pass also drops the FDEs describing collected
    /// functions: nothing relocates into unwind data, so [`Self::gc_sections`]
    /// keeps every `.eh_frame` member whole, and without this those records
    /// would survive with them, wasting image bytes and relocating against
    /// unplaced sections. Runs after [`Self::icf_sections`] so the member set
    /// is final.
    pub fn split_eh_frame(&mut self) -> Result<()> {
        let plan = crate::ehframe::build_plan(self)?;
        self.commit_eh(plan);
        Ok(())
    }

    /// Installs a built `.eh_frame` plan: the members' output sizes shrink
    /// to their surviving records.
    fn commit_eh(&mut self, plan: EhFramePlan) {
        self.outputs
            .set_member_sizes(OutKind::EhFrame, plan.sizes());
        self.eh = plan;
    }

    /// Pass 2e: deduplicate `SHF_MERGE` sections.
    ///
    /// Every translation unit emits its own copy of each string literal it
    /// uses, so this is most of `.rodata` on real C and C++ input. The
    /// surviving content of each pool is carried by its first contributing
    /// section; the others are aliased onto it through the same fold map ICF
    /// uses, so layout resolves their references to the carrier's address and
    /// the writer copies their bytes only once.
    ///
    /// Runs after garbage collection and before folding, so only sections
    /// that reach the output contribute and ICF can compare a relocation
    /// into a mergeable section by where the pool put its content -- the
    /// order lld finalises the two passes in
    /// (`lld/ELF/Driver.cpp`).
    pub fn merge_sections(&mut self) -> Result<()> {
        if !self.has_mergeable {
            return Ok(());
        }
        let plan = crate::merge::build_plan(self)?;
        self.commit_merge(plan);
        Ok(())
    }

    /// Installs a built merge plan: contributors fold onto their carriers
    /// and the carriers grow to the whole pool.
    fn commit_merge(&mut self, plan: MergePlan) {
        if plan.is_empty() {
            return;
        }
        let folded: FxHashSet<(usize, u16)> =
            plan.folded().iter().map(|&(at, _)| at).collect();
        for (at, carrier) in plan.folded() {
            self.folding.alias(at, carrier);
        }
        self.outputs.drop_members(&folded);
        // The carrier now stands for the whole pool.
        for (file, section, size) in plan.carrier_sizes() {
            self.outputs.set_member_size(file, section, size);
        }
        // Debug contributors are not output-section members, so they are not
        // folded; their sizes are rewritten in place and the aggregated
        // sections laid out again around them.
        self.debug
            .relayout(|file, section| plan.member_size(file, section));
        self.merge = plan;
    }

    /// Pass 2f: gather the members of each `__start_`/`__stop_` run into one
    /// contiguous block, so the bounds [`crate::defsym`] defines span those
    /// members and nothing else.
    ///
    /// Runs last of the section passes: garbage collection, folding and
    /// merging have all had their say, so the member list this reorders is
    /// the one layout will place. A link that bounds no section leaves every
    /// member exactly where it was.
    pub fn group_start_stop(&mut self) {
        if self.start_stop.is_empty() {
            return;
        }
        let Self {
            outputs,
            start_stop,
            ..
        } = self;
        outputs.group_runs(|file, section| {
            start_stop.run_of_section(file, section)
        });
    }

    /// Prints a short resolution and layout summary.
    pub fn report(&self) {
        println!("xold: {} input file(s)", self.files.len());

        let mut rows: Vec<(&[u8], &Symbol)> = self.symbols.entries().collect();
        rows.sort_by(|a, b| a.0.cmp(b.0));

        let mut defined = 0usize;
        let mut common = 0usize;
        let mut undefined = 0usize;
        println!("global symbols:");
        for (name, sym) in &rows {
            match &sym.kind {
                SymbolKind::Defined(def) => {
                    defined = defined.saturating_add(1);
                    println!(
                        "  {:<20} defined   {}",
                        show(name),
                        self.describe_definition(def)
                    );
                }
                SymbolKind::Common { size, .. } => {
                    common = common.saturating_add(1);
                    println!("  {:<20} common    size={size}", show(name));
                }
                SymbolKind::Undefined { .. } => {
                    undefined = undefined.saturating_add(1);
                    println!("  {:<20} undefined", show(name));
                }
            }
        }
        println!(
            "  -> {defined} defined, {common} common, {undefined} undefined"
        );

        println!("output sections:");
        for section in self.outputs.iter() {
            if section.members.is_empty() {
                continue;
            }
            println!(
                "  {:<10} size={:<6} align={:<3} members={}",
                section.kind.name(),
                section.size,
                section.align,
                section.members.len(),
            );
        }
    }

    /// Renders the origin of a definition for the summary.
    fn describe_definition(&self, def: &Definition) -> String {
        let file = self.files.get(def.file).map_or("?", InputFile::basename);
        match &def.source {
            DefSource::Section { index, .. } => {
                format!("{} {}", file, self.section_name(def.file, *index))
            }
            DefSource::Absolute { value } => {
                format!("{file} (absolute 0x{value:x})")
            }
        }
    }

    /// Looks up an input section name by file and section index.
    fn section_name(&self, file: usize, section: u16) -> String {
        let Some(input) = self.files.get(file) else {
            return String::new();
        };
        let Ok(obj) = input.object() else {
            return String::new();
        };
        obj.sections().get(usize::from(section)).map_or_else(
            || format!("#{section}"),
            |shdr| show(obj.section_name(shdr)),
        )
    }
}

/// Whether an unresolved reference of this kind extracts an archive member,
/// before the shared-dependency test in [`Context::extracts_member`].
///
/// Only a strong one does. An undefined *weak* reference is the feature-probe
/// idiom: the program tests the symbol's address and takes the other branch
/// when it is zero, so extracting a member to define it answers a question the
/// program asked in order to hear "no". lld states the rule the same way in
/// `Symbol::resolve` ("an undefined weak will not extract archive members").
///
/// The reference is not dropped, only made ineligible: a member extracted for
/// some other name may carry a *strong* reference to the same symbol, which
/// outranks the weak one in the fold and makes the name extractable on the
/// next round of [`Context::pull_archives`]'s fixpoint.
const fn strong_undefined(kind: &SymbolKind) -> bool {
    matches!(kind, SymbolKind::Undefined { weak: false })
}

/// Runs the full pipeline over `inputs` and prints a summary.
///
/// Archive inputs (`.a`) are set aside and mined lazily after the direct
/// objects are resolved; everything else is parsed as an ELF object up front.
pub fn link(inputs: &[Input<'_>]) -> Result<()> {
    let opened = open_all(inputs)?;
    let bytes: Vec<&[u8]> = opened.iter().map(InputBytes::bytes).collect();
    pool::run(&bytes, || {
        let (mut ctx, archives, _deps) = assemble_full(inputs, &bytes)?;
        ctx.resolve_symbols()?;
        ctx.pull_archives(&archives)?;
        // No further input can arrive, so any of the linker's own names still
        // undefined are the linker's to define.
        crate::defsym::define(&mut ctx)?;
        ctx.assign_sections()?;
        ctx.report();
        Ok(())
    })
}

/// One link: what to read, what to write, and how.
///
/// The pipeline is the same for every flavour, so this is the whole of what
/// distinguishes one from another. Build it from [`Self::exec`],
/// [`Self::shared`] or [`Self::dyn_exec`] and override the rest by field:
///
/// ```no_run
/// use std::path::Path;
///
/// use xold::{
///     input::Input,
///     linker::{Link, link_image},
/// };
///
/// let inputs = [
///     Input::Path(Path::new("crt1.o")),
///     Input::Memory {
///         name: Path::new("hello.c"),
///         bytes: &[],
///     },
///     Input::Path(Path::new("libc.so.6")),
/// ];
/// link_image(&Link {
///     gc: true,
///     ..Link::dyn_exec(&inputs, Path::new("hello"), b"/lib64/ld.so")
/// })?;
/// # Ok::<(), xold::Error>(())
/// ```
#[derive(Clone, Copy)]
// A link request is a flat description of one command line, and each of these
// is one option a caller wrote. Grouping them into sub-structures would move
// the options away from the names they are known by without making any of
// them clearer.
#[allow(clippy::struct_excessive_bools)]
pub struct Link<'a> {
    /// What to link, in command-line order. A library only satisfies
    /// references made by the inputs before it, so the order is part of the
    /// request; [`Input::Memory`] is legal at any position.
    pub inputs: &'a [Input<'a>],
    /// Where the finished image is written.
    pub output: &'a Path,
    /// Static executable, shared object, or dynamic executable.
    pub mode: LinkMode,
    /// The entry-point symbol. Empty for a shared object, which has none.
    pub entry: &'a [u8],
    /// `DT_SONAME`, for a shared object.
    pub soname: Option<&'a [u8]>,
    /// The `.interp` path, for a dynamic executable.
    pub interpreter: Option<&'a [u8]>,
    /// Drop input sections unreachable from the link roots (`--gc-sections`).
    pub gc: bool,
    /// `--export-dynamic`: put every default-visibility definition of an
    /// executable into `.dynsym`, so a plugin loaded at runtime can bind to
    /// the program's own symbols.
    pub export_dynamic: bool,
    /// Fold identical code sections (`--icf`).
    pub icf: IcfMode,
    /// `--strip-debug` or `--strip-all`: how much of the image's non-loaded
    /// information to keep.
    pub strip: Strip,
    /// `--hash-style`: which symbol hash tables a dynamic image carries.
    pub hash_style: HashStyle,
    /// `-rpath`: the runtime search path, already joined with `:` when the
    /// command line named several. Recorded as `DT_RUNPATH`.
    pub rpath: &'a [u8],
    /// `--build-id`: which digest the image's `NT_GNU_BUILD_ID` note carries,
    /// or `None` for no note.
    pub build_id: Option<&'a BuildId>,
    /// `-u NAME`: names entered as undefined before the link starts, so an
    /// archive member defining one is pulled in and garbage collection keeps
    /// it. What a program that loads its plugins by name, or a runtime whose
    /// entry is reached only from outside the image, needs.
    pub undefined: &'a [&'a [u8]],
    /// `--version-script`: which of the image's globals stay exported.
    pub version_script: Option<&'a VersionScript>,
    /// Whether a version script may name a symbol the link does not define.
    /// `--no-undefined-version` clears it, which is the default in current
    /// lld and what most build systems pass.
    pub undefined_version: bool,
    /// `-pie` or `-no-pie`, when the caller wrote one.
    ///
    /// `None` is the default and leaves the choice to the link: a dynamic
    /// executable is position-independent unless one of its inputs forces a
    /// fixed base. `Some(true)` over such an input is a contradiction and is
    /// refused rather than silently downgraded.
    pub pie: Option<bool>,
    /// `-z defs` (`--no-undefined`): refuse a link that leaves a strong
    /// reference unresolved.
    ///
    /// An executable is refused either way -- a relocation against an
    /// undefined name has no value to store. A shared object is not: its
    /// unresolved names are normally left for whoever loads it. This option
    /// says the library is meant to be self-contained, so a name no input and
    /// no dependency defines is a link error rather than a runtime one.
    pub no_undefined: bool,
    /// The `-z` keywords that describe the image: `-z now`, `-z nodelete`,
    /// `-z stack-size=N` and the rest. See [`ZOptions`].
    pub z: ZOptions,
    /// Rewrite relaxable relocations into cheaper forms.
    ///
    /// On by default, as it is in lld, mold and GNU ld: a `GOTPCRELX` site
    /// that can become a direct reference costs an indirection and a GOT slot
    /// otherwise, on every access, in every image. `--no-relax` turns it off.
    pub relax: bool,
}

/// The entry symbol an executable is given unless the caller names another.
const DEFAULT_ENTRY: &[u8] = b"_start";

impl<'a> Link<'a> {
    /// A static executable (`ET_EXEC`).
    pub const fn exec(inputs: &'a [Input<'a>], output: &'a Path) -> Self {
        Self {
            inputs,
            output,
            mode: LinkMode::Static,
            entry: DEFAULT_ENTRY,
            soname: None,
            interpreter: None,
            gc: false,
            export_dynamic: false,
            icf: IcfMode::None,
            relax: true,
            no_undefined: false,
            z: ZOptions::new(),
            pie: None,
            strip: Strip::None,
            hash_style: HashStyle::Both,
            rpath: b"",
            build_id: None,
            undefined: &[],
            version_script: None,
            undefined_version: true,
        }
    }

    /// A shared object (`ET_DYN`), which has no entry point.
    pub const fn shared(inputs: &'a [Input<'a>], output: &'a Path) -> Self {
        Self {
            mode: LinkMode::Shared,
            entry: b"",
            ..Self::exec(inputs, output)
        }
    }

    /// A dynamic executable loaded under `interpreter`. Shared-object inputs
    /// become `DT_NEEDED` dependencies resolved at runtime.
    pub const fn dyn_exec(
        inputs: &'a [Input<'a>],
        output: &'a Path,
        interpreter: &'a [u8],
    ) -> Self {
        Self {
            mode: LinkMode::DynExec,
            interpreter: Some(interpreter),
            ..Self::exec(inputs, output)
        }
    }
}

/// Runs the resolve, layout and write pipeline and stores the image.
///
/// Every flavour follows the same passes; [`Link::mode`] selects the image
/// kind and the dynamic tables that come with it. Shared-object inputs become
/// `DT_NEEDED` dependencies, which only a dynamic executable consumes.
pub fn link_image(job: &Link<'_>) -> Result<()> {
    let opened = open_all(job.inputs)?;
    let bytes: Vec<&[u8]> = opened.iter().map(InputBytes::bytes).collect();
    // The one place a link decides how many threads it runs on. Every entry
    // point that produces an ELF image funnels through here, and the inputs
    // are open by now, so the workload the choice is made from is in hand.
    pool::run(&bytes, || link_opened(job, &bytes))
}

/// The pipeline itself, over inputs that are already open.
fn link_opened<'a>(job: &Link<'_>, bytes: &'a [&'a [u8]]) -> Result<()> {
    let Link {
        inputs,
        output,
        mode,
        entry,
        soname,
        interpreter,
        gc,
        export_dynamic,
        icf,
        relax,
        no_undefined,
        z,
        pie,
        strip,
        hash_style,
        rpath,
        build_id,
        undefined,
        version_script,
        undefined_version,
    } = *job;
    let (mut ctx, archives, deps) = assemble_full(inputs, bytes)?;
    ctx.gc = gc;
    ctx.export_dynamic = export_dynamic;
    ctx.icf = icf;
    ctx.resolve_symbols()?;
    // `-u` names are entered before the archives are searched: that is what
    // makes an archive member defining one join the link.
    force_undefined(&mut ctx, undefined)?;
    ctx.pull_archives(&archives)?;
    // No further input can arrive, so any of the linker's own names still
    // undefined are the linker's to define.
    crate::defsym::define(&mut ctx)?;
    if no_undefined {
        check_no_undefined(&ctx)?;
    }
    if let Some(script) = version_script {
        apply_version_script(&mut ctx, script, undefined_version)?;
    }
    ctx.assign_sections()?;
    ctx.gc_sections(entry, undefined, mode)?;
    ctx.merge_sections()?;
    ctx.icf_sections(mode)?;
    ctx.split_eh_frame()?;
    ctx.group_start_stop();
    let target = derive_target(&ctx)?;
    check_dep_machines(&deps, target)?;
    let mut needed: Vec<&[u8]> = Vec::new();
    needed_sonames(&ctx, &deps, mode, &mut needed);
    let config = DynConfig {
        relax,
        z,
        pie,
        strip,
        hash_style,
        rpath,
        build_id,
        soname,
        interpreter,
        needed: &needed,
    };
    let mut layout = crate::layout::build(&ctx, target, mode, entry, &config)?;
    crate::writer::write(&ctx, target, &mut layout, mode, &config, output)
}

/// Enters each `-u` name as a strong undefined reference.
///
/// A name already interned keeps whatever it resolved to: `-u` asks for the
/// name to be looked for, not for a definition to be discarded.
fn force_undefined(ctx: &mut Context<'_>, names: &[&[u8]]) -> Result<()> {
    for &name in names {
        if ctx.symbols.find(name).is_some() {
            continue;
        }
        let inc = InputSymbol {
            name,
            binding: STB_GLOBAL,
            type_: STT_NOTYPE,
            visibility: STV_DEFAULT,
            shndx: SHN_UNDEF,
            value: 0,
            size: 0,
            sym_idx: 0,
            name_hash: crate::symbol::name_hash(name),
        };
        // File zero is the attribution a diagnostic would print, and no input
        // named this reference: the command line did. Nothing reads the file
        // of an undefined symbol except a duplicate-definition message, which
        // this can never provoke.
        ctx.symbols.intern(&inc, 0)?;
    }
    Ok(())
}

/// Applies a version script: every defined global the script's `local` list
/// claims is hidden, so it stays out of `.dynsym`.
///
/// This runs after resolution and before layout, which is what keeps the
/// sizing pass, the dynamic symbol table and the output symbol table from
/// disagreeing about which names are exported: all three read the visibility
/// this sets.
///
/// With `undefined_version` false (`--no-undefined-version`), a name the
/// script writes out in full and the link does not define is an error. A
/// script naming a symbol that is not there is nearly always a stale export
/// list, and the library it produces is missing an export its consumers
/// expect.
fn apply_version_script(
    ctx: &mut Context<'_>,
    script: &VersionScript,
    undefined_version: bool,
) -> Result<()> {
    if !undefined_version {
        for name in script.exact_names() {
            let defined = ctx
                .symbols
                .find(name)
                .and_then(|id| ctx.symbols.symbol(id))
                .is_some_and(|sym| {
                    !matches!(sym.kind, SymbolKind::Undefined { .. })
                });
            if !defined {
                return Err(Error::CommandLine(format!(
                    "the version script names `{}`, which this link does not \
                     define; --no-undefined-version is in force",
                    String::from_utf8_lossy(name)
                )));
            }
        }
    }
    if !script.hides_anything() {
        return Ok(());
    }
    // Only definitions are hidden. A version script says what this image
    // exports; an undefined name is what it imports, and hiding one would
    // make the reference unresolvable rather than private -- which is exactly
    // what happened to `_Unwind_Resume` in a `local: *;` library.
    let hide: Vec<SymbolId> = ctx
        .symbols
        .entries()
        .filter(|(_, sym)| !matches!(sym.kind, SymbolKind::Undefined { .. }))
        .filter(|(name, _)| script.visibility(name) == VersionVisibility::Local)
        .filter_map(|(name, _)| ctx.symbols.find(name))
        .collect();
    for id in hide {
        ctx.symbols.hide(id);
    }
    Ok(())
}

/// Refuses a link that leaves a strong reference unresolved, as `-z defs`
/// asks.
///
/// A name is unresolved when no input defines it and no dependency exports
/// it. Weak references are not: a weak undefined name resolves to zero by
/// design, which is what the reference asked for.
///
/// Every offender is collected rather than the first one found, so the
/// diagnostic says how much is missing, and the order is the symbol table's
/// -- first-seen input order -- so two runs of the same link report the same
/// name first.
fn check_no_undefined(ctx: &Context<'_>) -> Result<()> {
    let mut missing: Vec<&[u8]> = Vec::new();
    for (name, sym) in ctx.symbols.entries() {
        if !matches!(sym.kind, SymbolKind::Undefined { weak: false }) {
            continue;
        }
        if ctx.dep_exports.contains(name) {
            continue;
        }
        missing.push(name);
    }
    let Some(&first) = missing.first() else {
        return Ok(());
    };
    let name = String::from_utf8_lossy(first);
    let report = match missing.len() {
        1 => name.into_owned(),
        n => format!("{name} (and {} more)", n - 1),
    };
    Err(Error::UndefinedReference(report))
}

/// Links `paths` into an executable written to `output`.
///
/// `entry` names the entry-point symbol. When `gc` is set, unreachable
/// `SHF_ALLOC` input sections are dropped before layout (`--gc-sections`).
/// When `relax` is set, relaxable relocations (x86-64 `GOTPCRELX`) are
/// rewritten into cheaper PC-relative forms; otherwise the link is
/// byte-identical to the default.
///
/// A convenience over [`link_image`] for the common case of linking files;
/// pass [`Link`] directly to mix in inputs already held in memory.
pub fn link_to(
    paths: &[PathBuf],
    output: &Path,
    entry: &[u8],
    gc: bool,
    icf: IcfMode,
    relax: bool,
) -> Result<()> {
    let inputs = Input::from_paths(paths);
    link_image(&Link {
        entry,
        gc,
        icf,
        relax,
        ..Link::exec(&inputs, output)
    })
}

/// Links `paths` into a shared object (`ET_DYN`) written to `output`.
///
/// `soname`, if given, is recorded as `DT_SONAME`. When `gc` is set,
/// unreachable `SHF_ALLOC` input sections are dropped before layout
/// (`--gc-sections`); every definition the object exports is a root, since the
/// loader and `dlopen` reach those through `.dynsym`. When `relax` is set,
/// relaxable relocations are rewritten into cheaper forms.
///
/// A convenience over [`link_image`] for the common case of linking files;
/// pass [`Link`] directly to mix in inputs already held in memory.
pub fn link_shared(
    paths: &[PathBuf],
    output: &Path,
    soname: Option<&[u8]>,
    gc: bool,
    icf: IcfMode,
    relax: bool,
) -> Result<()> {
    let inputs = Input::from_paths(paths);
    link_image(&Link {
        soname,
        gc,
        icf,
        relax,
        ..Link::shared(&inputs, output)
    })
}

/// Links `paths` into a position-independent dynamic executable (`ET_DYN`)
/// written to `output`.
///
/// Shared object inputs (`.so`) are recorded as `DT_NEEDED` dependencies; the
/// executable resolves their exported functions at runtime through a PLT.
/// `entry` names the entry symbol; `interpreter` is the `.interp` path the
/// kernel hands to `ld.so`. When `gc` is set, unreachable `SHF_ALLOC` input
/// sections are dropped before layout (`--gc-sections`). When `relax` is set,
/// relaxable relocations are rewritten into cheaper forms.
///
/// A convenience over [`link_image`] for the common case of linking files;
/// pass [`Link`] directly to mix in inputs already held in memory.
pub fn link_dyn_exec(
    paths: &[PathBuf],
    output: &Path,
    entry: &[u8],
    interpreter: &[u8],
    gc: bool,
    icf: IcfMode,
    relax: bool,
) -> Result<()> {
    let inputs = Input::from_paths(paths);
    link_image(&Link {
        entry,
        gc,
        icf,
        relax,
        ..Link::dyn_exec(&inputs, output, interpreter)
    })
}

/// One input section that survived classification, with the output list it
/// belongs to. Produced per file by [`classify_sections`] and folded into the
/// context in file order.
struct Assigned<'a> {
    file: usize,
    section: u16,
    /// On-disk size (`sh_size`) and alignment (`sh_addralign`, at least 1).
    /// Copied out here because the header is in cache during classification
    /// and cold again by the time the serial fold runs.
    size: u64,
    align: u64,
    /// The output section this member joins, or `None` for a non-allocated
    /// `.debug_*` section. Decided here so the serial fold is a plain push.
    kind: Option<OutKind>,
    /// The `__start_`/`__stop_` run this member contributes to, if the link
    /// bounds its name. Looked up here because the section name is in hand.
    run: Option<usize>,
    /// Whether the section is `SHF_MERGE`. Noted here because the flags are
    /// already in hand; it saves the merge pass a walk over every input
    /// section in the link just to discover there is nothing to merge.
    mergeable: bool,
    /// The initialisation priority the section name encodes, for the three
    /// array sections that are ordered by it. Read here because the name is
    /// in hand; see [`array_priority`].
    priority: i32,
    /// Read only on the debug path, which needs the whole header.
    shdr: &'a Shdr64,
    name: &'a [u8],
}

/// The one section name that changes an allocated section's routing.
const EH_FRAME: &[u8] = b".eh_frame";

/// The two sections whose fragments form one function apiece.
const INIT: &[u8] = b".init";
const FINI: &[u8] = b".fini";

/// The prefix shared by the relocated-read-only-data family: `.data.rel.ro`
/// itself, `.data.rel.ro.local`, and the per-object `.data.rel.ro.*` names a
/// `-fdata-sections` compiler emits. A prefix is what lld matches too, after
/// it has folded the suffix away (`isRelRoDataSection`, ELF/Writer.cpp:557).
const DATA_REL_RO: &[u8] = b".data.rel.ro";

/// Which of the specially named sections this is, if any. A name is only
/// compared for a section whose flags could carry it, so an ordinary
/// `.text.*` or `.data.*` costs one length check.
fn named_section(obj: ObjectFile<'_>, shdr: &Shdr64) -> Named {
    if obj.section_name_is(shdr, EH_FRAME) {
        return Named::EhFrame;
    }
    let flags = shdr.sh_flags.get();
    if flags & SHF_EXECINSTR != 0 {
        if obj.section_name_is(shdr, INIT) {
            return Named::Init;
        }
        if obj.section_name_is(shdr, FINI) {
            return Named::Fini;
        }
    }
    // Only initialised writable data can be `.data.rel.ro`: the family holds
    // relocated pointers, so it is `SHT_PROGBITS` and never thread-local.
    // Gating on that keeps `.text.*`, `.rodata.*` and `.bss.*` from paying for
    // the compare.
    if flags & (SHF_WRITE | SHF_TLS) == SHF_WRITE
        && shdr.sh_type.get() == SHT_PROGBITS
        && obj.section_name_starts_with(shdr, DATA_REL_RO)
    {
        return Named::DataRelRo;
    }
    Named::Other
}

/// The initialisation priority of a section bound for one of the three array
/// output sections, or [`NO_PRIORITY`] for anything else.
///
/// Only those three are ordered by priority, and only they pay for the name:
/// the classifier deliberately avoids materialising section names, so this
/// asks the output kind first and reads the name for at most a handful of
/// sections per input.
fn array_priority(
    kind: Option<OutKind>,
    obj: ObjectFile<'_>,
    shdr: &Shdr64,
) -> i32 {
    match kind {
        Some(
            OutKind::PreinitArray | OutKind::InitArray | OutKind::FiniArray,
        ) => init_priority(obj.section_name(shdr)),
        _ => NO_PRIORITY,
    }
}

/// Classifies one input's sections, dropping the ones the link ignores:
/// the null section, members of discarded COMDAT groups, and every
/// non-allocated section that is not DWARF (`.symtab`, `.strtab`, comments --
/// xold synthesises its own tables).
fn classify_sections<'a>(
    file: usize,
    input: &'a InputFile<'_>,
    groups: &GroupDedup,
    start_stop: &StartStop,
) -> Result<Vec<Assigned<'a>>> {
    let obj = input.object()?;
    let sections = obj.sections();
    // Roughly half the sections of a `-ffunction-sections` input are its
    // relocation sections, which are dropped here; sizing to the full count
    // would allocate several times what survives, on every input.
    let mut out = Vec::with_capacity(sections.len() / 2);
    // Whether this input holds GCC LTO bytecode, and whether it holds any
    // real content beside it. See `refuse_thin_lto`.
    let (mut lto, mut content) = (false, false);
    for (index, shdr) in sections.iter().enumerate() {
        if shdr.sh_type.get() == SHT_NULL {
            continue;
        }
        let section = u16::try_from(index)
            .map_err(|_| Error::OutOfRange("section index"))?;
        if groups.is_discarded(file, section) {
            continue;
        }
        if obj.section_name_starts_with(shdr, LTO_PREFIX) {
            lto = true;
        }
        if alloc_content(shdr) {
            content = true;
        }
        // `SHF_EXCLUDE` is the producer asking that the section not survive
        // the final link -- `.text.unlikely` under a profile, the `.discard`
        // families the kernel and libc build with. lld drops it wherever it
        // finds one, keeping it only for `-r`, which relinks rather than
        // links (`lld/ELF/InputFiles.cpp`); this linker only
        // produces final links, so the drop is unconditional.
        if shdr.sh_flags.get() & SHF_EXCLUDE != 0 {
            continue;
        }
        // A compressed section's bytes are a deflate stream behind an
        // `Elf64_Chdr`, and its relocation offsets address the content the
        // stream expands to. Concatenating the stream and patching it at
        // those offsets produces debug info that no reader can decompress,
        // with nothing to say it happened -- so say it. lld decompresses
        // transparently (`InputFiles.cpp`, `contentMaybeDecompress`), which
        // is the answer to implement when such objects need to link.
        if shdr.sh_flags.get() & SHF_COMPRESSED != 0 {
            return Err(Error::Format(
                "compressed section (SHF_COMPRESSED): decompression is not \
                 implemented",
            ));
        }
        // The alignment is used as a mask by every placement cursor, so it
        // has to be one: a power of two, and small enough that rounding a
        // cursor to it cannot run off the end of the address space.
        crate::util::check_align(
            shdr.sh_addralign.get(),
            "section alignment is not a usable power of two",
        )?;
        // The extent has to be in the file before anything sizes a region
        // from it. A `SHT_NOBITS` section occupies no file bytes and states
        // its size freely; every other kind is bounded by what was mapped,
        // and a size the file cannot back reaches the writer as a slice it
        // cannot take. Reading the payload here is a bounds check on the
        // mapping, not a copy.
        if shdr.sh_type.get() != SHT_NOBITS {
            obj.section_data(shdr)?;
        }
        let alloc = shdr.sh_flags.get() & SHF_ALLOC != 0;
        if !alloc {
            // The legacy compressed spelling signals its deflate stream by
            // its name alone, so concatenating it would ship debug info no
            // reader can decompress. Refuse instead, as the
            // `SHF_COMPRESSED` spelling already does.
            if obj.section_name_starts_with(shdr, crate::debug::ZDEBUG_PREFIX) {
                return Err(Error::Format(
                    "legacy zlib-compressed debug section (.zdebug_*): \
                     decompression is not implemented",
                ));
            }
            if !keeps_non_alloc(&obj, shdr) {
                continue;
            }
        }
        let kind =
            alloc.then(|| OutKind::from_shdr(shdr, named_section(obj, shdr)));
        out.push(Assigned {
            file,
            section,
            size: shdr.sh_size.get(),
            align: shdr.sh_addralign.get().max(1),
            kind,
            run: alloc.then(|| start_stop.run_of(obj, shdr)).flatten(),
            mergeable: shdr.sh_flags.get() & SHF_MERGE != 0,
            priority: array_priority(kind, obj, shdr),
            shdr,
            // The aggregated debug sections are grouped by name; nothing reads
            // it for an allocated member.
            name: if alloc { &[] } else { obj.section_name(shdr) },
        });
    }
    refuse_thin_lto(lto, content)?;
    Ok(out)
}

/// Whether a section holds content the image would carry: allocated, not
/// empty, and not a note.
///
/// `SHT_NOBITS` counts -- a `.bss` of a hundred bytes is content the program
/// has, even though the file does not store it. An allocated note does not:
/// every object a current gcc emits carries `.note.gnu.property`, including
/// the LTO objects that hold no code at all, so counting one would answer
/// that every input has content.
fn alloc_content(shdr: &Shdr64) -> bool {
    shdr.sh_flags.get() & SHF_ALLOC != 0
        && shdr.sh_size.get() != 0
        && shdr.sh_type.get() != SHT_NOTE
}

/// Refuses an input whose only content is GCC LTO bytecode.
///
/// `gcc -flto` emits an object whose `.text` and `.data` are empty and whose
/// real code lives in `SHF_EXCLUDE`-flagged `.gnu.lto_*` sections, for the
/// plugin to compile. Dropping those sections, which the exclude flag asks
/// for, leaves an object contributing nothing -- and the link then fails
/// somewhere else entirely, reporting `main` as undefined.
///
/// `gcc -flto -ffat-lto-objects` emits both: real code and the bytecode
/// beside it. That one links exactly as a non-LTO object does, which is why
/// the test is for content rather than for the bytecode's presence.
fn refuse_thin_lto(lto: bool, content: bool) -> Result<()> {
    if lto && !content {
        return Err(Error::Format(
            "input holds GCC LTO bytecode and nothing else: link-time \
             optimisation is not implemented, so build with -fno-lto or \
             -ffat-lto-objects",
        ));
    }
    Ok(())
}

/// Whether a non-allocated section is one the image carries through.
///
/// The loader never reads any of them, but their readers are the tools around
/// the program: a debugger reads `.debug_*`, `rustc` reads a Rust library's
/// metadata out of `.rustc`, and a build system reads `.comment` to see what
/// produced the file. A linker that keeps only the sections it recognises
/// drops the rest silently, and the failure surfaces far away -- a proc-macro
/// library links clean and then cannot be loaded, because the metadata the
/// compiler put in it is gone.
///
/// So the rule is by kind rather than by name: any non-empty `SHT_PROGBITS`
/// section is concatenated with its same-named peers and written out. The
/// tables the linker rebuilds itself -- symbol tables, string tables,
/// relocations, groups -- are other section types and never reach here.
/// `.note.GNU-stack` is the one name excluded: it is an empty marker that
/// says the object wants no executable stack, which `PT_GNU_STACK` already
/// states, and every linker drops it.
fn keeps_non_alloc(obj: &ObjectFile<'_>, shdr: &Shdr64) -> bool {
    shdr.sh_type.get() == SHT_PROGBITS
        && shdr.sh_size.get() != 0
        && obj.section_name(shdr) != GNU_STACK_NOTE
}

/// The empty marker section a compiler emits to ask for a non-executable
/// stack.
const GNU_STACK_NOTE: &[u8] = b".note.GNU-stack";

/// The name prefix of the sections a GCC LTO object keeps its bytecode in.
const LTO_PREFIX: &[u8] = b".gnu.lto_";

/// Whether `bytes` is an `ET_DYN` shared object (a loadable dependency), as
/// opposed to a relocatable object or archive.
pub(super) fn is_shared_object(bytes: &[u8]) -> bool {
    ObjectFile::parse(bytes)
        .is_ok_and(|obj| obj.header().e_type.get() == ET_DYN)
}

/// Replaces the file indices in a duplicate-symbol error with the paths the
/// caller named.
///
/// The symbol table folds inputs by index and has no names to report with; a
/// message that identifies neither definer leaves the reader to find them.
/// lld prints both, with archive and member attribution.
fn name_duplicate_files(err: Error, files: &[InputFile<'_>]) -> Error {
    let Error::DuplicateSymbol(d) = err else {
        return err;
    };
    let path = |raw: &str| {
        raw.parse::<usize>()
            .ok()
            .and_then(|i| files.get(i))
            .map_or_else(|| raw.to_string(), |f| f.path().display().to_string())
    };
    Error::duplicate_symbol(d.name, path(&d.first), path(&d.second))
}

/// Reduces a per-input parallel result to the first failure in input order.
///
/// `collect::<Result<_>>` over a parallel iterator surfaces whichever failure
/// the runtime noticed first, so a link with two malformed inputs named `a.o`
/// on one run and `b.o` on the next. Output bytes were never at stake -- a
/// failed link writes nothing -- but a diagnostic that changes between
/// identical runs is the same defect as an image that does, and it is the one
/// a person actually reads.
///
/// The parse itself stays parallel; only the reduction is ordered. The cost is
/// that a failing link no longer short-circuits, so every input is parsed
/// before the first error is reported. That is work spent only on links that
/// were going to fail.
pub(crate) fn in_input_order<T>(results: Vec<Result<T>>) -> Result<Vec<T>> {
    results.into_iter().collect()
}

/// The single link target folded from the inputs' `e_machine`; see
/// [`assemble`]. Reachable for the ICF pass, which keys code comparison on
/// the architecture.
pub(crate) fn derive_target(ctx: &Context<'_>) -> Result<crate::reloc::Target> {
    derive_target_inner(ctx)
}

/// Derives the output ELF header's `e_flags` by folding every input's.
///
/// The word says which variant of the architecture the image is built for --
/// on RISC-V, the floating-point calling convention and whether compressed
/// instructions are present -- so it is part of the ABI, not a copy of one
/// input's header: glibc's loader refuses a hard-float program that reports the
/// soft-float ABI. Only relocatable inputs are folded, matching lld, which
/// walks `ctx.objectFiles`; a shared dependency is not linked in and so does
/// not shape what the image is.
pub(crate) fn derive_eflags(
    ctx: &Context<'_>,
    target: crate::reloc::Target,
) -> Result<u32> {
    let mut flags: Option<u32> = None;
    for input in &ctx.files {
        let incoming = input.object()?.header().e_flags.get();
        let merged = target.merge_eflags(flags, incoming).map_err(|what| {
            Error::incompatible_input(input.path().display().to_string(), what)
        })?;
        flags = Some(merged);
    }
    Ok(flags.unwrap_or(0))
}
