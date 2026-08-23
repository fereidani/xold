//! Dynamic linking emission: the synthetic tables a shared object or a dynamic
//! executable needs so the runtime loader can relocate the image and resolve
//! its imports and exports.
//!
//! In a static link every address is fixed at link time and no dynamic
//! relocation survives. In a shared link the image is position independent
//! (`ET_DYN`, base zero): the loader chooses the load base, so anything that
//! stores an absolute pointer must be described by a dynamic relocation
//! instead of being patched outright. This module builds the synthetic
//! sections that contract requires:
//!
//! - `.dynsym` / `.dynstr`: the exported global/weak symbols (and any globals
//!   imported through a GOT slot), with names in a parallel string table.
//! - `.hash`: the classic `SysV` symbol hash table over `.dynsym`, which the
//!   loader probes to resolve symbols by name.
//! - `.rela.dyn`: one `R_*_RELATIVE` per internal pointer slot and one
//!   `R_*_GLOB_DAT` per global GOT slot.
//! - `.dynamic`: the `_DYNAMIC` array pointing the loader at every table above.
//!
//! The plan is built in two phases. [`DynamicPlan::build`] produces the symbol
//! and relocation bytes plus their sizes; layout then places the regions and
//! stamps section indices; [`DynamicPlan::finalize`] then serialises the
//! `.dynamic` array from the now-known addresses.
//!
//! The work is split across focused submodules: [`dynsym`] owns the symbol
//! table and the `SysV`/GNU hash tables, [`version`] owns symbol versioning,
//! [`rela`] builds `.rela.dyn`, [`dynamic_section`] serialises `.dynamic`, and
//! [`sizes`] is the sizing-only probe pass. Types, placement and the
//! build/finalise orchestration live here.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::util::align_up;

mod dynamic_section;
mod dynsym;
mod rela;
mod sizes;
mod version;

use dynamic_section::{
    array_refs, build_dynamic, init_fini_refs, intern_needed, plt_refs,
};
use dynsym::{
    DynSym, add_imports, build_dynstr, build_gnu_hash, build_hash, names_by_id,
};
use rela::{build_rela_dyn, count_relative};
use sizes::{Sizing, compute_sizes};

use crate::{
    buildid::BuildId,
    elf::{
        Dyn64, Rela64, Sym64,
        constants::{
            DF_1_GLOBAL, DF_1_INITFIRST, DF_1_INTERPOSE, DF_1_NODELETE,
            DF_1_NOOPEN, DF_1_NOW, DF_1_ORIGIN, DF_BIND_NOW, DF_ORIGIN,
            SHN_UNDEF,
        },
    },
    error::Result,
    layout::{Layout, Region, Sect},
    linker::Context,
    reloc::{DynRelocs, Target},
    symbol::{SymbolId, SymbolKind, exports_to_dynsym},
};

/// On-disk size of one `Sym64` entry; recorded as `DT_SYMENT`.
pub(super) const SYM_SIZE: u64 = core::mem::size_of::<Sym64>() as u64;
/// On-disk size of one `Rela64` entry; recorded as `DT_RELAENT`.
pub(super) const RELA_SIZE: usize = core::mem::size_of::<Rela64>();
/// On-disk size of one `Dyn64` (`.dynamic` entry).
const DYNAMIC_SIZE: u64 = core::mem::size_of::<Dyn64>() as u64;

/// The initialisation function `DT_INIT` names. The runtime calls it before it
/// walks `.init_array`; the C runtime assembles its body from the `.init`
/// fragments in crti.o and crtn.o.
///
/// This is the name lld defaults `ctx.arg.init` to, and the point an
/// `-init NAME` command-line option would attach once xold grows one.
pub const INIT_SYMBOL: &[u8] = b"_init";
/// The termination function `DT_FINI` names, the `.fini` counterpart of
/// [`INIT_SYMBOL`] and the attachment point for a `-fini NAME` option.
pub const FINI_SYMBOL: &[u8] = b"_fini";

/// The link flavour, threaded through layout and the writer so the mode
/// switch lives at the `link_to` / `link_shared` boundary.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum LinkMode {
    /// `ET_EXEC`: every address fixed at link time, GOT prefilled.
    Static,
    /// `ET_DYN` shared object: position independent, loader fills GOT via
    /// dynamic relocs, no entry point, exports its globals.
    Shared,
    /// Dynamic executable: loader applies `RELATIVE` and `JUMP_SLOT`
    /// relocations, has an entry (`_start`) and a `PT_INTERP`, and resolves
    /// imported functions through a PLT at runtime. Position independent
    /// (`ET_DYN`) by default; drops to a fixed base (`ET_EXEC`) when it
    /// copy-relocates an absolute data import.
    DynExec,
}

impl LinkMode {
    /// Whether this mode produces a position-independent `ET_DYN` image with
    /// a dynamic table (either `-shared` or a dynamic executable).
    pub const fn is_dynamic(self) -> bool {
        matches!(self, Self::Shared | Self::DynExec)
    }
}

/// Dynamic-link inputs that shape the synthetic tables, threaded from the
/// driver through layout and the dynamic plan.
///
/// Static links pass the default (all fields empty); the fields are only
/// consulted when [`LinkMode::is_dynamic`] holds.
#[derive(Clone, Copy, Default)]
pub struct DynConfig<'a> {
    /// `DT_SONAME`: the name a `-shared` object advertises for itself.
    pub soname: Option<&'a [u8]>,
    /// `.interp` / `PT_INTERP`: the NUL-terminated interpreter path a dynamic
    /// executable is loaded under. Only emitted for [`LinkMode::DynExec`].
    pub interpreter: Option<&'a [u8]>,
    /// `DT_NEEDED`: the sonames of the shared objects this image imports
    /// symbols from. The loader loads them before resolving the image's
    /// `GLOB_DAT` and `JUMP_SLOT` relocations, so a shared object records
    /// its own dependencies exactly as an executable does.
    pub needed: &'a [&'a [u8]],
    /// The `-z` keywords that reach the image.
    pub z: ZOptions,
    /// How much of the symbolic and debug information the image keeps.
    pub strip: Strip,
    /// Which symbol hash tables the image carries.
    pub hash_style: HashStyle,
    /// `--build-id`: which digest the `NT_GNU_BUILD_ID` note carries. `None`
    /// for a link that asked for no note, which is the default.
    pub build_id: Option<&'a BuildId>,
    /// `-rpath`: the runtime search path recorded as `DT_RUNPATH`, already
    /// joined with `:` when several were written. Empty for a link that
    /// named none.
    pub rpath: &'a [u8],
    /// `-pie` or `-no-pie`, when the caller wrote one. `None` leaves the
    /// choice to the link: a dynamic executable is position-independent
    /// unless an absolute reference in its input forces a fixed base.
    pub pie: Option<bool>,
    /// Whether relaxation is on for this link.
    ///
    /// The scan reads it to decline a GOT slot for a site the writer will
    /// certainly rewrite into one that reads no slot. With relaxation off,
    /// every site keeps its slot, because every site keeps its GOT access.
    pub relax: bool,
}

/// Which of the two symbol hash tables a dynamic image carries.
///
/// The loader looks a name up through one of them. `.gnu.hash` is what every
/// current loader prefers -- it carries a bloom filter, so a name the object
/// does not define is usually rejected without a probe -- and `.hash` is the
/// original System V table, which is what an old or minimal loader reads.
/// Carrying both costs bytes and answers both readers, which is why it is the
/// default here.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum HashStyle {
    /// `--hash-style=both`: emit `.hash` and `.gnu.hash`.
    #[default]
    Both,
    /// `--hash-style=sysv`: emit `.hash` alone.
    Sysv,
    /// `--hash-style=gnu`: emit `.gnu.hash` alone.
    Gnu,
}

impl HashStyle {
    /// Whether the System V `.hash` table is emitted.
    pub const fn sysv(self) -> bool {
        matches!(self, Self::Both | Self::Sysv)
    }

    /// Whether `.gnu.hash` is emitted.
    pub const fn gnu(self) -> bool {
        matches!(self, Self::Both | Self::Gnu)
    }
}

/// How much of the image's non-loaded information the writer keeps.
///
/// None of it is loaded, so none of it changes what the program does; what it
/// changes is what a debugger, a profiler or a crash reporter can say about
/// the program afterwards, and how large the file is.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum Strip {
    /// Keep everything: the symbol table, its strings, and the `.debug_*`
    /// sections. The default.
    #[default]
    None,
    /// `--strip-debug`: drop the `.debug_*` sections and keep the symbol
    /// table, which is what a backtrace needs.
    Debug,
    /// `--strip-all`: drop those and the symbol table with them. The dynamic
    /// symbol table stays: the loader reads it, so an image without one does
    /// not run.
    All,
}

/// The `-z` keywords that describe the image rather than the link.
///
/// Each one sets a bit in `DT_FLAGS` or `DT_FLAGS_1`, or a field of a program
/// header, so each is carried through to the writer rather than acted on at
/// parse time. The keywords whose behaviour this linker already has -- `-z
/// noexecstack`, `-z relro`, `-z lazy` -- set nothing here: the image is
/// already what they ask for.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Default)]
pub struct ZOptions {
    /// `-z now`: bind every relocation at load time.
    pub now: bool,
    /// `-z origin`: `$ORIGIN` may appear in this image's runtime paths.
    pub origin: bool,
    /// `-z nodelete`: the object stays mapped once loaded.
    pub nodelete: bool,
    /// `-z nodlopen`: the object may not be `dlopen`ed.
    pub nodlopen: bool,
    /// `-z initfirst`: the object's initialisers run first.
    pub initfirst: bool,
    /// `-z interpose`: the object's definitions preempt every other's.
    pub interpose: bool,
    /// `-z global`: the object's symbols join the global search scope.
    pub global: bool,
    /// `-z stack-size=N`: the stack `PT_GNU_STACK` asks the loader for.
    /// `None` leaves the field zero, which means the system default.
    pub stack_size: Option<u64>,
}

impl ZOptions {
    /// No keyword given: the image every one of them would modify.
    ///
    /// `Default` says the same thing, and is what a caller building a
    /// [`Self`] by hand should use; this exists because [`crate::linker::Link`]
    /// builds its defaults in `const` context, where a trait method cannot be
    /// called.
    pub const fn new() -> Self {
        Self {
            now: false,
            origin: false,
            nodelete: false,
            nodlopen: false,
            initfirst: false,
            interpose: false,
            global: false,
            stack_size: None,
        }
    }

    /// The `DT_FLAGS` word these keywords ask for, before the bits the link
    /// itself decides are added.
    pub const fn flags(self) -> u64 {
        let mut bits = 0;
        if self.now {
            bits |= DF_BIND_NOW;
        }
        if self.origin {
            bits |= DF_ORIGIN;
        }
        bits
    }

    /// The `DT_FLAGS_1` word these keywords ask for.
    pub const fn flags_1(self) -> u64 {
        let mut bits = 0;
        if self.now {
            bits |= DF_1_NOW;
        }
        if self.origin {
            bits |= DF_1_ORIGIN;
        }
        if self.nodelete {
            bits |= DF_1_NODELETE;
        }
        if self.nodlopen {
            bits |= DF_1_NOOPEN;
        }
        if self.initfirst {
            bits |= DF_1_INITFIRST;
        }
        if self.interpose {
            bits |= DF_1_INTERPOSE;
        }
        if self.global {
            bits |= DF_1_GLOBAL;
        }
        bits
    }
}

/// The synthetic sections a shared object adds on top of the static layout.
///
/// Regions stay zeroed after [`DynamicPlan::build`]; layout fills them once
/// placement is known, then [`DynamicPlan::finalize`] builds `.dynamic`.
pub struct DynamicPlan {
    /// `.dynsym` entries: a null entry followed by exports and imports.
    pub dynsym: Vec<Sym64>,
    /// `.dynstr` bytes, with a leading NUL at offset zero.
    pub dynstr: Vec<u8>,
    /// `.hash` bytes: `nbucket`, `nchain`, `bucket[]`, `chain[]`.
    pub hash: Vec<u8>,
    /// `.gnu.hash` bytes: header, bloom filter, buckets, chain.
    pub gnu_hash: Vec<u8>,
    /// `.gnu.version` bytes: one `u16` per `.dynsym` entry (null entry first).
    /// Empty when no version information is emitted.
    pub versym: Vec<u8>,
    /// `.gnu.version_r` bytes: serialised `Elf_Verneed` + `Elf_Vernaux`
    /// records. Empty when no versioned dependency is referenced.
    pub verneed: Vec<u8>,
    /// `.rela.dyn` entries, with all `R_*_RELATIVE` first.
    pub rela_dyn: Vec<Rela64>,
    /// `.dynamic` entries; filled by [`DynamicPlan::finalize`].
    pub dynamic: Vec<Dyn64>,
    /// Offset of the SONAME string within `.dynstr`, if `-soname` was given.
    pub soname_off: Option<u32>,
    /// Offsets of the `DT_NEEDED` soname strings within `.dynstr`.
    pub needed_off: Vec<u32>,
    /// `DT_RUNPATH`: the `.dynstr` offset of the runtime search path, when
    /// one was given.
    pub rpath_off: Option<u32>,
    /// Number of `R_*_RELATIVE` entries at the front of `.rela.dyn`.
    pub relative_count: u32,
    /// Whether any version information (versym + VERNEED) is emitted. The
    /// dynamic table omits the `DT_VERSYM`/`DT_VERNEED`/`DT_VERNEEDNUM` tags
    /// when false.
    pub has_versions: bool,
    /// The number of `Elf_Verneed` records, recorded as `DT_VERNEEDNUM`.
    pub verneed_count: u32,
    /// The target's dynamic relocation types, captured so the PLT emitter
    /// (which reads this plan) emits the right `R_*_JUMP_SLOT` without
    /// having to thread `Target` separately.
    pub relocs: DynRelocs,
    /// Resolved symbol id -> 1-based dynsym index, for every imported symbol
    /// (whether reached through a GOT slot or a PLT entry). The PLT emitter
    /// reads this to name the symbol in each `JUMP_SLOT` relocation.
    pub import_index: FxHashMap<SymbolId, u32>,
    /// Section byte sizes, computed from the symbol set so they are stable
    /// across the address-resolution pass. Layout uses these to reserve space
    /// before the bytes are serialised.
    pub sizes: DynSizes,
    /// The by-name data-relocation imports the sizing survey found, carried
    /// from the probe to the full build so the one parallel walk over every
    /// allocated section's relocations is not repeated after address
    /// resolution. The survey classifies references, not addresses, so its
    /// answer cannot change between the two.
    data_imports: Vec<SymbolId>,
    /// The placed regions. Zeroed until layout assigns them. Section-header
    /// indices are not stored here: layout numbers every region of the image,
    /// synthetic or not, in [`crate::layout::Sect`] order.
    pub regions: DynRegions,
}

/// Where each synthetic section landed in the image.
///
/// Grouped so the probe/rebuild handoff in [`build_plan`] moves the placement
/// in one assignment: the probe plan exists only to size these sections, and
/// the real plan inherits every address the probe was placed at.
#[derive(Clone, Copy, Default)]
pub struct DynRegions {
    pub gnu_hash: Region,
    pub hash: Region,
    pub dynsym: Region,
    pub dynstr: Region,
    pub versym: Region,
    pub verneed: Region,
    pub rela_dyn: Region,
    pub dynamic: Region,
}

impl DynRegions {
    /// The placement of one synthetic section, or a zeroed region for a
    /// section this plan does not own.
    pub fn get(&self, sect: Sect) -> Region {
        match sect {
            Sect::GnuHash => self.gnu_hash,
            Sect::Hash => self.hash,
            Sect::Dynsym => self.dynsym,
            Sect::Dynstr => self.dynstr,
            Sect::Versym => self.versym,
            Sect::Verneed => self.verneed,
            Sect::RelaDyn => self.rela_dyn,
            Sect::Dynamic => self.dynamic,
            _ => Region::default(),
        }
    }
}

impl DynamicPlan {
    /// Sizes the plan without building it: every table stays empty and only
    /// [`Self::sizes`] (plus the survey's import list) is filled in.
    ///
    /// This is what layout builds before placement. The sizes depend on the
    /// symbol set and the GOT slot count, not on any address, so the regions
    /// placed from them still fit once the addresses exist. The full tables
    /// are built by [`build_plan`] after address resolution, which inherits
    /// the placed regions and the survey so the one parallel walk over every
    /// allocated section's relocations is not run a second time.
    pub fn probe(
        ctx: &Context<'_>,
        layout: &Layout,
        target: Target,
        mode: LinkMode,
        config: &DynConfig<'_>,
    ) -> Result<Self> {
        let sizing = compute_sizes(ctx, layout, target, mode, config)?;
        Ok(Self {
            dynsym: Vec::new(),
            dynstr: Vec::new(),
            hash: Vec::new(),
            gnu_hash: Vec::new(),
            versym: Vec::new(),
            verneed: Vec::new(),
            rela_dyn: Vec::new(),
            dynamic: Vec::new(),
            soname_off: None,
            needed_off: Vec::new(),
            rpath_off: None,
            relative_count: 0,
            has_versions: false,
            verneed_count: 0,
            relocs: target.dyn_relocs(),
            import_index: FxHashMap::default(),
            sizes: sizing.sizes,
            data_imports: sizing.data_imports,
            regions: DynRegions::default(),
        })
    }

    /// Builds the synthetic sections for `ctx` using the resolved addresses
    /// in `layout`. Regions stay zeroed; the caller assigns them. `sizing`
    /// is the probe's result, carried over so the sizes the regions were
    /// placed from and the tables emitted here agree by construction.
    ///
    /// `mode` selects which defined globals are exported: every one that is
    /// part of the ABI for `-shared`, and for a dynamic executable only those a
    /// dependency references. `config` supplies the SONAME and the `DT_NEEDED`
    /// list.
    fn build(
        ctx: &Context<'_>,
        layout: &Layout,
        target: Target,
        mode: LinkMode,
        config: &DynConfig<'_>,
        sizing: Sizing,
    ) -> Result<Self> {
        let relocs = target.dyn_relocs();
        let sizes = sizing.sizes;
        let names = names_by_id(ctx);
        let mut table = DynSym::new();
        // A shared object's own exported definition is preemptible, so a call
        // to it goes through a PLT stub whose `JUMP_SLOT` has to name its
        // `.dynsym` row. Recording the row here is what lets the PLT emitter
        // find it: `add_imports` only knows the rows it makes.
        let mut import_index: FxHashMap<SymbolId, u32> = FxHashMap::default();
        for e in &layout.exports {
            // An unresolved reference's row is for `.symtab` alone: its import
            // row, when the link has one, is added by `add_imports` below with
            // the versym and hash bookkeeping this loop does not do.
            if e.shndx == SHN_UNDEF {
                continue;
            }
            // `compute_sizes` puts the same question to the same predicate, so
            // the region it reserved still fits.
            let name = ctx.symbols.name(e.id);
            if !exports_definition(ctx, mode, e.visibility, name) {
                continue;
            }
            let idx = table.add_export(e, name);
            import_index.insert(e.id, idx);
        }
        import_index.extend(add_imports(
            ctx,
            layout,
            &names,
            &sizing.data_imports,
            &mut table,
        ));
        let mut dynstr = build_dynstr(&mut table, config.soname);
        let needed_off = intern_needed(&mut dynstr, config.needed);
        // The search path follows the `DT_NEEDED` strings, which is where the
        // sizing pass accounts for it.
        let rpath_off = (!config.rpath.is_empty())
            .then(|| dynsym::intern(&mut dynstr, config.rpath));
        // Reorder `.dynsym` so the defined (hashed) symbols form a contiguous
        // tail sorted by GNU hash bucket, then fix up every dynsym index the
        // import map still holds. Relocations resolve names through `by_name`,
        // which the reorder already rebuilt, so they see the new indices.
        let order = table.reorder_for_gnu_hash();
        for idx in import_index.values_mut() {
            if let Some(&new_idx) = order.remap.get(*idx as usize) {
                *idx = new_idx;
            }
        }
        // Build the symbol-version plan over the rows a dependency supplies:
        // each such name resolves to a `(soname, version)` via the dependency
        // VERDEF table. Versym indices follow the new dynsym order, so this
        // must run after the reorder.
        let mut dep_names: Vec<Vec<u8>> = Vec::new();
        version::dep_row_names(&table, layout, &mut dep_names);
        let vplan = version::build(ctx, &dep_names);
        let versym = if vplan.has_versions() {
            version::build_versym(&table, &vplan)
        } else {
            Vec::new()
        };
        // VERNEED interns its soname/version-name strings into `.dynstr`
        // after the symbol names, SONAME and DT_NEEDED entries, so the
        // earlier offsets stay valid.
        let verneed = if vplan.has_versions() {
            version::build_verneed(&vplan, &mut dynstr)
        } else {
            Vec::new()
        };
        let has_versions = vplan.has_versions();
        let verneed_count = vplan.verneed_count();
        // Each table is built only when the image carries it, so an omitted
        // one is zero bytes rather than a section nothing points at. The
        // sizing pass gates on the same setting, which is what keeps the
        // reservation and the emission in step.
        let hash = if config.hash_style.sysv() {
            build_hash(&table)
        } else {
            Vec::new()
        };
        let gnu_hash = if config.hash_style.gnu() {
            build_gnu_hash(&table, &order)
        } else {
            Vec::new()
        };
        let rela_dyn = build_rela_dyn(ctx, layout, target, &names, &table)?;
        let relative_count = count_relative(&rela_dyn, relocs.relative);
        let soname_off = table.soname_off;
        Ok(Self {
            dynsym: table.materialise(),
            dynstr,
            hash,
            gnu_hash,
            versym,
            verneed,
            rela_dyn,
            dynamic: Vec::new(),
            soname_off,
            needed_off,
            rpath_off,
            relative_count,
            has_versions,
            verneed_count,
            relocs,
            import_index,
            sizes,
            data_imports: sizing.data_imports,
            regions: DynRegions::default(),
        })
    }

    /// Serialises `.dynamic` from the placed regions. Call this after layout
    /// has filled every `*_region` field. `plt` supplies the addresses the
    /// PLT tags (`PLTGOT`, `JMPREL`, `PLTRELSZ`, `PLTREL`) reference, when a
    /// PLT was allocated. `arrays` supplies the function-pointer array regions
    /// for the `DT_*_ARRAY` tags, when the image carries pre-initialisation
    /// functions, constructors or destructors. `funcs` supplies the addresses
    /// of the initialisation and termination functions for `DT_INIT` and
    /// `DT_FINI`, when this link defines them.
    pub fn finalize(
        &mut self,
        mode: LinkMode,
        plt: Option<PltRefs>,
        arrays: ArrayRefs,
        funcs: InitFiniRefs,
    ) {
        self.dynamic = build_dynamic(self, mode, plt, arrays, funcs);
    }

    /// The dynsym index (1-based) recorded for an imported symbol, if any.
    /// Used by the PLT emitter to name the symbol in each `JUMP_SLOT`.
    pub fn import_index(&self, sym: SymbolId) -> Option<u32> {
        self.import_index.get(&sym).copied()
    }
}

/// The placed PLT addresses the `.dynamic` builder needs for the PLT tags.
/// Constructed from the layout regions by [`build_plan`].
#[derive(Clone, Copy)]
pub struct PltRefs {
    /// `DT_PLTGOT`: address of `.got.plt`.
    pub got_plt: u64,
    /// `DT_JMPREL`: address of `.rela.plt`.
    pub rela_plt: u64,
    /// `DT_PLTRELSZ`: byte size of `.rela.plt`.
    pub rela_plt_size: u64,
}

/// The placed function-pointer array regions for the `DT_*_ARRAY` tags.
///
/// Each entry that is present yields an `(address, byte size)` pair; absent
/// arrays emit no tag. Constructed from the layout regions by [`build_plan`].
#[derive(Clone, Copy, Default)]
pub struct ArrayRefs {
    /// `DT_PREINIT_ARRAY` / `DT_PREINIT_ARRAYSZ`, when the image carries
    /// pre-initialisation functions.
    pub preinit: Option<Region>,
    /// `DT_INIT_ARRAY` / `DT_INIT_ARRAYSZ`, when the image carries
    /// constructors.
    pub init: Option<Region>,
    /// `DT_FINI_ARRAY` / `DT_FINI_ARRAYSZ`, when the image carries
    /// destructors.
    pub fini: Option<Region>,
}

/// The placed addresses of the initialisation and termination functions, for
/// the `DT_INIT` and `DT_FINI` tags.
///
/// These name a single function apiece, not the arrays beside them: `DT_INIT`
/// is the `_init` the C runtime assembles from the `.init` fragments, whereas
/// `DT_INIT_ARRAY` in [`ArrayRefs`] describes a table of constructor pointers.
/// The runtime walks both.
///
/// A field is `Some` exactly when this link defines the corresponding name, a
/// question both the tag count and the serialiser put to the one predicate
/// [`dynamic_section::defined_function`]. Constructed from the resolved
/// exports by [`build_plan`].
#[derive(Clone, Copy, Default)]
pub struct InitFiniRefs {
    /// `DT_INIT`: the address of [`INIT_SYMBOL`].
    pub init: Option<u64>,
    /// `DT_FINI`: the address of [`FINI_SYMBOL`].
    pub fini: Option<u64>,
}

/// The byte sizes of the five synthetic sections, used by layout to reserve
/// space before addresses are known.
///
/// Computed by counting symbols and GOT slots, so it stays stable across the
/// address-resolution pass.
#[derive(Clone, Copy)]
pub struct DynSizes {
    /// `.dynsym` size in bytes (includes the null entry).
    pub dynsym: u64,
    /// `.dynstr` size in bytes.
    pub dynstr: u64,
    /// `.hash` size in bytes.
    pub hash: u64,
    /// `.gnu.hash` size in bytes.
    pub gnu_hash: u64,
    /// `.gnu.version` (versym) size in bytes: `2 * dynsym_count`.
    pub versym: u64,
    /// `.gnu.version_r` (VERNEED) size in bytes.
    pub verneed: u64,
    /// `.rela.dyn` size in bytes.
    pub rela_dyn: u64,
    /// `.dynamic` size in bytes.
    pub dynamic: u64,
    /// The `DT_FLAGS` word, or zero for no tag at all. Decided by the sizing
    /// pass so the emitter cannot disagree with the reserved size.
    pub flags: u64,
    /// The `DT_FLAGS_1` word, decided the same way.
    pub flags_1: u64,
}

/// The natural alignment of each synthetic section.
pub const DYNSTR_ALIGN: u64 = 1;
pub const HASH_ALIGN: u64 = 4;
/// `.gnu.hash` is word-aligned (its bloom filter is an array of 64-bit words).
pub const GNU_HASH_ALIGN: u64 = 8;
pub const DYNSYM_ALIGN: u64 = 8;
/// `.gnu.version` is an array of `u16`, so 2-byte alignment suffices.
pub const VERSYM_ALIGN: u64 = 2;
/// `.gnu.version_r` records contain `u32` fields; 4-byte alignment.
pub const VERNEED_ALIGN: u64 = 4;
pub const RELA_ALIGN: u64 = 8;
pub const DYNAMIC_ALIGN: u64 = 8;

/// Builds a `Region` with the identity-map invariant: `vaddr = base + offset`.
fn make_region(offset: u64, size: u64, base: u64) -> Region {
    Region {
        offset,
        vaddr: base.wrapping_add(offset),
        size,
    }
}

/// Places the read-only dynamic tables (`.hash`, `.dynsym`, `.dynstr`,
/// `.gnu.version`, `.gnu.version_r`, `.rela.dyn`) at `start`, extending the
/// read-execute segment.
///
/// Returns the offset one past the last byte placed. Sizes come from the
/// probe plan's `sizes` field, which is stable across address resolution.
pub fn place_ro(layout: &mut Layout, start: u64) -> u64 {
    let base = layout.base();
    let Some(plan) = layout.dynamic.as_mut() else {
        return start;
    };
    let s = &plan.sizes;
    // In placement order: `.gnu.hash` leads (the loader probes it first),
    // `.gnu.version` parallels `.dynsym` and so follows `.dynstr`, and
    // `.rela.dyn` closes the read-only tables.
    let r = &mut plan.regions;
    let mut cursor = start;
    for (align, size, slot) in [
        (GNU_HASH_ALIGN, s.gnu_hash, &mut r.gnu_hash),
        (HASH_ALIGN, s.hash, &mut r.hash),
        (DYNSYM_ALIGN, s.dynsym, &mut r.dynsym),
        (DYNSTR_ALIGN, s.dynstr, &mut r.dynstr),
        (VERSYM_ALIGN, s.versym, &mut r.versym),
        (VERNEED_ALIGN, s.verneed, &mut r.verneed),
        (RELA_ALIGN, s.rela_dyn, &mut r.rela_dyn),
    ] {
        let off = align_up(cursor, align);
        *slot = make_region(off, size, base);
        cursor = off.wrapping_add(size);
    }
    cursor
}

/// Places `.dynamic` at `start` (the start of the read-write segment, ahead of
/// `.got`). Returns the offset one past `.dynamic`.
pub fn place_rw(layout: &mut Layout, start: u64) -> u64 {
    let base = layout.base();
    let Some(plan) = layout.dynamic.as_mut() else {
        return start;
    };
    let off = align_up(start, DYNAMIC_ALIGN);
    plan.regions.dynamic = make_region(off, plan.sizes.dynamic, base);
    off + plan.sizes.dynamic
}

/// Builds the full dynamic plan after address resolution, inheriting the
/// regions the probe placed, and returns it without touching the layout.
///
/// The probe stays where it is while this runs, so a concurrent reader of
/// the layout -- the writer's copy pass runs alongside this -- still sees
/// the placed regions; the caller swaps the finished plan in afterwards.
/// The probe's sizes and its relocation-survey result are carried over
/// rather than recomputed: neither depends on an address, and the survey is
/// a walk over every allocated section's relocations that is not worth
/// running twice. `.dynamic` and every region are tightened to the
/// serialised lengths, exactly as before.
pub fn build_plan_from(
    ctx: &Context<'_>,
    layout: &Layout,
    target: Target,
    mode: LinkMode,
    config: &DynConfig<'_>,
) -> Result<DynamicPlan> {
    let placed = layout
        .dynamic
        .as_ref()
        .ok_or(crate::error::Error::Format("dynamic probe missing"))?;
    let sizing = Sizing {
        sizes: placed.sizes,
        data_imports: placed.data_imports.clone(),
    };
    let mut plan =
        DynamicPlan::build(ctx, layout, target, mode, config, sizing)?;
    plan.regions = placed.regions;
    let plt = plt_refs(layout);
    let arrays = array_refs(ctx, layout);
    let funcs = init_fini_refs(ctx, layout);
    plan.finalize(mode, plt, arrays, funcs);
    tighten_regions(&mut plan)?;
    Ok(plan)
}

/// Shrinks every synthetic region to the length actually serialised.
///
/// The sizes come from a probe that counts what the symbol set implies, and
/// emission can legitimately come up short of it: a data relocation whose
/// section `--gc-sections` dropped, a hidden weak undefined the emitter
/// resolves without a row, the gap between what the TLS counter reserves and
/// what the TLS emitter writes. What was left over stayed in `sh_size`, so
/// `readelf -r` listed trailing `R_*_NONE` rows that are not relocations and
/// `.dynsym` carried null entries past `.hash`'s `nchain`. Harmless to a
/// loader, wrong to every validator, and noise in any byte-level diff.
///
/// Nothing moves: the regions are already placed, so a shorter section leaves
/// padding before the next one rather than pulling it back. Emission longer
/// than the reservation is the opposite problem -- it would have overwritten
/// the neighbour -- so it ends the link rather than being clamped.
fn tighten_regions(plan: &mut DynamicPlan) -> Result<()> {
    let lens = [
        (
            ".dynamic",
            u64::try_from(plan.dynamic.len()).unwrap_or(0) * DYNAMIC_SIZE,
        ),
        (
            ".dynsym",
            u64::try_from(plan.dynsym.len()).unwrap_or(0) * SYM_SIZE,
        ),
        (".dynstr", u64::try_from(plan.dynstr.len()).unwrap_or(0)),
        (".hash", u64::try_from(plan.hash.len()).unwrap_or(0)),
        (".gnu.hash", u64::try_from(plan.gnu_hash.len()).unwrap_or(0)),
        (
            ".gnu.version",
            u64::try_from(plan.versym.len()).unwrap_or(0),
        ),
        (
            ".gnu.version_r",
            u64::try_from(plan.verneed.len()).unwrap_or(0),
        ),
        (
            ".rela.dyn",
            u64::try_from(plan.rela_dyn.len()).unwrap_or(0) * RELA_SIZE as u64,
        ),
    ];
    let regions = [
        &mut plan.regions.dynamic,
        &mut plan.regions.dynsym,
        &mut plan.regions.dynstr,
        &mut plan.regions.hash,
        &mut plan.regions.gnu_hash,
        &mut plan.regions.versym,
        &mut plan.regions.verneed,
        &mut plan.regions.rela_dyn,
    ];
    for (region, (name, len)) in regions.into_iter().zip(lens) {
        if len > region.size {
            debug_assert!(false, "{name} overran its reservation");
            return Err(crate::error::Error::Format(
                "a synthetic section serialised longer than it was reserved",
            ));
        }
        region.size = len;
    }
    Ok(())
}

/// Builds the set of symbol ids a `.bss` copy slot defines, so the GOT and
/// data-reloc emitters can route them to `RELATIVE` instead of `GLOB_DAT`.
/// A slot's aliases are in the set beside the symbol that selected it: they
/// resolve to the same link-time address.
pub(super) fn copy_id_set(layout: &Layout) -> FxHashSet<SymbolId> {
    layout.copy_bindings().map(|(id, _addr)| id).collect()
}

/// Whether a defined global named `name`, of visibility `visibility`, belongs
/// in `.dynsym` for a link in `mode`.
///
/// A shared object exports every definition that is part of its ABI: another
/// image may bind any of them. An executable is not a library, and its rule is
/// narrower -- it exports the names a `DT_NEEDED` dependency mentions at all.
/// That is what lets a dependency reach the program's own definition: a program
/// that interposes `malloc`, a callback a library looks up by name, a data
/// object a library writes through. It is deliberately not "export everything":
/// that is `--export-dynamic`, which is a separate request.
///
/// Mentioning it either way counts, and both halves are load-bearing. A
/// dependency that *references* the name undefined cannot be satisfied without
/// the row. A dependency that *defines* it is the interposition case: the
/// program's own definition has to win for every image in the process, and it
/// can only do that from `.dynsym`. Dropping the second half is what leaves an
/// interposing `malloc` unused while the program still links and runs.
///
/// lld arrives at the same set from the dependency's side, marking a symbol
/// exported both when it reads an undefined `.dynsym` row and when it resolves
/// a shared definition against an existing symbol
/// (`Symbol::resolve`, lld/ELF/Symbols.cpp), and its writer then puts every
/// symbol so marked into `.dynsym`.
///
/// Exporting a definition does not make it replaceable. lld's
/// `computeIsPreemptible` returns false for any definition in an executable,
/// which is what [`crate::symbol::Symbol::is_preemptible`] already answers, so
/// the relocation classifier keeps resolving these names to their addresses
/// rather than through a `GLOB_DAT`.
///
/// Only a shared-object input contributes such names; an archive or object
/// input has none, so a link with no dependency exports nothing this way.
///
/// Both lookups are membership tests against tables keyed by name, so neither
/// contributes an ordering: the caller walks the resolved symbols in id order
/// and asks about each.
///
/// Reachable from the whole crate because `--gc-sections` roots the exported
/// set and must ask the same question the table is built from: a definition
/// rooted by a second, hand-copied rule would be free to disagree with the row
/// that names it. See [`crate::gc`].
pub(crate) fn exports_definition(
    ctx: &Context<'_>,
    mode: LinkMode,
    visibility: u8,
    name: &[u8],
) -> bool {
    // A hidden or internal definition is private to the image whatever the
    // mode, which is what makes `-fvisibility=hidden` mean anything. A
    // dependency naming one is asking for a definition no loader may hand it.
    if !exports_to_dynsym(visibility) {
        return false;
    }
    match mode {
        LinkMode::Shared => true,
        // `--export-dynamic` widens an executable's export set to every
        // definition, which is what a program whose plugins bind back to it
        // needs: without it, only the names a dependency already asks for are
        // reachable, and a `dlopen`ed object binding to the program's own
        // symbol finds nothing.
        LinkMode::DynExec if ctx.export_dynamic => true,
        LinkMode::DynExec => {
            ctx.dep_undefs.contains(name) || ctx.dep_exports.get(name).is_some()
        }
        LinkMode::Static => false,
    }
}

/// Whether `id` belongs in `.dynsym` as an undefined import: a reference this
/// link has no definition for, spelled so the loader can look it up.
///
/// Hidden and internal visibility is a promise that the name is private to
/// this image, so no loader could ever satisfy it; an import row for one would
/// be a name that can never bind. lld gives such a symbol `STB_LOCAL` binding
/// and keeps it out of `.dynsym` for the same reason, and rejects a strong
/// reference to one outright -- which [`crate::layout::scan`] does too, so what
/// reaches here is the weak case that legitimately resolves to zero.
pub(super) fn is_runtime_import(ctx: &Context<'_>, id: SymbolId) -> bool {
    ctx.symbols.symbol(id).is_some_and(|s| {
        matches!(s.kind, SymbolKind::Undefined { .. })
            && exports_to_dynsym(s.visibility)
    })
}
