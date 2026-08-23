//! `.dynamic` array serialisation: the `_DYNAMIC` table that points the
//! runtime loader at every synthetic section this object emits, plus the
//! PLT/init/fini reference extraction the finaliser feeds it with.
//!
//! [`build_dynamic`] walks the placed regions and emits one `Dyn64` per tag,
//! ordered to match the sizing pass ([`dyn_tag_count`]) so the bytes reserved
//! before address resolution exactly match what is written here. The optional
//! PLT tags appear only when a PLT was allocated; the init/fini array tags
//! appear only when the image carries constructors or destructors, and the
//! `DT_INIT`/`DT_FINI` pair only when the link defines the functions they name.

use super::{
    ArrayRefs, DynamicPlan, FINI_SYMBOL, INIT_SYMBOL, InitFiniRefs, LinkMode,
    PltRefs, RELA_SIZE, SYM_SIZE, dynsym::intern,
};
use crate::{
    elf::{
        Dyn64,
        constants::{
            DT_DEBUG, DT_FINI, DT_FINI_ARRAY, DT_FINI_ARRAYSZ, DT_FLAGS,
            DT_FLAGS_1, DT_GNU_HASH, DT_HASH, DT_INIT, DT_INIT_ARRAY,
            DT_INIT_ARRAYSZ, DT_JMPREL, DT_NEEDED, DT_NULL, DT_PLTGOT,
            DT_PLTREL, DT_PLTRELSZ, DT_PREINIT_ARRAY, DT_PREINIT_ARRAYSZ,
            DT_RELA, DT_RELA_VAL, DT_RELACOUNT, DT_RELAENT, DT_RELASZ,
            DT_RUNPATH, DT_SONAME, DT_STRSZ, DT_STRTAB, DT_SYMENT, DT_SYMTAB,
            DT_VERNEED, DT_VERNEEDNUM, DT_VERSYM,
        },
    },
    endian::{I64, U64},
    layout::{Layout, Region, Sect},
    linker::Context,
    output::OutKind,
    symbol::{SymbolId, SymbolKind},
};

/// Serialises the `.dynamic` array from the placed regions. The PLT tags are
/// emitted when `plt` is present; the init/fini array tags are emitted when
/// `arrays` carries a region and the `DT_INIT`/`DT_FINI` pair when `funcs`
/// carries an address; a dynamic executable also emits `DT_DEBUG` (the loader
/// patches it with its `r_debug` address).
pub(super) fn build_dynamic(
    plan: &DynamicPlan,
    mode: LinkMode,
    plt: Option<PltRefs>,
    arrays: ArrayRefs,
    funcs: InitFiniRefs,
) -> Vec<Dyn64> {
    let mut out = Vec::with_capacity(16);
    for off in &plan.needed_off {
        out.push(entry(DT_NEEDED, u64::from(*off)));
    }
    if let Some(off) = plan.soname_off {
        out.push(entry(DT_SONAME, u64::from(off)));
    }
    // `DT_RUNPATH` rather than `DT_RPATH`: the older tag is searched before
    // `LD_LIBRARY_PATH` and is inherited by dependencies, which is why every
    // current linker emits the newer one.
    if let Some(off) = plan.rpath_off {
        out.push(entry(DT_RUNPATH, u64::from(off)));
    }
    // A tag is emitted for each table the image carries. Pointing `DT_HASH`
    // at a table that was not emitted would send the loader into whatever
    // section follows.
    if !plan.hash.is_empty() {
        out.push(entry(DT_HASH, plan.regions.hash.vaddr));
    }
    if !plan.gnu_hash.is_empty() {
        out.push(entry(DT_GNU_HASH, plan.regions.gnu_hash.vaddr));
    }
    out.push(entry(DT_SYMTAB, plan.regions.dynsym.vaddr));
    out.push(entry(DT_STRTAB, plan.regions.dynstr.vaddr));
    out.push(entry(
        DT_STRSZ,
        u64::try_from(plan.dynstr.len()).unwrap_or(u64::MAX),
    ));
    out.push(entry(DT_SYMENT, SYM_SIZE));
    // The relocation-table group names a table the loader walks; a link that
    // produced no dynamic relocations has nothing to walk, so no tag of the
    // three is emitted. A `DT_RELA` pointing at a zero-length table is noise
    // every strict consumer still has to parse. lld gates the group on
    // `part.relaDyn->isNeeded()`, which is false for an empty table
    // (`lld/ELF/SyntheticSections.cpp`).
    if !plan.rela_dyn.is_empty() {
        out.push(entry(DT_RELA, plan.regions.rela_dyn.vaddr));
        out.push(entry(
            DT_RELASZ,
            u64::try_from(plan.rela_dyn.len() * RELA_SIZE).unwrap_or(u64::MAX),
        ));
        out.push(entry(DT_RELAENT, RELA_SIZE as u64));
    }
    if plan.relative_count > 0 {
        out.push(entry(DT_RELACOUNT, u64::from(plan.relative_count)));
    }
    // `.preinit_array` leads the three arrays, as it does in the image and in
    // the order the runtime calls them. The gABI says a shared object's tag is
    // ignored, but lld emits it for every image kind and so does this: the
    // loader skipping a tag costs nothing, whereas dropping it would lose the
    // array for a `-shared` image whose consumer does walk it.
    if let Some(r) = arrays.preinit {
        out.push(entry(DT_PREINIT_ARRAY, r.vaddr));
        out.push(entry(DT_PREINIT_ARRAYSZ, r.size));
    }
    if let Some(r) = arrays.init {
        out.push(entry(DT_INIT_ARRAY, r.vaddr));
        out.push(entry(DT_INIT_ARRAYSZ, r.size));
    }
    if let Some(r) = arrays.fini {
        out.push(entry(DT_FINI_ARRAY, r.vaddr));
        out.push(entry(DT_FINI_ARRAYSZ, r.size));
    }
    // The two single-function tags follow the arrays and precede the version
    // tags, the neighbourhood lld puts them in (`SyntheticSections.cpp`,
    // `DynamicSection::computeContents`). The gABI fixes no tag order, so the
    // only requirements are that this one is derived from the inputs alone and
    // that [`dyn_tag_count`] counts the same conditions.
    if let Some(addr) = funcs.init {
        out.push(entry(DT_INIT, addr));
    }
    if let Some(addr) = funcs.fini {
        out.push(entry(DT_FINI, addr));
    }
    if let Some(plt) = plt {
        out.push(entry(DT_PLTGOT, plt.got_plt));
        out.push(entry(DT_JMPREL, plt.rela_plt));
        out.push(entry(DT_PLTRELSZ, plt.rela_plt_size));
        out.push(entry(DT_PLTREL, DT_RELA_VAL));
    }
    if plan.has_versions {
        out.push(entry(DT_VERSYM, plan.regions.versym.vaddr));
        out.push(entry(DT_VERNEED, plan.regions.verneed.vaddr));
        out.push(entry(DT_VERNEEDNUM, u64::from(plan.verneed_count)));
    }
    if mode == LinkMode::DynExec {
        // The loader fills `DT_DEBUG` with its `r_debug` address; emitting it
        // zero gives debuggers the slot they expect.
        out.push(entry(DT_DEBUG, 0));
    }
    // The two flag words, each emitted only when it has a bit set. What is in
    // them is settled by the sizing pass, which is what keeps the reserved
    // size and the written bytes in step; see `sizes::flag_words` for what
    // each bit means and who asked for it.
    if plan.sizes.flags != 0 {
        out.push(entry(DT_FLAGS, plan.sizes.flags));
    }
    if plan.sizes.flags_1 != 0 {
        out.push(entry(DT_FLAGS_1, plan.sizes.flags_1));
    }
    out.push(entry(DT_NULL, 0));
    out
}

/// One `Dyn64` row.
fn entry(tag: i64, value: u64) -> Dyn64 {
    Dyn64 {
        d_tag: I64::new(tag),
        d_un: U64::new(value),
    }
}

/// The optional `.dynamic` tags [`build_dynamic`] may emit, bundled so the
/// sizing helper stays under clippy's excessive-bool limit. Each flag mirrors
/// a conditional block in [`build_dynamic`]; the two must stay in lockstep.
///
/// The `_array` and `_func` suffixes are load bearing: an image can carry a
/// `.init_array` of constructors without defining `_init`, or the reverse, so
/// the two questions have separate answers and separate tags.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy)]
pub(super) struct DynFeatures {
    pub has_soname: bool,
    /// `DT_RUNPATH`: the link named a runtime search path.
    pub has_rpath: bool,
    pub has_plt: bool,
    /// `DT_PREINIT_ARRAY` and `DT_PREINIT_ARRAYSZ`.
    pub has_preinit_array: bool,
    /// `DT_INIT_ARRAY` and `DT_INIT_ARRAYSZ`.
    pub has_init_array: bool,
    /// `DT_FINI_ARRAY` and `DT_FINI_ARRAYSZ`.
    pub has_fini_array: bool,
    /// `DT_INIT`: this link defines [`INIT_SYMBOL`].
    pub has_init_func: bool,
    /// `DT_FINI`: this link defines [`FINI_SYMBOL`].
    pub has_fini_func: bool,
    pub has_versions: bool,
    /// `DT_HASH`: the image carries the System V hash table.
    pub has_hash: bool,
    /// `DT_GNU_HASH`: the image carries the GNU hash table.
    pub has_gnu_hash: bool,
    /// The `DT_FLAGS` word; no tag is emitted when it is zero.
    pub flags: u64,
    /// The `DT_FLAGS_1` word; no tag is emitted when it is zero.
    pub flags_1: u64,
}

/// The number of `Dyn64` entries `build_dynamic` emits for the given inputs.
/// Kept in lockstep with [`build_dynamic`] so the sizing pass reserves exactly
/// the bytes the serialiser writes.
pub(super) fn dyn_tag_count(
    mode: LinkMode,
    features: DynFeatures,
    needed: usize,
    relative_count: u64,
    rela_total: u64,
) -> u64 {
    // Core tags: SYMTAB, STRTAB, STRSZ, SYMENT, NULL, plus one per hash
    // table the image carries.
    let mut n = 5u64;
    if features.has_hash {
        n += 1;
    }
    if features.has_gnu_hash {
        n += 1;
    }
    // RELA, RELASZ, RELAENT -- only when a relocation table exists.
    if rela_total > 0 {
        n += 3;
    }
    if features.has_soname {
        n += 1;
    }
    if features.has_rpath {
        n += 1;
    }
    if relative_count > 0 {
        n += 1; // RELACOUNT
    }
    // One NEEDED per shared dependency.
    n += u64::try_from(needed).unwrap_or(0);
    if features.has_preinit_array {
        n += 2; // PREINIT_ARRAY, PREINIT_ARRAYSZ
    }
    if features.has_init_array {
        n += 2; // INIT_ARRAY, INIT_ARRAYSZ
    }
    if features.has_fini_array {
        n += 2; // FINI_ARRAY, FINI_ARRAYSZ
    }
    if features.has_init_func {
        n += 1; // INIT
    }
    if features.has_fini_func {
        n += 1; // FINI
    }
    if features.has_plt {
        // PLTGOT, JMPREL, PLTRELSZ, PLTREL.
        n += 4;
    }
    if features.has_versions {
        // VERSYM, VERNEED, VERNEEDNUM.
        n += 3;
    }
    if mode == LinkMode::DynExec {
        n += 1; // DEBUG
    }
    if features.flags != 0 {
        n += 1; // FLAGS
    }
    if features.flags_1 != 0 {
        n += 1; // FLAGS_1
    }
    n
}

/// Interns the `DT_NEEDED` soname strings into `dynstr`, returning their
/// offsets. The strings are appended after the symbol names and SONAME so the
/// offsets stay stable between the sizing and serialisation passes.
pub(super) fn intern_needed(
    dynstr: &mut Vec<u8>,
    needed: &[&[u8]],
) -> Vec<u32> {
    let mut out = Vec::with_capacity(needed.len());
    for name in needed {
        out.push(intern(dynstr, name));
    }
    out
}

/// Builds the [`PltRefs`] a `.dynamic` finaliser needs from the placed layout,
/// or `None` when no PLT entries were allocated.
pub(super) fn plt_refs(layout: &Layout) -> Option<PltRefs> {
    if layout.region(Sect::GotPlt).size == 0 {
        return None;
    }
    Some(PltRefs {
        got_plt: layout.region(Sect::GotPlt).vaddr,
        rela_plt: layout.region(Sect::RelaPlt).vaddr,
        rela_plt_size: layout.region(Sect::RelaPlt).size,
    })
}

/// Whether the image carries the function-pointer array `kind`, and so emits
/// its `DT_*_ARRAY` tag pair.
///
/// Both the sizing pass (through [`DynFeatures`]) and [`array_refs`] answer
/// through this one predicate. Asking the same question two ways is how the
/// reserved `.dynamic` byte count drifts from the tags actually written, which
/// leaves the array either unterminated or short.
pub(super) fn has_array(ctx: &Context<'_>, kind: OutKind) -> bool {
    ctx.outputs.section(kind).is_some()
}

/// Builds the [`ArrayRefs`] a `.dynamic` finaliser needs from the placed
/// array regions. An array contributes its tag pair exactly when
/// [`has_array`] holds for it.
pub(super) fn array_refs(ctx: &Context<'_>, layout: &Layout) -> ArrayRefs {
    let of = |kind, sect| array_region(ctx, layout, kind, sect);
    ArrayRefs {
        preinit: of(OutKind::PreinitArray, Sect::PreinitArray),
        init: of(OutKind::InitArray, Sect::InitArray),
        fini: of(OutKind::FiniArray, Sect::FiniArray),
    }
}

/// The placed region of one array, or `None` when the image has no such array.
fn array_region(
    ctx: &Context<'_>,
    layout: &Layout,
    kind: OutKind,
    sect: Sect,
) -> Option<Region> {
    has_array(ctx, kind).then(|| layout.region(sect))
}

/// The resolved id of `name` when this link defines it, and so emits the tag
/// that names it. `None` for a name nothing in the link defines, or one left
/// undefined for the loader to satisfy: a tag pointing at an unresolved
/// reference would send the runtime to address zero.
///
/// This mirrors lld's guard on `DT_INIT`/`DT_FINI`, which is on the symbol and
/// not on the presence of a `.init`/`.fini` section.
///
/// Both the sizing pass (through [`DynFeatures`]) and [`init_fini_refs`]
/// answer through this one predicate, for the reason spelled out on
/// [`has_array`]: two spellings of the same question are how the reserved
/// `.dynamic` byte count drifts from the tags actually written.
pub(super) fn defined_function(
    ctx: &Context<'_>,
    name: &[u8],
) -> Option<SymbolId> {
    let id = ctx.symbols.find(name)?;
    let sym = ctx.symbols.symbol(id)?;
    matches!(sym.kind, SymbolKind::Defined(_)).then_some(id)
}

/// Builds the [`InitFiniRefs`] a `.dynamic` finaliser needs, resolving each
/// name once and then taking its address from the placed exports in a single
/// pass.
///
/// A name this link defines always has an export here: layout emits one per
/// defined symbol, and the collector roots [`INIT_SYMBOL`]/[`FINI_SYMBOL`] so
/// `--gc-sections` cannot take the section either address lives in. The
/// fallback address is therefore unreachable, and is zero rather than a
/// dropped tag so the emitted tags stay in lockstep with [`dyn_tag_count`].
pub(super) fn init_fini_refs(
    ctx: &Context<'_>,
    layout: &Layout,
) -> InitFiniRefs {
    let init = defined_function(ctx, INIT_SYMBOL);
    let fini = defined_function(ctx, FINI_SYMBOL);
    let mut refs = InitFiniRefs {
        init: init.map(|_| 0),
        fini: fini.map(|_| 0),
    };
    for e in &layout.exports {
        if init == Some(e.id) {
            refs.init = Some(e.addr);
        }
        if fini == Some(e.id) {
            refs.fini = Some(e.addr);
        }
    }
    refs
}
