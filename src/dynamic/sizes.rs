//! Sizing pass for the synthetic dynamic sections: counts the byte sizes of
//! `.dynsym`, `.dynstr`, `.hash`, `.gnu.hash`, `.gnu.version`, `.gnu.version_r`
//! and `.rela.dyn` plus the `.dynamic` tag count, without resolving any
//! address.
//!
//! [`compute_sizes`] mirrors the structure [`super::DynamicPlan::build`] later
//! serialises, so layout can reserve space for every region up front (the
//! symbol set and GOT slot count are stable across the address-resolution
//! pass). It shares the data-reloc tally with the relocation builder via
//! [`super::rela::count_data_relocs`] and the version sizing via
//! [`super::version`].

use rustc_hash::FxHashSet;

use super::{
    DynConfig, DynSizes, FINI_SYMBOL, INIT_SYMBOL, LinkMode, RELA_SIZE,
    SYM_SIZE, copy_id_set,
    dynamic_section::{
        DynFeatures, defined_function, dyn_tag_count, has_array,
    },
    dynsym::{copy_rows, gnu_hash_size, names_by_id},
    exports_definition, is_runtime_import,
    rela::{
        FixedImports, SlotReloc, classify_slot, relative_applies,
        survey_data_relocs,
    },
    version::{self, VersionPlan},
};
use crate::{
    elf::{
        Dyn64,
        constants::{DF_1_PIE, DF_STATIC_TLS},
    },
    error::Result,
    layout::{GotKind, GotOwner, Layout},
    linker::Context,
    output::OutKind,
    reloc::Target,
    symbol::{SymbolId, SymbolKind},
};

/// The sizing pass's result: the byte sizes layout reserves, and the one
/// by-name data-import list [`super::dynsym::add_imports`] then adds rows for.
///
/// The list travels with the sizes because both come from the single
/// [`survey_data_relocs`] walk. Recomputing it in the emitter was a second full
/// pass over every allocated section's relocations that could only ever reach
/// the same answer.
pub(super) struct Sizing {
    pub(super) sizes: DynSizes,
    pub(super) data_imports: Vec<SymbolId>,
}

/// Counts the byte sizes of the synthetic sections from `ctx` and the scanned
/// GOT slots.
///
/// Mirrors the structure [`super::DynamicPlan::build`] later serialises,
/// without computing any address, so layout can place regions up front.
pub(super) fn compute_sizes(
    ctx: &Context<'_>,
    layout: &Layout,
    target: Target,
    mode: LinkMode,
    config: &DynConfig<'_>,
) -> Result<Sizing> {
    let relocs = target.dyn_relocs();
    let names = names_by_id(ctx);
    let copy_ids = copy_id_set(layout);
    let fixed = FixedImports::new(layout, &copy_ids);
    let DefinedCounts {
        mut nsyms,
        n_defined,
        mut dynstr,
    } = count_defined(ctx, layout, mode);
    let mut relative_count = 0u64;
    let mut glob_count = 0u64;
    let mut import_seen: FxHashSet<SymbolId> = FxHashSet::default();
    count_got_plt_imports(
        ctx,
        layout,
        &names,
        fixed,
        &mut nsyms,
        &mut dynstr,
        &mut import_seen,
        &mut relative_count,
        &mut glob_count,
    );
    // The third source of import rows, after the copy slots and the GOT/PLT
    // keys: the globals an absolute data reference resolves by name. The one
    // walk over those references also counts the `.rela.dyn` entries they add,
    // and its list is what [`super::dynsym::add_imports`] adds to the table --
    // so the row tally below, [`collect_import_names`] and the emitter all read
    // the same answer instead of recomputing it.
    let survey = survey_data_relocs(ctx, layout, fixed, relocs.abs64)?;
    let data_ids = survey.imports;
    count_data_imports(
        &names,
        &data_ids,
        &mut import_seen,
        &mut nsyms,
        &mut dynstr,
    );
    if let Some(s) = config.soname {
        dynstr = dynstr.saturating_add(s.len() as u64 + 1);
    }
    for n in config.needed {
        dynstr = dynstr.saturating_add(n.len() as u64 + 1);
    }
    if !config.rpath.is_empty() {
        dynstr = dynstr.saturating_add(config.rpath.len() as u64 + 1);
    }
    // Symbol-version sizing: mirror [`version::build`] over the same import
    // set so the versym/verneed region sizes and the dynstr bytes the
    // VERNEED strings add agree with the bytes the build pass emits.
    let import_names =
        collect_import_names(ctx, layout, &names, &copy_ids, &data_ids);
    let vplan = version::size_for(ctx, &import_names);
    let versym = if vplan.has_versions() {
        VersionPlan::versym_size(nsyms)
    } else {
        0
    };
    let verneed = vplan.verneed_size();
    dynstr = dynstr.saturating_add(vplan.dynstr_size());
    // Absolute data references (`R_X86_64_64`) in allocated sections add one
    // dynamic reloc each: RELATIVE for a local/section target or an import
    // this link fixes the address of, a symbol-based entry for a global the
    // loader resolves. Both occupy one `Rela64`, counted here so the region
    // sizing stays in lockstep with [`DynamicPlan::build`] -- including which
    // of the two halves this image emits at all, a question the counter and
    // the emitter put to the one classifier.
    relative_count = relative_count.saturating_add(survey.relative);
    let glob_total = glob_count.saturating_add(survey.glob);
    // One `R_X86_64_COPY` per copy-relocated symbol, after the RELATIVE prefix.
    let copy_count = u64::try_from(layout.copy_slots.len()).unwrap_or(0);
    let nbucket = nsyms.max(1);
    let sysv_hash = (2 + nbucket + nsyms) * core::mem::size_of::<u32>() as u64;
    // A table the image does not carry is reserved no bytes, which is what
    // leaves it without a section header: placement stamps an index only for
    // a region with a size.
    let hash = if config.hash_style.sysv() {
        sysv_hash
    } else {
        0
    };
    let gnu_hash = if config.hash_style.gnu() {
        gnu_hash_size(n_defined)
    } else {
        0
    };
    let rela_total = relative_count
        .saturating_add(glob_total)
        .saturating_add(copy_count);
    let rela_dyn = rela_total * RELA_SIZE as u64;
    let dynsym = nsyms * SYM_SIZE;
    // One Dyn64 per emitted tag; keep in sync with `build_dynamic`.
    let features =
        dyn_features(ctx, layout, config, mode, vplan.has_versions());
    let dyn_entries = dyn_tag_count(
        mode,
        features,
        config.needed.len(),
        relative_count,
        rela_total,
    );
    let dynamic = dyn_entries * core::mem::size_of::<Dyn64>() as u64;
    Ok(Sizing {
        sizes: DynSizes {
            dynsym,
            dynstr,
            hash,
            gnu_hash,
            versym,
            verneed,
            rela_dyn,
            dynamic,
            flags: features.flags,
            flags_1: features.flags_1,
        },
        data_imports: data_ids,
    })
}

/// What the defined `.dynsym` entries cost, before any import is counted.
struct DefinedCounts {
    /// `.dynsym` rows, including the leading null entry.
    nsyms: u64,
    /// Rows the GNU hash covers, per [`super::dynsym::DynSymEntry`]'s own
    /// test: the exported definitions, the copy slots, and the canonical PLT
    /// imports, which are undefined rows this image nonetheless answers for.
    /// The remaining imports form the unhashed prefix and are not counted.
    n_defined: u64,
    /// `.dynstr` bytes, including the leading NUL.
    dynstr: u64,
}

/// Counts the defined (hashed) symbols: the definitions this mode exports plus
/// every copy slot.
///
/// Which definitions those are is [`super::exports_definition`]'s to say, the
/// one predicate [`super::DynamicPlan::build`] filters the export list with, so
/// the reserved region cannot disagree with the emitted table. The walk is over
/// the resolved symbols rather than `layout.exports` because the export list is
/// still empty when this runs as a sizing probe; the two carry the same
/// definitions, in the same first-seen input order, save under `--gc-sections`:
/// a definition whose section was collected has no address to publish and is
/// left out of the export list, while placement is not known this early, so it
/// is counted here. That over-reserves the region rather than overrunning it,
/// as the data-relocation tally does for the same reason.
fn count_defined(
    ctx: &Context<'_>,
    layout: &Layout,
    mode: LinkMode,
) -> DefinedCounts {
    let mut out = DefinedCounts {
        nsyms: 1, // null entry
        n_defined: 0,
        dynstr: 1, // leading NUL
    };
    for (name, sym) in ctx.symbols.entries() {
        // A common is a definition the linker allocates, and the writer's
        // export loop treats it as one: `collect_exports` gives it a row
        // and the same predicate admits it. The count has to ask the same
        // question or the reservation comes up a row short.
        if matches!(
            sym.kind,
            SymbolKind::Defined(_) | SymbolKind::Common { .. }
        ) && exports_definition(ctx, mode, sym.visibility, name)
        {
            out.nsyms = out.nsyms.saturating_add(1);
            out.n_defined = out.n_defined.saturating_add(1);
            out.dynstr = out.dynstr.saturating_add(name.len() as u64 + 1);
        }
    }
    // Copy-relocated symbols are defined `.dynsym` entries (the executable
    // owns the copy in `.bss`). A slot carries one entry per name the
    // dependency gives the object it copies, which is more than one when the
    // object has aliases.
    for (_slot, name) in copy_rows(layout) {
        out.nsyms = out.nsyms.saturating_add(1);
        out.n_defined = out.n_defined.saturating_add(1);
        out.dynstr = out.dynstr.saturating_add(name.name.len() as u64 + 1);
    }
    // A canonical PLT row is an import -- its `.dynstr` name and its `.dynsym`
    // row are counted with the other imports below -- but it is hashed, so the
    // GNU hash table has to have room for it. One row per symbol, which is
    // what [`super::dynsym::add_imports`] emits.
    out.n_defined = out
        .n_defined
        .saturating_add(layout.canonical_plt.len() as u64);
    out
}

/// Which `.dynamic` tags the image emits, so the tag count matches what
/// [`super::dynamic_section`] later writes.
fn dyn_features(
    ctx: &Context<'_>,
    layout: &Layout,
    config: &DynConfig<'_>,
    mode: LinkMode,
    has_versions: bool,
) -> DynFeatures {
    DynFeatures {
        has_soname: config.soname.is_some(),
        // The PLT tags describe the table, not what is in it: a `.rela.plt`
        // holding only `IRELATIVE` entries for this image's own indirect
        // functions still has to be handed to the loader. This is the
        // condition [`super::dynamic_section::plt_refs`] emits them under,
        // which is what the count has to match.
        has_plt: !layout.plt_keys.is_empty(),
        // The array tags are counted through the same predicate
        // [`super::dynamic_section::array_refs`] emits them under, so the two
        // cannot disagree about how many `Dyn64` rows `.dynamic` holds.
        has_preinit_array: has_array(ctx, OutKind::PreinitArray),
        has_init_array: has_array(ctx, OutKind::InitArray),
        has_fini_array: has_array(ctx, OutKind::FiniArray),
        // `DT_INIT`/`DT_FINI` name one function each, not the arrays above:
        // an image can carry constructors without defining `_init`, and the
        // reverse. The tag is keyed on the symbol, through the predicate
        // [`super::dynamic_section::init_fini_refs`] also decides presence
        // with. That predicate reads only the resolved symbol table, so it
        // answers the same before address resolution (when this runs as a
        // sizing probe and the export list is still empty) as it does after.
        has_init_func: defined_function(ctx, INIT_SYMBOL).is_some(),
        has_fini_func: defined_function(ctx, FINI_SYMBOL).is_some(),
        has_versions,
        flags: flag_words(layout, config, mode).0,
        flags_1: flag_words(layout, config, mode).1,
        has_rpath: !config.rpath.is_empty(),
        has_hash: config.hash_style.sysv(),
        has_gnu_hash: config.hash_style.gnu(),
    }
}

/// The `DT_FLAGS` and `DT_FLAGS_1` words this image carries.
///
/// Two sources meet here. The link decides some bits from what it produced:
/// `DF_STATIC_TLS` says this object's thread-local offsets are fixed at load
/// time, which is what an initial-exec reference asks for -- it reads its
/// offset from a GOT slot the loader fills, and that offset exists only if the
/// object's block sits in the static TLS area, so a `GotKind::TlsOffset` key
/// is that reference (an executable needs no flag, being always in the static
/// area). `DF_1_PIE` says the image is a position-independent executable
/// rather than a library that happens to be `ET_DYN`, and is what makes glibc
/// refuse to `dlopen` a program and run its startup a second time.
///
/// The rest come from `-z` keywords, which the caller wrote and this linker
/// only records; see [`ZOptions`].
fn flag_words(
    layout: &Layout,
    config: &DynConfig<'_>,
    mode: LinkMode,
) -> (u64, u64) {
    let static_tls = mode == LinkMode::Shared
        && layout.got_keys.iter().any(|k| k.kind == GotKind::TlsOffset);
    let mut flags = config.z.flags();
    if static_tls {
        flags |= DF_STATIC_TLS;
    }
    let mut flags_1 = config.z.flags_1();
    if mode == LinkMode::DynExec && layout.is_pie() {
        flags_1 |= DF_1_PIE;
    }
    (flags, flags_1)
}

/// Collects the names of every undefined import dynsym entry, mirroring the
/// set [`super::dynsym::add_imports`] adds to the table. Used by the version
/// sizing pass to look each import up in the dependency VERDEF table without
/// building the real dynsym.
///
/// `data_ids` is the by-name data-relocation import list the caller already
/// walked, appended last so the order matches the table's.
fn collect_import_names(
    ctx: &Context<'_>,
    layout: &Layout,
    names: &[&[u8]],
    copy_ids: &FxHashSet<SymbolId>,
    data_ids: &[SymbolId],
) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut seen: FxHashSet<SymbolId> = FxHashSet::default();
    for (_slot, name) in copy_rows(layout) {
        out.push(name.name.clone());
        if let Some(id) = name.id {
            seen.insert(id);
        }
    }
    for keys in [&layout.got_keys, &layout.plt_keys] {
        for key in keys {
            let Some(id) = key.global_id() else {
                continue;
            };
            if copy_ids.contains(&id) || seen.contains(&id) {
                continue;
            }
            if !is_runtime_import(ctx, id) {
                continue;
            }
            let Some(name) = names.get(id.0) else {
                continue;
            };
            if name.is_empty() {
                continue;
            }
            out.push(name.to_vec());
            seen.insert(id);
        }
    }
    for_each_new_data_import(names, data_ids, &mut seen, |name| {
        out.push(name.to_vec());
    });
    out
}

/// Visits each by-name data-relocation import not already in `seen`, in
/// survey order, and records it.
///
/// Both the version-name survey and the row tally use this walk so they apply
/// the same empty-name and duplicate filters as the emitter.
fn for_each_new_data_import(
    names: &[&[u8]],
    data_ids: &[SymbolId],
    seen: &mut FxHashSet<SymbolId>,
    mut visit: impl FnMut(&[u8]),
) {
    for &id in data_ids {
        let Some(name) = names.get(id.0) else {
            continue;
        };
        if name.is_empty() || !seen.insert(id) {
            continue;
        }
        visit(name);
    }
}

/// Tallies the `.dynsym` row and `.dynstr` bytes each by-name data-relocation
/// import adds, skipping any the copy slots or the GOT/PLT keys already
/// counted through `seen`.
///
/// This is the sizing twin of the third loop in
/// [`super::dynsym::add_imports`], which skips the same names because the
/// table already holds them. Both consume the one [`data_import_ids`] walk, so
/// the reserved region and the emitted table cannot disagree.
fn count_data_imports(
    names: &[&[u8]],
    data_ids: &[SymbolId],
    seen: &mut FxHashSet<SymbolId>,
    nsyms: &mut u64,
    dynstr: &mut u64,
) {
    for_each_new_data_import(names, data_ids, seen, |name| {
        *nsyms = nsyms.saturating_add(1);
        *dynstr = dynstr.saturating_add(name.len() as u64 + 1);
    });
}

/// Tallies `.dynsym`/`.dynstr` growth and `RELATIVE`/`GLOB_DAT` counts over
/// the GOT and PLT key lists. A symbol reached both ways is added once; the
/// GOT slot of an import this link fixes the address of becomes `RELATIVE`
/// (not `GLOB_DAT`), and only a copy slot's names are left out of the row
/// tally, because the copy supplies its own defined row.
#[allow(clippy::too_many_arguments)]
fn count_got_plt_imports(
    ctx: &Context<'_>,
    layout: &Layout,
    names: &[&[u8]],
    fixed: FixedImports<'_>,
    nsyms: &mut u64,
    dynstr: &mut u64,
    seen: &mut FxHashSet<SymbolId>,
    relative_count: &mut u64,
    glob_count: &mut u64,
) {
    let shared = layout.mode() == LinkMode::Shared;
    let mut count_import = |id: SymbolId| {
        let is_import = is_runtime_import(ctx, id)
            && names.get(id.0).is_some_and(|n| !n.is_empty());
        if is_import && seen.insert(id) {
            *nsyms = nsyms.saturating_add(1);
            if let Some(n) = names.get(id.0) {
                *dynstr = dynstr.saturating_add(n.len() as u64 + 1);
            }
        }
        is_import
    };
    for key in &layout.got_keys {
        let global = key.global_id();
        match key.kind {
            // A local address slot holds a link-time address the loader
            // shifts by the load base -- when there is one to add. Both
            // `RELATIVE` arms below read
            // [`super::rela::relative_applies`], the same predicate
            // [`super::rela::push_got_entries`] filters its own two through,
            // so the count reserved here is the count emitted there.
            GotKind::Addr if global.is_none() => {
                if relative_applies(layout.pie) {
                    *relative_count = relative_count.saturating_add(1);
                }
            }
            // Every other address slot is bucketed by the classifier
            // [`super::rela::push_got_entries`] emits from, so the two cannot
            // disagree about which slot the loader has to fill by name.
            GotKind::Addr => {
                let Some(id) = global else { continue };
                match classify_slot(ctx, shared, fixed, id) {
                    Some(SlotReloc::ByName(_)) => {
                        count_import(id);
                        *glob_count = glob_count.saturating_add(1);
                    }
                    Some(SlotReloc::Relative)
                        if relative_applies(layout.pie) =>
                    {
                        *relative_count = relative_count.saturating_add(1);
                    }
                    Some(SlotReloc::Relative) | None => {}
                }
            }
            // A slot holding a thread-pointer offset takes a symbol-based
            // entry (`TPOFF64`) when a shared object owns the thread-local,
            // and nothing at all when this image places it: the offset is
            // fixed by this link and is not an address, so it must not be
            // shifted by the load base the way `RELATIVE` would.
            GotKind::TlsOffset => {
                if let Some(id) = global
                    && (layout.mode() == LinkMode::Shared
                        || is_runtime_import(ctx, id))
                {
                    count_import(id);
                    *glob_count = glob_count.saturating_add(1);
                }
            }
            // The module id of a general-dynamic pair always needs the loader
            // (`DTPMOD64`); the offset beside it needs one only when the
            // thread-local can be preempted, which is exactly when it has a
            // name to resolve.
            GotKind::TlsModule => {
                *glob_count = glob_count.saturating_add(1);
                if let Some(id) = global {
                    count_import(id);
                    *glob_count = glob_count.saturating_add(1);
                }
            }
        }
    }
    for key in &layout.plt_keys {
        if let GotOwner::Global(id) = key.owner
            && !fixed.is_copy(id)
        {
            count_import(id);
        }
    }
}
