//! Dead-section garbage collection for `--gc-sections`.
//!
//! A mark-sweep pass over the allocated input sections, run after
//! [`Context::assign_sections`] and before layout. Vertices are input sections
//! and edges are relocations: any section not reachable from a link root is
//! dropped from [`OutputSections`], shrinking the image.
//!
//! The algorithm is the standard one lld (`MarkLive.cpp`) and mold
//! (`gc-sections.cc`) share. [`run`] collects a root set, then walks
//! relocations to fixpoint over a worklist, then sweeps: the survivors are
//! retained and the unmarked members are removed. Only `SHF_ALLOC` sections
//! are collected; non-allocated sections (debug, symtab, comment) are never
//! placed in [`OutputSections`] and so are unaffected.
//!
//! Correctness rests on the invariant that every section a live byte of code
//! or data depends on is reachable through a relocation. By construction, if a
//! live section references a defined symbol, that symbol's defining section is
//! marked here, so a live section can never reference a dropped one. Undefined
//! references resolve to imports (no input section); commons and absolute
//! symbols synthesise storage without an input section, and are roots by other
//! means (`measure_commons` always allocates a `.bss` slot).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    dynamic::{FINI_SYMBOL, INIT_SYMBOL, LinkMode, exports_definition},
    ehframe::{Piece, is_fde_offset, records},
    elf::{
        ObjectFile, Shdr64, SymbolTable,
        constants::{SHF_ALLOC, SHF_EXECINSTR, SHF_LINK_ORDER},
    },
    error::Result,
    linker::Context,
    output::OutKind,
    symbol::{DefSource, Definition, SymbolKind},
};

/// The two output sections whose bytes are assembled from fragments in several
/// inputs, spelled through [`OutKind`] so they cannot drift from the names the
/// section classifier routes on.
const INIT_SECTION: &[u8] = OutKind::Init.name().as_bytes();
const FINI_SECTION: &[u8] = OutKind::Fini.name().as_bytes();

/// Section names a runtime reaches without any relocation in the link naming
/// them, matched in full. `.jcr` is the Java class registry a gcj runtime
/// walks; the other two are the `.init`/`.fini` fragments above.
const EXACT_ROOTS: [&[u8]; 3] = [INIT_SECTION, FINI_SECTION, b".jcr"];

/// The same, matched by prefix, because that is how the names occur: the
/// legacy constructor and destructor lists are spelled `.ctors.65535` and
/// `.dtors.65535` with the priority appended, and a compiler may emit
/// `.init_array.N` as `SHT_PROGBITS`, which the type test above misses.
const PREFIX_ROOTS: [&[u8]; 3] = [b".init_array", b".ctors", b".dtors"];

/// Whether a relocation symbol's `st_shndx` selects a real input section
/// rather than a reserved `SHN_*` value. Section, common, undefined and
/// absolute indices select nothing to mark.
///
/// Shared with identical code folding, which asks the same question of the
/// same field: a symbol naming no section is one neither pass can resolve to a
/// place in the image.
pub(crate) fn real_section(shndx: u16) -> Option<u16> {
    use crate::elf::constants::{
        SHN_ABS, SHN_COMMON, SHN_LORESERVE, SHN_UNDEF,
    };
    match shndx {
        SHN_UNDEF | SHN_ABS | SHN_COMMON => None,
        s if s >= SHN_LORESERVE => None,
        s => Some(s),
    }
}

/// Whether an allocated section is a GC root by its own attributes or name,
/// before any symbol-based roots are considered. Every section a runtime
/// reaches on its own is one: the constructor and destructor lists it walks,
/// notes a loader may inspect, explicit `SHF_GNU_RETAIN` pins, and the
/// `.init`/`.fini` fragments. Nothing in the link relocates into any of them,
/// so reachability alone would collect the lot.
///
/// This is lld's `isReserved` (`MarkLive.cpp`), less the note-in-a-group case
/// discussed below.
///
/// `.init` and `.fini` are one function each, assembled from fragments in
/// several inputs, and only the first fragment carries a name: crti.o opens
/// the function as `_init`, crtn.o closes it with no symbol of its own.
/// Rooting the symbol alone would keep the prologue and collect the return, so
/// the runtime would run off the end of the function `DT_INIT` points at.
///
/// An `SHT_NOTE` inside a section group is rooted here where lld collects it.
/// lld can tell the two apart because it threads every group member onto a
/// list; xold's [`crate::comdat::GroupDedup`] records only the members it
/// discarded, which is the question deduplication asks and not this one. The
/// difference costs image bytes for a note nothing references and never
/// correctness, so it stays until a pass needs the whole membership for its
/// own reasons.
fn is_root_section(obj: ObjectFile<'_>, shdr: &Shdr64) -> bool {
    use crate::elf::constants::{
        SHF_GNU_RETAIN, SHT_FINI_ARRAY, SHT_INIT_ARRAY, SHT_NOTE,
        SHT_PREINIT_ARRAY,
    };
    let ty = shdr.sh_type.get();
    if shdr.sh_flags.get() & SHF_GNU_RETAIN != 0
        || ty == SHT_INIT_ARRAY
        || ty == SHT_FINI_ARRAY
        || ty == SHT_PREINIT_ARRAY
        || ty == SHT_NOTE
    {
        return true;
    }
    EXACT_ROOTS.iter().any(|&n| obj.section_name_is(shdr, n))
        || PREFIX_ROOTS
            .iter()
            .any(|&p| obj.section_name_starts_with(shdr, p))
}

/// Marks `(file, section)` live and, if newly marked, pushes it on the
/// worklist for relocation scanning.
fn enqueue(
    file: usize,
    section: u16,
    live: &mut FxHashSet<(usize, u16)>,
    work: &mut Vec<(usize, u16)>,
) {
    if live.insert((file, section)) {
        work.push((file, section));
    }
}

/// Marks the defining input section of a resolved definition live, if it is
/// section-backed. Absolute definitions have no input section to collect.
fn mark_definition(
    def: &Definition,
    live: &mut FxHashSet<(usize, u16)>,
    work: &mut Vec<(usize, u16)>,
) {
    if let DefSource::Section { index, .. } = def.source {
        enqueue(def.file, index, live, work);
    }
}

/// Marks the defining section of the resolved global `name`, if it resolves to
/// a section-backed definition. Used for the entry symbol and the
/// `-shared` export root set.
fn mark_named(
    ctx: &Context<'_>,
    name: &[u8],
    live: &mut FxHashSet<(usize, u16)>,
    work: &mut Vec<(usize, u16)>,
) {
    let Some(id) = ctx.symbols.find(name) else {
        return;
    };
    let Some(sym) = ctx.symbols.symbol(id) else {
        return;
    };
    if let SymbolKind::Defined(def) = &sym.kind {
        mark_definition(def, live, work);
    }
}

/// Collects the root set into `live`/`work`:
///
/// - Every allocated section that is a root by its own attributes or name: the
///   `.preinit_array`/`.init_array`/`.fini_array` trio, `SHT_NOTE`,
///   `SHF_GNU_RETAIN`, the legacy `.ctors`/`.dtors` lists, `.jcr`, `.init` and
///   `.fini`. See [`is_root_section`].
/// - Every section a `__start_NAME`/`__stop_NAME` pair bounds. A program
///   reaches such a section by walking between the two symbols, with no
///   relocation naming it, so reachability alone would collect it and leave the
///   bounds describing nothing. This is lld's `-z nostart-stop-gc` rule
///   (`MarkLive.cpp`), the conservative one: its aggressive default drops a
///   bounded section that nothing else keeps, which silently empties the
///   program's table unless every entry was built with an explicit
///   `SHF_GNU_RETAIN`.
/// - Every `.eh_frame` member. Nothing relocates *into* unwind data, so
///   reachability alone would collect all of it and leave the runtime with no
///   index for a thrown PC (lld's `MarkLive` enqueues every `.eh_frame` section
///   for the same reason). The edges *out* of an FDE are filtered in
///   [`propagate`] so the unwind data does not in turn keep every described
///   function alive.
/// - The entry symbol's defining section.
/// - The defining sections of `_init` and `_fini`, which the loader calls
///   through `DT_INIT`/`DT_FINI` with nothing in this link relocating to them.
/// - Every defined global this image exports, because an exported row names an
///   address and another image may bind to it with no relocation in this link
///   to mark the bytes. See [`mark_exports`].
fn collect_roots(
    ctx: &Context<'_>,
    entry: &[u8],
    undefined: &[&[u8]],
    mode: LinkMode,
    live: &mut FxHashSet<(usize, u16)>,
    work: &mut Vec<(usize, u16)>,
) {
    for (file, input) in ctx.files.iter().enumerate() {
        let Ok(obj) = input.object() else { continue };
        for (i, shdr) in obj.sections().iter().enumerate() {
            if shdr.sh_flags.get() & SHF_ALLOC == 0 {
                continue;
            }
            let Ok(section) = u16::try_from(i) else {
                continue;
            };
            // A member of a COMDAT group that lost is not in the image, so it
            // is not a root. Seeding the walk from one keeps alive whatever
            // only the dead copy referenced -- bloat, in the conservative
            // direction, but it is also a walk over sections that were
            // already decided.
            if ctx.groups.is_discarded(file, section) {
                continue;
            }
            if is_root_section(obj, shdr)
                || ctx.start_stop.run_of_section(file, section).is_some()
            {
                enqueue(file, section, live, work);
            }
        }
    }
    if let Some(out) = ctx.outputs.section(OutKind::EhFrame) {
        for m in &out.members {
            enqueue(m.file, m.section, live, work);
        }
    }
    if !entry.is_empty() {
        mark_named(ctx, entry, live, work);
    }
    // A `-u` name was asked for by the command line and by nothing in the
    // image, so reachability alone would collect the very definition the
    // option pulled in.
    for &name in undefined {
        mark_named(ctx, name, live, work);
    }
    // The loader reaches these two through `DT_INIT`/`DT_FINI`, and no
    // relocation in this link names either, so reachability alone would leave
    // both tags pointing at collected bytes. lld roots the same two names.
    // They are the same constants the tags are emitted from, so the symbol the
    // loader is told to call is by construction the symbol kept here. This is
    // wider than the `.init`/`.fini` reservation in [`is_root_section`]: a
    // hand-written runtime may define `_init` in a section of any name.
    mark_named(ctx, INIT_SYMBOL, live, work);
    mark_named(ctx, FINI_SYMBOL, live, work);
    mark_exports(ctx, mode, live, work);
}

/// Marks the defining section of every definition this image exports.
///
/// An exported definition is reachable from outside the image with nothing in
/// this link naming it: a shared object publishes its whole ABI, and a dynamic
/// executable publishes the names a `DT_NEEDED` dependency mentions, which is
/// how a library's callback reaches the program's own definition. Collecting
/// one leaves a `.dynsym` row the loader can bind to pointing at bytes that are
/// no longer there.
///
/// lld roots exactly the exported set, in every mode, for the same reason
/// (`MarkLive::run`, `lld/ELF/MarkLive.cpp`).
///
/// The question is put to [`exports_definition`], the one predicate `.dynsym`
/// is built from, rather than to a copy of its rule here: a root set and an
/// export set derived separately are free to drift apart, and every way they
/// can differ is a defect. It already answers false for a static link, which
/// exports nothing and so may collect its unreferenced globals, and false for
/// hidden and internal visibility in any mode, which no loader may bind to.
///
/// The walk is `ctx.symbols.entries()`, which is id (first-seen input) order,
/// so the worklist this seeds is a function of the inputs rather than of how a
/// name happened to hash.
fn mark_exports(
    ctx: &Context<'_>,
    mode: LinkMode,
    live: &mut FxHashSet<(usize, u16)>,
    work: &mut Vec<(usize, u16)>,
) {
    for (name, sym) in ctx.symbols.entries() {
        if let SymbolKind::Defined(def) = &sym.kind
            && exports_definition(ctx, mode, sym.visibility, name)
        {
            mark_definition(def, live, work);
        }
    }
}

/// The input section a relocation's target symbol resolves to, or `None` when
/// it has none (an import, a common, an absolute value). A local symbol points
/// at a section in the same file; a global is resolved through the symbol table
/// so cross-file references reach the definition's true home even when this
/// reference comes from an input that only carries an undefined copy.
pub(crate) fn reloc_target(
    ctx: &Context<'_>,
    symtab: &SymbolTable<'_>,
    file: usize,
    sym_idx: u32,
) -> Option<(usize, u16)> {
    use crate::elf::constants::STB_LOCAL;
    let sym = symtab.syms.get(sym_idx as usize)?;
    if sym.bind() == STB_LOCAL {
        // A local symbol, including an `STT_SECTION` symbol, carries the
        // section index in `st_shndx` directly.
        return real_section(sym.st_shndx.get()).map(|section| (file, section));
    }
    // The table keys stems: a `foo@@VERS` row resolves as `foo`, so the
    // liveness edge must ask for the stem or a versioned definition's
    // section loses the edge and gets swept while still referenced.
    let name = crate::symbol::version_stem(symtab.name(sym));
    let id = ctx.symbols.find(name)?;
    let SymbolKind::Defined(def) = &ctx.symbols.symbol(id)?.kind else {
        return None;
    };
    match def.source {
        DefSource::Section { index, .. } => Some((def.file, index)),
        DefSource::Absolute { .. } => None,
    }
}

/// Whether an FDE's relocation to `(file, section)` must not be followed.
///
/// The executable half is the rule that keeps collection possible at all: an
/// FDE points at the function it describes, and following that edge would
/// keep every described function alive. The other two halves are lld's, and
/// its comment says why (`lld/ELF/MarkLive.cpp`): a table tied
/// to its function through a section group or an `SHF_LINK_ORDER` link is
/// retained with the function when the function is live, so marking the table
/// buys nothing there -- and when the function is dead it is worse than
/// nothing, because the group edge walks straight back out to the function
/// and revives it. An unreferenced `inline` function with a landing pad was
/// kept by exactly that cycle: its FDE's LSDA relocation marked
/// `.gcc_except_table`, the group bound that table to the `.text`, and the
/// dead function survived `--gc-sections`.
fn fde_target_travels_with_its_text(
    ctx: &Context<'_>,
    (file, section): (usize, u16),
) -> bool {
    has_flag(ctx, (file, section), SHF_EXECINSTR)
        || ctx.groups.is_grouped(file, section)
        || has_flag(ctx, (file, section), SHF_LINK_ORDER)
}

/// Whether the input section declares `flag` in its `sh_flags`.
fn has_flag(
    ctx: &Context<'_>,
    (file, section): (usize, u16),
    flag: u64,
) -> bool {
    ctx.files.get(file).is_some_and(|input| {
        input.object().is_ok_and(|obj| {
            obj.sections()
                .get(usize::from(section))
                .is_some_and(|s| s.sh_flags.get() & flag != 0)
        })
    })
}

/// Walks `(file, section)` into `pieces`/`fde` when it is an `.eh_frame`
/// member, reporting whether it was one. Both buffers are scratch, reused
/// across the worklist.
fn eh_frame_records(
    ctx: &Context<'_>,
    file: usize,
    section: u16,
    pieces: &mut Vec<Piece>,
    fde: &mut Vec<bool>,
) -> Result<bool> {
    let Some(input) = ctx.files.get(file) else {
        return Ok(false);
    };
    let obj = input.object()?;
    let Some(shdr) = obj.sections().get(usize::from(section)) else {
        return Ok(false);
    };
    if !crate::ehframe::is_eh_frame_name(obj.section_name(shdr)) {
        return Ok(false);
    }
    records(obj.section_data(shdr)?, pieces, fde);
    Ok(true)
}

/// Walks the worklist to fixpoint, marking the defining section of every
/// relocation in each live section. Errors from re-parsing an input or its
/// relocations are propagated; a missing symbol table or relocation section
/// simply ends that section's contribution.
///
/// Relocations inside an FDE that name an executable section are skipped: an
/// FDE points at the function it describes, and following that edge would keep
/// every described function alive, defeating collection. So is any FDE edge
/// to a section a group or an `SHF_LINK_ORDER` link ties to its function --
/// see [`fde_target_travels_with_its_text`]. The remaining FDE edges (a
/// free-standing LSDA) and every CIE edge (the personality routine) are
/// followed as usual. This is lld's `scanEhFrameSection` rule.
fn propagate(
    ctx: &Context<'_>,
    live: &mut FxHashSet<(usize, u16)>,
    work: &mut Vec<(usize, u16)>,
) -> Result<()> {
    let dependents = link_order_map(ctx)?;
    let mut pieces: Vec<Piece> = Vec::new();
    let mut fde: Vec<bool> = Vec::new();
    while let Some((file, section)) = work.pop() {
        // A section group says its members are kept or dropped together, and
        // a secondary member has no incoming relocation to be reached by:
        // the `.gcc_except_table` beside a function, a `.data.rel.ro` slice
        // the group's code indexes into. lld walks `nextInSectionGroup` when
        // it marks a member (`lld/ELF/MarkLive.cpp`); the same edge
        // is held here as a list.
        for &sibling in ctx.groups.group_members(file, section) {
            enqueue(file, sibling, live, work);
        }
        // A section ordered against this one describes it and carries no
        // relocation naming it, so nothing else can reach it. lld enqueues
        // the same edge, over `sec.dependentSections`
        // (`lld/ELF/MarkLive.cpp`).
        if !dependents.is_empty()
            && let Some(deps) = dependents.get(&(file, section))
        {
            for &dep in deps {
                enqueue(file, dep, live, work);
            }
        }
        let Some(input) = ctx.files.get(file) else {
            continue;
        };
        let Some(symtab) = input.symbol_table()? else {
            continue;
        };
        let Some(entries) = input.relocations(section)? else {
            continue;
        };
        let is_eh =
            eh_frame_records(ctx, file, section, &mut pieces, &mut fde)?;
        for r in entries {
            let Some(target) = reloc_target(ctx, &symtab, file, r.sym()) else {
                continue;
            };
            if is_eh
                && is_fde_offset(&pieces, &fde, r.r_offset.get())
                && fde_target_travels_with_its_text(ctx, target)
            {
                continue;
            }
            enqueue(target.0, target.1, live, work);
        }
    }
    Ok(())
}

/// Maps each section to the `SHF_LINK_ORDER` sections that name it in their
/// `sh_link`.
///
/// A link-order section is metadata about the section it points at:
/// `-fpatchable-function-entry` puts one entry per function in
/// `__patchable_function_entries`, and `sh_link` says which function. Nothing
/// relocates *to* such a section, so mark-and-sweep never reaches it and the
/// sweep drops it -- the nop pads survive in the live function and the table
/// describing them does not, which is exactly the runtime patcher's input
/// gone.
///
/// The edge runs the other way from every other edge in this pass, so it needs
/// its own index. Building it costs one walk of the section headers, taken
/// only when some input actually has such a section; a link with none pays the
/// scan that finds that out and nothing more, and `propagate` skips the lookup
/// on an empty map.
fn link_order_map(
    ctx: &Context<'_>,
) -> Result<FxHashMap<(usize, u16), Vec<u16>>> {
    let mut out: FxHashMap<(usize, u16), Vec<u16>> = FxHashMap::default();
    for (file, input) in ctx.files.iter().enumerate() {
        let obj = input.object()?;
        for (index, shdr) in obj.sections().iter().enumerate() {
            if shdr.sh_flags.get() & (SHF_LINK_ORDER | SHF_ALLOC)
                != SHF_LINK_ORDER | SHF_ALLOC
            {
                continue;
            }
            let Ok(section) = u16::try_from(index) else {
                continue;
            };
            let Ok(target) = u16::try_from(shdr.sh_link.get()) else {
                continue;
            };
            if target == 0 {
                continue;
            }
            out.entry((file, target)).or_default().push(section);
        }
    }
    Ok(out)
}

/// Runs mark-and-sweep over `ctx.outputs`: mark the root set and everything
/// reachable through relocations, then drop the unmarked members.
///
/// The layout and writer are unaffected beyond seeing fewer members.
pub fn run(
    ctx: &mut Context<'_>,
    entry: &[u8],
    undefined: &[&[u8]],
    mode: LinkMode,
) -> Result<()> {
    let mut live: FxHashSet<(usize, u16)> = FxHashSet::default();
    let mut work: Vec<(usize, u16)> = Vec::new();
    collect_roots(ctx, entry, undefined, mode, &mut live, &mut work);
    propagate(ctx, &mut live, &mut work)?;
    ctx.outputs.retain_live(&live);
    Ok(())
}
