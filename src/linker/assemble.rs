//! Input assembly: parsing the direct inputs into a [`Context`], and the
//! shared-object dependency records the dynamic tables are built from.
//!
//! Split from the driver so each half stays within the file budget: this is
//! everything that turns the opened command line into a populated context,
//! before resolution begins.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::{
    Context,
    deps::{DepTables, shared_soname},
    in_input_order, is_shared_object,
};
use crate::{
    archive::Archive,
    dynamic::LinkMode,
    elf::{Group, ObjectFile},
    error::{Error, Result},
    input::{Input, InputFile, InputSymbol},
    symbol::SymbolKind,
};

/// A shared object the executable links against: recorded as `DT_NEEDED` so
/// the loader loads it before resolving the executable's imports.
pub(super) struct SharedDep {
    soname: Vec<u8>,
    /// The dependency's `e_machine`. A dependency is part of the image's
    /// runtime, so it has to be built for the same architecture; nothing else
    /// reads its header, so the field is carried from the parse.
    machine: u16,
    /// Whether only an `AS_NEEDED` list asked for it, so the `DT_NEEDED` row
    /// is written just when the image binds a reference to it. See
    /// [`Input::AsNeeded`] and [`needed_sonames`].
    as_needed: bool,
}

/// Collects the `DT_NEEDED` list into `out`, in dependency order.
///
/// Every dynamic output records its dependencies: a shared object imports from
/// its own list just as an executable does, and one that omits them is
/// underlinked. Only a static link has no dynamic table, and so no list.
///
/// An `AS_NEEDED` dependency is left out unless the image binds a reference to
/// it. That is what the directive asks for, and it is the difference between
/// `-lc` recording `libc.so.6` alone, as every reference linker does, and
/// recording the loader beside it because the glibc script mentions it.
pub(super) fn needed_sonames<'a>(
    ctx: &Context<'_>,
    deps: &'a [SharedDep],
    mode: LinkMode,
    out: &mut Vec<&'a [u8]>,
) {
    out.clear();
    if !mode.is_dynamic() {
        return;
    }
    let mut used = vec![false; deps.len()];
    mark_used_deps(ctx, &mut used);
    for (i, dep) in deps.iter().enumerate() {
        if !dep.as_needed || used.get(i).copied().unwrap_or(true) {
            out.push(dep.soname.as_slice());
        }
    }
}

/// Marks every dependency this image has a reference for.
///
/// A dependency earns its `DT_NEEDED` row by answering a name the image does
/// not define: that is the loader's reason to open it. The undefined symbols
/// are exactly those names, and [`DepExports`] already says which dependency
/// each one resolves to, so this is one pass with one probe per undefined
/// symbol.
///
/// A weak reference counts here, where lld's `markLive` requires a strong one
/// (`lld/ELF/MarkLive.cpp`). lld can afford the narrower rule
/// because dropping a dependency also demotes its symbols, clearing the
/// version each carried; xold keeps its dependency tables whole, so a weak
/// import versioned by a dropped dependency would leave `.gnu.version_r`
/// naming a library `DT_NEEDED` does not. The cost of the wider rule is a row
/// that could have been omitted, which is what `--no-as-needed` asks for
/// anyway; the cost of the narrower one would be an image the loader rejects.
fn mark_used_deps(ctx: &Context<'_>, used: &mut [bool]) {
    for (name, sym) in ctx.symbols.entries() {
        if !matches!(sym.kind, SymbolKind::Undefined { .. }) {
            continue;
        }
        let Some(exp) = ctx.dep_exports.get(name) else {
            continue;
        };
        if let Ok(dep) = usize::try_from(exp.at.dep)
            && let Some(slot) = used.get_mut(dep)
        {
            *slot = true;
        }
    }
}

/// Rejects a shared dependency built for another architecture.
///
/// `derive_target` folds `e_machine` over the relocatable inputs only, so a
/// dependency never joined the vote: an `AArch64` `.so` on an x86-64 link
/// contributed its exports, satisfied references with them, and took a
/// `DT_NEEDED` row. The link succeeded and the loader refused the image, which
/// is the failure arriving as far from its cause as it can. lld rejects the
/// mismatch when it opens the file, in `isCompatible`.
///
/// The dependency's machine was read during the parallel parse, so this is one
/// comparison per dependency and no re-read.
pub(super) fn check_dep_machines(
    deps: &[SharedDep],
    target: crate::reloc::Target,
) -> Result<()> {
    for dep in deps {
        let ok = crate::reloc::Target::from_machine(dep.machine)
            .is_ok_and(|m| m == target);
        if !ok {
            return Err(Error::incompatible_input(
                String::from_utf8_lossy(&dep.soname).into_owned(),
                "architecture",
            ));
        }
    }
    Ok(())
}

/// Derives the single link target from the inputs' `e_machine`. Every input
/// must agree; a mix is a format error.
pub(super) fn derive_target(ctx: &Context<'_>) -> Result<crate::reloc::Target> {
    let mut target: Option<crate::reloc::Target> = None;
    for input in &ctx.files {
        let obj = input.object()?;
        let current = crate::reloc::Target::from_machine(obj.machine())?;
        match target {
            None => target = Some(current),
            Some(existing) if existing == current => {}
            Some(_) => {
                return Err(Error::Format("inputs disagree on e_machine"));
            }
        }
    }
    target.ok_or(Error::Format("no input objects to derive a machine"))
}

/// Parses the direct inputs into a context, collecting archive inputs and
/// shared-object dependencies. An `ET_DYN` input is treated as a shared
/// dependency (recorded as `DT_NEEDED`): its exports are read into the context
/// so a dynamic executable can size copy relocations, but its sections are not
/// linked in.
///
/// Parsing, archive-index building, and per-object global-symbol extraction
/// run in parallel (each input is independent; no shared mutable state). The
/// per-input results are then folded into the context serially in input order
/// so symbol precedence, file indices and dependency-export order are
/// deterministic and byte-identical to the serial baseline.
pub(super) fn assemble_full<'a>(
    inputs: &[Input<'_>],
    bytes: &'a [&'a [u8]],
) -> Result<(Context<'a>, Vec<Archive<'a>>, Vec<SharedDep>)> {
    let parsed: Vec<Result<ParsedDirect<'a>>> = inputs
        .par_iter()
        .zip(bytes.par_iter().copied())
        .map(|(input, data)| parse_direct(*input, data))
        .collect::<Vec<Result<ParsedDirect<'a>>>>();
    let parsed = in_input_order(parsed)?;
    let mut ctx = Context::new();
    let mut archives: Vec<Archive> = Vec::new();
    let mut deps: Vec<SharedDep> = Vec::new();
    // Soname to the index of the dependency that claimed it, so a repeat can
    // reach the first occurrence rather than only knowing there was one.
    let mut seen_sonames: FxHashMap<Vec<u8>, usize> = FxHashMap::default();
    for (input_pos, item) in parsed.into_iter().enumerate() {
        // The position among the direct inputs, in input order. An archive
        // keeps its own and each dependency records its, so the archive pass
        // can ask which of the two the command line reached first.
        let input_pos = u32::try_from(input_pos).unwrap_or(u32::MAX);
        match item {
            ParsedDirect::Archive(mut archive) => {
                archive.set_pos(input_pos);
                archives.push(archive);
            }
            ParsedDirect::Shared {
                soname,
                machine,
                tables,
                as_needed,
            } => {
                // A library reached twice -- named by path and again by `-l`,
                // or through two symlinks -- is one dependency. Recording it
                // twice writes two identical `DT_NEEDED` entries and burns a
                // second dependency index that copy relocations then key on,
                // so the same storage would be described as belonging to two
                // images. The soname is the name the loader resolves, so it
                // is the identity; the first occurrence wins, which keeps the
                // answer a function of input order. lld uniquifies DSOs the
                // same way, by soname as it loads them
                // (`lld/ELF/InputFiles.cpp`).
                //
                // The two spellings are reconciled rather than merely
                // deduplicated: a library named once outright and once from an
                // `AS_NEEDED` list is an unconditional dependency, whichever
                // occurrence came first.
                if let Some(&first) = seen_sonames.get(&soname) {
                    if !as_needed && let Some(dep) = deps.get_mut(first) {
                        dep.as_needed = false;
                    }
                    continue;
                }
                seen_sonames.insert(soname.clone(), deps.len());
                // The dependency's index is its position among the shared
                // inputs, which is only known here, in input order. Copy
                // relocations key on it to tell one dependency's addresses
                // from another's.
                let dep = u32::try_from(deps.len()).unwrap_or(u32::MAX);
                tables.merge_into(
                    &mut ctx.dep_exports,
                    &mut ctx.dep_versions,
                    &mut ctx.dep_undefs,
                    dep,
                );
                // The dependency's own input position, parallel to `deps`, so
                // the archive pass can compare it against an archive's.
                ctx.dep_order.push(input_pos);
                deps.push(SharedDep {
                    soname,
                    machine,
                    as_needed,
                });
            }
            ParsedDirect::Object {
                file,
                symbols,
                bins,
                groups,
            } => {
                ctx.files.push(file);
                ctx.extracted.push(Some(symbols));
                ctx.bins.push(bins);
                ctx.group_lists.push(groups);
                ctx.sym_id.push(Vec::new());
            }
        }
    }
    Ok((ctx, archives, deps))
}

/// One direct input's parse result, produced independently during the parallel
/// parse pass. The serial merge in [`assemble_full`] folds these into the
/// context in input order.
enum ParsedDirect<'data> {
    /// A static-library archive: parsed and held for lazy member extraction.
    Archive(Archive<'data>),
    /// A shared-object dependency: its exports and per-symbol version info
    /// are collected into private tables and merged serially so
    /// first-dependency precedence holds.
    Shared {
        soname: Vec<u8>,
        machine: u16,
        tables: DepTables<'data>,
        /// Whether an `AS_NEEDED` list named it. See [`Input::AsNeeded`].
        as_needed: bool,
    },
    /// A relocatable object: parsed, with its global symbols extracted and
    /// shard-partitioned and its section groups read.
    Object {
        file: InputFile<'data>,
        symbols: Vec<InputSymbol<'data>>,
        bins: Vec<(u32, u32)>,
        groups: Vec<Group<'data>>,
    },
}

/// Classifies, parses, and (for objects) extracts global symbols from one
/// direct input. Stateless and shareable across threads: it borrows only its
/// input bytes and writes nothing shared.
fn parse_direct<'d>(
    input: Input<'_>,
    data: &'d [u8],
) -> Result<ParsedDirect<'d>> {
    let path = input.name();
    if Archive::is_archive(data) {
        return Ok(ParsedDirect::Archive(Archive::parse(data, path)?));
    }
    if is_shared_object(data) {
        let soname = shared_soname(path, data, input.is_library_search());
        let machine = ObjectFile::parse(data).map_or(0, |o| o.machine());
        let tables = DepTables::read(data, &soname);
        return Ok(ParsedDirect::Shared {
            soname,
            machine,
            tables,
            as_needed: input.is_as_needed(),
        });
    }
    let (file, mut symbols, groups) = InputFile::open_with_symbols(path, data)?;
    // Partition while the records are hot from extraction; the bulk fold
    // consumes the runs directly.
    let bins = crate::symbol::partition_file(&mut symbols);
    Ok(ParsedDirect::Object {
        file,
        symbols,
        bins,
        groups,
    })
}
