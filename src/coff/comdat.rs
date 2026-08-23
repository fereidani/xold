//! COFF COMDAT selection.
//!
//! A COMDAT section is one the compiler may emit into several objects and the
//! linker is expected to keep exactly one copy of: an inline function, a
//! template instantiation, a vtable. The section carries
//! `IMAGE_SCN_LNK_COMDAT`, its section symbol's auxiliary record carries the
//! selection kind, and the symbol that follows names the group.
//!
//! Nothing read either. Every copy was placed, which is bloat, and every
//! definition was pushed into the global map, so a reference resolved to
//! whichever copy came first in input order -- an answer that changed with the
//! link line.
//!
//! The plan decides which copy wins and reports the losers so placement can
//! drop them and the address resolver can send their symbols to the winner.
//! Selection follows lld's `handleComdatSelection`: both copies' kinds are
//! compared, `ANY` paired with `LARGEST` answers as `LARGEST`, any other
//! kind disagreement is refused, `ANY` keeps the first, `LARGEST` keeps the
//! biggest, `SAME_SIZE` and `EXACT_MATCH` keep the first and refuse a copy
//! that does not match, `NODUPLICATES` refuses any second copy, and
//! `ASSOCIATIVE` lives or dies with the section it names.

use rustc_hash::FxHashMap;

use crate::{
    coff::{
        CoffFile,
        constants::{
            IMAGE_COMDAT_SELECT_ANY, IMAGE_COMDAT_SELECT_ASSOCIATIVE,
            IMAGE_COMDAT_SELECT_EXACT_MATCH, IMAGE_COMDAT_SELECT_LARGEST,
            IMAGE_COMDAT_SELECT_NODUPLICATES, IMAGE_COMDAT_SELECT_SAME_SIZE,
            IMAGE_SCN_LNK_COMDAT,
        },
        structs::SymbolAux,
    },
    error::{Error, Result},
};

/// One COMDAT section as the plan sees it.
struct Group<'data> {
    file: usize,
    /// 1-based section ordinal within its file.
    section: u32,
    /// The name that identifies the group across objects.
    key: &'data [u8],
    selection: u8,
    size: u32,
    /// For `ASSOCIATIVE`, the 1-based ordinal of the section this one follows.
    parent: u32,
    kept: bool,
}

/// Which COMDAT copies survive the link.
///
/// Built once from the inputs in input order, so the surviving copy is the
/// one the link line puts first, not the one a thread happened to reach.
pub struct ComdatPlan {
    /// `(file, 1-based section ordinal)` of every copy that lost, sorted, so
    /// the membership test is a binary search rather than a scan.
    dropped: Vec<(usize, u32)>,
}

impl ComdatPlan {
    /// Decides every COMDAT group across `inputs`.
    ///
    /// Fails when a selection kind is violated: a second `NODUPLICATES` copy,
    /// or a `SAME_SIZE` or `EXACT_MATCH` copy that does not match the one
    /// already kept.
    pub fn new(inputs: &[CoffFile<'_>]) -> Result<Self> {
        let mut groups = collect(inputs);
        select(&mut groups, inputs)?;
        drop_orphans(&mut groups);
        let mut dropped: Vec<(usize, u32)> = groups
            .iter()
            .filter(|g| !g.kept)
            .map(|g| (g.file, g.section))
            .collect();
        dropped.sort_unstable();
        Ok(Self { dropped })
    }

    /// Whether this input section lost its group and must not be placed.
    pub fn is_dropped(&self, file: usize, section: u32) -> bool {
        self.dropped.binary_search(&(file, section)).is_ok()
    }
}

/// Gathers every COMDAT section in input order with the metadata selection
/// needs.
fn collect<'data>(inputs: &[CoffFile<'data>]) -> Vec<Group<'data>> {
    let mut out = Vec::new();
    for (file, input) in inputs.iter().enumerate() {
        // One pass over the symbol table per file, not one per COMDAT
        // section: asking each section its own question walked the whole
        // table twice per section, which on sixteen objects of three thousand
        // functions was the link.
        let facts = section_facts(input);
        for section in input.sections() {
            if section.characteristics & IMAGE_SCN_LNK_COMDAT == 0 {
                continue;
            }
            let at = usize::try_from(section.index).unwrap_or(0);
            let Some(&(selection, parent, key)) = facts.get(at) else {
                continue;
            };
            let Some(selection) = selection else {
                continue;
            };
            out.push(Group {
                file,
                section: section.index,
                key: key.unwrap_or(section.name),
                selection,
                size: section.virtual_size.max(section.size_of_raw_data),
                parent,
                kept: true,
            });
        }
    }
    out
}

/// What one section's own symbol says about it: the selection kind, the
/// ordinal it is associated with, and the name of the first external defined
/// in it.
type SectionFacts<'data> = (Option<u8>, u32, Option<&'data [u8]>);

/// Per 1-based section ordinal: the selection kind and associated ordinal from
/// the section's own symbol, and the name of the first external defined in it.
///
/// The section symbol names the section (`.text`), which every COMDAT of that
/// kind shares; the external after it is the one the group is about.
fn section_facts<'data>(input: &CoffFile<'data>) -> Vec<SectionFacts<'data>> {
    let count = usize::try_from(input.number_of_sections()).unwrap_or(0);
    let mut out = vec![(None, 0u32, None); count + 1];
    for sym in input.symbols().iter() {
        let Ok(ord) = usize::try_from(sym.section_number) else {
            continue;
        };
        let Some(slot) = out.get_mut(ord) else {
            continue;
        };
        match sym.aux {
            SymbolAux::Section(aux) => {
                slot.0 = Some(aux.selection);
                slot.1 = aux.number;
            }
            _ if sym.is_external() && slot.2.is_none() => {
                slot.2 = Some(sym.name);
            }
            _ => {}
        }
    }
    out
}

/// Keeps one copy per key and marks the rest, refusing the kinds that forbid a
/// mismatch.
fn select(groups: &mut [Group<'_>], inputs: &[CoffFile<'_>]) -> Result<()> {
    // The winner so far for each key: an index into `groups`. A map, because
    // a list scanned per group is quadratic in the number of COMDATs, which
    // on a `-ffunction-sections` C++ link is the number of functions. Nothing
    // iterates it, so the hash order never reaches the output.
    let mut winners: FxHashMap<&[u8], usize> = FxHashMap::default();
    for i in 0..groups.len() {
        if groups[i].selection == IMAGE_COMDAT_SELECT_ASSOCIATIVE {
            continue;
        }
        let key = groups[i].key;
        let Some(&first) = winners.get(key) else {
            winners.insert(key, i);
            continue;
        };
        let loser = resolve_pair(groups, inputs, first, i)?;
        groups[loser].kept = false;
        if loser == first {
            winners.insert(key, i);
        }
    }
    Ok(())
}

/// Decides which of two copies of one group loses, and refuses the pairing
/// when the selection kind forbids it.
///
/// Both copies' kinds are read. `ANY` paired with `LARGEST` answers as
/// `LARGEST`: cl.exe picks `ANY` for vftables under `/GR-` and `LARGEST` under
/// `/GR`, and objects built with each must link. Any other disagreement
/// between the kinds is refused, whichever copy arrived first -- the same
/// symmetric rule lld applies (`lld/COFF/InputFiles.cpp`).
fn resolve_pair(
    groups: &[Group<'_>],
    inputs: &[CoffFile<'_>],
    first: usize,
    second: usize,
) -> Result<usize> {
    let (a, b) = (&groups[first], &groups[second]);
    let any_largest = |x: u8, y: u8| {
        (x == IMAGE_COMDAT_SELECT_ANY && y == IMAGE_COMDAT_SELECT_LARGEST)
            || (x == IMAGE_COMDAT_SELECT_LARGEST
                && y == IMAGE_COMDAT_SELECT_ANY)
    };
    if any_largest(a.selection, b.selection) {
        // The pair behaves as LARGEST: the larger copy wins, and a tie keeps
        // the one already placed so the answer follows input order.
        return Ok(if b.size > a.size { first } else { second });
    }
    if a.selection != b.selection {
        return Err(Error::Format(
            "conflicting COMDAT selection kinds for one group",
        ));
    }
    match a.selection {
        IMAGE_COMDAT_SELECT_NODUPLICATES => {
            Err(Error::Format("duplicate COMDAT declared NODUPLICATES"))
        }
        IMAGE_COMDAT_SELECT_SAME_SIZE if a.size != b.size => {
            Err(Error::Format("COMDAT copies differ in size"))
        }
        IMAGE_COMDAT_SELECT_EXACT_MATCH if !same_bytes(inputs, a, b) => {
            Err(Error::Format("COMDAT copies differ in content"))
        }
        // The larger copy wins, and a tie keeps the one already placed so the
        // answer follows input order.
        IMAGE_COMDAT_SELECT_LARGEST if b.size > a.size => Ok(first),
        _ => Ok(second),
    }
}

/// Whether two copies hold the same bytes, for `EXACT_MATCH`.
fn same_bytes(inputs: &[CoffFile<'_>], a: &Group<'_>, b: &Group<'_>) -> bool {
    let data = |g: &Group<'_>| {
        inputs
            .get(g.file)
            .and_then(|f| f.section_at(g.section))
            .map(|s| s.data)
    };
    match (data(a), data(b)) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

/// Drops every `ASSOCIATIVE` section whose parent lost.
///
/// Associations can chain, so the sweep repeats until it settles; one pass per
/// group is a bound, because each pass that changes anything drops at least
/// one more section.
fn drop_orphans(groups: &mut [Group<'_>]) {
    // Where each group sits, so following an association is a lookup rather
    // than a scan of every group.
    let at: FxHashMap<(usize, u32), usize> = groups
        .iter()
        .enumerate()
        .map(|(i, g)| ((g.file, g.section), i))
        .collect();
    let assoc: Vec<usize> = groups
        .iter()
        .enumerate()
        .filter(|(_, g)| g.selection == IMAGE_COMDAT_SELECT_ASSOCIATIVE)
        .map(|(i, _)| i)
        .collect();
    for _ in 0..=assoc.len() {
        let mut changed = false;
        for &i in &assoc {
            if !groups[i].kept {
                continue;
            }
            let key = (groups[i].file, groups[i].parent);
            let orphan = at.get(&key).is_some_and(|&p| !groups[p].kept);
            if orphan {
                groups[i].kept = false;
                changed = true;
            }
        }
        if !changed {
            return;
        }
    }
}
