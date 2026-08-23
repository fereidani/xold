//! The parallel bulk fold: interns every input's global symbols in one pass.
//!
//! Folding inputs one at a time is inherently serial, because a symbol's id is
//! the position at which its name was first seen. That ordering is worth
//! keeping -- it makes the output depend on the input order rather than on how
//! names happen to hash -- so this pass reproduces it exactly while doing the
//! expensive part concurrently.
//!
//! The work is split the way mold's `ShardedMap` splits it. A name belongs to
//! exactly one shard, chosen by its hash, so shards can be deduplicated in
//! parallel without any synchronisation. Ordering is restored afterwards: each
//! shard visits the files in order, so its list of names in first-seen order
//! is already sorted, and sorting the shards' lists together by
//! `(file, symbol index)` yields precisely the sequence a serial fold would
//! have interned in.
//!
//! Five phases:
//!
//! 1. Bin every input symbol by shard (parallel, per file).
//! 2. Deduplicate and apply precedence within each shard (parallel, per shard).
//!    Each shard walks the files in order, so precedence sees the same sequence
//!    it would serially.
//! 3. Sort the shards' first-occurrence lists together and hand out ids.
//! 4. Fill the symbol arena, the name arena and the shard indexes.
//! 5. Fill each file's `symbol index -> id` row (parallel, per file).

use hashbrown::HashTable;
use rayon::prelude::*;

use super::{
    NameRef, SHARDS, Symbol, SymbolId, SymbolKind, SymbolTable, merge_symbol,
    name_hash, shard_of,
};
use crate::{
    elf::constants::STV_DEFAULT,
    error::{Error, Result},
    input::InputSymbol,
};

/// A `(file, symbol index)` position, packed so it sorts as one integer. The
/// file is the high half, so ordering by this value is file-major, which is
/// the order a serial fold visits symbols in.
type Pos = u64;

/// Packs a position.
const fn pos(file: usize, index: u32) -> Pos {
    ((file as u64) << 32) | index as u64
}

/// The symbol index half of a position.
#[allow(clippy::cast_possible_truncation)]
const fn index_of(at: Pos) -> u32 {
    at as u32
}

/// One shard's deduplicated view of the names it owns.
struct Shard<'a> {
    /// Name -> slot within this shard. Holds slot indices while the shard is
    /// being built; [`assign_ids`] rewrites them to global ids in place, which
    /// needs no rehashing because the names do not move.
    index: HashTable<SymbolId>,
    /// Per slot: the name, borrowed from the input it was read from.
    names: Vec<&'a [u8]>,
    /// Per slot: the resolved symbol, merged across every occurrence. Holds
    /// the whole [`Symbol`] rather than just its kind because visibility is
    /// merged per occurrence too, and a losing occurrence still constrains it.
    syms: Vec<Symbol>,
    /// Per slot: where the name was first seen.
    first: Vec<Pos>,
    /// Per file: the index into `first` at which that file's run of
    /// first-seen names begins, with a trailing end marker. The shard walks
    /// files in order, so each run is contiguous and sorted.
    file_start: Vec<u32>,
    /// Per file: the slot each of this shard's records resolved to, parallel
    /// to that file's bin.
    slots: Vec<Vec<u32>>,
    /// Per slot: the global id it was numbered with. Empty until
    /// [`assign_ids`] runs.
    global: Vec<SymbolId>,
    /// The earliest merge error this shard saw, if any. Kept rather than
    /// returned so the caller can report the one a serial fold would have
    /// hit first.
    merge_error: Option<(Pos, Error)>,
}

/// Per file, per shard: the half-open range of that file's partitioned symbol
/// list the shard owns.
type Bins = [Vec<(u32, u32)>];

/// Folds every file's global symbols into `table`, returning each file's
/// `symbol index -> id` row.
///
/// The table must be empty: this pass assigns ids from zero.
pub(super) fn intern_all<'data>(
    table: &mut SymbolTable<'data>,
    files: &[Vec<InputSymbol<'data>>],
    bins: &Bins,
) -> Result<Vec<Vec<Option<SymbolId>>>> {
    debug_assert!(table.is_empty(), "bulk fold starts from an empty table");
    let mut shards = build_shards(files, bins)?;
    if let Some(err) = earliest_merge_error(&mut shards) {
        return Err(err);
    }
    let order = assign_ids(&mut shards, files.len());
    fill_table(table, &mut shards, &order);
    Ok(fill_rows(files, bins, &shards))
}

/// Phase 1: reorders each file's symbols so the ones belonging to a shard sit
/// together, and reports where each shard's run starts and ends.
///
/// Recording only the indices would leave each shard reading its file at a
/// 64-symbol stride, which is a cache miss per symbol; moving the records
/// themselves makes every shard's walk sequential. Input order is preserved
/// within a run, which is what fixes the order names are first seen in.
/// Partitions one file's symbols by shard, preserving input order within
/// each shard's run, and reports where each run starts and ends. Reachable
/// from the parse pass so a file's globals are partitioned while they are
/// still warm from extraction, rather than by a later pass copying them
/// cold.
pub(super) fn partition_file(
    syms: &mut Vec<InputSymbol<'_>>,
) -> Vec<(u32, u32)> {
    let mut counts = vec![0usize; SHARDS];
    for sym in syms.iter() {
        if let Some(c) = counts.get_mut(shard_of(sym.name_hash)) {
            *c = c.saturating_add(1);
        }
    }
    let mut buckets: Vec<Vec<InputSymbol<'_>>> =
        counts.iter().map(|&n| Vec::with_capacity(n)).collect();
    for sym in syms.iter() {
        if let Some(bucket) = buckets.get_mut(shard_of(sym.name_hash)) {
            bucket.push(*sym);
        }
    }
    let mut bounds = Vec::with_capacity(SHARDS);
    syms.clear();
    for bucket in buckets {
        let start = u32::try_from(syms.len()).unwrap_or(u32::MAX);
        syms.extend(bucket);
        let end = u32::try_from(syms.len()).unwrap_or(u32::MAX);
        bounds.push((start, end));
    }
    bounds
}

/// One shard's contiguous run of a file's partitioned symbols.
fn run<'a, 'd>(
    bins: &Bins,
    syms: &'a [InputSymbol<'d>],
    file: usize,
    shard: usize,
) -> &'a [InputSymbol<'d>] {
    let Some(&(lo, hi)) = bins.get(file).and_then(|b| b.get(shard)) else {
        return &[];
    };
    syms.get(lo as usize..hi as usize).unwrap_or_default()
}

/// Phase 2: deduplicates and merges within each shard.
///
/// Every shard walks the files in order, so the occurrences of any one name
/// arrive in the same sequence a serial fold would see them in, and precedence
/// resolves identically.
fn build_shards<'data>(
    files: &[Vec<InputSymbol<'data>>],
    bins: &Bins,
) -> Result<Vec<Shard<'data>>> {
    (0..SHARDS)
        .into_par_iter()
        .map(|s| {
            // Every record this shard will see, so its table is sized once
            // instead of rehashing its way up.
            let expect: usize = bins
                .iter()
                .filter_map(|b| b.get(s))
                .map(|&(lo, hi)| (hi.saturating_sub(lo)) as usize)
                .sum();
            let mut shard = Shard {
                index: HashTable::with_capacity(expect),
                names: Vec::with_capacity(expect),
                syms: Vec::with_capacity(expect),
                first: Vec::with_capacity(expect),
                file_start: Vec::with_capacity(files.len().saturating_add(1)),
                slots: vec![Vec::new(); files.len()],
                global: Vec::new(),
                merge_error: None,
            };
            for (file, syms) in files.iter().enumerate() {
                shard
                    .file_start
                    .push(u32::try_from(shard.first.len()).unwrap_or(u32::MAX));
                let run = run(bins, syms, file, s);
                let mut slots = Vec::with_capacity(run.len());
                for inc in run {
                    slots.push(shard.absorb(inc, file)?);
                }
                if let Some(row) = shard.slots.get_mut(file) {
                    *row = slots;
                }
            }
            shard
                .file_start
                .push(u32::try_from(shard.first.len()).unwrap_or(u32::MAX));
            Ok(shard)
        })
        .collect()
}

impl<'a> Shard<'a> {
    /// Folds one occurrence into the shard, returning its slot.
    fn absorb(&mut self, inc: &InputSymbol<'a>, file: usize) -> Result<u32> {
        // Order by the symbol's index in its input's table, not by its
        // position in the partitioned list: that index is the original input
        // order, which is what fixes which occurrence is the first one.
        let index = inc.sym_idx;
        let hash = inc.name_hash;
        let names = &self.names;
        if let Some(&slot) = self
            .index
            .find(hash, |&s| names.get(s.0).copied() == Some(inc.name))
        {
            let sym = self
                .syms
                .get_mut(slot.0)
                .ok_or(Error::OutOfRange("symbol shard slot"))?;
            if let Err(err) = merge_symbol(sym, inc, file) {
                self.record_merge_error(pos(file, index), err);
            }
            return u32::try_from(slot.0)
                .map_err(|_| Error::OutOfRange("symbol shard slot"));
        }
        let slot = self.names.len();
        self.names.push(inc.name);
        self.syms.push(Symbol::from_input(inc, file));
        self.first.push(pos(file, index));
        let names = &self.names;
        self.index.insert_unique(hash, SymbolId(slot), |&s| {
            names.get(s.0).copied().map_or(0, name_hash)
        });
        u32::try_from(slot).map_err(|_| Error::OutOfRange("symbol shard slot"))
    }

    /// Keeps the earliest merge error seen. Every kind is kept: an error a
    /// serial fold would have surfaced must not vanish because the parallel
    /// fold deferred it.
    fn record_merge_error(&mut self, at: Pos, err: Error) {
        if self.merge_error.as_ref().is_none_or(|(prev, _)| at < *prev) {
            self.merge_error = Some((at, err));
        }
    }
}

/// The merge error a serial fold would have reported first, moved out of its
/// shard.
fn earliest_merge_error(shards: &mut [Shard<'_>]) -> Option<Error> {
    shards
        .iter_mut()
        .filter_map(|s| s.merge_error.take())
        .min_by_key(|(at, _)| *at)
        .map(|(_, err)| err)
}

/// Phase 3: numbers every name in first-seen order and rewrites each shard's
/// index from slots to ids.
///
/// Returns the `(shard, slot)` of every id, in id order.
fn assign_ids(shards: &mut [Shard<'_>], file_count: usize) -> Vec<(u32, u32)> {
    // Every shard's first-seen list is already sorted and grouped by file, so
    // restoring the single sequence only needs the shards' runs for one file
    // merged at a time. Sorting each file's few thousand entries in parallel
    // is far cheaper than one sort over every name in the link.
    let per_file: Vec<Vec<(u32, u32, u32)>> = (0..file_count)
        .into_par_iter()
        .map(|file| {
            let mut names = Vec::new();
            for (s, shard) in shards.iter().enumerate() {
                let s = u32::try_from(s).unwrap_or(u32::MAX);
                let lo = shard.file_start.get(file).copied().unwrap_or(0);
                let hi = shard
                    .file_start
                    .get(file.saturating_add(1))
                    .copied()
                    .unwrap_or(lo);
                for slot in lo..hi {
                    let at =
                        shard.first.get(slot as usize).copied().unwrap_or(0);
                    names.push((index_of(at), s, slot));
                }
            }
            names.sort_unstable_by_key(|&(idx, _, _)| idx);
            names
        })
        .collect();

    let total: usize = per_file.iter().map(Vec::len).sum();
    let mut order = Vec::with_capacity(total);
    for shard in shards.iter_mut() {
        shard.global = vec![SymbolId(0); shard.names.len()];
    }
    for names in &per_file {
        for &(_, s, slot) in names {
            let id = SymbolId(order.len());
            if let Some(shard) = shards.get_mut(s as usize)
                && let Some(cell) = shard.global.get_mut(slot as usize)
            {
                *cell = id;
            }
            order.push((s, slot));
        }
    }
    // Rewrite each index from slots to ids. The names have not moved, so the
    // hashes are unchanged and no rehashing is needed.
    for shard in shards.iter_mut() {
        let global = &shard.global;
        for value in &mut shard.index {
            if let Some(&id) = global.get(value.0) {
                *value = id;
            }
        }
    }
    order
}

/// Phase 4: fills the symbol arena, the name arena and the shard indexes.
///
/// The shard indexes already hold global ids, so they are moved into the table
/// rather than rebuilt: re-inserting a million names would undo the whole
/// point of the parallel fold.
fn fill_table<'a>(
    table: &mut SymbolTable<'a>,
    shards: &mut [Shard<'a>],
    order: &[(u32, u32)],
) {
    // The names stay in the inputs; the table gathers one borrowed slice per
    // id, rather than copying tens of megabytes the mappings already hold.
    table.names = order
        .par_iter()
        .map(|&(s, slot)| {
            NameRef::Slice(
                shards
                    .get(s as usize)
                    .and_then(|sh| sh.names.get(slot as usize))
                    .copied()
                    .unwrap_or_default(),
            )
        })
        .collect();

    // Resolved symbols are independent per id, so the arena is gathered in
    // parallel; `collect` keeps them in id order.
    table.symbols = order
        .par_iter()
        .map(|&(s, slot)| {
            shards
                .get(s as usize)
                .and_then(|sh| sh.syms.get(slot as usize))
                .cloned()
                .unwrap_or(Symbol {
                    kind: SymbolKind::Undefined { weak: true },
                    visibility: STV_DEFAULT,
                })
        })
        .collect();

    for (dst, shard) in table.index.iter_mut().zip(shards.iter_mut()) {
        *dst = std::mem::take(&mut shard.index);
    }
}

/// Phase 5: fills each file's `symbol index -> id` row.
///
/// Every file owns its row and every record belongs to exactly one shard, so
/// the rows are filled concurrently with no lookups: the shard already
/// recorded which slot each record resolved to.
fn fill_rows(
    files: &[Vec<InputSymbol<'_>>],
    bins: &Bins,
    shards: &[Shard<'_>],
) -> Vec<Vec<Option<SymbolId>>> {
    files
        .par_iter()
        .enumerate()
        .map(|(file, syms)| {
            let len = syms
                .iter()
                .map(|s| s.sym_idx as usize)
                .max()
                .map_or(0, |m| m.saturating_add(1));
            let mut row = vec![None; len];
            for (s, shard) in shards.iter().enumerate() {
                let slots =
                    shard.slots.get(file).map_or(&[][..], Vec::as_slice);
                for (sym, &slot) in run(bins, syms, file, s).iter().zip(slots) {
                    let Some(&id) = shard.global.get(slot as usize) else {
                        continue;
                    };
                    if let Some(cell) = row.get_mut(sym.sym_idx as usize) {
                        *cell = Some(id);
                    }
                }
            }
            row
        })
        .collect()
}
