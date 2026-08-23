//! Mergeable section deduplication (`SHF_MERGE`, `SHF_STRINGS`).
//!
//! A section marked `SHF_MERGE` is a bag of independent, position-independent
//! constants: NUL-terminated strings when `SHF_STRINGS` is also set, otherwise
//! fixed-size entries of `sh_entsize` bytes. The producer promises nothing
//! refers to the middle of one, so the linker is free to split the section
//! apart, throw away duplicates and put the survivors back in any order.
//!
//! Every translation unit that uses `"the same literal"` emits its own copy,
//! so on real C and C++ input this is most of `.rodata`.
//!
//! # How references survive
//!
//! A reference names its piece in one of two ways, and both have to be
//! rewritten. Compilers often reference the section symbol and put the byte
//! offset in the addend. But a compiler is also free to emit a named local per
//! piece -- clang does for `-fPIE`, one `.L.str.N` per literal -- and then the
//! offset is the symbol's `st_value` and the addend is a bias on the site, not
//! part of the offset at all.
//!
//! This module records, per input section, the offset each surviving piece
//! landed at. [`MergePlan::addend`] looks up whichever of the two is the real
//! offset and rewrites the relocation's addend, which is what both the
//! writer's in-place apply and the dynamic-relocation emitter use as `A`.
//!
//! # Determinism
//!
//! Pieces are numbered in first-seen order over `(file, section, offset)`,
//! exactly as symbol ids are. The deduplication runs over a hash map, but the
//! map only decides *identity* -- whether two pieces are the same string --
//! and never the order they are emitted in.
//!
//! # What is not merged
//!
//! A section whose `sh_entsize` is zero, or whose fixed-size entries do not
//! divide its length, or whose strings are wide (`sh_entsize > 1` with
//! `SHF_STRINGS`), or that a relocation section applies to, is left alone and
//! concatenated as an ordinary section. That is always correct, just larger:
//! merging is an optimisation the linker may decline.
//!
//! One shape is refused outright rather than declined: a byte-string section
//! whose content does not end in a NUL. `SHF_STRINGS` says the content is
//! NUL-terminated strings, and a trailing run without a terminator is not one,
//! so emitting it unmerged would leave a consumer reading past the end of the
//! last string. Both reference linkers refuse it; see [`split`].
//!
//! Declining a relocated section is a correctness requirement rather than a
//! size trade. A pool is keyed on content alone, so two byte-identical pieces
//! carrying different relocations would fold onto one slot that both would
//! write; and every contributor but the carrier is dropped from its output
//! section, which takes its relocations with it. Both mold and lld decline the
//! same case.

use hashbrown::HashTable;
use rustc_hash::FxHashMap;

use crate::{
    elf::{
        ObjectFile, Rela64, Shdr64, SymbolTable,
        constants::{
            SHF_MERGE, SHF_STRINGS, SHT_NOBITS, STB_LOCAL, STT_SECTION,
        },
    },
    error::{Error, Result},
    input::InputFile,
    linker::Context,
    output::OutKind,
    symbol::name_hash,
    util::is_c_identifier,
};

/// What a mergeable section's pieces look like.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
enum Pieces {
    /// NUL-terminated byte strings.
    Strings,
    /// Fixed-size entries of this many bytes.
    Fixed(u64),
}

/// The output a pool's content lands in.
///
/// Mergeable content appears in both halves of the image. `.rodata.str1.1` is
/// allocated and joins an output section; `.debug_str` is not allocated and
/// joins an aggregated debug section, where it is usually the largest thing in
/// the file. Both deduplicate the same way and both are referenced the same
/// way -- by the section symbol with the offset in the addend -- so the only
/// difference is which list the members came from.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Scope {
    /// An allocated output section.
    Out(OutKind),
    /// An aggregated debug output section, by its index in
    /// [`crate::debug::DebugSections`].
    Debug(u32),
}

/// The identity a merged pool is shared over.
///
/// Only sections that agree on all of this may share a pool: the output they
/// land in, how their content is split, and the alignment their entries need.
/// Mixing alignments would mean padding between pieces, which defeats the
/// point.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
struct PoolKey {
    scope: Scope,
    pieces: Pieces,
    align: u64,
}

/// One input section's surviving pieces: the offset each piece had in the
/// input, paired with the offset it was given in the pool.
///
/// Sorted by input offset so a reference can be resolved with a binary search.
struct SectionPieces {
    /// The pool this section's content went into.
    pool: u32,
    /// `(input offset, pool offset)`, ascending by input offset.
    map: Vec<(u32, u32)>,
    /// One past the last byte the pieces cover, which is the section's own
    /// size: [`split`] tiles the whole section and [`pieces_of`] declines
    /// anything it could not. Bounds [`MergePlan::remap`], which would
    /// otherwise resolve an offset past the end against the last piece and
    /// hand back a pool offset that means nothing.
    end: u32,
}

/// The pieces a pool already holds, keyed by content.
///
/// An entry is the hash and the `(offset, length)` of a piece inside the
/// pool's own bytes, so the pool doubles as the arena: nothing is copied to
/// key the table. On a debug link the pool holds hundreds of thousands of
/// strings, and owning a copy of each just to look it up costs more than the
/// deduplication saves.
///
/// The hash is stored with the span because the dedup walk is serial (pool
/// layout is first-seen order): growing the table used to re-hash every kept
/// piece out of the pool right in the middle of that walk, and with the hash
/// stored a grow only moves entries.
#[derive(Default)]
struct PoolIndex {
    table: HashTable<(u64, u32, u32)>,
}

impl PoolIndex {
    /// The offset `piece` already sits at, or `None` if it is new.
    fn find(&self, pool: &[u8], piece: &[u8], hash: u64) -> Option<u32> {
        self.table
            .find(hash, |&(h, at, len)| {
                h == hash
                    && pool.get(at as usize..at as usize + len as usize)
                        == Some(piece)
            })
            .map(|&(_, at, _)| at)
    }

    /// Records a piece written at `at`.
    fn insert(&mut self, at: u32, len: u32, hash: u64) {
        self.table
            .insert_unique(hash, (hash, at, len), |&(h, ..)| h);
    }

    /// Makes room for `extra` more pieces ahead of one section's worth of
    /// inserts, so the serial walk never grows the table mid-stride.
    fn reserve(&mut self, extra: usize) {
        self.table.reserve(extra, |&(h, ..)| h);
    }
}

/// One deduplicated pool of merged content.
pub struct Pool {
    /// The deduplicated bytes, in first-seen order.
    bytes: Vec<u8>,
    /// The input section that carries the pool into the output. It keeps a
    /// member in its output section, sized to the whole pool; every other
    /// contributor is folded onto it and contributes nothing.
    carrier: (usize, u16),
    scope: Scope,
}

impl Pool {
    /// The deduplicated bytes this pool contributes to the image.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// The result of the merge pass: the pools and the per-section offset maps.
///
/// Empty when no input section is mergeable, in which case every lookup misses
/// and the link behaves exactly as it did before this pass existed.
#[derive(Default)]
pub struct MergePlan {
    pools: Vec<Pool>,
    /// The merged sections' piece maps, in the order the walk absorbed them.
    /// [`Self::rows`] names each one by `(file, section)`.
    sections: Vec<SectionPieces>,
    /// Per file, per section: the index into [`Self::sections`], or
    /// [`NOT_MERGED`]. The writer asks this of every relocation of a file
    /// with merges, so the row is an array read rather than a hash probe.
    rows: Vec<Vec<u32>>,
    /// Per file: one bit per symbol-table entry, set when the symbol is a
    /// local whose section was merged -- the only shape whose addend
    /// [`Self::addend`] rewrites. The apply loop asks about every relocation
    /// of every file with merges, and reading one bit beats reading the
    /// symbol row it would need to reach the same "no" for the vast majority
    /// that name nothing merged.
    local_bits: Vec<Vec<u64>>,
    /// Each pool's carrier, keyed by [`section_key`]. The writer asks whether
    /// a member is a carrier once per member it copies, which a scan over the
    /// pools would make quadratic. It decides identity only: the pool layout
    /// and the choice of carrier are fixed by [`build_plan`]'s walk order.
    carriers: FxHashMap<u64, u32>,
}

/// The row entry of a section that was not merged.
const NOT_MERGED: u32 = u32::MAX;

/// The map key for one input section. The file is the high half.
const fn section_key(file: usize, section: u16) -> u64 {
    ((file as u64) << 16) | section as u64
}

impl MergePlan {
    /// Whether any input section was merged.
    pub fn is_empty(&self) -> bool {
        self.pools.is_empty()
    }

    /// The pool bytes an input section carries into the output, if it is a
    /// carrier. Every other contributor carries nothing.
    pub fn carried_bytes(&self, file: usize, section: u16) -> Option<&[u8]> {
        let &pool = self.carriers.get(&section_key(file, section))?;
        self.pools.get(pool as usize).map(Pool::bytes)
    }

    /// The piece map of `(file, section)`, or `None` when it was not merged.
    fn entry(&self, file: usize, section: u16) -> Option<&SectionPieces> {
        let &idx = self.rows.get(file)?.get(usize::from(section))?;
        if idx == NOT_MERGED {
            return None;
        }
        self.sections.get(idx as usize)
    }

    /// Whether `(file, section)` was merged into a pool.
    pub fn is_merged(&self, file: usize, section: u16) -> bool {
        self.entry(file, section).is_some()
    }

    /// Whether any of `file`'s sections was merged. When not, no relocation
    /// of that file can name moved content, so [`Self::addend`] answers the
    /// entry's own addend without being asked.
    pub fn file_has_merges(&self, file: usize) -> bool {
        self.rows.get(file).is_some_and(|row| !row.is_empty())
    }

    /// Whether symbol `sym_idx` of `file` is a local in a merged section --
    /// the only symbol whose addend [`Self::addend`] would rewrite. A caller
    /// holding a relocation can skip the whole addend path on a clear bit.
    pub fn names_merged(&self, file: usize, sym_idx: u32) -> bool {
        let Some(bits) = self.local_bits.get(file) else {
            return false;
        };
        let word = (sym_idx / 64) as usize;
        bits.get(word).is_some_and(|w| w >> (sym_idx % 64) & 1 != 0)
    }

    /// Remaps an offset within a merged input section to its offset within the
    /// pool, or `None` if the section was not merged or the offset lies
    /// outside the content that was.
    ///
    /// An offset inside a piece keeps its distance from the piece start, so a
    /// reference to the middle of a string still lands in the middle of the
    /// surviving copy.
    ///
    /// The pieces tile the section from byte zero to its end -- [`split`]
    /// produces them that way and [`pieces_of`] declines any section it could
    /// not tile -- so an offset this turns away is one past the section's own
    /// end, which names no content in the input either. lld reports that case
    /// and carries on with its first piece
    /// (`MergeInputSection::getSectionPiece` : "offset is outside the
    /// section"); answering `None` is the same refusal without the guess,
    /// and leaves every caller on the unremapped offset it already falls
    /// back to for a section that was never merged.
    pub fn remap(&self, file: usize, section: u16, offset: u64) -> Option<u64> {
        let entry = self.entry(file, section)?;
        let at = u32::try_from(offset).ok()?;
        if at >= entry.end {
            return None;
        }
        debug_assert!(
            entry.map.first().is_none_or(|&(first, _)| first == 0),
            "a merged section's pieces tile it from byte zero"
        );
        // The piece containing `at` is the last one starting at or before it.
        let idx = match entry.map.binary_search_by_key(&at, |&(i, _)| i) {
            Ok(i) => i,
            Err(0) => return None,
            Err(i) => i.saturating_sub(1),
        };
        let &(in_off, out_off) = entry.map.get(idx)?;
        Some(u64::from(out_off).wrapping_add(u64::from(at - in_off)))
    }

    /// The addend to resolve `r` with: its own, or the offset the piece it
    /// names was moved to when its section was deduplicated.
    ///
    /// Only a local names merged content this way: it resolves to the pool base
    /// (see `local_addr`), so the in-section offset is remapped and becomes the
    /// addend. A global resolves to its own moved address already, and
    /// remapping again would count the move twice.
    ///
    /// Which part of the reference is the in-section offset depends on the
    /// symbol. A section symbol has no value of its own, so the compiler puts
    /// the whole offset in the addend and the lookup must fold it in. A named
    /// local -- `.L.str.1`, what clang emits for `-fPIE` -- already names its
    /// piece through `st_value`, and its addend is a bias on the site, such as
    /// the -4 of a PC-relative reference. Folding that in would look the piece
    /// up at an offset short of the section, miss, and leave the reference
    /// pointing at the pool base. This is lld's rule in `getSymVA`, which folds
    /// the addend into the section offset for a section symbol only.
    ///
    /// Every path that resolves a relocation reads its addend through this, so
    /// the in-place apply and the `R_*_RELATIVE` addend the loader applies
    /// cannot disagree about where a merged piece went.
    pub fn addend(
        &self,
        symtab: &SymbolTable<'_>,
        file: usize,
        r: &Rela64,
    ) -> i64 {
        self.moved_addend(symtab, file, r)
            .unwrap_or_else(|| r.r_addend.get())
    }

    /// The rewritten addend for a relocation whose symbol lives in a merged
    /// section, or `None` when it does not.
    fn moved_addend(
        &self,
        symtab: &SymbolTable<'_>,
        file: usize,
        r: &Rela64,
    ) -> Option<i64> {
        let sym = symtab.syms.get(r.sym() as usize)?;
        if sym.bind() != STB_LOCAL {
            return None;
        }
        let addend = r.r_addend.get();
        let shndx = sym.st_shndx.get();
        // One probe: `remap` misses for a section that was not merged, which is
        // the answer `is_merged` would have given. This runs per relocation.
        if sym.type_() == STT_SECTION {
            let inside =
                sym.st_value.get().wrapping_add(addend.cast_unsigned());
            return self.remap(file, shndx, inside).map(u64::cast_signed);
        }
        let moved = self.remap(file, shndx, sym.st_value.get())?;
        Some(moved.cast_signed().wrapping_add(addend))
    }

    /// The `(file, section)` a merged section was folded onto, so layout can
    /// stamp it with the carrier's address.
    pub fn carrier_of(
        &self,
        file: usize,
        section: u16,
    ) -> Option<(usize, u16)> {
        let entry = self.entry(file, section)?;
        self.pools.get(entry.pool as usize).map(|p| p.carrier)
    }

    /// Every merged *allocated* section that is not a carrier, paired with its
    /// carrier. Layout aliases these onto the carrier and drops them from the
    /// output section they were a member of.
    ///
    /// Debug contributors are not included: they are not members of an output
    /// section, and their references resolve through the pool offset rather
    /// than through a carrier address.
    pub fn folded(&self) -> Vec<((usize, u16), (usize, u16))> {
        let mut out = Vec::new();
        for (file, row) in self.rows.iter().enumerate() {
            for (section, &idx) in row.iter().enumerate() {
                if idx == NOT_MERGED {
                    continue;
                }
                let Some(pool) = self
                    .sections
                    .get(idx as usize)
                    .and_then(|e| self.pools.get(e.pool as usize))
                else {
                    continue;
                };
                if !matches!(pool.scope, Scope::Out(_)) {
                    continue;
                }
                let at = (file, u16::try_from(section).unwrap_or(u16::MAX));
                if pool.carrier != at {
                    out.push((at, pool.carrier));
                }
            }
        }
        // The dense walk is already in `(file, section)` order; the sort
        // stays so the contract does not rest on it.
        out.sort_unstable();
        out
    }

    /// Each allocated pool's carrier and the size it now contributes: the
    /// whole pool, since every other contributor was folded onto it.
    pub fn carrier_sizes(&self) -> Vec<(usize, u16, u64)> {
        self.pools
            .iter()
            .filter(|p| matches!(p.scope, Scope::Out(_)))
            .map(|p| {
                let (file, section) = p.carrier;
                (file, section, u64::try_from(p.bytes.len()).unwrap_or(0))
            })
            .collect()
    }

    /// The size a merged member now contributes: the whole pool for a carrier,
    /// nothing for any other contributor, and `None` if it was not merged.
    ///
    /// Used to relay out the aggregated debug sections, whose members are not
    /// output-section members and so are not folded.
    pub fn member_size(&self, file: usize, section: u16) -> Option<u64> {
        let entry = self.entry(file, section)?;
        let pool = self.pools.get(entry.pool as usize)?;
        Some(if pool.carrier == (file, section) {
            u64::try_from(pool.bytes.len()).unwrap_or(0)
        } else {
            0
        })
    }
}

/// How a mergeable section splits, or `None` if it is not mergeable or the
/// linker declines to split it.
///
/// `relocated` says whether a relocation section applies to `shdr`, which is a
/// decline on its own: such a section's pieces cannot be reordered or folded.
/// See the module documentation.
fn pieces_of(shdr: &Shdr64, relocated: bool) -> Option<Pieces> {
    if relocated {
        return None;
    }
    if shdr.sh_flags.get() & SHF_MERGE == 0 || shdr.sh_size.get() == 0 {
        return None;
    }
    // A section with no file content has nothing to deduplicate: its bytes are
    // the loader's zeroes, not an input's constants.
    if shdr.sh_type.get() == SHT_NOBITS {
        return None;
    }
    if shdr.sh_flags.get() & SHF_STRINGS != 0 {
        // Wide strings would need the terminator scanned at the entry width;
        // only byte strings are split.
        return (shdr.sh_entsize.get() == 1).then_some(Pieces::Strings);
    }
    let entsize = shdr.sh_entsize.get();
    if entsize == 0 || !shdr.sh_size.get().is_multiple_of(entsize) {
        return None;
    }
    Some(Pieces::Fixed(entsize))
}

/// Splits a mergeable section's bytes into its pieces, as `(offset, bytes)`.
///
/// The pieces tile the section: every byte belongs to exactly one, from offset
/// zero to `sh_size`. That is what lets [`MergePlan::remap`] resolve any
/// offset the input can name, and what makes the deduplicated pool a complete
/// replacement for the section's content.
///
/// A string section whose last byte is not a NUL cannot be tiled -- its
/// trailing run has no terminator, so there is no piece to copy and no way to
/// emit those bytes without inventing one. It is refused rather than
/// truncated: dropping the run would take content out of the image that an
/// input put there, and a reference into it would then resolve inside whatever
/// followed in the pool. lld refuses the same shape in
/// `MergeInputSection::splitStrings`.
///
/// The fixed-size form needs no such check: [`pieces_of`] already declines a
/// section whose size is not a whole number of entries.
fn split(data: &[u8], pieces: Pieces) -> Result<Vec<Piece>> {
    let piece = |start: usize, end: usize| {
        let off = u32::try_from(start)
            .map_err(|_| Error::Format("merged piece offset"))?;
        let len = u32::try_from(end - start)
            .map_err(|_| Error::Format("merge piece length"))?;
        Ok(Piece {
            off,
            len,
            hash: name_hash(data.get(start..end).unwrap_or_default()),
        })
    };
    match pieces {
        Pieces::Strings => {
            if data.last() != Some(&0) {
                return Err(Error::Format(
                    "mergeable string section is not NUL terminated",
                ));
            }
            let mut out = Vec::new();
            let mut start = 0usize;
            for (i, &b) in data.iter().enumerate() {
                if b == 0 {
                    let end = i.saturating_add(1);
                    out.push(piece(start, end)?);
                    start = end;
                }
            }
            Ok(out)
        }
        Pieces::Fixed(size) => {
            let n = usize::try_from(size).unwrap_or(1).max(1);
            data.chunks_exact(n)
                .enumerate()
                .map(|(i, _)| {
                    let start = i.saturating_mul(n);
                    piece(start, start + n)
                })
                .collect()
        }
    }
}

/// One piece of a mergeable section: where it sits in the section, and the
/// content hash the dedup probes with. Located and hashed in parallel, ahead
/// of the serial walk that decides where each piece lands.
struct Piece {
    off: u32,
    len: u32,
    hash: u64,
}

/// A mergeable section with its pieces split and hashed, ready for the
/// order-dependent dedup walk.
struct Prepared<'data> {
    scope: Scope,
    file: usize,
    section: u16,
    align: u64,
    pieces: Pieces,
    reserve: u64,
    data: &'data [u8],
    split: Vec<Piece>,
}

/// Deduplicates every mergeable input section of `ctx` into pools.
///
/// Runs after garbage collection and folding, so only sections that survive to
/// the output contribute. Sections are visited in `(file, section)` order and
/// pieces in offset order, which fixes both the pool layout and which section
/// carries each pool.
///
/// The walk itself cannot fan out -- a pool is laid out in first-seen order --
/// but most of its cost is not the walk: it is finding each candidate's
/// relocation section, splitting the bytes into pieces and hashing every one.
/// All of that reads one member alone, so it runs in parallel first and the
/// serial walk is left with the table probes and the byte copies.
pub fn build_plan(ctx: &Context<'_>) -> Result<MergePlan> {
    use rayon::prelude::*;
    let mut plan = MergePlan::default();
    let mut pool_of: FxHashMap<PoolKey, u32> = FxHashMap::default();
    // Per pool: content -> the offset the first copy was written at.
    let mut seen: Vec<PoolIndex> = Vec::new();
    // One object view per file rather than one per member: the walk visits
    // every input section in the link, and deriving the view is not free.
    let objs = ctx
        .files
        .iter()
        .map(|f| f.object().ok())
        .collect::<Vec<_>>();

    // Allocated members first, in output-section order, then the aggregated
    // debug sections. Both are visited in a fixed order, which is what fixes
    // the pool layout and which section carries each pool.
    let allocated = ctx.outputs.iter().flat_map(|out| {
        out.members
            .iter()
            .map(move |m| (Scope::Out(out.kind), m.file, m.section))
    });
    // The scope index distinguishes one debug output section's pool from
    // another's. Clamping it would give two sections the same scope and merge
    // their content, so the count is checked once and the indices come from
    // the checked range rather than from a per-item conversion.
    let debug_count = fits_u32(ctx.debug.sections.len(), "debug sections")?;
    let debug =
        (0..debug_count)
            .zip(&ctx.debug.sections)
            .flat_map(|(i, sec)| {
                let scope = Scope::Debug(i);
                sec.members.iter().map(move |m| (scope, m.file, m.section))
            });
    let members: Vec<(Scope, usize, u16)> = allocated.chain(debug).collect();
    let prepared: Vec<Result<Option<Prepared<'_>>>> = members
        .par_iter()
        .map(|&(scope, file, section)| {
            prepare(ctx, &objs, scope, file, section)
        })
        .collect();

    for entry in prepared {
        let Some(p) = entry? else { continue };
        let key = PoolKey {
            scope: p.scope,
            pieces: p.pieces,
            align: p.align,
        };
        // Checked before the entry so the closure cannot fail: two pools
        // sharing an index would have the carrier of one and the bytes of the
        // other.
        let next = fits_u32(plan.pools.len(), "merge pools")?;
        let pool = *pool_of.entry(key).or_insert_with(|| {
            let idx = next;
            plan.pools.push(Pool {
                bytes: Vec::new(),
                carrier: (p.file, p.section),
                scope: p.scope,
            });
            // A section carries at most one pool, so the first claim stands.
            plan.carriers
                .entry(section_key(p.file, p.section))
                .or_insert(idx);
            seen.push(PoolIndex::default());
            idx
        });
        // The pool never exceeds the bytes fed into it, and one contributor is
        // a fair guess at the deduplicated whole: `.debug_str` is largely the
        // same strings repeated per translation unit. Growing a multi-megabyte
        // pool from nothing costs more than over-reserving once.
        if let Some(dst) = plan.pools.get_mut(pool as usize)
            && dst.bytes.capacity() == 0
        {
            dst.bytes.reserve(usize::try_from(p.reserve).unwrap_or(0));
        }
        let (Some(dst), Some(index)) = (
            plan.pools.get_mut(pool as usize),
            seen.get_mut(pool as usize),
        ) else {
            continue;
        };
        let mut map = Vec::new();
        absorb(&mut dst.bytes, index, &p, &mut map)?;
        let end = u32::try_from(p.data.len())
            .map_err(|_| Error::Format("mergeable section size"))?;
        let entry = fits_u32(plan.sections.len(), "merged sections")?;
        plan.sections.push(SectionPieces { pool, map, end });
        if plan.rows.len() <= p.file {
            plan.rows.resize_with(p.file.saturating_add(1), Vec::new);
        }
        if let Some(row) = plan.rows.get_mut(p.file) {
            let at = usize::from(p.section);
            if row.len() <= at {
                row.resize(at.saturating_add(1), NOT_MERGED);
            }
            if let Some(slot) = row.get_mut(at) {
                *slot = entry;
            }
        }
    }
    plan.local_bits = local_bits(ctx, &plan)?;
    Ok(plan)
}

/// One bit per symbol of every file with merges: set when the symbol is a
/// local sitting in a merged section. See [`MergePlan::names_merged`].
fn local_bits(ctx: &Context<'_>, plan: &MergePlan) -> Result<Vec<Vec<u64>>> {
    use rayon::prelude::*;
    (0..plan.rows.len())
        .into_par_iter()
        .map(|file| {
            if !plan.file_has_merges(file) {
                return Ok(Vec::new());
            }
            let Some(symtab) = ctx
                .files
                .get(file)
                .map(InputFile::symbol_table)
                .transpose()?
            else {
                return Ok(Vec::new());
            };
            let Some(symtab) = symtab else {
                return Ok(Vec::new());
            };
            let mut bits = vec![0u64; symtab.syms.len().div_ceil(64)];
            for (i, sym) in symtab.syms.iter().enumerate() {
                if sym.bind() == STB_LOCAL
                    && plan.is_merged(file, sym.st_shndx.get())
                    && let Some(word) = bits.get_mut(i / 64)
                {
                    *word |= 1u64 << (i % 64);
                }
            }
            Ok(bits)
        })
        .collect()
}

/// Splits and hashes one member if it is a merge candidate, or answers `None`.
///
/// This is the per-member half of [`build_plan`]: everything here reads the
/// member alone, so it is safe to run for every member in parallel. Order
/// does not matter until the pieces are placed, which the serial walk does.
fn prepare<'data>(
    ctx: &Context<'data>,
    objs: &[Option<ObjectFile<'data>>],
    scope: Scope,
    file: usize,
    section: u16,
) -> Result<Option<Prepared<'data>>> {
    let (Some(input), Some(obj)) =
        (ctx.files.get(file), objs.get(file).and_then(Option::as_ref))
    else {
        return Ok(None);
    };
    let Some(shdr) = obj.sections().get(usize::from(section)) else {
        return Ok(None);
    };
    // A section whose name is a C identifier is one the program bounds
    // with `__start_NAME`/`__stop_NAME`, and those bounds are the extent
    // of *that section's* content in the image. Merging dissolves the
    // extent: the pool is keyed on how the content splits and what it
    // needs aligning to, not on which section it came from, so the
    // pieces land interleaved with `.rodata.str1.1`'s. The bounds then
    // either collapse to nothing or span foreign strings, and the walk
    // between them reads whatever else shares the pool.
    //
    // Declining to merge such a section costs the deduplication of a
    // section that is rooted anyway -- `__start_`/`__stop_` keep it whole
    // by definition -- and the name test runs only for a section that
    // already carries `SHF_MERGE`.
    let mergeable = shdr.sh_flags.get() & SHF_MERGE != 0
        && !is_c_identifier(obj.section_name(shdr));
    // Only a candidate is worth the relocation lookup: for an archive
    // member that means re-reading the member's section table.
    let relocated = mergeable && input.relocations(section)?.is_some();
    let Some(pieces) = pieces_of(shdr, relocated).filter(|_| mergeable) else {
        return Ok(None);
    };
    let data = obj.section_data(shdr)?;
    let split = split(data, pieces)?;
    Ok(Some(Prepared {
        scope,
        file,
        section,
        align: shdr.sh_addralign.get().max(1),
        pieces,
        reserve: shdr.sh_size.get(),
        data,
        split,
    }))
}

/// Writes one section's pieces into a pool, reusing the offset of any piece
/// `index` already holds, and records the input-to-pool offset map in `map`.
///
/// Pieces are visited in input offset order and a new one is appended, so the
/// pool layout follows the order the caller walks sections in.
///
/// Each new piece starts on `align`, the `sh_addralign` every contributor to
/// this pool shares. Deduplication reorders and repacks the content, so a
/// piece's offset within the pool is the only thing left describing where it
/// lands, and nothing else restores the alignment the input declared. gcc's
/// `.rodata.str1.8` is entsize 1 at align 8: packed end to end, every string
/// after the first one whose length is not a multiple of 8 sits misaligned,
/// and code compiled against the declared alignment -- a 16-byte load, an
/// `AArch64` atomic -- breaks on it. lld aligns every surviving piece the same
/// way, through the `llvm::Align(alignment)` its `StringTableBuilder` is
/// built with.
///
/// The padding is zero bytes, which is also what the gap between two input
/// sections would have been.
fn absorb(
    pool: &mut Vec<u8>,
    index: &mut PoolIndex,
    p: &Prepared<'_>,
    map: &mut Vec<(u32, u32)>,
) -> Result<()> {
    map.clear();
    map.reserve(p.split.len());
    index.reserve(p.split.len());
    let align = usize::try_from(p.align).unwrap_or(1).max(1);
    for piece in &p.split {
        let start = piece.off as usize;
        let bytes = p
            .data
            .get(start..start.saturating_add(piece.len as usize))
            .ok_or(Error::OutOfRange("merge piece bounds"))?;
        let mut out_off = index.find(pool, bytes, piece.hash);
        if out_off.is_none() {
            pool.resize(pool.len().next_multiple_of(align), 0);
            // A pool offset and a piece length are both `u32` because that is
            // what the piece map and the relocation addends carry. Clamping a
            // longer one to `u32::MAX` -- or a longer piece to zero -- mapped
            // every piece past 4 GiB to the same wrong place and remapped the
            // addends naming them to garbage, with nothing said. A `-g` link
            // that deduplicates more than 4 GiB of `.debug_str` reaches it.
            let at = fits_u32(pool.len(), "merge pool offset")?;
            let len = fits_u32(bytes.len(), "merge piece length")?;
            pool.extend_from_slice(bytes);
            index.insert(at, len, piece.hash);
            out_off = Some(at);
        }
        let Some(out_off) = out_off else {
            return Err(Error::OutOfRange("merge piece placement"));
        };
        map.push((piece.off, out_off));
    }
    Ok(())
}

/// Narrows a length or offset to the `u32` the piece map stores, or reports
/// the input this linker cannot represent.
fn fits_u32(value: usize, what: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::OutOfRange(what))
}
