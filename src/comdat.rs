//! ELF section-group (COMDAT) deduplication.
//!
//! A section group (`SHT_GROUP`) ties a set of input sections to a signature so
//! they are kept or discarded as a unit. Only a group whose flag word carries
//! `GRP_COMDAT` also asks the linker to keep just one group per signature
//! across all inputs; a group without it is a plain unit and every copy is
//! kept. C++ relies on the COMDAT form for the one-definition-rule: every
//! translation unit that instantiates a template or defines an `inline`
//! function emits its own copy inside a COMDAT group whose signature is the
//! mangled symbol name, and the linker discards every copy but the first.
//!
//! xold runs the dedup incrementally during symbol resolution: each file is
//! scanned before its symbols are folded. The first group seen for a signature
//! is kept; later groups with the same signature have all their member sections
//! recorded as discarded. Definitions sitting in a discarded section are then
//! folded as undefined so the surviving copy wins resolution naturally.
//!
//! Members of discarded groups are also skipped by
//! [`crate::linker::Context::assign_sections`], so they contribute neither
//! bytes nor relocations to the output. The effect is invisible for inputs
//! without groups: nothing is recorded as discarded and the link is
//! byte-identical to the pre-COMDAT behaviour.

use hashbrown::HashTable;
use rayon::prelude::*;

use crate::{
    elf::{Group, constants::GRP_COMDAT},
    error::{Error, Result},
    symbol::{SHARDS, shard_of},
};

/// The per-link COMDAT dedup state: the signatures already kept (first-seen
/// wins) and the `(file, section)` pairs that belong to discarded duplicate
/// groups.
///
/// The direct inputs are scanned as one batch by [`GroupDedup::scan_batch`];
/// archive members pulled later are scanned incrementally by
/// [`GroupDedup::scan`] and see the same first-wins answer.
pub struct GroupDedup {
    /// Signatures of groups already kept by an earlier group, sharded by
    /// hash so the batch scan can race the claims in parallel. A signature's
    /// shard is a pure function of its hash, so two signatures that could
    /// contend are always in the same shard and their order is decided
    /// there, in input order.
    kept: Vec<SignatureSet>,
    /// Per member section, the retained group it belongs to and whether it was
    /// discarded. Both questions are asked of every section of every input by
    /// the passes that follow, and both are answered by index.
    members: SectionRows,
    /// How many member sections [`Self::members`] marks discarded.
    discarded: usize,
    /// The members of each retained group: the file it came from and its
    /// section indices, in the order the group listed them.
    retained: Vec<(usize, Vec<u16>)>,
}

impl Default for GroupDedup {
    fn default() -> Self {
        Self {
            kept: (0..SHARDS).map(|_| SignatureSet::default()).collect(),
            members: SectionRows::default(),
            discarded: 0,
            retained: Vec::new(),
        }
    }
}

/// What a section group says about one input section: which retained group it
/// belongs to, and whether the group it came from was a discarded duplicate.
#[derive(Clone, Copy)]
struct Membership {
    /// The retained group, as an index into [`GroupDedup::retained`], or
    /// [`NO_GROUP`] when the section is in none.
    group: u32,
    discarded: bool,
}

impl Default for Membership {
    fn default() -> Self {
        Self {
            group: NO_GROUP,
            discarded: false,
        }
    }
}

/// The [`Membership::group`] of a section that belongs to no retained group.
const NO_GROUP: u32 = u32::MAX;

/// The COMDAT signatures already claimed, as one hash table over an arena of
/// signature bytes.
///
/// Every group in the link is put to this set once, and a C++ translation unit
/// carries one group per template instantiation and per inline function. A
/// `HashSet<Vec<u8>>` hashed each signature twice -- once to ask, once to
/// insert -- and allocated for every name it kept; this probes with the hash
/// the parallel group parse already computed and copies the bytes into the
/// arena only when the signature is new.
///
/// Each entry carries its hash. The dedup walk is the serial spine of a C++
/// link, and growing the table used to mean re-hashing every kept signature
/// out of the arena right in the middle of it; with the hash stored, a grow
/// only moves entries.
#[derive(Default)]
struct SignatureSet {
    table: HashTable<(u64, u32, u32)>,
    arena: Vec<u8>,
}

impl SignatureSet {
    /// Claims `signature` for the caller, reporting whether it was the first
    /// to do so. `hash` is the signature's [`crate::symbol::name_hash`].
    fn insert(&mut self, signature: &[u8], hash: u64) -> bool {
        let Self { table, arena } = self;
        let found = table.find(hash, |&(h, start, end)| {
            h == hash
                && arena.get(start as usize..end as usize) == Some(signature)
        });
        if found.is_some() {
            return false;
        }
        let Ok(start) = u32::try_from(arena.len()) else {
            // The arena cannot address this signature, so the set cannot tell
            // a later copy from it. Keeping the group is the safe answer: a
            // duplicate wastes bytes, a wrongly discarded one drops a
            // definition.
            return true;
        };
        arena.extend_from_slice(signature);
        let Ok(end) = u32::try_from(arena.len()) else {
            arena.truncate(start as usize);
            return true;
        };
        table.insert_unique(hash, (hash, start, end), |&(h, ..)| h);
        true
    }

    /// Makes room for `extra` more signatures ahead of a batch of inserts.
    fn reserve(&mut self, extra: usize) {
        self.table.reserve(extra, |&(h, ..)| h);
    }
}

/// Per input file, per section: what the group scan recorded about it.
///
/// A hash of a `(file, section)` pair used to answer this. Every section of
/// every input is asked about -- by symbol resolution, by section assignment,
/// by garbage collection and by the dynamic relocation walk -- and a C++ link
/// has thousands of sections per file, so the rows are indexed instead. A file
/// with no groups keeps an empty row and costs one bounds check.
#[derive(Default)]
struct SectionRows {
    rows: Vec<Vec<Membership>>,
}

impl SectionRows {
    /// What was recorded about `(file, section)`, or the default (no group,
    /// not discarded) for a section nothing recorded.
    fn get(&self, file: usize, section: u16) -> Membership {
        self.rows
            .get(file)
            .and_then(|row| row.get(usize::from(section)))
            .copied()
            .unwrap_or_default()
    }

    /// The slot for `(file, section)`, growing the rows to reach it.
    fn slot(&mut self, file: usize, section: u16) -> Option<&mut Membership> {
        if self.rows.len() <= file {
            self.rows.resize_with(file.saturating_add(1), Vec::new);
        }
        let row = self.rows.get_mut(file)?;
        let at = usize::from(section);
        if row.len() <= at {
            row.resize(at.saturating_add(1), Membership::default());
        }
        row.get_mut(at)
    }
}

/// One file's recorded verdicts: its retained groups as `(id, members)`,
/// and the member sections of its discarded duplicates.
type FileVerdicts = (Vec<(u32, Vec<u16>)>, Vec<u16>);

/// One COMDAT group's entry in the sharded first-wins race: where the group
/// sits, and the signature it claims.
#[derive(Clone, Copy)]
struct Claim<'data> {
    file: u32,
    group: u32,
    hash: u64,
    signature: &'data [u8],
}

impl GroupDedup {
    /// An empty state (no groups seen, nothing discarded).
    pub fn new() -> Self {
        Self::default()
    }

    /// Scans the direct inputs' groups as one batch, with the same result as
    /// [`Self::scan`] called per file in file order.
    ///
    /// First-per-signature is an ordered race, but only between groups whose
    /// signatures collide -- and colliding signatures share a shard, because
    /// the shard is a function of the hash. So the race runs one shard per
    /// thread, each shard walking its claims in input order, and the answers
    /// are exactly the serial walk's. The serial part left is recording the
    /// verdicts, which probes nothing.
    pub fn scan_batch(&mut self, files: &[Vec<Group<'_>>]) -> Result<()> {
        let mut shards: Vec<Vec<Claim<'_>>> = partition_claims(files);
        // The race itself: each shard claims its signatures in input order
        // and reports which claims won.
        let kept_lists: Vec<Vec<(u32, u32)>> = self
            .kept
            .par_iter_mut()
            .zip(&mut shards)
            .map(|(set, claims)| {
                set.reserve(claims.len());
                let mut kept = Vec::new();
                for c in claims.drain(..) {
                    if set.insert(c.signature, c.hash) {
                        kept.push((c.file, c.group));
                    }
                }
                kept
            })
            .collect();
        let mut kept: Vec<Vec<bool>> =
            files.iter().map(|g| vec![false; g.len()]).collect();
        for (file, group) in kept_lists.into_iter().flatten() {
            if let Some(slot) = kept
                .get_mut(file as usize)
                .and_then(|row| row.get_mut(group as usize))
            {
                *slot = true;
            }
        }
        self.record_batch(files, &kept)
    }

    /// Records every file's verdicts, with the same flag validation, id
    /// order and membership as [`Self::scan`] would in file order.
    ///
    /// The serial half validates and claims retained ids -- their order is
    /// meaning, an id indexes [`Self::retained`] -- and the membership rows
    /// fill in parallel afterwards: each row belongs to one file, and the
    /// ids it writes were fixed before the fan-out.
    fn record_batch(
        &mut self,
        files: &[Vec<Group<'_>>],
        kept: &[Vec<bool>],
    ) -> Result<()> {
        // Per file: the retained groups (id, members) and the members of
        // discarded duplicates, gathered serially in walk order.
        let mut work: Vec<FileVerdicts> = Vec::with_capacity(files.len());
        for (file, groups) in files.iter().enumerate() {
            let mut mine = (Vec::new(), Vec::new());
            for (g, group) in groups.iter().enumerate() {
                if group.flags & !GRP_COMDAT != 0 {
                    return Err(Error::Format("unsupported SHT_GROUP flags"));
                }
                // A group that asked for no deduplication is kept outright;
                // a deduplicating one is kept only if its claim won.
                let keep = group.flags & GRP_COMDAT == 0
                    || group.signature.is_empty()
                    || kept
                        .get(file)
                        .and_then(|row| row.get(g))
                        .copied()
                        .unwrap_or(false);
                let sections: Vec<u16> = group
                    .members
                    .iter()
                    .filter_map(|&m| u16::try_from(m).ok())
                    .collect();
                if !keep {
                    mine.1.extend(sections);
                } else if sections.len() >= 2 {
                    // A group of one has no sibling to keep alive; it
                    // claims no id, exactly as `record_group` declines.
                    let id = u32::try_from(self.retained.len())
                        .map_err(|_| Error::OutOfRange("section groups"))?;
                    self.retained.push((file, sections.clone()));
                    mine.0.push((id, sections));
                }
            }
            work.push(mine);
        }
        // Parallel half: each file's membership row, sized once to its
        // highest touched section.
        if self.members.rows.len() < files.len() {
            self.members.rows.resize_with(files.len(), Vec::new);
        }
        let counts: Vec<usize> = self
            .members
            .rows
            .par_iter_mut()
            .zip(&work)
            .map(|(row, (retained, discarded))| {
                let top = retained
                    .iter()
                    .flat_map(|(_, s)| s.iter())
                    .chain(discarded.iter())
                    .copied()
                    .max();
                if let Some(top) = top {
                    let need = usize::from(top).saturating_add(1);
                    if row.len() < need {
                        row.resize(need, Membership::default());
                    }
                }
                for (id, sections) in retained {
                    for &section in sections {
                        if let Some(slot) = row.get_mut(usize::from(section)) {
                            slot.group = *id;
                        }
                    }
                }
                let mut newly = 0usize;
                for &section in discarded {
                    if let Some(slot) = row.get_mut(usize::from(section))
                        && !slot.discarded
                    {
                        slot.discarded = true;
                        newly = newly.saturating_add(1);
                    }
                }
                newly
            })
            .collect();
        for n in counts {
            self.discarded = self.discarded.saturating_add(n);
        }
        Ok(())
    }

    /// Whether the input section `(file, section)` is a member of a discarded
    /// duplicate group. Consulted by symbol resolution and section assignment.
    pub fn is_discarded(&self, file: usize, section: u16) -> bool {
        // Inputs without duplicate groups discard nothing, which is the common
        // case; skip the lookup rather than indexing every section of every
        // file.
        self.discarded != 0 && self.members.get(file, section).discarded
    }

    /// The number of discarded member sections, for diagnostics.
    pub const fn discarded_count(&self) -> usize {
        self.discarded
    }

    /// The sections that belong to `(file, section)`'s group, or an empty
    /// slice when it is in none.
    ///
    /// A section group says its members are kept or dropped together. Garbage
    /// collection follows relocations, and a secondary member with no incoming
    /// relocation -- the `.gcc_except_table` beside a function, a
    /// `.data.rel.ro` slice the group's code indexes into -- has none, so it
    /// was swept while its siblings stayed. lld keeps the group whole by
    /// walking `nextInSectionGroup` when it marks a member
    /// (`lld/ELF/MarkLive.cpp`); this is the same edge, held as a
    /// list rather than a ring.
    ///
    /// Every retained group is recorded, not just the COMDAT ones: the flag
    /// decides whether duplicates may be folded, and says nothing about
    /// whether the members belong together, which the group itself already
    /// said.
    pub fn group_members(&self, file: usize, section: u16) -> &[u16] {
        if self.retained.is_empty() {
            return &[];
        }
        let group = self.members.get(file, section).group;
        self.retained
            .get(group as usize)
            .map_or(&[], |(_, members)| members.as_slice())
    }

    /// Whether `(file, section)` belongs to a retained group. Garbage
    /// collection asks the inverse question from the one
    /// [`Self::group_members`] answers -- "does membership bind this section"
    /// rather than "which sections does membership bind" -- when it decides
    /// whether following a relocation into a section would drag the section's
    /// whole group in behind it.
    pub fn is_grouped(&self, file: usize, section: u16) -> bool {
        !self.retained.is_empty()
            && self.members.get(file, section).group != NO_GROUP
    }

    /// Scans one file's section groups, keeping the first COMDAT group per
    /// signature and recording the members of every later duplicate as
    /// discarded.
    ///
    /// Only `GRP_COMDAT` asks for deduplication. The flag is what distinguishes
    /// "these sections belong together" from "these sections belong together
    /// and one copy is enough": a group without it is kept in full, however
    /// many inputs carry the same signature, because nothing has said the
    /// copies are interchangeable. Folding one anyway drops bytes a linker was
    /// never told were redundant.
    ///
    /// lld reads the same word before consulting the signature, and rejects a
    /// flag that is neither zero nor `GRP_COMDAT`
    /// (`lld/ELF/InputFiles.cpp`).
    ///
    /// That refusal is followed here rather than softened. A
    /// flag word carrying anything else is asking for semantics neither linker
    /// implements -- the remaining bits are the `GRP_MASKOS`/`GRP_MASKPROC`
    /// ranges, whose meaning belongs to an ABI xold does not know -- and
    /// guessing at it means guessing which copies of a definition may be
    /// dropped. There is no quiet answer to that: keeping the group could
    /// duplicate a definition the ABI wanted folded, and folding it could drop
    /// one it wanted kept.
    ///
    /// Groups with an empty signature are skipped: a nameless group cannot be
    /// matched across files, so it is neither kept nor discarded (its members
    /// pass through unchanged).
    pub fn scan(&mut self, file: usize, groups: &[Group<'_>]) -> Result<()> {
        for group in groups {
            if group.flags & !GRP_COMDAT != 0 {
                return Err(Error::Format("unsupported SHT_GROUP flags"));
            }
            if group.flags & GRP_COMDAT == 0 || group.signature.is_empty() {
                // Not a deduplication request, but still a group: its members
                // are kept or dropped together.
                self.record_group(file, group.members)?;
                continue;
            }
            let hash = group.signature_hash;
            let won = self
                .kept
                .get_mut(shard_of(hash))
                .is_some_and(|set| set.insert(group.signature, hash));
            if won {
                self.record_group(file, group.members)?;
            } else {
                self.record_members(file, group.members);
            }
        }
        Ok(())
    }

    /// Records a retained group's membership, so collection can keep it whole.
    ///
    /// Called for every group the scan does not discard, including the ones it
    /// passes over for deduplication: a group without `GRP_COMDAT`, and one
    /// with no signature to match across files. Both still say their members
    /// belong together.
    fn record_group(&mut self, file: usize, members: &[u32]) -> Result<()> {
        let id = u32::try_from(self.retained.len())
            .map_err(|_| Error::OutOfRange("section groups"))?;
        let sections: Vec<u16> = members
            .iter()
            .filter_map(|&m| u16::try_from(m).ok())
            .collect();
        if sections.len() < 2 {
            // A group of one has no sibling to keep alive.
            return Ok(());
        }
        for &section in &sections {
            if let Some(slot) = self.members.slot(file, section) {
                slot.group = id;
            }
        }
        self.retained.push((file, sections));
        Ok(())
    }

    /// Adds every member section index of a discarded group to the discard set.
    /// Indices that do not fit in `u16` are skipped: xold indexes sections by
    /// `u16`, so an out-of-range member is unreachable anyway.
    fn record_members(&mut self, file: usize, members: &[u32]) {
        for &raw in members {
            let Ok(section) = u16::try_from(raw) else {
                continue;
            };
            if let Some(slot) = self.members.slot(file, section)
                && !slot.discarded
            {
                slot.discarded = true;
                self.discarded = self.discarded.saturating_add(1);
            }
        }
    }
}

/// Buckets every valid COMDAT claim by its signature's shard, per file in
/// parallel, then joins the files' buckets in file order so each shard sees
/// its claims exactly as the serial walk would have.
fn partition_claims<'data>(
    files: &[Vec<Group<'data>>],
) -> Vec<Vec<Claim<'data>>> {
    let per_file: Vec<Vec<Vec<Claim<'data>>>> = files
        .par_iter()
        .enumerate()
        .map(|(file, groups)| {
            let mut buckets: Vec<Vec<Claim<'data>>> =
                (0..SHARDS).map(|_| Vec::new()).collect();
            for (g, group) in groups.iter().enumerate() {
                // Only a valid deduplication request races. Anything else is
                // either recorded unconditionally or refused, both of which
                // the verdict pass decides without a claim.
                if group.flags & !GRP_COMDAT != 0
                    || group.flags & GRP_COMDAT == 0
                    || group.signature.is_empty()
                {
                    continue;
                }
                let claim = Claim {
                    file: u32::try_from(file).unwrap_or(u32::MAX),
                    group: u32::try_from(g).unwrap_or(u32::MAX),
                    hash: group.signature_hash,
                    signature: group.signature,
                };
                if let Some(bucket) = buckets.get_mut(shard_of(claim.hash)) {
                    bucket.push(claim);
                }
            }
            buckets
        })
        .collect();
    // Joining is per shard, so it fans out too: shard `s` copies bucket `s`
    // of every file, in file order, which is the order the race depends on.
    (0..SHARDS)
        .into_par_iter()
        .map(|s| {
            let total =
                per_file.iter().map(|b| b.get(s).map_or(0, Vec::len)).sum();
            let mut shard = Vec::with_capacity(total);
            for buckets in &per_file {
                if let Some(bucket) = buckets.get(s) {
                    shard.extend_from_slice(bucket);
                }
            }
            shard
        })
        .collect()
}
