//! Record-level splitting of `.eh_frame`.
//!
//! A `.eh_frame` section is a sequence of independent CIE and FDE records
//! closed by a zero `length` word, and each FDE describes exactly one
//! function. Neither lld nor mold concatenates the input sections: both split
//! every `.eh_frame` into its records and re-emit them packed, then close the
//! output with a terminator of their own. This module builds the piece map
//! that lets xold do the same, and it runs on every link, for two reasons.
//!
//! Correctness first. Members are placed with their `sh_addralign` honoured,
//! and a `.eh_frame` whose size is not a multiple of that alignment is
//! followed by inter-member padding. Four zero bytes are a zero `length`,
//! which is the end-of-section terminator, so a concatenation truncates at the
//! first padded member for every walk over it -- `.eh_frame_hdr` construction
//! and the runtime unwinder alike. Packing the records leaves no padding to
//! misread, and dropping each input's own terminator leaves exactly one, at
//! the end, where it belongs.
//!
//! Then collection. When `--gc-sections` drops a function, its FDE describes
//! nothing: keeping it wastes image bytes and leaves a record whose
//! `initial_location` relocates against an unplaced section. A link that
//! collects nothing keeps every record; only the layout changes.
//!
//! For every `.eh_frame` member this module records each record's offset and
//! length, decides which records survive, and assigns the survivors their
//! offsets within the member's output contribution. The writer copies piece by
//! piece and rebases each relocation onto its piece's new offset; layout sizes
//! the member from the surviving bytes and `.eh_frame_hdr` from the surviving
//! FDE count.
//!
//! A CIE is kept unless another member already contributed the same one. It
//! describes a calling convention rather than a function, so every object
//! compiled with the same flags emits an identical copy, and the FDEs that
//! reference it by backwards offset can reach one copy as easily as their own.
//! [`cie`] decides which copies are the same; this module drops the rest and
//! records where their survivor went.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::cie::{CieKey, CiePool, CieRef, Claim};
use crate::{
    elf::{Rela64, SymbolTable},
    error::{Error, Result},
    gc::reloc_target,
    linker::{Context, in_input_order},
    output::{OutKind, OutputSections},
};

/// The zero `length` word that closes the output `.eh_frame`.
const TERMINATOR: u64 = 4;

/// One record of an input `.eh_frame` section.
#[derive(Clone, Copy, Debug)]
pub struct Piece {
    /// Offset of the record within the input section.
    pub in_off: u64,
    /// Length of the record in bytes.
    pub len: u64,
    /// Offset of the record within the member's output contribution, or `None`
    /// if the record was dropped.
    pub out_off: Option<u64>,
    /// Whether this record is an FDE (a CIE otherwise).
    pub fde: bool,
}

impl Piece {
    /// Whether `offset` (an input-section offset) falls inside this record.
    pub const fn covers(self, offset: u64) -> bool {
        offset >= self.in_off && offset < self.in_off.saturating_add(self.len)
    }
}

/// The surviving records of one input `.eh_frame` section.
pub struct EhMember {
    pieces: Vec<Piece>,
    live_size: u64,
    live_fdes: u64,
    /// The CIEs this member dropped as duplicates, by their input offset,
    /// paired with the copy that stands in for them. An FDE naming one of
    /// these reaches its CIE in another member, so the writer computes the
    /// distance from the two members' addresses rather than within this one.
    /// Empty for the overwhelming majority of members, which hold one or two
    /// CIEs and share both.
    cie_redirect: Vec<(u64, CieRef)>,
}

impl EhMember {
    /// The records of this member, in input order, kept and dropped alike.
    pub fn pieces(&self) -> &[Piece] {
        &self.pieces
    }

    /// Where the CIE this member dropped from `in_off` ended up, if it dropped
    /// one there.
    pub fn cie_redirect(&self, in_off: u64) -> Option<CieRef> {
        self.cie_redirect
            .iter()
            .find(|(off, _)| *off == in_off)
            .map(|(_, at)| *at)
    }

    /// Total bytes the surviving records occupy.
    pub const fn live_size(&self) -> u64 {
        self.live_size
    }

    /// The record covering `offset`, if any. Used by the writer to rebase a
    /// relocation onto its record's output offset.
    pub fn piece_at(&self, offset: u64) -> Option<Piece> {
        super::record_at(&self.pieces, offset)
            .and_then(|i| self.pieces.get(i))
            .copied()
    }
}

/// The `.eh_frame` piece map for a whole link. Empty when no input carried an
/// `.eh_frame`, in which case there is nothing to place and nothing to copy.
#[derive(Default)]
pub struct EhFramePlan {
    by_section: FxHashMap<(usize, u16), EhMember>,
    sizes: Vec<u64>,
    live_fdes: u64,
}

impl EhFramePlan {
    /// The piece map of one `.eh_frame` member, if the plan covers it.
    pub fn member(&self, file: usize, section: u16) -> Option<&EhMember> {
        self.by_section.get(&(file, section))
    }

    /// The number of FDE records the output will contain, which is the number
    /// of `.eh_frame_hdr` table entries to reserve.
    pub const fn live_fdes(&self) -> u64 {
        self.live_fdes
    }

    /// The output size of every covered member, in the order the members sit
    /// in their output section, so the caller can resize them in one pass.
    /// The last entry includes the terminator that closes the section.
    pub fn sizes(&self) -> &[u64] {
        &self.sizes
    }
}

/// Builds the piece map for `ctx`.
///
/// An FDE goes when the section its first relocation names did not reach the
/// output; a CIE, and a record with no relocation at all, always stays, since
/// there is nothing to prove either dead. Call after `--gc-sections` and ICF
/// have settled the member set: liveness is the member set itself, read
/// straight out of `ctx.outputs`. The caller then resizes the members to
/// [`EhFramePlan::sizes`].
///
/// The question is asked on every link rather than only under
/// `--gc-sections`, because collection is not the only thing that retires a
/// section. Identical code folding does: a folded function's bytes are gone
/// and its FDE would describe the representative's address, leaving two
/// records covering one PC with different LSDA pointers, and the unwinder
/// would take whichever the search table happened to keep. So does COMDAT
/// deduplication, whose discarded members leave FDEs behind whenever the
/// compiler puts `.eh_frame` outside the group. lld reads the same member set
/// for the same reason, and names both cases in doing so
/// (`EhFrameSection::isFdeLive`, `lld/ELF/SyntheticSections.cpp`:
/// "FDEs for garbage-collected or merged-by-ICF sections ... are dead").
pub fn build(ctx: &Context<'_>) -> Result<EhFramePlan> {
    let mut plan = EhFramePlan::default();
    let Some(out) = ctx.outputs.section(OutKind::EhFrame) else {
        return Ok(plan);
    };
    let live = LiveSections::new(&ctx.outputs);
    // Splitting one member reads that member alone, so the members split in
    // parallel and the plan is assembled from the rows in member order --
    // which is the order the output section places them in, and the order the
    // sizes have to arrive in.
    let mut split: Vec<Result<SplitMember>> = Vec::new();
    out.members
        .par_iter()
        .map(|m| split_member(ctx, &live, m.file, m.section))
        .collect_into_vec(&mut split);
    plan.sizes = Vec::with_capacity(out.members.len());
    // The CIE pool is filled here rather than in the parallel walk above, and
    // in this order: the copy a duplicate resolves to has to be one the image
    // places before it, since an FDE names its CIE by a backwards distance,
    // and it has to be the same copy on every run whatever the thread count.
    // Both follow from claiming the CIEs member by member, in output order.
    let mut pool = CiePool::default();
    for (m, member) in out.members.iter().zip(in_input_order(split)?) {
        let member = share_cies(&mut pool, m.file, m.section, member);
        plan.live_fdes = plan.live_fdes.saturating_add(member.live_fdes);
        plan.sizes.push(member.live_size);
        plan.by_section.insert((m.file, m.section), member);
    }
    // One zero `length` word closes the section. glibc's
    // `classify_object_over_fdes` reads it as the end of this object's unwind
    // data, and every input's own terminator was dropped above. lld appends
    // the same four bytes in `EhFrameSection::finalizeContents` ("Thus we add
    // one unconditionally"), mold in `EhFrameSection::construct` (".eh_frame
    // must end with a null word"). There is no synthetic member to hang them
    // on, so the last member carries them; the writer leaves them zero.
    if let Some(last) = plan.sizes.last_mut() {
        *last = last.saturating_add(TERMINATOR);
    }
    Ok(plan)
}

/// Every input section that survived into an output section, as one row of
/// flags per input file.
///
/// The set is dense and every FDE probes it: a `-ffunction-sections` build has
/// thousands of live sections per file, and one `.eh_frame` record per
/// function asks about one of them. Indexing the rows is what keeps that off a
/// hash of a `(file, section)` pair per record.
#[derive(Default)]
struct LiveSections {
    rows: Vec<Vec<bool>>,
}

impl LiveSections {
    /// The sections `outputs` kept.
    fn new(outputs: &OutputSections) -> Self {
        let mut live = Self::default();
        for out in outputs.iter() {
            for m in &out.members {
                live.insert(m.file, m.section);
            }
        }
        live
    }

    fn insert(&mut self, file: usize, section: u16) {
        if self.rows.len() <= file {
            self.rows.resize_with(file.saturating_add(1), Vec::new);
        }
        let Some(row) = self.rows.get_mut(file) else {
            return;
        };
        let at = usize::from(section);
        if row.len() <= at {
            row.resize(at.saturating_add(1), false);
        }
        if let Some(flag) = row.get_mut(at) {
            *flag = true;
        }
    }

    fn contains(&self, (file, section): (usize, u16)) -> bool {
        self.rows
            .get(file)
            .and_then(|row| row.get(usize::from(section)))
            .copied()
            .unwrap_or(false)
    }
}

/// One member's records and the two verdicts the split pass reaches about
/// them, before the CIE pool has had its say.
#[derive(Default)]
struct SplitMember {
    pieces: Vec<Piece>,
    fde: Vec<bool>,
    keep: Vec<bool>,
    /// The identity of every CIE this member holds, by record index. A record
    /// missing from this list is never shared: an FDE, or a CIE whose content
    /// after relocation this pass could not pin down.
    keys: Vec<(usize, CieKey)>,
}

/// Splits one `.eh_frame` member into records and decides which survive.
fn split_member(
    ctx: &Context<'_>,
    live: &LiveSections,
    file: usize,
    section: u16,
) -> Result<SplitMember> {
    let mut pieces = Vec::new();
    let mut fde = Vec::new();
    let Some(input) = ctx.files.get(file) else {
        return Ok(SplitMember::default());
    };
    let obj = input.object()?;
    let Some(shdr) = obj.sections().get(usize::from(section)) else {
        return Ok(SplitMember::default());
    };
    let data = obj.section_data(shdr)?;
    let end = super::records(data, &mut pieces, &mut fde);
    // Everything past the last record is dropped, so it has to be nothing but
    // the zero terminator an input may close its section with and whatever
    // padding follows it: all zero bytes. A tail with content in it is a
    // record the walk could not decode -- a 64-bit DWARF length, a record
    // running past the end of the section, a terminator with more records
    // behind it -- and re-emitting the section without it would leave the
    // image quietly short of unwind data. Refuse the link instead; lld
    // reports the decode failures it can see the same way, as "corrupted
    // .eh_frame" out of `EhInputSection::split`.
    if data.get(end..).unwrap_or_default().iter().any(|&b| b != 0) {
        return Err(Error::Format("undecodable .eh_frame record"));
    }
    // Both verdicts below read the same symbol table and relocation section,
    // so they are read once here. Either may be absent, which the callees
    // answer for themselves.
    let symtab = input.symbol_table()?;
    let entries = input.relocations(section)?;
    let keep =
        keep_flags(ctx, live, file, symtab.as_ref(), entries, &pieces, &fde);
    let keys = cie_keys(ctx, file, data, symtab.as_ref(), entries, &pieces);
    Ok(SplitMember {
        pieces,
        fde,
        keep,
        keys,
    })
}

/// The identity of every CIE in `pieces`, paired with its record index.
///
/// A CIE with no key is one this pass declines to share: the member carries no
/// symbol table to resolve its relocations through, or one of those
/// relocations names a symbol the link left undefined. Both leave the record's
/// final content unknown here, and an unknown is not a match.
fn cie_keys(
    ctx: &Context<'_>,
    file: usize,
    data: &[u8],
    symtab: Option<&SymbolTable<'_>>,
    entries: Option<&[Rela64]>,
    pieces: &[Piece],
) -> Vec<(usize, CieKey)> {
    let Some(symtab) = symtab else {
        return Vec::new();
    };
    let covering = cie_relocs(pieces, entries.unwrap_or_default());
    let mut keys = Vec::new();
    for (i, piece) in pieces.iter().enumerate() {
        if piece.fde {
            continue;
        }
        let start = usize::try_from(piece.in_off).unwrap_or(usize::MAX);
        let len = usize::try_from(piece.len).unwrap_or(0);
        let Some(bytes) = data.get(start..start.saturating_add(len)) else {
            continue;
        };
        let from = covering.partition_point(|(p, _, _)| *p < i);
        let to = covering.partition_point(|(p, _, _)| *p <= i);
        let rels = covering.get(from..to).unwrap_or_default();
        if let Some(key) = CieKey::new(ctx, symtab, file, bytes, rels) {
            keys.push((i, key));
        }
    }
    keys
}

/// The relocations landing in a CIE of `pieces`, as `(record index, offset
/// within the record, entry)` rows sorted by the first two fields.
///
/// `pieces` is ascending and non-overlapping, so each entry is placed by a
/// binary search, the same way [`super::first_reloc_syms`] places one. FDE
/// relocations are dropped here: only a CIE's take part in its identity, and a
/// member holds one or two CIEs against thousands of FDEs, so the rows are
/// collected flat rather than one growable row per record.
///
/// The sort makes the order the rows reach a [`CieKey`] a property of the
/// records rather than of the relocation table, which `ld -r` is free to
/// reorder. Without it two files holding the same CIE could disagree on their
/// key over nothing but the order their tables were written in.
fn cie_relocs(
    pieces: &[Piece],
    entries: &[Rela64],
) -> Vec<(usize, u64, Rela64)> {
    let mut covering: Vec<(usize, u64, Rela64)> = Vec::new();
    for r in entries {
        let at = r.r_offset.get();
        let Some(i) = super::record_at(pieces, at) else {
            continue;
        };
        let Some(piece) = pieces.get(i).filter(|p| !p.fde) else {
            continue;
        };
        covering.push((i, at.saturating_sub(piece.in_off), *r));
    }
    covering.sort_unstable_by_key(|(i, at, _)| (*i, *at));
    covering
}

/// Offers one member's CIEs to `pool` and assigns the survivors their offsets.
///
/// A CIE the pool already holds is dropped from this member and recorded as a
/// redirect, so the writer can still point its FDEs at a CIE. A CIE the pool
/// takes is placed by [`assign_offsets`] first and reported back afterwards,
/// since that is when it has an offset to report.
fn share_cies(
    pool: &mut CiePool,
    file: usize,
    section: u16,
    mut split: SplitMember,
) -> EhMember {
    let mut claimed: Vec<(usize, usize)> = Vec::new();
    let mut shared: Vec<(usize, CieRef)> = Vec::new();
    for (i, key) in split.keys.drain(..) {
        let at = CieRef {
            file,
            section,
            out_off: 0,
        };
        match pool.claim(key, at) {
            Claim::Kept(slot) => claimed.push((i, slot)),
            Claim::Shared(found) => {
                if let Some(flag) = split.keep.get_mut(i) {
                    *flag = false;
                }
                shared.push((i, found));
            }
        }
    }
    let mut member = assign_offsets(split.pieces, &split.fde, &split.keep);
    for (i, slot) in claimed {
        if let Some(off) = member.pieces.get(i).and_then(|p| p.out_off) {
            pool.place(slot, off);
        }
    }
    member.cie_redirect = shared
        .into_iter()
        .filter_map(|(i, at)| member.pieces.get(i).map(|p| (p.in_off, at)))
        .collect();
    member
}

/// Decides, per record, whether it survives.
///
/// A CIE is always kept: it is the format the FDEs sharing it are written in,
/// not a description of any one function. An FDE is kept when the section its
/// first relocation names reached the output, which `live` enumerates.
///
/// An FDE whose first relocation is absent, or names something no live section
/// backs, is dropped. Keeping one re-emits its `initial_location` verbatim,
/// written against the *input* file, so the image carries an unwind entry
/// claiming a range that belongs to nothing. `.eh_frame_hdr`'s own filter
/// hides such an entry from the binary search, but the search table is not the
/// only reader: a linear scanner walks the records themselves, as static
/// glibc's `classify_object_over_fdes` does, and ingests the bogus range. lld
/// drops the same shape in `isFdeLive`.
#[allow(clippy::too_many_arguments)]
fn keep_flags(
    ctx: &Context<'_>,
    live: &LiveSections,
    file: usize,
    symtab: Option<&SymbolTable<'_>>,
    entries: Option<&[Rela64]>,
    pieces: &[Piece],
    fde: &[bool],
) -> Vec<bool> {
    // A record is kept unless it is an FDE this pass can place. Starting from
    // "kept" is right for the CIEs, which this pass never revisits; whether
    // one of those is a duplicate of a CIE another member already contributed
    // is a question for the pool, after every member has been split.
    let mut keep = vec![true; pieces.len()];
    let is_fde = |i: usize| fde.get(i).copied().unwrap_or(false);
    // No symbol table or no relocation section means no FDE here can name its
    // function, so none of them can be shown to describe live code.
    let (Some(symtab), Some(entries)) = (symtab, entries) else {
        for (i, flag) in keep.iter_mut().enumerate() {
            *flag = !is_fde(i);
        }
        return keep;
    };
    // The record's first relocation names the function it describes, matching
    // lld's `EhSectionPiece::firstRelocation`. Later relocations point at the
    // LSDA, which lives or dies with that same function.
    let first = super::first_reloc_syms(pieces, entries);
    for (i, flag) in keep.iter_mut().enumerate() {
        if !is_fde(i) {
            continue;
        }
        *flag = first
            .get(i)
            .copied()
            .flatten()
            .and_then(|sym_idx| reloc_target(ctx, symtab, file, sym_idx))
            .is_some_and(|target| live.contains(target));
    }
    keep
}

/// Assigns each surviving record its offset within the member's output
/// contribution and totals the live bytes and FDE count.
fn assign_offsets(
    mut pieces: Vec<Piece>,
    fde: &[bool],
    keep: &[bool],
) -> EhMember {
    debug_assert_eq!(pieces.len(), fde.len(), "one kind flag per record");
    let mut cursor = 0u64;
    let mut live_fdes = 0u64;
    for (i, piece) in pieces.iter_mut().enumerate() {
        if !keep.get(i).copied().unwrap_or(true) {
            piece.out_off = None;
            continue;
        }
        piece.out_off = Some(cursor);
        cursor = cursor.saturating_add(piece.len);
        if fde.get(i).copied().unwrap_or(false) {
            live_fdes = live_fdes.saturating_add(1);
        }
    }
    EhMember {
        pieces,
        live_size: cursor,
        live_fdes,
        cie_redirect: Vec::new(),
    }
}
