//! Section copy and in-place relocation: the parallel engine that copies each
//! allocated input section's bytes into the image and patches its relocations.

use std::ops::Range;

use rayon::prelude::*;

use crate::{
    ehframe::{EhMember, Piece},
    elf::{
        Rela64, SymbolTable,
        constants::{
            SHF_ALLOC, SHN_LORESERVE, SHN_UNDEF, SHT_NOBITS, STB_LOCAL,
        },
    },
    error::{Error, Result},
    layout::{FileResolver, Layout},
    linker::Context,
    output::OutKind,
    reloc::{
        RelExpr, RelaxSite, Target, apply_target, relax_covers_target,
        relax_lead_target, relax_required_target, relax_span_target,
        relax_spans_type, relax_target, relax_trail_target, riscv, spec_target,
    },
    symbol::SymbolId,
};

/// Copies every allocated, file-backed input section and applies its
/// relocations in place.
///
/// The work is parallelised per input section: each member writes a disjoint
/// region of the image, so the applications never alias. The image is split
/// into one mutable slice per member serially (via `split_at_mut`), then the
/// per-member copy and relocation patching run concurrently with `rayon`. No
/// locking is required: a task only ever touches its own slice.
pub(super) fn copy_sections(
    ctx: &Context<'_>,
    target: Target,
    layout: &Layout,
    base: u64,
    relax: bool,
    image: &mut [u8],
) -> Result<()> {
    let mut items = copy_work_items(ctx, layout, base);
    // Which files lost a section to garbage collection, a COMDAT group or a
    // kind this linker does not emit. Computed once: the per-relocation check
    // is then a bool the member already carries.
    let discarded = files_with_discarded_sections(ctx, layout);
    for item in &mut items {
        item.has_discarded = discarded.get(item.file).copied().unwrap_or(false);
    }
    let mut slices = split_member_slices(image, &items)?;
    slices
        .par_iter_mut()
        .enumerate()
        .map(|(i, slice)| {
            copy_member_into(ctx, target, layout, relax, items[i], slice)
        })
        .collect::<Result<Vec<()>>>()?;
    Ok(())
}

/// One unit of copy work: a placed member and the disjoint image range it owns.
#[derive(Copy, Clone)]
pub(super) struct CopyItem {
    pub(super) file: usize,
    pub(super) section: u16,
    /// Absolute byte offset of the section within the image.
    pub(super) start: usize,
    /// On-disk size of the section in bytes.
    pub(super) len: usize,
    /// Virtual address of the section (used as the relocation placement base).
    pub(super) vaddr: u64,
    /// The value to write at an absolute site whose target section placement
    /// dropped, or `None` when such a site is not to be tombstoned. Set only
    /// for `.debug_*` members; see [`Patch::tombstone`].
    pub(super) tombstone: Option<u64>,
    /// Whether a reference into an ICF-folded section is tombstoned too.
    ///
    /// Folding leaves the losing section placed at the winner's address, so
    /// without this a second compilation unit describes code it no longer
    /// owns. lld tombstones the folded case everywhere except `.debug_line`,
    /// where keeping the real address is what lets a debugger still break on
    /// a folded-in function (`relocateNonAlloc`).
    pub(super) folded_tombstone: bool,
    /// Whether this member's file had any section placement dropped.
    ///
    /// The other half of the tombstone question. A debug member points a
    /// dropped reference at a tombstone; an allocated one cannot -- there is
    /// no value that means "nowhere" in executable code -- so the reference is
    /// an error. Recorded per file so the check costs one already-loaded bool
    /// per relocation instead of a symbol-table read.
    pub(super) has_discarded: bool,
}

/// Builds the flat list of copy work items in ascending image-offset order, so
/// the slices can be carved out with successive `split_at_mut` calls.
///
/// Output sections are walked in placement order rather than slot order, since
/// the TLS pair is placed after the init/fini arrays but numbered before them.
/// Within a section the members are already in ascending order, so the
/// concatenation is globally ascending.
fn copy_work_items(
    ctx: &Context<'_>,
    layout: &Layout,
    base: u64,
) -> Vec<CopyItem> {
    let mut out = Vec::new();
    for kind in OutKind::FILE_ORDER {
        let Some(section) = ctx.outputs.section(kind) else {
            continue;
        };
        out.extend(section.members.iter().filter_map(|m| {
            // `Member::size` is already the on-disk contribution: the
            // collector rewrote it for any `.eh_frame` member it split, so
            // there is nothing to re-derive from the input here.
            let len = usize::try_from(m.size).unwrap_or(0);
            let vaddr = layout.section_vaddr(m.file, m.section)?;
            (len != 0).then(|| CopyItem {
                tombstone: None,
                folded_tombstone: false,
                has_discarded: false,
                file: m.file,
                section: m.section,
                start: usize::try_from(vaddr.wrapping_sub(base))
                    .unwrap_or(usize::MAX),
                len,
                vaddr,
            })
        }));
    }
    out
}

/// Per file: whether any of its sections was dropped rather than placed.
///
/// A file with nothing dropped is the overwhelming common case, and the answer
/// short-circuits the whole check for it.
fn files_with_discarded_sections(
    ctx: &Context<'_>,
    layout: &Layout,
) -> Vec<bool> {
    (0..ctx.files.len())
        .map(|file| {
            let Some(input) = ctx.files.get(file) else {
                return false;
            };
            let Ok(obj) = input.object() else {
                return false;
            };
            obj.sections().iter().enumerate().any(|(i, s)| {
                let shndx = u16::try_from(i).unwrap_or(u16::MAX);
                s.sh_flags.get() & SHF_ALLOC != 0
                    && s.sh_type.get() != SHT_NOBITS
                    && s.sh_size.get() != 0
                    && !ctx.merge.is_merged(file, shndx)
                    && layout.section_vaddr(file, shndx).is_none()
            })
        })
        .collect()
}

/// Splits `image` into one mutable slice per work item. The items must be in
/// ascending `start` order with non-overlapping `[start, start + len)` ranges;
/// alignment padding between members is skipped (it was zero-filled when the
/// image was sized).
pub(super) fn split_member_slices<'a>(
    image: &'a mut [u8],
    items: &[CopyItem],
) -> Result<Vec<&'a mut [u8]>> {
    let mut out: Vec<&'a mut [u8]> = Vec::with_capacity(items.len());
    let mut cursor = 0usize;
    let mut remaining = image;
    for item in items {
        if item.start < cursor {
            return Err(Error::OutOfRange("member placement"));
        }
        let gap = item.start - cursor;
        if gap > 0 {
            // A member placed past the end of the sized image is a layout
            // bug, and `split_at_mut` would panic on it rather than report
            // one.
            if gap > remaining.len() {
                return Err(Error::OutOfRange("member placement"));
            }
            let (_, right) = remaining.split_at_mut(gap);
            remaining = right;
            cursor += gap;
        }
        if item.len > remaining.len() {
            return Err(Error::OutOfRange("member copy"));
        }
        let (chunk, right) = remaining.split_at_mut(item.len);
        out.push(chunk);
        remaining = right;
        cursor += item.len;
    }
    Ok(out)
}

/// Copies one member's bytes into its disjoint slice and patches the member's
/// relocations in place. Runs in parallel across members.
pub(super) fn copy_member_into(
    ctx: &Context<'_>,
    target: Target,
    layout: &Layout,
    relax: bool,
    item: CopyItem,
    slice: &mut [u8],
) -> Result<()> {
    let input = ctx
        .files
        .get(item.file)
        .ok_or(Error::OutOfRange("input file index"))?;
    let obj = input.object()?;
    let Some(shdr) = obj.sections().get(usize::from(item.section)) else {
        return Ok(());
    };
    // A merge-pool carrier contributes the deduplicated pool rather than its
    // own bytes; every other contributor was folded onto it and copies
    // nothing.
    let pool = ctx.merge.carried_bytes(item.file, item.section);
    let data = match pool {
        Some(bytes) => bytes,
        None => obj.section_data(shdr)?,
    };
    // An `.eh_frame` member contributes its records packed at the offsets the
    // splitter assigned; every other member is a straight copy.
    let split = ctx.eh.member(item.file, item.section);
    if let Some(member) = split {
        copy_pieces(member, layout, item.vaddr, data, slice)?;
    } else {
        if slice.len() < data.len() {
            return Err(Error::OutOfRange("section copy"));
        }
        slice[..data.len()].copy_from_slice(data);
    }

    let Some(entries) = input.relocations(item.section)? else {
        return Ok(());
    };
    // A carrier contributes the deduplicated pool, so every offset inside it
    // moved and `r_offset` names a byte that is no longer there. Remapping it
    // through `MergePlan::remap` would place this member's own relocations
    // correctly and still leave the link wrong: the pool is keyed on content
    // alone, so two byte-identical pieces with different relocations fold into
    // one slot that both would write, and every contributor other than the
    // carrier was dropped from the output section along with its relocations.
    // Refuse instead. mold declines to make such a section mergeable in the
    // first place (`relsec_idx != -1` in `convert_mergeable_sections`), which
    // is where the case belongs; until then this is the backstop that keeps it
    // from linking quietly.
    if pool.is_some() && !entries.is_empty() {
        return Err(Error::Format("relocations in a merged section"));
    }
    let resolver = layout.resolver(item.file);
    // The RISC-V PCREL_HI20/LO12 pair points the LO12's symbol at the paired
    // HI20 instruction, not the real target, so the LO12 can only be resolved
    // against its sibling relocations. Rewrite those entries per section
    // before the apply loop; the pass is deterministic and touches only this
    // member's slice of data, so it stays safe inside the parallel copy task.
    let rewritten = if target == Target::Riscv64 {
        riscv::rewrite_pcrel_pairs(entries, item.vaddr, &resolver)?
    } else {
        None
    };
    Patch {
        ctx,
        layout,
        target,
        relax,
        item,
        entries: rewritten.as_deref().unwrap_or(entries),
        // The symbol table is read for a file with merged sections, whose
        // relocation addends remap through it, and for a debug member, whose
        // tombstone check asks where each local symbol was defined. Neither
        // is the common case, and deriving it per member is not free --
        // neither is the addend probe per relocation it feeds, which a file
        // with nothing merged can never hit.
        symtab: if ctx.merge.file_has_merges(item.file)
            || item.tombstone.is_some()
            || item.has_discarded
        {
            input.symbol_table()?
        } else {
            None
        },
        split,
    }
    .apply_all(&resolver, slice)
}

/// One member's relocation pass: the entries to patch and the facts that fix
/// where each one lands.
///
/// The pass runs twice over the same entries -- once to collect the byte
/// ranges the lowerings will consume, once to apply what is left -- so both
/// walks resolve a site through [`Patch::site`] rather than each deriving the
/// offset and addend for itself.
struct Patch<'a, 'i> {
    ctx: &'a Context<'i>,
    layout: &'a Layout,
    target: Target,
    relax: bool,
    item: CopyItem,
    /// The member's RELA entries, after any per-arch rewrite.
    entries: &'a [Rela64],
    /// The file's symbol table, present only when the link merged something.
    symtab: Option<SymbolTable<'a>>,
    /// The `.eh_frame` split plan, when the collector split this member.
    split: Option<&'a EhMember>,
}

impl Patch<'_, '_> {
    /// Where one entry lands and the addend to resolve it with, or `None` when
    /// it has no slot at all.
    ///
    /// A split `.eh_frame` member moved its records, so a relocation lands at
    /// its record's new offset and one inside a dropped record lands nowhere.
    ///
    /// A reference into a merged section names the section and carries the
    /// byte offset in its addend. The content moved when it was deduplicated,
    /// so the offset is rewritten to where the surviving copy landed; the
    /// symbol itself resolves to the pool base.
    fn site(&self, r: &Rela64) -> Option<(u64, i64)> {
        let slot = match self.split {
            None => r.r_offset.get(),
            Some(member) => {
                let piece = member.piece_at(r.r_offset.get())?;
                let out_off = piece.out_off?;
                out_off
                    .wrapping_add(r.r_offset.get().wrapping_sub(piece.in_off))
            }
        };
        // One bit answers whether this relocation can name moved content at
        // all; only then is the symbol row read and the piece map probed.
        let addend = if self.ctx.merge.names_merged(self.item.file, r.sym()) {
            self.symtab.as_ref().map_or_else(
                || r.r_addend.get(),
                |symtab| self.ctx.merge.addend(symtab, self.item.file, r),
            )
        } else {
            r.r_addend.get()
        };
        Some((slot, addend))
    }

    /// Collects the byte ranges this member's lowerings will consume.
    ///
    /// Only a type that reports a range is examined, so a member with no
    /// dynamic TLS pays one type check per relocation and allocates nothing.
    /// The check comes before [`Self::site`] because a site is not free: it
    /// costs a merge-plan probe for the addend, which this pass would then
    /// throw away for every relocation but a handful.
    fn consumed(
        &self,
        resolver: &FileResolver<'_>,
        slice: &[u8],
        out: &mut Vec<Consumed>,
    ) {
        for r in self.entries {
            if !relax_spans_type(self.target, r.r_type()) {
                continue;
            }
            let Some((slot, addend)) = self.site(r) else {
                continue;
            };
            let site = RelaxSite {
                r_type: r.r_type(),
                sym: sym_of(r),
                addend,
                place: self.item.vaddr.wrapping_add(slot),
                data: slice,
                slot,
            };
            if let Some(range) = relax_span_target(self.target, site, resolver)
            {
                out.push(Consumed { range, slot });
            }
        }
    }

    /// Applies every relocation the lowerings did not consume.
    fn apply_all(
        &self,
        resolver: &FileResolver<'_>,
        slice: &mut [u8],
    ) -> Result<()> {
        let mut consumed = Vec::new();
        self.consumed(resolver, slice, &mut consumed);
        for r in self.entries {
            let Some((slot, addend)) = self.site(r) else {
                continue;
            };
            if consumed.iter().any(|c| c.swallows(slot)) {
                continue;
            }
            self.apply_entry(
                RelocApply {
                    r,
                    addend,
                    target: self.target,
                    relax: self.relax,
                    vaddr: self.item.vaddr,
                    out_off: slot,
                },
                resolver,
                slice,
            )?;
        }
        Ok(())
    }

    /// Applies one RELA entry to its slot within the member's slice.
    ///
    /// `site.out_off` is the slot's offset within that slice, which equals the
    /// relocation's `r_offset` except in a split `.eh_frame` member, whose
    /// records moved; the placement address the value is measured against is
    /// the section's virtual address plus that offset.
    ///
    /// A rewrite of the surrounding instructions runs first (see
    /// [`try_rewrite`]) and, when it happens, replaces the value computation
    /// entirely.
    fn apply_entry(
        &self,
        site: RelocApply<'_>,
        resolver: &FileResolver<'_>,
        slice: &mut [u8],
    ) -> Result<()> {
        let r_type = site.r.r_type();
        let spec = spec_target(site.target, r_type)?;
        if spec.expr == RelExpr::None {
            return Ok(());
        }
        let width = spec.write.width();
        if spec.expr == RelExpr::Abs
            && let Some(value) = self.tombstone(site.r)
        {
            return write_tombstone(site.out_off, width, value, slice);
        }
        self.check_not_discarded(site.r)?;
        let sym = sym_of(site.r);
        // Optional relaxations run only when the link asks for them; a
        // mandatory lowering runs regardless, because nothing else can resolve
        // the site.
        let required =
            relax_required_target(site.target, r_type, sym, resolver);
        if site.relax || required {
            match try_rewrite(site, width, resolver, slice)? {
                Rewrite::Done => return Ok(()),
                // A mandatory rewrite that did not happen ends the link. The
                // value computation below is not a fallback for one: nothing
                // allocated the storage such a site would name, so it would
                // measure a displacement from address zero and the image would
                // link clean and fault at run time. An optional rewrite has a
                // real fallback -- the unrelaxed form the input already holds
                // -- and falls through to it.
                Rewrite::Declined(why) if required => {
                    return Err(self.unlowered(site.r, why));
                }
                Rewrite::Declined(_) => {}
            }
        }
        let slot_off = usize::try_from(site.out_off).unwrap_or(usize::MAX);
        let Some(slot) =
            slice.get_mut(slot_off..slot_off.saturating_add(width))
        else {
            return Err(Error::OutOfRange("relocation slot"));
        };
        let place = site.vaddr.wrapping_add(site.out_off);
        apply_target(
            site.target,
            r_type,
            sym,
            site.addend,
            place,
            resolver,
            slot,
        )
    }

    /// The tombstone this site takes, or `None` when it resolves normally.
    ///
    /// A site is tombstoned when its member asked for one -- only a `.debug_*`
    /// member does -- and the symbol it names is a file-local one defined in a
    /// section placement dropped. A COMDAT group that lost, a section
    /// `--gc-sections` removed, and a kind this linker does not emit all land
    /// there; the addresses inside them were never assigned, so the site has
    /// nothing to point at.
    ///
    /// lld reaches the same sites from the other side: a symbol relative to a
    /// discarded section has already become undefined by the time it writes,
    /// and `relocateNonAlloc` tombstones an absolute reference to one
    /// (`lld/ELF/InputSection.cpp`).
    ///
    /// The addend is deliberately dropped. It is the offset of a place inside
    /// the section that is gone, and adding it to the tombstone would move the
    /// value back into the range of plausible addresses the tombstone exists
    /// to stay out of.
    fn tombstone(&self, r: &Rela64) -> Option<u64> {
        let value = self.item.tombstone?;
        let symtab = self.symtab.as_ref()?;
        let sym = symtab.syms.get(r.sym() as usize)?;
        if sym.bind() != STB_LOCAL {
            return None;
        }
        let shndx = sym.st_shndx.get();
        if shndx == SHN_UNDEF || shndx >= SHN_LORESERVE {
            return None;
        }
        // A section the merge pass folded into a pool was placed; it just
        // carries no stamped address of its own, because the writer supplies
        // the offset within the pool through the rewritten addend instead.
        // `.debug_str` and `.debug_line_str` are exactly that, and every
        // string reference in `.debug_info` and `.debug_line` names one.
        if self.ctx.merge.is_merged(self.item.file, shndx) {
            return None;
        }
        // An ICF-folded section keeps a placed address -- the winner's, which
        // `apply_folding` stamped into its slot -- so the absent-address test
        // below never catches it. Left alone, the loser's compilation unit
        // claims the winner's code and two units own one range, which is the
        // ambiguity the tombstone exists to prevent.
        if self.item.folded_tombstone
            && self.ctx.folding.is_folded((self.item.file, shndx))
        {
            return Some(value);
        }
        self.layout
            .section_vaddr(self.item.file, shndx)
            .is_none()
            .then_some(value)
    }

    /// Refuses a site in allocated code that names a symbol whose section was
    /// dropped.
    ///
    /// The other half of the tombstone question. A debug member points such a
    /// reference at a tombstone, because a debugger reads a value and can be
    /// told there is nothing there. Executable code has no such value: the
    /// site would resolve to `0 + st_value + addend`, a plausible-looking
    /// address in the first page. lld errors here too ("relocation refers to
    /// a discarded section").
    ///
    /// The check runs only for a file that actually lost a section, which is
    /// a bool the member already carries, so an ordinary link pays one
    /// predictable branch per relocation and no symbol-table read.
    fn check_not_discarded(&self, r: &Rela64) -> Result<()> {
        if !self.item.has_discarded {
            return Ok(());
        }
        let Some(symtab) = self.symtab.as_ref() else {
            return Ok(());
        };
        let Some(sym) = symtab.syms.get(r.sym() as usize) else {
            return Ok(());
        };
        if sym.bind() != STB_LOCAL {
            return Ok(());
        }
        let shndx = sym.st_shndx.get();
        if shndx == SHN_UNDEF
            || shndx >= SHN_LORESERVE
            || self.ctx.merge.is_merged(self.item.file, shndx)
            || self.layout.section_vaddr(self.item.file, shndx).is_some()
        {
            return Ok(());
        }
        Err(Error::UndefinedReference(
            String::from_utf8_lossy(symtab.name(sym)).into_owned(),
        ))
    }

    /// The error for a site the writer had to rewrite and could not.
    ///
    /// Every mandatory rewrite is a dynamic TLS lowering: the general-dynamic
    /// and local-dynamic sequences call a runtime `__tls_get_addr` that no
    /// image xold produces contains, so the sequence that reaches it has to go.
    /// Naming the thread-local is what makes the message actionable, since the
    /// offending bytes are the compiler's choice of TLS model for that one
    /// variable.
    fn unlowered(&self, r: &Rela64, reason: &'static str) -> Error {
        Error::unresolvable_tls(
            String::from_utf8_lossy(self.symbol_name(r)).into_owned(),
            reason,
        )
    }

    /// The name an entry's symbol carries in its own input file, or an empty
    /// slice when the file has no symbol table or the index is out of range.
    ///
    /// Read from the input rather than from the resolved symbol table because
    /// a relocation names a file-local index, which is what the resolver's
    /// per-file rows are keyed on too. Only the diagnostic path asks, so the
    /// re-parse it costs is never on the apply loop.
    fn symbol_name(&self, r: &Rela64) -> &[u8] {
        let Some(input) = self.ctx.files.get(self.item.file) else {
            return &[];
        };
        let Ok(Some(symtab)) = input.symbol_table() else {
            return &[];
        };
        symtab
            .syms
            .get(r.sym() as usize)
            .map_or(&[] as &[u8], |sym| symtab.name(sym))
    }
}

/// One lowered sequence's footprint in the member being patched.
///
/// A dynamic TLS sequence spans two instructions and carries a relocation on
/// each. Lowering it rewrites both, so the second relocation has nothing left
/// to patch and applying it would overwrite the replacement -- the local-exec
/// form ends with an offset field at exactly the displacement the dropped
/// `call` named. Which relocation that is comes from the sequence, never from
/// the call's own bytes.
#[derive(Clone)]
struct Consumed {
    /// The bytes the rewrite claims, as offsets within the member.
    range: Range<u64>,
    /// The slot of the relocation that claimed them. That one still applies:
    /// it is the one that performs the rewrite.
    slot: u64,
}

impl Consumed {
    /// Whether a relocation at `slot` is one this rewrite swallowed.
    fn swallows(&self, slot: u64) -> bool {
        slot != self.slot && self.range.contains(&slot)
    }
}

/// The resolved symbol an entry names, or `None` for the null symbol.
fn sym_of(r: &Rela64) -> Option<SymbolId> {
    match r.sym() {
        0 => None,
        idx => Some(SymbolId(usize::try_from(idx).unwrap_or(0))),
    }
}

/// Copies a split `.eh_frame` member: each surviving record is written at the
/// offset the plan assigned it, and the dropped records leave no bytes behind.
///
/// The records are packed, so anything left over in `slice` is the four-byte
/// zero terminator the plan reserved on the last member of the section. It is
/// written here rather than left to the zero-filled image, so the byte that
/// ends the section is one the copy actually put there.
fn copy_pieces(
    member: &EhMember,
    layout: &Layout,
    vaddr: u64,
    data: &[u8],
    slice: &mut [u8],
) -> Result<()> {
    let mut end = 0usize;
    for piece in member.pieces() {
        let Some(out_off) = piece.out_off else {
            continue;
        };
        let src_start = usize::try_from(piece.in_off).unwrap_or(usize::MAX);
        let len = usize::try_from(piece.len).unwrap_or(0);
        let dst_start = usize::try_from(out_off).unwrap_or(usize::MAX);
        let src = data
            .get(src_start..src_start.saturating_add(len))
            .ok_or(Error::OutOfRange("eh_frame record"))?;
        let dst = slice
            .get_mut(dst_start..dst_start.saturating_add(len))
            .ok_or(Error::OutOfRange("eh_frame record slot"))?;
        dst.copy_from_slice(src);
        end = end.max(dst_start.saturating_add(len));
        if piece.fde {
            rewrite_cie_pointer(member, layout, vaddr, *piece, src, dst)?;
        }
    }
    if let Some(tail) = slice.get_mut(end..) {
        tail.fill(0);
    }
    Ok(())
}

/// Rewrites one copied FDE's `CIE_pointer` (the record's second word) to the
/// distance from its new position back to its CIE's new position.
///
/// An FDE names its CIE by the backwards byte distance from that field, so
/// moving either record invalidates the stored value. lld's
/// `EhFrameSection::writeTo` patches the same field for the same reason.
///
/// The CIE is usually this member's own, and then both positions are member
/// offsets and their difference is the answer. When the splitter dropped it as
/// a duplicate the CIE it stands in for belongs to an earlier member, so both
/// positions are taken as addresses instead: within one output section a
/// distance between addresses is the distance between file offsets, and the
/// earlier member's placement is what makes the result positive.
fn rewrite_cie_pointer(
    member: &EhMember,
    layout: &Layout,
    vaddr: u64,
    piece: Piece,
    src: &[u8],
    dst: &mut [u8],
) -> Result<()> {
    let field: [u8; 4] = src
        .get(4..8)
        .and_then(|b| b.try_into().ok())
        .ok_or(Error::OutOfRange("FDE CIE pointer"))?;
    let old = u64::from(u32::from_le_bytes(field));
    // The stored value is `field position - CIE position` in the input.
    let cie_in = piece
        .in_off
        .wrapping_add(4)
        .checked_sub(old)
        .ok_or(Error::Format("FDE CIE pointer"))?;
    let cie_addr = if let Some(out_off) =
        member.piece_at(cie_in).and_then(|c| c.out_off)
    {
        vaddr.wrapping_add(out_off)
    } else {
        let at = member
            .cie_redirect(cie_in)
            .ok_or(Error::Format("FDE names a dropped CIE"))?;
        layout
            .section_vaddr(at.file, at.section)
            .ok_or(Error::Format("FDE names a dropped CIE"))?
            .wrapping_add(at.out_off)
    };
    let field_addr = vaddr
        .wrapping_add(piece.out_off.unwrap_or_default())
        .wrapping_add(4);
    let value = u32::try_from(field_addr.wrapping_sub(cie_addr))
        .map_err(|_| Error::OutOfRange("FDE CIE pointer"))?;
    let slot = dst
        .get_mut(4..8)
        .ok_or(Error::OutOfRange("FDE CIE pointer"))?;
    slot.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

/// One relocation ready to be applied: the entry, the addend to use (rewritten
/// when it points into a merged section), and where it lands.
#[derive(Copy, Clone)]
struct RelocApply<'a> {
    r: &'a Rela64,
    addend: i64,
    target: Target,
    relax: bool,
    /// Virtual address of the section being patched.
    vaddr: u64,
    /// Offset of the slot within that section.
    out_off: u64,
}

/// Writes `value` into the `width`-byte slot at `off`, truncating to the
/// field. The store is what an absolute relocation does with no symbol and no
/// addend, which is how lld writes a tombstone (`relocateNoSym`).
fn write_tombstone(
    off: u64,
    width: usize,
    value: u64,
    slice: &mut [u8],
) -> Result<()> {
    let at = usize::try_from(off).unwrap_or(usize::MAX);
    let Some(slot) = slice.get_mut(at..at.saturating_add(width)) else {
        return Err(Error::OutOfRange("relocation slot"));
    };
    let bytes = value.to_le_bytes();
    let Some(src) = bytes.get(..width) else {
        return Err(Error::OutOfRange("tombstone width"));
    };
    slot.copy_from_slice(src);
    Ok(())
}

/// What came of an attempted rewrite at one relocation site.
enum Rewrite {
    /// The site was rewritten, so its value computation is skipped.
    Done,
    /// Nothing was rewritten, and why. An optional rewrite falls back to the
    /// value computation; a mandatory one has no fallback, so the reason
    /// becomes the diagnostic.
    Declined(&'static str),
}

/// Why a rewrite the writer had to perform did not happen: the bytes around
/// the slot are not the sequence the lowering recognises, or the addend is not
/// the canonical one the encoding implies.
const NOT_LOWERABLE: &str = "the instruction sequence around the reference is not one this linker \
     knows how to rewrite into a form an executable can resolve";

/// The same, for a site whose sequence is not wholly inside the section that
/// carries it, so there is nothing to match against. A relocation cannot reach
/// into the section before or after its own.
const NO_WINDOW: &str = "the instruction sequence around the reference runs past the bounds of \
     the section holding it";

/// Runs the architecture's rewrite hook over the instructions around a site.
///
/// The window it is handed covers the opcode bytes that precede the value slot
/// ([`relax_lead_target`]), the slot itself, and whatever the rewrite reaches
/// past it ([`relax_trail_target`]), so the hook sees the whole sequence rather
/// than the field alone. `width` is the slot's width from the relocation's
/// spec.
fn try_rewrite(
    site: RelocApply<'_>,
    width: usize,
    resolver: &FileResolver<'_>,
    slice: &mut [u8],
) -> Result<Rewrite> {
    let r_type = site.r.r_type();
    let lead = relax_lead_target(site.target, r_type);
    let slot_off = usize::try_from(site.out_off).unwrap_or(usize::MAX);
    // A type no rewrite claims is not matched at all, and a site too close to
    // the start of the section cannot carry the opcode bytes a rewrite needs;
    // neither is matched against a short window.
    if !relax_covers_target(site.target, r_type) || slot_off < lead {
        return Ok(Rewrite::Declined(NO_WINDOW));
    }
    let start = slot_off - lead;
    let slot_end = slot_off.saturating_add(width);
    // The trail is read off the bytes that follow the slot for the one
    // sequence whose length depends on the instruction closing it; every other
    // type ignores them.
    let after = slice.get(slot_end..).unwrap_or(&[]);
    let slot = slice.get(slot_off..slot_end).unwrap_or(&[]);
    let end = slot_end.saturating_add(relax_trail_target(
        site.target,
        r_type,
        slot,
        after,
    ));
    let Some(window) = slice.get_mut(start..end) else {
        return Ok(Rewrite::Declined(NO_WINDOW));
    };
    let place = site.vaddr.wrapping_add(site.out_off);
    let done = relax_target(
        site.target,
        r_type,
        sym_of(site.r),
        site.addend,
        place,
        resolver,
        window,
    )?;
    Ok(if done {
        Rewrite::Done
    } else {
        Rewrite::Declined(NOT_LOWERABLE)
    })
}
