//! DWARF `.debug_*` section emission: plans file offsets for each aggregated
//! debug output section, then copies each member's bytes and applies its
//! `.rela.debug_*` relocations.

use rayon::prelude::*;

use super::sections::{CopyItem, copy_member_into, split_member_slices};
use crate::{
    error::Result,
    layout::{Layout, Region},
    linker::Context,
    reloc::Target,
    util::align_up,
};

/// File offsets of the laid-out `.debug_*` output sections, parallel to
/// `ctx.debug.sections`. Non-allocated: each region's `vaddr` is zero (debug
/// sections carry no load address); only `offset` and `size` are meaningful.
pub(super) type DebugPlan = Vec<Region>;

/// Computes file offsets for each aggregated `.debug_*` output section,
/// starting at `start`. Returns one region per `ctx.debug.sections`, in
/// order. Non-allocated: regions carry file offsets only (`vaddr` zero,
/// since debug sections are not loaded).
pub(super) fn plan_debug(ctx: &Context<'_>, start: u64) -> DebugPlan {
    let mut regions = Vec::with_capacity(ctx.debug.sections.len());
    let mut cursor = start;
    for sec in &ctx.debug.sections {
        let align = sec.align.max(1);
        cursor = align_up(cursor, align);
        regions.push(Region {
            offset: cursor,
            vaddr: 0,
            size: sec.size,
        });
        cursor = cursor.saturating_add(sec.size);
    }
    regions
}

/// One past the last byte of the planned debug sections, or `start` when the
/// link carries none.
pub(super) fn debug_end(plan: &DebugPlan, start: u64) -> u64 {
    plan.iter().map(|r| r.end()).fold(start, u64::max)
}

/// Copies each debug member's bytes into the image and applies its
/// `.rela.debug_*` relocations in place. Parallelised per member exactly like
/// the allocated-section copy: every member owns a disjoint image range, so
/// the applications never alias. Relaxation is disabled (debug sections hold
/// no relaxable sites).
pub(super) fn write_debug_bytes(
    ctx: &Context<'_>,
    target: Target,
    layout: &Layout,
    plan: &DebugPlan,
    image: &mut [u8],
) -> Result<()> {
    if ctx.debug.is_empty() {
        return Ok(());
    }
    let mut items = Vec::new();
    for (i, sec) in ctx.debug.sections.iter().enumerate() {
        let Some(region) = plan.get(i) else {
            continue;
        };
        let tombstone = Some(tombstone_for(&sec.name));
        // `.debug_line` deliberately keeps pointing at the surviving copy of
        // a folded function, so a debugger can still put a breakpoint on it.
        let folded_tombstone = sec.name.as_slice() != b".debug_line";
        for m in &sec.members {
            let len = usize::try_from(m.size).unwrap_or(0);
            if len == 0 {
                continue;
            }
            let start =
                usize::try_from(region.offset.wrapping_add(m.out_offset))
                    .unwrap_or(usize::MAX);
            items.push(CopyItem {
                // A debug member tombstones instead of erroring.
                has_discarded: false,
                file: m.file,
                section: m.section,
                start,
                len,
                // Relocation place base: the member's offset within its
                // output debug section. DWARF relocations are absolute (the
                // common case) and do not read the place; this fixes any
                // PC-relative site to a deterministic value.
                vaddr: m.out_offset,
                tombstone,
                folded_tombstone,
            });
        }
    }
    items.sort_by_key(|i| i.start);
    let mut slices = split_member_slices(image, &items)?;
    slices
        .par_iter_mut()
        .enumerate()
        .map(|(i, slice)| {
            copy_member_into(ctx, target, layout, false, items[i], slice)
        })
        .collect::<Result<Vec<()>>>()?;
    Ok(())
}

/// The value a `.debug_*` section carries in place of an address whose section
/// placement dropped.
///
/// Resolving such a site to its addend is what makes the defect: the result is
/// a small, valid-looking address, so a range list claims code at the bottom of
/// the image and two compilation units claim the same code. A tombstone is a
/// value no real range can be confused with.
///
/// The three answers are lld's, from `relocateNonAlloc`
/// (`lld/ELF/InputSection.cpp`). Most sections take zero. The
/// pre-DWARF-5 `.debug_loc` and `.debug_ranges` take one, because in those
/// formats an entry of all-ones is a base-address selection entry and a pair
/// of zeroes ends the list -- one is the smallest value that is neither.
/// `.debug_names` takes all-ones, which its own format reserves.
fn tombstone_for(name: &[u8]) -> u64 {
    match name {
        b".debug_loc" | b".debug_ranges" => 1,
        b".debug_names" => u64::MAX,
        _ => 0,
    }
}
