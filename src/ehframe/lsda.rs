//! The code sections an `.eh_frame` FDE describes with a language-specific
//! data area.
//!
//! Identical code folding compares a section's bytes and its relocations, and
//! two C++ functions can agree on both and still need different unwind
//! behaviour. The catch clauses of a `try` block are not encoded in the
//! function's instructions: the call-site table, the action table and the
//! type-info references all live in `.gcc_except_table`, which the FDE names
//! through its LSDA pointer. Two handlers that catch different types compile to
//! the same code and differ only there.
//!
//! Folding such a pair gives both functions the representative's LSDA, so the
//! other one's handler never matches and the exception escapes past a `catch`
//! that was written for it. The bytes the comparison reads are identical, so no
//! amount of care in the comparison itself can see the difference; the sections
//! have to be kept out of the partition altogether.
//!
//! lld does exactly that, before it collects a single candidate
//! (`ICF<ELFT>::run`, `lld/ELF/ICF.cpp`).
//!
//! The question is put to the CIE, not to the FDE, because an LSDA pointer is
//! present exactly when the CIE's augmentation string carries an `L`. A CIE
//! without one describes functions that cannot catch anything, and the
//! sections its FDEs describe stay eligible -- which is why an ordinary C
//! program still folds.

use rustc_hash::FxHashSet;

use crate::{
    error::Result, gc::reloc_target, linker::Context, output::OutKind,
};

/// Every input section an `.eh_frame` FDE describes whose CIE declares an
/// LSDA, and which identical code folding must therefore leave alone.
///
/// Walks the `.eh_frame` members of the output section, in member order, so
/// the result is a function of the inputs. Membership is all the caller reads,
/// so the set contributes no ordering of its own.
pub fn sections_with_lsda(
    ctx: &Context<'_>,
) -> Result<FxHashSet<(usize, u16)>> {
    let mut out = FxHashSet::default();
    let Some(eh) = ctx.outputs.section(OutKind::EhFrame) else {
        return Ok(out);
    };
    let mut pieces = Vec::new();
    let mut fde = Vec::new();
    for m in &eh.members {
        scan_member(ctx, m.file, m.section, &mut pieces, &mut fde, &mut out)?;
    }
    Ok(out)
}

/// Adds the sections one `.eh_frame` member's LSDA-bearing FDEs describe.
///
/// The record buffers are caller-owned so a link with hundreds of `.eh_frame`
/// sections allocates them once.
fn scan_member(
    ctx: &Context<'_>,
    file: usize,
    section: u16,
    pieces: &mut Vec<super::Piece>,
    fde: &mut Vec<bool>,
    out: &mut FxHashSet<(usize, u16)>,
) -> Result<()> {
    let Some(input) = ctx.files.get(file) else {
        return Ok(());
    };
    let obj = input.object()?;
    let Some(shdr) = obj.sections().get(usize::from(section)) else {
        return Ok(());
    };
    let data = obj.section_data(shdr)?;
    super::records(data, pieces, fde);
    // A record naming no symbol describes no section, and a member with no
    // relocations at all has no FDE that could name one. The malformed
    // `.eh_frame` shapes this walk cannot decode are reported by the record
    // splitter, which reads the same bytes in the same link.
    let Some(symtab) = input.symbol_table()? else {
        return Ok(());
    };
    let Some(entries) = input.relocations(section)? else {
        return Ok(());
    };
    let with_lsda = cies_with_lsda(data, pieces)?;
    if with_lsda.is_empty() {
        return Ok(());
    }
    let first = super::first_reloc_syms(pieces, entries);
    for (i, piece) in pieces.iter().enumerate() {
        let Some(cie) = super::cie_offset_of(data, *piece) else {
            continue;
        };
        if !with_lsda.contains(&cie) {
            continue;
        }
        // The FDE's first relocation names the function it describes, the same
        // field the record splitter reads for liveness.
        let Some(Some(sym_idx)) = first.get(i).copied() else {
            continue;
        };
        if let Some(target) = reloc_target(ctx, &symtab, file, sym_idx) {
            out.insert(target);
        }
    }
    Ok(())
}

/// The offsets of the CIEs in one member whose augmentation string carries an
/// `L`, so their FDEs hold an LSDA pointer.
fn cies_with_lsda(
    data: &[u8],
    pieces: &[super::Piece],
) -> Result<FxHashSet<u64>> {
    let mut out = FxHashSet::default();
    for piece in pieces {
        if piece.fde {
            continue;
        }
        let Ok(start) = usize::try_from(piece.in_off) else {
            continue;
        };
        let end = start
            .saturating_add(usize::try_from(piece.len).unwrap_or(usize::MAX));
        if super::cie_augmentation(data, start, end.min(data.len()))?.has_lsda {
            out.insert(piece.in_off);
        }
    }
    Ok(out)
}
