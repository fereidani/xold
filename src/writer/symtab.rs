//! Output symbol table (`.symtab`) and string table (`.strtab`/`.shstrtab`)
//! construction.
//!
//! The table is a null entry, then every local the inputs contribute (see
//! [`locals`]), then the defined globals layout collected. ELF requires that
//! order: locals first, with `sh_info` naming the index of the first global.
//!
//! # One difference from lld, on purpose
//!
//! lld's local block is wider than this one. Its `computeBinding` gives any
//! hidden or internal definition `STB_LOCAL` binding in the output, so a
//! symbol that is global in the inputs still lands among the locals -- that is
//! where its `_init`, `_fini`, `_GLOBAL_OFFSET_TABLE_`, `_DYNAMIC` and
//! `_dl_relocate_static_pie` rows come from. xold emits those with the binding
//! the inputs gave them, in the global block.
//!
//! Nothing symbolizes worse for it: the rows are present, at the right
//! addresses, and `sh_info` describes the table xold actually writes. Adopting
//! lld's rule would mean re-binding and re-ordering the *export* block, not
//! adding to the locals block below, which is why it is recorded here rather
//! than done in passing.

mod locals;

use bytemuck::Zeroable;
use rayon::prelude::*;

use crate::{
    elf::{
        Sym64,
        constants::{STB_GLOBAL, STB_GNU_UNIQUE, STB_WEAK},
    },
    endian::{U16, U32, U64},
    error::Result,
    layout::Layout,
    linker::Context,
};

/// A growing string table with a leading NUL byte at offset zero.
pub(super) struct StrTab {
    pub(super) bytes: Vec<u8>,
}

impl StrTab {
    pub(super) fn new() -> Self {
        Self { bytes: vec![0] }
    }

    pub(super) fn intern(&mut self, s: &[u8]) -> u32 {
        let off = u32::try_from(self.bytes.len()).unwrap_or(u32::MAX);
        self.bytes.extend_from_slice(s);
        self.bytes.push(0);
        off
    }
}

/// The built symbol table and the one fact its section header needs beyond
/// its size.
///
/// The default is the table `--strip-all` leaves behind: no entries at all,
/// and so no section header either.
#[derive(Default)]
pub(super) struct SymtabPlan {
    pub(super) syms: Vec<Sym64>,
    /// The index of the first non-local entry, which `.symtab`'s `sh_info`
    /// carries. Every reader splits the table on this value, so an off-by-one
    /// makes the table wrong rather than merely untidy.
    pub(super) first_global: u32,
}

/// Builds the output symbol table -- a null entry, the inputs' locals, then
/// the exports -- interning each name into `strtab`.
///
/// Every entry is independent except for its `st_name`, which is a running
/// offset into the string table. Taking those offsets with a prefix sum first
/// frees the entries themselves to be built in parallel; the names are then
/// appended in one pass. The result is byte-for-byte what interning one name
/// at a time produced.
///
/// The locals are gathered per input file in parallel too, so the pass stays
/// off the serial path as the table grows (see [`locals::collect`]).
pub(super) fn build_symtab(
    ctx: &Context<'_>,
    layout: &Layout,
    strtab: &mut StrTab,
) -> Result<SymtabPlan> {
    let locals = locals::collect(ctx, layout)?;
    let exports = &layout.exports;
    let total = locals.len().saturating_add(exports.len());
    let mut offsets = Vec::with_capacity(total);
    let mut cursor = u32::try_from(strtab.bytes.len()).unwrap_or(u32::MAX);
    let lengths = locals
        .iter()
        .map(|l| l.name.len())
        .chain(exports.iter().map(|e| ctx.symbols.name(e.id).len()));
    for len in lengths {
        offsets.push(cursor);
        let len = u32::try_from(len).unwrap_or(0);
        cursor = cursor.saturating_add(len).saturating_add(1);
    }
    let (local_offs, export_offs) = offsets.split_at(locals.len());

    let mut syms = Vec::with_capacity(total.saturating_add(1));
    syms.push(Sym64::zeroed());
    syms.par_extend(locals.par_iter().zip(local_offs).map(|(l, &st_name)| {
        Sym64 {
            st_name: U32::new(st_name),
            st_info: l.info,
            st_other: l.other,
            st_shndx: U16::new(l.shndx),
            st_value: U64::new(l.value),
            st_size: U64::new(l.size),
        }
    }));
    syms.par_extend(exports.par_iter().zip(export_offs).map(
        |(e, &st_name)| {
            let bind = if e.weak {
                STB_WEAK
            } else if e.unique {
                STB_GNU_UNIQUE
            } else {
                STB_GLOBAL
            };
            Sym64 {
                st_name: U32::new(st_name),
                st_info: (bind << 4) | e.sym_type,
                st_other: e.visibility,
                st_shndx: U16::new(e.shndx),
                st_value: U64::new(e.addr),
                st_size: U64::new(e.size),
            }
        },
    ));

    // One parallel pass appends every name: the batches tile the region the
    // sizes above measured, and the zeroes the resize wrote are the NUL
    // terminators. The bytes are exactly the serial append's.
    let names: Vec<&[u8]> = locals
        .iter()
        .map(|l| l.name)
        .chain(exports.iter().map(|e| ctx.symbols.name(e.id)))
        .collect();
    let base = strtab.bytes.len();
    let total: usize = names.iter().map(|n| n.len().saturating_add(1)).sum();
    strtab.bytes.resize(base.saturating_add(total), 0);
    write_names(strtab.bytes.get_mut(base..).unwrap_or(&mut []), &names);
    Ok(SymtabPlan {
        // The null entry is a local, so the first global sits one past the
        // block of locals that follows it.
        first_global: u32::try_from(locals.len().saturating_add(1))
            .unwrap_or(u32::MAX),
        syms,
    })
}

/// Copies NUL-terminated `names` into `region`, which tiles them exactly and
/// arrives zeroed (so the terminators are already in place).
///
/// The region is cut into one contiguous chunk per batch of names and the
/// chunks fill concurrently; a string table is often tens of megabytes and
/// this sits on the writer's critical path.
fn write_names(region: &mut [u8], names: &[&[u8]]) {
    /// Enough batches to keep every worker fed past uneven name lengths.
    const BATCHES: usize = 64;
    let per = names.len().div_ceil(BATCHES).max(1);
    let mut jobs: Vec<(&mut [u8], &[&[u8]])> =
        Vec::with_capacity(BATCHES.saturating_add(1));
    let mut rest = region;
    let mut idx = 0;
    while idx < names.len() {
        let end = idx.saturating_add(per).min(names.len());
        let batch = names.get(idx..end).unwrap_or_default();
        let bytes: usize =
            batch.iter().map(|n| n.len().saturating_add(1)).sum();
        let (chunk, tail) = rest.split_at_mut(bytes.min(rest.len()));
        rest = tail;
        jobs.push((chunk, batch));
        idx = end;
    }
    jobs.into_par_iter().for_each(|(chunk, batch)| {
        let mut at = 0usize;
        for name in batch {
            if let Some(dst) = chunk.get_mut(at..at.saturating_add(name.len()))
            {
                dst.copy_from_slice(name);
            }
            at = at.saturating_add(name.len()).saturating_add(1);
        }
    });
}
