//! GNU symbol versioning emission: the `.gnu.version` (versym) and
//! `.gnu.version_r` (VERNEED) sections, plus the matching `DT_VERSYM`,
//! `DT_VERNEED` and `DT_VERNEEDNUM` tags.
//!
//! The model is the first-cut design shared by gold and lld before a library
//! versions its own exports: imported symbols carry the version their
//! definition advertises in the dependency's `.gnu.version_d`, and the link's
//! own exports are reported as the default (`VER_NDX_GLOBAL`). A `.gnu.version`
//! table -- one `u16` per `.dynsym` entry, in the same (possibly GNU-hash
//! reordered) order -- records the version index of each symbol; the VERNEED
//! table lists one `Elf_Verneed` per dependency soname with a `Vernaux` for
//! every distinct version referenced from it.
//!
//! Inputs come from two sources, both already collected on the linker
//! [`Context`](crate::linker::Context):
//!
//! - Dependency VERDEF: `dep_versions` maps each imported symbol name to the
//!   `(soname, version_name, hash)` its definition carries. This is the primary
//!   source on toolchains (clang on glibc) whose `.o` files do not carry
//!   `.gnu.version_r`.
//! - Input-object `.gnu.version_r`: a `VersionTable` view over each input that
//!   emits one, propagated directly. Rare in practice but supported for
//!   correctness.
//!
//! The versym index space is local to the output object: 0 = local, 1 =
//! global (an unversioned export), and each `(soname, version)` referenced
//! takes the next index >= 2 in the order the VERNEED table lists them.

use bytemuck::Pod;
use rustc_hash::{FxHashMap, FxHashSet};

use super::dynsym::{DynSym, copy_rows, intern};
use crate::{
    elf::constants::{SHN_UNDEF, VER_NDX_GLOBAL, VER_NDX_LOCAL},
    endian::{U16, U32},
    layout::Layout,
    linker::Context,
    util::cstr_at,
};

/// One versioned dependency to emit: the soname and the distinct versions
/// referenced from it, with the name's ELF hash (the loader matches it
/// against the dependency's VERDEF) and the output-local versym index
/// assigned to that version.
#[derive(Clone)]
pub(super) struct VerneedEntry {
    pub soname: Vec<u8>,
    pub versions: Vec<VerneedVersion>,
}

/// One `Elf_Vernaux` row, plus the output-local index that symbols
/// referencing this version carry in `.gnu.version`.
#[derive(Clone)]
pub(super) struct VerneedVersion {
    pub name: Vec<u8>,
    pub hash: u32,
    pub out_index: u16,
}

/// The collected version requirements plus the per-dynsym-entry versym index,
/// ready to serialise. Built once and shared between the sizing pass (which
/// only needs the counts) and the byte emitter.
#[derive(Clone, Default)]
pub(super) struct VersionPlan {
    /// One `Elf_Verneed` per dependency soname, in a stable order.
    pub needs: Vec<VerneedEntry>,
    /// dynsym name -> output versym index, for the dynsym emitter to read.
    pub versym_by_name: FxHashMap<Vec<u8>, u16>,
}

impl VersionPlan {
    /// Whether any version information is to be emitted. When false the
    /// dynamic table omits the `DT_VERSYM`/`DT_VERNEED`/`DT_VERNEEDNUM` tags
    /// and the synthetic sections are not allocated.
    pub fn has_versions(&self) -> bool {
        !self.needs.is_empty()
    }

    /// The versym byte size: one `u16` per `.dynsym` entry (including the
    /// leading null entry), so always `2 * dynsym_count`.
    pub fn versym_size(dynsym_count: u64) -> u64 {
        dynsym_count.saturating_mul(2)
    }

    /// The VERNEED byte size: one 16-byte `Elf_Verneed` per soname plus one
    /// 16-byte `Elf_Vernaux` per version referenced.
    pub fn verneed_size(&self) -> u64 {
        let records = u64::try_from(self.needs.len()).unwrap_or(0);
        let auxes = self
            .needs
            .iter()
            .map(|n| u64::try_from(n.versions.len()).unwrap_or(0))
            .sum::<u64>();
        (records.saturating_add(auxes)).saturating_mul(16)
    }

    /// The number of `Elf_Verneed` records (`DT_VERNEEDNUM`).
    pub fn verneed_count(&self) -> u32 {
        u32::try_from(self.needs.len()).unwrap_or(0)
    }

    /// The byte size the version-name and soname strings add to `.dynstr`:
    /// sum of `len + 1` (NUL) over every distinct soname and version name.
    pub fn dynstr_size(&self) -> u64 {
        let mut total = 0u64;
        for n in &self.needs {
            total = total
                .saturating_add(u64::try_from(n.soname.len()).unwrap_or(0))
                .saturating_add(1);
            for v in &n.versions {
                total = total
                    .saturating_add(u64::try_from(v.name.len()).unwrap_or(0))
                    .saturating_add(1);
            }
        }
        total
    }
}

/// The deduplicated set of `(soname, version_name, hash)` triples referenced
/// by any name in `import_names`, in stable order (soname order of first
/// appearance, version order within each soname). Shared by the sizing and
/// byte-emission passes so they agree.
fn collect_requirements(
    ctx: &Context<'_>,
    import_names: &[Vec<u8>],
) -> Vec<(Vec<u8>, Vec<u8>, u32)> {
    let mut needs: Vec<(Vec<u8>, Vec<u8>, u32)> = Vec::new();
    for name in import_names {
        if name.is_empty() {
            continue;
        }
        let Some(ver) = ctx.dep_versions.get(name) else {
            continue;
        };
        if needs
            .iter()
            .any(|(s, n, _)| *s == ver.soname && *n == ver.name)
        {
            continue;
        }
        needs.push((ver.soname.clone(), ver.name.clone(), ver.hash));
    }
    needs
}

/// Groups the requirement triples into one [`VerneedEntry`] per soname,
/// numbering the versions from `VER_NDX_GLOBAL + 1` in encounter order. Also
/// returns the soname-to-position map the caller needs to look an entry up.
/// Shared by [`build`] and [`size_for`] so the two agree by construction.
fn build_needs(
    reqs: Vec<(Vec<u8>, Vec<u8>, u32)>,
) -> (Vec<VerneedEntry>, FxHashMap<Vec<u8>, usize>) {
    let mut needs: Vec<VerneedEntry> = Vec::new();
    let mut soname_pos: FxHashMap<Vec<u8>, usize> = FxHashMap::default();
    let mut next_index: u16 = VER_NDX_GLOBAL.checked_add(1).unwrap_or(2);
    for (soname, vname, vhash) in reqs {
        let pos = *soname_pos.entry(soname.clone()).or_insert_with(|| {
            let p = needs.len();
            needs.push(VerneedEntry {
                soname,
                versions: Vec::new(),
            });
            p
        });
        let Some(entry) = needs.get_mut(pos) else {
            continue;
        };
        let idx = next_index;
        next_index = next_index.saturating_add(1);
        entry.versions.push(VerneedVersion {
            name: vname,
            hash: vhash,
            out_index: idx,
        });
    }
    (needs, soname_pos)
}

/// Collects into `out` the names of the `.dynsym` rows whose definition lives
/// in a dependency, in the order `table` will be written.
///
/// Two kinds qualify. The undefined imports are the obvious one: the loader
/// resolves them out of a dependency, so each must record the version its
/// definition carries. The copy rows are the subtle one. A copy relocation
/// makes the row *defined* -- it points at the executable's own `.bss` slot --
/// yet the object it describes is still the dependency's, and picking the
/// wrong version of it picks the wrong size and the wrong bytes. Filtering on
/// `SHN_UNDEF` alone therefore offers the version plan every import except the
/// ones a copy relocation just took over, which is how a copy-relocated import
/// ends up with no version requirement at all.
///
/// lld keeps the two together for the same reason: its copy relocation
/// overwrites the shared symbol with a defined one but keeps the file it came
/// from and the version id it carried, and its writer asks for a VERNEED entry
/// for any `.dynsym` row a needed shared file supplied, defined or not.
///
/// This link's own exports are not offered: their definitions are the image's,
/// so they carry `VER_NDX_GLOBAL` even when a dependency happens to version a
/// name of its own that they interpose.
pub(super) fn dep_row_names(
    table: &DynSym,
    layout: &Layout,
    out: &mut Vec<Vec<u8>>,
) {
    // Identity only: which names came from a copy slot. The order is the
    // table's, walked below.
    let copies: FxHashSet<&[u8]> =
        copy_rows(layout).map(|(_, n)| n.name.as_slice()).collect();
    out.clear();
    for e in &table.entries {
        let name = cstr_at(&table.strtab, e.name_off);
        if e.shndx == SHN_UNDEF || copies.contains(name) {
            out.push(name.to_vec());
        }
    }
}

/// Builds the version plan for `ctx`. Names without a known version (an
/// unversioned fallback) are left as `VER_NDX_GLOBAL`; so are this link's own
/// exports, which are never offered here (xold emits no `.gnu.version_d`).
///
/// `import_names` is the set of dynsym names a dependency supplies, from
/// [`dep_row_names`], in their post-reorder order so the resulting versym
/// index map aligns with the emitted `.dynsym`.
pub(super) fn build(
    ctx: &Context<'_>,
    import_names: &[Vec<u8>],
) -> VersionPlan {
    let (needs, soname_pos) =
        build_needs(collect_requirements(ctx, import_names));
    let mut versym_by_name: FxHashMap<Vec<u8>, u16> = FxHashMap::default();
    // Assign per-import versym indices by looking each import up in the
    // (soname, version) table just built. Two imports of the same version
    // share an index; an unversioned import (no dep entry) falls back to
    // `VER_NDX_GLOBAL` and is omitted from the map.
    for name in import_names {
        if name.is_empty() {
            continue;
        }
        let Some(ver) = ctx.dep_versions.get(name) else {
            continue;
        };
        let Some(&pos) = soname_pos.get(&ver.soname) else {
            continue;
        };
        let Some(entry) = needs.get(pos) else {
            continue;
        };
        let Some(v) = entry.versions.iter().find(|v| v.name == ver.name) else {
            continue;
        };
        versym_by_name.insert(name.clone(), v.out_index);
    }
    VersionPlan {
        needs,
        versym_by_name,
    }
}

/// A sizing-only summary of the version plan, computed without the dynsym
/// table so the probe pass can reserve space before any bytes are built.
/// Mirrors [`build`] over the same import set; the two must agree.
pub(super) fn size_for(
    ctx: &Context<'_>,
    import_names: &[Vec<u8>],
) -> VersionPlan {
    let (needs, _soname_pos) =
        build_needs(collect_requirements(ctx, import_names));
    // The versym_by_name map is not needed for sizing; leave it empty so the
    // probe allocates no per-name storage.
    VersionPlan {
        needs,
        versym_by_name: FxHashMap::default(),
    }
}

/// Builds the `.gnu.version` byte blob: one little-endian `u16` per
/// `.dynsym` entry. The null entry at index 0 is `VER_NDX_LOCAL`; a row the
/// plan assigned a version keeps that index; everything else is
/// `VER_NDX_GLOBAL`, which covers this link's own exports and any dependency
/// row whose definition carried no version.
pub(super) fn build_versym(table: &DynSym, plan: &VersionPlan) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(
        table.entries.len().saturating_add(1).saturating_mul(2),
    );
    push_u16(&mut bytes, VER_NDX_LOCAL);
    for e in &table.entries {
        let name = cstr_at(&table.strtab, e.name_off);
        let v = plan
            .versym_by_name
            .get(name)
            .copied()
            .unwrap_or(VER_NDX_GLOBAL);
        push_u16(&mut bytes, v);
    }
    bytes
}

/// Builds the `.gnu.version_r` byte blob: one `Elf_Verneed` per soname, each
/// followed by its `Elf_Vernaux` rows. Records are linked by relative byte
/// offsets (`vn_next`, `vn_aux`, `vna_next`); the last of each chain
/// terminates with a zero offset. The soname and version-name strings are
/// interned into `dynstr` and referenced by offset.
pub(super) fn build_verneed(
    plan: &VersionPlan,
    dynstr: &mut Vec<u8>,
) -> Vec<u8> {
    use crate::elf::{Vernaux64, Verneed64};
    let n = plan.needs.len();
    if n == 0 {
        return Vec::new();
    }
    let mut soname_off = Vec::with_capacity(n);
    for need in &plan.needs {
        soname_off.push(intern(dynstr, &need.soname));
    }
    let mut aux_off: Vec<Vec<(u32, u32)>> = Vec::with_capacity(n);
    for need in &plan.needs {
        let mut row = Vec::with_capacity(need.versions.len());
        for v in &need.versions {
            row.push((v.hash, intern(dynstr, &v.name)));
        }
        aux_off.push(row);
    }
    // Compute the absolute byte offset of each Verneed so neighbours can be
    // linked by relative offset. Each Verneed is 16 bytes followed by its
    // Vernaux chain; the next Verneed follows the last Vernaux.
    let mut vernext_off = Vec::with_capacity(n);
    let mut cursor = 0usize;
    for need in &plan.needs {
        vernext_off.push(cursor);
        cursor += core::mem::size_of::<Verneed64>();
        cursor += need.versions.len() * core::mem::size_of::<Vernaux64>();
    }
    let mut bytes =
        Vec::with_capacity(usize::try_from(plan.verneed_size()).unwrap_or(0));
    for (i, need) in plan.needs.iter().enumerate() {
        let base = vernext_off[i];
        let next_verneed_off = if i + 1 < n {
            vernext_off[i + 1].saturating_sub(base)
        } else {
            0
        };
        let rec = Verneed64 {
            vn_version: U16::new(1),
            vn_cnt: U16::new(
                u16::try_from(need.versions.len()).unwrap_or(u16::MAX),
            ),
            vn_file: U32::new(soname_off[i]),
            vn_aux: U32::new(
                u32::try_from(core::mem::size_of::<Verneed64>()).unwrap_or(0),
            ),
            vn_next: U32::new(u32::try_from(next_verneed_off).unwrap_or(0)),
        };
        push_pod(&mut bytes, &rec);
        for (j, v) in need.versions.iter().enumerate() {
            let (hash, name_off) = aux_off[i][j];
            let next_vernaux_off = if j + 1 < need.versions.len() {
                u32::try_from(core::mem::size_of::<Vernaux64>()).unwrap_or(0)
            } else {
                0
            };
            let aux = Vernaux64 {
                vna_hash: U32::new(hash),
                vna_flags: U16::new(0),
                vna_other: U16::new(v.out_index),
                vna_name: U32::new(name_off),
                vna_next: U32::new(next_vernaux_off),
            };
            push_pod(&mut bytes, &aux);
        }
    }
    bytes
}

/// Appends one little-endian `u16` to `out`.
fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Appends one `Pod` struct serialised as little-endian bytes.
fn push_pod<T: Pod>(out: &mut Vec<u8>, value: &T) {
    out.extend_from_slice(bytemuck::bytes_of(value));
}
