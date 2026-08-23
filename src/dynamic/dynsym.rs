//! The dynamic symbol table under construction: `.dynsym`, `.dynstr` and the
//! `SysV` `.hash` plus the GNU `.gnu.hash`, split out of the main dynamic
//! module.
//!
//! [`DynSym`] collects export/import/copy rows in order, interns their names
//! into a parallel `.dynstr`, and materialises the on-disk `Sym64` slice. The
//! free functions populate it from the resolved symbol table (`add_imports`),
//! build the final `.dynstr`, `.hash` and `.gnu.hash` bytes, and recover a
//! `SymbolId -> name` lookup used by the relocation emitters.

use bytemuck::Zeroable;
use rustc_hash::FxHashMap;

use crate::{
    elf::{
        Sym64,
        constants::{
            SHN_UNDEF, STB_GLOBAL, STB_GNU_UNIQUE, STB_WEAK, STT_NOTYPE,
            STT_OBJECT, STT_TLS, STV_DEFAULT,
        },
    },
    endian::{U16, U32, U64},
    layout::{CopyName, CopySlot, Export, GotKind, Layout, Sect},
    linker::Context,
    symbol::{SymbolId, SymbolKind},
    util::{cstr_at, push_u32},
};

/// One row of the dynamic symbol table under construction.
#[derive(Clone, Copy)]
pub(super) struct DynSymEntry {
    pub name_off: u32,
    pub info: u8,
    /// `st_other`: the symbol's visibility (`STV_*`).
    pub other: u8,
    pub shndx: u16,
    pub value: u64,
    pub size: u64,
}

impl DynSymEntry {
    /// Whether the loader can find this row through `.gnu.hash`.
    ///
    /// The GNU hash indexes only the tail of `.dynsym` starting at
    /// `symoffset`, so a row placed before it is unfindable by name. An
    /// ordinary import belongs there -- the loader resolves it elsewhere and
    /// never looks it up in this image. A canonical PLT row does not: its
    /// `st_shndx` is `SHN_UNDEF` because the definition lives in a
    /// dependency, but `st_value` names this image's stub, and that stub is
    /// the one address every image in the process must agree on for
    /// `&shared_func` to compare equal across them. lld reaches the same
    /// answer from the other side, treating such a symbol as defined at the
    /// PLT while choosing the hash set and forcing only the emitted
    /// `st_shndx` back to `SHN_UNDEF`.
    const fn is_hashable(&self) -> bool {
        self.shndx != SHN_UNDEF || self.value != 0
    }
}

/// What an undefined `.dynsym` row carries besides its name and binding.
///
/// An ordinary import states a type and nothing else: the loader supplies the
/// address and the extent. A canonical PLT import states the address too --
/// see [`import_row`] for why the executable answers for a function a
/// dependency defines.
#[derive(Clone, Copy, Default)]
pub(super) struct ImportRow {
    /// `st_info`'s type half (`STT_*`).
    sym_type: u8,
    /// `st_value`: zero for an import the loader resolves outright.
    value: u64,
    /// `st_size`: zero unless the row states an extent of its own.
    size: u64,
}

/// The in-progress `.dynsym` table: entries in order plus the name-to-index
/// map that `GLOB_DAT` emission needs, and the string table under
/// construction.
pub(super) struct DynSym {
    pub entries: Vec<DynSymEntry>,
    /// Name bytes -> 1-based dynsym index (the null entry is index 0).
    pub by_name: FxHashMap<Vec<u8>, u32>,
    /// `.dynstr` bytes for every interned name.
    pub strtab: Vec<u8>,
    /// Offset of the SONAME within `.dynstr`, once interned.
    pub soname_off: Option<u32>,
}

impl DynSym {
    /// A fresh table with a leading NUL in the string table.
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            by_name: FxHashMap::default(),
            strtab: vec![0],
            soname_off: None,
        }
    }

    /// Spine of the three adders: interns `name`, takes the next 1-based
    /// index, pushes one entry built from the given fields, and indexes it by
    /// name so later lookups find it.
    #[allow(clippy::too_many_arguments)]
    fn add_row(
        &mut self,
        name: &[u8],
        weak: bool,
        unique: bool,
        sym_type: u8,
        other: u8,
        shndx: u16,
        value: u64,
        size: u64,
    ) -> u32 {
        let idx = next_index(self.entries.len());
        let name_off = intern(&mut self.strtab, name);
        // The winner's binding is what the row carries: the loader keys its
        // one-copy search for a GNU_UNIQUE symbol on this binding.
        let bind = if weak {
            STB_WEAK
        } else if unique {
            STB_GNU_UNIQUE
        } else {
            STB_GLOBAL
        };
        self.entries.push(DynSymEntry {
            name_off,
            info: (bind << 4) | sym_type,
            other,
            shndx,
            value,
            size,
        });
        self.by_name.insert(name.to_vec(), idx);
        idx
    }

    /// Adds one export, returning its 1-based dynsym index. `name` is the
    /// export's interned name, resolved by the caller through the symbol
    /// table (the export record carries only its id).
    pub(super) fn add_export(&mut self, e: &Export, name: &[u8]) -> u32 {
        self.add_row(
            name,
            e.weak,
            e.unique,
            e.sym_type,
            e.visibility,
            e.shndx,
            e.addr,
            e.size,
        )
    }

    /// Adds one import (an undefined reference resolved at runtime).
    pub(super) fn add_import(
        &mut self,
        name: &[u8],
        weak: bool,
        row: ImportRow,
    ) -> u32 {
        // An import the loader resolves by name is default-visibility by
        // construction: a hidden reference could never have been satisfied
        // from outside this image.
        self.add_row(
            name,
            weak,
            false,
            row.sym_type,
            STV_DEFAULT,
            SHN_UNDEF,
            row.value,
            row.size,
        )
    }

    /// Adds one copy-relocated symbol as a *defined* entry pointing at the
    /// executable's `.bss` copy slot. The loader fills the slot from the
    /// shared dependency, then resolves every reference to this address.
    pub(super) fn add_copy(
        &mut self,
        name: &[u8],
        weak: bool,
        shndx: u16,
        value: u64,
        size: u64,
    ) -> u32 {
        // The copy is this executable's storage for a dependency's
        // default-visibility data symbol, and keeps that visibility so every
        // other image binds the name to the copy.
        self.add_row(
            name,
            weak,
            false,
            STT_OBJECT,
            STV_DEFAULT,
            shndx,
            value,
            size,
        )
    }

    /// The dynsym index of `name`, if it has been added.
    pub(super) fn index_of(&self, name: &[u8]) -> Option<u32> {
        self.by_name.get(name).copied()
    }

    /// Materialises the on-disk `.dynsym`: null entry followed by every row.
    pub(super) fn materialise(self) -> Vec<Sym64> {
        let mut out = Vec::with_capacity(self.entries.len().saturating_add(1));
        out.push(Sym64::zeroed());
        for e in self.entries {
            out.push(Sym64 {
                st_name: U32::new(e.name_off),
                st_info: e.info,
                st_other: e.other,
                st_shndx: U16::new(e.shndx),
                st_value: U64::new(e.value),
                st_size: U64::new(e.size),
            });
        }
        out
    }

    /// Reorders `.dynsym` for the GNU hash table and reports the resulting
    /// layout.
    ///
    /// The GNU hash can only index a contiguous tail of symbols, and that tail
    /// must be sorted by hash bucket. This partitions the entries into an
    /// unhashed prefix (the imports the loader resolves elsewhere) followed by
    /// the rows this image answers for -- see [`DynSymEntry::is_hashable`] --
    /// sorted by `(bucket, name_off)`, then rebuilds `by_name` for the new
    /// order. The
    /// returned [`GnuHashOrdering`] carries the header parameters and an
    /// `old -> new` dynsym-index remap callers apply to any index they hold
    /// (the import map, and the `GLOB_DAT`/`COPY` relocs, resolve via
    /// `by_name`).
    pub(super) fn reorder_for_gnu_hash(&mut self) -> GnuHashOrdering {
        let n = self.entries.len();
        let mut undef_old: Vec<usize> = Vec::new();
        let mut def_old: Vec<usize> = Vec::new();
        for (i, e) in self.entries.iter().enumerate() {
            if e.is_hashable() {
                def_old.push(i);
            } else {
                undef_old.push(i);
            }
        }
        let n_defined = def_old.len();
        let n_buckets = buckets_for(n_defined);
        let mask_words = bloom_words(n_defined as u64);
        // Sort the hashed entries by (bucket, dynstr name offset). The offset
        // tiebreak keeps the order deterministic and matches the reference
        // linker, which keys on (bucket, strtab offset).
        //
        // The key is computed once per entry rather than inside the
        // comparison: a bucket is the symbol's GNU hash, and hashing the name
        // afresh on every comparison means hashing it a couple of dozen times
        // over the sort. The sort is unstable because the key is unique --
        // each name has its own offset in the string table.
        let mut keyed: Vec<(u32, u32, usize)> = def_old
            .iter()
            .map(|&i| {
                let e = &self.entries[i];
                (bucket_of(e, &self.strtab, n_buckets), e.name_off, i)
            })
            .collect();
        keyed.sort_unstable_by_key(|&(bucket, name_off, _)| (bucket, name_off));
        for (slot, &(_, _, i)) in def_old.iter_mut().zip(&keyed) {
            *slot = i;
        }
        // perm[k] = old 0-based index now sitting at new 0-based position k.
        let mut perm = Vec::with_capacity(n);
        perm.extend(undef_old.iter().copied());
        perm.extend(def_old.iter().copied());
        let mut remap = vec![0u32; n.saturating_add(1)];
        for (new0, &old0) in perm.iter().enumerate() {
            remap[old0.saturating_add(1)] =
                u32::try_from(new0.saturating_add(1)).unwrap_or(u32::MAX);
        }
        // Rebuild the entries and name index in the new order. DynSymEntry is
        // Copy, so a gather-by-index rebuilds without ownership gymnastics.
        let old_entries = core::mem::take(&mut self.entries);
        self.entries = perm.iter().map(|&i| old_entries[i]).collect();
        self.by_name.clear();
        for (k, e) in self.entries.iter().enumerate() {
            let name = cstr_at(&self.strtab, e.name_off);
            let idx = u32::try_from(k.saturating_add(1)).unwrap_or(u32::MAX);
            self.by_name.insert(name.to_vec(), idx);
        }
        GnuHashOrdering {
            symoffset: u32::try_from(perm.len() - n_defined + 1)
                .unwrap_or(u32::MAX),
            mask_words,
            n_buckets,
            bloom_shift: GNU_BLOOM_SHIFT,
            remap,
        }
    }
}

/// The GNU hash layout parameters and index remap produced by a reorder.
///
/// `symoffset` is the 1-based `.dynsym` index of the first hashed (defined)
/// symbol; symbols before it are exported-but-unhashed (the null entry plus
/// any undefined imports). `remap[old_1based] = new_1based` lets callers fix
/// up any dynsym index captured before the reorder.
pub(super) struct GnuHashOrdering {
    pub symoffset: u32,
    pub mask_words: u32,
    pub n_buckets: u32,
    pub bloom_shift: u32,
    pub remap: Vec<u32>,
}

/// The 1-based dynsym index for the entry that will sit at `entries.len`.
fn next_index(len: usize) -> u32 {
    u32::try_from(len + 1).unwrap_or(u32::MAX)
}

/// Builds a `SymbolId -> name` lookup borrowing every resolved name out of
/// the symbol arena, indexed by id.
///
/// The ids are dense and already in first-seen input order, so the vector is
/// filled by walking them; recovering each id from its name would cost one
/// hash probe per symbol for an answer the index already gives. The names
/// stay in the arena: copying half a million of them out was a visible slice
/// of the dynamic plan's cost, twice (once for the sizing probe, once for
/// the build).
pub(super) fn names_by_id<'ctx>(ctx: &'ctx Context<'_>) -> Vec<&'ctx [u8]> {
    ctx.symbols.ids().map(|id| ctx.symbols.name(id)).collect()
}

/// Every `.dynsym` row the copy slots contribute, in emission order: slot by
/// slot, and within a slot name by name in the dependency's symbol table
/// order.
///
/// [`add_imports`] adds these rows and [`super::sizes::compute_sizes`] counts
/// them. Both drive this one walk, so the reserved region cannot disagree with
/// the emitted table. Every row carries a name the dependency spelled, which
/// is never empty, so neither pass needs a filter of its own.
pub(super) fn copy_rows(
    layout: &Layout,
) -> impl Iterator<Item = (&CopySlot, &CopyName)> {
    layout
        .copy_slots
        .iter()
        .flat_map(|slot| slot.names.iter().map(move |name| (slot, name)))
}

/// Appends `name` to a string table and returns its offset.
///
/// Callers add each name at most once; an offset that overflows `u32` is
/// clamped so a malformed input cannot panic while serialising.
pub(super) fn intern(bytes: &mut Vec<u8>, name: &[u8]) -> u32 {
    let off = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    bytes.extend_from_slice(name);
    bytes.push(0);
    off
}

/// Adds every imported global the image needs a name for, returning the
/// resolved-symbol-id -> 1-based dynsym index map.
///
/// Three sources contribute, in this order, and a symbol reached more than one
/// way is added once with the index map recording the shared entry:
///
/// 1. The copy slots. Their names are added as *defined* entries pointing at
///    the executable's `.bss` copy slot (matching the system linker), and are
///    skipped by the other two so they are not duplicated as undefined imports.
///    Each name belongs to one slot -- a dependency gives one address one set
///    of names and a name one address -- so no two rows can collide.
/// 2. The GOT slots and PLT entries.
/// 3. The globals an absolute data reference resolves by name, which allocate
///    neither of the above. `data_ids` is that list, surveyed once by
///    [`super::rela::survey_data_relocs`] during sizing and handed here rather
///    than walked again: the two could only ever reach the same answer, and the
///    walk is over every allocated section's relocations.
///
/// Which references become import rows is [`super::is_runtime_import`]'s to
/// say, so a hidden reference -- one no loader could resolve -- gets no row
/// here and none reserved by [`super::sizes::compute_sizes`] either. The
/// sizing pass mirrors all three sources from the one survey.
pub(super) fn add_imports(
    ctx: &Context<'_>,
    layout: &Layout,
    names: &[&[u8]],
    data_ids: &[SymbolId],
    table: &mut DynSym,
) -> FxHashMap<SymbolId, u32> {
    let mut index: FxHashMap<SymbolId, u32> = FxHashMap::default();
    let copy_ids = super::copy_id_set(layout);
    for (slot, name) in copy_rows(layout) {
        let weak = name.id.is_some_and(|id| {
            ctx.symbols.symbol(id).is_some_and(|s| {
                matches!(s.kind, SymbolKind::Undefined { weak: true })
            })
        });
        let idx = table.add_copy(
            &name.name,
            weak,
            layout.shndx(Sect::Bss),
            slot.addr,
            name.size,
        );
        if let Some(id) = name.id {
            index.insert(id, idx);
        }
    }
    for keys in [&layout.got_keys, &layout.plt_keys] {
        for key in keys {
            let Some(id) = key.global_id() else {
                continue;
            };
            let Some(name) = names.get(id.0) else {
                continue;
            };
            if copy_ids.contains(&id) {
                continue;
            }
            let row = import_row(ctx, layout, id, name, key.kind);
            add_import_row(ctx, id, name, row, table, &mut index);
        }
    }
    for &id in data_ids {
        let Some(name) = names.get(id.0) else {
            continue;
        };
        let row = import_row(ctx, layout, id, name, GotKind::Addr);
        add_import_row(ctx, id, name, row, table, &mut index);
    }
    index
}

/// Adds the undefined import row for `id`, unless the table already holds
/// `name`, and records the row's index in `index` either way.
///
/// A reference no loader could resolve gets no row at all; see
/// [`super::is_runtime_import`].
fn add_import_row(
    ctx: &Context<'_>,
    id: SymbolId,
    name: &[u8],
    row: ImportRow,
    table: &mut DynSym,
    index: &mut FxHashMap<SymbolId, u32>,
) {
    if name.is_empty() || !super::is_runtime_import(ctx, id) {
        return;
    }
    let Some(sym) = ctx.symbols.symbol(id) else {
        return;
    };
    let idx = table.index_of(name).unwrap_or_else(|| {
        let weak = matches!(sym.kind, SymbolKind::Undefined { weak: true });
        table.add_import(name, weak, row)
    });
    index.insert(id, idx);
}

/// The row an import reached through a GOT slot, a PLT entry or an absolute
/// data reference takes.
///
/// A canonical PLT import is the one that states an address. This executable
/// took the address of a function a dependency defines from a slot that cannot
/// carry a dynamic relocation, so the link allocated a stub and resolved every
/// reference to it. The row stays `SHN_UNDEF`, which is what keeps the
/// executable's own `JUMP_SLOT` bound to the real function rather than looping
/// back on the stub, while the non-zero `st_value` tells the loader to resolve
/// the name to the stub everywhere else -- in this image and in any other that
/// binds it. That is what makes a function pointer taken from a read-only slot
/// compare equal to one taken from `.data`.
///
/// lld builds the same row: `lld/ELF/Relocations.cpp` explains
/// the mechanism, and `lld/ELF/SyntheticSections.cpp` keeps
/// `st_shndx` at `SHN_UNDEF` for a symbol carrying its `NEEDS_COPY` flag while
/// `st_value` holds the stub.
///
/// The size and the type come from the dependency, which is the only thing
/// that knows the function's extent. Only a TLS key names thread-local
/// storage, and it says so whether or not a dependency spells the name out;
/// everything else takes the type the dependency declares, since a slot
/// holding the address of an imported *object* is as ordinary as one holding
/// the address of a function, and typing both `STT_FUNC` misdescribes the data
/// ones.
fn import_row(
    ctx: &Context<'_>,
    layout: &Layout,
    id: SymbolId,
    name: &[u8],
    kind: GotKind,
) -> ImportRow {
    if matches!(kind, GotKind::TlsOffset | GotKind::TlsModule) {
        return ImportRow {
            sym_type: STT_TLS,
            ..ImportRow::default()
        };
    }
    let mut row = ImportRow {
        sym_type: dep_sym_type(ctx, name),
        ..ImportRow::default()
    };
    if layout.canonical_plt.contains(&id) {
        row.value = layout.plt_entry_addr(id).unwrap_or_default();
        row.size = ctx.dep_exports.get(name).map_or(0, |e| e.size);
    }
    row
}

/// The `STT_*` a dependency declares for `name`, or `STT_NOTYPE` when no
/// dependency names it -- which is what an unresolved weak reference is.
fn dep_sym_type(ctx: &Context<'_>, name: &[u8]) -> u8 {
    ctx.dep_exports.get(name).map_or(STT_NOTYPE, |e| e.sym_type)
}

/// Builds `.dynstr`: the interned names, followed by the SONAME if given.
/// Its offset is stashed on the table for `DT_SONAME`.
pub(super) fn build_dynstr(
    table: &mut DynSym,
    soname: Option<&[u8]>,
) -> Vec<u8> {
    if let Some(s) = soname {
        let off = intern(&mut table.strtab, s);
        table.soname_off = Some(off);
    }
    table.strtab.clone()
}

/// Builds the `SysV` `.hash` table over the dynsym names.
///
/// Layout is `nbucket, nchain, bucket[nbucket], chain[nchain]`, all
/// little-endian 32-bit. `chain[i]` links symbols hashing to the same bucket;
/// a zero entry terminates the chain (matching the null symbol at index 0).
pub(super) fn build_hash(table: &DynSym) -> Vec<u8> {
    let nsyms = table.entries.len().saturating_add(1);
    let nbucket = nsyms.max(1);
    let mut buckets = vec![0u32; nbucket];
    let mut chain = vec![0u32; nsyms];
    for (i, e) in table.entries.iter().enumerate() {
        let sym_idx = u32::try_from(i + 1).unwrap_or(u32::MAX);
        let name = cstr_at(&table.strtab, e.name_off);
        let b = (elf_hash(name) as usize) % nbucket;
        chain[sym_idx as usize] = buckets[b];
        buckets[b] = sym_idx;
    }
    let mut bytes = Vec::with_capacity(
        (2usize + nbucket + nsyms) * core::mem::size_of::<u32>(),
    );
    push_u32(&mut bytes, u32::try_from(nbucket).unwrap_or(u32::MAX));
    push_u32(&mut bytes, u32::try_from(nsyms).unwrap_or(u32::MAX));
    for b in &buckets {
        push_u32(&mut bytes, *b);
    }
    for c in &chain {
        push_u32(&mut bytes, *c);
    }
    bytes
}

/// The bloom shift written into the `.gnu.hash` header. The loader reads it
/// back from the header, so any constant works; GNU ld uses 6 and lld uses 26.
/// We match GNU ld (the value most binaries in the wild carry).
const GNU_BLOOM_SHIFT: u32 = 6;

/// The number of hash buckets for `n_defined` hashed symbols. A load factor of
/// 4 keeps the average chain short; a minimum of 1 avoids a zero-sized table
/// (some loaders reject one). Matches the reference linker.
fn buckets_for(n_defined: usize) -> u32 {
    u32::try_from(n_defined / 4).unwrap_or(u32::MAX).max(1)
}

/// The bloom filter size in 64-bit words for `n_defined` hashed symbols.
/// Allocates 12 bits per symbol and rounds up to a power of two, with a
/// minimum of one word so the table is never zero-sized.
fn bloom_words(n_defined: u64) -> u32 {
    let words = n_defined.saturating_mul(12) / 64;
    let pow = words.checked_next_power_of_two().unwrap_or(1 << 30);
    u32::try_from(pow.max(1)).unwrap_or(1)
}

/// The byte size of `.gnu.hash` for `n_defined` hashed symbols, used by the
/// sizing pass to reserve space before the table is built.
pub(super) fn gnu_hash_size(n_defined: u64) -> u64 {
    let mask_words = u64::from(bloom_words(n_defined));
    let n_buckets = n_defined / 4;
    let n_buckets = n_buckets.max(1);
    16 + 8 * mask_words + 4 * n_buckets + 4 * n_defined
}

/// The GNU hash bucket of `entry`'s name, using `n_buckets` buckets.
fn bucket_of(e: &DynSymEntry, strtab: &[u8], n_buckets: u32) -> u32 {
    let name = cstr_at(strtab, e.name_off);
    dl_new_hash(name) % n_buckets
}

/// Builds the `.gnu.hash` table from a reordered `.dynsym`.
///
/// Layout is the 16-byte header (`nbuckets`, `symoffset`, `bloom_size`,
/// `bloom_shift`), the `bloom_size` 64-bit bloom words, the `nbuckets` bucket
/// indices, then one `u32` per hashed symbol. Each chain word holds the
/// symbol's GNU hash with bit 0 (the LSB) clear, except the last symbol in a
/// bucket's chain sets the LSB to terminate the loader's walk.
pub(super) fn build_gnu_hash(
    table: &DynSym,
    order: &GnuHashOrdering,
) -> Vec<u8> {
    let mask_words = order.mask_words;
    let n_buckets = order.n_buckets.max(1);
    let symoffset = order.symoffset as usize;
    let start = symoffset.saturating_sub(1);
    let defined: &[DynSymEntry] = table.entries.get(start..).unwrap_or(&[]);
    // Hash and bucket each defined symbol once; they are already sorted by
    // bucket from the reorder, so equal buckets are contiguous.
    let hb: Vec<(u32, u32)> = defined
        .iter()
        .map(|e| {
            let name = cstr_at(&table.strtab, e.name_off);
            let h = dl_new_hash(name);
            (h, h % n_buckets)
        })
        .collect();
    let word_mask = u64::from(mask_words).saturating_sub(1);
    let mut bloom = vec![0u64; mask_words as usize];
    for &(h, _) in &hb {
        let word_idx = ((u64::from(h) / 64) & word_mask) as usize;
        let bit1 = u64::from(h % 64);
        let bit2 = u64::from((h >> order.bloom_shift) % 64);
        if let Some(w) = bloom.get_mut(word_idx) {
            *w |= 1u64 << bit1;
            *w |= 1u64 << bit2;
        }
    }
    let mut buckets = vec![0u32; n_buckets as usize];
    let mut chain: Vec<u32> = Vec::with_capacity(hb.len());
    let mut prev_bucket: i64 = -1;
    for (k, &(h, bucket)) in hb.iter().enumerate() {
        // The chain word clears bit 0 while another symbol in the same bucket
        // follows, and sets it (terminating the loader's walk) otherwise: the
        // next entry sits in a different bucket, or this is the last symbol.
        let continues = hb.get(k + 1).is_some_and(|&(_, nb)| nb == bucket);
        let val = if continues { h & !1 } else { h | 1 };
        chain.push(val);
        // First symbol of this bucket records its dynsym index.
        if i64::from(bucket) != prev_bucket {
            let idx = u32::try_from(symoffset + k).unwrap_or(u32::MAX);
            if let Some(b) = buckets.get_mut(bucket as usize) {
                *b = idx;
            }
            prev_bucket = i64::from(bucket);
        }
    }
    let mut bytes = Vec::with_capacity(
        16 + 8 * mask_words as usize + 4 * n_buckets as usize + 4 * hb.len(),
    );
    push_u32(&mut bytes, order.n_buckets);
    push_u32(&mut bytes, order.symoffset);
    push_u32(&mut bytes, order.mask_words);
    push_u32(&mut bytes, order.bloom_shift);
    for w in &bloom {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    for b in &buckets {
        push_u32(&mut bytes, *b);
    }
    for c in &chain {
        push_u32(&mut bytes, *c);
    }
    bytes
}

/// The GNU symbol hash (`dl_new_hash`): `h = 5381; h = h*33 + byte`. Distinct
/// from the `SysV` `elf_hash`; the loader uses it to pick a `.gnu.hash` bucket.
#[allow(clippy::cast_possible_truncation)]
fn dl_new_hash(name: &[u8]) -> u32 {
    let mut h: u32 = 5381;
    for &c in name {
        h = h.wrapping_shl(5).wrapping_add(h).wrapping_add(u32::from(c));
    }
    h
}

/// The ELF (`SysV`) symbol hash: a 32-bit rolling hash over a `NUL`-terminated
/// name. The loader uses it to pick a bucket in `.hash`.
#[allow(clippy::cast_possible_truncation)]
fn elf_hash(name: &[u8]) -> u32 {
    let mut h: u32 = 0;
    for &c in name {
        h = h.wrapping_shl(4).wrapping_add(u32::from(c));
        let g = h & 0xf000_0000;
        if g != 0 {
            h ^= g >> 24;
        }
        h &= !g;
    }
    h
}
