//! The GOT key types and the slot lookups built on them.
//!
//! The scan pass names every entry the image needs with a [`GotKey`]; layout
//! turns that ordered list into addresses, and the resolver reads a slot back
//! out by key. Keeping the identity type and the lookups together is what
//! stops the two from drifting: a key is only ever meaningful against the
//! index the same list built.

use super::{GOT_ENTRY, Layout};
use crate::{symbol::SymbolId, tls};

/// What a GOT entry holds, which decides both how many slots it takes and
/// which dynamic relocation the loader applies to it.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub enum GotKind {
    /// A symbol's address, in one slot.
    Addr,
    /// A thread-local's offset from the thread pointer, in one slot. The
    /// initial-exec model reads it directly.
    TlsOffset,
    /// A thread-local's module id and its offset within that module's block,
    /// in two slots and that order. This is the pair `__tls_get_addr` is
    /// handed by the general-dynamic model, and only a shared object needs
    /// one: an executable resolves the same reference without a runtime call.
    TlsModule,
}

impl GotKind {
    /// How many slots an entry of this kind occupies.
    pub const fn slots(self) -> u64 {
        match self {
            Self::Addr | Self::TlsOffset => 1,
            Self::TlsModule => 2,
        }
    }
}

/// Whose GOT entry it is. A global symbol has one entry shared by every
/// reference; a local symbol's entry is private to its file.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub enum GotOwner {
    /// One entry per resolved global symbol.
    Global(SymbolId),
    /// A file-private entry for a local symbol `(file, sym_idx)`.
    Local(usize, u32),
    /// The image's own entry, shared by every reference: the local-dynamic
    /// model asks the runtime for this module's block once and then reaches
    /// each thread-local by a fixed offset within it.
    Module,
}

impl GotOwner {
    /// The owner a relocation names: the resolved global when it has one, and
    /// the file-local symbol otherwise.
    pub fn of(id: Option<SymbolId>, file: usize, sym_idx: u32) -> Self {
        id.map_or(Self::Local(file, sym_idx), Self::Global)
    }

    /// The identity of this image's own module entry.
    pub const MODULE: Self = Self::Module;
}

/// The identity a GOT entry is shared over: what it holds, and for whom. Two
/// references share an entry when both agree.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct GotKey {
    pub kind: GotKind,
    pub owner: GotOwner,
}

impl core::hash::Hash for GotKey {
    /// Hashes the key as one word rather than field by field.
    ///
    /// Address resolution asks for a symbol's entries once per symbol per
    /// file, so this runs a few million times on a large link, and the
    /// per-field writes a derived implementation makes are the cost. The tag
    /// bits ride above the payload, which is a symbol id or a `(file, symbol)`
    /// pair and never reaches them.
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        const TAG_SHIFT: u32 = 60;
        let (tag, payload) = match self.owner {
            GotOwner::Global(id) => (0u64, id.0 as u64),
            GotOwner::Local(file, sym_idx) => {
                (1, (file as u64) << 32 | u64::from(sym_idx))
            }
            GotOwner::Module => (2, 0),
        };
        let kind = self.kind as u64;
        state.write_u64(payload ^ ((tag | (kind << 2)) << TAG_SHIFT));
    }
}

impl GotKey {
    /// An entry of `kind` for `owner`.
    pub const fn new(kind: GotKind, owner: GotOwner) -> Self {
        Self { kind, owner }
    }

    /// An address entry for a global symbol, the common case.
    pub const fn addr(id: SymbolId) -> Self {
        Self::new(GotKind::Addr, GotOwner::Global(id))
    }

    /// The global symbol identity, if this entry belongs to one.
    pub fn global_id(&self) -> Option<SymbolId> {
        match self.owner {
            GotOwner::Global(id) => Some(id),
            GotOwner::Local(..) | GotOwner::Module => None,
        }
    }

    /// This image's own module entry, which the local-dynamic model reads.
    pub const MODULE: Self = Self::new(GotKind::TlsModule, GotOwner::Module);
}

/// The index of the GOT entry the scan allocated for `key`, if it made one.
/// `key` is the precomputed identity (global id or file-local pair) the caller
/// derived from the per-file `sym_id` cache, so this is a plain index lookup
/// with no name probe.
pub(super) fn got_entry(layout: &Layout, key: GotKey) -> Option<u32> {
    if layout.got.size == 0 {
        return None;
    }
    layout.got_index.get(&key).copied()
}

/// The GOT slot address for `key`, or 0 if it has no entry.
pub(super) fn got_slot(layout: &Layout, key: GotKey) -> u64 {
    let Some(index) = got_entry(layout, key) else {
        return 0;
    };
    layout.got.vaddr.wrapping_add(u64::from(index) * GOT_ENTRY)
}

/// The PLT entry address for `key`, or 0 if it has no entry. Only imports
/// (undefined globals reached via a call) get a PLT entry; every other symbol
/// returns 0 so the resolver collapses its `PLT[sym]` to a direct call.
pub(super) fn plt_slot(layout: &Layout, key: GotKey) -> u64 {
    if layout.plt.size == 0 {
        return 0;
    }
    let Some(&index) = layout.plt_index.get(&key) else {
        return 0;
    };
    // `PLT[1 + index]`: skip the target's `PLT[0]` resolver trampoline
    // (16 bytes on x86-64, 32 on AArch64 and RISC-V).
    let spec = layout.target.plt_spec();
    let entry_off = spec
        .header_size
        .wrapping_add(u64::from(index) * spec.entry_size);
    layout.plt.vaddr.wrapping_add(entry_off)
}

/// Stores the resolved value of each GOT entry.
///
/// An address entry takes the symbol's address, and a `TlsOffset` entry its
/// offset from the thread pointer, which is what the resolved value of a TLS
/// symbol already is (and is zero for one a shared object owns, whose slot
/// the loader fills). A `TlsModule` pair holds the module id, which only the
/// loader knows and so starts at zero, followed by the symbol's offset within
/// that module's block.
pub(super) fn fill_got(global_addr: &[u64], layout: &mut Layout) {
    let block = layout.tls_block();
    let target = layout.target;
    // The values are taken out so the loop can read the symbol rows while it
    // writes; nothing else touches them in between.
    let mut values = core::mem::take(&mut layout.got_values);
    for (key, &index) in &layout.got_index {
        let value = match key.owner {
            GotOwner::Global(id) => {
                global_addr.get(id.0).copied().unwrap_or_default()
            }
            GotOwner::Local(file, sym_idx) => layout
                .sym
                .get(file)
                .and_then(|r| r.addr.get(sym_idx as usize))
                .copied()
                .unwrap_or_default(),
            // The module entry names no symbol: both its slots are the
            // loader's to fill.
            GotOwner::Module => continue,
        };
        let index = index as usize;
        if key.kind == GotKind::TlsModule {
            // The pair's second slot: the offset within the module's own
            // block, recovered from the thread-pointer-relative value above.
            let offset = block
                .map(|b| tls::block_offset(target, value, &b))
                .unwrap_or_default();
            if let Some(slot) = values.get_mut(index + 1) {
                *slot = offset;
            }
            continue;
        }
        if let Some(slot) = values.get_mut(index) {
            *slot = value;
        }
    }
    layout.got_values = values;
}
