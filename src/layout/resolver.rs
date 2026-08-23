//! The per-file relocation resolver: the addresses and facts one input file's
//! relocations are applied against.
//!
//! [`super::build`] fills a [`SymRows`] per input file, in parallel and with no
//! synchronisation, because the rows are owned per file rather than shared.
//! The writer then borrows one file's rows as a [`FileResolver`], which is the
//! [`Resolver`] the arch-neutral relocation driver runs against.

use crate::{
    reloc::{Resolver, Target},
    symbol::SymbolId,
    tls::{self, TlsBlock},
};

/// The per-symbol facts the relaxation pass needs that are not addresses.
///
/// Each is asked as its own question rather than being read off an address:
/// zero is a legal address in a position-independent image, so it can stand
/// neither for "no GOT slot" nor for "not defined here".
#[derive(Clone, Copy, Default)]
#[allow(clippy::struct_excessive_bools)]
pub(super) struct SymFlags {
    /// Whether a definition of this symbol could be replaced at load time by
    /// one from another image. See [`crate::symbol::Symbol::is_preemptible`].
    pub(super) preemptible: bool,
    /// Whether the symbol's value is a link-time constant rather than a place
    /// in the image. See [`GlobalFacts::absolute`].
    pub(super) absolute: bool,
    /// Whether the scan allocated a GOT slot for this symbol. It does so for
    /// every GOT-indirect reference except the `__tls_get_addr` call a lowered
    /// general-dynamic pair consumes.
    pub(super) got_slot: bool,
    /// Whether the symbol is a weak reference that nothing defines. A
    /// PC-relative site against one stores the architecture's answer rather
    /// than the plain `S + A - P`; see
    /// [`crate::reloc::Arch::undef_weak_pc`].
    pub(super) undef_weak: bool,
}

/// One input file's resolved per-symbol values.
///
/// The rows are parallel to the file's symbol table and are owned per file
/// rather than shared, so [`resolve_addresses`] fills every file's row
/// concurrently without any synchronisation.
#[derive(Default)]
pub(super) struct SymRows {
    /// The resolved value `S`. For a TLS symbol this is the
    /// thread-pointer-relative offset (TPOFF), not the vaddr.
    pub(super) addr: Vec<u64>,
    /// The GOT entry address, or 0. Empty when the link has no GOT.
    pub(super) got: Vec<u64>,
    /// The PLT entry address, or 0 when the symbol has no PLT entry (defined
    /// locally, so calls go direct). Empty when the link has no PLT.
    pub(super) plt: Vec<u64>,
    /// The address of the GOT slot holding the symbol's thread-pointer
    /// offset, or 0 when it has none. Empty unless the link imports a
    /// thread-local.
    pub(super) tls_got: Vec<u64>,
    /// The relaxation pass's per-symbol facts. Empty when the link has no
    /// GOT, which is exactly when no GOT-indirect site can be relaxed.
    pub(super) flags: Vec<SymFlags>,
    /// Each symbol's `st_size`, for the size relocations. Empty unless the
    /// scan saw one: a `u64` per symbol per file is real memory, and the
    /// relocation that reads it appears in almost no object.
    pub(super) size: Vec<u64>,
}

/// A relocation resolver bound to a single input file's symbol addresses.
pub struct FileResolver<'a> {
    pub(super) addr: &'a [u64],
    pub(super) got: &'a [u64],
    pub(super) plt: &'a [u64],
    pub(super) tls_got: &'a [u64],
    pub(super) flags: &'a [SymFlags],
    pub(super) size: &'a [u64],
    /// The address of this image's own module entry, or 0 when it has none.
    pub(super) tls_index: u64,
    /// The static TLS block and the target whose convention placed it, for
    /// converting a thread-pointer-relative value into a module-relative one.
    pub(super) tls: Option<(Target, TlsBlock)>,
    pub(super) got_base: u64,
    /// Whether the image is an executable, which fixes the thread pointer
    /// offset of the main module's thread-locals and so permits the
    /// general-dynamic TLS rewrite.
    pub(super) exec: bool,
}

impl Resolver for FileResolver<'_> {
    fn symbol_addr(&self, sym: SymbolId) -> u64 {
        self.addr.get(sym.0).copied().unwrap_or_default()
    }

    fn got_addr(&self, sym: SymbolId) -> u64 {
        self.got.get(sym.0).copied().unwrap_or_default()
    }

    fn sym_size(&self, sym: SymbolId) -> u64 {
        self.size.get(sym.0).copied().unwrap_or_default()
    }

    fn got_base(&self) -> u64 {
        self.got_base
    }

    fn plt_addr(&self, sym: SymbolId) -> u64 {
        // A symbol with a PLT entry (an imported function) routes through the
        // stub so the loader can bind it lazily; any other symbol collapses to
        // a direct call at its resolved address.
        let entry = self.plt.get(sym.0).copied().unwrap_or_default();
        if entry != 0 {
            entry
        } else {
            self.symbol_addr(sym)
        }
    }

    fn tls_got_addr(&self, sym: SymbolId) -> u64 {
        self.tls_got.get(sym.0).copied().unwrap_or_default()
    }

    fn tls_index_addr(&self) -> u64 {
        self.tls_index
    }

    fn tls_dtp_off(&self, sym: SymbolId) -> u64 {
        let value = self.symbol_addr(sym);
        // In an executable the local-dynamic sequence is lowered to read the
        // thread pointer directly, so the offset is measured from there --
        // which is what a TLS symbol resolves to already. A shared object
        // keeps the model, and its base is the module's own block.
        match self.tls {
            Some((target, block)) if !self.exec => {
                tls::block_offset(target, value, &block)
            }
            _ => value,
        }
    }

    fn is_exec(&self) -> bool {
        self.exec
    }

    fn is_preemptible(&self, sym: SymbolId) -> bool {
        // A symbol with no row is one the link never resolved; declining the
        // rewrite is the safe answer, and matches the trait's default.
        self.flags.get(sym.0).is_none_or(|f| f.preemptible)
    }

    fn is_absolute(&self, sym: SymbolId) -> bool {
        // As above: no row means no answer, so decline the rewrite.
        self.flags.get(sym.0).is_none_or(|f| f.absolute)
    }

    fn has_got_slot(&self, sym: SymbolId) -> bool {
        // An absent row means the link has no GOT at all, so no symbol has a
        // slot in it.
        self.flags.get(sym.0).is_some_and(|f| f.got_slot)
    }

    fn is_undef_weak(&self, sym: SymbolId) -> bool {
        self.flags.get(sym.0).is_some_and(|f| f.undef_weak)
    }
}
