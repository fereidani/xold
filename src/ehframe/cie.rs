//! Cross-member deduplication of `.eh_frame` CIE records.
//!
//! A CIE describes the calling convention its FDEs were compiled under, not
//! any one function, so every translation unit built with the same flags emits
//! the same one. A large C++ link carries thousands of byte-identical copies:
//! the 2868 objects of an LLVM build carry 5268 CIEs in exactly two distinct
//! shapes, 142 KiB of which is repetition. lld and mold both emit one copy per
//! distinct CIE (`EhFrameSection::addCie`, keyed on the record's bytes and its
//! personality symbol); this module is that key and the pool it feeds.
//!
//! Two CIEs may share a slot only when nothing distinguishes them after the
//! link, which is more than equal bytes. A CIE's personality routine reaches it
//! through a relocation, so the bytes hold a placeholder and the identity of
//! the routine lives in the relocation. The key therefore pairs the record's
//! bytes with every relocation that lands inside it, each reduced to what the
//! link will make of it: where in the record it applies, its type, its addend,
//! and the symbol it names. A global is named by its resolved [`SymbolId`], so
//! two files referring to one `__gxx_personality_v0` agree; a local is named by
//! its file, so two files' same-named private symbols never do. A relocation
//! that resolves to nothing makes the CIE unique, which keeps a reference this
//! pass cannot account for out of the pool.
//!
//! The pool is filled by one sequential walk in output-member order, so the
//! surviving copy of a CIE is always the earliest one in the image and the
//! backwards `CIE_pointer` of every FDE that shares it stays positive. Nothing
//! here reads the map in iteration order.

use rustc_hash::FxHashMap;

use crate::{
    elf::{Rela64, SymbolTable, constants::STB_LOCAL},
    linker::Context,
    symbol::SymbolId,
};

/// Where a surviving CIE landed: the member that kept it, and the offset it
/// was assigned within that member's output contribution.
#[derive(Clone, Copy, Debug)]
pub struct CieRef {
    /// Input file index of the member holding the surviving copy.
    pub file: usize,
    /// Input section index of that member.
    pub section: u16,
    /// The copy's offset within the member's output contribution.
    pub out_off: u64,
}

/// One relocation landing in a CIE, reduced to what survives the link.
#[derive(PartialEq, Eq, Hash)]
struct CieReloc {
    /// Offset of the relocated field within the record.
    at: u64,
    /// Relocation type.
    kind: u32,
    /// Relocation addend.
    addend: i64,
    /// The symbol the relocation names.
    target: CieTarget,
}

/// The identity of a symbol a CIE's relocation names.
#[derive(PartialEq, Eq, Hash)]
enum CieTarget {
    /// A global, by the definition the link resolved it to. Two files naming
    /// one routine agree here.
    Global(SymbolId),
    /// A local, by the file that defines it and its index there. Two files
    /// never agree here, which is the conservative answer: a private symbol is
    /// a different symbol in every file that spells it.
    Local(usize, u32),
}

/// What distinguishes one CIE from another after the link.
#[derive(PartialEq, Eq, Hash)]
pub struct CieKey {
    bytes: Vec<u8>,
    relocs: Vec<CieReloc>,
}

impl CieKey {
    /// The key for the record `bytes`, relocated by the entries of `covering`
    /// (given as `(record index, offset within the record, entry)` rows, of
    /// which only the last two fields are read here).
    ///
    /// `None` when a relocation names a symbol the link did not resolve to a
    /// definition, which leaves the record's final content unknown to this
    /// pass and so ineligible for sharing.
    pub fn new(
        ctx: &Context<'_>,
        symtab: &SymbolTable<'_>,
        file: usize,
        bytes: &[u8],
        covering: &[(usize, u64, Rela64)],
    ) -> Option<Self> {
        let mut relocs = Vec::with_capacity(covering.len());
        for (_, at, r) in covering {
            relocs.push(CieReloc {
                at: *at,
                kind: r.r_type(),
                addend: r.r_addend.get(),
                target: target_of(ctx, symtab, file, r.sym())?,
            });
        }
        Some(Self {
            bytes: bytes.to_vec(),
            relocs,
        })
    }
}

/// The symbol `sym_idx` names, as the link sees it.
fn target_of(
    ctx: &Context<'_>,
    symtab: &SymbolTable<'_>,
    file: usize,
    sym_idx: u32,
) -> Option<CieTarget> {
    let sym = symtab.syms.get(sym_idx as usize)?;
    if sym.bind() == STB_LOCAL {
        return Some(CieTarget::Local(file, sym_idx));
    }
    // The table keys stems: a versioned personality spelling resolves as
    // its stem.
    let name = crate::symbol::version_stem(symtab.name(sym));
    ctx.symbols.find(name).map(CieTarget::Global)
}

/// What the pool made of one offered CIE.
pub enum Claim {
    /// This copy is the survivor. The caller keeps it and, once the member's
    /// records have their output offsets, reports the one it landed at to
    /// [`CiePool::place`] under this slot.
    Kept(usize),
    /// An earlier copy already holds this key. The caller drops this one and
    /// points its FDEs at the returned location.
    Shared(CieRef),
}

/// The CIEs the output keeps, one per distinct [`CieKey`].
///
/// Filled member by member in output order, so the copy a key resolves to is
/// always the earliest one in the image. Claiming is two steps because a
/// member's records are given their output offsets only after its whole keep
/// set is known: [`CiePool::claim`] reserves the key and hands back a slot,
/// and [`CiePool::place`] records where that copy landed.
#[derive(Default)]
pub struct CiePool {
    seen: FxHashMap<CieKey, usize>,
    refs: Vec<CieRef>,
}

impl CiePool {
    /// Offers the CIE `key`, sitting in the member `at` names, to the pool.
    pub fn claim(&mut self, key: CieKey, at: CieRef) -> Claim {
        if let Some(found) =
            self.seen.get(&key).and_then(|&slot| self.refs.get(slot))
        {
            return Claim::Shared(*found);
        }
        let slot = self.refs.len();
        self.refs.push(at);
        self.seen.insert(key, slot);
        Claim::Kept(slot)
    }

    /// Records the offset a claimed survivor was assigned within its member.
    pub fn place(&mut self, slot: usize, out_off: u64) {
        if let Some(found) = self.refs.get_mut(slot) {
            found.out_off = out_off;
        }
    }
}
