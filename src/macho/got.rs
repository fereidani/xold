//! Scan-driven allocation of the Mach-O `__got` section.
//!
//! Mirrors the ELF scan pass in [`crate::layout::scan`]: every relocation of
//! every linked input section is classified through the arch-neutral `Needs`
//! table (see [`crate::reloc`]), and the external symbols referenced by a GOT
//! relocation are each awarded one non-lazy pointer slot. On `x86_64` those are
//! `X86_64_RELOC_GOT_LOAD` / `X86_64_RELOC_GOT` (a `movq sym@GOTPCREL(%rip),
//! %reg` and its siblings); on `arm64` the `GOT_LOAD_PAGE21` /
//! `GOT_LOAD_PAGEOFF12` / `POINTER_TO_GOT` family.
//!
//! Slots are de-duplicated by what identifies the referent. A global is its
//! name: defined in one file and referenced as an undefined extern from
//! several others, it occupies a single `__got` entry, matching the cross-file
//! resolution the symbol table already performs. A file-private symbol is its
//! file and index instead -- keying those by name too gave two files' `static
//! counter` one shared slot, which is one variable where the program declared
//! two. The walk is deterministic (inputs in order, first-seen wins), so
//! re-links of the same inputs reproduce the same `__got` byte for byte.

use rustc_hash::FxHashMap;

use crate::{
    error::Result,
    macho::{
        MachOFile,
        constants::N_EXT,
        layout::is_linkable,
        reloc::{MachoTarget, scan_needs},
    },
};

/// The planned `__got` content: external symbols that need a non-lazy pointer
/// slot, plus a name-to-slot index.
///
/// Slots are recorded in first-seen order. Both fields borrow the input bytes
/// (symbol names), so a plan outlives no input.
pub struct GotPlan<'d> {
    keys: Vec<GotKey<'d>>,
    index: FxHashMap<GotKey<'d>, u32>,
}

/// What a `__got` slot belongs to.
///
/// A global is named across files, so its name is the key. A file-private
/// symbol has meaning only inside its own object, so its file and index are.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum GotKey<'d> {
    Global(&'d [u8]),
    Local { file: usize, sym: u32 },
}

impl<'d> GotPlan<'d> {
    /// Walks every linked input section's relocations and records the external
    /// symbols referenced through a GOT relocation. Deferred sections (unwind,
    /// DWARF, dynamic-linker tables) are skipped via [`is_linkable`], matching
    /// the set of sections the writer actually copies.
    pub fn scan(
        inputs: &'d [MachOFile<'d>],
        target: MachoTarget,
    ) -> Result<Self> {
        let mut keys: Vec<GotKey<'d>> = Vec::new();
        let mut index: FxHashMap<GotKey<'d>, u32> = FxHashMap::default();
        for (file, input) in inputs.iter().enumerate() {
            // Cache the per-file symbol identities so each reloc's symbolnum
            // resolves in one indexed lookup instead of a re-walk of the
            // nlist array.
            let sym_keys: Vec<GotKey<'d>> = input
                .symbols()
                .iter()
                .enumerate()
                .map(|(i, s)| key_of(file, i, s.name, s.n_type & N_EXT != 0))
                .collect();
            for section in input.sections() {
                if !is_linkable(&section) {
                    continue;
                }
                for reloc in &section.relocations {
                    if !reloc.r_extern {
                        continue;
                    }
                    let needs = scan_needs(target, u32::from(reloc.r_type))?;
                    if !needs.got {
                        continue;
                    }
                    let Some(&key) = sym_keys.get(reloc.r_symbolnum as usize)
                    else {
                        continue;
                    };
                    if matches!(key, GotKey::Global(n) if n.is_empty())
                        || index.contains_key(&key)
                    {
                        continue;
                    }
                    let slot = u32::try_from(keys.len()).unwrap_or(u32::MAX);
                    index.insert(key, slot);
                    keys.push(key);
                }
            }
        }
        Ok(Self { keys, index })
    }

    /// The number of GOT slots (one per unique referenced symbol).
    pub fn count(&self) -> u32 {
        u32::try_from(self.keys.len()).unwrap_or(u32::MAX)
    }

    /// The slot index of one file's symbol, if it has a GOT entry.
    pub fn slot(
        &self,
        file: usize,
        sym: usize,
        name: &'d [u8],
        external: bool,
    ) -> Option<u32> {
        self.index.get(&key_of(file, sym, name, external)).copied()
    }

    /// An iterator over `(slot index, key)` in allocation order. Used by the
    /// writer to fill each slot with its resolved symbol address.
    pub fn iter(&self) -> impl Iterator<Item = (u32, GotKey<'d>)> {
        self.keys
            .iter()
            .enumerate()
            .map(|(i, &k)| (u32::try_from(i).unwrap_or(u32::MAX), k))
    }
}

/// The key one symbol occupies.
fn key_of(file: usize, sym: usize, name: &[u8], external: bool) -> GotKey<'_> {
    if external {
        GotKey::Global(name)
    } else {
        GotKey::Local {
            file,
            sym: u32::try_from(sym).unwrap_or(u32::MAX),
        }
    }
}
