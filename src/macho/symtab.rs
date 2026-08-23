//! Mach-O symbol resolution for a static link.
//!
//! The ELF `SymbolTable` is ELF-specific (it folds `InputSymbol` records that
//! carry ELF binding/type nibbles and `SHN_*` reserved indices), so the Mach-O
//! path keeps its own lightweight resolution: a name-to-definition map of the
//! external symbols that can be referenced across object boundaries, plus the
//! per-file address resolution that turns a symbol's `n_sect` / `n_value` into
//! a laid-out virtual address.
//!
//! Only the global table borrows the input bytes (symbol names); once every
//! symbol is resolved to a `u64` address the rest of the linker carries no
//! borrows, so layout and the writer are borrow-free.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    error::{Error, Result},
    macho::{
        MachOFile, MachSymbol,
        constants::{
            N_ABS, N_EXT, N_SECT, N_TYPE, N_UNDF, N_WEAK_DEF, N_WEAK_REF,
        },
    },
    util::show,
};

/// A defined global symbol: which file and which `nlist` index defines it,
/// whether the definition is weak, and whether it is a tentative
/// (`N_UNDF` + `n_value > 0`) definition.
#[derive(Clone, Copy)]
pub struct GlobalDef {
    pub file: usize,
    pub sym: u32,
    /// `N_WEAK_DEF`: a strong definition of the same name overrides this one.
    pub weak: bool,
    /// A tentative definition: `N_UNDF` with the size in `n_value`. The
    /// storage comes from the `__common` section, not from the defining
    /// file, so a `common` definition resolves through [`Globals::commons`].
    pub common: bool,
}

/// One merged common symbol: its name, size and alignment.
///
/// `n_desc` bits 8-11 of a tentative definition carry the alignment as a
/// power-of-two exponent.
pub struct CommonSym<'data> {
    pub name: &'data [u8],
    pub size: u64,
    pub align: u32,
}

/// The global symbol table: external defined names -> definition, plus the
/// merged common symbols in first-appearance order. Names borrow the mapped
/// input bytes, so this table outlives no input.
pub struct Globals<'d> {
    by_name: FxHashMap<&'d [u8], GlobalDef>,
    /// Merged tentative definitions, in the order the names first appear,
    /// so the `__common` layout is reproducible.
    pub commons: Vec<CommonSym<'d>>,
}

impl<'d> Globals<'d> {
    /// Builds the table from the parsed inputs. A strong definition displaces
    /// a weak one (`N_WEAK_DEF` in `n_desc`) and either displaces a
    /// tentative one; a tentative definition displaces a weak one but not a
    /// strong one; two tentative definitions merge into the larger size.
    /// Otherwise the first external definition of a name wins, the way a
    /// darwin link of two weak definitions keeps the first. darwin symbols
    /// carry a leading `_` on both the definition and every reference, so
    /// the comparison is on the full name.
    pub fn build(inputs: &'d [MachOFile<'d>]) -> Result<Self> {
        let mut by_name: FxHashMap<&'d [u8], GlobalDef> = FxHashMap::default();
        let mut commons: Vec<CommonSym<'d>> = Vec::new();
        let mut common_at: FxHashMap<&'d [u8], usize> = FxHashMap::default();
        for (file, input) in inputs.iter().enumerate() {
            let syms = input.symbols();
            for (i, sym) in syms.iter().enumerate() {
                if !is_global_definition(&sym) {
                    continue;
                }
                if sym.name.is_empty() {
                    continue;
                }
                let sym_idx = u32::try_from(i)
                    .map_err(|_| Error::OutOfRange("symbol index"))?;
                let typ = sym.n_type & N_TYPE;
                let common = typ == N_UNDF;
                let def = GlobalDef {
                    file,
                    sym: sym_idx,
                    weak: sym.n_desc & N_WEAK_DEF != 0,
                    common,
                };
                by_name
                    .entry(sym.name)
                    .and_modify(|slot| {
                        if outranks(&def, slot) {
                            *slot = def;
                        }
                    })
                    .or_insert(def);
                if common {
                    merge_common(&mut commons, &mut common_at, &sym);
                }
            }
        }
        let mut table = Self { by_name, commons };
        table.retain_won_commons();
        Ok(table)
    }

    /// The definition of `name`, if some input defines it as a global.
    pub fn get(&self, name: &[u8]) -> Option<GlobalDef> {
        self.by_name.get(name).copied()
    }

    /// Drops commons whose name a real definition won: their storage would
    /// sit in `__common` unreferenced. Called once the table is complete.
    fn retain_won_commons(&mut self) {
        let won: FxHashSet<&'d [u8]> = self
            .by_name
            .iter()
            .filter(|(_, def)| def.common)
            .map(|(name, _)| *name)
            .collect();
        self.commons.retain(|c| won.contains(c.name));
    }
}

/// Whether `incoming` displaces `current` for one name: a strong definition
/// outranks a tentative one, which outranks a weak one; equal ranks keep the
/// incumbent (two tentative definitions merge by size separately, and either
/// of their table entries names the same merged storage).
fn outranks(incoming: &GlobalDef, current: &GlobalDef) -> bool {
    fn rank(def: &GlobalDef) -> u8 {
        match (def.common, def.weak) {
            (false, false) => 2,
            (true, _) => 1,
            (false, true) => 0,
        }
    }
    rank(incoming) > rank(current)
}

/// Merges one tentative definition into the commons list: a new name is
/// appended (recording its slot for later merges), a repeated name keeps
/// the larger size and the stricter alignment.
fn merge_common<'d>(
    commons: &mut Vec<CommonSym<'d>>,
    common_at: &mut FxHashMap<&'d [u8], usize>,
    sym: &MachSymbol<'d>,
) {
    // `n_desc` bits 8-11 hold the alignment as a power-of-two exponent.
    let align = u32::from((sym.n_desc >> 8) & 0x0f);
    let Some(&at) = common_at.get(sym.name) else {
        common_at.insert(sym.name, commons.len());
        commons.push(CommonSym {
            name: sym.name,
            size: sym.n_value,
            align,
        });
        return;
    };
    if let Some(slot) = commons.get_mut(at) {
        slot.size = slot.size.max(sym.n_value);
        slot.align = slot.align.max(align);
    }
}

/// Whether `sym` is an external (`N_EXT`) defined symbol usable for cross-file
/// resolution. Debug stab entries and undefined references are excluded: stab
/// carries no link semantics, and undefined symbols are what resolution is
/// trying to satisfy. A tentative definition (`N_UNDF` with a size in
/// `n_value`) counts as defined.
///
/// `N_PEXT` does not disqualify a definition. lld reads scope off
/// `n_type & (N_EXT | N_PEXT)`: `N_EXT | N_PEXT` is linkage-unit scoped and
/// still takes part in the link, so that duplicates are reported or merged;
/// what `N_PEXT` withholds is a place in the output's export table, which
/// only a dylib has. `N_PEXT` without `N_EXT` is translation-unit scoped and
/// is already rejected by the `N_EXT` test below.
fn is_global_definition(sym: &MachSymbol<'_>) -> bool {
    if sym.is_stab() {
        return false;
    }
    if sym.n_type & N_EXT == 0 {
        return false;
    }
    let typ = sym.n_type & N_TYPE;
    if typ == N_UNDF {
        return sym.n_value > 0;
    }
    typ == N_SECT || typ == N_ABS
}

/// Resolves the virtual address of one symbol of one file.
///
/// Undefined globals resolve through [`Globals`] to their defining file; local
/// symbols and in-file definitions resolve directly from the symbol's
/// `n_sect` / `n_value`. An `N_SECT` symbol's `n_value` is an address in the
/// object's own section frame, so the frame base comes off before the placed
/// base goes on. A tentative definition (`N_UNDF` with a size) resolves to
/// its storage in the merged `__common` section through `common_addr`, which
/// the caller fills from the layout; a name absent from that map has no
/// common storage and falls through to the ordinary undefined path.
// The map is always the linker's own FxHashMap; a hasher parameter would
// generalise an internal API with exactly one caller.
#[allow(clippy::implicit_hasher)]
pub fn symbol_addr(
    inputs: &[MachOFile<'_>],
    globals: &Globals<'_>,
    common_addr: &FxHashMap<&[u8], u64>,
    sec_vaddr: &[Vec<u64>],
    file: usize,
    sym_idx: usize,
) -> Result<u64> {
    let input = inputs.get(file).ok_or(Error::OutOfRange("input file"))?;
    let sym = input
        .symbols()
        .nth(sym_idx)
        .ok_or(Error::OutOfRange("symbol index in file"))?;
    let typ = sym.n_type & N_TYPE;
    match typ {
        N_SECT => {
            let frame = input
                .section_addr(sym.n_sect)
                .ok_or(Error::OutOfRange("symbol section ordinal"))?;
            let offset = sym.n_value.wrapping_sub(frame);
            section_relative_addr(sec_vaddr, file, sym.n_sect, offset)
        }
        N_ABS => Ok(sym.n_value),
        N_UNDF => {
            // A tentative definition names storage in `__common`, not in its
            // own file: the address comes from the merged common table.
            if sym.n_value > 0
                && let Some(addr) = common_addr.get(sym.name)
            {
                return Ok(*addr);
            }
            // A weak reference (`N_WEAK_REF`) that nothing defines binds to
            // zero instead of failing the link; one that is defined resolves
            // normally, so the bit only opens the fallback.
            if sym.n_desc & N_WEAK_REF != 0 && globals.get(sym.name).is_none() {
                return Ok(0);
            }
            resolve_undefined(inputs, globals, common_addr, sec_vaddr, sym.name)
        }
        _ => Ok(0),
    }
}

/// `section_base + offset` for a section-relative symbol.
fn section_relative_addr(
    sec_vaddr: &[Vec<u64>],
    file: usize,
    n_sect: u8,
    offset: u64,
) -> Result<u64> {
    let zero_based = usize::from(n_sect).saturating_sub(1);
    let base = sec_vaddr
        .get(file)
        .and_then(|f| f.get(zero_based))
        .copied()
        .ok_or(Error::OutOfRange("symbol section ordinal"))?;
    Ok(base.wrapping_add(offset))
}

/// Resolves an undefined reference through the global table to its defining
/// file, then resolves that definition's address. A name with no global
/// definition is an unresolved reference in a static link.
fn resolve_undefined(
    inputs: &[MachOFile<'_>],
    globals: &Globals<'_>,
    common_addr: &FxHashMap<&[u8], u64>,
    sec_vaddr: &[Vec<u64>],
    name: &[u8],
) -> Result<u64> {
    let Some(def) = globals.get(name) else {
        return Err(Error::UndefinedReference(show(name)));
    };
    let usize_sym = usize::try_from(def.sym)
        .map_err(|_| Error::OutOfRange("global sym idx"))?;
    symbol_addr(inputs, globals, common_addr, sec_vaddr, def.file, usize_sym)
}
