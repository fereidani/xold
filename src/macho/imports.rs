//! Dynamic-import discovery for Mach-O executables.
//!
//! Undefined names supplied by a TAPI dependency become imports. A branch to
//! one receives an arm64 `__stubs` entry; GOT references and stub targets use
//! the shared [`GotPlan`](crate::macho::GotPlan) non-lazy pointer slot that
//! dyld fills from the classic bind opcode stream.

use rustc_hash::FxHashMap;

use crate::{
    error::Result,
    macho::{
        LinkOptions, MachOFile,
        constants::{N_EXT, N_TYPE, N_UNDF},
        layout::is_linkable,
        live::LiveSections,
        reloc::{MachoTarget, scan_needs},
        symtab::Globals,
    },
};

/// One imported symbol. `dylib` is the 1-based load-command ordinal.
#[derive(Clone, Copy)]
pub struct Import<'d> {
    pub name: &'d [u8],
    pub dylib: u32,
    pub stub: Option<u32>,
}

/// Imported names in first-appearance order plus their lookup table.
pub struct ImportPlan<'d> {
    imports: Vec<Import<'d>>,
    index: FxHashMap<&'d [u8], usize>,
    stub_count: u32,
}

impl<'d> ImportPlan<'d> {
    /// Finds every unresolved external named by a dependency, then scans
    /// branch relocations to allocate only the stubs that are actually used.
    pub fn build(
        inputs: &'d [MachOFile<'d>],
        globals: &Globals<'_>,
        target: MachoTarget,
        options: &LinkOptions<'_>,
        live: &LiveSections,
    ) -> Result<Self> {
        let mut plan = Self {
            imports: Vec::new(),
            index: FxHashMap::default(),
            stub_count: 0,
        };
        for (file, input) in inputs.iter().enumerate() {
            for (sym_idx, sym) in input.symbols().iter().enumerate() {
                if sym.is_stab()
                    || sym.n_type & N_EXT == 0
                    || sym.n_type & N_TYPE != N_UNDF
                    || sym.n_value != 0
                    || sym.name.is_empty()
                    || globals.get(sym.name).is_some()
                    || plan.index.contains_key(sym.name)
                    || !live.symbol(file, sym_idx)
                {
                    continue;
                }
                if let Some(dylib) = provider(options, sym.name) {
                    let at = plan.imports.len();
                    plan.index.insert(sym.name, at);
                    plan.imports.push(Import {
                        name: sym.name,
                        dylib,
                        stub: None,
                    });
                }
            }
        }

        for (file, input) in inputs.iter().enumerate() {
            let symbols: Vec<_> = input.symbols().iter().collect();
            for section in input.sections() {
                if !is_linkable(&section) || !live.section(file, section.index)
                {
                    continue;
                }
                for reloc in &section.relocations {
                    if !reloc.r_extern {
                        continue;
                    }
                    let Some(sym) = symbols.get(reloc.r_symbolnum as usize)
                    else {
                        continue;
                    };
                    let Some(&at) = plan.index.get(sym.name) else {
                        continue;
                    };
                    if scan_needs(target, u32::from(reloc.r_type))?.plt
                        && let Some(import) = plan.imports.get_mut(at)
                        && import.stub.is_none()
                    {
                        import.stub = Some(plan.stub_count);
                        plan.stub_count = plan.stub_count.saturating_add(1);
                    }
                }
            }
        }
        Ok(plan)
    }

    pub fn get(&self, name: &[u8]) -> Option<Import<'d>> {
        self.index
            .get(name)
            .and_then(|&at| self.imports.get(at))
            .copied()
    }

    pub const fn stub_count(&self) -> u32 {
        self.stub_count
    }

    pub fn stubs(&self) -> impl Iterator<Item = Import<'d>> + '_ {
        self.imports
            .iter()
            .copied()
            .filter(|entry| entry.stub.is_some())
    }

    pub fn iter(&self) -> impl Iterator<Item = Import<'d>> + '_ {
        self.imports.iter().copied()
    }
}

/// The first dylib exporting `name`, expressed as its Mach-O ordinal.
fn provider(options: &LinkOptions<'_>, name: &[u8]) -> Option<u32> {
    options.dylibs.iter().enumerate().find_map(|(i, dylib)| {
        dylib
            .exports
            .binary_search_by(|symbol| symbol.as_bytes().cmp(name))
            .ok()
            .map(|_| u32::try_from(i + 1).unwrap_or(u32::MAX))
    })
}
