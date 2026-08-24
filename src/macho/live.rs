//! Section reachability for ld64's `-dead_strip`.
//!
//! Mach-O object relocations form the same graph that ELF `--gc-sections`
//! follows: a live section keeps every section its relocations name.  This
//! backend lays out whole input sections rather than ld64 atoms, so the sweep
//! is deliberately section-granular, but it is a real mark/sweep and never a
//! flag-only no-op.

use crate::macho::{
    LinkOptions, MachOFile,
    constants::{
        N_EXT, N_PEXT, N_SECT, N_TYPE, N_UNDF, S_ATTR_NO_DEAD_STRIP,
        S_MOD_INIT_FUNC_POINTERS, S_MOD_TERM_FUNC_POINTERS, SECTION_TYPE,
    },
    layout::is_linkable,
    symtab::Globals,
};

/// Per-input section and symbol liveness, both indexed in input-file order.
pub struct LiveSections {
    sections: Vec<Vec<bool>>,
    symbols: Vec<Vec<bool>>,
}

impl LiveSections {
    pub fn build(
        inputs: &[MachOFile<'_>],
        globals: &Globals<'_>,
        entry: &[u8],
        options: &LinkOptions<'_>,
    ) -> Self {
        let mut live = Self {
            sections: inputs
                .iter()
                .map(|input| vec![false; input.sections().len()])
                .collect(),
            symbols: inputs
                .iter()
                .map(|input| vec![false; input.symbols().len()])
                .collect(),
        };
        if !options.dead_strip {
            for (file, input) in inputs.iter().enumerate() {
                for section in input.sections() {
                    if is_linkable(&section) {
                        live.mark(file, section.index, &mut Vec::new());
                    }
                }
                live.symbols[file].fill(true);
            }
            return live;
        }

        let mut work = Vec::new();
        live.root_name(inputs, globals, entry, &mut work);
        if options.dylib && options.exported_symbols.is_none() {
            for (file, input) in inputs.iter().enumerate() {
                for sym in input.symbols().iter() {
                    if sym.n_type & (N_EXT | N_PEXT) == N_EXT
                        && sym.n_type & N_TYPE == N_SECT
                    {
                        live.mark(file, u32::from(sym.n_sect), &mut work);
                    }
                }
            }
        } else if let Some(exports) = options.exported_symbols {
            for name in exports {
                live.root_name(inputs, globals, name.as_bytes(), &mut work);
            }
        }
        for (file, input) in inputs.iter().enumerate() {
            for section in input.sections() {
                let typ = section.flags & SECTION_TYPE;
                if section.flags & S_ATTR_NO_DEAD_STRIP != 0
                    || matches!(
                        typ,
                        S_MOD_INIT_FUNC_POINTERS | S_MOD_TERM_FUNC_POINTERS
                    )
                {
                    live.mark(file, section.index, &mut work);
                }
            }
        }

        while let Some((file, ordinal)) = work.pop() {
            let Some(input) = inputs.get(file) else {
                continue;
            };
            let Some(section) = input
                .sections()
                .into_iter()
                .find(|section| section.index == ordinal)
            else {
                continue;
            };
            let symbols: Vec<_> = input.symbols().iter().collect();
            for reloc in section.relocations {
                if reloc.r_extern {
                    let sym_idx = reloc.r_symbolnum as usize;
                    if let Some(slot) = live
                        .symbols
                        .get_mut(file)
                        .and_then(|row| row.get_mut(sym_idx))
                    {
                        *slot = true;
                    }
                    let Some(sym) = symbols.get(sym_idx) else {
                        continue;
                    };
                    if sym.n_type & N_TYPE == N_SECT {
                        live.mark(file, u32::from(sym.n_sect), &mut work);
                    } else if sym.n_type & N_TYPE == N_UNDF
                        && let Some(def) = globals.get(sym.name)
                        && !def.common
                        && let Some(target) =
                            inputs.get(def.file).and_then(|input| {
                                input.symbols().nth(def.sym as usize)
                            })
                        && target.n_type & N_TYPE == N_SECT
                    {
                        live.mark(
                            def.file,
                            u32::from(target.n_sect),
                            &mut work,
                        );
                    }
                } else {
                    live.mark(file, reloc.r_symbolnum, &mut work);
                }
            }
        }
        live.keep_unwind_metadata(inputs);
        live
    }

    /// Whether one 1-based input section ordinal survived the sweep.
    pub fn section(&self, file: usize, ordinal: u32) -> bool {
        let index = usize::try_from(ordinal).unwrap_or(0).saturating_sub(1);
        self.sections
            .get(file)
            .and_then(|row| row.get(index))
            .copied()
            .unwrap_or(false)
    }

    /// Whether a live relocation references this input symbol.
    pub fn symbol(&self, file: usize, index: usize) -> bool {
        self.symbols
            .get(file)
            .and_then(|row| row.get(index))
            .copied()
            .unwrap_or(false)
    }

    fn root_name(
        &mut self,
        inputs: &[MachOFile<'_>],
        globals: &Globals<'_>,
        name: &[u8],
        work: &mut Vec<(usize, u32)>,
    ) {
        if let Some(def) = globals.get(name)
            && !def.common
            && let Some(sym) = inputs
                .get(def.file)
                .and_then(|input| input.symbols().nth(def.sym as usize))
            && sym.n_type & N_TYPE == N_SECT
        {
            self.mark(def.file, u32::from(sym.n_sect), work);
            return;
        }
        for (file, input) in inputs.iter().enumerate() {
            for sym in input.symbols().iter() {
                if sym.name == name && sym.n_type & N_TYPE == N_SECT {
                    self.mark(file, u32::from(sym.n_sect), work);
                }
            }
        }
    }

    fn mark(
        &mut self,
        file: usize,
        ordinal: u32,
        work: &mut Vec<(usize, u32)>,
    ) {
        let index = usize::try_from(ordinal).unwrap_or(0).saturating_sub(1);
        let Some(slot) = self
            .sections
            .get_mut(file)
            .and_then(|row| row.get_mut(index))
        else {
            return;
        };
        if !*slot {
            *slot = true;
            work.push((file, ordinal));
        }
    }

    /// Retains an object's unwind payload when any of its code survived.
    /// These sections describe the code graph but must not themselves root
    /// every function they mention: doing that would turn `-dead_strip` into
    /// a no-op for the usual one-`__text` archive members.
    fn keep_unwind_metadata(&mut self, inputs: &[MachOFile<'_>]) {
        for (file, input) in inputs.iter().enumerate() {
            let has_live_code = input.sections().into_iter().any(|section| {
                section.segname == b"__TEXT"
                    && section.sectname != b"__eh_frame"
                    && self.section(file, section.index)
            });
            if !has_live_code {
                continue;
            }
            for section in input.sections() {
                if matches!(
                    section.sectname,
                    b"__eh_frame" | b"__gcc_except_tab"
                ) {
                    let index = usize::try_from(section.index)
                        .unwrap_or(0)
                        .saturating_sub(1);
                    if let Some(slot) = self
                        .sections
                        .get_mut(file)
                        .and_then(|row| row.get_mut(index))
                    {
                        *slot = true;
                    }
                }
                if section.segname == b"__LD"
                    && section.sectname == b"__compact_unwind"
                {
                    for reloc in section.relocations {
                        if reloc.r_extern && reloc.r_address % 32 == 16 {
                            if let Some(slot) =
                                self.symbols.get_mut(file).and_then(|row| {
                                    row.get_mut(reloc.r_symbolnum as usize)
                                })
                            {
                                *slot = true;
                            }
                        }
                    }
                }
            }
        }
    }
}
