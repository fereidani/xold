//! `__start_NAME` / `__stop_NAME`: the bounds a linker supplies for a section
//! whose name is a valid C identifier.
//!
//! A program that keeps a table -- a list of test cases, of module
//! descriptors, of tracepoints -- puts each entry in its own object file with
//! `__attribute__((section("mysec")))` and walks the result between
//! `__start_mysec` and `__stop_mysec`. Nothing in the inputs defines those two
//! names, and nothing gathers the entries: that is the linker's part of the
//! bargain, and it is what Linux-kernel-style tables and Rust crates such as
//! `linkme` and `inventory` are built on.
//!
//! xold has no linker-script-driven section map, so a custom section is not
//! given an output section of its own. It keeps the bargain within the output
//! section its flags already select:
//!
//! - the members named by one C identifier are gathered into a single
//!   contiguous run ([`OutputSections::group_runs`]), so a walk from the first
//!   byte to the last visits those members and nothing else;
//! - [`crate::defsym`] defines the two names at the run's bounds, on the same
//!   `PROVIDE` terms as the other linker-supplied symbols: only a name some
//!   input referenced, and never over an input's own definition;
//! - [`crate::gc`] treats a bounded section as a root, since a run reached only
//!   through its bounds has no relocation to be marked live by.
//!
//! Ordinary C and C++ input emits `.text`, `.rodata`, `.data.rel.ro` and
//! friends, none of which is an identifier (the leading dot disqualifies
//! them), so a link with no custom section never enters any of this.

use rustc_hash::FxHashMap;

use crate::{
    elf::{ObjectFile, Shdr64},
    symbol::SymbolTable,
    util::is_c_identifier,
};

/// The prefix of the symbol naming a run's first byte.
pub const START: &[u8] = b"__start_";
/// The prefix of the symbol naming one past a run's last byte.
pub const STOP: &[u8] = b"__stop_";

/// The placed bounds of one run.
#[derive(Clone, Copy, Default, Debug)]
pub struct Bounds {
    /// The address `__start_NAME` takes: the run's first byte.
    pub start: u64,
    /// The address `__stop_NAME` takes: one past the run's last byte.
    pub stop: u64,
}

/// The C-identifier section names this link bounds, and the input sections
/// contributing to each.
///
/// A run's index in [`Self::names`] is its identity everywhere else: in the
/// [`crate::defsym::DefSym`] entries that name its bounds and in the layout's
/// table of placed bounds.
#[derive(Default)]
pub struct StartStop {
    /// The bounded names, in first-seen symbol order.
    names: Vec<Vec<u8>>,
    /// Name -> run index. Empty on a link that mentions no such name, which
    /// is what makes [`Self::run_of`] free for ordinary input.
    by_name: FxHashMap<Vec<u8>, usize>,
    /// Input section -> the run it contributes to. Only populated for the
    /// sections whose name is bounded.
    of_section: FxHashMap<(usize, u16), usize>,
}

impl StartStop {
    /// Collects the section names this link is asked to bound: every name for
    /// which some input mentions `__start_NAME` or `__stop_NAME`.
    ///
    /// Run in symbol-id order, which is first-seen input order, so the run
    /// indices -- and therefore the order the runs are gathered in -- do not
    /// depend on how the names happened to hash.
    ///
    /// A name is collected whether the encapsulation symbol is referenced or
    /// defined: an input that defines `__start_NAME` for itself keeps that
    /// definition, but the members it names are still gathered, so the two
    /// spellings describe the same bytes.
    pub fn collect(&mut self, symbols: &SymbolTable) {
        for (name, _sym) in symbols.entries() {
            let Some(section) = bounded_name(name) else {
                continue;
            };
            if self.by_name.contains_key(section) {
                continue;
            }
            self.by_name.insert(section.to_vec(), self.names.len());
            self.names.push(section.to_vec());
        }
    }

    /// The bounded names, indexed by run.
    pub fn names(&self) -> &[Vec<u8>] {
        &self.names
    }

    /// The number of runs this link bounds.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether the link bounds no section at all, which is the common case.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// The run an input section contributes to, from its name.
    ///
    /// The leading-dot test comes first: it is one byte, and it rejects
    /// nearly every section a compiler emits without scanning the name to its
    /// terminator.
    pub fn run_of(&self, obj: ObjectFile<'_>, shdr: &Shdr64) -> Option<usize> {
        if self.by_name.is_empty() || obj.section_name_starts_with(shdr, b".") {
            return None;
        }
        self.by_name.get(obj.section_name(shdr)).copied()
    }

    /// Records that `(file, section)` contributes to `run`.
    pub fn record(&mut self, file: usize, section: u16, run: usize) {
        self.of_section.insert((file, section), run);
    }

    /// The run `(file, section)` contributes to, if it contributes to one.
    pub fn run_of_section(&self, file: usize, section: u16) -> Option<usize> {
        if self.of_section.is_empty() {
            return None;
        }
        self.of_section.get(&(file, section)).copied()
    }
}

/// The section name a `__start_`/`__stop_` symbol bounds, or `None` when the
/// name is not one of those.
fn bounded_name(symbol: &[u8]) -> Option<&[u8]> {
    let section = symbol
        .strip_prefix(START)
        .or_else(|| symbol.strip_prefix(STOP))?;
    is_c_identifier(section).then_some(section)
}
