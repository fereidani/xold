//! Answering the plugin: which definition won, and who can see it.
//!
//! The plugin has declared what every bitcode file defines and references.
//! What it cannot know is the rest of the link, and that is the whole of what
//! this module supplies: for each name, whether this file's definition is the
//! one that survives, and whether anything outside the bitcode can reach it.
//!
//! The second question is the one that has to be answered conservatively.
//! `used_in_regular_obj` is set by a *regular* object naming a symbol --
//! defining it or referencing it -- and by the names the command line pins
//! (`-u`, the entry symbol). A bitcode reference does not count, which is the
//! point: a name only bitcode mentions is one LTO may internalise or delete.
//! Report it invisible when a real object calls it and the definition is
//! quietly removed from under that call.

use rustc_hash::FxHashMap;

use crate::{
    elf::{
        ObjectFile,
        constants::{SHN_COMMON, SHN_UNDEF, STB_WEAK},
    },
    error::{Error, Result},
    input::{Format, InputFile},
    lto::{
        api::{
            LDPK_COMMON, LDPK_DEF, LDPK_WEAKDEF, LDPK_WEAKUNDEF, LDPV_DEFAULT,
            LDPV_PROTECTED,
        },
        resolve::{Facts, Winner, resolution},
        session,
    },
};

/// Names LTO may call without any input having referenced them.
///
/// The optimiser lowers loops and struct copies into calls to these, so a
/// definition can be needed by code that does not exist until codegen has
/// run. lld asks LLVM for the list (`lto::LTO::getRuntimeLibcallSymbols`);
/// that is a C++ entry point no plugin exposes, so this is the subset a
/// freestanding target actually gets lowered to. Naming one an archive does
/// not define costs nothing -- the lookup simply misses.
pub const RUNTIME_LIBCALLS: [&[u8]; 4] =
    [b"memcpy", b"memmove", b"memset", b"memcmp"];

/// What the inputs that are not bitcode contribute to one name.
#[derive(Clone, Copy, Default)]
struct Use {
    /// A regular object names it, or the command line pinned it. This is
    /// lld's `isUsedInRegularObj`.
    used: bool,
    /// The rank of the strongest regular definition of this name, in the
    /// order [`rank`] puts bitcode definitions in so the two compare
    /// directly, or `None` when no regular object defines it.
    ///
    /// The strength matters: a *weak* regular definition loses to a strong
    /// bitcode one, and treating any definition at all as blocking told the
    /// plugin its strong definition had been preempted, which lets LLVM drop
    /// the body the image was supposed to keep.
    def_rank: Option<u8>,
    /// A shared library defines it, and nothing else does.
    shared: bool,
}

/// Every name the non-bitcode half of the link knows about.
#[derive(Default)]
pub struct Regular {
    names: FxHashMap<Vec<u8>, Use>,
}

impl Regular {
    /// Records the names a regular input contributes.
    ///
    /// Only relocatable objects and shared libraries are read. An archive
    /// contributes nothing until a member is extracted, which matches what
    /// lld does: an unextracted member is not part of the link, so a name it
    /// would define is not yet visible to one.
    ///
    /// # Errors
    ///
    /// Reports an input that identifies as ELF and then does not parse.
    pub fn add_object(&mut self, bytes: &[u8]) -> Result<()> {
        if Format::detect(bytes) != Some(Format::Elf) {
            return Ok(());
        }
        let obj = ObjectFile::parse(bytes)?;
        if obj.is_relocatable() {
            return self.add_relocatable(bytes);
        }
        for name in exported_names(&obj) {
            self.entry(&name).shared = true;
        }
        Ok(())
    }

    /// Records the globals of one relocatable object.
    fn add_relocatable(&mut self, bytes: &[u8]) -> Result<()> {
        // Reading through `InputFile` rather than the raw object keeps this
        // on the same extraction the linker itself uses, so a symbol counted
        // here is one the link will see.
        let file =
            InputFile::from_member(std::path::Path::new("(lto)"), bytes)?;
        for sym in file.global_symbols()? {
            let rank = regular_rank(sym.binding, sym.shndx);
            let slot = self.entry(sym.name);
            slot.used = true;
            if sym.shndx != SHN_UNDEF {
                slot.def_rank =
                    Some(slot.def_rank.map_or(rank, |best| best.min(rank)));
            }
        }
        Ok(())
    }

    /// Pins a name the command line named: `-u`, `--entry`, and the reserved
    /// symbols the linker itself defines. lld marks these the same way, and
    /// for the same reason -- nothing in any object references them, yet the
    /// image needs them to survive.
    pub fn pin(&mut self, name: &[u8]) {
        self.entry(name).used = true;
    }

    fn entry(&mut self, name: &[u8]) -> &mut Use {
        self.names.entry(name.to_vec()).or_default()
    }

    fn get(&self, name: &[u8]) -> Use {
        self.names.get(name).copied().unwrap_or_default()
    }

    /// The names something references and nothing yet defines.
    ///
    /// A pinned name counts: `-u` and the entry symbol are references the
    /// command line made, and an archive member defining one joins the link
    /// exactly as it would for a reference in an object. Weak references do
    /// not, which is the rule everywhere else -- a weak undefined resolves to
    /// zero rather than pulling a member in.
    ///
    /// # Errors
    ///
    /// Reports a poisoned session lock.
    pub fn undefined(&self) -> Result<Vec<Vec<u8>>> {
        let state = session::session()
            .lock()
            .map_err(|_| Error::Format("LTO session lock was poisoned"))?;
        let mut out: FxHashMap<Vec<u8>, bool> = FxHashMap::default();
        for (name, seen) in &self.names {
            if seen.used {
                out.insert(name.clone(), seen.def_rank.is_some());
            }
        }
        for file in &state.claimed {
            for sym in &file.symbols {
                if is_definition(sym.kind) {
                    out.insert(sym.name.clone(), true);
                } else if sym.kind != LDPK_WEAKUNDEF {
                    out.entry(sym.name.clone()).or_insert(false);
                }
            }
        }
        Ok(out
            .into_iter()
            .filter_map(|(name, defined)| (!defined).then_some(name))
            .collect())
    }
}

/// The exported names of a shared object.
fn exported_names(obj: &ObjectFile<'_>) -> Vec<Vec<u8>> {
    let Ok(Some(dynsym)) = obj.dynamic_symbols() else {
        return Vec::new();
    };
    dynsym
        .iter()
        .filter(|sym| sym.st_shndx.get() != SHN_UNDEF)
        .map(|sym| dynsym.name(sym).to_vec())
        .collect()
}

/// How the linker ranks a definition the plugin declared.
///
/// The order is the link's own (`symbol::Class::rank`): a strong definition
/// outranks a tentative one, which outranks a weak one. Answering the plugin
/// with a different order than the link uses would have it compile the copy
/// the image does not keep.
const fn rank(kind: std::ffi::c_uint) -> u8 {
    match kind {
        LDPK_DEF => 1,
        LDPK_COMMON => 2,
        LDPK_WEAKDEF => 3,
        LDPK_WEAKUNDEF => 5,
        // `LDPK_UNDEF` and anything unrecognised: a reference, not a
        // definition.
        _ => 4,
    }
}

/// The rank of a regular ELF definition, in the same order [`rank`] puts
/// bitcode definitions in so that the two can be compared directly.
const fn regular_rank(binding: u8, shndx: u16) -> u8 {
    if shndx == SHN_COMMON {
        2
    } else if binding == STB_WEAK {
        3
    } else {
        1
    }
}

/// Whether a declared symbol is a definition rather than a reference.
const fn is_definition(kind: std::ffi::c_uint) -> bool {
    matches!(kind, LDPK_DEF | LDPK_WEAKDEF | LDPK_COMMON)
}

/// Computes every claimed symbol's resolution and records it for
/// `get_symbols` to hand back.
///
/// `export_all` is whether the link publishes its definitions -- a shared
/// object, or an executable built with `--export-dynamic`. It decides whether
/// a name only bitcode uses may still be interposed at load time, which is
/// what keeps LTO from assuming it has seen every reference.
///
/// # Errors
///
/// Reports a poisoned session lock, which means a callback panicked earlier.
pub fn resolve_claimed(regular: &Regular, export_all: bool) -> Result<()> {
    let mut state = session::session()
        .lock()
        .map_err(|_| Error::Format("LTO session lock was poisoned"))?;

    // Which bitcode file owns each name. Command-line order decides ties, so
    // the first file to reach a rank keeps it.
    let mut winner: FxHashMap<Vec<u8>, (u8, usize)> = FxHashMap::default();
    for file in &state.claimed {
        for sym in &file.symbols {
            if !is_definition(sym.kind) {
                continue;
            }
            let mine = rank(sym.kind);
            match winner.get(&sym.name) {
                Some((best, _)) if *best <= mine => {}
                _ => {
                    winner.insert(sym.name.clone(), (mine, file.handle));
                }
            }
        }
    }

    for file in &mut state.claimed {
        for sym in &mut file.symbols {
            let seen = regular.get(&sym.name);
            let defined = is_definition(sym.kind);
            let owner = winner.get(&sym.name).map(|(_, handle)| *handle);
            // A regular definition only displaces the bitcode one when it
            // would win the ordinary resolution. Equal ranks go to the
            // regular object, which was added to the link first. lld reaches
            // the same answer from the other side: it derives `Prevailing`
            // from whichever symbol the resolver already chose
            // (`lld/ELF/LTO.cpp`).
            let outranked =
                seen.def_rank.is_some_and(|best| best <= rank(sym.kind));
            let where_ = if outranked {
                Winner::Regular
            } else if owner.is_some() {
                Winner::Bitcode
            } else if seen.shared {
                Winner::Shared
            } else {
                Winner::Nowhere
            };
            let facts = Facts {
                defined,
                prevailing: defined && !outranked && owner == Some(file.handle),
                winner: where_,
                used_in_regular_obj: seen.used,
                // Protected is exported just as default is: it cannot be
                // interposed, but it is still in `.dynsym` and still
                // reachable from outside. lld gates this on the binding not
                // being local rather than on default visibility
                // (`lld/ELF/LTO.cpp`), and calling a protected definition
                // invisible lets LTO treat a real dynamic export as internal.
                exported: export_all
                    && matches!(sym.visibility, LDPV_DEFAULT | LDPV_PROTECTED),
            };
            sym.resolution = resolution(&facts).cast_signed();
        }
    }
    Ok(())
}
