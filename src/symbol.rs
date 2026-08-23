//! The resolved symbol model and the global symbol table.
//!
//! Symbols are folded from every input following ELF precedence: a strong
//! definition outranks a tentative (`common`) symbol, which outranks a weak
//! definition, which outranks a bare reference. Two strong definitions of the
//! same name are a duplicate-symbol error. See [`Class::rank`], which is where
//! that order is stated and where the tentative-versus-weak case is argued.
//!
//! The table is folded serially, in input order, so symbol precedence and the
//! resulting ids are deterministic. The parallel passes read it by index
//! through the per-file `sym_id` cache rather than probing it by name.

use std::hash::{BuildHasher, Hasher};

use hashbrown::HashTable;
use rustc_hash::FxBuildHasher;

mod bulk;

use crate::{
    elf::constants::{
        STB_GNU_UNIQUE, STB_WEAK, STT_NOTYPE, STT_TLS, STV_DEFAULT, STV_HIDDEN,
        STV_PROTECTED,
    },
    error::{Error, Result},
    input::InputSymbol,
};

/// Newtype index into the symbol arena. Always valid for an entry that exists.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SymbolId(pub usize);

/// A coarse precedence class used to order competing definitions.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum Class {
    DefinedStrong,
    DefinedWeak,
    Common,
    UndefStrong,
    UndefWeak,
}

impl Class {
    /// Lower rank wins: a symbol of this class replaces one ranked higher.
    ///
    /// A tentative definition outranks a *weak* one, which is the one place
    /// this order is not self-evident. lld spells the same precedence across
    /// two functions: `Symbol::shouldReplace` refuses to let an incoming weak
    /// definition displace a common (`if (isCommon()) return !other.isWeak();`,
    /// `lld/ELF/Symbols.cpp`), and `Symbol::resolve(const
    /// CommonSymbol &)` lets a common displace an existing definition when it
    /// is weak (`if (isDefined() && !isWeak()) return;`, `:579`). GNU ld
    /// agrees; mold does not, and prefers the weak definition.
    ///
    /// Checked against the three on this host rather than read off their
    /// sources alone: a translation unit with `int x;` and another with
    /// `__attribute__((weak)) int x = 7;` links to `x == 0` under both `ld.bfd`
    /// and `ld.lld`, in either input order, and to `x == 7` under `mold`. Two
    /// of the three, including the one this project reads against, give the
    /// storage to the tentative definition.
    const fn rank(self) -> u8 {
        match self {
            Self::DefinedStrong => 1,
            Self::Common => 2,
            Self::DefinedWeak => 3,
            Self::UndefStrong => 4,
            Self::UndefWeak => 5,
        }
    }
}

/// Where a defined symbol's bytes live.
#[derive(Clone, Debug)]
pub enum DefSource {
    /// Inside an input section.
    Section { index: u16, offset: u64 },
    /// An absolute value, not associated with any section.
    Absolute { value: u64 },
}

/// A resolved definition produced by folding one or more inputs.
#[derive(Clone, Debug)]
pub struct Definition {
    pub weak: bool,
    /// The definition arrived as `STB_GNU_UNIQUE`. It ranks with the weak
    /// ones ([`ranks_weak`]) but the binding itself survives into the output,
    /// because the loader keys its one-copy search on it.
    pub unique: bool,
    pub file: usize,
    pub source: DefSource,
    pub size: u64,
    pub sym_type: u8,
}

/// The outcome of resolving one name across all inputs.
#[derive(Clone, Debug)]
pub enum SymbolKind {
    /// A reference with no matching definition.
    Undefined { weak: bool },
    /// Backed by an input section or an absolute value.
    Defined(Definition),
    /// Tentative definition; storage is allocated lazily in `.bss`.
    Common { weak: bool, size: u64, align: u64 },
}

impl SymbolKind {
    /// The precedence class of this resolved kind.
    fn class(&self) -> Class {
        match self {
            Self::Defined(d) if d.weak || d.unique => Class::DefinedWeak,
            Self::Defined(_) => Class::DefinedStrong,
            Self::Common { .. } => Class::Common,
            Self::Undefined { weak: true } => Class::UndefWeak,
            Self::Undefined { .. } => Class::UndefStrong,
        }
    }

    /// Builds the initial resolved kind from a single input symbol.
    fn from_input(inc: &InputSymbol, file: usize) -> Self {
        if inc.is_undefined() {
            return Self::Undefined {
                weak: inc.binding == STB_WEAK,
            };
        }
        if inc.is_common() {
            // For common symbols `st_value` carries the required alignment.
            return Self::Common {
                weak: inc.binding == STB_WEAK,
                size: inc.size,
                align: inc.value,
            };
        }
        let source = if inc.is_absolute() {
            DefSource::Absolute { value: inc.value }
        } else {
            DefSource::Section {
                index: inc.shndx,
                offset: inc.value,
            }
        };
        Self::Defined(Definition {
            weak: inc.binding == STB_WEAK,
            unique: inc.binding == STB_GNU_UNIQUE,
            file,
            source,
            size: inc.size,
            sym_type: inc.type_,
        })
    }
}

/// The precedence class of an incoming input symbol.
fn input_class(inc: &InputSymbol) -> Class {
    if inc.is_undefined() {
        // An undefined `STB_GNU_UNIQUE` is deliberately not weak: it demands
        // a definition exactly as a strong reference does, so an archive
        // member carrying one is still pulled in.
        if inc.binding == STB_WEAK {
            Class::UndefWeak
        } else {
            Class::UndefStrong
        }
    } else if inc.is_common() {
        Class::Common
    } else if ranks_weak(inc.binding) {
        Class::DefinedWeak
    } else {
        Class::DefinedStrong
    }
}

/// Whether a definition with this binding ranks with the weak ones.
///
/// `-fgnu-unique` raises the binding of a COMDAT-held static from
/// `STB_WEAK` to `STB_GNU_UNIQUE`, and lld's `shouldReplace` treats the two
/// alike so the first among the weak and unique copies wins -- preferring an
/// incoming unique to an existing weak could pick a copy living in a
/// discarded COMDAT section (`lld/ELF/Symbols.cpp`). An
/// incoming `STB_GLOBAL` overrides both.
fn ranks_weak(binding: u8) -> bool {
    binding == STB_WEAK || binding == STB_GNU_UNIQUE
}

/// A resolved symbol. Its name lives in the table key and is not duplicated
/// here.
#[derive(Clone, Debug)]
pub struct Symbol {
    pub kind: SymbolKind,
    /// The merged visibility (`STV_*`) of every occurrence of the name, not
    /// just of the definition that won. See [`merge_visibility`].
    pub visibility: u8,
}

impl Symbol {
    /// Whether a definition of this symbol can be replaced at load time by one
    /// of the same name from another image. `shared` is whether this link
    /// produces a shared object.
    ///
    /// Mirrors lld's `computeIsPreemptible`. The clauses lld gates on
    /// `-Bsymbolic` and `--dynamic-list` are absent because xold implements
    /// neither, and with neither in force lld's answer for a definition in a
    /// shared object is an unconditional `true`.
    pub fn is_preemptible(&self, shared: bool) -> bool {
        // Anything but default visibility is a promise that no other image
        // supplies this definition, whatever the link mode.
        if self.visibility != STV_DEFAULT {
            return false;
        }
        // A reference this link has no definition for is resolved by the
        // loader, so which image satisfies it is not fixed here. A tentative
        // definition counts as a definition: it becomes this image's `.bss`
        // storage.
        if !matches!(
            self.kind,
            SymbolKind::Defined(_) | SymbolKind::Common { .. }
        ) {
            return true;
        }
        // Every definition an executable carries is final: nothing loaded
        // afterwards can take its place.
        shared
    }

    /// Whether this symbol's value is an absolute constant rather than a
    /// relocatable address. A TLS symbol's `st_value` is a thread-pointer
    /// offset, an absolute definition carries a literal value, and an
    /// undefined reference contributes `0`.
    pub fn is_absolute_value(&self) -> bool {
        match &self.kind {
            SymbolKind::Defined(def) => {
                def.sym_type == STT_TLS
                    || matches!(def.source, DefSource::Absolute { .. })
            }
            SymbolKind::Common { .. } => false,
            SymbolKind::Undefined { .. } => true,
        }
    }

    /// The resolved symbol a single input occurrence stands for on its own.
    fn from_input(inc: &InputSymbol, file: usize) -> Self {
        Self {
            kind: SymbolKind::from_input(inc, file),
            visibility: inc.visibility,
        }
    }
}

/// Whether a definition with visibility `vis` is part of the image's ABI and
/// so belongs in `.dynsym`.
///
/// Hidden and internal definitions are private to the image: keeping them out
/// is what makes `-fvisibility=hidden` mean anything. A protected definition
/// is exported like a default one; it merely cannot be preempted.
pub const fn exports_to_dynsym(vis: u8) -> bool {
    matches!(vis, STV_DEFAULT | STV_PROTECTED)
}

/// Folds an occurrence's visibility into the one resolved so far.
///
/// lld's rule (`Symbol::mergeProperties`): the most constrained visibility any
/// occurrence carries wins, including one carried by an undefined reference.
/// `STV_DEFAULT` is numerically the smallest yet the *least* constrained, so it
/// is special-cased rather than folded into the minimum.
///
/// The result is the minimum over the non-default values, which is commutative
/// and associative: the answer does not depend on the order the occurrences are
/// folded in, and so does not depend on how the work was scheduled.
const fn merge_visibility(current: u8, incoming: u8) -> u8 {
    if incoming == STV_DEFAULT {
        return current;
    }
    if current == STV_DEFAULT {
        return incoming;
    }
    if incoming < current {
        incoming
    } else {
        current
    }
}

/// Independent name indexes the table is split across, selected by the low
/// bits of a name's hash.
///
/// A name belongs to exactly one shard, so the bulk fold can give each shard
/// to a different thread and they never contend. The count is a power of two
/// so the selection is a mask.
pub(crate) const SHARDS: usize = 64;

/// The global name-to-symbol map, built by folding inputs into it.
///
/// Names stay where the inputs spell them: the table holds one borrowed
/// slice per id, in id (first-seen input) order, and the resolved symbols
/// live in a dense arena beside them. Copying every name into an owned
/// arena instead was tens of megabytes moved on a large link, for bytes the
/// mappings already hold for the link's whole life.
pub struct SymbolTable<'data> {
    /// Per id: the interned name. See [`NameRef`] for why two forms exist.
    names: Vec<NameRef<'data>>,
    /// Name -> id, split into [`SHARDS`] independent tables. Each entry is
    /// just the id: the name it stands for is `names[id]`, so an entry is a
    /// few bytes rather than an owned `Vec<u8>`. On a link with a million
    /// symbols that saves a million allocations, and the split is what lets
    /// [`Self::intern_all`] fold the inputs in parallel.
    index: Vec<HashTable<SymbolId>>,
    symbols: Vec<Symbol>,
}

impl Default for SymbolTable<'_> {
    fn default() -> Self {
        Self {
            names: Vec::new(),
            index: (0..SHARDS).map(|_| HashTable::new()).collect(),
            symbols: Vec::new(),
        }
    }
}

/// The shard a name hash belongs to. Truncation is the point: the shard is
/// selected by the hash's low bits.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn shard_of(hash: u64) -> usize {
    (hash as usize) & (SHARDS - 1)
}

/// The hash a name is filed under in [`SymbolTable::index`].
///
/// Public so extraction can compute it on the parallel parse pass and hand it
/// to [`SymbolTable::intern`] through [`InputSymbol::name_hash`].
/// Partitions one file's extracted globals by shard for the bulk fold; see
/// `bulk`. Exposed so the parallel parse pass can partition each file's
/// records while they are still warm from extraction.
pub(crate) fn partition_file(
    syms: &mut Vec<InputSymbol<'_>>,
) -> Vec<(u32, u32)> {
    bulk::partition_file(syms)
}

pub fn name_hash(name: &[u8]) -> u64 {
    let mut h = FxBuildHasher.build_hasher();
    h.write(name);
    h.finish()
}

/// Strips a version suffix off a symbol name: `foo@@VERS` and `foo@VERS`
/// both resolve as `foo`.
///
/// The assembler emits these spellings for `.symver` aliases and annotated
/// references, and the question they raise is which name a reference asks
/// for. lld answers that a version suffix is not part of the name: its table
/// keys a `foo@@VERS` definition under `foo` (`SymbolTable::insert` stems at
/// the first `@@`), it registers a dependency's exports under both `foo` and
/// `foo@VERS` (`SharedFile::parse`), and its output tables carry the bare
/// name (`Symbol::parseSymbolVersion` truncates at the `@`). Every spelling
/// of one symbol is therefore one symbol: a plain reference binds the
/// `foo@@VERS` definition, and a `foo@VERS` reference binds the dependency's
/// `foo`.
///
/// The suffix is dropped whole, which folds the spellings into the one name
/// this table has per symbol. The distinction lld can still draw -- which
/// version a `foo@VER` reference asked for, against a dependency offering
/// several -- is answered here from the dependency's own row for the name
/// (see `linker::deps`), since the table has no second slot to keep the
/// versioned spelling in.
///
/// A name whose stem would be empty (`@foo`) is a spelling of nothing and
/// keeps its bytes.
pub fn version_stem(name: &[u8]) -> &[u8] {
    match name.iter().position(|&b| b == b'@') {
        Some(0) | None => name,
        Some(pos) => &name[..pos],
    }
}

/// One interned name.
///
/// A direct input's mapping outlives the whole link, so the bulk fold
/// borrows the name straight from it -- copying every name into an owned
/// arena was tens of megabytes moved for bytes already held. An archive
/// member's bytes are owned by its `InputFile`, which the borrow cannot
/// name without tying the table to the file list it sits beside, so the
/// incremental intern path copies. Archives contribute thousands of names
/// where direct objects contribute a million, so the copy is the cold
/// side.
enum NameRef<'data> {
    /// Borrowed from a direct input's mapping.
    Slice(&'data [u8]),
    /// Copied out of an archive member (or any late-added file).
    Owned(Box<[u8]>),
}

impl NameRef<'_> {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Slice(s) => s,
            Self::Owned(o) => o,
        }
    }
}

/// The interned bytes of `id`. Free function so callers can hold it
/// alongside a mutable borrow of the index.
fn arena_name<'a>(names: &'a [NameRef<'_>], id: SymbolId) -> &'a [u8] {
    names.get(id.0).map(NameRef::bytes).unwrap_or_default()
}

impl<'data> SymbolTable<'data> {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserves room for `count` further symbols across every table.
    ///
    /// The name map is the hot structure in symbol resolution and grows to one
    /// entry per distinct name; sizing it once up front avoids the repeated
    /// rehash-and-reinsert of a table that doubles a couple of dozen times on
    /// a large link. `count` is an upper bound (the input symbol total), so the
    /// tables are never undersized.
    pub fn reserve(&mut self, count: usize) {
        let Self {
            names,
            index,
            symbols,
        } = self;
        let per_shard = count.div_ceil(SHARDS);
        for shard in index.iter_mut() {
            shard.reserve(per_shard, |&id| name_hash(arena_name(names, id)));
        }
        symbols.reserve(count);
        names.reserve(count);
    }

    /// Folds the global symbols of every input into the table in one pass,
    /// returning each file's `symbol index -> id` row.
    ///
    /// Equivalent to calling [`Self::intern`] for every symbol of every file
    /// in input order -- the ids and the precedence outcome are identical --
    /// but the deduplication runs across every available thread. See
    /// [`bulk`] for how the ordering is preserved. The table must be empty.
    /// `bins` carries each file's shard partition (see
    /// [`partition_file`]); every file must already be partitioned.
    pub fn intern_all(
        &mut self,
        files: &[Vec<InputSymbol<'data>>],
        bins: &[Vec<(u32, u32)>],
    ) -> Result<Vec<Vec<Option<SymbolId>>>> {
        bulk::intern_all(self, files, bins)
    }

    /// The number of resolved symbols (one per interned name).
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Whether no symbols have been interned.
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    /// Folds `inc` into the table, applying ELF resolution rules, and returns
    /// the resolved symbol's id. The caller records `(file, sym_idx) -> id`
    /// from this result so the layout passes never need to re-probe by name.
    pub fn intern(
        &mut self,
        inc: &InputSymbol<'_>,
        file: usize,
    ) -> Result<SymbolId> {
        let hash = inc.name_hash;
        let Self {
            names,
            index,
            symbols,
        } = self;
        let Some(shard) = index.get_mut(shard_of(hash)) else {
            return Err(Error::OutOfRange("symbol shard"));
        };
        if let Some(&id) =
            shard.find(hash, |&other| arena_name(names, other) == inc.name)
        {
            let sym = symbols
                .get_mut(id.0)
                .ok_or(Error::OutOfRange("symbol arena index"))?;
            merge_symbol(sym, inc, file)?;
            return Ok(id);
        }
        let id = SymbolId(symbols.len());
        // Copied, not borrowed: this path interns archive members, whose
        // bytes the table cannot outlive-borrow. See [`NameRef`].
        names.push(NameRef::Owned(inc.name.into()));
        shard.insert_unique(hash, id, |&other| {
            name_hash(arena_name(names, other))
        });
        symbols.push(Symbol::from_input(inc, file));
        Ok(id)
    }

    /// The interned name of a resolved symbol, or an empty slice for an
    /// out-of-range id.
    pub fn name(&self, id: SymbolId) -> &[u8] {
        arena_name(&self.names, id)
    }

    /// Every resolved id in first-seen input order.
    ///
    /// Passes that visit all symbols use this rather than iterating the name
    /// map: the ids are dense, so the symbol arena and any parallel per-id
    /// table are read sequentially, and the order does not depend on how the
    /// names happened to hash.
    pub fn ids(&self) -> impl Iterator<Item = SymbolId> {
        (0..self.symbols.len()).map(SymbolId)
    }

    /// Resolved entries in id (first-seen input) order.
    pub fn entries(&self) -> impl Iterator<Item = (&[u8], &Symbol)> {
        self.ids()
            .filter_map(move |id| Some((self.name(id), self.symbol(id)?)))
    }

    /// The id of a resolved symbol by name, if it was interned.
    pub fn find(&self, name: &[u8]) -> Option<SymbolId> {
        let hash = name_hash(name);
        self.index
            .get(shard_of(hash))?
            .find(hash, |&id| arena_name(&self.names, id) == name)
            .copied()
    }

    /// The resolved symbol for `id`.
    pub fn symbol(&self, id: SymbolId) -> Option<&Symbol> {
        self.symbols.get(id.0)
    }

    /// Hides a resolved symbol, as a version script's `local:` list asks.
    ///
    /// Visibility is what decides whether a definition reaches `.dynsym`
    /// ([`exports_to_dynsym`]) and whether another image may preempt it, so
    /// setting it here is what makes the script's answer the one every later
    /// pass reads. It is applied after resolution and before layout, so no
    /// sizing or emission pass sees the two disagree.
    pub fn hide(&mut self, id: SymbolId) {
        if let Some(sym) = self.symbols.get_mut(id.0) {
            sym.visibility = STV_HIDDEN;
        }
    }

    /// Replaces an undefined reference with an absolute definition the linker
    /// itself supplies, reporting whether it did.
    ///
    /// A name an input defines keeps that definition: this is how `PROVIDE`
    /// behaves, and the linker's own value for a name like `_end` is only
    /// correct in the absence of a real one. The value is filled in later,
    /// once the layout knows where the bound it names landed.
    ///
    /// `visibility` is folded in through [`merge_visibility`] rather than
    /// stamped over what the references carried, so the most constrained
    /// occurrence still wins. That is lld's rule too: its `addOptionalRegular`
    /// hands a visibility to `Symbol::resolve`, which merges it in
    /// `mergeProperties` exactly as it does for an input's.
    pub fn define_absolute(
        &mut self,
        id: SymbolId,
        sym_type: u8,
        visibility: u8,
    ) -> bool {
        let Some(sym) = self.symbols.get_mut(id.0) else {
            return false;
        };
        if !matches!(sym.kind, SymbolKind::Undefined { .. }) {
            return false;
        }
        sym.visibility = merge_visibility(sym.visibility, visibility);
        sym.kind = SymbolKind::Defined(Definition {
            weak: false,
            unique: false,
            file: 0,
            source: DefSource::Absolute { value: 0 },
            size: 0,
            sym_type,
        });
        true
    }
}

/// Folds an incoming occurrence into an existing resolved symbol.
///
/// Visibility is merged for every occurrence, whether or not it wins the
/// definition contest: a hidden reference constrains the name even when the
/// definition comes from elsewhere.
fn merge_symbol(
    existing: &mut Symbol,
    inc: &InputSymbol,
    file: usize,
) -> Result<()> {
    existing.visibility = merge_visibility(existing.visibility, inc.visibility);
    check_tls_mismatch(existing, inc)?;
    merge_kind(&mut existing.kind, inc, file)
}

/// Refuses to fold a plain occurrence into a resolved thread-local.
///
/// `st_value` of an `STT_TLS` symbol is an offset from the thread pointer,
/// not an address, so a plain reference to the name would resolve against a
/// number the code never meant. `STT_NOTYPE` is allowed through: assembler
/// references routinely carry no type. lld reports the same mismatch while
/// folding (`lld/ELF/InputFiles.cpp`).
fn check_tls_mismatch(existing: &Symbol, inc: &InputSymbol) -> Result<()> {
    let tls = matches!(
        &existing.kind,
        SymbolKind::Defined(def) if def.sym_type == STT_TLS
    );
    if tls && inc.type_ != STT_TLS && inc.type_ != STT_NOTYPE {
        return Err(Error::TlsMismatch(
            String::from_utf8_lossy(inc.name).into_owned(),
        ));
    }
    Ok(())
}

/// Folds an incoming symbol into an existing resolved kind.
fn merge_kind(
    existing: &mut SymbolKind,
    inc: &InputSymbol,
    file: usize,
) -> Result<()> {
    let prev = existing.class();
    let next = input_class(inc);
    match (prev, next) {
        // One file spelling a name twice at one place has aliased it, not
        // collided with itself: `.symver` emits the plain and the versioned
        // spelling of one symbol as two rows of one object, which
        // canonicalize ([`crate::input`]'s stemming) to the same name. The
        // first row stands, exactly as in lld, whose duplicate check accepts
        // a second definition from the file that already defined the name
        // (`ObjFile::postParse`). Rows at distinct places are two different
        // definitions folded under one stem -- `foo@V1` beside `foo@@V2`,
        // the compat-symbol pattern -- which this linker cannot keep apart
        // (it emits no VERDEF of its own), so they fall through to the
        // duplicate report instead of silently dropping one implementation.
        (Class::DefinedStrong, Class::DefinedStrong)
            if defining_file(existing) == file && same_place(existing, inc) =>
        {
            Ok(())
        }
        // Two absolute definitions that state the same value are not in
        // conflict: whichever row wins, every reference resolves to that
        // number. lld carves this exact case out of its duplicate report
        // for GNU ld compatibility, because assembler sources routinely
        // stamp the same constant twice
        // (`lld/ELF/Symbols.cpp`). Disagreeing values fall
        // through to the error below.
        (Class::DefinedStrong, Class::DefinedStrong)
            if same_absolute_value(existing, inc) =>
        {
            Ok(())
        }
        (Class::DefinedStrong, Class::DefinedStrong) => {
            // Both definers are in hand: the one already recorded and the one
            // being folded in. They are carried as indices and rendered as
            // paths by `intern_all`, which is the first place that knows what
            // the caller named each input.
            Err(Error::duplicate_symbol(
                String::from_utf8_lossy(inc.name).into_owned(),
                defining_file(existing).to_string(),
                file.to_string(),
            ))
        }
        (Class::Common, Class::Common) => {
            merge_common(existing, inc);
            Ok(())
        }
        (_, _) if next.rank() < prev.rank() => {
            *existing = SymbolKind::from_input(inc, file);
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Merges two tentative definitions: keep the larger size and alignment.
fn merge_common(existing: &mut SymbolKind, inc: &InputSymbol) {
    if let SymbolKind::Common { size, align, .. } = existing {
        if inc.size > *size {
            *size = inc.size;
        }
        // `st_value` is the alignment for common symbols.
        if inc.value > *align {
            *align = inc.value;
        }
    }
}

/// The index of the file a recorded definition came from, or `usize::MAX`
/// when the kind carries none -- which a strong definition always does, so the
/// fallback never renders.
fn defining_file(kind: &SymbolKind) -> usize {
    match kind {
        SymbolKind::Defined(def) => def.file,
        SymbolKind::Common { .. } | SymbolKind::Undefined { .. } => usize::MAX,
    }
}

/// Whether an incoming definition names the very place an existing one
/// already records: the alias case, where two spellings are one symbol.
fn same_place(existing: &SymbolKind, inc: &InputSymbol) -> bool {
    let SymbolKind::Defined(def) = existing else {
        return false;
    };
    match def.source {
        DefSource::Section { index, offset } => {
            !inc.is_absolute() && index == inc.shndx && offset == inc.value
        }
        DefSource::Absolute { value } => {
            inc.is_absolute() && value == inc.value
        }
    }
}

/// Whether both definitions are absolute and state the same value.
fn same_absolute_value(existing: &SymbolKind, inc: &InputSymbol) -> bool {
    let SymbolKind::Defined(Definition {
        source: DefSource::Absolute { value },
        ..
    }) = existing
    else {
        return false;
    };
    inc.is_absolute() && inc.value == *value
}
