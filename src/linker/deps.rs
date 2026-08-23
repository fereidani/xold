//! What the link reads out of its shared-object dependencies: the soname each
//! one advertises, the symbols it exports, and the versions those symbols
//! carry.
//!
//! A dependency is not linked in; only these tables of it survive into the
//! image. They answer four questions: which name goes in `DT_NEEDED`
//! ([`shared_soname`]), how large and how aligned a copy relocation's slot must
//! be ([`DepExports`]), which `(soname, version)` an import must record in
//! `.gnu.version_r` ([`DepVersions`]), and which of this link's own definitions
//! a dependency has to be able to reach ([`DepUndefs`]).
//!
//! [`DepExports`] also keeps the exports grouped by the address they sit at
//! inside the dependency, because a dependency may give one object several
//! names. That grouping is what lets a copy relocation put every alias on one
//! slot instead of handing each name its own.

use std::path::Path;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    elf::{
        ObjectFile, Sym64, SymbolTable, VersionTable,
        constants::{
            DT_NULL, DT_SONAME, SHN_ABS, SHN_UNDEF, SHT_DYNAMIC, STT_TLS,
            VER_NDX_HIDDEN,
        },
    },
    util::PAGE,
};

/// Identifies one object inside a dependency: which dependency defines it, and
/// where in that dependency it sits.
///
/// Two exports with the same origin are two names for one object. A copy
/// relocation keys on this rather than on the name, so every alias lands on
/// the one `.bss` slot.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DepAddr {
    /// Index of the defining dependency, in command-line order.
    pub dep: u32,
    /// The export's `st_value` inside that dependency.
    pub value: u64,
}

/// One exported symbol from a shared-object dependency: the attributes a copy
/// relocation needs to reserve a correctly sized and aligned slot.
#[derive(Clone, Copy)]
pub struct DynExport {
    /// `st_size` of the export, and so the size of the copy slot it takes.
    /// Zero when the dependency declared no extent for it, which is also what
    /// clears [`Self::placed`] and keeps it off the copy path entirely.
    pub size: u64,
    /// `st_info` type nibble (`STT_*`); only `STT_OBJECT` is copy-eligible.
    pub sym_type: u8,
    /// `st_other` visibility (`STV_*`), already masked.
    ///
    /// Only a default-visibility export can be copy-relocated. A protected one
    /// is a promise by the dependency that every reference inside it binds to
    /// its own definition, so an executable that takes over the name would
    /// leave the two halves of the process reading different storage; a hidden
    /// one is not reachable from outside the dependency at all. lld refuses
    /// both, in `canDefineSymbolInExecutable`.
    pub visibility: u8,
    /// Inferred alignment: the largest power of two dividing the export's
    /// `st_value` (its address inside the dependency), capped at a page.
    pub align: u64,
    /// Which object in which dependency the name refers to. Names sharing this
    /// are aliases of one object; see [`DepExports::aliases_at`].
    pub at: DepAddr,
    /// Whether the export names storage a copy relocation could take, and so
    /// whether sharing [`Self::at`] makes it an alias. Three things clear it:
    /// an `SHN_ABS` value, which is a constant rather than a place; a
    /// thread-local's value, which is an offset into a per-thread block; and a
    /// zero [`Self::size`], a place of no stated length. None is storage to
    /// copy, and none is made an alias by sharing a value, which is why lld's
    /// `getSymbolsAt` skips them.
    pub placed: bool,
    /// Whether the export names a real place in the dependency's image: a
    /// section-backed, non-thread-local definition. Unlike [`Self::placed`],
    /// a zero [`Self::size`] does not clear this: such an export still
    /// occupies the name, it only offers nothing a copy could take, and the
    /// copy path must refuse it rather than pass it by.
    pub storage: bool,
}

/// The merged exports of every `DT_NEEDED` dependency, keyed by name. Later
/// dependencies do not override earlier ones, matching symbol resolution order.
#[derive(Default)]
pub struct DepExports<'data> {
    /// Keyed by names borrowed straight from the dependency's mapping,
    /// which outlives the link. glibc alone exports ~3200 names, and a
    /// hello-world link used to spend a visible slice of its whole life
    /// copying every one into an owned key it would probe five times.
    by_name: FxHashMap<&'data [u8], DynExport>,
    /// The names a dependency defines at an address it gives more than one
    /// name, in that dependency's symbol table order. Addresses carrying a
    /// single name are left out: they need no group.
    aliases: FxHashMap<DepAddr, Vec<&'data [u8]>>,
}

impl<'data> DepExports<'data> {
    /// The export record for `name`, if any dependency defines it.
    pub fn get(&self, name: &[u8]) -> Option<DynExport> {
        self.by_name.get(name).copied()
    }

    /// Whether any dependency defines `name`.
    pub fn contains(&self, name: &[u8]) -> bool {
        self.by_name.contains_key(name)
    }

    /// Every name the defining dependency spells the object at `at` with, in
    /// that dependency's symbol table order, or `None` when the object has
    /// just the one name.
    ///
    /// The group is what the dependency declares, not what this link resolved:
    /// an earlier dependency may have exported one of these names first, in
    /// which case a reference to it does not reach this object at all. Callers
    /// filter the group with [`Self::get`] to drop those.
    pub fn aliases_at(&self, at: DepAddr) -> Option<&[&'data [u8]]> {
        self.aliases.get(&at).map(Vec::as_slice)
    }

    /// Inserts `name` unless it already exists (first dependency wins).
    fn insert(&mut self, name: &'data [u8], exp: DynExport) {
        self.by_name.entry(name).or_insert(exp);
    }

    /// Records `name` as one of the names `at` carries, keeping the symbol
    /// table order the caller walks in and rejecting an exact repeat.
    fn add_alias(&mut self, at: DepAddr, name: &'data [u8]) {
        let group = self.aliases.entry(at).or_default();
        if group.contains(&name) {
            return;
        }
        group.push(name);
    }

    /// Absorbs `other`, the tables of the dependency at index `dep`, stamping
    /// that index on every record it carries.
    ///
    /// Precedence is by first dependency: a name already present is not
    /// overwritten. The caller merges in input order so the result matches the
    /// serial baseline. Alias groups cannot collide, because `dep` makes each
    /// dependency's addresses its own.
    fn absorb(&mut self, other: Self, dep: u32) {
        for (name, mut exp) in other.by_name {
            exp.at.dep = dep;
            self.by_name.entry(name).or_insert(exp);
        }
        for (mut at, names) in other.aliases {
            at.dep = dep;
            self.aliases.insert(at, names);
        }
    }
}

/// The names the shared-object dependencies reference without defining: the
/// `SHN_UNDEF` rows of their `.dynsym` tables.
///
/// Such a row is not an export -- it is a reference the dependency makes, and
/// the loader has to satisfy it from somewhere. When this link defines the
/// name, that somewhere is the image being produced, so the definition has to
/// reach `.dynsym`; see [`crate::dynamic::exports_definition`].
#[derive(Default)]
pub struct DepUndefs<'data> {
    names: FxHashSet<&'data [u8]>,
}

impl<'data> DepUndefs<'data> {
    /// Whether any dependency carries an undefined reference to `name`.
    pub fn contains(&self, name: &[u8]) -> bool {
        self.names.contains(name)
    }

    /// Records `name` as referenced but not defined by a dependency. Empty
    /// names are dropped: there is no spelling for a loader to bind.
    fn insert(&mut self, name: &'data [u8]) {
        if !name.is_empty() {
            self.names.insert(name);
        }
    }

    /// Absorbs `other`. Membership is what the set answers, so folding two of
    /// them is a union and depends on no order.
    fn absorb(&mut self, other: Self) {
        self.names.extend(other.names);
    }
}

/// One dependency symbol's version: the soname the symbol lives in, the
/// version name it carries (e.g. `GLIBC_2.2.5`), and the ELF hash of that
/// version name (written into `.gnu.version_r`'s `vna_hash`).
#[derive(Clone)]
pub struct DepVersion {
    /// The soname of the dependency that exports the symbol.
    pub soname: Vec<u8>,
    /// The version name (e.g. `GLIBC_2.2.5`).
    pub name: Vec<u8>,
    /// ELF hash of `name`, taken from the dependency's `.gnu.version_d`.
    pub hash: u32,
}

/// Per-symbol version info for the shared-object dependencies, keyed by
/// symbol name.
///
/// The dynamic plan looks each undefined import up here to learn which
/// (soname, version) it must record in `.gnu.version_r`, and which
/// `.gnu.version` index to assign the import's dynsym entry.
#[derive(Default)]
pub struct DepVersions<'data> {
    by_name: FxHashMap<&'data [u8], DepVersion>,
}

impl<'data> DepVersions<'data> {
    /// The version info for `name`, if any dependency versioned it.
    pub fn get(&self, name: &[u8]) -> Option<&DepVersion> {
        self.by_name.get(name)
    }

    /// Inserts `name` unless it already exists (first dependency wins), to
    /// preserve symbol resolution precedence.
    fn insert(
        &mut self,
        name: &'data [u8],
        soname: &[u8],
        version: &[u8],
        hash: u32,
    ) {
        self.by_name.entry(name).or_insert_with(|| DepVersion {
            soname: soname.to_vec(),
            name: version.to_vec(),
            hash,
        });
    }

    /// Absorbs `other` into self, preserving first-dependency precedence.
    fn absorb(&mut self, other: Self) {
        for (name, v) in other.by_name {
            self.by_name.entry(name).or_insert(v);
        }
    }
}

/// The tables one shared-object dependency contributes, read before the
/// dependency's index is known. [`DepTables::merge_into`] stamps the index on
/// and folds them into the link's tables.
#[derive(Default)]
pub struct DepTables<'data> {
    exports: DepExports<'data>,
    versions: DepVersions<'data>,
    undefs: DepUndefs<'data>,
}

impl<'data> DepTables<'data> {
    /// Reads the export, version and undefined-reference tables out of one
    /// dependency's bytes. `soname` is recorded on every versioned export so
    /// `.gnu.version_r` can name the file each version came from.
    ///
    /// All three are read from the same `.dynsym` and `.gnu.version` pair: the
    /// exports need the version index to tell a dependency's default
    /// definition of a name from a superseded one, which is the column the
    /// version table is built from as well, and the undefined rows are the ones
    /// the export walk steps over. A dependency that carries no version
    /// information reads as all-default, which is what it is.
    pub fn read(bytes: &'data [u8], soname: &[u8]) -> Self {
        let mut tables = Self::default();
        if let Ok(obj) = ObjectFile::parse(bytes)
            && let Ok(Some(dynsym)) = obj.dynamic_symbols()
        {
            // Sized once: glibc contributes thousands of rows, and growing
            // through them re-hashes the maps a dozen times on the very
            // latency path a small link lives on.
            let rows = dynsym.syms.len();
            tables.exports.by_name.reserve(rows);
            tables.versions.by_name.reserve(rows);
            let vt = obj.version_table().unwrap_or_default();
            read_dep_exports(
                &dynsym,
                &vt,
                &mut tables.exports,
                &mut tables.undefs,
            );
            read_dep_versions(&dynsym, &vt, soname, &mut tables.versions);
        }
        tables
    }

    /// Folds these tables into the link's, as the dependency at index `dep`.
    /// Called in input order, so first-dependency precedence holds.
    pub fn merge_into(
        self,
        exports: &mut DepExports<'data>,
        versions: &mut DepVersions<'data>,
        undefs: &mut DepUndefs<'data>,
        dep: u32,
    ) {
        exports.absorb(self.exports, dep);
        versions.absorb(self.versions);
        undefs.absorb(self.undefs);
    }
}

/// Reads a shared object's defined `.dynsym` exports into `out`. Every named
/// entry the dependency defines is one, as in lld: `SHN_UNDEF` is the only
/// disqualifier, because such an entry is a reference the dependency makes
/// rather than a name it offers. An unnamed entry is skipped for want of a
/// spelling to bind to, not because it is not an export.
///
/// The `SHN_UNDEF` rows are not merely skipped: their names go to `undefs`,
/// because a reference a dependency makes is what obliges this link to export
/// its own definition of the name. lld collects the same set from the same
/// rows, marking each resulting symbol as exported
/// (`lld/ELF/InputFiles.cpp`).
///
/// A zero `st_size` in particular is not a disqualifier. It says the
/// dependency declared no extent for the object, which decides whether the
/// name can join an alias group and whether a copy relocation may size a slot
/// from it -- both of which hang off [`alias_eligible`] below -- and says
/// nothing about whether the export exists. An unsized export (a `.globl` with
/// no `.size`, common in hand-written assembly) is still an export.
///
/// The exports are also grouped by address, in symbol table order, so a copy
/// relocation can find an object's other names. That walk keeps only what
/// [`alias_eligible`] admits, exactly as lld's `getSymbolsAt` does.
///
/// A dependency may spell one name twice and tell the two apart by version
/// alone: `foo@GLIBC_2.2.5` beside `foo@@GLIBC_2.17` is two `.dynsym` rows
/// both named `foo`, with different sizes and different addresses. Only the
/// default one is recorded, since that is the definition an unversioned
/// reference binds to; see [`superseded`].
///
/// The dependency index is not known this early, so every record is written
/// with index zero; [`DepExports::absorb`] stamps the real one.
fn read_dep_exports<'data>(
    dynsym: &SymbolTable<'data>,
    vt: &VersionTable<'_>,
    out: &mut DepExports<'data>,
    undefs: &mut DepUndefs<'data>,
) {
    let mut defaults: FxHashSet<&'data [u8]> = FxHashSet::default();
    default_defined(dynsym, vt, &mut defaults);
    // How many names each address carries, so the second pass records only the
    // addresses that actually have aliases.
    let mut shared: FxHashMap<u64, u32> = FxHashMap::default();
    for (i, sym) in dynsym.iter().enumerate() {
        let name = dynsym.name(sym);
        let placed =
            has_storage(sym) && sym.st_size.get() != 0 && !name.is_empty();
        if placed && !superseded(vt, i, name, &defaults) {
            let n = shared.entry(sym.st_value.get()).or_insert(0);
            *n = n.saturating_add(1);
        }
    }
    for (i, sym) in dynsym.iter().enumerate() {
        if sym.st_shndx.get() == SHN_UNDEF {
            undefs.insert(dynsym.name(sym));
            continue;
        }
        let name = dynsym.name(sym);
        if name.is_empty() || superseded(vt, i, name, &defaults) {
            continue;
        }
        let at = DepAddr {
            dep: 0,
            value: sym.st_value.get(),
        };
        let storage = has_storage(sym);
        let placed = storage && sym.st_size.get() != 0;
        out.insert(
            name,
            DynExport {
                size: sym.st_size.get(),
                sym_type: sym.type_(),
                visibility: sym.visibility(),
                align: export_align(at.value),
                at,
                placed,
                storage,
            },
        );
        if placed && shared.get(&at.value).copied().unwrap_or(0) > 1 {
            out.add_alias(at, name);
        }
    }
}

/// Collects the names the dependency offers a default definition of.
///
/// The `.gnu.version` index of a defined row carries `VER_NDX_HIDDEN` when the
/// definition is not the default one for its name: a superseded
/// `foo@GLIBC_2.2.5` kept for binaries linked before the interface changed.
/// The row without the bit is the one an unversioned reference reaches.
fn default_defined<'data>(
    dynsym: &SymbolTable<'data>,
    vt: &VersionTable<'_>,
    out: &mut FxHashSet<&'data [u8]>,
) {
    for (i, sym) in dynsym.iter().enumerate() {
        if sym.st_shndx.get() == SHN_UNDEF
            || vt.version_of(i) & VER_NDX_HIDDEN != 0
        {
            continue;
        }
        let name = dynsym.name(sym);
        if !name.is_empty() {
            out.insert(name);
        }
    }
}

/// Whether the `.dynsym` row at `i` is a non-default definition of a name the
/// dependency also defines by default, and so not the row to record.
///
/// Such a row is skipped whole: recording it would make the export's size,
/// type and address an accident of which spelling `.dynsym` happened to list
/// first, and letting it into an alias group would put the name on storage
/// this link does not resolve it to.
///
/// A name with no default definition keeps the row it has. lld's answer is
/// larger: it names such a symbol `foo@VERSION`, so the two stay distinct
/// symbols and a reference can ask for either. xold has one name per symbol,
/// so the default definition is the one it can represent.
fn superseded(
    vt: &VersionTable<'_>,
    i: usize,
    name: &[u8],
    defaults: &FxHashSet<&[u8]>,
) -> bool {
    vt.version_of(i) & VER_NDX_HIDDEN != 0 && defaults.contains(name)
}

/// Whether `sym` names a real place in the dependency's image: what
/// [`DynExport::storage`] carries, and half of what [`DynExport::placed`]
/// asks.
///
/// An `SHN_ABS` value is a constant rather than a place and a thread-local's
/// is an offset into a per-thread block, so neither names storage. Size does
/// not enter: a zero-sized export still names its place, it only offers no
/// extent -- no object for a second name to alias, and no length for a copy
/// slot -- which is why `placed` additionally requires a size.
fn has_storage(sym: &Sym64) -> bool {
    let shndx = sym.st_shndx.get();
    shndx != SHN_UNDEF && shndx != SHN_ABS && sym.type_() != STT_TLS
}

/// Reads a shared-object dependency's per-symbol version info into `out`.
/// Walks `.dynsym` parallel to `.gnu.version`, and resolves each non-default
/// version index (>= 2) against the dependency's `.gnu.version_d`. The first
/// dependency to provide a version for a name wins, matching the existing
/// `dep_exports` precedence.
fn read_dep_versions<'data>(
    dynsym: &SymbolTable<'data>,
    vt: &VersionTable<'_>,
    soname: &[u8],
    out: &mut DepVersions<'data>,
) {
    use crate::elf::constants::VER_NDX_GLOBAL;
    if vt.defs.is_empty() {
        return;
    }
    // version index -> (version name, hash), from the dep's VERDEF entries.
    let mut by_idx: FxHashMap<u16, (&[u8], u32)> = FxHashMap::default();
    for d in &vt.defs {
        if let Some(name) = d.names.first() {
            by_idx.insert(d.index, (*name, d.hash));
        }
    }
    // A hidden row is a definition an unversioned reference is not meant to
    // prefer, so when a name carries both a superseded and a default row the
    // default one records the version. A name the dependency defines only
    // hidden (glibc exports many, `pthread_mutexattr_getprotocol@GLIBC_2.4`
    // among them) still records that row: the high bit does not change which
    // `vd_ndx` the index names, it comes off before the lookup (lld masks the
    // same way, `InputFiles.cpp`).
    let mut hidden: Vec<(&'data [u8], u16)> = Vec::new();
    for (i, sym) in dynsym.iter().enumerate() {
        if sym.st_shndx.get() == SHN_UNDEF {
            continue;
        }
        let name = dynsym.name(sym);
        if name.is_empty() {
            continue;
        }
        let raw = vt.version_of(i);
        let vidx = raw & !VER_NDX_HIDDEN;
        // 0 = local, 1 = global/default: no version requirement. Anything
        // >= 2 names a real version recorded in the dep's VERDEF.
        if vidx <= VER_NDX_GLOBAL {
            continue;
        }
        if raw & VER_NDX_HIDDEN != 0 {
            hidden.push((name, vidx));
            continue;
        }
        if let Some((vname, vhash)) = by_idx.get(&vidx).copied() {
            out.insert(name, soname, vname, vhash);
        }
    }
    for (name, vidx) in hidden {
        if out.get(name).is_some() {
            continue;
        }
        if let Some((vname, vhash)) = by_idx.get(&vidx).copied() {
            out.insert(name, soname, vname, vhash);
        }
    }
}

/// The alignment a copy relocation needs for an export whose address inside
/// the dependency is `value`: the largest power of two dividing it, capped at
/// one page (matching lld's `getAlignment`, sans the section-alignment cap).
fn export_align(value: u64) -> u64 {
    if value == 0 {
        return 1;
    }
    let zeros = value.trailing_zeros();
    let align = 1u64.checked_shl(zeros).unwrap_or(0);
    align.clamp(1, PAGE)
}

/// Extracts the soname a shared object advertises (`DT_SONAME`), falling back
/// to how the object was named when it has none. The result is recorded as the
/// `DT_NEEDED` value.
///
/// The fallback is where the two spellings part. A library found by `-lfoo`
/// takes its basename, because that is what the loader's search will find
/// again. One named directly keeps the path as written -- an explicit
/// `/opt/vendor/libpriv.so` reduced to `libpriv.so` names something the loader
/// has no reason to find, so the image would not start. lld splits the same
/// way: `withLOption ? path::filename(path) : path`
/// (`lld/ELF/Driver.cpp`).
pub fn shared_soname(path: &Path, bytes: &[u8], from_l: bool) -> Vec<u8> {
    if let Ok(obj) = ObjectFile::parse(bytes)
        && let Some(name) = read_soname(&obj)
    {
        return name;
    }
    let named = if from_l {
        path.file_name().map_or(path, Path::new)
    } else {
        path
    };
    path_bytes(named)
}

/// The bytes of `path`, as a `DT_NEEDED` string.
///
/// An ELF soname is a byte string, and so is a Unix path, so the two map
/// across without a UTF-8 round trip. Going through `to_str` instead would
/// drop a non-UTF-8 path to an empty name and emit a `DT_NEEDED` the loader
/// cannot resolve, even though the command line accepts such paths. lld
/// carries the raw bytes here for the same reason.
#[cfg(unix)]
fn path_bytes(path: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;

    path.as_os_str().as_bytes().to_vec()
}

/// The bytes of `path`, where the platform's paths are not byte strings.
/// Nothing better than a lossy conversion is available, and it at least keeps
/// a resolvable name for the common all-ASCII case.
#[cfg(not(unix))]
fn path_bytes(path: &Path) -> Vec<u8> {
    path.to_string_lossy().into_owned().into_bytes()
}

/// Reads `DT_SONAME` from a shared object's `.dynamic` section, resolving the
/// string-table offset it carries. Returns `None` if there is no dynamic
/// section or no `DT_SONAME`.
fn read_soname(obj: &ObjectFile<'_>) -> Option<Vec<u8>> {
    let dyn_shdr = obj
        .sections()
        .iter()
        .find(|s| s.sh_type.get() == SHT_DYNAMIC)?;
    let dyn_bytes = obj.section_data(dyn_shdr).ok()?;
    let strtab = obj.sections().get(dyn_shdr.sh_link.get() as usize)?;
    let strtab_bytes = obj.section_data(strtab).ok()?;
    for chunk in dyn_bytes.chunks(16) {
        if chunk.len() < 16 {
            break;
        }
        let tag = i64::from_le_bytes(chunk[..8].try_into().ok()?);
        if tag == DT_NULL {
            break;
        }
        if tag == DT_SONAME {
            let val = u64::from_le_bytes(chunk[8..16].try_into().ok()?);
            let off = usize::try_from(val).ok()?;
            let rest = strtab_bytes.get(off..)?;
            let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
            return Some(rest.get(..end)?.to_vec());
        }
    }
    None
}
