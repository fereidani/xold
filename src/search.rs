//! Library search: resolves `-l NAME` to a concrete file by scanning the
//! directories named with `-L` followed by the host's default search paths.
//!
//! Within a directory the order is `libNAME.so`, then the highest versioned
//! `libNAME.so.M`, then `libNAME.a`. The bare name leads because that is the
//! one the developer package installs and the one every reference linker
//! resolves: a tree with a `libfoo.so` symlink to `libfoo.so.1` beside a
//! runtime `libfoo.so.2` must link against 1, as it does everywhere else,
//! rather than silently against a different ABI.
//!
//! A candidate is accepted as an ELF object, an archive, or a GNU ld script
//! ([`crate::script`]) -- on GNU systems the bare name is frequently the last
//! of those, `libc.so` being text that `GROUP`s `libc.so.6`. What a candidate
//! is, is read out of its bytes rather than assumed from its name, the
//! versioned pick included, and the versioned scan stays as the fallback for a
//! bare name that is none of the three.
//!
//! `-l:filename` names a file outright, with no `lib` prefix or suffix added,
//! and is searched for by that name alone.
//!
//! # Cross links
//!
//! [`DEFAULT_PATHS`] describes the host. A link that targets another
//! architecture must not resolve `-l` against the host's libraries, so
//! `--sysroot` re-roots those defaults at another tree. It does not move the
//! `-L` directories, which name themselves; a `-L` path that is meant to be
//! sysroot-relative says so by leading with `=`, as it does for GNU `ld`.

use std::{
    io::Read as _,
    path::{Path, PathBuf},
};

/// Default directories searched after any `-L` paths, matching a native
/// x86-64 glibc layout. `/usr/local/lib64` leads so locally installed
/// libraries win, as with the system linker. Under `--sysroot` each is taken
/// relative to that tree instead.
const DEFAULT_PATHS: [&str; 5] = [
    "/usr/local/lib64",
    "/usr/lib64",
    "/lib64",
    "/usr/local/lib",
    "/usr/lib",
];

/// The marker that makes a `-L` path sysroot-relative, following GNU `ld`.
const SYSROOT_PREFIX: char = '=';

/// Resolves `-l name` to a concrete file path, scanning `search` then the
/// default directories in order. Returns the first directory that yields a
/// candidate.
///
/// `sysroot` re-roots the default directories, and any `-L` path written as
/// `=/usr/lib`. Pass `None` for a native link.
pub fn find_library(
    name: &str,
    search: &[PathBuf],
    sysroot: Option<&Path>,
) -> Option<PathBuf> {
    find_library_kind(name, search, sysroot, LibKind::Any)
}

/// Which files a `-l` search will accept.
///
/// A command line switches between the two with `-Bstatic` and `-Bdynamic`,
/// so which one applies is a property of the position `-l` was written at,
/// not of the link. A driver uses it to pin one library to its archive while
/// the rest of the line stays dynamic: `gcc` wraps `-lgcc` that way, and
/// `rustc` wraps the Rust standard library.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum LibKind {
    /// `libNAME.so`, then the highest versioned `libNAME.so.M`, then
    /// `libNAME.a`. The default, and what `-Bdynamic` restores.
    #[default]
    Any,
    /// `libNAME.a` only, as `-Bstatic` asks. A shared object of the same name
    /// beside it is passed over rather than preferred.
    ArchiveOnly,
}

/// Resolves `-l name` under an explicit [`LibKind`].
pub fn find_library_kind(
    name: &str,
    search: &[PathBuf],
    sysroot: Option<&Path>,
    kind: LibKind,
) -> Option<PathBuf> {
    // `-l:filename` names the file itself. No prefix and no suffix are added,
    // and no versioned scan runs: the caller already said which file.
    let exact = name.strip_prefix(':');
    let look = |dir: &Path| match (exact, kind) {
        (Some(file), LibKind::Any) => find_exact(file, dir),
        (Some(file), LibKind::ArchiveOnly) => {
            let path = dir.join(file);
            has_magic(&path, ARCHIVE_MAGIC).then_some(path)
        }
        (None, LibKind::Any) => find_in(name, dir),
        (None, LibKind::ArchiveOnly) => find_archive(name, dir),
    };
    for dir in search.iter().map(|dir| rooted_search(sysroot, dir)) {
        if let Some(found) = look(&dir) {
            return Some(found);
        }
    }
    for dir in DEFAULT_PATHS {
        if let Some(found) = look(&rooted(sysroot, dir)) {
            return Some(found);
        }
    }
    None
}

/// Finds `file` by that exact name in the `-L` directories and then the
/// defaults, which is how a linker script's `INPUT` and `GROUP` lists resolve
/// a relative name that is not beside the script.
///
/// Existence is the whole test, as it is in lld's `findFromSearchPaths`
/// (`lld/ELF/DriverUtils.cpp`): the script said which file it
/// wants, so a candidate that turns out to be the wrong kind is an error to
/// report against that file rather than a reason to keep looking.
pub fn find_in_paths(
    file: &str,
    search: &[PathBuf],
    sysroot: Option<&Path>,
) -> Option<PathBuf> {
    let named = |dir: &Path| {
        let path = dir.join(file);
        path.exists().then_some(path)
    };
    for dir in search.iter().map(|dir| rooted_search(sysroot, dir)) {
        if let Some(found) = named(&dir) {
            return Some(found);
        }
    }
    for dir in DEFAULT_PATHS {
        if let Some(found) = named(&rooted(sysroot, dir)) {
            return Some(found);
        }
    }
    None
}

/// A `-L` directory as it should be searched: unchanged, unless it leads with
/// `=`, which asks for the remainder to be taken relative to the sysroot.
fn rooted_search(sysroot: Option<&Path>, dir: &Path) -> PathBuf {
    let Some(rest) = dir.to_str().and_then(|d| d.strip_prefix(SYSROOT_PREFIX))
    else {
        return dir.to_path_buf();
    };
    rooted(sysroot, rest)
}

/// Joins a default path onto the sysroot, or returns it unchanged when there
/// is none. The leading separator is trimmed so the join does not discard the
/// root.
pub fn rooted(sysroot: Option<&Path>, path: &str) -> PathBuf {
    sysroot.map_or_else(
        || PathBuf::from(path),
        |root| root.join(path.trim_start_matches('/')),
    )
}

/// Looks for a `-l:filename` candidate: that exact name, accepted as a shared
/// object, an archive or a linker script according to its own contents.
fn find_exact(file: &str, dir: &Path) -> Option<PathBuf> {
    let path = dir.join(file);
    (has_magic(&path, ARCHIVE_MAGIC) || is_linkable(&path)).then_some(path)
}

/// Looks for `name`'s library inside one directory: a real-ELF
/// `libNAME.so`, else the highest versioned `libNAME.so.M`, else a real
/// `libNAME.a`.
///
/// The bare name leads because it is the one every reference linker resolves,
/// and picking a versioned file over it silently swaps the ABI: a developer
/// symlink `libfoo.so -> libfoo.so.1` beside a runtime `libfoo.so.2` must
/// still mean 1. A bare name that is a linker script leads just the same,
/// since that script is how the system spells which files the library is made
/// of; the versioned scan is the fallback for a bare name that is neither.
///
/// One `read_dir` serves both shared-object spellings: the versioned names can
/// only be found by scanning, and the bare one is recognised on the way past.
/// The scan is reached only when the bare name is not there: on a development
/// system it nearly always is, and one direct lookup is far cheaper than
/// walking a library directory of several thousand entries per `-l`.
fn find_in(name: &str, dir: &Path) -> Option<PathBuf> {
    let prefix = format!("lib{name}.so.");
    let bare = format!("lib{name}.so");
    let lead = dir.join(&bare);
    if is_linkable(&lead) {
        return Some(lead);
    }
    let entries = std::fs::read_dir(dir).ok()?;
    let mut best: Option<(PathBuf, Vec<u64>, String)> = None;
    let mut bare_so: Option<PathBuf> = None;
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let Some(fname) = file_name.to_str() else {
            continue;
        };
        if fname == bare {
            bare_so = Some(entry.path());
        } else if let Some(rest) = fname.strip_prefix(&prefix) {
            let key = version_key(rest);
            // The name breaks a tie in the numeric key. Without it the winner
            // between `libfoo.so.1.0` and `libfoo.so.1.x` is whichever
            // `read_dir` handed over first, which is the filesystem's
            // business and not an ordering.
            let better =
                best.as_ref().is_none_or(|(_, prev_key, prev_name)| {
                    (&key, fname) > (prev_key, prev_name.as_str())
                });
            if better {
                best = Some((entry.path(), key, fname.to_owned()));
            }
        }
    }
    // Every candidate is read before it is accepted: nothing about a name
    // makes a file an object, and on GNU systems `libNAME.so` and `libNAME.a`
    // are both names a linker script takes.
    if let Some(path) = bare_so.filter(|p| is_linkable(p)) {
        return Some(path);
    }
    if let Some(path) = best.map(|(p, _, _)| p).filter(|p| is_linkable(p)) {
        return Some(path);
    }
    let archive = dir.join(format!("lib{name}.a"));
    (has_magic(&archive, ARCHIVE_MAGIC) || is_script(&archive))
        .then_some(archive)
}

/// Looks for `name`'s archive inside one directory, which is all a `-Bstatic`
/// `-l` will take.
///
/// A linker script under the archive name is accepted for the same reason the
/// shared search accepts one: on GNU systems that text is how a library says
/// which files it is made of.
fn find_archive(name: &str, dir: &Path) -> Option<PathBuf> {
    let archive = dir.join(format!("lib{name}.a"));
    (has_magic(&archive, ARCHIVE_MAGIC) || is_script(&archive))
        .then_some(archive)
}

/// Whether the shared-object candidate at `path` is one the link can consume:
/// a real ELF object, or a linker script naming the files that are.
fn is_linkable(path: &Path) -> bool {
    has_magic(path, ELF_MAGIC) || is_script(path)
}

/// Whether the file at `path` is a GNU ld script.
///
/// Decided by content like every other candidate here. `libNAME.so` being a
/// script is a convention rather than a rule, so the bytes are what answer;
/// see [`crate::script::looks_like`], which the driver applies to the same
/// leading bytes when it expands the file.
fn is_script(path: &Path) -> bool {
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut head = [0u8; SCRIPT_PROBE];
    let mut filled = 0usize;
    while filled < head.len() {
        // A short read is the end of the file, not a failure: a script may be
        // shorter than the probe.
        let Ok(n @ 1..) = file.read(&mut head[filled..]) else {
            break;
        };
        filled = filled.saturating_add(n);
    }
    crate::script::looks_like(head.get(..filled).unwrap_or(&head))
}

/// How much of a candidate is read to tell script text from an object. Long
/// enough that a binary whose first bytes happen to be printable is still
/// caught, short enough to stay one page.
const SCRIPT_PROBE: usize = 64;

/// Parses the dotted suffix after `libNAME.so.` (for example `6` or `6.0.1`)
/// into a vector of numeric components, dropping any non-numeric trailing
/// parts. Non-numeric segments compare as zero, which is correct for the
/// common all-numeric glibc versions. Comparing component by component is
/// what puts `.10` above `.9` rather than below it.
fn version_key(suffix: &str) -> Vec<u64> {
    suffix
        .split('.')
        .map(|part| part.parse::<u64>().unwrap_or(0))
        .collect()
}

/// The bytes an ELF object opens with.
const ELF_MAGIC: &[u8] = b"\x7fELF";

/// The bytes an `ar` archive opens with.
const ARCHIVE_MAGIC: &[u8] = b"!<arch>\n";

/// The longest magic this module probes for, which fixes the read buffer.
const MAGIC_MAX: usize = 8;

/// Whether the file at `path` opens with `magic`.
///
/// Only the leading bytes are read, into the stack: this runs once per
/// candidate directory, and the files it probes are whole shared libraries,
/// so reading one to look at four bytes would be megabytes of pure waste.
fn has_magic(path: &Path, magic: &[u8]) -> bool {
    debug_assert!(magic.len() <= MAGIC_MAX, "magic fits the probe buffer");
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut head = [0u8; MAGIC_MAX];
    let Some(slot) = head.get_mut(..magic.len()) else {
        return false;
    };
    // `read` may stop short of the buffer even when the file is longer, so
    // the head is filled in a loop rather than in one call.
    let mut filled = 0usize;
    while filled < slot.len() {
        // A zero-length read is end of file: the candidate is shorter than
        // the magic and so is not what it claims to be.
        let Ok(n @ 1..) = file.read(&mut slot[filled..]) else {
            return false;
        };
        filled = filled.saturating_add(n);
    }
    slot == magic
}
