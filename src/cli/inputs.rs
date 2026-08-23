//! The input list under construction: `-l` resolution, the `-L` search path
//! and the sysroot it resolves against.

use std::{
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
};

use xold::search::{self, LibKind};

use super::next_os;

/// The input list under construction, with the `-L` search path and sysroot
/// it resolves `-l` against.
///
/// `paths`, `from_l` and `as_needed` stay the same length because only
/// [`Self::file`] and [`Self::library`] append, and each appends to all three.
/// The provenance is carried because a shared object with no `DT_SONAME`
/// takes its `DT_NEEDED` from how it was named, and only a `-l` library takes
/// the basename.
///
/// [`Self::kind`] and [`Self::as_needed`] are the state `-Bstatic`,
/// `-Bdynamic`, `--as-needed` and `--no-as-needed` set. They apply to the
/// inputs written after them and to no others, which is why they are recorded
/// per input here rather than kept as one setting for the link.
#[derive(Default)]
pub struct Inputs {
    pub paths: Vec<PathBuf>,
    pub from_l: Vec<bool>,
    pub search: Vec<PathBuf>,
    pub sysroot: Option<PathBuf>,
    /// Whether a `--start-lib` group is open, so the plain files that follow
    /// join it instead of the input list.
    pub in_lib: bool,
    /// The paths collected by the open `--start-lib` group.
    pub group: Vec<PathBuf>,
    /// The index ranges of `paths` each closed group occupies. `Options`
    /// carries them so the input walk can hand each group to the archive
    /// builder as one unit.
    pub lib_groups: Vec<(usize, usize)>,
    /// Parallel to `paths`: whether `--as-needed` was in force where each was
    /// written. A shared object marked this way earns its `DT_NEEDED` only by
    /// being bound to.
    pub as_needed_flags: Vec<bool>,
    /// Which files the next `-l` will accept: `-Bstatic` narrows it to
    /// archives, `-Bdynamic` restores the default.
    pub kind: LibKind,
    /// Whether `--as-needed` is in force at this point in the command line.
    pub as_needed: bool,
    /// The `(kind, as_needed)` pairs `--push-state` saved, innermost last.
    pub state: Vec<(LibKind, bool)>,
}

impl Inputs {
    /// A file named directly on the command line.
    pub fn file(&mut self, path: &OsStr) {
        let path = PathBuf::from(path);
        if self.in_lib {
            self.group.push(path);
        } else {
            self.paths.push(path);
            self.from_l.push(false);
            self.as_needed_flags.push(self.as_needed);
        }
    }

    /// A `-lNAME` library, resolved against the search path.
    pub fn library(&mut self, name: &str) -> std::result::Result<(), String> {
        if self.in_lib {
            // A group is one library built from the objects inside it; a
            // second library cannot become a member of the first.
            return Err(
                "xold: -l cannot appear between --start-lib and --end-lib"
                    .into(),
            );
        }
        let path =
            library(name, &self.search, self.sysroot.as_deref(), self.kind)?;
        self.paths.push(path);
        self.from_l.push(true);
        self.as_needed_flags.push(self.as_needed);
        Ok(())
    }

    /// Saves the `-Bstatic`/`--as-needed` state, as `--push-state` asks.
    pub fn push_state(&mut self) {
        self.state.push((self.kind, self.as_needed));
    }

    /// Restores the state `--push-state` saved, or reports an unmatched
    /// `--pop-state`.
    pub fn pop_state(&mut self) -> std::result::Result<(), String> {
        let Some((kind, as_needed)) = self.state.pop() else {
            return Err("xold: --pop-state without --push-state".into());
        };
        self.kind = kind;
        self.as_needed = as_needed;
        Ok(())
    }

    /// Opens a `--start-lib` group.
    pub const fn begin_lib(&mut self) {
        self.in_lib = true;
    }

    /// Closes the open group, recording the input range it occupies.
    pub fn end_lib(&mut self) {
        let start = self.paths.len();
        let as_needed = self.as_needed;
        for path in self.group.drain(..) {
            self.paths.push(path);
            self.from_l.push(false);
            self.as_needed_flags.push(as_needed);
        }
        if self.paths.len() > start {
            self.lib_groups.push((start, self.paths.len()));
        }
        self.in_lib = false;
    }
}
/// The `--sysroot` value, taken in a pass of its own before the arguments are
/// read in order.
///
/// `-l` is resolved where it appears, so a sysroot found later would arrive
/// too late to redirect it. Reading it first makes the option describe the
/// whole link rather than the part of the command line that follows it.
pub fn parse_sysroot(
    args: &[OsString],
) -> std::result::Result<Option<PathBuf>, String> {
    let mut sysroot = None;
    let mut i = 0;
    while i < args.len() {
        // Only the spellings are matched; a non-UTF-8 argument is a path
        // and cannot start a `--sysroot` form.
        let Some(arg) = args[i].to_str() else {
            i += 1;
            continue;
        };
        if arg == "--sysroot" {
            i += 1;
            sysroot = Some(PathBuf::from(next_os(args, i, "--sysroot")?));
        } else if let Some(path) = arg.strip_prefix("--sysroot=") {
            sysroot = Some(PathBuf::from(path));
        }
        i += 1;
    }
    Ok(sysroot)
}

/// Collects every `-L` directory before any `-l` is resolved.
///
/// Where a `-L` sits on the command line does not narrow which `-l` it serves:
/// GNU ld documents that all of them apply to all libraries, and lld gathers
/// them in `readConfigs` before it opens a single input. Resolving `-l`
/// against only the directories seen so far made `main.o -lfoo -L/opt/lib`
/// either fail outright or, worse, quietly pick up a system copy of `libfoo`
/// -- a different library, with no diagnostic. `--sysroot` already gets a
/// pre-pass for exactly this reason.
pub fn parse_search_path(
    args: &[OsString],
) -> std::result::Result<Vec<PathBuf>, String> {
    let mut search = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let Some(arg) = args[i].to_str() else {
            i += 1;
            continue;
        };
        if arg == "-L" {
            i += 1;
            search.push(PathBuf::from(next_os(args, i, "-L")?));
        } else if let Some(dir) =
            arg.strip_prefix("-L").filter(|d| !d.is_empty())
        {
            search.push(PathBuf::from(dir));
        }
        i += 1;
    }
    Ok(search)
}
/// Resolves `-lNAME` against the search path, or reports that it was not found.
pub fn library(
    name: &str,
    search_path: &[PathBuf],
    sysroot: Option<&Path>,
    kind: LibKind,
) -> std::result::Result<PathBuf, String> {
    search::find_library_kind(name, search_path, sysroot, kind).ok_or_else(
        || match kind {
            LibKind::Any => format!("xold: cannot find -l{name}"),
            LibKind::ArchiveOnly => format!(
                "xold: cannot find -l{name}: -Bstatic is in force, so only \
                 lib{name}.a was looked for"
            ),
        },
    )
}
