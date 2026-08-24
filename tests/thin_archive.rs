//! A thin archive, whose members are files beside it, still links.
//!
//! `ar rcsT` writes an archive that records where each member lives
//! instead of copying it in: the magic is `!<thin>\n`, a member's header
//! names the file (through the `//` long-name table, even for a short
//! name), and its size is the external file's -- no data bytes follow.
//! A distribution ships one where the members are kept close and the
//! library small, and `ld` links it by reading each member from the
//! path its header names, taken relative to the archive's own
//! directory (`lld/ELF/ArchiveFiles.cpp`).
//!
//! xold recognised only `!<arch>\n`, so a thin archive fell through the
//! format dispatch and the link died on what is still a library: the
//! members were never offered at all. And the header's size field
//! cannot be walked the fat way, because the bytes it counts are not
//! in the file -- the member walk has to know the archive is thin
//! before it can even find the next header.
//!
//! The fixture asks the host `ar` for a real thin archive, so the
//! shape under test is the one the tool writes.
//!
//! Gated on `clang` and `ar`; if either is missing the test prints a
//! note and returns.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{archive_tool, which};
use xold::{icf::IcfMode, linker::link_to};

mod common;

/// The member that defines the symbol the program wants.
const LIB_SRC: &[u8] = b"int from_member(void) { return 3; }\n";

/// The program: one undefined reference only the archive can answer.
const MAIN_SRC: &[u8] =
    b"extern int from_member(void);\nvoid _start(void) { from_member(); }\n";

/// A link against a thin archive reads the member from beside it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_thin_archive_member_is_read_from_its_path() {
    let Some(dir) = workdir() else {
        return;
    };
    let Some((member, main)) = compile(&dir) else {
        return;
    };
    let lib = dir.join("libthin.a");
    let archived = Command::new(archive_tool().expect("checked in workdir"))
        .args(["rcsT"])
        .arg(&lib)
        .arg(&member)
        .current_dir(&dir)
        .status()
        .expect("ar must run")
        .success();
    assert!(archived, "ar builds the thin archive");
    let out = dir.join("prog");
    let inputs = [lib, main];
    let res = link_to(&inputs, &out, b"_start", false, IcfMode::None, false);
    assert!(
        res.is_ok(),
        "the member is where the header says: {:?}",
        res.err()
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir() -> Option<PathBuf> {
    if which("clang").is_none() || archive_tool().is_none() {
        eprintln!("skipping thin-archive: clang or ar unavailable");
        return None;
    }
    let dir =
        std::env::temp_dir().join(format!("xold_thin_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the member and the program.
fn compile(dir: &Path) -> Option<(PathBuf, PathBuf)> {
    let clang = which("clang")?;
    let member = build(dir, &clang, "member.c", LIB_SRC)?;
    let main = build(dir, &clang, "main.c", MAIN_SRC)?;
    Some((member, main))
}

/// Compiles one source to a static object.
fn build(dir: &Path, clang: &Path, name: &str, src: &[u8]) -> Option<PathBuf> {
    let src_path = dir.join(name);
    let obj = dir.join(name.replace(".c", ".o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping thin-archive: clang cannot build {name}");
        return None;
    }
    Some(obj)
}
