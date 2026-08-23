//! An archive without a symbol index still gives up its members.
//!
//! `ar` writes a `/` symbol index when it is asked to (`ar rcs`), and the
//! index is only an accelerator: the members themselves carry the
//! definitions. GNU `ld` links an archive whose index is missing or stale
//! by scanning the members, and `ranlib` exists precisely to repair one.
//! xold read only the index, so an archive built without one answered no
//! lookup at all -- the members sat in the file and the link failed with
//! an undefined reference to a symbol the archive defines.
//!
//! The fix mirrors what `ld --whole-archive` and `ld` on an index-less
//! archive both do: when no `/` member is present, the index is rebuilt
//! from each member's own defined globals.
//!
//! The fixture builds the archive bytes by hand, so the shape under test
//! -- members, no `/`, no `//` -- is exact and not whatever the host `ar`
//! happens to write.
//!
//! Gated on `clang`; if it is missing the test prints a note and returns.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{icf::IcfMode, linker::link_to};

mod common;

/// The member that defines the symbol the program wants.
const LIB_SRC: &[u8] = b"int from_member(void) { return 3; }\n";

/// The program: one undefined reference only the archive can answer.
const MAIN_SRC: &[u8] =
    b"extern int from_member(void);\nvoid _start(void) { from_member(); }\n";

/// A link against a hand-built index-less archive resolves the member.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_archive_without_an_index_is_scanned() {
    let Some(dir) = workdir() else {
        return;
    };
    let Some((lib_obj, main_obj)) = compile(&dir) else {
        return;
    };
    let lib_bytes = fs::read(&lib_obj).expect("read the member object");
    let archive = build_indexless(&lib_bytes, "member.o");
    let arch_path = dir.join("libnox.a");
    fs::write(&arch_path, &archive).expect("write the archive");
    let out = dir.join("prog");
    let inputs = [arch_path, main_obj];
    let res = link_to(&inputs, &out, b"_start", false, IcfMode::None, false);
    assert!(
        res.is_ok(),
        "the member defines what the program wants: {:?}",
        res.err()
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir() -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping archive-no-index: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_arnoidx_{}", std::process::id()));
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
        eprintln!("skipping archive-no-index: clang cannot build {name}");
        return None;
    }
    Some(obj)
}

/// Packs `member` into a GNU archive with no `/` symbol index and no `//`
/// long-name table: the magic, one 60-byte header, the bytes, even padding.
fn build_indexless(member: &[u8], name: &str) -> Vec<u8> {
    let mut out = b"!<arch>\n".to_vec();
    let mut field = [b' '; 16];
    field[..name.len()].copy_from_slice(name.as_bytes());
    field[name.len()] = b'/';
    out.extend_from_slice(&field);
    out.extend_from_slice(b"0           ");
    out.extend_from_slice(b"0     ");
    out.extend_from_slice(b"0     ");
    out.extend_from_slice(b"644     ");
    out.extend_from_slice(format!("{:>10}", member.len()).as_bytes());
    out.extend_from_slice(b"`\n");
    out.extend_from_slice(member);
    if !member.len().is_multiple_of(2) {
        out.push(b'\n');
    }
    out
}
