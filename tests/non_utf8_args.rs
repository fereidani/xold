//! An input path that is not valid UTF-8 still links.
//!
//! `main` collected the command line with `env::args()`, which panics on
//! any argument that is not valid UTF-8. A filename in any of the many
//! byte-oriented locales -- a latin-1 accented name, a Shift-JIS name --
//! killed the link with a panic before a single input was opened. lld
//! keeps every argument as bytes and only ever compares flag spellings,
//! which are ASCII (`lld/Common/Args.cpp`).
//!
//! The fix keeps each argument as an `OsString`: a non-UTF-8 argument
//! cannot be an option (every flag this linker knows is ASCII), so it is
//! an input path and reaches the file system with its bytes intact.
//!
//! Gated on `clang`; if it is missing the test prints a note and returns.

use std::{
    ffi::OsString,
    fs,
    os::unix::ffi::OsStringExt,
    path::{Path, PathBuf},
    process::Command,
};

use common::{which, xold_bin};

mod common;

const SRC: &[u8] =
    b"int ext(void) { return 7; }\nint _start(void) { return ext(); }\n";

/// The object is filed under a name with a byte that is not valid UTF-8,
/// and the link naming it on the command line succeeds.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_non_utf8_input_path_links() {
    let Some(dir) = workdir() else {
        return;
    };
    let Some(obj) = compile(&dir) else {
        return;
    };
    let out = dir.join("prog");
    let ok = Command::new(xold_bin())
        .arg(&obj)
        .arg("--entry")
        .arg("_start")
        .arg("-o")
        .arg(&out)
        .status()
        .expect("xold must run")
        .success();
    assert!(ok, "a path that is not UTF-8 is a path, not a panic");
    assert!(out.exists(), "the image was written");
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir() -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping non-utf8-args: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_nonutf8_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture under a name carrying the byte `0xff`.
fn compile(dir: &Path) -> Option<OsString> {
    let clang = which("clang")?;
    let src = dir.join("plain.c");
    fs::write(&src, SRC).ok()?;
    let obj = OsString::from_vec(vec![b'a', 0xff, b'.', b'o']);
    let obj_path = dir.join(&obj);
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj_path)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping non-utf8-args: clang cannot build the fixture");
        return None;
    }
    Some(obj_path.into_os_string())
}
