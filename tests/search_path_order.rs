//! Every `-L` serves every `-l`, wherever it sits on the command line.
//!
//! GNU ld documents that all `-L` directories apply to all libraries, and lld
//! gathers them in `readConfigs` before it opens a single input. xold resolved
//! each `-l` against only the directories seen so far.
//!
//! `main.o -lfoo -L/opt/lib` therefore either failed outright or, when a
//! library of that name also existed in a default directory, quietly picked
//! that one up instead -- a different library, possibly a different ABI, with
//! no diagnostic. `--sysroot` already got a pre-pass for exactly this reason;
//! `-L` now gets the same one.
//!
//! These tests drive the built `xold` binary, because the ordering is a
//! command-line property and the library API takes resolved paths.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{which, xold_bin};

mod common;

const LIB_SRC: &[u8] = b"int searched_fn(void) { return 5; }\n";

/// `_start` is here because an executable with no entry point no longer
/// links, and these tests are about where the library came from.
const MAIN_SRC: &[u8] = b"extern int searched_fn(void);\n\
    int main(void) { return searched_fn() - 5; }\n\
    void _start(void) { }\n";

/// `-L` after the `-l` it serves still resolves the library.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_search_directory_after_the_library_still_serves_it() {
    let Some(dir) = workdir("after") else {
        return;
    };
    let Some(()) = build(&dir) else {
        return;
    };
    let out = link(&dir, &["-lsearched", &lflag(&dir)]);
    assert!(
        out.0,
        "a -L after the -l it serves must still resolve it: {}",
        out.1
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The two orders produce the same image, so the directory is a set member
/// rather than a position on the line.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn both_orders_produce_the_same_image() {
    let Some(dir) = workdir("same") else {
        return;
    };
    let Some(()) = build(&dir) else {
        return;
    };
    let before = link_to_file(&dir, &[&lflag(&dir), "-lsearched"], "before");
    let after = link_to_file(&dir, &["-lsearched", &lflag(&dir)], "after");
    let (Some(a), Some(b)) = (before, after) else {
        panic!("both orders must link");
    };
    assert_eq!(a, b, "the order of -L and -l must not change the image");
    let _ = fs::remove_dir_all(&dir);
}

/// The control: with no `-L` at all the library is not found, so the search
/// path is what resolves it and not some fallback.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn without_the_directory_the_library_is_not_found() {
    let Some(dir) = workdir("none") else {
        return;
    };
    let Some(()) = build(&dir) else {
        return;
    };
    let out = link(&dir, &["-lsearched"]);
    assert!(
        !out.0,
        "with no -L the library must not resolve, or these tests prove \
         nothing about the search path"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping search-path-order {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_searchpath_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    fs::create_dir_all(dir.join("libs")).ok()?;
    Some(dir)
}

/// The `-L` flag naming this fixture's library directory.
fn lflag(dir: &Path) -> String {
    format!("-L{}", dir.join("libs").display())
}

/// Builds the library and the object that references it.
fn build(dir: &Path) -> Option<()> {
    let clang = which("clang")?;
    let lib_src = dir.join("searched.c");
    fs::write(&lib_src, LIB_SRC).ok()?;
    let built = Command::new(&clang)
        .args(["-fPIC", "-shared", "-Wl,-soname,libsearched.so"])
        .arg(&lib_src)
        .arg("-o")
        .arg(dir.join("libs").join("libsearched.so"))
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping search-path-order: clang cannot build the library");
        return None;
    }
    let main_src = dir.join("main.c");
    fs::write(&main_src, MAIN_SRC).ok()?;
    Command::new(&clang)
        .args(["--target=x86_64-linux-gnu", "-fPIE", "-c"])
        .arg(&main_src)
        .arg("-o")
        .arg(dir.join("main.o"))
        .status()
        .ok()?
        .success()
        .then_some(())
}

/// Runs `xold` with the given library-related flags, returning success and the
/// diagnostic it printed.
fn link(dir: &Path, flags: &[&str]) -> (bool, String) {
    run(dir, flags, "prog")
}

/// The same, returning the linked bytes when it succeeded.
fn link_to_file(dir: &Path, flags: &[&str], stem: &str) -> Option<Vec<u8>> {
    let (ok, _) = run(dir, flags, stem);
    ok.then(|| fs::read(dir.join(stem)).expect("read output"))
}

/// One `xold` invocation over the fixture.
fn run(dir: &Path, flags: &[&str], stem: &str) -> (bool, String) {
    let out = Command::new(xold_bin())
        .arg("--dynamic-exec")
        .arg("-dynamic-linker")
        .arg("/lib64/ld-linux-x86-64.so.2")
        .arg("-o")
        .arg(dir.join(stem))
        .arg(dir.join("main.o"))
        .args(flags)
        .arg("-lc")
        .output()
        .expect("xold must run");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).trim().to_string(),
    )
}
