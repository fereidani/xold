//! Response files, group markers, and the spellings compilers actually emit.
//!
//! Three ways a real build reached this linker and was turned away.
//!
//! A driver that would exceed `ARG_MAX` -- which any large link does, and
//! which is why gcc and cmake do it -- writes the command line to a file and
//! passes `@file`. That reached the input list as a path, and the link died
//! reporting a format error about a file nobody named.
//!
//! `--start-group`/`--end-group` were refused as unimplemented although the
//! semantics already held: archives are iterated to a fixpoint, which is a
//! superset of what a group asks for. `gcc -static` emits them
//! unconditionally.
//!
//! And several spellings of options that do exist were refused -- `-e`,
//! `-soname=x`, `-dynamic-linker=x`, an attached `-ofile`. `-Wl,-soname=x` is
//! what compilers write.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{which, xold_bin};

mod common;

const SRC: &[u8] = b"void _start(void) { }\n";

/// A response file supplies the whole command line.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_response_file_supplies_the_arguments() {
    let Some(dir) = workdir("rsp") else {
        return;
    };
    let Some(obj) = compile(&dir) else {
        return;
    };
    let out = dir.join("prog");
    let rsp = dir.join("args.rsp");
    // Quoted, because a real driver quotes paths that may hold spaces.
    fs::write(
        &rsp,
        format!(
            "-o \"{}\"\n\"{}\"\n--entry _start\n",
            out.display(),
            obj.display()
        ),
    )
    .expect("write response file");

    let (ok, err) = xold(&[format!("@{}", rsp.display())]);
    assert!(ok, "a response file must be expanded, not linked: {err}");
    assert!(out.exists(), "and the link must produce its output");
    let _ = fs::remove_dir_all(&dir);
}

/// One response file may name another.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_response_file_may_name_another() {
    let Some(dir) = workdir("nested") else {
        return;
    };
    let Some(obj) = compile(&dir) else {
        return;
    };
    let out = dir.join("prog");
    let inner = dir.join("inner.rsp");
    fs::write(
        &inner,
        format!(
            "-o \"{}\" \"{}\" --entry _start\n",
            out.display(),
            obj.display()
        ),
    )
    .expect("write inner");
    let outer = dir.join("outer.rsp");
    fs::write(&outer, format!("@{}\n", inner.display())).expect("write outer");

    let (ok, err) = xold(&[format!("@{}", outer.display())]);
    assert!(ok, "a nested response file must be followed: {err}");
    assert!(out.exists(), "and the link must produce its output");
    let _ = fs::remove_dir_all(&dir);
}

/// A response file that names itself stops with a message that says so,
/// rather than recursing.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_self_naming_response_file_is_bounded() {
    let Some(dir) = workdir("cycle") else {
        return;
    };
    let rsp = dir.join("self.rsp");
    fs::write(&rsp, format!("@{}\n", rsp.display())).expect("write");
    let (ok, err) = xold(&[format!("@{}", rsp.display())]);
    assert!(!ok, "the link must stop");
    assert!(err.contains("nested"), "and say what happened, got {err:?}");
    let _ = fs::remove_dir_all(&dir);
}

/// Group markers are accepted, because the fixpoint already provides what
/// they ask for.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn group_markers_are_accepted() {
    let Some(dir) = workdir("group") else {
        return;
    };
    let Some(obj) = compile(&dir) else {
        return;
    };
    let out = dir.join("prog");
    let (ok, err) = xold(&[
        "--start-group".into(),
        obj.display().to_string(),
        "--end-group".into(),
        "-o".into(),
        out.display().to_string(),
        "--entry".into(),
        "_start".into(),
    ]);
    assert!(ok, "gcc -static writes these unconditionally: {err}");
    assert!(out.exists(), "and the link must produce its output");
    let _ = fs::remove_dir_all(&dir);
}

/// The short and attached spellings work.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_alternative_spellings_are_recognised() {
    let Some(dir) = workdir("spelling") else {
        return;
    };
    let Some(obj) = compile(&dir) else {
        return;
    };
    let out = dir.join("prog");
    let (ok, err) = xold(&[
        "-e".into(),
        "_start".into(),
        format!("-o{}", out.display()),
        obj.display().to_string(),
    ]);
    assert!(
        ok,
        "`-e` and an attached `-o` are ordinary ld spellings: {err}"
    );
    assert!(out.exists(), "and the link must produce its output");
    let _ = fs::remove_dir_all(&dir);
}

/// So does the attached `-soname=`, which is what `-Wl,-soname=x` becomes.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_attached_soname_spelling_is_recognised() {
    let Some(dir) = workdir("soname") else {
        return;
    };
    let Some(obj) = compile(&dir) else {
        return;
    };
    let out = dir.join("lib.so");
    let (ok, err) = xold(&[
        "-shared".into(),
        "-soname=libnamed.so".into(),
        "-o".into(),
        out.display().to_string(),
        obj.display().to_string(),
    ]);
    assert!(ok, "compilers write this form: {err}");
    assert!(out.exists(), "and the link must produce its output");
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping command-line-forms {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_cmdforms_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture object.
fn compile(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("r.c");
    let obj = dir.join("r.o");
    fs::write(&src, SRC).ok()?;
    Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIC", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success()
        .then_some(obj)
}

/// Runs the built linker with the given arguments.
fn xold(args: &[String]) -> (bool, String) {
    let out = Command::new(xold_bin())
        .args(args)
        .output()
        .expect("xold must run");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).trim().to_string(),
    )
}
