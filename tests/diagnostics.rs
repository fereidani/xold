//! Three diagnostics that named too little to act on.
//!
//! A duplicate definition reported the symbol and neither of the two files
//! that defined it, though both indices are in hand where the clash is found.
//! lld prints both definers.
//!
//! An `AArch64` dynamic-TLS reference reported `unsupported relocation 563`:
//! true, and useless, since it names neither the symbol nor the fact that it
//! is a thread-local. The refusal is deliberate -- this linker does not lower
//! general-dynamic or descriptor TLS -- but gcc defaults to TLSDESC on that
//! target, so the bare number is the first thing a user with any TLS at all
//! sees. Routing those types through the rewrite path, as x86-64 already
//! does, makes the refusal name the variable.
//!
//! And a section table too long for `e_shnum` wrote a zero there, which means
//! "read the count from `sh_size` of header zero" -- an escape this linker
//! does not implement, so the header described a table that was not there.
//! That one is refused rather than written; there is no fixture small enough
//! to build it here, so only the first two are tested.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{icf::IcfMode, linker::link_to};

const DUP_A: &[u8] = b"int shared_def = 1;\n\
    void _start(void) { }\n";
const DUP_B: &[u8] = b"int shared_def = 2;\n";

/// A thread-local reached the only way gcc reaches one by default.
const TLS_SRC: &[u8] = b"_Thread_local int tv;\n\
    int get(void) { return tv; }\n\
    void _start(void) { }\n";

mod common;

/// The duplicate-symbol error names both definers.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_duplicate_definition_names_both_files() {
    let Some(dir) = workdir("dup") else {
        return;
    };
    let Some(a) = compile(&dir, DUP_A, "a", "x86_64-linux-gnu") else {
        return;
    };
    let Some(b) = compile(&dir, DUP_B, "b", "x86_64-linux-gnu") else {
        return;
    };
    let err = link(&dir, &[a, b], "dup")
        .expect_err("two strong definitions of one name is an error");
    let text = format!("{err}");
    assert!(
        text.contains("shared_def"),
        "the message names the symbol, got {text:?}"
    );
    assert!(
        text.contains("a.o") && text.contains("b.o"),
        "and both files that define it, got {text:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The dynamic-TLS refusal names the thread-local.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_unlowered_tls_reference_names_the_variable() {
    let Some(dir) = workdir("tls") else {
        return;
    };
    let Some(obj) = compile(&dir, TLS_SRC, "t", "aarch64-linux-gnu") else {
        return;
    };
    let err = link(&dir, &[obj], "tls").expect_err(
        "this linker does not lower descriptor TLS, which is what gcc emits \
         by default on this target",
    );
    let text = format!("{err}");
    assert!(
        text.contains("'tv'"),
        "the refusal must name the thread-local, got {text:?}"
    );
    assert!(
        text.contains("thread-local"),
        "and say what kind of thing it is, got {text:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping diagnostics {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_diag_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles one fixture for `target`.
fn compile(
    dir: &Path,
    src: &[u8],
    stem: &str,
    target: &str,
) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src_path = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clang)
        .args([&format!("--target={target}"), "-fPIC", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping diagnostics: clang cannot target {target}");
        return None;
    }
    Some(obj)
}

/// Links the objects as a static executable.
fn link(dir: &Path, objs: &[PathBuf], stem: &str) -> Result<(), xold::Error> {
    link_to(
        objs,
        &dir.join(stem),
        b"_start",
        false,
        IcfMode::None,
        false,
    )
}
