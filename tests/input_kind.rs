//! Only a relocatable object can be linked into an image.
//!
//! A finished executable named on the command line was read as though it were
//! a `.o`. Its symbols carry absolute addresses, and that path reads a
//! symbol's value as an offset within its section, so the definitions came out
//! as nonsense -- or, when a name matched a real input's, as a
//! duplicate-symbol error pointing at the wrong thing.
//!
//! `e_type` was never checked; `is_relocatable` existed and had no callers. A
//! shared object is recognised before this point and taken as a dependency,
//! so what reaches here and is not `ET_REL` is a mistake worth saying out
//! loud. lld answers it in one line, "unknown file type".
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{icf::IcfMode, linker::link_to};

mod common;

const SRC: &[u8] = b"int helper(int x) { return x + 1; }\n\
    void _start(void) { }\n";

/// An executable in the input list ends the link.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_linked_executable_is_not_an_input_object() {
    let Some(dir) = workdir("exe") else {
        return;
    };
    let Some(obj) = compile(&dir) else {
        return;
    };
    let exe = dir.join("first");
    link(std::slice::from_ref(&obj), &exe)
        .expect("the ordinary link must succeed");

    let err = link(&[obj, exe], &dir.join("second")).expect_err(
        "a finished executable has absolute symbol values, which this path \
         reads as section offsets; the link must stop rather than invent \
         definitions from them",
    );
    assert!(
        matches!(err, xold::Error::Format(_)),
        "the refusal must be a format error, got {err:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: the object alone still links, so the check is about the file
/// kind and not about the input list.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_relocatable_object_still_links() {
    let Some(dir) = workdir("rel") else {
        return;
    };
    let Some(obj) = compile(&dir) else {
        return;
    };
    let res = link(&[obj], &dir.join("prog"));
    assert!(res.is_ok(), "an ET_REL input must link: {:?}", res.err());
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping input-kind {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_inputkind_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture object.
fn compile(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("m.c");
    let obj = dir.join("m.o");
    fs::write(&src, SRC).ok()?;
    Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fno-pic", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success()
        .then_some(obj)
}

/// Links the given inputs as a static executable.
fn link(paths: &[PathBuf], out: &Path) -> Result<(), xold::Error> {
    link_to(paths, out, b"_start", false, IcfMode::None, false)
}
