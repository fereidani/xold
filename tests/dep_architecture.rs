//! A shared dependency must be built for the architecture being linked.
//!
//! The link target is folded from the relocatable inputs' `e_machine`. A
//! dependency never joined that vote and nothing else read its header, so an
//! `AArch64` `.so` on an x86-64 link contributed its exports, satisfied
//! references with them, and took a `DT_NEEDED` row of its own. The link
//! succeeded; the loader refused the image. That is the failure arriving as
//! far from its cause as it can get.
//!
//! lld rejects the mismatch when it opens the file, in `isCompatible`. The
//! dependency's machine is read during the parallel parse here, so the check
//! is one comparison per dependency and no second read.
//!
//! Gated on a `clang` that can target `aarch64`; without one the tests print a
//! note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
};

mod common;

const LIB_SRC: &[u8] = b"int cross_fn(void) { return 9; }\n";

const MAIN_SRC: &[u8] = b"extern int cross_fn(void);\n\
    int main(void) { return cross_fn() - 9; }\n\
    void _start(void) { }\n";

/// A dependency for another architecture ends the link, naming it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_foreign_dependency_fails_the_link() {
    let Some(dir) = workdir("foreign") else {
        return;
    };
    let Some(lib) = library(&dir, "aarch64-linux-gnu", "aa") else {
        return;
    };
    let Some(obj) = main_object(&dir) else {
        return;
    };
    let err = link(&dir, &obj, &lib, "foreign").expect_err(
        "an AArch64 dependency cannot satisfy an x86-64 link; the loader \
         would refuse the image, so the linker must",
    );
    let text = format!("{err}");
    assert!(
        text.contains("libcross.so"),
        "the refusal must name the dependency, got {text:?}"
    );
    assert!(
        text.contains("architecture"),
        "and say what disagreed, got {text:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: the same program links against the same library built for the
/// right architecture, so the check is about the machine and nothing else.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_matching_dependency_still_links() {
    let Some(dir) = workdir("matching") else {
        return;
    };
    let Some(lib) = library(&dir, "x86_64-linux-gnu", "x86") else {
        return;
    };
    let Some(obj) = main_object(&dir) else {
        return;
    };
    let res = link(&dir, &obj, &lib, "matching");
    assert!(
        res.is_ok(),
        "a same-architecture dependency must link: {:?}",
        res.err()
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping dep-architecture {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_deparch_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the dependency for `target` and links it with xold, so the fixture
/// needs no cross sysroot or cross runtime.
fn library(dir: &Path, target: &str, stem: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src, LIB_SRC).ok()?;
    let built = Command::new(clang)
        .args([&format!("--target={target}"), "-fPIC", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping dep-architecture: clang cannot target {target}");
        return None;
    }
    let lib = dir.join(format!("lib{stem}.so"));
    let res = link_shared(
        std::slice::from_ref(&obj),
        &lib,
        Some(b"libcross.so"),
        false,
        IcfMode::None,
        false,
    );
    assert!(
        res.is_ok(),
        "the fixture library must link: {:?}",
        res.err()
    );
    Some(lib)
}

/// Compiles the x86-64 program that references the library.
fn main_object(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("m.c");
    let obj = dir.join("m.o");
    fs::write(&src, MAIN_SRC).ok()?;
    Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIE", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success()
        .then_some(obj)
}

/// Links the program against the library as a dynamic executable.
fn link(
    dir: &Path,
    obj: &Path,
    lib: &Path,
    stem: &str,
) -> Result<(), xold::Error> {
    let interp = interpreter().unwrap_or_else(|| b"/lib64/ld.so".to_vec());
    link_dyn_exec(
        &[obj.to_path_buf(), lib.to_path_buf()],
        &dir.join(stem),
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    )
}
