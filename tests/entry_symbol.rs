//! An executable whose entry symbol is undefined does not link.
//!
//! `entry_addr` answered zero for an entry symbol nothing defined, and said
//! nothing. The image linked clean, `readelf -h` showed `Entry point address:
//! 0x0`, and the program died on `exec` -- the failure arriving at the point
//! furthest from its cause.
//!
//! lld warns here, and GNU ld warns and falls back to the start of `.text`.
//! xold stops. An executable that cannot start is not a smaller problem than a
//! command line the linker cannot honour, which it already refuses outright,
//! and neither fallback produces the image the caller asked for.
//!
//! A shared object legitimately has no entry: it is entered through its
//! symbols, so `e_entry` stays zero unless `--entry` named something.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    icf::IcfMode,
    linker::{link_shared, link_to},
};

mod common;

/// No `_start` anywhere.
const NO_ENTRY_SRC: &[u8] = b"int helper(int x) { return x + 1; }\n";

/// With one.
const WITH_ENTRY_SRC: &[u8] = b"int helper(int x) { return x + 1; }\n\
    void _start(void) { }\n";

/// An executable with no entry symbol fails, naming the symbol.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_missing_entry_symbol_fails_the_link() {
    let Some(dir) = workdir("missing") else {
        return;
    };
    let Some(obj) = compile(&dir, NO_ENTRY_SRC, "noentry") else {
        return;
    };
    let err = link_to(
        std::slice::from_ref(&obj),
        &dir.join("prog"),
        b"_start",
        false,
        IcfMode::None,
        false,
    )
    .expect_err(
        "an executable with no entry point links and then dies on exec, so \
         the link must stop instead",
    );
    let text = format!("{err}");
    assert!(
        text.contains("_start"),
        "the refusal must name the symbol it looked for, got {text:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A named `--entry` that nothing defines fails the same way, so the check is
/// about the symbol and not about the default name.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_named_entry_that_is_undefined_fails_too() {
    let Some(dir) = workdir("named") else {
        return;
    };
    let Some(obj) = compile(&dir, WITH_ENTRY_SRC, "withentry") else {
        return;
    };
    let err = link_to(
        std::slice::from_ref(&obj),
        &dir.join("prog"),
        b"no_such_entry",
        false,
        IcfMode::None,
        false,
    )
    .expect_err("an entry symbol that is not defined has no address");
    assert!(
        format!("{err}").contains("no_such_entry"),
        "the refusal must name what was asked for"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: the same object with an entry symbol links, and `e_entry` is
/// where the symbol is.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_defined_entry_symbol_still_links() {
    let Some(dir) = workdir("present") else {
        return;
    };
    let Some(obj) = compile(&dir, WITH_ENTRY_SRC, "withentry") else {
        return;
    };
    let out = dir.join("prog");
    let res = link_to(
        std::slice::from_ref(&obj),
        &out,
        b"_start",
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let bytes = fs::read(&out).expect("read output");
    assert_ne!(e_entry(&bytes), 0, "e_entry must name the entry symbol");
    let _ = fs::remove_dir_all(&dir);
}

/// And a shared object needs none: it is entered through its symbols.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_shared_object_needs_no_entry_symbol() {
    let Some(dir) = workdir("shared") else {
        return;
    };
    let Some(obj) = compile(&dir, NO_ENTRY_SRC, "noentry") else {
        return;
    };
    let out = dir.join("lib.so");
    let res = link_shared(
        std::slice::from_ref(&obj),
        &out,
        Some(b"libnoentry.so"),
        false,
        IcfMode::None,
        false,
    );
    assert!(
        res.is_ok(),
        "a shared object has no entry point: {:?}",
        res.err()
    );
    let bytes = fs::read(&out).expect("read output");
    assert_eq!(e_entry(&bytes), 0, "and its e_entry stays zero");
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping entry-symbol {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_entrysym_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles one fixture object.
fn compile(dir: &Path, src: &[u8], stem: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src_path = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src_path, src).ok()?;
    Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIC", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success()
        .then_some(obj)
}

// --- readers ---------------------------------------------------------------

/// The ELF header's `e_entry`.
fn e_entry(bytes: &[u8]) -> u64 {
    bytes
        .get(24..32)
        .and_then(|c| <[u8; 8]>::try_from(c).ok())
        .map_or(0, u64::from_le_bytes)
}
