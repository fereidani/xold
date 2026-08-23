//! `--gc-sections` keeps a `SHF_LINK_ORDER` section while its target lives.
//!
//! A link-order section is metadata about the section its `sh_link` names.
//! `-fpatchable-function-entry` is the case that matters: each function gets a
//! `__patchable_function_entries` row holding the address of its nop pad, and
//! `sh_link` says which function the row is about.
//!
//! Nothing relocates *to* such a section, so mark-and-sweep never reached one
//! and the sweep dropped every last one. The nop pads stayed in the live
//! functions and the table describing them did not, which is the whole of what
//! a runtime patcher reads. No diagnostic, and the failure only shows when
//! something tries to patch.
//!
//! lld follows the edge from the other side, enqueueing `dependentSections`
//! when it marks a section (`lld/ELF/MarkLive.cpp`). The rows for
//! dead functions still go: keeping the target alive is the condition, not
//! keeping everything.
//!
//! Gated on a `clang` that accepts `-fpatchable-function-entry`; without one
//! the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_to};

mod common;

/// Three functions, one of them unreachable. Every one gets a
/// `__patchable_function_entries` row; the dead one's row must go with it.
const SRC: &[u8] = b"__attribute__((noinline)) int live_fn(int x)\n\
    { return x + 1; }\n\
    __attribute__((noinline)) int dead_fn(int x) { return x + 2; }\n\
    int main(void) { return live_fn(41); }\n";

/// One 8-byte row per surviving function.
const ROW: u64 = 8;

/// The rows for the two live functions survive collection.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_link_order_section_survives_while_its_target_does() {
    let Some(dir) = workdir("keep") else {
        return;
    };
    let Some(obj) = compile(&dir) else {
        return;
    };
    let kept = patchable_bytes(&link(&dir, &obj, "gc", true));
    assert_eq!(
        kept,
        2 * ROW,
        "main and live_fn survive, so their two rows must too; dropping them \
         leaves the nop pads in the image with nothing describing them"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The row for the collected function goes with it, so the edge keeps what is
/// needed rather than pinning the whole table.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_row_of_a_collected_function_goes_too() {
    let Some(dir) = workdir("drop") else {
        return;
    };
    let Some(obj) = compile(&dir) else {
        return;
    };
    let all = patchable_bytes(&link(&dir, &obj, "nogc", false));
    let kept = patchable_bytes(&link(&dir, &obj, "gc", true));
    assert_eq!(all, 3 * ROW, "all three functions have a row without --gc");
    assert_eq!(
        kept,
        all - ROW,
        "exactly one row must go -- dead_fn's. Keeping all three would pin \
         the table, and keeping none is the defect: dropping every row was \
         also `kept < all`"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping gc-link-order {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_linkorder_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture with one section per function and a patch pad on each.
fn compile(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("patch.c");
    let obj = dir.join("patch.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args([
            "--target=x86_64-linux-gnu",
            "-fno-pic",
            "-ffunction-sections",
            "-fpatchable-function-entry=2",
            "-c",
        ])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping gc-link-order: no -fpatchable-function-entry");
        return None;
    }
    Some(obj)
}

/// Links `obj`, with or without collection, and returns the image bytes.
fn link(dir: &Path, obj: &Path, stem: &str, gc: bool) -> Vec<u8> {
    let out = dir.join(stem);
    let res = link_to(
        std::slice::from_ref(&obj.to_path_buf()),
        &out,
        b"main",
        gc,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).expect("read output")
}

// --- readers ---------------------------------------------------------------

/// The bytes contributed by `__patchable_function_entries`.
///
/// The sections are writable and allocated, so they land in `.data`, which in
/// this fixture holds nothing else -- the program has no other writable data.
/// Measuring `.data` therefore measures the table.
fn patchable_bytes(bytes: &[u8]) -> u64 {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return 0;
    };
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == b".data")
        .map_or(0, |s| s.sh_size.get())
}
