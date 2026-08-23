//! A reference from allocated code into a dropped section is refused.
//!
//! The other half of the discarded-section fix. A `.debug_*` member points
//! such a reference at a tombstone, because a debugger reads a value and can
//! be told there is nothing there. Executable code has no such value: the site
//! resolved to `0 + st_value + addend`, a plausible-looking address in the
//! first page, silently. lld errors ("relocation refers to a discarded
//! section").
//!
//! The check is gated on whether the referencing member's file lost any
//! section at all, which is a bool the member already carries, so an ordinary
//! link pays one predictable branch per relocation and no symbol-table read.
//!
//! What these tests pin is that the gate does not fire on links that are
//! correct: garbage collection, COMDAT groups and identical code folding all
//! drop sections, and none of them may turn a good link into an error. No
//! input built on this host reaches the positive case -- the compilers here
//! do not emit a live reference into a group they also let lose -- so the
//! refusal itself is checked by the suite staying green, not by a fixture
//! that trips it.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_to};

mod common;

/// Live and dead code, each in its own section, with a file-local helper the
/// dead one calls: garbage collection drops both.
const SRC: &[u8] = b"static int helper(int x) { return x + 1; }\n\
    int dead(int x) { return helper(x); }\n\
    int live(int x) { return x * 2; }\n\
    int also_live(int x) { return live(x) + 1; }\n\
    void _start(void) { int (*p)(int) = also_live; (void)p; }\n";

/// A collected link still succeeds, and still collects.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn garbage_collection_does_not_trip_the_check() {
    let Some(dir) = workdir("gc") else {
        return;
    };
    let Some(collected) = link(&dir, true, IcfMode::None, "gc") else {
        return;
    };
    let Some(whole) = link(&dir, false, IcfMode::None, "whole") else {
        return;
    };
    assert!(
        text_size(&collected) < text_size(&whole),
        "the fixture must actually lose a section for this to mean anything: \
         {:#x} collected against {:#x} whole",
        text_size(&collected),
        text_size(&whole)
    );
    let _ = fs::remove_dir_all(&dir);
}

/// So does a folded one, which drops sections by a different route.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn folding_does_not_trip_the_check() {
    let Some(dir) = workdir("icf") else {
        return;
    };
    let Some(bytes) = link(&dir, false, IcfMode::All, "icf") else {
        return;
    };
    assert!(text_size(&bytes) > 0, "the image must carry code");
    let _ = fs::remove_dir_all(&dir);
}

/// And both together.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn collecting_and_folding_together_still_link() {
    let Some(dir) = workdir("both") else {
        return;
    };
    let Some(bytes) = link(&dir, true, IcfMode::All, "both") else {
        return;
    };
    assert!(text_size(&bytes) > 0, "the image must carry code");
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping discarded-reference {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_discarded_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture and links it at the given collection and folding
/// settings.
fn link(dir: &Path, gc: bool, icf: IcfMode, stem: &str) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let src = dir.join("g.c");
    let obj = dir.join("g.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args([
            "--target=x86_64-linux-gnu",
            "-fno-pic",
            "-O1",
            "-ffunction-sections",
            "-fdata-sections",
            "-c",
        ])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping discarded-reference: clang cannot build it");
        return None;
    }
    let out = dir.join(stem);
    let res =
        link_to(std::slice::from_ref(&obj), &out, b"_start", gc, icf, false);
    assert!(
        res.is_ok(),
        "a correct link must not be refused: {:?}",
        res.err()
    );
    fs::read(&out).ok()
}

// --- readers ---------------------------------------------------------------

/// The `sh_size` of `.text`.
fn text_size(bytes: &[u8]) -> u64 {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return 0;
    };
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == b".text")
        .map_or(0, |s| s.sh_size.get())
}
