//! Two latent holes in identical code folding.
//!
//! Neither is reachable today, and both are the kind that stops being latent
//! the moment something nearby changes.
//!
//! The equality test compared flags, content and relocations but never which
//! output section the two candidates land in. Every eligible candidate routes
//! to `.text` at the moment, so it never decided anything -- but nothing
//! enforced that, and a future split (`.text.hot` beside `.text`) would have
//! silently permitted a fold across two regions, moving one of them out of the
//! range its references measure against. lld opens its comparison with the
//! same question, `a->getParent() != b->getParent()`.
//!
//! The alias map resolved one hop. Two passes write into it -- identical code
//! folding and the merge pool -- and their eligibility sets overlap, so an
//! X to R to C chain is buildable; resolving it a hop at a time leaves the
//! answer depending on which entry the reader reached first, and the reader
//! walks a hash map. Entries are flattened on insert now, so every one names a
//! final representative and the walk has no order to depend on.
//!
//! These tests pin what is observable: folding still folds, the folded copies
//! resolve to one address, and the result does not move with the thread count.
//! Neither hole is reproducible from an input, which is what "latent" means.
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

/// Four identical functions across two output-section kinds, plus a mergeable
/// string section -- the shapes whose eligibility sets overlap.
const SRC: &[u8] = b"const char *s1(void) { return \"pooled\"; }\n\
    const char *s2(void) { return \"pooled\"; }\n\
    int f1(int x) { return x * 11; }\n\
    int f2(int x) { return x * 11; }\n\
    int f3(int x) { return x * 11; }\n\
    int use(int x) { return f1(x) + f2(x) + f3(x); }\n\
    void _start(void) { }\n";

/// The folded copies all resolve to one address.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn every_folded_copy_resolves_to_the_representative() {
    let Some(dir) = workdir("fold") else {
        return;
    };
    let Some(bytes) = link(&dir, IcfMode::All, "all") else {
        return;
    };
    let f1 = symbol_value(&bytes, b"f1").expect("f1 present");
    let f2 = symbol_value(&bytes, b"f2").expect("f2 present");
    let f3 = symbol_value(&bytes, b"f3").expect("f3 present");
    assert_eq!(f1, f2, "identical functions fold onto one address");
    assert_eq!(
        f2, f3,
        "and a third resolves to the same one, not to whichever entry the \
         map was read at first"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Folding still happens: the image is smaller than without it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn folding_still_folds() {
    let Some(dir) = workdir("size") else {
        return;
    };
    let Some(folded) = link(&dir, IcfMode::All, "all") else {
        return;
    };
    let Some(whole) = link(&dir, IcfMode::None, "none") else {
        return;
    };
    assert!(
        text_size(&folded) < text_size(&whole),
        "comparing the output section must not make everything ineligible: \
         {:#x} folded against {:#x} unfolded",
        text_size(&folded),
        text_size(&whole)
    );
    let _ = fs::remove_dir_all(&dir);
}

/// And the answer does not depend on the thread count, which is where a
/// hash-order dependency would show.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_folded_image_is_the_same_at_every_thread_count() {
    let Some(dir) = workdir("threads") else {
        return;
    };
    let Some(first) = link_with_threads(&dir, "t1", "1") else {
        return;
    };
    for threads in ["2", "4", "8", "16"] {
        let Some(other) = link_with_threads(&dir, threads, threads) else {
            return;
        };
        assert_eq!(
            first, other,
            "the image must be identical at {threads} threads"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping icf-fold-map {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_foldmap_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the fixture and links it at the given folding mode.
fn link(dir: &Path, icf: IcfMode, stem: &str) -> Option<Vec<u8>> {
    let obj = compile(dir)?;
    let out = dir.join(stem);
    let res = link_to(
        std::slice::from_ref(&obj),
        &out,
        b"_start",
        false,
        icf,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

/// The same, in a child process pinned to a thread count.
fn link_with_threads(dir: &Path, stem: &str, threads: &str) -> Option<Vec<u8>> {
    let obj = compile(dir)?;
    let out = dir.join(format!("th_{stem}"));
    let ok = Command::new(common::xold_bin())
        .env("RAYON_NUM_THREADS", threads)
        .arg("--icf=all")
        .arg("-o")
        .arg(&out)
        .arg(&obj)
        .arg("--entry")
        .arg("_start")
        .status()
        .ok()?
        .success();
    assert!(ok, "the fixture must link at {threads} threads");
    fs::read(&out).ok()
}

/// Compiles the fixture with one section per function and per datum.
fn compile(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("f.c");
    let obj = dir.join("f.o");
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
        eprintln!("skipping icf-fold-map: clang cannot build the fixture");
        return None;
    }
    Some(obj)
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

/// The `st_value` of a symbol in the output's `.symtab`.
fn symbol_value(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab
        .iter()
        .find(|s| symtab.name(s) == name)
        .map(|s| s.st_value.get())
}
