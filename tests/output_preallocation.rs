//! The output file's blocks are reserved before anything is written to it.
//!
//! `set_len` makes a sparse file: the blocks are allocated when the pages
//! fault in during writing, and a failed allocation there -- a full disk, a
//! quota -- kills the process with SIGBUS. No message, and the temporary left
//! behind. mold preallocates for this reason, and lld's `FileOutputBuffer`
//! reports the error instead of taking the signal.
//!
//! These tests do not discriminate, and say so rather than implying
//! otherwise: the writer fills every byte of the image, so the blocks end up
//! allocated either way -- the difference is only whether the allocation can
//! report a failure or has to take a signal, and filling a disk to find out
//! is not something a test suite here can do. What they pin is the absence of
//! a regression: the image is still whole, still an ELF file, and still
//! occupies the blocks its size implies on a filesystem that reports them.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{icf::IcfMode, linker::link_to};

mod common;

/// Enough data that the block count is unambiguous: a sparse file of this size
/// reports far fewer blocks than a reserved one.
const SRC: &[u8] = b"long big[65536] = { 1, 2, 3 };\n\
    long touch(void) { return big[65535]; }\n\
    void _start(void) { }\n";

/// The image occupies the blocks its size implies.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_image_is_not_sparse() {
    let Some(dir) = workdir("blocks") else {
        return;
    };
    let Some(path) = link(&dir) else {
        return;
    };
    let meta = fs::metadata(&path).expect("the image must exist");
    let size = meta.len();
    assert!(size > 256 * 1024, "the fixture must be large: {size} bytes");
    // `blocks()` counts 512-byte units regardless of the filesystem's own
    // block size.
    let held = meta.blocks() * 512;
    if held == 0 {
        eprintln!("skipping: this filesystem reports no block count");
        let _ = fs::remove_dir_all(&dir);
        return;
    }
    assert!(
        held * 2 >= size,
        "the image holds {held} bytes of blocks for {size} bytes of file: \
         the reservation did not happen, and a full disk during writing would \
         be a SIGBUS rather than an error"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// And it still contains what it should.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_image_is_still_correct() {
    let Some(dir) = workdir("content") else {
        return;
    };
    let Some(path) = link(&dir) else {
        return;
    };
    let bytes = fs::read(&path).expect("readable");
    assert_eq!(
        bytes.get(..4),
        Some(b"\x7fELF".as_slice()),
        "preallocating must not disturb the bytes"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping output-preallocation {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_prealloc_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture and links it, returning the output path.
fn link(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("b.c");
    let obj = dir.join("b.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fno-pic", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping output-preallocation: clang cannot build it");
        return None;
    }
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
    Some(out)
}
