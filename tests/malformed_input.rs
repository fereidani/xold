//! Bounds and alignment an input file gets to choose.
//!
//! Four numbers in an object file are the file's to state and the linker's to
//! doubt: a section's offset, its size, its entry count and its alignment. All
//! four were used in arithmetic that assumed they were sane.
//!
//! `offset + size` and `count * entsize` were added and multiplied plainly, in
//! a module whose own header says nothing there panics -- so under a profile
//! with overflow checks they did, and without them they wrapped to a small,
//! in-bounds range and the view succeeded over the wrong bytes.
//!
//! `sh_addralign` reached the placement cursors unvalidated. The rounding is a
//! mask, and a mask only describes the alignment it came from when exactly one
//! bit is set: a section declaring 24 was quietly aligned to 8. A value near
//! `u64::MAX` overflowed the rounding itself.
//!
//! Every one of these is a malformed or hostile input rather than something a
//! compiler writes, which is exactly why the answer has to be a diagnostic and
//! not a panic or a wrong view.
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

const SRC: &[u8] = b"int datum = 7;\n\
    void _start(void) { }\n";

/// A section whose offset and size overflow when added is refused.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_overflowing_span_is_refused() {
    let Some(dir) = workdir("span") else {
        return;
    };
    let Some(obj) = patched(&dir, "span", Patch::Span) else {
        return;
    };
    let err = link(&dir, &obj, "span")
        .expect_err("a span that cannot be added is not a span");
    assert!(
        matches!(err, xold::Error::OutOfRange(_) | xold::Error::Format(_)),
        "the refusal must be a range or format error, got {err:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A non-power-of-two alignment is refused rather than rounded down to one.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_non_power_of_two_alignment_is_refused() {
    let Some(dir) = workdir("align") else {
        return;
    };
    let Some(obj) = patched(&dir, "align", Patch::Align(24)) else {
        return;
    };
    let err = link(&dir, &obj, "align").expect_err(
        "24 is not an alignment a mask can express, and masking it silently \
         aligns to 8",
    );
    assert!(
        matches!(err, xold::Error::Format(_)),
        "the refusal must be a format error, got {err:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// So is one large enough to overflow the rounding.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_enormous_alignment_is_refused() {
    let Some(dir) = workdir("huge") else {
        return;
    };
    let Some(obj) = patched(&dir, "huge", Patch::Align(1 << 63)) else {
        return;
    };
    let err = link(&dir, &obj, "huge")
        .expect_err("no section needs eight exabytes of alignment");
    assert!(
        matches!(err, xold::Error::Format(_)),
        "the refusal must be a format error, got {err:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: the unpatched object links, so the checks reject malformed
/// inputs and nothing else.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_unpatched_object_still_links() {
    let Some(dir) = workdir("plain") else {
        return;
    };
    let Some(obj) = patched(&dir, "plain", Patch::None) else {
        return;
    };
    let res = link(&dir, &obj, "plain");
    assert!(res.is_ok(), "an ordinary object must link: {:?}", res.err());
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// What to corrupt in the fixture's `.data` section header.
#[derive(Clone, Copy)]
enum Patch {
    None,
    /// `sh_size` set so `sh_offset + sh_size` overflows.
    Span,
    /// `sh_addralign` set to this value.
    Align(u64),
}

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping malformed-input {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_malformed_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture and applies `patch` to its `.data` section header.
///
/// Patching the header is exact: the object is otherwise the one clang wrote,
/// so what the link reacts to is the one field under test.
fn patched(dir: &Path, stem: &str, patch: Patch) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("m.c");
    let obj = dir.join(format!("{stem}.o"));
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
        eprintln!("skipping malformed-input: clang cannot build the fixture");
        return None;
    }
    if matches!(patch, Patch::None) {
        return Some(obj);
    }
    let mut bytes = fs::read(&obj).ok()?;
    let at = data_header_offset(&bytes)?;
    // Elf64_Shdr: sh_addralign at +32 within the second half, sh_size at +32
    // from the start. Both are 8-byte little-endian fields.
    let (field, value) = match patch {
        Patch::Span => (32, u64::MAX - 8),
        Patch::Align(a) => (48, a),
        Patch::None => return Some(obj),
    };
    bytes
        .get_mut(at + field..at + field + 8)?
        .copy_from_slice(&value.to_le_bytes());
    fs::write(&obj, bytes).ok()?;
    Some(obj)
}

/// Links the object as a static executable.
fn link(dir: &Path, obj: &Path, stem: &str) -> Result<(), xold::Error> {
    link_to(
        std::slice::from_ref(&obj.to_path_buf()),
        &dir.join(stem),
        b"_start",
        false,
        IcfMode::None,
        false,
    )
}

// --- readers ---------------------------------------------------------------

/// The file offset of `.data`'s section header.
fn data_header_offset(bytes: &[u8]) -> Option<usize> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let index = obj
        .sections()
        .iter()
        .position(|s| obj.section_name(s) == b".data")?;
    let shoff = usize::try_from(read_u64(bytes, 0x28)?).ok()?;
    let entsize = usize::from(read_u16(bytes, 0x3a)?);
    Some(shoff + index * entsize)
}

fn read_u16(bytes: &[u8], at: usize) -> Option<u16> {
    bytes
        .get(at..at + 2)
        .and_then(|c| <[u8; 2]>::try_from(c).ok())
        .map(u16::from_le_bytes)
}

fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
    bytes
        .get(at..at + 8)
        .and_then(|c| <[u8; 8]>::try_from(c).ok())
        .map(u64::from_le_bytes)
}
