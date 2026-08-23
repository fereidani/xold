//! A link whose only allocated content is unwind data still places
//! `.eh_frame` after the headers.
//!
//! `.eh_frame` was anchored on the ends of `.rodata`, `.plt`, `.fini` and
//! `.text`. Every one of those regions reports offset zero while it is
//! unplaced, so an input contributing none of them handed `.eh_frame` the
//! start of the file -- the ELF header's own bytes. The placement cursor is
//! the running high-water mark of everything already placed and never drops
//! below the end of the headers, so anchoring on it is what keeps the two
//! apart.
//!
//! An empty `.text` is not enough to show this: it still takes a region and
//! reports a non-zero end. The section has to be absent, which is why the
//! fixture strips it.
//!
//! Gated on `as` and `objcopy`; if either is missing the test prints a note
//! and returns.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{icf::IcfMode, linker::link_to};

mod common;

/// An absolute `_start` keeps the entry symbol defined without contributing a
/// single byte of text, so `.eh_frame` is the only allocated section left once
/// `.text` is stripped.
const SRC: &[u8] = b"    .globl _start\n\
    .set _start, 0x401000\n\
    .section .eh_frame,\"a\",@progbits\n\
    .long 0\n";

/// The image keeps its ELF header and puts `.eh_frame` after it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_unwind_only_link_keeps_eh_frame_off_the_elf_header() {
    let Some(obj) = fixture() else {
        return;
    };
    let out = obj.with_file_name("unwind-only.out");
    let res = link_to(
        std::slice::from_ref(&obj),
        &out,
        b"_start",
        false,
        IcfMode::None,
        false,
    );
    assert!(
        res.is_ok(),
        "an unwind-only link must succeed: {:?}",
        res.err()
    );

    let image = fs::read(&out).expect("the image is readable");
    assert!(
        image.starts_with(b"\x7fELF"),
        "the ELF magic was overwritten: {:02x?}",
        image.get(..4)
    );
    let eh = section_range(&image, b".eh_frame")
        .expect("the image has an .eh_frame section");
    assert!(
        eh.0 >= 64,
        ".eh_frame starts at file offset {}, inside the 64-byte ELF header",
        eh.0
    );
}

/// Assembles `SRC` and removes `.text`, leaving an object whose only
/// allocated section is `.eh_frame`. `None` when the tools are missing.
fn fixture() -> Option<PathBuf> {
    let (Some(as_bin), Some(objcopy)) = (which("as"), which("objcopy")) else {
        println!("as or objcopy not found; skipping");
        return None;
    };
    let dir = std::env::temp_dir().join("xold_eh_frame_only");
    fs::create_dir_all(&dir).ok()?;
    let src = dir.join("only.s");
    fs::write(&src, SRC).ok()?;
    let with_text = dir.join("with-text.o");
    let status = Command::new(as_bin)
        .arg(&src)
        .arg("-o")
        .arg(&with_text)
        .status()
        .ok()?;
    if !status.success() {
        println!("as failed; skipping");
        return None;
    }
    let stripped = dir.join("no-text.o");
    let status = Command::new(objcopy)
        .arg("--remove-section=.text")
        .arg(&with_text)
        .arg(&stripped)
        .status()
        .ok()?;
    if !status.success() {
        println!("objcopy failed; skipping");
        return None;
    }
    Some(stripped)
}

/// The `(offset, size)` of the section named `name`, read straight out of the
/// image's section header table.
fn section_range(image: &[u8], name: &[u8]) -> Option<(u64, u64)> {
    let shoff = u64_at(image, 0x28)?;
    let shentsize = usize::from(u16_at(image, 0x3a)?);
    let shnum = usize::from(u16_at(image, 0x3c)?);
    let shstrndx = usize::from(u16_at(image, 0x3e)?);
    let base = usize::try_from(shoff).ok()?;
    let str_hdr = base.checked_add(shstrndx.checked_mul(shentsize)?)?;
    let str_off = usize::try_from(u64_at(image, str_hdr + 0x18)?).ok()?;
    for i in 0..shnum {
        let hdr = base.checked_add(i.checked_mul(shentsize)?)?;
        let name_off = usize::try_from(u32_at(image, hdr)?).ok()?;
        let start = str_off.checked_add(name_off)?;
        let bytes = image.get(start..)?;
        let end = bytes.iter().position(|&b| b == 0)?;
        if bytes.get(..end)? == name {
            return Some((
                u64_at(image, hdr + 0x18)?,
                u64_at(image, hdr + 0x20)?,
            ));
        }
    }
    None
}

fn u16_at(image: &[u8], off: usize) -> Option<u16> {
    let b = image.get(off..off.checked_add(2)?)?;
    Some(u16::from_le_bytes([b[0], b[1]]))
}

fn u32_at(image: &[u8], off: usize) -> Option<u32> {
    let b = image.get(off..off.checked_add(4)?)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn u64_at(image: &[u8], off: usize) -> Option<u64> {
    let b = image.get(off..off.checked_add(8)?)?;
    let mut v = [0u8; 8];
    v.copy_from_slice(b);
    Some(u64::from_le_bytes(v))
}
