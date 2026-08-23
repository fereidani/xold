//! A zero-size section at its region's end is still in that region.
//!
//! The output section index stamped on each input section was derived from the
//! member's address, by asking which region *covers* it over a half-open
//! range. A zero-size member placed at its region's exact end is covered by
//! nothing: `vaddr < region.end()` is false for it, and the lookup fell
//! through to zero.
//!
//! A global defined there was then published `SHN_UNDEF` -- the symbol table
//! said the program's own definition was an unresolved import. The shape is an
//! assembler end-marker: a `.data.zmarker` section holding no bytes, with a
//! label at the end of the data it marks, which is how hand-written assembly
//! and generated tables name their own extent.
//!
//! The index now comes from the output section the member belongs to, which is
//! the fact placement already recorded. That is cheaper as well as right: one
//! walk of the members instead of a scan of every content region per input
//! section.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::{ObjectFile, constants::SHN_UNDEF},
    icf::IcfMode,
    linker::link_to,
};

mod common;

/// `.data.zmarker` holds no bytes and is placed after `.data.payload`, so it
/// lands at the exact end of the aggregated `.data`.
const SRC: &[u8] = b"    .section .data.payload,\"aw\",@progbits\n\
    .globl payload\n\
payload:\n\
    .quad 0x1122334455667788\n\
    .section .data.zmarker,\"aw\",@progbits\n\
    .globl marker\n\
marker:\n\
    .text\n\
    .globl _start\n\
_start:\n\
    movq  payload(%rip), %rax\n\
    ret\n";

/// The marker is published in the section that holds it, not as an import.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_marker_at_the_region_end_keeps_its_section() {
    let Some(dir) = workdir("marker") else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    let payload = symbol(&bytes, b"payload").expect("payload is defined");
    let marker = symbol(&bytes, b"marker").expect("marker is defined");
    assert_ne!(
        marker.shndx, SHN_UNDEF,
        "a definition this link placed must not be published as an \
         unresolved import"
    );
    assert_eq!(
        marker.shndx, payload.shndx,
        "the marker is in the same output section as the data it marks"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// And it marks the right place: one past the data.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_marker_sits_at_the_end_of_what_it_marks() {
    let Some(dir) = workdir("addr") else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    let payload = symbol(&bytes, b"payload").expect("payload is defined");
    let marker = symbol(&bytes, b"marker").expect("marker is defined");
    assert_eq!(
        marker.value,
        payload.value + 8,
        "the marker names one past the eight bytes of payload"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping zero-size-section {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_zerosize_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Assembles and links the fixture.
fn link(dir: &Path) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let src = dir.join("marker.S");
    let obj = dir.join("marker.o");
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
        eprintln!("skipping zero-size-section: clang cannot assemble it");
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
    fs::read(&out).ok()
}

// --- readers ---------------------------------------------------------------

/// The fields of one `.symtab` row these tests reason about.
struct Row {
    shndx: u16,
    value: u64,
}

/// Reads a symbol out of the output's `.symtab`.
fn symbol(bytes: &[u8], name: &[u8]) -> Option<Row> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab.iter().find(|s| symtab.name(s) == name).map(|s| Row {
        shndx: s.st_shndx.get(),
        value: s.st_value.get(),
    })
}
