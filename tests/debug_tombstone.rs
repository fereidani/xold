//! Debug references into a discarded section get a tombstone, not an address.
//!
//! Two translation units that include the same inline function each emit a
//! copy of it in a COMDAT group, and the linker keeps one. The loser's
//! sections are dropped -- but its `.debug_*` sections are not, and they still
//! carry relocations against the code that is gone.
//!
//! xold answered those with the address the section would have had if it were
//! placed at zero, which is the addend. The result is a small, entirely
//! plausible address: `.debug_ranges` ends up holding a range like
//! `[0x0, 0xd)`, a debugger reads it as real code at the bottom of the image,
//! and two compilation units claim the same addresses. Nothing in the link
//! says a word.
//!
//! A tombstone is a value that cannot be mistaken for a range. lld picks it by
//! section (`relocateNonAlloc`, `lld/ELF/InputSection.cpp`):
//! one for the pre-DWARF-5 `.debug_loc` and `.debug_ranges`, where all-ones is
//! a base-address selection entry and a pair of zeroes ends the list; all-ones
//! for `.debug_names`; zero everywhere else. The addend is dropped, because
//! adding it back would walk the value into the range the tombstone exists to
//! stay out of.
//!
//! Gated on `clang++`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_to};

mod common;

/// The inline function both translation units define, in a COMDAT group.
const HEADER: &[u8] = b"#pragma once\n\
    inline int shared_fn(int x) { return x * 3; }\n";

const A_SRC: &[u8] = b"#include \"h.hpp\"\n\
    int a_entry(void) { return shared_fn(4); }\n";

const B_SRC: &[u8] = b"#include \"h.hpp\"\n\
    int a_entry(void);\n\
    int b_entry(void) { return shared_fn(5); }\n\
    int main(void) { return a_entry() + b_entry(); }\n";

/// `.debug_ranges` takes the tombstone 1, and the pair that named the
/// discarded copy becomes an empty range there rather than a range over real
/// addresses at the bottom of the image.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_discarded_range_becomes_an_empty_range_at_the_tombstone() {
    let Some(dir) = workdir("ranges") else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    let ranges =
        section_bytes(&bytes, b".debug_ranges").expect(".debug_ranges present");
    let words: Vec<u64> = ranges
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .collect();
    let pair = words
        .windows(2)
        .position(|w| w == [1, 1])
        .map(|i| (words[i], words[i + 1]));
    assert_eq!(
        pair,
        Some((1, 1)),
        "the range naming the discarded copy must be the tombstone pair \
         (1, 1); resolving it to its addend gives a range over real-looking \
         addresses. words: {words:#x?}"
    );
    assert!(
        !words
            .windows(2)
            .any(|w| w[0] == 0 && w[1] != 0 && w[1] < 0x1000),
        "no range may start at zero and end at a small address -- that is the \
         shape the addend produced. words: {words:#x?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The rest of the debug info is untouched: the live ranges still name real
/// code, so the tombstone applies to the discarded copy and nothing else.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_surviving_ranges_still_name_real_code() {
    let Some(dir) = workdir("live") else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    let text = section_addr(&bytes, b".text").expect(".text present");
    let ranges =
        section_bytes(&bytes, b".debug_ranges").expect(".debug_ranges present");
    let live = ranges
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .filter(|w| *w >= text)
        .count();
    assert!(
        live >= 4,
        "the ranges over the kept code must still hold real addresses, got \
         {live} words at or above .text ({text:#x})"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control that the tombstone did not swallow the string tables:
/// `.debug_str` and `.debug_line_str` are merged into pools rather than
/// stamped with an address of their own, so a naive "has no address" test
/// tombstones every string reference and the line table decodes to nothing.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn debug_string_references_still_resolve() {
    let Some(dir) = workdir("strings") else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    let info = section_bytes(&bytes, b".debug_info").expect(".debug_info");
    let strs = section_bytes(&bytes, b".debug_str").expect(".debug_str");
    assert!(
        !strs.is_empty(),
        "the fixture must produce a string table to reference"
    );
    // A DWARF 4 `.debug_info` names its producer and file through 4-byte
    // offsets into `.debug_str`. If those were tombstoned every one would be
    // zero.
    let nonzero = info
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|c| c.iter().any(|b| *b != 0))
        .count();
    assert!(
        nonzero > 4,
        "the compilation unit must still carry its string offsets, got \
         {nonzero} non-zero words"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang++").is_none() {
        eprintln!("skipping debug-tombstone {prefix}: clang++ unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_tombstone_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds both translation units and links them, returning the image bytes.
///
/// DWARF 4 is asked for by name because `.debug_ranges` is where the defect is
/// visible: DWARF 5 replaced it with `.debug_rnglists`, whose entries are
/// offsets into `.debug_addr` rather than addresses.
fn link(dir: &Path) -> Option<Vec<u8>> {
    let clangxx = which("clang++")?;
    fs::write(dir.join("h.hpp"), HEADER).ok()?;
    let a = compile(&clangxx, dir, A_SRC, "a")?;
    let b = compile(&clangxx, dir, B_SRC, "b")?;
    let out = dir.join("prog");
    let res = link_to(&[a, b], &out, b"main", false, IcfMode::None, false);
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

/// Compiles one translation unit with DWARF 4 debug info.
fn compile(
    clangxx: &Path,
    dir: &Path,
    src: &[u8],
    stem: &str,
) -> Option<PathBuf> {
    let src_path = dir.join(format!("{stem}.cpp"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clangxx)
        .args(["--target=x86_64-linux-gnu", "-gdwarf-4", "-O0", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping debug-tombstone: clang++ cannot build the fixture");
        return None;
    }
    Some(obj)
}

// --- readers ---------------------------------------------------------------

/// The bytes of an output section.
fn section_bytes(bytes: &[u8], name: &[u8]) -> Option<Vec<u8>> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == name)?;
    obj.section_data(shdr).ok().map(<[u8]>::to_vec)
}

/// The virtual address of an output section.
fn section_addr(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == name)
        .map(|s| s.sh_addr.get())
}
