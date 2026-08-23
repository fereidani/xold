//! Resolving one Mach-O symbol does not walk the whole table.
//!
//! `symbol_addr` collected the symbol iterator into a `Vec` and indexed that,
//! so every lookup allocated and decoded the file's entire `nlist` array. It
//! is called once per symbol and once per undefined reference, which made a
//! file's resolution quadratic in its symbol count and allocated once per
//! call besides.
//!
//! The entries are fixed width, so the one asked for is a slice index away.
//! The table exposes that directly now.
//!
//! Gated on a `clang` that can target darwin; without one the tests print a
//! note and return.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{macho::MachOFile, mmap_file::MappedFile};

mod common;

/// Several definitions, so the indices being tested are distinct.
const SRC: &[u8] = b"int a = 1;\n\
    int b = 2;\n\
    int c(void) { return a + b; }\n\
    int main(void) { return c(); }\n";

/// The indexed accessor agrees with the iterator, entry for entry.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn indexing_agrees_with_iterating() {
    let Some((dir, obj)) = darwin_object("agree") else {
        return;
    };
    let mapped = MappedFile::open(&obj).expect("the fixture must map");
    let file = MachOFile::parse(mapped.bytes()).expect("must parse");
    let syms = file.symbols();
    assert!(syms.len() > 2, "the fixture must carry several symbols");
    for (i, want) in syms.iter().enumerate() {
        let got = syms.nth(i).expect("every index in range resolves");
        assert_eq!(got.name, want.name, "entry {i} name");
        assert_eq!(got.n_type, want.n_type, "entry {i} type");
        assert_eq!(got.n_sect, want.n_sect, "entry {i} section");
        assert_eq!(got.n_value, want.n_value, "entry {i} value");
    }
    let _ = fs::remove_dir_all(&dir);
}

/// An index past the end is `None`, not a panic and not entry zero.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_index_past_the_end_resolves_to_nothing() {
    let Some((dir, obj)) = darwin_object("bounds") else {
        return;
    };
    let mapped = MappedFile::open(&obj).expect("the fixture must map");
    let file = MachOFile::parse(mapped.bytes()).expect("must parse");
    let syms = file.symbols();
    assert!(syms.nth(syms.len()).is_none(), "one past the end");
    assert!(syms.nth(usize::MAX).is_none(), "and far past it");
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Compiles the fixture for darwin, or `None` (after printing a note) when
/// clang cannot target it.
fn darwin_object(prefix: &str) -> Option<(PathBuf, PathBuf)> {
    let clang = which("clang")?;
    let dir = std::env::temp_dir()
        .join(format!("xold_machosym_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    let src = dir.join("s.c");
    let obj = dir.join("s.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-apple-darwin", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping macho-symbol-lookup {prefix}: no darwin target");
        let _ = fs::remove_dir_all(&dir);
        return None;
    }
    Some((dir, obj))
}
