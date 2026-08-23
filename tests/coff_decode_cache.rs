//! The COFF input tables are decoded once per file, not once per question.
//!
//! `sections()` and `symbols()` rebuilt their whole result on every call:
//! a `Vec` of section records with their relocations decoded, and a `Vec` of
//! symbol records with their auxiliary entries resolved. Placement asks once
//! per input, relocation collection asks again, the three TLS walks ask three
//! more times, member copying asks per member and export naming asks per name,
//! so the decode ran a fixed number of times over every section of every
//! input -- work linear in the input repeated once per pass.
//!
//! Both are memoised now. On sixteen objects of three thousand functions each
//! (48000 sections) the link goes from 7.06s to 1.61s.
//!
//! The property under test is that the second call hands back the first call's
//! storage rather than a fresh decode of the same bytes, which is what makes
//! the repetition free.
//!
//! Gated on a `clang` that can target `x86_64-pc-windows-msvc`; without one
//! the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::coff::CoffFile;

mod common;

/// Enough sections and symbols that a re-decode would be visible work.
const SRC: &[u8] = b"int a(int x) { return x + 1; }\n\
    int b(int x) { return x + 2; }\n\
    int c(int x) { return x + 3; }\n\
    const char *s(void) { return \"text\"; }\n\
    int main(void) { return a(1) + b(2) + c(3); }\n";

/// Asking twice yields the same storage, not a second decode.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_section_table_is_decoded_once() {
    let Some(dir) = workdir("sections") else {
        return;
    };
    let Some(bytes) = object(&dir) else {
        return;
    };
    let file = CoffFile::parse(&bytes).expect("the fixture must parse");
    let first = file.sections();
    let second = file.sections();
    assert!(
        !first.is_empty(),
        "the fixture must carry sections for this to mean anything"
    );
    assert!(
        std::ptr::eq(first.as_ptr(), second.as_ptr()),
        "the second call must reuse the first call's decode"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// And so does the symbol table.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_symbol_table_is_decoded_once() {
    let Some(dir) = workdir("symbols") else {
        return;
    };
    let Some(bytes) = object(&dir) else {
        return;
    };
    let file = CoffFile::parse(&bytes).expect("the fixture must parse");
    let first = file.symbols();
    let second = file.symbols();
    assert!(first.len() > 4, "the fixture must carry symbols");
    assert!(
        std::ptr::eq(first, second),
        "the second call must reuse the first call's decode"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The cache must not change what the tables say.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_cached_tables_are_the_ones_the_file_describes() {
    let Some(dir) = workdir("content") else {
        return;
    };
    let Some(bytes) = object(&dir) else {
        return;
    };
    let file = CoffFile::parse(&bytes).expect("the fixture must parse");
    assert_eq!(
        file.sections().len(),
        usize::try_from(file.number_of_sections()).unwrap_or(0),
        "every section in the table is decoded"
    );
    let named = |want: &[u8]| file.symbols().iter().any(|s| s.name == want);
    for want in [b"main".as_slice(), b"a", b"b", b"c"] {
        assert!(
            named(want),
            "the symbol table must still name every definition"
        );
    }
    // The indices are 1-based section numbers, in table order: a stale or
    // shared cache would show up here first.
    for (i, s) in file.sections().iter().enumerate() {
        assert_eq!(
            usize::try_from(s.index).unwrap_or(0),
            i + 1,
            "section numbering follows the table"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping coff-decode-cache {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_coffcache_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture for Windows and returns the object bytes.
fn object(dir: &Path) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let src = dir.join("c.c");
    let obj = dir.join("c.obj");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args([
            "--target=x86_64-pc-windows-msvc",
            "-O1",
            "-ffunction-sections",
            "-c",
        ])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping coff-decode-cache: clang cannot target Windows");
        return None;
    }
    fs::read(&obj).ok()
}
