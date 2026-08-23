//! `$`-suffixed input sections reach the output in name order.
//!
//! The PE/COFF convention drops `$` and everything after it when deciding
//! which output section an input belongs to, and the CRT leans on the part
//! it drops: `.CRT$XCA` opens the initializer table, `.CRT$XCU` fills it,
//! `.CRT$XLZ` and `.CRT$XZ` close it, and `.tls$aaa`/`.tls$zzz` bracket the
//! TLS template. The table is a contiguous run of pointers, so the members
//! must sit in the order their suffixes spell, whatever order the objects
//! arrived in.
//!
//! The layout concatenated members in input order, so a `.text$bbb` from
//! the first object landed ahead of a `.text$aaa` from the second. lld
//! collects the inputs into a map keyed by full section name and walks it
//! in key order, which is what makes the suffixes order the output
//! (`lld/COFF/Writer.cpp`); within one name it keeps the
//! input order, which is what `sortCRTSectionChunks` then makes explicit
//! for `.CRT`.
//!
//! The fixtures put a `.text$bbb` function in the first object and a
//! `.text$aaa` function in the second, and find both by their distinctive
//! `mov eax, imm32; ret` bodies in the linked `.text`.
//!
//! Gated on a `clang` that can target `x86_64-pc-windows-msvc`; without
//! one the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{coff::link_coff, input::Input};

mod common;

/// The first object on the command line, holding the lexically later
/// section: exactly the input order a name sort must undo.
const LATE_SRC: &[u8] = b"__attribute__((section(\".text$bbb\"), noinline))\n\
    int late_b(void) { return 0xbad1bee5; }\n\
    int keep_late(int x) { return late_b() + x; }\n";

/// The second object, holding the lexically earlier section.
const EARLY_SRC: &[u8] = b"__attribute__((section(\".text$aaa\"), noinline))\n\
    int early_a(void) { return 0x600d1dee; }\n\
    int keep_early(int x) { return early_a() + x; }\n";

/// Same shapes, plain `.text` both: no suffix to order by, so the input
/// order is the one that must survive.
const PLAIN_LATE_SRC: &[u8] = b"__attribute__((noinline))\n\
    int late_b(void) { return 0xbad1bee5; }\n\
    int keep_late(int x) { return late_b() + x; }\n";

const PLAIN_EARLY_SRC: &[u8] = b"__attribute__((noinline))\n\
    int early_a(void) { return 0x600d1dee; }\n\
    int keep_early(int x) { return early_a() + x; }\n";

/// `.text$aaa` lands ahead of `.text$bbb` although its object came second.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn suffixed_members_are_ordered_by_name() {
    let Some(dir) = workdir("suffix") else {
        return;
    };
    let Some(text) =
        link_and_read_text(&dir, "late", "early", LATE_SRC, EARLY_SRC)
    else {
        return;
    };
    let (a, b) = (find_body(&text, 0x600d_1dee), find_body(&text, 0xbad1_bee5));
    assert!(a.is_some() && b.is_some(), "both bodies are in .text");
    assert!(
        a < b,
        "the lexically earlier suffix lands first in .text, whatever the \
         input order"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: same names, no suffix -- the input order survives, so the
/// ordering above is the name sort's doing and not a fixture property.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn same_named_members_keep_the_input_order() {
    let Some(dir) = workdir("plain") else {
        return;
    };
    let Some(text) = link_and_read_text(
        &dir,
        "late",
        "early",
        PLAIN_LATE_SRC,
        PLAIN_EARLY_SRC,
    ) else {
        return;
    };
    let (a, b) = (find_body(&text, 0x600d_1dee), find_body(&text, 0xbad1_bee5));
    assert!(a.is_some() && b.is_some(), "both bodies are in .text");
    assert!(
        b < a,
        "members of one name keep the order their objects arrived in"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping coff-suffix-order {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_coffsfx_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles `late_src` and `early_src` in that order and links both into a
/// PE executable, returning its `.text` bytes.
fn link_and_read_text(
    dir: &Path,
    late_name: &str,
    early_name: &str,
    late_src: &[u8],
    early_src: &[u8],
) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let late = compile(dir, &clang, late_name, late_src)?;
    let early = compile(dir, &clang, early_name, early_src)?;
    let out = dir.join("a.exe");
    let files = [Input::Path(&late), Input::Path(&early)];
    let res = link_coff(&files, &out, b"keep_late", false);
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let bytes = fs::read(&out).ok()?;
    read_text(&bytes)
}

/// Compiles one source to a `.obj` for x86-64 Windows.
fn compile(
    dir: &Path,
    clang: &Path,
    name: &str,
    src: &[u8],
) -> Option<PathBuf> {
    let src_path = dir.join(format!("{name}.c"));
    let obj = dir.join(format!("{name}.obj"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-pc-windows-msvc", "-O1", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping coff-suffix-order: clang cannot build {name}");
        return None;
    }
    Some(obj)
}

// --- readers ---------------------------------------------------------------

/// The `.text` bytes of a PE image.
fn read_text(bytes: &[u8]) -> Option<Vec<u8>> {
    let pe = usize::try_from(read_u32(bytes, 0x3c)?).ok()?;
    let num = usize::from(read_u16(bytes, pe + 6)?);
    let opt = usize::from(read_u16(bytes, pe + 20)?);
    let first = pe + 24 + opt;
    (0..num).find_map(|i| {
        let at = first + i * 40;
        let cell = bytes.get(at..at + 8)?;
        let trimmed = cell.split(|&b| b == 0).next().unwrap_or(cell);
        if trimmed != b".text" {
            return None;
        }
        let ptr = read_u32(bytes, at + 20)? as usize;
        let size = read_u32(bytes, at + 16)? as usize;
        bytes.get(ptr..ptr + size).map(<[u8]>::to_vec)
    })
}

/// Where `mov eax, imm32; ret` for `imm` sits in `text`, if it is there.
fn find_body(text: &[u8], imm: u32) -> Option<usize> {
    let mut at = 0;
    while at + 6 <= text.len() {
        if text[at] == 0xb8
            && text[at + 1..at + 5] == imm.to_le_bytes()
            && text[at + 5] == 0xc3
        {
            return Some(at);
        }
        at += 1;
    }
    None
}

fn read_u16(bytes: &[u8], at: usize) -> Option<u16> {
    bytes
        .get(at..at + 2)
        .and_then(|c| <[u8; 2]>::try_from(c).ok())
        .map(u16::from_le_bytes)
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .and_then(|c| <[u8; 4]>::try_from(c).ok())
        .map(u32::from_le_bytes)
}
