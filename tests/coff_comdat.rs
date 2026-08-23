//! COFF COMDAT selection, and what it makes possible.
//!
//! `IMAGE_SCN_LNK_COMDAT` was read nowhere. Every copy of an inline function,
//! a template instantiation or a vtable was placed, and every one was pushed
//! into the global map, so a reference resolved to whichever copy the link
//! line happened to put first. Two objects that both include a header with an
//! inline function carried two identical bodies in `.text`.
//!
//! It also meant genuine duplicates could not be told apart from COMDAT ones,
//! so two strong definitions of the same name linked silently and the winner
//! depended on argument order. With selection in place the two cases separate:
//! one copy of the COMDAT survives, and a real clash is the error it always
//! was. lld does both in `handleComdatSelection`.
//!
//! Gated on a `clang` that can target `x86_64-pc-windows-msvc`, and the
//! running test additionally on `wine`; without them the tests print a note
//! and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{coff::link_coff, input::Input};

mod common;

/// A header with an inline function, included by both translation units, so
/// each object carries its own COMDAT copy of the body.
const HEADER: &[u8] =
    b"inline int shared_inline(int x) { return x * 3 + 1; }\n";

const A: &[u8] = b"#include \"h.h\"\n\
    int use_a(void) { return shared_inline(2); }\n\
    int use_b(void);\n\
    int main(void) { return use_a() + use_b() == 11 ? 0 : 1; }\n";

const B: &[u8] = b"#include \"h.h\"\n\
    int use_b(void) { return shared_inline(1); }\n";

/// Two strong definitions of one name, which is not a COMDAT and not legal.
const CLASH_A: &[u8] = b"int clash = 1;\n\
    int main(void) { return clash; }\n";
const CLASH_B: &[u8] = b"int clash = 2;\n";

/// One copy of the inline body survives, not two.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_comdat_is_kept_once() {
    let Some(dir) = workdir("dedup") else {
        return;
    };
    let Some((image, objs)) = link(&dir) else {
        return;
    };
    // The inline body has no relocations, so its bytes reach the image
    // unchanged and can be counted there.
    let body = objs
        .iter()
        .find_map(|o| comdat_text(o))
        .expect("both objects carry the inline body as a COMDAT");
    assert!(
        body.len() >= 8,
        "the body must be distinctive enough to count"
    );
    assert_eq!(
        occurrences(&image, &body),
        1,
        "exactly one copy of the inline body survives the link"
    );
    assert!(text_size(&image) > 0, ".text must be present");
    let _ = fs::remove_dir_all(&dir);
}

/// And the program still computes with it, so the surviving copy is the one
/// every reference reaches.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn both_callers_reach_the_surviving_copy() {
    let Some(dir) = workdir("run") else {
        return;
    };
    if which("wine").is_none() {
        eprintln!("skipping coff-comdat run: wine unavailable");
        return;
    }
    let Some(_) = link(&dir) else {
        return;
    };
    let out = Command::new("wine")
        .arg(dir.join("out.exe"))
        .output()
        .expect("wine must run the image");
    assert_eq!(
        out.status.code(),
        Some(0),
        "both callers must reach a body that computes 3x + 1"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A real duplicate is an error, not a first-wins race.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn two_strong_definitions_are_an_error() {
    let Some(dir) = workdir("clash") else {
        return;
    };
    let Some(a) = compile(&dir, CLASH_A, "x") else {
        return;
    };
    let Some(b) = compile(&dir, CLASH_B, "y") else {
        return;
    };
    let files = [Input::Path(&a), Input::Path(&b)];
    let err = link_coff(&files, &dir.join("clash.exe"), b"main", false)
        .expect_err("two strong definitions of one name is an error");
    let text = format!("{err}");
    assert!(
        text.contains("clash"),
        "the message must name the symbol, got {text:?}"
    );
    assert!(
        text.contains("input 0") && text.contains("input 1"),
        "and both inputs that define it, got {text:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping coff-comdat {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_coffcomdat_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Writes the header, compiles both translation units and links them.
///
/// Returns the image and the two objects' bytes, so a test can compare what
/// the inputs offered against what the link kept.
fn link(dir: &Path) -> Option<(Vec<u8>, Vec<Vec<u8>>)> {
    fs::write(dir.join("h.h"), HEADER).ok()?;
    let a = compile(dir, A, "a")?;
    let b = compile(dir, B, "b")?;
    let files = [Input::Path(&a), Input::Path(&b)];
    let out = dir.join("out.exe");
    let res = link_coff(&files, &out, b"main", false);
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let objs = vec![fs::read(&a).ok()?, fs::read(&b).ok()?];
    Some((fs::read(&out).ok()?, objs))
}

/// The raw bytes of the first COMDAT `.text` section in one object, which is
/// where the inline body lives.
fn comdat_text(bytes: &[u8]) -> Option<Vec<u8>> {
    let num = usize::from(read_u16(bytes, 2)?);
    let opt = usize::from(read_u16(bytes, 16)?);
    let first = 20 + opt;
    (0..num).find_map(|i| {
        let at = first + i * 40;
        let cell = bytes.get(at..at + 8)?;
        let trimmed = cell.split(|&b| b == 0).next().unwrap_or(cell);
        // IMAGE_SCN_LNK_COMDAT, and no relocations, so the bytes are final.
        if trimmed != b".text"
            || read_u32(bytes, at + 36)? & 0x1000 == 0
            || read_u16(bytes, at + 32)? != 0
        {
            return None;
        }
        let off = usize::try_from(read_u32(bytes, at + 20)?).ok()?;
        let size = usize::try_from(read_u32(bytes, at + 16)?).ok()?;
        bytes.get(off..off + size).map(<[u8]>::to_vec)
    })
}

/// How many non-overlapping times `needle` occurs in `hay`.
fn occurrences(hay: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() || hay.len() < needle.len() {
        return 0;
    }
    let mut count = 0;
    let mut at = 0;
    while at + needle.len() <= hay.len() {
        if &hay[at..at + needle.len()] == needle {
            count += 1;
            at += needle.len();
        } else {
            at += 1;
        }
    }
    count
}

/// Compiles one fixture for Windows at `-O0`, which is what leaves the inline
/// function as a COMDAT rather than folding it into its callers.
fn compile(dir: &Path, src: &[u8], stem: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let path = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.obj"));
    fs::write(&path, src).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-pc-windows-msvc", "-O0", "-c"])
        .arg(&path)
        .arg("-o")
        .arg(&obj)
        .arg("-I")
        .arg(dir)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping coff-comdat: clang cannot target Windows");
        return None;
    }
    Some(obj)
}

// --- readers ---------------------------------------------------------------

/// The `VirtualSize` of `.text`.
fn text_size(bytes: &[u8]) -> u32 {
    let Some(pe) = read_u32(bytes, 0x3c).and_then(|v| usize::try_from(v).ok())
    else {
        return 0;
    };
    let Some(num) = read_u16(bytes, pe + 6).map(usize::from) else {
        return 0;
    };
    let Some(opt) = read_u16(bytes, pe + 20).map(usize::from) else {
        return 0;
    };
    let first = pe + 24 + opt;
    (0..num)
        .find_map(|i| {
            let at = first + i * 40;
            let cell = bytes.get(at..at + 8)?;
            let trimmed = cell.split(|&b| b == 0).next().unwrap_or(cell);
            if trimmed != b".text" {
                return None;
            }
            read_u32(bytes, at + 8)
        })
        .unwrap_or(0)
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
