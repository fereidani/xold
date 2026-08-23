//! An undefined weak reference binds to zero in the Mach-O link.
//!
//! Mach-O marks a weak reference with `N_WEAK_REF` in `n_desc`: the
//! program is written to test the address before calling through it, so
//! the symbol may be absent. The resolver treated every undefined
//! reference alike and refused the link -- `xold: undefined reference to
//! _missing` -- exactly the case the weak bit exists to allow. lld
//! resolves an unresolved weak reference to zero
//! (`lld/MachO/Symbols.cpp`, `macho::replacer`).
//!
//! The bit only opens the fallback: when a definition does exist the
//! reference resolves to it, the same as a strong one.
//!
//! Gated on a `clang` that can target darwin; without one the tests
//! print a note and return.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{input::Input, macho::link_macho};

mod common;

const LC_SEGMENT_64: u32 = 0x19;

/// The reader: `pick` returns the address of a function nothing defines.
/// darwin accesses an undefined external through a GOT slot, so the link
/// must synthesize one and fill it.
const WEAK_SRC: &[u8] = b"extern int missing(void) __attribute__((weak));\n\
    int (*pick(void))(void) { return missing; }\n\
    int main(void) { return 0; }\n";

/// The definition, so the second link can check the fallback closes only
/// when nothing defines the name.
const DEF_SRC: &[u8] = b"int missing(void) { return 7; }\n";

/// A weak reference with no definition links, and the slot it reads is 0.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_undefined_weak_reference_binds_to_zero() {
    let Some((dir, image)) = link(false, "undef") else {
        return;
    };
    let slot = got_slot(&image).expect("the weak reference reads a GOT slot");
    assert_eq!(
        slot, 0,
        "an undefined weak reference must bind to zero, not fail the link"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A weak reference to a symbol that is defined resolves to the
/// definition; the weak bit must not force zero.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_weak_reference_to_a_definition_resolves_to_it() {
    let Some((dir, image)) = link(true, "defined") else {
        return;
    };
    let slot = got_slot(&image).expect("the weak reference reads a GOT slot");
    assert_ne!(
        slot, 0,
        "a weak reference to a defined symbol must resolve to its address"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Compiles the fixture, and (when `with_def`) the definition, and links.
fn link(with_def: bool, tag: &str) -> Option<(PathBuf, Vec<u8>)> {
    let clang = which("clang")?;
    let dir = std::env::temp_dir()
        .join(format!("xold_machoweakref_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    let mut objects = vec![compile(&clang, &dir, WEAK_SRC, "m")?];
    if with_def {
        objects.push(compile(&clang, &dir, DEF_SRC, "d")?);
    }
    let out = dir.join("prog");
    let files: Vec<Input<'_>> =
        objects.iter().map(|p| Input::Path(p.as_path())).collect();
    let res = link_macho(&files, &out, b"_main");
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let image = fs::read(&out).ok()?;
    Some((dir, image))
}

/// Compiles one fixture for darwin.
fn compile(
    clang: &PathBuf,
    dir: &std::path::Path,
    src: &[u8],
    stem: &str,
) -> Option<PathBuf> {
    let path = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&path, src).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-apple-darwin", "-O1", "-c"])
        .arg(&path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping macho weak-ref: no darwin target");
        return None;
    }
    Some(obj)
}

// --- readers ---------------------------------------------------------------

/// The first `__got` slot's value, where the fixture's only GOT
/// relocation points.
fn got_slot(image: &[u8]) -> Option<u64> {
    let ncmds = usize::try_from(read_u32(image, 16)?).ok()?;
    let mut at = 32;
    for _ in 0..ncmds {
        let cmd = read_u32(image, at)?;
        let size = usize::try_from(read_u32(image, at + 4)?).ok()?;
        if cmd == LC_SEGMENT_64 {
            let nsects = usize::try_from(read_u32(image, at + 64)?).ok()?;
            for i in 0..nsects {
                let sh = at + 72 + i * 80;
                let cell = image.get(sh..sh + 16)?;
                let trimmed = cell.split(|&b| b == 0).next().unwrap_or(cell);
                if trimmed == b"__got" {
                    // `section_64::offset` is a u32 at +48; `align` follows
                    // at +52, so an 8-byte read would pack the pair.
                    let off =
                        usize::try_from(read_u32(image, sh + 48)?).ok()?;
                    let slot: [u8; 8] = image
                        .get(off..off + 8)
                        .and_then(|c| c.try_into().ok())?;
                    return Some(u64::from_le_bytes(slot));
                }
            }
        }
        if size < 8 {
            return None;
        }
        at += size;
    }
    None
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .and_then(|c| <[u8; 4]>::try_from(c).ok())
        .map(u32::from_le_bytes)
}
