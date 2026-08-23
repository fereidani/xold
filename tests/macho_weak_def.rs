//! A strong definition overrides a weak one in the Mach-O link.
//!
//! Mach-O carries the weak/strong distinction in `n_desc`'s `N_WEAK_DEF`
//! bit, not in `n_type`: a weak definition (`__attribute__((weak))`) and a
//! strong definition of one name can both arrive at the link. The
//! `Globals` table kept whichever definition it saw first, so a weak
//! definition ahead of a strong one on the link line silently shadowed
//! it -- every reference resolved to the weak symbol, the opposite of
//! the rule. lld ranks a strong definition above a weak one
//! (`lld/MachO/Symbols.cpp`, `macho::overrides`).
//!
//! Darwin images cannot run on this host, so the test reads the output
//! symbol table and sections: `_tag` must resolve to the address holding
//! the strong definition's pointer, whose target string is `"strong"`.
//!
//! Gated on a `clang` that can target darwin; without one the test
//! prints a note and returns.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{input::Input, macho::link_macho};

mod common;

const LC_SEGMENT_64: u32 = 0x19;

/// The weak definition, first on the link line, with the reader: darwin
/// accesses a weak definition through a GOT slot because its address is
/// not final until the link decides the winner.
const WEAK_SRC: &[u8] = b"__attribute__((weak)) const char *tag = \"weak\";\n\
    const char *pick(void) { return tag; }\n";

/// The strong definition, second, with the entry point.
const STRONG_SRC: &[u8] = b"const char *tag = \"strong\";\n\
    int main(void) { return 0; }\n";

/// The reference resolves to the strong definition's storage.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_strong_definition_overrides_a_weak_one() {
    let Some((dir, image)) = link() else {
        return;
    };
    // The output symtab carries BOTH definitions (each object's nlist is
    // copied), so the resolution is read where the program reads it: the
    // `__got` slot `pick`'s `movq (%rip)` loads through. It holds the
    // winning `_tag` address; the datum it names holds the string.
    let (_, _, got_off) =
        section(&image, b"__got").expect("pick reads _tag through a GOT slot");
    let cell: [u8; 8] = image
        .get(got_off..got_off + 8)
        .and_then(|c| c.try_into().ok())
        .expect("the GOT slot must be file-backed");
    let datum = u64::from_le_bytes(cell);
    let pointer: [u8; 8] = read_at_vaddr(&image, datum, 8)
        .and_then(|c| c.try_into().ok())
        .expect("the datum's address must be mapped");
    let bytes = read_at_vaddr(&image, u64::from_le_bytes(pointer), 8)
        .expect("the string must be mapped");
    let string = bytes.split(|&b| b == 0).next().unwrap_or(bytes);
    assert_eq!(
        string, b"strong",
        "the weak definition must not shadow the strong one"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Compiles both objects for darwin and links them, weak first.
fn link() -> Option<(PathBuf, Vec<u8>)> {
    let clang = which("clang")?;
    let dir = std::env::temp_dir()
        .join(format!("xold_machoweakdef_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    let weak = compile(&clang, &dir, WEAK_SRC, "w")?;
    let strong = compile(&clang, &dir, STRONG_SRC, "s")?;
    let out = dir.join("prog");
    let files: Vec<Input<'_>> =
        vec![Input::Path(weak.as_path()), Input::Path(strong.as_path())]
            .into_iter()
            .collect();
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
        eprintln!("skipping macho weak-def: no darwin target");
        return None;
    }
    Some(obj)
}

// --- readers ---------------------------------------------------------------

/// `(addr, size, file offset)` of a section in the linked image, by name.
fn section(image: &[u8], name: &[u8]) -> Option<(u64, u64, usize)> {
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
                if trimmed == name {
                    // `section_64::offset` is a u32 at +48; `align` sits
                    // right after it at +52, so an 8-byte read would pack
                    // the pair into one bogus offset.
                    return Some((
                        read_u64(image, sh + 32)?,
                        read_u64(image, sh + 40)?,
                        usize::try_from(read_u32(image, sh + 48)?).ok()?,
                    ));
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

/// Reads through an `LC_SEGMENT_64` mapping: up to `len` file bytes at
/// virtual address `addr`.
fn read_at_vaddr(image: &[u8], addr: u64, len: usize) -> Option<&[u8]> {
    let ncmds = usize::try_from(read_u32(image, 16)?).ok()?;
    let mut at = 32;
    for _ in 0..ncmds {
        let cmd = read_u32(image, at)?;
        let size = usize::try_from(read_u32(image, at + 4)?).ok()?;
        if cmd == LC_SEGMENT_64 {
            let vmaddr = read_u64(image, at + 24)?;
            let vmsize = read_u64(image, at + 32)?;
            let fileoff = read_u64(image, at + 40)?;
            let filesize = read_u64(image, at + 48)?;
            if addr >= vmaddr && addr < vmaddr.saturating_add(vmsize) {
                let into = usize::try_from(addr - vmaddr).ok()?;
                let start = usize::try_from(fileoff).ok()?;
                let bound = usize::try_from(filesize).ok()?;
                let end = (into.checked_add(len)?).min(bound);
                return image.get(start + into..start + end);
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

fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
    bytes
        .get(at..at + 8)
        .and_then(|c| <[u8; 8]>::try_from(c).ok())
        .map(u64::from_le_bytes)
}
