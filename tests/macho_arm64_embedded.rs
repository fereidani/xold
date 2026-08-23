//! An `arm64` `UNSIGNED` relocation carries its addend in the stored bytes.
//!
//! Every other arm64 type overwrites scattered instruction bits, so its
//! embedded addend is zero and the explicit `ARM64_RELOC_ADDEND` hint is the
//! whole story. `UNSIGNED` is the exception: `ld` and lld read the field
//! itself (`ARM64Common::getEmbeddedAddend`), so `.quad _baz + 123` stores
//! 123 in the quad and expects the linker to add it. Dropping it made every
//! `sym+K` pointer initializer and every pointer to a private `L` symbol
//! (whose relocs are extern-form with a stored addend) land `K` bytes low.
//!
//! Darwin images cannot run on this host, so this reads the linked bytes.
//!
//! Gated on a `clang` that can target darwin; without one the test prints a
//! note and returns.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{input::Input, macho::link_macho};

mod common;

const LC_SEGMENT_64: u32 = 0x19;

/// A pointer initializer with a nonzero constant term, plus an extern-form
/// reference the linker must resolve.
const SRC: &[u8] = b".globl _main\n\
    .text\n\
    _main:\n\
      ret\n\
    .data\n\
    .globl _baz\n\
    _baz:\n\
      .quad 7\n\
    .globl _ptr\n\
    _ptr:\n\
      .quad _baz + 123\n";

/// `_ptr` holds `_baz + 123`, not `_baz`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_unsigned_fixup_keeps_its_stored_addend() {
    let Some((dir, image)) = link() else {
        return;
    };
    let (addr, size, offset) =
        section(&image, b"__data").expect("the fixture's data must be placed");
    assert!(size >= 16, "two quads of data, found {size} bytes");
    let cell = image
        .get(offset + 8..offset + 16)
        .expect("the pointer initializer must be in the file");
    let value = u64::from_le_bytes(
        <[u8; 8]>::try_from(cell).expect("eight bytes of pointer"),
    );
    assert_eq!(
        value,
        addr + 123,
        "_ptr must hold _baz + 123: the 123 is stored in the field, and \
         dropping it wrote a plain _baz"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Assembles the source for darwin arm64 and links it.
fn link() -> Option<(PathBuf, Vec<u8>)> {
    let clang = which("clang")?;
    let dir = std::env::temp_dir()
        .join(format!("xold_machoemb_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    let path = dir.join("u.s");
    let obj = dir.join("u.o");
    fs::write(&path, SRC).ok()?;
    let built = Command::new(&clang)
        .args(["--target=arm64-apple-darwin", "-c"])
        .arg(&path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping macho arm64 embedded: no darwin target");
        let _ = fs::remove_dir_all(&dir);
        return None;
    }
    let files: Vec<Input<'_>> =
        vec![Input::Path(obj.as_path())].into_iter().collect();
    let out = dir.join("prog");
    let res = link_macho(&files, &out, b"_main");
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let image = fs::read(&out).ok()?;
    Some((dir, image))
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
