//! A Mach-O symbol's `n_value` is an address in the object's own frame, not
//! an offset within its section.
//!
//! An `MH_OBJECT` lays its sections out sequentially in a mini frame
//! (`__text` at 0, `__data` at 0xd, ...), and an `N_SECT` symbol's
//! `n_value` is expressed in that frame. Resolving it as
//! `placed_section_vaddr + n_value` counts the input section's own
//! address twice: a datum in a file's second section landed `addr` bytes
//! past its real place, so every reference -- the GOT slot, the output
//! symtab, relocations -- pointed outside the section. Symbols in a
//! file's first section accidentally worked, because that section sits
//! at frame address zero.
//!
//! lld normalizes first: `sym.n_value - sectionAddr` is the offset the
//! linker places (`MachO/InputFiles.cpp`).
//!
//! Darwin images cannot run on this host, so this reads the output
//! symbol table and the section table.
//!
//! Gated on a `clang` that can target darwin; without one the test
//! prints a note and returns.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{input::Input, macho::link_macho};

mod common;

const LC_SEGMENT_64: u32 = 0x19;
const LC_SYMTAB: u32 = 0x2;

/// A datum in the object's second section (`__text` comes first), plus a
/// reader that reaches it, so the symbol must be resolved to be used.
const SRC: &[u8] = b"int probe(void) { return 1; }\n\
    int in_data = 0x5150;\n\
    int *where(void) { return &in_data; }\n\
    int main(void) { return *where() == 0x5150 ? 0 : 1; }\n";

/// The resolved `_in_data` lands inside the placed `__data`, not past it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_second_section_symbol_resolves_inside_its_section() {
    let Some((dir, image)) = link() else {
        return;
    };
    let (data_addr, data_size) = section(&image, b"__data")
        .expect("the fixture's datum must be placed in __data");
    let value = symbol_value(&image, b"_in_data")
        .expect("the resolved datum must appear in the output symtab");
    assert!(
        value >= data_addr && value < data_addr + data_size,
        "_in_data resolved to {value:#x}, but __data spans \
         {data_addr:#x}..{:#x}: the input-frame address was added on top \
         of the placed base",
        data_addr + data_size
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Compiles the source for darwin and links it.
fn link() -> Option<(PathBuf, Vec<u8>)> {
    let clang = which("clang")?;
    let dir = std::env::temp_dir()
        .join(format!("xold_machonval_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    let path = dir.join("u.c");
    let obj = dir.join("u.o");
    fs::write(&path, SRC).ok()?;
    let built = Command::new(&clang)
        .args(["--target=x86_64-apple-darwin", "-O1", "-c"])
        .arg(&path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping macho n_value: no darwin target");
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

/// `(addr, size)` of a section in the linked image, by name.
fn section(image: &[u8], name: &[u8]) -> Option<(u64, u64)> {
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

/// The `n_value` of the first symbol named `name`, from `LC_SYMTAB`.
fn symbol_value(image: &[u8], name: &[u8]) -> Option<u64> {
    let ncmds = usize::try_from(read_u32(image, 16)?).ok()?;
    let mut at = 32;
    for _ in 0..ncmds {
        let cmd = read_u32(image, at)?;
        let size = usize::try_from(read_u32(image, at + 4)?).ok()?;
        if cmd == LC_SYMTAB {
            let symoff = usize::try_from(read_u32(image, at + 8)?).ok()?;
            let nsyms = usize::try_from(read_u32(image, at + 12)?).ok()?;
            let stroff = usize::try_from(read_u32(image, at + 16)?).ok()?;
            for i in 0..nsyms {
                let sym = symoff + i * 16;
                let strx = usize::try_from(read_u32(image, sym)?).ok()?;
                let start = stroff + strx;
                let rest = image.get(start..)?;
                let sym_name = rest.split(|&b| b == 0).next().unwrap_or(rest);
                if sym_name == name {
                    return read_u64(image, sym + 8);
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
