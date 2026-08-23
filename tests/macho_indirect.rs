//! The indirect symbol table is written for the `__got` section.
//!
//! Every `S_NON_LAZY_SYMBOL_POINTERS` slot (the darwin GOT) names the
//! symbol it holds through the indirect symbol table: `LC_DYSYMTAB`'s
//! `indirectsymoff` / `nindirectsyms` locate one `u32` per slot, each
//! indexing a row of the output symbol table. The writer left both
//! fields zero, so tools decoding the image (`llvm-objdump --macho
//! --indirect-symbols`, debuggers walking a GOT slot to its symbol) saw
//! a `__got` with no symbol bindings at all. lld writes the table for
//! every indirect section (`lld/MachO/OutputSegment.cpp`).
//!
//! The test links a fixture whose only GOT relocation reads a defined
//! global, then walks slot to row to name: the row must sit in the
//! external-defined range the same command reports. A second test pins
//! the table's file placement: its u32 rows must start 4-aligned even
//! when the string table before them ends off a 4-byte boundary.
//!
//! Gated on a `clang` that can target darwin; without one the test
//! prints a note and returns.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{input::Input, macho::link_macho};

mod common;

const LC_SEGMENT_64: u32 = 0x19;
const LC_SYMTAB: u32 = 0x02;
const LC_DYSYMTAB: u32 = 0x0b;

/// The reader: `main` loads `target` through a GOT slot at `-O1`.
const READER_SRC: &[u8] =
    b"extern int target;\nint main(void) { return target; }\n";
const DEF_SRC: &[u8] = b"int target = 5;\n";

/// Each `__got` slot has a row, and the row names the slot's symbol as
/// an external definition.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn each_got_slot_names_its_symbol() {
    let Some((dir, image)) = link("rows") else {
        return;
    };
    let (_, got_size, _) =
        section(&image, b"__got").expect("the reference reads a GOT slot");
    let slots = got_size / 8;
    let slots = usize::try_from(slots).unwrap_or(0);
    let Some((symoff, nsyms, stroff, _)) = symtab(&image) else {
        panic!("the image must carry LC_SYMTAB");
    };
    let dysym = dysymtab(&image).expect("the image must carry LC_DYSYMTAB");
    assert_eq!(
        dysym.nindirectsyms,
        u32::try_from(slots).unwrap_or(u32::MAX),
        "one indirect row per GOT slot must be counted"
    );
    assert_ne!(dysym.indirectsymoff, 0, "the table must be located");
    for slot in 0..slots {
        let at = dysym.indirectsymoff + slot * 4;
        let row = read_u32(&image, at).expect("the row must be file-backed");
        assert!(
            row < nsyms,
            "slot {slot} row {row} must index the symbol table"
        );
        let entry = symoff + usize::try_from(row).unwrap_or(0) * 16;
        let strx = read_u32(&image, entry).expect("the row must exist");
        let n_type = image[entry + 4];
        let name = name_at(&image, stroff, strx);
        assert_eq!(
            name, b"_target",
            "slot {slot} must name the symbol it holds"
        );
        assert_eq!(
            n_type, 0x0f,
            "slot {slot} row must be an external definition"
        );
        assert!(
            row >= dysym.iextdefsym
                && row < dysym.iextdefsym + dysym.nextdefsym,
            "slot {slot} row must sit in the external-defined range"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

/// The indirect rows are u32s: `indirectsymoff` must be 4-aligned even
/// when the string table before them ends off a 4-byte boundary.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn indirect_table_sits_four_aligned() {
    let Some((dir, image)) = link("align") else {
        return;
    };
    let (_, _, _, strsize) =
        symtab(&image).expect("the image must carry LC_SYMTAB");
    assert_ne!(
        strsize % 4,
        0,
        "the fixture must end its string table off a 4-byte boundary"
    );
    let dysym = dysymtab(&image).expect("the image must carry LC_DYSYMTAB");
    assert_eq!(
        dysym.indirectsymoff % 4,
        0,
        "the u32 indirect rows must start 4-aligned"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Compiles both objects for darwin and links them. `tag` keeps each
/// test's scratch directory distinct within one test process.
fn link(tag: &str) -> Option<(PathBuf, Vec<u8>)> {
    let clang = which("clang")?;
    let dir = std::env::temp_dir()
        .join(format!("xold_machoindirect_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    let objects = [
        compile(&clang, &dir, READER_SRC, "r")?,
        compile(&clang, &dir, DEF_SRC, "d")?,
    ];
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
        eprintln!("skipping macho indirect: no darwin target");
        return None;
    }
    Some(obj)
}

// --- readers ---------------------------------------------------------------

/// `(address, size, file offset)` of a section, by name.
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
                    // `section_64::offset` is a u32 at +48; `align` follows
                    // at +52, so an 8-byte read would pack the pair.
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

/// `(symoff, nsyms, stroff, strsize)` from `LC_SYMTAB`.
fn symtab(image: &[u8]) -> Option<(usize, u32, usize, u32)> {
    let ncmds = usize::try_from(read_u32(image, 16)?).ok()?;
    let mut at = 32;
    for _ in 0..ncmds {
        let cmd = read_u32(image, at)?;
        let size = usize::try_from(read_u32(image, at + 4)?).ok()?;
        if cmd == LC_SYMTAB {
            return Some((
                usize::try_from(read_u32(image, at + 8)?).ok()?,
                read_u32(image, at + 12)?,
                usize::try_from(read_u32(image, at + 16)?).ok()?,
                read_u32(image, at + 20)?,
            ));
        }
        if size < 8 {
            return None;
        }
        at += size;
    }
    None
}

/// The `LC_DYSYMTAB` fields the test reads.
struct Dysym {
    iextdefsym: u32,
    nextdefsym: u32,
    indirectsymoff: usize,
    nindirectsyms: u32,
}

/// Reads those fields from `LC_DYSYMTAB`.
fn dysymtab(image: &[u8]) -> Option<Dysym> {
    let ncmds = usize::try_from(read_u32(image, 16)?).ok()?;
    let mut at = 32;
    for _ in 0..ncmds {
        let cmd = read_u32(image, at)?;
        let size = usize::try_from(read_u32(image, at + 4)?).ok()?;
        if cmd == LC_DYSYMTAB {
            return Some(Dysym {
                iextdefsym: read_u32(image, at + 16)?,
                nextdefsym: read_u32(image, at + 20)?,
                indirectsymoff: usize::try_from(read_u32(image, at + 56)?)
                    .ok()?,
                nindirectsyms: read_u32(image, at + 60)?,
            });
        }
        if size < 8 {
            return None;
        }
        at += size;
    }
    None
}

/// The NUL-terminated name at a string-table offset.
fn name_at(image: &[u8], stroff: usize, strx: u32) -> &[u8] {
    let start = stroff + usize::try_from(strx).unwrap_or(usize::MAX);
    let rest = image.get(start..).unwrap_or(&[]);
    rest.split(|&b| b == 0).next().unwrap_or(rest)
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
