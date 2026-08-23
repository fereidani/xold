//! Tentative definitions get storage in the Mach-O link.
//!
//! A tentative definition (`-fcommon`) arrives as `N_UNDF` with the size
//! in `n_value`: the symbol is defined, but no section of any input holds
//! it. The resolver only accepted `N_SECT` and `N_ABS` definitions, so
//! `-fcommon` objects failed with `xold: undefined reference to _g`.
//!
//! The rules the link must apply (lld merges commons the same way,
//! `lld/MachO/SymbolTable.cpp`):
//!
//! * every tentative definition of one name merges into one storage of the
//!   largest size and strictest alignment,
//! * a real definition displaces the common and no storage is allocated,
//! * references (through a GOT slot, on darwin) resolve to the storage or to
//!   the real definition.
//!
//! Gated on a `clang` that can target darwin; without one the tests
//! print a note and return.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{input::Input, macho::link_macho};

mod common;

const LC_SEGMENT_64: u32 = 0x19;
const LC_SYMTAB: u32 = 0x02;
const LC_DYSYMTAB: u32 = 0x0b;

/// A tentative definition with a second, larger tentative in another
/// file, plus a reference.
const TENTATIVE_SRC: &[u8] = b"int g;\nint use(void) { return g; }\n";
const LARGER_SRC: &[u8] = b"int g[128];\n";
const READER_SRC: &[u8] = b"extern int g;\nint main(void) { return g; }\n";

/// A tentative definition of the name the real definition claims, and
/// the reference both must answer.
const TENTATIVE2_SRC: &[u8] = b"int g2;\n";
const REAL_SRC: &[u8] = b"int g2 = 7;\n";
const REAL_READER_SRC: &[u8] =
    b"extern int g2;\nint main(void) { return g2; }\n";

/// The larger tentative size wins: `__common` is 512 bytes, and the GOT
/// slot the reference reads holds the section's base.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn tentative_definitions_merge_into_one_storage() {
    let Some((dir, image)) = link(&["t", "l", "r"], "merge") else {
        return;
    };
    let common = section(&image, b"__common").expect("a common must exist");
    assert_eq!(
        common.1, 512,
        "two tentative definitions must merge into the larger size"
    );
    let slot = got_slot(&image).expect("the reference reads a GOT slot");
    assert_eq!(
        slot, common.0,
        "the reference must resolve to the common storage"
    );
    // The definition must also read as one in the output symbol table: a
    // defined row in `__common`, with the undefined range empty.
    assert_eq!(
        undefined_rows(&image),
        0,
        "a defined name must not sit in the undefined range"
    );
    let (symoff, stroff) = symtab_extent(&image);
    let rows = symtab_rows(&image, symoff, stroff);
    let g = rows
        .iter()
        .find(|(name, _, _, _)| name == b"_g")
        .expect("the common must have a symbol row");
    assert_eq!(g.1, 0x0f, "the row must be a defined external");
    assert_eq!(g.3, common.0, "the row must carry the storage address");
    assert_ne!(g.2, 0, "the row must name a section");
    let _ = fs::remove_dir_all(&dir);
}

/// A real definition displaces the tentative one: no `__common` section
/// is emitted, and the reference resolves to the definition's storage.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_real_definition_displaces_a_tentative_one() {
    let Some((dir, image)) = link(&["t2", "def", "dr"], "real") else {
        return;
    };
    assert!(
        section(&image, b"__common").is_none(),
        "no storage may be allocated for a displaced common"
    );
    let slot = got_slot(&image).expect("the reference reads a GOT slot");
    let data = section(&image, b"__data").expect("the definition has data");
    assert_eq!(
        slot, data.0,
        "the reference must resolve to the real definition"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Which source each stem names.
fn source(stem: &str) -> &'static [u8] {
    match stem {
        "t" => TENTATIVE_SRC,
        "t2" => TENTATIVE2_SRC,
        "l" => LARGER_SRC,
        "r" => READER_SRC,
        "dr" => REAL_READER_SRC,
        _ => REAL_SRC,
    }
}

/// Compiles the named sources with `-fcommon` and links them, in order.
fn link(stems: &[&str], tag: &str) -> Option<(PathBuf, Vec<u8>)> {
    let clang = which("clang")?;
    let dir = std::env::temp_dir()
        .join(format!("xold_machocommon_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    let mut objects = Vec::new();
    for stem in stems {
        objects.push(compile(&clang, &dir, source(stem), stem)?);
    }
    let out = dir.join("prog");
    let files: Vec<Input<'_>> =
        objects.iter().map(|p| Input::Path(p.as_path())).collect();
    let res = link_macho(&files, &out, b"_main");
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let image = fs::read(&out).ok()?;
    Some((dir, image))
}

/// Compiles one fixture for darwin, with `-fcommon` so the tentative
/// definitions stay tentative.
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
        .args(["--target=x86_64-apple-darwin", "-fcommon", "-O0", "-c"])
        .arg(&path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping macho common: no darwin target");
        return None;
    }
    Some(obj)
}

// --- readers ---------------------------------------------------------------

/// `(address, size)` of a section in the linked image, by name. A
/// zero-fill section like `__common` has no file offset, so only the
/// virtual extent is reported.
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

/// `nundefsym` from `LC_DYSYMTAB`, or `u32::MAX` when absent.
fn undefined_rows(image: &[u8]) -> u32 {
    let ncmds = usize::try_from(read_u32(image, 16).unwrap_or(0)).unwrap_or(0);
    let mut at = 32;
    for _ in 0..ncmds {
        let Some(cmd) = read_u32(image, at) else {
            return u32::MAX;
        };
        let Some(size) = read_u32(image, at + 4) else {
            return u32::MAX;
        };
        if cmd == LC_DYSYMTAB {
            // `iundefsym` at +24, `nundefsym` at +28.
            return read_u32(image, at + 28).unwrap_or(u32::MAX);
        }
        if size < 8 {
            return u32::MAX;
        }
        at += usize::try_from(size).unwrap_or(0);
    }
    u32::MAX
}

/// `(symoff, stroff)` from `LC_SYMTAB`.
fn symtab_extent(image: &[u8]) -> (usize, usize) {
    let ncmds = usize::try_from(read_u32(image, 16).unwrap_or(0)).unwrap_or(0);
    let mut at = 32;
    for _ in 0..ncmds {
        let Some(cmd) = read_u32(image, at) else {
            return (0, 0);
        };
        let Some(size) = read_u32(image, at + 4) else {
            return (0, 0);
        };
        if cmd == LC_SYMTAB {
            let symoff = usize::try_from(read_u32(image, at + 8).unwrap_or(0))
                .unwrap_or(0);
            let stroff = usize::try_from(read_u32(image, at + 16).unwrap_or(0))
                .unwrap_or(0);
            return (symoff, stroff);
        }
        if size < 8 {
            return (0, 0);
        }
        at += usize::try_from(size).unwrap_or(0);
    }
    (0, 0)
}

/// `(name, n_type, n_sect, n_value)` per row.
fn symtab_rows(
    image: &[u8],
    symoff: usize,
    stroff: usize,
) -> Vec<(Vec<u8>, u8, u8, u64)> {
    let mut out = Vec::new();
    let mut at = symoff;
    while let Some(strx) = read_u32(image, at) {
        let Some(n_type) = image.get(at + 4).copied() else {
            break;
        };
        let Some(n_sect) = image.get(at + 5).copied() else {
            break;
        };
        let value = read_u64(image, at + 8).unwrap_or(0);
        let start = stroff + usize::try_from(strx).unwrap_or(0);
        let name = image
            .get(start..)
            .unwrap_or(&[])
            .split(|&b| b == 0)
            .next()
            .unwrap_or(&[])
            .to_vec();
        out.push((name, n_type, n_sect, value));
        at += 16;
        if out.len() > 4096 {
            break;
        }
    }
    out
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
