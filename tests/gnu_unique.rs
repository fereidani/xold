//! An `STB_GNU_UNIQUE` definition ranks like a weak one, and the first
//! among the unique copies wins.
//!
//! `-fgnu-unique` marks vague-linkage data a program must share
//! process-wide: the compiler raises the binding of a COMDAT-held static
//! from `STB_WEAK` to `STB_GNU_UNIQUE` so the loader can keep one copy
//! alive however many shared objects define it. For the linker the
//! binding means what lld spells out in `shouldReplace`: an incoming
//! `STB_GLOBAL` overrides `STB_WEAK`/`STB_GNU_UNIQUE`, and the first
//! among the weak and unique copies is the one kept -- preferring an
//! incoming unique to an existing weak picks a copy a discarded COMDAT
//! section may have lived in (`lld/ELF/Symbols.cpp`).
//! An undefined unique reference is not weak at all: it demands a
//! definition like a strong one does.
//!
//! xold ranked every unique definition as strong, so the second object
//! to define the name was a duplicate-symbol error on input the ABI
//! allows -- and a legal `-fgnu-unique` program refused to link.
//!
//! No tool on this host emits the binding, so the fixture compiles
//! ordinary definitions and rewrites the binding nibble of the chosen
//! row in each object's `.symtab` -- ELF surgery on one byte, keeping
//! the rest of the object exactly what the compiler wrote.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_to};

mod common;

/// The first object's definition.
const A_SRC: &[u8] = b"double u = 1.5;\ndouble first(void) { return u; }\n";

/// The second object's definition of the same name.
const B_SRC: &[u8] = b"double u = 2.5;\ndouble second(void) { return u; }\n";

/// A plain strong definition, which outranks a unique one.
const STRONG_SRC: &[u8] =
    b"double u = 7.0;\ndouble strong(void) { return u; }\n";

/// Two unique definitions of one name link; the first object's is kept.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_first_unique_definition_wins() {
    let Some(dir) = workdir("first") else {
        return;
    };
    let Some(clang) = which("clang") else {
        return;
    };
    let Some(a) = compile(&dir, &clang, "a", A_SRC, STB_GNU_UNIQUE) else {
        return;
    };
    let Some(b) = compile(&dir, &clang, "b", B_SRC, STB_GNU_UNIQUE) else {
        return;
    };
    let out = dir.join("prog");
    let res = link_to(&[a, b], &out, b"first", false, IcfMode::None, false);
    assert!(res.is_ok(), "one copy is the whole point: {:?}", res.err());
    let (bits, bind) = read_u(&out);
    assert_eq!(bits, 1.5_f64.to_bits(), "the first object's copy is kept");
    assert_eq!(bind, STB_GNU_UNIQUE, "the binding survives the link");
    let _ = fs::remove_dir_all(&dir);
}

/// Reversed input order keeps the other copy: first among equals.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_winner_follows_the_input_order() {
    let Some(dir) = workdir("second") else {
        return;
    };
    let Some(clang) = which("clang") else {
        return;
    };
    let Some(a) = compile(&dir, &clang, "a", A_SRC, STB_GNU_UNIQUE) else {
        return;
    };
    let Some(b) = compile(&dir, &clang, "b", B_SRC, STB_GNU_UNIQUE) else {
        return;
    };
    let out = dir.join("prog");
    let res = link_to(&[b, a], &out, b"second", false, IcfMode::None, false);
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let (bits, _) = read_u(&out);
    assert_eq!(bits, 2.5_f64.to_bits(), "the first object's copy is kept");
    let _ = fs::remove_dir_all(&dir);
}

/// A strong definition overrides a unique one, whichever arrived first.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_strong_definition_overrides_a_unique_one() {
    let Some(dir) = workdir("strong") else {
        return;
    };
    let Some(clang) = which("clang") else {
        return;
    };
    let Some(unique) = compile(&dir, &clang, "u", B_SRC, STB_GNU_UNIQUE) else {
        return;
    };
    let Some(strong) = compile(&dir, &clang, "s", STRONG_SRC, STB_GLOBAL)
    else {
        return;
    };
    for inputs in [vec![unique.clone(), strong.clone()], vec![strong, unique]] {
        let out = dir.join("prog");
        let res =
            link_to(&inputs, &out, b"strong", false, IcfMode::None, false);
        assert!(res.is_ok(), "global beats unique: {:?}", res.err());
        let (bits, bind) = read_u(&out);
        assert_eq!(bits, 7.0_f64.to_bits(), "the strong copy is kept");
        assert_eq!(bind, STB_GLOBAL, "the winner's binding is what is written");
    }
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// `STB_GNU_UNIQUE`, the binding the loader keys its one-copy search on.
const STB_GNU_UNIQUE: u8 = 10;
/// `STB_GLOBAL`.
const STB_GLOBAL: u8 = 1;

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping gnu-unique {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_gnuunique_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles one source and rewrites `u`'s binding nibble to `bind`.
/// `STB_GNU_UNIQUE` no tool here emits; `STB_GLOBAL` asks for no rewrite
/// at all.
fn compile(
    dir: &Path,
    clang: &Path,
    name: &str,
    src: &[u8],
    bind: u8,
) -> Option<PathBuf> {
    let src_path = dir.join(format!("{name}.c"));
    let obj = dir.join(format!("{name}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c", "-fno-pie"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping gnu-unique: clang cannot build {name}");
        return None;
    }
    if bind != STB_GLOBAL {
        patch_binding(&obj, b"u", bind)
            .expect("the object carries a row for u");
    }
    Some(obj)
}

/// Rewrites the binding of `name`'s row in an object's `.symtab`.
fn patch_binding(obj: &Path, name: &[u8], bind: u8) -> Option<()> {
    let mut bytes = fs::read(obj).ok()?;
    let shoff = usize::try_from(read_u64(&bytes, 0x28)?).unwrap_or(0);
    let shnum = read_u16(&bytes, 0x3c)?;
    let sh = |i: usize| shoff + i * 64;
    let symtab = (1..usize::from(shnum))
        .find(|&i| read_u32(&bytes, sh(i) + 4) == Some(2))?;
    let off = read_u64(&bytes, sh(symtab) + 0x18)?;
    let size = read_u64(&bytes, sh(symtab) + 0x20)?;
    let link = read_u32(&bytes, sh(symtab) + 0x28)? as usize;
    let stroff = read_u64(&bytes, sh(link) + 0x18)?;
    for at in (0..usize::try_from(size).unwrap_or(0)).step_by(24) {
        let row = usize::try_from(off).unwrap_or(0) + at;
        let st_name = usize::try_from(read_u32(&bytes, row)?).ok()?;
        let start = usize::try_from(stroff).unwrap_or(0) + st_name;
        let end = bytes.get(start..)?.iter().position(|&b| b == 0)? + start;
        if &bytes[start..end] != name {
            continue;
        }
        let info = bytes[row + 4];
        bytes[row + 4] = (bind << 4) | (info & 0xf);
        fs::write(obj, &bytes).ok()?;
        return Some(());
    }
    None
}

/// Reads `u`'s value bits and binding out of a linked image.
fn read_u(out: &Path) -> (u64, u8) {
    let bytes = fs::read(out).expect("read the image");
    let image = ObjectFile::parse(&bytes).expect("parse the image");
    let table = image.symbol_table().ok().flatten().expect(".symtab");
    let sym = table
        .syms
        .iter()
        .find(|s| table.name(s) == b"u")
        .expect("u is defined");
    let shndx = usize::from(sym.st_shndx.get());
    let sect = image.sections().get(shndx).expect("the holding section");
    let data = image.section_data(sect).expect("the section's bytes");
    let value = sym.st_value.get().wrapping_sub(sect.sh_addr.get());
    let at = usize::try_from(value).expect("u sits in its section");
    let cell = data.get(at..at + 8).expect("eight bytes of u");
    (
        u64::from_le_bytes(<[u8; 8]>::try_from(cell).expect("a cell")),
        sym.st_info >> 4,
    )
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

fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
    bytes
        .get(at..at + 8)
        .and_then(|c| <[u8; 8]>::try_from(c).ok())
        .map(u64::from_le_bytes)
}
