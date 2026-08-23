//! The Mach-O symbol table covers what the header says it covers.
//!
//! Index 0 was a zeroed `nlist`, borrowed from ELF. Mach-O has no null symbol,
//! and the invented one sat outside every `LC_DYSYMTAB` range: `ilocalsym`
//! started at 1, so the table the header described did not cover the table
//! that was written. `strip` and `ld -r` consumers reject that.
//!
//! And every resolved external reference was still emitted as an undefined
//! row while the header claimed `MH_NOUNDEFS`. `nm` showed `U _foo` in a
//! static image that had `_foo` right there in `__text`. A reference the link
//! satisfied is not undefined, so those rows are dropped.
//!
//! Darwin images cannot run on this host, so the test reads the tables the
//! consumers would.
//!
//! Gated on a `clang` that can target darwin; without one the tests print a
//! note and return.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{input::Input, macho::link_macho};

mod common;

/// Two files: one defines `_shared`, the other references it, so the reference
/// is satisfied inside the image.
const DEF: &[u8] = b"int shared_value = 41;\n\
    int shared(void) { return shared_value; }\n";
const USE: &[u8] = b"int shared(void);\n\
    int main(void) { return shared() - 41; }\n";

const LC_SYMTAB: u32 = 0x02;
const LC_DYSYMTAB: u32 = 0x0b;
const N_TYPE: u8 = 0x0e;
const N_UNDF: u8 = 0x00;

/// The dysymtab ranges cover every entry in the table.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_ranges_cover_the_whole_table() {
    let Some((dir, image)) = link("cover") else {
        return;
    };
    let (_, nsyms, _, _) = symtab(&image).expect("LC_SYMTAB present");
    let (ilocal, nlocal, iext, next, iundef, nundef) =
        dysymtab(&image).expect("LC_DYSYMTAB present");
    assert_eq!(ilocal, 0, "the locals start the table, with nothing ahead");
    assert_eq!(iext, ilocal + nlocal, "the externals follow the locals");
    assert_eq!(iundef, iext + next, "the undefined follow the externals");
    assert_eq!(
        iundef + nundef,
        nsyms,
        "and the three ranges account for every entry"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A reference the link satisfied is not emitted as undefined.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_satisfied_reference_is_not_undefined() {
    let Some((dir, image)) = link("undef") else {
        return;
    };
    let (symoff, nsyms, stroff, _) = symtab(&image).expect("LC_SYMTAB");
    let mut undefined = Vec::new();
    for i in 0..nsyms {
        let at =
            usize::try_from(symoff + u64::from(i) * 16).expect("entry offset");
        let n_type = *image.get(at + 4).expect("n_type");
        if n_type & N_TYPE != N_UNDF {
            continue;
        }
        let strx = read_u32(&image, at).expect("n_strx");
        undefined.push(name(&image, stroff + u64::from(strx)));
    }
    assert!(
        !undefined.iter().any(|n| n == b"_shared"),
        "the image defines _shared, so a row calling it undefined \
         contradicts MH_NOUNDEFS: {undefined:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The first entry is a real symbol with a name, not an invented blank.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_table_starts_with_a_real_symbol() {
    let Some((dir, image)) = link("first") else {
        return;
    };
    let (symoff, nsyms, stroff, _) = symtab(&image).expect("LC_SYMTAB");
    assert!(nsyms > 0, "the fixture must produce symbols");
    let at = usize::try_from(symoff).expect("entry offset");
    let strx = read_u32(&image, at).expect("n_strx");
    assert!(
        !name(&image, stroff + u64::from(strx)).is_empty(),
        "index 0 is a symbol the inputs named, not a zeroed entry Mach-O has \
         no concept of"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Compiles both fixtures for darwin and links them.
fn link(prefix: &str) -> Option<(PathBuf, Vec<u8>)> {
    let clang = which("clang")?;
    let dir = std::env::temp_dir()
        .join(format!("xold_machosymtab_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    let mut objs = Vec::new();
    for (stem, src) in [("d", DEF), ("u", USE)] {
        let path = dir.join(format!("{stem}.c"));
        let obj = dir.join(format!("{stem}.o"));
        fs::write(&path, src).ok()?;
        let built = Command::new(&clang)
            .args(["--target=x86_64-apple-darwin", "-O1", "-c"])
            .arg(&path)
            .arg("-o")
            .arg(&obj)
            .status()
            .ok()?
            .success();
        if !built {
            eprintln!("skipping macho-symtab-shape {prefix}: no darwin target");
            let _ = fs::remove_dir_all(&dir);
            return None;
        }
        objs.push(obj);
    }
    let files: Vec<Input<'_>> =
        objs.iter().map(|p| Input::Path(p.as_path())).collect();
    let out = dir.join("prog");
    let res = link_macho(&files, &out, b"_main");
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let image = fs::read(&out).ok()?;
    Some((dir, image))
}

// --- readers ---------------------------------------------------------------

/// `(symoff, nsyms, stroff, strsize)` from `LC_SYMTAB`.
fn symtab(image: &[u8]) -> Option<(u64, u32, u64, u32)> {
    let at = command(image, LC_SYMTAB)?;
    Some((
        u64::from(read_u32(image, at + 8)?),
        read_u32(image, at + 12)?,
        u64::from(read_u32(image, at + 16)?),
        read_u32(image, at + 20)?,
    ))
}

/// The six range fields of `LC_DYSYMTAB` this test reasons about.
fn dysymtab(image: &[u8]) -> Option<(u32, u32, u32, u32, u32, u32)> {
    let at = command(image, LC_DYSYMTAB)?;
    Some((
        read_u32(image, at + 8)?,
        read_u32(image, at + 12)?,
        read_u32(image, at + 16)?,
        read_u32(image, at + 20)?,
        read_u32(image, at + 24)?,
        read_u32(image, at + 28)?,
    ))
}

/// The file offset of the first load command of type `want`.
fn command(image: &[u8], want: u32) -> Option<usize> {
    let ncmds = usize::try_from(read_u32(image, 16)?).ok()?;
    let mut at = 32;
    for _ in 0..ncmds {
        let cmd = read_u32(image, at)?;
        let size = usize::try_from(read_u32(image, at + 4)?).ok()?;
        if cmd == want {
            return Some(at);
        }
        if size < 8 {
            return None;
        }
        at += size;
    }
    None
}

/// The NUL-terminated name at a string-table offset.
fn name(image: &[u8], off: u64) -> Vec<u8> {
    let at = usize::try_from(off).unwrap_or(usize::MAX);
    let rest = image.get(at..).unwrap_or_default();
    rest.split(|&b| b == 0).next().unwrap_or_default().to_vec()
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .and_then(|c| <[u8; 4]>::try_from(c).ok())
        .map(u32::from_le_bytes)
}
