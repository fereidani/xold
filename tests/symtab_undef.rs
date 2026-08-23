//! The output `.symtab` carries a row for every name the image imports.
//!
//! A reference the link resolves from a shared dependency is still part of
//! the program's interface: `nm` prints it as a `U` entry, a debugger lists
//! it among what the image needs, and `ld -r` style tooling walks the rows
//! to see the imports. lld writes one `SHN_UNDEF` row per resolved
//! unresolved name, `STT_NOTYPE` with the reference's binding, beside the
//! defined globals (`SymtabSection::finalize` walks the whole symbol table).
//!
//! xold's table stopped at the definitions, so the import side of a dynamic
//! program was invisible to every tool that reads `.symtab` -- the rows
//! simply were not there.
//!
//! The fixture links a program against a shared library and reads the image
//! back: the imported name has an `SHN_UNDEF` row with `STB_GLOBAL`
//! binding, and the weak probe beside it has one with `STB_WEAK`. Before
//! the fix both names were absent from the table.
//!
//! Gated on `clang`; if it is missing the test prints a note and returns.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_dyn_exec};

mod common;

/// The definitions the shared library carries.
const LIB_SRC: &[u8] = b"\
long probe(void) { return 11; }
long fallback(void) { return 5; }
";

/// The program: a strong import and a weak one, both used.
const MAIN_SRC: &[u8] = b"\
extern long probe(void);
__attribute__((weak)) extern long fallback(void);

void _start(void) {
    long r = probe() + (fallback ? fallback() : 0);
    __asm__ volatile (\"syscall\"
                      :
                      : \"a\"(60L), \"D\"(r)
                      : \"memory\", \"rcx\", \"r11\");
    __builtin_unreachable();
}
";

/// `STB_GLOBAL`.
const STB_GLOBAL: u8 = 1;
/// `STB_WEAK`.
const STB_WEAK: u8 = 2;

/// The imported names carry `SHN_UNDEF` rows with their bindings.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn imported_names_get_symtab_rows() {
    let Some(dir) = workdir() else {
        return;
    };
    let Some([main, lib]) = build(&dir) else {
        return;
    };
    let Some(interp) = interpreter() else {
        eprintln!("skipping symtab-undef: no system interpreter found");
        return;
    };
    let out = dir.join("prog");
    let res = link_dyn_exec(
        &[main, lib],
        &out,
        b"_start",
        &interp,
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let rows = read_rows(&out);
    let probe = row(&rows, b"probe").expect("probe has a .symtab row");
    assert_eq!(
        probe.shndx, 0,
        "an unresolved reference is SHN_UNDEF, not placed"
    );
    assert_eq!(
        probe.bind, STB_GLOBAL,
        "the strong import keeps its binding"
    );
    assert_eq!(probe.value, 0, "no definition means no value");
    let weak = row(&rows, b"fallback").expect("fallback has a .symtab row");
    assert_eq!(weak.shndx, 0, "the weak import is SHN_UNDEF too");
    assert_eq!(weak.bind, STB_WEAK, "the weak import keeps its binding");
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// One `.symtab` row the test read back.
struct Row {
    bind: u8,
    shndx: u16,
    value: u64,
}

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir() -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping symtab-undef: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_symtabund_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the shared library and the program that imports from it.
fn build(dir: &Path) -> Option<[PathBuf; 2]> {
    let clang = which("clang")?;
    let lib_src = dir.join("lib.c");
    let lib = dir.join("libprobe.so");
    fs::write(&lib_src, LIB_SRC).ok()?;
    let built = Command::new(&clang)
        .args(["--target=x86_64-linux-gnu", "-shared", "-fPIC"])
        .arg(&lib_src)
        .arg("-o")
        .arg(&lib)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping symtab-undef: clang cannot build the library");
        return None;
    }
    let main_src = dir.join("main.c");
    let main = dir.join("main.o");
    fs::write(&main_src, MAIN_SRC).ok()?;
    let built = Command::new(&clang)
        .args(["--target=x86_64-linux-gnu", "-c", "-fPIE"])
        .arg(&main_src)
        .arg("-o")
        .arg(&main)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping symtab-undef: clang cannot build the program");
        return None;
    }
    Some([main, lib])
}

/// Reads `name -> Row` out of the image's `.symtab`.
fn read_rows(out: &Path) -> Vec<(Vec<u8>, Row)> {
    let bytes = fs::read(out).expect("read the image");
    let image = ObjectFile::parse(&bytes).expect("parse the image");
    let table = image
        .symbol_table()
        .ok()
        .flatten()
        .expect("the image has a .symtab");
    table
        .syms
        .iter()
        .filter(|s| s.st_name.get() != 0)
        .map(|s| {
            (
                table.name(s).to_vec(),
                Row {
                    bind: s.st_info >> 4,
                    shndx: s.st_shndx.get(),
                    value: s.st_value.get(),
                },
            )
        })
        .collect()
}

/// Finds the row for `name`.
fn row<'a>(rows: &'a [(Vec<u8>, Row)], name: &[u8]) -> Option<&'a Row> {
    rows.iter().find(|(n, _)| n == name).map(|(_, r)| r)
}
