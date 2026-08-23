//! An import's version index must be unmasked before the VERDEF lookup.
//!
//! A `.gnu.version` halfword is a version index with flags in its high bits:
//! `VER_NDX_HIDDEN` (0x8000) marks a definition that is not the default one
//! to bind an unversioned reference to. The index itself still names the
//! `vd_ndx` it always did, so the flag comes off before the dependency's
//! version-definition table is consulted (lld masks the same way,
//! `InputFiles.cpp`). Using the raw halfword as the lookup key missed the
//! table for every hidden-version definition -- `bar@VER_1` below, and on
//! this host's libc `pthread_mutexattr_getprotocol@GLIBC_2.4` -- so the
//! import's `.dynsym` row came out unversioned with no Vernaux recorded,
//! and the loader could not tell which definition the link had chosen.
//!
//! Gated on `clang` and the system linker's `--version-script`; if either
//! is missing the test prints a note and returns.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_dyn_exec};

mod common;

/// The dependency: `bar` as a single-`@` `.symver` alias, which the system
/// linker marks `VER_NDX_HIDDEN` -- a definition an unversioned reference is
/// not supposed to prefer. `def@@VER_2` gives the library a default version
/// so the two nodes sort.
const LIB_SRC: &str = "__asm__(\".symver bar_impl,bar@VER_1\");\n\
     int bar_impl = 0x51;\n\
     __asm__(\".symver def_impl,def@@VER_2\");\n\
     int def_impl = 2;\n";

/// Two nodes: `VER_1` holds the hidden definition the fixture imports.
const LIB_MAP: &str = "VER_1 { global: bar; local: *; };\n\
     VER_2 { global: def; local: *; } VER_1;\n";

/// The consumer: an unversioned reference to the hidden-version definition,
/// which is how a plain C program names it.
const MAIN_SRC: &str = "extern int bar;\n\
     int main(void) { return bar == 0x51 ? 0 : 1; }\n";

const START_SRC: &str = "    .text\n    .globl _start\n_start:\n\
     call main\n movl %eax, %edi\n movl $60, %eax\n syscall\n";

const DEP_SONAME: &[u8] = b"libhid.so";
const HIDDEN_VERSION: &[u8] = b"VER_1";

/// The import records the hidden version, not "unversioned".
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_hidden_version_import_still_records_its_version() {
    let dir = workdir("hidden");
    let Some(interp) = interpreter() else {
        eprintln!("skipping hidden-version test: interpreter path unknown");
        return;
    };
    let Some(lib) = build_dependency(&dir) else {
        eprintln!("skipping hidden-version test: host toolchain unavailable");
        return;
    };
    assert_dependency_shape(&lib);

    let main_o = dir.join("hid_main.o");
    let start_o = dir.join("hid_start.o");
    build(MAIN_SRC, "c", &main_o, &dir, &["-fno-pie", "-fno-pic"])
        .expect("host clang compiles -fno-pie");
    build(START_SRC, "S", &start_o, &dir, &[])
        .expect("host clang assembles the entry stub");
    let prog = dir.join("hid_prog");
    link_dyn_exec(
        &[main_o, start_o, lib],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-exec link must succeed");

    let bytes = fs::read(&prog).expect("image is readable");
    let versym =
        versym_of(&bytes, b"bar").expect("bar must reach the output .dynsym");
    let rows = verneed_rows(&bytes);
    let named = rows
        .iter()
        .find(|(soname, name, _)| {
            soname == DEP_SONAME && name.as_slice() == HIDDEN_VERSION
        })
        .unwrap_or_else(|| {
            panic!("the link must record VER_1 of {DEP_SONAME:?}: {rows:?}")
        });
    assert_ne!(
        versym, 1,
        "an unversioned row cannot name which definition was chosen"
    );
    assert_eq!(
        versym, named.2,
        "the row carries the VERNEED index of the hidden version"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_dephid_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

fn build(
    src: &str,
    ext: &str,
    obj: &Path,
    dir: &Path,
    args: &[&str],
) -> Option<()> {
    let clang = which("clang")?;
    let stem = obj.file_stem()?.to_str().unwrap_or("unit");
    let file = dir.join(format!("{stem}.{ext}"));
    fs::write(&file, src).ok()?;
    Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c"])
        .args(args)
        .arg("-o")
        .arg(obj)
        .arg(&file)
        .status()
        .ok()?
        .success()
        .then_some(())
}

/// Builds the versioned dependency with the *system* linker, which is what
/// applies the version script. `None` when the host cannot build it.
fn build_dependency(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let obj = dir.join("libhid.o");
    build(LIB_SRC, "c", &obj, dir, &["-fPIC"])?;
    let map = dir.join("libhid.map");
    fs::write(&map, LIB_MAP).ok()?;
    let so = dir.join("libhid.so");
    let ok = Command::new(clang)
        .arg("-shared")
        .arg("-o")
        .arg(&so)
        .arg(&obj)
        .arg(format!("-Wl,--version-script={}", map.display()))
        .arg("-Wl,-soname,libhid.so")
        .status()
        .ok()?
        .success();
    ok.then_some(so)
}

/// The dependency's `bar` row really is hidden: the fixture must exercise the
/// masked lookup or it proves nothing.
fn assert_dependency_shape(lib: &Path) {
    let bytes = fs::read(lib).expect("dependency is readable");
    let obj = ObjectFile::parse(&bytes).expect("dependency parses");
    let dynsym = obj
        .dynamic_symbols()
        .ok()
        .flatten()
        .expect("dependency has a .dynsym");
    let vt = obj.version_table().expect("version table parses");
    let (i, _) = dynsym
        .iter()
        .enumerate()
        .find(|(_, sym)| dynsym.name(sym) == b"bar")
        .expect("the dependency defines bar");
    let raw = vt.version_of(i);
    assert_eq!(
        raw & 0x8000,
        0x8000,
        "bar's versym must carry VER_NDX_HIDDEN, found {raw:#x}"
    );
    assert_eq!(raw & !0x8000, 2, "and name version index 2 (VER_1)");
}

// --- readers ---------------------------------------------------------------

/// The `.gnu.version` index carried by the `.dynsym` row named `name`.
fn versym_of(bytes: &[u8], name: &[u8]) -> Option<u16> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let dynsym = obj.dynamic_symbols().ok()??;
    let vt = obj.version_table().ok()?;
    dynsym
        .iter()
        .enumerate()
        .find(|(_, sym)| dynsym.name(sym) == name)
        .map(|(i, _)| vt.version_of(i))
}

/// The `(soname, version name, index)` rows of `.gnu.version_r`.
fn verneed_rows(bytes: &[u8]) -> Vec<(Vec<u8>, Vec<u8>, u16)> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Ok(vt) = obj.version_table() else {
        return Vec::new();
    };
    vt.needs
        .iter()
        .flat_map(|need| {
            need.aux.iter().map(move |aux| {
                (need.soname.to_vec(), aux.name.to_vec(), aux.index)
            })
        })
        .collect()
}
