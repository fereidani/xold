//! A dependency's default-version definition is the one the link records.
//!
//! A shared object may spell one name twice and tell the two rows apart by
//! version alone. `foo@GLIBC_2.2.5` beside `foo@@GLIBC_2.17` is two `.dynsym`
//! entries both named `foo`, at different addresses and usually of different
//! sizes: the first is a superseded definition kept for binaries linked before
//! the interface changed, the second the default one an unversioned reference
//! binds to. Keying the export table on the bare name makes which of the two
//! xold records an accident of `.dynsym` order, and the size it takes from it
//! decides how large a copy relocation's slot is.
//!
//! The fixture is that shape built small: `foo@VER_1` is four bytes and
//! `foo@@VER_2` is sixteen, and the system linker lists the superseded row
//! first. The observable is the copy slot: sized from the default row it is
//! sixteen bytes, sized from the superseded one it is four, and a four-byte
//! slot for a sixteen-byte object leaves the loader copying a quarter of it.
//!
//! The slot alone is not the whole answer. A copy relocation makes the import's
//! `.dynsym` row *defined* -- it names the executable's own `.bss` slot -- yet
//! the object behind it is still the dependency's, so the row has to record
//! which version of it the program was built against. Without that, glibc binds
//! the unversioned reference to whichever definition it reaches first and
//! copies four bytes into a sixteen-byte slot; the sentinel in the last word
//! never arrives. Both system linkers record it: `readelf -V` on their output
//! shows `foo@VER_2 (2)` against a `.gnu.version_r` entry naming `VER_2` in
//! `libver.so`. So the tests below assert the version *and* run the program,
//! which is the behaviour the slot size only stands in for.
//!
//! xold has one name per symbol, so the default definition is the one it can
//! represent. lld's answer is larger -- it names a non-default definition
//! `foo@VERSION` so the two stay distinct symbols -- and is a separate feature.
//!
//! Gated on `clang` and the system linker's `--version-script`; if either is
//! missing the test prints a note and returns.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{
    elf::{ObjectFile, constants::VER_NDX_HIDDEN},
    icf::IcfMode,
    linker::link_dyn_exec,
    reloc::x86_64::R_X86_64_COPY,
};

mod common;

/// The dependency: one name, two definitions. `foo_old` is the superseded
/// four-byte one and `foo_new` the sixteen-byte default; the sentinel sits in
/// the last word, where only a correctly sized copy reaches it.
const LIB_SRC: &str = "__asm__(\".symver foo_old,foo@VER_1\");\n\
     __asm__(\".symver foo_new,foo@@VER_2\");\n\
     int foo_old = 5;\n\
     int foo_new[4] = { 5, 0, 0, 7 };\n";

/// The version script that makes `VER_2` the default and keeps `VER_1` as a
/// non-default definition of the same name.
const LIB_MAP: &str = "VER_1 { global: foo; local: *; };\n\
     VER_2 { global: foo; local: *; } VER_1;\n";

/// The consumer: a non-PIC reference, which is what selects a copy relocation
/// over a GOT slot.
const MAIN_SRC: &str = "extern int foo[4];\n\
     int main(void) { return foo[3]; }\n";

/// The freestanding entry stub: calls `main` and exits with its return value.
const START_SRC: &str = "    .text\n    .globl _start\n_start:\n\
     call main\n movl %eax, %edi\n movl $60, %eax\n syscall\n";

/// The size of the default definition, and so of the copy slot.
const DEFAULT_SIZE: u64 = 16;

/// The size of the superseded definition, which the slot must not take.
const SUPERSEDED_SIZE: u64 = 4;

/// The sentinel in the last word of `foo@@VER_2`, which the program returns.
/// Only a sixteen-byte copy of the default definition reaches it.
const SENTINEL: i32 = 7;

/// The version name the copy-relocated row must record, and the soname the
/// `.gnu.version_r` entry must name it under.
const DEFAULT_VERSION: &[u8] = b"VER_2";
const DEP_SONAME: &[u8] = b"libver.so";

/// A private working directory. The name carries the process id so concurrent
/// test binaries cannot delete each other's files.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_depvers_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Compiles or assembles `src` to `obj` with the host clang, passing `args`
/// through. Returns `None` when clang is unavailable.
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
    let obj = dir.join("libver.o");
    build(LIB_SRC, "c", &obj, dir, &["-fPIC"])?;
    let map = dir.join("libver.map");
    fs::write(&map, LIB_MAP).ok()?;
    let so = dir.join("libver.so");
    let ok = Command::new(clang)
        .arg("-shared")
        .arg("-o")
        .arg(&so)
        .arg(&obj)
        .arg(format!("-Wl,--version-script={}", map.display()))
        .arg("-Wl,-soname,libver.so")
        .status()
        .ok()?
        .success();
    ok.then_some(so)
}

/// The `.dynsym` rows named `foo`, as `(is_default_version, st_size)`, in the
/// order the dependency lists them.
fn foo_rows(bytes: &[u8]) -> Vec<(bool, u64)> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let (Ok(Some(dynsym)), Ok(vt)) =
        (obj.dynamic_symbols(), obj.version_table())
    else {
        return Vec::new();
    };
    dynsym
        .iter()
        .enumerate()
        .filter(|(_, sym)| dynsym.name(sym) == b"foo")
        .map(|(i, sym)| {
            (vt.version_of(i) & VER_NDX_HIDDEN == 0, sym.st_size.get())
        })
        .collect()
}

/// The `(st_value, st_size)` of the `.dynsym` row named `name`, which for a
/// copy-relocated import is the slot the link reserved.
fn dynsym_slot(bytes: &[u8], name: &[u8]) -> Option<(u64, u64)> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let dynsym = obj.dynamic_symbols().ok()??;
    dynsym
        .iter()
        .find(|sym| dynsym.name(sym) == name)
        .map(|sym| (sym.st_value.get(), sym.st_size.get()))
}

/// The `.gnu.version` index carried by the `.dynsym` row named `name`.
fn versym_of(bytes: &[u8], name: &[u8]) -> Option<u16> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let dynsym = obj.dynamic_symbols().ok()??;
    let vt = obj.version_table().ok()?;
    let i = dynsym.iter().position(|sym| dynsym.name(sym) == name)?;
    Some(vt.version_of(i))
}

/// Every `.gnu.version_r` row, as `(soname, version name, version index)`.
fn verneed_rows(bytes: &[u8]) -> Vec<(Vec<u8>, Vec<u8>, u16)> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Ok(vt) = obj.version_table() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for need in &vt.needs {
        for aux in &need.aux {
            out.push((need.soname.to_vec(), aux.name.to_vec(), aux.index));
        }
    }
    out
}

/// The `r_offset` of every `R_X86_64_COPY` in `.rela.dyn`.
fn copy_relocations(bytes: &[u8]) -> Vec<u64> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(data) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".rela.dyn")
        .and_then(|s| obj.section_data(s).ok())
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for row in data.as_chunks::<24>().0 {
        let off = u64::from_le_bytes(row[..8].try_into().unwrap_or([0; 8]));
        let info = u64::from_le_bytes(row[8..16].try_into().unwrap_or([0; 8]));
        if u32::try_from(info & 0xffff_ffff) == Ok(R_X86_64_COPY) {
            out.push(off);
        }
    }
    out
}

/// Asserts the fixture has the shape the defect needs: the superseded row
/// first, the default one second. Recording the first row wins by accident
/// when the order is the other way round.
fn assert_dependency_shape(lib: &Path) {
    let lib_bytes = fs::read(lib).expect("dependency is readable");
    let rows = foo_rows(&lib_bytes);
    assert_eq!(
        rows.len(),
        2,
        "the dependency must define foo twice: {rows:?}"
    );
    assert_eq!(
        rows[0],
        (false, SUPERSEDED_SIZE),
        "the superseded row must come first, else the fixture does not \
         exercise the defect"
    );
    assert_eq!(rows[1], (true, DEFAULT_SIZE), "the default row is second");
}

/// Compiles the consumer and the entry stub and links them against `lib` with
/// xold, returning the image path.
fn link_program(dir: &Path, lib: PathBuf, interp: &[u8]) -> PathBuf {
    let main_o = dir.join("depvers_main.o");
    let start_o = dir.join("depvers_start.o");
    build(MAIN_SRC, "c", &main_o, dir, &["-fno-pie", "-fno-pic"])
        .expect("host clang compiles -fno-pie");
    build(START_SRC, "S", &start_o, dir, &[])
        .expect("host clang assembles the entry stub");

    let prog = dir.join("depvers_prog");
    link_dyn_exec(
        &[main_o, start_o, lib],
        &prog,
        b"_start",
        interp,
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-exec link must succeed");
    prog
}

/// The proof: the copy slot xold reserves for `foo` is sized from the default
/// definition, not from the superseded row `.dynsym` happens to list first,
/// and the row records that version so the loader copies from the same one.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_copy_slot_is_sized_from_the_default_version() {
    let dir = workdir("copy");
    let Some(interp) = interpreter() else {
        eprintln!("skipping dep-version test: interpreter path unknown");
        return;
    };
    let Some(lib) = build_dependency(&dir) else {
        eprintln!("skipping dep-version test: host toolchain unavailable");
        return;
    };
    assert_dependency_shape(&lib);
    let prog = link_program(&dir, lib, interp.as_slice());

    let bytes = fs::read(&prog).expect("image is readable");
    let copies = copy_relocations(&bytes);
    assert_eq!(
        copies.len(),
        1,
        "an absolute data import takes one R_X86_64_COPY"
    );
    let slot = dynsym_slot(&bytes, b"foo").expect("foo must reach .dynsym");
    assert_eq!(
        slot.0, copies[0],
        "the dynsym entry must define foo at the slot the relocation fills"
    );
    assert_eq!(
        slot.1, DEFAULT_SIZE,
        "the copy slot must be sized from foo@@VER_2, not from foo@VER_1"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A copy-relocated import records the version its definition carries, exactly
/// as GNU ld and lld do: the `.dynsym` row's `.gnu.version` index names a
/// `.gnu.version_r` entry for `VER_2` in `libver.so`.
///
/// Being defined at the executable's `.bss` slot is what used to keep the row
/// out of the version plan, which offered it only the undefined rows.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_copy_relocated_import_records_its_version() {
    let dir = workdir("versioned");
    let Some(interp) = interpreter() else {
        eprintln!("skipping dep-version test: interpreter path unknown");
        return;
    };
    let Some(lib) = build_dependency(&dir) else {
        eprintln!("skipping dep-version test: host toolchain unavailable");
        return;
    };
    assert_dependency_shape(&lib);
    let prog = link_program(&dir, lib, interp.as_slice());

    let bytes = fs::read(&prog).expect("image is readable");
    let rows = verneed_rows(&bytes);
    assert_eq!(
        rows.len(),
        1,
        "one version of one dependency is referenced: {rows:?}"
    );
    assert_eq!(rows[0].0, DEP_SONAME, "the VERNEED names the dependency");
    assert_eq!(rows[0].1, DEFAULT_VERSION, "and the default version of foo");

    let versym = versym_of(&bytes, b"foo").expect("foo must reach .dynsym");
    assert_eq!(
        versym, rows[0].2,
        "the copy-relocated row must carry the VERNEED index, not \
         VER_NDX_GLOBAL"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The behaviour the slot size stands in for: with the version recorded, the
/// loader copies the sixteen bytes of `foo@@VER_2` and the program reads the
/// sentinel in the last word. Bound to `foo@VER_1` instead it copies four
/// bytes and reads whatever `.bss` was zeroed to.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_copied_object_is_the_default_version_at_runtime() {
    let dir = workdir("run");
    let Some(interp) = interpreter() else {
        eprintln!("skipping dep-version test: interpreter path unknown");
        return;
    };
    let Some(lib) = build_dependency(&dir) else {
        eprintln!("skipping dep-version test: host toolchain unavailable");
        return;
    };
    assert_dependency_shape(&lib);
    let prog = link_program(&dir, lib, interp.as_slice());

    let status = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(SENTINEL),
        "the program must read the last word of the sixteen-byte definition"
    );
    let _ = fs::remove_dir_all(&dir);
}
