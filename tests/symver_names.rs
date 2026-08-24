//! A symbol spelled with a version suffix is still that symbol.
//!
//! `.symver` in assembly or C makes the assembler emit the versioned spelling
//! beside the plain one: a definition comes out as its implementation name
//! next to `foo@@VERS_1`, and an annotated reference as `bar@VER_2`. Kept
//! verbatim, those names missed each other in the symbol table: a plain
//! reference to `foo` reported `foo` undefined while `foo@@VERS_1` sat
//! defined in the same link, and a reference spelled `bar@VER_2` did not see
//! the dependency's `bar@@VER_2` export at all.
//!
//! lld folds the spellings by keying its table on the name before the first
//! `@` (`SymbolTable::insert` stems `foo@@VERS_1` to `foo`) and registering a
//! dependency's exports under both spellings (`SharedFile::parse` adds
//! `name@ver` beside `name`). A same-file redefinition is not a duplicate
//! there (`ObjFile::postParse` accepts a second definition from the file
//! that already defined the name), which is what lets the paired rows of one
//! `.symver` directive coexist.
//!
//! Gated on `clang` (and the system linker for the versioned dependency);
//! when either is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{
    elf::ObjectFile,
    icf::IcfMode,
    linker::{link_dyn_exec, link_to},
};

mod common;

/// The definition: `.symver` renames the implementation symbol to
/// `foo@@VERS_1` (keeping `foo_impl` beside it), which a plain `call foo`
/// must bind to.
const DEF_SRC: &str = "    .text\n    .globl foo_impl\n\
     .symver foo_impl,foo@@VERS_1\n\
     foo_impl:\n    movl $42, %eax\n    ret\n";

/// The reference: a plain undefined `foo`.
const START_SRC: &str = "    .text\n    .globl _start\n_start:\n\
     call foo\n movl %eax, %edi\n movl $60, %eax\n syscall\n";

/// The dependency: `bar@@VER_2` as the default version of `bar`.
const LIB_SRC: &str = "__asm__(\".symver bar_impl,bar@@VER_2\");\n\
     int bar_impl = 0x61;\n";

const LIB_MAP: &str = "VER_2 { global: bar; local: *; };\n";

/// The consumer names the import with its version: `.symver` rewrites the
/// reference to `bar@VER_2`, so the object carries an undefined row spelled
/// with the suffix.
const MAIN_SRC: &str = "extern int bar;\n\
     __asm__(\".symver bar,bar@VER_2\");\n\
     int main(void) { return bar == 0x61 ? 0 : 1; }\n";

const DYN_START_SRC: &str = "    .text\n    .globl _start\n_start:\n\
     call main\n movl %eax, %edi\n movl $60, %eax\n syscall\n";

const DEP_SONAME: &[u8] = b"libver.so";
const VERSION: &[u8] = b"VER_2";

/// A plain reference binds a definition spelled `foo@@VERS_1`: the alias and
/// the plain name are one symbol, not two that miss each other.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_plain_reference_binds_a_version_suffixed_definition() {
    let dir = workdir("def");
    let Some(def) = assemble(DEF_SRC, &dir.join("sv_def.o"), &dir) else {
        eprintln!("skipping symver def test: host toolchain unavailable");
        return;
    };
    let Some(start) = assemble(START_SRC, &dir.join("sv_start.o"), &dir) else {
        eprintln!("skipping symver def test: host toolchain unavailable");
        return;
    };
    let prog = dir.join("sv_prog");
    link_to(&[def, start], &prog, b"_start", false, IcfMode::None, false)
        .expect("the plain reference must bind the foo@@VERS_1 definition");
    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        assert_plain_reference_contract(
            &fs::read(&prog).expect("read linked image"),
        );
        let _ = fs::remove_dir_all(&dir);
        return;
    }
    let status = Command::new(&prog).status().expect("the program runs");
    assert_eq!(
        status.code(),
        Some(42),
        "the call must reach the versioned definition's code"
    );
    let _ = fs::remove_dir_all(&dir);
}

fn assert_plain_reference_contract(bytes: &[u8]) {
    let obj = ObjectFile::parse(bytes).expect("valid ELF");
    let symtab = obj.symbol_table().expect("read symtab").expect("symtab");
    let value = |name: &[u8]| {
        symtab
            .syms
            .iter()
            .find(|sym| symtab.name(sym) == name)
            .map(|sym| sym.st_value.get())
            .unwrap_or_else(|| {
                panic!("{} is defined", String::from_utf8_lossy(name))
            })
    };
    let foo = value(b"foo");
    assert_eq!(
        foo,
        value(b"foo_impl"),
        "the versioned alias names its implementation"
    );
    assert_eq!(
        image_at(&obj, foo, 6),
        Some(b"\xb8\x2a\0\0\0\xc3".as_slice()),
        "the definition returns 42"
    );
    let start = value(b"_start");
    let body = image_at(&obj, start, 5).expect("_start call");
    assert_eq!(body[0], 0xe8, "_start has a direct call");
    let disp = i32::from_le_bytes(body[1..5].try_into().unwrap());
    assert_eq!(
        start
            .wrapping_add(5)
            .wrapping_add(i64::from(disp).cast_unsigned()),
        foo,
        "the plain call resolves to foo@@VERS_1"
    );
}

fn image_at<'a>(
    obj: &ObjectFile<'a>,
    addr: u64,
    len: usize,
) -> Option<&'a [u8]> {
    for sec in obj.sections() {
        let base = sec.sh_addr.get();
        if addr < base || addr >= base.saturating_add(sec.sh_size.get()) {
            continue;
        }
        let at = usize::try_from(addr - base).ok()?;
        return obj.section_data(sec).ok()?.get(at..at.checked_add(len)?);
    }
    None
}

/// A reference spelled `bar@VER_2` resolves against the dependency's
/// `bar@@VER_2` export and records that version for the import.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_version_suffixed_reference_binds_the_dependency_export() {
    let dir = workdir("ref");
    let Some(interp) = interpreter() else {
        eprintln!("skipping symver ref test: interpreter path unknown");
        return;
    };
    let Some(lib) = build_dependency(&dir) else {
        eprintln!("skipping symver ref test: host toolchain unavailable");
        return;
    };
    let Some(main) = compile(MAIN_SRC, &dir.join("svr_main.o"), &dir) else {
        eprintln!("skipping symver ref test: host toolchain unavailable");
        return;
    };
    let Some(start) = assemble(DYN_START_SRC, &dir.join("svr_start.o"), &dir)
    else {
        eprintln!("skipping symver ref test: host toolchain unavailable");
        return;
    };
    let prog = dir.join("svr_prog");
    link_dyn_exec(
        &[main, start, lib],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    )
    .expect("the bar@VER_2 reference must bind the dependency's bar");

    let bytes = fs::read(&prog).expect("image is readable");
    let versym =
        versym_of(&bytes, b"bar").expect("bar must reach the output .dynsym");
    let rows = verneed_rows(&bytes);
    let named = rows
        .iter()
        .find(|(soname, name, _)| {
            soname == DEP_SONAME && name.as_slice() == VERSION
        })
        .unwrap_or_else(|| {
            panic!(
                "the link must record {VERSION:?} of {DEP_SONAME:?}: {rows:?}"
            )
        });
    assert_eq!(
        versym, named.2,
        "the import's version index names VER_2, the version it asked for"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures ---------------------------------------------------------------

fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_symver_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Assembles `src` to `obj` with the host clang. Returns `None` when
/// unavailable.
fn assemble(src: &str, obj: &Path, dir: &Path) -> Option<PathBuf> {
    build(src, "S", obj, dir, &[])
}

/// Compiles `src` to `obj` with the host clang.
fn compile(src: &str, obj: &Path, dir: &Path) -> Option<PathBuf> {
    build(src, "c", obj, dir, &["-fno-pie", "-fno-pic"])
}

/// Compiles or assembles `src` to `obj` with the host clang, passing `args`
/// through. Returns `None` when unavailable.
fn build(
    src: &str,
    ext: &str,
    obj: &Path,
    dir: &Path,
    args: &[&str],
) -> Option<PathBuf> {
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
        .then(|| obj.to_path_buf())
}

/// Builds the versioned dependency with the system linker, which applies the
/// version script. Returns `None` when the host cannot build it.
fn build_dependency(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let obj = dir.join("libver.o");
    build(LIB_SRC, "c", &obj, dir, &["-fPIC"])?;
    let map = dir.join("libver.map");
    fs::write(&map, LIB_MAP).ok()?;
    let so = dir.join("libver.so");
    let ok = Command::new(clang)
        .arg("-shared")
        .arg("-fPIC")
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

// --- readers ----------------------------------------------------------------

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
