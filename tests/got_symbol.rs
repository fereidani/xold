//! `_GLOBAL_OFFSET_TABLE_` is a name this linker supplies.
//!
//! An object may reach the GOT through the symbol rather than through a
//! relocation that names a slot: hand-written position-independent assembly
//! does, and so does anything built the way a 32-bit ABI reaches its table.
//! Nothing defined it, so such an object died with "undefined reference".
//!
//! `R_X86_64_GOTPC32` appeared to work only because it ignores the symbol's
//! value and measures from the GOT base directly -- which left the two free to
//! disagree the moment the symbol did exist. They cannot now: the symbol is
//! read from the same accessor the relocation's base comes from.
//!
//! That base is `.got`, where lld's is `.got.plt`
//! (`lld/ELF/Arch/X86_64.cpp`). The difference is not observable
//! here: every GOT slot xold allocates lives in `.got`, so measuring from
//! `.got` measures from the table the slots are in, while lld's choice is an
//! x86-64 convention for code indexing the lazy-binding table directly.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::{ObjectFile, constants::SHN_UNDEF},
    icf::IcfMode,
    linker::link_to,
};

mod common;

/// Reaches the GOT through the symbol, and allocates a slot so the link has a
/// real table to name.
const SRC: &[u8] = b"    .text\n\
    .globl _start\n\
    .type _start,@function\n\
_start:\n\
    leaq  _GLOBAL_OFFSET_TABLE_(%rip), %rax\n\
    movq  $60, %rax\n\
    movq  $0, %rdi\n\
    syscall\n";

/// A slot-allocating translation unit, so `.got` exists.
const SLOT_SRC: &[u8] = b"extern int ext;\n\
    int *take(void) { return &ext; }\n\
    int ext = 3;\n";

/// The symbol resolves, and to the start of `.got`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_got_symbol_is_defined_at_the_table() {
    let Some(dir) = workdir("defined") else {
        return;
    };
    let Some(bytes) = link(&dir, true) else {
        return;
    };
    let sym = symbol(&bytes, b"_GLOBAL_OFFSET_TABLE_")
        .expect("an object may name the GOT symbol, so the linker defines it");
    assert_ne!(
        sym.shndx, SHN_UNDEF,
        "a name this linker supplies must not be published undefined"
    );
    let got = section_addr(&bytes, b".got").expect(".got present");
    assert_eq!(
        sym.value, got,
        "the symbol is the base every GOT-relative expression measures from"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// With no GOT at all the symbol is still defined, at the load base -- where
/// an empty table would begin.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_link_with_no_got_still_defines_it() {
    let Some(dir) = workdir("empty") else {
        return;
    };
    let Some(bytes) = link(&dir, false) else {
        return;
    };
    assert!(
        section_addr(&bytes, b".got").is_none(),
        "this fixture must allocate no GOT slot"
    );
    let sym = symbol(&bytes, b"_GLOBAL_OFFSET_TABLE_")
        .expect("the symbol is defined whether or not a slot was allocated");
    assert_ne!(sym.shndx, SHN_UNDEF, "and it is not an undefined import");
    assert_ne!(
        sym.value, 0,
        "zero is not an address: an empty table begins at the load base"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The symbol is hidden: it describes where this linker put its own table,
/// which is not something another image binds to.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_got_symbol_is_hidden() {
    let Some(dir) = workdir("hidden") else {
        return;
    };
    let Some(bytes) = link(&dir, true) else {
        return;
    };
    let sym = symbol(&bytes, b"_GLOBAL_OFFSET_TABLE_").expect("defined");
    assert_eq!(sym.other & 3, 2, "STV_HIDDEN, as lld's is");
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping got-symbol {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_gotsym_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Links the fixture, optionally with a translation unit that allocates a GOT
/// slot.
fn link(dir: &Path, with_slot: bool) -> Option<Vec<u8>> {
    let mut paths = vec![build(dir, SRC, "g", "S")?];
    if with_slot {
        paths.push(build(dir, SLOT_SRC, "slot", "c")?);
    }
    let out = dir.join(if with_slot { "with" } else { "without" });
    let res = link_to(&paths, &out, b"_start", false, IcfMode::None, false);
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

/// Compiles or assembles one input.
fn build(dir: &Path, src: &[u8], stem: &str, ext: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src_path = dir.join(format!("{stem}.{ext}"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIC", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping got-symbol: clang cannot build the fixture");
        return None;
    }
    Some(obj)
}

// --- readers ---------------------------------------------------------------

/// The fields of one `.symtab` row these tests reason about.
struct Row {
    value: u64,
    shndx: u16,
    other: u8,
}

/// Reads a symbol out of the output's `.symtab`.
fn symbol(bytes: &[u8], name: &[u8]) -> Option<Row> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab.iter().find(|s| symtab.name(s) == name).map(|s| Row {
        value: s.st_value.get(),
        shndx: s.st_shndx.get(),
        other: s.st_other,
    })
}

/// The virtual address of a named output section.
fn section_addr(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == name)
        .map(|s| s.sh_addr.get())
}
