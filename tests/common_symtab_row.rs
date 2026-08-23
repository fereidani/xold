//! A common symbol is published in `.symtab`.
//!
//! `export_for` returned `None` for anything that was not `Defined`, so a
//! common -- a tentative definition the linker allocates itself -- got an
//! address in `.bss` and no symbol-table row at all. `nm` showed nothing for a
//! `-fcommon` global, and neither did any debugger. lld emits a row at the
//! allocated address.
//!
//! Found while writing the regression test for the common-alignment fix,
//! which had to assert on `.bss`'s placement because the symbol it wanted to
//! name was not in the table.
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

/// One common of each of two sizes, so the row's size field means something.
const SRC: &[u8] = b"int small;\n\
    double wide[8];\n\
    int touch(void) { small = 1; wide[7] = 2.0; return small; }\n\
    void _start(void) { }\n";

/// The commons are in the table, at their allocated addresses.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_common_gets_a_symbol_table_row() {
    let Some(dir) = workdir("row") else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    let small = row(&bytes, b"small").expect("`int small;` must be published");
    let wide = row(&bytes, b"wide").expect("`double wide[8];` too");
    assert_eq!(small.size, 4, "the row carries the object's size");
    assert_eq!(wide.size, 64, "and so does the larger one");
    assert!(small.value > 0, "at the address it was allocated");
    assert!(wide.value > 0, "and so is the other");
    assert_ne!(small.value, wide.value, "at addresses of their own");
    let _ = fs::remove_dir_all(&dir);
}

/// Each row names `.bss`, which is where the storage is.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_row_names_the_section_the_storage_is_in() {
    let Some(dir) = workdir("shndx") else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    let bss = section_index(&bytes, b".bss").expect(".bss must be present");
    for name in [b"small".as_slice(), b"wide"] {
        let sym = row(&bytes, name).expect("published");
        assert_eq!(
            sym.shndx,
            bss,
            "{} must name .bss, not SHN_UNDEF or SHN_COMMON: a common in a \
             linked image is ordinary storage",
            String::from_utf8_lossy(name)
        );
        // STT_OBJECT is 1.
        assert_eq!(sym.info & 0xf, 1, "and it is an object, not a function");
    }
    let _ = fs::remove_dir_all(&dir);
}

/// The address is inside `.bss`, so the row and the storage agree.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_address_is_inside_the_section_it_names() {
    let Some(dir) = workdir("inside") else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    let (addr, size) =
        section_extent(&bytes, b".bss").expect(".bss must be present");
    for name in [b"small".as_slice(), b"wide"] {
        let sym = row(&bytes, name).expect("published");
        assert!(
            (addr..addr + size).contains(&sym.value),
            "{} is published at {:#x}, outside .bss [{addr:#x}, {:#x})",
            String::from_utf8_lossy(name),
            sym.value,
            addr + size
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping common-symtab-row {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_commonrow_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture with `-fcommon` and links it statically.
fn link(dir: &Path) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let src = dir.join("c.c");
    let obj = dir.join("c.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fno-pic", "-fcommon", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping common-symtab-row: clang cannot build it");
        return None;
    }
    let out = dir.join("prog");
    let res = link_to(
        std::slice::from_ref(&obj),
        &out,
        b"_start",
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

// --- readers ---------------------------------------------------------------

/// The fields of one `.symtab` row these tests reason about.
struct Row {
    value: u64,
    size: u64,
    shndx: u16,
    info: u8,
}

/// A `.symtab` row by name.
fn row(bytes: &[u8], name: &[u8]) -> Option<Row> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab.iter().find(|s| symtab.name(s) == name).map(|s| Row {
        value: s.st_value.get(),
        size: s.st_size.get(),
        shndx: s.st_shndx.get(),
        info: s.st_info,
    })
}

/// The section-header index of a section by name.
fn section_index(bytes: &[u8], name: &[u8]) -> Option<u16> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let at = obj
        .sections()
        .iter()
        .position(|s| obj.section_name(s) == name)?;
    u16::try_from(at).ok()
}

/// The address and size of a section by name.
fn section_extent(bytes: &[u8], name: &[u8]) -> Option<(u64, u64)> {
    let obj = ObjectFile::parse(bytes).ok()?;
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == name)
        .map(|s| (s.sh_addr.get(), s.sh_size.get()))
}
