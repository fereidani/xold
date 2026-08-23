//! The GOT-offset relocations store an offset, not an address.
//!
//! `R_X86_64_GOT64` and `R_X86_64_GOT32` spell `G + A` in the psABI: the
//! symbol's slot *offset from the base of the GOT*, the number a register
//! holding `_GLOBAL_OFFSET_TABLE_` is added to. xold stored the slot's
//! absolute address instead -- the whole virtual address of the table entry --
//! so code that then added the base back computed an address far past the
//! table. `R_X86_64_GOT32` was worse off still: the type had no row, and a
//! `.long sym@GOT` ended the link as "unsupported relocation 3".
//!
//! lld maps both to `R_GOTPLT`, which computes `GOT[sym] + A - GOT`
//! (`lld/ELF/Arch/X86_64.cpp`,
//! `lld/ELF/InputSection.cpp`). Its base is `.got.plt` where xold's
//! is `.got`, the same base `_GLOBAL_OFFSET_TABLE_` publishes in each -- the
//! value is measured from the table the code was told to add.
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

/// A defined datum reached only through the table, at both offsets' widths.
const SRC: &[u8] = b"    .text\n\
    .globl _start\n\
    .type _start,@function\n\
_start:\n\
    ret\n\
    .data\n\
    .globl foo\n\
    .type foo,@object\n\
foo: .quad 0x1234\n\
    .globl g64\n\
    .type g64,@object\n\
g64: .quad foo@GOT\n\
    .globl g32\n\
    .type g32,@object\n\
g32: .long foo@GOT\n";

/// Both widths store the slot's offset from the GOT base, which is the number
/// the table actually holds at that offset -- the datum's address.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn got_offsets_are_measured_from_the_table_base() {
    let Some(dir) = workdir() else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    let obj = parse(&bytes);
    let (got_addr, got) =
        section(&obj, b".got").expect("a GOT reference allocates the table");
    let foo_addr = symbol_value(&obj, b"foo").expect("foo defined");

    // The slot xold allocated for `foo` holds its address; its offset is the
    // value both relocations were supposed to store.
    let slot = got
        .as_chunks::<8>()
        .0
        .iter()
        .position(|c| u64::from_le_bytes(*c) == foo_addr)
        .expect("the slot holds the datum's address");
    let offset = u64::try_from(slot * 8).expect("slot offset in range");
    debug_assert!(offset < got.len() as u64);
    debug_assert!(got_addr + offset < got_addr + got.len() as u64);

    let g64 = symbol_value(&obj, b"g64").expect("g64 defined");
    let g32 = symbol_value(&obj, b"g32").expect("g32 defined");
    let data_addr = section_addr(&obj, b".data").expect(".data present");
    let data = section_bytes(&obj, b".data").expect(".data bytes");
    let at64 = usize::try_from(g64 - data_addr).expect("g64 in .data");
    let at32 = usize::try_from(g32 - data_addr).expect("g32 in .data");
    assert_eq!(
        read_u64(&data, at64),
        Some(offset),
        "GOT64 stores G + A: the slot's offset from _GLOBAL_OFFSET_TABLE_, \
         not the slot's address"
    );
    assert_eq!(
        read_u32(&data, at32),
        Some(u32::try_from(offset).expect("offset fits 32 bits")),
        "GOT32 stores the same offset at four bytes"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures ---------------------------------------------------------------

/// Creates a fresh working directory, or `None` (after printing a note) when
/// the host cannot build the inputs.
fn workdir() -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping got-relative-value: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_gotrel_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Assembles and links the fixture.
fn link(dir: &Path) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let src = dir.join("s.S");
    let obj = dir.join("s.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fno-pic", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping got-relative-value: clang cannot assemble it");
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

// --- readers ----------------------------------------------------------------

fn parse(bytes: &[u8]) -> ObjectFile<'_> {
    ObjectFile::parse(bytes).expect("a linked image parses")
}

/// The `st_value` of a symbol in the output's `.symtab`.
fn symbol_value(obj: &ObjectFile<'_>, name: &[u8]) -> Option<u64> {
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab
        .iter()
        .find(|s| symtab.name(s) == name)
        .map(|s| s.st_value.get())
}

/// The virtual address of a named output section.
fn section_addr(obj: &ObjectFile<'_>, name: &[u8]) -> Option<u64> {
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == name)
        .map(|s| s.sh_addr.get())
}

/// `(address, bytes)` of a named output section.
fn section(obj: &ObjectFile<'_>, name: &[u8]) -> Option<(u64, Vec<u8>)> {
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == name)?;
    let bytes = obj.section_data(shdr).ok().map(<[u8]>::to_vec)?;
    Some((shdr.sh_addr.get(), bytes))
}

/// The bytes of a named output section.
fn section_bytes(obj: &ObjectFile<'_>, name: &[u8]) -> Option<Vec<u8>> {
    section(obj, name).map(|(_, bytes)| bytes)
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
