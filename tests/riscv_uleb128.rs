//! The RISC-V `SET_ULEB128`/`SUB_ULEB128` pair.
//!
//! GAS emits the pair for `.uleb128 a - b` whenever relaxation is enabled,
//! which is the default, and a `.gcc_except_table` call-site table is exactly
//! that. So every C++ object with exception handling carries them, and every
//! one of them was refused as an unsupported relocation type.
//!
//! The two entries share an offset and encode one difference, which no single
//! relocation expression spells. They are folded in the per-section pre-pass
//! that already pairs `PCREL_HI20` with its `LO12`: the difference lands in
//! the `SET` entry's addend against no symbol, and the `SUB` entry becomes
//! `R_RISCV_NONE`. lld folds the pair at the same point, in its own apply loop
//! (`lld/ELF/Arch/RISCV.cpp`).
//!
//! Writing it back is a rewrite rather than a store: the field is a ULEB128
//! whose length the bytes already there fix, because the table's other entries
//! follow it. That length is read from the field itself -- every byte but the
//! last carries a continuation bit -- which is why the rewrite hook needs the
//! slot in hand and not only what follows it. A value with more bits than the
//! field holds is an overflow, as it is in lld.
//!
//! Gated on a `clang` that can target `riscv64`; without one the tests print a
//! note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_to};

mod common;

/// Two one-byte ULEB fields, each holding `b - a` -- twelve, the distance
/// between the two labels. The `.reloc` directives spell the pair out, because
/// an assembler resolves a same-section difference itself unless relaxation
/// forces it not to.
const SRC: &[u8] = b"    .text\n\
    .globl _start\n\
_start:\n\
    .globl a\n\
a:\n\
    nop\n\
    nop\n\
    nop\n\
    .globl b\n\
b:\n\
    ret\n\
    .section .gcc_except_table,\"a\",@progbits\n\
    .globl tbl\n\
tbl:\n\
    .byte 0x00\n\
    .byte 0x00\n\
    .reloc tbl, R_RISCV_SET_ULEB128, b\n\
    .reloc tbl, R_RISCV_SUB_ULEB128, a\n\
    .reloc tbl+1, R_RISCV_SET_ULEB128, b\n\
    .reloc tbl+1, R_RISCV_SUB_ULEB128, a\n";

/// A two-byte field, so the rewrite has to keep a continuation bit rather than
/// collapse the encoding.
const WIDE_SRC: &[u8] = b"    .text\n\
    .globl _start\n\
_start:\n\
    .globl a\n\
a:\n\
    nop\n\
    .globl b\n\
b:\n\
    ret\n\
    .section .gcc_except_table,\"a\",@progbits\n\
    .globl tbl\n\
tbl:\n\
    .byte 0x80\n\
    .byte 0x00\n\
    .reloc tbl, R_RISCV_SET_ULEB128, b\n\
    .reloc tbl, R_RISCV_SUB_ULEB128, a\n";

/// Each field holds the difference between the two labels.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_pair_writes_the_label_difference() {
    let Some(dir) = workdir("pair") else {
        return;
    };
    let Some(bytes) = link(&dir, SRC, "pair") else {
        return;
    };
    let a = symbol_value(&bytes, b"a").expect("a present");
    let b = symbol_value(&bytes, b"b").expect("b present");
    let table = table_bytes(&bytes).expect("the table is in the image");
    let want = u8::try_from(b - a).expect("the fixture keeps it small");
    assert_eq!(
        table.first().copied(),
        Some(want),
        "the field holds b - a, which is {want}"
    );
    assert_eq!(
        table.get(1).copied(),
        Some(want),
        "and so does the second, which is its own one-byte field"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A two-byte field stays two bytes: the encoding's length belongs to the
/// table, not to the linker.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_wider_field_keeps_its_length() {
    let Some(dir) = workdir("wide") else {
        return;
    };
    let Some(bytes) = link(&dir, WIDE_SRC, "wide") else {
        return;
    };
    let a = symbol_value(&bytes, b"a").expect("a present");
    let b = symbol_value(&bytes, b"b").expect("b present");
    let table = table_bytes(&bytes).expect("the table is in the image");
    let diff = b - a;
    assert!(diff < 0x80, "the fixture keeps the value inside seven bits");
    #[allow(clippy::cast_possible_truncation)]
    let low = 0x80 | (diff as u8 & 0x7f);
    assert_eq!(
        table.first().copied(),
        Some(low),
        "the first byte keeps its continuation bit"
    );
    assert_eq!(
        table.get(1).copied(),
        Some(0),
        "and the second holds what is left, which is nothing"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping riscv-uleb128 {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_uleb_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Assembles and links one fixture.
fn link(dir: &Path, src: &[u8], stem: &str) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let src_path = dir.join(format!("{stem}.S"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clang)
        .args(["--target=riscv64-unknown-elf", "-march=rv64g", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping riscv-uleb128: no riscv64 assembler");
        return None;
    }
    let out = dir.join(stem);
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

/// The bytes the `.gcc_except_table` contributed, which land in `.rodata`.
fn table_bytes(bytes: &[u8]) -> Option<Vec<u8>> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".rodata")?;
    obj.section_data(shdr).ok().map(<[u8]>::to_vec)
}

/// The `st_value` of a symbol in the output's `.symtab`.
fn symbol_value(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab
        .iter()
        .find(|s| symtab.name(s) == name)
        .map(|s| s.st_value.get())
}
