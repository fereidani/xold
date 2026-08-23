//! The x86-64 relocation types that were falling through to "unsupported".
//!
//! Each of these is something a compiler or assembler emits and lld
//! classifies, and each ended the link outright.
//!
//! The narrow widths -- `R_X86_64_8`, `PC8`, `PC16` -- are table rows and
//! nothing more. The size pair, `SIZE32` and `SIZE64`, needed one new
//! expression: they read the symbol's `st_size` rather than its address, which
//! is how `sizeof` an object the compiler cannot see the definition of is
//! spelled. The large code model's GOT family -- `GOT64`, `GOTPCREL64`,
//! `GOTPC64` -- is the existing GOT expressions at 64-bit width. And the APX
//! forms differ from `GOTPCRELX` only in how many prefix bytes precede the
//! displacement, which matters to relaxation and not to the value, so they are
//! classified and left unrelaxed.
//!
//! The per-symbol size rows are built only when the scan saw a size
//! relocation. A `u64` per symbol per file is real memory and almost no object
//! has one, so every other link allocates nothing.
//!
//! `PLTOFF64` is deliberately still refused: it is `PLT[sym] - GOT`, which no
//! existing expression spells, and a wrong large-model call target is worse
//! than a refusal.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::ObjectFile,
    icf::IcfMode,
    linker::link_to,
    reloc::{
        RelExpr, Target, WriteKind, spec_target,
        x86_64::{
            R_X86_64_8, R_X86_64_CODE_4_GOTPCRELX, R_X86_64_GOT32,
            R_X86_64_GOT64, R_X86_64_GOTPC64, R_X86_64_GOTPCREL64,
            R_X86_64_PC8, R_X86_64_PC16, R_X86_64_SIZE32, R_X86_64_SIZE64,
        },
    },
};

mod common;

/// The object is 24 bytes, and the program stores that size two ways.
const SRC: &[u8] = b"    .globl tiny\n\
    .set   tiny, 7\n\
    .data\n\
    .globl obj\n\
    .type obj,@object\n\
obj:\n\
    .zero 24\n\
    .size obj, 24\n\
    .globl sz32\n\
sz32:\n\
    .long obj@SIZE\n\
    .globl sz64\n\
sz64:\n\
    .quad obj@SIZE\n\
    .globl b8\n\
b8: .byte tiny\n\
    .globl w16\n\
w16: .word tiny\n\
    .text\n\
    .globl _start\n\
_start:\n\
    ret\n";

/// The object's size reaches the image, at both widths.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_size_relocations_store_the_symbols_size() {
    let Some(dir) = workdir("size") else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    let sz32 = symbol_value(&bytes, b"sz32").expect("sz32 present");
    let sz64 = symbol_value(&bytes, b"sz64").expect("sz64 present");
    let data_addr = section_addr(&bytes, b".data").expect(".data present");
    let data = section_bytes(&bytes, b".data").expect(".data bytes");

    let at32 = usize::try_from(sz32 - data_addr).expect("in range");
    let at64 = usize::try_from(sz64 - data_addr).expect("in range");
    assert_eq!(
        read_u32(&data, at32),
        Some(24),
        "SIZE32 stores the object's st_size, which is 24"
    );
    assert_eq!(
        read_u64(&data, at64),
        Some(24),
        "and SIZE64 stores the same value at eight bytes"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Every added type classifies rather than ending the link.
#[test]
fn the_added_types_are_classified() {
    let expect = [
        (R_X86_64_8, RelExpr::Abs, WriteKind::W8),
        (R_X86_64_PC8, RelExpr::Pc, WriteKind::W8),
        (R_X86_64_PC16, RelExpr::Pc, WriteKind::W16SU),
        (R_X86_64_SIZE32, RelExpr::Size, WriteKind::W32),
        (R_X86_64_SIZE64, RelExpr::Size, WriteKind::W64),
        (R_X86_64_GOT32, RelExpr::GotOffset, WriteKind::W32),
        (R_X86_64_GOT64, RelExpr::GotOffset, WriteKind::W64),
        (R_X86_64_GOTPCREL64, RelExpr::GotPc, WriteKind::W64),
        (R_X86_64_GOTPC64, RelExpr::GotBase, WriteKind::W64),
        (R_X86_64_CODE_4_GOTPCRELX, RelExpr::GotPc, WriteKind::W32S),
    ];
    for (r_type, expr, width) in expect {
        let spec = spec_target(Target::X86_64, r_type)
            .unwrap_or_else(|e| panic!("type {r_type} must classify: {e:?}"));
        assert_eq!(spec.expr, expr, "type {r_type} expression");
        assert_eq!(
            spec.write.width(),
            width.width(),
            "type {r_type} slot width"
        );
    }
}

/// A size relocation says so through `Needs`, which is what tells the layout
/// to build the per-symbol size rows at all.
#[test]
fn a_size_relocation_asks_for_the_size_rows() {
    let plain = xold::reloc::scan_target(Target::X86_64, R_X86_64_8)
        .expect("classifies");
    assert!(!plain.size, "an ordinary absolute store reads no size");
    let sized = xold::reloc::scan_target(Target::X86_64, R_X86_64_SIZE64)
        .expect("classifies");
    assert!(
        sized.size,
        "a size relocation must ask for the rows, or the value reads zero"
    );
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping reloc-x86-gaps {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_x86gaps_{prefix}_{}", std::process::id()));
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
        eprintln!("skipping reloc-x86-gaps: clang cannot assemble it");
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

/// The `st_value` of a symbol in the output's `.symtab`.
fn symbol_value(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab
        .iter()
        .find(|s| symtab.name(s) == name)
        .map(|s| s.st_value.get())
}

/// The virtual address of a named output section.
fn section_addr(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == name)
        .map(|s| s.sh_addr.get())
}

/// The bytes of a named output section.
fn section_bytes(bytes: &[u8], name: &[u8]) -> Option<Vec<u8>> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == name)?;
    obj.section_data(shdr).ok().map(<[u8]>::to_vec)
}
