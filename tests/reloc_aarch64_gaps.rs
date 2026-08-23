//! The `AArch64` relocation types that were falling through to "unsupported".
//!
//! Hand-written position-independent assembly reaches a symbol through a
//! `movz`/`movk` pair over `:prel_g1:`/`:prel_g0_nc:`, and a literal-pool load
//! reaches its datum through `ldr xN, label`. Both refused, though the
//! machinery each needs was already here: the signed MOVW writer that the
//! thread-pointer slices use, and the 19-bit field a conditional branch uses.
//!
//! Added with it: `MOVW_SABS_*`, the absolute counterpart of the PC-relative
//! set; `GOT_LD_PREL19`, the same 19-bit field reaching a GOT slot; and
//! `PLT32`, a 32-bit PC-relative reference that routes through the PLT, which
//! is what makes a relative vtable or a jump table over imported functions
//! work.
//!
//! The slice widths are lld's -- 17, 33 and 49 signed bits, one wider than the
//! field, because the sign is carried by the choice of `movz` or `movn` rather
//! than by a bit in the immediate
//! (`lld/ELF/Arch/AArch64.cpp`).
//!
//! Gated on a `clang` that can target `aarch64`; without one the tests print a
//! note and return.

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
        RelExpr, Target,
        aarch64::{
            R_AARCH64_GOT_LD_PREL19, R_AARCH64_LD_PREL_LO19,
            R_AARCH64_MOVW_PREL_G0_NC, R_AARCH64_MOVW_PREL_G1,
            R_AARCH64_MOVW_SABS_G0, R_AARCH64_PLT32,
        },
        spec_target,
    },
};

mod common;

/// A `movz`/`movk` pair over the PC-relative slices, and a literal-pool load.
const SRC: &[u8] = b"    .text\n\
    .globl _start\n\
_start:\n\
    movz x0, #:prel_g1:target\n\
    movk x0, #:prel_g0_nc:target\n\
    ldr  x1, target\n\
    ret\n\
    .data\n\
    .globl target\n\
    .align 3\n\
target:\n\
    .quad 0x1122334455667788\n";

/// Every added type classifies rather than ending the link.
#[test]
fn the_added_types_are_classified() {
    let expect = [
        (R_AARCH64_MOVW_SABS_G0, RelExpr::Escape),
        (R_AARCH64_MOVW_PREL_G1, RelExpr::Escape),
        (R_AARCH64_MOVW_PREL_G0_NC, RelExpr::Escape),
        (R_AARCH64_LD_PREL_LO19, RelExpr::Pc),
        (R_AARCH64_GOT_LD_PREL19, RelExpr::GotPc),
        (R_AARCH64_PLT32, RelExpr::PltPc),
    ];
    for (r_type, expr) in expect {
        let spec = spec_target(Target::AArch64, r_type).unwrap_or_else(|e| {
            panic!("type {r_type:#x} must classify: {e:?}")
        });
        assert_eq!(spec.expr, expr, "type {r_type:#x} expression");
    }
}

/// The pair and the load resolve to the datum's address.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_prel_pair_and_literal_load_reach_the_symbol() {
    let Some(dir) = workdir("prel") else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    let target = symbol_value(&bytes, b"target").expect("target present");
    let start = symbol_value(&bytes, b"_start").expect("_start present");
    let text = section_bytes(&bytes, b".text").expect(".text bytes");
    let text_addr = section_addr(&bytes, b".text").expect(".text addr");
    let at = usize::try_from(start - text_addr).expect("in range");

    // Each relocation measures from its own instruction, so the two slices
    // come from two different differences.
    let g1 = (target - start) >> 16;
    let g0 = (target - (start + 4)) & 0xffff;
    assert_eq!(
        movw_imm(&text, at),
        Some(u32::try_from(g1 & 0xffff).expect("slice fits")),
        "the movz carries the high slice of target - P"
    );
    assert_eq!(
        movw_imm(&text, at + 4),
        Some(u32::try_from(g0).expect("slice fits")),
        "and the movk the low slice, measured from its own place"
    );

    // `ldr xN, label` stores the offset in units of four in bits [5..23].
    let ldr = read_u32(&text, at + 8).expect("the load decodes");
    let off = i64::from((ldr >> 5) & 0x7ffff) * 4;
    assert_eq!(
        u64::try_from(i64::try_from(start + 8).expect("fits") + off)
            .expect("fits"),
        target,
        "the literal load reaches the datum"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping reloc-aarch64-gaps {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_aagaps_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Assembles and links the fixture.
fn link(dir: &Path) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let src = dir.join("a.S");
    let obj = dir.join("a.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args(["--target=aarch64-linux-gnu", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping reloc-aarch64-gaps: no aarch64 assembler");
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

/// The 16-bit immediate of a `movz`/`movk`/`movn` at `at`, as stored.
fn movw_imm(text: &[u8], at: usize) -> Option<u32> {
    read_u32(text, at).map(|cell| (cell >> 5) & 0xffff)
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .and_then(|c| <[u8; 4]>::try_from(c).ok())
        .map(u32::from_le_bytes)
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
