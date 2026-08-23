//! An odd RISC-V branch or jump displacement is refused, not truncated.
//!
//! Every RISC-V branch field encodes a byte offset with bit 0 removed: the
//! immediate is stored from bit 1 up, because instructions are 2-byte aligned
//! and the bit would always be zero. `write_jal`, `write_branch`,
//! `write_cbtype` and `write_cjtype` range-checked the value and then scattered
//! its bits, so an odd displacement simply lost its low bit and the branch
//! encoded `target - 1`.
//!
//! Nothing reported it. A `jal ra, sym+1` -- the shape `objcopy` leaves behind,
//! and the shape hand-written assembly reaches for when it means to land on the
//! second half of a compressed pair -- linked cleanly and jumped one byte short
//! of where the input asked. On RISC-V that is not a near miss: it is the
//! middle of an instruction.
//!
//! lld rejects the same values, pairing every one of the four encoders with
//! `checkAlignment(ctx, loc, val, 2, rel)`
//! (`lld/ELF/Arch/RISCV.cpp`).
//!
//! The four writer tests need no toolchain. The link test is gated on a clang
//! that can target `riscv64`; without one it prints a note and returns.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    icf::IcfMode,
    linker::link_to,
    reloc::{
        Resolver, apply,
        riscv::{
            R_RISCV_BRANCH, R_RISCV_JAL, R_RISCV_RVC_BRANCH, R_RISCV_RVC_JUMP,
            Riscv64,
        },
    },
    symbol::SymbolId,
};

mod common;

/// A resolver placing the symbol at `at`, which is all these tests vary.
struct At(u64);

impl Resolver for At {
    fn symbol_addr(&self, _: SymbolId) -> u64 {
        self.0
    }
    fn got_addr(&self, _: SymbolId) -> u64 {
        0
    }
    fn got_base(&self) -> u64 {
        0
    }
    fn plt_addr(&self, _: SymbolId) -> u64 {
        self.0
    }
}

/// `beq a0, a1, .` -- a B-type cell.
const BEQ: [u8; 4] = [0x63, 0x00, 0xB5, 0x00];
/// `jal ra, .` -- a J-type cell.
const JAL: [u8; 4] = [0xEF, 0x00, 0x00, 0x00];
/// `c.j .` -- a CJ-type cell.
const C_J: [u8; 2] = [0x01, 0xA0];
/// `c.beqz a0, .` -- a CB-type cell.
const C_BEQZ: [u8; 2] = [0x01, 0xC1];

/// Applies `r_type` to a copy of `slot` with the symbol at `at` and the site
/// at address 0, so the displacement is exactly `at`.
fn apply_at(r_type: u32, slot: &[u8], at: u64) -> Result<Vec<u8>, xold::Error> {
    let mut out = slot.to_vec();
    apply::<Riscv64, _>(r_type, Some(SymbolId(1)), 0, 0, &At(at), &mut out)?;
    Ok(out)
}

/// Each of the four encoders refuses an odd displacement rather than dropping
/// the bit it cannot store.
#[test]
fn an_odd_displacement_is_refused_by_every_branch_encoder() {
    for (name, r_type, slot) in [
        ("jal", R_RISCV_JAL, &JAL[..]),
        ("branch", R_RISCV_BRANCH, &BEQ[..]),
        ("c.j", R_RISCV_RVC_JUMP, &C_J[..]),
        ("c.beqz", R_RISCV_RVC_BRANCH, &C_BEQZ[..]),
    ] {
        let err = apply_at(r_type, slot, 0x11)
            .expect_err("an odd displacement cannot be encoded");
        assert!(
            matches!(err, xold::Error::RelocMisaligned(t) if t == r_type),
            "{name} must report the misalignment, not {err:?}"
        );
    }
}

/// The refusal is about the low bit and nothing else: the even displacements
/// on either side of a rejected one still encode, and encode differently from
/// each other.
///
/// The check matters because the encoders drop bit 0 rather than round, so
/// before the fix 0x11 wrote exactly what 0x10 writes. Two neighbouring even
/// values landing on distinct cells is what says the immediate is still being
/// stored, and stored from bit 1 up.
#[test]
fn even_displacements_still_encode() {
    for (name, r_type, slot) in [
        ("jal", R_RISCV_JAL, &JAL[..]),
        ("branch", R_RISCV_BRANCH, &BEQ[..]),
        ("c.j", R_RISCV_RVC_JUMP, &C_J[..]),
        ("c.beqz", R_RISCV_RVC_BRANCH, &C_BEQZ[..]),
    ] {
        let low = apply_at(r_type, slot, 0x10).expect("0x10 encodes");
        let high = apply_at(r_type, slot, 0x12).expect("0x12 encodes");
        assert_ne!(low, slot, "{name} must write the displacement");
        assert_ne!(
            low, high,
            "{name} must distinguish 0x10 from 0x12, or the immediate is not \
             reaching the cell"
        );
    }
}

/// End to end: a real `riscv64` object whose `jal` names `far_target + 1` is
/// refused. Before, it linked and the jump landed on `far_target`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_odd_jal_addend_fails_the_link() {
    let Some(dir) = workdir("jal") else {
        return;
    };
    let Some(objs) = build(&dir) else {
        return;
    };
    let out = dir.join("odd");
    let res = link_to(&objs, &out, b"_start", false, IcfMode::None, false);
    let err = res.expect_err(
        "a jal to an odd address cannot be encoded, so the link must fail \
         rather than silently jump one byte short",
    );
    assert!(
        matches!(err, xold::Error::RelocMisaligned(t) if t == R_RISCV_JAL),
        "the refusal must name the relocation that could not be written, \
         got {err:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// The caller: a jump whose target is one byte past a function.
const ODD_SRC: &[u8] = b"    .text\n\
    .globl _start\n\
    .type _start,@function\n\
_start:\n\
    jal     ra, far_target+1\n\
    ret\n\
    .size _start, .-_start\n";

/// The target, in its own section so the assembler cannot resolve the jump
/// itself and has to leave a relocation for the linker.
const TARGET_SRC: &[u8] = b"    .section .text.far,\"ax\",@progbits\n\
    .globl far_target\n\
    .type far_target,@function\n\
far_target:\n\
    ret\n\
    .size far_target, .-far_target\n";

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping riscv-branch-alignment {prefix}: no clang");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_rvbranch_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Assembles both inputs, or `None` when this clang cannot target `riscv64`.
fn build(dir: &Path) -> Option<Vec<PathBuf>> {
    let odd = assemble(dir, ODD_SRC, "odd")?;
    let target = assemble(dir, TARGET_SRC, "far")?;
    Some(vec![odd, target])
}

/// Assembles one `riscv64` source with relaxation off, so the `jal` stays a
/// `jal` and keeps its `R_RISCV_JAL`.
fn assemble(dir: &Path, src: &[u8], stem: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src_path = dir.join(format!("{stem}.S"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clang)
        .args([
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-mno-relax",
            "-c",
        ])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping riscv-branch-alignment: no riscv64 assembler");
        return None;
    }
    Some(obj)
}
