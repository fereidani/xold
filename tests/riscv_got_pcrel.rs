//! RISC-V GOT-indirect addressing: `R_RISCV_GOT_HI20` and its word-sized kin.
//!
//! Distro riscv64 compilers default to PIE, and PIE code reaches extern data
//! through the GOT: `auipc` carrying `R_RISCV_GOT_HI20` (spelled
//! `%got_pcrel_hi` in assembly) followed by a load carrying
//! `R_RISCV_PCREL_LO12_I`. That pair is in essentially every object such a
//! toolchain produces, and xold refused all of them -- the type was not in the
//! table at all, so the link ended with `UnsupportedReloc(20)`.
//!
//! Only the expression is new. `GOT_HI20` computes `GOT[sym] + A - P` where
//! `PCREL_HI20` computes `S + A - P`; the U-type encoding, the `+0x800`
//! sign-compensation and the 20-bit range check are the same writer, which is
//! why lld handles both in one `case` (`Arch/RISCV.cpp:448`). The pairing pass
//! also has to know about it: a `PCREL_LO12_I` names the instruction rather
//! than the target, so it can be paired with any flavour of `auipc`, and lld
//! keeps the same set in `RISCVPCRel::isHiReloc`.
//!
//! `R_RISCV_PLT32` and `R_RISCV_GOT32_PCREL` are the word-sized forms of the
//! same two ideas, range-checked as signed 32 exactly as `R_RISCV_32_PCREL`
//! is (`Arch/RISCV.cpp:545`).
//!
//! There is no riscv64 sysroot on this host, so the runtime case is
//! freestanding: a hand-written `_start` calls `main` and exits with its
//! return value through `ecall`. That is enough to prove the address the GOT
//! slot holds is the one the data actually lives at, which is what the
//! encoding exists for.
//!
//! Gated on `clang` with a riscv64 target and `qemu-riscv64-static`; if either
//! is missing the tests print a note and return.

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
        Arch, Needs, RelExpr, Write, WriteKind,
        riscv::{
            R_RISCV_GOT_HI20, R_RISCV_GOT32_PCREL, R_RISCV_PLT32, Riscv64,
        },
    },
};

mod common;

/// The value the program exits with: the datum plus one.
const EXPECTED_EXIT: i32 = 42;

/// The translation unit under test. `-fPIE` turns the `datum` reference into
/// a GOT-indirect access.
const MAIN_SRC: &[u8] =
    b"extern int datum;\nint main(void) { return datum + 1; }\n";

/// The datum, in a second unit so the reference really is extern.
const DATA_SRC: &[u8] = b"int datum = 41;\n";

/// A freestanding entry point: call `main`, then exit with its return value
/// (RISC-V syscall 93).
const START_SRC: &[u8] = b"    .text\n    .globl _start\n_start:\n\
    call main\n    li a7, 93\n    ecall\n";

/// The table classifies the GOT-indirect `auipc` the way lld does.
#[test]
fn got_hi20_is_a_got_relative_reference_that_allocates_a_slot() {
    let spec = Riscv64::spec(R_RISCV_GOT_HI20)
        .expect("R_RISCV_GOT_HI20 must be in the table");
    assert_eq!(
        spec.expr,
        RelExpr::Escape,
        "the U-type hi20 encoding needs the +0x800 compensation, which no \
         plain bitfield expresses, so it goes through the escape"
    );
    let needs = Riscv64::scan_needs(R_RISCV_GOT_HI20)
        .expect("R_RISCV_GOT_HI20 must classify");
    assert_eq!(
        needs,
        Needs {
            got: true,
            ..Needs::NONE
        },
        "an escape still has to say what it references, or the scan allocates \
         no slot and the auipc points at nothing"
    );
}

/// The two word-sized forms take the expressions and the signed-32 check lld
/// gives them.
#[test]
fn the_word_sized_got_and_plt_forms_are_signed_32_bit() {
    for (r_type, expr, name) in [
        (R_RISCV_PLT32, RelExpr::PltPc, "R_RISCV_PLT32"),
        (R_RISCV_GOT32_PCREL, RelExpr::GotPc, "R_RISCV_GOT32_PCREL"),
    ] {
        let spec =
            Riscv64::spec(r_type).unwrap_or_else(|_| panic!("{name} in table"));
        assert_eq!(spec.expr, expr, "{name} computes the wrong value");
        assert_eq!(
            spec.write,
            Write::Bytes(WriteKind::W32S),
            "{name} stores a signed 32-bit word, as R_RISCV_32_PCREL does"
        );
    }
}

/// The end of it: a PIE object's GOT-indirect load reads the right memory.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_pie_object_links_and_reads_its_extern_datum() {
    let Some(qemu) = which("qemu-riscv64-static") else {
        eprintln!("skipping riscv got run: qemu-riscv64-static missing");
        return;
    };
    let Some(dir) = workdir("run") else {
        return;
    };
    let Some(prog) = link(&dir) else {
        return;
    };
    let status = Command::new(&qemu)
        .arg(&prog)
        .status()
        .expect("qemu must start");
    assert_eq!(
        status.code(),
        Some(EXPECTED_EXIT),
        "the GOT slot must hold the address the datum lives at"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The fixture really carries the relocation under test, so the run above is
/// not passing for some other reason.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_fixture_carries_a_got_hi20() {
    let Some(dir) = workdir("shape") else {
        return;
    };
    let obj = dir.join("main.o");
    if compile(MAIN_SRC, "c", &obj).is_none() {
        eprintln!("skipping riscv got shape: no riscv64 clang target");
        return;
    }
    let bytes = fs::read(&obj).expect("read object");
    assert!(
        has_reloc(&bytes, R_RISCV_GOT_HI20),
        "a -fPIE riscv64 object referencing extern data must carry \
         R_RISCV_GOT_HI20, or this test is about nothing"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping riscv got {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_riscvgot_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the three inputs and links them into a freestanding executable.
fn link(dir: &Path) -> Option<PathBuf> {
    let main_o = dir.join("main.o");
    let data_o = dir.join("data.o");
    let start_o = dir.join("start.o");
    if compile(MAIN_SRC, "c", &main_o).is_none() {
        eprintln!("skipping riscv got run: no riscv64 clang target");
        return None;
    }
    compile(DATA_SRC, "c", &data_o)?;
    compile(START_SRC, "S", &start_o)?;

    let prog = dir.join("rvprog");
    let res = link_to(
        &[main_o, data_o, start_o],
        &prog,
        b"_start",
        false,
        IcfMode::None,
        false,
    );
    assert!(
        res.is_ok(),
        "a -fPIE riscv64 object must link: {:?}",
        res.err()
    );
    Some(prog)
}

/// Compiles or assembles `src` (written beside `obj` with extension `ext`)
/// for riscv64.
fn compile(src: &[u8], ext: &str, obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension(ext);
    fs::write(&src_path, src).ok()?;
    Command::new(clang)
        .args(["--target=riscv64-linux-gnu", "-fPIE", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success()
        .then_some(())
}

// --- readers ---------------------------------------------------------------

/// Whether any relocation section of `bytes` carries `r_type`.
fn has_reloc(bytes: &[u8], r_type: u32) -> bool {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return false;
    };
    (0..obj.sections().len()).any(|i| {
        u16::try_from(i).is_ok_and(|shndx| {
            matches!(obj.relocations(shndx), Ok(Some(rs))
                if rs.iter().any(|r| r.r_type() == r_type))
        })
    })
}
