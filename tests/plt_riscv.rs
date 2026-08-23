//! The RISC-V `PLT[0]` resolver trampoline, pinned instruction by
//! instruction.
//!
//! `PLT[0]` works out which entry trapped into it, and the last step of that
//! computation is a *logical* right shift: `t1` holds the byte distance
//! between two `.plt` stubs and has to become an index into `.got.plt`. The
//! arithmetic shift `srai` lives in the same encoding, one bit away, and
//! disassembles as a perfectly plausible instruction, so only the exact word
//! catches the substitution: the image still links, still loads, and binds the
//! wrong symbol on the first lazy call. lld emits `itype(SRLI, X_T1, X_T1, 1)`
//! in `RISCV::writePltHeader`; this pins the same word.
//!
//! Gated on clang having a `riscv64-linux-gnu` target, because the linker only
//! builds a PLT once some input makes a call it cannot resolve at link time.
//! Without the cross target the test prints a note and returns, matching the
//! other cross-arch tests.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::{ObjectFile, constants::EM_RISCV},
    icf::IcfMode,
    linker::link_shared,
    mmap_file::MappedFile,
};

mod common;

/// `srli t1, t1, 1`: opcode `0x13`, funct3 `0x5`, rd and rs1 both `t1` (x6),
/// shift amount 1, and the shift-kind selector above the amount left clear.
/// Setting bit 30 of the immediate field instead would make this `srai`.
const SRLI_T1_T1_1: u32 = 0x0013_5313;

/// Byte offset of that instruction within `PLT[0]`: the sixth of the eight
/// words the header is made of.
const SHIFT_OFFSET: usize = 20;

/// The size of the RISC-V `PLT[0]` header, in bytes.
const PLT_HEADER: usize = 32;

/// A call to a function this translation unit does not define, so the link has
/// an import to build a PLT entry for.
const SRC: &[u8] = b"int helper(int x);\n\
                     int entry(int x) { return helper(x) + 2; }\n";

/// Locates a clang that can produce rv64 objects, mirroring the probe in
/// `tests/link.rs`.
fn riscv_clang() -> Option<PathBuf> {
    let clang = which("clang")?;
    let probe = std::env::temp_dir().join("xold_plt_riscv_probe.o");
    let ok = Command::new(&clang)
        .args(["--target=riscv64-linux-gnu", "-c", "-x", "c", "-", "-o"])
        .arg(&probe)
        .stdin(std::process::Stdio::null())
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&probe);
    ok.then_some(clang)
}

/// Compiles [`SRC`] into `obj` as position-independent rv64 code. Returns
/// whether the compile succeeded.
fn compile_riscv(clang: &Path, obj: &Path) -> bool {
    let src = obj.with_extension("c");
    if fs::write(&src, SRC).is_err() {
        return false;
    }
    let ok = Command::new(clang)
        .args([
            "--target=riscv64-linux-gnu",
            "-fPIC",
            "-ffreestanding",
            "-nostdlib",
            "-c",
        ])
        .arg(&src)
        .arg("-o")
        .arg(obj)
        .status()
        .is_ok_and(|s| s.success());
    let _ = fs::remove_file(&src);
    ok
}

/// The `.plt` bytes of the image at `path`, with the image checked to be the
/// rv64 shared object the test asked for.
fn plt_bytes(path: &Path) -> Vec<u8> {
    let mapped = MappedFile::open(path).expect("output must be readable");
    let image =
        ObjectFile::parse(mapped.bytes()).expect("output must be valid ELF");
    assert_eq!(image.machine(), EM_RISCV);
    let plt = image
        .sections()
        .iter()
        .find(|s| image.section_name_is(s, b".plt"))
        .expect("a call to an import must produce a .plt");
    image
        .section_data(plt)
        .expect("the .plt contents must be readable")
        .to_vec()
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn riscv_plt_header_shifts_logically() {
    let Some(clang) = riscv_clang() else {
        eprintln!(
            "skipping RISC-V PLT header check: clang with \
             riscv64-linux-gnu target not found"
        );
        return;
    };
    let obj = std::env::temp_dir().join("xold_plt_riscv.o");
    let out = std::env::temp_dir().join("xold_plt_riscv.so");
    if !compile_riscv(&clang, &obj) {
        eprintln!(
            "skipping RISC-V PLT header check: clang could not compile \
             for riscv64-linux-gnu"
        );
        return;
    }

    link_shared(
        std::slice::from_ref(&obj),
        &out,
        None,
        false,
        IcfMode::None,
        false,
    )
    .expect("RISC-V shared link must succeed");
    let plt = plt_bytes(&out);
    let _ = fs::remove_file(&obj);
    let _ = fs::remove_file(&out);

    assert!(
        plt.len() >= PLT_HEADER,
        "the PLT must carry its resolver header, got {} bytes",
        plt.len()
    );
    let word = u32::from_le_bytes(
        plt[SHIFT_OFFSET..SHIFT_OFFSET + 4]
            .try_into()
            .expect("four bytes make a word"),
    );
    assert_eq!(
        word, SRLI_T1_T1_1,
        "PLT[0]+{SHIFT_OFFSET} must be srli t1, t1, 1 ({SRLI_T1_T1_1:#010x}), \
         got {word:#010x}"
    );
}
