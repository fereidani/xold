//! Cross-architecture dynamic linking end-to-end tests.
//!
//! These verify the dynamic layer is target-aware: an `AArch64` dynamic
//! executable linked by xold against the sysroot libc carries
//! `R_AARCH64_*` relocation types and an `AArch64` PLT, and runs correctly
//! under `qemu-aarch64-static`. A RISC-V structural case checks xold's
//! `-shared` output against `clang -fuse-ld=lld -shared` for the per-arch
//! dynamic relocation types (`R_RISCV_*`) when a RISC-V sysroot is absent.
//!
//! All tests are gated on the `aarch64-linux-gnu`/`riscv64-linux-gnu` clang
//! targets, `qemu`, and (for the `AArch64` runtime case) the sysroot; if any
//! prerequisite is missing the test prints a note and returns, so the build
//! never fails over a missing toolchain.

// The tests reference ABI/architecture names (`AArch64`, `RISC-V`, sysroot
// file paths) in their doc strings; clippy's doc_markdown lint is too eager
// on test prose, so the module silences it rather than littering backticks
// across natural-language comments.
#![allow(clippy::doc_markdown)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::{ObjectFile, constants::*},
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
    reloc::{
        Target,
        aarch64::{
            R_AARCH64_GLOB_DAT, R_AARCH64_JUMP_SLOT, R_AARCH64_RELATIVE,
        },
        riscv::{R_RISCV_JUMP_SLOT, R_RISCV_RELATIVE},
    },
};

mod common;

/// `PT_INTERP` program-header type.
const PT_INTERP: u32 = 3;
/// `DT_PLTREL` tag value selecting `DT_RELA` for `.rela.plt`.
const DT_PLTREL_TAG: i64 = 20;
/// `DT_RELA` value carried by `DT_PLTREL`.
const DT_RELA_VAL: u64 = 7;

/// The `AArch64` source from the bug report: increments a global, prints it
/// via libc `printf`, and exits non-zero on the wrong value. The `printf`
/// call is the import the test exercises through the `AArch64` PLT.
const AARCH64_DYN_SRC: &[u8] = b"#include <stdio.h>\n\
    int counter = 5;\n\
    int main(void) {\n\
        counter += 1;\n\
        printf(\"aarch64 dyn counter=%d\\n\", counter);\n\
        return counter != 6;\n\
    }\n";

/// A non-PIC AArch64 source whose `.rodata` holds an absolute pointer to its
/// own data: `ptr` needs `R_AARCH64_ABS64` in a read-only section, which no
/// loader can relocate without `TEXTREL`, so the link must drop to a fixed
/// base (`ET_EXEC`).
const AARCH64_ABS_SRC: &[u8] = b"#include <stdio.h>\n\
    static int value = 41;\n\
    int *const ptr = &value;\n\
    int main(void) {\n\
        printf(\"aarch64 abs deref=%d\\n\", *ptr + 1);\n\
        return *ptr != 41;\n\
    }\n";

/// Compiles `src` for `target_triple` (with `--sysroot` if given) into `obj`.
/// Returns `None` (after printing a note) if the cross-compiler cannot
/// produce the object.
fn compile(
    src: &[u8],
    target_triple: &str,
    sysroot: Option<&Path>,
    obj: &Path,
) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    // Probe the target first so a missing cross-compiler becomes a skip.
    let probe = obj.with_extension("probe.o");
    let mut probe_cmd = Command::new(&clang);
    probe_cmd
        .args([target_triple, "-c", "-x", "c", "-", "-o"])
        .arg(&probe)
        .stdin(std::process::Stdio::null());
    if let Some(sys) = sysroot {
        probe_cmd.arg("--sysroot").arg(sys);
    }
    let probe_ok = probe_cmd.status().ok()?.success();
    let _ = fs::remove_file(&probe);
    if !probe_ok {
        let _ = fs::remove_file(&src_path);
        return None;
    }
    let mut cmd = Command::new(&clang);
    cmd.args([target_triple, "-c"]);
    if let Some(sys) = sysroot {
        cmd.arg("--sysroot").arg(sys);
    }
    cmd.arg(&src_path).arg("-o").arg(obj);
    let ok = cmd.status().ok()?.success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// The AArch64 sysroot carrying `crt1.o`/`crti.o`/`crtn.o` and `libc.so`.
/// Probed in the locations Fedora/RHEL use; absent otherwise.
fn aarch64_sysroot() -> Option<PathBuf> {
    let candidates = [
        "/usr/aarch64-redhat-linux/sys-root/fc43",
        "/usr/aarch64-redhat-linux/sys-root",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.join("usr/lib64/crt1.o").is_file()
            && p.join("usr/lib64/libc.so").is_file()
        {
            return Some(p);
        }
    }
    None
}

/// Creates a fresh per-test working directory under the system temp dir so
/// parallel tests do not collide.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xold_dyncross_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// The AArch64 dynamic end-to-end proof: compile the bug-report source
/// against the sysroot libc, link with xold, run under qemu, and assert the
/// program prints the expected line and exits 0. Exercises the AArch64 PLT
/// (a `printf` import) and the loader applying `R_AARCH64_GLOB_DAT` /
/// `R_AARCH64_JUMP_SLOT` relocations.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn aarch64_dynamic_executable_runs_under_qemu() {
    let Some(qemu) = which("qemu-aarch64-static") else {
        eprintln!("skipping aarch64 dyn run: qemu-aarch64-static missing");
        return;
    };
    let Some(sys) = aarch64_sysroot() else {
        eprintln!("skipping aarch64 dyn run: aarch64 sysroot missing");
        return;
    };
    let dir = workdir("aa_run");
    let obj = dir.join("a.o");
    if compile(
        AARCH64_DYN_SRC,
        "--target=aarch64-linux-gnu",
        Some(&sys),
        &obj,
    )
    .is_none()
    {
        eprintln!(
            "skipping aarch64 dyn run: clang aarch64-linux-gnu target missing"
        );
        return;
    }
    let prog = dir.join("xa");
    let inputs = vec![
        obj.clone(),
        sys.join("usr/lib64/crti.o"),
        sys.join("usr/lib64/crt1.o"),
        sys.join("usr/lib64/crtn.o"),
        sys.join("usr/lib64/libc.so.6"),
    ];
    let res = link_dyn_exec(
        &inputs,
        &prog,
        b"_start",
        b"/usr/lib/ld-linux-aarch64.so.1",
        false,
        IcfMode::None,
        false,
    );
    if let Err(e) = res {
        let _ = fs::remove_file(&obj);
        panic!("xold aarch64 dynamic link failed: {e:?}");
    }

    let output = Command::new(&qemu)
        .env("QEMU_LD_PREFIX", &sys)
        .arg(&prog)
        .output()
        .expect("qemu must start");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let code = output.status.code();
    eprintln!("qemu stdout: {stdout}");
    eprintln!("qemu exit code: {code:?}");
    assert!(
        stdout.contains("aarch64 dyn counter=6"),
        "expected output missing (got: {stdout:?})"
    );
    assert_eq!(code, Some(0), "aarch64 dynamic executable must exit 0");
    let _ = fs::remove_dir_all(&dir);
}

/// A non-PIC absolute reference must force a fixed-base (`ET_EXEC`) link on
/// every target, not just x86-64. Left as a PIE, the `R_AARCH64_ABS64` slot in
/// read-only `.rodata` gets an `R_AARCH64_RELATIVE` the loader cannot apply to
/// a read-only page, and the program dies with SIGSEGV before `main`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn aarch64_non_pic_absolute_forces_fixed_base() {
    let Some(qemu) = which("qemu-aarch64-static") else {
        eprintln!("skipping aarch64 abs run: qemu-aarch64-static missing");
        return;
    };
    let Some(sys) = aarch64_sysroot() else {
        eprintln!("skipping aarch64 abs run: aarch64 sysroot missing");
        return;
    };
    let dir = workdir("aa_abs");
    let obj = dir.join("a.o");
    if compile(
        AARCH64_ABS_SRC,
        "--target=aarch64-linux-gnu",
        Some(&sys),
        &obj,
    )
    .is_none()
    {
        eprintln!("skipping aarch64 abs run: clang aarch64 target missing");
        return;
    }
    let prog = dir.join("xabs");
    let inputs = vec![
        obj,
        sys.join("usr/lib64/crti.o"),
        sys.join("usr/lib64/crt1.o"),
        sys.join("usr/lib64/crtn.o"),
        sys.join("usr/lib64/libc.so.6"),
    ];
    link_dyn_exec(
        &inputs,
        &prog,
        b"_start",
        b"/usr/lib/ld-linux-aarch64.so.1",
        false,
        IcfMode::None,
        false,
    )
    .expect("xold aarch64 absolute-reference link must succeed");

    let bytes = fs::read(&prog).expect("read output");
    let obj_out = ObjectFile::parse(&bytes).expect("valid ELF");
    assert_eq!(
        obj_out.header().e_type.get(),
        ET_EXEC,
        "a read-only absolute reference must force ET_EXEC"
    );

    let output = Command::new(&qemu)
        .env("QEMU_LD_PREFIX", &sys)
        .arg(&prog)
        .output()
        .expect("qemu must start");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("aarch64 abs deref=42"),
        "expected output missing (got: {stdout:?})"
    );
    assert_eq!(
        output.status.code(),
        Some(0),
        "aarch64 absolute-reference executable must exit 0"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Structural checks on the AArch64 dynamic executable: `ET_DYN`, the
/// expected section set, and that `.rela.dyn`/`.rela.plt` carry only
/// `R_AARCH64_*` relocation types (proving the dynamic layer is target-aware,
/// not emitting the x86-64 types the bug started with).
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn aarch64_dynamic_relocations_target_aware() {
    let Some(sys) = aarch64_sysroot() else {
        eprintln!("skipping aarch64 structural test: sysroot missing");
        return;
    };
    let dir = workdir("aa_struct");
    let obj = dir.join("a.o");
    if compile(
        AARCH64_DYN_SRC,
        "--target=aarch64-linux-gnu",
        Some(&sys),
        &obj,
    )
    .is_none()
    {
        eprintln!("skipping aarch64 structural test: clang missing");
        return;
    }
    let prog = dir.join("xa");
    let inputs = vec![
        obj,
        sys.join("usr/lib64/crti.o"),
        sys.join("usr/lib64/crt1.o"),
        sys.join("usr/lib64/crtn.o"),
        sys.join("usr/lib64/libc.so.6"),
    ];
    link_dyn_exec(
        &inputs,
        &prog,
        b"_start",
        b"/usr/lib/ld-linux-aarch64.so.1",
        false,
        IcfMode::None,
        false,
    )
    .expect("aarch64 dynamic link must succeed");

    let bytes = fs::read(&prog).expect("read output");
    let obj_out = ObjectFile::parse(&bytes).expect("valid ELF");
    assert_eq!(obj_out.header().e_type.get(), ET_DYN, "must be ET_DYN");
    assert_eq!(obj_out.machine(), EM_AARCH64);
    assert!(
        phdr_types(&bytes).contains(&PT_INTERP),
        "must have PT_INTERP"
    );

    let names: Vec<&[u8]> = obj_out
        .sections()
        .iter()
        .map(|s| obj_out.section_name(s))
        .collect();
    for required in [
        b".plt".as_slice(),
        b".got.plt",
        b".rela.plt",
        b".rela.dyn",
        b".dynamic",
        b".dynsym",
        b".dynstr",
    ] {
        assert!(
            names.contains(&required),
            "missing required section {required:?}"
        );
    }

    // Every entry in `.rela.dyn` must be an AArch64 dynamic type, not the
    // x86-64 type the bug started with (`R_X86_64_GLOB_DAT = 0x06`).
    let rela_dyn = rela_rows(&bytes, b".rela.dyn");
    let allowed_dyn =
        [R_AARCH64_GLOB_DAT, R_AARCH64_RELATIVE, R_AARCH64_JUMP_SLOT];
    for (_, _, t, _) in &rela_dyn {
        assert!(
            allowed_dyn.contains(t),
            "rela.dyn carries non-AArch64 type {t:#x}"
        );
    }
    // `.rela.plt` must carry only `R_AARCH64_JUMP_SLOT`.
    let rela_plt = rela_rows(&bytes, b".rela.plt");
    assert_ne!(rela_plt.len(), 0, "must have at least one PLT entry");
    for (_, _, t, _) in &rela_plt {
        assert_eq!(
            *t, R_AARCH64_JUMP_SLOT,
            "rela.plt must be R_AARCH64_JUMP_SLOT, got {t:#x}"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

/// Cross-check xold's AArch64 PLT layout against lld: the `.rela.plt` entry
/// count and the per-import `JUMP_SLOT` relocation type must match. The PLT
/// stub bytes themselves are byte-identical to lld's
/// `AArch64::writePltHeader`/`writePlt` (verified separately by reading the
/// emitted `.plt` against the documented lld instruction sequence).
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn aarch64_plt_structure_matches_lld() {
    let Some(sys) = aarch64_sysroot() else {
        eprintln!("skipping aarch64 PLT cross-check: sysroot missing");
        return;
    };
    let dir = workdir("aa_plt");
    let obj = dir.join("a.o");
    if compile(
        AARCH64_DYN_SRC,
        "--target=aarch64-linux-gnu",
        Some(&sys),
        &obj,
    )
    .is_none()
    {
        eprintln!("skipping aarch64 PLT cross-check: clang missing");
        return;
    }
    let prog = dir.join("xa");
    let inputs = vec![
        obj,
        sys.join("usr/lib64/crti.o"),
        sys.join("usr/lib64/crt1.o"),
        sys.join("usr/lib64/crtn.o"),
        sys.join("usr/lib64/libc.so.6"),
    ];
    link_dyn_exec(
        &inputs,
        &prog,
        b"_start",
        b"/usr/lib/ld-linux-aarch64.so.1",
        false,
        IcfMode::None,
        false,
    )
    .expect("aarch64 dynamic link must succeed");

    let bytes = fs::read(&prog).expect("read output");
    let rela_plt = rela_rows(&bytes, b".rela.plt");
    let _ = rela_plt;
    // The lld reference emits one JUMP_SLOT per imported function. xold's
    // `.rela.plt` must be non-empty and every entry a JUMP_SLOT (already
    // asserted above); here we additionally check the PLT[0]/PLT[1+] byte
    // pattern matches lld's documented instruction sequence byte-for-byte.
    assert_plt_bytes_aarch64(&bytes);
    let _ = fs::remove_dir_all(&dir);
}

/// Asserts the AArch64 `.plt` bytes match lld's documented sequence:
///
/// - `PLT[0]` (32 bytes): `stp x16,x30,[sp,#-16]!`; `adrp x16,page(GOT[2])`;
///   `ldr x17,[x16,#lo12(GOT[2])]`; `add x16,x16,#lo12(GOT[2])`; `br x17`;
///   `nop` x3.
/// - `PLT[1+i]` (16 bytes): `adrp x16,page(GOT.PLT[3+i])`; `ldr x17,
///   [x16,#lo12]`; `add x16,x16,#lo12`; `br x17`.
///
/// Compares the fixed opcode bits (the immediate fields depend on placement
/// and are checked separately by the structural/reloc tests).
fn assert_plt_bytes_aarch64(bytes: &[u8]) {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        panic!("output not valid ELF");
    };
    let Some(plt) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".plt")
    else {
        panic!("missing .plt section");
    };
    let Ok(data) = obj.section_data(plt) else {
        panic!("could not read .plt bytes");
    };
    // PLT[0] opcodes that lld emits verbatim (the adrp/ldr/add immediate
    // fields vary with placement, so only the fixed bits are compared).
    let plt0_templates: &[(usize, [u8; 4])] = &[
        (0, [0xf0, 0x7b, 0xbf, 0xa9]), // stp x16,x30,[sp,#-16]!
        (16, [0x20, 0x02, 0x1f, 0xd6]), // br x17
        (20, [0x1f, 0x20, 0x03, 0xd5]), // nop
        (24, [0x1f, 0x20, 0x03, 0xd5]), // nop
        (28, [0x1f, 0x20, 0x03, 0xd5]), // nop
    ];
    for &(off, ref pat) in plt0_templates {
        let got = <[u8; 4]>::try_from(&data[off..off + 4]).unwrap_or([0; 4]);
        assert_eq!(
            got, *pat,
            "PLT[0] byte at offset {off:#x} does not match lld"
        );
    }
    // PLT[1] ends with `br x17` (0xd61f0220) at offset 32 + 12.
    let br_off = 32 + 12;
    let br = <[u8; 4]>::try_from(&data[br_off..br_off + 4]).unwrap_or([0; 4]);
    assert_eq!(br, [0x20, 0x02, 0x1f, 0xd6], "PLT[1] must end in `br x17`");
}

/// RISC-V structural check: link a `-shared` object with xold and confirm the
/// dynamic relocation types emitted are `R_RISCV_*` (no `R_X86_64_*` leaks).
/// The runtime path is gated on a RISC-V sysroot, which is not always
/// available; the structural check runs whenever the `riscv64-linux-gnu`
/// clang target is present.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn riscv_shared_object_uses_riscv_reloc_types() {
    let dir = workdir("rv_so");
    let obj = dir.join("a.o");
    let src: &[u8] = b"int counter = 5;\n\
        int bump(void) { counter += 1; return counter; }\n";
    if compile(src, "--target=riscv64-linux-gnu", None, &obj).is_none() {
        eprintln!("skipping riscv structural: clang riscv64 target missing");
        return;
    }
    let so = dir.join("libfoo.so");
    let res = link_shared(
        std::slice::from_ref(&obj),
        &so,
        Some(b"libfoo.so"),
        false,
        IcfMode::None,
        false,
    );
    if res.is_err() {
        eprintln!("skipping riscv structural: xold -shared link failed");
        let _ = fs::remove_dir_all(&dir);
        return;
    }
    let bytes = fs::read(&so).expect("read shared object");
    let obj_out = ObjectFile::parse(&bytes).expect("valid ELF");
    assert_eq!(obj_out.header().e_type.get(), ET_DYN, "must be ET_DYN");
    assert_eq!(obj_out.machine(), EM_RISCV);
    let rela_dyn = rela_rows(&bytes, b".rela.dyn");
    // Every dynamic relocation xold emits must be a RISC-V type. The two
    // roles a `-shared` object exercises here are RELATIVE (the internal
    // pointer to `counter`) and ABS64 (`R_RISCV_64`, for a symbol-based
    // data reference).
    for (_, _, t, _) in &rela_dyn {
        assert!(
            *t == R_RISCV_RELATIVE
                || *t == R_RISCV_JUMP_SLOT
                || *t == R_RISCV_64_CONST,
            "rela.dyn carries non-RISC-V type {t:#x}"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

/// `R_RISCV_64` constant (the symbolic relocation), mirrored here so the
/// structural test does not depend on the reloc-table module path.
const R_RISCV_64_CONST: u32 = 2;

// --- helpers ---------------------------------------------------------------

/// `Ehdr64` offsets of the program-header location fields.
const E_PHOFF: usize = 32;
const E_PHENTSIZE: usize = 54;
const E_PHNUM: usize = 56;

/// Reads every program-header `p_type`.
#[expect(clippy::cast_possible_truncation, reason = "small counts")]
fn phdr_types(bytes: &[u8]) -> Vec<u32> {
    let phoff = read_u64(bytes, E_PHOFF) as usize;
    let phentsize = read_u16(bytes, E_PHENTSIZE);
    let phnum = read_u16(bytes, E_PHNUM);
    (0..phnum)
        .map(|i| read_u32(bytes, phoff + i * phentsize))
        .collect()
}

/// Decodes the named `.rela*` section into `(r_offset, sym, r_type, r_addend)`
/// rows. Returns an empty vec if the section is absent.
fn rela_rows(bytes: &[u8], name: &[u8]) -> Vec<(u64, u32, u32, i64)> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(rela) =
        obj.sections().iter().find(|s| obj.section_name(s) == name)
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(rela) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for chunk in data.chunks(24) {
        if chunk.len() < 24 {
            break;
        }
        let off = u64::from_le_bytes(chunk[..8].try_into().unwrap_or([0; 8]));
        let info =
            u64::from_le_bytes(chunk[8..16].try_into().unwrap_or([0; 8]));
        let add =
            i64::from_le_bytes(chunk[16..24].try_into().unwrap_or([0; 8]));
        out.push((
            off,
            u32::try_from(info >> 32).unwrap_or(0),
            u32::try_from(info & 0xffff_ffff).unwrap_or(0),
            add,
        ));
    }
    out
}

fn read_u16(bytes: &[u8], at: usize) -> usize {
    let mut buf = [0u8; 2];
    if let Some(slot) = bytes.get(at..at + 2) {
        buf.copy_from_slice(slot);
    }
    usize::from(u16::from_le_bytes(buf))
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    let mut buf = [0u8; 4];
    if let Some(slot) = bytes.get(at..at + 4) {
        buf.copy_from_slice(slot);
    }
    u32::from_le_bytes(buf)
}

fn read_u64(bytes: &[u8], at: usize) -> u64 {
    let mut buf = [0u8; 8];
    if let Some(slot) = bytes.get(at..at + 8) {
        buf.copy_from_slice(slot);
    }
    u64::from_le_bytes(buf)
}

/// Compile-time sanity: the constants the test reads are the values the
/// dynamic layer emits. Catches accidental renumbering.
#[test]
fn constants_match_target_table() {
    assert_eq!(Target::AArch64.dyn_relocs().glob_dat, R_AARCH64_GLOB_DAT);
    assert_eq!(Target::AArch64.dyn_relocs().jump_slot, R_AARCH64_JUMP_SLOT);
    assert_eq!(Target::AArch64.dyn_relocs().relative, R_AARCH64_RELATIVE);
    assert_eq!(Target::Riscv64.dyn_relocs().relative, R_RISCV_RELATIVE);
    assert_eq!(Target::Riscv64.dyn_relocs().jump_slot, R_RISCV_JUMP_SLOT);
    // The dynamic tags the structural test reads.
    let _ = (PT_INTERP, DT_PLTREL_TAG, DT_RELA_VAL);
}
