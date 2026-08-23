//! Static TLS (local-exec) end-to-end tests.
//!
//! These compile a real C source carrying `__thread`/`_Thread_local` globals
//! and link it with xold, then check the output structurally (a `PT_TLS`
//! segment with the right sizes/alignment, `.tdata`/`.tbss` sections, and no
//! surviving relocation sections) for `x86-64`, `AArch64` and `RISC-V`. The
//! x86-64 case additionally cross-checks the resolved `R_X86_64_TPOFF32` bytes
//! against the system linker (`ld.lld`) and runs a freestanding program that
//! stands up a minimal TLS image by hand and reads a `__thread` variable.
//!
//! Cross-architecture cases are gated on clang knowing the target triple; if
//! the cross-compiler is absent the test prints a note and returns, so the
//! build never fails over a missing toolchain.

use std::{fs, path::Path, process::Command};

use common::which;
use xold::{
    elf::{ObjectFile, constants::*},
    icf::IcfMode,
    linker::link_to,
    reloc::x86_64::R_X86_64_TPOFF32,
};

mod common;

// --- program-header field offsets (ELF64) ---------------------------------

/// `Ehdr64` offset of `e_phoff`.
const E_PHOFF: usize = 32;
/// `Ehdr64` offset of `e_phentsize`.
const E_PHENTSIZE: usize = 54;
/// `Ehdr64` offset of `e_phnum`.
const E_PHNUM: usize = 56;

/// A decoded program header, just the fields the tests inspect. Field names
/// mirror `Phdr64`.
#[derive(Clone, Copy, Debug)]
#[expect(clippy::struct_field_names, reason = "mirror ELF Phdr64 names")]
struct Phdr {
    p_type: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
}

/// Reads every program header from an ELF64 image.
#[expect(
    clippy::cast_possible_truncation,
    reason = "file offsets and counts are small"
)]
fn phdrs(bytes: &[u8]) -> Vec<Phdr> {
    let phoff = read_u64(bytes, E_PHOFF) as usize;
    let phentsize = read_u16(bytes, E_PHENTSIZE);
    let phnum = read_u16(bytes, E_PHNUM);
    let mut out = Vec::with_capacity(phnum);
    for i in 0..phnum {
        let base = phoff + i * phentsize;
        out.push(Phdr {
            p_type: read_u32(bytes, base),
            p_offset: read_u64(bytes, base + 8),
            p_vaddr: read_u64(bytes, base + 16),
            p_filesz: read_u64(bytes, base + 32),
            p_memsz: read_u64(bytes, base + 40),
            p_align: read_u64(bytes, base + 48),
        });
    }
    out
}

/// The first program header of `p_type`, if any.
fn find_phdr(bytes: &[u8], p_type: u32) -> Option<Phdr> {
    phdrs(bytes).into_iter().find(|p| p.p_type == p_type)
}

/// Little-endian readers over a byte slice; out-of-range becomes 0.
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

/// Reads the 4-byte little-endian value at `at` as a signed 32-bit integer.
fn read_i32(bytes: &[u8], at: usize) -> i32 {
    i32::from_le_bytes(read_u32(bytes, at).to_le_bytes())
}

/// Asserts that no relocation section survives into the linked output: every
/// relocation was applied.
fn assert_no_reloc_sections(obj: &ObjectFile<'_>, msg: &str) {
    assert!(
        obj.sections()
            .iter()
            .all(|s| s.sh_type.get() != SHT_RELA && s.sh_type.get() != SHT_REL),
        "{msg}: output must contain no relocation sections"
    );
}

/// The simple TLS source every arch links: one initialised thread-local.
const TLS_SRC: &[u8] = b"__thread int tvar = 9;\n\
                        int g(void) { return tvar; }\n";

/// Compiles `src` for `target_triple` into `obj`. Returns `None` (after
/// printing a note) if the cross-compiler cannot produce the object. The
/// source is written next to `obj` under a unique name so the parallel test
/// runner does not collide on a shared source filename.
fn compile(
    src: &[u8],
    target_triple: &str,
    extra: &[&str],
    obj: &Path,
) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write TLS source");
    // Probe the target first so a missing cross-compiler becomes a skip
    // rather than a half-written object.
    let probe = std::env::temp_dir().join(format!(
        "xold_tls_probe_{}.o",
        obj.file_stem().and_then(|s| s.to_str()).unwrap_or("x")
    ));
    let probe_ok = Command::new(&clang)
        .args([target_triple, "-c", "-x", "c", "-", "-o"])
        .arg(&probe)
        .stdin(std::process::Stdio::null())
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&probe);
    if !probe_ok {
        let _ = fs::remove_file(&src_path);
        return None;
    }
    let ok = Command::new(&clang)
        .args([target_triple, "-fno-pic", "-ffreestanding", "-c"])
        .args(extra)
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// Links `obj` with xold (entry `g`), reads the output back, and returns the
/// mapped bytes. `tag` isolates the temp output path per test so the parallel
/// test runner does not collide on a shared filename.
fn link_with_xold(obj: &Path, tag: &str) -> Vec<u8> {
    let out = std::env::temp_dir().join(format!("xold_tls_{tag}.out"));
    let inputs = [obj.to_path_buf()];
    link_to(&inputs, &out, b"g", false, IcfMode::None, false)
        .expect("xold TLS link must succeed");
    let bytes = fs::read(&out).expect("output readable");
    let _ = fs::remove_file(&out);
    bytes
}

// --- x86-64: structural, byte cross-check, and runtime --------------------

/// x86-64 native: a `__thread` program links into a valid static executable
/// with a correctly sized `PT_TLS`, a `.tdata` section, and no surviving
/// relocation sections.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_x86_64_tls_with_a_tls_segment() {
    let obj = std::env::temp_dir().join("xold_tls_x86_64.o");
    // x86-64 is the host; this never skips.
    compile(TLS_SRC, "--target=x86_64-linux-gnu", &[], &obj)
        .expect("host clang must compile x86-64 TLS");

    let bytes = link_with_xold(&obj, "x86_struct");
    let _ = fs::remove_file(&obj);
    let obj_out = ObjectFile::parse(&bytes).expect("xold output is valid ELF");
    assert_eq!(obj_out.header().e_type.get(), ET_EXEC);
    assert_eq!(obj_out.machine(), EM_X86_64);

    let tls = find_phdr(&bytes, PT_TLS).expect("output has a PT_TLS");
    // One `int` initialised thread-local: 4 bytes file and memory, 4-byte
    // alignment. The TLS block shares the identity map with the RW segment,
    // so `p_vaddr - p_offset` is the load base.
    assert_eq!(tls.p_filesz, 4, "PT_TLS filesz is .tdata size");
    assert_eq!(tls.p_memsz, 4, "PT_TLS memsz is .tdata + .tbss");
    assert_eq!(tls.p_align, 4, "PT_TLS align is the tdata alignment");
    // Identity-map invariant: the TLS block sits inside the RW `PT_LOAD`, so
    // its virtual address minus its file offset equals the load base, and the
    // block address itself is nonzero.
    assert!(
        tls.p_vaddr > tls.p_offset && tls.p_offset > 0,
        "PT_TLS placed within the loaded image"
    );
    assert_ne!(obj_out.header().e_entry.get(), 0, "entry `g` is resolved");

    assert!(
        obj_out
            .sections()
            .iter()
            .map(|s| obj_out.section_name(s))
            .any(|n| n == b".tdata"),
        "has .tdata"
    );

    assert_no_reloc_sections(&obj_out, "x86-64 TLS");
}

/// x86-64: the resolved `R_X86_64_TPOFF32` bytes the linker writes must match
/// the system linker byte-for-byte. Both `xold` and `ld.lld` link the same
/// object; the 4 bytes at the relocation site are read out of each output and
/// compared. This is the authoritative correctness oracle for the TPOFF sign
/// and magnitude.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn x86_64_tpoff_bytes_match_the_system_linker() {
    let lld = which("ld.lld").or_else(|| which("ld"));
    let obj = std::env::temp_dir().join("xold_tls_xoff.o");
    compile(TLS_SRC, "--target=x86_64-linux-gnu", &[], &obj)
        .expect("host clang must compile x86-64 TLS");

    // Recover the TPOFF32 relocation offset within `.text` from the object.
    let obj_bytes = fs::read(&obj).expect("read obj");
    let in_obj = ObjectFile::parse(&obj_bytes).expect("obj is valid ELF");
    let text_idx = in_obj
        .sections()
        .iter()
        .position(|s| in_obj.section_name(s) == b".text")
        .expect("obj has .text");
    let text_shndx =
        u16::try_from(text_idx).expect("section index fits in u16");
    let entries = in_obj
        .relocations(text_shndx)
        .expect("relocations readable")
        .expect("obj .text has a relocation section");
    let tpoff_off = entries
        .iter()
        .find(|r| r.r_type() == R_X86_64_TPOFF32)
        .map(|r| r.r_offset.get())
        .expect("object has an R_X86_64_TPOFF32");

    // xold output: read the 4 TPOFF bytes at `.text`'s file offset + the
    // relocation offset (the first `.text` member sits at the section start).
    let xold_bytes = link_with_xold(&obj, "x86_xoff");
    let xold_out = ObjectFile::parse(&xold_bytes).expect("xold ELF valid");
    let xold_text = xold_out
        .sections()
        .iter()
        .find(|s| xold_out.section_name(s) == b".text")
        .expect("xold output has .text");
    let xold_tpoff = read_i32(
        &xold_bytes,
        usize::try_from(xold_text.sh_offset.get() + tpoff_off)
            .expect("offset fits"),
    );

    // System linker output: link the same object with ld.lld (falling back to
    // the gold/bfd `ld`), then read the same site.
    let Some(linker) = lld.as_ref() else {
        let _ = fs::remove_file(&obj);
        eprintln!("skipping TPOFF cross-check: no system linker on PATH");
        return;
    };
    let sys_out = std::env::temp_dir().join("xold_tls_sys.out");
    let ok = Command::new(linker)
        .args(["-static", "-e", "g"])
        .arg(&obj)
        .arg("-o")
        .arg(&sys_out)
        .status()
        .is_ok_and(|s| s.success());
    if !ok {
        let _ = fs::remove_file(&obj);
        eprintln!("skipping TPOFF cross-check: system linker unavailable");
        return;
    }
    let sys_bytes = fs::read(&sys_out).expect("read system output");
    let _ = fs::remove_file(&sys_out);
    let sys_obj = ObjectFile::parse(&sys_bytes).expect("system ELF valid");
    let sys_text = sys_obj
        .sections()
        .iter()
        .find(|s| sys_obj.section_name(s) == b".text")
        .expect("system output has .text");
    let sys_tpoff = read_i32(
        &sys_bytes,
        usize::try_from(sys_text.sh_offset.get() + tpoff_off)
            .expect("offset fits"),
    );

    let _ = fs::remove_file(&obj);

    // For this source xold and lld both assign tvar the thread-pointer offset
    // -4 (one 4-byte TLS block below the TP, Variant II).
    assert_eq!(
        xold_tpoff, sys_tpoff,
        "xold TPOFF {xold_tpoff} must match system linker {sys_tpoff}"
    );
    assert_eq!(xold_tpoff, -4, "expected TPOFF for a single 4-byte TLS var");
    eprintln!(
        "TPOFF32 cross-check OK: xold={xold_tpoff} system={sys_tpoff} \
         (bytes {:?})",
        &xold_bytes[usize::try_from(xold_text.sh_offset.get() + tpoff_off)
            .expect("offset fits")..][..4]
    );
}

/// x86-64 runtime: a freestanding program stands up a minimal TLS image by
/// hand (TCB self-pointer at `%fs:0`, the `.tdata` template below the thread
/// pointer), reads a `__thread int tvar = 9`, and exits with the value. A
/// correct local-exec link makes `tvar` resolve to the seeded value.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn x86_64_tls_program_runs_and_reads_the_thread_local() {
    let obj = std::env::temp_dir().join("xold_tls_run.o");
    // The setup seeds exactly one 4-byte TLS block (tvar), so xold assigns
    // tpoff = -4; the C side computes the same layout from the TLS convention.
    let src = b"__thread int tvar = 9;\n\
                static volatile unsigned long tcb[4];\n\
                static long set_fs(unsigned long a) {\n\
                    long r;\n\
                    __asm__ volatile(\"syscall\"\n\
                        : \"=a\"(r)\n\
                        : \"0\"(158L), \"D\"(0x1002L), \"S\"(a)\n\
                        : \"rcx\", \"r11\", \"memory\");\n\
                    return r;\n\
                }\n\
                void _start(void) {\n\
                    unsigned long tp = (unsigned long)&tcb[0];\n\
                    tcb[0] = tp;\n\
                    *((volatile int *)(tp - 4)) = 9;\n\
                    set_fs(tp);\n\
                    int v = tvar;\n\
                    __asm__ volatile(\"syscall\" : : \"a\"(60L), \"D\"((long)v));\n\
                    __builtin_unreachable();\n\
                }\n";
    let src_path = std::env::temp_dir().join("xold_tls_run.c");
    fs::write(&src_path, src).expect("write runtime source");
    let ok = Command::new(which("clang").expect("clang present"))
        .args([
            "--target=x86_64-linux-gnu",
            "-fno-pic",
            "-ffreestanding",
            "-c",
        ])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("clang runs")
        .success();
    let _ = fs::remove_file(&src_path);
    if !ok {
        let _ = fs::remove_file(&obj);
        eprintln!("skipping TLS runtime: clang could not compile");
        return;
    }

    let out = std::env::temp_dir().join("xold_tls_run.out");
    link_to(
        std::slice::from_ref(&obj),
        &out,
        b"_start",
        false,
        IcfMode::None,
        false,
    )
    .expect("xold TLS runtime link must succeed");
    let _ = fs::remove_file(&obj);
    let status = Command::new(&out)
        .status()
        .expect("linked TLS program must be runnable");
    let _ = fs::remove_file(&out);
    assert_eq!(
        status.code(),
        Some(9),
        "the __thread variable must read back as 9"
    );
}

// --- AArch64: structural --------------------------------------------------

/// `AArch64` cross: a `__thread` program links into a valid static executable
/// with `PT_TLS`, `.tdata`, and no surviving relocations. Gated on clang with
/// an `aarch64-linux-gnu` target.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_aarch64_tls_with_a_tls_segment() {
    let obj = std::env::temp_dir().join("xold_tls_aarch64.o");
    if compile(TLS_SRC, "--target=aarch64-linux-gnu", &[], &obj).is_none() {
        eprintln!(
            "skipping AArch64 TLS: clang with aarch64-linux-gnu target \
             not found"
        );
        return;
    }

    let bytes = link_with_xold(&obj, "aarch64");
    let _ = fs::remove_file(&obj);
    let obj_out =
        ObjectFile::parse(&bytes).expect("xold AArch64 output is valid ELF");
    assert_eq!(obj_out.header().e_type.get(), ET_EXEC);
    assert_eq!(obj_out.machine(), EM_AARCH64);

    let tls = find_phdr(&bytes, PT_TLS).expect("output has a PT_TLS");
    assert_eq!(tls.p_filesz, 4);
    assert_eq!(tls.p_memsz, 4);
    assert_eq!(tls.p_align, 4);

    assert!(
        obj_out
            .sections()
            .iter()
            .map(|s| obj_out.section_name(s))
            .any(|n| n == b".tdata"),
        "has .tdata"
    );
    assert_no_reloc_sections(&obj_out, "AArch64 TLS");
    assert_ne!(obj_out.header().e_entry.get(), 0, "entry `g` is resolved");
}

// --- RISC-V: structural ---------------------------------------------------

/// RISC-V cross: a `__thread` program links into a valid static executable
/// with `PT_TLS`, `.tdata`, and no surviving relocations. Built with
/// `-mno-relax` so the local-exec `TPREL` sequence is not rewritten by linker
/// relaxation (which xold does not implement). Gated on clang with a
/// `riscv64-linux-gnu` target.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_riscv64_tls_with_a_tls_segment() {
    let obj = std::env::temp_dir().join("xold_tls_riscv.o");
    if compile(TLS_SRC, "--target=riscv64-linux-gnu", &["-mno-relax"], &obj)
        .is_none()
    {
        eprintln!(
            "skipping RISC-V TLS: clang with riscv64-linux-gnu target \
             not found"
        );
        return;
    }

    let bytes = link_with_xold(&obj, "riscv");
    let _ = fs::remove_file(&obj);
    let obj_out =
        ObjectFile::parse(&bytes).expect("xold RISC-V output is valid ELF");
    assert_eq!(obj_out.header().e_type.get(), ET_EXEC);
    assert_eq!(obj_out.machine(), EM_RISCV);

    let tls = find_phdr(&bytes, PT_TLS).expect("output has a PT_TLS");
    assert_eq!(tls.p_filesz, 4);
    assert_eq!(tls.p_memsz, 4);
    assert_eq!(tls.p_align, 4);

    assert!(
        obj_out
            .sections()
            .iter()
            .map(|s| obj_out.section_name(s))
            .any(|n| n == b".tdata"),
        "has .tdata"
    );
    assert_no_reloc_sections(&obj_out, "RISC-V TLS");
    assert_ne!(obj_out.header().e_entry.get(), 0, "entry `g` is resolved");
}

// --- qemu runtime: AArch64 + RISC-V local-exec TLS -----------------------

/// The `AArch64` local-exec TLS access sequence (`mrs TPIDR_EL0`; the
/// `TLSLE_ADD_TPREL_HI12`/`LO12_NC` pair baked by the linker) computes a
/// variable address as `TP + TPOFF`. qemu-user does not set up the guest
/// thread pointer for a static binary (that is the CRT's job), so `_start`
/// stands up a minimal TLS image itself: it points `TPIDR_EL0` at a static
/// buffer and seeds the `.tdata` value at the Variant 1 gap (`TP + 16`).
/// For one 4-byte variable `p_align = 4` makes the alignment pad zero, so the
/// TPOFF is exactly the 16-byte gap and the seed lands where the linker-baked
/// offset reads. `counter` starts at 5, `run` increments and returns it, so a
/// correct link leaves the qemu exit code at 6.
const AARCH64_TLS_RUN_SRC: &[u8] = b"__thread int counter = 5;\n\
                static volatile unsigned long tls_area[32];\n\
                int run(void) { counter += 1; return counter; }\n\
                void _start(void) {\n\
                    unsigned long tp = (unsigned long)&tls_area[0];\n\
                    *((volatile int *)(tp + 16)) = 5;\n\
                    __asm__ volatile(\"msr tpidr_el0, %0\" :: \"r\"(tp) : \"memory\");\n\
                    long v = run();\n\
                    register long x8 __asm__(\"x8\") = 93;\n\
                    register long x0 __asm__(\"x0\") = v;\n\
                    __asm__ volatile(\"svc #0\" : : \"r\"(x8), \"r\"(x0));\n\
                    __builtin_unreachable();\n\
                }\n";

/// The RISC-V counterpart. Variant 1 with no gap puts the first variable at
/// `TP + 0`, so `_start` seeds the buffer base and copies it into `tp` (x4);
/// the unrelaxed `TPREL_HI20`/`LO12_I`/`LO12_S` sequence the linker leaves
/// then resolves `counter` to `TP + 0`. Built without `-mno-relax` so the
/// `R_RISCV_RELAX` hints are present and ignored (xold does not relax),
/// matching real compiler output.
const RISCV_TLS_RUN_SRC: &[u8] = b"__thread int counter = 5;\n\
                static volatile int tls_area[32];\n\
                int run(void) { counter += 1; return counter; }\n\
                void _start(void) {\n\
                    unsigned long tp = (unsigned long)&tls_area[0];\n\
                    *((volatile int *)(tp + 0)) = 5;\n\
                    __asm__ volatile(\"mv tp, %0\" :: \"r\"(tp) : \"memory\");\n\
                    long v = run();\n\
                    register long a7 __asm__(\"a7\") = 93;\n\
                    register long a0 __asm__(\"a0\") = v;\n\
                    __asm__ volatile(\"ecall\" : : \"r\"(a7), \"r\"(a0));\n\
                    __builtin_unreachable();\n\
                }\n";

/// Two `__thread` variables exercise the per-variable TPOFF arithmetic: on
/// `AArch64` `a` sits at `TP + 16` and `b` at `TP + 20` (gap 16 plus the 4-byte
/// stride); on RISC-V `a` is at `TP + 0` and `b` at `TP + 4`. `_start` seeds
/// both; `run` mutates each and returns `a + b` (11 + 21 = 32).
const AARCH64_TLS_TWO_SRC: &[u8] = b"__thread int a = 10;\n\
                __thread int b = 20;\n\
                static volatile unsigned long tls_area[32];\n\
                int run(void) { a += 1; b += 1; return a + b; }\n\
                void _start(void) {\n\
                    unsigned long tp = (unsigned long)&tls_area[0];\n\
                    *((volatile int *)(tp + 16)) = 10;\n\
                    *((volatile int *)(tp + 20)) = 20;\n\
                    __asm__ volatile(\"msr tpidr_el0, %0\" :: \"r\"(tp) : \"memory\");\n\
                    long v = run();\n\
                    register long x8 __asm__(\"x8\") = 93;\n\
                    register long x0 __asm__(\"x0\") = v;\n\
                    __asm__ volatile(\"svc #0\" : : \"r\"(x8), \"r\"(x0));\n\
                    __builtin_unreachable();\n\
                }\n";

const RISCV_TLS_TWO_SRC: &[u8] = b"__thread int a = 10;\n\
                __thread int b = 20;\n\
                static volatile int tls_area[32];\n\
                int run(void) { a += 1; b += 1; return a + b; }\n\
                void _start(void) {\n\
                    unsigned long tp = (unsigned long)&tls_area[0];\n\
                    *((volatile int *)(tp + 0)) = 10;\n\
                    *((volatile int *)(tp + 4)) = 20;\n\
                    __asm__ volatile(\"mv tp, %0\" :: \"r\"(tp) : \"memory\");\n\
                    long v = run();\n\
                    register long a7 __asm__(\"a7\") = 93;\n\
                    register long a0 __asm__(\"a0\") = v;\n\
                    __asm__ volatile(\"ecall\" : : \"r\"(a7), \"r\"(a0));\n\
                    __builtin_unreachable();\n\
                }\n";

/// Compiles a self-contained TLS C program (its `_start` sets up the thread
/// pointer), links it with xold (entry `_start`), runs it under `qemu_name`,
/// and asserts the exit code. `tag` namespaces the temp files so the parallel
/// test runner does not collide. Compilation failure or a missing cross-target
/// /qemu prints a note and returns (skip); a link or qemu failure after a
/// successful compile panics, since that is a regression.
fn run_tls_under_qemu(
    src: &[u8],
    target_triple: &str,
    extra: &[&str],
    qemu_name: &str,
    tag: &str,
    expected: i32,
    msg: &str,
) {
    let dir = std::env::temp_dir();
    let obj = dir.join(format!("xold_tlsrt_{tag}.o"));
    let out = dir.join(format!("xold_tlsrt_{tag}.out"));
    let Some(qemu) = which(qemu_name) else {
        eprintln!("skipping {tag} TLS qemu run: {qemu_name} missing");
        return;
    };
    if compile(src, target_triple, extra, &obj).is_none() {
        eprintln!(
            "skipping {tag} TLS qemu run: clang ({target_triple}) missing"
        );
        return;
    }
    link_to(
        std::slice::from_ref(&obj),
        &out,
        b"_start",
        false,
        IcfMode::None,
        false,
    )
    .unwrap_or_else(|_| panic!("{tag} TLS link failed"));
    let status = Command::new(&qemu)
        .arg(&out)
        .status()
        .unwrap_or_else(|_| panic!("{tag} TLS qemu failed to start"));
    for f in [&obj, &out] {
        let _ = fs::remove_file(f);
    }
    assert_eq!(status.code(), Some(expected), "{tag}: {msg}");
}

/// `AArch64` local-exec TLS under qemu: `counter` starts at 5 and `run`
/// returns 6. Gated on the `aarch64-linux-gnu` clang target and
/// `qemu-aarch64-static`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn aarch64_tls_program_runs_under_qemu() {
    run_tls_under_qemu(
        AARCH64_TLS_RUN_SRC,
        "--target=aarch64-linux-gnu",
        &[],
        "qemu-aarch64-static",
        "aa_run",
        6,
        "__thread counter 5 -> 6",
    );
}

/// The same program, compiled so the compiler reaches for the MOVZ/MOVK
/// slices instead of the `ADD_TPREL` pair.
///
/// `-mtls-size=32` says the thread-pointer offset may not fit the pair's 24
/// bits, so clang emits `movz`/`movk` carrying `TLSLE_MOVW_TPREL_G1` and
/// `_G0_NC`. Those are the relocations whose slice is signed and whose sign
/// selects the instruction, so this is the set that cannot be a plain masked
/// bitfield; running the result is what shows the encoder agrees with the
/// hardware and not merely with the disassembler.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn aarch64_movw_tls_program_runs_under_qemu() {
    run_tls_under_qemu(
        AARCH64_TLS_RUN_SRC,
        "--target=aarch64-linux-gnu",
        &["-mtls-size=32"],
        "qemu-aarch64-static",
        "aa_movw",
        6,
        "__thread counter 5 -> 6 through the MOVZ/MOVK slices",
    );
}

/// RISC-V local-exec TLS under qemu: `counter` starts at 5 and `run` returns
/// 6. Gated on the `riscv64-linux-gnu` clang target and `qemu-riscv64-static`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn riscv_tls_program_runs_under_qemu() {
    run_tls_under_qemu(
        RISCV_TLS_RUN_SRC,
        "--target=riscv64-linux-gnu",
        &["-march=rv64gc", "-mabi=lp64d"],
        "qemu-riscv64-static",
        "rv_run",
        6,
        "__thread counter 5 -> 6",
    );
}

/// Two `__thread` variables on `AArch64`: verifies the per-variable TPOFF
/// stride (gap 16 + 4-byte offsets). `run` returns `a + b = 32`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn aarch64_tls_two_variables_run_under_qemu() {
    run_tls_under_qemu(
        AARCH64_TLS_TWO_SRC,
        "--target=aarch64-linux-gnu",
        &[],
        "qemu-aarch64-static",
        "aa_two",
        32,
        "two __thread vars a+b = 11+21 = 32",
    );
}

/// Two `__thread` variables on RISC-V: verifies the no-gap TPOFF stride.
/// `run` returns `a + b = 32`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn riscv_tls_two_variables_run_under_qemu() {
    run_tls_under_qemu(
        RISCV_TLS_TWO_SRC,
        "--target=riscv64-linux-gnu",
        &["-march=rv64gc", "-mabi=lp64d"],
        "qemu-riscv64-static",
        "rv_two",
        32,
        "two __thread vars a+b = 11+21 = 32",
    );
}
