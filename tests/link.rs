//! End-to-end linking: produces an executable with the driver and checks it
//! both runs (the freestanding smoke test) and is structurally valid (the
//! GOT-exercising `min.o` + `ext.o` link).

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::{ObjectFile, constants::*},
    icf::IcfMode,
    linker::link_to,
    mmap_file::MappedFile,
};

mod common;

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    p
}

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

/// Links the freestanding program and runs it. `counter` starts at 5, `entry`
/// increments it, so a correct link leaves the process exit code at 6.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_and_runs_a_freestanding_program() {
    let out = temp("xold_prog.out");
    link_to(
        &[fixture("prog.o"), fixture("start.o")],
        &out,
        b"_start",
        false,
        IcfMode::None,
        false,
    )
    .expect("link must succeed");

    let status = Command::new(&out)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(6),
        "entry() should return counter + 1 = 6"
    );
    let _ = std::fs::remove_file(&out);
}

/// Links `min.o` + `ext.o`, which route `global_counter` access through a GOT
/// entry, and checks the result against our own reader.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_got_relocations_into_a_valid_executable() {
    let out = temp("xold_min.out");
    link_to(
        &[fixture("min.o"), fixture("ext.o")],
        &out,
        b"entry",
        false,
        IcfMode::None,
        false,
    )
    .expect("link must succeed");

    let mapped = MappedFile::open(&out).expect("output must be readable");
    let obj =
        ObjectFile::parse(mapped.bytes()).expect("output must be valid ELF");

    assert_eq!(obj.header().e_type.get(), ET_EXEC);
    assert_eq!(obj.machine(), EM_X86_64);

    let names: Vec<&[u8]> =
        obj.sections().iter().map(|s| obj.section_name(s)).collect();
    for required in [b".text".as_slice(), b".got", b".data", b".bss"] {
        assert!(names.contains(&required), "missing section {required:?}");
    }

    // No relocation sections survive into the output: every relocation was
    // applied.
    assert!(
        obj.sections()
            .iter()
            .all(|s| s.sh_type.get() != SHT_RELA && s.sh_type.get() != SHT_REL),
        "output must contain no relocation sections"
    );

    let symtab = obj
        .symbol_table()
        .expect("symtab readable")
        .expect("output has a symbol table");
    let entry = symtab
        .iter()
        .find(|s| symtab.name(s) == b"entry")
        .expect("`entry` is exported");
    assert_eq!(
        obj.header().e_entry.get(),
        entry.st_value.get(),
        "the entry point must be `entry`'s address"
    );

    // The GOT entry for `global_counter` must hold the symbol's address.
    let counter = symtab
        .iter()
        .find(|s| symtab.name(s) == b"global_counter")
        .expect("`global_counter` is exported");
    let got = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".got")
        .expect("output has a .got");
    let bytes = obj.section_data(got).expect("got bytes");
    let mut value = 0u64;
    for (i, b) in bytes.iter().take(8).enumerate() {
        value |= u64::from(*b) << (i * 8);
    }
    assert_eq!(
        value,
        counter.st_value.get(),
        "GOT entry must point at global_counter"
    );

    let _ = std::fs::remove_file(&out);
}

/// The `AArch64` cross end-to-end link. This is gated on clang with an
/// `aarch64-linux-gnu` target being available, because that is what lets the
/// test compile a real `AArch64` object carrying `R_AARCH64_ADR_PREL_PG_HI21`
/// and `R_AARCH64_LDST32_ABS_LO12_NC` relocations. `qemu-aarch64` is not
/// required: the link output is checked structurally (it is a valid
/// `EM_AARCH64` executable with every relocation applied), not by running it.
/// If the cross-compiler is absent the test prints a note and returns, so the
/// build never fails over a missing toolchain.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_an_aarch64_object_into_a_valid_executable() {
    let Some(clang) = aarch64_clang() else {
        eprintln!(
            "skipping AArch64 end-to-end: clang with aarch64-linux-gnu \
             target not found"
        );
        return;
    };

    // `counter` lives in `.data` and is reached from `entry` via an
    // ADRP + LDST32 pair, exercising the page-relative split immediate and a
    // scaled 12-bit displacement.
    let src = b"volatile int counter = 5;\n\
                int entry(void) { counter = counter + 1; return counter; }\n";
    let src_path = std::env::temp_dir().join("xold_aarch64_c.c");
    let obj = std::env::temp_dir().join("xold_aarch64_c.o");
    let out = std::env::temp_dir().join("xold_aarch64.out");

    fs::write(&src_path, src).expect("write AArch64 source");
    let compiled = Command::new(clang)
        .args([
            "--target=aarch64-linux-gnu",
            "-ffreestanding",
            "-nostdlib",
            "-fno-pic",
            "-c",
        ])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status();
    let ok = compiled.is_ok_and(|s| s.success());
    if !ok {
        eprintln!(
            "skipping AArch64 end-to-end: \
             clang could not compile for aarch64-linux-gnu"
        );
        let _ = fs::remove_file(&src_path);
        return;
    }

    link_to(
        std::slice::from_ref(&obj),
        &out,
        b"entry",
        false,
        IcfMode::None,
        false,
    )
    .expect("AArch64 link must succeed");

    let mapped = MappedFile::open(&out).expect("output must be readable");
    let obj_out =
        ObjectFile::parse(mapped.bytes()).expect("output must be valid ELF");
    assert_eq!(obj_out.header().e_type.get(), ET_EXEC);
    assert_eq!(obj_out.machine(), EM_AARCH64);

    // Every relocation was applied: none survive into the output.
    assert!(
        obj_out
            .sections()
            .iter()
            .all(|s| s.sh_type.get() != SHT_RELA && s.sh_type.get() != SHT_REL),
        "output must contain no relocation sections"
    );

    // The entry symbol resolved to a non-zero address.
    assert_ne!(obj_out.header().e_entry.get(), 0, "entry must be resolved");

    let _ = fs::remove_file(&src_path);
    let _ = fs::remove_file(&obj);
    let _ = fs::remove_file(&out);
}

/// The RISC-V cross end-to-end link. Gated on clang with a `riscv64-linux-gnu`
/// target being available, which is what lets the test compile a real rv64
/// object carrying an `R_RISCV_CALL_PLT` relocation (and its paired
/// `R_RISCV_RELAX` hint). `qemu-riscv64` is not required: the link output is
/// checked structurally (a valid `EM_RISCV` executable with every relocation
/// applied), not by running it. If the cross-compiler is absent the test prints
/// a note and returns, so the build never fails over a missing toolchain. The
/// source is written to use a function call rather than a global variable so it
/// stays within the rv64 reloc set this phase reduces (`PCREL_LO12_*`, which
/// needs a paired-HI20 lookup, is still deferred).
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_a_riscv_object_into_a_valid_executable() {
    let Some(clang) = riscv_clang() else {
        eprintln!(
            "skipping RISC-V end-to-end: clang with riscv64-linux-gnu \
             target not found"
        );
        return;
    };

    // `entry` calls `helper`, producing an `auipc`+`jalr` pair relocated by
    // `R_RISCV_CALL_PLT` (plus an `R_RISCV_RELAX` hint at the same offset).
    let src = b"int helper(int x) { return x + 1; }\n\
                int entry(int x) { return helper(x) + 2; }\n";
    let src_path = std::env::temp_dir().join("xold_riscv_c.c");
    let obj = std::env::temp_dir().join("xold_riscv_c.o");
    let out = std::env::temp_dir().join("xold_riscv.out");

    fs::write(&src_path, src).expect("write RISC-V source");
    let compiled = Command::new(clang)
        .args([
            "--target=riscv64-linux-gnu",
            "-ffreestanding",
            "-nostdlib",
            "-fno-pic",
            "-c",
        ])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status();
    let ok = compiled.is_ok_and(|s| s.success());
    if !ok {
        eprintln!(
            "skipping RISC-V end-to-end: \
             clang could not compile for riscv64-linux-gnu"
        );
        let _ = fs::remove_file(&src_path);
        return;
    }

    link_to(
        std::slice::from_ref(&obj),
        &out,
        b"entry",
        false,
        IcfMode::None,
        false,
    )
    .expect("RISC-V link must succeed");

    let mapped = MappedFile::open(&out).expect("output must be readable");
    let obj_out =
        ObjectFile::parse(mapped.bytes()).expect("output must be valid ELF");
    assert_eq!(obj_out.header().e_type.get(), ET_EXEC);
    assert_eq!(obj_out.machine(), EM_RISCV);

    // Every relocation was applied: none survive into the output.
    assert!(
        obj_out
            .sections()
            .iter()
            .all(|s| s.sh_type.get() != SHT_RELA && s.sh_type.get() != SHT_REL),
        "output must contain no relocation sections"
    );

    // The entry symbol resolved to a non-zero address.
    assert_ne!(obj_out.header().e_entry.get(), 0, "entry must be resolved");

    let _ = fs::remove_file(&src_path);
    let _ = fs::remove_file(&obj);
    let _ = fs::remove_file(&out);
}

/// The RISC-V cross end-to-end link with a global variable access. Gated on
/// clang with a `riscv64-linux-gnu` target, which lets the test compile a real
/// rv64 object whose `.text` reaches a global through `R_RISCV_HI20` /
/// `R_RISCV_LO12_I` / `R_RISCV_LO12_S` (each paired with `R_RISCV_RELAX`), and
/// whose `.eh_frame` carries `R_RISCV_32_PCREL`, the `ADD32`/`SUB32` pair and
/// the `SET6`/`SUB6` pair. The link must fold `.eh_frame` into `.rodata` and
/// resolve every relocation so none survive into the output. If the
/// cross-compiler is absent the test prints a note and returns, so the build
/// never fails over a missing toolchain.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_a_riscv_global_object_into_a_valid_executable() {
    let Some(clang) = riscv_clang() else {
        eprintln!(
            "skipping RISC-V global end-to-end: clang with \
             riscv64-linux-gnu target not found"
        );
        return;
    };

    // `counter` lives in `.data` and is read and written from `bump`, which
    // forces absolute hi/lo global addressing plus a full `.eh_frame`.
    let src = b"int counter = 5;\n\
                int bump(void) { counter += 1; return counter; }\n";
    let src_path = std::env::temp_dir().join("xold_riscv_glob_c.c");
    let obj = std::env::temp_dir().join("xold_riscv_glob_c.o");
    let out = std::env::temp_dir().join("xold_riscv_glob.out");

    fs::write(&src_path, src).expect("write RISC-V source");
    // `-fno-pic` selects absolute HI20/LO12 global addressing; without
    // `-ffreestanding` clang keeps emitting `.eh_frame`, which is what lets
    // this case exercise the ADD32/SUB32 and SET6/SUB6 pairs.
    let compiled = Command::new(clang)
        .args(["--target=riscv64-linux-gnu", "-fno-pic", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status();
    let ok = compiled.is_ok_and(|s| s.success());
    if !ok {
        eprintln!(
            "skipping RISC-V global end-to-end: \
             clang could not compile for riscv64-linux-gnu"
        );
        let _ = fs::remove_file(&src_path);
        return;
    }

    link_to(
        std::slice::from_ref(&obj),
        &out,
        b"bump",
        false,
        IcfMode::None,
        false,
    )
    .expect("RISC-V global link must succeed");

    let mapped = MappedFile::open(&out).expect("output must be readable");
    let obj_out =
        ObjectFile::parse(mapped.bytes()).expect("output must be valid ELF");
    assert_eq!(obj_out.header().e_type.get(), ET_EXEC);
    assert_eq!(obj_out.machine(), EM_RISCV);

    // Every relocation was applied: none survive into the output, including
    // the eh_frame paired relocations that this object exercises.
    assert!(
        obj_out
            .sections()
            .iter()
            .all(|s| s.sh_type.get() != SHT_RELA && s.sh_type.get() != SHT_REL),
        "output must contain no relocation sections"
    );

    // The expected allocated sections are present: code, the read-only
    // `.eh_frame` (kept as its own section so `.eh_frame_hdr` can index it),
    // and the initialised global in `.data`.
    let names: Vec<&[u8]> = obj_out
        .sections()
        .iter()
        .map(|s| obj_out.section_name(s))
        .collect();
    for required in [b".text".as_slice(), b".eh_frame", b".data"] {
        assert!(names.contains(&required), "missing section {required:?}");
    }

    // The entry symbol resolved to a non-zero address.
    assert_ne!(obj_out.header().e_entry.get(), 0, "entry must be resolved");

    let _ = fs::remove_file(&src_path);
    let _ = fs::remove_file(&obj);
    let _ = fs::remove_file(&out);
}

/// The RISC-V cross end-to-end link with PC-relative global addressing.
/// `-mcmodel=medany` makes clang reach the global with an `auipc` + `addi`
/// pair relocated by `R_RISCV_PCREL_HI20` and `R_RISCV_PCREL_LO12_I`, where the
/// LO12's symbol points at the `auipc` rather than the global. The writer's
/// PCREL pairing pass must rewrite the LO12 onto the HI20's target so the pair
/// reconstructs the global's address. Gated on clang with a `riscv64-linux-gnu`
/// target; if absent the test prints a note and returns, so the build never
/// fails over a missing toolchain.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_a_riscv_pcrel_object_into_a_valid_executable() {
    let Some(clang) = riscv_clang() else {
        eprintln!(
            "skipping RISC-V PCREL end-to-end: clang with \
             riscv64-linux-gnu target not found"
        );
        return;
    };

    let src = b"int counter = 5;\n\
                int bump(void) { counter += 1; return counter; }\n";
    let src_path = std::env::temp_dir().join("xold_riscv_pcrel_c.c");
    let obj = std::env::temp_dir().join("xold_riscv_pcrel_c.o");
    let out = std::env::temp_dir().join("xold_riscv_pcrel.out");

    fs::write(&src_path, src).expect("write RISC-V source");
    let compiled = Command::new(clang)
        .args([
            "--target=riscv64-linux-gnu",
            "-mcmodel=medany",
            "-fno-pic",
            "-c",
        ])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status();
    let ok = compiled.is_ok_and(|s| s.success());
    if !ok {
        eprintln!(
            "skipping RISC-V PCREL end-to-end: \
             clang could not compile for riscv64-linux-gnu"
        );
        let _ = fs::remove_file(&src_path);
        return;
    }

    link_to(
        std::slice::from_ref(&obj),
        &out,
        b"bump",
        false,
        IcfMode::None,
        false,
    )
    .expect("RISC-V PCREL link must succeed");

    let mapped = MappedFile::open(&out).expect("output must be readable");
    let obj_out =
        ObjectFile::parse(mapped.bytes()).expect("output must be valid ELF");
    assert_eq!(obj_out.header().e_type.get(), ET_EXEC);
    assert_eq!(obj_out.machine(), EM_RISCV);

    // The paired HI20/LO12 pass resolved every relocation.
    assert!(
        obj_out
            .sections()
            .iter()
            .all(|s| s.sh_type.get() != SHT_RELA && s.sh_type.get() != SHT_REL),
        "output must contain no relocation sections"
    );
    assert_ne!(obj_out.header().e_entry.get(), 0, "entry must be resolved");

    let _ = fs::remove_file(&src_path);
    let _ = fs::remove_file(&obj);
    let _ = fs::remove_file(&out);
}

/// Compiles `src` into `obj` with the cross `clang` + `cflags`. `src_path`
/// names the temporary source (its extension selects the language). Returns
/// whether compilation succeeded.
fn compile_obj(
    clang: &Path,
    cflags: &[&str],
    src: &[u8],
    src_path: &Path,
    obj: &Path,
) -> bool {
    if fs::write(src_path, src).is_err() {
        return false;
    }
    Command::new(clang)
        .args(cflags)
        .arg("-c")
        .arg(src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .is_ok_and(|s| s.success())
}

/// The `AArch64` runtime end-to-end test, the first that actually executes the
/// cross-arch output. `counter` lives in `.data` and is reached from `main`
/// through an ADRP + LDST32 pair, exactly the relocation whose
/// mask-before-shift encoding this phase fixes (a shift-first encode would
/// load from `0x1000` past the page and segfault). Gated on both the
/// `aarch64-linux-gnu` clang target and `qemu-aarch64-static`; if either is
/// absent the test prints a note and returns, so the build never fails over a
/// missing toolchain. `counter` starts at 5 and `main` increments and returns
/// it, so a correct link leaves the qemu exit code at 6.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn runs_an_aarch64_program_under_qemu() {
    let Some(clang) = aarch64_clang() else {
        eprintln!(
            "skipping AArch64 qemu run: \
             clang (aarch64-linux-gnu) missing"
        );
        return;
    };
    let Some(qemu) = which("qemu-aarch64-static") else {
        eprintln!("skipping AArch64 qemu run: qemu-aarch64-static missing");
        return;
    };
    // `_start` calls `main` and exits with its return value (syscall 93).
    let start =
        b".global _start\n_start:\n\tbl main\n\tmov x8, #93\n\tsvc #0\n";
    let prog = b"volatile int counter = 5;\n\
                 int main(void) { counter += 1; return counter; }\n";
    run_cross_and_assert(&clang, start, prog, &qemu, "aa", 6, "counter 5 -> 6");
}

/// The RISC-V runtime end-to-end test. `main` mutates a global (`counter`)
/// inside a loop, reaching it through `R_RISCV_HI20` / `R_RISCV_LO12_I` and
/// producing a `.eh_frame` whose paired relocations (`32_PCREL`, `SET6`/
/// `SUB6`, `ADD32`/`SUB32`) must all resolve, plus a `BRANCH` for the loop
/// and a `CALL_PLT` pair for `_start` -> `main`. Gated on the
/// `riscv64-linux-gnu` clang target and `qemu-riscv64-static`; if either is
/// absent the test returns. `counter` starts at 5 and the loop adds 3, so the
/// exit code is 8.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn runs_a_riscv_program_under_qemu() {
    let Some(clang) = riscv_clang() else {
        eprintln!(
            "skipping RISC-V qemu run: \
             clang (riscv64-linux-gnu) missing"
        );
        return;
    };
    let Some(qemu) = which("qemu-riscv64-static") else {
        eprintln!("skipping RISC-V qemu run: qemu-riscv64-static missing");
        return;
    };
    // `_start` calls `main` (a0 = return value) and exits (syscall 93).
    let start = b".global _start\n\
                 _start:\n\tli a0, 0\n\tcall main\n\tli a7, 93\n\tecall\n";
    let prog = b"int counter = 5;\n\
                 int main(void) {\n\
                 \tfor (int i = 0; i < 3; i++) counter += 1;\n\
                 \treturn counter;\n\
                 }\n";
    run_cross_and_assert(
        &clang,
        start,
        prog,
        &qemu,
        "rv",
        8,
        "counter 5 + 3 -> 8",
    );
}

/// Compiles `start` (assembly) and `prog` (C) with the arch's cross clang,
/// links them with xold (entry `_start`), runs the result under `qemu`, and
/// asserts the exit code. `tag` namespaces the temp files (tests run in
/// parallel). Compilation failure prints a note and returns (skip); a link or
/// qemu failure after a successful compile panics, since that is a regression.
fn run_cross_and_assert(
    clang: &Path,
    start: &[u8],
    prog: &[u8],
    qemu: &Path,
    tag: &str,
    expected: i32,
    msg: &str,
) {
    let dir = std::env::temp_dir();
    let s_src = dir.join(format!("xold_{tag}_run_start.s"));
    let c_src = dir.join(format!("xold_{tag}_run_prog.c"));
    let s_obj = dir.join(format!("xold_{tag}_run_start.o"));
    let c_obj = dir.join(format!("xold_{tag}_run_prog.o"));
    let out = dir.join(format!("xold_{tag}_run.out"));
    let cflags = cross_cflags(tag);
    let made = compile_obj(clang, cflags, start, &s_src, &s_obj)
        && compile_obj(clang, cflags, prog, &c_src, &c_obj);
    if !made {
        eprintln!("skipping {tag} qemu run: clang could not compile");
        for f in [&s_src, &c_src, &s_obj, &c_obj, &out] {
            let _ = fs::remove_file(f);
        }
        return;
    }
    link_to(
        &[s_obj.clone(), c_obj.clone()],
        &out,
        b"_start",
        false,
        IcfMode::None,
        false,
    )
    .unwrap_or_else(|_| panic!("{tag} link failed"));
    let status = Command::new(qemu)
        .arg(&out)
        .status()
        .unwrap_or_else(|_| panic!("{tag} qemu failed to start"));
    let code = status.code();
    for f in [&s_src, &c_src, &s_obj, &c_obj, &out] {
        let _ = fs::remove_file(f);
    }
    assert_eq!(code, Some(expected), "{tag}: {msg}");
}

/// The cross-clang flags for each arch `tag`. Both builds are non-PIC so the
/// compiler reaches globals through absolute (`AArch64`) or HI20/LO12 (RISC-V)
/// addressing rather than a GOT.
fn cross_cflags(tag: &str) -> &'static [&'static str] {
    match tag {
        "aa" => &[
            "--target=aarch64-linux-gnu",
            "-ffreestanding",
            "-nostdlib",
            "-fno-pic",
        ],
        "rv" => &[
            "--target=riscv64-linux-gnu",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-fno-pic",
        ],
        _ => &[],
    }
}

/// Returns the clang binary if it can produce an `aarch64-linux-gnu` object,
/// else `None`.
fn aarch64_clang() -> Option<PathBuf> {
    let clang = which("clang")?;
    let ok = Command::new(&clang)
        .args(["--target=aarch64-linux-gnu", "-c", "-x", "c", "-", "-o"])
        .arg(std::env::temp_dir().join("xold_aarch64_probe.o"))
        .stdin(std::process::Stdio::null())
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(std::env::temp_dir().join("xold_aarch64_probe.o"));
    ok.then_some(clang)
}

/// Returns the clang binary if it can produce a `riscv64-linux-gnu` object,
/// else `None`.
fn riscv_clang() -> Option<PathBuf> {
    let clang = which("clang")?;
    let ok = Command::new(&clang)
        .args(["--target=riscv64-linux-gnu", "-c", "-x", "c", "-", "-o"])
        .arg(std::env::temp_dir().join("xold_riscv_probe.o"))
        .stdin(std::process::Stdio::null())
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(std::env::temp_dir().join("xold_riscv_probe.o"));
    ok.then_some(clang)
}
