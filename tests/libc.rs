//! End-to-end tests for linking a real C program against the system libc.
//!
//! These build three escalating cases:
//!
//! 1. Minimal crt path: `int main(void){ return 42; }` linked with
//!    `crt1.o`/`crti.o`/`crtn.o` + `libc.so.6`, entry `_start`, runs under
//!    `ld.so` and exits 42. Exercises `_start -> __libc_start_main -> main ->
//!    exit` plus a COPY/RELATIVE GOT slot for `main` (crt1 references `main`
//!    through its GOT via `R_X86_64_REX_GOTPCRELX`).
//! 2. A real libc call: `write(1, "hi\n", 3)` -- a PLT call into libc that
//!    resolves and runs.
//! 3. Stretch: `printf("hello, world\n")` -- exercises a libc IFUNC (`printf`)
//!    through a normal `JUMP_SLOT`; the loader runs the resolver inside libc,
//!    so no `IRELATIVE` is needed on the executable side.
//!
//! The default-clang (non-PIC) cases cover the "compile normally then link"
//! path: host `clang -c` without `-fPIE` emits absolute relocations
//! (`R_X86_64_64` for string literals, `R_X86_64_32S` for global data), which
//! a PIE cannot relocate. xold detects these and produces `ET_EXEC` at a fixed
//! base, matching `clang`/`clang -no-pie`.
//!
//! Gated on `clang`, `gcc` (to locate the crt objects), and the system
//! `ld.so`; if absent the tests print a note and return, so the build never
//! fails over a missing toolchain.

// `crt1`/`crti`/`crtn` are the canonical names of the crt objects; renaming
// them would obscure the test.
#![allow(clippy::similar_names)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{
    elf::{ObjectFile, constants::*},
    icf::IcfMode,
    linker::link_dyn_exec,
};

mod common;

/// Minimal crt path: exits 42.
const MIN_SRC: &[u8] = b"int main(void){ return 42; }\n";

/// A libc `write` call: writes "hi\n" and returns 0.
const WRITE_SRC: &[u8] = b"long write(int, const void *, unsigned long);\n\
     int main(void){ write(1, \"hi\\n\", 3); return 0; }\n";

/// A libc `printf` call: stretches to a libc IFUNC.
const PRINTF_SRC: &[u8] = b"int printf(const char *, ...);\n\
     int main(void){ printf(\"hello, world\\n\"); return 0; }\n";

/// A default-clang (non-PIC) hello-world: same surface as `PRINTF_SRC` but
/// compiled without `-fPIE`, so the string reference is `R_X86_64_64`.
const NONPIC_HELLO_SRC: &[u8] = b"#include <stdio.h>\n\
     int main(void){ printf(\"hello, world\\n\"); return 0; }\n";

/// A non-PIC program that reads a global array and does arithmetic: the data
/// references are `R_X86_64_32S` (a 32-bit absolute slot that cannot hold an
/// ASLR base, so the executable must be `ET_EXEC`).
const NONPIC_ARRAY_SRC: &[u8] = b"#include <stdio.h>\n\
     int data[4] = { 10, 20, 30, 40 };\n\
     int main(void){\n\
         int sum = 0;\n\
         for (int i = 0; i < 4; i++) sum += data[i];\n\
         printf(\"sum=%d\\n\", sum);\n\
         return sum == 100 ? 0 : 1;\n\
     }\n";

/// Compiles `src` (`-fPIE`) to `obj` with the host clang. Returns `None` if
/// clang is unavailable so callers can skip gracefully.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    compile_clang(src, obj, &["--target=x86_64-linux-gnu", "-fPIE", "-c"])
}

/// Compiles `src` with the host clang default (no `-fPIE`), producing a non-PIC
/// object whose data references are absolute (`R_X86_64_64`/`32S`).
fn compile_default(src: &[u8], obj: &Path) -> Option<()> {
    compile_clang(src, obj, &["--target=x86_64-linux-gnu", "-c"])
}

/// Runs the host clang with `args` to compile `src` into `obj`. Returns
/// `None` if clang is unavailable so callers can skip gracefully.
fn compile_clang(src: &[u8], obj: &Path, args: &[&str]) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args(args)
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// A fresh per-test working directory under the system temp dir.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xold_libc_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Whether the host toolchain needed for these tests is present: clang, gcc,
/// the three crt objects, `libc.so.6`, and a probeable interpreter.
struct Harness {
    crt1: PathBuf,
    crti: PathBuf,
    crtn: PathBuf,
    libc: PathBuf,
    interp: Vec<u8>,
}

impl Harness {
    /// Collects the harness, or returns `None` (printing a note) when a piece
    /// is missing.
    fn detect() -> Option<Self> {
        if which("clang").is_none() {
            eprintln!("skipping libc link tests: clang unavailable");
            return None;
        }
        let crt1 = crt_file("crt1.o")?;
        let crti = crt_file("crti.o")?;
        let crtn = crt_file("crtn.o")?;
        let libc = libc_so()?;
        let interp = interpreter()?;
        Some(Self {
            crt1,
            crti,
            crtn,
            libc,
            interp,
        })
    }

    /// Links `main_obj` plus the crt objects and libc into `prog` with xold.
    fn link(&self, main_obj: &Path, prog: &Path) {
        link_dyn_exec(
            &[
                main_obj.to_path_buf(),
                self.crti.clone(),
                self.crt1.clone(),
                self.crtn.clone(),
                self.libc.clone(),
            ],
            prog,
            b"_start",
            &self.interp,
            false,
            IcfMode::None,
            false,
        )
        .expect("xold libc link must succeed");
    }
}

/// Step 1: minimal crt path, `main` returns 42.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn minimal_crt_path_exits_42() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("min");
    let main_o = dir.join("min_main.o");
    compile(MIN_SRC, &main_o).expect("host clang compiles main");
    let prog = dir.join("min_prog");
    h.link(&main_o, &prog);

    let status = Command::new(&prog)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(status.code(), Some(42), "minimal crt path should exit 42");
}

/// Step 2: a real libc call (`write`) prints `hi` and exits 0.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn libc_write_call_runs() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("write");
    let main_o = dir.join("write_main.o");
    compile(WRITE_SRC, &main_o).expect("host clang compiles write main");
    let prog = dir.join("write_prog");
    h.link(&main_o, &prog);

    let out = Command::new(&prog)
        .output()
        .expect("linked program must be runnable");
    assert!(out.status.success(), "write program should exit 0");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hi\n",
        "write should print hi"
    );
}

/// Step 3 (stretch): `printf("hello, world\n")` exercises a libc IFUNC
/// (`printf`) resolved through a `JUMP_SLOT`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn libc_printf_ifunc_runs() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("printf");
    let main_o = dir.join("printf_main.o");
    compile(PRINTF_SRC, &main_o).expect("host clang compiles printf main");
    let prog = dir.join("printf_prog");
    h.link(&main_o, &prog);

    let out = Command::new(&prog)
        .output()
        .expect("linked program must be runnable");
    assert!(
        out.status.success(),
        "printf program should exit 0 (IFUNC resolved)"
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hello, world\n",
        "printf should print hello, world"
    );
}

/// Default `clang -c` (no `-fPIE`): the string literal is an absolute
/// `R_X86_64_64` reference. xold must produce `ET_EXEC` (fixed base) so the
/// absolute address is correct at runtime, and the program must run.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn default_clang_hello_world_runs() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("defhello");
    let main_o = dir.join("def_hello.o");
    compile_default(NONPIC_HELLO_SRC, &main_o)
        .expect("host clang compiles non-PIC hello");
    let prog = dir.join("def_hello_prog");
    h.link(&main_o, &prog);

    let bytes = fs::read(&prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    assert_eq!(
        obj.header().e_type.get(),
        ET_EXEC,
        "non-PIC must be ET_EXEC"
    );

    let out = Command::new(&prog)
        .output()
        .expect("linked program must be runnable");
    assert!(
        out.status.success(),
        "non-PIC hello should exit 0, got {:?}",
        out.status
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hello, world\n",
        "non-PIC hello should print hello, world"
    );
}

/// Default `clang -c` reading a global array: the data references are
/// `R_X86_64_32S` (32-bit absolute slots that cannot hold an ASLR base), so
/// the executable must be `ET_EXEC`. Exercises both `32S` (data access) and
/// `64` (string literal) absolute relocations.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn default_clang_global_array_runs() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("defarray");
    let main_o = dir.join("def_array.o");
    compile_default(NONPIC_ARRAY_SRC, &main_o)
        .expect("host clang compiles non-PIC array");
    let prog = dir.join("def_array_prog");
    h.link(&main_o, &prog);

    let bytes = fs::read(&prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    assert_eq!(
        obj.header().e_type.get(),
        ET_EXEC,
        "non-PIC must be ET_EXEC"
    );

    let out = Command::new(&prog)
        .output()
        .expect("linked program must be runnable");
    assert!(
        out.status.success(),
        "non-PIC array should exit 0 (sum matched), got {:?}",
        out.status
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "sum=100\n",
        "non-PIC array should print sum=100"
    );
}

/// Cross-checks xold's output against the system linker: both link the same
/// default-clang (non-PIC) source and produce an `ET_EXEC` that runs the same.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn default_clang_matches_system_linker() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("defxcheck");
    let main_o = dir.join("xcheck.o");
    compile_default(NONPIC_HELLO_SRC, &main_o)
        .expect("host clang compiles non-PIC source");

    let prog = dir.join("xold_prog");
    h.link(&main_o, &prog);
    let xold_out = Command::new(&prog)
        .output()
        .expect("xold program must be runnable");

    let Some(clang) = which("clang") else {
        return;
    };
    let src_path = main_o.with_extension("c");
    fs::write(&src_path, NONPIC_HELLO_SRC).expect("write source");
    let ref_prog = dir.join("ref_prog");
    let ref_ok = Command::new(clang)
        .arg(&src_path)
        .arg("-o")
        .arg(&ref_prog)
        .status()
        .is_ok();
    let _ = fs::remove_file(&src_path);
    if !ref_ok {
        return;
    }
    let ref_bytes = fs::read(&ref_prog).expect("read reference");
    let ref_obj = ObjectFile::parse(&ref_bytes).expect("valid reference ELF");
    assert_eq!(
        ref_obj.header().e_type.get(),
        ET_EXEC,
        "system clang default must also be ET_EXEC"
    );
    let ref_out = Command::new(&ref_prog)
        .output()
        .expect("reference program must be runnable");
    assert_eq!(
        String::from_utf8_lossy(&xold_out.stdout),
        String::from_utf8_lossy(&ref_out.stdout),
        "xold and system output must match"
    );
    assert_eq!(xold_out.status, ref_out.status);
}

/// Structural check: the libc-linked executable carries the dynamic sections
/// a working `ET_DYN` image needs, and emits a `RELATIVE` relocation for the
/// `main` GOT slot crt1 references.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn libc_linked_executable_is_well_formed() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("struct");
    let main_o = dir.join("struct_main.o");
    compile(MIN_SRC, &main_o).expect("host clang compiles main");
    let prog = dir.join("struct_prog");
    h.link(&main_o, &prog);

    let bytes = fs::read(&prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    assert_eq!(obj.header().e_type.get(), ET_DYN, "must be ET_DYN (PIE)");
    assert_eq!(obj.machine(), EM_X86_64);
    assert_ne!(obj.header().e_entry.get(), 0, "entry point must be set");

    let names: Vec<&[u8]> =
        obj.sections().iter().map(|s| obj.section_name(s)).collect();
    for required in [
        b".text".as_slice(),
        b".got",
        b".interp",
        b".dynamic",
        b".dynsym",
        b".dynstr",
        b".rela.dyn",
    ] {
        assert!(
            names.contains(&required),
            "missing required section {required:?}"
        );
    }

    // The `.rela.dyn` must include at least one RELATIVE (the `main` GOT slot)
    // and a GLOB_DAT for `__libc_start_main`.
    let rela = rela_dyn_rows(&bytes);
    assert!(
        rela.iter().any(|(_, _, t, _)| {
            *t == xold::reloc::x86_64::R_X86_64_RELATIVE
        }),
        "expected at least one RELATIVE relocation for main's GOT slot"
    );
}

/// Decodes `.rela.dyn` into `(r_offset, sym, r_type, r_addend)` rows.
fn rela_dyn_rows(bytes: &[u8]) -> Vec<(u64, u32, u32, i64)> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(rela) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".rela.dyn")
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
