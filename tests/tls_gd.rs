//! General-dynamic TLS lowering.
//!
//! A compiler emits the general-dynamic sequence for a `_Thread_local` it
//! cannot prove is local -- which is what `-fPIC` gives you for any ordinary
//! C program -- so the object carries `R_X86_64_TLSGD` plus a `PLT32` call to
//! `__tls_get_addr`. In an executable the variable lives in the main module's
//! static TLS block, so its offset from the thread pointer is fixed at link
//! time and the pair is rewritten into the local-exec form.
//!
//! This is a lowering, not an optimisation: xold has no runtime
//! `__tls_get_addr` to fall back on, so it must happen whether or not
//! relaxation was requested. These tests link *without* `--relax` for exactly
//! that reason.
//!
//! The programs are freestanding, so `_start` seeds the thread pointer with
//! `arch_prctl` before touching a thread-local -- that is the C runtime's job
//! in a hosted program, and there is no C runtime here.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{Error, icf::IcfMode, linker::link_to};

mod common;

fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_tlsgd_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Two thread-locals exercised through the general-dynamic sequence, with the
/// thread pointer set up by hand. Exits 0 only if every access reached the
/// right storage.
const SRC: &str = r#"
_Thread_local int counter = 7;
_Thread_local long wide = 100;

static volatile unsigned long tcb[16];

static long set_fs(unsigned long a) {
    long r;
    __asm__ volatile("syscall" : "=a"(r) : "0"(158L), "D"(0x1002L), "S"(a)
                     : "rcx", "r11", "memory");
    return r;
}

int bump(void) { counter += 1; return counter; }
long widen(void) { wide += 5; return wide; }

static int check(void) {
    counter = 7;
    wide = 100;
    if (bump() != 8) return 1;
    if (widen() != 105) return 2;
    if (counter != 8) return 3;
    /* The address must be stable across accesses. */
    int *p = &counter;
    *p = 42;
    if (counter != 42) return 4;
    return 0;
}

void _start(void) {
    unsigned long tp = (unsigned long)&tcb[8];
    tcb[8] = tp;
    set_fs(tp);
    int rc = check();
    __asm__ volatile("syscall" : : "a"(60L), "D"((long)rc));
    __builtin_unreachable();
}
"#;

/// Compiles `SRC` position-independently, which is what makes the compiler
/// choose the general-dynamic model.
fn compile(obj: &Path, dir: &Path) -> Option<()> {
    let clang = which("clang")?;
    let file = dir.join("tlsgd.c");
    fs::write(&file, SRC).ok()?;
    Command::new(clang)
        .args([
            "-c",
            "-O1",
            "-fPIC",
            "-ftls-model=global-dynamic",
            "-ffreestanding",
            "-o",
        ])
        .arg(obj)
        .arg(&file)
        .status()
        .ok()?
        .success()
        .then_some(())
}

/// A hand-written general-dynamic pair whose `lea` is missing the `data16`
/// prefix the psABI sequence carries. Everything else is canonical: the
/// relocation, its `-4` addend, and the `call` that closes the pair. The
/// relocations are spelled with `.reloc` so the assembler emits the pair
/// exactly as written rather than the encoding it would choose itself.
const DECLINED_SRC: &str = r#"
        .section .tdata,"awT",@progbits
        .globl  tv
tv:     .long   42

        .text
        .globl  _start
_start:
        nop
        .byte   0x48, 0x8d, 0x3d
        .reloc  ., R_X86_64_TLSGD, tv-4
        .long   0
        .byte   0x66, 0x66, 0x48, 0xe8
        .reloc  ., R_X86_64_PLT32, __tls_get_addr-4
        .long   0
        movl    $60, %eax
        xorl    %edi, %edi
        syscall
"#;

/// Assembles [`DECLINED_SRC`] into an object, or `None` without clang.
fn assemble(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let file = dir.join("declined.s");
    let obj = dir.join("declined.o");
    fs::write(&file, DECLINED_SRC).ok()?;
    Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c", "-o"])
        .arg(&obj)
        .arg(&file)
        .status()
        .ok()?
        .success()
        .then_some(obj)
}

/// Links the object with xold, returning the image path.
fn build(dir: &Path, relax: bool) -> Option<PathBuf> {
    let obj = dir.join("tlsgd.o");
    compile(&obj, dir)?;
    // The object must actually carry the general-dynamic pair, or the test
    // would pass without exercising the lowering at all.
    let relocs = Command::new("readelf")
        .args(["-rW"])
        .arg(&obj)
        .output()
        .ok()?;
    assert!(
        String::from_utf8_lossy(&relocs.stdout).contains("R_X86_64_TLSGD"),
        "the fixture must carry a general-dynamic pair to be a valid test"
    );
    let prog = dir.join(if relax { "prog_relax" } else { "prog" });
    link_to(
        std::slice::from_ref(&obj),
        &prog,
        b"_start",
        false,
        IcfMode::None,
        relax,
    )
    .expect("a program with a general-dynamic thread-local must link");
    Some(prog)
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_thread_local_links_and_runs_without_relaxation() {
    let dir = workdir("plain");
    let Some(prog) = build(&dir, false) else {
        eprintln!("skipping general-dynamic TLS test: clang unavailable");
        return;
    };
    let status = Command::new(&prog).status().expect("program runs");
    assert_eq!(
        status.code(),
        Some(0),
        "every general-dynamic access must reach the right storage"
    );
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn relaxation_does_not_change_the_outcome() {
    let dir = workdir("relax");
    let Some(prog) = build(&dir, true) else {
        eprintln!("skipping general-dynamic TLS relax test: clang unavailable");
        return;
    };
    let status = Command::new(&prog).status().expect("program runs");
    assert_eq!(status.code(), Some(0));
}

/// A general-dynamic pair whose opening `lea` is not the canonical encoding
/// must end the link, naming the thread-local.
///
/// The lowering is not an optimisation: an executable has no runtime
/// `__tls_get_addr` to fall back on, and the reference names a pair of GOT
/// slots this image never allocated. Falling through to the ordinary value
/// computation would measure a displacement from address zero and produce an
/// image that links clean and faults, which is what this pins down. lld
/// lowers the same bytes because it rewrites them without looking; xold
/// checks them, so refusing is the honest answer for a sequence it does not
/// recognise.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_pair_that_cannot_be_lowered_is_refused() {
    let dir = workdir("declined");
    let Some(obj) = assemble(&dir) else {
        eprintln!("skipping declined general-dynamic test: clang unavailable");
        return;
    };
    let prog = dir.join("prog");
    let err = link_to(
        std::slice::from_ref(&obj),
        &prog,
        b"_start",
        false,
        IcfMode::None,
        false,
    )
    .expect_err("an unlowerable general-dynamic pair must not link");
    assert!(
        matches!(&err, Error::UnresolvableTls(e) if e.symbol == "tv"),
        "the diagnostic must name the thread-local it could not resolve: {err}"
    );
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_lowered_pair_becomes_the_local_exec_form() {
    let dir = workdir("lowered");
    let Some(prog) = build(&dir, false) else {
        eprintln!("skipping general-dynamic TLS lowering test: no clang");
        return;
    };
    let out = Command::new("objdump")
        .args(["-d"])
        .arg(&prog)
        .output()
        .expect("objdump runs");
    let text = String::from_utf8_lossy(&out.stdout);
    // The pair is replaced wholesale: the thread pointer is read directly and
    // the offset folded into a `lea`, with no call left behind.
    assert!(
        text.contains("mov    %fs:0x0,%rax"),
        "the lowered form must read the thread pointer directly"
    );
    assert!(
        !text.contains("__tls_get_addr"),
        "the lowered image must not reference the runtime TLS resolver"
    );
}
