//! TLS descriptor lowering (`-mtls-dialect=gnu2`).
//!
//! A descriptor sequence is the general-dynamic model told with two
//! instructions: `lea x@tlsdesc(%rip), %rax` names a descriptor the loader
//! fills with a resolver function and its argument, and `call *x@tlscall(%rax)`
//! jumps through it, leaving the offset from the thread pointer behind. It is
//! what clang emits by default on several distributions, so an object built
//! with the system compiler can carry it without anyone asking for it.
//!
//! An executable resolves neither half at run time -- xold writes no descriptor
//! and no `R_X86_64_TLSDESC` relocation for a loader to fill one from -- so
//! both instructions are lowered, exactly as the general-dynamic pair is. That
//! makes this a lowering rather than an optimisation, which is why these tests
//! link *without* `--relax`.
//!
//! The programs are freestanding, so `_start` seeds the thread pointer with
//! `arch_prctl` before touching a thread-local; in a hosted program that is the
//! C runtime's job.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{
    Error,
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared, link_to},
};

mod common;

fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_tlsdesc_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Thread-locals reached through descriptor sequences, with the thread pointer
/// set up by hand. Exits 0 only if every access reached the right storage.
///
/// `local` is `static`, so the compiler reaches it through
/// `_TLS_MODULE_BASE_` -- the descriptor spelling of the local-dynamic model,
/// where one descriptor supplies the module's block and each variable sits at a
/// fixed offset from it. `counter` and `wide` are exported, so each gets a
/// descriptor of its own.
const SRC: &str = r#"
_Thread_local int counter = 7;
_Thread_local long wide = 100;
static _Thread_local int local = 3;

static volatile unsigned long tcb[16];

static long set_fs(unsigned long a) {
    long r;
    __asm__ volatile("syscall" : "=a"(r) : "0"(158L), "D"(0x1002L), "S"(a)
                     : "rcx", "r11", "memory");
    return r;
}

int bump(void) { counter += 1; return counter; }
long widen(void) { wide += 5; return wide; }
int nudge(void) { local += 2; return local; }

static int check(void) {
    counter = 7;
    wide = 100;
    local = 3;
    if (bump() != 8) return 1;
    if (widen() != 105) return 2;
    if (nudge() != 5) return 3;
    if (counter != 8) return 4;
    /* The address must be stable across accesses. */
    int *p = &counter;
    *p = 42;
    if (counter != 42) return 5;
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

/// Compiles `SRC` with the descriptor dialect, or `None` when the host
/// compiler cannot produce one.
fn compile(obj: &Path, dir: &Path, pic: bool) -> Option<()> {
    let clang = which("clang")?;
    let file = dir.join("tlsdesc.c");
    fs::write(&file, SRC).ok()?;
    let mut cmd = Command::new(clang);
    cmd.args(["-c", "-O1", "-mtls-dialect=gnu2", "-ffreestanding"]);
    if pic {
        cmd.arg("-fPIC");
    }
    cmd.arg("-o").arg(obj).arg(&file);
    cmd.status().ok()?.success().then_some(())
}

/// Compiles the fixture and checks it carries what the test is about, so a
/// compiler that quietly chose another TLS dialect skips instead of passing.
fn fixture(dir: &Path, pic: bool) -> Option<PathBuf> {
    let obj = dir.join("tlsdesc.o");
    compile(&obj, dir, pic)?;
    let relocs = Command::new("readelf")
        .args(["-rW"])
        .arg(&obj)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&relocs.stdout).into_owned();
    if !text.contains("R_X86_64_GOTPC32_TLSDESC")
        || !text.contains("R_X86_64_TLSDESC_CALL")
    {
        return None;
    }
    Some(obj)
}

/// A descriptor reference whose `lea` is not the canonical encoding: the
/// opcode is `mov` where the sequence calls for `lea`. Everything else is
/// canonical, including the `-4` addend, so only the bytes are wrong.
const DECLINED_SRC: &str = r#"
        .section .tdata,"awT",@progbits
        .globl  tv
tv:     .long   42

        .text
        .globl  _start
_start:
        .byte   0x48, 0x8b, 0x05
        .reloc  ., R_X86_64_GOTPC32_TLSDESC, tv-4
        .long   0
        movl    $60, %eax
        xorl    %edi, %edi
        syscall
"#;

/// Assembles [`DECLINED_SRC`], or `None` without clang.
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

/// Links the fixture into an executable, or `None` when the host cannot build
/// one to link.
fn build(dir: &Path, relax: bool) -> Option<PathBuf> {
    let obj = fixture(dir, true)?;
    let prog = dir.join(if relax { "prog_relax" } else { "prog" });
    link_to(
        std::slice::from_ref(&obj),
        &prog,
        b"_start",
        false,
        IcfMode::None,
        relax,
    )
    .expect("a program with TLS descriptors must link");
    Some(prog)
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn descriptors_link_and_run_without_relaxation() {
    let dir = workdir("plain");
    let Some(prog) = build(&dir, false) else {
        eprintln!("skipping TLS descriptor test: no clang with gnu2 dialect");
        return;
    };
    let status = Command::new(&prog).status().expect("program runs");
    assert_eq!(
        status.code(),
        Some(0),
        "every descriptor access must reach the right storage"
    );
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn relaxation_does_not_change_the_outcome() {
    let dir = workdir("relax");
    let Some(prog) = build(&dir, true) else {
        eprintln!("skipping TLS descriptor relax test: no clang with gnu2");
        return;
    };
    let status = Command::new(&prog).status().expect("program runs");
    assert_eq!(status.code(), Some(0));
}

/// The lowered form reads the thread pointer directly and calls nothing: the
/// `lea` becomes a `mov` of a constant and the call through the resolver
/// becomes the two-byte `nop`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_lowered_sequence_calls_no_resolver() {
    let dir = workdir("lowered");
    let Some(prog) = build(&dir, false) else {
        eprintln!("skipping TLS descriptor lowering test: no clang with gnu2");
        return;
    };
    let out = Command::new("objdump")
        .args(["-d"])
        .arg(&prog)
        .output()
        .expect("objdump runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("xchg   %ax,%ax"),
        "the call through the descriptor must become a nop"
    );
    assert!(
        !text.contains("call   *(%rax)"),
        "no call through a descriptor may survive into the image"
    );
}

/// A descriptor reference whose bytes are not the canonical `lea` must end the
/// link, naming the thread-local.
///
/// The lowering is the only way the site can be resolved: the image holds no
/// descriptor, so falling through to the ordinary value computation would
/// measure a displacement from storage nothing allocated and produce an image
/// that links clean and faults.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_sequence_that_cannot_be_lowered_is_refused() {
    let dir = workdir("declined");
    let Some(obj) = assemble(&dir) else {
        eprintln!("skipping declined TLS descriptor test: clang unavailable");
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
    .expect_err("an unlowerable descriptor reference must not link");
    assert!(
        matches!(&err, Error::UnresolvableTls(e) if e.symbol == "tv"),
        "the diagnostic must name the thread-local it could not resolve: {err}"
    );
}

/// A thread-local a shared object owns, reached from a program that was built
/// with the descriptor dialect. The program is freestanding, but its image has
/// an interpreter, so the loader has placed both modules' blocks and seeded the
/// thread pointer before `_start` runs.
const DEP: &str = r"
__thread int dep_var = 21;
int dep_touch(void) { return dep_var; }
";

const DEP_USER: &str = r#"
extern __thread int dep_var;

void _start(void) {
    dep_var += 4;
    int rc = dep_var == 25 ? 0 : 1;
    __asm__ volatile("syscall" : : "a"(60L), "D"((long)rc));
    __builtin_unreachable();
}
"#;

/// Compiles one source with the given flags, or `None` without clang.
fn compile_src(
    dir: &Path,
    name: &str,
    src: &str,
    extra: &[&str],
) -> Option<PathBuf> {
    let clang = which("clang")?;
    let file = dir.join(format!("{name}.c"));
    let obj = dir.join(format!("{name}.o"));
    fs::write(&file, src).ok()?;
    Command::new(clang)
        .args(["-c", "-O1", "-fPIC"])
        .args(extra)
        .arg("-o")
        .arg(&obj)
        .arg(&file)
        .status()
        .ok()?
        .success()
        .then_some(obj)
}

/// The C library, which a shared object with thread-locals needs for
/// `__tls_get_addr`.
fn libc_path() -> Option<PathBuf> {
    ["/usr/lib64/libc.so.6", "/lib/x86_64-linux-gnu/libc.so.6"]
        .into_iter()
        .map(PathBuf::from)
        .find(|p| p.is_file())
}

/// A descriptor reference to a thread-local this image does not place becomes
/// the initial-exec form, not the local-exec one: the offset is the loader's to
/// choose, so the `lea` becomes a load from the GOT slot it will fill rather
/// than a constant this link made up.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_dependency_s_thread_local_lowers_to_initial_exec() {
    let dir = workdir("initexec");
    let (Some(dep), Some(libc), Some(interp)) = (
        compile_src(&dir, "dep", DEP, &["-ftls-model=global-dynamic"]),
        libc_path(),
        interpreter(),
    ) else {
        eprintln!("skipping descriptor initial-exec test: host inputs missing");
        return;
    };
    let so = dir.join("libdep.so");
    link_shared(
        &[dep, libc.clone()],
        &so,
        Some(b"libdep.so"),
        false,
        IcfMode::None,
        false,
    )
    .expect("a shared object with a thread-local must link");
    let Some(user) = compile_src(
        &dir,
        "user",
        DEP_USER,
        &["-mtls-dialect=gnu2", "-ffreestanding"],
    ) else {
        eprintln!("skipping descriptor initial-exec test: no gnu2 dialect");
        return;
    };
    let prog = dir.join("prog");
    link_dyn_exec(
        &[user, so, libc],
        &prog,
        b"_start",
        &interp,
        false,
        IcfMode::None,
        false,
    )
    .expect("a descriptor reference to a dependency's thread-local must link");

    let out = Command::new("objdump")
        .args(["-d"])
        .arg(&prog)
        .output()
        .expect("objdump runs");
    // The instruction ahead of the nopped call is what separates the two
    // lowerings: a GOT load for initial-exec, where the local-exec form would
    // have folded a constant this link is in no position to choose.
    let text = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = text.lines().collect();
    let call = lines
        .iter()
        .position(|line| line.contains("xchg   %ax,%ax"))
        .expect("the call through the descriptor must become a nop");
    let opened = lines[call.saturating_sub(1)];
    assert!(
        opened.contains("mov") && opened.contains("(%rip)"),
        "the descriptor must be lowered to a load of the loader's offset, \
         not to a constant: {opened}"
    );

    let status = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("program runs");
    assert_eq!(
        status.code(),
        Some(0),
        "the descriptor must reach the dependency's own storage"
    );
}

/// A shared object cannot keep the descriptor model, and must say so.
///
/// Filling a descriptor takes an `R_X86_64_TLSDESC` dynamic relocation this
/// linker does not emit, and lowering is not open to a shared object either:
/// its thread-locals are placed by whoever loads it. Refusing is the honest
/// answer, and the message names the thread-local and the dialect to rebuild
/// with.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_shared_object_refuses_the_descriptor_model() {
    let dir = workdir("shared");
    let Some(obj) = fixture(&dir, true) else {
        eprintln!("skipping shared TLS descriptor test: no clang with gnu2");
        return;
    };
    let out = dir.join("libtlsdesc.so");
    let err = link_shared(
        std::slice::from_ref(&obj),
        &out,
        None,
        false,
        IcfMode::None,
        false,
    )
    .expect_err("a shared object must not keep TLS descriptors");
    assert!(
        matches!(&err, Error::UnresolvableTls(_)),
        "the refusal must name the thread-local it could not place: {err}"
    );
}
