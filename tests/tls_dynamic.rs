//! Thread-locals a shared object owns, and the initial-exec model.
//!
//! A thread-local the image places itself has an offset from the thread
//! pointer that this link fixes, so a reference to it is resolved here. One a
//! shared object defines does not: the loader chooses where the module's
//! block lands, so the reference has to go through a GOT slot the loader
//! fills with an `R_X86_64_TPOFF64` relocation. Getting that wrong is not a
//! link failure but a program that reads the wrong storage, which is why
//! these tests run the images rather than only inspecting them.
//!
//! The programs are freestanding apart from the library under test: they end
//! in `exit_group` rather than returning, so no C runtime is involved.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    icf::IcfMode,
    linker::{link_dyn_exec, link_to},
};

mod common;

/// The library whose thread-local the programs below reach for.
const LIB: &str = r"
__thread int libvar = 111;
int get_libvar(void) { return libvar; }
void set_libvar(int v) { libvar = v; }
";

/// Reads and writes a thread-local of the library, checking that both sides
/// see one variable. Exits 0 only if every access reached the right storage.
const USER: &str = r#"
extern __thread int libvar;
int get_libvar(void);
void set_libvar(int v);

static void bye(long code) {
    __asm__ volatile("syscall" : : "a"(231L), "D"(code));
    __builtin_unreachable();
}

void _start(void) {
    if (libvar != 111) bye(1);
    if (get_libvar() != 111) bye(2);
    libvar = 42;
    if (get_libvar() != 42) bye(3);
    set_libvar(7);
    if (libvar != 7) bye(4);
    bye(0);
}
"#;

fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_tlsdyn_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Compiles `src` to an object under `model`, which selects the TLS access
/// sequence the compiler emits.
fn compile(dir: &Path, name: &str, src: &str, model: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let file = dir.join(format!("{name}.c"));
    let obj = dir.join(format!("{name}.o"));
    fs::write(&file, src).ok()?;
    Command::new(clang)
        .args([
            "-c",
            "-O1",
            "-fPIC",
            "-ffreestanding",
            &format!("-ftls-model={model}"),
            "-o",
        ])
        .arg(&obj)
        .arg(&file)
        .status()
        .ok()?
        .success()
        .then_some(obj)
}

/// Builds the shared library holding `libvar`, with the system toolchain: the
/// point is to link against a real dependency, not one xold produced.
fn build_library(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let file = dir.join("lib.c");
    let lib = dir.join("libtls.so");
    fs::write(&file, LIB).ok()?;
    Command::new(clang)
        .args(["-shared", "-fPIC", "-o"])
        .arg(&lib)
        .arg(&file)
        .status()
        .ok()?
        .success()
        .then_some(lib)
}

/// The host's dynamic loader, or `None` on a system without the usual one.
fn interpreter() -> Option<&'static str> {
    ["/lib64/ld-linux-x86-64.so.2", "/lib/ld-linux-x86-64.so.2"]
        .into_iter()
        .find(|p| Path::new(p).is_file())
}

/// Runs `prog` with `dir` on the library search path, returning its exit code.
fn run(prog: &Path, dir: &Path) -> Option<i32> {
    Command::new(prog)
        .env("LD_LIBRARY_PATH", dir)
        .status()
        .ok()?
        .code()
}

/// Whether the image carries a dynamic relocation of the named type.
fn has_reloc(prog: &Path, name: &str) -> bool {
    let Some(readelf) = which("readelf") else {
        return false;
    };
    let Ok(out) = Command::new(readelf).arg("-rW").arg(prog).output() else {
        return false;
    };
    String::from_utf8_lossy(&out.stdout).contains(name)
}

/// Links a program that reaches for the library's thread-local under `model`,
/// runs it, and returns `(exit code, image path)`.
fn link_and_run(prefix: &str, model: &str) -> Option<(i32, PathBuf, PathBuf)> {
    let dir = workdir(prefix);
    let lib = build_library(&dir)?;
    let obj = compile(&dir, "user", USER, model)?;
    let interp = interpreter()?;
    let prog = dir.join("prog");
    link_dyn_exec(
        &[obj, lib],
        &prog,
        b"_start",
        interp.as_bytes(),
        false,
        IcfMode::None,
        false,
    )
    .expect("a program using a shared object's thread-local must link");
    let code = run(&prog, &dir)?;
    Some((code, prog, dir))
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_general_dynamic_reference_to_an_imported_thread_local_runs() {
    let Some((code, prog, _dir)) = link_and_run("gd", "global-dynamic") else {
        eprintln!("skipping imported-TLS test: clang or loader unavailable");
        return;
    };
    assert_eq!(code, 0, "every access must reach the library's own storage");
    assert!(
        has_reloc(&prog, "R_X86_64_TPOFF64"),
        "the offset is the loader's to fill in, so it takes a TPOFF64 entry"
    );
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_initial_exec_reference_to_an_imported_thread_local_runs() {
    let Some((code, prog, _dir)) = link_and_run("ie", "initial-exec") else {
        eprintln!("skipping imported-TLS test: clang or loader unavailable");
        return;
    };
    assert_eq!(code, 0);
    assert!(has_reloc(&prog, "R_X86_64_TPOFF64"));
}

/// A thread-local this image places needs no help from the loader: the offset
/// is known here, so the GOT slot holds it outright and no dynamic relocation
/// is emitted for it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_initial_exec_reference_to_an_own_thread_local_needs_no_loader() {
    const OWN: &str = r#"
__thread int own = 5;
static __thread int hidden = 9;

static volatile unsigned long tcb[16];

static void bye(long code) {
    __asm__ volatile("syscall" : : "a"(231L), "D"(code));
    __builtin_unreachable();
}

void _start(void) {
    /* The thread block is made by hand here, and nothing copies the initial
       images into it, so the program writes before it reads. */
    unsigned long tp = (unsigned long)&tcb[8];
    tcb[8] = tp;
    __asm__ volatile("syscall" : : "a"(158L), "D"(0x1002L), "S"(tp)
                     : "rcx", "r11", "memory");
    own = 5;
    hidden = 9;
    if (own != 5) bye(1);
    if (hidden != 9) bye(2);
    own = 21;
    hidden = 22;
    if (own != 21 || hidden != 22) bye(3);
    bye(0);
}
"#;
    let dir = workdir("own");
    let Some(obj) = compile(&dir, "own", OWN, "initial-exec") else {
        eprintln!("skipping own-TLS test: clang unavailable");
        return;
    };
    let prog = dir.join("prog");
    link_to(&[obj], &prog, b"_start", false, IcfMode::None, false)
        .expect("an initial-exec reference to an own thread-local must link");
    let status = Command::new(&prog).status().expect("program runs");
    assert_eq!(status.code(), Some(0));
    assert!(
        !has_reloc(&prog, "R_X86_64"),
        "the offset is fixed by this link, so nothing is left for a loader"
    );
}

/// A static link has no loader, so a thread-local it does not define itself
/// cannot be resolved at all.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_static_link_refuses_an_imported_thread_local() {
    let dir = workdir("static");
    let Some(obj) = compile(&dir, "user", USER, "global-dynamic") else {
        eprintln!("skipping static-TLS test: clang unavailable");
        return;
    };
    let out = dir.join("prog");
    let err = link_to(&[obj], &out, b"_start", false, IcfMode::None, false)
        .expect_err("a static link cannot resolve an imported thread-local");
    assert!(err.to_string().contains("libvar"));
}
