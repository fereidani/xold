//! Thread-locals in a shared object xold produces, and the sequences an
//! executable rewrites.
//!
//! A shared object cannot resolve its own thread-pointer offsets: the loader
//! decides where each module's block lands, so the general-dynamic and
//! local-dynamic models stay whole, and the GOT carries the module id and
//! in-module offset the runtime reads. An executable has no `__tls_get_addr`
//! to call, so the same sequences are rewritten to reach the thread pointer
//! directly.
//!
//! The libraries here are linked by xold and then used from a program the
//! system toolchain builds, which is the only check that matters: the two
//! sides must agree on one variable.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    icf::IcfMode,
    linker::{link_shared, link_to},
};

mod common;

/// Two thread-locals a shared object owns: one visible to its users, and one
/// private to the object, which the compiler reaches by the local-dynamic
/// model.
const LIB: &str = r"
__thread int shared_var = 111;
static __thread int hidden = 7;
int get_shared(void) { return shared_var; }
void set_shared(int v) { shared_var = v; }
int bump_hidden(void) { hidden += 1; return hidden; }
";

/// Uses the library from both sides: through its functions and directly.
const USER: &str = r#"
#include <stdio.h>
extern __thread int shared_var;
int get_shared(void);
void set_shared(int v);
int bump_hidden(void);

int main(void) {
    if (get_shared() != 111) return 1;
    set_shared(42);
    if (get_shared() != 42) return 2;
    if (shared_var != 42) return 3;
    shared_var = 55;
    if (get_shared() != 55) return 4;
    if (bump_hidden() != 8) return 5;
    if (bump_hidden() != 9) return 6;
    printf("shared=%d hidden=%d\n", get_shared(), bump_hidden());
    return 0;
}
"#;

fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_tlsshared_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Compiles `src` position-independently under `model`, plus any extra flags.
fn compile(
    dir: &Path,
    name: &str,
    src: &str,
    model: &str,
    extra: &[&str],
) -> Option<PathBuf> {
    let clang = which("clang")?;
    let file = dir.join(format!("{name}.c"));
    let obj = dir.join(format!("{name}.o"));
    fs::write(&file, src).ok()?;
    Command::new(clang)
        .args(["-c", "-O1", "-fPIC"])
        .arg(format!("-ftls-model={model}"))
        .args(extra)
        .arg("-o")
        .arg(&obj)
        .arg(&file)
        .status()
        .ok()?
        .success()
        .then_some(obj)
}

/// The C library, needed by a shared object only for `__tls_get_addr`.
fn libc_path() -> Option<PathBuf> {
    ["/usr/lib64/libc.so.6", "/lib/x86_64-linux-gnu/libc.so.6"]
        .into_iter()
        .map(PathBuf::from)
        .find(|p| p.is_file())
}

/// The dynamic relocations of `image`, as readelf prints them.
fn relocs(image: &Path) -> String {
    let Some(readelf) = which("readelf") else {
        return String::new();
    };
    let Ok(out) = Command::new(readelf).arg("-rW").arg(image).output() else {
        return String::new();
    };
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Builds the library with xold and a user program with the system toolchain,
/// then runs it. Returns `(exit code, library path)`.
fn build_and_run(prefix: &str) -> Option<(i32, PathBuf)> {
    let dir = workdir(prefix);
    let obj = compile(&dir, "lib", LIB, "global-dynamic", &[])?;
    let libc = libc_path()?;
    let so = dir.join("libtls.so");
    link_shared(
        &[obj, libc],
        &so,
        Some(b"libtls.so"),
        false,
        IcfMode::None,
        false,
    )
    .expect("a shared object with thread-locals must link");

    let clang = which("clang")?;
    let src = dir.join("user.c");
    let user = dir.join("user");
    fs::write(&src, USER).ok()?;
    let built = Command::new(clang)
        .arg(&src)
        .arg("-o")
        .arg(&user)
        .arg("-l:libtls.so")
        .arg(format!("-L{}", dir.display()))
        .arg(format!("-Wl,-rpath,{}", dir.display()))
        .status()
        .ok()?
        .success();
    if !built {
        return None;
    }
    let code = Command::new(&user)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .ok()?
        .code()?;
    Some((code, so))
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_shared_object_serves_its_thread_locals_to_a_user_program() {
    let Some((code, so)) = build_and_run("gd") else {
        eprintln!("skipping shared-TLS test: clang, libc or loader missing");
        return;
    };
    assert_eq!(
        code, 0,
        "the library and its user must see one variable per thread"
    );
    let table = relocs(&so);
    // The exported thread-local is reachable by name, so both halves of its
    // pair are the loader's to fill.
    assert!(
        table.contains("R_X86_64_DTPMOD64") && table.contains("shared_var"),
        "the exported thread-local needs a module id: {table}"
    );
    assert!(
        table.contains("R_X86_64_DTPOFF64"),
        "and an offset within that module: {table}"
    );
    // The private one is reached through the module's own entry, which names
    // no symbol at all.
    // readelf prints four columns for a relocation against no symbol, and
    // adds the value and name when there is one.
    assert!(
        table.lines().any(|l| l.contains("R_X86_64_DTPMOD64")
            && l.split_whitespace().count() == 4),
        "the module's own entry names no symbol: {table}"
    );
}

/// An executable has no runtime helper to call, so a local-dynamic sequence
/// is rewritten to read the thread pointer and the call disappears.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_executable_lowers_a_local_dynamic_sequence() {
    const SRC: &str = r#"
static __thread int a = 1;
static __thread long b = 2;
__thread int c = 3;

static volatile unsigned long tcb[16];

static void bye(long code) {
    __asm__ volatile("syscall" : : "a"(231L), "D"(code));
    __builtin_unreachable();
}

int touch(void) { a += 10; b += 20; c += 30; return a; }

void _start(void) {
    unsigned long tp = (unsigned long)&tcb[8];
    tcb[8] = tp;
    __asm__ volatile("syscall" : : "a"(158L), "D"(0x1002L), "S"(tp)
                     : "rcx", "r11", "memory");
    a = 1; b = 2; c = 3;
    if (touch() != 11) bye(1);
    if (a != 11) bye(2);
    if (b != 22) bye(3);
    if (c != 33) bye(4);
    a = 100;
    if (b != 22 || c != 33) bye(5);
    bye(0);
}
"#;
    let dir = workdir("ld");
    let Some(obj) =
        compile(&dir, "ld", SRC, "local-dynamic", &["-ffreestanding"])
    else {
        eprintln!("skipping local-dynamic test: clang unavailable");
        return;
    };
    // The fixture must carry the model under test.
    let table = relocs(&obj);
    assert!(
        table.contains("R_X86_64_TLSLD"),
        "the fixture must be local-dynamic to be a valid test"
    );
    let prog = dir.join("prog");
    link_to(&[obj], &prog, b"_start", false, IcfMode::None, false)
        .expect("an executable lowers the local-dynamic sequence");
    let status = Command::new(&prog).status().expect("program runs");
    assert_eq!(status.code(), Some(0));

    let Some(objdump) = which("objdump") else {
        return;
    };
    let out = Command::new(objdump)
        .args(["-d"])
        .arg(&prog)
        .output()
        .expect("objdump runs");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        !text.contains("__tls_get_addr"),
        "the lowered image must not reference the runtime TLS resolver"
    );
}

/// Under `-fno-plt` the general-dynamic pair reaches its helper through the
/// GOT rather than the PLT. The lowering consumes that call either way, so
/// nothing is allocated for a helper the image never calls.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_no_plt_general_dynamic_encoding_lowers_and_allocates_nothing() {
    const SRC: &str = r#"
__thread int own = 5;

static volatile unsigned long tcb[16];

static void bye(long code) {
    __asm__ volatile("syscall" : : "a"(231L), "D"(code));
    __builtin_unreachable();
}

int bump(void) { own += 1; return own; }

void _start(void) {
    unsigned long tp = (unsigned long)&tcb[8];
    tcb[8] = tp;
    __asm__ volatile("syscall" : : "a"(158L), "D"(0x1002L), "S"(tp)
                     : "rcx", "r11", "memory");
    own = 5;
    if (bump() != 6) bye(1);
    if (own != 6) bye(2);
    bye(0);
}
"#;
    let dir = workdir("noplt");
    let Some(obj) = compile(
        &dir,
        "np",
        SRC,
        "global-dynamic",
        &["-fno-plt", "-ffreestanding"],
    ) else {
        eprintln!("skipping -fno-plt test: clang unavailable");
        return;
    };
    let table = relocs(&obj);
    assert!(
        table.contains("R_X86_64_TLSGD")
            && table.contains("R_X86_64_GOTPCRELX"),
        "the fixture must use the GOT-indirect call to be a valid test"
    );

    let prog = dir.join("prog");
    link_to(&[obj], &prog, b"_start", false, IcfMode::None, false)
        .expect("the -fno-plt general-dynamic pair must lower");
    let status = Command::new(&prog).status().expect("program runs");
    assert_eq!(status.code(), Some(0), "the lowered access must reach own");

    // The helper claims no entry: the tables would otherwise carry a slot for
    // a symbol nothing reaches.
    let Some(readelf) = which("readelf") else {
        return;
    };
    let out = Command::new(readelf)
        .args(["-sW"])
        .arg(&prog)
        .output()
        .expect("readelf runs");
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("__tls_get_addr"),
        "a dropped reference must not reach the symbol table"
    );
}

/// A shared object can use the initial-exec model for a thread-local it
/// exports: the slot takes a `TPOFF64` naming the symbol, and the loader
/// fills it when it places the module.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_shared_object_may_use_initial_exec_for_what_it_exports() {
    const SRC: &str = r"
__thread int ievar = 77;
int get_ie(void) { return ievar; }
void set_ie(int v) { ievar = v; }
";
    let dir = workdir("ieshared");
    let Some(obj) = compile(&dir, "ie", SRC, "initial-exec", &[]) else {
        eprintln!("skipping shared initial-exec test: clang unavailable");
        return;
    };
    let so = dir.join("libie.so");
    link_shared(&[obj], &so, Some(b"libie.so"), false, IcfMode::None, false)
        .expect("initial-exec in a shared object must link");
    assert!(
        relocs(&so).contains("R_X86_64_TPOFF64"),
        "the offset is the loader's to fill in"
    );
}

/// The same model cannot reach a thread-local the object does not export:
/// there is no name for the loader to resolve, which is why compilers use the
/// local-dynamic model there. Refusing beats resolving it against an offset
/// that will not hold.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_shared_object_refuses_initial_exec_for_what_it_hides() {
    const SRC: &str = r"
static __thread int hidden = 1;
int bump(void) { hidden += 1; return hidden; }
";
    let dir = workdir("iehidden");
    let Some(obj) = compile(&dir, "hidden", SRC, "initial-exec", &[]) else {
        eprintln!("skipping hidden initial-exec test: clang unavailable");
        return;
    };
    let so = dir.join("libhidden.so");
    let err = link_shared(&[obj], &so, None, false, IcfMode::None, false)
        .expect_err("a hidden thread-local has no name for the loader");
    let text = err.to_string();
    assert!(
        text.contains("hidden") && text.contains("export"),
        "the diagnostic must name the symbol and the reason, got: {text}"
    );
}
