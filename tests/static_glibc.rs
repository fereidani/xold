//! Linking against a static C library, which asks more of a linker than any
//! other input does.
//!
//! A static glibc has no loader behind it, so the linker owes it three things
//! the dynamic case gets for free: the symbols that bound its constructor
//! arrays and its image (`__init_array_start` and friends), the
//! `R_X86_64_IRELATIVE` table its startup walks to resolve its own indirect
//! functions, and `.init`/`.fini` kept whole so the function each is
//! assembled from returns instead of running on into the next.
//!
//! These tests link real `libc.a` and run the result. Nothing else proves the
//! image is right: every failure mode here produces a linker that reports
//! success and a program that crashes or prints nonsense.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{icf::IcfMode, linker::link_to};

mod common;

fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_static_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// The startup objects and archives a static link needs, in the order the
/// system compiler passes them. `None` on a host without a static C library.
fn static_inputs(obj: PathBuf) -> Option<Vec<PathBuf>> {
    let clang = which("clang")?;
    let libgcc = Command::new(&clang)
        .arg("-print-libgcc-file-name")
        .output()
        .ok()?;
    let gcc_dir = PathBuf::from(
        String::from_utf8_lossy(&libgcc.stdout).trim().to_owned(),
    )
    .parent()?
    .to_path_buf();
    let libc_dir = ["/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib"]
        .into_iter()
        .map(PathBuf::from)
        .find(|d| d.join("libc.a").is_file())?;
    let mut inputs = vec![
        libc_dir.join("crt1.o"),
        libc_dir.join("crti.o"),
        gcc_dir.join("crtbeginT.o"),
        obj,
        gcc_dir.join("libgcc.a"),
        gcc_dir.join("libgcc_eh.a"),
        libc_dir.join("libc.a"),
        gcc_dir.join("crtend.o"),
        libc_dir.join("crtn.o"),
    ];
    if !inputs.iter().all(|p| p.is_file()) {
        return None;
    }
    inputs.retain(|p| p.is_file());
    Some(inputs)
}

/// Compiles `src` and links it statically. Returns the image path.
fn build(dir: &Path, name: &str, src: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let file = dir.join(format!("{name}.c"));
    let obj = dir.join(format!("{name}.o"));
    fs::write(&file, src).ok()?;
    let compiled = Command::new(clang)
        .args(["-c", "-O1", "-o"])
        .arg(&obj)
        .arg(&file)
        .status()
        .ok()?
        .success();
    if !compiled {
        return None;
    }
    let inputs = static_inputs(obj)?;
    let prog = dir.join(name);
    link_to(&inputs, &prog, b"_start", false, IcfMode::None, false)
        .expect("a static link against libc.a must succeed");
    Some(prog)
}

/// Runs `prog`, returning its exit code and what it wrote.
fn run(prog: &Path) -> (Option<i32>, String) {
    let out = Command::new(prog).output().expect("program runs");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

/// The whole point: a static C program that formats, allocates and runs its
/// constructors. Each of those exercises a different thing the linker owes a
/// static C library, and all of them are silent when wrong.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_static_c_program_links_and_behaves() {
    const SRC: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ctor_ran;
__attribute__((constructor)) static void ctor(void) { ctor_ran = 1; }

int main(void) {
    /* The constructor array bounds: a wrong pair runs nothing, or garbage. */
    if (!ctor_ran) return 1;
    /* Indirect functions: `memcpy` and friends are resolved by the startup
       code walking the IRELATIVE table. */
    char buf[32];
    memset(buf, 0, sizeof buf);
    memcpy(buf, "copied", 7);
    if (strcmp(buf, "copied") != 0) return 2;
    /* The heap starts at the end of the image, which the linker names. */
    char *p = malloc(4096);
    if (!p) return 3;
    memset(p, 'x', 4096);
    free(p);
    /* Integer formatting reads a digit table defined inside a merged
       section, which is only at the address the symbol names if merging
       moved the symbol with its contents. */
    printf("ok %d %s\n", 42, "str");
    return 0;
}
"#;
    let dir = workdir("program");
    let Some(prog) = build(&dir, "program", SRC) else {
        eprintln!("skipping static test: clang or a static libc is missing");
        return;
    };
    let (code, out) = run(&prog);
    assert_eq!(code, Some(0), "static program failed: {out}");
    assert_eq!(out, "ok 42 str\n", "static program printed the wrong thing");
}

/// The bounds the C library asks for. A section that does not exist still
/// needs a pair, and the pair must be equal: the runtime walks the range
/// between them, so a start past its end walks off the image.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_linker_defines_the_symbols_a_c_library_asks_for() {
    const SRC: &str = r#"
#include <stdio.h>
extern char __executable_start[], __ehdr_start[], _end[], __bss_start[];
extern char __init_array_start[], __init_array_end[];
extern char __fini_array_start[], __fini_array_end[];
extern char __preinit_array_start[], __preinit_array_end[];

__attribute__((constructor)) static void ctor(void) { }

int main(void) {
    if (__executable_start != __ehdr_start) return 1;
    if (__bss_start > _end) return 2;
    if (__init_array_start > __init_array_end) return 3;
    if (__fini_array_start > __fini_array_end) return 4;
    /* No `.preinit_array` exists, so its range must be empty rather than
       backwards or unbounded. */
    if (__preinit_array_start != __preinit_array_end) return 5;
    /* The constructor above is in the range. */
    if (__init_array_start == __init_array_end) return 6;
    puts("bounds ok");
    return 0;
}
"#;
    let dir = workdir("bounds");
    let Some(prog) = build(&dir, "bounds", SRC) else {
        eprintln!("skipping bounds test: clang or a static libc is missing");
        return;
    };
    let (code, out) = run(&prog);
    assert_eq!(code, Some(0), "bounds check failed: {out}");
}

/// A static image has no loader, so its own startup walks the range between
/// `__preinit_array_start` and `__preinit_array_end` before it walks the
/// constructors. Both halves have to be right for this to print `pim`: the
/// section has to exist as its own region rather than folded into `.data`, and
/// the two bounds have to name it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_preinit_array_runs_before_the_constructors() {
    // `volatile` is load bearing: at `-O1` clang evaluates a constructor whose
    // effects it can predict and folds the result into the initial data, which
    // would report an order no function ever ran in.
    const SRC: &str = r#"
#include <stdio.h>

static volatile char order[8];
static volatile int at;

static void preinit(int argc, char **argv, char **envp) {
    (void)argc; (void)argv; (void)envp;
    if (at < 7) order[at++] = 'p';
}
__attribute__((section(".preinit_array"), used))
static void (*preinit_slot)(int, char **, char **) = preinit;

__attribute__((constructor)) static void ctor(void) {
    if (at < 7) order[at++] = 'i';
}

extern char __preinit_array_start[], __preinit_array_end[];

int main(void) {
    char seen[8];
    int i;
    if (at < 7) order[at++] = 'm';
    /* One entry contributed, so the bounds span exactly one pointer. */
    if (__preinit_array_end - __preinit_array_start != sizeof(void *))
        return 1;
    for (i = 0; i < 8; i++) seen[i] = order[i];
    seen[7] = '\0';
    printf("order=%s\n", seen);
    return 0;
}
"#;
    let dir = workdir("preinit");
    let Some(prog) = build(&dir, "preinit", SRC) else {
        eprintln!("skipping preinit test: clang or a static libc is missing");
        return;
    };
    let (code, out) = run(&prog);
    assert_eq!(code, Some(0), "preinit program failed: {out}");
    assert_eq!(
        out, "order=pim\n",
        "the preinit function must run before the constructor and main"
    );
}

/// `_init` is one function assembled from the C runtime's prologue and
/// epilogue, which sit in different objects. Anything placed between them
/// would run as part of it, so its fragments get an output section of their
/// own rather than joining `.text`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_init_and_fini_functions_are_contiguous() {
    let dir = workdir("init");
    let Some(prog) = build(&dir, "init", "int main(void) { return 0; }") else {
        eprintln!("skipping init test: clang or a static libc is missing");
        return;
    };
    let Some(readelf) = which("readelf") else {
        return;
    };
    let out = Command::new(readelf)
        .args(["-SW"])
        .arg(&prog)
        .output()
        .expect("readelf runs");
    let text = String::from_utf8_lossy(&out.stdout);
    for name in [" .init ", " .fini "] {
        assert!(
            text.contains(name),
            "{name} needs an output section of its own:\n{text}"
        );
    }
}

/// Naming one archive twice is how a toolchain breaks a circular dependency
/// between archives. The second copy must add nothing: every member it could
/// supply is already in.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn naming_an_archive_twice_changes_nothing() {
    let clang = which("clang");
    let dir = workdir("twice");
    let file = dir.join("twice.c");
    fs::write(
        &file,
        "#include <stdio.h>\nint main(void){ puts(\"x\"); }\n",
    )
    .expect("write source");
    let obj = dir.join("twice.o");
    let Some(clang) = clang else {
        eprintln!("skipping repeated-archive test: clang is missing");
        return;
    };
    let compiled = Command::new(clang)
        .args(["-c", "-O1", "-o"])
        .arg(&obj)
        .arg(&file)
        .status()
        .expect("clang runs")
        .success();
    assert!(compiled);
    let Some(once) = static_inputs(obj) else {
        eprintln!("skipping repeated-archive test: no static libc");
        return;
    };
    // The archives again, in the order a compiler driver repeats them.
    let mut twice = once.clone();
    let repeated: Vec<PathBuf> = once
        .iter()
        .filter(|p| p.extension().is_some_and(|e| e == "a"))
        .cloned()
        .collect();
    let tail = twice.len() - 2;
    for (i, archive) in repeated.into_iter().enumerate() {
        twice.insert(tail + i, archive);
    }

    let a = dir.join("once");
    let b = dir.join("twice");
    link_to(&once, &a, b"_start", false, IcfMode::None, false)
        .expect("one copy of each archive links");
    link_to(&twice, &b, b"_start", false, IcfMode::None, false)
        .expect("repeating an archive must not be a duplicate definition");
    assert_eq!(
        fs::read(&a).expect("read image"),
        fs::read(&b).expect("read image"),
        "repeating an archive must not change the image"
    );
}
