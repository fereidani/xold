//! A weak reference the loader can still bind calls through its PLT entry.
//!
//! A weak reference nothing defines is the feature-probe idiom, and the
//! undefined-weak rules answer it: the branch goes to the next
//! instruction, the site takes its own address, or the call collapses
//! to `A - P`. But a weak undefined in a dynamic link is not nothing's
//! -- the loader may bind it to a definition in a `DT_NEEDED` library,
//! which is exactly what `crtbegin`'s guarded `__cxa_finalize` call
//! does against a libc it may find. A PLT entry exists for that call,
//! and the site must target the entry so the binding can happen: lld's
//! `R_PLT_PC` computes `PLT[sym] + A - P` without ever asking whether
//! the symbol is an undefined weak
//! (`lld/ELF/InputSection.cpp`); only the entry-less
//! site is demoted to `R_PC`, where the architecture's answer applies.
//!
//! xold's `PLT32` funnel consulted the undefined-weak answer first, so
//! a dynamic link wrote the absent-symbol target over a call whose PLT
//! entry the loader was about to fill -- a static-libc link through
//! `--dynamic-exec` segfaulted in `__do_global_dtors_aux` on exactly
//! this. The entry now decides: through the stub when one exists, the
//! architecture's answer only when none does.
//!
//! The fixture links a program whose weak `probe` is defined by a
//! shared library beside it, and runs the image: the guarded call
//! reaches the definition -- through the PLT entry -- and the exit
//! status carries the answer back. Before the fix the guard read the
//! address as null-shaped and the program took the miss path, exiting 3.
//!
//! Gated on `clang` and a readable system interpreter; if either is
//! missing the test prints a note and returns.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{icf::IcfMode, linker::link_dyn_exec};

mod common;

/// The definition the weak reference finds in the shared library.
const LIB_SRC: &[u8] = b"long probe(void) { return 7; }\n";

/// The program: a weak reference, guarded the way a feature probe is,
/// with the answer carried out in the exit status.
const MAIN_SRC: &[u8] = b"\
__attribute__((weak)) extern long probe(void);

void _start(void) {
    long r = probe ? probe() : 3;
    __asm__ volatile (\"syscall\"
                      :
                      : \"a\"(60L), \"D\"(r)
                      : \"memory\", \"rcx\", \"r11\");
    __builtin_unreachable();
}
";

/// The guarded call reaches the definition through its PLT entry.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_weak_reference_with_a_plt_entry_calls_through_it() {
    let Some(dir) = workdir() else {
        return;
    };
    let Some((lib, main)) = build(&dir) else {
        return;
    };
    let Some(interp) = interpreter() else {
        eprintln!("skipping weak-plt: no system interpreter found");
        return;
    };
    let out = dir.join("prog");
    let inputs = [main, lib];
    let res = link_dyn_exec(
        &inputs,
        &out,
        b"_start",
        &interp,
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let run = Command::new(&out)
        .env("LD_LIBRARY_PATH", &dir)
        .output()
        .expect("the image must run");
    assert_eq!(
        run.status.code(),
        Some(7),
        "the weak reference binds to the library's definition"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir() -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping weak-plt: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_weakplt_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the shared library and the program that probes it.
fn build(dir: &Path) -> Option<(PathBuf, PathBuf)> {
    let clang = which("clang")?;
    let lib_src = dir.join("lib.c");
    let lib = dir.join("libprobe.so");
    fs::write(&lib_src, LIB_SRC).ok()?;
    let built = Command::new(&clang)
        .args(["--target=x86_64-linux-gnu", "-shared", "-fPIC"])
        .arg(&lib_src)
        .arg("-o")
        .arg(&lib)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping weak-plt: clang cannot build the library");
        return None;
    }
    let main_src = dir.join("main.c");
    let main = dir.join("main.o");
    fs::write(&main_src, MAIN_SRC).ok()?;
    let built = Command::new(&clang)
        .args(["--target=x86_64-linux-gnu", "-c", "-fPIE"])
        .arg(&main_src)
        .arg("-o")
        .arg(&main)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping weak-plt: clang cannot build the program");
        return None;
    }
    Some((lib, main))
}
