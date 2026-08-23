//! Relaxation is on unless the link says otherwise.
//!
//! `GOTPCRELX` marks a GOT load the linker may rewrite into a direct
//! reference when the symbol turns out to be reachable without one:
//! `movq sym@GOTPCREL(%rip), %reg` becomes `leaq sym(%rip), %reg`. lld, mold
//! and GNU ld all do it by default. xold defaulted it off, so the image it
//! produced carried an indirection per GOT access, and a larger RELRO GOT,
//! that no reference linker on the same inputs would have.
//!
//! This is a deliberate change of output, not a repair of a wrong one: the
//! unrelaxed images were correct, just slower and bigger. `--no-relax`
//! restores them.
//!
//! Gated on `clang` and the system crt objects; without them the tests print
//! a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{icf::IcfMode, linker::link_dyn_exec};

mod common;

/// A datum reached through the GOT and a `main` whose address the startup code
/// takes -- the shape `GOTPCRELX` marks.
const SRC: &[u8] = b"#include <stdio.h>\n\
    int shared_datum = 41;\n\
    int bump(void) { return ++shared_datum; }\n\
    int main(void)\n\
    {\n\
        printf(\"%d %d\\n\", bump(), shared_datum);\n\
        return shared_datum == 42 ? 0 : 1;\n\
    }\n";

/// The default link relaxes, and the relaxed program runs.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_default_link_relaxes() {
    let Some(dir) = workdir("default") else {
        return;
    };
    let Some(relaxed) = link(&dir, true, "relaxed") else {
        return;
    };
    let Some(plain) = link(&dir, false, "plain") else {
        return;
    };
    assert_ne!(
        relaxed, plain,
        "the fixture must contain a relaxable site for this to mean anything"
    );
    let out = Command::new(dir.join("relaxed"))
        .output()
        .expect("must run");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "42 42\n",
        "the relaxed program must compute what the unrelaxed one does"
    );
    assert_eq!(out.status.code(), Some(0), "and exit 0");
    let _ = fs::remove_dir_all(&dir);
}

/// A relaxed site is a direct reference, not a load through the GOT.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_relaxed_site_stops_reading_the_got() {
    let Some(dir) = workdir("bytes") else {
        return;
    };
    let Some(relaxed) = link(&dir, true, "relaxed") else {
        return;
    };
    let Some(plain) = link(&dir, false, "plain") else {
        return;
    };
    // `48 8b 3d` is `movq off(%rip), %rdi`; relaxation turns the two opcode
    // bytes into `48 8d 3d`, a `leaq`. The operand changes with it, so the
    // difference is small and exact.
    let loads = count(&plain, &[0x48, 0x8b, 0x3d]);
    let leas = count(&relaxed, &[0x48, 0x8d, 0x3d]);
    assert!(
        leas > count(&plain, &[0x48, 0x8d, 0x3d]),
        "relaxation must produce a direct reference where the unrelaxed image \
         has a load: {leas} against {loads}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// And the unrelaxed program still runs, so `--no-relax` is a real escape.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn no_relax_still_links_and_runs() {
    let Some(dir) = workdir("norelax") else {
        return;
    };
    let Some(_) = link(&dir, false, "plain") else {
        return;
    };
    let out = Command::new(dir.join("plain")).output().expect("must run");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42 42\n");
    assert_eq!(out.status.code(), Some(0));
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping relax-default {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_relaxdef_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds and links the fixture against the host crt objects.
fn link(dir: &Path, relax: bool, stem: &str) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let interp = interpreter()?;
    let src = dir.join("r.c");
    let obj = dir.join("r.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIE", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping relax-default: clang cannot build it");
        return None;
    }
    let paths = vec![
        crt_file("Scrt1.o")?,
        crt_file("crti.o")?,
        obj,
        libc_so()?,
        crt_file("crtn.o")?,
    ];
    let out = dir.join(stem);
    let res = link_dyn_exec(
        &paths,
        &out,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        relax,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

/// How many times `needle` occurs in `hay`.
fn count(hay: &[u8], needle: &[u8]) -> usize {
    hay.windows(needle.len()).filter(|w| *w == needle).count()
}
