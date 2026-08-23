//! The published image honours the process umask.
//!
//! A fixed `0o755` is not "executable permissions"; it is a decision to
//! override the one setting whose entire job is to make that decision. Under
//! `umask 077` every other linker produces `0o700`, and xold produced a
//! world-readable, world-executable image -- on a build of something the user
//! had asked the system to keep to themselves.
//!
//! The mode is `0o777` less the umask now. It is read from
//! `/proc/self/status`, because there is no way to read it through libc
//! without also setting it, and setting it -- even to put it straight back --
//! would race every other thread in the process that creates a file, which a
//! linker used as a library is not entitled to do.
//!
//! These tests drive the built binary in a shell with a chosen umask, since
//! the umask is a property of the process doing the linking.
//!
//! Gated on `clang` and `sh`; without them the tests print a note and return.

use std::{
    fs,
    os::unix::fs::PermissionsExt as _,
    path::{Path, PathBuf},
    process::Command,
};

use common::{which, xold_bin};

mod common;

const SRC: &[u8] = b"void _start(void) { }\n";

/// A restrictive umask keeps the image to its owner.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_restrictive_umask_is_honoured() {
    let Some(dir) = workdir("restrictive") else {
        return;
    };
    let Some(mode) = link_under_umask(&dir, "077", "p077") else {
        return;
    };
    assert_eq!(
        mode & 0o777,
        0o700,
        "under `umask 077` the image must be the owner's alone; {mode:o} \
         publishes it to everyone"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The usual umask gives the usual mode, so the change is about honouring the
/// setting rather than tightening everything.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_default_umask_still_gives_0755() {
    let Some(dir) = workdir("default") else {
        return;
    };
    let Some(mode) = link_under_umask(&dir, "022", "p022") else {
        return;
    };
    assert_eq!(
        mode & 0o777,
        0o755,
        "under `umask 022` the image is 0755, as it always was"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The image is executable by its owner whatever the umask withholds from
/// others -- a linker that produced a non-executable program would be no use.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_owner_can_always_execute_it() {
    let Some(dir) = workdir("exec") else {
        return;
    };
    for (mask, stem) in [("077", "e077"), ("027", "e027")] {
        let Some(mode) = link_under_umask(&dir, mask, stem) else {
            return;
        };
        assert_eq!(
            mode & 0o700,
            0o700,
            "umask {mask} leaves the owner read, write and execute"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping output-permissions {prefix}: clang unavailable");
        return None;
    }
    if which("sh").is_none() {
        eprintln!("skipping output-permissions {prefix}: no shell");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_perms_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Links the fixture in a shell running under `mask`, and returns the mode of
/// the image it produced.
fn link_under_umask(dir: &Path, mask: &str, stem: &str) -> Option<u32> {
    let obj = compile(dir)?;
    let out = dir.join(stem);
    let script =
        format!("umask {mask}; exec \"$1\" -o \"$2\" \"$3\" --entry _start");
    let ok = Command::new("sh")
        .arg("-c")
        .arg(&script)
        .arg("sh")
        .arg(xold_bin())
        .arg(&out)
        .arg(&obj)
        .status()
        .ok()?
        .success();
    assert!(ok, "the fixture must link under umask {mask}");
    Some(fs::metadata(&out).ok()?.permissions().mode())
}

/// Compiles the fixture object.
fn compile(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("u.c");
    let obj = dir.join("u.o");
    fs::write(&src, SRC).ok()?;
    Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fno-pic", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success()
        .then_some(obj)
}
