//! A relaxed GOT reference costs no GOT slot.
//!
//! Relaxation rewrites `movq sym@GOTPCREL(%rip), %reg` into
//! `leaq sym(%rip), %reg`: the slot the load would have read is never looked
//! at again. Allocation happens in the scan, which ran before anyone knew the
//! rewrite would apply, so every relaxed site left a filled and unread slot
//! behind -- in the RELRO GOT, which is address space and a page the loader
//! write-protects.
//!
//! The scan predicts the rewrite now, the way mold does: the canonical `-4`
//! addend, an instruction form the rewrite covers, and a symbol that is
//! neither preemptible nor a constant. The one thing it cannot settle is
//! whether the relaxed displacement fits in 32 bits, which needs addresses the
//! layout has not assigned -- but an image where it does not fit is one whose
//! ordinary PC-relative references do not fit either.
//!
//! Gated on `clang` and the system crt objects; without them the tests print a
//! note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_dyn_exec};

mod common;

/// A datum and a function address, both reached through the GOT at `-fPIE`.
const SRC: &[u8] = b"#include <stdio.h>\n\
    int shared_datum = 41;\n\
    int bump(void) { return ++shared_datum; }\n\
    int main(void)\n\
    {\n\
        printf(\"%d %d\\n\", bump(), shared_datum);\n\
        return shared_datum == 42 ? 0 : 1;\n\
    }\n";

/// Relaxing shrinks the GOT.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_relaxed_reference_reserves_no_slot() {
    let Some(dir) = workdir("size") else {
        return;
    };
    let Some(relaxed) = link(&dir, true, "relaxed") else {
        return;
    };
    let Some(plain) = link(&dir, false, "plain") else {
        return;
    };
    let want = got_size(&plain);
    let got = got_size(&relaxed);
    assert!(
        want > 0,
        "the fixture must have a GOT without relaxation for this to mean \
         anything"
    );
    assert!(
        got < want,
        "a relaxed site must give its slot back: {got:#x} against {want:#x}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// And the program still runs, so nothing that was reclaimed was needed.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_program_still_runs() {
    let Some(dir) = workdir("run") else {
        return;
    };
    let Some(_) = link(&dir, true, "relaxed") else {
        return;
    };
    let out = Command::new(dir.join("relaxed"))
        .output()
        .expect("must run");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "42 42\n");
    assert_eq!(out.status.code(), Some(0), "and it exits 0");
    let _ = fs::remove_dir_all(&dir);
}

/// `--no-relax` keeps every slot, so the reclamation follows the rewrite
/// rather than happening on its own.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn no_relax_keeps_its_slots() {
    let Some(dir) = workdir("norelax") else {
        return;
    };
    let Some(plain) = link(&dir, false, "plain") else {
        return;
    };
    assert!(
        got_size(&plain) > 0,
        "without the rewrite there is nothing to reclaim, and the slot stays"
    );
    let out = Command::new(dir.join("plain")).output().expect("must run");
    assert_eq!(out.status.code(), Some(0));
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping got-reclaim {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_gotreclaim_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds and links the fixture against the host crt objects.
fn link(dir: &Path, relax: bool, stem: &str) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let interp = interpreter()?;
    let src = dir.join("g.c");
    let obj = dir.join("g.o");
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
        eprintln!("skipping got-reclaim: clang cannot build it");
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

// --- readers ---------------------------------------------------------------

/// The `sh_size` of `.got`.
fn got_size(bytes: &[u8]) -> u64 {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return 0;
    };
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == b".got")
        .map_or(0, |s| s.sh_size.get())
}
