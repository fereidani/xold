//! A dynamic executable's reference to a name no dependency exports.
//!
//! Having a loader is not the same as the loader being able to find something.
//! `ld.so` looks a name up in the `DT_NEEDED` list the image carries, so a
//! strong reference to a name none of those libraries defines fails at
//! startup, with the program already launched. xold linked such an image
//! clean: the refusal covered a static link (no loader at all) and a hidden
//! name (no loader could resolve it), and let everything else through on the
//! strength of the loader existing.
//!
//! That is how a link over the LLVM archives succeeded here while ld, mold,
//! wild and lld each refused it -- the corpus references libffi, libedit and
//! libxml2 and named none of them. lld reports the same shape from
//! `maybeReportUndefined`, whose policy is "ignore" only for `-shared`.
//!
//! The three references below are the whole of the rule: one that no
//! dependency can satisfy is refused, one that a dependency exports is not,
//! and a weak one is exempt because resolving to zero is the answer the
//! program tests for.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{
    error::Error,
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
};

mod common;

/// A call to a name nothing in the link and no dependency defines.
const MISSING: &[u8] = b"extern int nowhere_at_all(int);\n\
    int main(void) { return nowhere_at_all(3); }\n";

/// The same shape against a name libc exports.
const PRESENT: &[u8] = b"extern int abs(int);\n\
    int main(void) { return abs(-3); }\n";

/// A weak reference is a probe: the program tests the address for null, so
/// resolving it to zero is the answer, not an error.
const WEAK: &[u8] = b"extern int nowhere_at_all(int) __attribute__((weak));\n\
    int main(void) { return nowhere_at_all ? nowhere_at_all(3) : 0; }\n";

const START: &[u8] = b".globl _start\n_start:\n\
    call main\n movl %eax, %edi\n movl $60, %eax\n syscall\n";

const ABS_DEP: &[u8] = b"int abs(int x) { return x < 0 ? -x : x; }\n";

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_reference_no_dependency_exports_is_refused() {
    let Some(dir) = workdir("missing") else {
        return;
    };
    let Some(err) = link(&dir, MISSING).err() else {
        panic!("a reference nothing can satisfy must not link");
    };
    match err {
        Error::UndefinedReference(name) => {
            assert_eq!(name, "nowhere_at_all", "the diagnostic names it");
        }
        other => panic!("expected an undefined reference, got {other:?}"),
    }
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_reference_a_dependency_exports_still_links() {
    let Some(dir) = workdir("present") else {
        return;
    };
    assert!(
        link(&dir, PRESENT).is_ok(),
        "a name libc exports is what the loader is for"
    );
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_weak_reference_is_exempt() {
    let Some(dir) = workdir("weak") else {
        return;
    };
    assert!(
        link(&dir, WEAK).is_ok(),
        "a weak reference resolves to zero rather than failing the link"
    );
}

// --- fixtures --------------------------------------------------------------

/// Compiles `src` and links it as a dynamic executable against libc alone.
fn link(dir: &Path, src: &[u8]) -> Result<(), Error> {
    let obj = dir.join("u.o");
    compile(src, &obj)
        .ok_or(Error::Format("clang could not build the input"))?;
    if !cfg!(target_os = "linux") {
        let start = dir.join("start.o");
        assemble(START, &start)
            .ok_or(Error::Format("clang could not build _start"))?;
        let dep_obj = dir.join("abs.o");
        compile_pic(ABS_DEP, &dep_obj)
            .ok_or(Error::Format("clang could not build dependency"))?;
        let dep = dir.join("libc-standin.so");
        link_shared(
            &[dep_obj],
            &dep,
            Some(b"libc-standin.so"),
            false,
            IcfMode::None,
            false,
        )?;
        return link_dyn_exec(
            &[obj, start, dep],
            &dir.join("prog"),
            b"_start",
            b"/lib64/ld-linux-x86-64.so.2",
            false,
            IcfMode::None,
            false,
        );
    }
    let inputs = vec![
        crt_file("Scrt1.o").ok_or(Error::Format("no Scrt1.o"))?,
        crt_file("crti.o").ok_or(Error::Format("no crti.o"))?,
        obj,
        libc_so().ok_or(Error::Format("no libc.so.6"))?,
        crt_file("crtn.o").ok_or(Error::Format("no crtn.o"))?,
    ];
    let interp = interpreter().ok_or(Error::Format("no interpreter"))?;
    link_dyn_exec(
        &inputs,
        &dir.join("prog"),
        b"_start",
        &interp,
        false,
        IcfMode::None,
        false,
    )
}

/// Compiles `src` to `obj` with the host `clang`.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    compile_c(src, obj, "-fPIE")
}

fn compile_pic(src: &[u8], obj: &Path) -> Option<()> {
    compile_c(src, obj, "-fPIC")
}

fn compile_c(src: &[u8], obj: &Path, model: &str) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).ok()?;
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", model, "-O1", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

fn assemble(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("S");
    fs::write(&src_path, src).ok()?;
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// A fresh per-test working directory, or `None` (after a note) when the host
/// cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping undefined-import {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_undefimport_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}
