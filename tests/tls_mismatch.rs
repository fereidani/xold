//! A plain occurrence of a name a thread-local owns is refused, loudly.
//!
//! `st_value` of an `STT_TLS` symbol is an offset from the thread pointer,
//! not an address, and only the special TLS relocations are allowed to name
//! one. A plain reference -- a definition in `.data`, an ordinary
//! `R_X86_64_PC32` -- folded into the name silently resolves against that
//! offset as though it were an address, and the program reads a number the
//! code never meant. lld reports the mismatch while folding occurrences:
//! "TLS attribute mismatch", with `STT_NOTYPE` let through because
//! assembler references routinely carry no type
//! (`lld/ELF/InputFiles.cpp`).
//!
//! xold folded every occurrence without asking, so a thread-local and a
//! plain symbol of one name linked clean and produced an image that was
//! wrong at run time; the parallel fold then also discarded the error
//! after the check was added. `STT_NOTYPE` occurrences still fold
//! without complaint, exactly as in lld.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{error::Error, icf::IcfMode, linker::link_to};

mod common;

/// A plain definition folded into a thread-local name is refused.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_plain_definition_of_a_tls_name_is_refused() {
    let Some(dir) = workdir("def") else {
        return;
    };
    let Some([tls, plain]) = build(&dir) else {
        return;
    };
    let out = dir.join("prog");
    let res =
        link_to(&[tls, plain], &out, b"start", false, IcfMode::None, false);
    assert!(
        matches!(&res, Err(Error::TlsMismatch(name)) if name == "shared_name"),
        "a plain definition of a thread-local is a mismatch: {:?}",
        res.err()
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A typed plain reference is refused too: this is the case that used to
/// link clean and read the offset as an address at run time.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_typed_plain_reference_is_refused() {
    let Some(dir) = workdir("ref") else {
        return;
    };
    let Some([tls, ref_obj]) = build_reference(&dir) else {
        return;
    };
    let out = dir.join("prog");
    let res =
        link_to(&[tls, ref_obj], &out, b"start", false, IcfMode::None, false);
    assert!(
        matches!(&res, Err(Error::TlsMismatch(name)) if name == "shared_name"),
        "a plain reference to a thread-local is a mismatch: {:?}",
        res.err()
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(test: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping tls-mismatch: clang unavailable");
        return None;
    }
    // Keyed by test name as well as pid: the two tests run in parallel
    // threads of one process, and each wipes its directory on entry.
    let dir = std::env::temp_dir()
        .join(format!("xold_tlsmismatch_{test}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// The reference fixture: an `STT_OBJECT` undefined row naming the
/// thread-local. The `.type` directive survives on an undefined symbol, so
/// the row carries the type a C `extern` declaration would not.
const REF_SRC: &[u8] = b"\
.type shared_name,@object
.globl start
start:
movq shared_name(%rip), %rdi
movq $60, %rax
syscall
";

/// Builds the thread-local definition and a typed plain reference to it.
fn build_reference(dir: &Path) -> Option<[PathBuf; 2]> {
    let clang = which("clang")?;
    let tls = assemble(dir, &clang, "tls", TLS_SRC)?;
    let obj = assemble(dir, &clang, "ref", REF_SRC)?;
    Some([tls, obj])
}

/// The thread-local definition both fixtures fold against. The explicit
/// `.type` is what stamps `STT_TLS`: a bare label in `.tbss` stays
/// `STT_NOTYPE`, which the mismatch check rightly lets through.
const TLS_SRC: &[u8] = b"\
.section .tbss,\"awT\",@nobits
.globl shared_name
.type shared_name,@tls_object
shared_name:
.zero 8
.size shared_name, 8
";

/// Builds one thread-local definition and one typed plain definition of the
/// same name, each in its own object.
fn build(dir: &Path) -> Option<[PathBuf; 2]> {
    let clang = which("clang")?;
    let tls = assemble(dir, &clang, "tls", TLS_SRC)?;
    let plain = assemble(
        dir,
        &clang,
        "plain",
        b"\
.data
.globl shared_name
.type shared_name,@object
shared_name:
.quad 2
.text
.globl start
start:
xorq %rdi, %rdi
movq $60, %rax
syscall
",
    )?;
    Some([tls, plain])
}

/// Assembles one source into an object.
fn assemble(
    dir: &Path,
    clang: &Path,
    name: &str,
    src: &[u8],
) -> Option<PathBuf> {
    let src_path = dir.join(format!("{name}.s"));
    let obj = dir.join(format!("{name}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c", "-fno-pie"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    // A fixture that fails to assemble is a broken test, not a host gap:
    // skipping here once let both tests pass without exercising anything.
    assert!(built, "fixture {name} must assemble");
    Some(obj)
}
