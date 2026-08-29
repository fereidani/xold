//! Two objects may define the same `SHN_ABS` symbol with the same value.
//!
//! An absolute symbol carries its value in `st_value` and belongs to no
//! section, so two files that agree on the value are not in conflict:
//! whichever row the link picks, every reference resolves to the same
//! number. lld carves exactly this case out of its duplicate report --
//! `reportDuplicate` returns early when the existing definition has no
//! section, the incoming one has none either, and the values match
//! (`lld/ELF/Symbols.cpp`) -- for GNU ld compatibility,
//! because assembler sources routinely stamp the same constant twice.
//!
//! xold errored on every pair of strong definitions from distinct files,
//! absolute or not, so a program whose objects agreed on a constant was
//! refused. Disagreeing values are still an error: that is a genuine
//! contradiction, and the fix keeps it one.
//!
//! On arm64 macOS the agreement case is linked as a native dynamic Mach-O
//! image and executed; other hosts retain the original `x86_64` ELF fixture.
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
// Only the arm64 Mach-O path drives the built binary; elsewhere the check
// links in-process through `link_to`.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use common::xold_bin;
use xold::{icf::IcfMode, linker::link_to};

mod common;

/// Two objects that agree on the absolute value of one name link.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn same_value_absolute_definitions_link() {
    same_value_link();
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn same_value_link() {
    let Some(dir) = workdir("same") else {
        return;
    };
    let Some([a, b]) = build_elf(&dir, "42", "42") else {
        return;
    };
    let out = dir.join("prog");
    let res = link_to(&[a, b], &out, b"start", false, IcfMode::None, false);
    assert!(res.is_ok(), "agreement is not a conflict: {:?}", res.err());
    assert_eq!(read_answer(&out), 42, "the value both files stated");
    let _ = fs::remove_dir_all(&dir);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn same_value_link() {
    let Some(dir) = workdir("same") else {
        return;
    };
    let Some([a, b]) = build_macho(&dir, "42", "42") else {
        return;
    };
    let out = dir.join("prog");
    let linked = Command::new(xold_bin())
        .args([a.as_os_str(), b.as_os_str()])
        .args(["-o"])
        .arg(&out)
        .args(["-arch", "arm64", "-dynamic"])
        .status()
        .expect("run xold")
        .success();
    assert!(linked, "agreement is not a conflict");
    assert_eq!(read_answer(&out), 42, "the value both files stated");
    let _ = fs::remove_dir_all(&dir);
}

/// Two objects that disagree remain the error they always were.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn different_values_still_collide() {
    let Some(dir) = workdir("clash") else {
        return;
    };
    let Some([a, b]) = build_elf(&dir, "42", "43") else {
        return;
    };
    let out = dir.join("prog");
    let res = link_to(&[a, b], &out, b"start", false, IcfMode::None, false);
    assert!(
        matches!(res, Err(xold::error::Error::DuplicateSymbol(_))),
        "a contradiction stays an error: {:?}",
        res.err()
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping abs-duplicate {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_absdup_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds two objects that each define `answer` as `SHN_ABS` with the given
/// value, plus the tiny entry point the link needs.
fn build_elf(dir: &Path, first: &str, second: &str) -> Option<[PathBuf; 2]> {
    let clang = which("clang")?;
    let a = assemble(
        dir,
        &clang,
        "a",
        b"\
.globl answer
answer = ~
.globl start
start:
    movq $~, %rdi
    movq $60, %rax
    syscall
",
        first,
        "--target=x86_64-linux-gnu",
        &["-fno-pie"],
    )?;
    let b = assemble(
        dir,
        &clang,
        "b",
        b"\
.globl answer
answer = ~
",
        second,
        "--target=x86_64-linux-gnu",
        &["-fno-pie"],
    )?;
    Some([a, b])
}

/// Builds the native arm64 Mach-O form of the agreement fixture. `_main`
/// exits through the Darwin syscall ABI so no C runtime or dylib is needed.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn build_macho(dir: &Path, first: &str, second: &str) -> Option<[PathBuf; 2]> {
    let clang = which("clang")?;
    let a = assemble(
        dir,
        &clang,
        "a",
        b"\
.globl _answer
_answer = ~
.globl _main
.p2align 2
_main:
    mov x0, #_answer
    mov x16, #1
    svc #0x80
",
        first,
        "--target=arm64-apple-darwin",
        &[],
    )?;
    let b = assemble(
        dir,
        &clang,
        "b",
        b"\
.globl _answer
_answer = ~
",
        second,
        "--target=arm64-apple-darwin",
        &[],
    )?;
    Some([a, b])
}

/// Assembles one source with every `~` replaced by `value`. The marker is
/// `~` because no other byte of the template collides with it.
fn assemble(
    dir: &Path,
    clang: &Path,
    name: &str,
    template: &[u8],
    value: &str,
    target: &str,
    extra: &[&str],
) -> Option<PathBuf> {
    let src = dir.join(format!("{name}.s"));
    let obj = dir.join(format!("{name}.o"));
    let body = String::from_utf8_lossy(template).replace('~', value);
    fs::write(&src, body).ok()?;
    let built = Command::new(clang)
        .arg(target)
        .args(extra)
        .arg("-c")
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping abs-duplicate: clang cannot build {name}");
        return None;
    }
    Some(obj)
}

/// Reads `start`'s exit status out of the linked image by running it.
fn read_answer(out: &Path) -> i32 {
    let run = std::process::Command::new(out)
        .output()
        .expect("run the image");
    run.status.code().unwrap_or_default()
}
