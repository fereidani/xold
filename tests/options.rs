//! Command-line option handling.
//!
//! xold implements a fixed set of options. An argument that looks like an
//! option and is not one used to fall through to the input list, so a link
//! that named a feature xold lacks was reported as `No such file or
//! directory` naming a path nobody wrote, and a
//! reader had no way to tell that from a genuinely missing input. These tests
//! pin the two halves of the answer: an unknown option ends the link and says
//! which one it was, and an option that names behaviour xold already has is
//! accepted rather than refused.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{which, xold_bin};

mod common;

fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_options_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// The smallest thing that links: an entry point that exits. It carries call
/// frame information, so the image it links into has an `.eh_frame` for
/// `--eh-frame-hdr` to describe.
const SRC: &str = r"
        .text
        .globl  _start
_start:
        .cfi_startproc
        movl    $60, %eax
        xorl    %edi, %edi
        syscall
        .cfi_endproc
";

/// Assembles [`SRC`], or `None` without clang.
fn object(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let file = dir.join("start.s");
    let obj = dir.join("start.o");
    fs::write(&file, SRC).ok()?;
    Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c", "-o"])
        .arg(&obj)
        .arg(&file)
        .status()
        .ok()?
        .success()
        .then_some(obj)
}

/// Runs the linker over `args` plus one object, returning its exit status and
/// what it printed.
fn link(dir: &Path, args: &[&str], out: &str) -> Option<(bool, String)> {
    let obj = object(dir)?;
    let result = Command::new(xold_bin())
        .args(args)
        .arg(&obj)
        .args(["-o"])
        .arg(dir.join(out))
        .output()
        .ok()?;
    let printed = String::from_utf8_lossy(&result.stderr).into_owned();
    Some((result.status.success(), printed))
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_unknown_option_ends_the_link_and_names_itself() {
    let dir = workdir("unknown");
    let Some((ok, printed)) = link(&dir, &["--print-map"], "prog") else {
        eprintln!("skipping unknown option test: clang unavailable");
        return;
    };
    assert!(!ok, "an option xold does not implement must end the link");
    assert!(
        printed.contains("unknown option `--print-map`"),
        "the diagnostic must name the option: {printed}"
    );
    assert!(
        !printed.contains("No such file"),
        "an option must not be reported as a missing input file: {printed}"
    );
}

/// The option is named even when it is one that takes a value, so
/// `--defsym x=1` does not read as a complaint about `x=1`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_unknown_option_with_a_value_names_the_option() {
    let dir = workdir("valued");
    let Some((ok, printed)) = link(&dir, &["--defsym", "x=1"], "prog") else {
        eprintln!("skipping valued option test: clang unavailable");
        return;
    };
    assert!(!ok);
    assert!(
        printed.contains("unknown option `--defsym`"),
        "the diagnostic must name the option, not its value: {printed}"
    );
}

/// A lone `-` is a filename by convention, not an option, so it keeps
/// reporting what it is: an input that is not there.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_lone_dash_is_still_an_input_path() {
    let dir = workdir("dash");
    let Some((ok, printed)) = link(&dir, &["-"], "prog") else {
        eprintln!("skipping lone dash test: clang unavailable");
        return;
    };
    assert!(!ok);
    assert!(
        !printed.contains("unknown option"),
        "a lone dash is not an option: {printed}"
    );
}

/// Options that name what xold already does are accepted: it never collects
/// unreferenced sections unless asked, and it always writes `.eh_frame_hdr`.
/// Accepting them is not the same as ignoring an option, because the image is
/// the one the option describes.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn options_that_name_the_default_behaviour_are_accepted() {
    let dir = workdir("accepted");
    let Some((ok, printed)) =
        link(&dir, &["--no-gc-sections", "--eh-frame-hdr"], "prog")
    else {
        eprintln!("skipping accepted option test: clang unavailable");
        return;
    };
    assert!(ok, "xold must accept options it already honours: {printed}");
    let bytes = fs::read(dir.join("prog")).expect("read linked image");
    assert!(
        phdr_types(&bytes).contains(&0x6474_e550),
        "--eh-frame-hdr must describe an image that really carries one"
    );
}

fn phdr_types(bytes: &[u8]) -> Vec<u32> {
    let phoff = bytes
        .get(32..40)
        .and_then(|cell| cell.try_into().ok())
        .map(u64::from_le_bytes)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(0);
    let entsize = bytes
        .get(54..56)
        .and_then(|cell| cell.try_into().ok())
        .map(u16::from_le_bytes)
        .map_or(0, usize::from);
    let count = bytes
        .get(56..58)
        .and_then(|cell| cell.try_into().ok())
        .map(u16::from_le_bytes)
        .map_or(0, usize::from);
    (0..count)
        .filter_map(|i| {
            let at = phoff.checked_add(i.checked_mul(entsize)?)?;
            bytes
                .get(at..at.checked_add(4)?)
                .and_then(|cell| cell.try_into().ok())
                .map(u32::from_le_bytes)
        })
        .collect()
}
