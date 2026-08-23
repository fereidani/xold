//! A `SHF_COMPRESSED` section is refused, not concatenated and patched.
//!
//! `gcc -gz` and `clang -gz` leave each `.debug_*` section as an `Elf64_Chdr`
//! followed by a deflate stream, with `SHF_COMPRESSED` set. The relocations in
//! `.rela.debug_*` address the content that stream expands to, not the stream.
//!
//! Nothing here looked at the flag. The compressed bytes were concatenated
//! into the output like any other contribution, and the relocations were
//! applied at offsets into the middle of a deflate stream. The link succeeded
//! and produced debug info that no reader can decompress -- the header claims
//! a size the corrupted stream will not yield.
//!
//! lld decompresses transparently (`InputFiles.cpp`,
//! `contentMaybeDecompress`), which is the answer to implement when such
//! objects need to link. Until then the failure has to be loud: a linker that
//! cannot read its input must say so rather than write bytes that look like
//! output.
//!
//! Gated on a `clang` that supports `-gz=zlib`; without one the tests print a
//! note and return.

use std::{
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_to};

mod common;

/// The fixture has to be big enough that compressing it wins: clang leaves a
/// section uncompressed when the deflate stream would not be smaller, and a
/// two-line program's `.debug_info` is well under that.
fn source() -> Vec<u8> {
    let mut src = String::from("int main(void) { return 0; }\n");
    for i in 0..120 {
        let _ = writeln!(
            src,
            "int helper_with_a_long_descriptive_name_{i}(int a, int b) \
             {{ return a * {i} + b; }}"
        );
    }
    src.into_bytes()
}

/// The compressed input is refused, and the message names the flag.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_compressed_debug_section_fails_the_link() {
    let Some(dir) = workdir("refuse") else {
        return;
    };
    let Some(obj) = compile(&dir, true, "gz") else {
        return;
    };
    assert!(
        has_compressed_section(&obj),
        "the fixture must actually carry a SHF_COMPRESSED section, or this \
         test proves nothing about compression"
    );
    let err = link(&dir, &obj, "gz").expect_err(
        "relocations in a compressed section address content this linker \
         never expanded, so the link must fail rather than patch the deflate \
         stream",
    );
    let text = format!("{err}");
    assert!(
        text.contains("SHF_COMPRESSED"),
        "the refusal must name what it cannot read, got {text:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: the same source with uncompressed debug info still links, so
/// the refusal is about the flag and not about debug info.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_uncompressed_build_still_links() {
    let Some(dir) = workdir("plain") else {
        return;
    };
    let Some(obj) = compile(&dir, false, "plain") else {
        return;
    };
    assert!(
        !has_compressed_section(&obj),
        "the control must carry no compressed section"
    );
    let res = link(&dir, &obj, "plain");
    assert!(
        res.is_ok(),
        "an ordinary -g object must link: {:?}",
        res.err()
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping compressed-section {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_compressed_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture with debug info, compressed or not.
fn compile(dir: &Path, gz: bool, stem: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src, source()).ok()?;
    let mut cmd = Command::new(clang);
    cmd.args(["--target=x86_64-linux-gnu", "-g", "-fno-pic", "-c"]);
    if gz {
        cmd.arg("-gz=zlib");
    }
    let built = cmd.arg(&src).arg("-o").arg(&obj).status().ok()?.success();
    if !built {
        eprintln!("skipping compressed-section: clang cannot build with -gz");
        return None;
    }
    Some(obj)
}

/// Links `obj` as a static executable.
fn link(dir: &Path, obj: &Path, stem: &str) -> Result<(), xold::Error> {
    link_to(
        std::slice::from_ref(&obj.to_path_buf()),
        &dir.join(stem),
        b"main",
        false,
        IcfMode::None,
        false,
    )
}

// --- readers ---------------------------------------------------------------

/// Whether any section of `obj` carries `SHF_COMPRESSED`.
fn has_compressed_section(obj: &Path) -> bool {
    let Ok(bytes) = fs::read(obj) else {
        return false;
    };
    let Ok(parsed) = ObjectFile::parse(&bytes) else {
        return false;
    };
    parsed
        .sections()
        .iter()
        .any(|s| s.sh_flags.get() & COMPRESSED != 0)
}

/// `SHF_COMPRESSED`, spelled here so the test does not depend on the constant
/// it is checking.
const COMPRESSED: u64 = 0x800;
