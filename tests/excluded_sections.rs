//! `SHF_EXCLUDE` sections are dropped, and `.zdebug_*` refuses loudly.
//!
//! `SHF_EXCLUDE` is how a producer asks that a section not survive the
//! final link -- `.text.unlikely` under a profile, the `.discard`
//! families the kernel and libc build with. lld discards such a section
//! wherever it finds one (`lld/ELF/InputFiles.cpp`),
//! keeping only the `-r` case, which relinks rather than links. xold
//! treated the flag as unknown: the section was laid out and written
//! as though it had never asked to be dropped.
//!
//! `.zdebug_*` is the older spelling of compressed debug info: deflate
//! behind no `Elf64_Chdr`, signalled by the name alone. The
//! `SHF_COMPRESSED` spelling already refuses with the reason; this one
//! was dropped by the non-alloc filter without a word, and the linked
//! image simply had no debug info. lld decompresses both spellings;
//! until that is implemented, the legacy one refuses the same way the
//! modern one does rather than losing the data silently.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{icf::IcfMode, linker::link_to};

mod common;

/// One live and one excluded section, each carrying a run of its own
/// byte: the link keeps the first run and drops the second.
const EXCLUDE_SRC: &[u8] = b"\
.section .kept,\"ax\",@progbits
.fill 8,1,0x11
.section .excluded,\"axe\",@progbits
.fill 8,1,0x22
.section .text,\"ax\"
.globl _start
_start:
    ret
";

/// A non-allocated section that is neither debug info nor anything this
/// linker knows by name: the shape `rustc` puts a library's metadata in.
const METADATA_SRC: &[u8] = b"\
.section .rustc,\"\",@progbits
.ascii \"metadata-payload\"
.section .text,\"ax\"
.globl _start
_start:
    ret
";

/// A legacy compressed debug section, carrying deflate-shaped bytes.
const ZDEBUG_SRC: &[u8] = b"\
.section .zdebug_info,\"\",@progbits
.byte 0x78, 0x9c, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01
.section .text,\"ax\"
.globl _start
_start:
    ret
";

/// The excluded section does not reach the image; the live one does.
///
/// xold folds an unnamed alloc section into the region its flags pick,
/// so what survives of either section is its byte run, not its name:
/// the assertions read the image for the runs themselves.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_excluded_section_is_dropped() {
    let Some(dir) = workdir("exclude") else {
        return;
    };
    let Some(obj) = assemble(&dir, "exclude", EXCLUDE_SRC) else {
        return;
    };
    let out = dir.join("prog");
    let res = link_to(&[obj], &out, b"_start", false, IcfMode::None, false);
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let bytes = fs::read(&out).expect("read the image");
    assert!(
        bytes.windows(8).any(|w| w == [0x11; 8]),
        "the section without the flag still arrives"
    );
    assert!(
        !bytes.windows(8).any(|w| w == [0x22; 8]),
        "the section asked to be excluded stays out of the image"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A non-allocated section the linker has no name for still reaches the
/// image.
///
/// Nothing at runtime reads one, so dropping it costs nothing a program
/// notices -- and everything a toolchain does. `rustc` keeps a library's
/// metadata in `.rustc`, and a proc-macro library linked without it compiles,
/// links, and then cannot be loaded by the compiler that asked for it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_non_allocated_section_survives_the_link() {
    let Some(dir) = workdir("metadata") else {
        return;
    };
    let Some(obj) = assemble(&dir, "metadata", METADATA_SRC) else {
        return;
    };
    let out = dir.join("prog");
    let res = link_to(&[obj], &out, b"_start", false, IcfMode::None, false);
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let bytes = fs::read(&out).expect("read the image");
    assert!(
        bytes
            .windows(b"metadata-payload".len())
            .any(|w| w == b"metadata-payload"),
        "the section's bytes must reach the image"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A `.zdebug_*` section refuses the link with the reason, rather than
/// vanishing.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_legacy_compressed_debug_section_refuses() {
    let Some(dir) = workdir("zdebug") else {
        return;
    };
    let Some(obj) = assemble(&dir, "zdebug", ZDEBUG_SRC) else {
        return;
    };
    let out = dir.join("prog");
    let res = link_to(&[obj], &out, b"_start", false, IcfMode::None, false);
    let msg = format!("{err:?}", err = res.expect_err("zlib debug refuses"));
    assert!(
        msg.contains("zdebug") || msg.contains("compressed"),
        "the refusal names the section's shape: {msg}"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping excluded-sections {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_excl_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Assembles one fixture to a static object.
fn assemble(dir: &Path, name: &str, src: &[u8]) -> Option<PathBuf> {
    let src_path = dir.join(format!("{name}.s"));
    let obj = dir.join(format!("{name}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(which("clang").expect("checked in workdir"))
        .args(["--target=x86_64-linux-gnu", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping excluded-sections: clang cannot build {name}");
        return None;
    }
    Some(obj)
}
