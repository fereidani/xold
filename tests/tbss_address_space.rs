//! `.tbss` costs no address space.
//!
//! `.tbss` is a template, not storage: the loader copies it into each thread's
//! block, and no address in the image is ever read through it. The placement
//! cursor advanced past it anyway, so `.bss`, `_end` and the read-write
//! segment's `p_memsz` all grew by its whole size -- a
//! `__thread char big[1 << 20]` cost a megabyte of image address space that
//! nothing occupies.
//!
//! Its members still get addresses, because `PT_TLS` needs the extent and the
//! TLS block reads it from the assigned range. What changed is that whatever
//! follows starts where `.tbss` did, and the segment's memory extent leaves it
//! out. lld, mold and GNU ld all do both.
//!
//! Gated on `clang` and the system crt objects; without them the tests print a
//! note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{icf::IcfMode, linker::link_dyn_exec};

mod common;

/// A megabyte of zero-initialised thread-local, which is what makes the
/// difference measurable rather than incidental.
const SRC: &[u8] = b"#include <stdio.h>\n\
    __thread char big[1 << 20];\n\
    int normal = 7;\n\
    int main(void)\n\
    {\n\
        big[0] = 1;\n\
        big[(1 << 20) - 1] = 2;\n\
        printf(\"%d %d %d\\n\", (int)big[0], (int)big[(1 << 20) - 1], normal);\n\
        return 0;\n\
    }\n";

/// One megabyte of thread-local template, which the segment must not reserve.
const TLS_SIZE: u64 = 1 << 20;

/// The read-write segment does not reserve the template.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_writable_segment_excludes_the_template() {
    let Some(dir) = workdir("memsz") else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    let tls = segment(&bytes, PT_TLS, 0).expect("PT_TLS present");
    assert!(
        tls.memsz >= TLS_SIZE,
        "the fixture must carry a megabyte of .tbss, got {:#x}",
        tls.memsz
    );
    let load = writable_load(&bytes).expect("a writable PT_LOAD");
    assert!(
        load.memsz < TLS_SIZE,
        "the segment must not reserve the thread-local template: {:#x} of \
         memory for {:#x} of file",
        load.memsz,
        load.filesz
    );
    let _ = fs::remove_dir_all(&dir);
}

/// And the program still reads and writes both ends of its thread-local, so
/// the extent `PT_TLS` declares is still the real one.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_thread_local_still_works() {
    let Some(dir) = workdir("run") else {
        return;
    };
    let Some(_) = link(&dir) else {
        return;
    };
    let out = Command::new(dir.join("prog")).output().expect("must run");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "1 2 7\n",
        "both ends of the thread-local, and the ordinary datum beside it"
    );
    assert_eq!(out.status.code(), Some(0), "and the program exits 0");
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping tbss-address-space {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_tbss_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds and links the fixture against the host crt objects.
fn link(dir: &Path) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let interp = interpreter()?;
    let src = dir.join("t.c");
    let obj = dir.join("t.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args([
            "--target=x86_64-linux-gnu",
            "-fPIE",
            "-ftls-model=local-exec",
            "-c",
        ])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping tbss-address-space: clang cannot build it");
        return None;
    }
    let paths = vec![
        crt_file("Scrt1.o")?,
        crt_file("crti.o")?,
        obj,
        libc_so()?,
        crt_file("crtn.o")?,
    ];
    let out = dir.join("prog");
    let res = link_dyn_exec(
        &paths,
        &out,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

// --- readers ---------------------------------------------------------------

const PT_LOAD: u32 = 1;
const PT_TLS: u32 = 7;
const PF_W: u32 = 2;

/// The fields of one program header these tests reason about.
struct Segment {
    filesz: u64,
    memsz: u64,
}

/// The first writable `PT_LOAD`.
fn writable_load(bytes: &[u8]) -> Option<Segment> {
    segment(bytes, PT_LOAD, PF_W)
}

/// The first program header of type `want` whose flags include `flags`.
fn segment(bytes: &[u8], want: u32, flags: u32) -> Option<Segment> {
    let phoff = usize::try_from(read_u64(bytes, 32)?).ok()?;
    let entsize = usize::from(read_u16(bytes, 54)?);
    let count = usize::from(read_u16(bytes, 56)?);
    (0..count).find_map(|i| {
        let at = phoff + i * entsize;
        if read_u32(bytes, at)? != want {
            return None;
        }
        if read_u32(bytes, at + 4)? & flags != flags {
            return None;
        }
        Some(Segment {
            filesz: read_u64(bytes, at + 32)?,
            memsz: read_u64(bytes, at + 40)?,
        })
    })
}

fn read_u16(bytes: &[u8], at: usize) -> Option<u16> {
    bytes
        .get(at..at + 2)
        .and_then(|c| <[u8; 2]>::try_from(c).ok())
        .map(u16::from_le_bytes)
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .and_then(|c| <[u8; 4]>::try_from(c).ok())
        .map(u32::from_le_bytes)
}

fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
    bytes
        .get(at..at + 8)
        .and_then(|c| <[u8; 8]>::try_from(c).ok())
        .map(u64::from_le_bytes)
}
