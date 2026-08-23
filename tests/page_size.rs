//! The page size the segments are separated at.
//!
//! Two `PT_LOAD` segments with different permissions may not share a system
//! page. If they do, the kernel maps that page twice and the later, writable
//! mapping's protection covers the whole of it, so the tail of `.text` in the
//! shared page loses `PROT_EXEC`. A small binary can lose it on all of
//! `.text` and die at its own entry point.
//!
//! Which page that is depends on the kernel, not on the file. Linux runs
//! `AArch64` with 4K, 16K (Asahi, much of Android) and 64K (RHEL and Fedora on
//! ARM) pages, so an image separated at 4K is loadable on one of the three.
//! lld handles this with a per-target maximum page size -- 65536 on `AArch64`
//! (`Arch/AArch64.cpp:131`), 4096 elsewhere -- and that is what these tests
//! pin: `AArch64` separates and declares 64K, x86-64 stays at 4K.
//!
//! The failure mode itself cannot be reproduced here: qemu-user runs with the
//! host's page size, so a 4K-separated image loads fine under it. What qemu
//! does prove is the other direction -- that the wider separation did not
//! break the produced binary -- and the segment arithmetic proves the rest.
//!
//! Gated on `clang` (plus the sysroot and qemu for the `AArch64` cases); a
//! missing tool prints a note and returns.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{elf::constants::PT_LOAD, icf::IcfMode, linker::link_dyn_exec};

mod common;

/// lld's `defaultMaxPageSize` for `AArch64`.
const AARCH64_MAX_PAGE: u64 = 0x1_0000;
/// lld's `defaultMaxPageSize` everywhere else this linker targets.
const DEFAULT_MAX_PAGE: u64 = 0x1000;

/// A program small enough that its whole `.text` fits in one 64K page, which
/// is the shape that loses execute permission outright.
const SRC: &[u8] = b"#include <stdio.h>\n\
    int datum = 5;\n\
    int main(void) { printf(\"page %d\\n\", datum + 1); return 0; }\n";

/// `AArch64` separates the segments at 64K and says so in `p_align`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn aarch64_segments_are_separated_at_the_largest_page() {
    let Some(sys) = aarch64_sysroot() else {
        eprintln!("skipping aarch64 page size: sysroot missing");
        return;
    };
    let Some(prog) = link_aarch64(&sys, "aa_align") else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");
    let loads = load_segments(&bytes);
    assert_eq!(loads.len(), 2, "expected one RX and one RW PT_LOAD");

    for seg in &loads {
        assert_eq!(
            seg.align, AARCH64_MAX_PAGE,
            "a PT_LOAD's p_align states the page granularity the image is \
             laid out for; on `AArch64` that is 64K"
        );
    }
    assert_no_shared_page(&loads, AARCH64_MAX_PAGE);
    let _ = fs::remove_dir_all(prog.parent().unwrap_or_else(|| Path::new(".")));
}

/// The produced image still runs. qemu cannot show the 64K-page failure, but
/// it does show the wider separation did not break anything else.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_aarch64_image_still_runs() {
    let Some(qemu) = which("qemu-aarch64-static") else {
        eprintln!("skipping aarch64 page run: qemu-aarch64-static missing");
        return;
    };
    let Some(sys) = aarch64_sysroot() else {
        eprintln!("skipping aarch64 page run: sysroot missing");
        return;
    };
    let Some(prog) = link_aarch64(&sys, "aa_run") else {
        return;
    };
    let out = Command::new(&qemu)
        .env("QEMU_LD_PREFIX", &sys)
        .arg(&prog)
        .output()
        .expect("qemu must start");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("page 6"),
        "the 64K-separated image must still run (got {stdout:?})"
    );
    assert_eq!(out.status.code(), Some(0), "and exit 0");
    let _ = fs::remove_dir_all(prog.parent().unwrap_or_else(|| Path::new(".")));
}

/// x86-64 keeps 4K, so its bytes do not move. Kernels for it are configured
/// with 4K pages and nothing else, and a wider separation would only pad the
/// image.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn x86_64_keeps_the_four_kilobyte_page() {
    let Some(prog) = link_x86_64() else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");
    let loads = load_segments(&bytes);
    assert_eq!(loads.len(), 2, "expected one RX and one RW PT_LOAD");
    for seg in &loads {
        assert_eq!(
            seg.align, DEFAULT_MAX_PAGE,
            "x86-64 kernels use 4K pages; widening this would pad every image \
             for nothing"
        );
    }
    assert_no_shared_page(&loads, DEFAULT_MAX_PAGE);
    let _ = fs::remove_dir_all(prog.parent().unwrap_or_else(|| Path::new(".")));
}

/// Asserts no two segments with different flags occupy the same `page`-sized
/// system page.
fn assert_no_shared_page(loads: &[Load], page: u64) {
    for (i, a) in loads.iter().enumerate() {
        for b in loads.iter().skip(i + 1) {
            if a.flags == b.flags {
                continue;
            }
            let a_last = (a.vaddr + a.memsz.saturating_sub(1)) / page;
            let b_first = b.vaddr / page;
            assert!(
                a_last < b_first || b.vaddr + b.memsz <= a.vaddr,
                "PT_LOAD {a:?} and {b:?} share a {page:#x} page; the writable \
                 mapping's protection would cover the executable one's tail"
            );
        }
    }
}

// --- fixtures --------------------------------------------------------------

/// The `AArch64` sysroot carrying the crt objects and `libc.so.6`, or `None`.
fn aarch64_sysroot() -> Option<PathBuf> {
    for c in [
        "/usr/aarch64-redhat-linux/sys-root/fc43",
        "/usr/aarch64-redhat-linux/sys-root",
    ] {
        let p = PathBuf::from(c);
        if p.join("usr/lib64/crt1.o").is_file()
            && p.join("usr/lib64/libc.so.6").is_file()
        {
            return Some(p);
        }
    }
    None
}

/// Creates a fresh per-test working directory.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping page size {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_pagesize_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Links the fixture for `AArch64` against the sysroot libc, in a directory
/// named for the calling test so concurrent tests do not share one.
fn link_aarch64(sys: &Path, prefix: &str) -> Option<PathBuf> {
    let dir = workdir(prefix)?;
    let obj = dir.join("page.o");
    compile(SRC, "--target=aarch64-linux-gnu", Some(sys), &obj)?;
    let prog = dir.join("page_aa");
    let inputs = vec![
        obj,
        sys.join("usr/lib64/crti.o"),
        sys.join("usr/lib64/crt1.o"),
        sys.join("usr/lib64/crtn.o"),
        sys.join("usr/lib64/libc.so.6"),
    ];
    let res = link_dyn_exec(
        &inputs,
        &prog,
        b"_start",
        b"/usr/lib/ld-linux-aarch64.so.1",
        false,
        IcfMode::None,
        false,
    );
    assert!(
        res.is_ok(),
        "xold aarch64 link must succeed: {:?}",
        res.err()
    );
    Some(prog)
}

/// Links the same fixture for the host x86-64 toolchain.
fn link_x86_64() -> Option<PathBuf> {
    let dir = workdir("x86_64")?;
    let Some(interp) = interpreter() else {
        eprintln!("skipping page size x86_64: interpreter path unknown");
        return None;
    };
    let obj = dir.join("page.o");
    compile(SRC, "--target=x86_64-linux-gnu", None, &obj)?;
    let start = common::crt_file("Scrt1.o")?;
    let prologue = common::crt_file("crti.o")?;
    let epilogue = common::crt_file("crtn.o")?;
    let libc = common::libc_so()?;
    let prog = dir.join("page_x86");
    let res = link_dyn_exec(
        &[start, prologue, obj, libc, epilogue],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    );
    assert!(
        res.is_ok(),
        "xold x86-64 link must succeed: {:?}",
        res.err()
    );
    Some(prog)
}

/// Compiles `src` for `triple`, optionally against `sysroot`.
fn compile(
    src: &[u8],
    triple: &str,
    sysroot: Option<&Path>,
    obj: &Path,
) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).ok()?;
    let mut cmd = Command::new(clang);
    cmd.args([triple, "-c"]);
    if let Some(sys) = sysroot {
        cmd.arg("--sysroot").arg(sys);
    }
    cmd.arg(&src_path).arg("-o").arg(obj);
    cmd.status().ok()?.success().then_some(())
}

// --- readers ---------------------------------------------------------------

/// The fields of one `PT_LOAD` these tests reason about.
#[derive(Debug)]
struct Load {
    flags: u32,
    vaddr: u64,
    memsz: u64,
    align: u64,
}

/// `Ehdr64` offset of `e_phoff`.
const E_PHOFF: usize = 32;
/// `Ehdr64` offset of `e_phentsize`.
const E_PHENTSIZE: usize = 54;
/// `Ehdr64` offset of `e_phnum`.
const E_PHNUM: usize = 56;

/// Reads the `PT_LOAD` program headers out of a linked image.
///
/// Read from the raw bytes rather than through the object reader, which
/// exposes section headers only: the whole question here is what the *segment*
/// table says.
fn load_segments(bytes: &[u8]) -> Vec<Load> {
    let Ok(phoff) = usize::try_from(read_u64(bytes, E_PHOFF)) else {
        return Vec::new();
    };
    let phentsize = usize::from(read_u16(bytes, E_PHENTSIZE));
    let phnum = usize::from(read_u16(bytes, E_PHNUM));
    let mut out = Vec::new();
    for i in 0..phnum {
        let at = phoff + i * phentsize;
        if read_u32(bytes, at) != PT_LOAD {
            continue;
        }
        out.push(Load {
            flags: read_u32(bytes, at + 4),
            vaddr: read_u64(bytes, at + 16),
            memsz: read_u64(bytes, at + 40),
            align: read_u64(bytes, at + 48),
        });
    }
    out
}

/// Little-endian `u16` at `off`.
fn read_u16(bytes: &[u8], off: usize) -> u16 {
    let s = bytes.get(off..off + 2).unwrap_or(&[0; 2]);
    u16::from_le_bytes(s.try_into().unwrap_or([0; 2]))
}

/// Little-endian `u32` at `off`.
fn read_u32(bytes: &[u8], off: usize) -> u32 {
    let s = bytes.get(off..off + 4).unwrap_or(&[0; 4]);
    u32::from_le_bytes(s.try_into().unwrap_or([0; 4]))
}

/// Little-endian `u64` at `off`.
fn read_u64(bytes: &[u8], off: usize) -> u64 {
    let s = bytes.get(off..off + 8).unwrap_or(&[0; 8]);
    u64::from_le_bytes(s.try_into().unwrap_or([0; 8]))
}
