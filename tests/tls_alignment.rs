//! `PT_TLS` starts where the alignment it declares says it does.
//!
//! The segment declares one alignment for `.tdata` and `.tbss` together, so
//! the block's first byte has to sit at the larger of the two. xold placed
//! `.tdata` at its own member alignment while computing `p_align` as the
//! maximum of the pair, so `.tdata` at align 1 beside `.tbss` at align 64 left
//! `p_vaddr % 64 != 0`.
//!
//! A loader that lays the block down by `roundup(memsz, align)` without
//! compensating for the first byte then reads every thread-local at the wrong
//! offset, and the alignment the block declares is not the one it gets. lld
//! aligns the segment start for exactly this in `fixSectionAlignments`, citing
//! glibc PR/24606, FreeBSD's rtld, and musl before 1.1.23.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{elf::constants::PT_TLS, icf::IcfMode, linker::link_dyn_exec};

mod common;

/// The alignment `.tbss` declares, larger than anything `.tdata` asks for.
const WIDE: u64 = 64;

/// A byte of `.tdata` at alignment 1, so the block opens on a section that
/// asks for nothing, and a widely aligned `.tbss` behind it. That is the pair
/// whose maximum `PT_TLS` declares.
const SRC: &[u8] = b"_Thread_local char tdata_byte = 1;\n\
    _Thread_local __attribute__((aligned(64))) int tbss_wide[16];\n\
    int main(void)\n\
    {\n\
        return ((unsigned long)(void *)&tbss_wide[0] % 64) != 0;\n\
    }\n";

/// `PT_TLS`'s `p_vaddr` is congruent to the alignment it declares.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_tls_segment_start_is_congruent_to_its_alignment() {
    let Some(dir) = workdir("tls") else {
        return;
    };
    let Some(prog) = link(&dir, "tls") else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");
    let tls = tls_segment(&bytes).expect("the image has a PT_TLS");
    assert_eq!(
        tls.align, WIDE,
        "PT_TLS declares the pair's alignment, which .tbss sets here"
    );
    assert_eq!(
        tls.vaddr % tls.align,
        0,
        "p_vaddr ({:#x}) must be congruent to p_align ({:#x}), or a loader \
         placing the block by roundup(memsz, align) reads every thread-local \
         at the wrong offset",
        tls.vaddr,
        tls.align
    );
    let _ = fs::remove_dir_all(&dir);
}

/// And the thread-local really is aligned once the loader has laid the block
/// down, which is the property the congruence exists for.
///
/// This does not fail on a host with a current glibc even when `p_vaddr` is
/// not congruent: glibc has compensated for the first byte since the fix for
/// PR/24606. It stands in for the loaders that do not, and it checks that
/// raising `.tdata`'s placement alignment did not move the block somewhere the
/// program disagrees with. The structural test above is what pins the
/// congruence.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_thread_local_is_aligned_at_run_time() {
    let Some(dir) = workdir("tlsrun") else {
        return;
    };
    let Some(prog) = link(&dir, "tlsrun") else {
        return;
    };
    assert_eq!(
        Command::new(&prog)
            .status()
            .expect("linked program must be runnable")
            .code(),
        Some(0),
        "the program's own modulo check on its thread-local must pass"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping tls-alignment {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_tlsalign_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture and links it into a dynamic executable against the
/// host libc, so the produced program can actually run.
fn link(dir: &Path, stem: &str) -> Option<PathBuf> {
    let Some(interp) = interpreter() else {
        eprintln!("skipping tls-alignment {stem}: interpreter path unknown");
        return None;
    };
    let obj = dir.join(format!("{stem}.o"));
    compile(SRC, &obj)?;
    let start = common::crt_file("Scrt1.o")?;
    let prologue = common::crt_file("crti.o")?;
    let epilogue = common::crt_file("crtn.o")?;
    let libc = common::libc_so()?;

    let prog = dir.join(stem);
    let res = link_dyn_exec(
        &[start, prologue, obj, libc, epilogue],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "xold link must succeed: {:?}", res.err());
    Some(prog)
}

/// Compiles `src` with the host clang.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).ok()?;
    Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIE", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success()
        .then_some(())
}

// --- readers ---------------------------------------------------------------

/// The `PT_TLS` fields these tests reason about.
struct Tls {
    vaddr: u64,
    align: u64,
}

/// `Ehdr64` offset of `e_phoff`.
const E_PHOFF: usize = 32;
/// `Ehdr64` offset of `e_phentsize`.
const E_PHENTSIZE: usize = 54;
/// `Ehdr64` offset of `e_phnum`.
const E_PHNUM: usize = 56;

/// Reads the `PT_TLS` program header, or `None` when the image has none.
///
/// Read from the raw bytes: the object reader exposes section headers only,
/// and the question here is what the segment table says.
fn tls_segment(bytes: &[u8]) -> Option<Tls> {
    let phoff = usize::try_from(read_u64(bytes, E_PHOFF)).ok()?;
    let phentsize = usize::from(read_u16(bytes, E_PHENTSIZE));
    let phnum = usize::from(read_u16(bytes, E_PHNUM));
    (0..phnum)
        .map(|i| phoff + i * phentsize)
        .find(|&at| read_u32(bytes, at) == PT_TLS)
        .map(|at| Tls {
            vaddr: read_u64(bytes, at + 16),
            align: read_u64(bytes, at + 48),
        })
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
