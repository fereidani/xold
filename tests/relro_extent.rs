//! `PT_GNU_RELRO` never describes memory outside the segment that maps it.
//!
//! The RELRO header rounds its end up to a page: glibc reads `p_memsz` into
//! `l_relro_size` and `_dl_protect_relro` mprotects the page-aligned interior,
//! so an unrounded run shorter than a page collapses to nothing and a longer
//! one loses its final partial page.
//!
//! Those rounded-up bytes have to be inside a `PT_LOAD`. Placement pads to a
//! page when a non-RELRO writable region follows the run, but when the run is
//! the whole writable segment -- code plus `.dynamic` and `.got` and nothing
//! else, which is what a small shared object is -- there was nothing to pad
//! against, and RELRO ran up to a page past the only segment covering it.
//! glibc mprotects it anyway; `eu-elflint` flags it, and it is a header
//! describing memory nothing maps.
//!
//! The read-write `PT_LOAD` now runs to the same ceiling. Only `p_memsz`
//! grows: the extra is zero memory, exactly like `.bss`. lld reaches the same
//! extent by putting a `.relro_padding` section inside both segments.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{icf::IcfMode, linker::link_shared};

mod common;

/// A shared object with no writable data at all, so its RELRO run is the whole
/// read-write segment and nothing follows to pad against.
const TAIL_SRC: &[u8] = b"int add_one(int x) { return x + 1; }\n";

/// One with writable data after the run, so a non-RELRO region follows it.
const AFTER_SRC: &[u8] = b"int counter = 0;\n\
    int bump(void) { return ++counter; }\n";

const PT_LOAD: u32 = 1;
const PT_GNU_RELRO: u32 = 0x6474_e552;
const PF_W: u32 = 2;

/// The RELRO run stays inside the writable segment when it is that segment's
/// tail.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn relro_stays_inside_the_load_that_maps_it() {
    let Some(dir) = workdir("tail") else {
        return;
    };
    let Some(bytes) = link(&dir, TAIL_SRC, "tail") else {
        return;
    };
    let relro = segment(&bytes, PT_GNU_RELRO, 0).expect("PT_GNU_RELRO present");
    assert!(
        relro.memsz > relro.filesz,
        "the fixture must have a RELRO run that rounds up to a page, or it \
         proves nothing"
    );
    let load = writable_load(&bytes).expect("a writable PT_LOAD");
    assert!(
        relro.vaddr >= load.vaddr
            && relro.vaddr + relro.memsz <= load.vaddr + load.memsz,
        "RELRO [{:#x}, {:#x}) must lie inside the writable LOAD [{:#x}, \
         {:#x}); outside it, the header describes memory nothing maps",
        relro.vaddr,
        relro.vaddr + relro.memsz,
        load.vaddr,
        load.vaddr + load.memsz
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Only `p_memsz` grew: the padding is zero memory, so nothing was added to
/// the file.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_extra_extent_costs_no_file_bytes() {
    let Some(dir) = workdir("filesz") else {
        return;
    };
    let Some(bytes) = link(&dir, TAIL_SRC, "tail") else {
        return;
    };
    let load = writable_load(&bytes).expect("a writable PT_LOAD");
    assert!(
        load.memsz > load.filesz,
        "the ceiling must come from p_memsz alone"
    );
    assert!(
        load.offset + load.filesz <= bytes.len() as u64,
        "and the file extent must still be in the file"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: with a writable region after the run, placement already padded
/// and the extent is unchanged by this rule.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_region_after_the_run_is_unaffected() {
    let Some(dir) = workdir("after") else {
        return;
    };
    let Some(bytes) = link(&dir, AFTER_SRC, "after") else {
        return;
    };
    let relro = segment(&bytes, PT_GNU_RELRO, 0).expect("PT_GNU_RELRO present");
    let load = writable_load(&bytes).expect("a writable PT_LOAD");
    assert!(
        relro.vaddr + relro.memsz <= load.vaddr + load.memsz,
        "the run must stay inside its segment here too"
    );
    assert!(
        load.filesz > 0,
        "the fixture must actually carry writable data after the run"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping relro-extent {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_relro_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles one source and links it as a shared object.
fn link(dir: &Path, src: &[u8], stem: &str) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let src_path = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIC", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping relro-extent: clang cannot build the fixture");
        return None;
    }
    let out = dir.join(format!("lib{stem}.so"));
    let res = link_shared(
        std::slice::from_ref(&obj),
        &out,
        Some(b"librelro.so"),
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

// --- readers ---------------------------------------------------------------

/// The fields of one program header these tests reason about.
struct Segment {
    offset: u64,
    vaddr: u64,
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
            offset: read_u64(bytes, at + 8)?,
            vaddr: read_u64(bytes, at + 16)?,
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
