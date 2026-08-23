//! `PT_PHDR` and `PT_INTERP` come before the first `PT_LOAD`.
//!
//! The gABI states the requirement twice over: `PT_INTERP`, if present, must
//! precede any loadable segment entry, and `PT_PHDR` must precede any entry it
//! appears alongside. xold pushed the read-execute `PT_LOAD` first and
//! `PT_INTERP` after it.
//!
//! Linux scans the whole table, so the images loaded and ran; the cost was a
//! header table a conforming reader is entitled to reject, which `eu-elflint`
//! does. Conforming is free: the entries carry their own offsets, so only
//! their position in the table changes.
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

const SRC: &[u8] = b"int main(void) { return 0; }\n";

const PT_LOAD: u32 = 1;
const PT_INTERP: u32 = 3;
const PT_PHDR: u32 = 6;

/// `PT_INTERP` precedes every `PT_LOAD`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn interp_precedes_the_first_load() {
    let Some(dir) = workdir("interp") else {
        return;
    };
    let Some(types) = phdr_types(&dir) else {
        return;
    };
    let interp = types
        .iter()
        .position(|t| *t == PT_INTERP)
        .expect("a dynamic executable has a PT_INTERP");
    let load = types
        .iter()
        .position(|t| *t == PT_LOAD)
        .expect("and at least one PT_LOAD");
    assert!(
        interp < load,
        "the gABI requires PT_INTERP to precede any loadable entry; the \
         table is {types:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `PT_PHDR` leads the table.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn phdr_leads_the_table() {
    let Some(dir) = workdir("phdr") else {
        return;
    };
    let Some(types) = phdr_types(&dir) else {
        return;
    };
    let phdr = types
        .iter()
        .position(|t| *t == PT_PHDR)
        .expect("a position-independent executable has a PT_PHDR");
    assert_eq!(
        phdr, 0,
        "PT_PHDR must precede every entry it appears alongside; the table is \
         {types:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// And the image still loads and runs, which is what says the reorder moved
/// entries and not the regions they describe.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_reordered_image_still_runs() {
    let Some(dir) = workdir("run") else {
        return;
    };
    let Some(prog) = link(&dir) else {
        return;
    };
    let code = Command::new(&prog).status().expect("must run").code();
    assert_eq!(code, Some(0), "the program must still exit 0");
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping phdr-order {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_phdrorder_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Links a dynamic executable against the host crt objects.
fn link(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let interp = interpreter()?;
    let src = dir.join("p.c");
    let obj = dir.join("p.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIE", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping phdr-order: clang cannot build the fixture");
        return None;
    }
    let start = crt_file("Scrt1.o")?;
    let prologue = crt_file("crti.o")?;
    let epilogue = crt_file("crtn.o")?;
    let libc = libc_so()?;
    let prog = dir.join("prog");
    let res = link_dyn_exec(
        &[start, prologue, obj, libc, epilogue],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    Some(prog)
}

// --- readers ---------------------------------------------------------------

/// The `p_type` of every program header, in table order.
fn phdr_types(dir: &Path) -> Option<Vec<u32>> {
    let prog = link(dir)?;
    let bytes = fs::read(&prog).ok()?;
    let phoff = usize::try_from(read_u64(&bytes, 32)?).ok()?;
    let entsize = usize::from(read_u16(&bytes, 54)?);
    let count = usize::from(read_u16(&bytes, 56)?);
    (0..count)
        .map(|i| read_u32(&bytes, phoff + i * entsize))
        .collect()
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
