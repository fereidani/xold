//! COFF commons and weak externals resolve to real addresses.
//!
//! A common is a tentative definition: section number `IMAGE_SYM_UNDEFINED`
//! with a non-zero value that states the byte size, and the linker is expected
//! to allocate the storage. Nothing did. A weak external is storage class
//! `IMAGE_SYM_CLASS_WEAK_EXTERNAL` with an auxiliary record naming the default
//! definition by table index; the index was decoded nowhere.
//!
//! Both landed in the resolver's undefined arm, resolved to address zero, and
//! the link died with "its section was dropped or it is undefined" -- true of
//! neither. lld allocates commons in `.bss` and resolves weak externals
//! through the aux default.
//!
//! The commons block goes past every `.bss` member, so no member offset moves;
//! its slots are laid out in input order, so the addresses do not depend on
//! which file was parsed first.
//!
//! Gated on a `clang` that can target `x86_64-pc-windows-msvc`, and the
//! running test additionally on `wine`; without them the tests print a note
//! and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{coff::link_coff, input::Input};

mod common;

/// A tentative definition and a weak external in one translation unit. The
/// program returns 0 only if both resolved: the store through the common must
/// survive a load, and the weak call must reach its default.
const SRC: &[u8] = b"int tentative;\n\
    int weak_target(void) { return 77; }\n\
    extern int weak_ref(void) __attribute__((weak, alias(\"weak_target\")));\n\
    int main(void)\n\
    {\n\
        tentative = 5;\n\
        if (tentative != 5) return 1;\n\
        return weak_ref() == 77 ? 0 : 2;\n\
    }\n";

/// A second file declaring the same tentative definition, larger.
const OTHER: &[u8] = b"char tentative[64];\n\
    int touch(void) { tentative[63] = 1; return tentative[63]; }\n";

/// The link succeeds where it used to refuse.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_common_and_a_weak_external_link() {
    let Some(dir) = workdir("link") else {
        return;
    };
    let Some(bytes) = link(&dir, false) else {
        return;
    };
    assert_ne!(bytes.len(), 0, "the image must have content");
    let _ = fs::remove_dir_all(&dir);
}

/// `.bss` grows by the commons block, and the storage is real address space.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_common_gets_storage_in_bss() {
    let Some(dir) = workdir("bss") else {
        return;
    };
    let Some(bytes) = link(&dir, true) else {
        return;
    };
    let bss = section(&bytes, b".bss").expect("commons must produce a .bss");
    // 64 bytes for the larger declaration, on a 32-byte boundary, past
    // whatever `.bss` members the inputs contributed.
    assert!(
        bss.virtual_size >= 64,
        "the block must hold the largest declaration of the name: {:#x}",
        bss.virtual_size
    );
    assert_eq!(bss.raw_size, 0, "and it stays uninitialised, as .bss is");
    let _ = fs::remove_dir_all(&dir);
}

/// And the program runs: the store lands in the allocation and the weak call
/// reaches its default.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_program_reads_back_what_it_wrote() {
    let Some(dir) = workdir("run") else {
        return;
    };
    if which("wine").is_none() {
        eprintln!("skipping coff-commons-weak run: wine unavailable");
        return;
    }
    let Some(_) = link(&dir, false) else {
        return;
    };
    let out = Command::new("wine")
        .arg(dir.join("c.exe"))
        .output()
        .expect("wine must run the image");
    assert_eq!(
        out.status.code(),
        Some(0),
        "1 means the common did not hold the store, 2 means the weak external \
         did not reach its default"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping coff-commons-weak {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_coffcommon_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture with `-fcommon` and links it into an executable.
/// `two_files` adds a second, larger declaration of the same name.
fn link(dir: &Path, two_files: bool) -> Option<Vec<u8>> {
    let first = compile(dir, SRC, "c")?;
    let second = if two_files {
        Some(compile(dir, OTHER, "d")?)
    } else {
        None
    };
    let out = dir.join("c.exe");
    let mut files = vec![Input::Path(&first)];
    if let Some(path) = second.as_ref() {
        files.push(Input::Path(path));
    }
    let res = link_coff(&files, &out, b"main", false);
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

/// Compiles one fixture for Windows.
fn compile(dir: &Path, src: &[u8], stem: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let path = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.obj"));
    fs::write(&path, src).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-pc-windows-msvc", "-O1", "-fcommon", "-c"])
        .arg(&path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping coff-commons-weak: clang cannot target Windows");
        return None;
    }
    Some(obj)
}

// --- readers ---------------------------------------------------------------

/// The fields of one PE section header these tests reason about.
struct Section {
    virtual_size: u32,
    raw_size: u32,
}

/// Reads a section header out of a PE image by name.
fn section(bytes: &[u8], name: &[u8]) -> Option<Section> {
    let pe = usize::try_from(read_u32(bytes, 0x3c)?).ok()?;
    let num = usize::from(read_u16(bytes, pe + 6)?);
    let opt = usize::from(read_u16(bytes, pe + 20)?);
    let first = pe + 24 + opt;
    (0..num).find_map(|i| {
        let at = first + i * 40;
        let cell = bytes.get(at..at + 8)?;
        let trimmed = cell.split(|&b| b == 0).next().unwrap_or(cell);
        if trimmed != name {
            return None;
        }
        Some(Section {
            virtual_size: read_u32(bytes, at + 8)?,
            raw_size: read_u32(bytes, at + 16)?,
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
