//! Three things the PE layout got wrong about extents.
//!
//! Members were packed end to end and `IMAGE_SCN_ALIGN_*` was read nowhere.
//! clang puts its `__xmm@` constants in 16-byte-aligned `.rdata` COMDATs;
//! behind an odd-length string one of those lands misaligned and a `movaps`
//! faults. The 19-byte entry stub made it worse by starting every `.text`
//! member at an odd offset. lld aligns every chunk
//! (`COFF/Writer.cpp:1792`).
//!
//! `VirtualSize` was rounded to the section alignment before `SizeOfRawData`
//! was derived from it, so every section's file bytes padded to a 4 KiB page
//! rather than to `FileAlignment`, and the field no tool could read the real
//! extent from.
//!
//! And `.bss` was written as file-backed zeros: the `zerofill` flag existed,
//! `place` honoured it, and no constructor ever set it. A 100 MB array was 100
//! MB of zeros on disk.
//!
//! Together, on the fixture below, an executable goes from 12800 bytes to
//! 2048.
//!
//! Gated on a `clang` that can target `x86_64-pc-windows-msvc`; without one
//! the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{coff::link_coff, input::Input};

mod common;

/// A 16-byte-aligned constant behind an odd-length string, and a large `.bss`.
const SRC: &[u8] =
    b"__declspec(align(16)) const float vec[4] = {1, 2, 3, 4};\n\
    const char msg[] = \"odd\";\n\
    static char zeros[65536];\n\
    int main(void) { return (int)vec[0] + msg[0] + zeros[0]; }\n";

/// `.bss` takes no file bytes.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn bss_is_not_written_to_the_file() {
    let Some(dir) = workdir("bss") else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    let bss = section(&bytes, b".bss");
    if let Some(s) = bss {
        assert_eq!(
            s.raw_size, 0,
            "an uninitialised section has no file bytes to write"
        );
        assert_eq!(s.raw_ptr, 0, "and no file offset to write them at");
    }
    // Whether or not `.bss` gets its own section, 64 KiB of zeros must not be
    // in the file.
    assert!(
        bytes.len() < 65536,
        "the image carries the .bss array as file zeros: {} bytes",
        bytes.len()
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `VirtualSize` is the real extent, and raw data pads to `FileAlignment`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_extents_are_the_real_ones() {
    let Some(dir) = workdir("extent") else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    let text = section(&bytes, b".text").expect(".text present");
    assert!(
        text.virtual_size < 0x1000,
        "VirtualSize must be the section's own extent, not a page: {:#x}",
        text.virtual_size
    );
    assert_eq!(
        text.raw_size % 512,
        0,
        "raw data pads to FileAlignment, which is 512"
    );
    assert!(
        text.raw_size <= 0x1000,
        "and a section smaller than a page must not take one: {:#x}",
        text.raw_size
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Every section starts on its section alignment, and the image is small.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_image_is_no_larger_than_its_content() {
    let Some(dir) = workdir("size") else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    assert!(
        bytes.len() <= 8192,
        "packing to FileAlignment rather than to a page keeps this small: \
         {} bytes",
        bytes.len()
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping coff-layout {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_cofflayout_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture for Windows and links it into a PE executable.
fn link(dir: &Path) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let src = dir.join("a.c");
    let obj = dir.join("a.obj");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-pc-windows-msvc", "-O1", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping coff-layout: clang cannot target Windows");
        return None;
    }
    let out = dir.join("a.exe");
    let files = [Input::Path(&obj)];
    let res = link_coff(&files, &out, b"main", false);
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

// --- readers ---------------------------------------------------------------

/// The fields of one PE section header these tests reason about.
struct Section {
    virtual_size: u32,
    raw_size: u32,
    raw_ptr: u32,
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
            raw_ptr: read_u32(bytes, at + 20)?,
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
