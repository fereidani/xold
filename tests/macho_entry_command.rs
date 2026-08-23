//! A static Mach-O image launches without a dynamic linker, and covers the
//! null page.
//!
//! `LC_MAIN` sets `LC_REQ_DYLD`, and XNU's `load_main` demands a
//! `LC_LOAD_DYLINKER` beside it. The images this linker produces are static:
//! `MH_NOUNDEFS`, no dylinker command. Carrying `LC_MAIN` meant they could
//! never exec on darwin. lld pairs `LC_MAIN` with a dylinker only; a true
//! static image gets `LC_UNIXTHREAD`, whose register state says where to
//! begin.
//!
//! And `__TEXT` sat at 4 GiB with nothing covering `[0, 4 GiB)`, so the null
//! region was left mappable. ld64 and lld emit `__PAGEZERO` unconditionally
//! for an executable; strict tooling treats an image without it as malformed.
//!
//! Darwin images cannot run on this host, so these read the load commands the
//! kernel would.
//!
//! Gated on a `clang` that can target darwin; without one the tests print a
//! note and return.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{input::Input, macho::link_macho};

mod common;

const SRC: &[u8] = b"int g(void) { return 7; }\n\
    int main(void) { return g(); }\n";

const LC_SEGMENT_64: u32 = 0x19;
const LC_UNIXTHREAD: u32 = 0x05;
const LC_MAIN: u32 = 0x28 | 0x8000_0000;
const TEXT_BASE: u64 = 0x1_0000_0000;

/// The image carries `LC_UNIXTHREAD`, not `LC_MAIN`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_entry_is_a_thread_state() {
    let Some((dir, image)) = link("thread") else {
        return;
    };
    assert!(
        command(&image, LC_MAIN).is_none(),
        "LC_MAIN needs a dylinker beside it, and this image has none"
    );
    let at = command(&image, LC_UNIXTHREAD).expect("LC_UNIXTHREAD present");
    let count = read_u32(&image, at + 12).expect("state word count");
    let size = read_u32(&image, at + 4).expect("cmdsize");
    assert_eq!(
        size,
        16 + 4 * count,
        "the command covers its own header and the state that follows"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Its program counter is the entry's virtual address.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_program_counter_points_at_the_entry() {
    let Some((dir, image)) = link("pc") else {
        return;
    };
    let at = command(&image, LC_UNIXTHREAD).expect("LC_UNIXTHREAD present");
    // x86_64: rip is the 17th of 21 registers, counting from rax.
    let pc = read_u64(&image, at + 16 + 16 * 8).expect("rip");
    assert!(
        pc > TEXT_BASE,
        "the program counter is a virtual address in __TEXT, not the file \
         offset LC_MAIN took: {pc:#x}"
    );
    assert!(
        pc < TEXT_BASE + 0x10_0000,
        "and it is inside this small image: {pc:#x}"
    );
    // Every other register starts zeroed.
    for slot in [0usize, 7, 20] {
        let value = read_u64(&image, at + 16 + slot * 8).expect("register");
        assert_eq!(value, 0, "register {slot} starts zeroed");
    }
    let _ = fs::remove_dir_all(&dir);
}

/// And `__PAGEZERO` covers the low 4 GiB with no protection.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_null_page_is_covered() {
    let Some((dir, image)) = link("pagezero") else {
        return;
    };
    let at = segment(&image, b"__PAGEZERO").expect("__PAGEZERO present");
    assert_eq!(read_u64(&image, at + 24), Some(0), "it starts at zero");
    assert_eq!(
        read_u64(&image, at + 32),
        Some(TEXT_BASE),
        "and runs to where __TEXT begins"
    );
    assert_eq!(read_u64(&image, at + 48), Some(0), "it has no file bytes");
    assert_eq!(read_u32(&image, at + 56), Some(0), "and no protection");
    assert_eq!(read_u32(&image, at + 60), Some(0), "initial or maximum");
    // It comes first, so nothing else can claim the low addresses.
    assert_eq!(at, 32, "it is the first load command after the header");
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Compiles and links the fixture for darwin, or `None` (after printing a
/// note) when clang cannot target it.
fn link(prefix: &str) -> Option<(PathBuf, Vec<u8>)> {
    let clang = which("clang")?;
    let dir = std::env::temp_dir()
        .join(format!("xold_machoentry_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    let src = dir.join("e.c");
    let obj = dir.join("e.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-apple-darwin", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping macho-entry-command {prefix}: no darwin target");
        let _ = fs::remove_dir_all(&dir);
        return None;
    }
    let out = dir.join("prog");
    let files = [Input::Path(&obj)];
    let res = link_macho(&files, &out, b"_main");
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let bytes = fs::read(&out).ok()?;
    Some((dir, bytes))
}

// --- readers ---------------------------------------------------------------

/// The file offset of the first load command of type `want`.
fn command(image: &[u8], want: u32) -> Option<usize> {
    let ncmds = usize::try_from(read_u32(image, 16)?).ok()?;
    let mut at = 32;
    for _ in 0..ncmds {
        let cmd = read_u32(image, at)?;
        let size = usize::try_from(read_u32(image, at + 4)?).ok()?;
        if cmd == want {
            return Some(at);
        }
        if size < 8 {
            return None;
        }
        at += size;
    }
    None
}

/// The file offset of the `LC_SEGMENT_64` named `name`.
fn segment(image: &[u8], name: &[u8]) -> Option<usize> {
    let ncmds = usize::try_from(read_u32(image, 16)?).ok()?;
    let mut at = 32;
    for _ in 0..ncmds {
        let cmd = read_u32(image, at)?;
        let size = usize::try_from(read_u32(image, at + 4)?).ok()?;
        if cmd == LC_SEGMENT_64 {
            let cell = image.get(at + 8..at + 24)?;
            let trimmed = cell.split(|&b| b == 0).next().unwrap_or(cell);
            if trimmed == name {
                return Some(at);
            }
        }
        if size < 8 {
            return None;
        }
        at += size;
    }
    None
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
