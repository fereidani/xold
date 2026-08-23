//! The synthesized TLS directory's pointers get base relocations.
//!
//! `ImageTlsDirectory64` holds four absolute virtual addresses: the start and
//! end of the template, the address of `_tls_index`, and the callback array.
//! All four are stamped from the preferred image base. A DLL advertises
//! `DYNAMIC_BASE`, so the loader is free to map it elsewhere -- and then
//! dereferences four pointers into whatever now occupies the preferred base.
//!
//! The base-relocation collector walked input-section `ADDR64` sites only, and
//! the directory is synthesized rather than read from an input, so its four
//! fields were reachable by nothing. In lld the directory is the CRT's
//! `_tls_used` chunk and its own `ADDR64` relocations flow through the same
//! pass as everything else; here the sites are stated explicitly.
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

/// A thread-local, which is what makes the linker synthesize a directory.
const SRC: &[u8] = b"__declspec(thread) int slot = 3;\n\
    __declspec(dllexport) int get(void) { return slot; }\n\
    __declspec(dllexport) int set(int v) { slot = v; return slot; }\n";

/// Each of the four directory pointers carries a `DIR64` fixup.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn every_directory_pointer_is_relocated() {
    let Some(dir) = workdir("dll") else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    let (tlsdir_rva, _) = section_rva(&bytes, b".tlsdir")
        .expect("a TLS input must produce a .tlsdir section");
    let sites = fixup_rvas(&bytes);
    assert!(
        !sites.is_empty(),
        "a DLL with absolute addresses must carry a .reloc table"
    );
    for field in 0..4u32 {
        let want = tlsdir_rva + field * 8;
        assert!(
            sites.contains(&want),
            "the directory pointer at +{} has no base relocation; the loader \
             would read it at the preferred base after a rebase. Sites: {:x?}",
            field * 8,
            sites
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

/// The fixups name the directory, not the bytes after it: `_tls_index` and the
/// callbacks terminator are not addresses.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_trailer_dwords_are_not_relocated() {
    let Some(dir) = workdir("trailer") else {
        return;
    };
    let Some(bytes) = link(&dir) else {
        return;
    };
    let (tlsdir_rva, size) =
        section_rva(&bytes, b".tlsdir").expect(".tlsdir present");
    let sites = fixup_rvas(&bytes);
    let past = tlsdir_rva + 32;
    let end = tlsdir_rva + size;
    for site in &sites {
        assert!(
            !(past..end).contains(site),
            "{site:#x} is in the trailer, which holds a callback slot and a \
             dword, neither of which is an absolute address"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping coff-tls-basereloc {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_cofftlsreloc_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture for Windows and links it into a DLL.
fn link(dir: &Path) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let src = dir.join("t.c");
    let obj = dir.join("t.obj");
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
        eprintln!("skipping coff-tls-basereloc: clang cannot target Windows");
        return None;
    }
    let out = dir.join("t.dll");
    let files = [Input::Path(&obj)];
    let res = link_coff(&files, &out, b"get", true);
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

// --- readers ---------------------------------------------------------------

/// The RVA and virtual size of a section, by name.
fn section_rva(bytes: &[u8], name: &[u8]) -> Option<(u32, u32)> {
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
        Some((read_u32(bytes, at + 12)?, read_u32(bytes, at + 8)?))
    })
}

/// Every RVA the base-relocation table fixes up, decoded from the blocks.
fn fixup_rvas(bytes: &[u8]) -> Vec<u32> {
    let Some((rva, _)) = section_rva(bytes, b".reloc") else {
        return Vec::new();
    };
    let Some(off) = file_offset(bytes, rva) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut at = off;
    // One block per page: page RVA, block size, then WORD entries. The loop is
    // bounded by the image length, and every step advances by at least 8.
    while let (Some(page), Some(size)) =
        (read_u32(bytes, at), read_u32(bytes, at + 4))
    {
        let size = usize::try_from(size).unwrap_or(0);
        if size < 8 {
            break;
        }
        for e in (8..size).step_by(2) {
            let Some(entry) = read_u16(bytes, at + e) else {
                break;
            };
            // Type 10 is DIR64; type 0 is the ABSOLUTE padding the loader
            // ignores.
            if entry >> 12 == 10 {
                out.push(page + u32::from(entry & 0x0fff));
            }
        }
        at += size;
        if at >= bytes.len() {
            break;
        }
    }
    out
}

/// Maps an image RVA to its file offset through the section table.
fn file_offset(bytes: &[u8], rva: u32) -> Option<usize> {
    let pe = usize::try_from(read_u32(bytes, 0x3c)?).ok()?;
    let num = usize::from(read_u16(bytes, pe + 6)?);
    let opt = usize::from(read_u16(bytes, pe + 20)?);
    let first = pe + 24 + opt;
    (0..num).find_map(|i| {
        let at = first + i * 40;
        let start = read_u32(bytes, at + 12)?;
        let size = read_u32(bytes, at + 8)?.max(read_u32(bytes, at + 16)?);
        if rva < start || rva >= start + size {
            return None;
        }
        let raw = read_u32(bytes, at + 20)?;
        usize::try_from(raw + (rva - start)).ok()
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
