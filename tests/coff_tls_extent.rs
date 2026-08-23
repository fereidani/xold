//! The PE TLS directory's extent must cover the laid-out `.tls`, alignment
//! gaps included.
//!
//! `.tls$` members land in the output `.tls` at offsets rounded to each
//! member's alignment (the PE default is 16), so the section's virtual size
//! is a sum of members *plus gaps*. The directory's
//! `EndAddressOfRawData - StartAddressOfRawData` is the byte count the
//! loader copies into every new thread's block. Computing it as the sum of
//! raw member sizes under-counted by the gaps: two 4-byte variables at
//! offsets 0 and 16 described an 8-byte template, and the loader handed every
//! thread an 8-byte block while code addressed the second variable at +16 --
//! a silent out-of-bounds access in each thread. lld derives the extent from
//! the CRT's `_tls_used` whose bounds relocations point at the merged
//! section (`COFF/Writer.cpp`).
//!
//! Gated on `clang --target=x86_64-pc-windows-msvc`; without it the test
//! prints a note and returns.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{which, xold_bin};

mod common;

/// Two undersized `.tls$` members at 16-byte alignment: the second lands at
/// offset 16, so the laid-out section is 20 bytes while the raw members sum
/// to 8.
const SRC: &[u8] = b".globl main\n\
    .text\n\
    main:\n\
    \tret\n\
    .section .tls$aaa,\"dw\"\n\
    .align 16\n\
    .long 0x11111111\n\
    .section .tls$bbb,\"dw\"\n\
    .align 16\n\
    .long 0x22222222\n";

/// The template extent equals the laid-out `.tls` virtual size.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_tls_directory_covers_the_laid_out_section() {
    let Some(ctx) = LinkContext::discover() else {
        return;
    };
    let obj = ctx.compile("xold_coff_tls_extent", SRC);
    let exe = ctx.link("xold_coff_tls_extent", &obj);
    let image = fs::read(&exe).expect("read the linked image");

    let tls = section(&image, b".tls").expect("the .tls template section");
    let (start, end) = tls_directory(&image).expect("the TLS directory");
    let extent = end
        .checked_sub(start)
        .expect("the directory's end is past its start");
    assert_eq!(
        extent,
        tls.1,
        "the per-thread block is the laid-out template: {extent} bytes for \
         a {size}-byte section leaves variables past the gap unreadable",
        size = tls.1
    );
}

// --- fixtures --------------------------------------------------------------

struct LinkContext {
    clang: PathBuf,
    dir: PathBuf,
}

impl LinkContext {
    /// Builds the context when the Windows clang target is available.
    fn discover() -> Option<Self> {
        let triple = "--target=x86_64-pc-windows-msvc";
        let clang = which("clang")?;
        let dir = std::env::temp_dir()
            .join(format!("xold_coff_tlsext_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).ok()?;
        let probe_src = dir.join("probe.c");
        fs::write(&probe_src, b"int probe(void);\n").ok()?;
        let probe = dir.join("probe.obj");
        let ok = Command::new(&clang)
            .args([triple, "-c"])
            .arg(&probe_src)
            .arg("-o")
            .arg(&probe)
            .status()
            .ok()?
            .success();
        if !ok {
            eprintln!("skipping coff tls extent: no msvc target");
            let _ = fs::remove_dir_all(&dir);
            return None;
        }
        Some(Self { clang, dir })
    }

    fn compile(&self, stem: &str, src: &[u8]) -> PathBuf {
        let triple = "--target=x86_64-pc-windows-msvc";
        let src_path = self.dir.join(format!("{stem}.s"));
        fs::write(&src_path, src).expect("write source");
        let obj = self.dir.join(format!("{stem}.obj"));
        let ok = Command::new(&self.clang)
            .args([triple, "-c"])
            .arg(&src_path)
            .arg("-o")
            .arg(&obj)
            .status()
            .is_ok_and(|s| s.success());
        assert!(ok, "clang assemble failed for {stem}");
        obj
    }

    fn link(&self, stem: &str, obj: &Path) -> PathBuf {
        let exe = self.dir.join(format!("{stem}.exe"));
        let ok = Command::new(xold_bin())
            .arg(obj)
            .args(["-o"])
            .arg(&exe)
            .status()
            .is_ok_and(|s| s.success());
        assert!(ok, "xold link failed for {stem}");
        exe
    }
}

// --- readers ---------------------------------------------------------------

/// `(virtual address, virtual size)` of a section, by name.
fn section(image: &[u8], name: &[u8]) -> Option<(u64, u64)> {
    let (_, headers) = headers(image)?;
    for sh in headers {
        let field = &sh[..8];
        let trimmed = field.split(|&b| b == 0).next().unwrap_or(field);
        if trimmed == name {
            return Some((
                u64::from(read_u32(&sh[12..])?),
                u64::from(read_u32(&sh[8..])?),
            ));
        }
    }
    None
}

/// `(StartAddressOfRawData, EndAddressOfRawData)` of the TLS directory
/// (data directory 9).
fn tls_directory(image: &[u8]) -> Option<(u64, u64)> {
    let (optional, headers) = headers(image)?;
    let dir_count = usize::try_from(read_u32(&optional[108..])?).ok()?;
    if dir_count <= 9 {
        return None;
    }
    let entry = &optional[112 + 9 * 8..];
    let rva = u64::from(read_u32(entry)?);
    let file = rva_to_offset(headers, rva)?;
    let at = usize::try_from(file).ok()?;
    let start = read_u64_at(image, at)?;
    let end = read_u64_at(image, at.checked_add(8)?)?;
    Some((start, end))
}

/// `(optional header, section headers)` of the image.
fn headers(image: &[u8]) -> Option<(&[u8], Vec<&[u8]>)> {
    let pe = usize::try_from(read_u32(&image[0x3c..])?).ok()?;
    let sig = image.get(pe..pe + 4)?;
    if sig != b"PE\0\0" {
        return None;
    }
    let coff = pe + 4;
    let header_size = usize::from(read_u16(&image[coff + 16..])?);
    let optional = image.get(coff + 20..coff + 20 + header_size)?;
    let count = usize::from(read_u16(&image[coff + 2..])?);
    let table = coff + 20 + header_size;
    let mut out = Vec::new();
    for i in 0..count {
        let at = table + i * 40;
        out.push(image.get(at..at + 40)?);
    }
    Some((optional, out))
}

/// The file offset of an RVA, via the section that contains it.
fn rva_to_offset(headers: Vec<&[u8]>, rva: u64) -> Option<u64> {
    for sh in headers {
        let start = u64::from(read_u32(&sh[12..])?);
        let size = u64::from(read_u32(&sh[8..])?);
        if rva >= start && rva < start.checked_add(size)? {
            let raw = u64::from(read_u32(&sh[20..])?);
            return raw.checked_add(rva - start);
        }
    }
    None
}

fn read_u16(bytes: &[u8]) -> Option<u16> {
    bytes
        .get(..2)
        .and_then(|c| <[u8; 2]>::try_from(c).ok())
        .map(u16::from_le_bytes)
}

fn read_u32(bytes: &[u8]) -> Option<u32> {
    bytes
        .get(..4)
        .and_then(|c| <[u8; 4]>::try_from(c).ok())
        .map(u32::from_le_bytes)
}

fn read_u64_at(image: &[u8], at: usize) -> Option<u64> {
    image
        .get(at..at + 8)
        .and_then(|c| <[u8; 8]>::try_from(c).ok())
        .map(u64::from_le_bytes)
}
