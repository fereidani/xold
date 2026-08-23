//! A Mach-O section-relative relocation is stored in the input's frame.
//!
//! For `r_extern == 0` the entry names its referent by section ordinal, and
//! the bytes at the site encode the target's address *in the input file*: an
//! absolute one for a non-PC-relative entry, and one relative to the end of
//! its own field for a PC-relative entry. The resolver answers with the
//! referent section's *output* address, so the difference between those two
//! frames has to come out of the addend, which is what lld's two formulas do
//! (`MachO/InputFiles.cpp:588-603`).
//!
//! Neither was applied. Every `-O0` string literal -- an `L_` label with no
//! nlist entry, reached by a `SIGNED` relocation against `__cstring` --
//! landed off by the distance between the two sections in the input file. On
//! the fixture below that is 34 bytes.
//!
//! Darwin images cannot run on this host, so the test computes where the
//! relocation points and checks that it lands inside the section it names.
//!
//! Gated on a `clang` that can target darwin; without one the tests print a
//! note and return.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{
    input::Input,
    macho::{MachOFile, link_macho},
    mmap_file::MappedFile,
};

mod common;

/// A string literal, which is what clang puts in `__cstring` behind an `L_`
/// label with no symbol-table entry of its own.
const SRC: &[u8] =
    b"const char *msg(void) { return \"hello from a cstring\"; }\n\
    int main(void) { return msg()[0]; }\n";

/// The relocation points into the section it names.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_section_relative_reference_lands_in_its_referent() {
    let Some((dir, obj, image)) = link("cstring") else {
        return;
    };
    let mapped = MappedFile::open(&obj).expect("the fixture must map");
    let input = MachOFile::parse(mapped.bytes()).expect("must parse");
    let sections = input.sections();
    let text_in = sections
        .iter()
        .find(|s| s.sectname == b"__text")
        .expect("__text in the input");
    let site = text_in
        .relocations
        .iter()
        .find(|r| !r.r_extern && r.r_pcrel)
        .expect("a PC-relative section reference, which -O0 produces");

    let (text_out, _) =
        segment_section(&image, b"__text").expect("__text in the image");
    let (cstring_out, cstring_size) =
        segment_section(&image, b"__cstring").expect("__cstring in the image");

    let at = usize::try_from(text_out.1 + u64::from(site.r_address))
        .expect("file offset");
    let stored = read_i32(&image, at).expect("the fixup site");
    // A 4-byte PC-relative field addresses its own end.
    let target = text_out
        .0
        .wrapping_add(u64::from(site.r_address))
        .wrapping_add(4)
        .wrapping_add(i64::from(stored).cast_unsigned());

    assert!(
        (cstring_out.0..cstring_out.0 + cstring_size).contains(&target),
        "the reference must land inside __cstring [{:#x}, {:#x}), got \
         {target:#x} -- not subtracting the referent's input address leaves \
         it off by the distance between the two sections in the input file",
        cstring_out.0,
        cstring_out.0 + cstring_size
    );
    let _ = fs::remove_dir_all(&dir);
}

/// And it lands at the start of the string, not somewhere inside it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn it_lands_on_the_string_itself() {
    let Some((dir, _obj, image)) = link("exact") else {
        return;
    };
    let (cstring_out, _) =
        segment_section(&image, b"__cstring").expect("__cstring in the image");
    let at = usize::try_from(cstring_out.1).expect("file offset");
    let bytes = image.get(at..at + 5).expect("the string bytes");
    assert_eq!(
        bytes, b"hello",
        "the fixture's only string is at the start of __cstring, so a \
         reference to it must resolve to that address"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Compiles and links the fixture, returning the object path and the image.
fn link(prefix: &str) -> Option<(PathBuf, PathBuf, Vec<u8>)> {
    let clang = which("clang")?;
    let dir = std::env::temp_dir()
        .join(format!("xold_machosec_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    let src = dir.join("s.c");
    let obj = dir.join("s.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-apple-darwin", "-O0", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping macho-section-reloc {prefix}: no darwin target");
        let _ = fs::remove_dir_all(&dir);
        return None;
    }
    let out = dir.join("prog");
    let files = [Input::Path(&obj)];
    let res = link_macho(&files, &out, b"_main");
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let image = fs::read(&out).ok()?;
    Some((dir, obj, image))
}

// --- readers ---------------------------------------------------------------

/// `((vmaddr, fileoff), size)` of a section in the linked image, by name.
fn segment_section(image: &[u8], name: &[u8]) -> Option<((u64, u64), u64)> {
    const LC_SEGMENT_64: u32 = 0x19;
    let ncmds = usize::try_from(read_u32(image, 16)?).ok()?;
    let mut at = 32;
    for _ in 0..ncmds {
        let cmd = read_u32(image, at)?;
        let size = usize::try_from(read_u32(image, at + 4)?).ok()?;
        if cmd == LC_SEGMENT_64 {
            let nsects = usize::try_from(read_u32(image, at + 64)?).ok()?;
            for i in 0..nsects {
                let sh = at + 72 + i * 80;
                let cell = image.get(sh..sh + 16)?;
                let trimmed = cell.split(|&b| b == 0).next().unwrap_or(cell);
                if trimmed == name {
                    return Some((
                        (
                            read_u64(image, sh + 32)?,
                            read_u32(image, sh + 48)?.into(),
                        ),
                        read_u64(image, sh + 40)?,
                    ));
                }
            }
        }
        if size < 8 {
            return None;
        }
        at += size;
    }
    None
}

fn read_i32(bytes: &[u8], at: usize) -> Option<i32> {
    bytes
        .get(at..at + 4)
        .and_then(|c| <[u8; 4]>::try_from(c).ok())
        .map(i32::from_le_bytes)
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
