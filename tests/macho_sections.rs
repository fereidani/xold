//! Mach-O sections outside a fixed list are linked, and private GOT referents
//! get their own slots.
//!
//! `is_linkable` whitelisted eleven section names. Anything else in `__TEXT`
//! or `__DATA` -- a user's `__attribute__((section("__DATA,__mine")))`, an
//! `__mod_init_func`, any `__objc_*` -- was dropped with no diagnostic, and a
//! reference into a dropped section resolved against a base of zero. The
//! question is which sections the link *defers* (unwind, debug, dynamic-linker
//! metadata a static image has no use for), so that is what the check asks
//! now.
//!
//! `__got` slots were keyed by symbol name alone. A file-private symbol means
//! nothing outside its own object, so two files' `static counter` shared one
//! slot -- one variable where the program declared two. A private referent is
//! keyed by its file and index instead.
//!
//! Darwin images cannot run on this host, so these read the section table and
//! the GOT.
//!
//! Gated on a `clang` that can target darwin; without one the tests print a
//! note and return.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{input::Input, macho::link_macho};

mod common;

/// A datum in a section the linker was never told about.
const NAMED: &[u8] =
    b"__attribute__((section(\"__DATA,__mine\"))) int marked = 0x5150;\n\
    int read_marked(void) { return marked; }\n\
    int main(void) { return read_marked() == 0x5150 ? 0 : 1; }\n";

/// Two files, each with a file-private datum of the same name reached through
/// the GOT.
const PRIV_A: &[u8] = b"static int counter = 11;\n\
    int *a_ptr(void) { return &counter; }\n\
    int *b_ptr(void);\n\
    int main(void) { return a_ptr() == b_ptr(); }\n";
const PRIV_B: &[u8] = b"static int counter = 22;\n\
    int *b_ptr(void) { return &counter; }\n";

const LC_SEGMENT_64: u32 = 0x19;

/// A section the linker has no name for is still placed.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_unknown_section_is_linked() {
    let Some((dir, image)) = link("named", &[("u", NAMED)]) else {
        return;
    };
    let (addr, size) =
        section(&image, b"__mine").expect("the user's section must be placed");
    assert_eq!(size, 4, "it carries its one int");
    assert!(addr > 0x1_0000_0000, "at a real address, not zero");
    let _ = fs::remove_dir_all(&dir);
}

/// The deferred sections are still deferred.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_deferred_sections_are_still_dropped() {
    let Some((dir, image)) = link("deferred", &[("u", NAMED)]) else {
        return;
    };
    for name in [
        b"__eh_frame".as_slice(),
        b"__compact_unwind",
        b"__debug_str",
    ] {
        assert!(
            section(&image, name).is_none(),
            "{} needs machinery this phase does not have",
            String::from_utf8_lossy(name)
        );
    }
    // And the ordinary ones are still linked.
    assert!(section(&image, b"__text").is_some(), "__text is linked");
    let _ = fs::remove_dir_all(&dir);
}

/// Two files' private symbols of one name do not share a GOT slot.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn private_referents_get_their_own_slots() {
    let Some((dir, image)) = link("private", &[("a", PRIV_A), ("b", PRIV_B)])
    else {
        return;
    };
    let Some((_, size)) = section(&image, b"__got") else {
        // No GOT relocations on this host's codegen: nothing to check, and
        // saying so beats asserting a property the fixture never exercised.
        eprintln!("skipping private-referent check: no __got in the image");
        let _ = fs::remove_dir_all(&dir);
        return;
    };
    assert!(
        size >= 16,
        "two distinct private referents need two slots, not one: {size} bytes"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Compiles the sources for darwin and links them.
fn link(prefix: &str, sources: &[(&str, &[u8])]) -> Option<(PathBuf, Vec<u8>)> {
    let clang = which("clang")?;
    let dir = std::env::temp_dir()
        .join(format!("xold_machosecs_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    let mut objs = Vec::new();
    for (stem, src) in sources {
        let path = dir.join(format!("{stem}.c"));
        let obj = dir.join(format!("{stem}.o"));
        fs::write(&path, src).ok()?;
        let built = Command::new(&clang)
            .args(["--target=x86_64-apple-darwin", "-O1", "-c"])
            .arg(&path)
            .arg("-o")
            .arg(&obj)
            .status()
            .ok()?
            .success();
        if !built {
            eprintln!("skipping macho-sections {prefix}: no darwin target");
            let _ = fs::remove_dir_all(&dir);
            return None;
        }
        objs.push(obj);
    }
    let files: Vec<Input<'_>> =
        objs.iter().map(|p| Input::Path(p.as_path())).collect();
    let out = dir.join("prog");
    let res = link_macho(&files, &out, b"_main");
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let image = fs::read(&out).ok()?;
    Some((dir, image))
}

// --- readers ---------------------------------------------------------------

/// `(addr, size)` of a section in the linked image, by name.
fn section(image: &[u8], name: &[u8]) -> Option<(u64, u64)> {
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
                        read_u64(image, sh + 32)?,
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
