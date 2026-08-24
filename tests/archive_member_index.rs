//! An archive member's relocations are indexed once, not found by re-parsing.
//!
//! A borrowed input carries a `target section -> relocations` table built when
//! it is opened. A member owns its bytes, so a slice into them cannot be
//! stored beside them, and the lookup re-parsed the member and re-walked its
//! whole section table for every query instead -- once per section, per pass,
//! per member. A `-ffunction-sections` static archive is exactly the shape
//! that makes that quadratic.
//!
//! Offsets can be stored beside the bytes, and they answer the same question.
//! On forty members of four hundred functions each the link goes from 0.0602s
//! to 0.0382s.
//!
//! What is checked here is that the index answers what the re-parse did: the
//! same entries for a section that has them, nothing for one that does not,
//! and the member still links.
//!
//! Gated on `clang` and an ELF-capable `ar`; without them the tests print a
//! note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{archive_tool, which};
use xold::{icf::IcfMode, input::InputFile, linker::link_to};

mod common;

/// A member with relocations in `.text` and a section with none.
const MEMBER: &[u8] = b"int datum = 7;\n\
    extern int outside(int);\n\
    int pulled(int x) { return outside(x) + datum; }\n";
const MAIN: &[u8] = b"int pulled(int);\n\
    int outside(int x) { return x + 1; }\n\
    int main(void) { return pulled(1) - 9; }\n\
    void _start(void) { }\n";

/// The cached index reports the same relocations the object does.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_index_agrees_with_the_object() {
    let Some(dir) = workdir("agree") else {
        return;
    };
    let Some(bytes) = member_bytes(&dir) else {
        return;
    };
    let file = InputFile::from_member(Path::new("(m.o)"), &bytes)
        .expect("the member must parse");
    let obj = file.object().expect("the member must view");
    let count = obj.sections().len();
    assert!(count > 2, "the fixture must carry sections");
    let mut with_relocs = 0;
    for i in 0..count {
        let shndx = u16::try_from(i).expect("section index");
        let cached = file.relocations(shndx).expect("index must answer");
        let direct = obj.relocations(shndx).expect("object must answer");
        assert_eq!(
            cached.map(<[_]>::len),
            direct.map(<[_]>::len),
            "section {i}: the index and the object must agree on the count"
        );
        if let (Some(a), Some(b)) = (cached, direct) {
            with_relocs += 1;
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(x.r_offset.get(), y.r_offset.get());
                assert_eq!(x.r_info.get(), y.r_info.get());
                assert_eq!(x.r_addend.get(), y.r_addend.get());
            }
        }
    }
    assert!(
        with_relocs > 0,
        "the fixture must have at least one relocated section for this to \
         mean anything"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// An index past the section table answers nothing rather than panicking.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_section_past_the_end_has_no_relocations() {
    let Some(dir) = workdir("bounds") else {
        return;
    };
    let Some(bytes) = member_bytes(&dir) else {
        return;
    };
    let file = InputFile::from_member(Path::new("(m.o)"), &bytes)
        .expect("the member must parse");
    assert!(
        file.relocations(u16::MAX).expect("must answer").is_none(),
        "a section number no file has cannot have relocations"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// And a real archive link still resolves through a pulled member.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_archive_member_still_links() {
    let Some(dir) = workdir("link") else {
        return;
    };
    let Some(image) = link_archive(&dir) else {
        return;
    };
    assert_eq!(
        image.get(..4),
        Some(b"\x7fELF".as_slice()),
        "the link must produce an image"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping archive-member-index {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_arindex_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the member and returns its bytes.
fn member_bytes(dir: &Path) -> Option<Vec<u8>> {
    let obj = compile(dir, MEMBER, "m")?;
    fs::read(&obj).ok()
}

/// Builds an archive around the member and links a program against it.
fn link_archive(dir: &Path) -> Option<Vec<u8>> {
    let member = compile(dir, MEMBER, "m")?;
    let main = compile(dir, MAIN, "main")?;
    let ar = archive_tool()?;
    let lib = dir.join("lib.a");
    let made = Command::new(ar)
        .arg("rcs")
        .arg(&lib)
        .arg(&member)
        .status()
        .ok()?
        .success();
    if !made {
        eprintln!("skipping archive-member-index: ar failed");
        return None;
    }
    let out = dir.join("prog");
    let res =
        link_to(&[main, lib], &out, b"_start", false, IcfMode::None, false);
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

/// Compiles one fixture.
fn compile(dir: &Path, src: &[u8], stem: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let path = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&path, src).ok()?;
    let built = Command::new(clang)
        .args([
            "--target=x86_64-linux-gnu",
            "-fno-pic",
            "-O1",
            "-ffunction-sections",
            "-fdata-sections",
            "-c",
        ])
        .arg(&path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping archive-member-index: clang cannot build it");
        return None;
    }
    Some(obj)
}
