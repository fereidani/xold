//! An image with no dynamic relocations carries no `.rela.dyn` tags.
//!
//! `DT_RELA`, `DT_RELASZ` and `DT_RELAENT` describe a table the loader walks;
//! when the link produced no dynamic relocations there is nothing to walk, and
//! lld emits none of the three -- its `.dynamic` writer gates the group on
//! `part.relaDyn->isNeeded()`, which is false for an empty table
//! (`lld/ELF/SyntheticSections.cpp`). A `DT_RELA` that points
//! at a zero-length table is noise every strict consumer still has to parse,
//! and a zero-sized `.rela.dyn` section beside it is a section header carrying
//! no bytes.
//!
//! xold pushed the three tags unconditionally, so a shared object whose code
//! touched nothing outside itself advertised an empty relocation table
//! anyway.
//!
//! The fixture links a leaf shared object -- one function returning a
//! constant, no globals, no imports -- and reads `.dynamic` back: none of the
//! three tags is present, and no `.rela.dyn` section header exists.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_shared};

mod common;

/// `DT_RELA`.
const DT_RELA: u64 = 7;
/// `DT_RELAENT`.
const DT_RELAENT: u64 = 9;
/// `DT_RELASZ`.
const DT_RELASZ: u64 = 8;

/// A shared object with nothing to relocate names no relocation table.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_empty_rela_table_is_not_advertised() {
    let Some(dir) = workdir() else {
        return;
    };
    let Some(lib) = build(&dir) else {
        return;
    };
    let out = dir.join("leaf.so");
    let res = link_shared(
        &[lib],
        &out,
        Some(b"leaf.so"),
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let tags = dynamic_tags(&out);
    assert!(
        !tags.contains(&DT_RELA),
        "DT_RELA names a table that does not exist"
    );
    assert!(
        !tags.contains(&DT_RELASZ),
        "DT_RELASZ sizes a table that does not exist"
    );
    assert!(
        !tags.contains(&DT_RELAENT),
        "DT_RELAENT sizes a table that does not exist"
    );
    assert!(
        !has_section(&out, b".rela.dyn"),
        "no relocation rows means no .rela.dyn section"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir() -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping empty-rela: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_emptyrela_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the leaf object: a function touching nothing outside itself.
fn build(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("leaf.c");
    let obj = dir.join("leaf.o");
    fs::write(&src, b"int leaf(void) { return 3; }\n").ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c", "-fPIC"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping empty-rela: clang cannot build the object");
        return None;
    }
    Some(obj)
}

/// Reads the `d_tag` column of `.dynamic`.
fn dynamic_tags(out: &Path) -> Vec<u64> {
    let bytes = fs::read(out).expect("read the image");
    let image = ObjectFile::parse(&bytes).expect("parse the image");
    let Some(dynamic) = image
        .sections()
        .iter()
        .find(|s| image.section_name(s) == b".dynamic")
    else {
        return Vec::new();
    };
    let data = image.section_data(dynamic).expect("dynamic bytes");
    let mut tags = Vec::new();
    for chunk in data.as_chunks::<16>().0 {
        let tag = u64::from_le_bytes(
            <[u8; 8]>::try_from(&chunk[..8]).expect("a tag"),
        );
        if tag == 0 {
            break;
        }
        tags.push(tag);
    }
    tags
}

/// Whether the image carries a section with this name.
fn has_section(out: &Path, name: &[u8]) -> bool {
    let bytes = fs::read(out).expect("read the image");
    let image = ObjectFile::parse(&bytes).expect("parse the image");
    image
        .sections()
        .iter()
        .any(|s| image.section_name(s) == name)
}
