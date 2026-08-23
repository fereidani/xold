//! A synthetic region that tightens to nothing keeps its header row.
//!
//! The dynamic regions are sized by a probe that runs before placement, and
//! the probe over-counts on purpose: it surveys the data relocations of every
//! allocated section, including the ones `--gc-sections` is about to drop,
//! because placement is not known yet. Emission walks only the sections the
//! collection kept, so `.rela.dyn` can shrink -- all the way to zero when
//! every data relocation lived in a dropped section.
//!
//! Placement stamped a section-header index for the region while the probe's
//! size still stood, and every later region's index was numbered beside it.
//! The header table was rebuilt after the shrink and skipped the now-empty
//! region, so every later header slid one row down: a symbol in `.data` named
//! `.dynamic`'s header in `st_shndx`, and `e_shnum` disagreed with the count
//! the image was planned for. The row is emitted for as long as its index is
//! stamped, with the tightened size, which is what lld's one-pass numbering
//! makes true by construction.
//!
//! Gated on `clang`; if it is missing the test prints a note and returns.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_shared};

mod common;

/// A live exported datum with no pointer in it, and a dead one whose only
/// content is a pointer. The collection drops `.data.dead_ptr`, and with it
/// the one `R_X86_64_64` the probe counted: after emission `.rela.dyn` is
/// empty while the region was reserved and indexed.
const SRC: &[u8] = b"int keep_fn(void) { return 7; }\n\
    int live = 3;\n\
    __attribute__((used)) static int *dead_ptr = &keep_fn;\n";

/// The emptied `.rela.dyn` keeps a header row, at the index stamped for it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_region_that_empties_keeps_its_header_row() {
    let Some(dir) = workdir("gc") else {
        return;
    };
    let Some(bytes) = link(&dir, true, "gc.so") else {
        return;
    };
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let rela_at = find_section(&obj, b".rela.dyn")
        .expect("the emptied region keeps its header row");
    let rela = &obj.sections()[rela_at];
    assert_eq!(
        rela.sh_size.get(),
        0,
        "the collection dropped the only data relocation"
    );
    // Every later index still names the section it was stamped for. `live`
    // sits in `.data`, whose header was numbered after `.rela.dyn`.
    let symtab = obj.symbol_table().ok().flatten().expect(".symtab");
    let live = symtab
        .iter()
        .find(|s| symtab.name(s) == b"live")
        .expect("live is defined");
    let named = obj
        .sections()
        .get(usize::from(live.st_shndx.get()))
        .expect("the index selects a header");
    assert_eq!(
        obj.section_name(named),
        b".data",
        "a symbol's st_shndx names the section that holds it"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: without collection the same link emits the row the probe
/// counted, so the empty region above is the tightening and not a fixture
/// that never had a data relocation.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_uncollected_link_still_emits_it() {
    let Some(dir) = workdir("plain") else {
        return;
    };
    let Some(bytes) = link(&dir, false, "plain.so") else {
        return;
    };
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let rela_at = find_section(&obj, b".rela.dyn")
        .expect("the uncollected link reserves and fills the region");
    let rela = &obj.sections()[rela_at];
    assert!(
        rela.sh_size.get() > 0,
        "the dead pointer's relocation is emitted when nothing is collected"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping dynamic-zero-cross {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_zerocross_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the fixture and links it as a shared object, with or without
/// collection.
fn link(dir: &Path, gc: bool, name: &str) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let src = dir.join("z.c");
    let obj = dir.join("z.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args([
            "--target=x86_64-linux-gnu",
            "-fPIC",
            "-ffunction-sections",
            "-fdata-sections",
            "-c",
        ])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping dynamic-zero-cross: clang cannot build it");
        return None;
    }
    let out = dir.join(name);
    let res = link_shared(
        std::slice::from_ref(&obj),
        &out,
        Some(b"libzero.so"),
        gc,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

// --- readers ---------------------------------------------------------------

/// The header index of a named output section.
fn find_section(obj: &ObjectFile<'_>, name: &[u8]) -> Option<usize> {
    obj.sections()
        .iter()
        .position(|s| obj.section_name(s) == name)
}
