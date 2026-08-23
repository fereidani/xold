//! A synthetic section's `sh_size` is what was written into it.
//!
//! The sizes of `.rela.dyn`, `.dynsym`, `.dynstr`, `.hash`, `.gnu.hash`,
//! `.gnu.version` and `.gnu.version_r` come from a probe that counts what the
//! symbol set implies, run before placement so the regions can be reserved.
//! Emission can legitimately come up short of that count -- a data relocation
//! whose section `--gc-sections` dropped, a hidden weak undefined the emitter
//! resolves without a row, the gap between what the TLS counter reserves and
//! what the TLS emitter writes.
//!
//! Only `.dynamic` was tightened afterwards. The rest kept the probe's number,
//! so `readelf -r` listed trailing `R_X86_64_NONE` rows that are not
//! relocations, and `.dynsym` carried null entries past the count `.hash`'s
//! `nchain` gives for the same table. Harmless to a loader, wrong to every
//! validator, and noise in any byte-level comparison of two images.
//!
//! Nothing moves when a region shrinks: they are already placed, so a shorter
//! section leaves padding before the next rather than pulling it back.
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

/// A dead pointer beside a live one. `--gc-sections` drops the dead data
/// section, and with it the `RELATIVE` relocation the probe counted for it.
/// `used` keeps the compiler from removing it first, so the collection is the
/// linker's.
const SRC: &[u8] = b"int live_val = 1;\n\
    int *live_ptr = &live_val;\n\
    static int dead_val = 2;\n\
    __attribute__((used)) static int *dead_ptr = &dead_val;\n\
    int keep(void) { return *live_ptr; }\n";

/// `.rela.dyn` holds no row that is not a relocation.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn rela_dyn_carries_no_trailing_null_rows() {
    let Some(dir) = workdir("rela") else {
        return;
    };
    let Some(bytes) = link(&dir, true) else {
        return;
    };
    let rows = rela_dyn_types(&bytes);
    assert_ne!(rows.len(), 0, "the fixture must emit some relocations");
    assert!(
        rows.iter().all(|t| *t != 0),
        "every row must be a relocation; a zero type is reserved space the \
         probe counted and the emitter did not use: {rows:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `sh_size` is a whole number of entries and matches the rows that decode.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn every_synthetic_section_is_sized_to_its_content() {
    let Some(dir) = workdir("sizes") else {
        return;
    };
    let Some(bytes) = link(&dir, true) else {
        return;
    };
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let dynsym = size_of_section(&obj, b".dynsym");
    let hash_nchain = hash_nchain(&bytes).expect(".hash present");
    assert_eq!(
        dynsym / 24,
        u64::from(hash_nchain),
        ".dynsym must hold exactly the symbols .hash chains; a longer table \
         is null entries past nchain"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: without collection the same link emits every row the probe
/// counted, so the tightening removes slack rather than content.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn nothing_is_lost_when_the_probe_was_right() {
    let Some(dir) = workdir("nogc") else {
        return;
    };
    let Some(with_gc) = link(&dir, true) else {
        return;
    };
    let Some(without) = link(&dir, false) else {
        return;
    };
    let kept = rela_dyn_types(&without);
    let after = rela_dyn_types(&with_gc);
    assert!(
        kept.len() > after.len(),
        "collection must actually drop a relocation, or this fixture proves \
         nothing: {} without, {} with",
        kept.len(),
        after.len()
    );
    assert!(
        kept.iter().all(|t| *t != 0),
        "and the uncollected link must be exact too: {kept:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping dynamic-region-size {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_regionsize_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the fixture and links it as a shared object, with or without
/// collection.
fn link(dir: &Path, gc: bool) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let src = dir.join("g.c");
    let obj = dir.join("g.o");
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
        eprintln!("skipping dynamic-region-size: clang cannot build it");
        return None;
    }
    let out = dir.join(if gc { "gc.so" } else { "plain.so" });
    let res = link_shared(
        std::slice::from_ref(&obj),
        &out,
        Some(b"libregion.so"),
        gc,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

// --- readers ---------------------------------------------------------------

/// The `r_type` of every `.rela.dyn` row, read over the section's `sh_size`.
fn rela_dyn_types(bytes: &[u8]) -> Vec<u32> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(shdr) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".rela.dyn")
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(shdr) else {
        return Vec::new();
    };
    data.as_chunks::<24>()
        .0
        .iter()
        .filter_map(|c| {
            let info = u64::from_le_bytes(<[u8; 8]>::try_from(&c[8..16]).ok()?);
            #[allow(clippy::cast_possible_truncation)]
            Some(info as u32)
        })
        .collect()
}

/// The `sh_size` of a named section, or zero when it is absent.
fn size_of_section(obj: &ObjectFile<'_>, name: &[u8]) -> u64 {
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == name)
        .map_or(0, |s| s.sh_size.get())
}

/// `.hash`'s `nchain`, the number of symbols the table chains. It is the
/// second word of the section.
fn hash_nchain(bytes: &[u8]) -> Option<u32> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".hash")?;
    let data = obj.section_data(shdr).ok()?;
    data.get(4..8)
        .and_then(|c| <[u8; 4]>::try_from(c).ok())
        .map(u32::from_le_bytes)
}
