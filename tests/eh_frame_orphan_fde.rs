//! An FDE that cannot name its function does not reach the image.
//!
//! An FDE is kept when the section its first relocation names reached the
//! output. A record whose first relocation is absent, or names something no
//! live section backs, could not be resolved at all -- and was kept anyway:
//! the keep-flags started all-true and an unresolvable record fell through.
//!
//! Its `initial_location` was written against the *input* file and re-emitted
//! verbatim, so the image carried an unwind entry claiming a range that
//! belongs to nothing. `.eh_frame_hdr`'s own filter drops such an entry from
//! the binary search, which is why it went unnoticed, but the search table is
//! not the only reader: a linear scanner walks the records themselves, and
//! static glibc's `classify_object_over_fdes` is one. It ingests the bogus
//! range.
//!
//! lld drops the same shape in `isFdeLive`.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_to};

mod common;

/// Three functions, so three FDEs; one of them is orphaned by the patch below.
const SRC: &[u8] = b"int p(int x) { return x + 1; }\n\
    int q(int x) { return x * 3; }\n\
    int main(void) { return p(1) + q(2) - 8; }\n\
    void _start(void) { }\n";

/// The orphaned record is dropped, so every FDE in the image describes code
/// that is in the image.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_unresolvable_fde_is_dropped() {
    let Some(dir) = workdir("drop") else {
        return;
    };
    let Some(intact) = link(&dir, false) else {
        return;
    };
    let Some(patched) = link(&dir, true) else {
        return;
    };
    let before = count_fdes(&intact);
    let after = count_fdes(&patched);
    assert!(before >= 3, "the fixture must produce FDEs: {before}");
    assert_eq!(
        after,
        before - 1,
        "the record whose relocation names no part of the section cannot be \
         placed, so it must not be written; keeping it leaves an unwind entry \
         over a range from the input file"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The rest survive: the drop is about the one record that cannot be resolved.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_resolvable_records_are_untouched() {
    let Some(dir) = workdir("keep") else {
        return;
    };
    let Some(patched) = link(&dir, true) else {
        return;
    };
    assert!(
        count_fdes(&patched) >= 2,
        "the functions whose relocations do resolve keep their unwind records"
    );
    assert_eq!(
        count_cies(&patched),
        1,
        "and the CIE stays: it is the record format, not a description of any \
         one function"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping eh-frame-orphan-fde {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_ehorphan_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the fixture, optionally orphaning one FDE's relocation, and links
/// it.
fn link(dir: &Path, orphan: bool) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let src = dir.join("c.c");
    let stem = if orphan { "orphan" } else { "intact" };
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args([
            "--target=x86_64-linux-gnu",
            "-fno-pic",
            "-ffunction-sections",
            "-c",
        ])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping eh-frame-orphan-fde: clang cannot build it");
        return None;
    }
    if orphan {
        orphan_first_reloc(&obj)?;
    }
    let out = dir.join(stem);
    let res = link_to(
        std::slice::from_ref(&obj),
        &out,
        b"_start",
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

/// Moves the first `.rela.eh_frame` entry's `r_offset` past the end of the
/// section, so the record it belonged to has no relocation naming its
/// function.
///
/// The relocation is still there and still well formed; it simply covers no
/// record, which is the state an FDE reaches when the tooling that produced
/// the object dropped or rewrote its reference.
fn orphan_first_reloc(obj: &Path) -> Option<()> {
    let mut bytes = fs::read(obj).ok()?;
    let off = {
        let parsed = ObjectFile::parse(&bytes).ok()?;
        let shdr = parsed
            .sections()
            .iter()
            .find(|s| parsed.section_name(s) == b".rela.eh_frame")?;
        usize::try_from(shdr.sh_offset.get()).ok()?
    };
    bytes
        .get_mut(off..off + 8)?
        .copy_from_slice(&u64::MAX.to_le_bytes());
    fs::write(obj, bytes).ok()
}

// --- readers ---------------------------------------------------------------

/// The number of FDE records in the output's `.eh_frame`.
fn count_fdes(bytes: &[u8]) -> usize {
    walk_eh_frame(bytes).1
}

/// The number of CIE records.
fn count_cies(bytes: &[u8]) -> usize {
    walk_eh_frame(bytes).0
}

/// Walks `.eh_frame`, returning `(cies, fdes)`. A record whose `CIE_pointer`
/// is zero is a CIE; anything else is an FDE.
fn walk_eh_frame(bytes: &[u8]) -> (usize, usize) {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return (0, 0);
    };
    let Some(shdr) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".eh_frame")
    else {
        return (0, 0);
    };
    let Ok(data) = obj.section_data(shdr) else {
        return (0, 0);
    };
    let (mut cies, mut fdes, mut at) = (0, 0, 0usize);
    while at + 8 <= data.len() {
        let Some(len) = read_u32(data, at) else {
            break;
        };
        if len == 0 {
            break;
        }
        match read_u32(data, at + 4) {
            Some(0) => cies += 1,
            Some(_) => fdes += 1,
            None => break,
        }
        let Some(next) = (len as usize)
            .checked_add(4)
            .and_then(|n| at.checked_add(n))
        else {
            break;
        };
        at = next;
    }
    (cies, fdes)
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .and_then(|c| <[u8; 4]>::try_from(c).ok())
        .map(u32::from_le_bytes)
}
