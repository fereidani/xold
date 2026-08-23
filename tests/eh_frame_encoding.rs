//! An FDE's `initial_location` base is three bits, and an unknown one is an
//! error.
//!
//! A CIE's `R` augmentation byte says how the FDEs sharing it spell their
//! `initial_location`: the low nibble is the width, bits 4 to 6 are the base,
//! and bit 7 is `DW_EH_PE_indirect`. `fde_pc` compared `enc & 0xf0` against
//! `DW_EH_PE_pcrel`, which folds the indirect bit into the base.
//!
//! Two things followed. `indirect|pcrel` (0x90) missed the pcrel arm and was
//! read as an absolute address, as was every base the function has no formula
//! for. The decoded PC is then nowhere near the function, the range filter
//! drops the record, and the function disappears from the `.eh_frame_hdr`
//! search table -- the count goes to zero and a `throw` through any of those
//! functions terminates the program. The link says nothing at any point.
//!
//! lld masks `0x70` and reports "unknown FDE size relative encoding" for a
//! base it cannot compute (`lld/ELF/SyntheticSections.cpp`).
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

const SRC: &[u8] = b"int f(int x) { return x + 1; }\n\
    int main(void) { return f(41); }\n";

/// `DW_EH_PE_sdata4 | DW_EH_PE_pcrel`, what clang writes.
const PCREL: u8 = 0x1b;
/// The same with `DW_EH_PE_indirect` set.
const INDIRECT_PCREL: u8 = 0x9b;
/// `DW_EH_PE_sdata4 | DW_EH_PE_datarel`, a base with no formula here.
const DATAREL: u8 = 0x3b;

/// An `indirect|pcrel` CIE still yields a search table: the indirect bit
/// changes what the decoded address points at, not how it is decoded.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_indirect_pcrel_encoding_still_reaches_the_search_table() {
    let Some(dir) = workdir("indirect") else {
        return;
    };
    let Some(obj) = patched(&dir, "ind", INDIRECT_PCREL) else {
        return;
    };
    let out = dir.join("ind");
    let res = link(&obj, &out);
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let bytes = fs::read(&out).expect("read output");
    assert!(
        fde_count(&bytes) > 0,
        "reading the pcrel value as an absolute address puts the PC nowhere \
         near the function, and the range filter then drops the record: the \
         search table ends up empty and a throw through the function \
         terminates"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A base with no formula is refused rather than read as absolute.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_unknown_base_encoding_fails_the_link() {
    let Some(dir) = workdir("datarel") else {
        return;
    };
    let Some(obj) = patched(&dir, "datarel", DATAREL) else {
        return;
    };
    let err = link(&obj, &dir.join("datarel")).expect_err(
        "a base this linker cannot compute must end the link, not silently \
         produce an unwind table the records fall out of",
    );
    assert!(
        matches!(err, xold::Error::Format(_)),
        "the refusal must be a format error, got {err:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: the encoding clang actually writes links and indexes both
/// functions, so the stricter reading costs nothing an ordinary build needs.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_ordinary_pcrel_encoding_is_unaffected() {
    let Some(dir) = workdir("plain") else {
        return;
    };
    let Some(obj) = patched(&dir, "plain", PCREL) else {
        return;
    };
    let out = dir.join("plain");
    let res = link(&obj, &out);
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let bytes = fs::read(&out).expect("read output");
    assert_eq!(
        fde_count(&bytes),
        2,
        "both functions must be indexed for unwinding"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping eh-frame-encoding {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_ehenc_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// The leading bytes of the CIE clang emits for this fixture: length, a zero
/// CIE id, version 1, the augmentation string `zR`, code alignment 1, data
/// alignment -8, return-address column 16, and a one-byte augmentation data
/// length. The `R` byte is the one after them.
const CIE_PROLOGUE: [u8; 16] = [
    0x14, 0, 0, 0, 0, 0, 0, 0, 1, 0x7a, 0x52, 0, 1, 0x78, 0x10, 1,
];

/// Compiles the fixture and rewrites the CIE's `R` byte to `enc`.
///
/// Every base but `pcrel` needs a CIE no compiler on this host will emit, so
/// the byte is patched. It is one field of one record, and the record stays
/// well formed: what changes is only the claim about how its FDEs are spelled.
fn patched(dir: &Path, stem: &str, enc: u8) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fno-pic", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping eh-frame-encoding: clang cannot build the fixture");
        return None;
    }
    let mut bytes = fs::read(&obj).ok()?;
    let at = find(&bytes, &CIE_PROLOGUE)?;
    *bytes.get_mut(at + CIE_PROLOGUE.len())? = enc;
    fs::write(&obj, bytes).ok()?;
    Some(obj)
}

/// The offset of the first occurrence of `needle` in `hay`.
fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Links `obj` as a static executable with an unwind search table.
fn link(obj: &Path, out: &Path) -> Result<(), xold::Error> {
    link_to(
        std::slice::from_ref(&obj.to_path_buf()),
        out,
        b"main",
        false,
        IcfMode::None,
        false,
    )
}

// --- readers ---------------------------------------------------------------

/// The `fde_count` field of `.eh_frame_hdr`, which sits after the four
/// encoding bytes and the 4-byte `eh_frame_ptr`.
fn fde_count(bytes: &[u8]) -> u32 {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return 0;
    };
    let Some(shdr) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".eh_frame_hdr")
    else {
        return 0;
    };
    let Ok(data) = obj.section_data(shdr) else {
        return 0;
    };
    data.get(8..12)
        .and_then(|c| <[u8; 4]>::try_from(c).ok())
        .map_or(0, u32::from_le_bytes)
}
