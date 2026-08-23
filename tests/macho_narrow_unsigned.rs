//! A 4-byte Mach-O `UNSIGNED` is legal, and a missing entry is not.
//!
//! Mach-O sizes an absolute fixup through the entry's `r_length`, not through
//! its type, so `.long _sym` and `.quad _sym` arrive as the same relocation
//! number. The table pinned `UNSIGNED` to a 64-bit store, so the narrow form
//! asked the driver to write eight bytes into a four-byte slot and died with
//! `OutOfRange("relocation output slot")` -- an internal error for a legal
//! input. The reader normalises the narrow form to its own type now, which
//! keeps the width stated in the table rather than overridden in the driver.
//!
//! Separately, `has_entry` was computed and read nowhere: an image whose entry
//! symbol was undefined got `entryoff` zero, which points at the Mach header.
//! It linked, and then executed its own magic number. lld hard-errors.
//!
//! Darwin images cannot run on this host, so these check the tables and the
//! refusal rather than the result of executing anything.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{
    error::Error,
    input::Input,
    macho::{
        MachReloc, link_macho,
        reloc::{MachoTarget, table_type},
    },
    reloc::{
        Arch, WriteKind,
        macho_arm64::{
            ARM64_RELOC_UNSIGNED, ARM64_RELOC_UNSIGNED_4, MachoArm64,
        },
        macho_x86_64::{
            MachoX86_64, X86_64_RELOC_UNSIGNED, X86_64_RELOC_UNSIGNED_4,
        },
    },
};

mod common;

/// A 4-byte entry maps to the narrow type, an 8-byte one does not.
#[test]
fn the_width_comes_from_the_entry() {
    for (target, wide, narrow) in [
        (
            MachoTarget::X86_64,
            X86_64_RELOC_UNSIGNED,
            X86_64_RELOC_UNSIGNED_4,
        ),
        (
            MachoTarget::Arm64,
            ARM64_RELOC_UNSIGNED,
            ARM64_RELOC_UNSIGNED_4,
        ),
    ] {
        assert_eq!(
            table_type(target, &unsigned(3)),
            wide,
            "a pointer-width absolute keeps the type it arrived as"
        );
        assert_eq!(
            table_type(target, &unsigned(2)),
            narrow,
            "`.long _sym` is legal and needs a 4-byte store"
        );
    }
}

/// And the narrow type stores four bytes.
#[test]
fn the_narrow_type_writes_four_bytes() {
    let wide = MachoX86_64::spec(X86_64_RELOC_UNSIGNED).expect("wide spec");
    let narrow =
        MachoX86_64::spec(X86_64_RELOC_UNSIGNED_4).expect("narrow spec");
    assert!(
        matches!(kind(wide), Some(WriteKind::W64)),
        "the 8-byte form is unchanged"
    );
    assert!(
        matches!(kind(narrow), Some(WriteKind::W32SU)),
        "the 4-byte form stores four bytes and range-checks them"
    );
    let arm = MachoArm64::spec(ARM64_RELOC_UNSIGNED_4).expect("arm64 spec");
    assert!(
        matches!(kind(arm), Some(WriteKind::W32SU)),
        "and so does the arm64 one"
    );
}

/// Only `UNSIGNED` is affected: every other type keeps its number whatever the
/// entry's length says.
#[test]
fn no_other_type_is_rewritten() {
    for r_type in [1u8, 2, 3, 4, 5, 9] {
        let mut reloc = unsigned(2);
        reloc.r_type = r_type;
        assert_eq!(
            table_type(MachoTarget::X86_64, &reloc),
            u32::from(r_type),
            "type {r_type} is sized by its own spec, not by r_length"
        );
    }
}

/// A missing entry symbol is refused rather than pointed at the header.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_missing_entry_is_an_error() {
    let Some((dir, obj)) = darwin_object("entry") else {
        return;
    };
    let files = [Input::Path(&obj)];
    let err = link_macho(&files, &dir.join("out"), b"_absent")
        .expect_err("an entry that is defined nowhere is an error");
    assert!(
        matches!(err, Error::UndefinedEntry(ref name) if name == "_absent"),
        "the refusal must name the entry it could not find, got {err:?}"
    );
    // The control: the real entry still links, so the check refuses a missing
    // entry and nothing else.
    let res = link_macho(&files, &dir.join("ok"), b"_main");
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let _ = fs::remove_dir_all(&dir);
}

// --- darwin fixture --------------------------------------------------------

/// A defined entry and one call, which is the smallest thing that lays out.
const SRC: &[u8] = b"int g(void) { return 7; }\n\
    int main(void) { return g(); }\n";

/// Compiles the fixture for darwin, or `None` (after printing a note) when
/// clang cannot target it.
fn darwin_object(prefix: &str) -> Option<(PathBuf, PathBuf)> {
    let clang = which("clang")?;
    let dir = std::env::temp_dir()
        .join(format!("xold_machoentry_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    let src = dir.join("m.c");
    let obj = dir.join("m.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-apple-darwin", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping macho-entry {prefix}: clang cannot target darwin");
        let _ = fs::remove_dir_all(&dir);
        return None;
    }
    Some((dir, obj))
}

// --- fixtures --------------------------------------------------------------

/// An absolute relocation of `1 << length` bytes.
const fn unsigned(r_length: u8) -> MachReloc {
    MachReloc {
        r_address: 0,
        r_symbolnum: 0,
        r_pcrel: false,
        r_length,
        r_extern: true,
        r_type: 0,
        r_scattered: false,
    }
}

/// The byte-store kind of a spec, if it is a plain byte store.
const fn kind(spec: xold::reloc::Spec) -> Option<WriteKind> {
    match spec.write {
        xold::reloc::Write::Bytes(k) => Some(k),
        xold::reloc::Write::Field(_) => None,
    }
}
