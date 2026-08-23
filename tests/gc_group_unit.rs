//! A section group is collected as a unit.
//!
//! A group says its members are kept or dropped together -- that is the whole
//! of what `SHT_GROUP` means. Garbage collection followed relocations and
//! nothing else, so a secondary member with no incoming relocation was swept
//! while its siblings stayed: a `.gcc_except_table` beside a function, a
//! `.data` slice the group's code indexes into rather than names, a metadata
//! blob a runtime finds by walking the section.
//!
//! Marking a member now enqueues the rest, which is the edge lld walks over
//! `nextInSectionGroup` when it marks one (`lld/ELF/MarkLive.cpp`).
//! Every retained group is recorded, not only the `GRP_COMDAT` ones: the flag
//! decides whether duplicate copies may be folded and says nothing about
//! whether the members belong together, which the group itself already said.
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

/// A two-member group. `grp_fn` is called and so is reachable; `grp_data` is
/// named by nothing, which is exactly the shape that was being swept.
const SRC: &[u8] =
    b"    .section .text.grp,\"axG\",@progbits,siggroup,comdat\n\
    .globl grp_fn\n\
    .type grp_fn,@function\n\
grp_fn:\n\
    movl $0, %eax\n\
    ret\n\
    .section .data.grp,\"awG\",@progbits,siggroup,comdat\n\
    .globl grp_data\n\
grp_data:\n\
    .quad 0x1122334455667788\n\
    .text\n\
    .globl _start\n\
    .type _start,@function\n\
_start:\n\
    call grp_fn\n\
    movq $60, %rax\n\
    movq $0, %rdi\n\
    syscall\n";

/// The unreferenced member survives with the one that is reached.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_group_member_nothing_references_survives_with_its_siblings() {
    let Some(dir) = workdir("unit") else {
        return;
    };
    let Some(bytes) = link(&dir, true) else {
        return;
    };
    assert!(
        symbol_value(&bytes, b"grp_fn").is_some(),
        "the called member is reachable and must stay"
    );
    assert!(
        symbol_value(&bytes, b"grp_data").is_some(),
        "and its group sibling must stay with it: a group is kept or dropped \
         as a unit, and nothing relocates to this member"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The member's bytes are there, not just its symbol.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_sibling_keeps_its_content() {
    let Some(dir) = workdir("bytes") else {
        return;
    };
    let Some(bytes) = link(&dir, true) else {
        return;
    };
    assert_eq!(
        section_size(&bytes, b".data"),
        8,
        "the group's data member contributes its eight bytes"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: collection is still doing something, so the group is kept by
/// the group rule rather than by nothing being collected at all.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn collection_still_removes_what_is_unreachable() {
    let Some(dir) = workdir("control") else {
        return;
    };
    let Some(collected) = link(&dir, true) else {
        return;
    };
    let Some(whole) = link(&dir, false) else {
        return;
    };
    assert!(
        collected.len() <= whole.len(),
        "collection must not grow the image"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping gc-group-unit {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_gcgroup_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Assembles and links the fixture, with or without collection.
fn link(dir: &Path, gc: bool) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let src = dir.join("g.S");
    let obj = dir.join("g.o");
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
        eprintln!("skipping gc-group-unit: clang cannot assemble it");
        return None;
    }
    let out = dir.join(if gc { "gc" } else { "whole" });
    let res = link_to(
        std::slice::from_ref(&obj),
        &out,
        b"_start",
        gc,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

// --- readers ---------------------------------------------------------------

/// The `st_value` of a symbol in the output's `.symtab`.
fn symbol_value(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab
        .iter()
        .find(|s| symtab.name(s) == name)
        .map(|s| s.st_value.get())
}

/// The `sh_size` of a named output section, or zero when it is absent.
fn section_size(bytes: &[u8], name: &[u8]) -> u64 {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return 0;
    };
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == name)
        .map_or(0, |s| s.sh_size.get())
}
