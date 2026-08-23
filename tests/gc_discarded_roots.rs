//! A discarded COMDAT member is not a garbage-collection root.
//!
//! The root walk visited every allocated section of every input and rooted the
//! ones that are roots by kind -- `.init_array`, a note, an explicit
//! `SHF_GNU_RETAIN` pin. It did not ask whether the section was in the image.
//!
//! A COMDAT group that lost is not: its members were decided against before
//! collection ran. Seeding the walk from one keeps alive everything only the
//! dead copy referenced. An inline variable with a dynamic initialiser is the
//! everyday shape -- every translation unit that includes the header emits the
//! group, `.init_array` among its members, and all but one copy is retired.
//!
//! The direction is conservative, so nothing breaks; the image simply carries
//! code that collection was asked to remove and that nothing can reach.
//!
//! Gated on `clang++` and a system `libstdc++`; without them the tests print a
//! note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_dyn_exec};

mod common;

/// An inline variable with a dynamic initialiser: both translation units emit
/// the COMDAT group, and the group carries `.init_array`, which is a root by
/// kind.
const HEADER: &[u8] = b"#pragma once\n\
    int side_effect(void);\n\
    inline int inline_global = side_effect();\n";

const A_SRC: &[u8] = b"#include \"h.hpp\"\n\
    int side_effect(void) { return 5; }\n\
    int a_entry(void) { return inline_global; }\n";

const B_SRC: &[u8] = b"#include \"h.hpp\"\n\
    int a_entry(void);\n\
    int main(void) { return a_entry() - 5; }\n";

/// Collection keeps less code once the dead copy stops seeding the walk.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_retired_group_member_seeds_nothing() {
    let Some(dir) = workdir("seed") else {
        return;
    };
    let Some(collected) = link(&dir, true) else {
        return;
    };
    let Some(whole) = link(&dir, false) else {
        return;
    };
    let kept = text_size(&collected);
    let all = text_size(&whole);
    assert!(kept > 0 && all > 0, "the fixture must produce code");
    assert!(
        kept < all,
        "--gc-sections must remove something here: {kept:#x} of {all:#x}"
    );
    // The surviving group's initialiser stays, the retired copy's does not.
    // The bound is this fixture's live code on this toolchain: rooting the
    // dead copy's `.init_array` adds its 16-byte startup routine, taking
    // `.text` from 0x90 to 0xa0. An exact number is what makes the difference
    // visible; if a toolchain change shifts it the test says so rather than
    // passing on a larger image.
    assert!(
        kept <= 0x90,
        ".text is {kept:#x}: rooting the retired copy's .init_array leaves \
         its startup routine in the image, which is code nothing can reach"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The surviving copy is untouched: the program still has its initialiser and
/// the function it calls.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_surviving_copy_keeps_its_initialiser() {
    let Some(dir) = workdir("keep") else {
        return;
    };
    let Some(collected) = link(&dir, true) else {
        return;
    };
    assert!(
        symbol_value(&collected, b"__cxx_global_var_init").is_some(),
        "the surviving group's initialiser is reachable and must stay"
    );
    assert_eq!(
        init_array_size(&collected),
        8,
        "and exactly one of the two copies' .init_array entries reaches the \
         image"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang++").is_none() {
        eprintln!("skipping gc-discarded-roots {prefix}: no clang++");
        return None;
    }
    if cxx_runtime().is_none() {
        eprintln!("skipping gc-discarded-roots {prefix}: no libstdc++");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_gcroots_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds both translation units and links them, with or without collection.
fn link(dir: &Path, gc: bool) -> Option<Vec<u8>> {
    fs::write(dir.join("h.hpp"), HEADER).ok()?;
    let a = compile(dir, A_SRC, "a")?;
    let b = compile(dir, B_SRC, "b")?;
    let interp = common::interpreter()?;
    let paths = vec![
        common::crt_file("Scrt1.o")?,
        common::crt_file("crti.o")?,
        a,
        b,
        cxx_runtime()?,
        common::libc_so()?,
        common::crt_file("crtn.o")?,
    ];
    let out = dir.join(if gc { "gc" } else { "whole" });
    let res = link_dyn_exec(
        &paths,
        &out,
        b"_start",
        interp.as_slice(),
        gc,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

/// Compiles one translation unit with one section per function and per datum.
fn compile(dir: &Path, src: &[u8], stem: &str) -> Option<PathBuf> {
    let clangxx = which("clang++")?;
    let src_path = dir.join(format!("{stem}.cpp"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clangxx)
        .args([
            "--target=x86_64-linux-gnu",
            "-std=c++17",
            "-O1",
            "-ffunction-sections",
            "-fdata-sections",
            "-c",
        ])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping gc-discarded-roots: clang++ cannot build it");
        return None;
    }
    Some(obj)
}

/// The C++ runtime as a real shared object.
///
/// The `libstdc++.so` beside the compiler is a GNU linker script on this
/// distribution. The driver expands those; this test drives the library, which
/// takes a settled input list, so the versioned object is named directly.
fn cxx_runtime() -> Option<PathBuf> {
    ["/usr/lib64/libstdc++.so.6", "/usr/lib/libstdc++.so.6"]
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
}

// --- readers ---------------------------------------------------------------

/// The `sh_size` of `.text`.
fn text_size(bytes: &[u8]) -> u64 {
    section_size(bytes, b".text")
}

/// The `sh_size` of `.init_array`.
fn init_array_size(bytes: &[u8]) -> u64 {
    section_size(bytes, b".init_array")
}

fn section_size(bytes: &[u8], name: &[u8]) -> u64 {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return 0;
    };
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == name)
        .map_or(0, |s| s.sh_size.get())
}

/// The `st_value` of a symbol in the output's `.symtab`.
fn symbol_value(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab
        .iter()
        .find(|s| symtab.name(s) == name)
        .map(|s| s.st_value.get())
}
