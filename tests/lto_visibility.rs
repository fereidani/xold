//! `used_in_regular_obj` decides what LTO may delete.
//!
//! This is the flag whose failure mode is silence. A bitcode definition that
//! nothing outside the bitcode names may be internalised or dropped, and
//! that is the optimisation LTO exists for. The same definition, when a
//! regular object calls it, must survive -- and if the linker reports it as
//! invisible, LTO removes it from under a call that is still there.
//!
//! So the check is a pair. The same `helper` is compiled twice: once with
//! nothing but bitcode naming it, where it must be internalised, and once
//! with a native object referencing it, where it must stay global. One case
//! alone would pass against a linker that always answered the same way.
//!
//! Codegen shuts LLVM down inside the plugin, so a process gets one run. The
//! pair therefore lives in two binaries; this one holds the case that must
//! keep the symbol.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{
    input::InputFile,
    lto::{Job, Output, compile, plugin},
};

mod common;

/// A regular object's reference keeps a bitcode definition global.
#[test]
#[cfg_attr(miri, ignore = "needs a host plugin and toolchain")]
fn a_native_reference_keeps_a_bitcode_definition() {
    if plugin::resolve(None).is_none() {
        eprintln!("skipping visibility: no LLVMgold.so on this host");
        return;
    }
    let Some(dir) = workdir() else {
        return;
    };
    // The bitcode half defines `helper` and `main`; the native half calls
    // `helper` and nothing else. Only the native reference makes `helper`
    // visible, so it is the single fact under test.
    let Some(bitcode) = build(
        &dir,
        "lib",
        b"int helper(void){return 7;}\nint main(void){return helper();}\n",
        &["-flto", "-c"],
    ) else {
        return;
    };
    let Some(native) = build(
        &dir,
        "user",
        b"int helper(void);\nint use_helper(void){return helper();}\n",
        &["-fno-lto", "-c"],
    ) else {
        return;
    };

    let entry: &[u8] = b"main";
    let objects = compile(&Job {
        named_plugin: None,
        options: &[],
        output: &dir.join("prog"),
        kind: Output::Executable,
        inputs: &[bitcode, native],
        pinned: &[entry],
        export_all: false,
    })
    .expect("the plugin must compile the bitcode")
    .objects;
    assert!(!objects.is_empty(), "codegen must produce an object");

    let bytes = fs::read(&objects[0]).expect("the produced object is readable");
    let file = InputFile::from_member(&objects[0], &bytes)
        .expect("the produced object parses");
    let globals = file.global_symbols().expect("its symbols are readable");
    let helper = globals.iter().find(|sym| sym.name == b"helper");
    assert!(
        helper.is_some_and(|sym| sym.shndx != 0),
        "a native object calls `helper`, so LTO must keep it global; \
         globals were {:?}",
        globals
            .iter()
            .map(|s| String::from_utf8_lossy(s.name).into_owned())
            .collect::<Vec<_>>()
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

fn workdir() -> Option<PathBuf> {
    which("clang")?;
    let dir = std::env::temp_dir()
        .join(format!("xold_lto_vis_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles one source with `flags`, or `None` when the host clang cannot.
fn build(
    dir: &std::path::Path,
    stem: &str,
    source: &[u8],
    flags: &[&str],
) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src, source).ok()?;
    let built = Command::new(clang)
        .args(flags)
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping visibility: clang cannot build {stem}");
        return None;
    }
    Some(obj)
}
