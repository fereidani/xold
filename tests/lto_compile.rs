//! The full LTO handoff, in a binary of its own.
//!
//! Codegen is a once-per-process event: the plugin shuts LLVM down when it
//! finishes, and the session that holds the claimed files is global because
//! the plugin ABI gives its callbacks nowhere else to look. That is right for
//! a linker, which links once -- but it means this check cannot share a
//! process with the tests that claim files without compiling them, so it gets
//! its own.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{
    input::Format,
    lto::{Output, plugin},
};

mod common;

/// The whole handoff: claim, resolve, compile.
///
/// The plugin only produces an object if it accepted every resolution it was
/// given, so this is the check that the verdicts are well-formed as well as
/// that codegen ran. The object it returns must be a real relocatable ELF
/// defining the entry symbol, because that is what the link will consume.
#[test]
#[cfg_attr(miri, ignore = "needs a host plugin and toolchain")]
fn the_plugin_compiles_claimed_bitcode_into_an_object() {
    if plugin::resolve(None).is_none() {
        eprintln!("skipping compile: no LLVMgold.so on this host");
        return;
    }
    let Some((dir, obj)) = compile_bitcode("compile") else {
        return;
    };
    let out = dir.join("prog");
    let entry: &[u8] = b"main";
    let produced = xold::lto::compile(&xold::lto::Job {
        named_plugin: None,
        options: &[],
        output: &out,
        kind: Output::Executable,
        inputs: &[obj],
        pinned: &[entry],
        groups: &[],
        export_all: false,
    });
    let objects = match produced {
        Ok(compiled) => compiled.objects,
        Err(err) => panic!("the plugin must compile the bitcode: {err}"),
    };
    assert!(!objects.is_empty(), "codegen must produce an object");
    let bytes = fs::read(&objects[0]).expect("the produced object is readable");
    assert_eq!(
        Format::detect(&bytes),
        Some(Format::Elf),
        "the plugin returns native objects, not more bitcode"
    );
    let file = xold::input::InputFile::from_member(&objects[0], &bytes)
        .expect("the produced object parses");
    let defines_main = file
        .global_symbols()
        .expect("its symbols are readable")
        .iter()
        .any(|sym| sym.name == entry && sym.shndx != 0);
    assert!(defines_main, "the pinned entry symbol survived LTO");
    let _ = fs::remove_dir_all(&dir);
}

/// Compiles a tiny translation unit to bitcode, or `None` when the host
/// clang cannot.
fn compile_bitcode(tag: &str) -> Option<(PathBuf, PathBuf)> {
    let clang = which("clang")?;
    let dir = std::env::temp_dir()
        .join(format!("xold_lto_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    let src = dir.join("t.c");
    let obj = dir.join("t.o");
    fs::write(
        &src,
        b"int helper(void){return 7;}\nint main(void){return helper();}\n",
    )
    .ok()?;
    let built = Command::new(clang)
        .args(["-flto", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping lto {tag}: clang cannot emit bitcode");
        let _ = fs::remove_dir_all(&dir);
        return None;
    }
    if !fs::read(&obj).is_ok_and(|b| b.starts_with(b"BC\xc0\xde")) {
        eprintln!("skipping lto {tag}: this clang emits fat LTO objects");
        let _ = fs::remove_dir_all(&dir);
        return None;
    }
    Some((dir, obj))
}
