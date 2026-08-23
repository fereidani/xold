//! A weak reference a dependency defines resolves like any other import.
//!
//! The undefined-weak answer at a PC-relative site (`A - P` and friends,
//! see `Arch::undef_weak_pc`) is for a name nothing in the link defines.
//! A weak data reference a shared dependency does define gets a copy
//! relocation and must resolve to its `.bss` slot -- the fact table used
//! to class it by its table kind alone (`Undefined { weak: true }`), so
//! the site stored the undefined-weak arithmetic over the slot address
//! and read a number that was never the data.
//!
//! Gated on `clang`; when it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{icf::IcfMode, linker::link_dyn_exec};

mod common;

/// The dependency defines the data the executable weakly references.
const LIB_SRC: &str = "int weak_data = 0x2a;\n";

/// A weak undefined data reference read PC-relatively: the copy slot is
/// what the load must see.
const START_SRC: &str = "    .text\n    .globl _start\n\
     .weak weak_data\n\
     _start:\n\
     movl weak_data(%rip), %edi\n movl $60, %eax\n syscall\n";

/// The loaded value is the dependency's, via the copy slot -- not the
/// architecture's undefined-weak arithmetic.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_dep_defined_weak_data_reference_reads_the_copy_slot() {
    let dir = workdir();
    let Some(interp) = interpreter() else {
        eprintln!("skipping weak-dep test: interpreter path unknown");
        return;
    };
    let Some(lib) = build_dependency(&dir) else {
        eprintln!("skipping weak-dep test: host toolchain unavailable");
        return;
    };
    let Some(start) = assemble(START_SRC, &dir.join("wdd_start.o"), &dir)
    else {
        eprintln!("skipping weak-dep test: host toolchain unavailable");
        return;
    };
    let prog = dir.join("wdd_prog");
    link_dyn_exec(
        &[start, lib],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    )
    .expect("the weak reference binds the dependency's definition");
    let status = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("the program runs");
    assert_eq!(
        status.code(),
        Some(0x2a),
        "the load must read the copy slot the dependency's value fills"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures ---------------------------------------------------------------

fn workdir() -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_weakdep_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Assembles `src` to `obj` with the host clang. Returns `None` when the
/// toolchain is unavailable; a fixture that fails to assemble panics.
fn assemble(src: &str, obj: &Path, dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let stem = obj.file_stem()?.to_str().unwrap_or("unit");
    let file = dir.join(format!("{stem}.S"));
    fs::write(&file, src).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c", "-o"])
        .arg(obj)
        .arg(&file)
        .status()
        .ok()?
        .success();
    assert!(built, "fixture {stem} must assemble");
    Some(obj.to_path_buf())
}

/// Builds the shared dependency with the system toolchain.
fn build_dependency(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("libwd.c");
    fs::write(&src, LIB_SRC).ok()?;
    let so = dir.join("libwd.so");
    let built = Command::new(clang)
        .args(["-shared", "-fPIC", "-Wl,-soname,libwd.so", "-o"])
        .arg(&so)
        .arg(&src)
        .status()
        .ok()?
        .success();
    assert!(built, "fixture dependency must build");
    Some(so)
}
