//! Runtime coverage for final-image Mach-O unwind tables.
//!
//! The structural section tests catch malformed headers, but the contract is
//! ultimately libunwind behavior: Rust must catch a panic, an uncaught panic
//! must reach its normal exit path, and a personality/LSDA carried only by
//! compact unwind must work across an xold-linked dylib boundary.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::{fs, process::Command};

use common::{which, xold_bin};

mod common;

#[test]
#[cfg_attr(miri, ignore = "needs the host Rust and C++ toolchains")]
fn arm64_executables_and_dylibs_unwind_at_runtime() {
    let Some(rustc) = which("rustc") else {
        eprintln!("skipping Mach-O unwind runtime test: rustc unavailable");
        return;
    };
    let Some(clang) = which("clang") else {
        eprintln!("skipping Mach-O unwind runtime test: clang unavailable");
        return;
    };
    let Some(clangxx) = which("clang++") else {
        eprintln!("skipping Mach-O unwind runtime test: clang++ unavailable");
        return;
    };
    let dir = std::env::temp_dir()
        .join(format!("xold_macho_unwind_runtime_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create unwind workdir");
    let clang_linker = format!("-Clinker={}", clang.display());
    let linker = format!("-Clink-arg=--ld-path={}", xold_bin());

    let caught_src = dir.join("caught.rs");
    let caught = dir.join("caught");
    fs::write(
        &caught_src,
        "fn main() {\n    let caught = std::panic::catch_unwind(|| panic!(\"boom\"));\n    println!(\"caught: {}\", caught.is_err());\n}\n",
    )
    .expect("write caught fixture");
    let built = Command::new(&rustc)
        .arg("-Cpanic=unwind")
        .arg(&clang_linker)
        .arg(&linker)
        .arg(&caught_src)
        .arg("-o")
        .arg(&caught)
        .output()
        .expect("rustc builds caught fixture");
    assert!(
        built.status.success(),
        "xold links caught Rust fixture: {}",
        String::from_utf8_lossy(&built.stderr)
    );
    let run = Command::new(&caught).output().expect("caught fixture runs");
    assert_eq!(run.status.code(), Some(0));
    assert!(
        String::from_utf8_lossy(&run.stdout).contains("caught: true"),
        "panic crossed its frames and reached catch_unwind"
    );

    let uncaught_src = dir.join("uncaught.rs");
    let uncaught = dir.join("uncaught");
    fs::write(&uncaught_src, "fn main() { panic!(\"uncaught\"); }\n")
        .expect("write uncaught fixture");
    let built = Command::new(&rustc)
        .arg("-Cpanic=unwind")
        .arg(&clang_linker)
        .arg(&linker)
        .arg(&uncaught_src)
        .arg("-o")
        .arg(&uncaught)
        .output()
        .expect("rustc builds uncaught fixture");
    assert!(
        built.status.success(),
        "xold links uncaught Rust fixture: {}",
        String::from_utf8_lossy(&built.stderr)
    );
    let run = Command::new(&uncaught)
        .output()
        .expect("uncaught fixture runs");
    assert_eq!(run.status.code(), Some(101), "normal Rust panic exit");

    let throw_src = dir.join("throw.cc");
    let throw_obj = dir.join("throw.o");
    let dylib = dir.join("libthrow.dylib");
    fs::write(
        &throw_src,
        "extern \"C\" __attribute__((noinline)) void thrower() { throw 42; }\n",
    )
    .expect("write throwing dylib fixture");
    assert!(
        Command::new(&clangxx)
            .args(["-std=c++17", "-O1", "-fexceptions", "-fPIC", "-c"])
            .arg(&throw_src)
            .arg("-o")
            .arg(&throw_obj)
            .status()
            .expect("compile throwing object")
            .success()
    );
    let linked = Command::new(&clangxx)
        .arg(format!("--ld-path={}", xold_bin()))
        .args(["-dynamiclib", "-nodefaultlibs", "-lc++", "-lSystem"])
        .arg(format!("-Wl,-install_name,{}", dylib.display()))
        .arg("-o")
        .arg(&dylib)
        .arg(&throw_obj)
        .output()
        .expect("link throwing dylib");
    assert!(
        linked.status.success(),
        "xold links throwing dylib: {}",
        String::from_utf8_lossy(&linked.stderr)
    );

    let main_src = dir.join("main.cc");
    let main = dir.join("main");
    fs::write(
        &main_src,
        "#include <cstdio>\nextern \"C\" void thrower();\nint main() { try { thrower(); } catch (int x) { std::printf(\"caught dylib: %d\\n\", x); return x == 42 ? 0 : 2; } return 3; }\n",
    )
    .expect("write dylib caller");
    let linked = Command::new(&clangxx)
        .args(["-std=c++17", "-O1", "-fexceptions"])
        .arg(&main_src)
        .arg(format!("-L{}", dir.display()))
        .arg("-lthrow")
        .arg("-o")
        .arg(&main)
        .output()
        .expect("system linker builds dylib caller");
    assert!(
        linked.status.success(),
        "system linker links dylib caller: {}",
        String::from_utf8_lossy(&linked.stderr)
    );
    let run = Command::new(&main).output().expect("dylib caller runs");
    assert_eq!(run.status.code(), Some(0));
    assert_eq!(String::from_utf8_lossy(&run.stdout), "caught dylib: 42\n");

    let _ = fs::remove_dir_all(&dir);
}
