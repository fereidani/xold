//! Probes for the host toolchain, shared by the end-to-end link tests.
//!
//! Every test that links a runnable executable needs the same handful of host
//! inputs: the C runtime objects, `libc.so.6`, and the `ld.so` path to record
//! in `.interp`. Each test file used to carry its own copy of these probes,
//! and the copies drifted: most spelled `gcc -print-file-name=` correctly, but
//! several passed the name as a separate argument, which gcc rejects. Those
//! files then found no crt objects, skipped every test, and still reported
//! success. One definition each, here, is what keeps that from recurring.
//!
//! Cargo compiles this module into every test binary that declares
//! `mod common;`, so no single binary uses all of it; hence the blanket
//! `dead_code` allowance.

#![allow(dead_code)]

use std::{path::PathBuf, process::Command};

/// Resolves `cmd` on `PATH`, or `None` when it is not there.
pub fn which(cmd: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&paths) {
        let candidate = dir.join(cmd);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// The host path of `name` (`crt1.o`, `libc.so.6`, ...) as gcc resolves it,
/// or `None` when gcc is absent or does not know the file.
///
/// The name belongs in the option itself. gcc accepts only the joined
/// `-print-file-name=NAME` spelling and rejects the split form outright, which
/// would make this return `None` on every host and silently disable the caller
/// rather than skip for a real reason.
///
/// gcc echoes the bare name back when it cannot place the file, so a result
/// equal to the input, or one that does not name an existing file, is treated
/// as "not found".
pub fn crt_file(name: &str) -> Option<PathBuf> {
    let gcc = which("gcc")?;
    let out = Command::new(gcc)
        .arg(format!("-print-file-name={name}"))
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let trimmed = stdout.trim();
    if trimmed.is_empty() || trimmed == name {
        return None;
    }
    let path = PathBuf::from(trimmed);
    path.is_file().then_some(path)
}

/// Locates `libc.so.6` on the host.
pub fn libc_so() -> Option<PathBuf> {
    crt_file("libc.so.6")
}

/// Locates the `libstdc++` linker input on the host.
pub fn libstdcxx_so() -> Option<PathBuf> {
    crt_file("libstdc++.so")
}

/// Locates `libgcc_s.so.1` on the host.
///
/// A C++ program that can throw references `_Unwind_Resume`, which lives here
/// and not in `libstdc++`: `libstdc++.so.6` names it undefined and reaches it
/// through its own `DT_NEEDED`. A link that omits this library is underlinked,
/// and every linker says so -- what a compiler driver adds as `-lgcc_s` a test
/// driving the linker directly has to name itself.
pub fn libgcc_s_so() -> Option<PathBuf> {
    crt_file("libgcc_s.so.1")
}

/// The `xold` binary built for the profile the test is running under.
///
/// Cargo sets `CARGO_BIN_EXE_<name>` for every integration test and builds the
/// binary before the test runs, so this is right under `cargo test` and
/// `cargo test --release` alike. Spelling the path out by hand is not: a
/// hard-coded `target/debug/xold` does not exist under `--release`, and when
/// it does exist it is whatever an earlier build left behind, so the test
/// passes against a binary that is not the one under test.
pub const fn xold_bin() -> &'static str {
    env!("CARGO_BIN_EXE_xold")
}

/// The interpreter path recorded in `/bin/true`, so a test-linked executable
/// asks for the same `ld.so` the host itself uses.
pub fn interpreter() -> Option<Vec<u8>> {
    let out = Command::new("readelf")
        .args(["-l", "/bin/true"])
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().find(|l| l.contains("interpreter:"))?;
    let start = line.find('/')?;
    let end = line.rfind(']').unwrap_or(line.len());
    Some(line.as_bytes()[start..end].to_vec())
}

/// The loader itself, as a linker input.
///
/// `ld.so` is a shared object like any other and exports names a program may
/// reference directly -- `__tls_get_addr` is the one that matters here. Since
/// glibc 2.34 `libc.so.6` does not define it: libc's own dynamic symbol table
/// names it undefined, exactly as a program's does. A link that calls it and
/// lists only libc is underlinked, and every linker says so.
pub fn loader_so() -> Option<PathBuf> {
    let path = PathBuf::from(String::from_utf8(interpreter()?).ok()?);
    path.is_file().then_some(path)
}
