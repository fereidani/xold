//! PE DLL output: `-shared` produces a PE32+ x86-64 DLL with an export table,
//! verified structurally and at runtime under `wine`.
//!
//! A DLL is built from a `__declspec(dllexport)` source, then a loader EXE
//! (built with xold's Phase-20 rich imports) calls `LoadLibraryA` on the DLL,
//! `GetProcAddress("add")`, invokes it, and exits with the result. The DLL is
//! also checked with `file`, xold's own `PeImage` reader and `llvm-readobj`.
//!
//! Each test is gated on `clang --target=x86_64-pc-windows-msvc` being
//! available; if the cross target is absent the test prints a note and returns,
//! so the build never fails over a missing toolchain. The `wine` runtime check
//! is gated separately on `wine64`/`wine` being present.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use common::{which, xold_bin};
use xold::coff::{
    PeImage,
    constants::{
        IMAGE_DIRECTORY_ENTRY_EXPORT, IMAGE_FILE_DLL, IMAGE_FILE_MACHINE_AMD64,
    },
};

mod common;

/// The DLL source: exports `add` via `__declspec(dllexport)`.
const DLL_SRC: &[u8] =
    b"__declspec(dllexport) int add(int a, int b){ return a + b; }\n";

/// The loader source: `LoadLibrary` + `GetProcAddress` + call, exit with
/// result.
const EXE_SRC: &[u8] = b"__attribute__((dllimport)) void* LoadLibraryA(const char*);\n\
                         __attribute__((dllimport)) void* GetProcAddress(void*, const char*);\n\
                         __attribute__((dllimport)) void ExitProcess(unsigned);\n\
                         typedef int (*add_t)(int,int);\n\
                         int main(void){\n\
                             void *h = LoadLibraryA(\"lib.dll\");\n\
                             add_t add = (add_t)GetProcAddress(h, \"add\");\n\
                             int r = add(40, 2);\n\
                             ExitProcess((unsigned)r);\n\
                         }\n";

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_pe_dll_with_export_table() {
    let Some(dir) = setup() else {
        return;
    };
    let dll = dir.join("lib.dll");
    let exe = dir.join("use.exe");
    assert!(build_dll(&dll), "xold -shared failed");

    // `file` reports a PE32+ x86-64 DLL.
    let kind = file_type(&dll);
    eprintln!("file lib.dll: {kind}");
    assert!(kind.contains("PE32+"), "expected PE32+, got: {kind}");
    assert!(kind.contains("x86-64"), "expected x86-64, got: {kind}");
    assert!(
        kind.contains("DLL"),
        "expected DLL characteristics, got: {kind}"
    );

    // xold's own reader sees IMAGE_FILE_DLL, the EXPORT data directory, and the
    // right machine.
    assert_dll_round_trips(&dll);

    // llvm-readobj walks the export table and reports `add`.
    assert!(
        llvm_readobj_lists_export(&dll, "add"),
        "export `add` missing"
    );

    // Build the loader EXE and run the LoadLibrary + GetProcAddress + call
    // round-trip under wine: add(40, 2) -> exit 42.
    assert!(build_exe(&exe), "loader link failed");
    if let Some((code, out)) = run_wine(&exe, &dir) {
        let _ = out;
        assert_eq!(code, 42, "wine exit code (expected add(40,2) == 42)");
    }
}

/// Compiles the DLL source with clang `-msvc` and links it with `xold -shared`.
fn build_dll(dll: &Path) -> bool {
    let triple = "--target=x86_64-pc-windows-msvc";
    let Some(clang) = windows_clang(triple) else {
        return false;
    };
    let dir = parent_of(dll);
    let src = dir.join("xold_pe_dll_lib.c");
    let obj = dir.join("xold_pe_dll_lib.obj");
    fs::write(&src, DLL_SRC).is_ok()
        && compile(&clang, triple, &src, &obj)
        && link_shared(&obj, dll)
}

/// Compiles the loader source and links it with `xold` into an EXE.
fn build_exe(exe: &Path) -> bool {
    let triple = "--target=x86_64-pc-windows-msvc";
    let Some(clang) = windows_clang(triple) else {
        return false;
    };
    let dir = parent_of(exe);
    let src = dir.join("xold_pe_dll_use.c");
    let obj = dir.join("xold_pe_dll_use.obj");
    fs::write(&src, EXE_SRC).is_ok()
        && compile(&clang, triple, &src, &obj)
        && link_exe(&obj, exe)
}

/// The parent directory of `path`, or the current directory if it has none.
fn parent_of(path: &Path) -> PathBuf {
    path.parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
}

/// Runs clang to compile `src` into `obj` for `triple`.
fn compile(clang: &Path, triple: &str, src: &Path, obj: &Path) -> bool {
    Command::new(clang)
        .args([triple, "-c"])
        .arg(src)
        .arg("-o")
        .arg(obj)
        .status()
        .is_ok_and(|s| s.success())
}

/// Links `obj` into a DLL `out` with `xold -shared`.
fn link_shared(obj: &Path, out: &Path) -> bool {
    Command::new(xold_bin())
        .arg(obj)
        .args(["-shared", "-o"])
        .arg(out)
        .status()
        .is_ok_and(|s| s.success())
}

/// Links `obj` into an EXE `out` with `xold`.
fn link_exe(obj: &Path, out: &Path) -> bool {
    Command::new(xold_bin())
        .arg(obj)
        .args(["-o"])
        .arg(out)
        .status()
        .is_ok_and(|s| s.success())
}

/// Creates an isolated temp directory for this test, returning `None` (skip)
/// when the clang cross target is absent.
fn setup() -> Option<PathBuf> {
    let triple = "--target=x86_64-pc-windows-msvc";
    windows_clang(triple)?;
    let dir = std::env::temp_dir().join("xold_pe_dll");
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// xold's `PeImage` reader round-trips the DLL: machine, DLL flag, and a
/// populated EXPORT data directory.
fn assert_dll_round_trips(dll: &Path) {
    let bytes = fs::read(dll).expect("read linked DLL");
    let img = PeImage::parse(&bytes).expect("reader round-trips the DLL");
    assert_eq!(img.machine(), IMAGE_FILE_MACHINE_AMD64);
    assert!(
        img.characteristics() & IMAGE_FILE_DLL != 0,
        "IMAGE_FILE_DLL not set (0x{:x})",
        img.characteristics()
    );
    let export_dir = img
        .data_directory(IMAGE_DIRECTORY_ENTRY_EXPORT)
        .expect("export directory present");
    assert!(
        export_dir.0 != 0 && export_dir.1 != 0,
        "export directory not populated"
    );
}

/// Whether `llvm-readobj --coff-exports` lists `name` as an exported symbol.
fn llvm_readobj_lists_export(dll: &Path, name: &str) -> bool {
    let Some(readobj) = which("llvm-readobj") else {
        eprintln!("skipping llvm-readobj check: not installed");
        return true;
    };
    let out = Command::new(readobj)
        .args(["--coff-exports"])
        .arg(dll)
        .output()
        .expect("run llvm-readobj");
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout.lines().any(|line| {
        line.trim().starts_with("Name:") && line.trim().ends_with(name)
    })
}

/// Runs `exe` under wine with its directory on the path so `LoadLibraryA` finds
/// the sibling DLL. Returns `(exit_code, stdout)`.
fn run_wine(exe: &Path, dir: &Path) -> Option<(i32, String)> {
    let Some(wine) = which("wine64").or_else(|| which("wine")) else {
        eprintln!("skipping wine run: wine not installed");
        return None;
    };
    // Wine's LoadLibraryA resolves a bare name against the loaded executable's
    // directory, so run from `dir` with the EXE addressed relatively.
    let status = Command::new(&wine)
        .arg(exe)
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("run wine");
    let code = status.code().unwrap_or(-1);
    eprintln!("wine {} -> exit {code}", exe.display());
    Some((code, String::new()))
}

// --- tooling helpers ------------------------------------------------------

/// Returns the clang binary if it can compile for `triple`, else `None`.
fn windows_clang(triple: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let ok = Command::new(&clang)
        .args([triple, "-c", "-x", "c", "-", "-o", "/dev/null"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .ok()?
        .success();
    ok.then_some(clang)
}

/// Runs `file` on `path` and returns its stdout.
fn file_type(path: &Path) -> String {
    String::from_utf8_lossy(
        &Command::new("file")
            .arg(path)
            .output()
            .map(|o| o.stdout)
            .unwrap_or_default(),
    )
    .into_owned()
}
