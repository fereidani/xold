//! A DLL that calls `ExitProcess` keeps the import.
//!
//! `detect_imports` skipped every `__imp_ExitProcess` it saw in the inputs,
//! on the theory that the executable's entry stub already imports it. A DLL
//! has no entry stub and never terminates the process itself, so for
//! `-shared` the skip dropped the only import there was: the IAT slot the
//! object's code calls through was never allocated and the reference
//! resolved to nothing, leaving an image whose export calls into an
//! unbound slot. lld's rule is the plain one -- an undefined `__imp_<func>`
//! is an import, whatever the output kind (`lld/COFF/DLL.cpp` builds
//! the import table from exactly those undefineds).
//!
//! The skip also swallowed a legitimate duplicate: an executable whose own
//! code calls `ExitProcess` got one import row seeded by the stub, which
//! happens to be right, but only by coincidence of the skip.
//!
//! Gated on the `clang` Windows cross target; if it is absent the tests
//! print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{which, xold_bin};
use xold::coff::PeImage;

mod common;

/// A DLL that exports a wrapper terminating the process: the object carries
/// an undefined `__imp_ExitProcess`, the import under test.
const SRC_STOP: &[u8] = b"__attribute__((dllimport)) void ExitProcess(unsigned);\n\
                          __declspec(dllexport) int stop(unsigned code){ ExitProcess(code); return 0; }\n";

/// The control: the same DLL shape without the call, so the import below is
/// the object's doing and not a fixture that always carried it.
const SRC_PLAIN: &[u8] =
    b"__declspec(dllexport) int add(int a, int b){ return a + b; }\n";

/// The DLL's import table lists `kernel32.dll!ExitProcess`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_dll_calling_exit_process_imports_it() {
    let Some((dir, clang)) = setup("stop") else {
        return;
    };
    let obj = compile(&dir, &clang, "stop.c", SRC_STOP);
    let dll = dir.join("stop.dll");
    assert!(link_shared(&obj, &dll), "the fixture links");
    let imports = read_imports(&dll);
    assert!(
        imports
            .iter()
            .any(|(dll, func)| dll == b"kernel32.dll" && func == b"ExitProcess"),
        "a DLL that calls ExitProcess imports it, entry stub or none; got \
         {imports:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: without the call, no `kernel32.dll` import at all.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_dll_without_the_call_imports_nothing_from_kernel32() {
    let Some((dir, clang)) = setup("plain") else {
        return;
    };
    let obj = compile(&dir, &clang, "plain.c", SRC_PLAIN);
    let dll = dir.join("plain.dll");
    assert!(link_shared(&obj, &dll), "the fixture links");
    let imports = read_imports(&dll);
    assert!(
        !imports.iter().any(|(dll, _)| dll == b"kernel32.dll"),
        "the control DLL reaches for no kernel32 function; got {imports:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// A fresh working directory and the Windows-targeting clang, or `None`
/// (after a note) when the host cannot build these.
fn setup(prefix: &str) -> Option<(PathBuf, PathBuf)> {
    let clang = which("clang")?;
    let dir = std::env::temp_dir()
        .join(format!("xold_coff_exit_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some((dir, clang))
}

/// Compiles `src` to `name.obj` for x86-64 Windows.
fn compile(dir: &Path, clang: &Path, name: &str, src: &[u8]) -> PathBuf {
    let src_path = dir.join(name);
    let obj = dir.join(name.replace(".c", ".obj"));
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args(["--target=x86_64-pc-windows-msvc", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("clang runs")
        .success();
    assert!(ok, "clang builds {name}");
    obj
}

/// Links `obj` into a DLL.
fn link_shared(obj: &Path, out: &Path) -> bool {
    Command::new(xold_bin())
        .arg(obj)
        .args(["-shared", "-o"])
        .arg(out)
        .status()
        .is_ok_and(|s| s.success())
}

// --- readers ---------------------------------------------------------------

/// The image's `(dll, func)` import rows.
fn read_imports(dll: &Path) -> Vec<(Vec<u8>, Vec<u8>)> {
    let bytes = fs::read(dll).expect("read linked DLL");
    let img = PeImage::parse(&bytes).expect("parse linked DLL");
    img.imports()
}
