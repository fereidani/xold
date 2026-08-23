//! Regression for a PE `REL32` overflow that appeared with loose-typed
//! `void*`/`long` declarations of the Win32 thread and wait functions combined
//! with `__declspec(thread)` storage and a static worker.
//!
//! The program is the failing shape from the bug report: `CreateThread` is
//! declared with `void*` and `long` arguments (LLP64 sizes) and `worker`
//! returns `void*`, so clang emits a 64-bit store (`movq`) for the last
//! `void*` argument instead of a 32-bit store (`movl`). xold must still link
//! it and the result must run under `wine`: `counter` starts at 5, `main`
//! increments it to 6, a spawned thread sets its own per-thread copy to 99,
//! and after the join `main` exits 0 only if its copy is still 6 (per-thread
//! TLS isolation).
//!
//! The root cause of the original overflow was an unresolved relocation target
//! resolving to address 0 (notably the synthesised `_tls_index` slot before TLS
//! support landed): a `REL32` of the form `S + A - P` with `S = 0` overflows a
//! signed 32-bit field and surfaced as the opaque `relocation overflow
//! (type 4)`. xold now resolves `_tls_index` through the TLS plan and reports a
//! named `undefined reference` if any relocation target ever resolves to 0, so
//! a future regression is immediately diagnosable rather than a bare type
//! number.
//!
//! Gated on `clang --target=x86_64-pc-windows-msvc`; the `wine` runtime check
//! is gated separately on `wine` being present. When a tool is missing the test
//! prints a note and returns, so the build never fails over a missing
//! toolchain.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use common::{which, xold_bin};

mod common;

/// The failing source shape: loose `void*`/`long` Win32 declarations, a
/// `void*`-returning static worker, and `__declspec(thread)` storage. Exit 0
/// only if `main`'s `counter` is still 6 after the worker set its own copy.
const SRC_LOOSE: &[u8] = b"__declspec(dllimport) void* CreateThread(\n\
    void*, long, void* (*)(void*), void*, long, void*);\n\
    __declspec(dllimport) long WaitForSingleObject(void*, long);\n\
    __declspec(dllimport) void ExitProcess(unsigned);\n\
    __declspec(thread) int counter = 5;\n\
    static void* worker(void* arg){ counter = 99; return 0; }\n\
    int main(void){\n\
        counter += 1;\n\
        void* h = CreateThread(0, 0, worker, 0, 0, 0);\n\
        WaitForSingleObject(h, -1);\n\
        ExitProcess(counter == 6 ? 0 : 1);\n\
    }\n";

/// The well-typed counterpart (`unsigned long`), which always linked. Linked in
/// the same test to guard both shapes against future regressions.
const SRC_TYPED: &[u8] = b"__declspec(dllimport) void *CreateThread(\n\
    void*, unsigned long, unsigned long (*)(void*),\n\
    void*, unsigned long, unsigned long);\n\
    __declspec(dllimport) unsigned long\n\
        WaitForSingleObject(void*, unsigned long);\n\
    __declspec(dllimport) void ExitProcess(unsigned);\n\
    __declspec(thread) int counter = 5;\n\
    static unsigned long worker(void *a){ counter = 99; return 0; }\n\
    int main(void){\n\
        counter += 1;\n\
        void *h = CreateThread(0, 0, worker, 0, 0, 0);\n\
        WaitForSingleObject(h, 0xFFFFFFFF);\n\
        ExitProcess(counter == 6 ? 0 : 1);\n\
    }\n";

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_loose_typed_tls_program_with_imports() {
    let Some(ctx) = LinkContext::discover() else {
        return;
    };
    let obj = ctx.compile("xold_coff_tls_loose", SRC_LOOSE);
    let exe = ctx.link("xold_coff_tls_loose", &obj);

    let kind = file_type(&exe);
    eprintln!("file: {kind}");
    assert!(kind.contains("PE32+"), "expected PE32+, got: {kind}");
    assert!(kind.contains("x86-64"), "expected x86-64, got: {kind}");

    // The loose-typed shape must link without a `relocation overflow` and the
    // program must observe `counter == 6` after the worker ran.
    assert_wine_exit(&exe, 0);
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn well_typed_tls_program_with_imports_still_links() {
    let Some(ctx) = LinkContext::discover() else {
        return;
    };
    let obj = ctx.compile("xold_coff_tls_typed", SRC_TYPED);
    let exe = ctx.link("xold_coff_tls_typed", &obj);
    assert_wine_exit(&exe, 0);
}

// --- helpers --------------------------------------------------------------

/// Runs `exe` under `wine` and asserts the exit code.
fn assert_wine_exit(exe: &Path, expected: i32) {
    let Some(wine) = which("wine64").or_else(|| which("wine")) else {
        eprintln!("skipping wine run: wine not installed");
        return;
    };
    let status = Command::new(wine)
        .arg(exe)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("run wine");
    let code = status.code().unwrap_or(-1);
    eprintln!("wine {} -> exit {code}", exe.display());
    assert_eq!(code, expected, "wine should report exit code {expected}");
}

/// Collected tool paths and the temp directory for one test run.
struct LinkContext {
    clang: PathBuf,
    dir: PathBuf,
}

impl LinkContext {
    /// Builds the context when the Windows clang target is available.
    fn discover() -> Option<Self> {
        let triple = "--target=x86_64-pc-windows-msvc";
        let clang = which("clang")?;
        let probe = std::env::temp_dir().join("xold_coff_tls_loose_probe.o");
        let ok = Command::new(&clang)
            .args([triple, "-c", "-x", "c", "-", "-o"])
            .arg(&probe)
            .stdin(Stdio::null())
            .status()
            .ok()?
            .success();
        let _ = fs::remove_file(&probe);
        if !ok {
            eprintln!("skipping: clang msvc target cannot compile");
            return None;
        }
        Some(Self {
            clang,
            dir: std::env::temp_dir(),
        })
    }

    /// Compiles `src` into `<stem>.obj` and returns the object path.
    fn compile(&self, stem: &str, src: &[u8]) -> PathBuf {
        let triple = "--target=x86_64-pc-windows-msvc";
        let src_path = self.dir.join(format!("{stem}.c"));
        fs::write(&src_path, src).expect("write source");
        let obj = self.dir.join(format!("{stem}.obj"));
        let ok = Command::new(&self.clang)
            .args([triple, "-c"])
            .arg(&src_path)
            .arg("-o")
            .arg(&obj)
            .status()
            .is_ok_and(|s| s.success());
        assert!(ok, "clang compile failed for {stem}");
        obj
    }

    /// Links `obj` with xold into `<stem>.exe` and returns the exe path.
    fn link(&self, stem: &str, obj: &Path) -> PathBuf {
        let exe = self.dir.join(format!("{stem}.exe"));
        let ok = Command::new(xold_bin())
            .arg(obj)
            .args(["-o"])
            .arg(&exe)
            .status()
            .is_ok_and(|s| s.success());
        assert!(ok, "xold link failed for {stem}");
        exe
    }
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
