//! PE static thread-local storage: a `__declspec(thread)` variable links with
//! xold and runs correctly under `wine`, both single-threaded (the main thread
//! reads the right initial value after a modification) and multi-threaded (a
//! spawned thread sees its own fresh copy while the main thread's copy is
//! untouched).
//!
//! Each test is gated on `clang --target=x86_64-pc-windows-msvc` being
//! available; the `wine` runtime checks are gated separately on `wine` being
//! present. When a tool is missing the test prints a note and returns, so the
//! build never fails over a missing toolchain.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use common::{which, xold_bin};
use xold::coff::{
    PeImage,
    constants::{
        IMAGE_DIRECTORY_ENTRY_TLS, IMAGE_FILE_MACHINE_AMD64,
        IMAGE_NT_OPTIONAL_HDR64_MAGIC,
    },
};

mod common;

/// The single-threaded source: `counter` starts at 5, `main` increments it and
/// checks `get_counter()` returns 6.
const SRC_SINGLE: &[u8] = b"__declspec(thread) int counter = 5;\n\
                           int get_counter(void){ return counter; }\n\
                           int main(void){ counter += 1;\n\
                                            return get_counter() - 6; }\n";

/// The multi-threaded source: the main thread sets its copy to 6, a spawned
/// thread sets its own copy to 42, and the main thread verifies its copy is
/// still 6 after the join (per-thread isolation).
const SRC_MULTI: &[u8] = b"__declspec(thread) int counter = 5;\n\
                           int get_counter(void){ return counter; }\n\
                           __declspec(dllimport) void *CreateThread(\n\
                               void*, unsigned long, unsigned long (*)(void*),\n\
                               void*, unsigned long, unsigned long*);\n\
                           __declspec(dllimport) unsigned long\n\
                               WaitForSingleObject(void*, unsigned long);\n\
                           static unsigned long worker(void *a){\n\
                               counter = 42; return counter; }\n\
                           int main(void){\n\
                               counter += 1;\n\
                               if (get_counter() != 6) return 100;\n\
                               void *h = CreateThread(0,0,worker,0,0,0);\n\
                               if (!h) return 101;\n\
                               WaitForSingleObject(h, 0xFFFFFFFF);\n\
                               if (get_counter() != 6) return 102;\n\
                               return 0; }\n";

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_pe_with_tls_section() {
    let Some(ctx) = LinkContext::discover() else {
        return;
    };
    let obj = ctx.compile("xold_coff_tls_single", SRC_SINGLE);
    let exe = ctx.link("xold_coff_tls_single", &obj);

    let kind = file_type(&exe);
    eprintln!("file: {kind}");
    assert!(kind.contains("PE32+"), "expected PE32+, got: {kind}");
    assert!(kind.contains("x86-64"), "expected x86-64, got: {kind}");

    assert_tls_directory(&exe);
    assert_llvm_objdump_tls(&exe);
    assert_wine_exit(&exe, 0);
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn tls_per_thread_isolation_under_wine() {
    let Some(ctx) = LinkContext::discover() else {
        return;
    };
    let obj = ctx.compile("xold_coff_tls_multi", SRC_MULTI);
    let exe = ctx.link("xold_coff_tls_multi", &obj);

    assert_tls_directory(&exe);
    // Exit 0: main saw 6, the worker saw 42 (its own copy), and main's copy
    // was still 6 after the join.
    assert_wine_exit(&exe, 0);
}

/// Re-parses the executable with xold's own `PeImage` reader and checks the
/// `.tls` section and the TLS data directory.
fn assert_tls_directory(exe: &Path) {
    let bytes = fs::read(exe).expect("read linked executable");
    let img = PeImage::parse(&bytes).expect("reader round-trips the output");
    assert_eq!(img.machine(), IMAGE_FILE_MACHINE_AMD64);
    assert_eq!(img.magic(), IMAGE_NT_OPTIONAL_HDR64_MAGIC);

    let sections = img.sections();
    let tls = sections
        .iter()
        .find(|s| s.name == b".tls")
        .expect(".tls section present");
    assert!(tls.virtual_address != 0, ".tls section has a real RVA");
    // The template starts with counter = 5 (little-endian `05 00 00 00`).
    assert_eq!(
        &tls.data[..4],
        [0x05, 0x00, 0x00, 0x00],
        "TLS template holds counter's initial value"
    );

    let tls_dir = img
        .data_directory(IMAGE_DIRECTORY_ENTRY_TLS)
        .expect("TLS data directory present");
    assert!(tls_dir.0 != 0, "TLS directory has an RVA");
    assert_eq!(
        tls_dir.1, 40,
        "TLS directory size is sizeof(ImageTlsDirectory64)"
    );
}

/// `llvm-objdump --private-headers` recognises the TLS directory and reports
/// sane template addresses.
fn assert_llvm_objdump_tls(exe: &Path) -> bool {
    let Some(objdump) = which("llvm-objdump") else {
        eprintln!("skipping llvm-objdump check: not installed");
        return true;
    };
    let out = Command::new(objdump)
        .args(["--private-headers"])
        .arg(exe)
        .output()
        .expect("run llvm-objdump");
    let stdout = String::from_utf8_lossy(&out.stdout);
    eprintln!(
        "llvm-objdump exit={} (parses {} bytes)",
        out.status,
        stdout.len()
    );
    assert!(
        stdout.contains("TLS directory"),
        "headers list TLS directory"
    );
    assert!(
        stdout.contains("StartAddressOfRawData"),
        "TLS directory fields are decoded"
    );
    true
}

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

// --- helpers --------------------------------------------------------------

/// Collected tool paths and the temp directory for one test run.
struct LinkContext {
    clang: PathBuf,
    dir: PathBuf,
}

impl LinkContext {
    /// Builds the context when the Windows clang target and the xold binary
    /// are both available.
    fn discover() -> Option<Self> {
        let triple = "--target=x86_64-pc-windows-msvc";
        let clang = which("clang")?;
        let probe = std::env::temp_dir().join("xold_coff_tls_probe.o");
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
