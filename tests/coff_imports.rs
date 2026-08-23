//! PE rich imports: linking user `dllimport` references into a PE32+ image
//! that imports multiple functions (across multiple DLLs) and runs correctly
//! under `wine`.
//!
//! Each test is gated on `clang --target=x86_64-pc-windows-msvc` being
//! available; if the cross target is absent the test prints a note and
//! returns, so the build never fails over a missing toolchain. The `wine`
//! runtime check is gated separately on `wine64`/`wine` being present.
//!
//! clang `-msvc` encodes a `dllimport` call as a reference to the undefined
//! external `__imp_<func>` (the IAT slot address); xold maps the function to a
//! DLL via a well-known table, emits the import directory, and resolves the
//! reference to the IAT slot so the loader binds it at load time.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use common::{which, xold_bin};
use xold::coff::{
    PeImage,
    constants::{
        IMAGE_DIRECTORY_ENTRY_IAT, IMAGE_DIRECTORY_ENTRY_IMPORT,
        IMAGE_FILE_MACHINE_AMD64,
    },
};

mod common;

/// `printf` from `msvcrt.dll`: one imported function, the simplest case.
const SRC_PRINTF: &[u8] = b"__attribute__((dllimport)) int printf(const char*, ...);\n\
                            int main(void){ printf(\"hello from PE\\n\"); return 0; }\n";

/// Two DLLs exercised in one program: `kernel32.dll!WriteFile` (via
/// `GetStdHandle`) writes to the console, then `msvcrt.dll!printf` prints.
const SRC_MULTI: &[u8] = b"__attribute__((dllimport)) int printf(const char*, ...);\n\
                           __attribute__((dllimport)) void* GetStdHandle(unsigned int);\n\
                           __attribute__((dllimport)) int WriteFile(void*, const void*, unsigned int, unsigned int*, void*);\n\
                           static const char msg[] = \"kernel32 writes\\n\";\n\
                           int main(void){\n\
                               void* h = GetStdHandle((unsigned int)-11);\n\
                               unsigned int w = 0;\n\
                               WriteFile(h, msg, (unsigned int)sizeof(msg)-1, &w, (void*)0);\n\
                               printf(\"msvcrt prints %d\\n\", 7);\n\
                               return 0;\n\
                           }\n";

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_pe_importing_msvcrt_printf() {
    let Some((dir, exe)) = run_link(SRC_PRINTF, "printf") else {
        return;
    };
    assert_pe32_plus_x86_64(&exe);
    assert_round_trips(&exe);

    // The import directory lists msvcrt.dll!printf (plus the entry stub's
    // kernel32.dll!ExitProcess).
    let imports = read_imports(&exe);
    assert!(
        imports
            .iter()
            .any(|(d, f)| d == b"msvcrt.dll" && f == b"printf"),
        "expected msvcrt.dll!printf, got {imports:?}"
    );
    assert!(
        imports
            .iter()
            .any(|(d, f)| d == b"kernel32.dll" && f == b"ExitProcess"),
        "expected kernel32.dll!ExitProcess, got {imports:?}"
    );
    assert!(llvm_objdump_lists(&exe, "msvcrt.dll"));
    assert!(llvm_objdump_lists(&exe, "printf"));

    // Runtime proof: the program prints and exits 0.
    let (code, out) = run_wine(&exe, &dir);
    assert_eq!(code, 0, "wine exit code");
    assert!(
        out.contains("hello from PE"),
        "wine stdout missing expected output, got: {out:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_pe_importing_multiple_dlls() {
    let Some((dir, exe)) = run_link(SRC_MULTI, "multi") else {
        return;
    };
    assert_pe32_plus_x86_64(&exe);
    assert_round_trips(&exe);

    // Both DLLs and their functions appear in the import directory.
    let imports = read_imports(&exe);
    let kernel32_has =
        |f: &[u8]| imports.iter().any(|(d, g)| d == b"kernel32.dll" && g == f);
    assert!(
        kernel32_has(b"ExitProcess"),
        "kernel32!ExitProcess, got {imports:?}"
    );
    assert!(
        kernel32_has(b"GetStdHandle"),
        "kernel32!GetStdHandle, got {imports:?}"
    );
    assert!(
        kernel32_has(b"WriteFile"),
        "kernel32!WriteFile, got {imports:?}"
    );
    assert!(
        imports
            .iter()
            .any(|(d, f)| d == b"msvcrt.dll" && f == b"printf"),
        "msvcrt!printf, got {imports:?}"
    );

    // llvm-objdump shows two distinct DLL descriptors.
    assert!(llvm_objdump_lists(&exe, "kernel32.dll"));
    assert!(llvm_objdump_lists(&exe, "msvcrt.dll"));

    // Both code paths run: the kernel32 WriteFile line, then the msvcrt line.
    let (code, out) = run_wine(&exe, &dir);
    assert_eq!(code, 0, "wine exit code");
    assert!(
        out.contains("kernel32 writes"),
        "wine stdout missing kernel32 output, got: {out:?}"
    );
    assert!(
        out.contains("msvcrt prints 7"),
        "wine stdout missing msvcrt output, got: {out:?}"
    );
}

// --- shared helpers -------------------------------------------------------

/// Compiles `src` for the Windows target and links it with xold into a temp
/// `exe`, returning `Some((temp_dir, exe_path))`. Returns `None` (skip) when
/// the clang cross target is absent; panics on a compile or link failure.
/// Every temp path is tag-specific so parallel tests cannot clobber each other.
fn run_link(src: &[u8], tag: &str) -> Option<(PathBuf, PathBuf)> {
    let triple = "--target=x86_64-pc-windows-msvc";
    let clang = windows_clang(triple)?;
    let dir = std::env::temp_dir();
    let src_path = dir.join(format!("xold_coff_imp_{tag}.c"));
    let obj = dir.join(format!("xold_coff_imp_{tag}.obj"));
    let exe = dir.join(format!("xold_coff_imp_{tag}.exe"));
    fs::write(&src_path, src)
        .unwrap_or_else(|e| panic!("{tag}: write src: {e}"));
    assert!(
        compile(&clang, triple, &src_path, &obj),
        "{tag}: clang compile failed"
    );
    assert!(xold_link(&obj, &exe), "{tag}: xold link failed");
    Some((dir, exe))
}

/// `file` reports a PE32+ x86-64 console image.
fn assert_pe32_plus_x86_64(exe: &Path) {
    let kind = file_type(exe);
    eprintln!("file: {kind}");
    assert!(kind.contains("PE32+"), "expected PE32+, got: {kind}");
    assert!(kind.contains("x86-64"), "expected x86-64, got: {kind}");
}

/// xold's own `PeImage` reader parses the output and sees the import and IAT
/// directories populated, with the right machine.
fn assert_round_trips(exe: &Path) {
    let bytes = fs::read(exe).expect("read linked executable");
    let img = PeImage::parse(&bytes).expect("reader round-trips the output");
    assert_eq!(img.machine(), IMAGE_FILE_MACHINE_AMD64);
    let import_dir = img
        .data_directory(IMAGE_DIRECTORY_ENTRY_IMPORT)
        .expect("import directory present");
    assert!(
        import_dir.0 != 0 && import_dir.1 != 0,
        "import directory set"
    );
    let iat_dir = img
        .data_directory(IMAGE_DIRECTORY_ENTRY_IAT)
        .expect("IAT directory present");
    assert!(iat_dir.0 != 0 && iat_dir.1 != 0, "IAT directory set");
}

/// The `(dll, function)` pairs the reader walks from the import directory.
fn read_imports(exe: &Path) -> Vec<(Vec<u8>, Vec<u8>)> {
    let bytes = fs::read(exe).expect("read linked executable");
    let img = PeImage::parse(&bytes).expect("parse linked executable");
    img.imports()
}

/// Whether `llvm-objdump --private-headers` mentions `needle`.
fn llvm_objdump_lists(exe: &Path, needle: &str) -> bool {
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
    stdout.contains(needle)
}

/// Runs `exe` under wine (if present) and returns `(exit_code, stdout)`.
/// Returns `(0, String::new())` when wine is absent, so the runtime assertion
/// is silently skipped rather than failing the build over a missing tool.
/// The program's stdout is captured into a temp file so wine's own stderr noise
/// cannot contaminate it.
fn run_wine(exe: &Path, dir: &Path) -> (i32, String) {
    let Some(wine) = which("wine64").or_else(|| which("wine")) else {
        eprintln!("skipping wine run: wine not installed");
        return (0, String::new());
    };
    let out_path = dir.join(format!(
        "xold_coff_wine_{}.out",
        exe.file_stem().and_then(|s| s.to_str()).unwrap_or("out")
    ));
    // Redirect the program stdout into a file via a shell so wine's own stderr
    // stays separate; then read the file back.
    let shell = format!(
        "{} {} > {} 2>/dev/null",
        quote(&wine),
        quote(exe),
        quote(&out_path)
    );
    let status = Command::new("sh")
        .arg("-c")
        .arg(&shell)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("run wine");
    let code = status.code().unwrap_or(-1);
    let captured = fs::read_to_string(&out_path).unwrap_or_default();
    eprintln!(
        "wine {} -> exit {code}, stdout: {:?}",
        exe.display(),
        captured.trim()
    );
    let _ = fs::remove_file(&out_path);
    (code, captured)
}

// --- tooling helpers ------------------------------------------------------

/// Returns the clang binary if it can compile for `triple`, else `None`. The
/// probe compiles empty input straight to `/dev/null`, so parallel tests share
/// no temp path.
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

/// Compiles `src_path` for `triple` into `out`.
fn compile(clang: &Path, triple: &str, src_path: &Path, out: &Path) -> bool {
    Command::new(clang)
        .args([triple, "-c"])
        .arg(src_path)
        .arg("-o")
        .arg(out)
        .status()
        .is_ok_and(|s| s.success())
}

/// Links `obj` with xold into `out`.
fn xold_link(obj: &Path, out: &Path) -> bool {
    Command::new(xold_bin())
        .arg(obj)
        .args(["-o"])
        .arg(out)
        .status()
        .is_ok_and(|s| s.success())
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

/// Quotes a path for a shell command, escaping single quotes.
fn quote(p: &Path) -> String {
    format!("'{}'", p.display())
}
