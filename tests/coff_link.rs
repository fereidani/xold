//! COFF/PE static link: produces a PE32+ x86-64 executable from a Windows
//! `.obj` input and verifies it structurally, via xold's own `PeImage` reader,
//! via `llvm-objdump`, and (stretch) by running it under `wine`.
//!
//! Each test is gated on `clang --target=x86_64-pc-windows-msvc` being
//! available; if the cross target is absent the test prints a note and returns,
//! so the build never fails over a missing toolchain. The `wine` runtime check
//! is gated separately on `wine64` being present and runs only when the stretch
//! path is reached.

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
        IMAGE_FILE_EXECUTABLE_IMAGE, IMAGE_FILE_LARGE_ADDRESS_AWARE,
        IMAGE_FILE_MACHINE_AMD64, IMAGE_NT_OPTIONAL_HDR64_MAGIC,
        IMAGE_SUBSYSTEM_WINDOWS_CUI,
    },
};

mod common;

/// The C source compiled for the Windows target: `g` returns 7 and `main`
/// calls it, producing one `REL32` relocation in `.text`.
const SRC: &[u8] = b"int g(void){ return 7; }\n\
                    int main(void){ return g(); }\n";

/// The conventional load address of an x86-64 executable image.
const IMAGE_BASE: u64 = 0x0000_0001_4000_0000;

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_x86_64_pe_executable() {
    let triple = "--target=x86_64-pc-windows-msvc";
    let Some(clang) = windows_clang(triple) else {
        eprintln!("skipping: clang msvc target not found");
        return;
    };
    let dir = std::env::temp_dir();
    let obj = dir.join("xold_coff_link_x86.obj");
    let exe = dir.join("xold_coff_link_x86.exe");
    if !compile(&clang, triple, &obj) {
        eprintln!("skipping: clang compile failed");
        return;
    }
    assert!(xold_link(&obj, &exe), "xold link failed");

    let kind = file_type(&exe);
    eprintln!("file: {kind}");
    assert!(kind.contains("PE32+"), "expected PE32+, got: {kind}");
    assert!(
        kind.contains("console") || kind.contains("Console"),
        "expected console subsystem, got: {kind}"
    );
    assert!(kind.contains("x86-64"), "expected x86-64, got: {kind}");

    assert_round_trips(&exe);
    assert!(llvm_objdump_parses(&exe));
    assert_wine_exit_code(&exe);
}

/// Re-parses the executable with xold's own `PeImage` reader and checks the
/// machine, optional header, sections, entry and import table.
fn assert_round_trips(exe: &Path) {
    let bytes = fs::read(exe).expect("read linked executable");
    let img = PeImage::parse(&bytes).expect("reader round-trips the output");

    assert_eq!(img.machine(), IMAGE_FILE_MACHINE_AMD64);
    assert!(img.is_x86_64());
    assert_eq!(img.magic(), IMAGE_NT_OPTIONAL_HDR64_MAGIC);
    assert_eq!(img.image_base(), IMAGE_BASE);
    assert_eq!(img.subsystem(), IMAGE_SUBSYSTEM_WINDOWS_CUI);
    assert!(
        img.characteristics() & IMAGE_FILE_EXECUTABLE_IMAGE != 0,
        "executable image flag set"
    );
    assert!(
        img.characteristics() & IMAGE_FILE_LARGE_ADDRESS_AWARE != 0,
        "large address aware flag set"
    );

    // `.text` is the first section and carries the entry point.
    let sections = img.sections();
    let text = sections
        .iter()
        .find(|s| s.name == b".text")
        .expect(".text section present");
    assert_eq!(text.virtual_address, 0x1000, ".text maps at 0x1000");
    assert_eq!(img.address_of_entry_point(), text.virtual_address);
    assert!(
        text.data.iter().any(|&b| b != 0),
        ".text has non-zero (relocated) bytes"
    );

    // The entry stub calls the ExitProcess IAT slot, so the entry must not be
    // the user `main`/`g` (both follow the stub). The call leaves a relative
    // displacement; at minimum the relocated bytes differ from a zeroed stub.
    let stub = text
        .data
        .get(..19)
        .expect("entry stub occupies the first 19 bytes");
    assert_eq!(
        stub[0..4],
        [0x48, 0x83, 0xec, 0x28],
        "sub rsp, 0x28 prologue"
    );
    assert_eq!(stub[4], 0xe8, "direct call opcode to entry");
    assert_eq!(stub[9..11], [0x89, 0xc1], "mov ecx, eax");
    assert_eq!(stub[11..13], [0xff, 0x15], "indirect call to IAT slot");
    assert_eq!(stub[17..19], [0x0f, 0x0b], "ud2 sentinel");

    // Import and IAT data directories are populated.
    let import_dir = img
        .data_directory(IMAGE_DIRECTORY_ENTRY_IMPORT)
        .expect("import directory present");
    assert!(import_dir.0 != 0, "import directory has an RVA");
    assert!(import_dir.1 != 0, "import directory has a size");
    let iat_dir = img
        .data_directory(IMAGE_DIRECTORY_ENTRY_IAT)
        .expect("IAT directory present");
    assert!(iat_dir.0 != 0, "IAT directory has an RVA");

    // The import table names kernel32.dll!ExitProcess.
    let imports = img.imports();
    assert!(
        imports
            .iter()
            .any(|(dll, func)| dll == b"kernel32.dll" && func == b"ExitProcess"),
        "imports kernel32.dll!ExitProcess, got: {imports:?}"
    );
}

/// Whether `llvm-objdump --private-headers` parses the executable (and reports
/// the PE32+ magic and the `ExitProcess` import).
fn llvm_objdump_parses(exe: &Path) -> bool {
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
    stdout.contains("PE32+") && stdout.contains("ExitProcess")
}

/// Stretch goal: run the executable under `wine` and verify it exits 7 (the
/// value `g` returns). Skipped silently if `wine64` is absent.
fn assert_wine_exit_code(exe: &Path) {
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
    eprintln!("wine ./p.exe -> exit {code}");
    assert_eq!(code, 7, "wine should report exit code 7 (ExitProcess)");
}

// --- helpers --------------------------------------------------------------

/// Returns the clang binary if it can compile for `triple`, else `None`.
fn windows_clang(triple: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let probe = std::env::temp_dir().join("xold_coff_link_probe.o");
    let ok = Command::new(&clang)
        .args([triple, "-c", "-x", "c", "-", "-o"])
        .arg(&probe)
        .stdin(Stdio::null())
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&probe);
    ok.then_some(clang)
}

/// Compiles `SRC` for `triple` into `out`.
fn compile(clang: &Path, triple: &str, out: &Path) -> bool {
    let src = std::env::temp_dir().join("xold_coff_link_src.c");
    let _ = fs::write(&src, SRC);
    Command::new(clang)
        .args([triple, "-c"])
        .arg(&src)
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
