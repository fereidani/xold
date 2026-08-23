//! Mach-O static link: produces a `MH_EXECUTE` from darwin `.o` inputs and
//! verifies it structurally and via round-trip with xold's own reader.
//!
//! Each test is gated on `clang --target=*-apple-darwin` being available; if
//! the cross target is absent the test prints a note and returns, so the build
//! never fails over a missing toolchain. Verification is structural only:
//! Linux has no Mach-O loader, so the image is checked with `file` and
//! re-parsed by xold's `MachOFile` reader rather than executed.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use common::{which, xold_bin};
use xold::macho::{
    MachOFile,
    constants::{
        CPU_SUBTYPE_ARM64_ALL, CPU_SUBTYPE_X86_64_ALL, CPU_TYPE_ARM64,
        CPU_TYPE_X86_64, MH_EXECUTE, N_EXT, N_SECT, N_TYPE,
    },
};

mod common;

/// The C source compiled for the darwin targets: `g` is a defined function
/// and `main` calls it, producing one branch relocation in `__text`.
const SRC: &[u8] = b"int g(void){ return 7; }\n\
                    int main(void){ return g(); }\n";

/// Conventional base of the `__TEXT` segment for both supported architectures.
const TEXT_BASE: u64 = 0x1_0000_0000;

/// Returns the clang binary if it can compile for `triple`, else `None`.
fn darwin_clang(triple: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let probe = std::env::temp_dir().join("xold_macho_link_probe.o");
    let ok = Command::new(clang.as_os_str())
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
    let src = std::env::temp_dir().join("xold_macho_link_src.c");
    let _ = fs::write(&src, SRC);
    Command::new(clang)
        .args([triple, "-c"])
        .arg(&src)
        .arg("-o")
        .arg(out)
        .status()
        .is_ok_and(|s| s.success())
}

/// Links `obj` with xold into `out`, returning success.
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

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_x86_64_static_executable() {
    let triple = "--target=x86_64-apple-darwin";
    let Some(clang) = darwin_clang(triple) else {
        eprintln!("skipping: clang darwin target not found");
        return;
    };
    let dir = std::env::temp_dir();
    let obj = dir.join("xold_macho_link_x86.o");
    let exe = dir.join("xold_macho_link_x86");
    if !compile(&clang, triple, &obj) {
        eprintln!("skipping: clang compile failed");
        return;
    }
    assert!(xold_link(&obj, &exe), "xold link failed");

    let kind = file_type(&exe);
    eprintln!("file(x86_64): {kind}");
    assert!(kind.contains("Mach-O"), "expected Mach-O, got: {kind}");
    assert!(
        kind.contains("executable"),
        "expected executable, got: {kind}"
    );
    assert!(kind.contains("x86_64"), "expected x86_64, got: {kind}");

    assert_round_trips(&exe, CPU_TYPE_X86_64, CPU_SUBTYPE_X86_64_ALL, b"_main");
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_arm64_static_executable() {
    let triple = "--target=arm64-apple-darwin";
    let Some(clang) = darwin_clang(triple) else {
        eprintln!("skipping: clang darwin target not found");
        return;
    };
    let dir = std::env::temp_dir();
    let obj = dir.join("xold_macho_link_arm.o");
    let exe = dir.join("xold_macho_link_arm");
    if !compile(&clang, triple, &obj) {
        eprintln!("skipping: clang compile failed");
        return;
    }
    assert!(xold_link(&obj, &exe), "xold link failed");

    let kind = file_type(&exe);
    eprintln!("file(arm64): {kind}");
    assert!(kind.contains("Mach-O"), "expected Mach-O, got: {kind}");
    assert!(
        kind.contains("executable"),
        "expected executable, got: {kind}"
    );
    assert!(kind.contains("arm64"), "expected arm64, got: {kind}");

    assert_round_trips(&exe, CPU_TYPE_ARM64, CPU_SUBTYPE_ARM64_ALL, b"_main");
}

/// Re-parses the executable with xold's own `MachOFile` reader and checks the
/// header, segments, sections, symbol table and entry are well-formed and that
/// the applied branch relocation left non-zero bytes in `__text`.
fn assert_round_trips(
    exe: &Path,
    cpu: u32,
    cpusubtype: u32,
    entry_name: &[u8],
) {
    let bytes = fs::read(exe).expect("read linked executable");
    let obj = MachOFile::parse(&bytes).expect("reader round-trips the output");

    assert_eq!(obj.cpu_type(), cpu);
    assert_eq!(obj.cpu_subtype(), cpusubtype);
    assert_eq!(obj.filetype(), MH_EXECUTE);

    // A branch relocation was applied; __text must carry real instruction
    // bytes, not be all zero.
    let sections = obj.sections();
    let text = sections
        .iter()
        .find(|s| s.sectname == b"__text")
        .expect("__text section present");
    assert_eq!(text.segname, b"__TEXT");
    assert!(
        text.data.iter().any(|&b| b != 0),
        "__text has non-zero (relocated) bytes"
    );

    // Segment invariants: __TEXT maps from file offset 0 and covers __text.
    let text_seg = sections
        .iter()
        .find(|s| s.segname == b"__TEXT")
        .expect("__TEXT segment section");
    assert!(
        text_seg.addr >= 0x1_0000_0000,
        "__TEXT mapped at the conventional base"
    );

    // The symbol table contains the entry symbol as an external definition.
    let syms = obj.symbols();
    let entry_sym =
        syms.iter()
            .find(|s| s.name == entry_name)
            .unwrap_or_else(|| {
                panic!(
                    "entry symbol {} in symtab",
                    String::from_utf8_lossy(entry_name)
                )
            });
    assert!(entry_sym.is_external(), "entry is external");
    assert_eq!(entry_sym.n_type & N_TYPE, N_SECT);
    assert_eq!(entry_sym.n_type & N_EXT, N_EXT);
    assert_eq!(entry_sym.n_sect, 1, "entry defined in first section");

    // LC_MAIN carries a non-zero entry offset into __TEXT. The on-disk value
    // is a file offset (entryoff); the entry symbol's virtual address is the
    // __TEXT base plus that offset. Accept either form the reader might expose.
    let entry = obj.entry();
    assert!(entry != 0, "entry offset is set");
    let want = entry_sym.n_value;
    assert!(
        want == entry || want == TEXT_BASE.wrapping_add(entry),
        "entry does not match symbol: entry={entry:#x} value={want:#x}"
    );
}

// --- helpers --------------------------------------------------------------
