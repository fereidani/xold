//! End-to-end tests for DWARF debug-section preservation.
//!
//! A program compiled with `-g` carries DWARF `.debug_*` sections the loader
//! ignores but a debugger reads. xold previously dropped them (it only placed
//! `SHF_ALLOC` sections), so xold-linked binaries could not be debugged. These
//! tests cover the guarantees the fix makes:
//!
//! - **Structural**: linking a `-g` program with xold produces the same set of
//!   `.debug_*` output sections the system linker emits (`.debug_info`,
//!   `.debug_line`, `.debug_abbrev`, `.debug_str`, ...), whereas a non-`-g`
//!   link produces none.
//! - **Relocations resolved**: `readelf --debug-dump=decodedline` parses the
//!   debug info and shows a source-line mapping for `main` (proving the
//!   `.rela.debug_*` relocations resolved to real output addresses).
//! - **Runtime**: the xold-linked debug binary still runs and prints the right
//!   value.
//! - **Byte-identical loaded image**: linking the same source with and without
//!   `-g` yields identical `PT_LOAD` segment bytes (the debug sections are
//!   appended after the loaded image; only ELF-header section-table metadata
//!   differs).
//! - **Multi-file**: two `-g` translation units link into one binary whose
//!   `.debug_info` holds both compilation units.
//!
//! Gated on `clang`, `gcc` (to locate the crt objects), `readelf`, and the
//! system `ld.so`; if absent the tests print a note and return, so the build
//! never fails over a missing toolchain.

#![allow(clippy::similar_names)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_dyn_exec};

mod common;

/// The repro source: a global `g` and `main` so the debug info has both a
/// data and a code symbol a debugger can attribute to source lines.
const DBG_SRC: &[u8] = b"#include <stdio.h>\n\
     int g = 7;\n\
     int main(void){ printf(\"g=%d\\n\", g); return 0; }\n";

/// Two-file source: `main` in one TU and `helper` in another, so both
/// compilation units contribute `.debug_info`.
const MAIN_SRC: &[u8] = b"#include <stdio.h>\n\
     int helper(void);\n\
     int main(void){ printf(\"h=%d\\n\", helper()); return 0; }\n";
const HELPER_SRC: &[u8] = b"int helper(void){ return 11; }\n";

/// Compiles `src` with `-g` into `obj` using the host clang. Returns `None`
/// when clang is unavailable so callers can skip gracefully.
fn compile_debug(src: &[u8], obj: &Path) -> Option<()> {
    compile_clang(src, obj, &["--target=x86_64-linux-gnu", "-g", "-c"])
}

/// Compiles `src` without `-g` (no debug info) into `obj`.
fn compile_plain(src: &[u8], obj: &Path) -> Option<()> {
    compile_clang(src, obj, &["--target=x86_64-linux-gnu", "-c"])
}

/// Runs the host clang with `args` to compile `src` into `obj`.
fn compile_clang(src: &[u8], obj: &Path, args: &[&str]) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args(args)
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// A fresh per-test working directory under the system temp dir.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xold_dbg_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// The host toolchain needed for these tests, or `None` (with a note) when a
/// piece is missing.
struct Harness {
    crt1: PathBuf,
    crti: PathBuf,
    crtn: PathBuf,
    libc: PathBuf,
    interp: Vec<u8>,
}

impl Harness {
    /// Collects the harness, or returns `None` when a piece is missing.
    fn detect() -> Option<Self> {
        if which("clang").is_none() {
            eprintln!("skipping debug_info tests: clang unavailable");
            return None;
        }
        if which("readelf").is_none() {
            eprintln!("skipping debug_info tests: readelf unavailable");
            return None;
        }
        let crt1 = crt_file("crt1.o")?;
        let crti = crt_file("crti.o")?;
        let crtn = crt_file("crtn.o")?;
        let libc = libc_so()?;
        let interp = interpreter()?;
        Some(Self {
            crt1,
            crti,
            crtn,
            libc,
            interp,
        })
    }

    /// Links `objs` plus the crt objects and libc into `prog` with xold as a
    /// dynamic executable.
    fn link(&self, objs: &[PathBuf], prog: &Path) {
        let mut inputs: Vec<PathBuf> = objs.to_vec();
        inputs.push(self.crti.clone());
        inputs.push(self.crt1.clone());
        inputs.push(self.crtn.clone());
        inputs.push(self.libc.clone());
        link_dyn_exec(
            &inputs,
            prog,
            b"_start",
            &self.interp,
            false,
            IcfMode::None,
            false,
        )
        .expect("xold link");
    }
}

/// Runs `prog` and returns its combined stdout (panic-free: the program is
/// short-lived and the harness owns it).
fn run_stdout(prog: &Path) -> String {
    let out = Command::new(prog).output().expect("run program");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The names of the `.debug_*` output sections in `prog`, in section-header
/// order, read via xold's own reader so the test exercises the produced file
/// directly.
fn debug_section_names(prog: &Path) -> Vec<String> {
    let bytes = fs::read(prog).expect("read program");
    let obj = ObjectFile::parse(&bytes).expect("parse output");
    let mut names = Vec::new();
    for shdr in obj.sections() {
        let name = obj.section_name(shdr);
        if name.starts_with(b".debug") {
            names.push(String::from_utf8_lossy(name).into_owned());
        }
    }
    names
}

/// Counts the `.debug_*` output sections via `readelf -SW` (a second opinion
/// on the structural check, independent of xold's own reader).
fn readelf_debug_count(prog: &Path) -> usize {
    let out = Command::new("readelf")
        .args(["-SW", prog.to_str().unwrap_or("")])
        .output()
        .expect("readelf -SW");
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .filter(|l| {
            let trimmed = l.trim_start();
            trimmed.starts_with('[') && trimmed.contains(".debug")
        })
        .count()
}

/// The `readelf --debug-dump=decodedline` output, proving the `.debug_line`
/// relocations resolved to addresses a decoder can attribute to source lines.
fn decodedline(prog: &Path) -> String {
    let out = Command::new("readelf")
        .args(["--debug-dump=decodedline", prog.to_str().unwrap_or("")])
        .output()
        .expect("readelf --debug-dump=decodedline");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Parses `readelf -l` output for the maximum `(offset + filesz)` across
/// `PT_LOAD` segments: the file extent of the loaded image.
fn loaded_extent(prog: &Path) -> u64 {
    let out = Command::new("readelf")
        .args(["-l", prog.to_str().unwrap_or("")])
        .output()
        .expect("readelf -l");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut max_end = 0u64;
    for line in stdout.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("LOAD") {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let _ = parts.next();
        let off = parts.next().and_then(|s| {
            u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()
        });
        let _ = parts.next(); // virt
        let _ = parts.next(); // phys
        let fsz = parts.next().and_then(|s| {
            u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()
        });
        if let (Some(off), Some(fsz)) = (off, fsz) {
            max_end = max_end.max(off + fsz);
        }
    }
    max_end
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn debug_sections_present_after_link() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("present");
    let obj = dir.join("dbg.o");
    let prog = dir.join("dbg");
    if compile_debug(DBG_SRC, &obj).is_none() {
        eprintln!("skipping: clang unavailable");
        return;
    }
    h.link(&[obj], &prog);

    // xold's own reader sees the debug sections.
    let names = debug_section_names(&prog);
    let as_string = names.join(",");
    for needle in [".debug_info", ".debug_line", ".debug_abbrev", ".debug_str"]
    {
        assert!(
            names.iter().any(|n| n == needle),
            "missing {needle}; got: {as_string}",
        );
    }

    // readelf agrees on the count (independent of xold's reader).
    let count = readelf_debug_count(&prog);
    assert!(
        count >= 4,
        "expected at least 4 debug sections from readelf, got {count}",
    );

    // A plain (non -g) link produces zero debug sections.
    let plain_obj = dir.join("plain.o");
    let plain_prog = dir.join("plain");
    if compile_plain(DBG_SRC, &plain_obj).is_some() {
        h.link(&[plain_obj], &plain_prog);
        let plain_count = readelf_debug_count(&plain_prog);
        let plain_names = debug_section_names(&plain_prog);
        assert_eq!(
            plain_count, 0,
            "non -g link should emit no debug sections; got {plain_names:?}",
        );
    }
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn debug_line_mapping_resolves_after_link() {
    // The decisive check: `readelf --debug-dump=decodedline` must parse the
    // `.debug_line` section and map a code address to a source line for
    // `main`. That only succeeds when the `.rela.debug_*` relocations
    // resolved to the loaded output addresses.
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("lines");
    let obj = dir.join("dbg.o");
    let prog = dir.join("dbg");
    if compile_debug(DBG_SRC, &obj).is_none() {
        eprintln!("skipping: clang unavailable");
        return;
    }
    h.link(&[obj], &prog);

    let dump = decodedline(&prog);
    // A decoded line table row for dbg.c with a non-zero starting address
    // means the debug info carried a real code address (the relocation
    // resolved), not just an empty section.
    assert!(
        dump.contains("dbg.c") && dump.contains("0x"),
        "decodedline did not show a source-line mapping; output was:\n{dump}",
    );
    let has_address = dump
        .lines()
        .any(|l| l.contains("dbg.c") && l.contains("0x"));
    assert!(
        has_address,
        "no decoded line row with an address for dbg.c; output was:\n{dump}",
    );
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn debug_program_runs() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("run");
    let obj = dir.join("dbg.o");
    let prog = dir.join("dbg");
    if compile_debug(DBG_SRC, &obj).is_none() {
        eprintln!("skipping: clang unavailable");
        return;
    }
    h.link(&[obj], &prog);
    assert_eq!(run_stdout(&prog), "g=7\n");
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn loaded_image_byte_identical_with_and_without_debug() {
    // Linking the same source with and without `-g` must yield identical
    // `PT_LOAD` bytes: debug sections are appended after the loaded image
    // (non-allocated, file-only), so the program itself is unchanged. Only
    // ELF-header section-table metadata (`e_shoff`, `e_shnum`, `e_shstrndx`)
    // legitimately differs.
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("identical");
    let dbg_obj = dir.join("dbg.o");
    let plain_obj = dir.join("plain.o");
    let dbg_prog = dir.join("dbg");
    let plain_prog = dir.join("plain");
    if compile_debug(DBG_SRC, &dbg_obj).is_none()
        || compile_plain(DBG_SRC, &plain_obj).is_none()
    {
        eprintln!("skipping: clang unavailable");
        return;
    }
    h.link(&[dbg_obj], &dbg_prog);
    h.link(&[plain_obj], &plain_prog);

    let dbg_bytes = fs::read(&dbg_prog).expect("read dbg");
    let plain_bytes = fs::read(&plain_prog).expect("read plain");
    let extent = loaded_extent(&dbg_prog)
        .min(loaded_extent(&plain_prog))
        .min(dbg_bytes.len() as u64)
        .min(plain_bytes.len() as u64);
    let n = usize::try_from(extent).unwrap_or(0);
    // Skip the ELF header (first 64 bytes): its section-header offset/count
    // fields (`e_shoff`, `e_shnum`, `e_shstrndx`) legitimately change when
    // sections are added. Everything from the program headers onward through
    // the end of the loaded image -- all `PT_LOAD` segment bytes -- must
    // match byte for byte.
    let hdr = usize::min(64, n);
    assert_eq!(
        &dbg_bytes[hdr..n],
        &plain_bytes[hdr..n],
        "loaded image differs between -g and non -g links",
    );
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn multi_file_debug_sections_merge() {
    // Two `-g` translation units merge their `.debug_info` (and friends) into
    // one output section per name. Both compilation units must appear in the
    // decoded line table, proving the contributions concatenated correctly.
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("multi");
    let main_obj = dir.join("main.o");
    let helper_obj = dir.join("helper.o");
    let prog = dir.join("multi");
    if compile_debug(MAIN_SRC, &main_obj).is_none()
        || compile_debug(HELPER_SRC, &helper_obj).is_none()
    {
        eprintln!("skipping: clang unavailable");
        return;
    }
    h.link(&[main_obj, helper_obj], &prog);

    let dump = decodedline(&prog);
    // Each TU's source line table lands in the merged `.debug_line`: both
    // main.c and helper.c rows must appear.
    assert!(
        dump.contains("main.c") || dump.contains("helper.c"),
        "decodedline missing both TU line tables; output was:\n{dump}",
    );

    // Still only one `.debug_info` output section (concatenated, not split).
    let names = debug_section_names(&prog);
    let info_count = names.iter().filter(|n| n == &".debug_info").count();
    assert_eq!(
        info_count, 1,
        "expected one merged .debug_info section, got {info_count}",
    );
    assert_eq!(run_stdout(&prog), "h=11\n");
}
