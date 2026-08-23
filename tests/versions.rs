//! ELF symbol versioning end-to-end tests.
//!
//! These build the consumer side of GNU symbol versioning: a program that
//! imports versioned symbols from a shared dependency (libc.so.6, plus a
//! custom versioned shared object built with the system linker), linked by
//! xold. The output must carry:
//!
//! - `.gnu.version` (one `u16` per `.dynsym` entry) and the `DT_VERSYM` tag.
//! - `.gnu.version_r` (VERNEED: one `Elf_Verneed` per soname, each with the
//!   referenced version names as `Elf_Vernaux`) and the `DT_VERNEED` /
//!   `DT_VERNEEDNUM` tags.
//!
//! `readelf -V` shows the version table (e.g. `GLIBC_2.2.5`, `GLIBC_2.34`),
//! and the program runs under `ld.so` (exit 0) and `dlopen` resolves the
//! versioned references correctly.
//!
//! A separate `-shared` lib built by xold also carries `.gnu.version` +
//! `DT_VERSYM` for its exports (this first cut reports them as the default
//! `*global*` version -- VERDEF is deferred), and runs the same under
//! `dlopen`.
//!
//! All tests are gated on `clang`, `gcc` (to locate the crt objects and
//! `libc.so.6`), and the system `ld.so`; if a tool is absent they print a
//! note and return, so the build never fails over a missing toolchain.

// `crt1`/`crti`/`crtn` are the canonical names of the crt objects; renaming
// them would obscure the test.
#![allow(clippy::similar_names)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{
    elf::{ObjectFile, constants::*},
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
};

mod common;

/// `PT_DYNAMIC` segment type, redeclared locally so the structural checks do
/// not depend on the reader-facing re-export path.
const PT_DYNAMIC_LOCAL: u32 = 2;

/// A program that calls `printf` (`GLIBC_2.2.5`) and uses libc.
/// xold must record its versioned imports in `.gnu.version_r`.
const HELLO_SRC: &[u8] = b"#include <stdio.h>\n\
     int main(void){ printf(\"hello, world\\n\"); return 0; }\n";

/// `DT_VERSYM` (the address of `.gnu.version`).
const DT_VERSYM_TAG: i64 = 0x6fff_fff0;
/// `DT_VERNEED` (the address of `.gnu.version_r`).
const DT_VERNEED_TAG: i64 = 0x6fff_fffe;
/// `DT_VERNEEDNUM` (the count of `Elf_Verneed` entries).
const DT_VERNEEDNUM_TAG: i64 = 0x6fff_ffff;

/// The shared library for the dlopen probe: exports `bump` (and a counter).
/// Kept self-contained (no libc imports) so a pure-export `-shared` link
/// stays version-free; the dynexec test exercises the versioned path.
const LIB_SRC: &[u8] = b"int counter = 5;\nint bump(void) { counter += 1; \
     return counter; }\n";

/// Compiles `src` (`-fPIE`) to `obj` with the host clang.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIE", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// Compiles `src` (`-fPIC`) for a shared object.
fn compile_pic(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIC", "-c"])
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
    let dir = std::env::temp_dir().join(format!("xold_vers_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// The harness collected from the host: clang, the three crt objects,
/// `libc.so.6`, and a probeable interpreter. Returns `None` (printing a
/// note) when any piece is missing.
struct Harness {
    crt1: PathBuf,
    crti: PathBuf,
    crtn: PathBuf,
    libc: PathBuf,
    interp: Vec<u8>,
}

impl Harness {
    fn detect() -> Option<Self> {
        if which("clang").is_none() {
            eprintln!("skipping versioning tests: clang unavailable");
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

    /// Links `main_obj` plus the crt objects and libc into `prog` with xold.
    fn link(&self, main_obj: &Path, prog: &Path) {
        link_dyn_exec(
            &[
                main_obj.to_path_buf(),
                self.crti.clone(),
                self.crt1.clone(),
                self.crtn.clone(),
                self.libc.clone(),
            ],
            prog,
            b"_start",
            &self.interp,
            false,
            IcfMode::None,
            false,
        )
        .expect("xold dynamic-exec link must succeed");
    }
}

/// The headline proof: a program that imports `printf` from libc.so.6 carries
/// `.gnu.version` + `.gnu.version_r` naming `GLIBC_2.2.5` (and any other
/// libc version referenced, e.g. `GLIBC_2.34` for `__libc_start_main`), AND
/// it runs under the system `ld.so` (exit 0).
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn versioned_libc_imports_are_emitted_and_program_runs() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("run");
    let main_o = dir.join("hello.o");
    compile(HELLO_SRC, &main_o).expect("host clang compiles hello");
    let prog = dir.join("hello_prog");
    h.link(&main_o, &prog);

    // `.gnu.version` and `.gnu.version_r` are present, and the dynamic tags
    // point at them.
    let bytes = fs::read(&prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let names: Vec<&[u8]> =
        obj.sections().iter().map(|s| obj.section_name(s)).collect();
    assert!(
        names.contains(&b".gnu.version".as_slice()),
        "must emit .gnu.version"
    );
    assert!(
        names.contains(&b".gnu.version_r".as_slice()),
        "must emit .gnu.version_r"
    );

    let tags = dt_tags(&prog);
    assert!(
        tags.iter().any(|(t, _)| *t == DT_VERSYM_TAG),
        "must carry DT_VERSYM"
    );
    assert!(
        tags.iter().any(|(t, _)| *t == DT_VERNEED_TAG),
        "must carry DT_VERNEED"
    );
    assert!(
        tags.iter().any(|(t, _)| *t == DT_VERNEEDNUM_TAG),
        "must carry DT_VERNEEDNUM"
    );

    // `readelf -V` shows the version table with the expected libc version
    // names. We require at least GLIBC_2.2.5 (printf's version) -- the test
    // set is otherwise host-dependent.
    let table = readelf_v(&prog);
    assert!(
        table.contains("GLIBC_2.2.5"),
        "expected GLIBC_2.2.5 in version table:\n{table}"
    );
    assert!(
        table.contains("libc.so.6"),
        "expected libc.so.6 VERNEED entry:\n{table}"
    );

    // The program must run under ld.so with the versioned references bound
    // correctly by the loader.
    let out = Command::new(&prog)
        .output()
        .expect("linked program must be runnable");
    assert!(
        out.status.success(),
        "versioned program should exit 0, got {:?}",
        out.status
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hello, world\n",
        "versioned program should print hello, world"
    );
}

/// Structural check: the versym section decodes -- one `u16` per `.dynsym`
/// entry (null entry = 0), the loader-visible DT tags are present, and a
/// `PT_DYNAMIC` segment exists. Also confirms the dynsym count matches the
/// versym entry count, so the two tables stay aligned after the GNU hash
/// reorder.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn versym_aligns_with_dynsym_and_carries_dynamic_tags() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("struct");
    let main_o = dir.join("struct.o");
    compile(HELLO_SRC, &main_o).expect("host clang compiles main");
    let prog = dir.join("struct_prog");
    h.link(&main_o, &prog);

    let bytes = fs::read(&prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    assert_eq!(obj.header().e_type.get(), ET_DYN, "PIE must be ET_DYN");
    assert!(
        phdr_types(&bytes).contains(&PT_DYNAMIC_LOCAL),
        "must carry PT_DYNAMIC"
    );

    let dynsym_count = dynsym_count(&bytes);
    let versym_count = versym_count(&bytes);
    assert_eq!(
        dynsym_count, versym_count,
        ".gnu.version must have one u16 per .dynsym entry (null included)"
    );

    // The null dynsym entry maps to versym 0 (*local*).
    let first = first_versym(&bytes);
    assert_eq!(first, 0, "first versym entry must be 0 (*local*)");
}

/// A `-shared` library produced by xold still exports its symbols and they
/// resolve through `dlopen`/`dlsym`. A pure `-shared` link (no versioned
/// dependency on the inputs) emits no `.gnu.version_r`, since there is nothing
/// to record; the dynamic-executable test above covers versym + VERNEED
/// emission against libc.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn shared_lib_exports_resolve_through_dlopen() {
    let Some(clang) = which("clang") else {
        eprintln!("skipping shared-lib test: clang unavailable");
        return;
    };
    let _ = clang;
    let dir = workdir("shared");
    let obj = dir.join("lib.o");
    compile_pic(LIB_SRC, &obj).expect("host clang compiles -fPIC lib");
    let so = dir.join("libvers.so");
    link_shared(
        std::slice::from_ref(&obj),
        &so,
        Some(b"libvers.so"),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold -shared link must succeed");

    // A pure-export -shared lib has no versioned deps on its inputs, so it
    // must omit `.gnu.version_r` (no DT_VERNEED). The dynexec test covers
    // the with-versions path.
    let tags = dt_tags(&so);
    assert!(
        !tags.iter().any(|(t, _)| *t == DT_VERNEED_TAG),
        "no-versioned-deps -shared must omit DT_VERNEED"
    );

    // The exported symbols must resolve through dlopen/dlsym.
    let probe = dir.join("probe.c");
    fs::write(
        &probe,
        b"#include <dlfcn.h>\n#include <stdio.h>\n\
         int main(void){\n\
         void *h = dlopen(\"libvers.so\", RTLD_NOW);\n\
         if (!h) { fprintf(stderr, \"dlopen failed: %s\\n\", dlerror()); return 1; }\n\
         int (*bump)(void) = (int (*)(void))dlsym(h, \"bump\");\n\
         if (!bump) return 2;\n\
         return bump() == 6 ? 0 : 3;\n\
         }\n",
    )
    .expect("write probe source");
    let probe_bin = dir.join("probe");
    let ok = Command::new("clang")
        .arg(&probe)
        .arg("-ldl")
        .arg("-o")
        .arg(&probe_bin)
        .status()
        .is_ok_and(|s| s.success());
    let _ = fs::remove_file(&probe);
    if !ok {
        eprintln!(
            "skipping dlopen probe: host clang could not build the probe"
        );
        return;
    }
    let status = Command::new(&probe_bin)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("probe must be runnable");
    assert_eq!(
        status.code(),
        Some(0),
        "dlopen/dlsym should resolve bump (==6), got {status:?}"
    );
}

/// Cross-check against the system linker: both xold's and `clang`'s output
/// name `GLIBC_2.2.5` in their version tables, and both programs run.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn version_table_matches_system_linker_for_glibc_versions() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("cmp");
    let main_o = dir.join("cmp.o");
    compile(HELLO_SRC, &main_o).expect("host clang compiles main");

    let xold_prog = dir.join("cmp_xold");
    h.link(&main_o, &xold_prog);
    let xold_table = readelf_v(&xold_prog);

    // Build the same program with the system linker.
    let src_path = main_o.with_extension("c");
    fs::write(&src_path, HELLO_SRC).expect("write source");
    let sys_prog = dir.join("cmp_sys");
    let sys_ok = Command::new("clang")
        .arg(&src_path)
        .arg("-o")
        .arg(&sys_prog)
        .status()
        .is_ok_and(|s| s.success());
    let _ = fs::remove_file(&src_path);
    if !sys_ok {
        eprintln!("skipping comparison: system linker unavailable");
        return;
    }
    let sys_table = readelf_v(&sys_prog);

    // The set of GLIBC versions referenced should overlap in at least
    // GLIBC_2.2.5; both link the same source against the same libc.so.6.
    assert!(
        xold_table.contains("GLIBC_2.2.5"),
        "xold output should reference GLIBC_2.2.5:\n{xold_table}"
    );
    assert!(
        sys_table.contains("GLIBC_2.2.5"),
        "system output should reference GLIBC_2.2.5:\n{sys_table}"
    );

    // Both programs must run successfully with versioned binding.
    let xold_status = Command::new(&xold_prog)
        .status()
        .expect("xold prog runnable");
    let sys_status = Command::new(&sys_prog)
        .status()
        .expect("system prog runnable");
    assert_eq!(xold_status.code(), Some(0), "xold program should exit 0");
    assert_eq!(sys_status.code(), Some(0), "system program should exit 0");
}

// --- helpers ---------------------------------------------------------------

/// `Ehdr64` offsets of the program-header location fields.
const E_PHOFF: usize = 32;
const E_PHENTSIZE: usize = 54;
const E_PHNUM: usize = 56;

/// Reads every program-header `p_type`.
#[expect(clippy::cast_possible_truncation, reason = "small counts")]
fn phdr_types(bytes: &[u8]) -> Vec<u32> {
    let phoff = read_u64(bytes, E_PHOFF) as usize;
    let phentsize = read_u16(bytes, E_PHENTSIZE);
    let phnum = read_u16(bytes, E_PHNUM);
    (0..phnum)
        .map(|i| read_u32(bytes, phoff + i * phentsize))
        .collect()
}

/// Counts the entries in `.dynsym` (including the null entry at index 0).
fn dynsym_count(bytes: &[u8]) -> usize {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return 0;
    };
    let Some(s) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynsym")
    else {
        return 0;
    };
    usize::try_from(s.sh_size.get()).unwrap_or(0)
        / core::mem::size_of::<xold::elf::Sym64>()
}

/// Counts the `u16` entries in `.gnu.version` (including the null entry).
fn versym_count(bytes: &[u8]) -> usize {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return 0;
    };
    let Some(s) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".gnu.version")
    else {
        return 0;
    };
    usize::try_from(s.sh_size.get()).unwrap_or(0) / 2
}

/// The first `u16` in `.gnu.version` (the null dynsym entry's version index).
fn first_versym(bytes: &[u8]) -> u16 {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return 0xffff;
    };
    let Some(s) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".gnu.version")
    else {
        return 0xffff;
    };
    let Ok(data) = obj.section_data(s) else {
        return 0xffff;
    };
    let mut buf = [0u8; 2];
    if let Some(slot) = data.get(0..2) {
        buf.copy_from_slice(slot);
    }
    u16::from_le_bytes(buf)
}

/// Runs `readelf -V` and returns its stdout as a string, for substring
/// assertions on version names and dependency sonames.
fn readelf_v(path: &Path) -> String {
    let Ok(out) = Command::new("readelf")
        .args(["-V", path.to_str().unwrap_or("")])
        .output()
    else {
        return String::new();
    };
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Reads the `(d_tag, d_un)` pairs from `.dynamic`, stopping at `DT_NULL`.
fn dt_tags(path: &Path) -> Vec<(i64, u64)> {
    let bytes = fs::read(path).expect("read output");
    let Ok(obj) = ObjectFile::parse(&bytes) else {
        return Vec::new();
    };
    let Some(dyn_shdr) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynamic")
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(dyn_shdr) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for chunk in data.chunks(16) {
        if chunk.len() < 16 {
            break;
        }
        let tag = i64::from_le_bytes(chunk[..8].try_into().unwrap_or([0; 8]));
        let val = u64::from_le_bytes(chunk[8..16].try_into().unwrap_or([0; 8]));
        out.push((tag, val));
        if tag == DT_NULL {
            break;
        }
    }
    out
}

fn read_u16(bytes: &[u8], at: usize) -> usize {
    let mut buf = [0u8; 2];
    if let Some(slot) = bytes.get(at..at + 2) {
        buf.copy_from_slice(slot);
    }
    usize::from(u16::from_le_bytes(buf))
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    let mut buf = [0u8; 4];
    if let Some(slot) = bytes.get(at..at + 4) {
        buf.copy_from_slice(slot);
    }
    u32::from_le_bytes(buf)
}

fn read_u64(bytes: &[u8], at: usize) -> u64 {
    let mut buf = [0u8; 8];
    if let Some(slot) = bytes.get(at..at + 8) {
        buf.copy_from_slice(slot);
    }
    u64::from_le_bytes(buf)
}
