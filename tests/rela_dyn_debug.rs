//! Regression tests for `.rela.dyn` when the inputs carry debug info.
//!
//! A `-g` object holds its DWARF in non-allocated `.debug_*` sections, and
//! those come with `R_X86_64_64` relocation tables of their own
//! (`.rela.debug_addr`, `.rela.debug_line`, ... all store code addresses).
//! Those sections have no runtime address, so they must contribute nothing to
//! `.rela.dyn`: the loader has no slot to fix up, and an entry naming one lands
//! at a bogus `r_offset`.
//!
//! xold used to size `.rela.dyn` from a walk that skipped them and fill it from
//! a walk that did not, so a `-g` dynamic link wrote more entries than the
//! region held and overran `.rela.plt`, faulting before `main`. These tests
//! pin the invariants that fix must keep:
//!
//! - The `.rela.dyn` section header size and `DT_RELASZ` agree, so the table
//!   the loader reads is exactly the table the layout reserved room for.
//! - No dynamic relocation points below the end of the program headers, which
//!   is where a `.debug_*` contribution offset would land when mistaken for a
//!   virtual address.
//! - Compiling the same source with `-g` adds no dynamic relocations at all.
//!
//! Gated on `clang` (and, for the executable case, the host crt objects and
//! `libc.so.6`); if a piece is missing the test prints a note and returns.

#![allow(clippy::similar_names, reason = "crt1/crti/crtn are the host names")]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, which};
use xold::{
    elf::{ObjectFile, constants::ET_DYN},
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
};

mod common;

/// On-disk size of one `Rela64`.
const RELA_SIZE: usize = 24;

/// A `-fPIC` fixture that stores absolute pointers in `.data`: `gp` points at
/// a global (an `R_X86_64_64` dynamic reloc) and `lp` at a static (an
/// `R_X86_64_RELATIVE`). Compiled `-g` it also emits `.debug_*` sections whose
/// relocation tables carry `R_X86_64_64` entries of their own.
const LIB_SRC: &[u8] = b"int counter = 5;\n\
    static int local = 7;\n\
    int *gp = &counter;\n\
    static int *lp = &local;\n\
    int bump(void) { counter += 1; return counter; }\n\
    int readlp(void) { return *lp; }\n";

/// A second translation unit. Two of them are what makes the case bite: each
/// output `.debug_*` section aggregates its members in order, and it is the
/// members after the first that carry a non-zero contribution offset -- the
/// value that used to be mistaken for a placed address.
const LIB2_SRC: &[u8] = b"static int other = 11;\n\
    static int *op = &other;\n\
    int readop(void) { return *op; }\n";

/// The library half of the executable fixture. Same as [`LIB_SRC`] without the
/// pointer to a defined global: an executable exports no defined globals, so
/// such a reference reaches no `.dynsym` entry and is counted but not emitted,
/// which would leave `.rela.dyn` over-reserved for a reason this test is not
/// about.
const EXEC_LIB_SRC: &[u8] = b"int counter = 5;\n\
    static int local = 7;\n\
    static int *lp = &local;\n\
    int bump(void) { counter += 1; return counter; }\n\
    int readlp(void) { return *lp; }\n";

/// The executable fixture: calls into the library half and prints the result,
/// so the linked image exercises a PLT alongside `.rela.dyn`.
const MAIN_SRC: &[u8] = b"#include <stdio.h>\n\
    int bump(void);\n\
    int readlp(void);\n\
    int main(void) { printf(\"%d\\n\", bump() + readlp()); return 0; }\n";

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_shared_object_built_from_debug_objects_sizes_rela_dyn_correctly() {
    let Some(dir) = workdir("shared") else {
        return;
    };
    let objs = vec![dir.join("lib.o"), dir.join("lib2.o")];
    let out = dir.join("lib.so");
    if compile_all(&objs, true).is_none() {
        eprintln!("skipping: clang unavailable");
        return;
    }
    link_shared(&objs, &out, None, false, IcfMode::None, false)
        .expect("xold -shared link must succeed");
    check_rela_dyn(&out, "-shared -g");
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn debug_info_adds_no_dynamic_relocations() {
    // The decisive invariant: DWARF lives in non-allocated sections, so it has
    // no runtime slot to relocate. Linking the same source with and without
    // `-g` must therefore produce the same number of dynamic relocations.
    let Some(dir) = workdir("count") else {
        return;
    };
    let debug_objs = vec![dir.join("dbg.o"), dir.join("dbg2.o")];
    let plain_objs = vec![dir.join("plain.o"), dir.join("plain2.o")];
    let debug_out = dir.join("dbg.so");
    let plain_out = dir.join("plain.so");
    if compile_all(&debug_objs, true).is_none()
        || compile_all(&plain_objs, false).is_none()
    {
        eprintln!("skipping: clang unavailable");
        return;
    }
    link_shared(&debug_objs, &debug_out, None, false, IcfMode::None, false)
        .expect("xold -shared link of the -g objects must succeed");
    link_shared(&plain_objs, &plain_out, None, false, IcfMode::None, false)
        .expect("xold -shared link of the plain objects must succeed");

    let with_debug = rela_dyn_offsets(&read(&debug_out));
    let without = rela_dyn_offsets(&read(&plain_out));
    assert!(
        !without.is_empty(),
        "fixture must produce dynamic relocations for the check to mean \
         anything",
    );
    assert_eq!(
        with_debug.len(),
        without.len(),
        "a -g build added {} dynamic relocation(s); DWARF is not allocated \
         and must add none",
        with_debug.len().saturating_sub(without.len()),
    );
    assert_eq!(with_debug, without, "the relocated slots must be the same");
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_dynamic_executable_built_from_debug_objects_runs() {
    // The failure this pins: the surplus `.rela.dyn` entries overwrote
    // `.rela.plt`, destroying the `JUMP_SLOT` relocations, and the program
    // faulted before reaching `main`.
    let Some(dir) = workdir("dynexec") else {
        return;
    };
    let Some(h) = Harness::detect() else {
        return;
    };
    let lib_obj = dir.join("lib.o");
    let main_obj = dir.join("main.o");
    let prog = dir.join("prog");
    if compile(EXEC_LIB_SRC, &lib_obj, true).is_none()
        || compile(MAIN_SRC, &main_obj, true).is_none()
    {
        eprintln!("skipping: clang unavailable");
        return;
    }
    let inputs = vec![
        h.crt1.clone(),
        h.crti.clone(),
        main_obj,
        lib_obj,
        h.libc.clone(),
        h.crtn.clone(),
    ];
    link_dyn_exec(
        &inputs,
        &prog,
        b"_start",
        &h.interp,
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-executable link must succeed");
    check_rela_dyn(&prog, "--dynamic-exec -g");

    let out = Command::new(&prog)
        .output()
        .expect("run the linked program");
    assert!(
        out.status.success(),
        "the linked program exited with {:?}",
        out.status.code(),
    );
    // `bump()` is 6 (the global incremented once) and `readlp()` is 7, read
    // through the static pointer the loader relocated.
    assert_eq!(String::from_utf8_lossy(&out.stdout), "13\n");
}

// --- checks ----------------------------------------------------------------

/// Asserts the two `.rela.dyn` invariants on the image at `path`: its section
/// header size equals `DT_RELASZ`, and no entry relocates a slot below the end
/// of the program headers.
fn check_rela_dyn(path: &Path, what: &str) {
    let bytes = read(path);
    let obj = ObjectFile::parse(&bytes).expect("output must be valid ELF");
    // The `r_offset` check below compares against a file offset, which only
    // the identity map of a position-independent image (load base zero) makes
    // meaningful. A PIE is also the only image kind that emits data
    // relocations at all, so a fixed-base result would mean the fixture
    // stopped covering the case.
    assert_eq!(
        obj.header().e_type.get(),
        ET_DYN,
        "{what}: image must be position independent",
    );

    let section = rela_dyn_section_size(&bytes);
    let relasz = dt_relasz(&bytes);
    assert_eq!(
        section, relasz,
        "{what}: .rela.dyn is {section} bytes but DT_RELASZ says {relasz}; \
         the loader would read past the section",
    );

    let limit = phdr_end(&bytes);
    for off in rela_dyn_offsets(&bytes) {
        assert!(
            off >= limit,
            "{what}: dynamic relocation at r_offset {off:#x} lands inside the \
             ELF or program headers (which end at {limit:#x})",
        );
    }
}

// --- image readers ---------------------------------------------------------

/// The `sh_size` of `.rela.dyn`, or zero when the image has no such section.
fn rela_dyn_section_size(bytes: &[u8]) -> u64 {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return 0;
    };
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == b".rela.dyn")
        .map_or(0, |s| s.sh_size.get())
}

/// The `DT_RELASZ` value from `.dynamic`, or zero when the tag is absent.
fn dt_relasz(bytes: &[u8]) -> u64 {
    /// `DT_RELASZ`.
    const TAG: i64 = 8;
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return 0;
    };
    let Some(shdr) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynamic")
    else {
        return 0;
    };
    let Ok(data) = obj.section_data(shdr) else {
        return 0;
    };
    for chunk in data.as_chunks::<16>().0 {
        let tag = read_i64(chunk, 0);
        if tag == TAG {
            return read_u64(chunk, 8);
        }
    }
    0
}

/// The `r_offset` of every entry in `.rela.dyn`, in table order.
fn rela_dyn_offsets(bytes: &[u8]) -> Vec<u64> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(shdr) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".rela.dyn")
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(shdr) else {
        return Vec::new();
    };
    data.as_chunks::<RELA_SIZE>()
        .0
        .iter()
        .map(|entry| read_u64(entry, 0))
        .collect()
}

/// One past the last program-header byte. Nothing loaded lives below it, so no
/// relocated slot may either.
fn phdr_end(bytes: &[u8]) -> u64 {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return 0;
    };
    let hdr = obj.header();
    let count = u64::from(hdr.e_phnum.get());
    let size = u64::from(hdr.e_phentsize.get());
    hdr.e_phoff.get().saturating_add(count.saturating_mul(size))
}

fn read_u64(bytes: &[u8], at: usize) -> u64 {
    let mut buf = [0u8; 8];
    if let Some(slot) = bytes.get(at..at + 8) {
        buf.copy_from_slice(slot);
    }
    u64::from_le_bytes(buf)
}

fn read_i64(bytes: &[u8], at: usize) -> i64 {
    read_u64(bytes, at).cast_signed()
}

/// Reads a linked image, failing the test if it is unreadable.
fn read(path: &Path) -> Vec<u8> {
    fs::read(path).expect("linked image must be readable")
}

// --- host toolchain --------------------------------------------------------

/// The crt objects, libc and interpreter a dynamic-executable link needs.
struct Harness {
    /// The position-independent C runtime startup object (`Scrt1.o`). The
    /// non-PIE `crt1.o` would force a fixed-base image, which emits no data
    /// relocations and so would not exercise `.rela.dyn` at all.
    crt1: PathBuf,
    crti: PathBuf,
    crtn: PathBuf,
    libc: PathBuf,
    interp: Vec<u8>,
}

impl Harness {
    /// Collects the harness, or returns `None` (with a note) when a piece is
    /// missing.
    fn detect() -> Option<Self> {
        let crt1 = crt_file("Scrt1.o")?;
        let crti = crt_file("crti.o")?;
        let crtn = crt_file("crtn.o")?;
        let libc = crt_file("libc.so.6")?;
        let interp = interpreter()?;
        Some(Self {
            crt1,
            crti,
            crtn,
            libc,
            interp,
        })
    }
}

/// Compiles the two library translation units into `objs`, with DWARF when
/// `debug`. `objs` is `[LIB_SRC, LIB2_SRC]` in that order.
fn compile_all(objs: &[PathBuf], debug: bool) -> Option<()> {
    for (src, obj) in [LIB_SRC, LIB2_SRC].iter().zip(objs) {
        compile(src, obj, debug)?;
    }
    Some(())
}

/// Compiles `src` into `obj` as a `-fPIC` object, with DWARF when `debug`.
fn compile(src: &[u8], obj: &Path, debug: bool) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write fixture source");
    let mut cmd = Command::new(clang);
    cmd.args(["--target=x86_64-linux-gnu", "-fPIC", "-c"]);
    if debug {
        cmd.arg("-g");
    }
    let ok = cmd
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// A fresh per-test working directory, or `None` when clang is unavailable.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping rela_dyn_debug tests: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("xold_rela_dyn_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}
