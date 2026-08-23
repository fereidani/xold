//! End-to-end tests for `DT_INIT` and `DT_FINI`, the two tags that name the
//! single initialisation and termination functions the runtime calls around a
//! program's lifetime.
//!
//! These are not the `DT_*_ARRAY` tags (covered by `tests/init_array.rs`): the
//! arrays hold constructor pointers a compiler emits, whereas `_init` and
//! `_fini` are one function apiece, assembled from the `.init`/`.fini`
//! fragments in crti.o and crtn.o. An image can carry either without the
//! other, so the tags are keyed on the symbol, as lld keys them.
//!
//! The two guarantees the tests pin down are coupled:
//!
//! - The tags exist and hold the addresses of `_init`/`_fini`, which lie inside
//!   the `.init`/`.fini` sections.
//! - Under `--gc-sections` both sections survive whole and the tags still name
//!   the same bytes. Nothing in the link relocates to `_init`, so without an
//!   explicit root the collector takes the sections and leaves `DT_INIT`
//!   pointing at whatever the layout put in their place -- a crash before
//!   `main`. Only crti.o's fragment carries a name, so the section reservation
//!   matters as much as the symbol root: keeping the prologue and collecting
//!   crtn.o's return would run the runtime off the end of the function.
//!
//! An image that defines neither name must emit neither tag, with `.dynamic`
//! still exactly as long as the tags it does hold. The tag count is reserved
//! before addresses are known and the array is written afterwards; a
//! disagreement truncates the array or leaves it unterminated.
//!
//! Gated on `clang`, `gcc` (to locate the crt objects), and the system
//! `ld.so`; if absent the tests print a note and return, so the build never
//! fails over a missing toolchain.

#![allow(clippy::similar_names)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{
    elf::{ObjectFile, Shdr64, constants::*},
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
};

mod common;

/// A program that prints a marker, so a link whose `DT_INIT` points at
/// collected bytes shows up as a crash before the marker rather than as a
/// silently different exit status.
const MAIN_SRC: &[u8] = b"#include <stdio.h>\n\
     int main(void){ printf(\"ran\\n\"); return 0; }\n";

/// A translation unit with no C runtime in sight: it defines neither `_init`
/// nor `_fini`, so an image built from it alone must carry neither tag.
const PLAIN_SRC: &[u8] = b"int gv = 7;\n\
     int add(int x){ return x + gv; }\n";

/// Compiles `src` with the host clang using `args` into `obj`. Returns `None`
/// when clang is unavailable so callers can skip gracefully.
fn compile(src: &[u8], obj: &Path, args: &[&str]) -> Option<()> {
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
    let dir = std::env::temp_dir().join(format!("xold_initfini_{prefix}"));
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
            eprintln!("skipping init_fini tests: clang unavailable");
            return None;
        }
        Some(Self {
            crt1: crt_file("crt1.o")?,
            crti: crt_file("crti.o")?,
            crtn: crt_file("crtn.o")?,
            libc: libc_so()?,
            interp: interpreter()?,
        })
    }

    /// Links `main_obj` plus the crt objects and libc into `prog` with xold,
    /// with or without `--gc-sections`.
    fn link(&self, main_obj: &Path, prog: &Path, gc: bool) {
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
            gc,
            IcfMode::None,
            false,
        )
        .expect("xold libc link must succeed");
    }
}

/// Decodes the `.dynamic` section into `(d_tag, d_un)` rows, stopping at
/// `DT_NULL`.
fn dynamic_tags(obj: &ObjectFile<'_>) -> Vec<(i64, u64)> {
    let mut out = Vec::new();
    for chunk in dynamic_bytes(obj).as_chunks::<16>().0 {
        let tag = i64::from_le_bytes(chunk[..8].try_into().unwrap_or([0; 8]));
        if tag == DT_NULL {
            break;
        }
        let val = u64::from_le_bytes(chunk[8..16].try_into().unwrap_or([0; 8]));
        out.push((tag, val));
    }
    out
}

/// The raw `.dynamic` bytes, or empty when the image has no such section.
fn dynamic_bytes<'a>(obj: &ObjectFile<'a>) -> &'a [u8] {
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynamic")
        .and_then(|s| obj.section_data(s).ok())
        .unwrap_or(&[])
}

/// The value of `want`, or `None` when the image does not carry the tag.
fn tag(obj: &ObjectFile<'_>, want: i64) -> Option<u64> {
    dynamic_tags(obj)
        .into_iter()
        .find(|(t, _)| *t == want)
        .map(|(_, v)| v)
}

/// The section named `want`, which must exist.
fn section<'a>(obj: &'a ObjectFile<'_>, want: &[u8]) -> &'a Shdr64 {
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == want)
        .unwrap_or_else(|| {
            panic!("{} section must exist", String::from_utf8_lossy(want))
        })
}

/// The address of the defined symbol `want` in the output symbol table.
fn symbol_addr(obj: &ObjectFile<'_>, want: &[u8]) -> Option<u64> {
    let symtab = obj.symbol_table().ok()??;
    symtab
        .syms
        .iter()
        .find(|s| symtab.name(s) == want)
        .map(|s| s.st_value.get())
}

/// Asserts `.dynamic` holds exactly its tags plus one `DT_NULL` terminator,
/// and that its bytes do not run into the section placed after it.
///
/// This is what a tag count out of step with the serialiser breaks: reserving
/// too little makes `.dynamic` overrun its neighbour, reserving too much
/// leaves the array unterminated where the loader stops reading.
fn assert_dynamic_well_formed(obj: &ObjectFile<'_>) {
    let bytes = dynamic_bytes(obj);
    assert_ne!(bytes.len(), 0, ".dynamic must exist and be non-empty");
    assert_eq!(bytes.len() % 16, 0, ".dynamic must hold whole Dyn64 rows");
    let rows = bytes.len() / 16;
    let tags = dynamic_tags(obj).len();
    assert_eq!(
        rows,
        tags + 1,
        ".dynamic must hold its tags plus one DT_NULL and nothing else"
    );
    let last = &bytes[bytes.len() - 16..];
    let tag = i64::from_le_bytes(last[..8].try_into().unwrap_or([0; 8]));
    assert_eq!(tag, DT_NULL, ".dynamic must end with DT_NULL");

    let dynamic = section(obj, b".dynamic");
    let end = dynamic.sh_addr.get() + dynamic.sh_size.get();
    for s in obj.sections() {
        let addr = s.sh_addr.get();
        if s.sh_flags.get() & SHF_ALLOC == 0 || addr <= dynamic.sh_addr.get() {
            continue;
        }
        assert!(
            addr >= end,
            ".dynamic must not overrun {}",
            String::from_utf8_lossy(obj.section_name(s))
        );
    }
}

/// Asserts `addr` is the address of `sym` and lies inside the section `sect`.
fn assert_names_function(
    obj: &ObjectFile<'_>,
    addr: u64,
    sym: &[u8],
    sect: &[u8],
) {
    let name = String::from_utf8_lossy(sym).into_owned();
    assert_eq!(
        Some(addr),
        symbol_addr(obj, sym),
        "the tag must hold the address of {name}"
    );
    let s = section(obj, sect);
    let (start, size) = (s.sh_addr.get(), s.sh_size.get());
    assert!(
        addr >= start && addr < start + size,
        "{name} at {addr:#x} must lie inside {} [{start:#x}, {:#x})",
        String::from_utf8_lossy(sect),
        start + size
    );
}

/// A dynamic executable linked with crti.o/crtn.o must carry `DT_INIT` and
/// `DT_FINI` holding the addresses of `_init` and `_fini`, and must run.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn dyn_exec_names_init_and_fini() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("tags");
    let main_o = dir.join("tags.o");
    compile(
        MAIN_SRC,
        &main_o,
        &["--target=x86_64-linux-gnu", "-fPIE", "-c"],
    )
    .expect("host clang compiles the main source");
    let prog = dir.join("tags_prog");
    h.link(&main_o, &prog, false);

    let bytes = fs::read(&prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    assert_dynamic_well_formed(&obj);

    let init = tag(&obj, DT_INIT).expect("DT_INIT must exist");
    let fini = tag(&obj, DT_FINI).expect("DT_FINI must exist");
    assert_names_function(&obj, init, b"_init", b".init");
    assert_names_function(&obj, fini, b"_fini", b".fini");

    let out = Command::new(&prog)
        .output()
        .expect("linked program must be runnable");
    assert!(
        out.status.success(),
        "program should exit 0: {:?}",
        out.status
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ran\n");
}

/// `--gc-sections` must keep `.init` and `.fini` whole -- byte-for-byte what
/// the same link without the flag produced -- and the tags must still name
/// them. Only crti.o's fragment carries `_init`, so a symbol root alone would
/// keep the prologue and drop crtn.o's return.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn gc_sections_keeps_init_and_fini() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("gc");
    let main_o = dir.join("gc.o");
    compile(
        MAIN_SRC,
        &main_o,
        &["--target=x86_64-linux-gnu", "-fPIE", "-c"],
    )
    .expect("host clang compiles the main source");
    let plain = dir.join("gc_plain");
    let collected = dir.join("gc_prog");
    h.link(&main_o, &plain, false);
    h.link(&main_o, &collected, true);

    let plain_bytes = fs::read(&plain).expect("read plain output");
    let gc_bytes = fs::read(&collected).expect("read gc output");
    let plain_obj = ObjectFile::parse(&plain_bytes).expect("valid ELF");
    let gc_obj = ObjectFile::parse(&gc_bytes).expect("valid ELF");
    assert_dynamic_well_formed(&gc_obj);

    for (sym, name) in [(&b"_init"[..], &b".init"[..]), (b"_fini", b".fini")] {
        let want = section(&plain_obj, name);
        let got = section(&gc_obj, name);
        assert_eq!(
            got.sh_size.get(),
            want.sh_size.get(),
            "{} must survive gc whole",
            String::from_utf8_lossy(name)
        );
        assert_eq!(
            gc_obj.section_data(got).ok(),
            plain_obj.section_data(want).ok(),
            "{} must hold the same bytes with and without gc",
            String::from_utf8_lossy(name)
        );
        let which = if sym == b"_init" { DT_INIT } else { DT_FINI };
        let addr = tag(&gc_obj, which).expect("tag must survive gc");
        assert_names_function(&gc_obj, addr, sym, name);
    }

    let out = Command::new(&collected)
        .output()
        .expect("gc-linked program must be runnable");
    assert!(
        out.status.success(),
        "a DT_INIT pointing at collected bytes crashes before main: {:?}",
        out.status
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "ran\n");
}

/// An image whose inputs define neither `_init` nor `_fini` must emit neither
/// tag, and `.dynamic` must still be exactly as long as the tags it does hold.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn image_without_init_fini_emits_no_tags() {
    if which("clang").is_none() {
        eprintln!("skipping init_fini tests: clang unavailable");
        return;
    }
    let dir = workdir("none");
    let obj_path = dir.join("none.o");
    compile(
        PLAIN_SRC,
        &obj_path,
        &["--target=x86_64-linux-gnu", "-fPIC", "-c"],
    )
    .expect("host clang compiles the plain source");
    let so = dir.join("libnone.so");
    link_shared(
        &[obj_path],
        &so,
        Some(b"libnone.so"),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold shared link must succeed");

    let bytes = fs::read(&so).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    assert_eq!(
        symbol_addr(&obj, b"_init"),
        None,
        "the fixture must not define _init"
    );
    assert_eq!(tag(&obj, DT_INIT), None, "DT_INIT must not be emitted");
    assert_eq!(tag(&obj, DT_FINI), None, "DT_FINI must not be emitted");
    assert_dynamic_well_formed(&obj);
}
