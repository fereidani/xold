//! Initial-exec access to a *hidden* thread-local in a shared object.
//!
//! An initial-exec reference loads its offset from a GOT slot. In a shared
//! object that offset is the loader's to choose -- the object does not know
//! where in the process's static TLS area its block will land -- so the slot
//! needs a dynamic relocation. xold emitted one only when the thread-local had
//! a `.dynsym` row to name, and a hidden-visibility global has none.
//!
//! The slot was then left holding what `fill_got` put there: the offset from
//! an *executable's* thread pointer, computed from this image's own block as
//! if it were the main module. Nothing corrected it, so the program read
//! whatever happened to live at that offset from the real thread pointer. A
//! link that succeeds and reads the wrong thread-local memory is the exact
//! failure this project's TLS notes say must never be produced quietly.
//!
//! A hidden thread-local needs no name, and that is the point: a name no other
//! image can bind cannot be preempted, so the offset within this image's block
//! is the whole answer, and it goes in the relocation's addend with `r_sym`
//! zero. lld emits the same row from
//! `addAddendOnlyRelocIfNonPreemptible`; checked against it on this host, and
//! against glibc by running.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::ObjectFile, icf::IcfMode, linker::link_shared,
    reloc::x86_64::R_X86_64_TPOFF64,
};

mod common;

/// Two ordinary thread-locals ahead of the hidden one, so its offset within
/// the block is not zero and a wrong addend is visible.
const LIB_SRC: &[u8] = b"_Thread_local int pad_a = 1;\n\
    _Thread_local int pad_b = 2;\n\
    __attribute__((visibility(\"hidden\"))) _Thread_local int hidden_tl = 7;\n\
    int read_hidden(void) { return hidden_tl; }\n\
    void set_hidden(int v) { hidden_tl = v; }\n";

/// The harness: read the hidden thread-local, write it, read it back. Both
/// halves have to reach the same storage, and the first has to see the
/// initialiser.
const MAIN_SRC: &[u8] = b"#include <stdio.h>\n\
    extern int read_hidden(void);\n\
    extern void set_hidden(int);\n\
    /* The program's own thread-locals take the storage closest to the thread\n\
       pointer, so the library's block is pushed away from where it would sit\n\
       if it were the main module. Without them the executable-convention\n\
       constant xold used to leave in the GOT slot happens to name the right\n\
       memory, and this test proves nothing. */\n\
    _Thread_local long prog_tls[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };\n\
    int main(void)\n\
    {\n\
        int first = read_hidden();\n\
        set_hidden(99);\n\
        int second = read_hidden();\n\
        printf(\"ie %d %d %ld\\n\", first, second, prog_tls[0]);\n\
        return (first == 7 && second == 99 && prog_tls[0] == 1) ? 0 : 1;\n\
    }\n";

/// `hidden_tl`'s offset within the TLS block: two 4-byte thread-locals ahead
/// of it, and it is 4-byte aligned.
const HIDDEN_OFFSET: i64 = 8;

/// The GOT slot carries a `TPOFF64` naming no symbol, with the block offset in
/// the addend.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_hidden_thread_locals_slot_carries_an_addend_only_tpoff() {
    let Some(dir) = workdir("reloc") else {
        return;
    };
    let Some(lib) = link_library(&dir) else {
        return;
    };
    let bytes = fs::read(&lib).expect("read shared object");
    let rows = rela_dyn(&bytes);
    let tpoff: Vec<_> = rows
        .iter()
        .filter(|r| r.r_type == R_X86_64_TPOFF64)
        .collect();
    assert_eq!(
        tpoff.len(),
        1,
        "the one initial-exec slot needs exactly one TPOFF64; got {rows:#x?}"
    );
    let row = tpoff[0];
    assert_eq!(
        row.sym, 0,
        "a hidden thread-local has no .dynsym row to name, and needs none: \
         nothing outside this image can preempt it"
    );
    assert_eq!(
        row.addend, HIDDEN_OFFSET,
        "the addend is the offset within this image's TLS block, which is the \
         whole of what the loader needs"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The behaviour behind the row: the accesses reach the storage the thread-
/// local actually occupies.
///
/// This is the test that matters. A wrong constant in the slot still links and
/// still runs; it just reads somewhere else in the thread area, which only
/// executing the program shows.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_hidden_thread_local_reads_and_writes_its_own_storage() {
    let Some(dir) = workdir("run") else {
        return;
    };
    let Some(lib) = link_library(&dir) else {
        return;
    };
    let Some(prog) = build_harness(&dir, &lib) else {
        return;
    };
    let out = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .output()
        .expect("harness must run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ie 7 99 1"),
        "the initial-exec accesses must reach the hidden thread-local's own \
         storage (got {stdout:?})"
    );
    assert_eq!(out.status.code(), Some(0), "and the harness must exit 0");
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping tls-hidden-ie {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_tlshiddenie_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the library with the initial-exec model and links it with
/// `xold -shared`.
fn link_library(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("ietls.c");
    let obj = dir.join("ietls.o");
    fs::write(&src, LIB_SRC).ok()?;
    let built = Command::new(&clang)
        .args([
            "--target=x86_64-linux-gnu",
            "-fPIC",
            "-ftls-model=initial-exec",
            "-c",
        ])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping tls-hidden-ie: clang cannot build the fixture");
        return None;
    }
    let lib = dir.join("libtlshiddenie.so");
    let res = link_shared(
        std::slice::from_ref(&obj),
        &lib,
        Some(b"libtlshiddenie.so"),
        false,
        IcfMode::None,
        false,
    );
    assert!(
        res.is_ok(),
        "an initial-exec hidden thread-local must link: {:?}",
        res.err()
    );
    Some(lib)
}

/// Builds the harness against the produced library with the host toolchain.
fn build_harness(dir: &Path, lib: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("iemain.c");
    fs::write(&src, MAIN_SRC).ok()?;
    let bin = dir.join("ieharness");
    let ok = Command::new(clang)
        .arg("-fPIE")
        .arg(&src)
        .arg(lib)
        .arg("-o")
        .arg(&bin)
        .arg("-Wl,-rpath")
        .arg(dir)
        .status()
        .ok()?
        .success();
    ok.then_some(bin)
}

// --- readers ---------------------------------------------------------------

/// The fields of one `.rela.dyn` row these tests reason about.
#[derive(Debug)]
struct Row {
    r_type: u32,
    sym: u32,
    addend: i64,
}

/// Reads `.rela.dyn` out of a linked image.
fn rela_dyn(bytes: &[u8]) -> Vec<Row> {
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
    data.as_chunks::<24>()
        .0
        .iter()
        .filter_map(|c| {
            let info = u64::from_le_bytes(<[u8; 8]>::try_from(&c[8..16]).ok()?);
            let addend =
                i64::from_le_bytes(<[u8; 8]>::try_from(&c[16..24]).ok()?);
            Some(Row {
                #[allow(clippy::cast_possible_truncation)]
                r_type: info as u32,
                #[allow(clippy::cast_possible_truncation)]
                sym: (info >> 32) as u32,
                addend,
            })
        })
        .collect()
}
