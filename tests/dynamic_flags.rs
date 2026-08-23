//! `DT_FLAGS` and `DT_FLAGS_1` say what the loader has to know up front.
//!
//! Two facts about an image cannot be recovered from its relocations in time
//! to act on them, so the ABI puts them in flag words the loader reads first.
//!
//! `DF_STATIC_TLS` says the object's thread-local offsets are fixed at load
//! time, which is what an initial-exec reference asks for: it reads its offset
//! from a GOT slot the loader fills, and that offset only exists if the
//! object's block sits in the static TLS area. glibc reserves a slot when it
//! opens such an object and refuses the `dlopen` cleanly if it cannot; without
//! the flag the failure arrives later and less clearly, when a relocation
//! cannot be applied.
//!
//! `DF_1_PIE` says the image is a position-independent executable rather than
//! a library that happens to be `ET_DYN`. It is what makes glibc refuse to
//! `dlopen` a program, which is what keeps one from being loaded as a library
//! and running its startup a second time.
//!
//! xold emitted neither word. Both facts were already decided by the time
//! `.dynamic` was built; nothing wrote them down.
//!
//! Gated on `clang` and the system crt objects; without them the tests print a
//! note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{
    elf::ObjectFile,
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
};

mod common;

/// A shared object with an initial-exec thread-local.
const IE_SRC: &[u8] = b"_Thread_local int ie_var = 5;\n\
    int read_ie(void) { return ie_var; }\n";

/// A shared object with no thread-local storage at all.
const PLAIN_SRC: &[u8] = b"int plain(int x) { return x + 1; }\n";

const EXEC_SRC: &[u8] = b"int main(void) { return 0; }\n";

const DT_FLAGS: i64 = 30;
const DT_FLAGS_1: i64 = 0x6fff_fffb;
const DF_STATIC_TLS: u64 = 0x10;
const DF_1_PIE: u64 = 0x0800_0000;

/// A shared object with an initial-exec reference carries `DF_STATIC_TLS`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_initial_exec_shared_object_declares_static_tls() {
    let Some(dir) = workdir("statictls") else {
        return;
    };
    let Some(lib) = link_library(&dir, IE_SRC, "ie") else {
        return;
    };
    let bytes = fs::read(&lib).expect("read shared object");
    let flags = tag(&bytes, DT_FLAGS).expect(
        "an initial-exec reference needs a slot in the static TLS area, and \
         DT_FLAGS is where the loader is told to reserve one",
    );
    assert_eq!(
        flags & DF_STATIC_TLS,
        DF_STATIC_TLS,
        "DT_FLAGS must carry DF_STATIC_TLS, got {flags:#x}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A shared object without thread-local storage carries no `DT_FLAGS`, so the
/// flag reports a property rather than being stamped on everything.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_shared_object_without_tls_declares_nothing() {
    let Some(dir) = workdir("plain") else {
        return;
    };
    let Some(lib) = link_library(&dir, PLAIN_SRC, "plain") else {
        return;
    };
    let bytes = fs::read(&lib).expect("read shared object");
    assert_eq!(
        tag(&bytes, DT_FLAGS),
        None,
        "an object with no thread-local storage must not claim static TLS"
    );
    assert_eq!(tag(&bytes, DT_FLAGS_1), None, "and a library is not a PIE");
    let _ = fs::remove_dir_all(&dir);
}

/// A position-independent executable carries `DF_1_PIE`, and still runs.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_pie_executable_declares_itself_one() {
    let Some(dir) = workdir("pie") else {
        return;
    };
    let Some(prog) = link_exec(&dir) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");
    let flags = tag(&bytes, DT_FLAGS_1).expect(
        "glibc refuses to dlopen a PIE, and DT_FLAGS_1 is how it knows the \
         image is one",
    );
    assert_eq!(
        flags & DF_1_PIE,
        DF_1_PIE,
        "DT_FLAGS_1 must carry DF_1_PIE, got {flags:#x}"
    );
    let code = Command::new(&prog).status().expect("must run").code();
    assert_eq!(code, Some(0), "and the program must still run");
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping dynamic-flags {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_dynflags_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles one position-independent object, with the initial-exec model when
/// the source declares a thread-local.
fn compile(dir: &Path, src: &[u8], stem: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src_path = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clang)
        .args([
            "--target=x86_64-linux-gnu",
            "-fPIC",
            "-ftls-model=initial-exec",
            "-c",
        ])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping dynamic-flags: clang cannot build the fixture");
        return None;
    }
    Some(obj)
}

/// Links one source as a shared object.
fn link_library(dir: &Path, src: &[u8], stem: &str) -> Option<PathBuf> {
    let obj = compile(dir, src, stem)?;
    let lib = dir.join(format!("lib{stem}.so"));
    let res = link_shared(
        std::slice::from_ref(&obj),
        &lib,
        Some(b"libflags.so"),
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    Some(lib)
}

/// Links a position-independent dynamic executable against the host crt
/// objects.
fn link_exec(dir: &Path) -> Option<PathBuf> {
    let interp = interpreter()?;
    let obj = compile(dir, EXEC_SRC, "exec")?;
    let start = crt_file("Scrt1.o")?;
    let prologue = crt_file("crti.o")?;
    let epilogue = crt_file("crtn.o")?;
    let libc = libc_so()?;
    let prog = dir.join("prog");
    let res = link_dyn_exec(
        &[start, prologue, obj, libc, epilogue],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    Some(prog)
}

// --- readers ---------------------------------------------------------------

/// The value of the first `.dynamic` entry with tag `want`, if present.
fn tag(bytes: &[u8], want: i64) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynamic")?;
    let data = obj.section_data(shdr).ok()?;
    data.as_chunks::<16>().0.iter().find_map(|c| {
        let d_tag = i64::from_le_bytes(<[u8; 8]>::try_from(&c[0..8]).ok()?);
        if d_tag != want {
            return None;
        }
        Some(u64::from_le_bytes(<[u8; 8]>::try_from(&c[8..16]).ok()?))
    })
}
