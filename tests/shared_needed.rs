//! `DT_NEEDED` in a shared object.
//!
//! A `-shared` link is not self-contained. When it imports a name from another
//! shared file it emits a `GLOB_DAT` or `JUMP_SLOT` against that name, and the
//! loader can only bind it if the library says which file to look in. That is
//! `DT_NEEDED`, and it belongs in every dynamic output, not only executables:
//! a library that omits it is underlinked, and `dlopen`ing it fails unless the
//! process happens to have loaded the dependency already for other reasons.
//!
//! The two tests below cover the tag and the behaviour it stands for: the
//! soname is recorded, and the produced library actually loads on its own.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_shared};

mod common;

/// `DT_NULL`, the terminator of the dynamic array.
const DT_NULL: i64 = 0;
/// `DT_NEEDED`: the `.dynstr` offset of a dependency's soname.
const DT_NEEDED: i64 = 1;
/// `DT_STRTAB`: the address of `.dynstr`, used here only to confirm the tag
/// set is otherwise well formed.
const DT_STRTAB: i64 = 5;

/// The dependency, built by the host toolchain.
const DEP_SRC: &[u8] = b"int dep_value(void) { return 42; }\n";

/// The library under test: it calls into the dependency, so the link imports
/// `dep_value` and the output must name the file it comes from.
const MID_SRC: &[u8] = b"extern int dep_value(void);\n\
    int mid_read(void) { return dep_value() + 1; }\n";

/// A `dlopen`/`dlsym` harness: it loads the xold-linked library by path with
/// `RTLD_NOW`, so every import must bind at load time, and calls through.
const HARNESS_SRC: &[u8] = b"#include <dlfcn.h>\n\
    #include <stdio.h>\n\
    int main(int argc, char **argv) {\n\
        if (argc < 2) return 2;\n\
        void *h = dlopen(argv[1], RTLD_NOW);\n\
        if (!h) { fprintf(stderr, \"%s\\n\", dlerror()); return 3; }\n\
        int (*f)(void) = (int (*)(void))dlsym(h, \"mid_read\");\n\
        if (!f) return 4;\n\
        return f();\n\
    }\n";

/// What the harness exits with once the dependency binds: 42 from the
/// dependency plus 1.
const EXPECTED_EXIT: i32 = 43;

/// The soname of a shared input reaches the output's `DT_NEEDED` list.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_shared_object_records_its_dependencies() {
    let Some(dir) = workdir("tag") else {
        return;
    };
    let Some(lib) = link_library(&dir) else {
        return;
    };
    let bytes = fs::read(&lib).expect("read shared object");
    let needed = needed_names(&bytes);
    assert!(
        needed.iter().any(|n| n == b"libsharedneed.so"),
        "a -shared output importing from a dependency must record it as \
         DT_NEEDED; got {needed:?}"
    );
    assert!(
        dt_tags(&bytes).contains(&DT_STRTAB),
        "the dynamic array must still be well formed around the new tag"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The behaviour the tag stands for: the library loads by itself. Without
/// `DT_NEEDED` the loader has no file to resolve `dep_value` in and `RTLD_NOW`
/// fails the open outright.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_shared_object_loads_without_its_dependency_preloaded() {
    let Some(dir) = workdir("run") else {
        return;
    };
    let Some(lib) = link_library(&dir) else {
        return;
    };
    let Some(harness) = build_harness(&dir) else {
        return;
    };
    let status = Command::new(&harness)
        .arg(&lib)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("harness must run");
    assert_eq!(
        status.code(),
        Some(EXPECTED_EXIT),
        "dlopen of the xold-linked library must resolve its imports"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping shared-needed {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_sharedneed_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the host dependency, then links `MID_SRC` against it with xold
/// `-shared`. Returns the produced library's path.
fn link_library(dir: &Path) -> Option<PathBuf> {
    let dep = build_dependency(dir)?;
    let obj = dir.join("sharedneed_mid.o");
    compile(MID_SRC, "c", &obj, &["-fPIC"])?;

    let lib = dir.join("libsharedneedmid.so");
    let linked = link_shared(
        &[obj, dep],
        &lib,
        Some(b"libsharedneedmid.so"),
        false,
        IcfMode::None,
        false,
    );
    assert!(
        linked.is_ok(),
        "xold -shared link must succeed: {:?}",
        linked.err()
    );
    Some(lib)
}

/// Builds the shared dependency with the host toolchain.
fn build_dependency(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let obj = dir.join("sharedneed_dep.o");
    compile(DEP_SRC, "c", &obj, &["-fPIC"])?;
    let so = dir.join("libsharedneed.so");
    let ok = Command::new(clang)
        .arg("-shared")
        .arg("-o")
        .arg(&so)
        .arg(&obj)
        .arg("-Wl,-soname,libsharedneed.so")
        .status()
        .ok()?
        .success();
    ok.then_some(so)
}

/// Builds the `dlopen` harness with the host toolchain.
fn build_harness(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("sharedneed_harness.c");
    fs::write(&src, HARNESS_SRC).ok()?;
    let bin = dir.join("sharedneed_harness");
    let ok = Command::new(clang)
        .arg(&src)
        .arg("-o")
        .arg(&bin)
        .arg("-ldl")
        .status()
        .ok()?
        .success();
    ok.then_some(bin)
}

/// Compiles `src` (written beside `obj` with extension `ext`) with the host
/// clang, passing `args` through.
fn compile(src: &[u8], ext: &str, obj: &Path, args: &[&str]) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension(ext);
    fs::write(&src_path, src).ok()?;
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c"])
        .args(args)
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    ok.then_some(())
}

// --- readers ---------------------------------------------------------------

/// The `.dynamic` array of `bytes` as (tag, value) pairs, terminated at
/// `DT_NULL`.
fn dyn_entries(bytes: &[u8]) -> Vec<(i64, u64)> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(shdr) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynamic")
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(shdr) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for chunk in data.as_chunks::<16>().0 {
        let (Ok(tag), Ok(val)) = (
            <[u8; 8]>::try_from(&chunk[..8]),
            <[u8; 8]>::try_from(&chunk[8..]),
        ) else {
            break;
        };
        let tag = i64::from_le_bytes(tag);
        out.push((tag, u64::from_le_bytes(val)));
        if tag == DT_NULL {
            break;
        }
    }
    out
}

/// The `DT_*` tags of `bytes`, in order.
fn dt_tags(bytes: &[u8]) -> Vec<i64> {
    dyn_entries(bytes).into_iter().map(|(t, _)| t).collect()
}

/// The sonames named by `DT_NEEDED`, read out of `.dynstr`.
fn needed_names(bytes: &[u8]) -> Vec<Vec<u8>> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let strtab = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynstr")
        .and_then(|s| obj.section_data(s).ok())
        .unwrap_or(&[]);
    let mut out = Vec::new();
    for (tag, val) in dyn_entries(bytes) {
        if tag != DT_NEEDED {
            continue;
        }
        let Ok(off) = usize::try_from(val) else {
            continue;
        };
        let Some(rest) = strtab.get(off..) else {
            continue;
        };
        let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
        out.push(rest[..end].to_vec());
    }
    out
}
