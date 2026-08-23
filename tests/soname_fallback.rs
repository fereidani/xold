//! A `DT_SONAME`-less library keeps the name it was given.
//!
//! When a shared object carries no `DT_SONAME`, the `DT_NEEDED` the image
//! records is how the object was named on the command line. The two spellings
//! part there. A library found by `-lfoo` takes its basename, because that is
//! what the loader's own search will find again. One named directly keeps the
//! path as written.
//!
//! xold took the basename either way, so an explicitly named
//! `/opt/vendor/libpriv.so` was recorded as `libpriv.so` -- a name the loader
//! has no reason to find. The link succeeded and the program did not start,
//! with a runtime error naming a library that was right there on disk.
//!
//! lld splits the same way: `withLOption ? path::filename(path) : path`
//! (`lld/ELF/Driver.cpp`).
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
    input::Input,
    linker::{Link, link_image},
};

mod common;

const LIB_SRC: &[u8] = b"int vendor_fn(void) { return 3; }\n";

const MAIN_SRC: &[u8] = b"extern int vendor_fn(void);\n\
    int main(void) { return vendor_fn() - 3; }\n";

/// A library named by path keeps the path, so the loader can find it again.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_explicitly_named_library_keeps_its_path() {
    let Some(dir) = workdir("path") else {
        return;
    };
    let Some((prog, lib)) = link(&dir, false) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");
    let names = needed(&bytes);
    let want = lib.to_str().expect("path is utf-8").as_bytes().to_vec();
    assert!(
        names.contains(&want),
        "the DT_NEEDED must be the path as written, or the loader has nothing \
         to search for; got {names:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The behaviour behind the name: the program starts without being told where
/// to look.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_program_starts_without_a_search_path() {
    let Some(dir) = workdir("run") else {
        return;
    };
    let Some((prog, _)) = link(&dir, false) else {
        return;
    };
    let out = Command::new(&prog).output().expect("must be runnable");
    assert!(
        out.status.success(),
        "the image must start: {}",
        String::from_utf8_lossy(&out.stderr).trim()
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A library found by `-l` takes its basename, which is what the loader's
/// search path is for. The rule is about how the library was named, not about
/// always keeping the longest spelling.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_library_found_by_search_takes_its_basename() {
    let Some(dir) = workdir("basename") else {
        return;
    };
    let Some((prog, _)) = link(&dir, true) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");
    let names = needed(&bytes);
    assert!(
        names.iter().any(|n| n == b"libpriv.so"),
        "a -l library is recorded by basename; got {names:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping soname-fallback {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_soname_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds a `DT_SONAME`-less library in a subdirectory and links a program
/// against it, either naming it directly or as a `-l` search result.
fn link(dir: &Path, as_search: bool) -> Option<(PathBuf, PathBuf)> {
    let clang = which("clang")?;
    let interp = interpreter()?;
    let vendor = dir.join("vendor");
    fs::create_dir_all(&vendor).ok()?;
    let lib_src = dir.join("priv.c");
    fs::write(&lib_src, LIB_SRC).ok()?;
    let lib = vendor.join("libpriv.so");
    // No `-Wl,-soname`: the whole point is a library that declares no name.
    let built = Command::new(&clang)
        .args(["-fPIC", "-shared"])
        .arg(&lib_src)
        .arg("-o")
        .arg(&lib)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping soname-fallback: clang cannot build the library");
        return None;
    }
    assert!(
        soname(&fs::read(&lib).ok()?).is_none(),
        "the fixture library must declare no DT_SONAME"
    );

    let main_src = dir.join("main.c");
    let obj = dir.join("main.o");
    fs::write(&main_src, MAIN_SRC).ok()?;
    Command::new(&clang)
        .args(["--target=x86_64-linux-gnu", "-fPIE", "-c"])
        .arg(&main_src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success()
        .then_some(())?;

    let start = crt_file("Scrt1.o")?;
    let prologue = crt_file("crti.o")?;
    let epilogue = crt_file("crtn.o")?;
    let libc = libc_so()?;
    let prog = dir.join(if as_search { "prog_l" } else { "prog_path" });
    let lib_input = if as_search {
        Input::Library(&lib)
    } else {
        Input::Path(&lib)
    };
    let files = [
        Input::Path(&start),
        Input::Path(&prologue),
        Input::Path(&obj),
        lib_input,
        Input::Path(&libc),
        Input::Path(&epilogue),
    ];
    let res = link_image(&Link {
        relax: false,
        ..Link::dyn_exec(&files, &prog, &interp)
    });
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    Some((prog, lib))
}

// --- readers ---------------------------------------------------------------

const DT_NEEDED: i64 = 1;
const DT_SONAME: i64 = 14;

/// The `DT_NEEDED` names of a linked image, in table order.
fn needed(bytes: &[u8]) -> Vec<Vec<u8>> {
    dyn_strings(bytes, DT_NEEDED)
}

/// The image's own `DT_SONAME`, if it declares one.
fn soname(bytes: &[u8]) -> Option<Vec<u8>> {
    dyn_strings(bytes, DT_SONAME).into_iter().next()
}

/// Every `.dynstr` string named by a `.dynamic` entry with tag `want`.
fn dyn_strings(bytes: &[u8], want: i64) -> Vec<Vec<u8>> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(dynamic) = section(&obj, b".dynamic") else {
        return Vec::new();
    };
    let Some(dynstr) = section(&obj, b".dynstr") else {
        return Vec::new();
    };
    dynamic
        .as_chunks::<16>()
        .0
        .iter()
        .filter_map(|c| {
            let d_tag = i64::from_le_bytes(<[u8; 8]>::try_from(&c[0..8]).ok()?);
            if d_tag != want {
                return None;
            }
            let off = usize::try_from(u64::from_le_bytes(
                <[u8; 8]>::try_from(&c[8..16]).ok()?,
            ))
            .ok()?;
            let tail = dynstr.get(off..)?;
            let end = tail.iter().position(|b| *b == 0)?;
            Some(tail[..end].to_vec())
        })
        .collect()
}

/// The bytes of a named section.
fn section(obj: &ObjectFile<'_>, name: &[u8]) -> Option<Vec<u8>> {
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == name)?;
    obj.section_data(shdr).ok().map(<[u8]>::to_vec)
}
