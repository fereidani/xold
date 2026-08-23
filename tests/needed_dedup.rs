//! The same library named twice is one dependency.
//!
//! A shared object reached twice -- named by path and again through `-l`, or
//! through two symlinks to the same file -- was recorded twice. The image then
//! carried two identical `DT_NEEDED` entries, and, worse, the second
//! occurrence burned a second dependency index. Copy relocations key on that
//! index to tell one dependency's addresses from another's, so the same
//! storage ended up described as belonging to two images.
//!
//! The soname is the name the loader resolves, so it is the identity. The
//! first occurrence wins, which keeps the answer a function of input order
//! rather than of which path spelled it. lld uniquifies DSOs the same way, by
//! soname as it loads them (`lld/ELF/InputFiles.cpp`).
//!
//! Gated on `clang` and the system crt objects; without them the tests print a
//! note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_dyn_exec};

mod common;

const LIB_SRC: &[u8] = b"int dup_fn(void) { return 7; }\n";

const MAIN_SRC: &[u8] = b"extern int dup_fn(void);\n\
    int main(void) { return dup_fn() - 7; }\n";

/// Naming the library twice yields one `DT_NEEDED`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_same_soname_is_recorded_once() {
    let Some(dir) = workdir("once") else {
        return;
    };
    let Some(prog) = link(&dir, true) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");
    let names = needed(&bytes);
    let dups = names.iter().filter(|n| *n == b"libdup.so").count();
    assert_eq!(
        dups, 1,
        "one library is one dependency however many paths reach it; the \
         DT_NEEDED list is {names:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// And the deduplicated image still runs, so the dependency that was kept is
/// the one the program needs.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_deduplicated_image_runs() {
    let Some(dir) = workdir("run") else {
        return;
    };
    let Some(prog) = link(&dir, true) else {
        return;
    };
    let code = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("must run")
        .code();
    assert_eq!(code, Some(0), "the program must find dup_fn and exit 0");
    let _ = fs::remove_dir_all(&dir);
}

/// The control: naming it once gives the same list, so deduplication removed a
/// duplicate rather than a dependency.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn naming_it_once_gives_the_same_list() {
    let Some(dir) = workdir("single") else {
        return;
    };
    let Some(once) = link(&dir, false) else {
        return;
    };
    let Some(twice) = link(&dir, true) else {
        return;
    };
    assert_eq!(
        needed(&fs::read(&once).expect("read")),
        needed(&fs::read(&twice).expect("read")),
        "naming the library twice must produce the dependency list naming it \
         once produces"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping needed-dedup {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_needed_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the library and the program, and links the program against the
/// library once or twice.
///
/// The second reference is a copy of the file under a different name, not a
/// symlink: it is the soname inside that has to decide identity, and a copy
/// makes that the only thing the two paths share.
fn link(dir: &Path, twice: bool) -> Option<PathBuf> {
    let clang = which("clang")?;
    let interp = interpreter()?;
    let lib_src = dir.join("dup.c");
    fs::write(&lib_src, LIB_SRC).ok()?;
    let lib = dir.join("libdup.so");
    let built = Command::new(&clang)
        .args(["-fPIC", "-shared", "-Wl,-soname,libdup.so"])
        .arg(&lib_src)
        .arg("-o")
        .arg(&lib)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping needed-dedup: clang cannot build the library");
        return None;
    }
    let alias = dir.join("libalias.so");
    fs::copy(&lib, &alias).ok()?;

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
    let mut paths = vec![start, prologue, obj, lib];
    if twice {
        paths.push(alias);
    }
    paths.push(libc);
    paths.push(epilogue);

    let prog = dir.join(if twice { "prog_twice" } else { "prog_once" });
    let res = link_dyn_exec(
        &paths,
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

/// `DT_NEEDED`'s value is 1; the value is an offset into `.dynstr`.
const DT_NEEDED: i64 = 1;

/// The `DT_NEEDED` names of a linked image, in table order.
fn needed(bytes: &[u8]) -> Vec<Vec<u8>> {
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
            if d_tag != DT_NEEDED {
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
