//! The bounds the linker defines itself, and the one property that makes them
//! usable in an image the loader places: they name a section.
//!
//! `_end`, `__start_NAME` and the rest are recorded as absolute definitions --
//! the layout owns their values, no input section does -- but they name places
//! in the image, and every one of those places moves with the load base. The
//! only thing that tells a loader so is `st_shndx`: glibc's `SYMBOL_ADDRESS`
//! skips the load base for an `SHN_ABS` row, so a bound published as absolute
//! is read back at its link-time value in a module mapped anywhere else.
//!
//! The check that matters is the runtime one: a shared object xold links,
//! loaded by a program the system toolchain built, has to report bounds that
//! land inside its own mapping. The structural tests beside it pin the two
//! halves that make that work -- the section index, and the visibility that
//! keeps a bound the C runtime uses out of the dynamic ABI.
//!
//! Gated on `clang` and a host `libc.so.6`; absent either, the tests print a
//! note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{libc_so, which};
use xold::{
    elf::{ObjectFile, constants::SHN_ABS},
    icf::IcfMode,
    linker::{link_shared, link_to},
};

mod common;

/// A library that reports the bounds of a section it owns, plus the address of
/// the one entry in that section, so the caller can compare the two.
///
/// The probe is `__start_`/`__stop_` rather than `_end`, because `_end` is a
/// default-visibility definition and therefore preemptible: a program linked
/// against this library exports its own `_end` and the loader binds the
/// library's reference to that one, which is what every linker does and says
/// nothing about how the bound was published. `__start_`/`__stop_` are
/// protected, so the library always reads its own.
///
/// `_end` is referenced all the same, so the structural test beside the runtime
/// one has a default-visibility bound to inspect in `.dynsym`.
const LIB: &str = r#"
extern char __start_myreg[], __stop_myreg[];
extern char _end[];

__attribute__((section("myreg"), used)) static long entry = 0x5eed;

char *get_reg_start(void) { return __start_myreg; }
char *get_reg_stop(void) { return __stop_myreg; }
char *get_end(void) { return _end; }
long *get_entry(void) { return &entry; }
"#;

/// Uses the library through its own accessors and reports what it found.
///
/// Every check compares two addresses the *library* produced, so the program
/// never needs to know where the loader put it. A bound that kept its
/// link-time value fails the first comparison outright: `entry` is a mapped
/// address and the stale bound is a small offset from zero.
const USER: &str = r#"
#include <stdio.h>
#include <stdlib.h>

char *get_reg_start(void);
char *get_reg_stop(void);
long *get_entry(void);

int main(void) {
    char *start = get_reg_start();
    char *stop = get_reg_stop();
    long *entry = get_entry();

    /* The run holds exactly the one entry, at exactly its address. */
    if (start != (char *)entry) return 1;
    if (stop - start != (long)sizeof(long)) return 2;
    if (*entry != 0x5eed) return 3;
    printf("ok\n");
    return 0;
}
"#;

fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_defsym_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Compiles `src` with `clang`, returning the object path.
fn compile(
    dir: &Path,
    name: &str,
    src: &str,
    extra: &[&str],
) -> Option<PathBuf> {
    let clang = which("clang")?;
    let file = dir.join(format!("{name}.c"));
    let obj = dir.join(format!("{name}.o"));
    fs::write(&file, src).ok()?;
    Command::new(clang)
        .args(["-c", "-O1"])
        .args(extra)
        .arg("-o")
        .arg(&obj)
        .arg(&file)
        .status()
        .ok()?
        .success()
        .then_some(obj)
}

/// Links `LIB` into a shared object with xold, returning its path.
fn build_library(dir: &Path) -> Option<PathBuf> {
    let obj = compile(dir, "lib", LIB, &["-fPIC"])?;
    let libc = libc_so()?;
    let so = dir.join("libbounds.so");
    link_shared(
        &[obj, libc],
        &so,
        Some(b"libbounds.so"),
        false,
        IcfMode::None,
        false,
    )
    .expect("a shared object referring to linker bounds must link");
    Some(so)
}

/// The `(st_shndx, st_other)` of a name in one of `image`'s symbol tables, or
/// `None` when the table has no such row. `dynamic` selects `.dynsym`.
fn row(image: &Path, want: &[u8], dynamic: bool) -> Option<(u16, u8)> {
    let bytes = fs::read(image).ok()?;
    let obj = ObjectFile::parse(&bytes).ok()?;
    let table = if dynamic {
        obj.dynamic_symbols().ok()?
    } else {
        obj.symbol_table().ok()?
    }?;
    table
        .syms
        .iter()
        .find(|s| table.name(s) == want)
        .map(|s| (s.st_shndx.get(), s.st_other))
}

/// The whole point: a program built by the system toolchain reads the bounds
/// out of an xold-linked shared object and finds them where the loader put the
/// module, not where the link left them.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_shared_object_reports_bounds_that_moved_with_it() {
    let dir = workdir("run");
    let Some(so) = build_library(&dir) else {
        eprintln!("skipping bounds test: clang or libc missing");
        return;
    };
    let Some(clang) = which("clang") else {
        return;
    };
    let src = dir.join("user.c");
    let user = dir.join("user");
    fs::write(&src, USER).expect("write user source");
    let built = Command::new(clang)
        .arg(&src)
        .arg("-o")
        .arg(&user)
        .arg("-l:libbounds.so")
        .arg(format!("-L{}", dir.display()))
        .arg(format!("-Wl,-rpath,{}", dir.display()))
        .status()
        .expect("clang must run")
        .success();
    assert!(built, "the user program must link against the library");
    let code = Command::new(&user)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("the user program must run")
        .code()
        .expect("the user program must exit normally");
    assert_eq!(
        code, 0,
        "every bound must land inside the module the loader mapped"
    );
    let _ = so;
}

/// A bound that reaches `.dynsym` names a section there too. `SHN_ABS` is what
/// tells the loader to leave the value alone, which is the bug this pins.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_exported_bound_is_not_absolute() {
    let dir = workdir("dynsym");
    let Some(so) = build_library(&dir) else {
        eprintln!("skipping bounds test: clang or libc missing");
        return;
    };
    for name in [&b"_end"[..], b"__start_myreg", b"__stop_myreg"] {
        let (shndx, _) = row(&so, name, true).unwrap_or_else(|| {
            panic!("{} is exported", String::from_utf8_lossy(name))
        });
        assert_ne!(
            shndx,
            SHN_ABS,
            "{} must name a section so the loader relocates it",
            String::from_utf8_lossy(name)
        );
    }
}

/// The bounds a C runtime uses to walk its own image are private to it, so
/// they carry hidden visibility and stay out of `.dynsym` -- which is also
/// what keeps them from being preempted. lld gives the same set the same
/// visibility.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_runtime_bounds_are_private_to_the_image() {
    const HIDDEN: u8 = 2;
    let dir = workdir("hidden");
    let Some(obj) = compile(
        &dir,
        "arrays",
        r"
extern char __init_array_start[], __init_array_end[];
extern char __ehdr_start[];
long span(void) { return __init_array_end - __init_array_start; }
char *hdr(void) { return __ehdr_start; }
",
        &["-fPIC"],
    ) else {
        eprintln!("skipping bounds test: clang missing");
        return;
    };
    let Some(libc) = libc_so() else {
        return;
    };
    let so = dir.join("libarrays.so");
    link_shared(
        &[obj, libc],
        &so,
        Some(b"libarrays.so"),
        false,
        IcfMode::None,
        false,
    )
    .expect("the library must link");

    for name in [
        &b"__init_array_start"[..],
        b"__init_array_end",
        b"__ehdr_start",
    ] {
        let spelled = String::from_utf8_lossy(name).into_owned();
        assert!(
            row(&so, name, true).is_none(),
            "{spelled} describes this image's own layout and must not be exported"
        );
        let (shndx, other) = row(&so, name, false)
            .unwrap_or_else(|| panic!("{spelled} is defined in .symtab"));
        assert_eq!(other, HIDDEN, "{spelled} is hidden");
        assert_ne!(shndx, SHN_ABS, "{spelled} names a section");
    }
}

/// `_edata` is the end of initialised data, which exists whether or not the
/// image has any `.bss`. Reading it off `.bss`'s start left it at address zero
/// for an image with none.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn data_end_is_defined_without_a_bss() {
    let dir = workdir("edata");
    let Some(obj) = compile(
        &dir,
        "nobss",
        r"
extern char _edata[], __bss_start[];
int payload = 7;
void _start(void) { }
char *edata(void) { return _edata; }
char *bss(void) { return __bss_start; }
",
        &["-fno-pie"],
    ) else {
        eprintln!("skipping bounds test: clang missing");
        return;
    };
    let out = dir.join("nobss");
    link_to(&[obj], &out, b"_start", false, IcfMode::None, false)
        .expect("an image with no .bss must link");

    let bytes = fs::read(&out).expect("read output");
    let image = ObjectFile::parse(&bytes).expect("parse output");
    assert!(
        image
            .sections()
            .iter()
            .all(|s| image.section_name(s) != b".bss"),
        "the fixture must produce no .bss for this to be the case under test"
    );
    for name in [&b"_edata"[..], b"__bss_start"] {
        let spelled = String::from_utf8_lossy(name).into_owned();
        let (shndx, _) = row(&out, name, false)
            .unwrap_or_else(|| panic!("{spelled} is defined"));
        assert_ne!(shndx, SHN_ABS, "{spelled} names a section");
        assert_ne!(shndx, 0, "{spelled} names a section that exists");
    }
}
