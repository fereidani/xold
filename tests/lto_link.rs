//! End-to-end LTO links, driven through the binary.
//!
//! Codegen shuts LLVM down inside the plugin, so a process gets one link.
//! Driving `xold` as a subprocess gives each case its own process, which is
//! also how a build system uses it -- and it means these check the whole
//! path, from the command line to a program that runs.
//!
//! Every case is gated: on `clang`, on a plugin being installed, and for the
//! Windows case on `wine`. A missing one prints a note and returns, because
//! it is a property of the machine rather than of this linker.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, which, xold_bin};
use xold::lto::plugin;

mod common;

/// `helper` is internalised, `main` is kept, and the program runs.
#[test]
#[cfg_attr(miri, ignore = "needs a host plugin and toolchain")]
fn an_lto_program_links_and_runs() {
    let Some(dir) = workdir("run") else {
        return;
    };
    let Some(obj) = bitcode(
        &dir,
        "t",
        b"int helper(void){return 7;}\nint main(void){return helper();}\n",
        "x86_64-linux-gnu",
    ) else {
        return;
    };
    let (Some(start), Some(open), Some(close)) =
        (crt_file("crt1.o"), crt_file("crti.o"), crt_file("crtn.o"))
    else {
        eprintln!("skipping lto run: no crt objects on this host");
        return;
    };
    let out = dir.join("prog");
    let linked = Command::new(xold_bin())
        .args([&start, &open, &obj])
        .arg("-lc")
        .arg(&close)
        .arg("-o")
        .arg(&out)
        .arg("--dynamic-exec")
        .output()
        .expect("run xold");
    assert!(
        linked.status.success(),
        "the LTO link must succeed: {}",
        String::from_utf8_lossy(&linked.stderr)
    );
    let run = Command::new(&out).status().expect("the image must run");
    assert_eq!(run.code(), Some(7), "helper() returns 7 through main()");
    let _ = fs::remove_dir_all(&dir);
}

/// A native object linked beside bitcode reaches the compiled code.
#[test]
#[cfg_attr(miri, ignore = "needs a host plugin and toolchain")]
fn bitcode_and_native_objects_link_together() {
    let Some(dir) = workdir("mixed") else {
        return;
    };
    let Some(lib) = bitcode(
        &dir,
        "lib",
        b"int helper(void){return 7;}\n",
        "x86_64-linux-gnu",
    ) else {
        return;
    };
    let Some(user) = native(
        &dir,
        "user",
        b"int helper(void);\nint main(void){return helper();}\n",
    ) else {
        return;
    };
    let (Some(start), Some(open), Some(close)) =
        (crt_file("crt1.o"), crt_file("crti.o"), crt_file("crtn.o"))
    else {
        eprintln!("skipping lto mixed: no crt objects on this host");
        return;
    };
    let out = dir.join("prog");
    let linked = Command::new(xold_bin())
        .args([&start, &open, &lib, &user])
        .arg("-lc")
        .arg(&close)
        .arg("-o")
        .arg(&out)
        .arg("--dynamic-exec")
        .output()
        .expect("run xold");
    assert!(
        linked.status.success(),
        "a mixed link must succeed: {}",
        String::from_utf8_lossy(&linked.stderr)
    );
    let run = Command::new(&out).status().expect("the image must run");
    assert_eq!(
        run.code(),
        Some(7),
        "the native caller reaches the bitcode definition"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Windows-targeted bitcode becomes a PE that runs under `wine`.
///
/// The entry is not `main`: a mingw `main` calls `__main` from a C runtime
/// this link does not pull in, and the point here is the LTO path rather than
/// the runtime.
#[test]
#[cfg_attr(miri, ignore = "needs a host plugin and toolchain")]
fn a_windows_lto_program_links_and_runs_under_wine() {
    let Some(dir) = workdir("wine") else {
        return;
    };
    let Some(wine) = which("wine").or_else(|| which("wine64")) else {
        eprintln!("skipping windows lto run: no wine on this host");
        return;
    };
    let Some(obj) = bitcode(
        &dir,
        "e",
        b"int helper(void){return 7;}\nint entrypoint(void){return helper();}\n",
        "x86_64-pc-windows-gnu",
    ) else {
        return;
    };
    let out = dir.join("e.exe");
    let linked = Command::new(xold_bin())
        .arg(&obj)
        .arg("-o")
        .arg(&out)
        .args(["--entry", "entrypoint"])
        .output()
        .expect("run xold");
    assert!(
        linked.status.success(),
        "the windows LTO link must succeed: {}",
        String::from_utf8_lossy(&linked.stderr)
    );
    let image = fs::read(&out).expect("the image is readable");
    assert_eq!(&image[..2], b"MZ", "LTO on a windows target produces a PE");
    let run = Command::new(wine)
        .arg(&out)
        .env("WINEDEBUG", "-all")
        .status()
        .expect("wine runs the image");
    assert_eq!(run.code(), Some(7), "the PE returns helper()'s value");
    let _ = fs::remove_dir_all(&dir);
}

/// A bitcode member is pulled out of an archive, compiled, and linked.
///
/// The plugin cannot see inside an archive, so the linker has to find the
/// member itself: the same rule as for a native one -- it defines a name
/// something references and nothing else does. `llvm-ar` is required because
/// GNU `ar` writes no index entries for bitcode without its own plugin, and
/// an archive with no index cannot be searched by name.
#[test]
#[cfg_attr(miri, ignore = "needs a host plugin and toolchain")]
fn a_bitcode_archive_member_is_extracted_and_linked() {
    let Some(dir) = workdir("archive") else {
        return;
    };
    let Some(ar) = which("llvm-ar") else {
        eprintln!("skipping lto archive: no llvm-ar on this host");
        return;
    };
    let Some(lib) = bitcode(
        &dir,
        "lib",
        b"int helper(void){return 7;}\n",
        "x86_64-linux-gnu",
    ) else {
        return;
    };
    let Some(user) = bitcode(
        &dir,
        "user",
        b"int helper(void);\nint main(void){return helper();}\n",
        "x86_64-linux-gnu",
    ) else {
        return;
    };
    let archive = dir.join("libhelper.a");
    let made = Command::new(ar)
        .arg("rcs")
        .arg(&archive)
        .arg(&lib)
        .status()
        .is_ok_and(|s| s.success());
    assert!(made, "llvm-ar builds an archive of bitcode");

    let (Some(start), Some(open), Some(close)) =
        (crt_file("crt1.o"), crt_file("crti.o"), crt_file("crtn.o"))
    else {
        eprintln!("skipping lto archive: no crt objects on this host");
        return;
    };
    let out = dir.join("prog");
    let linked = Command::new(xold_bin())
        .args([&start, &open, &user, &archive])
        .arg("-lc")
        .arg(&close)
        .arg("-o")
        .arg(&out)
        .arg("--dynamic-exec")
        .output()
        .expect("run xold");
    assert!(
        linked.status.success(),
        "the archive member must be found and compiled: {}",
        String::from_utf8_lossy(&linked.stderr)
    );
    let run = Command::new(&out).status().expect("the image must run");
    assert_eq!(run.code(), Some(7), "the extracted member's code runs");
    let _ = fs::remove_dir_all(&dir);
}

/// Bitcode inside a `--start-lib` group links, and an unused member does not
/// reach the image.
///
/// A group's index is built from what the linker can read, which is not
/// bitcode, so the LTO pass takes every bitcode member rather than searching
/// for one. The second half of this is what makes that safe: a member
/// nothing references is dropped by LTO, which is the same image a lazy
/// search would have produced.
#[test]
#[cfg_attr(miri, ignore = "needs a host plugin and toolchain")]
fn a_start_lib_group_of_bitcode_links_without_its_dead_members() {
    let Some(dir) = workdir("group") else {
        return;
    };
    let Some(lib) = bitcode(
        &dir,
        "lib",
        b"int helper(void){return 7;}\n",
        "x86_64-linux-gnu",
    ) else {
        return;
    };
    let Some(dead) = bitcode(
        &dir,
        "dead",
        b"int unused_thing(void){return 1;}\n",
        "x86_64-linux-gnu",
    ) else {
        return;
    };
    let Some(user) = bitcode(
        &dir,
        "user",
        b"int helper(void);\nint main(void){return helper();}\n",
        "x86_64-linux-gnu",
    ) else {
        return;
    };
    let (Some(start), Some(open), Some(close)) =
        (crt_file("crt1.o"), crt_file("crti.o"), crt_file("crtn.o"))
    else {
        eprintln!("skipping lto group: no crt objects on this host");
        return;
    };
    let out = dir.join("prog");
    let linked = Command::new(xold_bin())
        .args([&start, &open, &user])
        .arg("--start-lib")
        .args([&lib, &dead])
        .arg("--end-lib")
        .arg("-lc")
        .arg(&close)
        .arg("-o")
        .arg(&out)
        .arg("--dynamic-exec")
        .output()
        .expect("run xold");
    assert!(
        linked.status.success(),
        "a group of bitcode must link: {}",
        String::from_utf8_lossy(&linked.stderr)
    );
    let run = Command::new(&out).status().expect("the image must run");
    assert_eq!(run.code(), Some(7), "the group member's code runs");
    let image = fs::read(&out).expect("the image is readable");
    assert!(
        !contains(&image, b"unused_thing"),
        "a member nothing references must not reach the image"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Darwin-targeted bitcode becomes a Mach-O image.
///
/// It cannot be run here, so this checks the shape: LTO routed the compiled
/// objects to the Mach-O writer rather than to the ELF one, which is decided
/// by the objects the plugin returned and not by anything the input scan saw.
#[test]
#[cfg_attr(miri, ignore = "needs a host plugin and toolchain")]
fn darwin_bitcode_produces_a_macho_image() {
    let Some(dir) = workdir("macho") else {
        return;
    };
    let Some(obj) = bitcode(
        &dir,
        "m",
        b"int helper(void){return 7;}\nint main(void){return helper();}\n",
        "x86_64-apple-darwin",
    ) else {
        return;
    };
    let out = dir.join("m.out");
    let linked = Command::new(xold_bin())
        .arg(&obj)
        .arg("-o")
        .arg(&out)
        .args(["-arch", "x86_64"])
        .output()
        .expect("run xold");
    assert!(
        linked.status.success(),
        "a darwin LTO link must succeed: {}",
        String::from_utf8_lossy(&linked.stderr)
    );
    let image = fs::read(&out).expect("the image is readable");
    assert_eq!(
        image.get(..4),
        Some([0xcf, 0xfa, 0xed, 0xfe].as_slice()),
        "LTO on a darwin target produces a 64-bit Mach-O"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// A working directory, or `None` when the host cannot run these at all.
fn workdir(tag: &str) -> Option<PathBuf> {
    which("clang")?;
    if plugin::resolve(None).is_none() {
        eprintln!("skipping lto {tag}: no LLVMgold.so on this host");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_ltolink_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles `source` to bitcode for `target`.
fn bitcode(
    dir: &Path,
    stem: &str,
    source: &[u8],
    target: &str,
) -> Option<PathBuf> {
    let obj = build(
        dir,
        stem,
        source,
        &[&format!("--target={target}"), "-flto", "-c"],
    )?;
    // A clang configured for fat objects gives a native file, which is a
    // different shape than the one under test.
    if !fs::read(&obj).is_ok_and(|b| b.starts_with(b"BC\xc0\xde")) {
        eprintln!("skipping: this clang emits fat LTO objects");
        return None;
    }
    Some(obj)
}

/// Compiles `source` to an ordinary host object.
fn native(dir: &Path, stem: &str, source: &[u8]) -> Option<PathBuf> {
    build(dir, stem, source, &["-fno-lto", "-c"])
}

/// Whether `needle` appears anywhere in `haystack`.
fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

fn build(
    dir: &Path,
    stem: &str,
    source: &[u8],
    flags: &[&str],
) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src, source).ok()?;
    let built = Command::new(clang)
        .args(flags)
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping: clang cannot build {stem} with {flags:?}");
        return None;
    }
    Some(obj)
}
