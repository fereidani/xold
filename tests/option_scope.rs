//! An option this linker accepts is an option it acts on.
//!
//! xold refuses an option it does not implement, on the stated ground that
//! accepting one and ignoring it produces an image other than the one the
//! command line described. Several accepted options were then dropped anyway,
//! depending on the output kind.
//!
//! `-shared --entry f` parsed and threw the entry away, so the shared object
//! got `e_entry = 0`. `-soname` outside `-shared` was dropped. On the COFF and
//! Mach-O paths `--gc-sections`, `--icf`, `--relax` and `-soname` were all
//! dropped, and `-shared` too on Mach-O.
//!
//! Both halves are fixed the same way round: what the ELF path can honour, it
//! honours in every mode, and what another format's path cannot honour is
//! refused by name rather than swallowed. lld honours `--entry` and `-soname`
//! in every output type.
//!
//! These tests drive the built binary, since the scoping is a command-line
//! property.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{which, xold_bin};

mod common;

/// `_start` is here because an executable with no entry point no longer
/// links: the `--dynamic-exec` case below needs one to reach the assertion it
/// is actually about.
const SRC: &[u8] = b"int f(void) { return 1; }\n\
    void _start(void) { }\n";

/// `--entry` is honoured for a shared object.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn entry_is_honoured_for_a_shared_object() {
    let Some(dir) = workdir("entry") else {
        return;
    };
    let Some(obj) = compile(&dir) else {
        return;
    };
    let out = dir.join("with_entry.so");
    let (ok, err) = xold(&["-shared", "--entry", "f"], &obj, &out);
    assert!(ok, "the link must succeed: {err}");
    let bytes = fs::read(&out).expect("read output");
    let want = symbol_value(&bytes, b"f").expect("f is defined");
    assert_ne!(want, 0, "the fixture must place f somewhere");
    assert_eq!(
        e_entry(&bytes),
        want,
        "--entry names the entry point whatever the output kind; dropping it \
         leaves e_entry at zero"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// And without `--entry` a shared object still has none, so the default is not
/// applied to something that has no entry point.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_shared_object_without_entry_keeps_none() {
    let Some(dir) = workdir("noentry") else {
        return;
    };
    let Some(obj) = compile(&dir) else {
        return;
    };
    let out = dir.join("plain.so");
    let (ok, err) = xold(&["-shared"], &obj, &out);
    assert!(ok, "the link must succeed: {err}");
    let bytes = fs::read(&out).expect("read output");
    assert_eq!(
        e_entry(&bytes),
        0,
        "a shared object has no entry point unless one is asked for"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `-soname` is honoured for an executable rather than dropped.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn soname_is_honoured_outside_shared() {
    let Some(dir) = workdir("soname") else {
        return;
    };
    let Some(obj) = compile(&dir) else {
        return;
    };
    let out = dir.join("prog");
    let (ok, err) =
        xold(&["--dynamic-exec", "-soname", "named.so"], &obj, &out);
    assert!(ok, "the link must succeed: {err}");
    let bytes = fs::read(&out).expect("read output");
    assert_eq!(
        dyn_string(&bytes, DT_SONAME).as_deref(),
        Some(&b"named.so"[..]),
        "an accepted -soname must reach the image"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// An option the COFF path cannot honour is refused by name.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_elf_only_option_is_refused_on_the_coff_path() {
    let Some(dir) = workdir("coff") else {
        return;
    };
    let Some(obj) = compile_coff(&dir) else {
        return;
    };
    let out = dir.join("prog.exe");
    let (ok, err) = xold(&["--gc-sections"], &obj, &out);
    assert!(
        !ok,
        "an option the COFF path drops must end the link, not be swallowed"
    );
    assert!(
        err.contains("--gc-sections") && err.contains("COFF"),
        "the refusal must name the option and the format, got {err:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping option-scope {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_optscope_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture as an ELF object.
fn compile(dir: &Path) -> Option<PathBuf> {
    build(dir, "x86_64-linux-gnu", "elf.o")
}

/// Compiles the fixture as a COFF object, or `None` when this clang cannot
/// target Windows.
fn compile_coff(dir: &Path) -> Option<PathBuf> {
    let obj = build(dir, "x86_64-pc-windows-msvc", "coff.o");
    if obj.is_none() {
        eprintln!("skipping option-scope: clang cannot target COFF");
    }
    obj
}

fn build(dir: &Path, target: &str, name: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("f.c");
    let obj = dir.join(name);
    fs::write(&src, SRC).ok()?;
    // No `-fPIC`: clang rejects it for the Windows target, and the ELF
    // fixture does not need it either -- nothing here is linked shared
    // except through a `-shared` this test drives explicitly.
    Command::new(clang)
        .args([&format!("--target={target}"), "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success()
        .then_some(obj)
}

/// Runs the built linker, returning success and what it printed.
fn xold(flags: &[&str], obj: &Path, out: &Path) -> (bool, String) {
    let res = Command::new(xold_bin())
        .args(flags)
        .arg("-o")
        .arg(out)
        .arg(obj)
        .output()
        .expect("xold must run");
    (
        res.status.success(),
        String::from_utf8_lossy(&res.stderr).trim().to_string(),
    )
}

// --- readers ---------------------------------------------------------------

const DT_SONAME: i64 = 14;

/// The ELF header's `e_entry`.
fn e_entry(bytes: &[u8]) -> u64 {
    bytes
        .get(24..32)
        .and_then(|c| <[u8; 8]>::try_from(c).ok())
        .map_or(0, u64::from_le_bytes)
}

/// The `st_value` of a symbol in the output's `.symtab`.
fn symbol_value(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = xold::elf::ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab
        .iter()
        .find(|s| symtab.name(s) == name)
        .map(|s| s.st_value.get())
}

/// The `.dynstr` string named by the first `.dynamic` entry with tag `want`.
fn dyn_string(bytes: &[u8], want: i64) -> Option<Vec<u8>> {
    let obj = xold::elf::ObjectFile::parse(bytes).ok()?;
    let dynamic = section(&obj, b".dynamic")?;
    let dynstr = section(&obj, b".dynstr")?;
    dynamic.as_chunks::<16>().0.iter().find_map(|c| {
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
}

/// The bytes of a named section.
fn section(obj: &xold::elf::ObjectFile<'_>, name: &[u8]) -> Option<Vec<u8>> {
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == name)?;
    obj.section_data(shdr).ok().map(<[u8]>::to_vec)
}
