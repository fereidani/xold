//! `--version-script`: the list of what a library promises.
//!
//! A `cdylib`, a `dylib` and a proc macro are all linked with one of these:
//! `rustc` writes a script naming the items it exports and hides everything
//! else with `local: *;`. A linker that cannot read it cannot build any of
//! them, and one that reads it and applies it to the wrong symbols produces a
//! library that fails to load.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{which, xold_bin};
use xold::elf::ObjectFile;

mod common;

/// Two functions, one of which the script promises.
const SRC: &[u8] = b"int pub_one(void) { return 1; }\n\
    int priv_one(void) { return 2; }\n";

/// A library that calls a name nothing in the link defines, so the reference
/// stays an import. A `local: *;` script must not touch it: hiding an import
/// makes it unresolvable rather than private.
const IMPORTING_SRC: &[u8] = b"extern int elsewhere(void);\n\
    int calls_out(void) { return elsewhere(); }\n";

/// The script every generated one looks like.
const SCRIPT: &[u8] = b"{\n global:\n  pub_one;\n local:\n  *;\n};\n";

/// The promised name is exported and everything else is not.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_script_decides_what_is_exported() {
    let Some(dir) = workdir("exports") else {
        return;
    };
    let Some(obj) = compile(&dir, "vs.c", SRC) else {
        return;
    };
    let script = dir.join("vs.map");
    fs::write(&script, SCRIPT).expect("write the version script");

    let out = dir.join("libvs.so");
    let (ok, err) = xold(&[
        "-shared".into(),
        format!("--version-script={}", script.display()),
        "-o".into(),
        path(&out),
        path(&obj),
    ]);
    assert!(ok, "the link must succeed: {err}");
    let exported = dynamic_symbols(&fs::read(&out).unwrap_or_default());
    assert!(
        exported.iter().any(|n| n == b"pub_one"),
        "the promised name must be exported"
    );
    assert!(
        !exported.iter().any(|n| n == b"priv_one"),
        "and `local: *;` must hide the rest"
    );

    // Without the script both are exported, which is what makes the test
    // above about the script rather than about visibility defaults.
    let bare = dir.join("libbare.so");
    let (ok, err) =
        xold(&["-shared".into(), "-o".into(), path(&bare), path(&obj)]);
    assert!(ok, "the unscripted link must succeed: {err}");
    let exported = dynamic_symbols(&fs::read(&bare).unwrap_or_default());
    assert!(
        exported.iter().any(|n| n == b"priv_one"),
        "without a script every global is exported"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `local: *;` hides definitions and leaves imports alone.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn hiding_does_not_reach_imports() {
    let Some(dir) = workdir("imports") else {
        return;
    };
    let Some(obj) = compile(&dir, "imp.c", IMPORTING_SRC) else {
        return;
    };
    let script = dir.join("hide-all.map");
    fs::write(&script, b"{\n local:\n  *;\n};\n").expect("write the script");

    let out = dir.join("libimp.so");
    let (ok, err) = xold(&[
        "-shared".into(),
        format!("--version-script={}", script.display()),
        "-o".into(),
        path(&out),
        path(&obj),
    ]);
    assert!(ok, "the link must succeed: {err}");
    let symbols = dynamic_symbols(&fs::read(&out).unwrap_or_default());
    assert!(
        symbols.iter().any(|n| n == b"elsewhere"),
        "the import must stay in .dynsym for the loader to resolve"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `--no-undefined-version` refuses a script naming something the link does
/// not define, and a named version node is refused with its reason.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_script_that_cannot_be_honoured_is_refused() {
    let Some(dir) = workdir("refused") else {
        return;
    };
    let Some(obj) = compile(&dir, "vs.c", SRC) else {
        return;
    };
    let out = dir.join("libvs.so");

    let stale = dir.join("stale.map");
    fs::write(&stale, b"{\n global:\n  gone_away;\n local:\n  *;\n};\n")
        .expect("write the script");
    let (ok, err) = xold(&[
        "-shared".into(),
        format!("--version-script={}", stale.display()),
        "--no-undefined-version".into(),
        "-o".into(),
        path(&out),
        path(&obj),
    ]);
    assert!(!ok, "a script naming an absent symbol must be refused");
    assert!(
        err.contains("gone_away"),
        "and the diagnostic must name it: {err}"
    );

    let versioned = dir.join("versioned.map");
    fs::write(&versioned, b"LIB_1.0 {\n global:\n  pub_one;\n};\n")
        .expect("write the script");
    let (ok, err) = xold(&[
        "-shared".into(),
        format!("--version-script={}", versioned.display()),
        "-o".into(),
        path(&out),
        path(&obj),
    ]);
    assert!(
        !ok,
        "a named version node must be refused, not half-applied"
    );
    assert!(
        err.contains("version_d") || err.contains("LIB_1.0"),
        "and must say what is missing: {err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Compiles one position-independent source file with the host `clang`.
fn compile(dir: &Path, name: &str, src: &[u8]) -> Option<PathBuf> {
    let clang = which("clang")?;
    let file = dir.join(name);
    fs::write(&file, src).ok()?;
    let obj = dir.join(format!("{name}.o"));
    let ok = Command::new(clang)
        .args(["-c", "-fPIC"])
        .arg(&file)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    ok.then_some(obj)
}

/// Runs xold, returning whether it succeeded and what it printed.
fn xold(args: &[String]) -> (bool, String) {
    let out = Command::new(xold_bin())
        .arg("--no-fork")
        .args(args)
        .output()
        .expect("run xold");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// A path as a command-line argument.
fn path(p: &Path) -> String {
    p.display().to_string()
}

/// A fresh directory for one test, or `None` when clang is absent.
fn workdir(tag: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        println!("clang not found; skipping");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("xold-vs-{tag}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// The names in an image's `.dynsym`.
fn dynamic_symbols(bytes: &[u8]) -> Vec<Vec<u8>> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let dynsym = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynsym")
        .and_then(|s| obj.section_data(s).ok())
        .unwrap_or(&[]);
    let dynstr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynstr")
        .and_then(|s| obj.section_data(s).ok())
        .unwrap_or(&[]);
    let mut out = Vec::new();
    for row in dynsym.as_chunks::<24>().0 {
        let off = u32::from_le_bytes(row[..4].try_into().unwrap_or_default());
        let Some(rest) = dynstr.get(off as usize..) else {
            continue;
        };
        let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
        if end != 0 {
            out.push(rest[..end].to_vec());
        }
    }
    out
}
