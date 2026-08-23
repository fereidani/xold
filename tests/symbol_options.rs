//! The options that decide which symbols a link looks for and publishes.
//!
//! `-u` names something the inputs never reference, so an archive member
//! defining it joins the link and garbage collection keeps it: a runtime whose
//! entry is reached only from outside the image depends on it. `-E` publishes
//! an executable's own definitions, which is what a program whose plugins bind
//! back to it needs. And `--version` answers the question every build system
//! asks a linker before it decides to use it.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, which, xold_bin};

mod common;

/// A program that neither calls nor names the archive member below.
const MAIN_SRC: &[u8] = b"void _start(void) { }\n";

/// The member `-u` is written to pull in.
const EXTRA_SRC: &[u8] = b"int extra_fn(void) { return 7; }\n";

/// `-u` pulls an archive member nothing references, and `--gc-sections`
/// keeps it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn undefined_pulls_an_archive_member() {
    let Some(dir) = workdir("undefined") else {
        return;
    };
    let Some(main) = compile(&dir, "m.c", MAIN_SRC, &[]) else {
        return;
    };
    let Some(extra) = compile(&dir, "x.c", EXTRA_SRC, &["-ffunction-sections"])
    else {
        return;
    };
    if archive(&dir, &extra).is_none() {
        return;
    }

    let bare = dir.join("bare");
    let (ok, err) = xold(&[
        "-o".into(),
        path(&bare),
        path(&main),
        "-L".into(),
        path(&dir),
        "-lextra".into(),
    ]);
    assert!(ok, "the link without -u must succeed: {err}");
    assert!(
        !defines(&bare, "extra_fn"),
        "a member nothing references stays out of the image"
    );

    for extra_args in [
        vec!["-u", "extra_fn"],
        vec!["--gc-sections", "-u", "extra_fn"],
    ] {
        let out = dir.join("pulled");
        let mut args: Vec<String> =
            extra_args.iter().map(|a| (*a).to_string()).collect();
        args.extend([
            "-o".into(),
            path(&out),
            path(&main),
            "-L".into(),
            path(&dir),
            "-lextra".into(),
        ]);
        let (ok, err) = xold(&args);
        assert!(ok, "the -u link must succeed: {err}");
        assert!(
            defines(&out, "extra_fn"),
            "-u must pull the member in and keep it, even under {extra_args:?}"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

/// `--export-dynamic` publishes an executable's own definitions, so an object
/// loaded at runtime can bind to them.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn export_dynamic_publishes_the_program() {
    let Some(dir) = workdir("export-dynamic") else {
        return;
    };
    let Some(obj) = compile(&dir, "h.c", HOST_SRC, &["-fPIE"]) else {
        return;
    };
    let Some(start) = crt_file("Scrt1.o") else {
        return;
    };
    let Some(open) = crt_file("crti.o") else {
        return;
    };
    let Some(close) = crt_file("crtn.o") else {
        return;
    };
    let link = |flags: &[&str], name: &str| -> PathBuf {
        let out = dir.join(name);
        let mut args: Vec<String> =
            flags.iter().map(|f| (*f).to_string()).collect();
        args.extend([
            "-o".into(),
            path(&out),
            path(&start),
            path(&open),
            path(&obj),
            "-lc".into(),
            path(&close),
        ]);
        let (ok, err) = xold(&args);
        assert!(ok, "the {name} link must succeed: {err}");
        out
    };

    let published = link(&["--export-dynamic"], "published");
    assert!(
        dynamic_defines(&published, "host_value"),
        "--export-dynamic puts the program's definitions in .dynsym"
    );
    let plain = link(&[], "plain");
    assert!(
        !dynamic_defines(&plain, "host_value"),
        "and without it an executable publishes only what a dependency asks \
         for"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The program under test defines a name a plugin would bind to.
const HOST_SRC: &[u8] = b"int host_value(void) { return 41; }\n\
    int main(void) { return 0; }\n";

/// `--version` and `--help` answer without linking, which is how a build
/// system identifies a linker before it uses one.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_linker_identifies_itself() {
    let out = Command::new(xold_bin())
        .arg("--version")
        .output()
        .expect("run xold");
    assert!(out.status.success(), "--version must succeed on its own");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.starts_with("xold "),
        "and must name the linker first: {text}"
    );

    let out = Command::new(xold_bin())
        .arg("--help")
        .output()
        .expect("run xold");
    assert!(out.status.success(), "--help must succeed on its own");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("usage: xold"),
        "and must print the usage: {text}"
    );
}

/// Whether the image's symbol table defines `name`.
fn defines(path: &Path, name: &str) -> bool {
    symbol_present(path, name, false)
}

/// Whether the image's dynamic symbol table defines `name`.
fn dynamic_defines(path: &Path, name: &str) -> bool {
    symbol_present(path, name, true)
}

/// Whether `name` is a defined symbol of the image's `.symtab` or `.dynsym`.
fn symbol_present(path: &Path, name: &str, dynamic: bool) -> bool {
    use xold::elf::ObjectFile;
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    let Ok(obj) = ObjectFile::parse(&bytes) else {
        return false;
    };
    let (table, strings) = if dynamic {
        (b".dynsym".as_slice(), b".dynstr".as_slice())
    } else {
        (b".symtab".as_slice(), b".strtab".as_slice())
    };
    let syms = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == table)
        .and_then(|s| obj.section_data(s).ok())
        .unwrap_or(&[]);
    let strtab = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == strings)
        .and_then(|s| obj.section_data(s).ok())
        .unwrap_or(&[]);
    syms.as_chunks::<24>().0.iter().any(|row| {
        let off = u32::from_le_bytes(row[..4].try_into().unwrap_or_default());
        let shndx =
            u16::from_le_bytes(row[6..8].try_into().unwrap_or_default());
        if shndx == 0 {
            return false;
        }
        strtab
            .get(off as usize..)
            .map(|rest| {
                let end = rest.iter().position(|&b| b == 0).unwrap_or(0);
                &rest[..end]
            })
            .is_some_and(|found| found == name.as_bytes())
    })
}

/// Packs `member` into `libextra.a`.
fn archive(dir: &Path, member: &Path) -> Option<PathBuf> {
    let ar = which("ar")?;
    let out = dir.join("libextra.a");
    let ok = Command::new(ar)
        .arg("rcs")
        .arg(&out)
        .arg(member)
        .status()
        .ok()?
        .success();
    ok.then_some(out)
}

/// Compiles one source file with the host `clang`.
fn compile(
    dir: &Path,
    name: &str,
    src: &[u8],
    extra: &[&str],
) -> Option<PathBuf> {
    let clang = which("clang")?;
    let file = dir.join(name);
    fs::write(&file, src).ok()?;
    let obj = dir.join(format!("{name}.o"));
    let ok = Command::new(clang)
        .arg("-c")
        .args(extra)
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
    let dir = std::env::temp_dir().join(format!("xold-sym-{tag}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}
