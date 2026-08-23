//! The `-z` keywords a driver writes, and the ones it must not be able to
//! write silently.
//!
//! `clang` hardens by default: `-z relro`, `-z now` and `-z noexecstack` are
//! on every link it drives. Two of those name what xold already does, and one
//! -- `-z now` -- changes the image, so it has to reach `DT_FLAGS`. The
//! keywords that ask for an image xold does not produce are refused by name,
//! because a hardening flag that is accepted and dropped is worse than one
//! that is refused: the build believes it got what it asked for.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, which, xold_bin};
use xold::elf::ObjectFile;

mod common;

/// `DT_NULL`, the terminator of the dynamic array.
const DT_NULL: i64 = 0;
/// `DT_FLAGS`.
const DT_FLAGS: i64 = 30;
/// `DT_FLAGS_1`.
const DT_FLAGS_1: i64 = 0x6fff_fffb;
/// `DF_BIND_NOW`.
const DF_BIND_NOW: u64 = 0x08;
/// `DF_1_NOW`.
const DF_1_NOW: u64 = 0x01;
/// `PT_GNU_STACK`.
const PT_GNU_STACK: u32 = 0x6474_e551;

/// A program with something to bind: the call reaches libc through the PLT,
/// which is what `-z now` changes the binding time of.
const MAIN_SRC: &[u8] = b"int puts(const char *);\n\
    int main(void) { return puts(\"x\"); }\n";

/// A library with a reference nothing in the link defines, which is what
/// `-z defs` refuses and a plain `-shared` link allows.
const OPEN_SRC: &[u8] = b"extern int elsewhere(void);\n\
    int here(void) { return elsewhere(); }\n";

/// `-z now` reaches both flag words.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn now_is_recorded_in_the_flag_words() {
    let Some(dir) = workdir("now") else {
        return;
    };
    let Some(out) = dynamic_link(&dir, MAIN_SRC, &["-z".into(), "now".into()])
    else {
        return;
    };
    let image = fs::read(&out).unwrap_or_default();
    assert!(
        flag_word(&image, DT_FLAGS) & DF_BIND_NOW != 0,
        "-z now must set DF_BIND_NOW"
    );
    assert!(
        flag_word(&image, DT_FLAGS_1) & DF_1_NOW != 0,
        "and DF_1_NOW, which is the word current loaders read"
    );

    let Some(lazy) = dynamic_link(&dir, MAIN_SRC, &[]) else {
        return;
    };
    let image = fs::read(&lazy).unwrap_or_default();
    assert!(
        flag_word(&image, DT_FLAGS) & DF_BIND_NOW == 0,
        "and a link that did not ask for it must not carry it"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `-z stack-size=N` reaches `PT_GNU_STACK`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn stack_size_reaches_the_program_header() {
    let Some(dir) = workdir("stack") else {
        return;
    };
    let Some(out) = dynamic_link(
        &dir,
        MAIN_SRC,
        &["-z".into(), "stack-size=0x200000".into()],
    ) else {
        return;
    };
    let image = fs::read(&out).unwrap_or_default();
    assert_eq!(
        gnu_stack_size(&image),
        Some(0x0020_0000),
        "the stack size asked for must be the one PT_GNU_STACK carries"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A keyword describing an image xold does not produce is refused, and so is
/// one nobody implements.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_image_changing_keyword_is_refused() {
    let Some(dir) = workdir("refused") else {
        return;
    };
    let Some(obj) = compile(&dir, "m.c", MAIN_SRC, &[]) else {
        return;
    };
    let out = dir.join("prog");
    for (kw, expected) in [
        ("execstack", "executable stack"),
        ("nosuchkeyword", "unknown"),
    ] {
        let (ok, err) = xold(&[
            "-z".into(),
            kw.into(),
            "-o".into(),
            path(&out),
            path(&obj),
        ]);
        assert!(!ok, "-z {kw} must not be accepted silently");
        assert!(err.contains(expected), "and must say why: {err}");
    }
    let _ = fs::remove_dir_all(&dir);
}

/// `-z defs` refuses a shared object with a name nothing defines; without it
/// the same link is fine, because a library's unresolved names are normally
/// the loader's problem.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn defs_refuses_an_unresolved_reference() {
    let Some(dir) = workdir("defs") else {
        return;
    };
    let Some(obj) = compile(&dir, "open.c", OPEN_SRC, &["-fPIC"]) else {
        return;
    };
    let out = dir.join("libopen.so");
    let (ok, err) =
        xold(&["-shared".into(), "-o".into(), path(&out), path(&obj)]);
    assert!(ok, "a shared object may leave a name to the loader: {err}");

    let (ok, err) = xold(&[
        "-shared".into(),
        "-z".into(),
        "defs".into(),
        "-o".into(),
        path(&out),
        path(&obj),
    ]);
    assert!(!ok, "-z defs must refuse the same link");
    assert!(
        err.contains("elsewhere"),
        "and must name the symbol nothing defines: {err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Links `src` into a dynamic executable with `extra` on the command line.
fn dynamic_link(dir: &Path, src: &[u8], extra: &[String]) -> Option<PathBuf> {
    let obj = compile(dir, "m.c", src, &[])?;
    let start = crt_file("crt1.o")?;
    let open = crt_file("crti.o")?;
    let close = crt_file("crtn.o")?;
    let out = dir.join(format!("prog{}", extra.len()));
    let mut args = vec!["-o".into(), path(&out)];
    args.extend(extra.iter().cloned());
    args.push(path(&start));
    args.push(path(&open));
    args.push(path(&obj));
    args.push("-lc".into());
    args.push(path(&close));
    let (ok, err) = xold(&args);
    assert!(ok, "the fixture link must succeed: {err}");
    Some(out)
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
    let dir = std::env::temp_dir().join(format!("xold-z-{tag}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// The value of one `.dynamic` tag, or zero when it is not there.
fn flag_word(bytes: &[u8], want: i64) -> u64 {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return 0;
    };
    let Some(data) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynamic")
        .and_then(|s| obj.section_data(s).ok())
    else {
        return 0;
    };
    for entry in data.as_chunks::<16>().0 {
        let tag = i64::from_le_bytes(entry[..8].try_into().unwrap_or_default());
        let val = u64::from_le_bytes(entry[8..].try_into().unwrap_or_default());
        if tag == DT_NULL {
            break;
        }
        if tag == want {
            return val;
        }
    }
    0
}

/// The `p_memsz` of `PT_GNU_STACK`, read straight out of the image.
fn gnu_stack_size(bytes: &[u8]) -> Option<u64> {
    let phoff = u64::from_le_bytes(bytes.get(0x20..0x28)?.try_into().ok()?);
    let phentsize =
        u16::from_le_bytes(bytes.get(0x36..0x38)?.try_into().ok()?) as usize;
    let phnum =
        u16::from_le_bytes(bytes.get(0x38..0x3a)?.try_into().ok()?) as usize;
    let base = usize::try_from(phoff).ok()?;
    for i in 0..phnum {
        let at = base.checked_add(i.checked_mul(phentsize)?)?;
        let phdr = bytes.get(at..at.checked_add(phentsize)?)?;
        let kind = u32::from_le_bytes(phdr.get(0..4)?.try_into().ok()?);
        if kind == PT_GNU_STACK {
            return Some(u64::from_le_bytes(
                phdr.get(40..48)?.try_into().ok()?,
            ));
        }
    }
    None
}
