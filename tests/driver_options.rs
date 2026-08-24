//! The options a compiler driver writes on every link.
//!
//! `clang` and `gcc` do not hand a linker a minimal command line. They spell
//! out the target (`-m elf_x86_64`), an optimisation level, the archive or
//! shared preference around individual libraries (`-Bstatic`, `-Bdynamic`,
//! `--push-state`, `--pop-state`), and whether a dependency has to be bound to
//! before it earns its `DT_NEEDED` (`--as-needed`). A linker that refuses any
//! of them cannot be reached through a driver at all, and one that accepts
//! them and ignores them links something other than what was asked for.
//!
//! These tests cover both halves: the options are accepted, and each one
//! changes the image the way its name promises.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

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
/// `DT_NEEDED`: the `.dynstr` offset of a dependency's soname.
const DT_NEEDED: i64 = 1;
/// `DT_RUNPATH`: the `.dynstr` offset of the runtime search path.
const DT_RUNPATH: i64 = 29;

/// The program under test: it uses nothing from the extra library, so whether
/// that library is recorded is exactly the question `--as-needed` answers.
const MAIN_SRC: &[u8] = b"int main(void) { return 0; }\n";

/// A definition of the default entry symbol, for the links that name no crt
/// objects and so would otherwise have no entry point.
const START_SRC: &[u8] = b"void _start(void) { }\n";

/// An input whose stored address this link has to fix, which is what makes a
/// position-independent image impossible.
const ABSOLUTE_SRC: &[u8] = b"int gv = 1;\nint *p = &gv;\n\
    int main(void) { return *p; }\n";

/// `ET_EXEC`, the fixed-base executable type.
const ET_EXEC: u16 = 2;
/// `ET_DYN`, the type a position-independent executable carries.
const ET_DYN: u16 = 3;

/// The library the command line names but the program never calls.
const DEP_SRC: &[u8] = b"int dep_value(void) { return 42; }\n";

/// `-m` naming another machine than the inputs carry is refused.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_wrong_emulation_is_refused() {
    let Some(dir) = workdir("emulation") else {
        return;
    };
    let Some(obj) = compile(&dir, "m.c", MAIN_SRC, &[]) else {
        return;
    };
    let out = dir.join("prog");
    let (ok, err) = xold(&[
        "-m".into(),
        "aarch64linux".into(),
        "-o".into(),
        path(&out),
        path(&obj),
    ]);
    assert!(!ok, "an emulation the inputs contradict must not link");
    assert!(
        err.contains("aarch64linux") && err.contains("x86-64"),
        "and the diagnostic must name both sides: {err}"
    );

    let (ok, err) = xold(&[
        "-m".into(),
        "elf_i386".into(),
        "-o".into(),
        path(&out),
        path(&obj),
    ]);
    assert!(
        !ok,
        "and an emulation this linker cannot produce is refused"
    );
    assert!(
        err.contains("unknown emulation"),
        "with a message that says so: {err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `-m` naming this link's own machine, an `-O` level and the state markers
/// are accepted and change nothing.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_driver_preamble_links() {
    let Some(dir) = workdir("preamble") else {
        return;
    };
    let Some(obj) = compile(&dir, "m.c", START_SRC, &[]) else {
        return;
    };
    let plain = dir.join("plain");
    let (ok, err) = xold(&["-o".into(), path(&plain), path(&obj)]);
    assert!(ok, "the plain link must succeed first: {err}");

    let driven = dir.join("driven");
    let (ok, err) = xold(&[
        "-m".into(),
        "elf_x86_64".into(),
        "-O1".into(),
        "--threads".into(),
        "1".into(),
        "--push-state".into(),
        "--as-needed".into(),
        "--pop-state".into(),
        "-o".into(),
        path(&driven),
        path(&obj),
    ]);
    assert!(
        ok,
        "and so must the same link with the driver preamble: {err}"
    );
    assert_eq!(
        fs::read(&plain).ok(),
        fs::read(&driven).ok(),
        "none of those options describes a different image, so the bytes \
         must match"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `--as-needed` withholds `DT_NEEDED` from a dependency nothing binds to,
/// and `--no-as-needed` restores it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn as_needed_decides_dt_needed() {
    let Some(dir) = workdir("as-needed") else {
        return;
    };
    let Some(obj) = compile(&dir, "m.c", MAIN_SRC, &[]) else {
        return;
    };
    let Some(dep) = shared_library(&dir) else {
        return;
    };
    let Some(crt) = crt_objects() else {
        return;
    };

    let mut kept = crt.prefix.clone();
    kept.push(path(&obj));
    kept.push("-L".into());
    kept.push(path(&dir));
    kept.push("-ldep".into());
    kept.extend(crt.suffix.clone());
    let out = dir.join("kept");
    let mut args = vec!["-o".into(), path(&out)];
    args.extend(kept.clone());
    let (ok, err) = xold(&args);
    assert!(ok, "the unconditional link must succeed: {err}");
    assert!(
        needed_names(&fs::read(&out).unwrap_or_default())
            .iter()
            .any(|n| n == b"libdep.so"),
        "and record the library it was given"
    );

    let out = dir.join("dropped");
    let mut args = vec!["-o".into(), path(&out)];
    args.extend(crt.prefix.clone());
    args.push(path(&obj));
    args.push("-L".into());
    args.push(path(&dir));
    args.push("--as-needed".into());
    args.push("-ldep".into());
    args.push("--no-as-needed".into());
    args.extend(crt.suffix);
    let (ok, err) = xold(&args);
    assert!(ok, "the --as-needed link must succeed too: {err}");
    assert!(
        !needed_names(&fs::read(&out).unwrap_or_default())
            .iter()
            .any(|n| n == b"libdep.so"),
        "and must not record a library nothing in the image binds to"
    );
    let _ = dep;
    let _ = fs::remove_dir_all(&dir);
}

/// `-Bstatic` narrows `-l` to archives: a directory holding only the shared
/// object no longer satisfies it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn bstatic_looks_only_for_an_archive() {
    let Some(dir) = workdir("bstatic") else {
        return;
    };
    let Some(obj) = compile(&dir, "m.c", START_SRC, &[]) else {
        return;
    };
    if shared_library(&dir).is_none() {
        return;
    }
    let out = dir.join("prog");
    let (ok, err) = xold(&[
        "-o".into(),
        path(&out),
        path(&obj),
        "-L".into(),
        path(&dir),
        "-Bstatic".into(),
        "-ldep".into(),
    ]);
    assert!(!ok, "-Bstatic must not fall back to the shared object");
    assert!(
        err.contains("libdep.a"),
        "and must say which file it looked for: {err}"
    );

    let (ok, err) = xold(&[
        "-o".into(),
        path(&out),
        path(&obj),
        "-L".into(),
        path(&dir),
        "-Bstatic".into(),
        "-Bdynamic".into(),
        "-ldep".into(),
    ]);
    assert!(ok, "-Bdynamic must restore the default search: {err}");
    let _ = fs::remove_dir_all(&dir);
}

/// `-static` and a shared object among the inputs contradict each other.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn static_refuses_a_shared_input() {
    let Some(dir) = workdir("static") else {
        return;
    };
    let Some(obj) = compile(&dir, "m.c", MAIN_SRC, &[]) else {
        return;
    };
    let Some(dep) = shared_library(&dir) else {
        return;
    };
    let out = dir.join("prog");
    let (ok, err) = xold(&[
        "-static".into(),
        "-o".into(),
        path(&out),
        path(&obj),
        path(&dep),
    ]);
    assert!(!ok, "a static image cannot depend on a shared object");
    assert!(
        err.contains("-static"),
        "and the diagnostic must name the option: {err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `-pie` and `-no-pie` decide the image type, and `-pie` over an input that
/// cannot be loaded at a random base is refused rather than downgraded.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn pie_and_no_pie_decide_the_image_type() {
    let Some(dir) = workdir("pie") else {
        return;
    };
    let Some(obj) = compile(&dir, "m.c", MAIN_SRC, &["-fPIE"]) else {
        return;
    };
    let Some(crt) = crt_objects() else {
        return;
    };

    for (flag, want) in [("-pie", ET_DYN), ("-no-pie", ET_EXEC)] {
        let out = dir.join(flag.trim_start_matches('-'));
        let mut args = vec![flag.to_string(), "-o".into(), path(&out)];
        args.extend(crt.prefix.clone());
        args.push(path(&obj));
        args.extend(crt.suffix.clone());
        let (ok, err) = xold(&args);
        assert!(ok, "{flag} must link: {err}");
        assert_eq!(
            e_type(&fs::read(&out).unwrap_or_default()),
            Some(want),
            "{flag} must decide the image type"
        );
    }

    let Some(fixed) = compile(&dir, "abs.c", ABSOLUTE_SRC, &["-fno-pic"])
    else {
        return;
    };
    let out = dir.join("refused");
    let mut args = vec!["-pie".to_string(), "-o".into(), path(&out)];
    args.extend(crt.prefix.clone());
    args.push(path(&fixed));
    args.extend(crt.suffix);
    let (ok, err) = xold(&args);
    assert!(
        !ok,
        "-pie over a non-PIC absolute reference must be refused"
    );
    assert!(
        err.contains("-pie"),
        "and the diagnostic must name the option: {err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `--strip-debug` drops the debug sections and keeps the symbol table;
/// `--strip-all` drops both, and the image still runs.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn stripping_drops_what_it_names() {
    let Some(dir) = workdir("strip") else {
        return;
    };
    let Some(obj) = compile(&dir, "m.c", MAIN_SRC, &["-g", "-fPIE"]) else {
        return;
    };
    let Some(crt) = crt_objects() else {
        return;
    };
    let link = |flags: &[&str], name: &str| -> Vec<Vec<u8>> {
        let out = dir.join(name);
        let mut args: Vec<String> =
            flags.iter().map(|f| (*f).to_string()).collect();
        args.push("-o".into());
        args.push(path(&out));
        args.extend(crt.prefix.clone());
        args.push(path(&obj));
        args.extend(crt.suffix.clone());
        let (ok, err) = xold(&args);
        assert!(ok, "the {name} link must succeed: {err}");
        section_names(&fs::read(&out).unwrap_or_default())
    };

    let kept = link(&[], "kept");
    assert!(
        kept.iter().any(|n| n.starts_with(b".debug_")),
        "an unstripped link keeps its debug sections"
    );
    assert!(kept.iter().any(|n| n == b".symtab"), "and its symbol table");

    let no_debug = link(&["--strip-debug"], "no-debug");
    assert!(
        !no_debug.iter().any(|n| n.starts_with(b".debug_")),
        "--strip-debug drops the debug sections"
    );
    assert!(
        no_debug.iter().any(|n| n == b".symtab"),
        "and keeps the symbol table a backtrace needs"
    );

    let bare = link(&["--strip-all"], "bare");
    assert!(
        !bare.iter().any(|n| n == b".symtab" || n == b".strtab"),
        "--strip-all drops the symbol table too"
    );
    assert!(
        bare.iter().any(|n| n == b".dynsym"),
        "but never the dynamic symbol table, which the loader reads"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `--hash-style` decides which symbol hash tables the image carries, and
/// the image still runs on either one.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn hash_style_decides_the_tables() {
    let Some(dir) = workdir("hash") else {
        return;
    };
    let Some(obj) = compile(&dir, "m.c", MAIN_SRC, &["-fPIE"]) else {
        return;
    };
    let Some(crt) = crt_objects() else {
        return;
    };
    for (style, sysv, gnu) in [
        ("both", true, true),
        ("sysv", true, false),
        ("gnu", false, true),
    ] {
        let out = dir.join(style);
        let mut args =
            vec![format!("--hash-style={style}"), "-o".into(), path(&out)];
        args.extend(crt.prefix.clone());
        args.push(path(&obj));
        args.extend(crt.suffix.clone());
        let (ok, err) = xold(&args);
        assert!(ok, "--hash-style={style} must link: {err}");
        let names = section_names(&fs::read(&out).unwrap_or_default());
        assert_eq!(
            names.iter().any(|n| n == b".hash"),
            sysv,
            "--hash-style={style} decides whether .hash is emitted"
        );
        assert_eq!(
            names.iter().any(|n| n == b".gnu.hash"),
            gnu,
            "--hash-style={style} decides whether .gnu.hash is emitted"
        );
        assert!(
            Command::new(&out).status().is_ok_and(|s| s.success()),
            "and the image must still run under the loader"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

/// `-rpath` is recorded as `DT_RUNPATH`, with every directory written joined
/// by `:` in the order they were given.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn rpath_becomes_dt_runpath() {
    let Some(dir) = workdir("rpath") else {
        return;
    };
    let Some(obj) = compile(&dir, "m.c", MAIN_SRC, &["-fPIE"]) else {
        return;
    };
    let Some(crt) = crt_objects() else {
        return;
    };
    let out = dir.join("prog");
    let mut args = vec![
        "-rpath".into(),
        "/opt/lib".into(),
        "-rpath=/opt/other".into(),
        "-o".into(),
        path(&out),
    ];
    args.extend(crt.prefix.clone());
    args.push(path(&obj));
    args.extend(crt.suffix);
    let (ok, err) = xold(&args);
    assert!(ok, "the link must succeed: {err}");
    assert_eq!(
        runpath(&fs::read(&out).unwrap_or_default()),
        Some(b"/opt/lib:/opt/other".to_vec()),
        "both directories must be recorded, in the order written"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `--build-id` records a note that names the image, the same note on every
/// run, and a different one once the image changes.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn build_id_names_the_image() {
    let Some(dir) = workdir("build-id") else {
        return;
    };
    let Some(obj) = compile(&dir, "m.c", MAIN_SRC, &["-fPIE"]) else {
        return;
    };
    let Some(other) = compile(&dir, "o.c", ABSOLUTE_SRC, &["-fPIE"]) else {
        return;
    };
    let Some(crt) = crt_objects() else {
        return;
    };
    let link = |style: &str, obj: &Path, name: &str| -> Option<Vec<u8>> {
        let out = dir.join(name);
        let mut args = vec![style.to_string(), "-o".into(), path(&out)];
        args.extend(crt.prefix.clone());
        args.push(path(obj));
        args.extend(crt.suffix.clone());
        let (ok, err) = xold(&args);
        assert!(ok, "the {name} link must succeed: {err}");
        build_id(&fs::read(&out).unwrap_or_default())
    };

    let first = link("--build-id=sha1", &obj, "first");
    assert_eq!(
        first.as_ref().map(Vec::len),
        Some(20),
        "--build-id=sha1 records a 20-byte note"
    );
    assert_eq!(
        first,
        link("--build-id=sha1", &obj, "again"),
        "and the same inputs must produce the same id"
    );
    assert_ne!(
        first,
        link("--build-id=sha1", &other, "other"),
        "while a different image must get a different one"
    );
    assert_eq!(
        link("--build-id=0xdeadbeef", &obj, "literal"),
        Some(vec![0xde, 0xad, 0xbe, 0xef]),
        "and a literal id is recorded as written"
    );
    assert_eq!(
        link("--build-id=fast", &obj, "fast").map(|id| id.len()),
        Some(16),
        "the default style records a 16-byte id"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `gcc` passes its LTO plugin on every link, so the option is accepted --
/// and an input that actually holds LTO bytecode is refused by name.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_lto_plugin_is_accepted_and_lto_input_is_not() {
    let Some(dir) = workdir("plugin") else {
        return;
    };
    let Some(obj) = compile(&dir, "m.c", MAIN_SRC, &["-fPIE"]) else {
        return;
    };
    let Some(crt) = crt_objects() else {
        return;
    };
    let out = dir.join("prog");
    let mut args = vec![
        "-plugin".into(),
        "/usr/lib/gcc/liblto_plugin.so".into(),
        "-plugin-opt=-fresolution=/tmp/whatever.res".into(),
        "-o".into(),
        path(&out),
    ];
    args.extend(crt.prefix.clone());
    args.push(path(&obj));
    args.extend(crt.suffix.clone());
    let (ok, err) = xold(&args);
    assert!(
        ok,
        "the plugin options must not stop an ordinary link: {err}"
    );

    let Some(lto) = compile(&dir, "l.c", MAIN_SRC, &["-flto"]) else {
        return;
    };
    // A fat LTO object carries real code beside the bytecode, and links the
    // way any other object does.
    if let Some(fat) =
        compile(&dir, "f.c", MAIN_SRC, &["-flto", "-ffat-lto-objects"])
    {
        let mut args = vec!["-o".into(), path(&out)];
        args.extend(crt.prefix.clone());
        args.push(path(&fat));
        args.extend(crt.suffix);
        let (ok, err) = xold(&args);
        assert!(ok, "a fat LTO object must link like any other: {err}");
    }
    // clang -flto emits bitcode rather than an ELF object, which the reader
    // rejects for another reason; only the gcc shape is under test here.
    if fs::read(&lto).is_ok_and(|b| b.starts_with(b"\x7fELF")) {
        let (ok, err) = xold(&[
            "-o".into(),
            path(&out),
            path(&lto),
            "--entry".into(),
            "main".into(),
        ]);
        assert!(!ok, "an LTO object must not link to a silent half-image");
        assert!(
            err.contains("lto"),
            "and the diagnostic must name what it is: {err}"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

/// The crt objects a dynamic executable is linked with, split around the
/// inputs the way a driver writes them.
struct Crt {
    prefix: Vec<String>,
    suffix: Vec<String>,
}

/// The host's crt objects, or `None` when they are not installed.
fn crt_objects() -> Option<Crt> {
    let start = crt_file("crt1.o")?;
    let open = crt_file("crti.o")?;
    let close = crt_file("crtn.o")?;
    Some(Crt {
        prefix: vec![path(&start), path(&open)],
        suffix: vec!["-lc".into(), path(&close)],
    })
}

/// Builds `libdep.so` in `dir`, or `None` when the host toolchain cannot.
fn shared_library(dir: &Path) -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        let obj = compile(dir, "dep.c", DEP_SRC, &["-fPIC"])?;
        let out = dir.join("libdep.so");
        let (ok, _) = xold(&[
            "-shared".into(),
            "-soname".into(),
            "libdep.so".into(),
            "-o".into(),
            path(&out),
            path(&obj),
        ]);
        return ok.then_some(out);
    }
    #[cfg(not(target_os = "macos"))]
    {
        let clang = which("clang")?;
        let src = dir.join("dep.c");
        fs::write(&src, DEP_SRC).ok()?;
        let out = dir.join("libdep.so");
        let ok = Command::new(clang)
            .args(["-fPIC", "-shared"])
            .arg(&src)
            .arg("-o")
            .arg(&out)
            .status()
            .ok()?
            .success();
        ok.then_some(out)
    }
}

/// Compiles one source file with the host `clang`.
fn compile(
    dir: &Path,
    name: &str,
    src: &[u8],
    extra: &[&str],
) -> Option<PathBuf> {
    let clang = which("clang")?;
    let path = dir.join(name);
    fs::write(&path, src).ok()?;
    let obj = dir.join(format!("{name}.o"));
    let ok = Command::new(clang)
        .args([
            "--target=x86_64-linux-gnu",
            "-c",
            "-fno-asynchronous-unwind-tables",
        ])
        .args(extra)
        .arg(&path)
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

/// The `DT_RUNPATH` string of an image, if it has one.
fn runpath(bytes: &[u8]) -> Option<Vec<u8>> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return None;
    };
    let strtab = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynstr")
        .and_then(|s| obj.section_data(s).ok())?;
    let (_, val) = dyn_entries(bytes)
        .into_iter()
        .find(|(tag, _)| *tag == DT_RUNPATH)?;
    let rest = strtab.get(usize::try_from(val).ok()?..)?;
    let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
    Some(rest[..end].to_vec())
}

/// The descriptor of the image's `NT_GNU_BUILD_ID` note, if it has one.
fn build_id(bytes: &[u8]) -> Option<Vec<u8>> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let data = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".note.gnu.build-id")
        .and_then(|s| obj.section_data(s).ok())?;
    // n_namesz, n_descsz, n_type, then the padded name, then the descriptor.
    let descsz = u32::from_le_bytes(data.get(4..8)?.try_into().ok()?) as usize;
    Some(data.get(16..16 + descsz)?.to_vec())
}

/// The section names of an image.
fn section_names(bytes: &[u8]) -> Vec<Vec<u8>> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    obj.sections()
        .iter()
        .map(|s| obj.section_name(s).to_vec())
        .collect()
}

/// The `e_type` of an image: the half-word at offset 16.
fn e_type(bytes: &[u8]) -> Option<u16> {
    bytes
        .get(16..18)
        .and_then(|b| <[u8; 2]>::try_from(b).ok())
        .map(u16::from_le_bytes)
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
    let dir = std::env::temp_dir().join(format!("xold-driver-{tag}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// The dynamic array of an image, as `(tag, value)` pairs.
fn dyn_entries(bytes: &[u8]) -> Vec<(i64, u64)> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(data) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynamic")
        .and_then(|s| obj.section_data(s).ok())
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in data.as_chunks::<16>().0 {
        let tag = i64::from_le_bytes(entry[..8].try_into().unwrap_or_default());
        let val = u64::from_le_bytes(entry[8..].try_into().unwrap_or_default());
        if tag == DT_NULL {
            break;
        }
        out.push((tag, val));
    }
    out
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
