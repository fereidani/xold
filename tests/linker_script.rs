//! GNU ld linker scripts: the file `-lc` actually resolves to.
//!
//! `/usr/lib64/libc.so` is not a library. It is text that `GROUP`s
//! `libc.so.6`, `libc_nonshared.a` and, `AS_NEEDED`, the loader. A linker that
//! cannot read it resolves `-lc` to `libc.so.6` alone and has no definition of
//! `atexit`, which lives only in the archive, nor of `__tls_get_addr`, which
//! since glibc 2.34 lives only in the loader -- so the link either fails or,
//! worse, produces an image that dies at startup.
//!
//! The parser tests below fix the grammar; the link tests fix what a real
//! `-lc` has to produce, including the `AS_NEEDED` rule, which is what keeps
//! `DT_NEEDED` naming the same libraries GNU ld names.
//!
//! The link tests are gated on `clang` and on a host `libc.so` that really is
//! a script; if either is missing they print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, which, xold_bin};
use xold::script::{Name, Search, looks_like, parse};

mod common;

/// The host's own `libc.so`, verbatim.
const GLIBC: &str = "/* GNU ld script\n   Use the shared library, but some \
                     functions are only in\n   the static library, so try \
                     that secondarily.  */\nOUTPUT_FORMAT(elf64-x86-64)\nGROUP \
                     ( /lib64/libc.so.6 /usr/lib64/libc_nonshared.a  AS_NEEDED \
                     ( /lib64/ld-linux-x86-64.so.2 ) )\n";

// --- the grammar -----------------------------------------------------------

/// The shape every glibc install ships: a comment, an `OUTPUT_FORMAT` that is
/// read and discarded, and a `GROUP` whose last member is `AS_NEEDED`.
#[test]
fn the_glibc_script_names_three_files() {
    let mut names = Vec::new();
    parse(Path::new("libc.so"), GLIBC, &mut names).expect("parses");
    assert_eq!(
        names,
        vec![
            file("/lib64/libc.so.6", false),
            file("/usr/lib64/libc_nonshared.a", false),
            file("/lib64/ld-linux-x86-64.so.2", true),
        ]
    );
}

/// `INPUT` names files the same way `GROUP` does, and either list may hold the
/// `-l` spelling.
#[test]
fn input_takes_both_spellings() {
    let mut names = Vec::new();
    parse(
        Path::new("t"),
        "INPUT ( -lfoo, \"a b.o\" ; ../rel.o )",
        &mut names,
    )
    .expect("parses");
    assert_eq!(
        names,
        vec![
            Name {
                text: "foo",
                library: true,
                as_needed: false,
            },
            file("a b.o", false),
            file("../rel.o", false),
        ],
        "a comma and a semicolon separate, and a quoted name may hold a space"
    );
}

/// A `/` only ends a word when it opens a comment; every path in a real script
/// is full of them.
#[test]
fn a_slash_belongs_to_a_path_unless_it_starts_a_comment() {
    let mut names = Vec::new();
    parse(Path::new("t"), "GROUP(/a/b.so/* here */ /c.so)", &mut names)
        .expect("parses");
    assert_eq!(names, vec![file("/a/b.so", false), file("/c.so", false)]);
}

/// A directive that describes the image is refused rather than skipped.
///
/// Skipping it would produce an image the script explicitly asked not to be
/// produced, and say nothing -- the trade this linker refuses everywhere else
/// it meets an option it does not implement.
#[test]
fn a_directive_that_places_sections_is_refused() {
    let mut names = Vec::new();
    for text in ["SECTIONS { .text : { *(.text) } }", "ENTRY(main)"] {
        let err = parse(Path::new("t"), text, &mut names)
            .expect_err("must not be ignored");
        let shown = err.to_string();
        assert!(
            shown.contains("not implemented"),
            "the diagnostic must say so outright, got {shown}"
        );
    }
}

/// Malformed input ends the link instead of silently naming fewer files.
#[test]
fn an_unterminated_script_is_refused() {
    let mut names = Vec::new();
    for text in [
        "/* forever",
        "GROUP ( a.o",
        "GROUP a.o",
        "GROUP ( \"unclosed )",
    ] {
        assert!(
            parse(Path::new("t"), text, &mut names).is_err(),
            "`{text}` is not a script this linker can follow"
        );
    }
}

/// What the driver hands to the parser, and what it must not.
#[test]
fn only_text_reads_as_a_script() {
    assert!(looks_like(b"/* GNU ld script */\nINPUT(a.o)\n"));
    assert!(!looks_like(b""), "an empty file names nothing");
    assert!(!looks_like(b"\x7fELF\x02\x01\x01"), "an object is not text");
    assert!(
        !looks_like(b"!<arch>\ndebian-binary   1234567890  "),
        "an archive header is printable and is still not a script"
    );
}

/// A relative name is looked for beside the script that named it, which is how
/// a sysroot's own `libc.so` reaches the files next to it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_relative_name_resolves_beside_its_script() {
    let dir = workdir("beside").expect("temp dir");
    fs::write(dir.join("real.so"), b"\x7fELF").expect("write");
    let script = dir.join("libt.so");
    let search = Search {
        paths: &[],
        sysroot: None,
    };
    let found = search
        .resolve(&file("real.so", false), &script)
        .expect("resolves beside the script");
    assert_eq!(found, dir.join("real.so"));
    let _ = fs::remove_dir_all(&dir);
}

// --- what a real `-lc` has to produce --------------------------------------

/// `-lc` alone has to reach `atexit`, and `atexit` is in no shared object.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_plain_lc_reaches_the_static_half_of_libc() {
    let Some(dir) = link_workdir("atexit") else {
        return;
    };
    let Some(out) = link(&dir, "atexit", ATEXIT) else {
        return;
    };
    let run = Command::new(&out).output().expect("the image must run");
    assert!(run.status.success(), "and exit cleanly");
    assert_eq!(
        String::from_utf8_lossy(&run.stdout),
        "hi\nbye\n",
        "the handler `atexit` registered has to have run"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The `AS_NEEDED` member is recorded only when the image binds to it.
///
/// Both halves matter. Recording the loader unconditionally would make every
/// program depend on a library it never calls into, which is what the
/// directive exists to prevent; not recording it when `__tls_get_addr` is
/// called would leave the reference unresolvable at load time. GNU ld draws
/// the line in exactly these two places.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_as_needed_member_is_recorded_only_when_used() {
    let Some(dir) = link_workdir("asneeded") else {
        return;
    };
    let Some(plain) = link(&dir, "plain", ATEXIT) else {
        return;
    };
    assert!(
        !needed(&plain).iter().any(|n| n.starts_with("ld-linux")),
        "a program that calls nothing the loader defines must not depend on it"
    );
    let Some(helper) = link(&dir, "helper", TLS_HELPER) else {
        return;
    };
    assert!(
        needed(&helper).iter().any(|n| n.starts_with("ld-linux")),
        "and one that calls `__tls_get_addr` must, since only the loader \
         defines it"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A program whose exit handler lives in `libc_nonshared.a`.
const ATEXIT: &[u8] = b"#include <stdio.h>\n#include <stdlib.h>\n\
    static void bye(void) { printf(\"bye\\n\"); }\n\
    int main(void) { atexit(bye); printf(\"hi\\n\"); return 0; }\n";

/// A reference to the one function only the loader exports.
const TLS_HELPER: &[u8] = b"extern void *__tls_get_addr(void *);\n\
    int main(void) { return __tls_get_addr(0) != 0; }\n";

// --- fixtures --------------------------------------------------------------

/// A plain file name, as a script's list would hold it.
const fn file(text: &str, as_needed: bool) -> Name<'_> {
    Name {
        text,
        library: false,
        as_needed,
    }
}

/// Links `src` as a dynamic executable against `-lc` and nothing else, so the
/// script is the only thing that can supply the rest of libc.
fn link(dir: &Path, name: &str, src: &[u8]) -> Option<PathBuf> {
    let obj = dir.join(format!("{name}.o"));
    compile(src, &obj)?;
    let out = dir.join(name);
    let crt = |name: &str| crt_file(name).map(|p| p.display().to_string());
    let args = [
        crt("Scrt1.o")?,
        crt("crti.o")?,
        crt("crtbeginS.o")?,
        obj.display().to_string(),
        "-lc".to_owned(),
        crt("crtendS.o")?,
        crt("crtn.o")?,
        "--dynamic-exec".to_owned(),
        "-o".to_owned(),
        out.display().to_string(),
    ];
    let status = Command::new(xold_bin())
        .args(args)
        .output()
        .expect("xold must run");
    assert!(
        status.status.success(),
        "-lc has to be enough on its own: {}",
        String::from_utf8_lossy(&status.stderr).trim()
    );
    Some(out)
}

/// The `DT_NEEDED` names an image records, as `readelf` reports them.
fn needed(image: &Path) -> Vec<String> {
    let out = Command::new("readelf")
        .arg("-d")
        .arg(image)
        .output()
        .expect("readelf must run");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| l.contains("(NEEDED)"))
        .filter_map(|l| {
            let start = l.find('[')?;
            let end = l.rfind(']')?;
            l.get(start.saturating_add(1)..end).map(str::to_owned)
        })
        .collect()
}

/// Compiles `src` to `obj` with the host `clang`.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).ok()?;
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIE", "-O1", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// A fresh per-test working directory.
fn workdir(prefix: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir()
        .join(format!("xold_script_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// The same, for a test that links: `None` (after a note) when the host cannot
/// build the inputs, or when its `libc.so` is not the script these tests are
/// about.
fn link_workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() || !host_libc_is_a_script() {
        eprintln!("skipping linker-script {prefix}: host cannot build it");
        return None;
    }
    workdir(prefix)
}

/// Whether the `libc.so` a `-lc` on this host resolves to is a linker script.
/// A musl or BSD host names a real object there and has nothing to expand.
fn host_libc_is_a_script() -> bool {
    crt_file("libc.so").is_some_and(|path| {
        fs::read(path)
            .is_ok_and(|bytes| looks_like(bytes.get(..64).unwrap_or(&bytes)))
    })
}
