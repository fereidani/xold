//! End-to-end tests for `--gc-sections`: dead-section garbage
//! collection.
//!
//! With `--gc-sections` an unused function compiled with `-ffunction-sections`
//! (its own input section) is dropped from the output, while the program still
//! links and runs identically. These tests cover the four guarantees the
//! feature makes:
//!
//! - **Functional**: the repro drops the unused section's bytes (its `.text`
//!   content vanishes) and the binary still runs (`used=7`, exit 0).
//! - **Byte-identical default**: linking the same input *without* the flag is
//!   deterministic (two runs produce byte-equal output); the wider suite guards
//!   byte-identity against the pre-feature behaviour.
//! - **No false drops**: a function reached indirectly -- through an
//!   `.init_array` constructor pointer, and through a function pointer stored
//!   in `.data` -- is kept and the program runs, as is a section reached only
//!   by walking between `__start_NAME` and `__stop_NAME`.
//! - **Size**: with gc the binary is strictly smaller than without.
//!
//! xold folds input sections into canonical output sections (`.text`,
//! `.rodata`, ...), so a dropped function's name never appears as an output
//! section header. Its absence is therefore asserted on its code bytes inside
//! `.text` and on the `.text` size shrinking, which is the meaningful
//! equivalent under this output model.
//!
//! Gated on `clang`, `gcc` (to locate the crt objects), and the system
//! `ld.so`; if absent the tests print a note and return, so the build never
//! fails over a missing toolchain.

#![allow(clippy::similar_names)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_dyn_exec};

mod common;

/// The repro source. `unused_fn` is `static` and unreferenced; clang would
/// elide a plain unused static at compile time, so `__attribute__((used))`
/// forces emission into `.text.unused_fn` -- exactly the case `--gc-sections`
/// exists to handle. `used_fn` is called by `main` and must survive.
const GC_SRC: &[u8] = b"#include <stdio.h>\n\
     static __attribute__((used)) int unused_fn(void){ return 42; }\n\
     int used_fn(void){ return 7; }\n\
     int main(void){ printf(\"used=%d\\n\", used_fn()); return 0; }\n";

/// A constructor that stamps a buffer; `main` reads it, so the program prints
/// `buf=ok` only if the constructor survived gc and ran before `main`.
const CTOR_SRC: &[u8] = b"#include <stdio.h>\n\
     #include <stdlib.h>\n\
     #include <string.h>\n\
     static char *buf;\n\
     __attribute__((constructor)) static void init(void){\n\
         buf = malloc(4); strcpy(buf, \"ok\");\n\
     }\n\
     int main(void){ printf(\"buf=%s\\n\", buf); return buf == NULL; }\n";

/// A table in a section named by a C identifier, reached only by walking
/// between `__start_mysec` and `__stop_mysec`. No relocation names the
/// entries, so reachability alone would collect them and leave the bounds
/// describing nothing. `unused_thing` is unreferenced and disposable.
const START_STOP_SRC: &[u8] = b"#include <stdio.h>\n\
     static int e1 __attribute__((used, section(\"mysec\"))) = 4;\n\
     static int e2 __attribute__((used, section(\"mysec\"))) = 5;\n\
     extern char __start_mysec[], __stop_mysec[];\n\
     int unused_thing(void){ return -1; }\n\
     int main(void){\n\
         const int *p = (const int *)__start_mysec;\n\
         const int *e = (const int *)__stop_mysec;\n\
         long n = e - p;\n\
         int sum = 0;\n\
         for (; p < e; p++) sum += *p;\n\
         printf(\"n=%ld sum=%d\\n\", n, sum);\n\
         return 0;\n\
     }\n";

/// `real_fn` is reached only through the function pointer `fp` in `.data`; it
/// must not be collected. `unused_thing` is unreferenced and disposable.
const FPTR_SRC: &[u8] = b"#include <stdio.h>\n\
     int real_fn(void){ return 99; }\n\
     static int (*fp)(void) = real_fn;\n\
     int unused_thing(void){ return -1; }\n\
     int main(void){ printf(\"fp=%d\\n\", fp()); return fp()==99?0:1; }\n";

/// A dependency that calls back into the program. It is built by the host
/// toolchain, so the reference to `hook` reaches xold as an undefined
/// `.dynsym` row of a `DT_NEEDED` input.
const DEP_SRC: &[u8] = b"extern int hook(void);\n\
     int use_hook(void) { return hook(); }\n";

/// The program the dependency calls back into. `hook` is the only definition
/// at issue: `main` calls `use_hook` in the dependency, so no relocation in
/// this link names `hook` at all and reachability alone collects it. The
/// executable exports it because a dependency references it, and an exported
/// definition has to survive the sweep or the loader binds the dependency's
/// call to nothing.
const HOOK_SRC: &[u8] = b"#include <stdio.h>\n\
     extern int use_hook(void);\n\
     int hook(void){ return 42; }\n\
     int main(void){ printf(\"hook=%d\\n\", use_hook()); return 0; }\n";

/// `clang -O0 -ffunction-sections -fdata-sections`: per-symbol sections are
/// the input gc operates on.
const CFLAGS: &[&str] = &["--target=x86_64-linux-gnu", "-c", "-O0"];

/// Compiles `src` into `obj` with the host clang. Returns `None` when clang is
/// unavailable so callers can skip gracefully.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    compile_with(src, obj, &[])
}

/// The same, passing `extra` through to clang. Every input is compiled to
/// per-symbol sections, which is what gives the collector something to take.
fn compile_with(src: &[u8], obj: &Path, extra: &[&str]) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args(CFLAGS)
        .args(["-ffunction-sections", "-fdata-sections"])
        .args(extra)
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// A fresh per-test working directory under the system temp dir.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xold_gc_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// The host toolchain needed for these tests, or `None` (with a note) when a
/// piece is missing.
struct Harness {
    crt1: PathBuf,
    crti: PathBuf,
    crtn: PathBuf,
    libc: PathBuf,
    interp: Vec<u8>,
}

impl Harness {
    /// Collects the harness, or returns `None` when a piece is missing.
    fn detect() -> Option<Self> {
        if which("clang").is_none() {
            eprintln!("skipping gc_sections tests: clang unavailable");
            return None;
        }
        let crt1 = crt_file("crt1.o")?;
        let crti = crt_file("crti.o")?;
        let crtn = crt_file("crtn.o")?;
        let libc = libc_so()?;
        let interp = interpreter()?;
        Some(Self {
            crt1,
            crti,
            crtn,
            libc,
            interp,
        })
    }

    /// Links `main_obj` plus the crt objects and libc into `prog` with xold,
    /// enabling `--gc-sections` when `gc` is set.
    fn link(&self, main_obj: &Path, prog: &Path, gc: bool) {
        self.link_with(main_obj, None, prog, gc);
    }

    /// The same, with an extra shared-object dependency when `dep` is given.
    fn link_with(
        &self,
        main_obj: &Path,
        dep: Option<&Path>,
        prog: &Path,
        gc: bool,
    ) {
        let mut inputs = vec![
            main_obj.to_path_buf(),
            self.crti.clone(),
            self.crt1.clone(),
            self.crtn.clone(),
            self.libc.clone(),
        ];
        inputs.extend(dep.map(Path::to_path_buf));
        link_dyn_exec(
            &inputs,
            prog,
            b"_start",
            &self.interp,
            gc,
            IcfMode::None,
            false,
        )
        .expect("xold link must succeed");
    }
}

/// Builds `DEP_SRC` into `libgchook.so` with the host toolchain, giving it a
/// SONAME so the `DT_NEEDED` name is one `LD_LIBRARY_PATH` can resolve. The
/// callback is left undefined in the library, which is the whole point, so the
/// host linker is told to allow it.
fn build_dependency(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let obj = dir.join("gcdep.o");
    compile_with(DEP_SRC, &obj, &["-fPIC"])?;
    let so = dir.join("libgchook.so");
    Command::new(clang)
        .args(["-shared", "-fPIC", "-Wl,-soname,libgchook.so"])
        .arg("-Wl,--allow-shlib-undefined")
        .arg(&obj)
        .arg("-o")
        .arg(&so)
        .status()
        .ok()?
        .success()
        .then_some(so)
}

/// The on-disk size of the `.text` output section, read via xold's own reader.
/// Returns 0 when the section is absent.
fn text_section_size(bytes: &[u8]) -> u64 {
    let obj = ObjectFile::parse(bytes).expect("valid ELF output");
    for shdr in obj.sections() {
        let name = obj.section_name(shdr);
        if name == b".text" {
            return shdr.sh_size.get();
        }
    }
    0
}

/// Whether the `.text` output section contains `needle`.
fn text_section_has(bytes: &[u8], needle: &[u8]) -> bool {
    let obj = ObjectFile::parse(bytes).expect("valid ELF output");
    for shdr in obj.sections() {
        if obj.section_name(shdr) != b".text" {
            continue;
        }
        let Ok(data) = obj.section_data(shdr) else {
            return false;
        };
        return data.windows(needle.len()).any(|w| w == needle);
    }
    false
}

/// Runs `prog` and returns its stdout and exit status.
fn run(prog: &Path) -> (Vec<u8>, std::process::ExitStatus) {
    let out = Command::new(prog)
        .output()
        .expect("linked program must be runnable");
    (out.stdout, out.status)
}

/// The same, with `dir` on the loader's search path so a `DT_NEEDED`
/// dependency built beside the program is found.
fn run_with_library_path(
    prog: &Path,
    dir: &Path,
) -> (Vec<u8>, std::process::ExitStatus) {
    let out = Command::new(prog)
        .env("LD_LIBRARY_PATH", dir)
        .output()
        .expect("linked program must be runnable");
    (out.stdout, out.status)
}

/// `unused_fn`'s `return 42` compiles to `mov eax, 42` = `B8 2A 00 00 00`,
/// a byte signature unique within this program's `.text`.
const UNUSED_FN_SIGNATURE: &[u8] = &[0xb8, 0x2a, 0x00, 0x00, 0x00];

/// The repro: with `--gc-sections` the unused function's bytes are gone from
/// `.text`, the `.text` and file sizes shrink, and the program still prints
/// `used=7` and exits 0. Without the flag the bytes are present.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn gc_sections_drops_unused_function_and_runs() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("repro");
    let main_o = dir.join("gc.o");
    compile(GC_SRC, &main_o).expect("host clang compiles gc.c");

    let prog_gc = dir.join("gc_prog");
    let prog_no = dir.join("nogc_prog");
    h.link(&main_o, &prog_gc, true);
    h.link(&main_o, &prog_no, false);

    let bytes_gc = fs::read(&prog_gc).expect("read gc output");
    let bytes_no = fs::read(&prog_no).expect("read no-gc output");

    // The unused function's signature byte sequence must survive in the
    // default link and vanish under `--gc-sections`.
    assert!(
        text_section_has(&bytes_no, UNUSED_FN_SIGNATURE),
        "default link must keep unused_fn's bytes in .text"
    );
    assert!(
        !text_section_has(&bytes_gc, UNUSED_FN_SIGNATURE),
        "--gc-sections must drop unused_fn's bytes from .text"
    );

    // GC must shrink both the .text section and the whole image.
    let text_gc = text_section_size(&bytes_gc);
    let text_no = text_section_size(&bytes_no);
    assert!(
        text_gc < text_no,
        ".text must shrink under gc: got gc={text_gc}, no-gc={text_no}"
    );
    assert!(
        bytes_gc.len() < bytes_no.len(),
        "file must shrink under gc: got gc={}, no-gc={}",
        bytes_gc.len(),
        bytes_no.len()
    );

    // Both binaries must run identically.
    let (out_gc, status_gc) = run(&prog_gc);
    let (out_no, status_no) = run(&prog_no);
    assert_eq!(out_gc, out_no, "gc and no-gc stdout must match");
    assert_eq!(out_gc, b"used=7\n", "program output");
    assert!(status_gc.success(), "gc run exits 0");
    assert!(status_no.success(), "no-gc run exits 0");
}

/// Linking the same input twice without `--gc-sections` produces byte-equal
/// output, proving the default path is deterministic and unaffected by the gc
/// plumbing (the `gc: bool` flag is `false` here).
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn default_link_is_byte_identical_across_runs() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("ident");
    let main_o = dir.join("ident.o");
    compile(GC_SRC, &main_o).expect("host clang compiles gc.c");

    let a = dir.join("a");
    let b = dir.join("b");
    h.link(&main_o, &a, false);
    h.link(&main_o, &b, false);
    let bytes_a = fs::read(&a).expect("read a");
    let bytes_b = fs::read(&b).expect("read b");
    assert_eq!(
        bytes_a, bytes_b,
        "two default links of the same input must be byte-identical"
    );
}

/// A constructor referenced from `.init_array` must survive `--gc-sections`
/// and run before `main` (the `.init_array` section is a root, and its
/// relocation marks the constructor's section).
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn constructor_survives_gc() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("ctor");
    let main_o = dir.join("ctor.o");
    compile(CTOR_SRC, &main_o).expect("host clang compiles ctor.c");
    let prog = dir.join("ctor_prog");
    h.link(&main_o, &prog, true);

    let (out, status) = run(&prog);
    assert_eq!(out, b"buf=ok\n", "constructor must run under gc");
    assert!(status.success(), "gc ctor run exits 0");
}

/// A section reached only through `__start_`/`__stop_` must survive
/// `--gc-sections`. Nothing relocates into it, so reachability alone would
/// collect it and leave the two bounds describing an empty range; the linker
/// roots a bounded section instead, as lld does under `-z nostart-stop-gc`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn start_stop_section_survives_gc() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("startstop");
    let main_o = dir.join("startstop.o");
    compile(START_STOP_SRC, &main_o).expect("host clang compiles startstop.c");
    let prog_gc = dir.join("startstop_gc");
    let prog_no = dir.join("startstop_nogc");
    h.link(&main_o, &prog_gc, true);
    h.link(&main_o, &prog_no, false);

    // Both links must see the same two entries: gc may not narrow the range.
    let (out_gc, status_gc) = run(&prog_gc);
    let (out_no, status_no) = run(&prog_no);
    assert_eq!(
        out_gc, b"n=2 sum=9\n",
        "--gc-sections must keep a section its bounds are the only way in to"
    );
    assert_eq!(out_no, out_gc, "gc must not change what the walk finds");
    assert!(status_gc.success(), "gc start/stop run exits 0");
    assert!(status_no.success(), "no-gc start/stop run exits 0");

    // The unreferenced function is still collected, so the link really did
    // run the collector rather than keeping everything.
    let bytes_gc = fs::read(&prog_gc).expect("read gc output");
    let bytes_no = fs::read(&prog_no).expect("read no-gc output");
    assert!(
        text_section_size(&bytes_gc) < text_section_size(&bytes_no),
        ".text must still shrink under gc"
    );
}

/// A definition a `DT_NEEDED` dependency binds to must survive
/// `--gc-sections`.
///
/// Nothing in this link relocates to `hook`: the program calls into the
/// library, and only the library calls back. The executable exports the
/// definition because the dependency references it, so the collector has to
/// root what the image exports or the sweep takes the section and leaves a
/// `.dynsym` row the loader cannot bind -- which fails at run time, not at
/// link time. lld roots exactly the exported set in `MarkLive::run`.
///
/// Asserted by running the program: a structural check on the `.dynsym` row
/// would pass over a row that names collected bytes.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_definition_a_dependency_binds_to_survives_gc() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("depcall");
    let Some(dep) = build_dependency(&dir) else {
        eprintln!("skipping gc dependency test: host cannot build the library");
        return;
    };
    let main_o = dir.join("hook.o");
    compile(HOOK_SRC, &main_o).expect("host clang compiles hook.c");

    let prog_gc = dir.join("hook_gc");
    let prog_no = dir.join("hook_nogc");
    h.link_with(&main_o, Some(&dep), &prog_gc, true);
    h.link_with(&main_o, Some(&dep), &prog_no, false);

    let (out_gc, status_gc) = run_with_library_path(&prog_gc, &dir);
    let (out_no, status_no) = run_with_library_path(&prog_no, &dir);
    assert_eq!(
        out_gc, b"hook=42\n",
        "the dependency's callback must reach the program's definition under \
         --gc-sections"
    );
    assert_eq!(out_no, out_gc, "gc must not change what the callback finds");
    assert!(status_gc.success(), "gc callback run exits 0");
    assert!(status_no.success(), "no-gc callback run exits 0");

    // The collector really ran: `unused_fn` from the other fixture is not in
    // this program, so size is the check that gc had an effect here.
    let bytes_gc = fs::read(&prog_gc).expect("read gc output");
    let bytes_no = fs::read(&prog_no).expect("read no-gc output");
    assert!(
        bytes_gc.len() < bytes_no.len(),
        "the gc link must still be smaller: got gc={}, no-gc={}",
        bytes_gc.len(),
        bytes_no.len()
    );
}

/// A function reached only through a function pointer stored in `.data` must
/// survive `--gc-sections`: the live `.data` section's relocation marks the
/// target. The program runs and prints `fp=99`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn function_pointer_in_data_survives_gc() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("fptr");
    let main_o = dir.join("fptr.o");
    compile(FPTR_SRC, &main_o).expect("host clang compiles fptr.c");
    let prog = dir.join("fptr_prog");
    h.link(&main_o, &prog, true);

    let (out, status) = run(&prog);
    assert_eq!(out, b"fp=99\n", "function-pointer target must survive gc");
    assert!(status.success(), "gc fptr run exits 0");
}
