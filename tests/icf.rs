//! End-to-end tests for ICF: identical code folding.
//!
//! With `--icf=all` two byte-identical functions (same opcodes, same relocation
//! signatures) collapse onto one survivor: both symbols resolve to the same
//! `st_value`, the duplicate's bytes leave `.text`, and the program still runs
//! correctly. `--icf=safe` does the same except for functions whose address is
//! observable (taken by an absolute or GOT relocation), which it leaves alone.
//! Off by default, the output is byte-identical to a link with no ICF pass.
//!
//! Covered guarantees:
//!
//! - **Functional**: `f1`/`f2` share `st_value` with `--icf=all` and differ
//!   without it; the program prints `6 6` and exits 0 either way.
//! - **Byte-identical default**: two default links of the same input produce
//!   byte-equal output (ICF plumbing is inert when the flag is absent).
//! - **Section order**: two identical functions calling a common target fold
//!   whether the target is emitted between them or before them.
//! - **Undefined targets**: two identical functions calling the same import
//!   fold, under `--icf=all` and `--icf=safe` alike.
//! - **No false fold**: two functions differing by a single opcode are not
//!   folded under `--icf=all`, and neither are two byte-identical functions
//!   that call different symbols.
//! - **Preemption**: in a shared object, two callers of different preemptible
//!   symbols stay apart, though the same pair folds in an executable.
//! - **Safe mode**: a function whose address is stored in a global is not
//!   folded under `--icf=safe` (its address is observable) but is folded under
//!   `--icf=all`.
//! - **Size**: the duplicate's bytes leave `.text`, so `.text` shrinks.
//!
//! Gated on `clang`, `gcc` (to locate the crt objects) and the system `ld.so`;
//! if absent the tests print a note and return, so the build never fails over a
//! missing toolchain.

#![allow(clippy::similar_names)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{
    elf::ObjectFile,
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared, link_to},
};

mod common;

/// Two byte-identical functions; `main` calls both. The ICF repro.
const ICF_SRC: &[u8] = b"#include <stdio.h>\n\
     int f1(int x){ return x + 1; }\n\
     int f2(int x){ return x + 1; }\n\
     int main(void){ printf(\"%d %d\\n\", f1(5), f2(5)); return 0; }\n";

/// `f1` and `f2` differ by one opcode (`+1` vs `+2`); they must never fold.
const DIFF_SRC: &[u8] = b"#include <stdio.h>\n\
     int f1(int x){ return x + 1; }\n\
     int f2(int x){ return x + 2; }\n\
     int main(void){ printf(\"%d %d\\n\", f1(5), f2(5)); return 0; }\n";

/// `f1` and `f2` are identical, but `f1`'s address is stored in the global
/// `gp`; under `--icf=safe` `f1` must stay unique (its address is observable),
/// while `--icf=all` still folds the pair.
const ADDR_SRC: &[u8] = b"#include <stdio.h>\n\
     int f1(int x){ return x + 1; }\n\
     int f2(int x){ return x + 1; }\n\
     int (*gp)(int) = f1;\n\
     int main(void){ printf(\"%d %d %d\\n\", f1(5), f2(5), gp(5)); return 0; }\n";

/// `g1` and `g2` are identical and both call `mid`, which the compiler emits
/// *between* them because it is defined after both. Where a common target sits
/// in the section order is not a property of the sections that reference it, so
/// the pair must still fold.
const MID_LAST_SRC: &[u8] = b"#include <stdio.h>\n\
     int mid(int);\n\
     int g1(int x){ return mid(x) + 1; }\n\
     int g2(int x){ return mid(x) + 1; }\n\
     int mid(int x){ return x * 2; }\n\
     int main(void){ printf(\"%d %d %d\\n\", g1(5), g2(5), mid(5)); return 0; }\n";

/// The same program with `mid` defined first, so it is emitted before both of
/// its callers. Must fold exactly as [`MID_LAST_SRC`] does.
const MID_FIRST_SRC: &[u8] = b"#include <stdio.h>\n\
     int mid(int x){ return x * 2; }\n\
     int g1(int x){ return mid(x) + 1; }\n\
     int g2(int x){ return mid(x) + 1; }\n\
     int main(void){ printf(\"%d %d %d\\n\", g1(5), g2(5), mid(5)); return 0; }\n";

/// `h1` and `h2` are identical and both call `atoi`, which is undefined in the
/// object and comes from libc. A relocation against an import must not keep the
/// pair apart.
const UNDEF_SRC: &[u8] = b"#include <stdio.h>\n\
     extern int atoi(const char *);\n\
     int h1(const char *s){ return atoi(s) + 1; }\n\
     int h2(const char *s){ return atoi(s) + 1; }\n\
     int main(void){ printf(\"%d %d\\n\", h1(\"5\"), h2(\"5\")); return 0; }\n";

/// Two pairs that are byte-identical and differ only in the symbol they call:
/// `n1`/`n2` call the non-identical `t1`/`t2`, `e1`/`e2` call two different
/// imports. Only the relocation targets tell either pair apart, so neither may
/// fold.
const CALLEE_SRC: &[u8] = b"#include <stdio.h>\n\
     extern int atoi(const char *);\n\
     extern long atol(const char *);\n\
     int t1(int x){ return x * 2; }\n\
     int t2(int x){ return x * 3; }\n\
     int n1(int x){ return t1(x) + 1; }\n\
     int n2(int x){ return t2(x) + 1; }\n\
     int e1(const char *s){ return atoi(s) + 1; }\n\
     int e2(const char *s){ return (int)atol(s) + 1; }\n\
     int main(void){\n\
       printf(\"%d %d %d %d\\n\", n1(5), n2(5), e1(\"7\"), e2(\"7\"));\n\
       return 0; }\n";

/// `clang -O0 -ffunction-sections -fdata-sections`: per-symbol sections are the
/// input ICF operates on.
const CFLAGS: &[&str] = &["--target=x86_64-linux-gnu", "-c", "-O0"];

/// Compiles `src` into `obj` with the host clang, or returns `None` when clang
/// is unavailable so callers can skip gracefully.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args(CFLAGS)
        .args(["-ffunction-sections", "-fdata-sections"])
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
///
/// The name carries the process id: two copies of this binary running at once
/// would otherwise share the path, and the `remove_dir_all` below would delete
/// files the other is still compiling into.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_icf_{prefix}_{}", std::process::id()));
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
            eprintln!("skipping icf tests: clang unavailable");
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
    /// selecting the ICF mode.
    fn link(&self, main_obj: &Path, prog: &Path, icf: IcfMode) {
        link_dyn_exec(
            &[
                main_obj.to_path_buf(),
                self.crti.clone(),
                self.crt1.clone(),
                self.crtn.clone(),
                self.libc.clone(),
            ],
            prog,
            b"_start",
            &self.interp,
            false,
            icf,
            false,
        )
        .expect("xold link must succeed");
    }
}

/// The `st_value` of a global symbol in the output symtab, or `None`.
fn symbol_value(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).expect("valid ELF output");
    let symtab = obj.symbol_table().ok().flatten()?;
    for sym in symtab.iter() {
        if sym.bind() == xold::elf::constants::STB_GLOBAL
            && symtab.name(sym) == name
        {
            return Some(sym.st_value.get());
        }
    }
    None
}

/// The on-disk size of the `.text` output section, read via xold's own reader.
/// Returns 0 when the section is absent.
fn text_section_size(bytes: &[u8]) -> u64 {
    let obj = ObjectFile::parse(bytes).expect("valid ELF output");
    for shdr in obj.sections() {
        if obj.section_name(shdr) == b".text" {
            return shdr.sh_size.get();
        }
    }
    0
}

/// Runs `prog` and returns its stdout and exit status.
fn run(prog: &Path) -> (Vec<u8>, std::process::ExitStatus) {
    let out = Command::new(prog)
        .output()
        .expect("linked program must be runnable");
    (out.stdout, out.status)
}

/// Compiles `src` into `dir/name.o`, links it into `dir/name` under `icf` and
/// runs it. Returns the linked image and the program's stdout; the run must
/// exit 0, since a fold that changes behaviour is worse than no fold at all.
fn build_and_run(
    h: &Harness,
    dir: &Path,
    name: &str,
    src: &[u8],
    icf: IcfMode,
) -> (Vec<u8>, Vec<u8>) {
    let obj = dir.join(format!("{name}.o"));
    compile(src, &obj).expect("host clang compiles the source");
    let prog = dir.join(name);
    h.link(&obj, &prog, icf);
    let bytes = fs::read(&prog).expect("read linked output");
    let (out, status) = run(&prog);
    assert!(status.success(), "{name} run exits 0");
    (bytes, out)
}

/// The repro: with `--icf=all` the two identical functions share `st_value`
/// and the program prints `6 6`; without ICF their values differ and the
/// program still prints `6 6`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn icf_all_folds_identical_functions() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("repro");
    let main_o = dir.join("icf.o");
    compile(ICF_SRC, &main_o).expect("host clang compiles icf.c");

    let prog_all = dir.join("icf_all");
    let prog_none = dir.join("icf_none");
    h.link(&main_o, &prog_all, IcfMode::All);
    h.link(&main_o, &prog_none, IcfMode::None);

    let bytes_all = fs::read(&prog_all).expect("read icf output");
    let bytes_none = fs::read(&prog_none).expect("read none output");

    let f1_all = symbol_value(&bytes_all, b"f1").expect("f1 present");
    let f2_all = symbol_value(&bytes_all, b"f2").expect("f2 present");
    let f1_none = symbol_value(&bytes_none, b"f1").expect("f1 present");
    let f2_none = symbol_value(&bytes_none, b"f2").expect("f2 present");

    assert_eq!(
        f1_all, f2_all,
        "--icf=all must fold f1 and f2 to one address"
    );
    assert_ne!(
        f1_none, f2_none,
        "default link must keep f1 and f2 at distinct addresses"
    );

    let (out_all, status_all) = run(&prog_all);
    let (out_none, status_none) = run(&prog_none);
    assert_eq!(out_all, b"6 6\n", "icf=all program output");
    assert_eq!(out_none, b"6 6\n", "default program output");
    assert!(status_all.success(), "icf=all run exits 0");
    assert!(status_none.success(), "default run exits 0");
}

/// Two default links of the same input are byte-equal: the ICF plumbing is
/// inert when the flag is absent.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn default_link_is_byte_identical_across_runs() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("ident");
    let main_o = dir.join("ident.o");
    compile(ICF_SRC, &main_o).expect("host clang compiles icf.c");
    let a = dir.join("a");
    let b = dir.join("b");
    h.link(&main_o, &a, IcfMode::None);
    h.link(&main_o, &b, IcfMode::None);
    let bytes_a = fs::read(&a).expect("read a");
    let bytes_b = fs::read(&b).expect("read b");
    assert_eq!(
        bytes_a, bytes_b,
        "two default links of the same input must be byte-identical"
    );
}

/// Two functions differing by a single opcode are never folded, even under
/// `--icf=all`. The program prints `6 7` either way.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn no_false_fold_on_differing_code() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("diff");
    let main_o = dir.join("diff.o");
    compile(DIFF_SRC, &main_o).expect("host clang compiles diff.c");
    let prog = dir.join("diff_prog");
    h.link(&main_o, &prog, IcfMode::All);
    let bytes = fs::read(&prog).expect("read diff output");
    let f1 = symbol_value(&bytes, b"f1").expect("f1 present");
    let f2 = symbol_value(&bytes, b"f2").expect("f2 present");
    assert_ne!(
        f1, f2,
        "functions differing by an opcode must not fold under --icf=all"
    );
    let (out, status) = run(&prog);
    assert_eq!(out, b"6 7\n", "differing functions keep their semantics");
    assert!(status.success(), "diff run exits 0");
}

/// A function whose address is taken (stored in a global) is kept unique under
/// `--icf=safe` (its address is observable) but is folded under `--icf=all`.
/// Both programs print `6 6 6`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn safe_mode_keeps_address_taken_unfolded() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("addr");
    let main_o = dir.join("addr.o");
    compile(ADDR_SRC, &main_o).expect("host clang compiles addr.c");

    let prog_safe = dir.join("addr_safe");
    let prog_all = dir.join("addr_all");
    h.link(&main_o, &prog_safe, IcfMode::Safe);
    h.link(&main_o, &prog_all, IcfMode::All);

    let bytes_safe = fs::read(&prog_safe).expect("read safe output");
    let bytes_all = fs::read(&prog_all).expect("read all output");

    let f1_safe = symbol_value(&bytes_safe, b"f1").expect("f1 present");
    let f2_safe = symbol_value(&bytes_safe, b"f2").expect("f2 present");
    assert_ne!(
        f1_safe, f2_safe,
        "--icf=safe must not fold a function whose address is taken",
    );

    let f1_all = symbol_value(&bytes_all, b"f1").expect("f1 present");
    let f2_all = symbol_value(&bytes_all, b"f2").expect("f2 present");
    assert_eq!(
        f1_all, f2_all,
        "--icf=all folds identical functions regardless of address-taking",
    );

    let (out_safe, status_safe) = run(&prog_safe);
    let (out_all, status_all) = run(&prog_all);
    assert_eq!(out_safe, b"6 6 6\n", "safe program output");
    assert_eq!(out_all, b"6 6 6\n", "all program output");
    assert!(status_safe.success(), "safe run exits 0");
    assert!(status_all.success(), "all run exits 0");
}

/// Two identical functions calling a common target fold whichever side of them
/// the compiler emits that target on: `mid` sits between `g1` and `g2` in one
/// object and before both in the other. Both programs print `11 11 10`, so the
/// survivor really is called through both names.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn icf_folds_whatever_the_target_section_order_is() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("order");
    for (name, src) in
        [("mid_last", MID_LAST_SRC), ("mid_first", MID_FIRST_SRC)]
    {
        let (bytes, out) = build_and_run(&h, &dir, name, src, IcfMode::All);
        let g1 = symbol_value(&bytes, b"g1").expect("g1 present");
        let g2 = symbol_value(&bytes, b"g2").expect("g2 present");
        let mid = symbol_value(&bytes, b"mid").expect("mid present");
        assert_eq!(g1, g2, "--icf=all must fold g1 and g2 in {name}");
        assert_ne!(g1, mid, "the callee must not fold onto its callers");
        assert_eq!(out, b"11 11 10\n", "{name} program output");
    }
}

/// Two identical functions calling the same undefined symbol fold: the pair
/// agrees on the import it names, which is all a relocation against an import
/// can be asked to agree on. `--icf=safe` folds them too, their address being
/// taken by nothing. Both programs print `6 6`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn icf_folds_calls_to_an_undefined_symbol() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("undef");
    for (name, mode) in
        [("undef_all", IcfMode::All), ("undef_safe", IcfMode::Safe)]
    {
        let (bytes, out) = build_and_run(&h, &dir, name, UNDEF_SRC, mode);
        let h1 = symbol_value(&bytes, b"h1").expect("h1 present");
        let h2 = symbol_value(&bytes, b"h2").expect("h2 present");
        assert_eq!(h1, h2, "{name} must fold two calls to the same import");
        assert_eq!(out, b"6 6\n", "{name} program output");
    }
}

/// Byte-identical functions that differ only in the symbol they call are never
/// folded: `n1`/`n2` call two different local functions, `e1`/`e2` two
/// different imports. The program prints `11 16 8 8`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn no_false_fold_on_differing_call_targets() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("callee");
    let (bytes, out) =
        build_and_run(&h, &dir, "callee", CALLEE_SRC, IcfMode::All);
    let n1 = symbol_value(&bytes, b"n1").expect("n1 present");
    let n2 = symbol_value(&bytes, b"n2").expect("n2 present");
    let e1 = symbol_value(&bytes, b"e1").expect("e1 present");
    let e2 = symbol_value(&bytes, b"e2").expect("e2 present");
    assert_ne!(n1, n2, "callers of two different functions must not fold");
    assert_ne!(e1, e2, "callers of two different imports must not fold");
    assert_eq!(out, b"11 16 8 8\n", "callee program output");
}

/// With `--icf=all` the duplicate's bytes leave `.text`, so the `.text` output
/// section is strictly smaller than in the default link.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn icf_shrinks_text_section() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("size");
    let main_o = dir.join("size.o");
    compile(ICF_SRC, &main_o).expect("host clang compiles icf.c");
    let prog_all = dir.join("size_all");
    let prog_none = dir.join("size_none");
    h.link(&main_o, &prog_all, IcfMode::All);
    h.link(&main_o, &prog_none, IcfMode::None);
    let bytes_all = fs::read(&prog_all).expect("read all output");
    let bytes_none = fs::read(&prog_none).expect("read none output");
    let text_all = text_section_size(&bytes_all);
    let text_none = text_section_size(&bytes_none);
    assert!(
        text_all < text_none,
        ".text must shrink under --icf=all: got all={text_all}, none={text_none}"
    );
}

/// `ca` and `cb` are byte-identical and differ only in which of the equally
/// byte-identical `a`/`b` they call. `_start` gives the static link an entry.
///
/// In a shared object `a` and `b` have default visibility, so the executable
/// that loads it may define either one. Folding `ca` onto `cb` would leave one
/// address for two calls, and interposing only `a` could then no longer be
/// expressed. In an executable nothing can be interposed and the pair folds.
const PREEMPT_SRC: &[u8] = b"int a(int x){ return x + 1; }\n\
     int b(int x){ return x + 1; }\n\
     int ca(int x){ return a(x); }\n\
     int cb(int x){ return b(x); }\n\
     void _start(void){ }\n";

/// Compiles `src` `-fPIC` with per-function sections, so both link flavours
/// read the same object.
fn compile_pic(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args(CFLAGS)
        .args(["-fPIC", "-ffunction-sections", "-fdata-sections"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// Two sections whose calls name different preemptible symbols must not fold,
/// though the same pair folds when nothing can be interposed.
///
/// The leaf pair `a`/`b` folds in both links: it carries no relocation, so
/// there is no symbol to tell the two apart. That is what makes this a test of
/// the call target rather than of folding being off.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_preemptible_call_target_keeps_two_callers_apart() {
    if which("clang").is_none() {
        eprintln!("skipping preemptible-fold test: clang unavailable");
        return;
    }
    let dir = workdir("preempt");
    let obj = dir.join("preempt.o");
    let Some(()) = compile_pic(PREEMPT_SRC, &obj) else {
        eprintln!("skipping preemptible-fold test: host clang unavailable");
        return;
    };

    let so = dir.join("libpreempt.so");
    link_shared(
        std::slice::from_ref(&obj),
        &so,
        None,
        false,
        IcfMode::All,
        false,
    )
    .expect("xold -shared --icf=all link must succeed");
    let shared_bytes = fs::read(&so).expect("read shared output");
    assert_eq!(
        symbol_value(&shared_bytes, b"a"),
        symbol_value(&shared_bytes, b"b"),
        "two identical leaf functions still fold in a shared object"
    );
    assert_ne!(
        symbol_value(&shared_bytes, b"ca"),
        symbol_value(&shared_bytes, b"cb"),
        "callers of two preemptible symbols must stay apart"
    );

    let exe = dir.join("preempt_exe");
    link_to(
        std::slice::from_ref(&obj),
        &exe,
        b"_start",
        false,
        IcfMode::All,
        false,
    )
    .expect("xold --icf=all link must succeed");
    let static_bytes = fs::read(&exe).expect("read static output");
    assert_eq!(
        symbol_value(&static_bytes, b"ca"),
        symbol_value(&static_bytes, b"cb"),
        "nothing can interpose an executable's definitions, so the pair folds"
    );
    assert!(
        symbol_value(&static_bytes, b"ca").is_some(),
        "the callers must be in the output symbol table at all"
    );
}

/// Two identical functions, one of which declares a stricter alignment than
/// the other. Three padding functions ahead of them put the representative at
/// an address that is 16-aligned but not 64-aligned, which is what makes the
/// difference between honouring the declaration and dropping it visible.
const ALIGN_SRC: &[u8] = b"#include <stdio.h>\n\
     int pad1(int x){ return x + 1; }\n\
     int pad2(int x){ return x + 2; }\n\
     int plain(int x){ return x * 3 + 7; }\n\
     __attribute__((aligned(64))) int wide(int x){ return x * 3 + 7; }\n\
     int main(void){\n\
       printf(\"%d %d %d %d\\n\", pad1(1), pad2(1), plain(1), wide(1));\n\
       return 0; }\n";

/// A folded section's alignment survives the fold: the representative inherits
/// the strictest requirement in its group, as lld's `InputSection::replace`
/// does. Without that, `wide` is given `plain`'s 16-byte-aligned address and
/// its `__attribute__((aligned(64)))` means nothing.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn folding_keeps_the_strictest_alignment() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("align");
    let (bytes, _) = build_and_run(&h, &dir, "align", ALIGN_SRC, IcfMode::All);
    let plain = symbol_value(&bytes, b"plain").expect("plain is exported");
    let wide = symbol_value(&bytes, b"wide").expect("wide is exported");
    assert_eq!(plain, wide, "the identical pair folds");
    assert_eq!(
        wide % 64,
        0,
        "the surviving address must satisfy the strictest alignment \
         in the group; got {wide:#x}"
    );
}

/// Two functions with identical bodies whose sections differ only in
/// `sh_flags`: `r1` carries `SHF_GNU_RETAIN`, `r2` does not.
const FLAGS_SRC: &[u8] = b"#include <stdio.h>\n\
     __attribute__((retain)) int r1(int x){ return x * 5 + 9; }\n\
     int r2(int x){ return x * 5 + 9; }\n\
     int main(void){ printf(\"%d %d\\n\", r1(1), r2(1)); return 0; }\n";

/// Sections that disagree on `sh_flags` are not folded, even when every byte
/// and every relocation matches. lld's `equalsConstant` opens with the same
/// test (`a->flags != b->flags`): the flags are what an input declares the
/// section to be, and two sections declared differently were not declared to
/// be interchangeable.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn no_fold_across_differing_section_flags() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("flags");
    let (bytes, out) =
        build_and_run(&h, &dir, "flags", FLAGS_SRC, IcfMode::All);
    assert_eq!(out, b"14 14\n", "both functions still compute their result");
    let r1 = symbol_value(&bytes, b"r1").expect("r1 is exported");
    let r2 = symbol_value(&bytes, b"r2").expect("r2 is exported");
    assert_ne!(
        r1, r2,
        "a retained section and a plain one are not the same section"
    );
}
