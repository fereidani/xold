//! End-to-end tests for ELF COMDAT / section-group deduplication.
//!
//! A multi-TU C++ program where every translation unit instantiates the same
//! template (or defines the same `inline` function) carries one `SHT_GROUP`
//! per instantiation, signed by the mangled symbol name. Without dedup xold
//! keeps every copy, bloating the output ~2.5x over the system linker. With
//! dedup only the first group per signature survives: duplicate template code
//! is folded to one copy while the program still links and runs identically.
//!
//! These tests cover the four guarantees:
//!
//! - **Size**: the 20-TU `std::vector` repro shrinks from ~73k (no dedup) to
//!   within 1.5x of the system linker's output.
//! - **Runs**: the deduped binary still computes the right answer (`f1()==3`,
//!   exit 0).
//! - **Inline**: a multi-TU inline-function program dedupes to one copy and
//!   runs correctly.
//! - **Byte-identical default**: a non-COMDAT link is unchanged (two runs
//!   produce byte-equal output), proving the dedup plumbing is invisible when
//!   no groups are present.
//!
//! Gated on `clang++`, `gcc` (to locate the crt objects), `libc.so.6`,
//! `libstdc++.so`, and the system `ld.so`; if absent the tests print a note
//! and return, so the build never fails over a missing toolchain.

#![allow(clippy::similar_names)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{
    crt_file, interpreter, libc_so, libgcc_s_so, libstdcxx_so, which,
};
use xold::{icf::IcfMode, linker::link_dyn_exec};

mod common;

/// One TU of the 20-TU repro: each instantiates `std::vector<int>` with a
/// distinct pair of values and sums them. All 20 share the same template
/// instantiations, producing identical `SHT_GROUP` sections.
const TU_SRC: &[u8] = b"#include <vector>\n\
     int f{N}(){std::vector<int>v{{{N},{M}}};int s=0;for(int x:v)s+=x;return s;}\n";

/// The main TU: calls `f1` and returns non-zero if the result is wrong.
const MAIN_SRC: &[u8] = b"extern int f1();\n\
     int main(){return f1()!=3;}\n";

/// An inline function in a header, included by several TUs. Each inclusion
/// produces a COMDAT group for `inline_add`; the linker must keep one copy.
const INLINE_HDR: &[u8] = b"#pragma once\n\
     inline int inline_add(int a, int b){ return a + b; }\n";

/// One TU using the inline function.
const INLINE_TU_SRC: &[u8] = b"#include \"hdr.h\"\n\
     int caller_{N}(){ return inline_add({N}, {N}); }\n";

/// The main TU for the inline test.
const INLINE_MAIN_SRC: &[u8] = b"#include \"hdr.h\"\n\
     extern int caller_1();\n\
     int main(){ return inline_add(1, 1) + caller_1() != 4; }\n";

/// One member of a section group, with `{FLAG}` selecting the group's flag
/// word and `{VALUE}` its payload.
///
/// `.section name,"awG",@progbits,signature` builds a group with a flag word
/// of zero; appending `,comdat` sets `GRP_COMDAT`. That is the whole
/// difference between the two variants, so the fixture needs no post-processing
/// to produce either -- the assembler writes the byte under test.
///
/// The member is a section named by a C identifier, so the program can walk it
/// between `__start_grpsec` and `__stop_grpsec` and count what survived. A
/// signature symbol is the usual way in, but every copy defines the same one,
/// so only the surviving copy would ever be reachable by name.
const GROUP_MEMBER_SRC: &[u8] =
    b"        .section grpsec,\"awG\",@progbits,tfn_sig{FLAG}\n\
        .p2align 3\n        .quad {VALUE}\n";

/// Walks the group's section and reports how many members reached the image
/// and what they hold. One COMDAT copy survives; two plain-group copies both
/// do.
const GROUP_MAIN_SRC: &[u8] = b"#include <cstdio>\n\
     extern \"C\" unsigned long __start_grpsec[], __stop_grpsec[];\n\
     int main(){\n\
         unsigned long n = 0, sum = 0;\n\
         for (const unsigned long *p = __start_grpsec; p < __stop_grpsec; p++)\
             { n++; sum += *p; }\n\
         printf(\"n=%lu sum=%lu\\n\", n, sum);\n\
         return 0;\n\
     }\n";

/// Compiles `src` (a C++ source) into `obj` with the host clang++. Returns
/// `None` when clang++ is unavailable so callers can skip gracefully. When
/// `hdr` is given it is written next to the source so `#include "hdr.h"`
/// resolves.
fn compile_cpp(
    src: &[u8],
    obj: &Path,
    hdr: Option<(&str, &[u8])>,
) -> Option<()> {
    let clang = which("clang++")?;
    let src_path = obj.with_extension("cpp");
    fs::write(&src_path, src).expect("write source");
    if let Some((name, bytes)) = hdr {
        fs::write(src_path.with_file_name(name), bytes).expect("write header");
    }
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c", "-O0"])
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
    let dir = std::env::temp_dir().join(format!("xold_comdat_{prefix}"));
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
    libstdcxx: PathBuf,
    libgcc_s: PathBuf,
    interp: Vec<u8>,
}

impl Harness {
    /// Collects the harness, or returns `None` when a piece is missing.
    fn detect() -> Option<Self> {
        if which("clang++").is_none() {
            eprintln!("skipping comdat tests: clang++ unavailable");
            return None;
        }
        let crt1 = crt_file("crt1.o")?;
        let crti = crt_file("crti.o")?;
        let crtn = crt_file("crtn.o")?;
        let libc = libc_so()?;
        let libstdcxx = libstdcxx_so()?;
        let libgcc_s = libgcc_s_so()?;
        let interp = interpreter()?;
        Some(Self {
            crt1,
            crti,
            crtn,
            libc,
            libstdcxx,
            libgcc_s,
            interp,
        })
    }

    /// Links `objs` plus the crt objects, libc and libstdc++ into `prog`.
    fn link(&self, objs: &[PathBuf], prog: &Path) {
        let mut paths: Vec<PathBuf> = objs.to_vec();
        paths.push(self.crti.clone());
        paths.push(self.crt1.clone());
        paths.push(self.crtn.clone());
        paths.push(self.libc.clone());
        paths.push(self.libstdcxx.clone());
        // `_Unwind_Resume` lives in `libgcc_s`, which `libstdc++` names
        // undefined; without it the link is underlinked.
        paths.push(self.libgcc_s.clone());
        link_dyn_exec(
            &paths,
            prog,
            b"_start",
            &self.interp,
            false,
            IcfMode::None,
            false,
        )
        .expect("xold link must succeed");
    }
}

/// Runs `prog` and returns its exit status.
fn run(prog: &Path) -> std::process::ExitStatus {
    Command::new(prog)
        .status()
        .expect("linked program must be runnable")
}

/// Runs `prog` and returns its stdout, for the tests that read what the
/// program found rather than only whether it agreed.
fn run_output(prog: &Path) -> String {
    let out = Command::new(prog)
        .output()
        .expect("linked program must be runnable");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Assembles `src` into `obj` with the host clang. Returns `None` when clang
/// is unavailable so callers can skip gracefully.
fn assemble(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("s");
    fs::write(&src_path, src).expect("write assembly");
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// The on-disk size of `path`, or 0 if it cannot be read.
fn file_size(path: &Path) -> u64 {
    fs::metadata(path).map_or(0, |m| m.len())
}

/// The slack the size comparison allows for the `PT_GNU_RELRO` boundary.
///
/// The protected run has to end on a page boundary or the loader's mprotect
/// rounds the last page away, and xold maps the image identically
/// (`vaddr == base + offset`), so that boundary is real file bytes -- up to one
/// page of them. The system reference pays nothing for the same boundary: lld
/// pads with an `SHT_NOBITS` `.relro_padding` and opens a second `PT_LOAD`
/// after it. Comparing the two without this term measures the padding, not the
/// deduplication the test is about.
const RELRO_PAGE: u64 = 0x1000;

/// Renders `tmpl` by substituting `{N}` and `{M}` placeholders.
fn render(tmpl: &[u8], n: usize, m: usize) -> Vec<u8> {
    let s = String::from_utf8_lossy(tmpl);
    let out = s
        .replace("{N}", &n.to_string())
        .replace("{M}", &m.to_string());
    out.into_bytes()
}

/// The 20-TU `std::vector` repro: each TU instantiates the same templates,
/// producing identical COMDAT groups. xold must dedup them so the output is
/// close to the system linker's size, and the binary must still run correctly
/// (exit 0, `f1()==3`).
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn comdat_dedup_shrinks_20tu_vector_repro_and_runs() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("vector");
    let n = 20;
    let mut objs = Vec::new();
    for i in 1..=n {
        let obj = dir.join(format!("t{i}.o"));
        let src = render(TU_SRC, i, i * 2);
        compile_cpp(&src, &obj, None).expect("host clang++ compiles tu");
        objs.push(obj);
    }
    let main_o = dir.join("m.o");
    compile_cpp(MAIN_SRC, &main_o, None).expect("host clang++ compiles main");
    objs.push(main_o);

    let prog = dir.join("xo");
    h.link(&objs, &prog);

    // The deduped binary must run correctly: f1() returns 3 and main returns 0.
    let status = run(&prog);
    assert!(
        status.success(),
        "deduped vector repro must exit 0 (f1()==3), got {status}"
    );

    let xold_size = file_size(&prog);

    // The system reference (fully deduped) sets the size floor. xold must be
    // within 1.5x of it, plus the one page the RELRO boundary costs: without
    // COMDAT dedup xold was ~2.5x.
    let sys_prog = dir.join("sysref");
    let sys_ok = Command::new("clang++")
        .args(["--target=x86_64-linux-gnu"])
        .args(&objs)
        .args(["-o", sys_prog.to_str().unwrap()])
        .status()
        .is_ok_and(|s| s.success());
    if sys_ok {
        let sys_size = file_size(&sys_prog);
        let limit = sys_size.saturating_mul(3) / 2 + RELRO_PAGE;
        assert!(
            xold_size < limit,
            "xold output ({xold_size}) must be within 1.5x of system \
             ({sys_size}) plus one RELRO page"
        );
    }

    // Weaker bound that needs no system linker: the pre-dedup output was ~73k,
    // so staying well below it proves the duplicate groups were dropped.
    assert!(
        xold_size < 70_000,
        "xold output ({xold_size}) must be well below the pre-dedup ~73k"
    );
}

/// A multi-TU inline-function program: each TU includes the same header with
/// an `inline` function, producing a COMDAT group for it. The deduped binary
/// keeps one copy and runs correctly.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn comdat_dedup_folds_inline_function_across_tus() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("inline");
    let n = 10;
    let mut objs = Vec::new();
    for i in 1..=n {
        let obj = dir.join(format!("c{i}.o"));
        let src = render(INLINE_TU_SRC, i, 0);
        compile_cpp(&src, &obj, Some(("hdr.h", INLINE_HDR)))
            .expect("host clang++ compiles inline tu");
        objs.push(obj);
    }
    let main_o = dir.join("m.o");
    compile_cpp(INLINE_MAIN_SRC, &main_o, Some(("hdr.h", INLINE_HDR)))
        .expect("host clang++ compiles inline main");
    objs.push(main_o);

    let prog = dir.join("xo");
    h.link(&objs, &prog);

    // inline_add(1,1) + caller_1() = 2 + (1+1) = 4; main returns 0 on success.
    let status = run(&prog);
    assert!(
        status.success(),
        "deduped inline test must exit 0, got {status}"
    );

    // With dedup, adding more TUs barely grows the binary; without it the
    // duplicate inline copies would inflate the output.
    let xold_size = file_size(&prog);
    assert!(
        xold_size < 60_000,
        "inline dedup output ({xold_size}) must stay small"
    );
}

/// Only `GRP_COMDAT` asks for deduplication: a section group without the flag
/// keeps every copy.
///
/// Both variants are the same two objects, signed the same way, differing in
/// one word: the group's flag. With `GRP_COMDAT` the second copy is discarded
/// and the program finds one member; without it, both are laid out and the
/// program finds two. `ld.lld` splits the same way -- it deduplicates only when
/// the flag is set (`lld/ELF/InputFiles.cpp`) -- and xold used to
/// fold both, dropping bytes no input had said were redundant.
///
/// The count is read out of the running program, walking the section between
/// `__start_grpsec` and `__stop_grpsec`, rather than off a section header: the
/// question is what reached the image, and the members are indistinguishable
/// in a header.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_group_without_grp_comdat_keeps_every_copy() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("groupflag");
    let main_o = dir.join("groupmain.o");
    if compile_cpp(GROUP_MAIN_SRC, &main_o, None).is_none() {
        eprintln!("skipping group-flag test: host clang++ unavailable");
        return;
    }

    let mut out = Vec::new();
    for (flag, label) in [(",comdat", "comdat"), ("", "plain")] {
        let mut objs = vec![main_o.clone()];
        for (i, value) in ["0x11", "0x22"].iter().enumerate() {
            let obj = dir.join(format!("{label}{i}.o"));
            let src = render_group(GROUP_MEMBER_SRC, flag, value);
            if assemble(&src, &obj).is_none() {
                eprintln!("skipping group-flag test: host clang unavailable");
                return;
            }
            objs.push(obj);
        }
        let prog = dir.join(format!("{label}_prog"));
        h.link(&objs, &prog);
        out.push(run_output(&prog));
    }

    assert_eq!(
        out.first().map(String::as_str),
        Some("n=1 sum=17\n"),
        "a COMDAT group keeps one copy per signature"
    );
    assert_eq!(
        out.get(1).map(String::as_str),
        Some("n=2 sum=51\n"),
        "a group without GRP_COMDAT is not a request to deduplicate, so both \
         copies must reach the image"
    );
}

/// Renders [`GROUP_MEMBER_SRC`] with the group flag suffix and payload.
fn render_group(tmpl: &[u8], flag: &str, value: &str) -> Vec<u8> {
    let s = String::from_utf8_lossy(tmpl);
    s.replace("{FLAG}", flag)
        .replace("{VALUE}", value)
        .into_bytes()
}

/// Linking the same non-COMDAT input twice produces byte-equal output, proving
/// the dedup plumbing is invisible when no section groups are present.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn non_comdat_link_is_byte_identical() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("ident");
    let src = b"int g = 7;\n\
         int helper(int x){ return x * 3; }\n\
         int main(){ return helper(g) - 21; }\n";
    let main_o = dir.join("plain.o");
    compile_cpp(src, &main_o, None).expect("host clang++ compiles plain");

    let a = dir.join("a");
    let b = dir.join("b");
    h.link(std::slice::from_ref(&main_o), &a);
    h.link(std::slice::from_ref(&main_o), &b);

    let bytes_a = fs::read(&a).expect("read a");
    let bytes_b = fs::read(&b).expect("read b");
    assert_eq!(
        bytes_a, bytes_b,
        "two links of the same non-COMDAT input must be byte-identical"
    );
    let status = run(&a);
    assert!(
        status.success(),
        "non-COMDAT binary must exit 0, got {status}"
    );
}
