//! Identical code folding and C++ exception tables.
//!
//! Two functions can agree on every byte and every relocation and still need
//! different unwind behaviour, because what separates them is not in the
//! section: a `catch` clause's type is recorded in `.gcc_except_table`, which
//! the FDE describing the function names through its LSDA pointer. Folding
//! such a pair gives both the survivor's table, so the other one's handler
//! never matches and the exception walks past a `catch` written for it.
//!
//! These tests link a program where exactly that happens and run it. A
//! structural check would not do: the two functions fold into one address
//! either way, and the only place the difference shows is which handler the
//! unwinder picks.
//!
//! Covered:
//!
//! - **The repro**: `fa` catches `A`, `fb` catches `B`, both compile to the
//!   same bytes. Under `--icf=none`, `--icf=safe` and `--icf=all` alike the
//!   program must report that `fb` caught its exception.
//! - **Not folded**: the two sections keep distinct addresses under
//!   `--icf=all`, which is what makes the run above possible.
//! - **Still folds**: a pair with no exception handling in the same image is
//!   folded, so the exclusion is the narrow one and not a disabled pass.
//! - **One row per PC**: `.eh_frame_hdr`'s search table holds no two rows for
//!   the same address, the invariant that folding used to break by leaving the
//!   retired function's FDE behind.
//!
//! Gated on `clang++`, `gcc` (for the crt objects) and the system `ld.so`; if
//! absent the tests print a note and return.

#![allow(clippy::similar_names)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, libstdcxx_so, which};
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_dyn_exec};

mod common;

/// `fa` and `fb` compile to identical bytes with identical relocations: the
/// same call sequence, the same three libstdc++ imports, the same immediate.
/// They differ only in the type each handler catches, which lives in
/// `.gcc_except_table` and is reached through the FDE's LSDA pointer.
///
/// `main` calls `fb` with the argument that throws `B`. A correct link prints
/// `caught=11`; a link that folded `fb` onto `fa` prints `caught=-1`, because
/// `fa`'s table admits only `A` and the exception escapes to the outer
/// `catch (...)`.
const LSDA_SRC: &[u8] = b"\
#include <cstdio>\n\
struct A { int x; };\n\
struct B { int x; };\n\
__attribute__((noinline)) void thrower(int k){\n\
    if (k == 1) throw A{1};\n\
    throw B{2};\n\
}\n\
__attribute__((noinline)) int fa(int k){\n\
    try { thrower(k); } catch (A &) { return 11; }\n\
    return 0;\n\
}\n\
__attribute__((noinline)) int fb(int k){\n\
    try { thrower(k); } catch (B &) { return 11; }\n\
    return 0;\n\
}\n\
int main(){\n\
    int r = 0;\n\
    try { r = fb(2); } catch (...) { r = -1; }\n\
    printf(\"caught=%d\\n\", r);\n\
    return 0;\n\
}\n";

/// The same program with two identical functions that catch nothing. Their
/// CIE carries no `L`, so nothing keeps them out of the partition and they
/// must still fold.
const PLAIN_SRC: &[u8] = b"\
#include <cstdio>\n\
__attribute__((noinline)) int p1(int x){ return x * 3 + 7; }\n\
__attribute__((noinline)) int p2(int x){ return x * 3 + 7; }\n\
int main(){ printf(\"%d %d\\n\", p1(1), p2(1)); return 0; }\n";

/// The host toolchain these tests need.
struct Harness {
    crt1: PathBuf,
    crti: PathBuf,
    crtn: PathBuf,
    libc: PathBuf,
    libstdcxx: PathBuf,
    interp: Vec<u8>,
}

impl Harness {
    /// Collects the harness, or returns `None` with a note when a piece is
    /// missing, so a host without a C++ toolchain skips rather than fails.
    fn detect() -> Option<Self> {
        if which("clang++").is_none() {
            eprintln!("skipping icf_lsda tests: clang++ unavailable");
            return None;
        }
        Some(Self {
            crt1: crt_file("Scrt1.o")?,
            crti: crt_file("crti.o")?,
            crtn: crt_file("crtn.o")?,
            libc: libc_so()?,
            libstdcxx: libstdcxx_so()?,
            interp: interpreter()?,
        })
    }

    /// Links `obj` plus the crt objects, libc and libstdc++ into `prog`.
    fn link(&self, obj: &Path, prog: &Path, icf: IcfMode) {
        link_dyn_exec(
            &[
                obj.to_path_buf(),
                self.crti.clone(),
                self.crt1.clone(),
                self.crtn.clone(),
                self.libc.clone(),
                self.libstdcxx.clone(),
            ],
            prog,
            b"_start",
            &self.interp,
            false,
            icf,
            false,
        )
        .expect("xold C++ link must succeed");
    }
}

/// A private working directory, named after the process so concurrent test
/// binaries cannot delete each other's files.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_icf_lsda_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Compiles `src` with the host `clang++` at `-O2 -ffunction-sections`, which
/// is what puts each function in a section of its own for folding to see.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    let clangxx = which("clang++")?;
    let src_path = obj.with_extension("cpp");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clangxx)
        .args([
            "--target=x86_64-linux-gnu",
            "-fPIE",
            "-O2",
            "-ffunction-sections",
            "-c",
        ])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// Compiles, links under `icf` and runs, returning the image bytes and stdout.
fn build_and_run(
    h: &Harness,
    dir: &Path,
    name: &str,
    src: &[u8],
    icf: IcfMode,
) -> (Vec<u8>, String) {
    let obj = dir.join(format!("{name}.o"));
    compile(src, &obj).expect("host clang++ compiles the source");
    let prog = dir.join(name);
    h.link(&obj, &prog, icf);
    let bytes = fs::read(&prog).expect("read linked output");
    let out = Command::new(&prog).output().expect("linked program runs");
    assert!(out.status.success(), "{name} exits 0");
    (bytes, String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The `st_value` of a symbol in the output `.symtab`, by name.
fn symbol_value(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).expect("valid ELF output");
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab
        .iter()
        .find(|s| symtab.name(s) == name)
        .map(|s| s.st_value.get())
}

/// The covered PCs in the `.eh_frame_hdr` binary-search table, in table order.
///
/// The table is `version, eh_frame_ptr_enc, fde_count_enc, table_enc`, then the
/// `eh_frame` pointer and the FDE count, then one `(pc, fde)` pair per row.
/// Every value is `DW_EH_PE_datarel | DW_EH_PE_sdata4`, a signed 32-bit offset
/// from the section's own address, which the reader here adds back.
fn eh_frame_hdr_pcs(bytes: &[u8]) -> Vec<u64> {
    let obj = ObjectFile::parse(bytes).expect("valid ELF output");
    let Some(shdr) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".eh_frame_hdr")
    else {
        return Vec::new();
    };
    let data = obj.section_data(shdr).expect("eh_frame_hdr readable");
    let base = shdr.sh_addr.get();
    let count = u32::from_le_bytes(
        data[8..12].try_into().expect(".eh_frame_hdr has a count"),
    ) as usize;
    (0..count)
        .filter_map(|i| {
            let at = 12 + i * 8;
            let raw = data.get(at..at + 4)?;
            let off = i32::from_le_bytes(raw.try_into().ok()?);
            Some(base.wrapping_add(i64::from(off).cast_unsigned()))
        })
        .collect()
}

/// The repro. `fb` must catch its own exception whatever the folding mode.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_handler_survives_folding_in_every_mode() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("handler");
    for (name, mode) in [
        ("none", IcfMode::None),
        ("safe", IcfMode::Safe),
        ("all", IcfMode::All),
    ] {
        let (_, out) = build_and_run(&h, &dir, name, LSDA_SRC, mode);
        assert_eq!(
            out.trim(),
            "caught=11",
            "--icf={name}: fb's own catch clause must match; \
             caught=-1 means it inherited fa's exception table"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

/// The two functions are not folded, which is what leaves each with its own
/// exception table.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn lsda_functions_keep_distinct_addresses() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("distinct");
    let (bytes, _) = build_and_run(&h, &dir, "lsda", LSDA_SRC, IcfMode::All);
    let fa = symbol_value(&bytes, b"_Z2fai").expect("fa is in .symtab");
    let fb = symbol_value(&bytes, b"_Z2fbi").expect("fb is in .symtab");
    assert_ne!(
        fa, fb,
        "a function an FDE describes with an LSDA must not fold"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The exclusion is narrow: a pair whose CIE declares no LSDA still folds, so
/// `--icf=all` has not simply stopped working on C++ input.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn functions_without_an_lsda_still_fold() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("plain");
    let (folded, _) = build_and_run(&h, &dir, "on", PLAIN_SRC, IcfMode::All);
    let (apart, _) = build_and_run(&h, &dir, "off", PLAIN_SRC, IcfMode::None);
    let p1 = symbol_value(&folded, b"_Z2p1i").expect("p1 is in .symtab");
    let p2 = symbol_value(&folded, b"_Z2p2i").expect("p2 is in .symtab");
    assert_eq!(p1, p2, "an ordinary identical pair still folds");
    let q1 = symbol_value(&apart, b"_Z2p1i").expect("p1 is in .symtab");
    let q2 = symbol_value(&apart, b"_Z2p2i").expect("p2 is in .symtab");
    assert_ne!(q1, q2, "and stays apart without folding");
    let _ = fs::remove_dir_all(&dir);
}

/// The unwind index holds one row per address. A folded function's FDE used to
/// survive into the image describing the representative's address, leaving two
/// rows for one PC with different LSDA pointers behind them.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn eh_frame_hdr_has_one_row_per_pc() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("rows");
    let (bytes, _) = build_and_run(&h, &dir, "plain", PLAIN_SRC, IcfMode::All);
    let pcs = eh_frame_hdr_pcs(&bytes);
    assert_ne!(pcs, [], "the image carries an unwind index");
    for pair in pcs.windows(2) {
        assert!(
            pair[0] < pair[1],
            "the search table is strictly ascending: {pair:x?}"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}
