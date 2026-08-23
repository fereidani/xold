//! Multi-file parallel parse + symbol extraction: correctness and determinism.
//!
//! These tests compile a chain of 24 cross-referencing ELF objects (each
//! defines a function that calls the next module's function, accumulating a
//! sum), link them with xold, and check the result runs correctly. They also
//! link the same inputs on private rayon pools of 1 and 8 threads and assert
//! the output is byte-identical, proving the parallel parse + extract + serial
//! merge is deterministic regardless of scheduling.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use rayon::ThreadPoolBuilder;
use xold::{icf::IcfMode, linker::link_to};

mod common;

/// The number of cross-referencing modules in the chain. 24 objects exercises
/// the parallel parse + extract over enough inputs to be meaningful.
const CHAIN_LEN: usize = 24;

/// The expected exit code: the sum 0 + 1 + ... + 23 = 276, whose low 8 bits
/// (the value `waitpid` reports as the process exit status) are 20.
const EXPECTED_EXIT: i32 = 20;

/// Compiles `src` into `obj` with the host clang as a freestanding `x86_64`
/// object. Returns `None` if clang is unavailable so callers can skip.
fn compile(src: &str, obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args([
            "--target=x86_64-linux-gnu",
            "-ffreestanding",
            "-fno-pic",
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

/// A fresh per-test working directory.
fn workdir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xold_parparse_{label}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Generates the C source for module `i` of the chain. Module `i` defines
/// `step_i` and calls `step_{i+1}`; the last module just returns its index.
fn module_source(i: usize) -> String {
    if i + 1 == CHAIN_LEN {
        return format!("int step_{i}(void) {{ return {i}; }}\n");
    }
    format!(
        "extern int step_{next}(void);\n\
         int step_{i}(void) {{ return {i} + step_{next}(); }}\n",
        next = i + 1
    )
}

/// The entry module: calls `step_0` so the program exit code is the chain sum.
const ENTRY_SRC: &str = "extern int step_0(void);\n\
     int entry(void) { return step_0(); }\n";

/// Compiles the full chain plus the entry module into `dir`, returning the
/// object paths in link order (entry first, then the chain). Returns `None` if
/// the host clang is unavailable.
fn build_chain(dir: &Path) -> Option<Vec<PathBuf>> {
    which("clang")?;
    let mut objs = Vec::with_capacity(CHAIN_LEN + 1);
    let entry = dir.join("entry.o");
    compile(ENTRY_SRC, &entry)?;
    objs.push(entry);
    for i in 0..CHAIN_LEN {
        let obj = dir.join(format!("mod{i:02}.o"));
        compile(&module_source(i), &obj)?;
        objs.push(obj);
    }
    Some(objs)
}

/// The freestanding `_start` fixture that calls `entry` and exits.
fn start_fixture() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push("start.o");
    p
}

/// Links `inputs` for `entry` on a fresh thread pool of `n` threads, returning
/// the output bytes. Mirrors the determinism harness in `tests/parallel.rs`.
fn link_bytes(
    inputs: &[PathBuf],
    entry: &[u8],
    n: usize,
    label: &str,
) -> Vec<u8> {
    let pool = ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("thread pool builds");
    let out = std::env::temp_dir().join(label);
    pool.install(|| link_to(inputs, &out, entry, false, IcfMode::None, false))
        .expect("link succeeds");
    fs::read(&out).expect("output readable")
}

/// Links a chain of 24 cross-referencing objects and checks the result runs
/// with the expected exit code (the accumulated sum). This proves the parallel
/// parse + extract + serial merge resolves every cross-file reference.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn multi_file_chain_links_and_runs_correctly() {
    let Some(objs) = build_chain(&workdir("build")) else {
        eprintln!("skipping parallel parse chain test: clang unavailable");
        return;
    };
    let mut inputs = objs;
    inputs.push(start_fixture());

    let out = std::env::temp_dir().join("xold_pp_chain.out");
    link_to(&inputs, &out, b"_start", false, IcfMode::None, false)
        .expect("link must succeed");

    let status = Command::new(&out)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(EXPECTED_EXIT),
        "chain sum 0+...+23 = 276, exit code = 276 & 0xff"
    );
    let _ = fs::remove_file(&out);
}

/// The 24-object chain must link to byte-identical output whether the parallel
/// parse + extract runs on one thread or eight. This proves the extract-then-
/// merge design (parallel parse, serial merge in input order) is deterministic
/// regardless of how rayon schedules the per-input work.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn multi_file_chain_is_byte_identical_across_thread_counts() {
    let Some(objs) = build_chain(&workdir("det")) else {
        eprintln!("skipping determinism test: clang unavailable");
        return;
    };
    let mut inputs = objs;
    inputs.push(start_fixture());

    let one = link_bytes(&inputs, b"_start", 1, "xold_pp_chain_1.out");
    let eight = link_bytes(&inputs, b"_start", 8, "xold_pp_chain_8.out");
    assert_eq!(
        one, eight,
        "multi-file chain output must not depend on thread count"
    );

    let _ = fs::remove_file(std::env::temp_dir().join("xold_pp_chain_1.out"));
    let _ = fs::remove_file(std::env::temp_dir().join("xold_pp_chain_8.out"));
}
