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

use common::{elf_fixture, which};
use rayon::ThreadPoolBuilder;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_to};

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
    elf_fixture("start.o")
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

    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        assert_chain_contract(&fs::read(&out).expect("read linked image"));
        let _ = fs::remove_file(&out);
        return;
    }

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

/// Verifies every applied cross-file call in the linked chain. The final
/// function returns 23; the generated source adds each preceding index, so a
/// complete `entry -> step_0 -> ... -> step_23` path computes 276.
fn assert_chain_contract(bytes: &[u8]) {
    let obj = ObjectFile::parse(bytes).expect("valid ELF");
    let symtab = obj.symbol_table().expect("read symtab").expect("symtab");
    let symbol = |name: &[u8]| {
        symtab
            .syms
            .iter()
            .find(|sym| symtab.name(sym) == name)
            .copied()
            .unwrap_or_else(|| {
                panic!("{} is defined", String::from_utf8_lossy(name))
            })
    };
    let entry = symbol(b"entry");
    let first = symbol(b"step_0");
    assert_eq!(
        call_target(&obj, entry.st_value.get(), entry.st_size.get()),
        Some(first.st_value.get())
    );
    for i in 0..CHAIN_LEN - 1 {
        let current = symbol(format!("step_{i}").as_bytes());
        let next = symbol(format!("step_{}", i + 1).as_bytes());
        assert_eq!(
            call_target(&obj, current.st_value.get(), current.st_size.get()),
            Some(next.st_value.get()),
            "step_{i} calls step_{}",
            i + 1
        );
    }
    let last = symbol(b"step_23");
    let body = image_at(&obj, last.st_value.get(), last.st_size.get())
        .expect("last body");
    assert!(
        body.windows(5).any(|w| w == [0xb8, 23, 0, 0, 0]),
        "step_23 returns 23"
    );
}

fn call_target(obj: &ObjectFile<'_>, addr: u64, size: u64) -> Option<u64> {
    let body = image_at(obj, addr, size)?;
    let at = body.iter().position(|&byte| byte == 0xe8)?;
    let disp = i32::from_le_bytes(body.get(at + 1..at + 5)?.try_into().ok()?);
    Some(
        addr.wrapping_add(u64::try_from(at + 5).ok()?)
            .wrapping_add(i64::from(disp).cast_unsigned()),
    )
}

fn image_at<'a>(
    obj: &ObjectFile<'a>,
    addr: u64,
    size: u64,
) -> Option<&'a [u8]> {
    let len = usize::try_from(size).ok()?;
    for sec in obj.sections() {
        let base = sec.sh_addr.get();
        if addr < base || addr >= base.saturating_add(sec.sh_size.get()) {
            continue;
        }
        let at = usize::try_from(addr - base).ok()?;
        return obj.section_data(sec).ok()?.get(at..at.checked_add(len)?);
    }
    None
}
