//! Determinism checks for the multi-threaded linker.
//!
//! The parallel scan and copy+apply passes must produce byte-identical output
//! regardless of thread count. Each test runs the same link twice on private
//! `rayon` thread pools (one single-threaded, one multi-threaded) installed
//! via `ThreadPoolBuilder::install`, so the parallel paths execute under
//! controlled scheduling. `RAYON_NUM_THREADS` is not used so the tests do not
//! depend on the ambient environment.

use std::{fs, path::PathBuf};

use common::elf_fixture;
use rayon::ThreadPoolBuilder;
use xold::{icf::IcfMode, linker::link_to};

mod common;

fn fixture(name: &str) -> PathBuf {
    elf_fixture(name)
}

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

/// Links `inputs` for `entry` on a fresh thread pool of `n` threads, returning
/// the output bytes. `pool.install` runs `link_to` with that pool as the
/// active rayon pool, so every internal `par_iter` is scheduled by it.
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
    let out = temp(label);
    pool.install(|| link_to(inputs, &out, entry, false, IcfMode::None, false))
        .expect("link succeeds");
    fs::read(&out).expect("output readable")
}

/// The GOT-exercising `min.o + ext.o` link must be byte-identical whether the
/// scan and copy passes run on one thread or many. This directly proves the
/// GOT allocation order and the per-section relocation writes are independent
/// of how rayon schedules the work items.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn got_link_is_byte_identical_across_thread_counts() {
    let inputs = vec![fixture("min.o"), fixture("ext.o")];
    let one = link_bytes(&inputs, b"entry", 1, "xold_par_min_1.out");
    let eight = link_bytes(&inputs, b"entry", 8, "xold_par_min_8.out");
    assert_eq!(
        one, eight,
        "GOT link output must not depend on thread count"
    );

    let _ = fs::remove_file(temp("xold_par_min_1.out"));
    let _ = fs::remove_file(temp("xold_par_min_8.out"));
}

/// The freestanding `prog.o + start.o` link (no GOT, but PC-relative and
/// absolute relocations through the parallel copy pass) must also be stable.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn freestanding_link_is_byte_identical_across_thread_counts() {
    let inputs = vec![fixture("prog.o"), fixture("start.o")];
    let one = link_bytes(&inputs, b"_start", 1, "xold_par_prog_1.out");
    let eight = link_bytes(&inputs, b"_start", 8, "xold_par_prog_8.out");
    assert_eq!(
        one, eight,
        "freestanding link output must not depend on thread count"
    );

    let _ = fs::remove_file(temp("xold_par_prog_1.out"));
    let _ = fs::remove_file(temp("xold_par_prog_8.out"));
}
