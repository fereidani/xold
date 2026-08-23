//! The workload threshold that sizes a link's thread pool must not change
//! what the link produces.
//!
//! `xold::pool` runs a small link on a private one-thread pool and leaves a
//! large one on rayon's global pool. That is only safe because the two paths
//! are the same code over the same data, so these tests link one fixture set
//! three ways -- on the pool the rule picks for it, on an explicit one-thread
//! pool, and on an explicit eight-thread pool -- and require the images to be
//! byte-identical.
//!
//! An installed pool wins over the rule, which is what keeps
//! `tests/parallel.rs` and `tests/parallel_parse.rs` testing the thread counts
//! they name, and what lets the eight-thread arm below stand for the
//! above-threshold path.

use std::{fs, path::PathBuf};

use rayon::ThreadPoolBuilder;
use xold::{icf::IcfMode, linker::link_to, pool::fans_out};

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    p
}

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

/// Links `inputs` for `entry` and returns the image bytes. `threads` of
/// `None` lets `xold::pool` choose, which is the below-threshold path for
/// every fixture here; `Some(n)` installs a pool of `n` threads, which the
/// rule defers to and which is therefore the above-threshold path.
fn link_bytes(
    inputs: &[PathBuf],
    entry: &[u8],
    threads: Option<usize>,
    label: &str,
) -> Vec<u8> {
    let out = temp(label);
    let link = || link_to(inputs, &out, entry, false, IcfMode::None, false);
    match threads {
        None => link().expect("link succeeds"),
        Some(n) => {
            let pool = ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("thread pool builds");
            pool.install(link).expect("link succeeds");
        }
    }
    let bytes = fs::read(&out).expect("output readable");
    let _ = fs::remove_file(&out);
    bytes
}

/// Links `inputs` on both sides of the threshold and requires one image.
fn assert_same_image_either_side(inputs: &[PathBuf], entry: &[u8], tag: &str) {
    let sizes: Vec<Vec<u8>> =
        inputs.iter().map(|p| fs::read(p).expect("input")).collect();
    let views: Vec<&[u8]> = sizes.iter().map(Vec::as_slice).collect();
    assert!(
        !fans_out(&views),
        "{tag}: the fixtures must sit below the threshold, or the \
         rule-chosen arm tests nothing"
    );

    let chosen = link_bytes(inputs, entry, None, &format!("xold_pt_{tag}_c"));
    let serial =
        link_bytes(inputs, entry, Some(1), &format!("xold_pt_{tag}_1"));
    let wide = link_bytes(inputs, entry, Some(8), &format!("xold_pt_{tag}_8"));

    assert_eq!(
        chosen, serial,
        "{tag}: the pool the rule picks must produce the one-thread image"
    );
    assert_eq!(
        chosen, wide,
        "{tag}: a link below the threshold must produce the same bytes as \
         the same link above it"
    );
}

/// The GOT-exercising `min.o + ext.o` link is one image on either side of the
/// threshold.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn got_link_is_identical_either_side_of_the_threshold() {
    let inputs = vec![fixture("min.o"), fixture("ext.o")];
    assert_same_image_either_side(&inputs, b"entry", "got");
}

/// So is the freestanding `prog.o + start.o` link, which relocates through
/// the parallel copy pass without a GOT.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn freestanding_link_is_identical_either_side_of_the_threshold() {
    let inputs = vec![fixture("prog.o"), fixture("start.o")];
    assert_same_image_either_side(&inputs, b"_start", "free");
}

/// A four-kilobyte page repeated as often as a test needs a distinct input.
const BLOCK: &[u8; 4096] = &[0u8; 4096];

/// The rule separates a link too small to fan out from one large enough,
/// whether the size comes from the inputs' bytes or from their number.
#[test]
fn the_rule_separates_small_workloads_from_large_ones() {
    let few = vec![&BLOCK[..]; 4];
    assert!(!fans_out(&few), "16 KiB over 4 inputs is not worth a pool");

    let one_large = vec![0u8; 4 << 20];
    assert!(
        fans_out(&[&one_large]),
        "4 MiB in one input is worth a pool"
    );

    let many = vec![&BLOCK[..]; 512];
    assert!(fans_out(&many), "512 inputs are worth a pool");
}

/// A shared object is a `DT_NEEDED` dependency whose sections are never
/// linked in, so its size must not count towards the estimate. Getting this
/// wrong puts a compiler driver's link -- a few kilobytes of objects next to a
/// multi-megabyte `libc.so.6` -- on the wrong side of the threshold.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn a_shared_object_does_not_count_towards_the_estimate() {
    let mut shared = vec![0u8; 4 << 20];
    shared[..4].copy_from_slice(b"\x7fELF");
    // `e_type`, little-endian, at the offset the ELF header keeps it: ET_DYN.
    shared[16..18].copy_from_slice(&3u16.to_le_bytes());
    assert!(
        !fans_out(&[&shared, &BLOCK[..]]),
        "a 4 MiB dependency next to one small object is a small link"
    );

    // The same bytes as a relocatable object do count.
    let mut object = shared.clone();
    object[16..18].copy_from_slice(&1u16.to_le_bytes());
    assert!(
        fans_out(&[&object, &BLOCK[..]]),
        "a 4 MiB relocatable input is a large link"
    );
}
