//! Linking inputs the caller already holds in memory.
//!
//! A caller that generated an object itself has the bytes in hand; making it
//! write them to a temporary so xold can map them back is a round trip with no
//! purpose. [`Input::Memory`] removes it, and these tests pin the contract that
//! makes the removal safe:
//!
//! - the same bytes produce the same image whether they arrive as a path or as
//!   a slice, at any thread count;
//! - a memory input is legal at any position, so a library still satisfies only
//!   the references made before it;
//! - a slice that does not start on a word boundary is realigned rather than
//!   rejected, because `Vec<u8>` promises alignment 1 and the caller cannot
//!   promise more;
//! - the name a memory input carries is used exactly where a real path's file
//!   name would be, including as the `DT_NEEDED` fallback for a shared object
//!   with no `DT_SONAME`.

use std::{
    fs,
    path::{Path, PathBuf},
};

use rayon::ThreadPoolBuilder;
use xold::{
    elf::ObjectFile,
    icf::IcfMode,
    input::Input,
    linker::{Link, link_image, link_shared},
};

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    p
}

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

/// The alignment a zero-copy structure view needs, mirrored from the linker so
/// the misalignment test can construct a slice that definitely lacks it.
const ALIGN: usize = 8;

/// `DT_NEEDED`, and the size of one `.dynamic` entry.
const DT_NEEDED: i64 = 1;
const DYN_ENTRY: usize = 16;

/// Links `inputs` into a static executable and returns the image bytes, with
/// the scratch file removed. `label` keeps concurrently running tests apart.
fn link_bytes(inputs: &[Input<'_>], entry: &[u8], label: &str) -> Vec<u8> {
    link_bytes_on(inputs, entry, label, 0)
}

/// As [`link_bytes`], but with the link driven by a private rayon pool of
/// `threads` workers (`0` means the ambient pool). `pool.install` makes that
/// pool the active one, so every internal `par_iter` is scheduled by it.
fn link_bytes_on(
    inputs: &[Input<'_>],
    entry: &[u8],
    label: &str,
    threads: usize,
) -> Vec<u8> {
    let out = temp(label);
    let job = Link {
        entry,
        ..Link::exec(inputs, &out)
    };
    if threads == 0 {
        link_image(&job).expect("link must succeed");
    } else {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool builds");
        pool.install(|| link_image(&job))
            .expect("link must succeed");
    }
    let bytes = fs::read(&out).expect("output must be readable");
    let _ = fs::remove_file(&out);
    bytes
}

/// Copies `bytes` into a buffer that deliberately does not start where a
/// structure view may begin, so the realigning path is the one under test.
/// Returns the buffer and the offset the copy starts at.
fn misaligned(bytes: &[u8]) -> (Vec<u8>, usize) {
    let mut buf = vec![0u8; bytes.len() + ALIGN];
    // Shifts the start to one byte past a word boundary, whatever the
    // allocator handed back.
    let slack = (ALIGN - buf.as_ptr() as usize % ALIGN) % ALIGN;
    let off = slack + 1;
    buf[off..off + bytes.len()].copy_from_slice(bytes);
    (buf, off)
}

/// The `DT_NEEDED` names recorded in the image at `path`.
fn dt_needed(path: &Path) -> Vec<Vec<u8>> {
    let bytes = fs::read(path).expect("output must be readable");
    let obj = ObjectFile::parse(&bytes).expect("output must be valid ELF");
    let section = |want: &[u8]| {
        obj.sections()
            .iter()
            .find(|s| obj.section_name(s) == want)
            .map(|s| obj.section_data(s).expect("section data"))
    };
    let Some(dynamic) = section(b".dynamic") else {
        return Vec::new();
    };
    let strtab = section(b".dynstr").unwrap_or_default();
    let mut names = Vec::new();
    for entry in dynamic.as_chunks::<DYN_ENTRY>().0 {
        let tag = i64::from_le_bytes(entry[..8].try_into().unwrap());
        if tag != DT_NEEDED {
            continue;
        }
        let raw = u64::from_le_bytes(entry[8..].try_into().unwrap());
        let val = usize::try_from(raw).expect("string offset fits");
        let rest = &strtab[val.min(strtab.len())..];
        let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
        names.push(rest[..end].to_vec());
    }
    names
}

/// The baseline every other test compares against: `min.o + ext.o` linked from
/// two real files.
///
/// `label` names this caller's own scratch file. The tests run concurrently and
/// [`link_bytes`] deletes what it read, so a label shared between two of them
/// would have one test remove the file the other is still reading.
fn baseline(label: &str) -> Vec<u8> {
    let paths = [fixture("min.o"), fixture("ext.o")];
    link_bytes(&Input::from_paths(&paths), b"entry", label)
}

#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn memory_input_matches_a_file_input_byte_for_byte() {
    let min = fs::read(fixture("min.o")).expect("fixture readable");
    let ext = fs::read(fixture("ext.o")).expect("fixture readable");
    let from_memory = link_bytes(
        &[
            Input::Memory {
                name: Path::new("min.o"),
                bytes: &min,
            },
            Input::Memory {
                name: Path::new("ext.o"),
                bytes: &ext,
            },
        ],
        b"entry",
        "xold_mem_both.out",
    );
    assert_eq!(
        from_memory,
        baseline("xold_mem_both_base.out"),
        "bytes linked from memory must equal the same bytes linked from files"
    );
}

#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn a_memory_input_is_legal_at_any_position() {
    let min = fs::read(fixture("min.o")).expect("fixture readable");
    let ext = fs::read(fixture("ext.o")).expect("fixture readable");
    let min_path = fixture("min.o");
    let ext_path = fixture("ext.o");
    let expected = baseline("xold_mem_position_base.out");

    let first = link_bytes(
        &[
            Input::Memory {
                name: Path::new("min.o"),
                bytes: &min,
            },
            Input::Path(&ext_path),
        ],
        b"entry",
        "xold_mem_first.out",
    );
    assert_eq!(first, expected, "memory input first");

    let last = link_bytes(
        &[
            Input::Path(&min_path),
            Input::Memory {
                name: Path::new("ext.o"),
                bytes: &ext,
            },
        ],
        b"entry",
        "xold_mem_last.out",
    );
    assert_eq!(last, expected, "memory input last");
}

/// A library only satisfies references made by the inputs before it, and that
/// has to keep holding when the referring object is a slice rather than a file:
/// `min.o` references `external_func`, which is defined by a member of
/// `libext.a` named after it.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn an_archive_after_a_memory_object_is_still_mined() {
    let min = fs::read(fixture("min.o")).expect("fixture readable");
    let archive = fixture("libext.a");
    let paths = [fixture("min.o"), fixture("libext.a")];

    let mixed = link_bytes(
        &[
            Input::Memory {
                name: Path::new("min.o"),
                bytes: &min,
            },
            Input::Path(&archive),
        ],
        b"entry",
        "xold_mem_archive.out",
    );
    let all_paths = link_bytes(
        &Input::from_paths(&paths),
        b"entry",
        "xold_mem_archive_ref.out",
    );
    assert_eq!(
        mixed, all_paths,
        "an archive must be mined for a memory input's undefined references"
    );
}

/// Cranelift and friends hand back a `Vec<u8>`, whose element type promises
/// alignment 1 and nothing more. A slice that starts mid-word must still link,
/// and must link to the same bytes.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn a_misaligned_memory_input_links_identically() {
    let min = fs::read(fixture("min.o")).expect("fixture readable");
    let ext = fs::read(fixture("ext.o")).expect("fixture readable");
    let (min_buf, min_off) = misaligned(&min);
    let (ext_buf, ext_off) = misaligned(&ext);
    let min_slice = &min_buf[min_off..min_off + min.len()];
    let ext_slice = &ext_buf[ext_off..ext_off + ext.len()];
    assert_ne!(
        min_slice.as_ptr() as usize % ALIGN,
        0,
        "the test input must actually be misaligned"
    );

    let linked = link_bytes(
        &[
            Input::Memory {
                name: Path::new("min.o"),
                bytes: min_slice,
            },
            Input::Memory {
                name: Path::new("ext.o"),
                bytes: ext_slice,
            },
        ],
        b"entry",
        "xold_mem_unaligned.out",
    );
    assert_eq!(
        linked,
        baseline("xold_mem_unaligned_base.out"),
        "a misaligned slice must be realigned, not rejected or misread"
    );
}

/// The determinism rule applies to memory inputs like any other: the image is a
/// pure function of the inputs, not of how rayon scheduled the passes.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn memory_input_is_byte_identical_across_thread_counts() {
    let min = fs::read(fixture("min.o")).expect("fixture readable");
    let ext = fs::read(fixture("ext.o")).expect("fixture readable");
    let inputs = [
        Input::Memory {
            name: Path::new("min.o"),
            bytes: &min,
        },
        Input::Memory {
            name: Path::new("ext.o"),
            bytes: &ext,
        },
    ];
    let one = link_bytes_on(&inputs, b"entry", "xold_mem_t1.out", 1);
    let eight = link_bytes_on(&inputs, b"entry", "xold_mem_t8.out", 8);
    assert_eq!(one, eight, "output must not depend on thread count");
}

/// The one place an input's *name* reaches the image: a shared object with no
/// `DT_SONAME` is recorded under its file name. A memory input must behave as
/// the same bytes in a file of that name would, which is what keeps the name a
/// documented input rather than a hidden one.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn a_memory_shared_object_is_needed_under_its_given_name() {
    // `ext.o` has no `DT_SONAME` of its own once linked with `-shared`.
    let so = temp("xold_mem_libext.so");
    link_shared(&[fixture("ext.o")], &so, None, false, IcfMode::None, false)
        .expect("shared link must succeed");
    let so_bytes = fs::read(&so).expect("shared object readable");

    let min_path = fixture("min.o");
    let out = temp("xold_mem_needed.out");
    let inputs = [
        Input::Path(&min_path),
        Input::Memory {
            name: Path::new("libmem.so"),
            bytes: &so_bytes,
        },
    ];
    link_image(&Link {
        entry: b"entry",
        ..Link::dyn_exec(&inputs, &out, b"/lib64/ld-linux-x86-64.so.2")
    })
    .expect("dynamic link must succeed");

    assert_eq!(
        dt_needed(&out),
        vec![b"libmem.so".to_vec()],
        "the memory input's name must be the DT_NEEDED fallback"
    );

    let _ = fs::remove_file(&so);
    let _ = fs::remove_file(&out);
}
