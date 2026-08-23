//! `.eh_frame_hdr` holds one row per covered PC.
//!
//! The table is a binary-search index the runtime unwinder uses to find an FDE
//! for a throwing PC. Two rows with the same PC describe the same bytes with,
//! in general, different LSDA pointers, and the unwinder takes whichever the
//! search happens to land on -- which handler runs then depends on the shape
//! of the table rather than on the program.
//!
//! `build_hdr` sorted and then `debug_assert!`ed uniqueness, so in a release
//! build a duplicate would simply be written. The assertion's premise was that
//! `split::keep_flags` drops every record that could pair up, but it cannot:
//! it resolves an FDE's first relocation through the *global* symbol table, so
//! a retired weak or COMDAT copy's FDE resolves to the prevailing section and
//! looks live. The table is now deduplicated as well as defended, which is
//! what lld does -- stable sort, then unique, with a comment naming the same
//! case (`lld/ELF/SyntheticSections.cpp`).
//!
//! A caveat worth stating: no input this host can produce actually reaches the
//! duplicate. Each of the three shapes below -- a COMDAT duplicate, an ICF
//! fold, a losing weak definition -- is handled by `keep_flags` before the
//! table is built, so these tests pass with the deduplication removed. They
//! guard the invariant across the cases that could break it; they do not
//! demonstrate the defect.
//!
//! Gated on `clang++` and a system `libstdc++`; without them the tests print a
//! note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_dyn_exec};

mod common;

/// Two translation units sharing an inline function that throws, so both emit
/// an FDE for it in a COMDAT group and one copy is retired.
const HEADER: &[u8] = b"#pragma once\n\
    struct Guard { ~Guard(); };\n\
    inline int thrower(int x)\n\
    { Guard g; if (x < 0) { throw x; } return x * 2; }\n";

const A_SRC: &[u8] = b"#include \"h.hpp\"\n\
    Guard::~Guard() {}\n\
    int a_go(int x) { try { return thrower(x); } catch (int e) { return -e; } }\n";

const B_SRC: &[u8] = b"#include \"h.hpp\"\n\
    int a_go(int);\n\
    int b_go(int x) { try { return thrower(x); } catch (int e) { return -e; } }\n\
    int main(void) { return a_go(3) + b_go(4) - 14; }\n";

/// Two identical throwing functions, so `--icf=all` folds them and both FDEs
/// name one address.
const ICF_SRC: &[u8] = b"struct Guard { ~Guard(); };\n\
    int f1(int x) { Guard g; if (x < 0) { throw 1; } return x + 7; }\n\
    int f2(int x) { Guard g; if (x < 0) { throw 1; } return x + 7; }\n\
    Guard::~Guard() {}\n\
    int main(void) { return f1(1) + f2(2) - 17; }\n";

/// A COMDAT duplicate leaves one row per PC.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_comdat_duplicate_leaves_one_row_per_pc() {
    let Some(dir) = workdir("comdat") else {
        return;
    };
    let Some(bytes) = link_comdat(&dir) else {
        return;
    };
    assert_unique(&bytes, "a retired COMDAT copy's FDE");
    let _ = fs::remove_dir_all(&dir);
}

/// So does an identical-code fold, which is the case lld's comment names.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_icf_fold_leaves_one_row_per_pc() {
    let Some(dir) = workdir("icf") else {
        return;
    };
    let Some(bytes) = link_one(&dir, ICF_SRC, "icf", IcfMode::All) else {
        return;
    };
    assert_unique(&bytes, "a folded function's second FDE");
    let _ = fs::remove_dir_all(&dir);
}

/// And the table is still a usable search index: sorted, and covering every
/// throwing function the image kept.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_table_is_sorted_and_not_empty() {
    let Some(dir) = workdir("sorted") else {
        return;
    };
    let Some(bytes) = link_comdat(&dir) else {
        return;
    };
    let pcs = table_pcs(&bytes);
    assert_ne!(pcs.len(), 0, "the fixture must produce unwind rows");
    assert!(
        pcs.windows(2).all(|w| w[0] < w[1]),
        "the table is binary-searched, so it must be strictly increasing: \
         {pcs:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Every row names a distinct PC.
fn assert_unique(bytes: &[u8], what: &str) {
    let pcs = table_pcs(bytes);
    let mut sorted = pcs.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        pcs.len(),
        "two rows for one PC means the unwinder picks a handler by where the \
         binary search lands ({what}); the table is {pcs:?}"
    );
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang++").is_none() {
        eprintln!("skipping eh-frame-hdr-unique {prefix}: no clang++");
        return None;
    }
    if cxx_runtime().is_none() {
        eprintln!("skipping eh-frame-hdr-unique {prefix}: no libstdc++");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_ehunique_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds and links the two-translation-unit COMDAT fixture.
fn link_comdat(dir: &Path) -> Option<Vec<u8>> {
    fs::write(dir.join("h.hpp"), HEADER).ok()?;
    let a = compile(dir, A_SRC, "a")?;
    let b = compile(dir, B_SRC, "b")?;
    link_objs(dir, &[a, b], "comdat", IcfMode::None)
}

/// Builds and links a single-source fixture.
fn link_one(
    dir: &Path,
    src: &[u8],
    stem: &str,
    icf: IcfMode,
) -> Option<Vec<u8>> {
    let obj = compile(dir, src, stem)?;
    link_objs(dir, &[obj], stem, icf)
}

/// Compiles one C++ translation unit with one section per function.
fn compile(dir: &Path, src: &[u8], stem: &str) -> Option<PathBuf> {
    let clangxx = which("clang++")?;
    let src_path = dir.join(format!("{stem}.cpp"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clangxx)
        .args([
            "--target=x86_64-linux-gnu",
            "-O1",
            "-ffunction-sections",
            "-c",
        ])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping eh-frame-hdr-unique: clang++ cannot build it");
        return None;
    }
    Some(obj)
}

/// Links the objects against the C++ runtime and libc.
fn link_objs(
    dir: &Path,
    objs: &[PathBuf],
    stem: &str,
    icf: IcfMode,
) -> Option<Vec<u8>> {
    let interp = common::interpreter()?;
    let mut paths =
        vec![common::crt_file("Scrt1.o")?, common::crt_file("crti.o")?];
    paths.extend(objs.iter().cloned());
    paths.push(cxx_runtime()?);
    // `_Unwind_Resume` lives in `libgcc_s`, which the C++ runtime names
    // undefined; without it the link is underlinked.
    paths.push(common::libgcc_s_so()?);
    paths.push(common::libc_so()?);
    paths.push(common::crt_file("crtn.o")?);
    let out = dir.join(stem);
    let res = link_dyn_exec(
        &paths,
        &out,
        b"_start",
        interp.as_slice(),
        false,
        icf,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

/// The C++ runtime as a real shared object.
///
/// The `libstdc++.so` beside the compiler is a GNU linker script on this
/// distribution. The driver expands those; this test drives the library, which
/// takes a settled input list, so the versioned object is named directly.
fn cxx_runtime() -> Option<PathBuf> {
    ["/usr/lib64/libstdc++.so.6", "/usr/lib/libstdc++.so.6"]
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
}

// --- readers ---------------------------------------------------------------

/// The covered PCs of `.eh_frame_hdr`'s search table, as stored (`datarel`
/// offsets, which are monotonic in the addresses they encode).
fn table_pcs(bytes: &[u8]) -> Vec<i32> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(shdr) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".eh_frame_hdr")
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(shdr) else {
        return Vec::new();
    };
    let Some(count) = data
        .get(8..12)
        .and_then(|c| <[u8; 4]>::try_from(c).ok())
        .map(u32::from_le_bytes)
    else {
        return Vec::new();
    };
    (0..count as usize)
        .filter_map(|i| {
            let at = 12 + i * 8;
            data.get(at..at + 4)
                .and_then(|c| <[u8; 4]>::try_from(c).ok())
                .map(i32::from_le_bytes)
        })
        .collect()
}
