//! Identical code that shares a string literal must fold across files.
//!
//! Every translation unit carries its own copy of each string literal it
//! uses, in its own `.rodata.str1.1` input section. Two identical functions
//! in two objects therefore reference the same string through two different
//! input sections, and ICF compared relocation targets by that `(file,
//! section)` pair: the classes stayed apart and the functions never folded,
//! even though the merge pass deduplicates both strings into one pool byte
//! range either way. The output carried two copies of code that differ only
//! in which identical copy of an identical string they load from.
//!
//! lld finalises the merge sections before it runs ICF
//! (`lld/ELF/Driver.cpp`) and compares a relocation into a
//! mergeable section by its offset in the parent output section
//! (`lld/ELF/ICF.cpp`) -- the pool offset, where the copies
//! meet. The same rule, applied here, makes the two functions name the same
//! target and fold.
//!
//! Gated on `clang`; if it is missing the test prints a note and returns.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_shared};

mod common;

/// One half of the fixture: a function whose only data reference is a string
/// literal, kept alive by a caller. The index is a runtime value so the
/// literal is not constant-folded away before it reaches the object.
const A_SRC: &[u8] = b"__attribute__((noinline, used)) static int f(int i) {\n\
    return \"a mergeable literal\"[i];\n\
    }\n\
    int keep_a(int i) { return f(i) + 1; }\n";

/// The other half: a byte-identical `f` in its own object, with a different
/// caller so only the string loaders -- not the callers -- may fold.
const B_SRC: &[u8] = b"__attribute__((noinline, used)) static int f(int i) {\n\
    return \"a mergeable literal\"[i];\n\
    }\n\
    int keep_b(int i) { return f(i) + f(i + 2); }\n";

/// Under `--icf=all` the two copies of `f` land on one address, because the
/// string each loads is the same pool content after the merge pass.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn identical_code_sharing_a_literal_folds() {
    let Some(dir) = workdir("fold") else {
        return;
    };
    let Some(objects) = compile(&dir) else {
        return;
    };
    let out = dir.join("folded.so");
    let res = link_shared(
        &objects,
        &out,
        Some(b"libicfm.so"),
        false,
        IcfMode::All,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let image = fs::read(&out).expect("read the image");
    let values = f_values(&image);
    assert_eq!(values.len(), 2, "both local copies of f are in .symtab");
    assert_eq!(
        values[0], values[1],
        "two functions that load the same pooled string are one address"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: with folding off the two copies stay apart, so the fold
/// above is the pass's doing and not a fixture that emitted one function.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn without_icf_they_stay_apart() {
    let Some(dir) = workdir("plain") else {
        return;
    };
    let Some(objects) = compile(&dir) else {
        return;
    };
    let out = dir.join("plain.so");
    let res = link_shared(
        &objects,
        &out,
        Some(b"libicfm.so"),
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let image = fs::read(&out).expect("read the image");
    let values = f_values(&image);
    assert_eq!(values.len(), 2, "both local copies of f are in .symtab");
    assert_ne!(
        values[0], values[1],
        "without folding each object keeps its own copy"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping icf-merge-fold {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_icf_merge_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles both halves to separate objects.
fn compile(dir: &Path) -> Option<Vec<PathBuf>> {
    let clang = which("clang")?;
    let mut objects = Vec::new();
    for (name, src) in [("a.c", A_SRC), ("b.c", B_SRC)] {
        let src_path = dir.join(name);
        let obj = dir.join(name.replace(".c", ".o"));
        fs::write(&src_path, src).ok()?;
        let built = Command::new(&clang)
            .args([
                "--target=x86_64-linux-gnu",
                "-fPIC",
                "-ffunction-sections",
                "-fdata-sections",
                "-c",
            ])
            .arg(&src_path)
            .arg("-o")
            .arg(&obj)
            .status()
            .ok()?
            .success();
        if !built {
            eprintln!("skipping icf-merge-fold: clang cannot build {name}");
            return None;
        }
        objects.push(obj);
    }
    Some(objects)
}

// --- readers ---------------------------------------------------------------

/// The addresses of the local symbols named `f`, in table order.
fn f_values(image: &[u8]) -> Vec<u64> {
    let obj = ObjectFile::parse(image).expect("valid ELF");
    let symtab = obj.symbol_table().ok().flatten().expect(".symtab");
    symtab
        .iter()
        .filter(|s| symtab.name(s) == b"f")
        .map(|s| s.st_value.get())
        .collect()
}
