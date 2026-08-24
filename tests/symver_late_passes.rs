//! The late passes ask the symbol table with stemmed names too.
//!
//! Resolution interns every versioned spelling under its stem, but the GC
//! walk, the archive indices and the CIE personality lookup used to ask with
//! the raw spelling: a liveness edge through a `foo@VERS_1` reference missed
//! the interned `foo` and the defining section was swept while still
//! referenced, and an `ar` index keying `foo@@VERS_1` never offered the
//! member to a plain `foo` reference.
//!
//! One object spelling a name at two distinct places -- `foo@VERS_1` beside
//! `foo@@VERS_2`, the compat-symbol pattern -- is refused loudly: this
//! linker emits no VERDEF of its own, so it cannot keep the two apart, and
//! silently dropping one implementation is worse than an error.
//!
//! Gated on `clang` (and `ar` for the archive case); when missing the tests
//! print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{archive_tool, which};
use xold::{error::Error, icf::IcfMode, linker::link_to};

mod common;

/// The definition, in its own section so `--gc-sections` can sweep it:
/// `.symver` renames the implementation to `foo@@VERS_1`.
const DEF_SRC: &str = "    .section .text.impl,\"ax\",@progbits\n\
     .globl foo_impl\n\
     .symver foo_impl,foo@@VERS_1\n\
     foo_impl:\n    movl $42, %eax\n    ret\n";

/// The consumer names the import with its version: `.symver` rewrites the
/// undefined reference to `foo@VERS_1`, so the relocation's symbol row
/// carries the suffixed spelling the late passes must stem.
const START_SRC: &str = "    .text\n    .globl _start\n\
     .symver foo,foo@VERS_1\n\
     _start:\n\
     call foo\n movl %eax, %edi\n movl $60, %eax\n syscall\n";

/// Two implementations folded under one stem: the compat pattern this
/// linker refuses.
const TWO_VERSIONS_SRC: &str = "    .text\n\
     .globl old_impl\n\
     .symver old_impl,foo@VERS_1\n\
     old_impl:\n    movl $1, %eax\n    ret\n\
     .globl new_impl\n\
     .symver new_impl,foo@@VERS_2\n\
     new_impl:\n    movl $2, %eax\n    ret\n\
     .globl _start\n\
     _start:\n    movl $60, %eax\n    syscall\n";

/// `--gc-sections` keeps the section a version-suffixed reference binds:
/// the liveness edge asks for the stem, exactly as resolution does.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn gc_keeps_a_versioned_definition_the_suffix_names() {
    let dir = workdir("gc");
    let Some(def) = assemble(DEF_SRC, &dir.join("svl_def.o"), &dir) else {
        eprintln!("skipping symver gc test: host toolchain unavailable");
        return;
    };
    let Some(start) = assemble(START_SRC, &dir.join("svl_start.o"), &dir)
    else {
        eprintln!("skipping symver gc test: host toolchain unavailable");
        return;
    };
    let prog = dir.join("svl_prog");
    link_to(&[def, start], &prog, b"_start", true, IcfMode::None, false)
        .expect("the versioned reference keeps the defining section alive");
    assert_returns_42(&prog);
    let _ = fs::remove_dir_all(&dir);
}

/// An `ar` index keys the raw `foo@@VERS_1` spelling; the member must still
/// be offered to a reference that resolves as `foo`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_archive_member_defining_a_versioned_name_is_extracted() {
    let dir = workdir("ar");
    let Some(def) = assemble(DEF_SRC, &dir.join("svm_def.o"), &dir) else {
        eprintln!("skipping symver archive test: toolchain unavailable");
        return;
    };
    let Some(start) = assemble(START_SRC, &dir.join("svm_start.o"), &dir)
    else {
        eprintln!("skipping symver archive test: toolchain unavailable");
        return;
    };
    let Some(lib) = archive(&dir, &def) else {
        eprintln!("skipping symver archive test: ar unavailable");
        return;
    };
    let prog = dir.join("svm_prog");
    link_to(&[start, lib], &prog, b"_start", false, IcfMode::None, false)
        .expect("the archive member defining foo@@VERS_1 must extract");
    assert_returns_42(&prog);
    let _ = fs::remove_dir_all(&dir);
}

/// One caller of the versioned name, duplicated verbatim across two files:
/// ICF must resolve the suffixed reference through the table and fold them.
const CALLER_SRC: &str = "    .section .text.CALLER,\"ax\",@progbits\n\
     .globl CALLER\n\
     .symver foo,foo@VERS_1\n\
     CALLER:\n    call foo\n    ret\n";

/// The entry keeps both callers referenced so only ICF can unify them.
const ICF_START_SRC: &str = "    .text\n    .globl _start\n_start:\n\
     call f1\n call f2\n movl $60, %eax\n syscall\n";

/// ICF folds two identical callers of a version-suffixed name: the
/// equivalence class comes from the resolved definition, which the fold
/// can only see by asking with the stem.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn icf_folds_identical_callers_of_a_versioned_name() {
    let dir = workdir("icf");
    let f1_src = CALLER_SRC.replace("CALLER", "f1");
    let f2_src = CALLER_SRC.replace("CALLER", "f2");
    let Some(def) = assemble(DEF_SRC, &dir.join("svi_def.o"), &dir) else {
        eprintln!("skipping symver icf test: host toolchain unavailable");
        return;
    };
    let Some(f1) = assemble(&f1_src, &dir.join("svi_f1.o"), &dir) else {
        eprintln!("skipping symver icf test: host toolchain unavailable");
        return;
    };
    let Some(f2) = assemble(&f2_src, &dir.join("svi_f2.o"), &dir) else {
        eprintln!("skipping symver icf test: host toolchain unavailable");
        return;
    };
    let Some(start) = assemble(ICF_START_SRC, &dir.join("svi_start.o"), &dir)
    else {
        eprintln!("skipping symver icf test: host toolchain unavailable");
        return;
    };
    let prog = dir.join("svi_prog");
    link_to(
        &[def, f1, f2, start],
        &prog,
        b"_start",
        false,
        IcfMode::All,
        false,
    )
    .expect("the callers link");
    let bytes = fs::read(&prog).expect("image is readable");
    let (a, b) = (
        symbol_value(&bytes, b"f1").expect("f1 defined"),
        symbol_value(&bytes, b"f2").expect("f2 defined"),
    );
    assert_eq!(a, b, "identical callers of one resolved name must fold");
    let _ = fs::remove_dir_all(&dir);
}

/// Two versioned definitions of one stem at distinct places in one file are
/// refused: without VERDEF emission the output cannot keep them apart.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn two_versioned_definitions_of_one_stem_are_refused() {
    let dir = workdir("two");
    let Some(obj) = assemble(TWO_VERSIONS_SRC, &dir.join("svt_two.o"), &dir)
    else {
        eprintln!("skipping symver refusal test: toolchain unavailable");
        return;
    };
    let out = dir.join("svt_prog");
    let res = link_to(&[obj], &out, b"_start", false, IcfMode::None, false);
    assert!(
        matches!(&res, Err(Error::DuplicateSymbol(_))),
        "two implementations under one stem must be refused: {:?}",
        res.err()
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures ---------------------------------------------------------------

fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_symver_late_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Assembles `src` to `obj` with the host clang. Returns `None` when the
/// toolchain is unavailable; a fixture that fails to assemble panics.
fn assemble(src: &str, obj: &Path, dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let stem = obj.file_stem()?.to_str().unwrap_or("unit");
    let file = dir.join(format!("{stem}.S"));
    fs::write(&file, src).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c", "-o"])
        .arg(obj)
        .arg(&file)
        .status()
        .ok()?
        .success();
    assert!(built, "fixture {stem} must assemble");
    Some(obj.to_path_buf())
}

/// The `st_value` of a symbol in the output's `.symtab`.
fn symbol_value(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = xold::elf::ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab
        .iter()
        .find(|s| symtab.name(s) == name)
        .map(|s| s.st_value.get())
}

/// Executes the fixture where its ELF architecture is native. Other hosts
/// verify the same result from the linked `foo` body instead of attempting to
/// launch a foreign executable.
fn assert_returns_42(prog: &Path) {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        let status = Command::new(prog).status().expect("the program runs");
        assert_eq!(status.code(), Some(42), "the linked program returns 42");
    }
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let bytes = fs::read(prog).expect("image is readable");
        let addr = symbol_value(&bytes, b"foo")
            .expect("the versioned definition is present under its stem");
        assert_eq!(
            image_bytes_at(&bytes, addr, 6).as_deref(),
            Some(b"\xb8\x2a\0\0\0\xc3".as_slice()),
            "the surviving or extracted definition returns 42"
        );
    }
}

/// `len` linked bytes at virtual address `addr`.
#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn image_bytes_at(bytes: &[u8], addr: u64, len: usize) -> Option<Vec<u8>> {
    let obj = xold::elf::ObjectFile::parse(bytes).ok()?;
    for sec in obj.sections() {
        let base = sec.sh_addr.get();
        if base == 0 || addr < base || addr >= base + sec.sh_size.get() {
            continue;
        }
        let data = obj.section_data(sec).ok()?;
        let at = usize::try_from(addr - base).ok()?;
        return data.get(at..at.checked_add(len)?).map(<[u8]>::to_vec);
    }
    None
}

/// Packs `member` into a fresh GNU archive with a real `ar` symbol index.
fn archive(dir: &Path, member: &Path) -> Option<PathBuf> {
    let ar = archive_tool()?;
    let lib = dir.join("libsvm.a");
    let built = Command::new(ar)
        .arg("rcs")
        .arg(&lib)
        .arg(member)
        .status()
        .ok()?
        .success();
    assert!(built, "fixture archive must build");
    Some(lib)
}
