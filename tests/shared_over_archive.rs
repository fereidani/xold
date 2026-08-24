//! A shared dependency is a definition, so it stops archive extraction.
//!
//! The gcc driver names both spellings of the same library on nearly every
//! link: `libgcc_s.so.1` beside `libgcc.a`, `libstdc++.so` beside its static
//! twin. Only one of the two may supply the code. If the archive member is
//! extracted anyway, the image carries a static copy of a unit it also records
//! a `DT_NEEDED` for, and whatever state that unit owns -- an unwinder's
//! registry, a lazily-built table, a `once` flag -- exists twice in one
//! process with each copy blind to the other.
//!
//! xold routed `ET_DYN` inputs into the dependency tables only, so a name a
//! shared object on the command line defined still read as undefined to the
//! archive pass and pulled the member. lld cannot reach that state: a
//! `SharedSymbol` is its own class in the lattice, so `resolve(LazySymbol)`
//! extracts nothing over it, and `resolve(SharedSymbol)` overwrites a
//! still-lazy symbol rather than extracting it -- the answer is the same
//! whichever of the two came first on the command line.
//!
//! Non-default visibility is the exception both linkers keep: a hidden
//! reference must be satisfied inside this image, so the archive member is
//! still the only thing that can answer it.
//!
//! Gated on `clang` and `ar`; if either is missing the tests print a note and
//! return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{archive_tool, which};
use xold::{icf::IcfMode, linker::link_shared};

mod common;

/// The shared spelling of the library: it defines `dual_impl`.
const DEP_SRC: &[u8] = b"int dual_impl(void) { return 1; }\n";

/// The static spelling of the same library. A distinct return value, so the
/// test can tell which copy the image ended up with without reading tables.
const ARCHIVE_SRC: &[u8] = b"int dual_impl(void) { return 2; }\n";

/// A second archive member nothing references, to confirm the fixpoint still
/// leaves unreferenced members alone.
const ARCHIVE_SPARE_SRC: &[u8] = b"int spare_impl(void) { return 3; }\n";

/// The library under test: it calls the dual-spelled name.
const USER_SRC: &[u8] = b"extern int dual_impl(void);\n\
    int user_read(void) { return dual_impl(); }\n";

/// The same call through a reference the translation unit keeps hidden. No
/// loader will bind it to the dependency, so the archive member must still be
/// extracted for it.
const HIDDEN_USER_SRC: &[u8] =
    b"__attribute__((visibility(\"hidden\"))) int dual_impl(void);\n\
    int user_read(void) { return dual_impl(); }\n";

/// A shared definition on the command line keeps the archive member out.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_shared_definition_stops_the_archive_from_being_extracted() {
    let Some(dir) = workdir("dso") else {
        return;
    };
    let Some(lib) = link_user(&dir, USER_SRC, true) else {
        return;
    };
    assert!(
        !defines_locally(&lib, b"dual_impl"),
        "the dependency defines dual_impl, so no archive member may be \
         extracted over it; the image would carry two copies of the unit"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Without the shared spelling, the archive member is still the answer: the
/// rule turns off exactly one path and leaves ordinary extraction alone.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn without_the_shared_definition_the_archive_still_supplies_it() {
    let Some(dir) = workdir("nodso") else {
        return;
    };
    let Some(lib) = link_user(&dir, USER_SRC, false) else {
        return;
    };
    assert!(
        defines_locally(&lib, b"dual_impl"),
        "with no dependency defining it, the archive member is what resolves \
         the reference"
    );
    assert!(
        !defines_locally(&lib, b"spare_impl"),
        "a member nothing references is still not pulled"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A hidden reference is private to this image, so a dependency cannot answer
/// it and the archive member is extracted as before.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_hidden_reference_still_extracts_the_archive_member() {
    let Some(dir) = workdir("hidden") else {
        return;
    };
    let Some(lib) = link_user(&dir, HIDDEN_USER_SRC, true) else {
        return;
    };
    assert!(
        defines_locally(&lib, b"dual_impl"),
        "a hidden reference must be satisfied inside this image, so the \
         archive member is the only thing that can supply it"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() || archive_tool().is_none() {
        eprintln!("skipping shared-over-archive {prefix}: toolchain missing");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_dsoarch_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Links `user_src` against the archive, and against the shared dependency too
/// when `with_dep` is set. Returns the produced library's path.
fn link_user(dir: &Path, user_src: &[u8], with_dep: bool) -> Option<PathBuf> {
    let archive = build_archive(dir)?;
    let user_o = dir.join("dsoarch_user.o");
    compile(user_src, &user_o)?;

    let mut inputs = vec![user_o];
    if with_dep {
        inputs.push(build_dependency(dir)?);
    }
    inputs.push(archive);

    let lib = dir.join("libdsoarchuser.so");
    let linked = link_shared(
        &inputs,
        &lib,
        Some(b"libdsoarchuser.so"),
        false,
        IcfMode::None,
        false,
    );
    assert!(
        linked.is_ok(),
        "xold -shared link must succeed: {:?}",
        linked.err()
    );
    Some(lib)
}

/// Builds the shared spelling of the library with the host toolchain.
fn build_dependency(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let obj = dir.join("dsoarch_dep.o");
    compile(DEP_SRC, &obj)?;
    let so = dir.join("libdsoarch.so");
    let ok = Command::new(clang)
        .arg("-shared")
        .arg("-o")
        .arg(&so)
        .arg(&obj)
        .arg("-Wl,-soname,libdsoarch.so")
        .status()
        .ok()?
        .success();
    ok.then_some(so)
}

/// Builds the static spelling: two members, one of them unreferenced.
fn build_archive(dir: &Path) -> Option<PathBuf> {
    let ar = archive_tool()?;
    let impl_o = dir.join("dsoarch_impl.o");
    let spare_o = dir.join("dsoarch_spare.o");
    compile(ARCHIVE_SRC, &impl_o)?;
    compile(ARCHIVE_SPARE_SRC, &spare_o)?;
    let archive = dir.join("libdsoarch.a");
    let ok = Command::new(ar)
        .arg("rcs")
        .arg(&archive)
        .arg(&impl_o)
        .arg(&spare_o)
        .status()
        .ok()?
        .success();
    ok.then_some(archive)
}

/// Compiles `src` with the host clang as position-independent code.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).ok()?;
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIC", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    ok.then_some(())
}

// --- readers ---------------------------------------------------------------

/// Whether the linked image carries its own definition of `name`, as opposed
/// to importing it. A `.symtab` row with a section index other than
/// `SHN_UNDEF` is one; that is exactly what an extracted archive member
/// leaves behind.
fn defines_locally(path: &Path, name: &[u8]) -> bool {
    use xold::elf::{ObjectFile, constants::SHN_UNDEF};

    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    let Ok(obj) = ObjectFile::parse(&bytes) else {
        return false;
    };
    let Ok(Some(symtab)) = obj.symbol_table() else {
        return false;
    };
    symtab
        .syms
        .iter()
        .any(|s| symtab.name(s) == name && s.st_shndx.get() != SHN_UNDEF)
}
