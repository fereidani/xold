//! An archive placed before a shared library is extracted from first.
//!
//! A shared object on the command line is a definition, and a definition
//! stops archive extraction -- but only a definition that is already in hand
//! when the archive is reached. Command-line order decides which of the two
//! spellings of one library supplies a name, and the decision is made at the
//! position of the *earlier* input: a shared library named after the archive
//! arrives too late, the member is already in the link, and the image keeps
//! the static copy while still recording the library in `DT_NEEDED`.
//!
//! lld works the same way through its symbol lattice: an undefined reference
//! meeting a lazy archive symbol extracts it (`resolve(const LazySymbol &)`),
//! and the shared symbol that arrives later cannot un-extract what is now a
//! defined name. Only the reverse order -- shared library first -- leaves the
//! reference for the loader, which is the case `tests/shared_over_archive.rs`
//! covers.
//!
//! Gated on `clang` and `ar`; if either is missing the test prints a note and
//! returns.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{archive_tool, which};
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_shared};

mod common;

/// The shared spelling of the library: it defines `dual_impl`, returning 1.
const DEP_SRC: &[u8] = b"int dual_impl(void) { return 1; }\n";

/// The static spelling of the same library, returning 2: a distinct value, so
/// the test can tell which copy the image ended up with.
const ARCHIVE_SRC: &[u8] = b"int dual_impl(void) { return 2; }\n";

/// The library under test: it calls the dual-spelled name.
const USER_SRC: &[u8] = b"extern int dual_impl(void);\n\
    int user_read(void) { return dual_impl(); }\n";

/// An archive named before the shared library supplies the name itself.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_archive_named_before_the_shared_library_is_extracted() {
    let Some(dir) = workdir() else {
        return;
    };
    let Some(archive) = build_archive(&dir) else {
        return;
    };
    let Some(dep) = build_dependency(&dir) else {
        return;
    };
    let user_o = dir.join("arord_user.o");
    if compile(USER_SRC, &user_o).is_none() {
        return;
    }

    // The archive comes first: the member is in the link before the shared
    // library is ever read.
    let lib = dir.join("libarorduser.so");
    let linked = link_shared(
        &[user_o, archive, dep],
        &lib,
        Some(b"libarorduser.so"),
        false,
        IcfMode::None,
        false,
    );
    assert!(
        linked.is_ok(),
        "xold -shared link must succeed: {:?}",
        linked.err()
    );
    assert!(
        defines_locally(&lib, b"dual_impl"),
        "the archive was named first, so its member is the definition the \
         link keeps; suppressing it for a library that came later leaves \
         the reference to a definition the image also carries statically"
    );
    assert!(
        names_dependency(&lib, b"libd.so"),
        "a shared library on the command line is recorded in DT_NEEDED \
         whether or not it supplied any code"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures ---------------------------------------------------------------

fn workdir() -> Option<PathBuf> {
    if which("clang").is_none() || archive_tool().is_none() {
        eprintln!("skipping archive-order test: toolchain missing");
        return None;
    }
    let dir =
        std::env::temp_dir().join(format!("xold_arord_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the static spelling of the library.
fn build_archive(dir: &Path) -> Option<PathBuf> {
    let ar = archive_tool()?;
    let obj = dir.join("arord_arch.o");
    compile(ARCHIVE_SRC, &obj)?;
    let archive = dir.join("libd.a");
    let ok = Command::new(ar)
        .arg("rcs")
        .arg(&archive)
        .arg(&obj)
        .status()
        .ok()?
        .success();
    ok.then_some(archive)
}

/// Builds the shared spelling of the library with the host toolchain.
fn build_dependency(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let obj = dir.join("arord_dep.o");
    compile(DEP_SRC, &obj)?;
    let so = dir.join("libd.so");
    let ok = Command::new(clang)
        .arg("-shared")
        .arg("-o")
        .arg(&so)
        .arg(&obj)
        .arg("-Wl,-soname,libd.so")
        .status()
        .ok()?
        .success();
    ok.then_some(so)
}

/// Compiles `src` with the host clang as position-independent code.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).ok()?;
    Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIC", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success()
        .then_some(())
}

// --- readers ----------------------------------------------------------------

/// Whether `name` is defined (not `SHN_UNDEF`) in the image's symbol table.
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

/// Whether the image's `DT_NEEDED` entries name `soname`.
fn names_dependency(path: &Path, soname: &[u8]) -> bool {
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    needed(&bytes).iter().any(|n| n == soname)
}

/// `DT_NEEDED`'s value is 1; the value is an offset into `.dynstr`.
const DT_NEEDED: i64 = 1;

/// The `DT_NEEDED` names of a linked image, in table order.
fn needed(bytes: &[u8]) -> Vec<Vec<u8>> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(dynamic) = section(&obj, b".dynamic") else {
        return Vec::new();
    };
    let Some(dynstr) = section(&obj, b".dynstr") else {
        return Vec::new();
    };
    dynamic
        .as_chunks::<16>()
        .0
        .iter()
        .filter_map(|c| {
            let d_tag = i64::from_le_bytes(<[u8; 8]>::try_from(&c[0..8]).ok()?);
            if d_tag != DT_NEEDED {
                return None;
            }
            let off = usize::try_from(u64::from_le_bytes(
                <[u8; 8]>::try_from(&c[8..16]).ok()?,
            ))
            .ok()?;
            let tail = dynstr.get(off..)?;
            let end = tail.iter().position(|b| *b == 0)?;
            Some(tail[..end].to_vec())
        })
        .collect()
}

/// The bytes of a named section.
fn section(obj: &ObjectFile<'_>, name: &[u8]) -> Option<Vec<u8>> {
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == name)?;
    obj.section_data(shdr).ok().map(<[u8]>::to_vec)
}
