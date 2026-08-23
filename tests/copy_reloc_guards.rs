//! A protected or hidden data export cannot be copied into an executable.
//!
//! A copy relocation is how an executable takes over a dependency's data
//! object: it owns the storage in its own `.bss`, and the loader copies the
//! initial bytes there. That only works if every reference in the process can
//! be made to agree, and two visibilities say it cannot.
//!
//! Protected is the dependency's promise that its own references bind to its
//! own definition. Copy the object and the two halves of the process read
//! different storage for one name. Hidden is not reachable from outside the
//! dependency at all.
//!
//! The candidate test asked only whether the export was a placed `STT_OBJECT`;
//! `DynExport` recorded no visibility to ask about. A protected object got a
//! copy slot -- and once the visibility test was added, skipping quietly was
//! no better: the reference then resolved to the address the dependency
//! happened to be linked at, which is not where it will be loaded, so the
//! image was wrong with no relocation to show for it. lld stops with "cannot
//! preempt symbol", in `canDefineSymbolInExecutable`, and so does this.
//!
//! A zero-sized export is refused just as loudly: a slot of no stated length
//! would take over the name with nothing behind it, and skipping it quietly
//! binds the reference to the address the dependency was linked at. lld
//! stops in `addCopyRelSymbol` ("cannot create a copy relocation for
//! symbol"), and so does this.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{
    elf::ObjectFile,
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
    reloc::x86_64::R_X86_64_COPY,
};

mod common;

/// One protected object and one ordinary one, at the same visibility except
/// for the attribute under test.
const LIB_SRC: &[u8] =
    b"__attribute__((visibility(\"protected\"))) int prot_obj = 5;\n\
    int plain_obj = 7;\n";

/// A reference to the protected object from a non-position-independent
/// executable, which is what asks for a copy relocation.
const PROT_SRC: &[u8] = b"extern int prot_obj;\n\
    int read_it(void) { return prot_obj; }\n\
    void _start(void) { }\n";

const PLAIN_SRC: &[u8] = b"extern int plain_obj;\n\
    int read_it(void) { return plain_obj; }\n\
    void _start(void) { }\n";

/// Referencing a protected data object ends the link, naming it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_protected_export_cannot_be_copied() {
    let Some(dir) = workdir("protected") else {
        return;
    };
    let Some(lib) = library(&dir) else {
        return;
    };
    let Some(obj) = compile(&dir, PROT_SRC, "prot") else {
        return;
    };
    let err = link(&dir, &obj, &lib, "prot").expect_err(
        "a protected object cannot be taken over, and resolving the reference \
         anyway binds it to the address the dependency was linked at",
    );
    let text = format!("{err}");
    assert!(
        text.contains("prot_obj"),
        "the refusal must name the symbol, got {text:?}"
    );
    assert!(
        text.contains("preempt"),
        "and say what cannot be done, got {text:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: an ordinary export from the same library is still copied, so
/// the refusal is about the visibility and not about copy relocations.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_default_export_is_still_copied() {
    let Some(dir) = workdir("default") else {
        return;
    };
    let Some(lib) = library(&dir) else {
        return;
    };
    let Some(obj) = compile(&dir, PLAIN_SRC, "plain") else {
        return;
    };
    let out = dir.join("plain");
    let res = link(&dir, &obj, &lib, "plain");
    assert!(res.is_ok(), "an ordinary import links: {:?}", res.err());
    let bytes = fs::read(&out).expect("read output");
    assert_eq!(
        copy_relocs(&bytes),
        1,
        "the ordinary data import still takes its copy slot"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The dependency's own definition of a data object with no `.size`
/// directive: bytes sit at the label, but the export declares no extent.
const ZERO_LIB_SRC: &[u8] = b"    .data\n\
     .globl zed\n\
     .type zed,@object\n\
zed:\n    .long 7\n";

const ZERO_REF_SRC: &[u8] = b"extern int zed;\n\
    int read_it(void) { return zed; }\n\
    void _start(void) { }\n";

/// Referencing a zero-sized data export ends the link, naming it: a copy
/// slot needs a length.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_zero_sized_export_cannot_be_copied() {
    let Some(dir) = workdir("zerosized") else {
        return;
    };
    let Some(lib_obj) = assemble(&dir, ZERO_LIB_SRC, "zlib") else {
        return;
    };
    let lib = dir.join("libzed.so");
    let res = link_shared(
        std::slice::from_ref(&lib_obj),
        &lib,
        Some(b"libzed.so"),
        false,
        IcfMode::None,
        false,
    );
    assert!(
        res.is_ok(),
        "the fixture library must link: {:?}",
        res.err()
    );
    assert_eq!(
        export_size(&lib, b"zed"),
        Some(0),
        "premise: the dependency must export zed with no stated size"
    );
    let Some(obj) = compile(&dir, ZERO_REF_SRC, "zref") else {
        return;
    };
    let err = link(&dir, &obj, &lib, "zref").expect_err(
        "a zero-sized object cannot be taken over: the slot would own the \
         name with nothing behind it",
    );
    let text = format!("{err}");
    assert!(
        text.contains("zed") && text.contains("copy relocation"),
        "the refusal must name the symbol and the mechanism, got {text:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping copy-reloc-guards {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_copyguard_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the dependency and links it as a shared object.
fn library(dir: &Path) -> Option<PathBuf> {
    let obj = compile_with(dir, LIB_SRC, "lib", "-fPIC")?;
    let lib = dir.join("libprot.so");
    let res = link_shared(
        std::slice::from_ref(&obj),
        &lib,
        Some(b"libprot.so"),
        false,
        IcfMode::None,
        false,
    );
    assert!(
        res.is_ok(),
        "the fixture library must link: {:?}",
        res.err()
    );
    Some(lib)
}

/// Compiles a non-position-independent referencing object, which is what asks
/// for a copy relocation rather than a GOT slot.
fn compile(dir: &Path, src: &[u8], stem: &str) -> Option<PathBuf> {
    compile_with(dir, src, stem, "-fno-pic")
}

/// Assembles a fixture, for the shapes C cannot state (a `.type` object
/// with no `.size`).
fn assemble(dir: &Path, src: &[u8], stem: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src_path = dir.join(format!("{stem}.S"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    assert!(built, "fixture {stem} must assemble");
    Some(obj)
}

/// The `st_size` of the named export in a library's `.dynsym`.
fn export_size(lib: &Path, name: &[u8]) -> Option<u64> {
    let bytes = fs::read(lib).ok()?;
    let obj = ObjectFile::parse(&bytes).ok()?;
    let dynsym = obj.dynamic_symbols().ok()??;
    dynsym
        .iter()
        .find(|s| dynsym.name(s) == name)
        .map(|s| s.st_size.get())
}

fn compile_with(
    dir: &Path,
    src: &[u8],
    stem: &str,
    model: &str,
) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src_path = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", model, "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping copy-reloc-guards: clang cannot build it");
        return None;
    }
    Some(obj)
}

/// Links the referencing object against the library.
fn link(
    dir: &Path,
    obj: &Path,
    lib: &Path,
    stem: &str,
) -> Result<(), xold::Error> {
    let interp = interpreter().unwrap_or_else(|| b"/lib64/ld.so".to_vec());
    link_dyn_exec(
        &[obj.to_path_buf(), lib.to_path_buf()],
        &dir.join(stem),
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    )
}

// --- readers ---------------------------------------------------------------

/// How many `R_X86_64_COPY` rows the image carries.
fn copy_relocs(bytes: &[u8]) -> usize {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return 0;
    };
    let Some(shdr) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".rela.dyn")
    else {
        return 0;
    };
    let Ok(data) = obj.section_data(shdr) else {
        return 0;
    };
    data.as_chunks::<24>()
        .0
        .iter()
        .filter(|c| {
            <[u8; 8]>::try_from(&c[8..16]).is_ok_and(|w| {
                #[allow(clippy::cast_possible_truncation)]
                let r_type = u64::from_le_bytes(w) as u32;
                r_type == R_X86_64_COPY
            })
        })
        .count()
}
