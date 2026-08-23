//! A symbol whose section index this linker cannot read is refused.
//!
//! `st_shndx` is 16 bits, and three of its reserved values carry their own
//! meaning: `SHN_UNDEF`, `SHN_ABS`, `SHN_COMMON`. Everything else in the
//! reserved range is not a section index, and the most common of them is
//! `SHN_XINDEX` -- the marker that says the real index is too large for the
//! field and lives in a `SHT_SYMTAB_SHNDX` table beside the symbol table. A
//! C++ translation unit built with `-ffunction-sections` reaches 0xff00
//! sections without trying.
//!
//! xold does not read that table, and took the field raw. The symbol was then
//! treated as defined in section 0xffff, which is never placed, so the address
//! lookup missed, fell back to zero, and the symbol resolved to `st_value`
//! bytes from the start of the image. Every reference to it pointed into the
//! ELF header. Nothing said so.
//!
//! Refusing is what the linker owes an input it cannot represent. lld reads
//! the table instead, which is the better answer and the one to implement if
//! such objects need to link; until then the failure is loud rather than
//! silent.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_to};

mod common;

/// One definition and one reference to it, so a wrong address for `target`
/// would reach the image rather than being optimised away.
const SRC: &[u8] = b"int target = 7;\n\
    int *reference(void) { return &target; }\n\
    void _start(void) { }\n";

/// The size of one `Elf64_Sym`, and the offset of `st_shndx` inside it.
const SYM_SIZE: usize = 24;
const SHNDX_OFF: usize = 6;

/// `SHN_XINDEX` is refused by name: the input is well formed and this linker
/// simply cannot read where the symbol lives.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_symbol_marked_shn_xindex_is_refused() {
    let Some(dir) = workdir("xindex") else {
        return;
    };
    let Some(obj) = patched(&dir, "xindex", 0xffff) else {
        return;
    };
    let err = link(&dir, &obj, "xindex").expect_err(
        "a symbol whose section index lives in SHT_SYMTAB_SHNDX cannot be \
         placed, so the link must fail rather than resolve it near zero",
    );
    let text = format!("{err}");
    assert!(
        text.contains("SHN_XINDEX"),
        "the refusal must name what it cannot read, got {text:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Any other reserved value is refused too. `0xff00` is `SHN_LORESERVE`
/// itself, the start of the processor- and OS-specific range.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn any_other_reserved_section_index_is_refused() {
    let Some(dir) = workdir("reserved") else {
        return;
    };
    let Some(obj) = patched(&dir, "reserved", 0xff00) else {
        return;
    };
    let err = link(&dir, &obj, "reserved")
        .expect_err("a reserved st_shndx is not a section index");
    assert!(
        matches!(err, xold::Error::Format(_)),
        "the refusal must be a format error, got {err:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: the same object links when `target` keeps its real section
/// index, so the check rejects the reserved values and nothing else.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_unpatched_object_still_links() {
    let Some(dir) = workdir("plain") else {
        return;
    };
    let Some(obj) = compile(&dir) else {
        return;
    };
    let res = link(&dir, &obj, "plain");
    assert!(res.is_ok(), "an ordinary object must link: {:?}", res.err());
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping symbol-xindex {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_xindex_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture.
fn compile(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("xindex.c");
    let obj = dir.join("xindex.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fno-pic", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping symbol-xindex: clang cannot build the fixture");
        return None;
    }
    Some(obj)
}

/// Compiles the fixture and rewrites `target`'s `st_shndx` to `shndx`.
///
/// Producing a genuine `SHN_XINDEX` object would mean emitting 0xff00
/// sections; patching the one field is the same input as far as every reader
/// is concerned, and it is exact.
fn patched(dir: &Path, stem: &str, shndx: u16) -> Option<PathBuf> {
    let obj = compile(dir)?;
    let mut bytes = fs::read(&obj).ok()?;
    let at = shndx_offset(&bytes, b"target")?;
    bytes
        .get_mut(at..at + 2)?
        .copy_from_slice(&shndx.to_le_bytes());
    let out = dir.join(format!("{stem}.o"));
    fs::write(&out, bytes).ok()?;
    Some(out)
}

/// Links `obj` as a static executable.
fn link(dir: &Path, obj: &Path, stem: &str) -> Result<(), xold::Error> {
    link_to(
        std::slice::from_ref(&obj.to_path_buf()),
        &dir.join(stem),
        b"_start",
        false,
        IcfMode::None,
        false,
    )
}

// --- readers ---------------------------------------------------------------

/// The byte offset of `name`'s `st_shndx` field inside the object file.
fn shndx_offset(bytes: &[u8], name: &[u8]) -> Option<usize> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok().flatten()?;
    let index = symtab.iter().position(|s| symtab.name(s) == name)?;
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".symtab")?;
    let base = usize::try_from(shdr.sh_offset.get()).ok()?;
    Some(base + index * SYM_SIZE + SHNDX_OFF)
}
