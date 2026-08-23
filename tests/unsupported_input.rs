//! Inputs xold refuses rather than links quietly.
//!
//! Both shapes here are ones a linker can only get wrong: it has no way to
//! produce a correct image, and the failure it would otherwise produce is a
//! running binary rather than a diagnostic. Each test builds the shape by
//! patching a real object, because no compiler emits either.
//!
//! - **`SHT_REL` relocations.** xold applies `SHT_RELA` only: a `REL` entry
//!   keeps its addend in the target bytes, and none of the value computations
//!   reads it back. A section relocated that way used to be treated as having
//!   no relocations at all, so the link succeeded and every call site in it was
//!   left unpatched.
//! - **A mergeable string section with no terminator.** `SHF_STRINGS` promises
//!   NUL-terminated content; a trailing run without a terminator is not a
//!   piece, so it can neither be deduplicated nor copied into a pool.
//!
//! The refusal is at the point of use, not at the parse: an object may carry a
//! `.rel.*` section for a section this link never places, and that costs the
//! link nothing.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{
    elf::{ObjectFile, Relocs},
    input::Input,
    linker::{Link, link_image},
};

mod common;

/// `SHT_REL`, the relocation section type xold does not apply.
const SHT_REL: u32 = 9;
/// `SHT_RELA`, the one it does.
const SHT_RELA: u32 = 4;

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    p
}

/// The byte offset of section header `index` within the file, read from the
/// ELF header's `e_shoff` and `e_shentsize`.
fn shdr_offset(bytes: &[u8], index: usize) -> usize {
    let e_shoff = u64::from_le_bytes(
        bytes[0x28..0x30].try_into().expect("e_shoff is in range"),
    );
    let e_shentsize = u16::from_le_bytes(
        bytes[0x3a..0x3c]
            .try_into()
            .expect("e_shentsize is in range"),
    );
    let at = usize::try_from(e_shoff).expect("section table is addressable");
    at + index * usize::from(e_shentsize)
}

/// Overwrites the `sh_type` of section header `index`.
fn set_sh_type(bytes: &mut [u8], index: usize, ty: u32) {
    let at = shdr_offset(bytes, index) + 4;
    bytes[at..at + 4].copy_from_slice(&ty.to_le_bytes());
}

/// The index of the first section satisfying `pick`.
fn find_section(
    bytes: &[u8],
    pick: impl Fn(&ObjectFile<'_>, &xold::elf::Shdr64) -> bool,
) -> Option<usize> {
    let obj = ObjectFile::parse(bytes).expect("fixture parses");
    obj.sections().iter().position(|s| pick(&obj, s))
}

/// Links `bytes` as the only input, returning the error message on failure.
///
/// The input is passed in memory, so the patched bytes never touch the disk
/// and two of these tests running at once cannot collide over a file name.
fn link_err(name: &str, bytes: &[u8]) -> Option<String> {
    let out = std::env::temp_dir()
        .join(format!("xold_unsupported_{name}_{}", std::process::id()));
    let inputs = [Input::Memory {
        name: std::path::Path::new("patched.o"),
        bytes,
    }];
    let result = link_image(&Link::exec(&inputs, &out));
    let _ = fs::remove_file(&out);
    result.err().map(|e| e.to_string())
}

/// Compiles a translation unit with two string literals into an object with a
/// `SHF_MERGE|SHF_STRINGS` section, or `None` when clang is unavailable.
fn compile_literals(dir: &std::path::Path) -> Option<Vec<u8>> {
    let clang = which("clang")?;
    let src = dir.join("lit.c");
    let obj = dir.join("lit.o");
    fs::write(
        &src,
        b"const char *a(void){ return \"alpha\"; }\n\
          const char *b(void){ return \"bravo\"; }\n" as &[u8],
    )
    .expect("write source");
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-O1", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    ok.then(|| fs::read(&obj).expect("read compiled object"))
}

/// The collapse every pass reads through: a `REL` section is an error, not an
/// empty relocation list.
#[test]
fn rel_relocations_are_an_error_not_an_absence() {
    assert!(matches!(Relocs::None.entries(), Ok(None)));
    let Err(err) = Relocs::Rel.entries() else {
        panic!("SHT_REL must have no readable entries");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("SHT_REL"),
        "the message names the form it refuses: {msg}"
    );
}

/// An object whose `.text` is relocated by an `SHT_REL` section is refused.
/// Before this it linked, and every relocation in `.text` was skipped.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_rel_relocated_section_fails_the_link() {
    let mut bytes = fs::read(fixture("min.o")).expect("read fixture");
    let rela = find_section(&bytes, |obj, s| {
        s.sh_type.get() == SHT_RELA && obj.section_name(s) == b".rela.text"
    })
    .expect("the fixture has a .rela.text");
    set_sh_type(&mut bytes, rela, SHT_REL);

    let err = link_err("rel", &bytes).expect("the link must be refused");
    assert!(
        err.contains("SHT_REL"),
        "the diagnostic names the unsupported form: {err}"
    );
}

/// A `SHF_MERGE|SHF_STRINGS` section whose last byte is not a NUL is refused
/// rather than silently truncated to its terminated prefix.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_unterminated_mergeable_string_section_fails_the_link() {
    let dir = std::env::temp_dir()
        .join(format!("xold_unsupported_lit_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create work dir");
    let Some(mut bytes) = compile_literals(&dir) else {
        eprintln!("skipping: clang unavailable");
        return;
    };
    let obj = ObjectFile::parse(&bytes).expect("compiled object parses");
    let (offset, size) = obj
        .sections()
        .iter()
        .find(|s| {
            let flags = s.sh_flags.get();
            // SHF_MERGE | SHF_STRINGS, byte-wide entries.
            flags & 0x10 != 0 && flags & 0x20 != 0 && s.sh_entsize.get() == 1
        })
        .map(|s| (s.sh_offset.get(), s.sh_size.get()))
        .expect("the fixture has a mergeable string section");
    let offset = usize::try_from(offset).expect("offset is addressable");
    let size = usize::try_from(size).expect("size is addressable");
    assert!(size > 0, "the section has content");
    // Overwrite the closing NUL, leaving the last string unterminated.
    bytes[offset + size - 1] = b'x';

    let err =
        link_err("unterminated", &bytes).expect("the link must be refused");
    assert!(
        err.contains("NUL terminated"),
        "the diagnostic names the broken promise: {err}"
    );
    let _ = fs::remove_dir_all(&dir);
}
