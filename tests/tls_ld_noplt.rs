//! The `-fno-plt` spelling of the local-dynamic TLS sequence.
//!
//! `TLSLD` has no fallback in an executable: there is no runtime
//! `__tls_get_addr` to reach, so lowering the sequence is the only way the
//! site can be resolved at all. A shape the lowering does not recognise is
//! therefore not a missed optimisation but a failed link, reported as
//! `UnresolvableTls`.
//!
//! xold recognised one shape: the twelve-byte form that closes with a direct
//! `call __tls_get_addr`. `-fno-plt` reaches the helper through the GOT
//! instead -- `call *__tls_get_addr@GOTPCREL(%rip)`, one byte wider -- and
//! that thirteen-byte form never matched. Arch Linux carries `-fno-plt` in its
//! default CFLAGS, so two file-local `_Thread_local`s in one translation unit
//! were enough to end the link. The general-dynamic pair was already handled,
//! because both of *its* call encodings are the same width; only the
//! local-dynamic one has two lengths.
//!
//! lld matches the same two shapes in `relaxTlsLdToLe` (`X86_64.cpp:696`) and
//! errors on anything else. The bytes the indirect form becomes are the
//! psABI's "Table 11.9: LD -> LE Code Transition (LP64)": one more `data16`
//! prefix in front of the same `mov %fs:0, %rax`, so the rewrite stays in
//! place.
//!
//! Verified by running: the fixture is compiled by clang, linked by xold, and
//! executed, because a lowered TLS sequence that assembles is not the same as
//! one that reads the right memory.
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
    linker::link_dyn_exec,
    reloc::x86_64::{R_X86_64_GOTPCRELX, R_X86_64_TLSLD},
};

mod common;

/// Two translation-unit-local thread-locals, which is what makes the compiler
/// choose the local-dynamic model: both live in this module, so one call can
/// find the block for the pair.
const SRC: &[u8] = b"#include <stdio.h>\n\
    static _Thread_local int a = 11;\n\
    static _Thread_local int b = 31;\n\
    int main(void) { printf(\"ld %d\\n\", a + b); return 0; }\n";

/// What the program prints once both thread-locals read correctly.
const EXPECTED: &str = "ld 42";

/// The opcode of the GOT-indirect call: `call *disp32(%rip)`.
const CALL_INDIRECT: &[u8] = &[0xff, 0x15];

/// The thirteen bytes the lowering produces: `.long 0x66666666` followed by
/// `movq %fs:0, %rax`.
const LOWERED: &[u8] = &[
    0x66, 0x66, 0x66, 0x66, 0x64, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00,
    0x00,
];

/// The link succeeds and the program reads both thread-locals.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_no_plt_local_dynamic_sequence_links_and_runs() {
    let Some(dir) = workdir("run") else {
        return;
    };
    let Some(prog) = link(&dir) else {
        return;
    };
    let out = Command::new(&prog).output().expect("program must run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(EXPECTED),
        "both thread-locals must read correctly (got {stdout:?})"
    );
    assert_eq!(out.status.code(), Some(0), "and the program must exit 0");
    let _ = fs::remove_dir_all(&dir);
}

/// The fixture really is the shape under test, and the image really carries
/// the lowered form.
///
/// Without the first half this test would pass on a compiler that emitted the
/// direct call, proving nothing; without the second it would pass on a linker
/// that left the sequence alone in a way glibc happened to tolerate.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_indirect_call_is_rewritten_in_place() {
    let Some(dir) = workdir("bytes") else {
        return;
    };
    let obj = dir.join("ld.o");
    if compile(SRC, &obj).is_none() {
        eprintln!("skipping tls-ld-noplt: clang cannot build the fixture");
        return;
    }
    let obj_bytes = fs::read(&obj).expect("read object");
    assert!(
        has_indirect_ld_sequence(&obj_bytes),
        "the fixture must carry a TLSLD closed by a GOT-indirect call, or \
         this test is not about -fno-plt at all"
    );

    let Some(prog) = link(&dir) else {
        return;
    };
    let prog_bytes = fs::read(&prog).expect("read output");
    let text = section_data(&prog_bytes, b".text").expect("the image has text");
    // One per thread-local. A surviving `lea sym@tlsld(%rip), %rdi` is not
    // worth searching for directly -- those three bytes are also every
    // ordinary RIP-relative load into `%rdi` -- but a sequence that was not
    // lowered leaves no copy of the replacement behind either.
    assert_eq!(
        count(text, LOWERED),
        2,
        "each lowered sequence must leave the psABI's Table 11.9 form in \
         .text, once per thread-local the compiler grouped into the pair"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping tls-ld-noplt {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_tlsldnoplt_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture and links it into a runnable dynamic executable.
fn link(dir: &Path) -> Option<PathBuf> {
    let Some(interp) = interpreter() else {
        eprintln!("skipping tls-ld-noplt: interpreter path unknown");
        return None;
    };
    let obj = dir.join("ld.o");
    compile(SRC, &obj)?;
    let start = common::crt_file("Scrt1.o")?;
    let prologue = common::crt_file("crti.o")?;
    let epilogue = common::crt_file("crtn.o")?;
    let libc = common::libc_so()?;

    let prog = dir.join("ldprog");
    let res = link_dyn_exec(
        &[start, prologue, obj, libc, epilogue],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    );
    assert!(
        res.is_ok(),
        "a -fno-plt local-dynamic sequence must link: {:?}",
        res.err()
    );
    Some(prog)
}

/// Compiles `src` with the host clang.
///
/// `-fno-plt` is the flag under test. `-fPIC` with an explicit
/// `local-dynamic` model is what keeps the compiler from picking the
/// local-exec form on its own: it knows the two statics are private to the
/// module, and for a `-fPIE` target it would resolve them at compile time and
/// emit no `TLSLD` at all.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).ok()?;
    Command::new(clang)
        .args([
            "--target=x86_64-linux-gnu",
            "-fno-plt",
            "-fPIC",
            "-ftls-model=local-dynamic",
            "-c",
        ])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success()
        .then_some(())
}

// --- readers ---------------------------------------------------------------

/// How many times `needle` occurs in `haystack`, counting overlaps.
fn count(haystack: &[u8], needle: &[u8]) -> usize {
    haystack
        .windows(needle.len())
        .filter(|w| *w == needle)
        .count()
}

/// The bytes of the named section in a linked image.
fn section_data<'a>(bytes: &'a [u8], name: &[u8]) -> Option<&'a [u8]> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == name)?;
    // The borrow is over `bytes`, which outlives the parsed view.
    let off = usize::try_from(shdr.sh_offset.get()).ok()?;
    let size = usize::try_from(shdr.sh_size.get()).ok()?;
    bytes.get(off..off.checked_add(size)?)
}

/// Whether the object carries a `TLSLD` site whose sequence closes with a
/// GOT-indirect call to `__tls_get_addr`.
///
/// The slot is four bytes wide, so the closing instruction starts four bytes
/// past the relocation's offset; the helper's own relocation sits two bytes
/// past that.
fn has_indirect_ld_sequence(bytes: &[u8]) -> bool {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return false;
    };
    let Some((shndx, text)) = obj
        .sections()
        .iter()
        .enumerate()
        .find(|(_, s)| obj.section_name(s) == b".text")
    else {
        return false;
    };
    let Ok(data) = obj.section_data(text) else {
        return false;
    };
    let Ok(shndx) = u16::try_from(shndx) else {
        return false;
    };
    let Ok(Some(relocs)) = obj.relocations(shndx) else {
        return false;
    };
    relocs
        .iter()
        .filter(|r| r.r_type() == R_X86_64_TLSLD)
        .any(|r| {
            let Ok(at) = usize::try_from(r.r_offset.get() + 4) else {
                return false;
            };
            data.get(at..at + 2) == Some(CALL_INDIRECT)
                && relocs.iter().any(|o| {
                    o.r_type() == R_X86_64_GOTPCRELX
                        && o.r_offset.get() == r.r_offset.get() + 6
                })
        })
}
