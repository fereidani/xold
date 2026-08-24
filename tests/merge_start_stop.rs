//! A section the program bounds with `__start_`/`__stop_` is not merged away.
//!
//! `__start_NAME`/`__stop_NAME` bound the extent of the section called `NAME`
//! in the image, and a program walks between them expecting to find that
//! section's content and nothing else. Registration tables are built this way.
//!
//! A merge pool is keyed on how its content splits and what it needs aligning
//! to, not on which section the content came from. So a `SHF_MERGE` section
//! whose name happens to be a C identifier shared a pool with
//! `.rodata.str1.1`, its pieces landed interleaved with foreign strings, and
//! the bounds either collapsed to nothing or spanned the lot. Merging runs
//! before the bounds are measured, so nothing downstream could notice.
//!
//! Such a section is no longer merged. It costs the deduplication of a section
//! that is rooted anyway -- `__start_`/`__stop_` keep it whole by definition
//! -- and the name is only looked at for a section that already carries
//! `SHF_MERGE`.
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

/// `regtab` is mergeable *and* C-identifier-named, and `.rodata.str1.1`
/// carries one of the same strings, so a shared pool would fold them together
/// and put a foreign string between the bounds.
///
/// The two entries are ten bytes each with their terminators, so the run is
/// twenty bytes and the program says so itself.
const SRC: &[u8] = b"    .section regtab,\"aMS\",@progbits,1\n\
    .asciz \"entry-one\"\n\
    .asciz \"entry-two\"\n\
    .section .rodata.str1.1,\"aMS\",@progbits,1\n\
    .asciz \"entry-one\"\n\
    .asciz \"unrelated filler string\"\n\
    .text\n\
    .globl _start\n\
_start:\n\
    leaq  __start_regtab(%rip), %rax\n\
    leaq  __stop_regtab(%rip), %rcx\n\
    subq  %rax, %rcx\n\
    movq  $60, %rax\n\
    movq  $0, %rdi\n\
    cmpq  $20, %rcx\n\
    je    1f\n\
    movq  $1, %rdi\n\
1:  syscall\n";

/// The program measures its own table and finds exactly its two entries.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_bounds_span_the_sections_own_content() {
    let Some(dir) = workdir("bounds") else {
        return;
    };
    let Some(prog) = link(&dir) else {
        return;
    };
    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        assert_bound_run(&fs::read(&prog).expect("read linked image"));
        let _ = fs::remove_dir_all(&dir);
        return;
    }
    let code = Command::new(&prog)
        .status()
        .expect("linked program must run")
        .code();
    assert_eq!(
        code,
        Some(0),
        "__stop_regtab - __start_regtab must be the twenty bytes of the two \
         entries; folding the section into the shared string pool makes the \
         run span whatever else landed in it"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The other section still merges, so the exemption is about the bounded
/// section and not about turning deduplication off.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_ordinary_string_pool_still_deduplicates() {
    let Some(dir) = workdir("dedup") else {
        return;
    };
    let Some(prog) = link(&dir) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");
    let rodata = section(&bytes, b".rodata").expect(".rodata present");
    assert_eq!(
        occurrences(&rodata, b"unrelated filler string"),
        1,
        "the ordinary mergeable section is still pooled"
    );
    // "entry-one" appears twice on purpose: once inside `regtab`, which is no
    // longer pooled, and once in `.rodata.str1.1`, which is. Keeping both is
    // the point -- the bounded section owns its copy.
    assert_eq!(
        occurrences(&rodata, b"entry-one"),
        2,
        "the bounded section keeps its own copy rather than sharing one"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping merge-start-stop {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_startstop_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Assembles and links the fixture.
fn link(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("t.S");
    let obj = dir.join("t.o");
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
        eprintln!("skipping merge-start-stop: clang cannot assemble it");
        return None;
    }
    let out = dir.join("prog");
    let res = link_to(
        std::slice::from_ref(&obj),
        &out,
        b"_start",
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    Some(out)
}

// --- readers ---------------------------------------------------------------

/// The bytes of a named output section.
fn section(bytes: &[u8], name: &[u8]) -> Option<Vec<u8>> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == name)?;
    obj.section_data(shdr).ok().map(<[u8]>::to_vec)
}

/// How many times `needle` appears in `hay`.
fn occurrences(hay: &[u8], needle: &[u8]) -> usize {
    hay.windows(needle.len()).filter(|w| *w == needle).count()
}

/// Reads the two linker-defined bounds and the exact bytes between them, which
/// is the same subtraction and content contract the foreign program checks.
fn assert_bound_run(bytes: &[u8]) {
    let obj = ObjectFile::parse(bytes).expect("valid ELF");
    let symtab = obj.symbol_table().expect("read symtab").expect("symtab");
    let value = |name: &[u8]| {
        symtab
            .syms
            .iter()
            .find(|sym| symtab.name(sym) == name)
            .map(|sym| sym.st_value.get())
            .unwrap_or_else(|| {
                panic!("{} is defined", String::from_utf8_lossy(name))
            })
    };
    let start = value(b"__start_regtab");
    let stop = value(b"__stop_regtab");
    assert_eq!(stop - start, 20, "the bounded run is exactly twenty bytes");
    assert_eq!(
        image_at(&obj, start, 20),
        Some(b"entry-one\0entry-two\0".as_slice()),
        "the run contains only its two registration entries"
    );
}

fn image_at<'a>(
    obj: &ObjectFile<'a>,
    addr: u64,
    len: usize,
) -> Option<&'a [u8]> {
    for sec in obj.sections() {
        let base = sec.sh_addr.get();
        if addr < base || addr >= base.saturating_add(sec.sh_size.get()) {
            continue;
        }
        let at = usize::try_from(addr - base).ok()?;
        return obj.section_data(sec).ok()?.get(at..at.checked_add(len)?);
    }
    None
}
