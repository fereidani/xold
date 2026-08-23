//! A mergeable section that carries relocations must not be merged.
//!
//! `SHF_MERGE` is a promise that a section is a bag of independent constants
//! the linker may reorder and deduplicate. A section with a relocation table
//! breaks that promise twice over: a pool is keyed on content alone, so two
//! byte-identical pieces carrying different relocations fold onto one slot that
//! both would write, and every contributor but the carrier is dropped from its
//! output section, which takes its relocations with it. mold and lld both
//! decline to split such a section; so does xold, which concatenates it as an
//! ordinary member instead.
//!
//! The fixture is the smallest shape that shows it. Two objects contribute one
//! eight-byte entry each to a single `.rodata.cst8` pool; both entries read as
//! eight zero bytes in the input, and only the second carries the
//! `R_X86_64_64` that fills them with an address. Folding the second onto the
//! first leaves that word zero and drops the relocation with the member, so the
//! program reads the word back rather than checking that the link succeeded --
//! it succeeded before this was fixed, too.
//!
//! Gated on `clang`; if absent the test prints a note and returns, so the build
//! never fails over a missing toolchain.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{icf::IcfMode, linker::link_to};

mod common;

/// The unrelocated contributor: one eight-byte entry of zeros. It is the first
/// section into the pool, so it is the one a fold would keep.
const POOL_A: &str = "    .section .rodata.cst8,\"aM\",@progbits,8\n\
     .p2align 3\n\
     .globl plain_entry\n\
     plain_entry:\n\
     .quad 0\n";

/// The relocated contributor: the same eight zero bytes in the file, plus the
/// `R_X86_64_64` that makes them the address of `target_value`.
const POOL_B: &str = "    .data\n\
     .p2align 3\n\
     .globl target_value\n\
     target_value:\n\
     .quad 42\n\
     .section .rodata.cst8,\"aM\",@progbits,8\n\
     .p2align 3\n\
     .globl ptr_entry\n\
     ptr_entry:\n\
     .quad target_value\n";

/// Reads both entries back. Each failure has its own exit status so a broken
/// link says which guarantee it broke.
///
/// The two addresses are compared through volatile locals because the compiler
/// knows two distinct objects cannot share an address and folds the comparison
/// away otherwise -- which is precisely the claim under test.
const MAIN: &str = r"
extern unsigned long plain_entry;
extern unsigned long ptr_entry;
extern unsigned long target_value;
int main(void) {
    unsigned long *volatile plain = &plain_entry;
    unsigned long *volatile ptr = &ptr_entry;
    /* A fold would put both entries on one address. */
    if (plain == ptr) return 1;
    /* The relocated entry must hold the address the relocation names. */
    if (ptr_entry != (unsigned long)&target_value) return 2;
    if (*(unsigned long *)ptr_entry != 42) return 3;
    if (plain_entry != 0) return 4;
    return 0;
}
";

/// The freestanding entry stub: calls `main` and exits with its return value.
const START: &str = "    .text\n    .globl _start\n_start:\n\
     call main\n movl %eax, %edi\n movl $60, %eax\n syscall\n";

/// A private working directory. The name carries the process id so concurrent
/// test binaries cannot delete each other's files.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_mergerel_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Assembles or compiles `src` to `obj`, choosing the language from `ext`.
/// Returns `None` when no host compiler exists.
fn build(src: &str, ext: &str, obj: &Path, dir: &Path) -> Option<()> {
    let clang = which("clang")?;
    let stem = obj.file_stem()?.to_str().unwrap_or("unit");
    let file = dir.join(format!("{stem}.{ext}"));
    fs::write(&file, src).ok()?;
    Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fno-pie", "-fno-pic"])
        .args(["-ffreestanding", "-c", "-o"])
        .arg(obj)
        .arg(&file)
        .status()
        .ok()?
        .success()
        .then_some(())
}

/// Builds the four inputs and links them into an executable, or `None` when
/// the host toolchain cannot build them.
fn link(dir: &Path) -> Option<PathBuf> {
    let a = dir.join("pool_a.o");
    let b = dir.join("pool_b.o");
    let m = dir.join("mergerel_main.o");
    let s = dir.join("start.o");
    build(POOL_A, "S", &a, dir)?;
    build(POOL_B, "S", &b, dir)?;
    build(MAIN, "c", &m, dir)?;
    build(START, "S", &s, dir)?;
    let out = dir.join("prog");
    link_to(&[a, b, m, s], &out, b"_start", false, IcfMode::None, false)
        .expect("xold links a relocated mergeable section");
    Some(out)
}

/// The headline proof: the relocated entry still holds the value its
/// relocation names. Before the decline it held zero, because the entry was
/// folded onto the byte-identical one in the other object and its own member
/// -- relocation and all -- was dropped from the output.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_relocated_merge_entry_keeps_its_value() {
    let dir = workdir("value");
    let Some(prog) = link(&dir) else {
        eprintln!("skipping relocated-merge test: host clang unavailable");
        return;
    };
    let status = Command::new(&prog)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(0),
        "a relocated .rodata.cst8 entry must survive with its own value"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The same link, twice: declining to merge must be as deterministic as
/// merging is.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_decline_is_independent_of_the_run() {
    let dir = workdir("determinism");
    let Some(first) = link(&dir) else {
        eprintln!("skipping relocated-merge determinism: clang unavailable");
        return;
    };
    let bytes = fs::read(&first).expect("image is readable");
    for _ in 0..3 {
        let again = link(&dir).expect("relink succeeds");
        assert_eq!(
            fs::read(&again).expect("image is readable"),
            bytes,
            "the output must be a function of the inputs alone"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}
