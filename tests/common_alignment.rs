//! `.bss` is placed as strictly as the commons it holds.
//!
//! An alignment a translation unit declares is a promise the code was
//! compiled against: an aligned SIMD load, a locked read-modify-write that
//! must not straddle a cache line, an `AArch64` exclusive pair. A common
//! symbol's alignment is the one the linker can lose without noticing,
//! because a common carries no input section header for the output section to
//! absorb an alignment from.
//!
//! Its offset is measured from `.bss`'s start by its own alignment, and
//! `.bss`'s members do not bound that: a 64-byte common in an image whose
//! `.bss` members ask for 16 landed at `base + 16 (mod 64)`. lld has no such
//! gap, because it converts commons into input sections and an output section
//! takes the largest alignment of anything in it.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_dyn_exec};

mod common;

/// The alignment the fixture declares, larger than `.bss`'s floor of 16 and
/// larger than any alignment its other members carry.
const WIDE: u64 = 64;

/// A tentative definition asking for 64-byte alignment, beside ordinary `.bss`
/// members that ask for less. Built `-fcommon`, so `wide_common` really is a
/// common symbol and not an ordinary definition.
const COMMON_SRC: &[u8] =
    b"__attribute__((aligned(64))) int wide_common[16];\n\
    int plain_bss[4];\n\
    char narrow_bss;\n\
    int main(void)\n\
    {\n\
        return ((unsigned long)(void *)wide_common % 64) != 0;\n\
    }\n";

/// `.bss` is placed at least as strictly as the strictest common in it, so a
/// common's computed address really is aligned.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_wide_common_lands_on_its_declared_alignment() {
    let Some(dir) = workdir("common") else {
        return;
    };
    let Some(prog) = link(&dir, COMMON_SRC, "common", &["-fcommon"]) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");
    // Asserted on the section rather than on a `.symtab` row: xold publishes
    // no row for a common symbol, so the section is where the placement is
    // visible. Commons are laid out from `.bss`'s start, so its address and
    // alignment are what decide theirs.
    let bss = section(&bytes, b".bss").expect("the image has .bss");
    assert!(
        bss.align >= WIDE,
        ".bss must be placed at least as strictly as its widest common; got \
         {}",
        bss.align
    );
    assert_eq!(
        bss.addr % WIDE,
        0,
        ".bss starts at {:#x}, which is not 64-byte aligned, so every offset \
         measured from it is off by the same amount",
        bss.addr
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The program agrees at run time, which is what the arithmetic is for.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_common_alignment_holds_at_run_time() {
    let Some(dir) = workdir("commonrun") else {
        return;
    };
    let Some(prog) = link(&dir, COMMON_SRC, "commonrun", &["-fcommon"]) else {
        return;
    };
    assert_eq!(
        run(&prog),
        Some(0),
        "the program's own modulo check must pass"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping rw-alignment {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_rwalign_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles `src` and links it into a dynamic executable against the host
/// libc, so the produced program can actually run.
fn link(dir: &Path, src: &[u8], stem: &str, args: &[&str]) -> Option<PathBuf> {
    let Some(interp) = interpreter() else {
        eprintln!("skipping rw-alignment {stem}: interpreter path unknown");
        return None;
    };
    let obj = dir.join(format!("{stem}.o"));
    compile(src, &obj, args)?;
    let start = common::crt_file("Scrt1.o")?;
    let prologue = common::crt_file("crti.o")?;
    let epilogue = common::crt_file("crtn.o")?;
    let libc = common::libc_so()?;

    let prog = dir.join(stem);
    let res = link_dyn_exec(
        &[start, prologue, obj, libc, epilogue],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "xold link must succeed: {:?}", res.err());
    Some(prog)
}

/// Compiles `src` with the host clang, passing `args` through.
fn compile(src: &[u8], obj: &Path, args: &[&str]) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).ok()?;
    Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIE", "-c"])
        .args(args)
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success()
        .then_some(())
}

/// Runs the linked program, returning its exit status.
fn run(prog: &Path) -> Option<i32> {
    Command::new(prog)
        .status()
        .expect("linked program must be runnable")
        .code()
}

// --- readers ---------------------------------------------------------------

/// The placement of one output section.
struct Placed {
    addr: u64,
    align: u64,
}

/// The address and alignment of the named section.
fn section(bytes: &[u8], name: &[u8]) -> Option<Placed> {
    let obj = ObjectFile::parse(bytes).ok()?;
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == name)
        .map(|s| Placed {
            addr: s.sh_addr.get(),
            align: s.sh_addralign.get(),
        })
}
