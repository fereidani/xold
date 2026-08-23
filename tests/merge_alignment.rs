//! Merged pieces keep the alignment their section declared.
//!
//! Deduplication reorders and repacks a `SHF_MERGE` section's content, so a
//! piece's offset within the pool is the only thing left describing where it
//! lands. Nothing else restores the alignment the input asked for.
//!
//! gcc's `.rodata.str1.8` is the case that matters: `sh_entsize` 1 with
//! `sh_addralign` 8, which every `-O2` compilation with a string literal in it
//! emits. Packed end to end, every string after the first one whose length is
//! not a multiple of 8 sits misaligned, and code compiled against the declared
//! alignment -- a 16-byte load over a literal, an `AArch64` atomic on a merged
//! constant -- breaks on data the linker placed correctly by its own
//! arithmetic. lld aligns every surviving piece through the
//! `llvm::Align(alignment)` its `StringTableBuilder` is constructed with.
//!
//! The fixture is hand-written assembly rather than compiled C, because the
//! point is to name the pieces: a compiler gives string literals local
//! `.L`-prefixed symbols the output does not publish, and the test has to read
//! the address of a specific one.
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

/// The alignment the fixture's section declares.
const WIDE: u64 = 8;

/// Three strings in one `.rodata.str1.8`, none of whose lengths is a multiple
/// of 8, so each one after the first is misaligned unless the pool pads. Named
/// globals, so the addresses are readable from `.symtab`.
///
/// `s_dup` repeats `s_a`'s content: it has to fold onto the same offset, so
/// the test also shows the padding did not cost the deduplication.
const MERGE_SRC: &[u8] = b"    .section .rodata.str1.8,\"aMS\",@progbits,1\n\
    .p2align 3\n\
    .globl s_a\ns_a:\n    .asciz \"abc\"\n\
    .globl s_b\ns_b:\n    .asciz \"defghij\"\n\
    .globl s_c\ns_c:\n    .asciz \"klmnopqrs\"\n\
    .globl s_dup\ns_dup:\n    .asciz \"abc\"\n";

/// A second translation unit contributing to the same pool, so the walk
/// crosses a section boundary with the cursor at an odd offset.
const MERGE_SRC2: &[u8] = b"    .section .rodata.str1.8,\"aMS\",@progbits,1\n\
    .p2align 3\n\
    .globl s_d\ns_d:\n    .asciz \"tuvwx\"\n\
    .globl s_e\ns_e:\n    .asciz \"yz\"\n";

/// The program: every named string must be 8-byte aligned, and the two
/// spellings of `"abc"` must be one address.
const MAIN_SRC: &[u8] = b"extern const char s_a[], s_b[], s_c[], s_dup[];\n\
    extern const char s_d[], s_e[];\n\
    int main(void)\n\
    {\n\
        /* Through volatile pointers: comparing two array names directly is\n\
           something the compiler folds, since distinct objects have distinct\n\
           addresses -- which is exactly the assumption under test. */\n\
        const char *volatile all[] = { s_a, s_b, s_c, s_d, s_e };\n\
        for (int i = 0; i < 5; i++)\n\
            if ((unsigned long)(void *)all[i] % 8) { return i + 1; }\n\
        const char *volatile a = s_a;\n\
        const char *volatile dup = s_dup;\n\
        if (a != dup) { return 9; }\n\
        return 0;\n\
    }\n";

/// Every merged piece starts on the alignment its section declared.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn every_merged_piece_starts_on_its_sections_alignment() {
    let Some(dir) = workdir("align") else {
        return;
    };
    let Some(prog) = link(&dir, "align") else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");
    for name in [
        b"s_a".as_slice(),
        b"s_b".as_slice(),
        b"s_c".as_slice(),
        b"s_d".as_slice(),
        b"s_e".as_slice(),
    ] {
        let addr = symbol_addr(&bytes, name)
            .unwrap_or_else(|| panic!("{} must be defined", show(name)));
        assert_eq!(
            addr % WIDE,
            0,
            "{} landed at {addr:#x}, which is not on the 8-byte boundary its \
             section declared",
            show(name)
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

/// The padding does not cost the deduplication: two spellings of the same
/// string still share one piece.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn identical_pieces_still_fold_onto_one_offset() {
    let Some(dir) = workdir("dedup") else {
        return;
    };
    let Some(prog) = link(&dir, "dedup") else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");
    let a = symbol_addr(&bytes, b"s_a").expect("s_a must be defined");
    let dup = symbol_addr(&bytes, b"s_dup").expect("s_dup must be defined");
    assert_eq!(
        a, dup,
        "two identical strings must still deduplicate to one address"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The program agrees at run time, and reads its own strings.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_merged_alignment_holds_at_run_time() {
    let Some(dir) = workdir("run") else {
        return;
    };
    let Some(prog) = link(&dir, "run") else {
        return;
    };
    assert_eq!(
        Command::new(&prog)
            .status()
            .expect("linked program must be runnable")
            .code(),
        Some(0),
        "the program's own alignment and identity checks must pass"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// A symbol name as text, for a message.
fn show(name: &[u8]) -> String {
    String::from_utf8_lossy(name).into_owned()
}

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping merge-alignment {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_mergealign_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the three inputs and links them into a runnable dynamic executable.
fn link(dir: &Path, stem: &str) -> Option<PathBuf> {
    let Some(interp) = interpreter() else {
        eprintln!("skipping merge-alignment {stem}: interpreter path unknown");
        return None;
    };
    let strings = dir.join(format!("{stem}_str.o"));
    let strings2 = dir.join(format!("{stem}_str2.o"));
    let main_o = dir.join(format!("{stem}_main.o"));
    build(MERGE_SRC, "S", &strings)?;
    build(MERGE_SRC2, "S", &strings2)?;
    build(MAIN_SRC, "c", &main_o)?;

    let start = common::crt_file("Scrt1.o")?;
    let prologue = common::crt_file("crti.o")?;
    let epilogue = common::crt_file("crtn.o")?;
    let libc = common::libc_so()?;

    let prog = dir.join(stem);
    let res = link_dyn_exec(
        &[start, prologue, main_o, strings, strings2, libc, epilogue],
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

/// Compiles or assembles `src` (written beside `obj` with extension `ext`).
fn build(src: &[u8], ext: &str, obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension(ext);
    fs::write(&src_path, src).ok()?;
    Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIE", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success()
        .then_some(())
}

// --- readers ---------------------------------------------------------------

/// The address of `name` in `.symtab`, or `None` when it is absent.
fn symbol_addr(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok()??;
    symtab
        .syms
        .iter()
        .find(|s| symtab.name(s) == name)
        .map(|s| s.st_value.get())
}
