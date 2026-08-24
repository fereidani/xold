//! `--gc-sections` must keep the legacy constructor and destructor lists.
//!
//! Before `.init_array` there were `.ctors` and `.dtors`: lists of function
//! pointers the C runtime walks at startup and exit, spelled `.ctors.65535`
//! and `.dtors.65535` with the priority appended. `.jcr` is the same shape for
//! the Java class registry. Nothing in a link relocates into any of them, so
//! reachability alone collects the lot, and with it every constructor they were
//! the only reference to. lld reserves all three in `isReserved`
//! (`MarkLive.cpp`); xold reserves the same set.
//!
//! The fixture keeps the two halves of that separate. `.ctors.65535` holds one
//! pointer and nothing names the section, so it survives only by being a root;
//! the function it points at sits in its own `.text` section and survives only
//! because the section's relocation was followed. A third function nothing
//! reaches proves the collector really ran.
//!
//! Gated on `clang`; if absent the test prints a note and returns, so the build
//! never fails over a missing toolchain.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_to};

mod common;

/// The fixture. Each function is its own section, so the collector can take
/// them one at a time, and the constructor list is a plain `.ctors.65535`
/// holding one pointer -- what a pre-`.init_array` toolchain emits.
const LEGACY: &str = "    .section .text.legacy_ctor,\"ax\",@progbits\n\
     .globl legacy_ctor\n\
     .type legacy_ctor,@function\n\
     legacy_ctor:\n\
     movl $0x5b, %eax\n\
     ret\n\
     .section .text.unused_leaf,\"ax\",@progbits\n\
     .globl unused_leaf\n\
     .type unused_leaf,@function\n\
     unused_leaf:\n\
     movl $0x6c, %eax\n\
     ret\n\
     .section .ctors.65535,\"aw\",@progbits\n\
     .p2align 3\n\
     .quad legacy_ctor\n";

/// A `main` that does nothing: the point is what survives around it.
const MAIN: &str = "int main(void) { return 0; }\n";

/// The freestanding entry stub: calls `main` and exits with its return value.
const START: &str = "    .text\n    .globl _start\n_start:\n\
     call main\n movl %eax, %edi\n movl $60, %eax\n syscall\n";

/// `legacy_ctor`'s body, `mov eax, 0x5b; ret`: a byte signature unique within
/// this program's `.text`.
const LEGACY_CTOR_SIGNATURE: &[u8] = &[0xb8, 0x5b, 0x00, 0x00, 0x00, 0xc3];

/// `unused_leaf`'s body, `mov eax, 0x6c; ret`.
const UNUSED_LEAF_SIGNATURE: &[u8] = &[0xb8, 0x6c, 0x00, 0x00, 0x00, 0xc3];

/// A private working directory. The name carries the process id so concurrent
/// test binaries cannot delete each other's files.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_ctors_{prefix}_{}", std::process::id()));
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

/// Builds the inputs and links them into `name`, with or without
/// `--gc-sections`. `None` when the host toolchain cannot build them.
fn link(dir: &Path, name: &str, gc: bool) -> Option<PathBuf> {
    let legacy = dir.join("legacy.o");
    let m = dir.join("ctors_main.o");
    let s = dir.join("start.o");
    build(LEGACY, "S", &legacy, dir)?;
    build(MAIN, "c", &m, dir)?;
    build(START, "S", &s, dir)?;
    let out = dir.join(name);
    link_to(&[legacy, m, s], &out, b"_start", gc, IcfMode::None, false)
        .expect("xold links the legacy constructor fixture");
    Some(out)
}

/// Whether the output's `.text` holds `needle`.
fn text_has(bytes: &[u8], needle: &[u8]) -> bool {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return false;
    };
    obj.sections()
        .iter()
        .filter(|s| obj.section_name(s) == b".text")
        .filter_map(|s| obj.section_data(s).ok())
        .any(|data| data.windows(needle.len()).any(|w| w == needle))
}

/// The address of the defined symbol `want` in the output symbol table.
fn symbol_addr(bytes: &[u8], want: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok()??;
    symtab
        .syms
        .iter()
        .find(|s| symtab.name(s) == want)
        .map(|s| s.st_value.get())
}

/// Whether any allocated section other than `.text` holds `addr` as a 64-bit
/// little-endian word, which is what a surviving `.ctors` entry looks like once
/// its relocation has been applied.
fn data_holds_pointer(bytes: &[u8], addr: u64) -> bool {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return false;
    };
    let word = addr.to_le_bytes();
    obj.sections()
        .iter()
        .filter(|s| obj.section_name(s) != b".text")
        .filter_map(|s| obj.section_data(s).ok())
        .any(|data| data.windows(word.len()).any(|w| w == word))
}

/// The headline proof: a `.ctors.65535` nothing references survives
/// `--gc-sections`, holding the pointer it was built with, and the constructor
/// it names survives with it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_legacy_ctors_section_survives_gc() {
    let dir = workdir("gc");
    let Some(prog) = link(&dir, "ctors_gc", true) else {
        eprintln!("skipping legacy-ctors test: host clang unavailable");
        return;
    };
    let bytes = fs::read(&prog).expect("image is readable");

    let ctor = symbol_addr(&bytes, b"legacy_ctor")
        .expect("legacy_ctor must reach the output symbol table");
    assert!(
        data_holds_pointer(&bytes, ctor),
        "the .ctors entry must survive gc holding legacy_ctor's address"
    );
    assert!(
        text_has(&bytes, LEGACY_CTOR_SIGNATURE),
        "the constructor the .ctors entry names must survive gc"
    );
    assert!(
        !text_has(&bytes, UNUSED_LEAF_SIGNATURE),
        "the collector must still drop what nothing reaches"
    );

    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        let status = Command::new(&prog)
            .status()
            .expect("linked program must be runnable");
        assert_eq!(status.code(), Some(0), "the gc-linked program must run");
    } else {
        let image = ObjectFile::parse(&bytes).expect("valid ELF executable");
        let start = symbol_addr(&bytes, b"_start").expect("_start is retained");
        assert_eq!(
            image.header().e_entry.get(),
            start,
            "the foreign executable's entry still points at _start"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

/// The same link without the flag keeps everything, so the assertions above
/// are about the collector rather than about the fixture.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_default_link_keeps_both_functions() {
    let dir = workdir("plain");
    let Some(prog) = link(&dir, "ctors_plain", false) else {
        eprintln!("skipping legacy-ctors default test: clang unavailable");
        return;
    };
    let bytes = fs::read(&prog).expect("image is readable");
    assert!(
        text_has(&bytes, LEGACY_CTOR_SIGNATURE),
        "the default link must keep legacy_ctor"
    );
    assert!(
        text_has(&bytes, UNUSED_LEAF_SIGNATURE),
        "the default link must keep unused_leaf"
    );
    let _ = fs::remove_dir_all(&dir);
}
