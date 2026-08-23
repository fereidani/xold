//! An ordinary `PLT32` call site whose preceding bytes happen to spell a
//! lowered TLS sequence must still be relocated.
//!
//! A general-dynamic TLS pair is lowered as a unit, and the replacement ends
//! with a 4-byte field exactly where the `call __tls_get_addr` displacement
//! was, so the relocation on that displacement has to be dropped. xold used to
//! find it by comparing the twelve bytes before every `PLT32` slot in the
//! program against the three sequences a lowering writes. That is a guess: any
//! site whose neighbourhood matches loses its relocation, whether a lowering
//! put those bytes there or not, and the whole image is measured against it.
//!
//! The decision now belongs to the `TLSGD`/`TLSLD` relocation, which reports
//! the bytes its lowering consumes, so a site that merely looks like one is
//! patched normally. These fixtures place each of the three sequences directly
//! in front of a `PLT32` slot and check the displacement lands.
//!
//! The relocation is placed with `.reloc` rather than by an instruction,
//! because no assembler emits a `PLT32` on a `lea` displacement -- which is
//! the point: the byte test could not tell the two apart, and nothing about
//! the site itself can.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_to};

mod common;

/// Three `PLT32` sites, each preceded by the bytes one of the TLS lowerings
/// writes, and none of them the product of a lowering.
///
/// - `le_probe`: the local-exec form a general-dynamic pair becomes when the
///   image places the thread-local itself, `mov %fs:0, %rax` then the opcode of
///   `lea off(%rax), %rax`.
/// - `ie_probe`: the initial-exec form it becomes when a shared object places
///   it, the same load then the opcode of `add off(%rip), %rax`.
/// - `ld_probe`: the twelve bytes a local-dynamic sequence becomes, which sit
///   four bytes further back because that sequence is shorter than the window.
///   The `nop`s in front keep the first byte from being the `%fs` prefix, so
///   the local-dynamic arm of the old test is the one this reaches.
///
/// Every probe is followed by `ret` so the fixture is still valid code, and
/// `_start` opens the section so no probe sits closer than twelve bytes to its
/// start.
const SRC: &[u8] = b"    .text\n\
    .globl _start\n\
_start:\n\
    ret\n\
    .globl le_probe\n\
le_probe:\n\
    .byte 0x64, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00\n\
    .byte 0x48, 0x8d, 0x80\n\
    .reloc ., R_X86_64_PLT32, callee-4\n\
    .long 0\n\
    ret\n\
    .globl ie_probe\n\
ie_probe:\n\
    .byte 0x64, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00\n\
    .byte 0x48, 0x03, 0x05\n\
    .reloc ., R_X86_64_PLT32, callee-4\n\
    .long 0\n\
    ret\n\
    .globl ld_probe\n\
ld_probe:\n\
    .byte 0x90, 0x90, 0x90, 0x90\n\
    .byte 0x66, 0x66, 0x66, 0x64, 0x48, 0x8b, 0x04, 0x25\n\
    .reloc ., R_X86_64_PLT32, callee-4\n\
    .long 0\n\
    ret\n\
    .globl callee\n\
callee:\n\
    ret\n";

/// The offset of the `PLT32` slot from the probe symbol: the twelve bytes the
/// old test matched sit in front of it.
const SLOT: u64 = 12;

/// The addend the fixture's relocations carry, the canonical `-4` of a
/// PC-relative reference whose displacement is measured from the end of its
/// own field.
const ADDEND: i64 = -4;

/// Assembles `SRC` with the host clang, or `None` when it is absent.
fn assemble(obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src = obj.with_extension("S");
    fs::write(&src, SRC).ok()?;
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src);
    ok.then_some(())
}

/// A fresh per-test working directory under the system temp dir.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xold_lookalike_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Assembles and links the fixture, returning the image bytes, or `None` when
/// the host has no clang.
fn build(prefix: &str, relax: bool) -> Option<Vec<u8>> {
    let dir = workdir(prefix);
    let obj = dir.join("lookalike.o");
    assemble(&obj)?;
    let prog = dir.join("prog");
    link_to(
        std::slice::from_ref(&obj),
        &prog,
        b"_start",
        false,
        IcfMode::None,
        relax,
    )
    .expect("a program whose calls only look like TLS must link");
    fs::read(&prog).ok()
}

/// The `st_value` of a global symbol in the output symtab.
fn symbol_value(obj: &ObjectFile<'_>, name: &[u8]) -> u64 {
    let symtab = obj
        .symbol_table()
        .ok()
        .flatten()
        .expect("output has a symbol table");
    for sym in symtab.iter() {
        if symtab.name(sym) == name {
            return sym.st_value.get();
        }
    }
    panic!(
        "output defines no symbol named {}",
        String::from_utf8_lossy(name)
    );
}

/// The `len` bytes stored at `vaddr`, read out of the section that covers it.
fn read_at<'a>(obj: &'a ObjectFile<'a>, vaddr: u64, len: u64) -> &'a [u8] {
    for shdr in obj.sections() {
        let addr = shdr.sh_addr.get();
        let end = addr.saturating_add(shdr.sh_size.get());
        if vaddr < addr || vaddr.saturating_add(len) > end {
            continue;
        }
        let Ok(data) = obj.section_data(shdr) else {
            continue;
        };
        let at = usize::try_from(vaddr - addr).unwrap_or(usize::MAX);
        let n = usize::try_from(len).unwrap_or(usize::MAX);
        if let Some(slice) = data.get(at..at.saturating_add(n)) {
            return slice;
        }
    }
    panic!("no section covers {len} bytes at {vaddr:#x}");
}

/// Checks one probe: the twelve bytes in front of the slot are still the ones
/// the fixture wrote, and the slot holds `S + A - P`.
fn check_probe(bytes: &[u8], probe: &[u8], lead: &[u8]) {
    let obj = ObjectFile::parse(bytes).expect("valid ELF output");
    let base = symbol_value(&obj, probe);
    let place = base + SLOT;
    let callee = symbol_value(&obj, b"callee");
    let want = i64::try_from(callee).expect("address fits") + ADDEND
        - i64::try_from(place).expect("address fits");
    let want = i32::try_from(want).expect("a displacement fits 32 bits");
    assert_ne!(want, 0, "the fixture must need a non-zero displacement");
    let stored = read_at(&obj, place, 4);
    let stored: [u8; 4] = stored.try_into().expect("four bytes");
    assert_eq!(
        i32::from_le_bytes(stored),
        want,
        "a PLT32 site that merely looks like a lowered TLS call must still \
         be relocated"
    );
    assert_eq!(
        read_at(&obj, base, SLOT),
        lead,
        "the bytes in front of the probe must be left alone"
    );
}

/// The local-exec bytes a lowered general-dynamic pair is rewritten into.
const LE_LEAD: &[u8] = &[
    0x64, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, 0x48, 0x8d, 0x80,
];

/// The initial-exec bytes it is rewritten into instead when a shared object
/// places the thread-local.
const IE_LEAD: &[u8] = &[
    0x64, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, 0x48, 0x03, 0x05,
];

/// The four `nop`s plus the first eight bytes of a lowered local-dynamic
/// sequence, which is what fills the window in front of that probe.
const LD_LEAD: &[u8] = &[
    0x90, 0x90, 0x90, 0x90, 0x66, 0x66, 0x66, 0x64, 0x48, 0x8b, 0x04, 0x25,
];

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_call_that_looks_like_a_lowered_gd_pair_is_still_relocated() {
    let Some(bytes) = build("le", false) else {
        eprintln!("skipping TLS look-alike test: clang unavailable");
        return;
    };
    check_probe(&bytes, b"le_probe", LE_LEAD);
    check_probe(&bytes, b"ie_probe", IE_LEAD);
    check_probe(&bytes, b"ld_probe", LD_LEAD);
}

/// The lowering runs whether or not relaxation was asked for, so the property
/// has to hold in both links.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn relaxation_does_not_reintroduce_the_byte_test() {
    let Some(bytes) = build("relax", true) else {
        eprintln!("skipping TLS look-alike relax test: clang unavailable");
        return;
    };
    check_probe(&bytes, b"le_probe", LE_LEAD);
    check_probe(&bytes, b"ie_probe", IE_LEAD);
    check_probe(&bytes, b"ld_probe", LD_LEAD);
}
