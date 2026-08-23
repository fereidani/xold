//! A PC-relative site against an undefined weak takes the architecture's
//! answer, not `A - P`.
//!
//! A weak reference nothing defines is the feature-probe idiom: the program
//! tests the symbol's address and takes the other branch when it is absent.
//! With `S = 0` the plain PC-relative arithmetic stores `A - P`, and a branch
//! field holding `-P` jumps to absolute address zero -- xold sent every
//! `AArch64` `bl w` and every `RISC-V` `j w` there, silently, because the value
//! still fit the field.
//!
//! lld resolves each architecture by rule beside its `R_PC` case
//! (`lld/ELF/InputSection.cpp`): an `AArch64` branch goes to the
//! next instruction and anything else PC-relative takes the place's own
//! address; a `RISC-V` branch goes to itself, the deliberate infinite loop that
//! makes the miss visible; x86-64 alone keeps `A - P`. The first test pins
//! those values through the relocation driver with a fixed resolver; the
//! second links an `AArch64` object and decodes the instructions it holds.
//!
//! Gated on `clang` for the linked fixture; the value test runs anywhere.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{
    icf::IcfMode,
    linker::{link_dyn_exec, link_to},
    reloc::{
        Resolver, Target,
        aarch64::{
            R_AARCH64_ADR_PREL_LO21, R_AARCH64_CALL26, R_AARCH64_CONDBR19,
            R_AARCH64_PLT32, R_AARCH64_PREL64,
        },
        apply_target,
        riscv::{R_RISCV_32_PCREL, R_RISCV_JAL},
        x86_64::R_X86_64_PC32,
    },
    symbol::SymbolId,
};

mod common;

/// A resolver over one symbol at address zero, standing in for the weak
/// reference nothing defines.
struct UndefWeak;

impl Resolver for UndefWeak {
    fn symbol_addr(&self, _sym: SymbolId) -> u64 {
        0
    }
    fn got_addr(&self, _sym: SymbolId) -> u64 {
        0
    }
    fn got_base(&self) -> u64 {
        0
    }
    fn plt_addr(&self, _sym: SymbolId) -> u64 {
        0
    }
    fn is_undef_weak(&self, _sym: SymbolId) -> bool {
        true
    }
}

/// The value each architecture stores for an undefined weak.
#[test]
fn each_architecture_answers_by_rule() {
    let weak = Some(SymbolId(0));
    let resolver = UndefWeak;
    let place = 0x0040_0000_u64;
    aarch64_branches_next_instruction(weak, &resolver, place);
    aarch64_data_forms_take_the_addend(weak, &resolver, place);
    riscv_and_x86_keep_the_plain_form(weak, &resolver, place);
}

/// `AArch64` branches go to the next instruction: the stored offset is 4.
fn aarch64_branches_next_instruction(
    weak: Option<SymbolId>,
    resolver: &UndefWeak,
    place: u64,
) {
    let mut out = 0x9400_0000u32.to_le_bytes();
    apply_target(
        Target::AArch64,
        R_AARCH64_CALL26,
        weak,
        0,
        place,
        resolver,
        &mut out,
    )
    .expect("CALL26 applies");
    assert_eq!(
        imm26(u32::from_le_bytes(out)),
        4,
        "a call to an undefined weak lands on the next instruction"
    );
    let mut out = 0x5400_0000u32.to_le_bytes();
    apply_target(
        Target::AArch64,
        R_AARCH64_CONDBR19,
        weak,
        0,
        place,
        resolver,
        &mut out,
    )
    .expect("CONDBR19 applies");
    assert_eq!(
        imm19(u32::from_le_bytes(out)),
        4,
        "and so does a conditional branch"
    );
}

/// Everything else PC-relative on `AArch64` stores the addend.
fn aarch64_data_forms_take_the_addend(
    weak: Option<SymbolId>,
    resolver: &UndefWeak,
    place: u64,
) {
    let mut out = [0u8; 8];
    apply_target(
        Target::AArch64,
        R_AARCH64_PREL64,
        weak,
        8,
        place,
        resolver,
        &mut out,
    )
    .expect("PREL64 applies");
    assert_eq!(
        u64::from_le_bytes(out),
        8,
        "a data reference to an undefined weak is its own addend"
    );
    let mut out = [0u8; 4];
    apply_target(
        Target::AArch64,
        R_AARCH64_PLT32,
        weak,
        8,
        place,
        resolver,
        &mut out,
    )
    .expect("PLT32 applies");
    assert_eq!(
        u32::from_le_bytes(out),
        8,
        "a PLT-class reference without a PLT entry is data here"
    );
    let mut out = 0x1000_0000u32.to_le_bytes();
    apply_target(
        Target::AArch64,
        R_AARCH64_ADR_PREL_LO21,
        weak,
        0x10,
        place,
        resolver,
        &mut out,
    )
    .expect("ADR_PREL_LO21 applies");
    assert_eq!(
        imm21(u32::from_le_bytes(out)),
        0x10,
        "the ADR form is data too: the address it forms is its own place \
         plus the addend"
    );
}

/// A `RISC-V` branch goes to itself, and the architectures with no special
/// rule keep `A - P`: `RISC-V` outside the branch set, and x86-64 everywhere.
fn riscv_and_x86_keep_the_plain_form(
    weak: Option<SymbolId>,
    resolver: &UndefWeak,
    place: u64,
) {
    let mut out = 0xFFFF_F06Fu32.to_le_bytes();
    apply_target(
        Target::Riscv64,
        R_RISCV_JAL,
        weak,
        0,
        place,
        resolver,
        &mut out,
    )
    .expect("JAL applies");
    assert_eq!(
        j_imm(u32::from_le_bytes(out)),
        0,
        "a jump to an undefined weak targets its own instruction, the \
         infinite loop that makes the miss visible"
    );
    for (target, r_type) in [
        (Target::Riscv64, R_RISCV_32_PCREL),
        (Target::X86_64, R_X86_64_PC32),
    ] {
        let mut out = [0u8; 4];
        apply_target(target, r_type, weak, 0, place, resolver, &mut out)
            .unwrap_or_else(|e| panic!("applies: {e:?}"));
        assert_eq!(
            u32::from_le_bytes(out),
            0u32.wrapping_sub(u32::try_from(place).expect("in range")),
            "the architecture with no special rule keeps A - P"
        );
    }
}

/// The linked `AArch64` fixture: a call, a conditional branch and a data
/// reference to one undefined weak.
const SRC: &[u8] = b"    .text\n\
    .globl _start\n\
    .type _start,@function\n\
_start:\n\
    bl w\n\
    b.eq w\n\
    adr x0, w\n\
    ret\n\
    .weak w\n\
    .data\n\
    .globl d64\n\
    .type d64,@object\n\
d64: .quad w - .\n";

/// The instructions and the datum carry the rule, in a real link.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_aarch64_link_resolves_the_weak_reference_by_rule() {
    let Some(dir) = workdir("rule") else {
        return;
    };
    let clang = which("clang").expect("workdir checked clang");
    let src = dir.join("s.S");
    let obj = dir.join("s.o");
    fs::write(&src, SRC).expect("write source");
    let ok = Command::new(clang)
        .args(["--target=aarch64-linux-gnu", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("clang runs")
        .success();
    assert!(ok, "clang assembles the fixture");
    let out = dir.join("prog");
    link_to(
        std::slice::from_ref(&obj),
        &out,
        b"_start",
        false,
        IcfMode::None,
        false,
    )
    .expect("the fixture links");
    let image = fs::read(&out).expect("read the image");
    let (text_addr, text) = section(&image, b".text").expect(".text present");
    let start = symbol_value(&image, b"_start").expect("_start defined");

    let base = usize::try_from(start - text_addr).expect("in .text");
    let word = |at: usize| {
        u32::from_le_bytes(
            <[u8; 4]>::try_from(&text[at..at + 4]).expect("four bytes"),
        )
    };
    assert_eq!(
        imm26(word(base)),
        4,
        "the call branches to the next instruction, not to address zero"
    );
    assert_eq!(imm19(word(base + 4)), 4, "so does the conditional branch");
    assert_eq!(
        imm21(word(base + 8)),
        0,
        "the ADR forms its own address: displacement zero"
    );
    let (_, data) = section(&image, b".data").expect(".data present");
    let d64 = symbol_value(&image, b"d64").expect("d64 defined");
    let data_addr = section_addr(&image, b".data").expect(".data present");
    let at = usize::try_from(d64 - data_addr).expect("in .data");
    let d64_bytes =
        <[u8; 8]>::try_from(&data[at..at + 8]).expect("eight bytes");
    assert_eq!(
        u64::from_le_bytes(d64_bytes),
        0,
        "the datum holds w - here, which is the addend alone"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A hidden weak reference beside a dependency that defines the name: the
/// hidden row never reaches `.dynsym`, so the loader cannot bind it and the
/// site must still take the architecture's undefined-weak answer.
const HIDDEN_SRC: &[u8] = b"    .text\n\
    .globl _start\n\
    .type _start,@function\n\
_start:\n\
    bl w\n\
    ret\n\
    .weak w\n\
    .hidden w\n";

/// The dependency's own definition of the name the hidden reference spells.
const HIDDEN_DEP_SRC: &str = "void w(void) {}\n";

/// A dependency's export must not demote the undefined-weak answer for a
/// hidden reference: hidden never reaches `.dynsym`, so nothing binds it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_hidden_weak_reference_ignores_a_dependency_export() {
    let Some(dir) = workdir("hidden") else {
        return;
    };
    let clang = which("clang").expect("workdir checked clang");
    let src = dir.join("h.S");
    let obj = dir.join("h.o");
    fs::write(&src, HIDDEN_SRC).expect("write source");
    let ok = Command::new(&clang)
        .args(["--target=aarch64-linux-gnu", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("clang runs")
        .success();
    assert!(ok, "clang assembles the fixture");
    let dep_src = dir.join("libw.c");
    let dep = dir.join("libw.so");
    fs::write(&dep_src, HIDDEN_DEP_SRC).expect("write dependency");
    let ok = Command::new(&clang)
        .args([
            "--target=aarch64-linux-gnu",
            "-shared",
            "-fPIC",
            "-nostdlib",
            // The host BFD linker has no aarch64 emulation; lld is the
            // cross-linker the test fixtures rely on throughout.
            "-fuse-ld=lld",
            "-Wl,-soname,libw.so",
            "-o",
        ])
        .arg(&dep)
        .arg(&dep_src)
        .status()
        .expect("clang runs")
        .success();
    if !ok {
        // Cross-linking the dependency needs lld; a host without it can
        // still run every other test here.
        eprintln!("skipping hidden-weak link: no aarch64 cross-linker");
        return;
    }
    let out = dir.join("hprog");
    link_dyn_exec(
        &[obj, dep],
        &out,
        b"_start",
        b"/lib/ld-linux-aarch64.so.1",
        false,
        IcfMode::None,
        false,
    )
    .expect("the fixture links");
    let image = fs::read(&out).expect("read the image");
    let (text_addr, text) = section(&image, b".text").expect(".text present");
    let start = symbol_value(&image, b"_start").expect("_start defined");
    let base = usize::try_from(start - text_addr).expect("in .text");
    let word = u32::from_le_bytes(
        <[u8; 4]>::try_from(&text[base..base + 4]).expect("four bytes"),
    );
    assert_eq!(
        imm26(word),
        4,
        "the hidden weak call branches to the next instruction, not to \
         whatever the plain arithmetic stores"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures ---------------------------------------------------------------

/// Keyed by test name as well as pid: the linked tests run in parallel
/// threads of one process, and each wipes its directory on entry.
fn workdir(test: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping undef-weak-pc link: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_uwpc_{test}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

// --- readers ----------------------------------------------------------------

/// The signed 26-bit immediate of a B/BL instruction, as a byte offset from
/// the place.
fn imm26(word: u32) -> i64 {
    let raw = i64::from(word & 0x03FF_FFFF);
    sign_extend(raw << 2, 28)
}

/// The signed 19-bit immediate of a B.cond instruction, as a byte offset.
fn imm19(word: u32) -> i64 {
    let raw = i64::from((word >> 5) & 0x7_FFFF);
    sign_extend(raw << 2, 21)
}

/// The signed 21-bit immediate of an ADR instruction, as a byte offset.
fn imm21(word: u32) -> i64 {
    let lo = u64::from((word >> 29) & 0x3);
    let hi = u64::from((word >> 5) & 0x7_FFFF);
    sign_extend((hi << 2 | lo).cast_signed(), 21)
}

/// The signed immediate of a J-type instruction, as a byte offset.
fn j_imm(word: u32) -> i64 {
    let imm = (u64::from(word >> 31) << 20)
        | (u64::from((word >> 21) & 0x3FF) << 1)
        | (u64::from((word >> 20) & 0x1) << 11)
        | (u64::from((word >> 12) & 0xFF) << 12);
    sign_extend(imm.cast_signed(), 21)
}

/// Sign-extends the low `bits` of `value`.
const fn sign_extend(value: i64, bits: u32) -> i64 {
    let shift = 64 - bits;
    (value << shift) >> shift
}

/// The `st_value` of a symbol in the output's `.symtab`.
fn symbol_value(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = xold::elf::ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab
        .iter()
        .find(|s| symtab.name(s) == name)
        .map(|s| s.st_value.get())
}

/// The virtual address of a named output section.
fn section_addr(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = xold::elf::ObjectFile::parse(bytes).ok()?;
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == name)
        .map(|s| s.sh_addr.get())
}

/// `(address, bytes)` of a named output section.
fn section(bytes: &[u8], name: &[u8]) -> Option<(u64, Vec<u8>)> {
    let obj = xold::elf::ObjectFile::parse(bytes).ok()?;
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == name)?;
    let data = obj.section_data(shdr).ok().map(<[u8]>::to_vec)?;
    Some((shdr.sh_addr.get(), data))
}
