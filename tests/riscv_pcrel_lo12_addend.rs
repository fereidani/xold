//! A `PCREL_LO12` names its `auipc` as symbol plus addend, and the addend
//! counts.
//!
//! `%pcrel_lo` does not name the target of the access; it names the `auipc`
//! that computed the high half, so the linker can recover the same PC the
//! `auipc` used. The assembler spells that as a local label, but a tool that
//! drops local symbols -- `objcopy -x`, `ld -r` -- rewrites the reference
//! against the section symbol and moves the whole offset into the addend.
//!
//! `rewrite_pcrel_pairs` looked the pair up at `symbol_addr(sym) -
//! section_base` and never read the addend, so every such relocation looked up
//! the *start of the section*. Usually that misses and the link fails with "no
//! paired HI20 at its label", which is merely a refusal of a valid input. When
//! an `auipc` does sit at offset 0 -- and in a `.text` that starts with one, it
//! does -- the LO12 pairs with the wrong `auipc` and stores the low bits of a
//! different symbol's displacement. The link succeeds and the access reads the
//! wrong address.
//!
//! lld folds the addend into the key at the same point:
//! `hiReloc.offset = d->value + addend` in `getPCRelHi20`
//! (`lld/ELF/InputSection.cpp`).
//!
//! The link tests are gated on a clang that can target `riscv64`; without one
//! they print a note and return. The unit test needs no toolchain.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::{ObjectFile, Rela64},
    endian::{I64, U64},
    icf::IcfMode,
    linker::link_to,
    reloc::{
        Resolver,
        riscv::{
            R_RISCV_PCREL_HI20, R_RISCV_PCREL_LO12_I, rewrite_pcrel_pairs,
        },
    },
    symbol::SymbolId,
};

mod common;

/// A resolver that puts every symbol at the section start, which is what a
/// section symbol resolves to. The addend is then the only thing that can
/// distinguish one `auipc` from another.
struct SectionStart(u64);

impl Resolver for SectionStart {
    fn symbol_addr(&self, _: SymbolId) -> u64 {
        self.0
    }
    fn got_addr(&self, _: SymbolId) -> u64 {
        0
    }
    fn got_base(&self) -> u64 {
        0
    }
    fn plt_addr(&self, _: SymbolId) -> u64 {
        0
    }
}

/// Packs an `r_info` word the way the ELF64 layout does.
const fn info(sym: u32, r_type: u32) -> u64 {
    ((sym as u64) << 32) | r_type as u64
}

/// Two `auipc` sites and one LO12 naming the second of them through the
/// section symbol plus an addend. The pair must be the second, not the first.
#[test]
fn the_addend_selects_which_auipc_a_lo12_pairs_with() {
    let section_base = 0x1000;
    let entries = [
        // auipc at offset 0, against symbol 7.
        Rela64 {
            r_offset: U64::new(0),
            r_info: U64::new(info(7, R_RISCV_PCREL_HI20)),
            r_addend: I64::new(0),
        },
        // auipc at offset 8, against symbol 9.
        Rela64 {
            r_offset: U64::new(8),
            r_info: U64::new(info(9, R_RISCV_PCREL_HI20)),
            r_addend: I64::new(0),
        },
        // The LO12 at offset 0xc names `.text + 8`, so it pairs with the
        // second auipc.
        Rela64 {
            r_offset: U64::new(0xc),
            r_info: U64::new(info(1, R_RISCV_PCREL_LO12_I)),
            r_addend: I64::new(8),
        },
    ];
    let rewritten = rewrite_pcrel_pairs(
        &entries,
        section_base,
        &SectionStart(section_base),
    )
    .expect("the pair resolves")
    .expect("a LO12 was rewritten");
    let lo12 = &rewritten[2];
    assert_eq!(
        lo12.sym(),
        9,
        "the addend names the auipc at offset 8, whose symbol is 9; pairing \
         with the one at offset 0 would silently store symbol 7's low bits"
    );
    // The rewritten addend carries the LO12 site back to the auipc's PC:
    // hi_addend + lo_offset - hi_off = 0 + 0xc - 8.
    assert_eq!(
        lo12.r_addend.get(),
        4,
        "the addend must measure from the paired auipc, not from the section"
    );
}

/// End to end, and the reason this is a tier-1 defect rather than a refusal:
/// the link succeeds either way, and only the encoded low bits differ.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_section_relative_lo12_reaches_its_own_symbol() {
    let Some(dir) = workdir("pair") else {
        return;
    };
    let Some(obj) = assemble(&dir, "pair") else {
        return;
    };
    let out = dir.join("pair");
    let res = link_to(
        std::slice::from_ref(&obj),
        &out,
        b"_start",
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let bytes = fs::read(&out).expect("read output");

    let alpha = symbol_value(&bytes, b"alpha").expect("alpha present");
    let beta = symbol_value(&bytes, b"beta").expect("beta present");
    assert_ne!(alpha, beta, "the two targets must be distinct");

    // Each pair is `auipc rd, hi` then `addi rd, rd, lo`, so the address the
    // sequence computes is the auipc's PC plus the two halves.
    let text = section_addr(&bytes, b".text").expect(".text present");
    let code = section_bytes(&bytes, b".text").expect(".text data");
    let first = computed(&code, text, 0).expect("first pair decodes");
    let second = computed(&code, text, 8).expect("second pair decodes");
    assert_eq!(first, alpha, "the first pair must reach alpha");
    assert_eq!(
        second, beta,
        "the second pair must reach beta; pairing its LO12 with the auipc at \
         offset 0 puts alpha's low bits on beta's high bits, which lands past \
         the end of beta"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Two `%pcrel_hi`/`%pcrel_lo` pairs whose LO12 relocations are written by
/// hand against the section symbol, which is the form a local-symbol strip
/// leaves behind. `rv64g` keeps every instruction 4 bytes wide so the offsets
/// in the `.reloc` directives are the ones written here.
const PAIR_SRC: &[u8] = b"    .text\n\
    .globl _start\n\
    .type _start,@function\n\
_start:\n\
    auipc   a0, %pcrel_hi(alpha)\n\
    addi    a0, a0, 0\n\
    auipc   a1, %pcrel_hi(beta)\n\
    addi    a1, a1, 0\n\
    ret\n\
    .size _start, .-_start\n\
    .reloc 4,  R_RISCV_PCREL_LO12_I, .text+0\n\
    .reloc 12, R_RISCV_PCREL_LO12_I, .text+8\n\
    .section .rodata,\"a\",@progbits\n\
    .globl alpha\n\
alpha:  .word 0x11111111\n\
    .globl beta\n\
beta:   .word 0x22222222\n";

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping riscv-pcrel-lo12-addend {prefix}: no clang");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_rvlo12_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Assembles the fixture, or `None` when this clang cannot target `riscv64`.
fn assemble(dir: &Path, stem: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join(format!("{stem}.S"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src, PAIR_SRC).ok()?;
    let built = Command::new(clang)
        .args([
            "--target=riscv64-unknown-elf",
            "-march=rv64g",
            "-mno-relax",
            "-c",
        ])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping riscv-pcrel-lo12-addend: no riscv64 assembler");
        return None;
    }
    Some(obj)
}

// --- readers ---------------------------------------------------------------

/// The address an `auipc`/`addi` pair starting at `off` within `.text`
/// computes: the `auipc`'s own PC plus the U-type hi20 plus the sign-extended
/// I-type lo12.
fn computed(code: &[u8], text: u64, off: usize) -> Option<u64> {
    let auipc = cell(code, off)?;
    let addi = cell(code, off + 4)?;
    let hi = u64::from(auipc & 0xFFFF_F000);
    let lo = i64::from(addi.cast_signed() >> 20);
    Some(
        text.wrapping_add(off as u64)
            .wrapping_add(hi)
            .wrapping_add_signed(lo),
    )
}

/// A 4-byte little-endian instruction cell at `off`.
fn cell(code: &[u8], off: usize) -> Option<u32> {
    let raw = code.get(off..off + 4)?;
    Some(u32::from_le_bytes(<[u8; 4]>::try_from(raw).ok()?))
}

/// The `st_value` of a global symbol in the output's `.symtab`.
fn symbol_value(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab
        .iter()
        .find(|s| symtab.name(s) == name)
        .map(|s| s.st_value.get())
}

/// The virtual address of an output section.
fn section_addr(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == name)
        .map(|s| s.sh_addr.get())
}

/// The bytes of an output section.
fn section_bytes(bytes: &[u8], name: &[u8]) -> Option<Vec<u8>> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == name)?;
    obj.section_data(shdr).ok().map(<[u8]>::to_vec)
}
