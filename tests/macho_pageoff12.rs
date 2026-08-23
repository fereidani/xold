//! The arm64 page-offset encodings read the instruction they patch.
//!
//! `PAGEOFF12` and `GOT_LOAD_PAGEOFF12` hold the in-page part of an address,
//! shifted right by the access size the opcode implies: zero for `ADD`, three
//! for a 64-bit load. That is two operations -- mask to the page, then scale
//! -- and a `Field` can only do one: it shifts the whole value. Encoding
//! `GOT_LOAD_PAGEOFF12` as a scaled field shifted the entire address right by
//! three, so address bits 12 to 14 landed inside the immediate and every
//! folded `adrp + ldr` global access read eight times its offset. `PAGEOFF12`
//! had no scale at all, so a 64-bit load got the byte offset where it needed
//! the doubleword index.
//!
//! Both go through the escape path now, which reads the opcode for the scale
//! and encodes only the in-page bits, as lld does
//! (`Arch/ARM64Common.h:88-104`). `write_adr` also range-checks the page
//! distance, which used to truncate silently.
//!
//! Darwin images cannot run on this host, so these check the encoded
//! instruction words.

use xold::{
    reloc::{
        Resolver, apply,
        macho_arm64::{
            ARM64_RELOC_GOT_LOAD_PAGE21, ARM64_RELOC_GOT_LOAD_PAGEOFF12,
            ARM64_RELOC_PAGE21, ARM64_RELOC_PAGEOFF12, MachoArm64,
        },
    },
    symbol::SymbolId,
};

/// `add x0, x0, #0`, the instruction a `PAGEOFF12` against an `ADD` patches.
const ADD: u32 = 0x9100_0000;
/// `ldr x0, [x0, #0]`, a 64-bit load: scale 3.
const LDR64: u32 = 0xf940_0000;
/// `ldrb w0, [x0, #0]`, a byte load: scale 0.
const LDRB: u32 = 0x3940_0000;

/// An `ADD` takes the in-page bits unshifted.
#[test]
fn an_add_takes_the_offset_as_it_is() {
    let out = encode(ADD, ARM64_RELOC_PAGEOFF12, 0x1_0000_2abc);
    assert_eq!(imm12(out), 0xabc, "the low twelve bits, and nothing above");
}

/// A 64-bit load takes them divided by eight.
#[test]
fn a_doubleword_load_takes_the_scaled_index() {
    let out = encode(LDR64, ARM64_RELOC_PAGEOFF12, 0x1_0000_2ab8);
    assert_eq!(
        imm12(out),
        0xab8 / 8,
        "a 64-bit load indexes doublewords, so the immediate is the offset \
         over eight"
    );
}

/// A byte load takes them unshifted, like the `ADD`.
#[test]
fn a_byte_load_takes_the_offset_as_it_is() {
    let out = encode(LDRB, ARM64_RELOC_PAGEOFF12, 0x1_0000_2abc);
    assert_eq!(imm12(out), 0xabc);
}

/// No address bit above eleven reaches the immediate.
#[test]
fn the_page_bits_do_not_leak_into_the_immediate() {
    // Two addresses that agree in their low twelve bits and differ above.
    let a = encode(LDR64, ARM64_RELOC_PAGEOFF12, 0x1_0000_0008);
    let b = encode(LDR64, ARM64_RELOC_PAGEOFF12, 0x1_0007_7008);
    assert_eq!(
        imm12(a),
        imm12(b),
        "only the in-page bits are encoded, so two addresses in the same \
         position on different pages encode identically"
    );
    assert_eq!(imm12(a), 1, "and eight bytes in is index one");
}

/// The GOT form scales the same way, from the same opcode.
#[test]
fn the_got_form_reads_its_own_opcode() {
    let out = encode(LDR64, ARM64_RELOC_GOT_LOAD_PAGEOFF12, 0);
    // The stub resolver puts the GOT slot at 0x1_0000_4020.
    assert_eq!(imm12(out), 0x20 / 8, "the slot's in-page index");
}

/// The rest of the instruction is untouched.
#[test]
fn the_opcode_survives_the_patch() {
    let out = encode(LDR64, ARM64_RELOC_PAGEOFF12, 0x1_0000_2ab8);
    assert_eq!(
        out & !(0xfff << 10),
        LDR64,
        "only the immediate field is written"
    );
}

/// An unaligned access is refused rather than encoded as a truncated index.
#[test]
fn an_unaligned_access_is_refused() {
    let mut slot = LDR64.to_le_bytes();
    let res = apply::<MachoArm64, Stub>(
        ARM64_RELOC_PAGEOFF12,
        Some(SymbolId(0)),
        0x1_0000_2ab9_i64.wrapping_sub(TARGET.cast_signed()),
        0,
        &Stub,
        &mut slot,
    );
    assert!(
        res.is_err(),
        "a 64-bit load at an odd address cannot be encoded, and rounding it \
         down would read the wrong doubleword"
    );
}

/// A page distance past what the instruction holds is refused.
#[test]
fn an_out_of_range_page_distance_is_refused() {
    let mut slot = 0x9000_0000u32.to_le_bytes();
    // ADRP holds 21 signed bits of page count: +/- 4 GiB. Place the site 8 GiB
    // away from the symbol.
    let res = apply::<MachoArm64, Stub>(
        ARM64_RELOC_PAGE21,
        Some(SymbolId(0)),
        0,
        TARGET.wrapping_add(0x2_0000_0000),
        &Stub,
        &mut slot,
    );
    assert!(
        res.is_err(),
        "truncating this silently points the ADRP at a page 4 GiB from the \
         one asked for"
    );
    // The in-range case still encodes.
    let mut near = 0x9000_0000u32.to_le_bytes();
    let ok = apply::<MachoArm64, Stub>(
        ARM64_RELOC_GOT_LOAD_PAGE21,
        Some(SymbolId(0)),
        0,
        TARGET,
        &Stub,
        &mut near,
    );
    assert!(ok.is_ok(), "an ordinary distance still encodes: {ok:?}");
}

// --- fixtures --------------------------------------------------------------

/// The address the stub resolver reports for every symbol.
const TARGET: u64 = 0x1_0000_2000;

/// A resolver with one symbol and one GOT slot at fixed addresses.
struct Stub;

impl Resolver for Stub {
    fn symbol_addr(&self, _sym: SymbolId) -> u64 {
        TARGET
    }
    fn got_addr(&self, _sym: SymbolId) -> u64 {
        0x1_0000_4020
    }
    fn got_base(&self) -> u64 {
        0
    }
    fn plt_addr(&self, sym: SymbolId) -> u64 {
        self.symbol_addr(sym)
    }
}

/// Applies `r_type` to `base` for the address `va`, returning the patched
/// instruction word.
fn encode(base: u32, r_type: u32, va: u64) -> u32 {
    let mut slot = base.to_le_bytes();
    let addend = va.cast_signed().wrapping_sub(TARGET.cast_signed());
    apply::<MachoArm64, Stub>(
        r_type,
        Some(SymbolId(0)),
        if r_type == ARM64_RELOC_GOT_LOAD_PAGEOFF12 {
            0
        } else {
            addend
        },
        0,
        &Stub,
        &mut slot,
    )
    .expect("the fixture must encode");
    u32::from_le_bytes(slot)
}

/// The 12-bit immediate at instruction bits [10..21].
const fn imm12(word: u32) -> u32 {
    (word >> 10) & 0xfff
}
