//! Two absolute-store range checks: one that was missing, one too narrow.
//!
//! `R_RISCV_SET32` stored its value with no range check at all. It is the only
//! `SET` that can overflow -- SET6, SET8 and SET16 write a label difference
//! the assembler already sized, and lld range-checks none of them -- but
//! SET32 shares its width with `32_PCREL` and `PLT32`, and lld checks all
//! three together (`lld/ELF/Arch/RISCV.cpp`). Without it a value
//! past 2^31 was stored truncated, so the difference a section records is not
//! the difference between the two labels it names.
//!
//! `R_X86_64_16` accepted only the unsigned range, so a small negative
//! constant in a 16-bit absolute store was refused where lld takes it
//! (`checkIntUInt`, `[-32768, 65535]`). mold agrees with the narrower reading,
//! which makes this a judgement call; it goes to lld, the reference this
//! project measures itself against, and the change only ever accepts more.
//!
//! Both are driven through the relocation appliers directly: the values are at
//! the edges of a 32-bit and a 16-bit field, which no fixture of ordinary size
//! reaches.

use xold::{
    reloc::{
        Resolver, apply,
        riscv::{R_RISCV_32, R_RISCV_SET16, R_RISCV_SET32, Riscv64},
        x86_64::{R_X86_64_16, X86_64},
    },
    symbol::SymbolId,
};

/// A resolver placing the symbol at a fixed address.
struct At(u64);

impl Resolver for At {
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
        self.0
    }
}

/// Applies `r_type` with the symbol at `at` and no addend.
fn write<A: xold::reloc::Arch>(
    r_type: u32,
    at: u64,
    slot: &mut [u8],
) -> Result<(), xold::Error> {
    apply::<A, _>(r_type, Some(SymbolId(1)), 0, 0, &At(at), slot)
}

/// A `SET32` value past the signed 32-bit range is refused.
#[test]
fn a_riscv_set32_past_its_field_is_refused() {
    let mut slot = [0u8; 4];
    let err = write::<Riscv64>(R_RISCV_SET32, 0x1_0000_0000, &mut slot)
        .expect_err(
            "a value wider than the field cannot be stored, and truncating it \
             makes the recorded difference not the difference between the two \
             labels",
        );
    assert!(
        matches!(err, xold::Error::RelocOverflow(t) if t == R_RISCV_SET32),
        "the refusal must name the relocation, got {err:?}"
    );
}

/// Values inside the field still store, at both ends.
#[test]
fn a_riscv_set32_inside_its_field_still_stores() {
    let mut low = [0u8; 4];
    write::<Riscv64>(R_RISCV_SET32, 0x7fff_ffff, &mut low)
        .expect("the largest positive value fits");
    assert_eq!(low, [0xff, 0xff, 0xff, 0x7f]);

    let mut neg = [0u8; 4];
    write::<Riscv64>(R_RISCV_SET32, u64::MAX, &mut neg)
        .expect("minus one fits the signed range");
    assert_eq!(neg, [0xff, 0xff, 0xff, 0xff]);
}

/// The narrower `SET`s are unaffected: they were never range-checked, here or
/// in lld, because the assembler sized what they carry.
#[test]
fn the_narrower_sets_are_unchanged() {
    let mut slot = [0u8; 2];
    write::<Riscv64>(R_RISCV_SET16, 0x1_2345, &mut slot)
        .expect("SET16 stores what it is given");
    assert_eq!(slot, [0x45, 0x23]);
}

/// A 32-bit RISC-V absolute store takes a negative label difference.
///
/// The psABI gives `R_RISCV_32` no signedness and lld writes it unchecked, so
/// `.word sym - bigger_sym` is ordinary; it was refused here and accepted
/// everywhere else.
#[test]
fn a_riscv_32_takes_a_negative_difference() {
    let mut slot = [0u8; 4];
    write::<Riscv64>(R_RISCV_32, u64::MAX, &mut slot)
        .expect("a negative difference is a legitimate 32-bit store");
    assert_eq!(slot, [0xff, 0xff, 0xff, 0xff]);

    let mut high = [0u8; 4];
    write::<Riscv64>(R_RISCV_32, 0xffff_ffff, &mut high)
        .expect("and so is a large unsigned address");
    assert_eq!(high, [0xff, 0xff, 0xff, 0xff]);

    let mut over = [0u8; 4];
    let err = write::<Riscv64>(R_RISCV_32, 0x1_0000_0000, &mut over)
        .expect_err("a value in neither range still does not fit");
    assert!(
        matches!(err, xold::Error::RelocOverflow(t) if t == R_RISCV_32),
        "the refusal must name the relocation, got {err:?}"
    );
}

/// A 16-bit absolute store takes a small negative constant.
#[test]
fn a_16_bit_absolute_store_takes_a_negative_value() {
    let mut slot = [0u8; 2];
    write::<X86_64>(R_X86_64_16, u64::MAX, &mut slot)
        .expect("minus one is a legitimate 16-bit store, spelled 0xffff");
    assert_eq!(slot, [0xff, 0xff]);

    let mut low = [0u8; 2];
    write::<X86_64>(R_X86_64_16, (-32768i64).cast_unsigned(), &mut low)
        .expect("the most negative 16-bit value fits");
    assert_eq!(low, [0x00, 0x80]);
}

/// And the unsigned range it always took is still taken, while a value in
/// neither range is still refused.
#[test]
fn the_16_bit_store_still_bounds_what_it_takes() {
    let mut slot = [0u8; 2];
    write::<X86_64>(R_X86_64_16, 0xffff, &mut slot)
        .expect("the largest unsigned value fits");
    assert_eq!(slot, [0xff, 0xff]);

    let mut over = [0u8; 2];
    let err = write::<X86_64>(R_X86_64_16, 0x1_0000, &mut over)
        .expect_err("a value in neither range does not fit the field");
    assert!(
        matches!(err, xold::Error::RelocOverflow(t) if t == R_X86_64_16),
        "the refusal must name the relocation, got {err:?}"
    );
}
