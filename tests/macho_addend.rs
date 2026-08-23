//! Two Mach-O addend defects, both off by a fixed amount.
//!
//! `X86_64_RELOC_SIGNED_1/2/4` name how many bytes of instruction follow the
//! fixup field. Darwin assemblers bake that count into the stored bytes, and a
//! linker that accounts for it has to add it to the addend and add it again to
//! the program counter it subtracts, so the two cancel: lld's net value is
//! `S + raw - (1 << length) - P` (`Arch/X86_64.cpp:101,109`). Subtracting it
//! once, as the PC-relative correction already did, left the value short by
//! that count -- every `movl $1, _g(%rip)` wrote to `_g - 4`.
//!
//! `ARM64_RELOC_ADDEND` carries an explicit addend in the 24-bit
//! `r_symbolnum` field, and the field is signed. Reading it unsigned turned
//! every negative addend into a value just under 16 MB. lld sign-extends it.
//!
//! Darwin images cannot run on this host, so these check the arithmetic
//! against the formulas rather than the result of executing it.

use xold::{
    macho::{
        MachReloc,
        reloc::{MachoTarget, addend_for, arm64_pending_addend},
    },
    reloc::macho_x86_64::{
        X86_64_RELOC_SIGNED, X86_64_RELOC_SIGNED_1, X86_64_RELOC_SIGNED_2,
        X86_64_RELOC_SIGNED_4,
    },
};

/// The `SIGNED_N` types get the same addend as plain `SIGNED`.
#[test]
fn the_signed_correction_is_not_applied_twice() {
    let slot = 0i32.to_le_bytes();
    let plain =
        addend_for(MachoTarget::X86_64, &pcrel(X86_64_RELOC_SIGNED), &slot, 0);
    assert_eq!(
        plain, -4,
        "a 4-byte PC-relative field addresses its own end, so the correction \
         is one field width"
    );
    for r_type in [
        X86_64_RELOC_SIGNED_1,
        X86_64_RELOC_SIGNED_2,
        X86_64_RELOC_SIGNED_4,
    ] {
        let got = addend_for(MachoTarget::X86_64, &pcrel(r_type), &slot, 0);
        assert_eq!(
            got, plain,
            "type {r_type} must net the same value as SIGNED: the trailing \
             byte count cancels between the addend and the program counter"
        );
    }
}

/// The stored bytes still reach the value, so the correction is the only
/// thing that changed.
#[test]
fn the_stored_bytes_are_still_the_addend() {
    let slot = 0x20i32.to_le_bytes();
    let got = addend_for(
        MachoTarget::X86_64,
        &pcrel(X86_64_RELOC_SIGNED_4),
        &slot,
        0,
    );
    assert_eq!(got, 0x20 - 4, "raw bytes minus one field width");
}

/// A non-PC-relative entry takes the bytes unchanged.
#[test]
fn an_absolute_entry_takes_the_bytes_as_they_are() {
    let slot = 0x11u64.to_le_bytes();
    let mut reloc = pcrel(0);
    reloc.r_pcrel = false;
    reloc.r_length = 3;
    let got = addend_for(MachoTarget::X86_64, &reloc, &slot, 0);
    assert_eq!(got, 0x11, "nothing to correct without a program counter");
}

/// An arm64 `UNSIGNED` fixup carries its addend in the stored bytes, on top
/// of any explicit hint.
#[test]
fn an_arm64_unsigned_keeps_the_stored_addend() {
    let mut reloc = pcrel(0);
    reloc.r_pcrel = false;
    reloc.r_length = 3;
    let slot = 123u64.to_le_bytes();
    let got = addend_for(MachoTarget::Arm64, &reloc, &slot, 0);
    assert_eq!(
        got, 123,
        "UNSIGNED overwrites whole bytes, so the field itself is the addend"
    );
    let with_hint = addend_for(MachoTarget::Arm64, &reloc, &slot, 7);
    assert_eq!(
        with_hint, 130,
        "the hint and the stored bytes add, matching lld's totalAddend"
    );
}

/// The scattered arm64 types have no embedded addend: only the hint counts.
#[test]
fn an_arm64_scattered_type_has_no_embedded_addend() {
    let reloc = pcrel(1);
    let slot = 0xffff_ffffu32.to_le_bytes();
    let got = addend_for(MachoTarget::Arm64, &reloc, &slot, 5);
    assert_eq!(
        got, 5,
        "instruction-bit fixups discard the stored bits, hint only"
    );
}

/// A negative explicit addend stays negative.
#[test]
fn a_negative_arm64_addend_is_sign_extended() {
    // -16 in 24 bits.
    let mut reloc = pcrel(0);
    reloc.r_symbolnum = 0x00ff_fff0;
    assert_eq!(
        arm64_pending_addend(&reloc),
        -16,
        "the field is 24 bits and signed; reading it unsigned makes this \
         16777200, which is 16 MB away from the referent"
    );
}

/// And a positive one is unchanged.
#[test]
fn a_positive_arm64_addend_is_unchanged() {
    let mut reloc = pcrel(0);
    reloc.r_symbolnum = 0x0000_0030;
    assert_eq!(arm64_pending_addend(&reloc), 0x30);
    reloc.r_symbolnum = 0x007f_ffff;
    assert_eq!(
        arm64_pending_addend(&reloc),
        0x007f_ffff,
        "the largest positive value the field holds"
    );
}

// --- fixtures --------------------------------------------------------------

/// A 4-byte PC-relative relocation of type `r_type`, which is what every
/// `SIGNED*` entry is.
fn pcrel(r_type: u32) -> MachReloc {
    MachReloc {
        r_address: 0,
        r_symbolnum: 0,
        r_pcrel: true,
        r_length: 2,
        r_extern: true,
        r_type: u8::try_from(r_type).unwrap_or(0),
        r_scattered: false,
    }
}
