//! The `AArch64` relocation table: classification, bitfield driver
//! correctness for contiguous and split immediates, and `scan_target` GOT
//! allocation.
//!
//! Each bitfield test starts from a real instruction opcode (BL, ADD, ADRP,
//! B.cond) and checks the exact mutated bytes, with the expected encoding
//! computed by hand from the `AArch64` instruction layout.

use xold::{
    reloc::{
        Arch, Check, Field, Needs, RelExpr, Resolver, Spec, Target, Write,
        WriteKind,
        aarch64::{
            AArch64, R_AARCH64_ABS16, R_AARCH64_ABS32, R_AARCH64_ABS64,
            R_AARCH64_ADD_ABS_LO12_NC, R_AARCH64_ADR_GOT_PAGE,
            R_AARCH64_ADR_PREL_LO21, R_AARCH64_ADR_PREL_PG_HI21,
            R_AARCH64_ADR_PREL_PG_HI21_NC, R_AARCH64_CALL26,
            R_AARCH64_CONDBR19, R_AARCH64_JUMP26, R_AARCH64_LD64_GOT_LO12_NC,
            R_AARCH64_LDST8_ABS_LO12_NC, R_AARCH64_LDST16_ABS_LO12_NC,
            R_AARCH64_LDST32_ABS_LO12_NC, R_AARCH64_LDST64_ABS_LO12_NC,
            R_AARCH64_LDST128_ABS_LO12_NC, R_AARCH64_MOVW_UABS_G1,
            R_AARCH64_PREL16, R_AARCH64_PREL32, R_AARCH64_TLSDESC,
            R_AARCH64_TLSLE_ADD_TPREL_HI12, R_AARCH64_TLSLE_ADD_TPREL_LO12_NC,
            R_AARCH64_TLSLE_MOVW_TPREL_G0, R_AARCH64_TLSLE_MOVW_TPREL_G0_NC,
            R_AARCH64_TLSLE_MOVW_TPREL_G1, R_AARCH64_TLSLE_MOVW_TPREL_G1_NC,
            R_AARCH64_TLSLE_MOVW_TPREL_G2,
        },
        apply, scan, scan_target,
    },
    symbol::SymbolId,
};

/// A resolver returning fixed addresses, mirroring `tests/reloc.rs`.
struct Fixed {
    symbol: u64,
    got: u64,
    plt: u64,
    got_base: u64,
}

impl Resolver for Fixed {
    fn symbol_addr(&self, _: SymbolId) -> u64 {
        self.symbol
    }
    fn got_addr(&self, _: SymbolId) -> u64 {
        self.got
    }
    fn got_base(&self) -> u64 {
        self.got_base
    }
    fn plt_addr(&self, _: SymbolId) -> u64 {
        self.plt
    }
}

/// `BL` (branch with link), opcode `0x94000000`, imm26 at bits [0..26].
const BL: [u8; 4] = [0x00, 0x00, 0x00, 0x94];
/// `ADD x0, x0, #0`, opcode `0x91000000`, imm12 at bits [10..21].
const ADD: [u8; 4] = [0x00, 0x00, 0x00, 0x91];
/// `ADRP x0`, opcode `0x90000000`, split imm21 (immlo [29..30], immhi [5..23]).
const ADRP: [u8; 4] = [0x00, 0x00, 0x00, 0x90];
/// `ADR x0`, opcode `0x10000000`, same split imm21 as `ADRP`.
const ADR: [u8; 4] = [0x00, 0x00, 0x00, 0x10];
/// `B.EQ`, opcode `0x54000000`, imm19 at bits [5..23].
const B_EQ: [u8; 4] = [0x00, 0x00, 0x00, 0x54];
/// `LDR W0, [X0, #0]`, opcode `0xB9400000`, imm12 at bits [10..21].
const LDR_W: [u8; 4] = [0x00, 0x00, 0x40, 0xB9];
/// `LDR X0, [X0, #0]`, opcode `0xF9400000`, imm12 at bits [10..21].
const LDR_X: [u8; 4] = [0x00, 0x00, 0x40, 0xF9];
/// `MOVZ X0, #0`, opcode `0xD2800000`, imm16 at bits [5..21].
const MOVZ: [u8; 4] = [0x00, 0x00, 0x80, 0xD2];
/// `MOVK X0, #0, lsl #16`, opcode `0xF2A00000`. Bits 30 and 29 are both set,
/// which is what marks a `movk`: it merges its slice into a register the
/// earlier instructions built, so it carries no sign of its own and the
/// relocation leaves its opcode alone.
const MOVK_16: [u8; 4] = [0x00, 0x00, 0xA0, 0xF2];
/// `MOVK X0, #0`, opcode `0xF2800000` -- the low slice of the same sequence.
const MOVK_0: [u8; 4] = [0x00, 0x00, 0x80, 0xF2];
/// `MOVN X0, #0`, opcode `0x92800000`. Bits 30 and 29 are both clear; it loads
/// the complement of its immediate, which is how a negative slice is encoded.
const MOVN: [u8; 4] = [0x00, 0x00, 0x80, 0x92];

#[test]
fn table_classifies_whole_byte_relocations() {
    assert_eq!(
        AArch64::spec(R_AARCH64_ABS64).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Bytes(WriteKind::W64),
        }
    );
    // The narrow ABS/PREL widths take either signedness, matching lld's
    // `checkIntUInt`: a 32-bit slot holds both `-1` and `0xffffffff`.
    assert_eq!(
        AArch64::spec(R_AARCH64_PREL32).unwrap(),
        Spec {
            expr: RelExpr::Pc,
            write: Write::Bytes(WriteKind::W32SU),
        }
    );
    assert_eq!(
        AArch64::spec(R_AARCH64_ABS32).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Bytes(WriteKind::W32SU),
        }
    );
    assert_eq!(
        AArch64::spec(R_AARCH64_ABS16).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Bytes(WriteKind::W16SU),
        }
    );
    assert_eq!(
        AArch64::spec(R_AARCH64_PREL16).unwrap(),
        Spec {
            expr: RelExpr::Pc,
            write: Write::Bytes(WriteKind::W16SU),
        }
    );
}

#[test]
fn table_classifies_contiguous_bitfield_relocations() {
    // ADD 12-bit displacement at instruction bits [10..21]. The ADD lo12 has
    // scale 0, so the Field's shift-then-mask coincides with the ABI's
    // mask-then-shift and a plain Field is correct.
    assert_eq!(
        AArch64::spec(R_AARCH64_ADD_ABS_LO12_NC).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Field(Field::new(4, 0, 10, 12, Check::None)),
        }
    );
    // 26-bit branch target at bits [0..26], scaled by the instruction size.
    assert_eq!(
        AArch64::spec(R_AARCH64_CALL26).unwrap(),
        Spec {
            expr: RelExpr::PltPc,
            write: Write::Field(Field::scaled(4, 2, 0, 26, Check::Signed)),
        }
    );
    // 19-bit B.cond target at bits [5..23].
    assert_eq!(
        AArch64::spec(R_AARCH64_CONDBR19).unwrap(),
        Spec {
            expr: RelExpr::PltPc,
            write: Write::Field(Field::scaled(4, 2, 5, 19, Check::Signed)),
        }
    );
    // 16-bit MOVZ slice for bits [16..32] of the value: right_shift 16.
    assert_eq!(
        AArch64::spec(R_AARCH64_MOVW_UABS_G1).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Field(Field::new(4, 16, 5, 16, Check::Unsigned)),
        }
    );
}

#[test]
fn table_routes_ldst_lo12_relocations_to_escape() {
    // The LDST low-12 relocations mask the address to the low 12 bits FIRST
    // and only then scale by the access size. The plain Field model shifts the
    // full value before masking, which is wrong for these, so they route to
    // escape (which writes bits [10..21] itself). The 4-byte placeholder only
    // sizes the instruction slot.
    let escape = Spec {
        expr: RelExpr::Escape,
        write: Write::Field(Field::new(4, 0, 0, 0, Check::None)),
    };
    assert_eq!(AArch64::spec(R_AARCH64_LDST8_ABS_LO12_NC).unwrap(), escape);
    assert_eq!(AArch64::spec(R_AARCH64_LDST16_ABS_LO12_NC).unwrap(), escape);
    assert_eq!(AArch64::spec(R_AARCH64_LDST32_ABS_LO12_NC).unwrap(), escape);
    assert_eq!(AArch64::spec(R_AARCH64_LDST64_ABS_LO12_NC).unwrap(), escape);
    assert_eq!(
        AArch64::spec(R_AARCH64_LDST128_ABS_LO12_NC).unwrap(),
        escape
    );
    // The GOT lo12 variant shares the mask-first writer; the escape handler
    // selects the GOT expression for it.
    assert_eq!(AArch64::spec(R_AARCH64_LD64_GOT_LO12_NC).unwrap(), escape);
}

#[test]
fn table_routes_split_and_tls_relocations_to_escape() {
    let escape = Spec {
        expr: RelExpr::Escape,
        write: Write::Field(Field::new(4, 0, 0, 0, Check::None)),
    };
    assert_eq!(AArch64::spec(R_AARCH64_ADR_PREL_PG_HI21).unwrap(), escape);
    assert_eq!(AArch64::spec(R_AARCH64_ADR_GOT_PAGE).unwrap(), escape);
    assert_eq!(AArch64::spec(R_AARCH64_TLSDESC).unwrap(), escape);
}

#[test]
fn unknown_type_is_unsupported() {
    assert!(matches!(
        scan::<AArch64>(999).unwrap_err(),
        xold::Error::UnsupportedReloc(999)
    ));
}

#[test]
fn call26_writes_branch_offset_into_imm26() {
    // PltPc collapses to S + A - P in a static link.
    let fixed = Fixed {
        symbol: 0x1000,
        got: 0,
        plt: 0x1000,
        got_base: 0,
    };
    // value = 0x1000 - 0x800 = 0x800; stored as (0x800 >> 2) = 0x200 at imm26.
    // BL | 0x200 = 0x94000200 -> [0x00, 0x02, 0x00, 0x94].
    let mut slot = BL;
    apply::<AArch64, _>(
        R_AARCH64_CALL26,
        Some(SymbolId(1)),
        0,
        0x800,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x00, 0x02, 0x00, 0x94]);

    // JUMP26 shares the imm26 layout.
    let mut slot = BL;
    apply::<AArch64, _>(
        R_AARCH64_JUMP26,
        Some(SymbolId(1)),
        0,
        0x800,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x00, 0x02, 0x00, 0x94]);
}

#[test]
fn call26_encodes_backward_branch_as_twos_complement() {
    let fixed = Fixed {
        symbol: 0x800,
        got: 0,
        plt: 0x800,
        got_base: 0,
    };
    // value = 0x800 - 0x1000 = -0x800 (mod 2^64); arithmetic >> 2 keeps the
    // sign, so imm26 = 0x3FFFE00 (low 26 bits of -2).
    let mut slot = BL;
    apply::<AArch64, _>(
        R_AARCH64_CALL26,
        Some(SymbolId(1)),
        0,
        0x1000,
        &fixed,
        &mut slot,
    )
    .unwrap();
    // BL | 0x3FFFE00 = 0x97FFFE00 -> [0x00, 0xfe, 0xff, 0x97].
    assert_eq!(slot, [0x00, 0xfe, 0xff, 0x97]);
}

#[test]
fn add_abs_lo12_writes_low_12_bits_into_imm12() {
    let fixed = Fixed {
        symbol: 0x1234,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    // value = 0x1234 + 0x10 = 0x1244; low 12 bits = 0x244; << 10 = 0x91000.
    // ADD | 0x91000 = 0x91091000 -> [0x00, 0x10, 0x09, 0x91].
    let mut slot = ADD;
    apply::<AArch64, _>(
        R_AARCH64_ADD_ABS_LO12_NC,
        Some(SymbolId(1)),
        0x10,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x00, 0x10, 0x09, 0x91]);
}

#[test]
fn ldst32_lo12_masks_to_page_offset_before_scaling() {
    let fixed = Fixed {
        symbol: 0x401_004,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    // value = 0x401004. The ABI masks to the low 12 bits first, then scales by
    // the 4-byte access: (0x004 >> 2) = 1, so imm12 = 1 -> field 1 << 10.
    // LDR_W | 0x400 = 0xB9400400 -> [0x00, 0x04, 0x40, 0xB9]. Shifting the
    // full address first (the old bug) gives (0x401004 >> 2) & 0xfff = 0x401,
    // a 4 KiB-too-high offset.
    let mut slot = LDR_W;
    apply::<AArch64, _>(
        R_AARCH64_LDST32_ABS_LO12_NC,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x00, 0x04, 0x40, 0xB9]);
}

#[test]
fn ldst_lo12_scales_each_access_size_from_the_page_offset() {
    // For a page offset of 0x80 the encoded imm12 is 0x80 >> scale. The
    // relocation patches only bits [10..21], so a single base instruction
    // exercises every access width.
    let fixed = Fixed {
        symbol: 0x400_080,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    for (r_type, imm12) in [
        (R_AARCH64_LDST8_ABS_LO12_NC, 0x80u64),
        (R_AARCH64_LDST16_ABS_LO12_NC, 0x40),
        (R_AARCH64_LDST32_ABS_LO12_NC, 0x20),
        (R_AARCH64_LDST64_ABS_LO12_NC, 0x10),
        (R_AARCH64_LDST128_ABS_LO12_NC, 0x08),
    ] {
        let mut slot = LDR_W;
        apply::<AArch64, _>(r_type, Some(SymbolId(1)), 0, 0, &fixed, &mut slot)
            .unwrap();
        assert_eq!(slot, ldr_w_with_imm12(imm12));
    }
}

#[test]
fn ld64_got_lo12_masks_the_got_entry_address() {
    let fixed = Fixed {
        symbol: 0,
        got: 0x401_008,
        plt: 0,
        got_base: 0,
    };
    // value = GOT[sym] + 0 = 0x401008; mask-first: (0x008 >> 3) = 1 -> imm12 1.
    // LDR_X | 0x400 = 0xF9400400 -> [0x00, 0x04, 0x40, 0xF9].
    let mut slot = LDR_X;
    apply::<AArch64, _>(
        R_AARCH64_LD64_GOT_LO12_NC,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x00, 0x04, 0x40, 0xF9]);
}

/// Builds the expected `LDR W0, [X0, #imm]` bytes for a given 12-bit immediate.
const fn ldr_w_with_imm12(imm12: u64) -> [u8; 4] {
    #[allow(clippy::cast_possible_truncation)]
    (0xB940_0000u32 | ((imm12 << 10) as u32)).to_le_bytes()
}

#[test]
fn condbr19_writes_target_into_imm19() {
    let fixed = Fixed {
        symbol: 0x1000,
        got: 0,
        plt: 0x1000,
        got_base: 0,
    };
    // value = 0x800; (0x800 >> 2) = 0x200; << 5 = 0x4000.
    // B.EQ | 0x4000 = 0x54004000 -> [0x00, 0x40, 0x00, 0x54].
    let mut slot = B_EQ;
    apply::<AArch64, _>(
        R_AARCH64_CONDBR19,
        Some(SymbolId(1)),
        0,
        0x800,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x00, 0x40, 0x00, 0x54]);
}

/// A branch counts 4-byte instructions, so `store` drops the low two bits of
/// the byte offset. A target that is not 4-byte aligned would therefore be
/// encoded as a different, nearby address; it has to be diagnosed instead.
/// lld does the same, through `checkAlignment` beside its range check.
#[test]
fn misaligned_branch_target_is_diagnosed() {
    let fixed = Fixed {
        symbol: 0x1002,
        got: 0,
        plt: 0x1002,
        got_base: 0,
    };
    // value = 0x1002 - 0x800 = 0x802, which is not a multiple of 4. Truncating
    // it would branch to 0x1000 without a word of complaint.
    let mut slot = BL;
    let err = apply::<AArch64, _>(
        R_AARCH64_CALL26,
        Some(SymbolId(1)),
        0,
        0x800,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(
        matches!(err, xold::Error::RelocMisaligned(R_AARCH64_CALL26)),
        "expected a misalignment diagnosis, got {err:?}"
    );
    assert_eq!(slot, BL, "a rejected relocation must not patch the cell");

    // The same holds for the 19-bit conditional form.
    let mut slot = B_EQ;
    let err = apply::<AArch64, _>(
        R_AARCH64_CONDBR19,
        Some(SymbolId(1)),
        0,
        0x800,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(
        matches!(err, xold::Error::RelocMisaligned(R_AARCH64_CONDBR19)),
        "expected a misalignment diagnosis, got {err:?}"
    );
    assert_eq!(slot, B_EQ, "a rejected relocation must not patch the cell");
}

/// The alignment check must not reach a field whose `right_shift` selects a
/// slice of a wider value rather than a scale. A `MOVW` slice takes bits
/// [16..32] of an arbitrary address, and the `ADD_TPREL_HI12` field takes bits
/// [12..24] of an arbitrary thread-pointer offset; neither says anything about
/// the bits below, and lld range-checks both without an alignment check. An
/// unscaled field (`right_shift == 0`) is likewise unconstrained.
#[test]
fn slice_fields_accept_a_value_with_low_bits_set() {
    let fixed = Fixed {
        symbol: 0x0001_2345,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    // MOVW G1 takes bits [16..32] = 0x0001, which sits at cell bits [5..21].
    // MOVZ | (1 << 5) = 0xD2800020.
    let mut slot = MOVZ;
    apply::<AArch64, _>(
        R_AARCH64_MOVW_UABS_G1,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .expect("a MOVW slice imposes no alignment on the value");
    assert_eq!(slot, [0x20, 0x00, 0x80, 0xD2]);

    // TPREL HI12 takes bits [12..24] = 0x012, at cell bits [10..21].
    // ADD | (0x12 << 10) = 0x91004800.
    let mut slot = ADD;
    apply::<AArch64, _>(
        R_AARCH64_TLSLE_ADD_TPREL_HI12,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .expect("a HI12 slice imposes no alignment on the value");
    assert_eq!(slot, [0x00, 0x48, 0x00, 0x91]);

    // An unscaled 12-bit ADD displacement takes the value as it stands.
    let mut slot = ADD;
    apply::<AArch64, _>(
        R_AARCH64_ADD_ABS_LO12_NC,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .expect("an unscaled field imposes no alignment on the value");
    // 0x12345 & 0xfff = 0x345, at cell bits [10..21]: ADD | 0xD1400.
    assert_eq!(slot, [0x00, 0x14, 0x0D, 0x91]);
}

#[test]
fn adr_prel_pg_hi21_writes_split_21_bit_immediate() {
    let fixed = Fixed {
        symbol: 0x9000,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    // page(0x9000) - page(0x4000) = 0x5000; >> 12 = 0x5 (page count).
    // imm 0x5: immlo = 0x5 & 0x3 = 1 -> bits [29..30]; immhi = 0x5 >> 2 = 1
    // -> bits [5..23]. ADRP | 0x20000000 | 0x20 = 0xB0000020.
    let mut slot = ADRP;
    apply::<AArch64, _>(
        R_AARCH64_ADR_PREL_PG_HI21,
        Some(SymbolId(1)),
        0,
        0x4000,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x20, 0x00, 0x00, 0xB0]);
}

#[test]
fn adr_got_page_writes_page_offset_of_got_entry() {
    let fixed = Fixed {
        symbol: 0,
        got: 0x9000,
        plt: 0,
        got_base: 0x9000,
    };
    // page(GOT[sym] + 0) - page(P) = page(0x9000) - page(0x4000) = 0x5000;
    // >> 12 = 0x5, same split encoding as the PC-relative ADRP case.
    let mut slot = ADRP;
    apply::<AArch64, _>(
        R_AARCH64_ADR_GOT_PAGE,
        Some(SymbolId(1)),
        0,
        0x4000,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x20, 0x00, 0x00, 0xB0]);
}

#[test]
fn abs64_writes_full_width_absolute() {
    let fixed = Fixed {
        symbol: 0x0123_4567_89ab,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 8];
    apply::<AArch64, _>(
        R_AARCH64_ABS64,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0xab, 0x89, 0x67, 0x45, 0x23, 0x01, 0x00, 0x00]);
}

#[test]
fn branch_overflow_is_detected() {
    // A 1 GiB branch is well past the +/- 128 MiB CALL26 range.
    let fixed = Fixed {
        symbol: 0x4000_0000,
        got: 0,
        plt: 0x4000_0000,
        got_base: 0,
    };
    let mut slot = BL;
    let err = apply::<AArch64, _>(
        R_AARCH64_CALL26,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(matches!(err, xold::Error::RelocOverflow(R_AARCH64_CALL26)));
}

#[test]
fn scan_target_allocates_got_for_got_relocations() {
    // LD64_GOT_LO12_NC references a GOT entry directly.
    assert_eq!(
        scan_target(Target::AArch64, R_AARCH64_LD64_GOT_LO12_NC).unwrap(),
        Needs {
            got: true,
            ..Needs::NONE
        }
    );
    // ADR_GOT_PAGE is Escape for writing but still needs a GOT entry, so the
    // scan override must allocate one.
    assert_eq!(
        scan_target(Target::AArch64, R_AARCH64_ADR_GOT_PAGE).unwrap(),
        Needs {
            got: true,
            ..Needs::NONE
        }
    );
}

#[test]
fn scan_target_allocates_no_got_for_call26() {
    // CALL26 is a direct branch: no GOT entry. (PLT collapses to the symbol in
    // a static link, and this linker never allocates a PLT, so a `plt: true`
    // need has no effect.)
    let needs = scan_target(Target::AArch64, R_AARCH64_CALL26).unwrap();
    assert!(!needs.got);
}

#[test]
fn target_round_trips_e_machine() {
    use xold::elf::constants::{EM_AARCH64, EM_RISCV, EM_X86_64};
    assert_eq!(Target::from_machine(EM_X86_64).unwrap(), Target::X86_64);
    assert_eq!(Target::from_machine(EM_AARCH64).unwrap(), Target::AArch64);
    assert_eq!(Target::from_machine(EM_RISCV).unwrap(), Target::Riscv64);
    assert!(Target::from_machine(0x9999).is_err());
    assert_eq!(Target::AArch64.machine(), EM_AARCH64);
}

#[test]
fn tls_relocation_is_recognised_but_not_reduced() {
    let fixed = Fixed {
        symbol: 0,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 4];
    let err = apply::<AArch64, _>(
        R_AARCH64_TLSDESC,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        xold::Error::UnsupportedReloc(R_AARCH64_TLSDESC)
    ));
}

#[test]
fn table_classifies_local_exec_tprel_relocations() {
    // The ADD_TPREL pair stores the thread-pointer-relative offset into the
    // shared 12-bit ADD immediate. HI12 selects val >> 12 with a 24-bit
    // unsigned check; LO12_NC takes the low 12 with no check.
    assert_eq!(
        AArch64::spec(R_AARCH64_TLSLE_ADD_TPREL_HI12).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Field(Field::new(4, 12, 10, 12, Check::Unsigned)),
        }
    );
    assert_eq!(
        AArch64::spec(R_AARCH64_TLSLE_ADD_TPREL_LO12_NC).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Field(Field::new(4, 0, 10, 12, Check::None)),
        }
    );
    // The MOVW_TPREL slices write the same 16 bits as `MOVW_UABS_*` but are
    // not the same relocation: the slice is signed and its sign selects the
    // instruction, which one mask cannot describe, so they route through
    // escape. The escape path carries their range checks.
    for r_type in [
        R_AARCH64_TLSLE_MOVW_TPREL_G0,
        R_AARCH64_TLSLE_MOVW_TPREL_G0_NC,
        R_AARCH64_TLSLE_MOVW_TPREL_G1,
        R_AARCH64_TLSLE_MOVW_TPREL_G1_NC,
        R_AARCH64_TLSLE_MOVW_TPREL_G2,
    ] {
        let spec = AArch64::spec(r_type).unwrap();
        assert_eq!(spec.expr, RelExpr::Escape, "type {r_type:#x}");
        assert_eq!(spec.write.width(), 4, "type {r_type:#x}");
    }
}

#[test]
fn add_tprel_pair_writes_hi12_and_lo12_of_offset() {
    // A single 4-byte TLS variable: the TPOFF is the 16-byte Variant 1 gap.
    // HI12 writes val >> 12 (= 0); LO12_NC writes val & 0xfff (= 16). The ADD
    // shift bit (sf/opc) sits outside the imm12 field and is preserved, so HI12
    // leaves the `lsl #12` set by the assembler untouched.
    let fixed = Fixed {
        symbol: 16,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    // HI12: ADD x0, x0, #0, lsl #12 -- imm12 = 0, opcode unchanged.
    let mut hi = ADD;
    apply::<AArch64, _>(
        R_AARCH64_TLSLE_ADD_TPREL_HI12,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut hi,
    )
    .unwrap();
    assert_eq!(hi, [0x00, 0x00, 0x00, 0x91]);
    // LO12_NC: ADD x0, x0, #16 -- imm12 = 16, << 10 = 0x4000.
    let mut lo = ADD;
    apply::<AArch64, _>(
        R_AARCH64_TLSLE_ADD_TPREL_LO12_NC,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut lo,
    )
    .unwrap();
    assert_eq!(lo, [0x00, 0x40, 0x00, 0x91]);
}

/// Applies `r_type` at `value` over `insn` and returns the patched bytes.
fn tprel_movw(r_type: u32, value: u64, insn: [u8; 4]) -> [u8; 4] {
    let fixed = Fixed {
        symbol: value,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut out = insn;
    apply::<AArch64, _>(r_type, Some(SymbolId(1)), 0, 0, &fixed, &mut out)
        .unwrap();
    out
}

#[test]
fn movw_tprel_slices_each_take_their_16_bit_value_window() {
    // A TPOFF that needs all three slices: 0x0003_0002_0010 splits as
    // G0 = 0x0010, G1 = 0x0002, G2 = 0x0003. A compiler emits the top slice as
    // a MOVZ and the rest as MOVKs, which is the sequence used here: a MOVK
    // merges into a register the earlier instructions built, so it carries no
    // sign and its opcode survives untouched.
    //
    // G2 (checked, MOVZ): 0x0003 << 5 = 0x60, and the slice is non-negative so
    // the MOVZ opcode stands.
    assert_eq!(
        tprel_movw(R_AARCH64_TLSLE_MOVW_TPREL_G2, 0x0003_0002_0010, MOVZ),
        [0x60, 0x00, 0x80, 0xD2]
    );
    // G1_NC (MOVK, lsl #16): 0x0002 << 5 = 0x40, opcode unchanged.
    assert_eq!(
        tprel_movw(R_AARCH64_TLSLE_MOVW_TPREL_G1_NC, 0x0003_0002_0010, MOVK_16),
        [0x40, 0x00, 0xA0, 0xF2]
    );
    // G0_NC (MOVK): 0x0010 << 5 = 0x200, opcode unchanged.
    assert_eq!(
        tprel_movw(R_AARCH64_TLSLE_MOVW_TPREL_G0_NC, 0x0003_0002_0010, MOVK_0),
        [0x00, 0x02, 0x80, 0xF2]
    );
}

/// A slice is a *signed* 17-bit quantity whose sign picks the instruction: the
/// immediate field has no sign bit, so a negative slice becomes a `MOVN`
/// carrying the bitwise complement. This is lld's `writeSMovWImm`, and it is
/// why the set cannot be a plain masked bitfield.
#[test]
fn a_negative_movw_tprel_slice_becomes_a_movn() {
    // -16 as a G0 slice: bit 16 of the slice is set, so the instruction becomes
    // MOVN (bit 30 cleared: 0xD2800000 -> 0x92800000) and it carries the
    // complement, 0xf, because MOVN loads `!imm`: 0xf << 5 = 0x1e0.
    assert_eq!(
        tprel_movw(
            R_AARCH64_TLSLE_MOVW_TPREL_G0,
            (-16i64).cast_unsigned(),
            MOVZ
        ),
        [0xE0, 0x01, 0x80, 0x92]
    );
    // The same slice reached through a MOVK is left as a MOVK: bits 30 and 29
    // are both set, so the sign fixup does not apply and the raw slice is
    // stored -- 0xfff0 << 5 = 0x1ffe00.
    assert_eq!(
        tprel_movw(
            R_AARCH64_TLSLE_MOVW_TPREL_G0_NC,
            (-16i64).cast_unsigned(),
            MOVK_0
        ),
        [0x00, 0xFE, 0x9F, 0xF2]
    );
    // A non-negative slice reaching a MOVN turns it back into a MOVZ, so the
    // instruction always matches the sign of the value the link resolved.
    assert_eq!(
        tprel_movw(R_AARCH64_TLSLE_MOVW_TPREL_G0, 0x10, MOVN),
        [0x00, 0x02, 0x80, 0xD2]
    );
}

/// The checked slices accept a signed quantity one bit wider than the field
/// they write, because the sign lives in the opcode. lld checks 17, 33 and 49
/// bits for G0, G1 and G2.
#[test]
fn checked_movw_tprel_slices_range_check_one_bit_wider_than_the_field() {
    // G0 takes the signed 17-bit range: 0xffff fits, 0x10000 does not.
    assert_eq!(
        tprel_movw(R_AARCH64_TLSLE_MOVW_TPREL_G0, 0xFFFF, MOVZ),
        [0xE0, 0xFF, 0x9F, 0xD2]
    );
    let fixed = Fixed {
        symbol: 0x1_0000,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut out = MOVZ;
    assert!(
        apply::<AArch64, _>(
            R_AARCH64_TLSLE_MOVW_TPREL_G0,
            Some(SymbolId(1)),
            0,
            0,
            &fixed,
            &mut out,
        )
        .is_err(),
        "an offset past the signed 17-bit range overflows G0"
    );
    // The `_NC` form of the same slice takes it, which is what makes it the
    // no-check variant.
    assert_eq!(
        tprel_movw(R_AARCH64_TLSLE_MOVW_TPREL_G0_NC, 0x1_0000, MOVK_0),
        [0x00, 0x00, 0x80, 0xF2]
    );
}

/// lld range-checks an ADRP against 33 signed bits (`checkInt(ctx, loc, val,
/// 33, rel)`) applied to the *unshifted* page delta, then stores `val >> 12`;
/// the `_NC` form is the fall-through case and carries no check. Checking the
/// shifted page count instead would accept a delta beyond ADRP's +/- 4 GiB
/// reach and truncate it into a nearby page.
#[test]
fn adr_prel_pg_hi21_range_checks_the_unshifted_page_delta() {
    // page(S) - page(P) = 0x1_0000_5000, one page past the 33-bit window.
    let fixed = Fixed {
        symbol: 0x1_0000_5000,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = ADRP;
    let err = apply::<AArch64, _>(
        R_AARCH64_ADR_PREL_PG_HI21,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(
        matches!(err, xold::Error::RelocOverflow(R_AARCH64_ADR_PREL_PG_HI21)),
        "expected an overflow diagnosis, got {err:?}"
    );
    assert_eq!(slot, ADRP, "a rejected relocation must not patch the cell");

    // The `_NC` form is defined to truncate the same value: page count
    // 0x10_0005 keeps immlo = 1 at bits [29..30] and immhi = 0x10_0004 at bits
    // [5..23], so ADRP | 0x2000_0000 | 0x0080_0020 = 0xB080_0020.
    let mut slot = ADRP;
    apply::<AArch64, _>(
        R_AARCH64_ADR_PREL_PG_HI21_NC,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .expect("the _NC form carries no range check");
    assert_eq!(slot, [0x20, 0x00, 0x80, 0xB0]);
}

/// The two ends of the 33-bit window are legal and must keep linking: a page
/// delta is a multiple of 4 KiB, so the extremes are `0xffff_f000` forwards and
/// `-0x1_0000_0000` backwards.
#[test]
fn adr_prel_pg_hi21_accepts_the_whole_33_bit_window() {
    let fixed = Fixed {
        symbol: 0xFFFF_F000,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    // Page count 0xf_ffff: immlo = 3 -> 0x6000_0000, immhi = 0xf_fffc << 3 =
    // 0x007f_ffe0. ADRP | both = 0xF07F_FFE0.
    let mut slot = ADRP;
    apply::<AArch64, _>(
        R_AARCH64_ADR_PREL_PG_HI21,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .expect("the largest forward page delta is in range");
    assert_eq!(slot, [0xE0, 0xFF, 0x7F, 0xF0]);

    // Backwards by exactly 4 GiB: page count 0xffff_f000 truncated to the
    // split immediate leaves immlo = 0 and immhi = 0x10_0000 << 3 =
    // 0x0080_0000.
    let fixed = Fixed {
        symbol: 0,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = ADRP;
    apply::<AArch64, _>(
        R_AARCH64_ADR_PREL_PG_HI21,
        Some(SymbolId(1)),
        0,
        0x1_0000_0000,
        &fixed,
        &mut slot,
    )
    .expect("the largest backward page delta is in range");
    assert_eq!(slot, [0x00, 0x00, 0x80, 0x90]);
}

/// `ADR_PREL_LO21` stores its value unshifted, so lld checks it against 21
/// signed bits (`checkInt(ctx, loc, val, 21, rel)`).
#[test]
fn adr_prel_lo21_range_checks_its_21_bit_immediate() {
    // 0x20_0000 is 2 MiB away, one bit past the +/- 1 MiB reach of an ADR.
    let fixed = Fixed {
        symbol: 0x20_0000,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = ADR;
    let err = apply::<AArch64, _>(
        R_AARCH64_ADR_PREL_LO21,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(
        matches!(err, xold::Error::RelocOverflow(R_AARCH64_ADR_PREL_LO21)),
        "expected an overflow diagnosis, got {err:?}"
    );
    assert_eq!(slot, ADR, "a rejected relocation must not patch the cell");

    // 0x1000 is inside the window: immlo = 0, immhi = 0x1000 << 3 = 0x8000.
    let fixed = Fixed {
        symbol: 0x1000,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = ADR;
    apply::<AArch64, _>(
        R_AARCH64_ADR_PREL_LO21,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .expect("a 4 KiB displacement fits the 21-bit window");
    assert_eq!(slot, [0x00, 0x80, 0x00, 0x10]);
}

/// A scaled load/store encodes its displacement in units of the access size,
/// so an address that is not a multiple of that size cannot be represented and
/// would be stored as a nearby one. lld pairs every scaled form with
/// `checkAlignment`; LDST8 counts single bytes and constrains nothing.
#[test]
fn misaligned_ldst_displacement_is_diagnosed() {
    let fixed = Fixed {
        symbol: 0x40_0004,
        got: 0x40_0004,
        plt: 0,
        got_base: 0,
    };
    // A doubleword load needs an 8-byte aligned address; 0x40_0004 is not one.
    let mut slot = LDR_X;
    let err = apply::<AArch64, _>(
        R_AARCH64_LDST64_ABS_LO12_NC,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            xold::Error::RelocMisaligned(R_AARCH64_LDST64_ABS_LO12_NC)
        ),
        "expected a misalignment diagnosis, got {err:?}"
    );
    assert_eq!(slot, LDR_X, "a rejected relocation must not patch the cell");

    // The GOT lo12 form shares the 8-byte rule.
    let mut slot = LDR_X;
    let err = apply::<AArch64, _>(
        R_AARCH64_LD64_GOT_LO12_NC,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            xold::Error::RelocMisaligned(R_AARCH64_LD64_GOT_LO12_NC)
        ),
        "expected a misalignment diagnosis, got {err:?}"
    );

    // A halfword load of an odd address is rejected for the same reason.
    let mut slot = LDR_W;
    let err = apply::<AArch64, _>(
        R_AARCH64_LDST16_ABS_LO12_NC,
        Some(SymbolId(1)),
        1,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            xold::Error::RelocMisaligned(R_AARCH64_LDST16_ABS_LO12_NC)
        ),
        "expected a misalignment diagnosis, got {err:?}"
    );

    // The same address is fine for a word load (4-byte aligned) and for a byte
    // load, which imposes no alignment at all.
    let mut slot = LDR_W;
    apply::<AArch64, _>(
        R_AARCH64_LDST32_ABS_LO12_NC,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .expect("0x40_0004 is 4-byte aligned");
    assert_eq!(slot, ldr_w_with_imm12(1));

    let mut slot = LDR_W;
    apply::<AArch64, _>(
        R_AARCH64_LDST8_ABS_LO12_NC,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .expect("a byte load imposes no alignment");
    assert_eq!(slot, ldr_w_with_imm12(4));
}

/// lld checks `R_AARCH64_ABS32` with `checkIntUInt`, which accepts a value that
/// fits either 32-bit range. A negative absolute value is legal input and must
/// not be reported as an overflow.
#[test]
fn abs32_accepts_either_signedness() {
    let fixed = Fixed {
        symbol: 0,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    // S + A = -8, stored as the low 32 bits of the two's-complement value.
    let mut slot = [0u8; 4];
    apply::<AArch64, _>(
        R_AARCH64_ABS32,
        Some(SymbolId(1)),
        -8,
        0,
        &fixed,
        &mut slot,
    )
    .expect("a negative 32-bit absolute value is in range");
    assert_eq!(slot, [0xF8, 0xFF, 0xFF, 0xFF]);

    // The large unsigned value with the same encoding is equally legal.
    let fixed = Fixed {
        symbol: 0xFFFF_FFF8,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 4];
    apply::<AArch64, _>(
        R_AARCH64_ABS32,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .expect("a large unsigned 32-bit absolute value is in range");
    assert_eq!(slot, [0xF8, 0xFF, 0xFF, 0xFF]);

    // Widening the check is not the same as dropping it: a value outside both
    // ranges still overflows.
    let fixed = Fixed {
        symbol: 0x1_0000_0000,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 4];
    let err = apply::<AArch64, _>(
        R_AARCH64_ABS32,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(
        matches!(err, xold::Error::RelocOverflow(R_AARCH64_ABS32)),
        "expected an overflow diagnosis, got {err:?}"
    );
    assert_eq!(slot, [0u8; 4], "a rejected relocation must not patch bytes");
}

/// `R_AARCH64_PREL32` takes the same `checkIntUInt`, so a large unsigned
/// displacement is legal alongside the negative one a backward reference
/// produces.
#[test]
fn prel32_accepts_either_signedness() {
    let fixed = Fixed {
        symbol: 0xFFFF_F000,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 4];
    apply::<AArch64, _>(
        R_AARCH64_PREL32,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .expect("a large unsigned 32-bit displacement is in range");
    assert_eq!(slot, [0x00, 0xF0, 0xFF, 0xFF]);

    // A backward reference lands on the same word from the other side.
    let fixed = Fixed {
        symbol: 0,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 4];
    apply::<AArch64, _>(
        R_AARCH64_PREL32,
        Some(SymbolId(1)),
        0,
        0x1000,
        &fixed,
        &mut slot,
    )
    .expect("a negative 32-bit displacement is in range");
    assert_eq!(slot, [0x00, 0xF0, 0xFF, 0xFF]);

    // A displacement outside both ranges still overflows.
    let fixed = Fixed {
        symbol: 0x1_0000_0000,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 4];
    let err = apply::<AArch64, _>(
        R_AARCH64_PREL32,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(
        matches!(err, xold::Error::RelocOverflow(R_AARCH64_PREL32)),
        "expected an overflow diagnosis, got {err:?}"
    );
}

/// The 16-bit widths follow the same rule, one width down.
#[test]
fn abs16_and_prel16_accept_either_signedness() {
    let fixed = Fixed {
        symbol: 0,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 2];
    apply::<AArch64, _>(
        R_AARCH64_ABS16,
        Some(SymbolId(1)),
        -8,
        0,
        &fixed,
        &mut slot,
    )
    .expect("a negative 16-bit absolute value is in range");
    assert_eq!(slot, [0xF8, 0xFF]);

    let fixed = Fixed {
        symbol: 0xFFF8,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 2];
    apply::<AArch64, _>(
        R_AARCH64_PREL16,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .expect("a large unsigned 16-bit displacement is in range");
    assert_eq!(slot, [0xF8, 0xFF]);

    let fixed = Fixed {
        symbol: 0x1_0000,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 2];
    let err = apply::<AArch64, _>(
        R_AARCH64_ABS16,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(
        matches!(err, xold::Error::RelocOverflow(R_AARCH64_ABS16)),
        "expected an overflow diagnosis, got {err:?}"
    );
}
