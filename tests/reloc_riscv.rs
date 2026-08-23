//! The RISC-V relocation table: classification, bitfield driver correctness
//! for the contiguous I-type field, and the escape encoders for the U-type
//! hi20, the scattered S/B/J immediates, and the 8-byte `CALL` pair.
//!
//! Each escape test starts from a real instruction opcode (LUI, SW, JAL, BEQ,
//! AUIPC, JALR) and checks the exact mutated bytes, with the expected encoding
//! computed by hand from the RISC-V instruction layout. The hi20 case is chosen
//! so its low 12 bits are >= 0x800, exercising the `+0x800` sign-compensation
//! that a plain bitfield cannot express.

use xold::{
    elf::Rela64,
    endian::{I64, U64},
    reloc::{
        Arch, Check, Field, Needs, RelExpr, Resolver, Spec, Target, Write,
        WriteKind, apply,
        riscv::{
            R_RISCV_32, R_RISCV_32_PCREL, R_RISCV_64, R_RISCV_ADD8,
            R_RISCV_ADD16, R_RISCV_ADD32, R_RISCV_ADD64, R_RISCV_ALIGN,
            R_RISCV_BRANCH, R_RISCV_CALL, R_RISCV_CALL_PLT, R_RISCV_HI20,
            R_RISCV_JAL, R_RISCV_LO12_I, R_RISCV_LO12_S, R_RISCV_PCREL_HI20,
            R_RISCV_PCREL_LO12_I, R_RISCV_PCREL_LO12_S, R_RISCV_RELAX,
            R_RISCV_RVC_BRANCH, R_RISCV_RVC_JUMP, R_RISCV_SET6, R_RISCV_SET8,
            R_RISCV_SET16, R_RISCV_SET32, R_RISCV_SUB6, R_RISCV_SUB8,
            R_RISCV_SUB16, R_RISCV_SUB32, R_RISCV_TLS_GD_HI20, Riscv64,
            rewrite_pcrel_pairs,
        },
        scan, scan_target,
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

/// A resolver with every address at zero, for relocations whose value comes
/// only from the addend or only from the slot's current bytes.
const fn zero() -> Fixed {
    Fixed {
        symbol: 0,
        got: 0,
        plt: 0,
        got_base: 0,
    }
}

/// `addi x0, x0, 0`, opcode `0x13`, I-type imm at bits [31..20].
const ADDI: [u8; 4] = [0x13, 0x00, 0x00, 0x00];
/// `lui x0, 0`, opcode `0x37`, U-type imm at bits [31..12].
const LUI: [u8; 4] = [0x37, 0x00, 0x00, 0x00];
/// `sw x0, 0(x0)`, opcode `0x23`, S-type imm split [31..25]/[11..7].
const SW: [u8; 4] = [0x23, 0x00, 0x00, 0x00];
/// `jal x0, 0`, opcode `0x6f`, J-type imm scattered.
const JAL: [u8; 4] = [0x6f, 0x00, 0x00, 0x00];
/// `beq x0, x0, 0`, opcode `0x63`, B-type imm scattered.
const BEQ: [u8; 4] = [0x63, 0x00, 0x00, 0x00];
/// `c.j x0, 0`, opcode `0xA002`, CJ-type 11-bit imm scattered in 2 bytes.
const C_J: [u8; 2] = [0x02, 0xA0];
/// `c.beqz x0, 0`, opcode `0xC001`, CB-type 9-bit imm scattered in 2 bytes.
const C_BEQZ: [u8; 2] = [0x01, 0xC0];
/// `auipc x0, 0`, opcode `0x17`, U-type imm at bits [31..12].
const AUIPC: [u8; 4] = [0x17, 0x00, 0x00, 0x00];
/// `jalr x0, x0, 0`, opcode `0x67`, I-type imm at bits [31..20].
const JALR: [u8; 4] = [0x67, 0x00, 0x00, 0x00];

#[test]
fn table_classifies_whole_byte_relocations() {
    assert_eq!(
        Riscv64::spec(R_RISCV_64).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Bytes(WriteKind::W64),
        }
    );
    // Signed or unsigned: the psABI gives `R_RISCV_32` no signedness and lld
    // writes it unchecked, so a negative label difference is as legitimate as
    // a large unsigned address.
    assert_eq!(
        Riscv64::spec(R_RISCV_32).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Bytes(WriteKind::W32SU),
        }
    );
}

#[test]
fn table_classifies_lo12_i_as_a_contiguous_field() {
    // I-type lo12: 12-bit contiguous field at instruction bits [31..20].
    assert_eq!(
        Riscv64::spec(R_RISCV_LO12_I).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Field(Field::new(4, 0, 20, 12, Check::None)),
        }
    );
}

#[test]
fn table_routes_hi20_scattered_and_tls_to_escape() {
    // The 4-byte escape placeholder sizes a single instruction slot; the value
    // and encoding are handled by `Arch::escape`.
    let escape4 = Spec {
        expr: RelExpr::Escape,
        write: Write::Field(Field::new(4, 0, 0, 0, Check::None)),
    };
    assert_eq!(Riscv64::spec(R_RISCV_HI20).unwrap(), escape4);
    assert_eq!(Riscv64::spec(R_RISCV_LO12_S).unwrap(), escape4);
    assert_eq!(Riscv64::spec(R_RISCV_PCREL_HI20).unwrap(), escape4);
    assert_eq!(Riscv64::spec(R_RISCV_JAL).unwrap(), escape4);
    assert_eq!(Riscv64::spec(R_RISCV_BRANCH).unwrap(), escape4);
    assert_eq!(Riscv64::spec(R_RISCV_TLS_GD_HI20).unwrap(), escape4);
    // CALL spans two instruction cells, so its slot is 8 bytes.
    assert_eq!(
        Riscv64::spec(R_RISCV_CALL).unwrap(),
        Spec {
            expr: RelExpr::Escape,
            write: Write::Field(Field::new(8, 0, 0, 0, Check::None)),
        }
    );
}

#[test]
fn table_classifies_relax_and_align_as_none() {
    assert_eq!(
        Riscv64::spec(R_RISCV_RELAX).unwrap(),
        Spec {
            expr: RelExpr::None,
            write: Write::Bytes(WriteKind::W64),
        }
    );
    assert_eq!(
        Riscv64::spec(R_RISCV_ALIGN).unwrap(),
        Spec {
            expr: RelExpr::None,
            write: Write::Bytes(WriteKind::W64),
        }
    );
}

#[test]
fn unknown_type_is_unsupported() {
    assert!(matches!(
        scan::<Riscv64>(999).unwrap_err(),
        xold::Error::UnsupportedReloc(999)
    ));
}

#[test]
fn lo12_i_writes_low_12_bits_into_the_imm_field() {
    let fixed = Fixed {
        symbol: 0x1234,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    // value = 0x1234 + 0x10 = 0x1244; low 12 bits = 0x244; << 20 = 0x24400000.
    // ADDI | 0x24400000 = 0x24400013 -> [0x13, 0x00, 0x40, 0x24].
    let mut slot = ADDI;
    apply::<Riscv64, _>(
        R_RISCV_LO12_I,
        Some(SymbolId(1)),
        0x10,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x13, 0x00, 0x40, 0x24]);
}

#[test]
fn hi20_applies_the_sign_compensation_rounding() {
    let fixed = Fixed {
        symbol: 0x1800,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    // value = 0x1800. The low 12 bits (0x800) have bit 11 set, so the hi20 must
    // round up: (0x1800 + 0x800) & 0xFFFFF000 = 0x2000, i.e. imm field 0x2.
    // LUI | 0x2000 = 0x00002037 -> [0x37, 0x20, 0x00, 0x00]. A plain >> 12
    // would instead give 0x1 (0x1037), reconstructing the wrong address.
    let mut slot = LUI;
    apply::<Riscv64, _>(
        R_RISCV_HI20,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x37, 0x20, 0x00, 0x00]);
}

#[test]
fn lo12_s_writes_the_scattered_s_type_immediate() {
    let fixed = Fixed {
        symbol: 0x1234,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    // value = 0x1234; lo12 = 0x234; imm[11:5] = 0x11 at [31..25], imm[4:0] =
    // 0x14 at [11..7]. (0x11 << 25) | (0x14 << 7) = 0x22000A00.
    // SW | 0x22000A00 = 0x22000A23 -> [0x23, 0x0A, 0x00, 0x22].
    let mut slot = SW;
    apply::<Riscv64, _>(
        R_RISCV_LO12_S,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x23, 0x0A, 0x00, 0x22]);
}

#[test]
fn pcrel_hi20_writes_a_pc_relative_hi20() {
    let fixed = Fixed {
        symbol: 0x9000,
        got: 0,
        plt: 0x9000,
        got_base: 0,
    };
    // value = 0x9000 - 0x4000 = 0x5000; (0x5000 + 0x800) & 0xFFFFF000 = 0x5000.
    // AUIPC | 0x5000 = 0x00005017 -> [0x17, 0x50, 0x00, 0x00].
    let mut slot = AUIPC;
    apply::<Riscv64, _>(
        R_RISCV_PCREL_HI20,
        Some(SymbolId(1)),
        0,
        0x4000,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x17, 0x50, 0x00, 0x00]);
}

#[test]
fn jal_writes_the_scattered_j_type_immediate() {
    let fixed = Fixed {
        symbol: 0x1100,
        got: 0,
        plt: 0x1100,
        got_base: 0,
    };
    // value = 0x1100 - 0x1000 = 0x100; imm[10:1] = 0x80 at [30..21].
    // JAL | (0x80 << 21) = 0x1000006F -> [0x6F, 0x00, 0x00, 0x10].
    let mut slot = JAL;
    apply::<Riscv64, _>(
        R_RISCV_JAL,
        Some(SymbolId(1)),
        0,
        0x1000,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x6F, 0x00, 0x00, 0x10]);
}

#[test]
fn branch_writes_the_scattered_b_type_immediate() {
    let fixed = Fixed {
        symbol: 0x1100,
        got: 0,
        plt: 0x1100,
        got_base: 0,
    };
    // value = 0x100; imm[10:5] = 0x8 at [30..25].
    // BEQ | (0x8 << 25) = 0x10000063 -> [0x63, 0x00, 0x00, 0x10].
    let mut slot = BEQ;
    apply::<Riscv64, _>(
        R_RISCV_BRANCH,
        Some(SymbolId(1)),
        0,
        0x1000,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x63, 0x00, 0x00, 0x10]);
}

#[test]
fn table_routes_compressed_branch_and_jump_to_escape() {
    // Compressed (16-bit) branch and jump carry scattered immediates in a
    // 2-byte cell, so they route to escape with a 2-byte placeholder.
    let escape2 = Spec {
        expr: RelExpr::Escape,
        write: Write::Field(Field::new(2, 0, 0, 0, Check::None)),
    };
    assert_eq!(Riscv64::spec(R_RISCV_RVC_BRANCH).unwrap(), escape2);
    assert_eq!(Riscv64::spec(R_RISCV_RVC_JUMP).unwrap(), escape2);
}

#[test]
fn rvc_jump_writes_the_scattered_cj_type_immediate() {
    let fixed = Fixed {
        symbol: 0x10,
        got: 0,
        plt: 0x10,
        got_base: 0,
    };
    // value = 0x10 - 0 = 0x10; imm[4] = 1 -> bit 11.
    // C_J | 0x800 = 0xA002 | 0x800 = 0xA802 -> [0x02, 0xA8].
    let mut slot = C_J;
    apply::<Riscv64, _>(
        R_RISCV_RVC_JUMP,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x02, 0xA8]);
}

#[test]
fn rvc_branch_writes_the_scattered_cb_type_immediate() {
    let fixed = Fixed {
        symbol: 0x10,
        got: 0,
        plt: 0x10,
        got_base: 0,
    };
    // value = 0x10; imm[4:3] = 0b10 -> bits [11:10] = 0x800.
    // C_BEQZ | 0x800 = 0xC001 | 0x800 = 0xC801 -> [0x01, 0xC8].
    let mut slot = C_BEQZ;
    apply::<Riscv64, _>(
        R_RISCV_RVC_BRANCH,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x01, 0xC8]);
}

#[test]
fn rvc_jump_overflow_is_detected() {
    // The c.j range is +/- 2 KiB (signed 12 bits); 4 KiB overflows.
    let fixed = Fixed {
        symbol: 0x1000,
        got: 0,
        plt: 0x1000,
        got_base: 0,
    };
    let mut slot = C_J;
    let err = apply::<Riscv64, _>(
        R_RISCV_RVC_JUMP,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(matches!(err, xold::Error::RelocOverflow(R_RISCV_RVC_JUMP)));
}

#[test]
fn call_writes_the_auipc_jalr_pair_with_a_nonzero_lo12() {
    let fixed = Fixed {
        symbol: 0x2234,
        got: 0,
        plt: 0x2234,
        got_base: 0,
    };
    // PltPc collapses to S + A - P in a static link: 0x2234 - 0x1000 = 0x1234.
    // auipc hi20: (0x1234 + 0x800) & 0xFFFFF000 = 0x1000 -> AUIPC | 0x1000.
    // jalr lo12: 0x1234 & 0xFFF = 0x234 -> JALR | (0x234 << 20) = 0x23400067.
    let mut slot = [AUIPC, JALR].concat();
    apply::<Riscv64, _>(
        R_RISCV_CALL,
        Some(SymbolId(1)),
        0,
        0x1000,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x17, 0x10, 0x00, 0x00, 0x67, 0x00, 0x40, 0x23]);
}

#[test]
fn call_plt_matches_call_in_a_static_link() {
    let fixed = Fixed {
        symbol: 0x2000,
        got: 0,
        plt: 0x2000,
        got_base: 0,
    };
    // value = 0x2000 - 0x1000 = 0x1000; auipc imm = 0x1000, jalr lo12 = 0.
    let mut slot = [AUIPC, JALR].concat();
    apply::<Riscv64, _>(
        R_RISCV_CALL_PLT,
        Some(SymbolId(1)),
        0,
        0x1000,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x17, 0x10, 0x00, 0x00, 0x67, 0x00, 0x00, 0x00]);
}

#[test]
fn pcrel_lo12_i_writes_the_low_12_bits_of_the_pc_value() {
    // After the writer's pairing pass has rewritten the LO12 entry to carry
    // the HI20's target, the escape stores the low 12 bits of `S + A - P` in
    // the I-type immediate field at [31..20]. The driver is exercised here as
    // if the rewrite had already happened.
    let fixed = Fixed {
        symbol: 0x1234,
        ..zero()
    };
    // value = 0x1234 - 0x4 = 0x1230; lo12 = 0x230; ADDI | (0x230 << 20) =
    // 0x23000013 -> [0x13, 0x00, 0x00, 0x23].
    let mut slot = ADDI;
    apply::<Riscv64, _>(
        R_RISCV_PCREL_LO12_I,
        Some(SymbolId(1)),
        0,
        0x4,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x13, 0x00, 0x00, 0x23]);
}

#[test]
fn pcrel_lo12_s_writes_the_scattered_s_type_low_12() {
    // Same as the absolute LO12_S but over the PC-relative value.
    let fixed = Fixed {
        symbol: 0x1234,
        ..zero()
    };
    // value = 0x1234; lo12 = 0x234; imm[11:5] = 0x11, imm[4:0] = 0x14.
    // SW | 0x22000A00 = 0x22000A23 -> [0x23, 0x0A, 0x00, 0x22].
    let mut slot = SW;
    apply::<Riscv64, _>(
        R_RISCV_PCREL_LO12_S,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x23, 0x0A, 0x00, 0x22]);
}

/// Packs `(sym, r_type)` into an ELF64 `r_info` word, mirroring the writer.
fn info(sym: u32, r_type: u32) -> u64 {
    (u64::from(sym) << 32) | u64::from(r_type)
}

#[test]
fn rewrite_pairs_each_pcrel_lo12_with_its_hi20() {
    // The LO12's symbol is a local label on the `auipc` (here section offset
    // 8); the resolver returns `section_base + 8` for it. The pass must move
    // the HI20's target symbol (5) onto the LO12 and shift the addend by
    // (lo_off - hi_off) so the Pc value at the LO12 reconstructs the auipc's
    // `S + A - P_hi`.
    let section_base = 0x1000u64;
    let label_addr = section_base + 8;
    let resolver = Fixed {
        symbol: label_addr,
        ..zero()
    };
    let entries = [
        Rela64 {
            r_offset: U64::new(8),
            r_info: U64::new(info(5, R_RISCV_PCREL_HI20)),
            r_addend: I64::new(0),
        },
        Rela64 {
            r_offset: U64::new(0xc),
            r_info: U64::new(info(2, R_RISCV_PCREL_LO12_I)),
            r_addend: I64::new(0),
        },
    ];
    let rewritten = rewrite_pcrel_pairs(&entries, section_base, &resolver)
        .expect("paired LO12 rewrites")
        .expect("at least one LO12 was paired");
    assert_eq!(rewritten.len(), 2);
    let lo12 = &rewritten[1];
    assert_eq!(lo12.r_offset.get(), 0xc);
    assert_eq!(lo12.sym(), 5);
    assert_eq!(lo12.r_type(), R_RISCV_PCREL_LO12_I);
    assert_eq!(lo12.r_addend.get(), 4);
}

#[test]
fn rewrite_returns_none_when_there_is_no_hi20() {
    // A section without any PCREL_HI20 cannot host a paired LO12; the pass
    // returns None so the writer keeps the borrowed slice unchanged.
    let resolver = zero();
    let entries = [Rela64 {
        r_offset: U64::new(0),
        r_info: U64::new(info(1, R_RISCV_LO12_I)),
        r_addend: I64::new(0),
    }];
    assert!(
        rewrite_pcrel_pairs(&entries, 0x1000, &resolver)
            .unwrap()
            .is_none()
    );
}

#[test]
fn rewrite_errors_when_a_lo12_has_no_matching_hi20() {
    // A LO12 whose label does not land on a HI20 site is malformed; rather
    // than write wrong bytes, the pass surfaces an error.
    let resolver = Fixed {
        symbol: 0x2000,
        ..zero()
    };
    let entries = [Rela64 {
        r_offset: U64::new(0xc),
        r_info: U64::new(info(2, R_RISCV_PCREL_LO12_I)),
        r_addend: I64::new(0),
    }];
    let res = rewrite_pcrel_pairs(&entries, 0x1000, &resolver);
    assert!(matches!(res, Err(xold::Error::Format(_))));
}

#[test]
fn jal_overflow_is_detected() {
    // The JAL range is +/- 1 MiB (signed 21 bits); a 2 MiB jump overflows.
    let fixed = Fixed {
        symbol: 0x0020_0000,
        got: 0,
        plt: 0x0020_0000,
        got_base: 0,
    };
    let mut slot = JAL;
    let err = apply::<Riscv64, _>(
        R_RISCV_JAL,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(matches!(err, xold::Error::RelocOverflow(R_RISCV_JAL)));
}

#[test]
fn branch_overflow_is_detected() {
    // The conditional branch range is +/- 4 KiB (signed 13 bits); 8 KiB
    // overflows.
    let fixed = Fixed {
        symbol: 0x2000,
        got: 0,
        plt: 0x2000,
        got_base: 0,
    };
    let mut slot = BEQ;
    let err = apply::<Riscv64, _>(
        R_RISCV_BRANCH,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(matches!(err, xold::Error::RelocOverflow(R_RISCV_BRANCH)));
}

#[test]
fn scan_target_allocates_plt_for_call_and_nothing_for_lo12_i() {
    // CALL is semantically PLT[sym] + A - P; the scan records the PLT need. (In
    // a static link the PLT collapses to the symbol, so nothing is actually
    // allocated, but the table is honest about the reference kind.)
    assert_eq!(
        scan_target(Target::Riscv64, R_RISCV_CALL).unwrap(),
        Needs {
            plt: true,
            ..Needs::NONE
        }
    );
    assert_eq!(
        scan_target(Target::Riscv64, R_RISCV_CALL_PLT).unwrap(),
        Needs {
            plt: true,
            ..Needs::NONE
        }
    );
    // LO12_I is a direct absolute reference: no GOT, no PLT.
    assert_eq!(
        scan_target(Target::Riscv64, R_RISCV_LO12_I).unwrap(),
        Needs::NONE
    );
    // The RELAX hint forces no allocation.
    assert_eq!(
        scan_target(Target::Riscv64, R_RISCV_RELAX).unwrap(),
        Needs::NONE
    );
}

#[test]
fn target_round_trips_e_machine() {
    use xold::elf::constants::EM_RISCV;
    assert_eq!(Target::from_machine(EM_RISCV).unwrap(), Target::Riscv64);
    assert_eq!(Target::Riscv64.machine(), EM_RISCV);
}

#[test]
fn none_relocations_leave_the_slot_untouched() {
    let fixed = Fixed {
        symbol: 0,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0xab; 8];
    apply::<Riscv64, _>(
        R_RISCV_RELAX,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0xab; 8]);
}

#[test]
fn table_classifies_eh_frame_relocations() {
    // `R_RISCV_32_PCREL` is a plain signed 32-bit `S + A - P` store.
    assert_eq!(
        Riscv64::spec(R_RISCV_32_PCREL).unwrap(),
        Spec {
            expr: RelExpr::Pc,
            write: Write::Bytes(WriteKind::W32S),
        }
    );
    // ADD/SUB/SET6/SUB6 are read-modify-write, so they route to escape. The
    // `Bytes` placeholder only sizes the slot; the table groups them by width.
    let esc = |w| Spec {
        expr: RelExpr::Escape,
        write: Write::Bytes(w),
    };
    assert_eq!(Riscv64::spec(R_RISCV_ADD8).unwrap(), esc(WriteKind::W8));
    assert_eq!(Riscv64::spec(R_RISCV_SUB8).unwrap(), esc(WriteKind::W8));
    assert_eq!(Riscv64::spec(R_RISCV_SET6).unwrap(), esc(WriteKind::W8));
    assert_eq!(Riscv64::spec(R_RISCV_SUB6).unwrap(), esc(WriteKind::W8));
    assert_eq!(Riscv64::spec(R_RISCV_SET8).unwrap(), esc(WriteKind::W8));
    assert_eq!(Riscv64::spec(R_RISCV_ADD16).unwrap(), esc(WriteKind::W16));
    assert_eq!(Riscv64::spec(R_RISCV_SUB16).unwrap(), esc(WriteKind::W16));
    assert_eq!(Riscv64::spec(R_RISCV_SET16).unwrap(), esc(WriteKind::W16));
    assert_eq!(Riscv64::spec(R_RISCV_ADD32).unwrap(), esc(WriteKind::W32));
    assert_eq!(Riscv64::spec(R_RISCV_SUB32).unwrap(), esc(WriteKind::W32));
    assert_eq!(Riscv64::spec(R_RISCV_SET32).unwrap(), esc(WriteKind::W32));
    assert_eq!(Riscv64::spec(R_RISCV_ADD64).unwrap(), esc(WriteKind::W64));
}

#[test]
fn pcrel_32_stores_a_signed_pc_relative_value() {
    let fixed = Fixed {
        symbol: 0x1000,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    // S + A - P = 0x1000 - 0x800 = 0x800.
    let mut slot = [0u8; 4];
    apply::<Riscv64, _>(
        R_RISCV_32_PCREL,
        Some(SymbolId(1)),
        0,
        0x800,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x00, 0x08, 0x00, 0x00]);

    // Negative: 0x800 - 0x1000 = -0x800, stored as signed 32-bit.
    let mut slot = [0u8; 4];
    apply::<Riscv64, _>(
        R_RISCV_32_PCREL,
        Some(SymbolId(1)),
        0,
        0x1000,
        &Fixed {
            symbol: 0x800,
            ..zero()
        },
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x00, 0xf8, 0xff, 0xff]);
}

#[test]
fn add_sub_pair_computes_the_difference_like_eh_frame_range() {
    // An FDE address_range field is built from ADD32(end) then SUB32(start)
    // applied to the same zero-initialised slot: the result is end - start.
    let end = Fixed {
        symbol: 0x4000_0022,
        ..zero()
    };
    let start = Fixed {
        symbol: 0x4000_0000,
        ..zero()
    };
    let mut slot = [0u8; 4];
    apply::<Riscv64, _>(
        R_RISCV_ADD32,
        Some(SymbolId(1)),
        0,
        0,
        &end,
        &mut slot,
    )
    .unwrap();
    apply::<Riscv64, _>(
        R_RISCV_SUB32,
        Some(SymbolId(1)),
        0,
        0,
        &start,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x22, 0x00, 0x00, 0x00]);
}

#[test]
fn add_wraps_modulo_the_slot_width() {
    // An all-ones slot plus one wraps to zero at every width.
    let one = Fixed {
        symbol: 1,
        ..zero()
    };
    let mut s8 = [0xffu8; 1];
    apply::<Riscv64, _>(R_RISCV_ADD8, Some(SymbolId(1)), 0, 0, &one, &mut s8)
        .unwrap();
    assert_eq!(s8, [0x00]);

    let mut s16 = [0xffu8; 2];
    apply::<Riscv64, _>(R_RISCV_ADD16, Some(SymbolId(1)), 0, 0, &one, &mut s16)
        .unwrap();
    assert_eq!(s16, [0x00, 0x00]);

    let mut s64 = [0xffu8; 8];
    apply::<Riscv64, _>(R_RISCV_ADD64, Some(SymbolId(1)), 0, 0, &one, &mut s64)
        .unwrap();
    assert_eq!(s64, [0x00; 8]);
}

#[test]
fn sub_wraps_modulo_the_slot_width() {
    // A zero slot minus one wraps to all-ones at every width.
    let one = Fixed {
        symbol: 1,
        ..zero()
    };
    let mut s8 = [0u8; 1];
    apply::<Riscv64, _>(R_RISCV_SUB8, Some(SymbolId(1)), 0, 0, &one, &mut s8)
        .unwrap();
    assert_eq!(s8, [0xff]);

    let mut s16 = [0u8; 2];
    apply::<Riscv64, _>(R_RISCV_SUB16, Some(SymbolId(1)), 0, 0, &one, &mut s16)
        .unwrap();
    assert_eq!(s16, [0xff, 0xff]);
}

#[test]
fn set6_writes_low_six_bits_and_masks_high_value_bits() {
    // High 2 bits are preserved; only the low 6 bits of the value are stored.
    let val = Fixed {
        symbol: 0x2a,
        ..zero()
    };
    let mut slot = [0xc0u8]; // high bits 0b11, low bits 0
    apply::<Riscv64, _>(R_RISCV_SET6, Some(SymbolId(1)), 0, 0, &val, &mut slot)
        .unwrap();
    assert_eq!(slot, [0xc0 | 0x2a]);

    // Bits above bit 5 of the value are dropped: 0xff -> low6 0x3f.
    let val = Fixed {
        symbol: 0xff,
        ..zero()
    };
    let mut slot = [0x80u8];
    apply::<Riscv64, _>(R_RISCV_SET6, Some(SymbolId(1)), 0, 0, &val, &mut slot)
        .unwrap();
    assert_eq!(slot, [0x80 | 0x3f]);
}

#[test]
fn sub6_subtracts_from_low_six_bits_with_wraparound() {
    // (low6 - val) mod 64, high 2 bits preserved.
    let val = Fixed {
        symbol: 0x05,
        ..zero()
    };
    let mut slot = [0xd0u8]; // low6 = 0x10
    apply::<Riscv64, _>(R_RISCV_SUB6, Some(SymbolId(1)), 0, 0, &val, &mut slot)
        .unwrap();
    assert_eq!(slot, [0xc0 | ((0x10 - 0x05) & 0x3f)]);

    // Wraparound: 0x02 - 0x05 wraps to 0x3d in 6 bits.
    let mut slot = [0xc2u8]; // low6 = 0x02
    apply::<Riscv64, _>(R_RISCV_SUB6, Some(SymbolId(1)), 0, 0, &val, &mut slot)
        .unwrap();
    assert_eq!(slot, [0xc0 | 0x3d]);
}

#[test]
fn set6_sub6_pair_matches_a_real_eh_frame_augmentation_field() {
    // The same SET6/SUB6 pair the assembler emits for an eh_frame
    // augmentation length: SET6 writes end's low bits, SUB6 subtracts
    // start's, yielding (end - start) mod 64 in the low 6 bits while the
    // high 2 bits survive from the original byte.
    let end = Fixed {
        symbol: 0x4000_0102,
        ..zero()
    };
    let start = Fixed {
        symbol: 0x4000_00f0,
        ..zero()
    };
    let mut slot = [0x40u8]; // high bits 0b01, low bits 0
    apply::<Riscv64, _>(R_RISCV_SET6, Some(SymbolId(1)), 0, 0, &end, &mut slot)
        .unwrap();
    apply::<Riscv64, _>(
        R_RISCV_SUB6,
        Some(SymbolId(1)),
        0,
        0,
        &start,
        &mut slot,
    )
    .unwrap();
    // (0x02 - 0x30) mod 64 = 0x12; high bits 0b01 -> 0x52.
    assert_eq!(slot, [0x52]);
}

#[test]
fn set8_overwrites_the_full_byte() {
    // SET8 is a plain store, not read-modify-write: the previous contents are
    // discarded and the low 8 bits of `S + A` are written.
    let val = Fixed {
        symbol: 0x41,
        ..zero()
    };
    let mut slot = [0xffu8];
    apply::<Riscv64, _>(R_RISCV_SET8, Some(SymbolId(1)), 0, 0, &val, &mut slot)
        .unwrap();
    assert_eq!(slot, [0x41]);

    // Bits above bit 7 are dropped: 0x142 -> 0x42.
    let val = Fixed {
        symbol: 0x142,
        ..zero()
    };
    let mut slot = [0x00u8];
    apply::<Riscv64, _>(R_RISCV_SET8, Some(SymbolId(1)), 0, 0, &val, &mut slot)
        .unwrap();
    assert_eq!(slot, [0x42]);
}

#[test]
fn set16_and_set32_overwrite_the_full_slot() {
    let val = Fixed {
        symbol: 0x0102_0304,
        ..zero()
    };
    let mut s16 = [0xffu8; 2];
    apply::<Riscv64, _>(R_RISCV_SET16, Some(SymbolId(1)), 0, 0, &val, &mut s16)
        .unwrap();
    assert_eq!(s16, [0x04, 0x03]);

    let mut s32 = [0xffu8; 4];
    apply::<Riscv64, _>(R_RISCV_SET32, Some(SymbolId(1)), 0, 0, &val, &mut s32)
        .unwrap();
    assert_eq!(s32, [0x04, 0x03, 0x02, 0x01]);
}

#[test]
fn set_pair_overwrites_then_subtracts_like_eh_frame() {
    // A SET8/SUB8 pair overwrites with end's low byte, then subtracts start's,
    // yielding (end - start) mod 256: the byte-width analogue of the
    // ADD32/SUB32 address-range pair.
    let end = Fixed {
        symbol: 0x4000_0022,
        ..zero()
    };
    let start = Fixed {
        symbol: 0x4000_0010,
        ..zero()
    };
    let mut slot = [0xabu8];
    apply::<Riscv64, _>(R_RISCV_SET8, Some(SymbolId(1)), 0, 0, &end, &mut slot)
        .unwrap();
    apply::<Riscv64, _>(
        R_RISCV_SUB8,
        Some(SymbolId(1)),
        0,
        0,
        &start,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x12]);
}

#[test]
fn add_sub_relocations_force_no_got_or_plt_allocation() {
    // ADD/SUB/SET6/SUB6 reference a symbol but need no GOT or PLT entry.
    for r_type in [
        R_RISCV_ADD32,
        R_RISCV_SUB32,
        R_RISCV_SET6,
        R_RISCV_SUB6,
        R_RISCV_SET8,
        R_RISCV_SET16,
        R_RISCV_SET32,
        R_RISCV_32_PCREL,
    ] {
        assert_eq!(scan_target(Target::Riscv64, r_type).unwrap(), Needs::NONE);
    }
}
