//! Mach-O relocation tables: classification of every handled `X86_64_RELOC_*`
//! and `ARM64_RELOC_*` type, the bitfield driver for contiguous and split
//! arm64 immediates, the scan GOT allocation, and a real-darwin-object
//! classification check.
//!
//! The bitfield tests start from real instruction opcodes (BL, ADD, ADRP) and
//! check the exact mutated bytes, mirroring `tests/reloc_aarch64.rs`. The
//! real-object checks are gated on `clang --target=*-apple-darwin`.

use std::{
    fs,
    path::PathBuf,
    process::{Command, Stdio},
};

use common::which;
use xold::{
    macho::MachOFile,
    mmap_file::MappedFile,
    reloc::{
        Arch, Check, Field, Needs, RelExpr, Resolver, Spec, Write, WriteKind,
        apply,
        macho_arm64::{
            ARM64_RELOC_ADDEND, ARM64_RELOC_BRANCH26,
            ARM64_RELOC_GOT_LOAD_PAGE21, ARM64_RELOC_GOT_LOAD_PAGEOFF12,
            ARM64_RELOC_PAGE21, ARM64_RELOC_PAGEOFF12,
            ARM64_RELOC_POINTER_TO_GOT, ARM64_RELOC_SUBTRACTOR,
            ARM64_RELOC_TLVP_LOAD_PAGE21, ARM64_RELOC_UNSIGNED, MachoArm64,
        },
        macho_x86_64::{
            MachoX86_64, X86_64_RELOC_BRANCH, X86_64_RELOC_GOT,
            X86_64_RELOC_GOT_LOAD, X86_64_RELOC_SIGNED, X86_64_RELOC_SIGNED_1,
            X86_64_RELOC_SIGNED_2, X86_64_RELOC_SIGNED_4,
            X86_64_RELOC_SUBTRACTOR, X86_64_RELOC_TLV, X86_64_RELOC_UNSIGNED,
        },
        scan,
    },
    symbol::SymbolId,
};

mod common;

/// A resolver returning fixed addresses, mirroring `tests/reloc_aarch64.rs`.
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

/// `BL`, opcode `0x94000000`, imm26 at bits [0..26].
const BL: [u8; 4] = [0x00, 0x00, 0x00, 0x94];
/// `ADD x0, x0, #0`, opcode `0x91000000`, imm12 at bits [10..21].
const ADD: [u8; 4] = [0x00, 0x00, 0x00, 0x91];
/// `ADRP x0`, opcode `0x90000000`, split imm21 (immlo [29..30], immhi [5..23]).
const ADRP: [u8; 4] = [0x00, 0x00, 0x00, 0x90];

#[test]
fn x86_64_table_classifies_each_type() {
    assert_eq!(
        MachoX86_64::spec(X86_64_RELOC_UNSIGNED).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Bytes(WriteKind::W64),
        }
    );
    assert_eq!(
        MachoX86_64::spec(X86_64_RELOC_SIGNED).unwrap(),
        Spec {
            expr: RelExpr::Pc,
            write: Write::Bytes(WriteKind::W32S),
        }
    );
    assert_eq!(
        MachoX86_64::spec(X86_64_RELOC_BRANCH).unwrap(),
        Spec {
            expr: RelExpr::PltPc,
            write: Write::Bytes(WriteKind::W32S),
        }
    );
    for ty in [X86_64_RELOC_GOT_LOAD, X86_64_RELOC_GOT] {
        assert_eq!(
            MachoX86_64::spec(ty).unwrap(),
            Spec {
                expr: RelExpr::GotPc,
                write: Write::Bytes(WriteKind::W32S),
            }
        );
    }
    for ty in [
        X86_64_RELOC_SIGNED_1,
        X86_64_RELOC_SIGNED_2,
        X86_64_RELOC_SIGNED_4,
    ] {
        assert_eq!(
            MachoX86_64::spec(ty).unwrap(),
            Spec {
                expr: RelExpr::Pc,
                write: Write::Bytes(WriteKind::W32S),
            }
        );
    }
    let escape = Spec {
        expr: RelExpr::Escape,
        write: Write::Bytes(WriteKind::W64),
    };
    assert_eq!(MachoX86_64::spec(X86_64_RELOC_SUBTRACTOR).unwrap(), escape);
    assert_eq!(MachoX86_64::spec(X86_64_RELOC_TLV).unwrap(), escape);
}

#[test]
fn x86_64_table_rejects_unknown_type() {
    assert!(matches!(
        scan::<MachoX86_64>(50).unwrap_err(),
        xold::Error::UnsupportedReloc(50)
    ));
}

#[test]
fn x86_64_scan_allocates_got_for_got_relocations() {
    for ty in [X86_64_RELOC_GOT_LOAD, X86_64_RELOC_GOT] {
        assert_eq!(
            scan::<MachoX86_64>(ty).unwrap(),
            Needs {
                got: true,
                ..Needs::NONE
            }
        );
    }
    // A branch is a direct call: no GOT entry.
    let needs = scan::<MachoX86_64>(X86_64_RELOC_BRANCH).unwrap();
    assert!(!needs.got);
}

#[test]
fn arm64_table_classifies_each_type() {
    assert_eq!(
        MachoArm64::spec(ARM64_RELOC_UNSIGNED).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Bytes(WriteKind::W64),
        }
    );
    // BRANCH26: contiguous imm26 field, scaled and signed.
    assert_eq!(
        MachoArm64::spec(ARM64_RELOC_BRANCH26).unwrap(),
        Spec {
            expr: RelExpr::PltPc,
            write: Write::Field(Field::scaled(4, 2, 0, 26, Check::Signed)),
        }
    );
    assert_eq!(
        MachoArm64::spec(ARM64_RELOC_POINTER_TO_GOT).unwrap(),
        Spec {
            expr: RelExpr::GotPc,
            write: Write::Bytes(WriteKind::W32S),
        }
    );
    let escape = Spec {
        expr: RelExpr::Escape,
        write: Write::Field(Field::new(4, 0, 0, 0, Check::None)),
    };
    assert_eq!(MachoArm64::spec(ARM64_RELOC_PAGE21).unwrap(), escape);
    // Both page-offset forms escape: the shift comes from the opcode, which a
    // field cannot read.
    assert_eq!(MachoArm64::spec(ARM64_RELOC_PAGEOFF12).unwrap(), escape);
    assert_eq!(
        MachoArm64::spec(ARM64_RELOC_GOT_LOAD_PAGEOFF12).unwrap(),
        escape
    );
    assert_eq!(
        MachoArm64::spec(ARM64_RELOC_GOT_LOAD_PAGE21).unwrap(),
        escape
    );
    assert_eq!(MachoArm64::spec(ARM64_RELOC_SUBTRACTOR).unwrap(), escape);
    assert_eq!(
        MachoArm64::spec(ARM64_RELOC_TLVP_LOAD_PAGE21).unwrap(),
        escape
    );
    // ADDEND is an annotation with no fixup at its own site.
    assert_eq!(
        MachoArm64::spec(ARM64_RELOC_ADDEND).unwrap(),
        Spec {
            expr: RelExpr::None,
            write: Write::Bytes(WriteKind::W64),
        }
    );
}

#[test]
fn arm64_scan_allocates_got_for_got_relocations() {
    // GOT_LOAD_PAGEOFF12 references the GOT directly.
    assert_eq!(
        scan::<MachoArm64>(ARM64_RELOC_GOT_LOAD_PAGEOFF12).unwrap(),
        Needs {
            got: true,
            ..Needs::NONE
        }
    );
    // GOT_LOAD_PAGE21 is Escape for writing but still needs a GOT entry.
    assert_eq!(
        scan::<MachoArm64>(ARM64_RELOC_GOT_LOAD_PAGE21).unwrap(),
        Needs {
            got: true,
            ..Needs::NONE
        }
    );
    // A direct branch needs no GOT.
    let needs = scan::<MachoArm64>(ARM64_RELOC_BRANCH26).unwrap();
    assert!(!needs.got);
}

#[test]
fn arm64_branch26_writes_imm26() {
    let fixed = Fixed {
        symbol: 0x1000,
        got: 0,
        plt: 0x1000,
        got_base: 0,
    };
    // value = 0x1000 - 0x800 = 0x800; stored as (0x800 >> 2) = 0x200 at imm26.
    // BL | 0x200 = 0x94000200 -> [0x00, 0x02, 0x00, 0x94].
    let mut slot = BL;
    apply::<MachoArm64, _>(
        ARM64_RELOC_BRANCH26,
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
fn arm64_pageoff12_writes_low_12_bits_into_imm12() {
    let fixed = Fixed {
        symbol: 0x1234,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    // value = 0x1234 + 0x10 = 0x1244; low 12 bits = 0x244; << 10 = 0x91000.
    // ADD | 0x91000 = 0x91091000 -> [0x00, 0x10, 0x09, 0x91].
    let mut slot = ADD;
    apply::<MachoArm64, _>(
        ARM64_RELOC_PAGEOFF12,
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
fn arm64_page21_writes_split_21_bit_immediate() {
    let fixed = Fixed {
        symbol: 0x9000,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    // page(0x9000) - page(0x4000) = 0x5000; >> 12 = 0x5 (page count).
    // imm 0x5: immlo = 1 -> bits [29..30]; immhi = 1 -> bits [5..23].
    // ADRP | 0x20000000 | 0x20 = 0xB0000020.
    let mut slot = ADRP;
    apply::<MachoArm64, _>(
        ARM64_RELOC_PAGE21,
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
fn arm64_got_load_page21_writes_page_offset_of_got_entry() {
    let fixed = Fixed {
        symbol: 0,
        got: 0x9000,
        plt: 0,
        got_base: 0x9000,
    };
    // page(GOT[sym] + 0) - page(P) = 0x5000; >> 12 = 0x5, same split encoding.
    let mut slot = ADRP;
    apply::<MachoArm64, _>(
        ARM64_RELOC_GOT_LOAD_PAGE21,
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
fn arm64_subtractor_and_tls_escape_is_rejected() {
    let fixed = Fixed {
        symbol: 0,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 4];
    for ty in [ARM64_RELOC_SUBTRACTOR, ARM64_RELOC_TLVP_LOAD_PAGE21] {
        let err = apply::<MachoArm64, _>(
            ty,
            Some(SymbolId(1)),
            0,
            0,
            &fixed,
            &mut slot,
        )
        .unwrap_err();
        assert!(matches!(err, xold::Error::UnsupportedReloc(t) if t == ty));
    }
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn classifies_relocations_of_a_real_darwin_object() {
    // The same source the reader test uses; the relocs it emits are decoded and
    // every fixup type must classify to a known spec.
    let src = b"extern int ext_var;\n\
               extern int ext_func(int);\n\
               int global = 42;\n\
               int call_ext(void) { return ext_func(ext_var); }\n\
               int *get_global(void) { return &global; }\n";
    let triple = "--target=arm64-apple-darwin";
    let Some(clang) = darwin_clang(triple) else {
        eprintln!(
            "skipping arm64 reloc classification: clang darwin target not found"
        );
        return;
    };
    let src_path = std::env::temp_dir().join("xold_macho_reloc_src.c");
    let path = std::env::temp_dir().join("xold_macho_reloc.o");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(&clang)
        .args([triple, "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(&path)
        .status()
        .is_ok_and(|s| s.success());
    if !ok {
        eprintln!("skipping arm64 reloc classification: clang compile failed");
        return;
    }
    let mapped = MappedFile::open(&path).expect("open darwin object");
    let obj = MachOFile::parse(mapped.bytes()).expect("parse darwin object");
    assert!(obj.is_arm64());

    let mut seen_branch = false;
    let mut seen_page = false;
    for section in obj.sections() {
        for reloc in &section.relocations {
            let spec = MachoArm64::spec(u32::from(reloc.r_type));
            // Every reloc clang emitted must map to a known spec; escape and
            // none specs are valid classifications too (SUBTRACTOR/ADDEND/TLS).
            assert!(
                spec.is_ok(),
                "unclassified arm64 reloc type {}",
                reloc.r_type
            );
            if reloc.r_type == 2 {
                seen_branch = true;
            }
            if reloc.r_type == 3 || reloc.r_type == 5 {
                seen_page = true;
            }
        }
    }
    assert!(seen_branch, "expected an ARM64_RELOC_BRANCH26");
    assert!(seen_page, "expected a page21 relocation");
}

// --- helpers --------------------------------------------------------------

/// Returns the clang binary if it can compile for `triple`, else `None`.
fn darwin_clang(triple: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let probe = std::env::temp_dir().join("xold_darwin_reloc_probe.o");
    let ok = Command::new(&clang)
        .args([triple, "-c", "-x", "c", "-", "-o"])
        .arg(&probe)
        .stdin(Stdio::null())
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&probe);
    ok.then_some(clang)
}
