//! COFF relocation tables: classification of every handled `IMAGE_REL_AMD64_*`
//! and `IMAGE_REL_I386_*` type, the scan allocation, the apply driver for the
//! standard value relocs, and a real-Windows-object classification check.
//!
//! The real-object checks are gated on `clang --target=*-pc-windows-msvc`.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use common::which;
use xold::{
    coff::CoffFile,
    mmap_file::MappedFile,
    reloc::{
        Arch, Check, Field, Needs, RelExpr, Resolver, Spec, Write, WriteKind,
        apply,
        coff_i386::{
            CoffI386, IMAGE_REL_I386_ABSOLUTE, IMAGE_REL_I386_DIR16,
            IMAGE_REL_I386_DIR32, IMAGE_REL_I386_DIR32NB, IMAGE_REL_I386_REL16,
            IMAGE_REL_I386_REL32, IMAGE_REL_I386_SECREL,
            IMAGE_REL_I386_SECREL7, IMAGE_REL_I386_SECTION,
        },
        coff_x86_64::{
            CoffX86_64, IMAGE_REL_AMD64_ABSOLUTE, IMAGE_REL_AMD64_ADDR32,
            IMAGE_REL_AMD64_ADDR32NB, IMAGE_REL_AMD64_ADDR64,
            IMAGE_REL_AMD64_PAIR, IMAGE_REL_AMD64_REL32,
            IMAGE_REL_AMD64_REL32_1, IMAGE_REL_AMD64_REL32_2,
            IMAGE_REL_AMD64_REL32_3, IMAGE_REL_AMD64_REL32_4,
            IMAGE_REL_AMD64_REL32_5, IMAGE_REL_AMD64_SECREL,
            IMAGE_REL_AMD64_SECREL7, IMAGE_REL_AMD64_SECTION,
            IMAGE_REL_AMD64_SSPAN32, IMAGE_REL_AMD64_TOKEN,
        },
        scan,
    },
    symbol::SymbolId,
};

mod common;

/// A resolver returning fixed addresses, mirroring `tests/reloc_macho.rs`.
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

#[test]
fn amd64_table_classifies_each_type() {
    assert_eq!(
        CoffX86_64::spec(IMAGE_REL_AMD64_ABSOLUTE).unwrap(),
        Spec {
            expr: RelExpr::None,
            write: Write::Bytes(WriteKind::W64),
        }
    );
    assert_eq!(
        CoffX86_64::spec(IMAGE_REL_AMD64_ADDR64).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Bytes(WriteKind::W64),
        }
    );
    assert_eq!(
        CoffX86_64::spec(IMAGE_REL_AMD64_ADDR32).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Bytes(WriteKind::W32),
        }
    );
    // REL32 and REL32_1..5 all compute S + A - P as a signed 32-bit store;
    // the extra -1..-5 byte correction is applied at reloc normalisation.
    for ty in [
        IMAGE_REL_AMD64_REL32,
        IMAGE_REL_AMD64_REL32_1,
        IMAGE_REL_AMD64_REL32_2,
        IMAGE_REL_AMD64_REL32_3,
        IMAGE_REL_AMD64_REL32_4,
        IMAGE_REL_AMD64_REL32_5,
    ] {
        assert_eq!(
            CoffX86_64::spec(ty).unwrap(),
            Spec {
                expr: RelExpr::Pc,
                write: Write::Bytes(WriteKind::W32S),
            }
        );
    }
    // RVA and section-relative types route through Escape.
    let escape32 = Spec {
        expr: RelExpr::Escape,
        write: Write::Bytes(WriteKind::W32),
    };
    assert_eq!(
        CoffX86_64::spec(IMAGE_REL_AMD64_ADDR32NB).unwrap(),
        escape32
    );
    assert_eq!(CoffX86_64::spec(IMAGE_REL_AMD64_SECREL).unwrap(), escape32);
    assert_eq!(
        CoffX86_64::spec(IMAGE_REL_AMD64_SECTION).unwrap(),
        Spec {
            expr: RelExpr::Escape,
            write: Write::Bytes(WriteKind::W16),
        }
    );
    assert_eq!(CoffX86_64::spec(IMAGE_REL_AMD64_TOKEN).unwrap(), escape32);
    assert_eq!(CoffX86_64::spec(IMAGE_REL_AMD64_SSPAN32).unwrap(), escape32);
    // SECREL7 patches the low 7 bits of a byte: a 1-byte cell, 7-bit field.
    assert_eq!(
        CoffX86_64::spec(IMAGE_REL_AMD64_SECREL7).unwrap(),
        Spec {
            expr: RelExpr::Escape,
            write: Write::Field(Field::new(1, 0, 0, 7, Check::None)),
        }
    );
}

#[test]
fn amd64_table_rejects_unknown_type() {
    assert!(matches!(
        scan::<CoffX86_64>(0x00ff).unwrap_err(),
        xold::Error::UnsupportedReloc(0x00ff)
    ));
}

#[test]
fn amd64_rel32_chain_includes_all_six_variants() {
    // The six REL32 forms carry consecutive type numbers 0x04..0x09.
    for (i, ty) in [
        IMAGE_REL_AMD64_REL32,
        IMAGE_REL_AMD64_REL32_1,
        IMAGE_REL_AMD64_REL32_2,
        IMAGE_REL_AMD64_REL32_3,
        IMAGE_REL_AMD64_REL32_4,
        IMAGE_REL_AMD64_REL32_5,
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(*ty, u32::try_from(0x04 + i).unwrap());
    }
}

#[test]
fn amd64_scan_needs_no_got_or_plt_for_standard_types() {
    // The standard COFF AMD64 relocs resolve directly; only escape relocs
    // would force allocation, and those are handled by the writer.
    for ty in [
        IMAGE_REL_AMD64_ADDR64,
        IMAGE_REL_AMD64_ADDR32,
        IMAGE_REL_AMD64_REL32,
    ] {
        assert_eq!(scan::<CoffX86_64>(ty).unwrap(), Needs::NONE, "ty {ty}");
    }
}

#[test]
fn amd64_addr64_stores_symbol_plus_addend() {
    let fixed = Fixed {
        symbol: 0x0123_4567_89ab_cdef,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 8];
    apply::<CoffX86_64, _>(
        IMAGE_REL_AMD64_ADDR64,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, 0x0123_4567_89ab_cdefu64.to_le_bytes());
}

#[test]
fn amd64_addr32_stores_low_32_bits() {
    let fixed = Fixed {
        symbol: 0x0000_0000_dead_beef,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 4];
    apply::<CoffX86_64, _>(
        IMAGE_REL_AMD64_ADDR32,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, 0xdead_beefu32.to_le_bytes());
}

#[test]
fn amd64_rel32_stores_signed_displacement() {
    let fixed = Fixed {
        symbol: 0x1000,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    // value = 0x1000 - 0x0800 = 0x800 (the addend is folded in at
    // normalisation; here the driver sees the net addend 0).
    let mut slot = [0u8; 4];
    apply::<CoffX86_64, _>(
        IMAGE_REL_AMD64_REL32,
        Some(SymbolId(1)),
        0,
        0x0800,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, 0x800u32.to_le_bytes());
}

#[test]
fn amd64_pair_classifies_as_escape() {
    // PAIR is the high-half annotation following an SREL32; it carries no
    // independent fixup but is routed through Escape so the driver can walk
    // the pair rather than skip it as None.
    assert_eq!(
        CoffX86_64::spec(IMAGE_REL_AMD64_PAIR).unwrap(),
        Spec {
            expr: RelExpr::Escape,
            write: Write::Bytes(WriteKind::W32),
        }
    );
}

#[test]
fn i386_table_classifies_each_type() {
    assert_eq!(
        CoffI386::spec(IMAGE_REL_I386_ABSOLUTE).unwrap(),
        Spec {
            expr: RelExpr::None,
            write: Write::Bytes(WriteKind::W32),
        }
    );
    assert_eq!(
        CoffI386::spec(IMAGE_REL_I386_DIR16).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Bytes(WriteKind::W16),
        }
    );
    assert_eq!(
        CoffI386::spec(IMAGE_REL_I386_REL16).unwrap(),
        Spec {
            expr: RelExpr::Pc,
            write: Write::Bytes(WriteKind::W16),
        }
    );
    assert_eq!(
        CoffI386::spec(IMAGE_REL_I386_DIR32).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Bytes(WriteKind::W32),
        }
    );
    assert_eq!(
        CoffI386::spec(IMAGE_REL_I386_REL32).unwrap(),
        Spec {
            expr: RelExpr::Pc,
            write: Write::Bytes(WriteKind::W32S),
        }
    );
    let escape32 = Spec {
        expr: RelExpr::Escape,
        write: Write::Bytes(WriteKind::W32),
    };
    assert_eq!(CoffI386::spec(IMAGE_REL_I386_DIR32NB).unwrap(), escape32);
    assert_eq!(CoffI386::spec(IMAGE_REL_I386_SECREL).unwrap(), escape32);
    assert_eq!(
        CoffI386::spec(IMAGE_REL_I386_SECTION).unwrap(),
        Spec {
            expr: RelExpr::Escape,
            write: Write::Bytes(WriteKind::W16),
        }
    );
    assert_eq!(
        CoffI386::spec(IMAGE_REL_I386_SECREL7).unwrap(),
        Spec {
            expr: RelExpr::Escape,
            write: Write::Field(Field::new(1, 0, 0, 7, Check::None)),
        }
    );
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn classifies_relocations_of_a_real_msvc_object() {
    let Some(clang) = windows_clang("--target=x86_64-pc-windows-msvc") else {
        eprintln!("skipping COFF reloc classification: clang target not found");
        return;
    };
    let path = std::env::temp_dir().join("xold_coff_reloc.o");
    if !compile(&clang, "--target=x86_64-pc-windows-msvc", &path) {
        eprintln!("skipping COFF reloc classification: clang compile failed");
        return;
    }
    let mapped = MappedFile::open(&path).expect("open COFF object");
    let obj = CoffFile::parse(mapped.bytes()).expect("parse COFF object");
    assert!(obj.is_x86_64());

    let mut seen_rel32 = false;
    let mut seen_addr32nb = false;
    let mut classified = 0u32;
    for section in obj.sections() {
        for reloc in &section.relocations {
            let spec = CoffX86_64::spec(u32::from(reloc.typ));
            assert!(
                spec.is_ok(),
                "unclassified AMD64 reloc type {}",
                reloc.typ
            );
            classified += 1;
            // REL32: a PC-relative code/data reference.
            if reloc.typ == 0x0004 {
                seen_rel32 = true;
            }
            // ADDR32NB: an RVA in .pdata/.xdata.
            if reloc.typ == 0x0003 {
                seen_addr32nb = true;
            }
        }
    }
    assert!(classified > 0, "expected relocations in the object");
    assert!(seen_rel32, "expected an IMAGE_REL_AMD64_REL32");
    assert!(seen_addr32nb, "expected an IMAGE_REL_AMD64_ADDR32NB (RVA)");
}

// --- helpers --------------------------------------------------------------

/// The source the real-object classification test compiles.
const SRC: &[u8] = b"extern int ext_var;\n\
                    extern int ext_func(int);\n\
                    int global = 42;\n\
                    int call_ext(void) { return ext_func(ext_var); }\n\
                    int *get_global(void) { return &global; }\n";

/// Returns the clang binary if it can compile for `triple`, else `None`.
fn windows_clang(triple: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let probe = std::env::temp_dir().join("xold_coff_reloc_probe.o");
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

/// Compiles `SRC` for `triple` into `out`, returning whether it succeeded.
fn compile(clang: &Path, triple: &str, out: &Path) -> bool {
    let src = std::env::temp_dir().join("xold_coff_reloc_src.c");
    let _ = fs::write(&src, SRC);
    Command::new(clang)
        .args([triple, "-c"])
        .arg(&src)
        .arg("-o")
        .arg(out)
        .status()
        .is_ok_and(|s| s.success())
}
