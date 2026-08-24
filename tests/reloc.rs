//! The declarative relocation table and its arch-neutral drivers.
//!
//! The table is checked both in isolation (each raw type maps to the expected
//! [`Spec`]) and against the relocations a real compiler emits: `min.o`'s
//! `.text` exercises `R_X86_64_PC32`, `R_X86_64_PLT32` and
//! `R_X86_64_REX_GOTPCRELX`, which the scan must route to no allocation, a PLT
//! entry and a GOT entry respectively.

use std::path::PathBuf;

use common::elf_fixture;
use rustc_hash::FxHashMap;
use xold::{
    elf::ObjectFile,
    mmap_file::MappedFile,
    reloc::{
        Arch, Needs, RelExpr, Resolver, Spec, Write, WriteKind, apply, scan,
        x86_64::{
            R_X86_64_32S, R_X86_64_64, R_X86_64_DTPMOD64, R_X86_64_NONE,
            R_X86_64_PC32, R_X86_64_PLT32, R_X86_64_REX_GOTPCRELX,
            R_X86_64_TLSGD, R_X86_64_TLSLD, X86_64,
        },
    },
    symbol::SymbolId,
};

mod common;

/// A resolver returning fixed addresses for every symbol, for value checks.
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

fn fixture(name: &str) -> PathBuf {
    elf_fixture(name)
}

/// The section index of `.text` in `obj`.
fn text_shndx(obj: &ObjectFile) -> u16 {
    let pos = obj
        .sections()
        .iter()
        .position(|s| obj.section_name(s) == b".text")
        .expect(".text section present");
    u16::try_from(pos).expect("section index fits")
}

#[test]
fn table_maps_common_x86_64_types() {
    assert_eq!(
        X86_64::spec(R_X86_64_PC32).unwrap(),
        Spec {
            expr: RelExpr::Pc,
            write: Write::Bytes(WriteKind::W32S)
        }
    );
    assert_eq!(
        X86_64::spec(R_X86_64_PLT32).unwrap(),
        Spec {
            expr: RelExpr::PltPc,
            write: Write::Bytes(WriteKind::W32S)
        }
    );
    assert_eq!(
        X86_64::spec(R_X86_64_REX_GOTPCRELX).unwrap(),
        Spec {
            expr: RelExpr::GotPc,
            write: Write::Bytes(WriteKind::W32S)
        }
    );
    assert_eq!(
        X86_64::spec(R_X86_64_32S).unwrap(),
        Spec {
            expr: RelExpr::Abs,
            write: Write::Bytes(WriteKind::W32S)
        }
    );
    // The general-dynamic slot is the 4-byte displacement of the `lea` that
    // opens the pair. It names the module entry a shared object hands to
    // `__tls_get_addr`; an executable never computes it, because the whole
    // pair is rewritten before the apply path is reached.
    assert_eq!(
        X86_64::spec(R_X86_64_TLSGD).unwrap(),
        Spec {
            expr: RelExpr::TlsGotPc,
            write: Write::Bytes(WriteKind::W32S)
        }
    );
}

#[test]
fn unknown_type_is_unsupported() {
    assert!(matches!(
        scan::<X86_64>(99).unwrap_err(),
        xold::Error::UnsupportedReloc(99)
    ));
}

#[test]
fn absolute_addend_only_when_symbol_is_null() {
    let fixed = Fixed {
        symbol: 0x1000,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let value = RelExpr::Abs.compute(None, 42, 0, &fixed).unwrap();
    assert_eq!(value, 42);
}

#[test]
fn pc_relative_subtracts_the_place() {
    let fixed = Fixed {
        symbol: 0x1000,
        got: 0x3000,
        plt: 0x1000,
        got_base: 0x3000,
    };
    // S + A - P = 0x1000 - 4 - 0x800 = 0x7fc.
    let value = RelExpr::Pc
        .compute(Some(SymbolId(1)), -4, 0x800, &fixed)
        .unwrap();
    assert_eq!(value, 0x7fc);
    // A static PLT equals the symbol, so PltPc collapses to Pc.
    let value = RelExpr::PltPc
        .compute(Some(SymbolId(1)), -4, 0x800, &fixed)
        .unwrap();
    assert_eq!(value, 0x7fc);
    // GOT[sym] + A - P = 0x3000 - 4 - 0x800 = 0x27fc.
    let value = RelExpr::GotPc
        .compute(Some(SymbolId(1)), -4, 0x800, &fixed)
        .unwrap();
    assert_eq!(value, 0x27fc);
}

#[test]
fn apply_writes_little_endian_and_patches_the_slot() {
    let fixed = Fixed {
        symbol: 0x1000,
        got: 0x3000,
        plt: 0x1000,
        got_base: 0x3000,
    };
    let mut slot = [0u8; 4];
    apply::<X86_64, _>(
        R_X86_64_PC32,
        Some(SymbolId(1)),
        -4,
        0x800,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0xfc, 0x07, 0x00, 0x00]);
}

#[test]
fn apply_writes_full_width_absolute() {
    let fixed = Fixed {
        symbol: 0x1000,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 8];
    apply::<X86_64, _>(
        R_X86_64_64,
        Some(SymbolId(1)),
        0x10,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0x10, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
}

#[test]
fn signed_field_overflow_is_an_error() {
    // 0x8000_0000 is one past i32::MAX, so the signed 32-bit field overflows.
    let fixed = Fixed {
        symbol: 0x8000_0000,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 4];
    let err = apply::<X86_64, _>(
        R_X86_64_32S,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(matches!(err, xold::Error::RelocOverflow(R_X86_64_32S)));
}

#[test]
fn none_relocation_leaves_the_slot_untouched() {
    let fixed = Fixed {
        symbol: 0,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0xab; 8];
    apply::<X86_64, _>(
        R_X86_64_NONE,
        Some(SymbolId(1)),
        -4,
        0x800,
        &fixed,
        &mut slot,
    )
    .unwrap();
    assert_eq!(slot, [0xab; 8]);
}

#[test]
fn escape_relocation_is_routed_to_the_hatch() {
    let fixed = Fixed {
        symbol: 0,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 8];
    // The module id of a thread-local is the loader's to supply, so an input
    // object naming it has no value this pass can compute.
    let err = apply::<X86_64, _>(
        R_X86_64_DTPMOD64,
        Some(SymbolId(1)),
        0,
        0,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        xold::Error::UnsupportedReloc(R_X86_64_DTPMOD64)
    ));
}

/// A local-dynamic reference with no module entry allocated is one the writer
/// was meant to rewrite. Storing an offset from zero would corrupt the
/// sequence silently, so the apply path refuses it.
#[test]
fn a_local_dynamic_reference_without_a_module_entry_is_refused() {
    let fixed = Fixed {
        symbol: 0,
        got: 0,
        plt: 0,
        got_base: 0,
    };
    let mut slot = [0u8; 4];
    let err = apply::<X86_64, _>(
        R_X86_64_TLSLD,
        Some(SymbolId(1)),
        -4,
        0x1000,
        &fixed,
        &mut slot,
    )
    .unwrap_err();
    assert!(matches!(err, xold::Error::UnsupportedReloc(_)));
}

#[test]
fn scan_routes_types_to_their_allocations() {
    assert_eq!(scan::<X86_64>(R_X86_64_PC32).unwrap(), Needs::NONE);
    assert_eq!(
        scan::<X86_64>(R_X86_64_PLT32).unwrap(),
        Needs {
            plt: true,
            ..Needs::NONE
        }
    );
    assert_eq!(
        scan::<X86_64>(R_X86_64_REX_GOTPCRELX).unwrap(),
        Needs {
            got: true,
            ..Needs::NONE
        }
    );
}

#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn classifies_relocations_from_a_real_object() {
    let min = MappedFile::open(&fixture("min.o")).unwrap();
    let obj = ObjectFile::parse(min.bytes()).unwrap();
    let target = text_shndx(&obj);
    let entries = obj
        .relocations(target)
        .unwrap()
        .expect(".text has a relocation section");

    // The classification depends only on the relocation type, so the first
    // answer recorded for a type stands for every entry that shares it.
    let mut needs_by_type: FxHashMap<u32, Needs> = FxHashMap::default();
    for entry in entries {
        let r_type = entry.r_type();
        let needs = scan::<X86_64>(r_type).unwrap();
        needs_by_type.entry(r_type).or_insert(needs);
    }

    assert_eq!(
        needs_by_type.get(&R_X86_64_PC32).copied().unwrap(),
        Needs::NONE
    );
    assert_eq!(
        needs_by_type.get(&R_X86_64_REX_GOTPCRELX).copied().unwrap(),
        Needs {
            got: true,
            ..Needs::NONE
        }
    );
    assert_eq!(
        needs_by_type.get(&R_X86_64_PLT32).copied().unwrap(),
        Needs {
            plt: true,
            ..Needs::NONE
        }
    );
}
