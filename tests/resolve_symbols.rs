//! Cross-file symbol resolution.
//!
//! Each fixture is compiled from `tests/fixtures/*.c`; the assertions exercise
//! the ELF precedence rules: a strong definition resolves an undefined
//! reference, beats a weak one, and two strong definitions collide.

use std::path::{Path, PathBuf};

use common::elf_fixture;
use xold::{
    archive::Archive,
    input::Input,
    linker::{Context, link},
    mmap_file::MappedFile,
    symbol::SymbolKind,
};

mod common;

fn fixture(name: &str) -> PathBuf {
    elf_fixture(name)
}

/// Finds a resolved symbol by exact name.
fn resolved<'a>(ctx: &'a Context<'a>, name: &[u8]) -> Option<&'a SymbolKind> {
    ctx.symbols
        .entries()
        .find(|(n, _)| *n == name)
        .map(|(_, s)| &s.kind)
}

#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn resolves_undefined_against_definition() {
    let min = MappedFile::open(&fixture("min.o")).unwrap();
    let ext = MappedFile::open(&fixture("ext.o")).unwrap();
    let mut ctx = Context::new();
    ctx.add_file(&fixture("min.o"), min.bytes()).unwrap();
    ctx.add_file(&fixture("ext.o"), ext.bytes()).unwrap();
    ctx.resolve_symbols().unwrap();

    let SymbolKind::Defined(def) = resolved(&ctx, b"external_func").unwrap()
    else {
        panic!("expected a definition for external_func");
    };
    assert!(!def.weak, "the strong definition must win");
}

#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn strong_overrides_weak_definition() {
    let min = MappedFile::open(&fixture("min.o")).unwrap();
    let weak = MappedFile::open(&fixture("weak_ext.o")).unwrap();
    let ext = MappedFile::open(&fixture("ext.o")).unwrap();
    let mut ctx = Context::new();
    ctx.add_file(&fixture("min.o"), min.bytes()).unwrap();
    ctx.add_file(&fixture("weak_ext.o"), weak.bytes()).unwrap();
    ctx.add_file(&fixture("ext.o"), ext.bytes()).unwrap();
    ctx.resolve_symbols().unwrap();

    let SymbolKind::Defined(def) = resolved(&ctx, b"external_func").unwrap()
    else {
        panic!("expected a definition for external_func");
    };
    assert!(!def.weak, "strong definition must outrank the weak one");
}

#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn weak_definition_resolves_undefined_reference() {
    let min = MappedFile::open(&fixture("min.o")).unwrap();
    let weak = MappedFile::open(&fixture("weak_ext.o")).unwrap();
    let mut ctx = Context::new();
    ctx.add_file(&fixture("min.o"), min.bytes()).unwrap();
    ctx.add_file(&fixture("weak_ext.o"), weak.bytes()).unwrap();
    ctx.resolve_symbols().unwrap();

    let SymbolKind::Defined(def) = resolved(&ctx, b"external_func").unwrap()
    else {
        panic!("a weak definition should still resolve the reference");
    };
    assert!(def.weak, "with only a weak definition present it is kept");
}

#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn unresolved_reference_stays_undefined() {
    let min = MappedFile::open(&fixture("min.o")).unwrap();
    let mut ctx = Context::new();
    ctx.add_file(&fixture("min.o"), min.bytes()).unwrap();
    ctx.resolve_symbols().unwrap();

    let kind = resolved(&ctx, b"external_func").unwrap();
    assert!(matches!(kind, SymbolKind::Undefined { .. }));
}

#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn duplicate_strong_definition_is_an_error() {
    let paths = vec![fixture("ext.o"), fixture("ext.o")];
    match link(&Input::from_paths(&paths)) {
        // Both definers are named now: the error carries the two inputs, so
        // a reader does not have to go looking for them.
        Err(xold::Error::DuplicateSymbol(d)) => {
            assert_eq!(d.name, "external_func");
            assert!(
                d.first.ends_with("ext.o") && d.second.ends_with("ext.o"),
                "the message names both inputs, got {:?} and {:?}",
                d.first,
                d.second
            );
        }
        other => panic!("expected a duplicate symbol error, got {other:?}"),
    }
}

#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn archive_index_exposes_member_symbols() {
    let lib = MappedFile::open(&fixture("libext.a")).unwrap();
    let archive = Archive::parse(lib.bytes(), Path::new("lib.a")).unwrap();
    assert_eq!(archive.lookup(b"external_func"), Some(0x5a));
    assert!(archive.lookup(b"does_not_exist").is_none());
}

#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn archive_member_is_pulled_for_undefined_symbol() {
    let min = MappedFile::open(&fixture("min.o")).unwrap();
    let lib = MappedFile::open(&fixture("libext.a")).unwrap();
    let mut ctx = Context::new();
    ctx.add_file(&fixture("min.o"), min.bytes()).unwrap();
    let archive = Archive::parse(lib.bytes(), Path::new("lib.a")).unwrap();
    ctx.resolve_symbols().unwrap();
    ctx.pull_archives(&[archive]).unwrap();

    let SymbolKind::Defined(_) = resolved(&ctx, b"external_func").unwrap()
    else {
        panic!("the archive member should have resolved external_func");
    };
}

#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn link_pulls_archive_member_end_to_end() {
    let paths = vec![fixture("min.o"), fixture("libext.a")];
    link(&Input::from_paths(&paths))
        .expect("archive pull must succeed end to end");
}

/// A tentative definition outranks a weak one, in either input order.
///
/// This is the one precedence rule that is not self-evident, and the reference
/// linkers disagree about it: `ld.bfd` and `ld.lld` give the storage to the
/// tentative definition, `mold` to the weak one. xold follows the first two;
/// see `symbol::Class::rank`, which records the evidence.
///
/// Both orders are checked because a precedence rule that only holds one way
/// round is not a precedence rule: it is the first-seen symbol winning.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn common_outranks_weak_definition() {
    for order in [
        ["common_obj.o", "weak_obj.o"],
        ["weak_obj.o", "common_obj.o"],
    ] {
        let first = MappedFile::open(&fixture(order[0])).unwrap();
        let second = MappedFile::open(&fixture(order[1])).unwrap();
        let mut ctx = Context::new();
        ctx.add_file(&fixture(order[0]), first.bytes()).unwrap();
        ctx.add_file(&fixture(order[1]), second.bytes()).unwrap();
        ctx.resolve_symbols().unwrap();

        let kind = resolved(&ctx, b"shared_obj")
            .expect("both inputs mention the symbol");
        let SymbolKind::Common { size, .. } = kind else {
            panic!(
                "{order:?}: the tentative definition must win, got {kind:?}"
            );
        };
        assert_eq!(*size, 4, "{order:?}: the tentative definition's size");
    }
}

/// A strong definition still outranks a tentative one, in either input order,
/// which is what keeps the rule above from reading as "the tentative
/// definition always wins".
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn strong_definition_outranks_common() {
    for order in [
        ["common_obj.o", "strong_obj.o"],
        ["strong_obj.o", "common_obj.o"],
    ] {
        let first = MappedFile::open(&fixture(order[0])).unwrap();
        let second = MappedFile::open(&fixture(order[1])).unwrap();
        let mut ctx = Context::new();
        ctx.add_file(&fixture(order[0]), first.bytes()).unwrap();
        ctx.add_file(&fixture(order[1]), second.bytes()).unwrap();
        ctx.resolve_symbols().unwrap();

        let kind = resolved(&ctx, b"shared_obj")
            .expect("both inputs mention the symbol");
        let SymbolKind::Defined(def) = kind else {
            panic!("{order:?}: the strong definition must win, got {kind:?}");
        };
        assert!(!def.weak, "{order:?}: and it is the strong one");
    }
}
