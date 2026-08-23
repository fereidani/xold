//! End-to-end check of the ELF reader against a compiled fixture.
//!
//! The fixture `tests/fixtures/min.c` is compiled to `min.o` with clang; the
//! expected counts below mirror `readelf -sW` / `readelf -SW` output so the
//! test guards against regressions in section and symbol parsing.

use std::path::Path;

use xold::{
    elf::{ObjectFile, Rela64, constants::*},
    mmap_file::MappedFile,
};

fn fixture() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/min.o");
    p
}

fn parse_fixture<'a>(path: &'a Path, bytes: &'a [u8]) -> ObjectFile<'a> {
    let _ = path;
    ObjectFile::parse(bytes).expect("fixture must parse")
}

#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn parses_header_and_sections() {
    let path = fixture();
    let mapped = MappedFile::open(&path).expect("open fixture");
    let obj = parse_fixture(&path, mapped.bytes());

    assert_eq!(obj.machine(), EM_X86_64);
    assert!(obj.is_relocatable());

    let names: Vec<&[u8]> =
        obj.sections().iter().map(|s| obj.section_name(s)).collect();
    for required in [
        b".text".as_slice(),
        b".data",
        b".bss",
        b".symtab",
        b".strtab",
    ] {
        assert!(names.contains(&required), "missing {required:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn reads_symbol_table() {
    let path = fixture();
    let mapped = MappedFile::open(&path).expect("open fixture");
    let obj = parse_fixture(&path, mapped.bytes());

    let symtab = obj
        .symbol_table()
        .expect("scan symtab")
        .expect("fixture has a symbol table");

    // One reserved null entry plus the eight symbols readelf reports.
    assert_eq!(symtab.syms.len(), 9);

    let undefined = symtab
        .iter()
        .filter(|s| s.st_shndx.get() == SHN_UNDEF && s.st_name.get() != 0)
        .count();
    assert_eq!(undefined, 1);

    let has_entry = symtab.iter().any(|s| symtab.name(s) == b"entry");
    assert!(has_entry, "expected to find the `entry` symbol");
}

#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn resolves_text_relocations() {
    let path = fixture();
    let mapped = MappedFile::open(&path).expect("open fixture");
    let obj = parse_fixture(&path, mapped.bytes());

    // `.text` is the second section (index 2) in the fixture.
    let text_index = 2u16;
    let entries: &[Rela64] = obj
        .relocations(text_index)
        .expect("scan relocs")
        .expect("`.text` has a relocation section");
    // readelf reports 5 relocations for `.rela.text`.
    assert_eq!(entries.len(), 5);
}

#[test]
fn rejects_non_elf() {
    let bytes = b"not an elf file at all";
    assert!(ObjectFile::parse(bytes).is_err());
}
