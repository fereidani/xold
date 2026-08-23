//! A section-header-free ELF parses instead of erroring.
//!
//! The gABI lets a file carry no section table at all: `e_shoff`,
//! `e_shnum` and `e_shstrndx` all zero, nothing left for tools to walk.
//! `strip`ped images and synthetic inputs look like this, and the system
//! tools read one without complaint.
//!
//! Two rejections stacked on such a file: the section-header re-view asked
//! `bytemuck` to test the alignment of a pointer with no byte behind it,
//! and the name-table lookup asked `object` for `e_shstrndx` -- which
//! refuses the reserved zero before learning the file has no sections to
//! name at all. Both answers called the file malformed; it was not.
//!
//! The fixture builds the 64-byte header by hand and asks the reader to
//! parse it: the parse succeeds, the section list is empty, and the symbol
//! table is absent rather than malformed.

use xold::{elf::ObjectFile, input::AlignedBytes};

/// The 64-byte little-endian ELF64 header of an `ET_REL` x86-64 object
/// with no section table.
fn bare_header() -> [u8; 64] {
    let mut h = [0u8; 64];
    h[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    h[4] = 2; // ELFCLASS64
    h[5] = 1; // ELFDATA2LSB
    h[6] = 1; // EV_CURRENT
    h[7] = 3; // EI_OSABI_LINUX
    h[16..18].copy_from_slice(&1u16.to_le_bytes()); // ET_REL
    h[18..20].copy_from_slice(&62u16.to_le_bytes()); // EM_X86_64
    h[20] = 1; // e_version
    h[52..56].copy_from_slice(&1u32.to_le_bytes()); // e_flags
    h[58..60].copy_from_slice(&64u16.to_le_bytes()); // e_shentsize
    h
}

/// The parse succeeds and describes a relocatable object with no sections.
#[test]
fn a_section_header_free_elf_parses() {
    let h = AlignedBytes::new(&bare_header());
    let obj = ObjectFile::parse(h.bytes())
        .expect("a header-only ELF is well formed and must parse");
    assert_eq!(obj.sections().len(), 0, "the file states no sections");
    assert!(obj.is_relocatable(), "e_type is ET_REL");
    let symtab = obj
        .symbol_table()
        .expect("the absence of a table is not a malformed one");
    assert!(symtab.is_none(), "no section means no symbol table");
}
