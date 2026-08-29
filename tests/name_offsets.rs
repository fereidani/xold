//! Name offsets past their string table are rejected, not read empty.
//!
//! `sh_name` and `st_name` are offsets into `.shstrtab` and `.strtab`.
//! The reader borrowed the name at the offset unchecked, and an offset
//! past the table's end borrowed nothing: the section or symbol read as
//! the empty name. A section dropped out of every name-based rule
//! silently, and two broken symbols merged under "" -- no diagnostic,
//! no refusal, just a differently-shaped link.
//!
//! lld resolves both through llvm's `ELFFile::getString`, which returns
//! an error for an out-of-range offset; the local checks adopt the same
//! answer. Zero stays legal in both columns: it is the unnamed slot
//! every section table leads with and every local symbol may use.
//!
//! The fixtures are hand-assembled ELFs whose one name column points
//! past its table. The parse (for `sh_name`) and the symbol-table read
//! (for `st_name`) both refuse; a well-formed control parses.

use xold::{elf::ObjectFile, input::AlignedBytes};

/// Builds a minimal ELF64 `ET_REL` object with a `.symtab` whose single
/// symbol carries `st_name`, and `.strtab`/`.shstrtab` tables holding
/// `strtab`/`shstrtab` bytes. `section_name_off` overrides the `.symtab`
/// row's `sh_name` so the caller can corrupt it.
fn build_elf(sym_name: u32, section_name_off: u32) -> AlignedBytes {
    let shstrtab: &[u8] = b"\0.shstrtab\0.strtab\0.symtab\0";
    let strtab: &[u8] = b"\0probe\0";
    let mut out = vec![0u8; 64];
    out[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    out[4] = 2; // ELFCLASS64
    out[5] = 1; // ELFDATA2LSB
    out[6] = 1; // EV_CURRENT
    out[16..18].copy_from_slice(&1u16.to_le_bytes()); // ET_REL
    out[18..20].copy_from_slice(&62u16.to_le_bytes()); // EM_X86_64
    out[20] = 1; // e_version
    out[58..60].copy_from_slice(&64u16.to_le_bytes()); // e_shentsize
    let shstr_off = out.len() as u64;
    out.extend_from_slice(shstrtab);
    let str_off = out.len() as u64;
    out.extend_from_slice(strtab);
    while !out.len().is_multiple_of(8) {
        out.push(0);
    }
    // One 24-byte symbol: named `sym_name`, defined, global, section 4.
    let mut sym = [0u8; 24];
    sym[0..4].copy_from_slice(&sym_name.to_le_bytes());
    sym[4] = 1 << 4; // st_info: STB_GLOBAL, STT_NOTYPE
    sym[6..8].copy_from_slice(&4u16.to_le_bytes()); // st_shndx
    let sym_off = out.len() as u64;
    out.extend_from_slice(&sym);
    let mut shdrs: Vec<[u8; 64]> = vec![[0; 64]];
    let row = |name: u32, sh_type: u32, off: u64, size: u64, link: u32| {
        let mut sh = [0u8; 64];
        sh[0..4].copy_from_slice(&name.to_le_bytes());
        sh[4..8].copy_from_slice(&sh_type.to_le_bytes());
        sh[24..32].copy_from_slice(&off.to_le_bytes());
        sh[32..40].copy_from_slice(&size.to_le_bytes());
        sh[40..44].copy_from_slice(&link.to_le_bytes());
        sh
    };
    shdrs.push(row(1, 3, shstr_off, shstrtab.len() as u64, 0));
    shdrs.push(row(11, 3, str_off, strtab.len() as u64, 0));
    shdrs.push(row(19, 2, sym_off, sym.len() as u64, 2));
    shdrs[3][0..4].copy_from_slice(&section_name_off.to_le_bytes());
    let shoff = out.len() as u64;
    for r in &shdrs {
        out.extend_from_slice(r);
    }
    out[40..48].copy_from_slice(&shoff.to_le_bytes());
    let shnum = u16::try_from(shdrs.len()).unwrap_or(u16::MAX);
    out[60..62].copy_from_slice(&shnum.to_le_bytes());
    out[62..64].copy_from_slice(&1u16.to_le_bytes()); // e_shstrndx
    AlignedBytes::new(&out)
}

/// An `sh_name` past `.shstrtab` is refused at the parse.
#[test]
fn a_section_name_offset_past_the_table_is_rejected() {
    let elf = build_elf(1, 999);
    let Err(err) = ObjectFile::parse(elf.bytes()) else {
        panic!("the name column points past its table")
    };
    assert!(
        err.to_string().contains("section name offset"),
        "the error must name the broken column: {err}"
    );
}

/// An `st_name` past `.strtab` is refused at the parse, as a section name is.
///
/// Both symbol tables are located and their name columns checked once, where
/// the file is parsed, so every later reader takes a table already known
/// sound. That is what keeps the check off the accessors, which each pass
/// that reads a symbol calls.
#[test]
fn a_symbol_name_offset_past_the_table_is_rejected() {
    let Err(err) = ObjectFile::parse(build_elf(999, 19).bytes()) else {
        panic!("the name column points past its table")
    };
    assert!(
        err.to_string().contains("symbol name offset"),
        "the error must name the broken column: {err}"
    );
}

/// The same object with in-range columns parses and names its symbol.
#[test]
fn in_range_name_columns_parse() {
    let elf = build_elf(1, 19);
    let obj =
        ObjectFile::parse(elf.bytes()).expect("in-range columns are sound");
    let table = obj
        .symbol_table()
        .expect("the symbol table reads")
        .expect("the object has one");
    assert_eq!(
        table.syms[0].st_name.get(),
        1,
        "the named symbol keeps its offset"
    );
    assert_eq!(table.name(&table.syms[0]), b"probe");
}
