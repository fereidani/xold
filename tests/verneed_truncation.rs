//! A cut-off version table is a malformed object, not an empty one.
//!
//! `.gnu.version_r` and `.gnu.version_d` are chains: each record's
//! `vn_next`/`vd_next` gives the byte offset of its successor, and a zero
//! ends the chain. The reader used to stop at the first record that no
//! longer fit the payload, keeping whatever it had collected -- so a file
//! whose chain ran past its bytes produced a partial table that resolved
//! some version references and silently mislabelled the rest. lld, reading
//! the same sections through `object`, reports the short read.
//!
//! The fixtures are hand-assembled ELFs (a header, a name table, and the
//! version sections under test). A record cut off mid-bytes and a chain
//! that promises a successor past the payload are both rejected; a
//! complete chain still parses.
//!
//! `u16::MAX` bounds the chain walk, so a file cannot spin the reader on
//! a cyclic chain either: it is reported as unterminated.

use xold::{elf::ObjectFile, input::AlignedBytes};

/// Builds a minimal ELF64 `ET_REL` object whose `.gnu.version_r` holds
/// `verneed` and whose `.gnu.version_d` holds `verdef`. An empty payload
/// means the section is left out entirely.
fn build_elf(verneed: &[u8], verdef: &[u8]) -> AlignedBytes {
    let names = b"\0.shstrtab\0.gnu.version_r\0.gnu.version_d\0";
    let mut out = vec![0u8; 64];
    out[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    out[4] = 2; // ELFCLASS64
    out[5] = 1; // ELFDATA2LSB
    out[6] = 1; // EV_CURRENT
    out[16..18].copy_from_slice(&1u16.to_le_bytes()); // ET_REL
    out[18..20].copy_from_slice(&62u16.to_le_bytes()); // EM_X86_64
    out[20] = 1; // e_version
    out[58..60].copy_from_slice(&64u16.to_le_bytes()); // e_shentsize
    let strtab_off = out.len() as u64;
    out.extend_from_slice(names);
    while !out.len().is_multiple_of(8) {
        out.push(0);
    }
    let mut shdrs: Vec<[u8; 64]> = vec![[0; 64]];
    // Section 1: .shstrtab.
    let mut sh = [0u8; 64];
    sh[0..4].copy_from_slice(&1u32.to_le_bytes()); // sh_name
    sh[4..8].copy_from_slice(&3u32.to_le_bytes()); // SHT_STRTAB
    shdrs.push(sh);
    let add_version_section = |out: &mut Vec<u8>,
                               shdrs: &mut Vec<[u8; 64]>,
                               name: u32,
                               sh_type: u32,
                               payload: &[u8]| {
        let offset = out.len() as u64;
        out.extend_from_slice(payload);
        while !out.len().is_multiple_of(8) {
            out.push(0);
        }
        let mut sh = [0u8; 64];
        sh[0..4].copy_from_slice(&name.to_le_bytes());
        sh[4..8].copy_from_slice(&sh_type.to_le_bytes());
        sh[24..32].copy_from_slice(&offset.to_le_bytes()); // sh_offset
        sh[32..40].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        sh[40..44].copy_from_slice(&1u32.to_le_bytes()); // sh_link: names
        sh[48..56].copy_from_slice(&4u64.to_le_bytes()); // sh_addralign
        shdrs.push(sh);
    };
    if !verneed.is_empty() {
        add_version_section(&mut out, &mut shdrs, 11, 0x6fff_fffe, verneed);
    }
    if !verdef.is_empty() {
        add_version_section(&mut out, &mut shdrs, 25, 0x6fff_fffd, verdef);
    }
    // Fix the .shstrtab offset now that payloads cannot move it.
    let strtab = shdrs.get_mut(1).expect("shstrtab row");
    strtab[24..32].copy_from_slice(&strtab_off.to_le_bytes());
    strtab[32..40].copy_from_slice(&(names.len() as u64).to_le_bytes());
    let shoff = out.len() as u64;
    for row in &shdrs {
        out.extend_from_slice(row);
    }
    let shnum = u16::try_from(shdrs.len()).unwrap_or(u16::MAX);
    out[40..48].copy_from_slice(&shoff.to_le_bytes());
    out[60..62].copy_from_slice(&shnum.to_le_bytes());
    out[62..64].copy_from_slice(&1u16.to_le_bytes()); // e_shstrndx
    AlignedBytes::new(&out)
}

/// One well-formed `Verneed` + `Vernaux` pair: the chain ends here.
fn intact_verneed() -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&1u16.to_le_bytes()); // vn_version
    v.extend_from_slice(&1u16.to_le_bytes()); // vn_cnt
    v.extend_from_slice(&0u32.to_le_bytes()); // vn_file
    v.extend_from_slice(&16u32.to_le_bytes()); // vn_aux
    v.extend_from_slice(&0u32.to_le_bytes()); // vn_next: end
    v.extend_from_slice(&0u32.to_le_bytes()); // vna_hash
    v.extend_from_slice(&0u16.to_le_bytes()); // vna_flags
    v.extend_from_slice(&2u16.to_le_bytes()); // vna_other
    v.extend_from_slice(&0u32.to_le_bytes()); // vna_name
    v.extend_from_slice(&0u32.to_le_bytes()); // vna_next: end
    v
}

/// A record whose bytes stop partway through the 16-byte header.
#[test]
fn a_record_cut_off_midway_is_rejected() {
    let elf = build_elf(&intact_verneed()[..10], &[]);
    let obj =
        ObjectFile::parse(elf.bytes()).expect("the ELF structure is sound");
    let err = version_error(&obj, "a cut-off record is malformed, not absent");
    assert!(
        err.to_string().contains("truncated .gnu.version_r"),
        "the error must name the truncated table: {err}"
    );
}

/// A chain link that promises a successor past the payload is rejected.
#[test]
fn a_dangling_chain_link_is_rejected() {
    let mut v = intact_verneed();
    // The one record present claims a sibling at +64; the payload holds 32.
    v[8..12].copy_from_slice(&64u32.to_le_bytes());
    let elf = build_elf(&v, &[]);
    let obj =
        ObjectFile::parse(elf.bytes()).expect("the ELF structure is sound");
    let err = version_error(&obj, "a promised record must be present");
    assert!(
        err.to_string().contains("truncated .gnu.version_r"),
        "the error must name the truncated table: {err}"
    );
}

/// The same rejection for `.gnu.version_d`, whose records chain the same way.
#[test]
fn a_cut_off_verdef_is_rejected() {
    let mut d = Vec::new();
    d.extend_from_slice(&1u16.to_le_bytes()); // vd_version
    d.extend_from_slice(&0u16.to_le_bytes()); // vd_flags
    d.extend_from_slice(&2u16.to_le_bytes()); // vd_ndx
    d.extend_from_slice(&1u16.to_le_bytes()); // vd_cnt
    d.extend_from_slice(&0u32.to_le_bytes()); // vd_hash
    d.extend_from_slice(&20u32.to_le_bytes()); // vd_aux
    d.extend_from_slice(&0u32.to_le_bytes()); // vd_next: end
    d.extend_from_slice(&0u32.to_le_bytes()); // vda_name
    d.extend_from_slice(&0u32.to_le_bytes()); // vda_next: end
    let elf = build_elf(&[], &d[..12]);
    let obj =
        ObjectFile::parse(elf.bytes()).expect("the ELF structure is sound");
    let err = version_error(&obj, "a cut-off verdef record is malformed");
    assert!(
        err.to_string().contains("truncated .gnu.version_d"),
        "the error must name the truncated table: {err}"
    );
}

/// A complete chain still parses: one need with one auxiliary entry.
#[test]
fn an_intact_chain_still_parses() {
    let elf = build_elf(&intact_verneed(), &[]);
    let obj =
        ObjectFile::parse(elf.bytes()).expect("the ELF structure is sound");
    let table = obj.version_table().expect("a well-formed table must parse");
    assert_eq!(table.needs.len(), 1, "one Verneed record was present");
    assert_eq!(
        table.needs[0].aux.len(),
        1,
        "one Vernaux record followed it"
    );
}

/// The `version_table` error, failing the test when the table parsed.
fn version_error(obj: &ObjectFile<'_>, why: &str) -> xold::error::Error {
    match obj.version_table() {
        Err(e) => e,
        Ok(_) => panic!("{why}"),
    }
}
