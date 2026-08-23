//! A shared object exports its common symbols to `.dynsym` -- sized for.
//!
//! A common is a definition the linker allocates, and the export half of
//! the writer treats it as one: `collect_exports` builds the row and the
//! exports predicate admits it in a `-shared` link. The sizing half
//! counted only `SymbolKind::Defined`, so the reservation came up one
//! row short per common and the write overran into the neighbouring
//! region, which the tighten pass catches and ends the link over. A
//! `-fcommon` global in a library could not be linked at all.
//!
//! The fixture is a hand-assembled object with one global common; the
//! link must succeed and `.dynsym` must carry the symbol's row.
//!
//! No toolchain is needed: the object is bytes and the image is read back
//! through the linker's own reader.

use std::path::Path;

use xold::{elf::ObjectFile, icf::IcfMode, linker::link_shared};

/// `SHN_COMMON`: the section index that marks a tentative definition.
const SHN_COMMON: u16 = 0xfff2;

/// Builds an ELF64 `ET_REL` object holding one global common symbol
/// named `tentative`, four bytes aligned to eight.
fn build_common() -> Vec<u8> {
    let shstrtab: &[u8] = b"\0.shstrtab\0.strtab\0.symtab\0";
    let strtab: &[u8] = b"\0tentative\0";
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
    // Two symbols: the null row, then the common.
    let mut syms = vec![[0u8; 24]; 2];
    syms[1][0..4].copy_from_slice(&1u32.to_le_bytes()); // st_name
    syms[1][4] = 1 << 4; // st_info: STB_GLOBAL, STT_NOTYPE
    syms[1][6..8].copy_from_slice(&SHN_COMMON.to_le_bytes());
    syms[1][8..16].copy_from_slice(&8u64.to_le_bytes()); // alignment
    syms[1][16..24].copy_from_slice(&4u64.to_le_bytes()); // st_size
    let sym_off = out.len() as u64;
    for s in &syms {
        out.extend_from_slice(s);
    }
    let row =
        |name: u32, sh_type: u32, off: u64, size: u64, link: u32, info: u32| {
            let mut sh = [0u8; 64];
            sh[0..4].copy_from_slice(&name.to_le_bytes());
            sh[4..8].copy_from_slice(&sh_type.to_le_bytes());
            sh[24..32].copy_from_slice(&off.to_le_bytes());
            sh[32..40].copy_from_slice(&size.to_le_bytes());
            sh[40..44].copy_from_slice(&link.to_le_bytes());
            sh[44..48].copy_from_slice(&info.to_le_bytes());
            sh
        };
    let shdrs = [
        [0u8; 64],
        row(1, 3, shstr_off, shstrtab.len() as u64, 0, 0),
        row(11, 3, str_off, strtab.len() as u64, 0, 0),
        row(19, 2, sym_off, 48, 2, 1),
    ];
    let shoff = out.len() as u64;
    for r in &shdrs {
        out.extend_from_slice(r);
    }
    out[40..48].copy_from_slice(&shoff.to_le_bytes());
    let shnum = u16::try_from(shdrs.len()).unwrap_or(u16::MAX);
    out[60..62].copy_from_slice(&shnum.to_le_bytes());
    out[62..64].copy_from_slice(&1u16.to_le_bytes()); // e_shstrndx
    out
}

/// The link succeeds and the common reaches `.dynsym`.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn a_shared_object_exports_its_commons() {
    let dir = std::env::temp_dir()
        .join(format!("xold_sharedcommon_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the workdir");
    let obj = dir.join("tentative.o");
    std::fs::write(&obj, build_common()).expect("write the object");
    let out = dir.join("libt.so");
    let res = link_shared(
        &[obj],
        &out,
        Some(b"libt.so"),
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "a common must link: {:?}", res.err());
    let names = dynsym_names(&out);
    assert!(
        names.iter().any(|n| n == b"tentative"),
        "the common's row must be in .dynsym"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The names in the image's `.dynsym`.
fn dynsym_names(out: &Path) -> Vec<Vec<u8>> {
    let bytes = std::fs::read(out).expect("read the image");
    let obj = ObjectFile::parse(&bytes).expect("parse the image");
    let table = obj
        .dynamic_symbols()
        .expect("the table reads")
        .expect("a shared object has one");
    table
        .iter()
        .map(|sym| table.name(sym).to_vec())
        .filter(|n| !n.is_empty())
        .collect()
}
