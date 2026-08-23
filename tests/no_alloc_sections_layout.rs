//! A link whose inputs contribute no allocated section still places its
//! synthetic regions past the ELF headers.
//!
//! Placement walks the read-execute segment with a cursor seeded at the
//! header table, but the segment's end -- the address the read-only
//! dynamic tables start at -- was computed as the maximum over the placed
//! regions. Every unplaced region reports offset zero, and a `-shared`
//! link over an object with no `SHF_ALLOC` section leaves all of them
//! unplaced: the maximum came out zero and `.gnu.hash`, `.hash`,
//! `.dynsym` and `.dynstr` were laid over the ELF header and the program
//! header table themselves. The link reported success and produced an
//! image whose own loader tables were the bytes of its headers.
//!
//! The fixture is a hand-assembled object holding one common symbol and
//! nothing else -- no allocated byte anywhere. The link must succeed and
//! every allocated section in the image must start past the header table.

use std::path::PathBuf;

use xold::{elf::ObjectFile, icf::IcfMode, linker::link_shared};

/// `SHN_COMMON`: the section index that marks a tentative definition.
const SHN_COMMON: u16 = 0xfff2;

/// `SHF_ALLOC`: the flag marking a section as occupying memory at run time.
const SHF_ALLOC: u64 = 0x2;

/// Builds an ELF64 `ET_REL` object holding one global common symbol named
/// `tentative` and no allocated section.
fn build_common_only() -> Vec<u8> {
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

/// One past the image's program header table: the first file offset a
/// placed region may start at.
fn headers_end(image: &[u8]) -> u64 {
    let phoff = u64::from_le_bytes(image[32..40].try_into().expect("e_phoff"));
    let phentsize =
        u16::from_le_bytes(image[54..56].try_into().expect("e_phentsize"));
    let phnum = u16::from_le_bytes(image[56..58].try_into().expect("e_phnum"));
    phoff + u64::from(phentsize) * u64::from(phnum)
}

/// The shared object built from the alloc-free input places every
/// allocated section past the header table.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn no_alloc_sections_still_places_past_the_headers() {
    let dir = std::env::temp_dir()
        .join(format!("xold_noalloc_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the workdir");
    let obj = dir.join("tentative.o");
    std::fs::write(&obj, build_common_only()).expect("write the object");
    let out: PathBuf = dir.join("libt.so");
    let res = link_shared(
        &[obj],
        &out,
        Some(b"libt.so"),
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the link must succeed: {:?}", res.err());
    let image = std::fs::read(&out).expect("read the image");
    let end = headers_end(&image);
    let parsed = ObjectFile::parse(&image).expect("parse the image");
    for shdr in parsed.sections() {
        if shdr.sh_flags.get() & SHF_ALLOC == 0 {
            continue;
        }
        assert!(
            shdr.sh_offset.get() >= end,
            "allocated section {} starts at {}, inside the header table \
             ending at {}",
            String::from_utf8_lossy(parsed.section_name(shdr)),
            shdr.sh_offset.get(),
            end,
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
