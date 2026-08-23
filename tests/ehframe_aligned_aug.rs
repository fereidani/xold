//! A CIE whose personality routine is encoded `DW_EH_PE_aligned` is
//! refused, not misparsed.
//!
//! The augmentation string's entries carry their data positionally, so
//! the only way past the `P` entry is to know the width of its value.
//! `DW_EH_PE_aligned` does not name one: it says the value sits at an
//! address rounded up to its own size, a width only knowable against
//! the record's runtime address, which a linker never has. The reader
//! used to skip nothing and carry on, which desynchronised the cursor
//! -- every later entry, `R` included, was read at the wrong offset, and
//! the FDE encoding came out as whatever byte happened to sit there.
//! The functions the record described then took a wrong `.eh_frame_hdr`
//! search table, and a `throw` through them terminated.
//!
//! The fixture is a hand-assembled object: one `.text` function, one
//! `.eh_frame` holding a CIE with augmentation `zPR` whose `P` encoding
//! byte is 0x50 (`DW_EH_PE_aligned`), and one FDE the `.rela.eh_frame`
//! row binds to the function so the record survives as live. The link
//! must refuse the object, naming the encoding.

use std::path::PathBuf;

use xold::{icf::IcfMode, linker::link_shared};

/// `SHF_ALLOC`: the flag marking a section as occupying memory at run time.
const SHF_ALLOC: u64 = 0x2;

/// `SHT_PROGBITS`.
const SHT_PROGBITS: u32 = 1;

/// `SHT_SYMTAB`.
const SHT_SYMTAB: u32 = 2;

/// `SHT_STRTAB`.
const SHT_STRTAB: u32 = 3;

/// `SHT_RELA`.
const SHT_RELA: u32 = 4;

/// `R_X86_64_PC32`.
const R_X86_64_PC32: u32 = 2;

/// `STB_GLOBAL << 4 | STT_FUNC`.
const GLOBAL_FUNC: u8 = (1 << 4) | 2;

/// Where the FDE's pc-begin field sits within the `.eh_frame`.
const FDE_PC_OFF: u64 = 40;

/// The section-name string table, shared by every fixture below.
const SHSTRTAB: &[u8] =
    b"\0.shstrtab\0.strtab\0.symtab\0.eh_frame\0.text\0.rela.eh_frame\0";

/// The `.eh_frame`: one CIE with augmentation `zPR` (the personality
/// encoded `DW_EH_PE_aligned`), one FDE, and the terminator.
fn eh_frame_with_aligned_personality() -> Vec<u8> {
    let mut body = Vec::new();
    body.push(1); // version
    body.extend_from_slice(b"zPR\0"); // augmentation
    body.push(1); // code_align (ULEB128)
    body.push(1); // data_align (ULEB128)
    body.push(1); // return-address register (CIE v1: one byte)
    body.push(9); // z aug-data length: P's encoding byte plus 8 value bytes
    body.push(0x50); // P encoding: DW_EH_PE_aligned
    body.extend_from_slice(&[0u8; 8]); // the personality address
    body.push(0x1b); // R encoding: pcrel | sdata4
    while body.len() % 8 != 0 {
        body.push(0); // records pad to the address size
    }
    let mut out = Vec::new();
    // The length word counts the bytes after it: the CIE_id included.
    let cie_len =
        u32::try_from(body.len() + 4).expect("the CIE fits a length word");
    out.extend_from_slice(&cie_len.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // CIE_id
    out.extend_from_slice(&body);
    // One FDE: the CIE_pointer is the offset from its own field back to
    // the CIE start, then the `R`-encoded pc-begin (relocated to the
    // function) and the address range.
    let cie_ptr = u32::try_from(out.len() + 4).expect("the CIE offset fits");
    out.extend_from_slice(&12u32.to_le_bytes()); // FDE length
    out.extend_from_slice(&cie_ptr.to_le_bytes());
    out.extend_from_slice(&0i32.to_le_bytes()); // pc-begin: pcrel, sdata4
    out.extend_from_slice(&0u32.to_le_bytes()); // address range
    out.extend_from_slice(&0u32.to_le_bytes()); // terminator
    out
}

/// A section name's offset in [`SHSTRTAB`].
fn name_off(name: &[u8]) -> u32 {
    let at = SHSTRTAB
        .windows(name.len())
        .position(|w| w == name)
        .expect("the name is in the table");
    u32::try_from(at).expect("the offset fits")
}

/// One section-header row.
#[allow(clippy::too_many_arguments)]
fn sh_row(
    name: &[u8],
    ty: u32,
    off: u64,
    size: u64,
    link: u32,
    info: u32,
    align: u64,
    flags: u64,
) -> [u8; 64] {
    let mut s = [0u8; 64];
    s[0..4].copy_from_slice(&name_off(name).to_le_bytes());
    s[4..8].copy_from_slice(&ty.to_le_bytes());
    s[8..16].copy_from_slice(&flags.to_le_bytes());
    s[24..32].copy_from_slice(&off.to_le_bytes());
    s[32..40].copy_from_slice(&size.to_le_bytes());
    s[40..44].copy_from_slice(&link.to_le_bytes());
    s[44..48].copy_from_slice(&info.to_le_bytes());
    s[48..56].copy_from_slice(&align.to_le_bytes());
    s
}

/// Where each payload landed in the file being assembled.
struct Offsets {
    shstr: u64,
    strtab: u64,
    strtab_len: u64,
    text: u64,
    eh: u64,
    eh_len: u64,
    sym: u64,
    sym_rows: usize,
    rela: u64,
}

/// The section header table, from where each payload landed.
fn section_rows(o: &Offsets) -> Vec<[u8; 64]> {
    let eh_idx = 5u32;
    let sym_idx = 6u32;
    vec![
        [0u8; 64],
        sh_row(
            b".shstrtab",
            SHT_STRTAB,
            o.shstr,
            u64::try_from(SHSTRTAB.len()).unwrap(),
            0,
            0,
            1,
            0,
        ),
        sh_row(b".strtab", SHT_STRTAB, o.strtab, o.strtab_len, 0, 0, 1, 0),
        sh_row(b".text", SHT_PROGBITS, o.text, 16, 0, 0, 1, SHF_ALLOC | 0x4),
        sh_row(
            b".eh_frame",
            SHT_PROGBITS,
            o.eh,
            o.eh_len,
            0,
            0,
            8,
            SHF_ALLOC,
        ),
        sh_row(
            b".symtab",
            SHT_SYMTAB,
            o.sym,
            u64::try_from(o.sym_rows * 24).unwrap(),
            2,
            1,
            8,
            0,
        ),
        sh_row(
            b".rela.eh_frame",
            SHT_RELA,
            o.rela,
            24,
            sym_idx,
            eh_idx,
            8,
            0,
        ),
    ]
}

/// Appends `shdrs` and stamps the header facts that describe them.
fn stamp_shdrs(out: &mut Vec<u8>, shdrs: &[[u8; 64]]) {
    let shoff = out.len() as u64;
    for r in shdrs {
        out.extend_from_slice(r);
    }
    out[40..48].copy_from_slice(&shoff.to_le_bytes());
    let shnum = u16::try_from(shdrs.len()).unwrap_or(u16::MAX);
    out[60..62].copy_from_slice(&shnum.to_le_bytes());
    out[62..64].copy_from_slice(&1u16.to_le_bytes()); // e_shstrndx
}

/// Builds the ELF64 `ET_REL` object.
fn build_object() -> Vec<u8> {
    let strtab: &[u8] = b"\0f\0";
    let eh_frame = eh_frame_with_aligned_personality();
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
    out.extend_from_slice(SHSTRTAB);
    let str_off = out.len() as u64;
    out.extend_from_slice(strtab);
    while !out.len().is_multiple_of(8) {
        out.push(0);
    }
    let text_off = out.len() as u64;
    out.extend_from_slice(&[0xccu8; 16]); // the function's bytes
    let eh_off = out.len() as u64;
    out.extend_from_slice(&eh_frame);
    while !out.len().is_multiple_of(8) {
        out.push(0);
    }
    // Two symbols: the null row, then `f` defined in `.text`.
    let mut syms = vec![[0u8; 24]; 2];
    syms[1][0..4].copy_from_slice(&1u32.to_le_bytes()); // st_name: "f"
    syms[1][4] = GLOBAL_FUNC;
    syms[1][6..8].copy_from_slice(&4u16.to_le_bytes()); // .text
    let sym_off = out.len() as u64;
    for s in &syms {
        out.extend_from_slice(s);
    }
    // One relocation: the FDE's pc-begin names `f`, which is what makes
    // the record live and the `.eh_frame_hdr` get built.
    let mut rela = [0u8; 24];
    rela[0..8].copy_from_slice(&FDE_PC_OFF.to_le_bytes());
    let r_info = (1u64 << 32) | u64::from(R_X86_64_PC32);
    rela[8..16].copy_from_slice(&r_info.to_le_bytes());
    let rela_off = out.len() as u64;
    out.extend_from_slice(&rela);
    let shdrs = section_rows(&Offsets {
        shstr: shstr_off,
        strtab: str_off,
        strtab_len: u64::try_from(strtab.len()).unwrap(),
        text: text_off,
        eh: eh_off,
        eh_len: u64::try_from(eh_frame.len()).unwrap(),
        sym: sym_off,
        sym_rows: syms.len(),
        rela: rela_off,
    });
    stamp_shdrs(&mut out, &shdrs);
    out
}

/// The link refuses a `DW_EH_PE_aligned` personality encoding.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn an_aligned_personality_encoding_is_refused() {
    let dir = std::env::temp_dir()
        .join(format!("xold_ehframealigned_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the workdir");
    let obj = dir.join("aligned.o");
    std::fs::write(&obj, build_object()).expect("write the object");
    let out: PathBuf = dir.join("libt.so");
    let res = link_shared(
        &[obj],
        &out,
        Some(b"libt.so"),
        false,
        IcfMode::None,
        false,
    );
    let Err(err) = res else {
        panic!("an aligned personality encoding must be refused, not guessed")
    };
    assert!(
        err.to_string().contains("DW_EH_PE_aligned"),
        "the error must name the encoding: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
