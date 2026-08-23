//! A common symbol's alignment must be a power of two, like any other.
//!
//! A common carries no section header, so its alignment rides in
//! `st_value`. Every placement cursor rounds with it as a mask, which is
//! the same contract `sh_addralign` has -- and `sh_addralign` is checked
//! where the file enters. A non-power-of-two slipped through and rounded
//! to the power below (a common "aligned" to 3 landed on a multiple of
//! 4), and a value near `u64::MAX` was an arithmetic hazard wearing the
//! shape of a request.
//!
//! The fixtures are hand-assembled objects with one common symbol whose
//! `st_value` the test sets. A power-of-two alignment links; three and
//! 2^31 are both refused, naming the column.

use std::path::Path;

use xold::{icf::IcfMode, linker::link_shared};

/// `SHN_COMMON`: the section index that marks a tentative definition.
const SHN_COMMON: u16 = 0xfff2;

/// Builds an ELF64 `ET_REL` object holding one common symbol with the
/// given `st_value` (its alignment) and size.
fn build_common(st_value: u64) -> Vec<u8> {
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
    syms[1][8..16].copy_from_slice(&st_value.to_le_bytes());
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

/// Writes `bytes` to `dir/tentative.o` and returns the path.
fn object(dir: &Path, bytes: &[u8]) -> std::path::PathBuf {
    let path = dir.join("tentative.o");
    std::fs::write(&path, bytes).expect("write the object");
    path
}

/// A power-of-two alignment links: the common lands in `.bss`.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn a_power_of_two_common_alignment_links() {
    let dir = std::env::temp_dir()
        .join(format!("xold_commonalign_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the workdir");
    let obj = object(&dir, &build_common(8));
    let out = dir.join("libt.so");
    let res = link_shared(
        &[obj],
        &out,
        Some(b"libt.so"),
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "an aligned common must link: {:?}", res.err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// An alignment that is not a power of two is refused.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn a_non_power_of_two_common_alignment_is_refused() {
    let dir = std::env::temp_dir()
        .join(format!("xold_commonalign2_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the workdir");
    let obj = object(&dir, &build_common(3));
    let out = dir.join("libt.so");
    let res = link_shared(
        &[obj],
        &out,
        Some(b"libt.so"),
        false,
        IcfMode::None,
        false,
    );
    let Err(err) = res else {
        panic!("an alignment of 3 is not a mask any cursor can use")
    };
    assert!(
        err.to_string().contains("common symbol alignment"),
        "the error must name the broken column: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// An alignment past the largest a linker has use for is refused.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn an_absurd_common_alignment_is_refused() {
    let dir = std::env::temp_dir()
        .join(format!("xold_commonalign3_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the workdir");
    let obj = object(&dir, &build_common(1 << 31));
    let out = dir.join("libt.so");
    let res = link_shared(
        &[obj],
        &out,
        Some(b"libt.so"),
        false,
        IcfMode::None,
        false,
    );
    let Err(err) = res else {
        panic!("2^31 is an arithmetic hazard, not a request")
    };
    assert!(
        err.to_string().contains("common symbol alignment"),
        "the error must name the broken column: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
