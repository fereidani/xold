//! COMDAT selection kinds are compared pairwise, both ways.
//!
//! lld's `handleComdatSelection` decides each arriving copy against
//! the leader with BOTH selection fields in hand
//! (`lld/COFF/InputFiles.cpp`):
//!
//! * `ANY` paired with `LARGEST` is treated as `LARGEST` -- cl.exe picks `ANY`
//!   for vftables under `/GR-` and `LARGEST` under `/GR`, and objects built
//!   with each must link.
//! * any other disagreement between the kinds is a duplicate-symbol error,
//!   whichever copy arrived first.
//!
//! xold looked only at the first copy's kind: an `ANY` leader silently
//! swallowed a larger `LARGEST` copy, and a kind disagreement (say
//! `ANY` against `SAME_SIZE`) linked without complaint.
//!
//! The fixtures hand-build one-object COMDATs (no compiler emits
//! mismatched kinds on purpose): a `.text` COMDAT carrying a marker
//! byte, a section symbol whose auxiliary record states the kind, and
//! the external `f` that names the group.

use std::{fs, path::PathBuf};

use xold::{coff::PeImage, input::Input};

/// Machine type and the flags of a `.text` COMDAT.
const AMD64: u16 = 0x8664;
const TEXT_COMDAT_FLAGS: u32 = 0x6000_1020;

/// The selection kinds the fixtures use.
const LARGEST: u8 = 6;
const ANY: u8 = 2;
const SAME_SIZE: u8 = 3;

/// Builds a one-COMDAT object whose `.text` holds `size` bytes of
/// `marker`, with the auxiliary section record declaring `selection`
/// and the external `f` naming the group.
fn comdat_object(selection: u8, size: u32, marker: u8) -> Vec<u8> {
    const DATA_OFF: u32 = 60;
    const SYM_OFF_PAD: u32 = 4; // past the data, back to an 8-byte line
    let data_end = DATA_OFF + size;
    let sym_off = data_end + (SYM_OFF_PAD - data_end % 8) % 8;
    let mut out = Vec::new();
    out.extend_from_slice(&AMD64.to_le_bytes()); // Machine
    out.extend_from_slice(&1u16.to_le_bytes()); // NumberOfSections
    out.extend_from_slice(&0u32.to_le_bytes()); // TimeDateStamp
    out.extend_from_slice(&sym_off.to_le_bytes()); // PointerToSymbolTable
    out.extend_from_slice(&3u32.to_le_bytes()); // NumberOfSymbols
    out.extend_from_slice(&0u16.to_le_bytes()); // SizeOfOptionalHeader
    out.extend_from_slice(&0u16.to_le_bytes()); // Characteristics
    // Section header: `.text`, COMDAT, `size` raw bytes.
    out.extend_from_slice(b".text\0\0\0");
    out.extend_from_slice(&size.to_le_bytes()); // VirtualSize
    out.extend_from_slice(&0u32.to_le_bytes()); // VirtualAddress
    out.extend_from_slice(&size.to_le_bytes()); // SizeOfRawData
    out.extend_from_slice(&DATA_OFF.to_le_bytes()); // PointerToRawData
    out.extend_from_slice(&0u32.to_le_bytes()); // PointerToRelocations
    out.extend_from_slice(&0u32.to_le_bytes()); // PointerToLinenumbers
    out.extend_from_slice(&0u16.to_le_bytes()); // NumberOfRelocations
    out.extend_from_slice(&0u16.to_le_bytes()); // NumberOfLinenumbers
    out.extend_from_slice(&TEXT_COMDAT_FLAGS.to_le_bytes());
    debug_assert_eq!(out.len(), DATA_OFF as usize);
    out.extend(vec![marker; usize::try_from(size).unwrap_or(0)]);
    while out.len() < sym_off as usize {
        out.push(0);
    }
    //   0: .text -- static, one auxiliary section record
    //   1: aux section (length, selection)
    //   2: f     -- external in section 1, the group's name
    out.extend_from_slice(b".text\0\0\0");
    out.extend_from_slice(&0u32.to_le_bytes()); // Value
    out.extend_from_slice(&1i16.to_le_bytes()); // SectionNumber
    out.extend_from_slice(&0u16.to_le_bytes()); // Type
    out.push(0x03); // STATIC
    out.push(1); // NumberOfAuxSymbols
    out.extend_from_slice(&size.to_le_bytes()); // Length
    out.extend_from_slice(&0u16.to_le_bytes()); // NumberOfRelocations
    out.extend_from_slice(&0u16.to_le_bytes()); // NumberOfLinenumbers
    out.extend_from_slice(&0u32.to_le_bytes()); // CheckSum
    out.extend_from_slice(&1u16.to_le_bytes()); // Number (association)
    out.push(selection); // Selection
    out.push(0); // bReserved
    out.extend_from_slice(&0u16.to_le_bytes()); // HighNumber
    out.extend_from_slice(&0u32.to_le_bytes()); // zero prefix: long name
    out.extend_from_slice(&4u32.to_le_bytes()); // "f" at string offset 4
    out.extend_from_slice(&0u32.to_le_bytes()); // Value
    out.extend_from_slice(&1i16.to_le_bytes()); // SectionNumber
    out.extend_from_slice(&0u16.to_le_bytes()); // Type
    out.push(0x02); // EXTERNAL
    out.push(0); // NumberOfAuxSymbols
    // String table: the total length then `f`.
    out.extend_from_slice(&6u32.to_le_bytes());
    out.extend_from_slice(b"f\0");
    out
}

/// `ANY` paired with `LARGEST` behaves as `LARGEST`: the bigger copy
/// wins, not the first.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn any_and_largest_merge_to_largest() {
    let (dir, a, b) = fixtures(
        "merge",
        comdat_object(ANY, 16, 0xaa),
        comdat_object(LARGEST, 32, 0xbb),
    );
    let out = dir.join("merge.dll");
    // A DLL link: no entry stub, so `.text` starts at the winning member.
    let files = [Input::Path(&a), Input::Path(&b)];
    let res = xold::coff::link_coff(&files, &out, b"f", true);
    assert!(res.is_ok(), "the pair must link: {:?}", res.err());
    let bytes = fs::read(&out).expect("read the image");
    let img = PeImage::parse(&bytes).expect("parse the image");
    let text = img
        .sections()
        .into_iter()
        .find(|s| s.name == b".text")
        .expect(".text must be placed");
    assert_eq!(
        &text.data[..1],
        &[0xbb],
        "the larger LARGEST copy must win over the earlier ANY copy"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Disagreeing selection kinds are refused whichever copy came first.
#[test]
#[cfg_attr(miri, ignore = "Miri does not support file-backed memory mappings")]
fn conflicting_selection_kinds_are_refused() {
    let (dir, a, b) = fixtures(
        "conflict",
        comdat_object(ANY, 16, 0xaa),
        comdat_object(SAME_SIZE, 16, 0xaa),
    );
    let out = dir.join("conflict.exe");
    let files = [Input::Path(&a), Input::Path(&b)];
    let res = xold::coff::link_coff(&files, &out, b"f", false);
    let Err(err) = res else {
        let _ = fs::remove_dir_all(&dir);
        panic!("ANY against SAME_SIZE must be refused");
    };
    let text = format!("{err}");
    assert!(
        text.contains("COMDAT"),
        "the refusal must name the COMDAT conflict: {text}"
    );
    assert!(!out.exists(), "nothing may be published on refusal");
    let _ = fs::remove_dir_all(&dir);
}

/// Writes both fixtures beside each other under a per-test directory.
fn fixtures(tag: &str, a: Vec<u8>, b: Vec<u8>) -> (PathBuf, PathBuf, PathBuf) {
    let dir = std::env::temp_dir()
        .join(format!("xold_comdatself_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create the workdir");
    let obj_a = dir.join("a.obj");
    let obj_b = dir.join("b.obj");
    fs::write(&obj_a, a).expect("write object a");
    fs::write(&obj_b, b).expect("write object b");
    (dir, obj_a, obj_b)
}
