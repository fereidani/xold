//! Malformed-archive handling: the `ar` reader must reject a corrupt symbol
//! index instead of trusting the counts inside it.
//!
//! The GNU index member starts with a symbol count read straight from the
//! file. A count larger than the member could possibly describe used to drive
//! the parse loop for as many iterations as the number said, reading empty
//! names off the end of the member; the reader now checks the count against
//! the member's own size first.

use std::path::Path;

use xold::{archive::Archive, error::Error};

/// Builds one member header: a 16-byte name field, the fixed metadata fields,
/// a right-padded decimal size, and the `` `\n `` terminator (60 bytes).
fn header(name: &str, size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(60);
    out.extend_from_slice(format!("{name:<16}").as_bytes());
    out.extend_from_slice(b"0           "); // mtime, 12
    out.extend_from_slice(b"0     "); // uid, 6
    out.extend_from_slice(b"0     "); // gid, 6
    out.extend_from_slice(b"100644  "); // mode, 8
    out.extend_from_slice(format!("{size:<10}").as_bytes());
    out.extend_from_slice(b"`\n");
    assert_eq!(out.len(), 60, "member header is 60 bytes");
    out
}

/// An archive whose symbol index claims more symbols than its bytes can hold
/// must be rejected, not walked. The count here (`0x3fff_ffff`) would take
/// over a billion iterations to exhaust.
#[test]
fn absurd_symbol_count_is_rejected() {
    let mut index = Vec::new();
    index.extend_from_slice(&0x3fff_ffffu32.to_be_bytes());
    index.extend_from_slice(&[0u8; 4]); // one offset, no name table

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"!<arch>\n");
    bytes.extend_from_slice(&header("/", index.len()));
    bytes.extend_from_slice(&index);

    let err = Archive::parse(&bytes, Path::new("lib.a"))
        .err()
        .expect("a symbol count larger than the member must be rejected");
    assert!(
        matches!(err, Error::Format(_) | Error::OutOfRange(_)),
        "unexpected error: {err:?}"
    );
}

/// A well-formed index still parses: one symbol, one member offset, one name.
#[test]
fn valid_symbol_index_parses() {
    let mut index = Vec::new();
    index.extend_from_slice(&1u32.to_be_bytes());
    index.extend_from_slice(&0x1234u32.to_be_bytes());
    index.extend_from_slice(b"sym\0");

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"!<arch>\n");
    bytes.extend_from_slice(&header("/", index.len()));
    bytes.extend_from_slice(&index);

    let archive = Archive::parse(&bytes, Path::new("lib.a"))
        .expect("valid index must parse");
    assert_eq!(archive.lookup(b"sym"), Some(0x1234));
    assert_eq!(archive.lookup(b"other"), None);
}

/// A member offset past the end of the mapping is an error from both accessors,
/// including one close enough to `usize::MAX` to overflow an unchecked add.
#[test]
fn out_of_range_member_offset_errors() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"!<arch>\n");
    bytes.extend_from_slice(&header("obj.o/", 4));
    bytes.extend_from_slice(b"data");

    let archive = Archive::parse(&bytes, Path::new("lib.a"))
        .expect("valid archive must parse");
    for offset in [u64::from(u32::MAX), u64::MAX] {
        assert!(
            archive.member(offset).is_err(),
            "member({offset:#x}) must error"
        );
        assert!(
            archive.member_name(offset).is_err(),
            "member_name({offset:#x}) must error"
        );
    }
    // The real member still reads back.
    assert_eq!(&archive.member(8).expect("member data")[..], b"data");
    assert_eq!(archive.member_name(8).expect("member name"), "obj.o");
}
