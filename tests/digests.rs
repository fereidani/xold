//! The digests the build-id note is built from.
//!
//! `--build-id=sha1` and `--build-id=md5` name published algorithms, and an
//! image's id is only useful if it is the value everything else computes for
//! the same bytes: a debuginfo server looks a program up by the digest its own
//! tools took of it. These are the standards' own test vectors.

use std::fmt::Write as _;

use xold::buildid::{fast, md5, sha1};

/// A digest as the hexadecimal string every tool prints.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// SHA-1 against the FIPS 180-4 examples, including the message long enough
/// to need a padding block of its own.
#[test]
fn sha1_matches_its_test_vectors() {
    assert_eq!(hex(&sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    assert_eq!(
        hex(&sha1(b"abc")),
        "a9993e364706816aba3e25717850c26c9cd0d89d"
    );
    assert_eq!(
        hex(&sha1(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        )),
        "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
    );
    let million = vec![b'a'; 1_000_000];
    assert_eq!(
        hex(&sha1(&million)),
        "34aa973cd4c4daa4f61eeb2bdbad27316534016f"
    );
}

/// MD5 against the RFC 1321 suite.
#[test]
fn md5_matches_its_test_vectors() {
    assert_eq!(hex(&md5(b"")), "d41d8cd98f00b204e9800998ecf8427e");
    assert_eq!(hex(&md5(b"abc")), "900150983cd24fb0d6963f7d28e17f72");
    assert_eq!(
        hex(&md5(b"message digest")),
        "f96b697d7cb7938d525a2f31aaf161d0"
    );
    assert_eq!(
        hex(&md5(b"12345678901234567890123456789012345678901234567890\
              123456789012345678901234567890")),
        "57edf4a22be3c955ac49da2e2107b67a"
    );
}

/// The cheap digest is a function of the bytes alone: the same input gives
/// the same answer, and a one-bit change gives another.
#[test]
fn fast_separates_what_it_is_given() {
    assert_eq!(fast(b"the image"), fast(b"the image"));
    assert_ne!(fast(b"the image"), fast(b"the imagf"));
    // A change beyond the first block still moves the answer, which a digest
    // that stopped early would not do.
    let mut long = vec![0u8; 4096];
    let first = fast(&long);
    long[4000] = 1;
    assert_ne!(first, fast(&long));
}
