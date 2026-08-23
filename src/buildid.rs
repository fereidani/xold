//! `--build-id`: the `NT_GNU_BUILD_ID` note that names one build of an image.
//!
//! The note holds a byte string that identifies the image. Nothing at runtime
//! reads it; what reads it is everything around the program -- a debugger
//! matching a core dump to its binary, a debuginfo server looking a build up,
//! a package manager checking that two files came from the same link. The
//! only property any of them needs is that two different images get different
//! strings and one image gets the same string every time it is linked.
//!
//! That second half is why `--build-id=uuid` is not implemented here: it
//! answers with a random string, so the same inputs would produce a different
//! image on every run, and byte-for-byte reproducibility is a property this
//! linker holds to. Every other style is a digest of the image itself, which
//! satisfies both halves at once.

use crate::util::write_pod;

/// Which digest the note carries.
#[derive(Clone, PartialEq, Eq, Default)]
pub enum BuildId {
    /// No note at all. The default, and what `--build-id=none` asks for.
    #[default]
    None,
    /// A cheap non-cryptographic digest of the image, and what a bare
    /// `--build-id` selects.
    ///
    /// It identifies a build; it is not a defence against a forged one. lld
    /// makes the same choice for the same reason: hashing a large image with
    /// SHA-1 costs more than the rest of the link, and nothing that reads a
    /// build id needs the digest to be cryptographic.
    Fast,
    /// MD5 of the image.
    Md5,
    /// SHA-1 of the image, when the caller asks for it by name.
    Sha1,
    /// The bytes the caller wrote, as `--build-id=0x...`.
    Hex(Vec<u8>),
}

/// The note's fixed prefix: `n_namesz`, `n_descsz`, `n_type`, then the name
/// `GNU\0`. The descriptor follows, and it is what carries the digest.
const HEADER: usize = 16;

/// `NT_GNU_BUILD_ID`.
const NT_GNU_BUILD_ID: u32 = 3;

/// The vendor name every build-id note carries, NUL-terminated and already
/// padded to four bytes.
const NAME: [u8; 4] = *b"GNU\0";

impl BuildId {
    /// How many bytes of digest the note carries.
    pub fn digest_len(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Fast | Self::Md5 => 16,
            Self::Sha1 => 20,
            Self::Hex(bytes) => bytes.len(),
        }
    }

    /// The whole note's size, rounded up so what follows it stays aligned.
    /// Zero when there is no note, which is what keeps the section unplaced.
    pub fn note_size(&self) -> u64 {
        let len = self.digest_len();
        if len == 0 {
            return 0;
        }
        let padded = len.next_multiple_of(4);
        u64::try_from(HEADER + padded).unwrap_or(0)
    }

    /// Writes the note's header and name into `note`, leaving the descriptor
    /// zeroed for [`Self::fill`] to complete.
    ///
    /// The two halves are separate because the digest covers the finished
    /// image: the header has to be in place before the image is hashed, and
    /// the descriptor can only be written afterwards.
    pub fn write_header(&self, image: &mut [u8], at: u64) {
        let len = self.digest_len();
        if len == 0 {
            return;
        }
        let namesz = u32::try_from(NAME.len()).unwrap_or(0);
        let descsz = u32::try_from(len).unwrap_or(0);
        write_pod(image, at, &namesz.to_le_bytes());
        write_pod(image, at + 4, &descsz.to_le_bytes());
        write_pod(image, at + 8, &NT_GNU_BUILD_ID.to_le_bytes());
        write_pod(image, at + 12, &NAME);
    }

    /// Computes the digest over the finished image and writes it into the
    /// note's descriptor at `at`.
    ///
    /// The descriptor is zero while the digest is taken, so the answer does
    /// not depend on itself. That is what lld and GNU `ld` both do, and it is
    /// what makes the same inputs produce the same note.
    pub fn fill(&self, image: &mut [u8], at: u64) {
        let Ok(off) = usize::try_from(at) else {
            return;
        };
        let digest = match self {
            Self::None => return,
            Self::Hex(bytes) => bytes.clone(),
            Self::Fast => fast(image).to_vec(),
            Self::Md5 => md5(image).to_vec(),
            Self::Sha1 => sha1(image).to_vec(),
        };
        let Some(slot) = image.get_mut(off + HEADER..) else {
            return;
        };
        let len = digest.len().min(slot.len());
        if let Some(dst) = slot.get_mut(..len) {
            dst.copy_from_slice(digest.get(..len).unwrap_or(&[]));
        }
    }
}

/// Reads a `--build-id=0x...` value into the bytes it names.
///
/// An odd number of digits, or one that is not hexadecimal, is an error
/// rather than a truncated note: the caller wrote an exact string, and half of
/// it identifies nothing.
pub fn parse_hex(text: &str) -> Option<Vec<u8>> {
    if text.is_empty() || !text.len().is_multiple_of(2) {
        return None;
    }
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(text.len() / 2);
    for pair in bytes.as_chunks::<2>().0 {
        let hi = (pair[0] as char).to_digit(16)?;
        let lo = (pair[1] as char).to_digit(16)?;
        out.push(u8::try_from(hi * 16 + lo).ok()?);
    }
    Some(out)
}

/// The non-cryptographic digest `--build-id=fast` asks for: two independent
/// FNV-1a lanes over the image, one per half of each 16-byte block, so the
/// two halves of the answer cannot agree by construction.
///
/// FNV-1a is chosen for being short enough to read and fixed for all time,
/// which a build id needs and a faster hash with a tuned constant table does
/// not give without pinning that table here too.
pub fn fast(image: &[u8]) -> [u8; 16] {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut lanes = [OFFSET, OFFSET ^ u64::MAX];
    for (i, chunk) in image.chunks(8).enumerate() {
        let lane = i % 2;
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        let value = u64::from_le_bytes(word);
        lanes[lane] ^= value;
        lanes[lane] = lanes[lane].wrapping_mul(PRIME);
    }
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&lanes[0].to_le_bytes());
    out[8..].copy_from_slice(&lanes[1].to_le_bytes());
    out
}

/// The MD5 of `data` (RFC 1321).
pub fn md5(data: &[u8]) -> [u8; 16] {
    let mut state: [u32; 4] =
        [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    for_each_block(data, false, |block| md5_block(&mut state, block));
    let mut out = [0u8; 16];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

/// The SHA-1 of `data` (FIPS 180-4).
pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut state: [u32; 5] = [
        0x6745_2301,
        0xefcd_ab89,
        0x98ba_dcfe,
        0x1032_5476,
        0xc3d2_e1f0,
    ];
    for_each_block(data, true, |block| sha1_block(&mut state, block));
    let mut out = [0u8; 20];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Hands `f` each 64-byte block of the message, then the padding both
/// algorithms share: a `0x80` byte, zeroes, and the bit length in the last
/// eight bytes -- big-endian for SHA-1, little-endian for MD5.
///
/// The message is walked in place rather than copied into a block list: the
/// message here is the whole output image, and a second copy of it is the one
/// allocation this must not make.
fn for_each_block(data: &[u8], big_endian: bool, mut f: impl FnMut(&[u8; 64])) {
    let bits = (data.len() as u64).wrapping_mul(8);
    let (whole, tail) = data.as_chunks::<64>();
    for block in whole {
        f(block);
    }
    let mut last = [0u8; 64];
    last[..tail.len()].copy_from_slice(tail);
    last[tail.len()] = 0x80;
    // A tail with no room for the length word takes a block of its own, and
    // the length goes in the next.
    if tail.len() >= 56 {
        f(&last);
        last = [0u8; 64];
    }
    let length = if big_endian {
        bits.to_be_bytes()
    } else {
        bits.to_le_bytes()
    };
    last[56..].copy_from_slice(&length);
    f(&last);
}

/// One MD5 round over a 64-byte block.
///
/// The working variables keep the specification's names: `a` through `d` are
/// what RFC 1321 calls them, and a reader checking this against the standard
/// is the only reader it has.
#[allow(clippy::many_single_char_names)]
fn md5_block(state: &mut [u32; 4], block: &[u8; 64]) {
    let mut m = [0u32; 16];
    for (i, word) in block.as_chunks::<4>().0.iter().enumerate() {
        m[i] = u32::from_le_bytes(*word);
    }
    let [mut a, mut b, mut c, mut d] = *state;
    for i in 0..64u32 {
        let idx = i as usize;
        let (f, g) = match idx / 16 {
            0 => ((b & c) | (!b & d), idx),
            1 => ((d & b) | (!d & c), (5 * idx + 1) % 16),
            2 => (b ^ c ^ d, (3 * idx + 5) % 16),
            _ => (c ^ (b | !d), (7 * idx) % 16),
        };
        let tmp = d;
        d = c;
        c = b;
        let sum = a
            .wrapping_add(f)
            .wrapping_add(MD5_K[idx])
            .wrapping_add(m[g]);
        b = b.wrapping_add(sum.rotate_left(MD5_SHIFT[idx]));
        a = tmp;
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
}

/// One SHA-1 round over a 64-byte block.
///
/// The working variables keep FIPS 180-4's names, for the reason
/// [`md5_block`] keeps RFC 1321's.
#[allow(clippy::many_single_char_names)]
fn sha1_block(state: &mut [u32; 5], block: &[u8; 64]) {
    let mut w = [0u32; 80];
    for (i, word) in block.as_chunks::<4>().0.iter().enumerate() {
        w[i] = u32::from_be_bytes(*word);
    }
    for i in 16..80 {
        w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
    }
    let [mut a, mut b, mut c, mut d, mut e] = *state;
    for (i, word) in w.iter().enumerate() {
        let (f, k) = match i / 20 {
            0 => ((b & c) | (!b & d), 0x5a82_7999),
            1 => (b ^ c ^ d, 0x6ed9_eba1),
            2 => ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc),
            _ => (b ^ c ^ d, 0xca62_c1d6),
        };
        let tmp = a
            .rotate_left(5)
            .wrapping_add(f)
            .wrapping_add(e)
            .wrapping_add(k)
            .wrapping_add(*word);
        e = d;
        d = c;
        c = b.rotate_left(30);
        b = a;
        a = tmp;
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
}

/// MD5's per-round additive constants: `floor(2^32 * abs(sin(i + 1)))`.
const MD5_K: [u32; 64] = [
    0xd76a_a478,
    0xe8c7_b756,
    0x2420_70db,
    0xc1bd_ceee,
    0xf57c_0faf,
    0x4787_c62a,
    0xa830_4613,
    0xfd46_9501,
    0x6980_98d8,
    0x8b44_f7af,
    0xffff_5bb1,
    0x895c_d7be,
    0x6b90_1122,
    0xfd98_7193,
    0xa679_438e,
    0x49b4_0821,
    0xf61e_2562,
    0xc040_b340,
    0x265e_5a51,
    0xe9b6_c7aa,
    0xd62f_105d,
    0x0244_1453,
    0xd8a1_e681,
    0xe7d3_fbc8,
    0x21e1_cde6,
    0xc337_07d6,
    0xf4d5_0d87,
    0x455a_14ed,
    0xa9e3_e905,
    0xfcef_a3f8,
    0x676f_02d9,
    0x8d2a_4c8a,
    0xfffa_3942,
    0x8771_f681,
    0x6d9d_6122,
    0xfde5_380c,
    0xa4be_ea44,
    0x4bde_cfa9,
    0xf6bb_4b60,
    0xbebf_bc70,
    0x289b_7ec6,
    0xeaa1_27fa,
    0xd4ef_3085,
    0x0488_1d05,
    0xd9d4_d039,
    0xe6db_99e5,
    0x1fa2_7cf8,
    0xc4ac_5665,
    0xf429_2244,
    0x432a_ff97,
    0xab94_23a7,
    0xfc93_a039,
    0x655b_59c3,
    0x8f0c_cc92,
    0xffef_f47d,
    0x8584_5dd1,
    0x6fa8_7e4f,
    0xfe2c_e6e0,
    0xa301_4314,
    0x4e08_11a1,
    0xf753_7e82,
    0xbd3a_f235,
    0x2ad7_d2bb,
    0xeb86_d391,
];

/// MD5's per-round left-rotation amounts.
const MD5_SHIFT: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4,
    11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6,
    10, 15, 21,
];
