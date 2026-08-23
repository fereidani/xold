//! On-disk structures must carry little-endian integers whatever the host is.
//!
//! Every container xold handles stores integers little-endian, and the structs
//! are viewed over mapped bytes with `bytemuck`, which never reorders. If the
//! fields were plain host-order integers these tests would pass on a
//! little-endian host and fail on a big-endian one -- which is exactly the bug
//! they exist to prevent, since almost nobody would run them there.

use bytemuck::Zeroable;
use xold::{
    coff::pe::{DataDirectory, NtFileHeader},
    elf::{Ehdr64, Rela64, Shdr64, Sym64},
    endian::{U16, U32, U64},
    macho::lc::MachHeader64,
};

#[test]
fn little_endian_integers_round_trip() {
    assert_eq!(U16::new(0x0102).get(), 0x0102);
    assert_eq!(U32::new(0x0102_0304).get(), 0x0102_0304);
    assert_eq!(U64::new(0x0102_0304_0506_0708).get(), 0x0102_0304_0506_0708);
}

#[test]
fn integers_are_stored_least_significant_byte_first() {
    assert_eq!(bytemuck::bytes_of(&U16::new(0x0102)), &[0x02, 0x01]);
    assert_eq!(
        bytemuck::bytes_of(&U32::new(0x0102_0304)),
        &[0x04, 0x03, 0x02, 0x01]
    );
    assert_eq!(
        bytemuck::bytes_of(&U64::new(0x0102_0304_0506_0708)),
        &[0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
    );
}

#[test]
fn section_headers_serialise_little_endian() {
    let mut shdr = Shdr64::zeroed();
    shdr.sh_size = U64::new(0x1122_3344_5566_7788);
    shdr.sh_type = U32::new(0x0000_0001);
    let bytes = bytemuck::bytes_of(&shdr);
    // sh_size is at offset 32 in the ELF64 section header.
    assert_eq!(
        &bytes[32..40],
        &[0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]
    );
    // sh_type is at offset 4.
    assert_eq!(&bytes[4..8], &[0x01, 0x00, 0x00, 0x00]);
}

#[test]
fn a_header_read_back_from_bytes_agrees() {
    let mut ehdr = Ehdr64::zeroed();
    ehdr.e_machine = U16::new(0x003e);
    ehdr.e_entry = U64::new(0x0040_1000);
    let bytes = bytemuck::bytes_of(&ehdr).to_vec();
    let back: Ehdr64 = bytemuck::pod_read_unaligned(&bytes);
    assert_eq!(back.e_machine.get(), 0x003e);
    assert_eq!(back.e_entry.get(), 0x0040_1000);

    let mut sym = Sym64::zeroed();
    sym.st_value = U64::new(0xdead_beef);
    let raw = bytemuck::bytes_of(&sym).to_vec();
    assert_eq!(
        bytemuck::pod_read_unaligned::<Sym64>(&raw).st_value.get(),
        0xdead_beef
    );

    let mut rela = Rela64::zeroed();
    rela.r_addend = xold::endian::I64::new(-4);
    let raw = bytemuck::bytes_of(&rela).to_vec();
    assert_eq!(
        bytemuck::pod_read_unaligned::<Rela64>(&raw).r_addend.get(),
        -4
    );
}

#[test]
fn pe_headers_serialise_little_endian() {
    let mut nt = NtFileHeader::zeroed();
    nt.signature = U32::new(0x0000_4550);
    nt.machine = U16::new(0x8664);
    let bytes = bytemuck::bytes_of(&nt);
    assert_eq!(&bytes[0..4], &[0x50, 0x45, 0x00, 0x00], "PE\0\0 signature");
    assert_eq!(&bytes[4..6], &[0x64, 0x86], "IMAGE_FILE_MACHINE_AMD64");

    let mut dir = DataDirectory::zeroed();
    dir.virtual_address = U32::new(0x0000_2000);
    assert_eq!(&bytemuck::bytes_of(&dir)[0..4], &[0x00, 0x20, 0x00, 0x00]);
}

#[test]
fn mach_headers_serialise_little_endian() {
    let mut h = MachHeader64::zeroed();
    h.magic = U32::new(0xfeed_facf);
    h.ncmds = U32::new(0x0000_0007);
    let bytes = bytemuck::bytes_of(&h);
    assert_eq!(&bytes[0..4], &[0xcf, 0xfa, 0xed, 0xfe], "MH_MAGIC_64");
    // ncmds is the fifth word of mach_header_64.
    assert_eq!(&bytes[16..20], &[0x07, 0x00, 0x00, 0x00]);
}
