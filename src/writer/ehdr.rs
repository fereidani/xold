//! ELF header (Ehdr) emission: writes the fixed `Ehdr64` at offset zero.

use super::{EHDR_SIZE, EV_CURRENT, PHDR_SIZE, SHDR_SIZE};
use crate::{
    elf::{
        Ehdr64,
        constants::{
            ELFCLASS64, ELFDATA2LSB, ELFMAG, ET_DYN, ET_EXEC, SHN_LORESERVE,
        },
    },
    endian::{U16, U32, U64},
    error::{Error, Result},
    layout::Layout,
    reloc::Target,
    util::write_pod,
};

/// Writes the ELF header at offset zero.
/// A table of `SHN_LORESERVE` or more headers cannot state its own length in
/// `e_shnum`: the field's values from there up are reserved, and the real
/// count moves into `sh_size` of header zero. That escape is not implemented,
/// and writing a truncated count -- or the zero a failed conversion produced,
/// which means "read the escape" -- would describe a table that is not there.
/// So the link stops instead.
pub(super) fn write_ehdr(
    image: &mut [u8],
    target: Target,
    layout: &Layout,
    shdr_off: u64,
    shdr_count: usize,
    shstrndx: u16,
) -> Result<()> {
    let shnum = u16::try_from(shdr_count)
        .ok()
        .filter(|n| *n < SHN_LORESERVE)
        .ok_or(Error::OutOfRange(
            "section count needs the SHN_LORESERVE escape, which is not \
             implemented",
        ))?;
    let mut ident = [0u8; 16];
    ident[..4].copy_from_slice(&ELFMAG);
    ident[4] = ELFCLASS64;
    ident[5] = ELFDATA2LSB;
    ident[6] = u8::try_from(EV_CURRENT).unwrap_or(0);
    let header = Ehdr64 {
        e_ident: ident,
        e_type: U16::new(if layout.is_pie() { ET_DYN } else { ET_EXEC }),
        e_machine: U16::new(target.machine()),
        e_version: U32::new(EV_CURRENT),
        e_entry: U64::new(layout.entry()),
        e_phoff: U64::new(EHDR_SIZE),
        e_shoff: U64::new(shdr_off),
        // Which variant of the architecture the image is built for, folded
        // from the inputs. On RISC-V this is the floating-point calling
        // convention, which the loader checks before it will run the image.
        e_flags: U32::new(layout.e_flags),
        e_ehsize: U16::new(u16::try_from(EHDR_SIZE).unwrap_or(0)),
        e_phentsize: U16::new(u16::try_from(PHDR_SIZE).unwrap_or(0)),
        e_phnum: U16::new(layout.phdr_count()),
        e_shentsize: U16::new(u16::try_from(SHDR_SIZE).unwrap_or(0)),
        e_shnum: U16::new(shnum),
        e_shstrndx: U16::new(shstrndx),
    };
    write_pod(image, 0, &header);
    Ok(())
}
