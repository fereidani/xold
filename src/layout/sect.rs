//! The output section table: one ordered list of every region the ELF image
//! can contain, with the section-header metadata each one carries.
//!
//! Three passes need this order and must agree on it: layout numbers the
//! section headers ([`super::place::assign_shndx`]), the writer emits them
//! ([`crate::writer`]), and symbol resolution stamps each input section with
//! the index of the region that holds it. Keeping the list here, once, is what
//! makes those three consistent by construction rather than by comment.
//!
//! A region is emitted only when its size is non-zero, so a static link
//! silently skips every dynamic entry and a link without TLS skips the TLS
//! pair. The order itself is the placement order: read-execute content first,
//! then the read-only dynamic tables, then the writable segment.
//!
//! The writable segment opens with the RELRO run ([`relro`]) -- the regions a
//! loader can re-protect read-only once it has applied the image's
//! relocations -- and only then the regions that must stay writable for the
//! life of the process. `PT_GNU_RELRO` describes a single range, so the run
//! has to be contiguous and has to come first.

use crate::elf::constants::{
    SHF_ALLOC, SHF_EXECINSTR, SHF_INFO_LINK, SHF_TLS, SHF_WRITE, SHT_DYNAMIC,
    SHT_DYNSYM, SHT_FINI_ARRAY, SHT_GNU_HASH, SHT_GNU_VERNEED, SHT_GNU_VERSYM,
    SHT_HASH, SHT_INIT_ARRAY, SHT_NOBITS, SHT_NOTE, SHT_PREINIT_ARRAY,
    SHT_PROGBITS, SHT_RELA, SHT_STRTAB,
};

/// On-disk size of one `Sym64`; the `.dynsym` `sh_entsize`.
const SYM_SIZE: u64 = 24;
/// On-disk size of one `Rela64`; the `.rela.*` `sh_entsize`.
const RELA_SIZE: u64 = 24;
/// On-disk size of one `Dyn64`; the `.dynamic` `sh_entsize`.
const DYN_SIZE: u64 = 16;

/// One region of the output image, in section-header order.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Sect {
    Interp,
    /// The image's allocated notes, gathered into one section so a single
    /// `PT_NOTE` can cover them. Notes are self-describing -- each carries its
    /// own name, type and length -- so a reader walks the run without needing
    /// the input section names back.
    Note,
    /// The `NT_GNU_BUILD_ID` note this link synthesises, kept apart from the
    /// notes the inputs contributed because its bytes come from the finished
    /// image rather than from any member.
    BuildId,
    Init,
    Text,
    Fini,
    Plt,
    Rodata,
    EhFrame,
    EhFrameHdr,
    GnuHash,
    Hash,
    Dynsym,
    Dynstr,
    Versym,
    Verneed,
    RelaDyn,
    RelaPlt,
    Dynamic,
    Got,
    /// Relocated read-only data: `const` objects whose initialisers need a
    /// relocation. In the protected run, so they are read-only once the loader
    /// has applied them.
    DataRelRo,
    /// `.preinit_array`, ahead of `.init_array`: the runtime walks the
    /// pre-initialisation array first, and keeping the image order the same as
    /// the call order is what the system linkers do.
    PreinitArray,
    InitArray,
    FiniArray,
    /// The PLT's GOT, and the first region past the protected run. xold binds
    /// lazily, so the loader writes a resolved address into a `.got.plt` slot
    /// on the first call through its stub -- long after `PT_GNU_RELRO` was
    /// applied. lld draws the same line: `isRelroSection` returns
    /// `ctx.arg.zNow` for `.got.plt` (ELF/Writer.cpp:627), so it joins the run
    /// only under `-z now`, which xold does not implement.
    GotPlt,
    Data,
    Tdata,
    Tbss,
    Bss,
}

/// The section a header's `sh_link` cross-references.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Link {
    /// No cross-reference (`sh_link = 0`).
    None,
    /// The symbol table the section indexes (`.hash`, `.rela.*`, versym).
    Dynsym,
    /// The string table the section's names live in (`.dynsym`, verneed).
    Dynstr,
}

/// The meaning of a header's `sh_info` field.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Info {
    /// Unused (`sh_info = 0`).
    Zero,
    /// Index of the first non-local symbol; `.dynsym` carries only the null
    /// local, so 1.
    FirstGlobal,
    /// The number of `Elf_Verneed` records (`.gnu.version_r` only).
    VerneedCount,
    /// The `.got.plt` section index, which is what `.rela.plt`'s entries
    /// patch. Paired with [`SHF_INFO_LINK`] on the same header.
    GotPlt,
}

/// The `align` of a region whose alignment is not a constant: it is the
/// largest its members ask for, which only measurement knows.
///
/// Placement records what it used and [`crate::layout::Layout::sect_align`]
/// reads it back. Stating it in the table rather than leaving the constant
/// there is what keeps `sh_addralign` from claiming an alignment the placement
/// cursor never honoured -- `.text` reporting 16 at an address 8 bytes off a
/// 16-byte boundary, which the ELF spec forbids.
pub const MEASURED: u64 = 0;

/// The fixed section-header metadata of one region.
#[derive(Copy, Clone)]
pub struct Spec {
    pub name: &'static [u8],
    pub sh_type: u32,
    pub flags: u64,
    /// `sh_addralign`, or [`MEASURED`] for a region aggregated from input
    /// members.
    pub align: u64,
    pub entsize: u64,
    pub link: Link,
    pub info: Info,
}

/// Builds a spec with the common `sh_link = 0`, `sh_info = 0` case.
const fn plain(
    name: &'static [u8],
    sh_type: u32,
    flags: u64,
    align: u64,
) -> Spec {
    Spec {
        name,
        sh_type,
        flags,
        align,
        entsize: 0,
        link: Link::None,
        info: Info::Zero,
    }
}

/// Every region in placement and section-header order, paired with its
/// metadata. Iterating this drives both the shndx assignment and the writer's
/// header emission.
pub const TABLE: [(Sect, Spec); 29] = [
    (Sect::Interp, plain(b".interp", SHT_PROGBITS, SHF_ALLOC, 1)),
    (
        Sect::BuildId,
        plain(b".note.gnu.build-id", SHT_NOTE, SHF_ALLOC, 4),
    ),
    (Sect::Note, plain(b".note", SHT_NOTE, SHF_ALLOC, MEASURED)),
    (
        Sect::Init,
        plain(b".init", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, MEASURED),
    ),
    (
        Sect::Text,
        plain(b".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, MEASURED),
    ),
    (
        Sect::Fini,
        plain(b".fini", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, MEASURED),
    ),
    (
        Sect::Plt,
        plain(b".plt", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 16),
    ),
    (
        Sect::Rodata,
        plain(b".rodata", SHT_PROGBITS, SHF_ALLOC, MEASURED),
    ),
    (
        Sect::EhFrame,
        plain(b".eh_frame", SHT_PROGBITS, SHF_ALLOC, MEASURED),
    ),
    (
        Sect::EhFrameHdr,
        plain(b".eh_frame_hdr", SHT_PROGBITS, SHF_ALLOC, 4),
    ),
    (
        Sect::GnuHash,
        Spec {
            name: b".gnu.hash",
            sh_type: SHT_GNU_HASH,
            flags: SHF_ALLOC,
            align: 8,
            entsize: 0,
            link: Link::Dynsym,
            info: Info::Zero,
        },
    ),
    (
        Sect::Hash,
        Spec {
            name: b".hash",
            sh_type: SHT_HASH,
            flags: SHF_ALLOC,
            align: 4,
            entsize: 4,
            link: Link::Dynsym,
            info: Info::Zero,
        },
    ),
    (
        Sect::Dynsym,
        Spec {
            name: b".dynsym",
            sh_type: SHT_DYNSYM,
            flags: SHF_ALLOC,
            align: 8,
            entsize: SYM_SIZE,
            link: Link::Dynstr,
            info: Info::FirstGlobal,
        },
    ),
    (Sect::Dynstr, plain(b".dynstr", SHT_STRTAB, SHF_ALLOC, 1)),
    (
        Sect::Versym,
        Spec {
            name: b".gnu.version",
            sh_type: SHT_GNU_VERSYM,
            flags: SHF_ALLOC,
            align: 2,
            entsize: 2,
            link: Link::Dynsym,
            info: Info::Zero,
        },
    ),
    (
        Sect::Verneed,
        Spec {
            name: b".gnu.version_r",
            sh_type: SHT_GNU_VERNEED,
            flags: SHF_ALLOC,
            align: 4,
            entsize: 0,
            link: Link::Dynstr,
            info: Info::VerneedCount,
        },
    ),
    (
        Sect::RelaDyn,
        Spec {
            name: b".rela.dyn",
            sh_type: SHT_RELA,
            flags: SHF_ALLOC,
            align: 8,
            entsize: RELA_SIZE,
            link: Link::Dynsym,
            info: Info::Zero,
        },
    ),
    (
        Sect::RelaPlt,
        // A relocation section names the section its entries apply to in
        // `sh_info`, and sets `SHF_INFO_LINK` to say the number is a section
        // index. `.rela.plt` patches `.got.plt`; `.rela.dyn` reaches wherever
        // its targets live and names nothing, which is the same split lld
        // makes in `RelocationBaseSection::finalizeContents`
        // (`lld/ELF/SyntheticSections.cpp`).
        //
        // lld gates both on `.got.plt` having been emitted. Here the flag is
        // constant because the gate is implied: `.rela.plt` holds one entry
        // per PLT slot and `.got.plt` one slot per PLT entry, so an empty
        // `.got.plt` means an empty `.rela.plt`, and a zero-sized region emits
        // no header at all.
        Spec {
            name: b".rela.plt",
            sh_type: SHT_RELA,
            flags: SHF_ALLOC | SHF_INFO_LINK,
            align: 8,
            entsize: RELA_SIZE,
            link: Link::Dynsym,
            info: Info::GotPlt,
        },
    ),
    (
        Sect::Dynamic,
        Spec {
            name: b".dynamic",
            sh_type: SHT_DYNAMIC,
            flags: SHF_ALLOC | SHF_WRITE,
            align: 8,
            entsize: DYN_SIZE,
            link: Link::Dynstr,
            info: Info::Zero,
        },
    ),
    (
        Sect::Got,
        plain(b".got", SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, 8),
    ),
    (
        Sect::DataRelRo,
        plain(
            b".data.rel.ro",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_WRITE,
            MEASURED,
        ),
    ),
    (
        Sect::PreinitArray,
        plain(
            b".preinit_array",
            SHT_PREINIT_ARRAY,
            SHF_ALLOC | SHF_WRITE,
            MEASURED,
        ),
    ),
    (
        Sect::InitArray,
        plain(
            b".init_array",
            SHT_INIT_ARRAY,
            SHF_ALLOC | SHF_WRITE,
            MEASURED,
        ),
    ),
    (
        Sect::FiniArray,
        plain(
            b".fini_array",
            SHT_FINI_ARRAY,
            SHF_ALLOC | SHF_WRITE,
            MEASURED,
        ),
    ),
    (
        Sect::GotPlt,
        plain(b".got.plt", SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, 8),
    ),
    (
        Sect::Data,
        plain(b".data", SHT_PROGBITS, SHF_ALLOC | SHF_WRITE, MEASURED),
    ),
    (
        Sect::Tdata,
        plain(
            b".tdata",
            SHT_PROGBITS,
            SHF_ALLOC | SHF_WRITE | SHF_TLS,
            MEASURED,
        ),
    ),
    (
        Sect::Tbss,
        plain(
            b".tbss",
            SHT_NOBITS,
            SHF_ALLOC | SHF_WRITE | SHF_TLS,
            MEASURED,
        ),
    ),
    (
        Sect::Bss,
        plain(b".bss", SHT_NOBITS, SHF_ALLOC | SHF_WRITE, MEASURED),
    ),
];

/// The regions that hold input-section content, in address order.
///
/// Symbol resolution maps a placed virtual address back to its section index
/// through this subset; the synthetic tables never hold an input section.
pub const CONTENT: [Sect; 15] = [
    Sect::Note,
    Sect::Init,
    Sect::Text,
    Sect::Fini,
    Sect::Rodata,
    Sect::EhFrame,
    Sect::Got,
    Sect::DataRelRo,
    Sect::PreinitArray,
    Sect::InitArray,
    Sect::FiniArray,
    Sect::Data,
    Sect::Tdata,
    Sect::Tbss,
    Sect::Bss,
];

/// Whether a region occupies file bytes. The `SHT_NOBITS` pair (`.bss`,
/// `.tbss`) takes up memory only, so the image never grows to cover it.
pub const fn file_backed(sect: Sect) -> bool {
    !matches!(sect, Sect::Bss | Sect::Tbss)
}

/// Whether a region rides in the read-write `PT_LOAD` segment. Everything else
/// is read-only or executable and rides in the read-execute one.
pub const fn writable(spec: &Spec) -> bool {
    spec.flags & SHF_WRITE != 0
}

/// Whether a region belongs to the RELRO run, the head of the read-write
/// segment that `PT_GNU_RELRO` covers.
///
/// The membership mirrors lld's `isRelroSection` (ELF/Writer.cpp:575): the
/// `.dynamic` table (line 636), `.got` (line 612), the `.data.rel.ro` family
/// (line 645) and the three function-pointer arrays (lines 605-607) are
/// written once, by the loader, and never again. `.got.plt` is deliberately
/// absent: lld admits it only under `-z now` (line 627), and xold binds
/// lazily, so the loader stores through a `.got.plt` slot the first time a
/// stub is called.
///
/// xold parts from lld over the TLS pair, which lld puts in the run because
/// `.tdata` is only ever a template (line 595). xold places `.tdata`/`.tbss`
/// after the run instead, so that `PT_TLS` and the unprotected tail stay one
/// contiguous block and the run needs no second page boundary.
pub const fn relro(sect: Sect) -> bool {
    matches!(
        sect,
        Sect::Dynamic
            | Sect::Got
            | Sect::DataRelRo
            | Sect::PreinitArray
            | Sect::InitArray
            | Sect::FiniArray
    )
}

/// Whether a region holds executable instructions.
pub const fn executable(spec: &Spec) -> bool {
    spec.flags & SHF_EXECINSTR != 0
}

/// The content regions that hold code, in address order.
///
/// `.init`, `.text` and `.fini` are separate output sections, so a caller
/// asking whether an address names code has to consider all of them. The set
/// is derived from [`TABLE`] rather than listed a second time, so a new
/// executable region joins it by construction.
pub fn code() -> impl Iterator<Item = Sect> {
    CONTENT
        .into_iter()
        .filter(|s| TABLE.iter().any(|(t, spec)| t == s && executable(spec)))
}
