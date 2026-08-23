//! `.eh_frame` parsing and `.eh_frame_hdr` generation.
//!
//! The runtime unwinder (libgcc's personality routine, driven by glibc's
//! `dl_iterate_phdr`) locates the FDE covering a thrown PC through the
//! `PT_GNU_EH_FRAME` segment, which points at `.eh_frame_hdr`. That header is a
//! binary-search table over the FDEs: each entry pairs the FDE's initial
//! location (the function's runtime address, encoded `DW_EH_PE_datarel` to the
//! header) with the FDE's offset within `.eh_frame`. Without it the loader has
//! no unwind index and a C++ `throw` aborts at `terminate` instead of reaching
//! its `catch` handler.
//!
//! This module walks the merged output `.eh_frame` (after the writer applied
//! the FDE relocations, so the initial-location slots already hold their
//! runtime-relative values), reconstructs each FDE's covered PC, and emits the
//! `.eh_frame_hdr` bytes.
//!
//! CIE/FDE record layout (DWARF32, the only form the host clang emits):
//!
//! - Common: 4-byte `length`, then `length` bytes of payload. A zero `length`
//!   word marks the section terminator.
//! - CIE: payload starts with a 4-byte `CIE_id` of zero.
//! - FDE: payload starts with a 4-byte `CIE_pointer` that is the byte offset
//!   from the field's own position back to the associated CIE. It is always
//!   non-zero (a CIE precedes its FDEs), so `cie_pointer == 0` distinguishes a
//!   CIE from an FDE.
//!
//! The FDE's `initial_location` (the covered PC) sits 8 bytes into the record
//! (after `length` and `CIE_pointer`) and is interpreted per the encoding the
//! associated CIE advertises in its augmentation string's `R` entry.
//!
//! Because a zero `length` ends the section, the output `.eh_frame` is not a
//! concatenation of its inputs: it is the records of every input re-emitted
//! back to back, with no padding between them and exactly one zero terminator
//! at the very end, which the `split` submodule arranges. Any interior run of
//! four zero bytes --
//! inter-member alignment padding, or an input's own terminator carried into
//! the middle of the section -- truncates every walk that reaches it, this
//! module's included.

use rayon::slice::ParallelSliceMut;
use rustc_hash::FxHashMap;

use crate::{
    elf::Rela64,
    error::{Error, Result},
    layout::{Layout, Region, Sect, sect},
    util::{push_rel32, push_u32},
};

// --- DW_EH_PE encoding constants -----------------------------------------

/// Value-encoding formats (low nibble of a `DW_EH_PE` byte).
const DW_EH_PE_ABSPTR: u8 = 0x00;
const DW_EH_PE_UDATA2: u8 = 0x02;
const DW_EH_PE_UDATA4: u8 = 0x03;
const DW_EH_PE_UDATA8: u8 = 0x04;
const DW_EH_PE_SDATA2: u8 = 0x0a;
const DW_EH_PE_SDATA4: u8 = 0x0b;
const DW_EH_PE_SDATA8: u8 = 0x0c;
/// Modifier bit: the value is relative to the field's own runtime address.
const DW_EH_PE_PCREL: u8 = 0x10;
/// Selects the base of a `DW_EH_PE` byte: three bits, so the `indirect` bit
/// (0x80) stays out of the comparison.
const DW_EH_PE_BASE_MASK: u8 = 0x70;
/// Modifier bit: the value is relative to `.eh_frame_hdr`'s load address. Used
/// for the binary-search table entries.
const DW_EH_PE_DATAREL: u8 = 0x30;
/// `DW_EH_PE_aligned`: a base that is not an address but a placement rule,
/// and the one base whose value width is unknowable at link time.
const DW_EH_PE_ALIGNED: u8 = 0x50;

/// `.eh_frame_hdr` is exactly: 4-byte header + 4-byte `eh_frame_ptr` + 4-byte
/// `fde_count` + 8 bytes per table entry.
const HDR_FIXED: u64 = 12;
/// One binary-search table entry: `(initial_location, fde_offset)`, both
/// sdata4.
const ENTRY_SIZE: u64 = 8;

mod cie;
mod lsda;
mod split;

pub use lsda::sections_with_lsda;
pub use split::{EhFramePlan, EhMember, Piece, build as build_plan};

/// Whether `name` is `.eh_frame`.
pub fn is_eh_frame_name(name: &[u8]) -> bool {
    name == b".eh_frame"
}

/// The offset of the CIE an FDE record is associated with, within the same
/// input section.
///
/// The `CIE_pointer` field sits four bytes into the record and holds the
/// distance from its own position back to the CIE, so the CIE starts at
/// `in_off + 4 - cie_pointer`. `None` for a CIE, for a record whose pointer
/// reaches before the start of the section, or for one whose field cannot be
/// read.
///
/// Stated here rather than at each caller: two passes read the association
/// (the record splitter's liveness and the LSDA scan below), and a linker that
/// spells the same arithmetic twice is one refactor away from spelling it
/// differently.
fn cie_offset_of(data: &[u8], piece: Piece) -> Option<u64> {
    if !piece.fde {
        return None;
    }
    let field = usize::try_from(piece.in_off).ok()?.checked_add(4)?;
    let cie_ptr = u64::from(read_u32(data, field).ok()?);
    u64::try_from(field).ok()?.checked_sub(cie_ptr)
}

/// The record covering `offset`, if the walk in `pieces` produced one.
fn record_at(pieces: &[Piece], offset: u64) -> Option<usize> {
    let idx = pieces.partition_point(|p| p.in_off <= offset);
    let i = idx.checked_sub(1)?;
    pieces.get(i).filter(|p| p.covers(offset)).map(|_| i)
}

/// The symbol index of the first relocation landing in each record of
/// `pieces`, in record order.
///
/// An FDE's first relocation names the function it describes; the ones after
/// it name the LSDA and, in a CIE, the personality routine. lld reaches the
/// same relocation through `EhSectionPiece::firstRelocation` (consumed by
/// `EhFrameSection::isFdeLive`) and then reads its symbol; both of xold's
/// `.eh_frame` passes want the symbol and nothing else about the relocation,
/// so that is what this yields.
///
/// "First" means lowest `r_offset` within the record, not first in the
/// relocation table. The two agree for a table an assembler wrote, and part
/// company after `ld -r`, which may reorder `.rela.eh_frame`. Taking table
/// order there picks the LSDA's relocation instead of the function's, and both
/// callers then ask their question about the wrong section: liveness is
/// decided by whether the LSDA survived, and `sections_with_lsda` marks the
/// `.gcc_except_table` rather than the function -- so the function stays
/// eligible for identical code folding despite having a handler, which is the
/// exact miscompile that exclusion exists to prevent. lld sorts the
/// relocations by offset before reading `firstRelocation` for the same reason.
///
/// `pieces` is ascending and non-overlapping, so each entry is placed by a
/// binary search; a relocation inside no record is ignored. Selecting the
/// minimum is one comparison per relocation, so nothing needs sorting.
fn first_reloc_syms(pieces: &[Piece], entries: &[Rela64]) -> Vec<Option<u32>> {
    let mut first: Vec<Option<(u64, u32)>> = vec![None; pieces.len()];
    for r in entries {
        let at = r.r_offset.get();
        let Some(i) = record_at(pieces, at) else {
            continue;
        };
        if let Some(slot) = first.get_mut(i)
            && slot.is_none_or(|(prev, _)| at < prev)
        {
            *slot = Some((at, r.sym()));
        }
    }
    first.into_iter().map(|s| s.map(|(_, sym)| sym)).collect()
}

/// Walks one `.eh_frame` slice, replacing `pieces` with each record's
/// `(offset, length)` and `fde` with whether that record is an FDE, and
/// returns the offset the walk stopped at.
///
/// The buffers are caller-owned so a pass over many sections reuses them. The
/// walk ends at the first word it cannot decode as a record length: the zero
/// terminator, a 64-bit DWARF length, or a record running past the end of the
/// section. The records decoded so far stand, and everything from the returned
/// offset on belongs to no record. The caller decides what that tail means:
/// the splitter accepts it only when it is all zero bytes, which is what a
/// terminator and its padding are.
pub fn records(
    data: &[u8],
    pieces: &mut Vec<Piece>,
    fde: &mut Vec<bool>,
) -> usize {
    pieces.clear();
    fde.clear();
    let mut cursor = 0usize;
    while cursor + 4 <= data.len() {
        let Ok(len) = read_u32(data, cursor) else {
            break;
        };
        if len == 0 || len == 0xffff_ffff {
            break;
        }
        let Ok(len_usize) = usize::try_from(len) else {
            break;
        };
        let Some(next) =
            cursor.checked_add(4).and_then(|x| x.checked_add(len_usize))
        else {
            break;
        };
        if next > data.len() {
            break;
        }
        let Ok(cie_ptr) = read_u32(data, cursor + 4) else {
            break;
        };
        pieces.push(Piece {
            in_off: u64::try_from(cursor).unwrap_or(u64::MAX),
            len: u64::try_from(next.saturating_sub(cursor)).unwrap_or(0),
            out_off: None,
            fde: cie_ptr != 0,
        });
        fde.push(cie_ptr != 0);
        cursor = next;
    }
    cursor
}

/// The `.eh_frame_hdr` byte size for `fde_count` table entries.
pub fn hdr_size(fde_count: u64) -> u64 {
    HDR_FIXED.saturating_add(fde_count.saturating_mul(ENTRY_SIZE))
}

/// Whether `offset` falls inside an FDE record of the walk [`records`] made.
///
/// `pieces` is ascending and non-overlapping, so this is a binary search. An
/// offset inside no record at all (a malformed section) reads as a CIE, the
/// conservative answer for the collector.
pub fn is_fde_offset(pieces: &[Piece], fde: &[bool], offset: u64) -> bool {
    record_at(pieces, offset)
        .and_then(|i| fde.get(i).copied())
        .unwrap_or(false)
}

/// Builds the `.eh_frame_hdr` bytes from the relocated `.eh_frame` image.
///
/// `image` is the full writer buffer; the relevant slice is the placed
/// `.eh_frame` region. The FDE `initial_location` slots have already been
/// patched by the writer's relocation pass, so each holds the value the
/// runtime unwinder will decode.
pub fn build_hdr(image: &[u8], layout: &Layout) -> Result<Vec<u8>> {
    let eh = layout.region(Sect::EhFrame);
    let hdr = layout.region(Sect::EhFrameHdr);
    let start = usize::try_from(eh.offset)
        .map_err(|_| Error::OutOfRange("eh_frame off"))?;
    let size = usize::try_from(eh.size)
        .map_err(|_| Error::OutOfRange("eh_frame size"))?;
    let bytes = image
        .get(start..start.saturating_add(size))
        .ok_or(Error::OutOfRange("eh_frame slice"))?;

    let mut entries = collect_entries(bytes, eh.vaddr)?;
    // Drop FDEs that cover no placed code. The splitter already removes the
    // records whose function was collected, so this is a backstop: a row whose
    // initial_location resolved against an unplaced section would sit at the
    // front of the binary-search table and claim the address range below the
    // first real function.
    //
    // Every placed code region counts, not `.text` alone: `.init` and `.fini`
    // are output sections of their own, and an FDE covering either is as real
    // as one covering `.text`. Testing `.text` only would drop it from the
    // search table, leaving the unwinder unable to find it.
    let mut code = [Region::default(); sect::CONTENT.len()];
    let n = collect_code(layout, &mut code);
    let placed = &code[..n];
    entries.retain(|e| {
        placed
            .iter()
            .any(|r| e.pc >= r.vaddr && e.pc < r.vaddr.wrapping_add(r.size))
    });
    // Sort by covered PC so the table is a valid binary-search index. The
    // sort is stable and the input order is fixed by the serial record walk
    // above, so the result is deterministic at any thread count; the fan-out
    // matters because this runs on the writer's critical path, after the
    // copy pass patched the slots and before the header can be written.
    entries.par_sort_by_key(|e| e.pc);
    // One row per covered PC. Two would describe the same bytes with, in
    // general, different LSDA pointers, and the unwinder would take whichever
    // the binary search happened to land on.
    //
    // [`split::keep_flags`] drops most of the records that could pair up, but
    // it cannot drop all of them: it resolves an FDE's first relocation
    // through the global symbol table, so a retired weak or COMDAT copy's FDE
    // resolves to the *prevailing* section and looks live. The invariant is
    // therefore restored here as well as defended there. The sort is stable,
    // so the survivor is the record earliest in input order, and the choice is
    // a function of the inputs rather than of the sort.
    //
    // lld does exactly this, sorting stably and then uniquing, with a comment
    // naming the same ICF case
    // (`lld/ELF/SyntheticSections.cpp`).
    entries.dedup_by_key(|e| e.pc);

    Ok(serialize_hdr(&entries, eh.vaddr, hdr.vaddr))
}

/// Fills `out` with every placed code region and returns how many it wrote.
///
/// Empty regions are skipped: a link with no `.init` places none, and a zeroed
/// region would otherwise claim the address range at zero.
fn collect_code(layout: &Layout, out: &mut [Region]) -> usize {
    let mut n = 0usize;
    for sect in sect::code() {
        let region = layout.region(sect);
        if region.size == 0 {
            continue;
        }
        let Some(slot) = out.get_mut(n) else {
            break;
        };
        *slot = region;
        n += 1;
    }
    n
}

/// One row of the `.eh_frame_hdr` table: the FDE's covered PC and the FDE's
/// offset within `.eh_frame`.
#[derive(Clone, Copy)]
struct Entry {
    /// Runtime address of the first instruction the FDE covers.
    pc: u64,
    /// Runtime address of the FDE record itself. Both are stored absolute and
    /// converted to `datarel` offsets when the table is serialised.
    fde_addr: u64,
}

/// Walks the merged `.eh_frame`, decoding each FDE's covered PC.
///
/// CIEs always precede their FDEs in the section, and the `CIE_pointer` is a
/// backwards offset from the FDE, so a single forward scan can record each
/// CIE's FDE-address encoding (`R` augmentation value) as it passes and look
/// it up by offset when the FDE arrives.
///
/// The walk stops at the first zero `length`, which the splitter guarantees is
/// the single terminator closing the section. The runtime unwinder walks the
/// same bytes the same way whenever it has to scan linearly, so a terminator
/// anywhere else would cost both this table and the unwinder every record
/// behind it.
fn collect_entries(bytes: &[u8], eh_vaddr: u64) -> Result<Vec<Entry>> {
    let mut cies: FxHashMap<usize, u8> = FxHashMap::default();
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor + 4 <= bytes.len() {
        let len = read_u32(bytes, cursor)?;
        if len == 0 {
            break;
        }
        if len == 0xffff_ffff {
            return Err(Error::Format("64-bit .eh_frame length"));
        }
        let len_usize =
            usize::try_from(len).map_err(|_| Error::Format("eh_frame len"))?;
        let next = cursor
            .checked_add(4)
            .and_then(|x| x.checked_add(len_usize))
            .ok_or(Error::Format("eh_frame length overflow"))?;
        if next > bytes.len() {
            return Err(Error::Format("truncated .eh_frame"));
        }
        let cie_ptr = read_u32(bytes, cursor + 4)?;
        if cie_ptr == 0 {
            let enc = cie_augmentation(bytes, cursor, next)?.fde_encoding;
            cies.insert(cursor, enc);
        } else {
            // CIE_pointer is the offset from the field back to the CIE.
            let field_at = cursor.saturating_add(4);
            let cie_off = field_at
                .checked_sub(
                    usize::try_from(cie_ptr)
                        .map_err(|_| Error::Format("cie ptr"))?,
                )
                .ok_or(Error::Format("FDE CIE pointer"))?;
            // A CIE this FDE points at that was never read is not a
            // missing default: the augmentation string of that CIE is what
            // says how the FDE spells its address, and without it the bytes
            // read here mean nothing. Assuming `absptr` produced a garbage
            // PC that the `retain` filter below then dropped, so the
            // function disappeared from the search table and a `throw`
            // through it terminated the program with no diagnostic.
            let enc = cies.get(&cie_off).copied().ok_or(Error::Format(
                "FDE points at a CIE this section does not contain",
            ))?;
            let pc = fde_pc(bytes, cursor, enc, eh_vaddr)?;
            out.push(Entry {
                pc,
                fde_addr: eh_vaddr
                    .wrapping_add(u64::try_from(cursor).unwrap_or(0)),
            });
        }
        cursor = next;
    }
    Ok(out)
}

/// Decodes the `initial_location` of the FDE at `fde_off`.
///
/// `enc` is the CIE's `R` augmentation value: its low nibble selects the value
/// width, bits 4 to 6 select the base, and bit 7 (`DW_EH_PE_indirect`) says
/// the decoded value is the address of a slot holding the real one. For
/// `DW_EH_PE_pcrel` the stored value is `func_addr - field_addr`, so the
/// field's runtime address is added back to recover `func_addr`. For
/// `DW_EH_PE_absptr` the stored value is the address directly.
///
/// The base occupies three bits, not four. Masking `0xf0` folded the indirect
/// bit into the comparison, so `indirect|pcrel` (0x90) missed the pcrel arm
/// and was read as an absolute address -- as was every base this function does
/// not implement. The resulting PC is nowhere near the function, and the
/// caller's range filter then drops the record: the function vanishes from the
/// `.eh_frame_hdr` search table and a `throw` through it terminates, silently.
///
/// Anything else is refused. lld draws the line in the same place, masking
/// `0x70` and reporting "unknown FDE size relative encoding" for a base it has
/// no formula for (`lld/ELF/SyntheticSections.cpp`).
fn fde_pc(bytes: &[u8], fde_off: usize, enc: u8, eh_vaddr: u64) -> Result<u64> {
    // The initial_location field begins 8 bytes into the FDE record (after the
    // 4-byte length and the 4-byte CIE_pointer).
    let field_at = fde_off.saturating_add(8);
    let raw = read_encoded(bytes, field_at, enc & 0x0f)?;
    let field_va = eh_vaddr.wrapping_add(u64::try_from(field_at).unwrap_or(0));
    match enc & DW_EH_PE_BASE_MASK {
        DW_EH_PE_ABSPTR => Ok(raw),
        DW_EH_PE_PCREL => Ok(raw.wrapping_add(field_va)),
        _ => Err(Error::Format("unknown FDE initial_location base encoding")),
    }
}

/// What one CIE's augmentation string declares.
#[derive(Copy, Clone, Debug, Default)]
struct Augmentation {
    /// The `R` value: how the FDEs sharing this CIE encode their
    /// `initial_location`. `DW_EH_PE_absptr` when the string names none.
    fde_encoding: u8,
    /// Whether the string carries an `L`, so every FDE sharing this CIE has an
    /// LSDA pointer in its augmentation data. lld asks exactly this question
    /// in `EhReader::hasLSDA` (`lld/ELF/EhFrame.cpp`), and asks it
    /// of the CIE rather than of the FDE: the pointer's presence is a
    /// property of the record format the CIE fixes.
    has_lsda: bool,
}

/// Reads the augmentation string of the CIE whose record starts at `cie_off`
/// and reports everything it declares.
///
/// The string is a sequence of single-character entries whose data is stored
/// positionally, not in tag-length-value form, so the only way past one entry
/// is to know its width: `z` carries an aug-data length, `P` a personality
/// routine (encoding byte plus value), `L` an LSDA encoding byte, `R` the FDE
/// encoding. `B`, `S` and `G` carry nothing.
///
/// lld reads the same string twice, once per question (`getFdeEncoding` and
/// `hasLSDA`, both in `lld/ELF/EhFrame.cpp`), with the skip arms spelled
/// out in each. Both answers come from one walk here, so the two cannot
/// disagree about how far into the record a given entry sits.
///
/// A character outside that set ends the walk, because its width is unknown
/// and everything after it would be read at the wrong offset -- lld reports
/// "unknown `.eh_frame` augmentation string" and breaks out at the same point.
/// The two answers then take opposite defaults, which is the conservative
/// direction for each: no `R` entry was reached, so the encoding is
/// `DW_EH_PE_absptr`, the value the ABI assigns when none is given; and an `L`
/// may well sit behind the character that stopped the walk, so `has_lsda` is
/// true, which keeps the sections those FDEs describe out of identical code
/// folding.
fn cie_augmentation(
    bytes: &[u8],
    cie_off: usize,
    end: usize,
) -> Result<Augmentation> {
    let mut cur = Cursor::new(bytes, cie_off, end);
    cur.skip(8)?; // length + CIE_id
    let version = cur.read_byte()?;
    let aug = cur.read_cstr()?;
    cur.skip_leb128()?; // code_align
    cur.skip_leb128()?; // data_align
    // Return-address register: one byte in CIE v1, ULEB128 in v3.
    if version >= 3 {
        cur.skip_leb128()?;
    } else {
        cur.skip(1)?;
    }
    let mut out = Augmentation::default();
    for ch in aug {
        match ch {
            b'z' => cur.skip_leb128()?,
            b'L' => {
                out.has_lsda = true;
                cur.skip(1)?;
            }
            b'P' => cur.skip_aug_p()?,
            b'R' => out.fde_encoding = cur.read_byte()?,
            // 'B' (B-key pointer), 'S' (signal handler), 'G' (PAC): no data.
            b'B' | b'S' | b'G' => {}
            _ => {
                out.has_lsda = true;
                break;
            }
        }
    }
    Ok(out)
}

/// A bounds-checked cursor over one CIE record's payload.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    end: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], pos: usize, end: usize) -> Self {
        Self { bytes, pos, end }
    }

    fn read_byte(&mut self) -> Result<u8> {
        if self.pos >= self.end {
            return Err(Error::Format("CIE truncated"));
        }
        let b = self.bytes[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn skip(&mut self, n: usize) -> Result<()> {
        let new = self
            .pos
            .checked_add(n)
            .filter(|&x| x <= self.end)
            .ok_or(Error::Format("CIE truncated"))?;
        self.pos = new;
        Ok(())
    }

    fn read_cstr(&mut self) -> Result<&'a [u8]> {
        let end = self.bytes[self.pos..self.end]
            .iter()
            .position(|&b| b == 0)
            .ok_or(Error::Format("CIE augmentation not terminated"))?;
        let s = &self.bytes[self.pos..self.pos + end];
        self.pos += end + 1;
        Ok(s)
    }

    fn skip_leb128(&mut self) -> Result<()> {
        while self.pos < self.end {
            let b = self.bytes[self.pos];
            self.pos += 1;
            if b & 0x80 == 0 {
                return Ok(());
            }
        }
        Err(Error::Format("CIE truncated LEB128"))
    }

    fn skip_aug_p(&mut self) -> Result<()> {
        let enc = self.read_byte()?;
        if enc & DW_EH_PE_BASE_MASK == DW_EH_PE_ALIGNED {
            // `DW_EH_PE_aligned` does not name a smaller value: it says the
            // value is laid down at an address rounded up to its own size,
            // so its width is only known against the record's runtime
            // address, which a linker never has. Skipping nothing (the old
            // answer) desynchronised the cursor -- every later entry,
            // `R` included, was read at the wrong offset and the FDE
            // encoding came out as whatever byte happened to sit there.
            // Refusing keeps the failure loud; nothing in practice emits
            // the form.
            return Err(Error::Format(
                "DW_EH_PE_aligned personality encoding is not supported",
            ));
        }
        let n = aug_value_size(enc)?;
        self.skip(n)
    }
}

/// The on-disk byte width of one `DW_EH_PE`-encoded value (low nibble only).
fn aug_value_size(enc: u8) -> Result<usize> {
    Ok(match enc & 0x0f {
        DW_EH_PE_ABSPTR | DW_EH_PE_UDATA8 | DW_EH_PE_SDATA8 => 8,
        DW_EH_PE_UDATA2 | DW_EH_PE_SDATA2 => 2,
        DW_EH_PE_UDATA4 | DW_EH_PE_SDATA4 => 4,
        _ => return Err(Error::Format("unknown DW_EH_PE format")),
    })
}

/// Reads a `DW_EH_PE`-encoded value at `offset`, sign-extending the signed
/// forms to `u64`.
fn read_encoded(bytes: &[u8], offset: usize, format: u8) -> Result<u64> {
    match format {
        DW_EH_PE_UDATA2 => Ok(u64::from(read_u16(bytes, offset)?)),
        DW_EH_PE_SDATA2 => {
            Ok(sign_extend(u64::from(read_u16(bytes, offset)?), 16))
        }
        DW_EH_PE_UDATA4 => Ok(u64::from(read_u32(bytes, offset)?)),
        DW_EH_PE_SDATA4 => {
            Ok(sign_extend(u64::from(read_u32(bytes, offset)?), 32))
        }
        DW_EH_PE_UDATA8 | DW_EH_PE_SDATA8 | DW_EH_PE_ABSPTR => {
            read_u64(bytes, offset)
        }
        _ => Err(Error::Format("unknown FDE value format")),
    }
}

/// Sign-extends the `bits`-wide value in `v` to a full `u64`.
fn sign_extend(v: u64, bits: u32) -> u64 {
    if bits >= 64 || v & (1u64 << (bits - 1)) == 0 {
        v
    } else {
        v | (!0u64 << bits)
    }
}

/// Serialises the `.eh_frame_hdr`: a 4-byte header, the PC-relative
/// `.eh_frame` pointer, the FDE count, and the `(pc, fde_offset)` table sorted
/// by PC. All table values are encoded `DW_EH_PE_datarel | DW_EH_PE_sdata4`,
/// i.e. signed 32-bit offsets from the `.eh_frame_hdr` load address.
fn serialize_hdr(entries: &[Entry], eh_vaddr: u64, hdr_vaddr: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        usize::try_from(HDR_FIXED).unwrap_or(0) + entries.len() * 8,
    );
    out.push(1); // version
    out.push(DW_EH_PE_PCREL | DW_EH_PE_SDATA4); // eh_frame_ptr_enc
    out.push(DW_EH_PE_UDATA4); // fde_count_enc
    out.push(DW_EH_PE_DATAREL | DW_EH_PE_SDATA4); // table_enc
    // `eh_frame_ptr` is PC-relative to this 4-byte field (header offset 4).
    push_rel32(&mut out, eh_vaddr.wrapping_sub(hdr_vaddr + 4));
    push_u32(&mut out, u32::try_from(entries.len()).unwrap_or(0));
    for e in entries {
        // datarel to the .eh_frame_hdr start.
        push_rel32(&mut out, e.pc.wrapping_sub(hdr_vaddr));
        push_rel32(&mut out, e.fde_addr.wrapping_sub(hdr_vaddr));
    }
    out
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let slot = bytes
        .get(offset..offset + 4)
        .ok_or(Error::Format("u32 bounds"))?;
    Ok(u32::from_le_bytes(
        slot.try_into().map_err(|_| Error::Format("u32 slice"))?,
    ))
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let slot = bytes
        .get(offset..offset + 2)
        .ok_or(Error::Format("u16 bounds"))?;
    Ok(u16::from_le_bytes(
        slot.try_into().map_err(|_| Error::Format("u16 slice"))?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let slot = bytes
        .get(offset..offset + 8)
        .ok_or(Error::Format("u64 bounds"))?;
    Ok(u64::from_le_bytes(
        slot.try_into().map_err(|_| Error::Format("u64 slice"))?,
    ))
}
