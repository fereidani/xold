//! Static TLS (local-exec) layout: the thread-pointer-relative offset
//! arithmetic shared by every architecture.
//!
//! A static executable carries its TLS template (`.tdata` for initialised
//! thread-locals, `.tbss` for zero-init ones) inside a `PT_TLS` program
//! header. At load time the runtime allocates one per-thread copy and points
//! the thread pointer (TP) at a fixed position relative to it. A local-exec
//! relocation stores the offset of a variable from the TP (`TPOFF`), so the
//! linker must convert each TLS symbol's position within the template into a
//! TP-relative offset following the per-arch TP convention.
//!
//! The conventions and the formulas below mirror lld's `getTlsTpOffset`
//! (`lld/ELF/InputSection.cpp`), which is the authoritative reference:
//!
//! * Variant 2 (x86-64): the static TLS block sits immediately below the TP, so
//!   a variable's `TPOFF` is typically negative. The block is padded so that
//!   `TP - p_memsz` is congruent to `p_vaddr` modulo `p_align`.
//! * Variant 1 (`AArch64`): the TP is followed by a two-word gap (16 bytes on
//!   `AArch64`), then alignment padding, then the block; offsets are
//!   non-negative.
//! * Variant 1 (RISC-V): like `AArch64` but with no gap; the TP is followed by
//!   alignment padding then the block.
//!
//! `p_vaddr`, `p_memsz` and `p_align` are the `PT_TLS` fields. Because xold
//! uses an identity map with page-aligned placement, `p_vaddr` is a multiple
//! of `p_align` in practice, but the formulas are applied verbatim so the
//! result matches lld for any placement.

use crate::reloc::Target;

/// The laid-out static TLS block, mirroring the fields of a `PT_TLS` program
/// header.
///
/// `offset`/`vaddr` locate the start of `.tdata` (or `.tbss` when no `.tdata`
/// is present); `filesz` is the file-backed size and `memsz` the in-memory size
/// including `.tbss`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TlsBlock {
    /// File offset of the block's start.
    pub offset: u64,
    /// Virtual address of the block's start (`p_vaddr`).
    pub vaddr: u64,
    /// File-backed size (`p_filesz`): the `.tdata` bytes.
    pub filesz: u64,
    /// In-memory size (`p_memsz`): `.tdata` plus `.tbss` with padding.
    pub memsz: u64,
    /// Block alignment (`p_align`): the max of `.tdata`/`.tbss` alignment.
    pub align: u64,
}

impl TlsBlock {
    /// Whether the block carries any TLS content.
    pub const fn is_empty(self) -> bool {
        self.memsz == 0
    }
}

/// The thread-pointer-relative offset (`TPOFF`) for a TLS symbol, following
/// the per-arch TP convention.
///
/// The arithmetic is modular (wrapping) to match lld's unsigned computation;
/// callers that need a signed interpretation (e.g. x86-64 `TPOFF32`)
/// reinterpreting the result when it is stored. `sym_off` is the symbol's
/// offset from the block start (`sym_vaddr - block.vaddr`).
pub fn tpoff(target: Target, sym_off: u64, block: &TlsBlock) -> u64 {
    let p_vaddr = block.vaddr;
    let p_memsz = block.memsz;
    // `align` is at least 1 by construction, so `align - 1` is a safe mask.
    let align_mask = block.align.saturating_sub(1);
    match target {
        // Variant 2: block below TP. Pad so that
        // `TP - p_memsz == p_vaddr (mod p_align)`, i.e. the padding is
        // `(-p_vaddr - p_memsz) & (p_align - 1)`.
        Target::X86_64 => {
            let pad = p_vaddr.wrapping_neg().wrapping_sub(p_memsz) & align_mask;
            sym_off.wrapping_sub(p_memsz).wrapping_sub(pad)
        }
        // Variant 1 with a two-word (16-byte) gap between TP and the block.
        Target::AArch64 => {
            let pad = p_vaddr.wrapping_sub(16) & align_mask;
            sym_off.wrapping_add(16).wrapping_add(pad)
        }
        // Variant 1 with no gap.
        Target::Riscv64 => sym_off.wrapping_add(p_vaddr & align_mask),
    }
}

/// The offset of a TLS symbol within its own module's block (`DTPOFF`), given
/// the thread-pointer-relative offset [`tpoff`] produced for it in `block`.
///
/// Both describe the same storage from different origins, so this is exactly
/// the inverse of that function: the general-dynamic model stores a
/// module-relative offset in the GOT, while every direct access uses a
/// thread-pointer-relative one.
pub fn block_offset(target: Target, tp: u64, block: &TlsBlock) -> u64 {
    let p_vaddr = block.vaddr;
    let p_memsz = block.memsz;
    let align_mask = block.align.saturating_sub(1);
    match target {
        Target::X86_64 => {
            let pad = p_vaddr.wrapping_neg().wrapping_sub(p_memsz) & align_mask;
            tp.wrapping_add(p_memsz).wrapping_add(pad)
        }
        Target::AArch64 => {
            let pad = p_vaddr.wrapping_sub(16) & align_mask;
            tp.wrapping_sub(16).wrapping_sub(pad)
        }
        Target::Riscv64 => {
            let pad = p_vaddr & align_mask;
            tp.wrapping_sub(pad)
        }
    }
}
