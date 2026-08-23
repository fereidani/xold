//! The procedure linkage table: the trampolines an image calls a function
//! through when the address to jump to is not fixed at link time.
//!
//! Two things need one. A function imported from a shared object is bound by
//! the loader, and an indirect function (`STT_GNU_IFUNC`) this image defines
//! is whatever its resolver returns when it runs. Both get one stub in `.plt`,
//! one 8-byte slot in `.got.plt` for the stub to jump through, and one
//! `.rela.plt` entry describing how the slot is filled.
//!
//! For an import the slot starts out holding a value that routes the first
//! call back into the resolver stub `PLT[0]`, which hands control to the
//! runtime resolver (`_dl_runtime_resolve`) with enough context to patch the
//! slot. Subsequent calls jump straight to the resolved address. An indirect
//! function is never bound lazily: whoever applies the `R_*_IRELATIVE` entry
//! (the loader, or a static image's own startup) calls the resolver and stores
//! what it returns before the first call reaches the stub.
//!
//! The stub bytes and the resolver-identification convention are
//! architecture-specific and follow lld:
//!
//! - x86-64: a 16-byte `PLT[0]` (`push GOT[1]; jmp *GOT[2]; nopl`) and 16-byte
//!   per-entry stubs (`jmp *GOT.PLT[n]; push reloc_idx; jmp PLT[0]`). The
//!   initial slot value is `PLT[n] + 6` (the address of the `push`), so the
//!   first jump falls through to `PLT[0]` with the entry's index pushed.
//! - `AArch64`: a 32-byte `PLT[0]` (`stp x16,x30; adrp x16,page(GOT[2])`; `ldr
//!   x17,[x16,lo12(GOT[2])]; add x16,x16,lo12(GOT[2]); br x17; nop; nop; nop`)
//!   and 16-byte per-entry stubs (`adrp x16,page(GOT.PLT[n]); ldr x17,
//!   [x16,lo12]; add x16,x16,lo12; br x17`). The initial slot value is `PLT[0]`
//!   (the resolver stub), so the loader receives `x16` = the slot's address to
//!   identify the entry.
//! - `RISC-V`: a 32-byte `PLT[0]` and 16-byte per-entry `auipc`/`ld`/`jalr`
//!   stubs. The initial slot value is `PLT[0]`; the resolver reads the entry
//!   index from `t1`, computed in the stub and adjusted in `PLT[0]`.
//!
//! `.got.plt` carries the `SysV` three header slots on x86-64/`AArch64`
//! (`_DYNAMIC`, `link_map`, resolver) and two on RISC-V (resolver,
//! `link_map`), matching lld's `gotPltHeaderEntriesNum`.
//!
//! [`build_plan`] produces the three synthetic sections (`.plt`, `.got.plt`,
//! `.rela.plt`) from the placed layout and the dynamic symbol table. Layout
//! has already fixed every region's offset and address; this module only
//! serialises bytes.

use crate::{
    dynamic::DynamicPlan,
    elf::Rela64,
    endian::{I64, U64},
    error::{Error, Result},
    layout::{Layout, Sect},
    reloc::Target,
    symbol::SymbolId,
    util::push_rel32,
};

/// Bytes in one GOT entry.
const GOT_ENTRY: u64 = 8;

/// The serialised PLT, GOT.PLT and `.rela.plt` bytes for one image. Regions
/// and section indices live on [`Layout`]; this struct carries only the bytes
/// the writer copies into them.
pub struct PltPlan {
    /// `.plt`: the `PLT[0]` trampoline followed by one stub per PLT entry.
    pub plt: Vec<u8>,
    /// `.got.plt`: the loader-owned header slots, then one slot per PLT entry
    /// pre-filled with the value the lazy-binding convention requires.
    pub got_plt: Vec<u8>,
    /// `.rela.plt`: one `R_*_JUMP_SLOT` per import and one `R_*_IRELATIVE`
    /// per indirect function this image defines.
    pub rela_plt: Vec<Rela64>,
}

/// Builds the PLT plan from the placed layout, reading dynsym indices from
/// the dynamic plan.
///
/// Call after [`crate::dynamic::build_plan`] so the dynamic symbol table and
/// every region are in their final state.
pub fn build_plan(
    layout: &Layout,
    dynamic: Option<&DynamicPlan>,
) -> Result<PltPlan> {
    let plt = build_plt_bytes(layout);
    let got_plt = build_got_plt_bytes(layout);
    let rela_plt = build_rela_plt(layout, dynamic)?;
    Ok(PltPlan {
        plt,
        got_plt,
        rela_plt,
    })
}

/// Builds the `.plt` bytes for `layout.target`: the `PLT[0]` resolver
/// trampoline plus one lazy stub per import. Dispatches to the per-arch
/// builder so the stub bytes match what the runtime loader expects.
fn build_plt_bytes(layout: &Layout) -> Vec<u8> {
    match layout.target {
        Target::X86_64 => build_plt_bytes_x86_64(layout),
        Target::AArch64 => build_plt_bytes_aarch64(layout),
        Target::Riscv64 => build_plt_bytes_riscv(layout),
    }
}

/// Builds the `.got.plt` bytes for `layout.target`: the loader-owned header
/// slots followed by one slot per import, pre-filled with the value the
/// lazy-binding convention requires (x86-64: the per-entry `push` address;
/// AArch64/RISC-V: the address of `PLT[0]`).
fn build_got_plt_bytes(layout: &Layout) -> Vec<u8> {
    match layout.target {
        Target::X86_64 | Target::AArch64 => build_got_plt_bytes_3hdr(layout),
        Target::Riscv64 => build_got_plt_bytes_2hdr(layout),
    }
}

/// Builds `.rela.plt`: one entry per PLT entry, in allocation order, each
/// naming the `.got.plt` slot its stub jumps through.
///
/// The split is per entry, not per image kind, and the two kinds share the one
/// table. An indirect function this image defines takes an `R_*_IRELATIVE`
/// carrying its resolver as the addend and naming no symbol: what the slot
/// must hold is whatever the resolver returns, which no symbol can stand for.
/// An imported function takes an `R_*_JUMP_SLOT` naming its `.dynsym` entry,
/// which the loader binds to the definition it finds.
///
/// Both kinds reach the runtime the same way. A dynamic image hands the table
/// to the loader through `DT_JMPREL`, which applies an `IRELATIVE` there by
/// calling the resolver rather than by resolving a name. A static image has no
/// loader and no dynamic symbol table, so its own startup applies the table
/// (see [`crate::defsym`]) and an import can never reach it.
fn build_rela_plt(
    layout: &Layout,
    dyn_plan: Option<&DynamicPlan>,
) -> Result<Vec<Rela64>> {
    let got_plt = layout.region(Sect::GotPlt).vaddr;
    let reserved = layout.target.plt_spec().got_plt_reserved;
    let mut out = Vec::with_capacity(layout.plt_keys.len());
    for (i, key) in layout.plt_keys.iter().enumerate() {
        let sym = key
            .global_id()
            .ok_or(Error::Format("PLT entry is not a global symbol"))?;
        let slot = got_plt
            .wrapping_add(reserved * GOT_ENTRY)
            .wrapping_add(u64::try_from(i).unwrap_or(0) * GOT_ENTRY);
        out.push(if layout.is_ifunc(sym) {
            irelative(layout, slot, sym)
        } else {
            jump_slot(dyn_plan, slot, sym)?
        });
    }
    Ok(out)
}

/// The `R_*_IRELATIVE` entry for an indirect function's slot: the resolver
/// address in the addend is the whole relocation.
fn irelative(layout: &Layout, slot: u64, sym: SymbolId) -> Rela64 {
    Rela64 {
        r_offset: U64::new(slot),
        r_info: U64::new(u64::from(layout.target.dyn_relocs().irelative)),
        r_addend: I64::new(layout.ifunc_resolver(sym).cast_signed()),
    }
}

/// The `R_*_JUMP_SLOT` entry for an imported function's slot, naming the
/// import's `.dynsym` index.
///
/// Fails when the link has no dynamic symbol table, or has one that does not
/// record this import: either would mean the layout scan and the dynamic plan
/// disagreed on which symbols are imports.
fn jump_slot(
    dyn_plan: Option<&DynamicPlan>,
    slot: u64,
    sym: SymbolId,
) -> Result<Rela64> {
    let plan = dyn_plan
        .ok_or(Error::Format("PLT entry for an import in a static image"))?;
    let sym_idx = plan.import_index(sym).ok_or(Error::Format(
        "PLT symbol has no dynamic symbol table entry",
    ))?;
    Ok(Rela64 {
        r_offset: U64::new(slot),
        r_info: U64::new(
            (u64::from(sym_idx) << 32) | u64::from(plan.relocs.jump_slot),
        ),
        r_addend: I64::new(0),
    })
}

// === x86-64 ===============================================================

/// PLT[0]/entry size in bytes (x86-64).
const X86_64_PLT: u64 = 16;

/// Builds the x86-64 `.plt` bytes: the 16-byte `PLT[0]` resolver trampoline
/// (`push GOT[1]; jmp *GOT[2]; nopl`) followed by one 16-byte lazy stub per
/// import (`jmp *GOT.PLT[i]; push reloc_idx; jmp PLT[0]`).
fn build_plt_bytes_x86_64(layout: &Layout) -> Vec<u8> {
    let p0 = layout.region(Sect::Plt).vaddr;
    let got_plt = layout.region(Sect::GotPlt).vaddr;
    let got1 = got_plt.wrapping_add(GOT_ENTRY);
    let got2 = got_plt.wrapping_add(2 * GOT_ENTRY);
    let mut out = Vec::with_capacity(plt_capacity(layout));
    // PLT[0]: push GOT[1]; jmp *GOT[2]; nopl.
    out.extend_from_slice(&[0xff, 0x35]);
    push_rel32(&mut out, got1.wrapping_sub(p0.wrapping_add(6)));
    out.extend_from_slice(&[0xff, 0x25]);
    push_rel32(&mut out, got2.wrapping_sub(p0.wrapping_add(12)));
    out.extend_from_slice(&[0x0f, 0x1f, 0x40, 0x00]);
    // PLT[1+i]: jmp *GOT.PLT[i]; push reloc_idx; jmp PLT[0].
    for (i, _key) in layout.plt_keys.iter().enumerate() {
        let i64 = u64::try_from(i).unwrap_or(0);
        let pn = p0.wrapping_add(X86_64_PLT).wrapping_add(i64 * X86_64_PLT);
        let slot = got_plt
            .wrapping_add(3 * GOT_ENTRY)
            .wrapping_add(i64 * GOT_ENTRY);
        out.extend_from_slice(&[0xff, 0x25]);
        push_rel32(&mut out, slot.wrapping_sub(pn.wrapping_add(6)));
        out.extend_from_slice(&[0x68]);
        push_rel32(&mut out, i64);
        out.extend_from_slice(&[0xe9]);
        push_rel32(&mut out, p0.wrapping_sub(pn.wrapping_add(16)));
    }
    out
}

/// Builds the x86-64/`AArch64` `.got.plt` bytes (three header slots).
/// `PLT[0]` addresses the initial per-entry slot value: x86-64 stores
/// `PLT[n] + 6` (the address of the entry's `push`), `AArch64` stores
/// `PLT[0]` (the resolver stub). The first two header slots are zero; the
/// loader fills them with the `link_map` pointer and the resolver address.
/// The third header slot is `_DYNAMIC` for x86-64 (per the psABI) and zero
/// for `AArch64` (matching lld, which leaves the default empty).
fn build_got_plt_bytes_3hdr(layout: &Layout) -> Vec<u8> {
    let mut out = Vec::with_capacity(got_plt_capacity(layout));
    let dynamic_addr = layout
        .dynamic
        .as_ref()
        .map_or(0, |p| p.regions.dynamic.vaddr);
    // x86-64 psABI: GOT[0] = _DYNAMIC. AArch64 leaves it zero (lld default).
    let got0 = if layout.target == Target::X86_64 {
        dynamic_addr
    } else {
        0
    };
    push_u64(&mut out, got0);
    push_u64(&mut out, 0); // GOT[1]: link_map, filled by the loader.
    push_u64(&mut out, 0); // GOT[2]: resolver, filled by the loader.
    let p0 = layout.region(Sect::Plt).vaddr;
    for (i, _key) in layout.plt_keys.iter().enumerate() {
        let i64 = u64::try_from(i).unwrap_or(0);
        // x86-64 routes the first call through PLT[0] via the per-entry push,
        // so the slot initially points at PLT[n] + 6. AArch64 routes via x16
        // (the slot address), so the slot initially points at PLT[0].
        let initial = match layout.target {
            Target::X86_64 => p0
                .wrapping_add(X86_64_PLT)
                .wrapping_add(i64 * X86_64_PLT)
                .wrapping_add(6),
            _ => p0,
        };
        push_u64(&mut out, initial);
    }
    out
}

// === `AArch64` ============================================================

/// Builds the `AArch64` `.plt` bytes: a 32-byte `PLT[0]` resolver trampoline
/// followed by one 16-byte lazy stub per import. Matches lld's
/// `AArch64::writePltHeader`/`writePlt`.
fn build_plt_bytes_aarch64(layout: &Layout) -> Vec<u8> {
    let p0 = layout.region(Sect::Plt).vaddr;
    let got_plt = layout.region(Sect::GotPlt).vaddr;
    // The resolver stub reads GOT.PLT[2] (offset 16) for `_dl_runtime_resolve`
    // and passes x16 = the per-entry GOT slot address so the loader can
    // identify the symbol.
    let resolver_slot = got_plt.wrapping_add(2 * GOT_ENTRY);
    let mut out = Vec::with_capacity(plt_capacity(layout));
    // PLT[0]: stp x16,x30,[sp,#-16]!; adrp x16,page(GOT[2]); ldr x17,[x16,
    // lo12(GOT[2])]; add x16,x16,lo12(GOT[2]); br x17; nop; nop; nop.
    out.extend_from_slice(&[0xf0, 0x7b, 0xbf, 0xa9]);
    // The adrp is the second instruction, so the page it is measured from is
    // the one holding `p0 + 4`, not the one holding `p0`.
    write_adrp(&mut out, p0.wrapping_add(4), resolver_slot);
    write_ldr_imm12(&mut out, resolver_slot);
    write_add_imm12(&mut out, resolver_slot);
    out.extend_from_slice(&[0x20, 0x02, 0x1f, 0xd6]);
    out.extend_from_slice(&[0x1f, 0x20, 0x03, 0xd5]);
    out.extend_from_slice(&[0x1f, 0x20, 0x03, 0xd5]);
    out.extend_from_slice(&[0x1f, 0x20, 0x03, 0xd5]);
    // PLT[1+i]: adrp x16,page(GOT.PLT[3+i]); ldr x17,[x16,lo12]; add x16,x16,
    // lo12; br x17.
    for (i, _key) in layout.plt_keys.iter().enumerate() {
        let i64 = u64::try_from(i).unwrap_or(0);
        let entry_addr = p0.wrapping_add(32).wrapping_add(i64 * 16);
        let slot = got_plt
            .wrapping_add(3 * GOT_ENTRY)
            .wrapping_add(i64 * GOT_ENTRY);
        write_adrp(&mut out, entry_addr, slot);
        write_ldr_imm12(&mut out, slot);
        write_add_imm12(&mut out, slot);
        out.extend_from_slice(&[0x20, 0x02, 0x1f, 0xd6]);
    }
    out
}

/// Writes an `adrp x16, page(sym) - page(place)` instruction (4 bytes) at the
/// end of `out`.
///
/// `place` is the address of the adrp itself, since that is the page the
/// difference is measured from. In `PLT[0]` the adrp is the second
/// instruction, so `place` is four bytes past the start of `.plt`. The 21-bit
/// immediate is split: `imm[0..2]` at bits [29..30] (`immlo`), `imm[2..21]`
/// at bits [5..23] (`immhi`). The opcode bits (`ADRP` = `1.immlo.10000`)
/// are fixed.
fn write_adrp(out: &mut Vec<u8>, place: u64, sym: u64) {
    // page(s_a) - page(p) stored as the page count (>> 12).
    let page_diff = (sym & !0xFFF).wrapping_sub(place & !0xFFF) >> 12;
    #[allow(clippy::cast_possible_truncation)]
    let imm = page_diff as u32 & 0x1F_FFFF;
    let imm_lo = (imm & 0x3) << 29;
    let imm_hi = (imm & 0x001F_FFFC) << 3;
    // 0x90 = ADRP opcode pattern with rd = x16 (bits [0..5] = 0x10).
    let insn: u32 = 0x9000_0010 | imm_lo | imm_hi;
    out.extend_from_slice(&insn.to_le_bytes());
}

/// Writes an `ldr x17, [x16, #lo12(sym)]` instruction (4 bytes) at the end of
/// `out`. The 12-bit unsigned offset is `(sym & 0xfff) >> 3`, scaled by the
/// 8-byte access size.
fn write_ldr_imm12(out: &mut Vec<u8>, sym: u64) {
    let disp = ((sym & 0xFFF) >> 3) as u32;
    // ldr x17, [x16, #disp]: 1111 1001 01 disp x16 x17
    let insn: u32 = 0xF940_0211 | ((disp & 0xFFF) << 10);
    out.extend_from_slice(&insn.to_le_bytes());
}

/// Writes an `add x16, x16, #lo12(sym)` instruction (4 bytes) at the end of
/// `out`. The 12-bit unsigned immediate is `sym & 0xfff`, unscaled.
fn write_add_imm12(out: &mut Vec<u8>, sym: u64) {
    let imm12 = (sym & 0xFFF) as u32;
    // add x16, x16, #imm12: 1001 0001 00 imm12 x16 x16
    let insn: u32 = 0x9100_0210 | (imm12 << 10);
    out.extend_from_slice(&insn.to_le_bytes());
}

// === RISC-V ===============================================================

/// Builds the RISC-V `.plt` bytes: a 32-byte `PLT[0]` resolver trampoline
/// followed by one 16-byte lazy stub per import. Matches lld's
/// `RISCV::writePltHeader`/`writePlt` (rv64 variant).
fn build_plt_bytes_riscv(layout: &Layout) -> Vec<u8> {
    let p0 = layout.region(Sect::Plt).vaddr;
    let got_plt = layout.region(Sect::GotPlt).vaddr;
    let mut out = Vec::with_capacity(plt_capacity(layout));
    // PLT[0]:
    //   1: auipc t2, %pcrel_hi(.got.plt)
    //      sub t1, t1, t3
    //      ld t3, %pcrel_lo(1b)(t2)         ; t3 = _dl_runtime_resolve
    //      addi t1, t1, -32-12              ; t1 = &.plt[i] - &.plt[0]
    //      addi t0, t2, %pcrel_lo(1b)       ; t0 = &.got.plt
    //      srli t1, t1, 1                   ; rv64: index by 8-byte slot
    //      ld t0, 8(t0)                     ; t0 = link_map
    //      jalr t3
    // The pcrel pair (1b at PLT[0]) computes the address of `.got.plt`; the
    // load picks up GOT[0] (resolver) and GOT[1] (link_map).
    let off = got_plt.wrapping_sub(p0);
    write_auipc(&mut out, XReg::T2, hi20_signed(off));
    write_rtype(&mut out, 0x33, 0, XReg::T1, XReg::T1, XReg::T3, 0x20);
    write_itype(&mut out, 0x03, 0x3, XReg::T3, XReg::T2, lo12_signed(off));
    write_itype(&mut out, 0x13, 0x0, XReg::T1, XReg::T1, -44);
    write_itype(&mut out, 0x13, 0x0, XReg::T0, XReg::T2, lo12_signed(off));
    write_itype_shift(&mut out, 0x13, 0x5, XReg::T1, XReg::T1, 1);
    write_itype(&mut out, 0x03, 0x3, XReg::T0, XReg::T0, 8);
    write_itype_jalr(&mut out, XReg::T3, XReg::Zero);
    // PLT[1+i]:
    //   1: auipc t3, %pcrel_hi(got.plt[n])
    //      ld t3, %pcrel_lo(1b)(t3)
    //      jalr t1, t3                       ; call resolver, return addr in t1
    //      nop
    for (i, _key) in layout.plt_keys.iter().enumerate() {
        let i64 = u64::try_from(i).unwrap_or(0);
        let entry_addr = p0.wrapping_add(32).wrapping_add(i64 * 16);
        let slot = got_plt
            .wrapping_add(2 * GOT_ENTRY)
            .wrapping_add(i64 * GOT_ENTRY);
        let off = slot.wrapping_sub(entry_addr);
        write_auipc(&mut out, XReg::T3, hi20_signed(off));
        write_itype(&mut out, 0x03, 0x3, XReg::T3, XReg::T3, lo12_signed(off));
        write_itype_jalr(&mut out, XReg::T3, XReg::T1);
        write_itype(&mut out, 0x13, 0x0, XReg::Zero, XReg::Zero, 0);
    }
    out
}

/// Builds the RISC-V `.got.plt` bytes (two loader-owned header slots followed
/// by per-import slots). The loader fills GOT[0] with the resolver address
/// and GOT[1] with the `link_map` pointer; each per-import slot starts out
/// pointing at `PLT[0]` so the first lazy call routes to the resolver.
fn build_got_plt_bytes_2hdr(layout: &Layout) -> Vec<u8> {
    let mut out = Vec::with_capacity(got_plt_capacity(layout));
    let p0 = layout.region(Sect::Plt).vaddr;
    push_u64(&mut out, 0); // GOT[0]: resolver, filled by the loader.
    push_u64(&mut out, 0); // GOT[1]: link_map, filled by the loader.
    for _ in &layout.plt_keys {
        push_u64(&mut out, p0);
    }
    out
}

/// A RISC-V integer register, encoded into the rd/rs1/rs2 fields.
#[derive(Copy, Clone)]
#[allow(non_camel_case_types, reason = "matches the ABI register names")]
enum XReg {
    Zero = 0,
    T0 = 5,
    T1 = 6,
    T2 = 7,
    T3 = 28,
}

/// Writes a U-type instruction (`lui`/`auipc`) at the end of `out`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "intentional bit-pattern reinterpret"
)]
fn write_auipc(out: &mut Vec<u8>, rd: XReg, imm20: i32) {
    let imm = (imm20 as u32 & 0xFFFFF) << 12;
    let opcode = 0x17; // auipc
    let insn = imm | (u32::from(rd as u8) << 7) | opcode;
    out.extend_from_slice(&insn.to_le_bytes());
}

/// Writes an I-type instruction (`addi`/etc.) at the end of `out`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "intentional bit-pattern reinterpret"
)]
fn write_itype(
    out: &mut Vec<u8>,
    opcode: u32,
    funct3: u32,
    rd: XReg,
    rs1: XReg,
    imm12: i32,
) {
    let imm = imm12 as u32 & 0xFFF;
    let insn = imm << 20
        | (u32::from(rs1 as u8) << 15)
        | (funct3 << 12)
        | (u32::from(rd as u8) << 7)
        | opcode;
    out.extend_from_slice(&insn.to_le_bytes());
}

/// Writes an I-type `jalr rd, rs1, imm` at the end of `out`. When `rd` is
/// `Zero` this is a plain jump; otherwise it is a call (the return address
/// lands in `rd`).
fn write_itype_jalr(out: &mut Vec<u8>, rs1: XReg, rd: XReg) {
    write_itype(out, 0x67, 0x0, rd, rs1, 0);
}

/// Writes the I-type immediate-shift variant used for `srli t1, t1, 1`.
///
/// The immediate field is not a plain 12-bit value here: on rv64 its low 6
/// bits at [25..20] are the shift amount, and the 6 bits above them at
/// [31..26] select the shift kind. That selector is zero for the logical
/// shifts (`slli`, `srli`); the arithmetic `srai` sets bit 30 of it. Leaving
/// it clear is what makes this a `srli` rather than a `srai`.
#[allow(clippy::cast_possible_truncation)]
fn write_itype_shift(
    out: &mut Vec<u8>,
    opcode: u32,
    funct3: u32,
    rd: XReg,
    rs1: XReg,
    shamt: u32,
) {
    let insn = ((shamt & 0x3F) << 20)
        | (u32::from(rs1 as u8) << 15)
        | (funct3 << 12)
        | (u32::from(rd as u8) << 7)
        | opcode;
    out.extend_from_slice(&insn.to_le_bytes());
}

/// Writes an R-type instruction at the end of `out`.
#[allow(clippy::cast_possible_truncation)]
fn write_rtype(
    out: &mut Vec<u8>,
    opcode: u32,
    funct3: u32,
    rd: XReg,
    rs1: XReg,
    rs2: XReg,
    funct7: u32,
) {
    let insn = (funct7 << 25)
        | (u32::from(rs2 as u8) << 20)
        | (u32::from(rs1 as u8) << 15)
        | (funct3 << 12)
        | (u32::from(rd as u8) << 7)
        | opcode;
    out.extend_from_slice(&insn.to_le_bytes());
}

/// The hi20 field of a 32-bit value, sign-adjusted to pair with a sign-
/// extending lo12 (RISC-V U-type + I-type pattern).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "intentional bit-pattern reinterpret"
)]
fn hi20_signed(value: u64) -> i32 {
    let adjusted = value.wrapping_add(0x800);
    ((adjusted.cast_signed()) >> 12) as i32
}

/// The low 12 bits of `value` as a sign-extended `i32` (the I-type immediate
/// form).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "intentional bit-pattern reinterpret"
)]
fn lo12_signed(value: u64) -> i32 {
    (value.cast_signed() & 0xFFF) as i32
}

// === shared helpers =======================================================

/// Byte capacity of the `.plt` section, for the stub builders' buffer.
fn plt_capacity(layout: &Layout) -> usize {
    usize::try_from(layout.region(Sect::Plt).size).unwrap_or(0)
}

/// Byte capacity of the `.got.plt` section, for the slot builders' buffer.
fn got_plt_capacity(layout: &Layout) -> usize {
    usize::try_from(layout.region(Sect::GotPlt).size).unwrap_or(0)
}

/// Appends a little-endian 64-bit value to `out`.
fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}
