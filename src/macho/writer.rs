//! Mach-O executable byte assembly and the static link driver.
//!
//! The writer is the Mach-O counterpart of the ELF `writer`: it turns a
//! [`Layout`](super::layout::MachLayout) into a `MH_EXECUTE` byte image. It
//! serialises the header and load commands, copies each linked section's bytes
//! while patching relocations in place through the arch-neutral
//! [`crate::reloc::apply`] driver, and appends `__LINKEDIT` (the `nlist_64`
//! symbol table and the string table).
//!
//! The Mach-O link is a path fully separate from the ELF linker: it owns its
//! own symbol resolution, layout and byte output, so the ELF writer and linker
//! are untouched and stay byte-identical.

use std::{hash::Hasher, path::Path};

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

use crate::{
    archive::{self, Archive},
    endian::{U16, U32, U64},
    error::{Error, Result},
    input::{AlignedBytes, Format, Input, InputBytes},
    macho::{
        self, GotPlan, LinkOptions, MachOFile,
        constants::{
            ARM_THREAD_STATE64, ARM_THREAD_STATE64_COUNT,
            ARM_THREAD_STATE64_PC, CPU_SUBTYPE_ARM64_ALL,
            CPU_SUBTYPE_X86_64_ALL, INDIRECT_SYMBOL_ABS, INDIRECT_SYMBOL_LOCAL,
            LC_BUILD_VERSION, LC_DYLD_INFO_ONLY, LC_DYSYMTAB, LC_LOAD_DYLIB,
            LC_LOAD_DYLINKER, LC_MAIN, LC_SEGMENT_64, LC_SYMTAB, LC_UNIXTHREAD,
            LC_UUID, MH_DYLDLINK, MH_EXECUTE, MH_HAS_TLV_DESCRIPTORS,
            MH_MAGIC_64, MH_NOUNDEFS, MH_PIE, MH_TWOLEVEL, N_ABS, N_EXT,
            N_SECT, N_TYPE, N_UNDF, N_WEAK_REF, PAGEZERO_SIZE, VM_PROT_READ,
            X86_THREAD_STATE64, X86_THREAD_STATE64_COUNT,
            X86_THREAD_STATE64_RIP,
        },
        got::GotKey,
        imports::ImportPlan,
        layout::{
            self, GotLayout, HEADER_SIZE, MachLayout, OutSegment, TEXT_BASE,
        },
        lc::{
            BuildVersionCommand, DyldInfoCommand, DylibCommand,
            DylinkerCommand, DysymtabCommand, EntryPointCommand, MachHeader64,
            Nlist64, SectionHeader64, SegmentCommand64, SymtabCommand,
            ThreadCommand, UuidCommand,
        },
        reloc::MachoTarget,
        sections,
        symtab::{self, Globals},
    },
    pool,
    util::{open_all, pad_name, trim_nul, write_output, write_pod},
};

/// On-disk size of one `nlist_64`.
const NLIST_SIZE: u64 = 16;
/// Links Mach-O `inputs` into a static executable written to `output`.
/// `entry` names the entry symbol (conventionally `_main` for darwin).
pub fn link_macho(
    inputs: &[Input<'_>],
    output: &Path,
    entry: &[u8],
) -> Result<()> {
    link_macho_with_options(inputs, output, entry, &LinkOptions::default())
}

/// Links a Mach-O executable with driver-selected architecture and load
/// command metadata.
pub fn link_macho_with_options(
    inputs: &[Input<'_>],
    output: &Path,
    entry: &[u8],
    options: &LinkOptions<'_>,
) -> Result<()> {
    let opened = open_all(inputs)?;
    let expanded = expand_archives(&opened)?;
    let bytes: Vec<&[u8]> = expanded.iter().map(MachInput::bytes).collect();
    // The Mach-O driver does not share the ELF pipeline, so it sizes its own
    // pool here for the same reason and by the same rule.
    pool::run(&bytes, || {
        let inputs = parse_all(&bytes)?;
        let image = write_executable(&inputs, entry, options)?;
        // Nothing may be published that a serialiser could not reach: the
        // writers size the image from the layout, so a write past its end is a
        // sizing bug.
        crate::util::check_truncated_write()?;
        write_output(output, &image)
    })
}

/// One direct Mach-O object or an aligned copy of an archive member.
enum MachInput<'a> {
    Borrowed(&'a [u8]),
    Owned(AlignedBytes),
}

impl MachInput<'_> {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Owned(bytes) => bytes.bytes(),
        }
    }
}

/// Expands Darwin `.rlib`/`.a` inputs. Non-object members are Rust metadata
/// or archive indexes and do not participate in the native link.
fn expand_archives<'a>(
    opened: &'a [InputBytes<'_>],
) -> Result<Vec<MachInput<'a>>> {
    let mut out = Vec::new();
    for input in opened {
        let bytes = input.bytes();
        if !Archive::is_archive(bytes) {
            out.push(MachInput::Borrowed(bytes));
            continue;
        }
        for member in archive::members(bytes)? {
            if Format::detect(member) == Some(Format::MachO) {
                out.push(MachInput::Owned(AlignedBytes::new(member)));
            }
        }
    }
    Ok(out)
}

/// Parses every input as a 64-bit Mach-O object, in parallel. Each parse is
/// independent (no shared mutable state); `par_iter().collect()` preserves
/// input order so the result matches the serial baseline exactly.
fn parse_all<'d>(bytes: &'d [&'d [u8]]) -> Result<Vec<MachOFile<'d>>> {
    use rayon::prelude::*;
    bytes.par_iter().map(|b| MachOFile::parse(b)).collect()
}

/// Builds the full `MH_EXECUTE` byte image for `inputs`.
fn write_executable(
    inputs: &[MachOFile<'_>],
    entry: &[u8],
    options: &LinkOptions<'_>,
) -> Result<Vec<u8>> {
    let target = derive_target(inputs)?;
    if options.arch.is_some_and(|arch| arch != target) {
        return Err(Error::CommandLine(
            "-arch disagrees with the Mach-O input objects".into(),
        ));
    }
    let globals = Globals::build(inputs)?;
    let imports = ImportPlan::build(inputs, &globals, target, options)?;
    let mut got_plan = GotPlan::scan(inputs, target)?;
    for import in imports.stubs() {
        got_plan.ensure_global(import.name);
    }
    let plan = SymtabPlan::build(inputs)?;
    // The indirect symbol table follows the symbol/string table. Dynamic
    // opcode streams are planned from the resulting section addresses, then
    // a second layout stamps their final __LINKEDIT size; earlier segments do
    // not depend on that size and therefore stay identical.
    let indirect_count = imports.stub_count().saturating_add(got_plan.count());
    let streams_relative = plan
        .linkedit_size()
        .saturating_add(u64::from(indirect_count) * 4);
    let provisional = layout::build(
        inputs,
        streams_relative,
        entry,
        got_plan.count(),
        &globals.commons,
        target,
        options,
        imports.stub_count(),
    )?;
    let dynamic = build_dynamic_info(
        inputs,
        &provisional,
        &got_plan,
        &imports,
        &globals,
    )?;
    let rebase_relative = streams_relative;
    let bind_relative = rebase_relative.saturating_add(
        u64::try_from(dynamic.rebase.len()).unwrap_or(u64::MAX),
    );
    let linkedit_size = bind_relative
        .saturating_add(u64::try_from(dynamic.bind.len()).unwrap_or(u64::MAX));
    let layout = layout::build(
        inputs,
        linkedit_size,
        entry,
        got_plan.count(),
        &globals.commons,
        target,
        options,
        imports.stub_count(),
    )?;
    let common_addr = common_addr_map(&globals, &layout);
    let sym_addr = resolve_all(
        inputs,
        &globals,
        &common_addr,
        &layout.sec_vaddr,
        &imports,
        layout.stubs,
    )?;
    let got_addr = build_got_addr(inputs, &got_plan, layout.got);
    let sec_ordinal = section_ordinal_map(&layout)?;

    let total = layout.linkedit_fileoff + layout.linkedit_filesize;
    let mut image = vec![0u8; usize::try_from(total).unwrap_or(usize::MAX)];
    write_header(&mut image, target, &layout, options);
    write_commands(
        &mut image,
        target,
        &layout,
        &plan,
        &sym_addr,
        &sec_ordinal,
        options,
        rebase_relative,
        u32::try_from(dynamic.rebase.len()).unwrap_or(u32::MAX),
        bind_relative,
        u32::try_from(dynamic.bind.len()).unwrap_or(u32::MAX),
        imports.stub_count(),
    )?;
    write_sections(&mut image, inputs, target, &layout, &sym_addr, &got_addr)?;
    write_tls_descriptors(&mut image, inputs, &layout, &sym_addr)?;
    write_stubs(&mut image, &layout, &imports, &got_plan)?;
    if let Some(got) = layout.got {
        fill_got(&mut image, got, &got_plan, &globals, &sym_addr);
    }
    write_linkedit(
        &mut image,
        &layout,
        &plan,
        &sym_addr,
        &sec_ordinal,
        &got_plan,
        &imports,
        rebase_relative,
        &dynamic.rebase,
        bind_relative,
        &dynamic.bind,
    );
    Ok(image)
}

/// Derives the single link target from the inputs' `cputype`. Every input must
/// agree; a mix is a format error.
fn derive_target(inputs: &[MachOFile<'_>]) -> Result<MachoTarget> {
    let mut target: Option<MachoTarget> = None;
    for input in inputs {
        let current = MachoTarget::from_cpu(input.cpu_type())?;
        match target {
            None => target = Some(current),
            Some(existing) if existing == current => {}
            Some(_) => return Err(Error::Format("inputs disagree on cputype")),
        }
    }
    target.ok_or(Error::Format("no Mach-O inputs to derive a target"))
}

/// Resolves the virtual address of every input symbol of every file.
fn resolve_all(
    inputs: &[MachOFile<'_>],
    globals: &Globals<'_>,
    common_addr: &FxHashMap<&[u8], u64>,
    sec_vaddr: &[Vec<u64>],
    imports: &ImportPlan<'_>,
    stubs: Option<layout::StubLayout>,
) -> Result<Vec<Vec<u64>>> {
    let mut out = Vec::with_capacity(inputs.len());
    for (file, input) in inputs.iter().enumerate() {
        let n = input.symbols().len();
        let mut addrs = vec![0u64; n];
        for (i, slot) in addrs.iter_mut().enumerate() {
            let Some(sym) = input.symbols().nth(i) else {
                continue;
            };
            if sym.n_type & N_TYPE == N_UNDF && sym.n_value == 0 {
                if let Some(import) = imports.get(sym.name) {
                    *slot =
                        import.stub.zip(stubs).map_or(0, |(stub, layout)| {
                            layout.addr.wrapping_add(u64::from(stub) * 12)
                        });
                    continue;
                }
                if sym.n_desc & N_WEAK_REF != 0 {
                    *slot = 0;
                    continue;
                }
                if matches!(sym.name, b"__mh_execute_header" | b"___dso_handle")
                {
                    *slot = TEXT_BASE;
                    continue;
                }
            }
            *slot = symtab::symbol_addr(
                inputs,
                globals,
                common_addr,
                sec_vaddr,
                file,
                i,
            )?;
        }
        out.push(addrs);
    }
    Ok(out)
}

struct DynamicInfo {
    rebase: Vec<u8>,
    bind: Vec<u8>,
}

/// Plans classic dyld rebases and eager binds. Absolute pointer relocations
/// in input data need to move with a PIE slide; an absolute reference to a
/// TAPI import (notably `__tlv_bootstrap` in `__thread_vars`) binds at its
/// final section location rather than through the synthetic GOT.
fn build_dynamic_info(
    inputs: &[MachOFile<'_>],
    layout: &MachLayout,
    got: &GotPlan<'_>,
    imports: &ImportPlan<'_>,
    globals: &Globals<'_>,
) -> Result<DynamicInfo> {
    let mut bind = Vec::new();
    if let Some(got_layout) = layout.got {
        let data = layout
            .data
            .as_ref()
            .ok_or(Error::Format("Mach-O GOT has no data segment"))?;
        for (slot, key) in got.iter() {
            let GotKey::Global(name) = key else {
                continue;
            };
            let Some(import) = imports.get(name) else {
                continue;
            };
            let offset = got_layout
                .addr
                .wrapping_sub(data.vmaddr)
                .wrapping_add(u64::from(slot) * 8);
            append_bind(&mut bind, import.dylib, name, 2, offset);
        }
    }

    let mut rebases = Vec::new();
    let mut seen_rebase = FxHashSet::default();
    let mut seen_bind = FxHashSet::default();
    for (segment_index, segment) in
        [(1u8, Some(&layout.text)), (2u8, layout.data.as_ref())]
    {
        let Some(segment) = segment else {
            continue;
        };
        for section in &segment.sections {
            for member in &section.members {
                let input = inputs
                    .get(member.file)
                    .ok_or(Error::OutOfRange("Mach-O fixup input"))?;
                let input_section = input
                    .sections()
                    .into_iter()
                    .find(|candidate| candidate.index == member.section)
                    .ok_or(Error::OutOfRange("Mach-O fixup section"))?;
                let symbols: Vec<_> = input.symbols().iter().collect();
                for reloc in &input_section.relocations {
                    if u32::from(reloc.r_type) != 0 || reloc.r_length != 3 {
                        continue;
                    }
                    let address = section
                        .addr
                        .wrapping_add(member.offset)
                        .wrapping_add(u64::from(reloc.r_address));
                    let offset = address.wrapping_sub(segment.vmaddr);
                    if reloc.r_extern {
                        let Some(sym) = symbols.get(reloc.r_symbolnum as usize)
                        else {
                            continue;
                        };
                        if let Some(import) = imports.get(sym.name) {
                            if seen_bind.insert((segment_index, offset)) {
                                append_bind(
                                    &mut bind,
                                    import.dylib,
                                    sym.name,
                                    segment_index,
                                    offset,
                                );
                            }
                            continue;
                        }
                        if input_section.sectname == b"__thread_vars"
                            && sym.name.ends_with(b"$tlv$init")
                        {
                            // This field is a TLS-template offset, not a
                            // virtual address to slide.
                            continue;
                        }
                        if sym.is_undefined()
                            && sym.n_value == 0
                            && globals.get(sym.name).is_none()
                            && sym.n_desc & N_WEAK_REF != 0
                        {
                            continue;
                        }
                    }
                    if seen_rebase.insert((segment_index, offset)) {
                        rebases.push((segment_index, offset));
                    }
                }
            }
        }
    }
    if !bind.is_empty() {
        bind.push(0);
    }
    Ok(DynamicInfo {
        rebase: encode_rebases(&rebases),
        bind,
    })
}

fn append_bind(
    out: &mut Vec<u8>,
    dylib: u32,
    name: &[u8],
    segment: u8,
    offset: u64,
) {
    const SET_DYLIB_ORDINAL_IMM: u8 = 0x10;
    const SET_DYLIB_ORDINAL_ULEB: u8 = 0x20;
    const SET_SYMBOL_TRAILING_FLAGS_IMM: u8 = 0x40;
    const SET_TYPE_POINTER: u8 = 0x51;
    const SET_SEGMENT_AND_OFFSET_ULEB: u8 = 0x70;
    const DO_BIND: u8 = 0x90;

    if dylib <= 15 {
        out.push(SET_DYLIB_ORDINAL_IMM | u8::try_from(dylib).unwrap_or(15));
    } else {
        out.push(SET_DYLIB_ORDINAL_ULEB);
        write_uleb(out, u64::from(dylib));
    }
    out.push(SET_SYMBOL_TRAILING_FLAGS_IMM);
    out.extend_from_slice(name);
    out.push(0);
    out.push(SET_TYPE_POINTER);
    out.push(SET_SEGMENT_AND_OFFSET_ULEB | segment);
    write_uleb(out, offset);
    out.push(DO_BIND);
}

fn encode_rebases(entries: &[(u8, u64)]) -> Vec<u8> {
    const SET_TYPE_POINTER: u8 = 0x11;
    const SET_SEGMENT_AND_OFFSET_ULEB: u8 = 0x20;
    const DO_REBASE_IMM_TIMES_ONE: u8 = 0x51;
    const DONE: u8 = 0x00;
    let mut out = Vec::new();
    if entries.is_empty() {
        return out;
    }
    out.push(SET_TYPE_POINTER);
    for &(segment, offset) in entries {
        out.push(SET_SEGMENT_AND_OFFSET_ULEB | segment);
        write_uleb(&mut out, offset);
        out.push(DO_REBASE_IMM_TIMES_ONE);
    }
    out.push(DONE);
    out
}

fn write_uleb(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = u8::try_from(value & 0x7f).unwrap_or(0);
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Writes arm64's three-instruction non-lazy symbol stubs:
/// `adrp x16, slot@page; ldr x16, [x16, slot@pageoff]; br x16`.
fn write_stubs(
    image: &mut [u8],
    layout: &MachLayout,
    imports: &ImportPlan<'_>,
    got_plan: &GotPlan<'_>,
) -> Result<()> {
    let Some(stubs) = layout.stubs else {
        return Ok(());
    };
    let got = layout
        .got
        .ok_or(Error::Format("Mach-O stubs require a GOT"))?;
    for import in imports.stubs() {
        let stub = import.stub.unwrap_or(u32::MAX);
        let slot = got_plan
            .global_slot(import.name)
            .ok_or(Error::Format("Mach-O import has no GOT slot"))?;
        let stub_addr = stubs.addr.wrapping_add(u64::from(stub) * 12);
        let got_addr = got.addr.wrapping_add(u64::from(slot) * 8);
        let off = stubs.offset.wrapping_add(u64::from(stub) * 12);
        let adrp = arm64_stub_adrp(stub_addr, got_addr)?;
        let pageoff = u32::try_from((got_addr & 0xfff) >> 3).unwrap_or(0);
        let ldr = 0xf940_0210 | (pageoff << 10);
        let br = 0xd61f_0200u32;
        let start = usize::try_from(off)
            .map_err(|_| Error::OutOfRange("Mach-O stub offset"))?;
        let dst = image
            .get_mut(start..start.saturating_add(12))
            .ok_or(Error::OutOfRange("Mach-O stub bytes"))?;
        dst[0..4].copy_from_slice(&adrp.to_le_bytes());
        dst[4..8].copy_from_slice(&ldr.to_le_bytes());
        dst[8..12].copy_from_slice(&br.to_le_bytes());
    }
    Ok(())
}

fn arm64_stub_adrp(place: u64, target: u64) -> Result<u32> {
    let target_page = target & !0xfff;
    let place_page = place & !0xfff;
    let delta = target_page
        .cast_signed()
        .wrapping_sub(place_page.cast_signed());
    if !(-(1i64 << 32)..(1i64 << 32)).contains(&delta) {
        return Err(Error::OutOfRange("arm64 stub ADRP distance"));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let imm = (delta >> 12) as u32;
    let imm_lo = (imm & 0x3) << 29;
    let imm_hi = (imm & 0x001f_fffc) << 3;
    Ok(0x9000_0010 | imm_lo | imm_hi)
}

/// Name to `__common` storage address, from the globals' merged commons and
/// the layout's parallel addresses.
fn common_addr_map<'a>(
    globals: &'a Globals<'a>,
    layout: &MachLayout,
) -> FxHashMap<&'a [u8], u64> {
    globals
        .commons
        .iter()
        .zip(layout.common_addr.iter())
        .map(|(c, addr)| (c.name, *addr))
        .collect()
}

/// Builds `(file, input section ordinal) -> 1-based output section ordinal`.
///
/// `nlist_64.n_sect` is one byte, so an image with more than 255 output
/// sections cannot name them all. Saturating at 255 would point every section
/// past the limit at the same wrong one, so the count is checked instead. The
/// walk visits every emitted section header, member-less synthetic ones
/// included, which is the same order and the same set
/// [`common_section_ordinal`] counts; a link that gets past here therefore
/// cannot overflow there either.
fn section_ordinal_map(layout: &MachLayout) -> Result<Vec<(usize, u32, u8)>> {
    let mut out = Vec::new();
    let mut ordinal: u32 = 1;
    for segment in iter_segments(layout) {
        for s in &segment.sections {
            let n = u8::try_from(ordinal).map_err(|_| {
                Error::OutOfRange("Mach-O output section ordinal")
            })?;
            for m in &s.members {
                out.push((m.file, m.section, n));
            }
            ordinal += 1;
        }
    }
    Ok(out)
}

/// Iterates the placed segments carrying section headers (`__TEXT`, `__DATA`).
fn iter_segments(layout: &MachLayout) -> impl Iterator<Item = &OutSegment> {
    [Some(&layout.text), layout.data.as_ref()]
        .into_iter()
        .flatten()
}

// --- symtab plan ---------------------------------------------------------

/// One planned `nlist_64` entry, before its address and output section are
/// stamped.
struct PlannedSym {
    file: usize,
    sym: u32,
    n_type: u8,
    n_desc: u16,
    n_strx: u32,
    class: SymClass,
    input_n_sect: u8,
    /// A tentative definition (`N_UNDF` with a size): planned as an
    /// external-defined row whose section is the synthetic `__common`.
    tentative: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SymClass {
    Local,
    Extdef,
    Undef,
}

/// The planned symbol table: the entries (partitioned local / external-defined
/// / undefined) and the string table, built before layout so the
/// `__LINKEDIT` size is known.
struct SymtabPlan<'d> {
    locals: Vec<PlannedSym>,
    extdef: Vec<PlannedSym>,
    undef: Vec<PlannedSym>,
    strtab: Vec<u8>,
    /// `(input file, input symbol index)` to the row's index in the output
    /// table, for the indirect symbol table.
    out_index: FxHashMap<(usize, u32), u32>,
    /// A global name to the output row that defines it (an undefined row
    /// when nothing defines the name), for the same purpose.
    by_name: FxHashMap<&'d [u8], u32>,
}

impl<'d> SymtabPlan<'d> {
    fn build(inputs: &'d [MachOFile<'d>]) -> Result<Self> {
        // Every name some input defines as a global. A reference to one of
        // these is satisfied inside the image, so emitting it as an undefined
        // row contradicts `MH_NOUNDEFS` -- `nm` showed `U _foo` in a static
        // image, and consumers that read the dysymtab ranges rejected it.
        let defined = global_names(inputs);
        let mut strtab = vec![0u8];
        let mut locals = Vec::new();
        let mut extdef = Vec::new();
        let mut undef = Vec::new();
        for (file, input) in inputs.iter().enumerate() {
            let sections = input.sections();
            let syms = input.symbols();
            for (i, sym) in syms.iter().enumerate() {
                if sym.is_stab() {
                    continue;
                }
                let input_n_sect = sym.n_sect;
                let typ = sym.n_type & N_TYPE;
                // A tentative definition is a definition: it takes an
                // external-defined row in the synthetic `__common` section,
                // not an undefined one.
                let tentative = typ == N_UNDF && sym.n_value > 0;
                let class = if tentative {
                    SymClass::Extdef
                } else if typ == N_UNDF {
                    SymClass::Undef
                } else if sym.n_type & N_EXT != 0 {
                    SymClass::Extdef
                } else {
                    SymClass::Local
                };
                // Drop section-relative symbols whose section is not linked
                // (e.g. a local label in a deferred unwind section).
                if typ == N_SECT && !section_linked(&sections, input_n_sect) {
                    continue;
                }
                if class == SymClass::Undef && defined.contains(&sym.name) {
                    continue;
                }
                let n_strx = intern(&mut strtab, sym.name);
                let sym_idx = u32::try_from(i)
                    .map_err(|_| Error::OutOfRange("planned sym idx"))?;
                let planned = PlannedSym {
                    file,
                    sym: sym_idx,
                    n_type: sym.n_type,
                    n_desc: sym.n_desc,
                    n_strx,
                    class,
                    input_n_sect,
                    tentative,
                };
                match class {
                    SymClass::Local => locals.push(planned),
                    SymClass::Extdef => extdef.push(planned),
                    SymClass::Undef => undef.push(planned),
                }
            }
        }
        // Output row indices are only known once the classes are collected:
        // the written table is locals, then external-defined, then undefined.
        let mut out_index: FxHashMap<(usize, u32), u32> = FxHashMap::default();
        let mut next = 0u32;
        for rows in [&locals, &extdef, &undef] {
            for sym in rows {
                out_index.insert((sym.file, sym.sym), next);
                next = next.saturating_add(1);
            }
        }
        // A global's row is the one that defines it; only a name nothing
        // defines keeps an undefined row.
        let mut by_name: FxHashMap<&[u8], u32> = FxHashMap::default();
        for sym in extdef.iter().chain(undef.iter()) {
            if let Some(&idx) = out_index.get(&(sym.file, sym.sym)) {
                by_name.entry(plan_name(inputs, sym)).or_insert(idx);
            }
        }
        Ok(Self {
            locals,
            extdef,
            undef,
            strtab,
            out_index,
            by_name,
        })
    }

    /// Total `__LINKEDIT` byte size: the `nlist_64` array followed by the
    /// string table, rounded up to 4 bytes so the indirect symbol table's
    /// u32 rows land aligned. The segment start is page-aligned by the
    /// layout, so aligning the content aligns the file offset too.
    fn linkedit_size(&self) -> u64 {
        let nsyms = self.nsyms();
        let symtab = u64::try_from(nsyms).unwrap_or(0) * NLIST_SIZE;
        align4(symtab + u64::try_from(self.strtab.len()).unwrap_or(0))
    }

    /// The number of rows in all three symbol-table ranges.
    fn nsyms(&self) -> usize {
        self.locals.len() + self.extdef.len() + self.undef.len()
    }

    /// Counts for `LC_DYSYMTAB`.
    fn count(rows: &[PlannedSym]) -> u32 {
        u32::try_from(rows.len()).unwrap_or(0)
    }
}

/// Whether the 1-based section ordinal `n_sect` of `sections` is linked.
fn section_linked(sections: &[macho::MachSection<'_>], n_sect: u8) -> bool {
    let idx = usize::from(n_sect).saturating_sub(1);
    sections.get(idx).is_some_and(layout::is_linkable)
}

/// Rounds a byte offset up to the 4-byte alignment the indirect symbol
/// table's u32 rows require. ld64 and lld 4-align the table the same way.
const fn align4(n: u64) -> u64 {
    (n + 3) & !3
}

/// Appends `name` plus a NUL to `strtab`, returning its starting offset.
fn intern(strtab: &mut Vec<u8>, name: &[u8]) -> u32 {
    let off = u32::try_from(strtab.len()).unwrap_or(u32::MAX);
    strtab.extend_from_slice(name);
    strtab.push(0);
    off
}

// --- image serialisation -------------------------------------------------

/// Writes the `mach_header_64` at offset zero.
fn write_header(
    image: &mut [u8],
    target: MachoTarget,
    layout: &MachLayout,
    options: &LinkOptions<'_>,
) {
    let (cputype, cpusubtype) = cpu_fields(target);
    let header = MachHeader64 {
        magic: U32::new(MH_MAGIC_64),
        cputype: U32::new(cputype),
        cpusubtype: U32::new(cpusubtype),
        filetype: U32::new(MH_EXECUTE),
        ncmds: U32::new(command_count(layout, options)),
        sizeofcmds: U32::new(layout.sizeofcmds),
        flags: U32::new(if options.dynamic {
            let tlv = layout.data.as_ref().is_some_and(|data| {
                data.sections
                    .iter()
                    .any(|s| trim_nul(&s.sectname) == b"__thread_vars")
            });
            MH_NOUNDEFS
                | MH_DYLDLINK
                | MH_TWOLEVEL
                | MH_PIE
                | if tlv { MH_HAS_TLV_DESCRIPTORS } else { 0 }
        } else {
            MH_NOUNDEFS
        }),
        reserved: U32::new(0),
    };
    write_pod(image, 0, &header);
}

/// The `(cputype, cpusubtype)` pair for `target`.
fn cpu_fields(target: MachoTarget) -> (u32, u32) {
    match target {
        MachoTarget::X86_64 => (
            crate::macho::constants::CPU_TYPE_X86_64,
            CPU_SUBTYPE_X86_64_ALL,
        ),
        MachoTarget::Arm64 => (
            crate::macho::constants::CPU_TYPE_ARM64,
            CPU_SUBTYPE_ARM64_ALL,
        ),
    }
}

/// The number of load commands the image emits.
fn command_count(layout: &MachLayout, options: &LinkOptions<'_>) -> u32 {
    // PAGEZERO, TEXT, LINKEDIT, maybe DATA.
    let nseg = 3 + u32::from(layout.data.is_some());
    let entry = if options.dynamic { 3 } else { 1 };
    nseg
        + 3 // SYMTAB, DYSYMTAB, UUID
        + entry
        + u32::from(options.platform.is_some())
        + u32::try_from(options.dylibs.len()).unwrap_or(u32::MAX)
}

/// Writes the load-command table right after the header.
fn write_commands(
    image: &mut [u8],
    target: MachoTarget,
    layout: &MachLayout,
    plan: &SymtabPlan,
    sym_addr: &[Vec<u64>],
    sec_ordinal: &[(usize, u32, u8)],
    options: &LinkOptions<'_>,
    rebase_relative: u64,
    rebase_size: u32,
    bind_relative: u64,
    bind_size: u32,
    stub_count: u32,
) -> Result<()> {
    let mut off = HEADER_SIZE;
    off = write_pagezero(image, off);
    off = write_segment(image, off, &layout.text)?;
    if let Some(data) = layout.data.as_ref() {
        off = write_segment(image, off, data)?;
    }
    off = write_linkedit_segment(image, off, layout);
    off = write_symtab(image, off, layout, plan);
    let got_count = layout.got.map_or(0, |g| g.count);
    off = write_dysymtab(
        image,
        off,
        plan,
        layout,
        got_count.saturating_add(stub_count),
    )?;
    if options.dynamic {
        off = write_dyld_info(
            image,
            off,
            layout,
            rebase_relative,
            rebase_size,
            bind_relative,
            bind_size,
        );
    }
    if let Some(platform) = options.platform {
        off = write_build_version(image, off, platform);
    }
    for dylib in options.dylibs {
        off = write_dylib(image, off, dylib.install_name)?;
    }
    if options.dynamic {
        off = write_dylinker(image, off)?;
        off = write_main(image, off, layout);
    } else {
        off = write_thread(image, off, target, layout);
    }
    write_uuid(image, off, target, layout, sym_addr, sec_ordinal);
    Ok(())
}

/// Writes `LC_BUILD_VERSION` without tool-version records.
fn write_build_version(
    image: &mut [u8],
    off: u64,
    version: crate::macho::PlatformVersion,
) -> u64 {
    write_pod(
        image,
        off,
        &BuildVersionCommand {
            cmd: U32::new(LC_BUILD_VERSION),
            cmdsize: U32::new(24),
            platform: U32::new(version.platform),
            min_os: U32::new(version.min_os),
            sdk: U32::new(version.sdk),
            ntools: U32::new(0),
        },
    )
}

/// Writes the classic dyld metadata command. This linker uses eager non-lazy
/// binds plus the rebase stream needed when arm64 PIE slides the image.
fn write_dyld_info(
    image: &mut [u8],
    off: u64,
    layout: &MachLayout,
    rebase_relative: u64,
    rebase_size: u32,
    bind_relative: u64,
    bind_size: u32,
) -> u64 {
    let rebase_off = layout.linkedit_fileoff.saturating_add(rebase_relative);
    let bind_off = layout.linkedit_fileoff.saturating_add(bind_relative);
    write_pod(
        image,
        off,
        &DyldInfoCommand {
            cmd: U32::new(LC_DYLD_INFO_ONLY),
            cmdsize: U32::new(48),
            rebase_off: U32::new(if rebase_size == 0 {
                0
            } else {
                u32::try_from(rebase_off).unwrap_or(0)
            }),
            rebase_size: U32::new(rebase_size),
            bind_off: U32::new(u32::try_from(bind_off).unwrap_or(0)),
            bind_size: U32::new(bind_size),
            weak_bind_off: U32::new(0),
            weak_bind_size: U32::new(0),
            lazy_bind_off: U32::new(0),
            lazy_bind_size: U32::new(0),
            export_off: U32::new(0),
            export_size: U32::new(0),
        },
    )
}

/// Writes dyld's path for a dynamically launched executable.
fn write_dylinker(image: &mut [u8], off: u64) -> Result<u64> {
    const PATH: &str = "/usr/lib/dyld";
    let size = layout::dylinker_command_size();
    write_pod(
        image,
        off,
        &DylinkerCommand {
            cmd: U32::new(LC_LOAD_DYLINKER),
            cmdsize: U32::new(size),
            name: U32::new(12),
        },
    );
    write_command_string(image, off, 12, size, PATH)
}

/// Writes one dynamic-library dependency by install name.
fn write_dylib(image: &mut [u8], off: u64, name: &str) -> Result<u64> {
    let size = layout::dylib_command_size(name);
    write_pod(
        image,
        off,
        &DylibCommand {
            cmd: U32::new(LC_LOAD_DYLIB),
            cmdsize: U32::new(size),
            name: U32::new(24),
            timestamp: U32::new(0),
            current_version: U32::new(0),
            compatibility_version: U32::new(0),
        },
    );
    write_command_string(image, off, 24, size, name)
}

/// Copies a NUL-terminated string into an already-sized load command.
fn write_command_string(
    image: &mut [u8],
    off: u64,
    fixed_size: u32,
    command_size: u32,
    value: &str,
) -> Result<u64> {
    let start = off.saturating_add(u64::from(fixed_size));
    let value_end =
        start.saturating_add(u64::try_from(value.len()).unwrap_or(u64::MAX));
    let start = usize::try_from(start)
        .map_err(|_| Error::OutOfRange("Mach-O load-command string"))?;
    let value_end = usize::try_from(value_end)
        .map_err(|_| Error::OutOfRange("Mach-O load-command string"))?;
    let slot = image
        .get_mut(start..value_end)
        .ok_or(Error::OutOfRange("Mach-O load-command string"))?;
    slot.copy_from_slice(value.as_bytes());
    Ok(off.saturating_add(u64::from(command_size)))
}

/// Writes `LC_MAIN`, whose entry offset is relative to `__TEXT`.
fn write_main(image: &mut [u8], off: u64, layout: &MachLayout) -> u64 {
    write_pod(
        image,
        off,
        &EntryPointCommand {
            cmd: U32::new(LC_MAIN),
            cmdsize: U32::new(24),
            entryoff: U64::new(layout.entry_off),
            stacksize: U64::new(0),
        },
    )
}

/// Writes one `LC_SEGMENT_64` and its embedded `section_64` headers.
fn write_segment(image: &mut [u8], off: u64, seg: &OutSegment) -> Result<u64> {
    let nsects = u32::try_from(seg.sections.len())
        .map_err(|_| Error::OutOfRange("segment section count"))?;
    let cmd = SegmentCommand64 {
        cmd: U32::new(LC_SEGMENT_64),
        cmdsize: U32::new(72 + 80 * nsects),
        segname: seg.name,
        vmaddr: U64::new(seg.vmaddr),
        vmsize: U64::new(seg.vmsize),
        fileoff: U64::new(seg.fileoff),
        filesize: U64::new(seg.filesize),
        maxprot: U32::new(seg.maxprot),
        initprot: U32::new(seg.initprot),
        nsects: U32::new(nsects),
        flags: U32::new(0),
    };
    let mut cursor = write_pod(image, off, &cmd);
    for s in &seg.sections {
        let hdr = SectionHeader64 {
            sectname: s.sectname,
            segname: s.segname,
            addr: U64::new(s.addr),
            size: U64::new(s.size),
            offset: U32::new(u32::try_from(s.offset).unwrap_or(0)),
            align: U32::new(s.align),
            reloff: U32::new(0),
            nreloc: U32::new(0),
            flags: U32::new(s.flags),
            reserved1: U32::new(s.reserved1),
            reserved2: U32::new(s.reserved2),
            reserved3: U32::new(0),
        };
        cursor = write_pod(image, cursor, &hdr);
    }
    Ok(cursor)
}

/// Writes the `LC_SEGMENT_64` for `__LINKEDIT` (no sections).
fn write_linkedit_segment(
    image: &mut [u8],
    off: u64,
    layout: &MachLayout,
) -> u64 {
    let cmd = SegmentCommand64 {
        cmd: U32::new(LC_SEGMENT_64),
        cmdsize: U32::new(72),
        segname: pad_name(b"__LINKEDIT"),
        vmaddr: U64::new(layout.linkedit_vmaddr),
        vmsize: U64::new(layout.linkedit_filesize),
        fileoff: U64::new(layout.linkedit_fileoff),
        filesize: U64::new(layout.linkedit_filesize),
        maxprot: U32::new(VM_PROT_READ),
        initprot: U32::new(VM_PROT_READ),
        nsects: U32::new(0),
        flags: U32::new(0),
    };
    write_pod(image, off, &cmd)
}

/// Writes `LC_SYMTAB`.
fn write_symtab(
    image: &mut [u8],
    off: u64,
    layout: &MachLayout,
    plan: &SymtabPlan,
) -> u64 {
    let nsyms = plan.nsyms();
    let stroff = layout.linkedit_fileoff
        + u64::try_from(nsyms).unwrap_or(0) * NLIST_SIZE;
    let cmd = SymtabCommand {
        cmd: U32::new(LC_SYMTAB),
        cmdsize: U32::new(24),
        symoff: U32::new(u32::try_from(layout.linkedit_fileoff).unwrap_or(0)),
        nsyms: U32::new(u32::try_from(nsyms).unwrap_or(0)),
        stroff: U32::new(u32::try_from(stroff).unwrap_or(0)),
        strsize: U32::new(u32::try_from(plan.strtab.len()).unwrap_or(0)),
    };
    write_pod(image, off, &cmd)
}

/// Writes `LC_DYSYMTAB` with the local/external/undefined ranges.
fn write_dysymtab(
    image: &mut [u8],
    off: u64,
    plan: &SymtabPlan<'_>,
    layout: &MachLayout,
    got_count: u32,
) -> Result<u64> {
    let nlocals = SymtabPlan::count(&plan.locals);
    let nextdef = SymtabPlan::count(&plan.extdef);
    let nundef = SymtabPlan::count(&plan.undef);
    // One row index per `__got` slot, written after the string table in
    // `__LINKEDIT`. The rows are u32s, so the offset is rounded up to
    // 4 bytes past the string table.
    let nsyms = plan.nsyms();
    let indirect = align4(
        layout.linkedit_fileoff
            + u64::try_from(nsyms).unwrap_or(0) * NLIST_SIZE
            + plan.strtab.len() as u64,
    );
    let indirectsymoff = u32::try_from(indirect)
        .map_err(|_| Error::OutOfRange("indirect symbol table offset"))?;
    // The locals start the table: there is no null entry ahead of them.
    let ilocalsym = 0u32;
    let iextdefsym = ilocalsym + nlocals;
    let iundefsym = iextdefsym + nextdef;
    let cmd = DysymtabCommand {
        cmd: U32::new(LC_DYSYMTAB),
        cmdsize: U32::new(80),
        ilocalsym: U32::new(ilocalsym),
        nlocalsym: U32::new(nlocals),
        iextdefsym: U32::new(iextdefsym),
        nextdefsym: U32::new(nextdef),
        iundefsym: U32::new(iundefsym),
        nundefsym: U32::new(nundef),
        tocoff: U32::new(0),
        ntoc: U32::new(0),
        modtaboff: U32::new(0),
        nmodtab: U32::new(0),
        extrefsymoff: U32::new(0),
        nextrefsyms: U32::new(0),
        indirectsymoff: U32::new(indirectsymoff),
        nindirectsyms: U32::new(got_count),
        extreloff: U32::new(0),
        nextrel: U32::new(0),
        locreloff: U32::new(0),
        nlocrel: U32::new(0),
    };
    Ok(write_pod(image, off, &cmd))
}

/// Writes the leading `__PAGEZERO` segment.
///
/// It covers `[0, 4 GiB)` with no protection and no file bytes, so a null
/// dereference faults rather than reaching a mappable page. ld64 and lld emit
/// it unconditionally for an executable, and strict tooling treats an image
/// without it as malformed.
fn write_pagezero(image: &mut [u8], off: u64) -> u64 {
    let cmd = SegmentCommand64 {
        cmd: U32::new(LC_SEGMENT_64),
        cmdsize: U32::new(72),
        segname: pad_name(b"__PAGEZERO"),
        vmaddr: U64::new(0),
        vmsize: U64::new(PAGEZERO_SIZE),
        fileoff: U64::new(0),
        filesize: U64::new(0),
        maxprot: U32::new(0),
        initprot: U32::new(0),
        nsects: U32::new(0),
        flags: U32::new(0),
    };
    write_pod(image, off, &cmd)
}

/// Writes `LC_UNIXTHREAD` with the entry address in the program counter.
///
/// `LC_MAIN` sets `LC_REQ_DYLD`, and XNU's `load_main` demands a
/// `LC_LOAD_DYLINKER` beside it, so a true-static image carrying `LC_MAIN`
/// can never exec. The thread state is the register file the kernel starts
/// with: every register zero except the one that says where to begin.
fn write_thread(
    image: &mut [u8],
    off: u64,
    target: MachoTarget,
    layout: &MachLayout,
) -> u64 {
    let (flavor, count, pc_slot) = match target {
        MachoTarget::X86_64 => (
            X86_THREAD_STATE64,
            X86_THREAD_STATE64_COUNT,
            X86_THREAD_STATE64_RIP,
        ),
        MachoTarget::Arm64 => (
            ARM_THREAD_STATE64,
            ARM_THREAD_STATE64_COUNT,
            ARM_THREAD_STATE64_PC,
        ),
    };
    let head = ThreadCommand {
        cmd: U32::new(LC_UNIXTHREAD),
        cmdsize: U32::new(layout::thread_command_size(target)),
        flavor: U32::new(flavor),
        count: U32::new(count),
    };
    let mut at = write_pod(image, off, &head);
    // The state is `count` 32-bit words, all zero but the program counter,
    // which is a virtual address rather than the file offset `LC_MAIN` took.
    let pc = TEXT_BASE.wrapping_add(layout.entry_off);
    for slot in 0..usize::try_from(count).unwrap_or(0) / 2 {
        let value = if slot == pc_slot { pc } else { 0 };
        at = write_pod(image, at, &U64::new(value));
    }
    at
}

/// Writes `LC_UUID` with a deterministic image identifier derived from the
/// layout, so re-links of the same inputs reproduce the same UUID.
fn write_uuid(
    image: &mut [u8],
    off: u64,
    target: MachoTarget,
    layout: &MachLayout,
    sym_addr: &[Vec<u64>],
    sec_ordinal: &[(usize, u32, u8)],
) -> u64 {
    let uuid = derive_uuid(target, layout, sym_addr, sec_ordinal);
    let cmd = UuidCommand {
        cmd: U32::new(LC_UUID),
        cmdsize: U32::new(24),
        uuid,
    };
    write_pod(image, off, &cmd)
}

/// Mixes layout-derived values into a 16-byte identifier.
fn derive_uuid(
    target: MachoTarget,
    layout: &MachLayout,
    sym_addr: &[Vec<u64>],
    sec_ordinal: &[(usize, u32, u8)],
) -> [u8; 16] {
    let mut h = FxHasher::default();
    h.write_u32(layout.sizeofcmds);
    h.write_u64(layout.entry_off);
    h.write_u64(layout.text.filesize);
    h.write_u8(match target {
        MachoTarget::X86_64 => 0,
        MachoTarget::Arm64 => 1,
    });
    for (file, _sec, ord) in sec_ordinal {
        h.write_usize(*file);
        h.write_u8(*ord);
    }
    for file in sym_addr {
        for a in file {
            h.write_u64(*a);
        }
    }
    mix_to_uuid(h.finish())
}

/// Spreads a 64-bit hash across a 16-byte buffer.
fn mix_to_uuid(hash: u64) -> [u8; 16] {
    let mut h = rustc_hash::FxHasher::default();
    h.write_u64(hash);
    let a = h.finish();
    h = rustc_hash::FxHasher::default();
    h.write_u64(a.wrapping_mul(0xff51_afd7_ed55_8ccd));
    let b = h.finish();
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&a.to_be_bytes());
    out[8..].copy_from_slice(&b.to_be_bytes());
    out
}

// --- linkedit (symbol + string tables) -----------------------------------

/// Writes `__LINKEDIT`: the `nlist_64` array (locals, external-defined, then
/// undefined) followed by the string table.
///
/// There is no leading null entry. Mach-O has no null symbol, and the one
/// invented here sat outside every `LC_DYSYMTAB` range, so the table the
/// header described did not cover the table that was written. Each entry's
/// `n_value` is the resolved symbol address; `n_sect` is the output section
/// ordinal (0 for absolute and undefined symbols).
fn write_linkedit(
    image: &mut [u8],
    layout: &MachLayout,
    plan: &SymtabPlan<'_>,
    sym_addr: &[Vec<u64>],
    sec_ordinal: &[(usize, u32, u8)],
    got_plan: &GotPlan<'_>,
    imports: &ImportPlan<'_>,
    rebase_relative: u64,
    rebase: &[u8],
    bind_relative: u64,
    bind: &[u8],
) {
    let common_sect = common_section_ordinal(layout);
    let mut off = layout.linkedit_fileoff;
    for class in [SymClass::Local, SymClass::Extdef, SymClass::Undef] {
        for sym in plan.iter(class) {
            let value = nlist_value(sym, sym_addr);
            let n_sect = nlist_sect(sym, sec_ordinal);
            // A tentative definition leaves the input as `N_UNDF`; in the
            // output it is a defined external in `__common`.
            let (n_type, n_sect) = if sym.tentative {
                (N_EXT | N_SECT, common_sect)
            } else {
                (sym.n_type, n_sect)
            };
            let entry = Nlist64 {
                n_strx: U32::new(sym.n_strx),
                n_type,
                n_sect,
                n_desc: U16::new(sym.n_desc),
                n_value: U64::new(value),
            };
            off = write_pod(image, off, &entry);
        }
    }
    let str_off = usize::try_from(off).unwrap_or(usize::MAX);
    if let Some(slot) =
        image.get_mut(str_off..str_off.saturating_add(plan.strtab.len()))
    {
        slot.copy_from_slice(&plan.strtab);
    }
    // The indirect rows are u32s: skip the padding that 4-aligns them
    // past the string table, matching `LC_DYSYMTAB`'s `indirectsymoff`.
    let ind_off = align4(off + plan.strtab.len() as u64);
    write_indirect_symbols(
        image,
        usize::try_from(ind_off).unwrap_or(usize::MAX),
        plan,
        got_plan,
        imports,
    );
    let rebase_off = layout.linkedit_fileoff.saturating_add(rebase_relative);
    let start = usize::try_from(rebase_off).unwrap_or(usize::MAX);
    if let Some(dst) = image.get_mut(start..start.saturating_add(rebase.len()))
    {
        dst.copy_from_slice(rebase);
    }
    let bind_off = layout.linkedit_fileoff.saturating_add(bind_relative);
    let start = usize::try_from(bind_off).unwrap_or(usize::MAX);
    if let Some(dst) = image.get_mut(start..start.saturating_add(bind.len())) {
        dst.copy_from_slice(bind);
    }
}

/// Writes the indirect symbol table after the string table: for each
/// `__got` slot, the output row of the symbol it holds. A key with no row
/// (nothing defined or planned the name) is marked absolute, the constant
/// consumers read as "no symbol".
fn write_indirect_symbols(
    image: &mut [u8],
    off: usize,
    plan: &SymtabPlan<'_>,
    got_plan: &GotPlan<'_>,
    imports: &ImportPlan<'_>,
) {
    for import in imports.stubs() {
        let stub = import.stub.unwrap_or(u32::MAX);
        let row = plan
            .by_name
            .get(import.name)
            .copied()
            .unwrap_or(INDIRECT_SYMBOL_ABS);
        let at =
            off.saturating_add(usize::try_from(stub).unwrap_or(usize::MAX) * 4);
        if let Some(dst) = image.get_mut(at..at.saturating_add(4)) {
            dst.copy_from_slice(&row.to_le_bytes());
        }
    }
    let got_base = imports.stub_count();
    for (slot, key) in got_plan.iter() {
        let row = match key {
            GotKey::Global(name) => plan
                .by_name
                .get(name)
                .copied()
                .unwrap_or(INDIRECT_SYMBOL_ABS),
            GotKey::Local { file, sym } => plan
                .out_index
                .get(&(file, sym))
                .copied()
                .unwrap_or(INDIRECT_SYMBOL_LOCAL),
        };
        let row_index = got_base.saturating_add(slot);
        let at = off.saturating_add(
            usize::try_from(row_index).unwrap_or(usize::MAX) * 4,
        );
        if let Some(dst) = image.get_mut(at..at.saturating_add(4)) {
            dst.copy_from_slice(&row.to_le_bytes());
        }
    }
}

impl SymtabPlan<'_> {
    /// Iterates the entries of one class.
    fn iter(&self, class: SymClass) -> impl Iterator<Item = &PlannedSym> {
        let vec: &[PlannedSym] = match class {
            SymClass::Local => &self.locals,
            SymClass::Extdef => &self.extdef,
            SymClass::Undef => &self.undef,
        };
        vec.iter()
    }
}

/// The 1-based output ordinal of the synthetic `__common` section, or 0
/// when the link allocated none. Counting follows the section headers the
/// writer emits, member-less synthetic sections included.
///
/// The saturating add cannot clamp: [`section_ordinal_map`] walks the same
/// sections in the same order and fails the link before the count reaches
/// 256.
fn common_section_ordinal(layout: &MachLayout) -> u8 {
    let mut ordinal = 0u8;
    for segment in iter_segments(layout) {
        for s in &segment.sections {
            ordinal = ordinal.saturating_add(1);
            if trim_nul(&s.sectname) == b"__common" {
                return ordinal;
            }
        }
    }
    0
}

/// The name of one planned symbol, borrowed from its input file.
fn plan_name<'d>(inputs: &'d [MachOFile<'d>], sym: &PlannedSym) -> &'d [u8] {
    let idx = usize::try_from(sym.sym).unwrap_or(usize::MAX);
    inputs
        .get(sym.file)
        .and_then(|f| f.symbols().nth(idx))
        .map_or(&b""[..], |s| s.name)
}

/// Every name some input defines as a global, for dropping the undefined rows
/// that the link itself satisfies.
fn global_names<'d>(inputs: &'d [MachOFile<'d>]) -> FxHashSet<&'d [u8]> {
    let mut out = FxHashSet::default();
    for input in inputs {
        #[allow(clippy::explicit_iter_loop)]
        // `&syms` would borrow the temporary table rather than the mapped
        // bytes, and the names outlive it.
        for sym in input.symbols().iter() {
            if sym.is_stab() || sym.n_type & N_EXT == 0 || sym.name.is_empty() {
                continue;
            }
            let typ = sym.n_type & N_TYPE;
            // A tentative definition (`N_UNDF` with a size) defines the
            // name through the merged `__common` storage.
            if typ == N_SECT
                || typ == N_ABS
                || (typ == N_UNDF && sym.n_value > 0)
            {
                out.insert(sym.name);
            }
        }
    }
    out
}

/// The resolved address for a planned symbol. Undefined symbols are zero.
fn nlist_value(sym: &PlannedSym, sym_addr: &[Vec<u64>]) -> u64 {
    if sym.class == SymClass::Undef {
        return 0;
    }
    let idx = usize::try_from(sym.sym).unwrap_or(0);
    sym_addr
        .get(sym.file)
        .and_then(|f| f.get(idx))
        .copied()
        .unwrap_or(0)
}

/// The output section ordinal for a planned symbol, or 0 (`NO_SECT`) for
/// absolute and undefined symbols.
fn nlist_sect(sym: &PlannedSym, sec_ordinal: &[(usize, u32, u8)]) -> u8 {
    if (sym.n_type & N_TYPE) != N_SECT {
        return 0;
    }
    sec_ordinal
        .iter()
        .find(|(f, s, _)| *f == sym.file && *s == sym.input_sect_ordinal())
        .map_or(0, |(_, _, o)| *o)
}

impl PlannedSym {
    /// The 1-based input section ordinal the symbol was defined in.
    fn input_sect_ordinal(&self) -> u32 {
        u32::from(self.input_n_sect)
    }
}

/// Writes the file-backed section bytes, applying each input section's
/// relocations in place. Delegates to the [`sections`] module.
fn write_sections(
    image: &mut [u8],
    inputs: &[MachOFile<'_>],
    target: MachoTarget,
    layout: &MachLayout,
    sym_addr: &[Vec<u64>],
    got_addr: &[Vec<u64>],
) -> Result<()> {
    let in_addr = input_section_addrs(inputs);
    let tables = sections::AddrTables {
        sym_addr,
        got_addr,
        sec_vaddr: &layout.sec_vaddr,
        in_addr: &in_addr,
    };
    sections::write(image, inputs, target, layout, &tables)
}

/// Rewrites the third word of each Darwin TLV descriptor. Object files spell
/// it as a relocation to `$tlv$init`; final images store an offset from the
/// start of the thread-local template, which `_tlv_bootstrap` consumes.
fn write_tls_descriptors(
    image: &mut [u8],
    inputs: &[MachOFile<'_>],
    layout: &MachLayout,
    sym_addr: &[Vec<u64>],
) -> Result<()> {
    let Some(data) = layout.data.as_ref() else {
        return Ok(());
    };
    let tls_base = data
        .sections
        .iter()
        .find(|s| trim_nul(&s.sectname) == b"__thread_data")
        .or_else(|| {
            data.sections
                .iter()
                .find(|s| trim_nul(&s.sectname) == b"__thread_bss")
        })
        .map(|s| s.addr);
    let Some(tls_base) = tls_base else {
        return Ok(());
    };
    let Some(vars) = data
        .sections
        .iter()
        .find(|s| trim_nul(&s.sectname) == b"__thread_vars")
    else {
        return Ok(());
    };
    for member in &vars.members {
        let input = inputs
            .get(member.file)
            .ok_or(Error::OutOfRange("TLV descriptor input"))?;
        let section = input
            .sections()
            .into_iter()
            .find(|s| s.index == member.section)
            .ok_or(Error::OutOfRange("TLV descriptor section"))?;
        let symbols: Vec<_> = input.symbols().iter().collect();
        for reloc in &section.relocations {
            if !reloc.r_extern
                || u32::from(reloc.r_type) != 0
                || reloc.r_length != 3
            {
                continue;
            }
            let Some(sym) = symbols.get(reloc.r_symbolnum as usize) else {
                continue;
            };
            if !sym.name.ends_with(b"$tlv$init") {
                continue;
            }
            let address = sym_addr
                .get(member.file)
                .and_then(|file| file.get(reloc.r_symbolnum as usize))
                .copied()
                .ok_or(Error::OutOfRange("TLV initializer symbol"))?;
            let value = address
                .checked_sub(tls_base)
                .ok_or(Error::Format("TLV initializer precedes template"))?;
            let off = vars
                .offset
                .wrapping_add(member.offset)
                .wrapping_add(u64::from(reloc.r_address));
            let start = usize::try_from(off)
                .map_err(|_| Error::OutOfRange("TLV descriptor offset"))?;
            let dst = image
                .get_mut(start..start.saturating_add(8))
                .ok_or(Error::OutOfRange("TLV descriptor bytes"))?;
            dst.copy_from_slice(&value.to_le_bytes());
        }
    }
    Ok(())
}

/// Per file, per 1-based section ordinal: the address the input file gave that
/// section.
///
/// A section-relative relocation stores its target as an address in the input
/// file's own frame, so turning it into an offset within the referent needs
/// the frame it was written in.
fn input_section_addrs(inputs: &[MachOFile<'_>]) -> Vec<Vec<u64>> {
    inputs
        .iter()
        .map(|input| input.sections().iter().map(|s| s.addr).collect())
        .collect()
}

/// Builds the per-file, per-symbol `__got` slot address table consumed by the
/// resolver. Entry `i` of file `f` is the address of the slot for that file's
/// `i`-th symbol (`got.addr + slot * 8` when the symbol has a slot, else zero).
/// Symbols are matched by name against the plan, so one global referenced from
/// several files resolves to the same slot in each.
fn build_got_addr(
    inputs: &[MachOFile<'_>],
    got_plan: &GotPlan<'_>,
    got: Option<GotLayout>,
) -> Vec<Vec<u64>> {
    let mut out = Vec::with_capacity(inputs.len());
    let Some(got) = got else {
        for input in inputs {
            out.push(vec![0u64; input.symbols().len()]);
        }
        return out;
    };
    for (file, input) in inputs.iter().enumerate() {
        let syms = input.symbols();
        let mut addrs = vec![0u64; syms.len()];
        for (i, sym) in syms.iter().enumerate() {
            let external = sym.n_type & N_EXT != 0;
            if let Some(slot) = got_plan.slot(file, i, sym.name, external)
                && let Some(dst) = addrs.get_mut(i)
            {
                *dst = got.addr.wrapping_add(u64::from(slot) * 8);
            }
        }
        out.push(addrs);
    }
    out
}

/// Writes the resolved address of each GOT entry's symbol into its 8-byte slot.
/// For a fully-static executable the entries are link-time-resolved (no dyld
/// bind opcodes), so each slot holds the symbol's runtime virtual address.
fn fill_got(
    image: &mut [u8],
    got: GotLayout,
    got_plan: &GotPlan<'_>,
    globals: &Globals<'_>,
    sym_addr: &[Vec<u64>],
) {
    let base = usize::try_from(got.offset).unwrap_or(usize::MAX);
    for (slot, key) in got_plan.iter() {
        let value = match key {
            GotKey::Global(name) => global_addr(globals, sym_addr, name),
            // A file-private referent resolves in its own file, which is the
            // whole point of keying it that way.
            GotKey::Local { file, sym } => sym_addr
                .get(file)
                .and_then(|f| f.get(usize::try_from(sym).unwrap_or(usize::MAX)))
                .copied()
                .unwrap_or(0),
        };
        let off =
            base.wrapping_add(usize::try_from(slot).unwrap_or(usize::MAX) * 8);
        if let Some(dst) = image.get_mut(off..off.saturating_add(8)) {
            dst.copy_from_slice(&value.to_le_bytes());
        }
    }
}

/// Resolves a global symbol's runtime address through the definition table.
/// Falls back to zero if the name has no global definition (matches the
/// `symbol_addr` path, which would have already reported an undefined ref).
fn global_addr(
    globals: &Globals<'_>,
    sym_addr: &[Vec<u64>],
    name: &[u8],
) -> u64 {
    let Some(def) = globals.get(name) else {
        return 0;
    };
    let sym = usize::try_from(def.sym).unwrap_or(usize::MAX);
    sym_addr
        .get(def.file)
        .and_then(|f| f.get(sym))
        .copied()
        .unwrap_or(0)
}
