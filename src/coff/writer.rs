//! PE32+ executable assembly and the COFF static link driver.
//!
//! The writer is the PE counterpart of the ELF and Mach-O writers: it turns a
//! [`PeLayout`] into a PE32+ x86-64 console image. It serialises the DOS stub,
//! the PE signature, the file and optional headers with their data directory,
//! and the section table; copies each linked section's bytes while patching
//! relocations in place through the arch-neutral [`crate::reloc::apply`]
//! driver; emits the import table (`.idata`); and writes the linker-generated
//! entry stub that calls the user entry and then `kernel32!ExitProcess`.
//!
//! Like the Mach-O link, this path owns its own resolution, layout and byte
//! output, so the ELF and Mach-O writers are untouched.

use std::path::Path;

use rustc_hash::FxHashMap;

use crate::{
    coff::{
        CoffFile, CoffSymbolTable,
        basereloc::RelocPlan,
        comdat::ComdatPlan,
        commons::CommonsPlan,
        constants::{
            FILE_ALIGNMENT, IMAGE_BASE_DLL_X86_64, IMAGE_BASE_X86_64,
            IMAGE_DIRECTORY_ENTRY_BASERELOC, IMAGE_DIRECTORY_ENTRY_EXPORT,
            IMAGE_DIRECTORY_ENTRY_IAT, IMAGE_DIRECTORY_ENTRY_IMPORT,
            IMAGE_DIRECTORY_ENTRY_RESOURCE, IMAGE_DIRECTORY_ENTRY_TLS,
            IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE,
            IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA,
            IMAGE_DLLCHARACTERISTICS_NX_COMPAT, IMAGE_FILE_DLL,
            IMAGE_FILE_EXECUTABLE_IMAGE, IMAGE_FILE_LARGE_ADDRESS_AWARE,
            IMAGE_FILE_LINE_NUMS_STRIPPED, IMAGE_FILE_LOCAL_SYMS_STRIPPED,
            IMAGE_FILE_MACHINE_AMD64, IMAGE_FILE_MACHINE_I386,
            IMAGE_NT_OPTIONAL_HDR64_MAGIC, IMAGE_NT_SIGNATURE,
            IMAGE_NUMBEROF_DIRECTORY_ENTRIES, IMAGE_SIZEOF_DOS_HEADER,
            IMAGE_SIZEOF_OPTIONAL_HEADER64, IMAGE_SUBSYSTEM_WINDOWS_CUI,
            IMAGE_SYM_ABSOLUTE, IMAGE_SYM_CLASS_EXTERNAL, IMAGE_SYM_UNDEFINED,
            SECTION_ALIGNMENT,
        },
        dllmap,
        exports::{ExportPlan, collect_directives},
        imports::{IMP_PREFIX, ImportEntry, ImportPlan},
        layout::{self, ENTRY_STUB_SIZE, OutSection, PeLayout, TLS_DIR_SIZE},
        pe::{
            DataDirectory, DosHeader, NtFileHeader, OptionalHeader64,
            SectionHeader,
        },
        reloc::{CoffTarget, global_symbol_addr},
        sections,
        tls::{self, TlsPlan},
    },
    endian::{U16, U32, U64},
    error::{Error, Result},
    input::{Input, InputBytes},
    pool,
    util::{open_all, show, trim_nul, write_output, write_pod},
};

/// The undefined external clang-msvc emits for the module's TLS slot index.
/// The linker synthesises a dword for it and the loader fills it at load time.
const TLS_INDEX_NAME: &[u8] = b"_tls_index";

/// The conventional name of the image base itself, which MSVC programs read
/// as `&__ImageBase`; the linker defines it to the base.
const IMAGE_BASE_NAME: &[u8] = b"__ImageBase";

/// Links COFF inputs at `paths` into a PE32+ x86-64 image written to `output`.
///
/// `entry` names the entry symbol (conventionally `main` for an executable).
/// When `shared` is set the image is a DLL: it carries `IMAGE_FILE_DLL`, an
/// export directory built from the inputs' `/EXPORT:` directives, and a
/// base-relocation table; its entry point is zero.
pub fn link_coff(
    inputs: &[Input<'_>],
    output: &Path,
    entry: &[u8],
    shared: bool,
) -> Result<()> {
    let opened = open_all(inputs)?;
    let bytes: Vec<&[u8]> = opened.iter().map(InputBytes::bytes).collect();
    // The COFF driver does not share the ELF pipeline, so it sizes its own
    // pool here for the same reason and by the same rule.
    pool::run(&bytes, || {
        let inputs = parse_all(&bytes)?;
        let dll_name = if shared {
            output_basename(output)
        } else {
            Vec::new()
        };
        let image = write_executable(&inputs, entry, &dll_name, shared)?;
        // Nothing may be published that a serialiser could not reach: the
        // writers size the image from the layout, so a write past its end is a
        // sizing bug.
        crate::util::check_truncated_write()?;
        write_output(output, &image)
    })
}

/// The output path's basename (the DLL file name), for the export directory's
/// `Name` field. Falls back to `xold.dll` if the path has no file name.
fn output_basename(output: &Path) -> Vec<u8> {
    output
        .file_name()
        .and_then(|n| n.to_str())
        .map_or(&b"xold.dll"[..], str::as_bytes)
        .to_vec()
}

/// Parses every input as a bare COFF object, in parallel. Each parse is
/// independent (no shared mutable state); `par_iter().collect()` preserves
/// input order so the result matches the serial baseline exactly.
fn parse_all<'d>(bytes: &'d [&'d [u8]]) -> Result<Vec<CoffFile<'d>>> {
    use rayon::prelude::*;
    bytes.par_iter().map(|b| CoffFile::parse(b)).collect()
}

/// Builds the full PE32+ byte image for `inputs`.
fn write_executable(
    inputs: &[CoffFile<'_>],
    entry: &[u8],
    dll_name: &[u8],
    shared: bool,
) -> Result<Vec<u8>> {
    let target = derive_target(inputs)?;
    let (mut imports, import_size) = plan_imports(inputs, shared)?;
    let (mut exports, export_size) = plan_exports(inputs, dll_name, shared)?;

    let entry_stub_size = if shared { 0 } else { ENTRY_STUB_SIZE };
    let image_base = if shared {
        IMAGE_BASE_DLL_X86_64
    } else {
        IMAGE_BASE_X86_64
    };

    // Detect `__declspec(thread)` storage. When present, reserve a `.tlsdir`
    // trailer for the TLS directory, the null callbacks and `_tls_index`.
    let needs_tls = tls::has_tls(inputs);
    let tls_meta_size = if needs_tls { TLS_DIR_SIZE } else { 0 };

    // Tentative definitions (`int g;` at file scope) and weak externals are
    // the linker's to allocate and to resolve; neither is an input section.
    let commons = CommonsPlan::new(inputs);
    // One copy of each inline function, template instantiation and vtable
    // survives; the rest are not placed and their references go to the winner.
    let comdats = ComdatPlan::new(inputs)?;

    let (layout, reloc) = build_layout(
        inputs,
        entry_stub_size,
        import_size,
        export_size,
        tls_meta_size,
        commons.size(),
        &comdats,
        image_base,
        shared,
        target,
    )?;

    let (tls, (sym_addr, sym_sec_rva)) = finalize_link(
        inputs,
        &layout,
        &mut imports,
        &mut exports,
        &commons,
        needs_tls,
    )?;

    let total = raw_image_size(&layout);
    let mut image = vec![0u8; usize::try_from(total).unwrap_or(usize::MAX)];
    write_headers(
        &mut image,
        &layout,
        target,
        &imports,
        &exports,
        &reloc,
        tls.as_ref(),
    );
    sections::write(
        &mut image,
        inputs,
        target,
        &layout,
        layout.image_base,
        &sym_addr,
        &sym_sec_rva,
    )?;
    write_imports(&mut image, &layout, &imports)?;
    // Only when there is a directory to write. An empty plan still has 41
    // bytes of header, and writing them looked for a `.edata` the layout no
    // longer reserves.
    if export_size != 0 {
        write_section_bytes(&mut image, &layout, b".edata", exports.bytes())?;
    }
    write_section_bytes(&mut image, &layout, b".reloc", reloc.bytes())?;
    if let Some(ref plan) = tls {
        write_section_bytes(&mut image, &layout, b".tlsdir", plan.bytes())?;
    }
    if !shared {
        write_entry_stub(
            &mut image, &layout, inputs, &sym_addr, entry, &imports,
        )?;
    }
    Ok(image)
}

/// Plans the import tables. The returned size is zero when nothing is
/// imported, so no `.idata` is reserved.
fn plan_imports(
    inputs: &[CoffFile<'_>],
    shared: bool,
) -> Result<(ImportPlan, u32)> {
    let entries = detect_imports(inputs, shared)?;
    let imports = ImportPlan::new(&entries);
    let size = if imports.is_empty() {
        0
    } else {
        imports.size()
    };
    Ok((imports, size))
}

/// Plans the export directory. The returned size is zero when nothing is
/// exported, so no `.edata` is reserved.
fn plan_exports(
    inputs: &[CoffFile<'_>],
    dll_name: &[u8],
    shared: bool,
) -> Result<(ExportPlan, u32)> {
    let entries = if shared {
        collect_directives(inputs)?
    } else {
        Vec::new()
    };
    let exports = ExportPlan::new(&entries, dll_name);
    // An empty plan still serialises its 41-byte header, so reserving on
    // `size() > 0` reserved it always: every executable carried an export
    // directory advertising nothing. Imports already ask the question this
    // way round.
    let size = if entries.is_empty() {
        0
    } else {
        exports.size()
    };
    Ok((exports, size))
}

/// Finalises the plans against the built layout: gives the import and export
/// directories their RVAs, builds the TLS plan and resolves every symbol to
/// its virtual address.
fn finalize_link<'data>(
    inputs: &[CoffFile<'data>],
    layout: &PeLayout,
    imports: &mut ImportPlan,
    exports: &mut ExportPlan,
    commons: &CommonsPlan<'data>,
    needs_tls: bool,
) -> Result<(Option<TlsPlan>, SymbolAddrs)> {
    let idata_rva = layout::section_rva(layout, b".idata").map(|(rva, _)| rva);
    if let Some(base) = idata_rva {
        imports.finalize(base);
    }
    let tls = finalize_tls(inputs, layout, needs_tls)?;
    let (sym_addr, sym_sec_rva) =
        resolve_symbol_addrs(inputs, layout, imports, tls.as_ref(), commons)?;
    let edata_rva = layout::section_rva(layout, b".edata").map(|(rva, _)| rva);
    if let Some(base) = edata_rva {
        exports
            .finalize(base, |name| export_rva(inputs, &sym_addr, name, layout));
        verify_exports(exports, &sym_addr, inputs, layout)?;
    }
    Ok((tls, (sym_addr, sym_sec_rva)))
}

/// One past the last raw byte any section occupies: the file's size. An
/// image with no raw section data is headers alone.
fn raw_image_size(layout: &PeLayout) -> u64 {
    layout
        .sections
        .iter()
        .map(|s| {
            u64::from(s.pointer_to_raw_data) + u64::from(s.size_of_raw_data)
        })
        .max()
        .unwrap_or_else(|| u64::from(layout.size_of_headers))
}

/// Builds and finalises the TLS plan when the inputs carry TLS storage. The
/// template RVA and size are read from the laid-out `.tls` section (its
/// virtual size covers the members' alignment gaps, which the per-thread
/// copy must span); the trailer RVA from `.tlsdir`. Returns `None` when
/// there is no TLS.
fn finalize_tls(
    inputs: &[CoffFile<'_>],
    layout: &PeLayout,
    needs_tls: bool,
) -> Result<Option<TlsPlan>> {
    if !needs_tls {
        return Ok(None);
    }
    let (template_rva, template_size) = layout::section_rva(layout, b".tls")
        .ok_or(Error::Format("TLS inputs present but no .tls section"))?;
    let meta_rva = layout::section_rva(layout, b".tlsdir")
        .map(|(rva, _)| rva)
        .ok_or(Error::Format(".tlsdir section missing for TLS directory"))?;
    let mut plan = TlsPlan::new(inputs, template_size, 0);
    plan.finalize(template_rva, meta_rva, layout.image_base);
    Ok(Some(plan))
}

/// Builds the final section layout and the base-relocation plan.
///
/// Base-relocation sites are the absolute (`ADDR64`) fixups in the linked
/// sections. Their RVAs depend only on sections that sort ahead of `.reloc`, so
/// a DLL first builds a provisional layout without `.reloc` to read each site's
/// final RVA, then rebuilds with the table reserved. Adding `.reloc` (the last
/// section) does not shift any earlier section. An executable skips the
/// provisional pass: it keeps a fixed base and carries no `.reloc`.
#[allow(clippy::too_many_arguments)]
fn build_layout(
    inputs: &[CoffFile<'_>],
    entry_stub_size: u32,
    import_size: u32,
    export_size: u32,
    tls_meta_size: u32,
    commons_size: u32,
    comdats: &ComdatPlan,
    image_base: u64,
    shared: bool,
    target: CoffTarget,
) -> Result<(PeLayout, RelocPlan)> {
    let prov = layout::build(
        inputs,
        entry_stub_size,
        import_size,
        export_size,
        0,
        tls_meta_size,
        commons_size,
        comdats,
        image_base,
        shared,
    )?;
    if !shared {
        // An executable keeps a fixed base and carries no `.reloc`.
        return Ok((prov, RelocPlan::new(&[])));
    }
    let mut sites = collect_reloc_sites(inputs, &prov, target);
    // The synthesized TLS directory holds four absolute VAs of its own. They
    // sit ahead of `.reloc`, which is placed last, so the provisional layout
    // already gives their final RVAs.
    if tls_meta_size != 0
        && let Some((meta_rva, _)) = layout::section_rva(&prov, b".tlsdir")
    {
        sites.extend_from_slice(&tls::directory_fixup_rvas(meta_rva));
    }
    let reloc = RelocPlan::new(&sites);
    let layout = layout::build(
        inputs,
        entry_stub_size,
        import_size,
        export_size,
        reloc.size(),
        tls_meta_size,
        commons_size,
        comdats,
        image_base,
        shared,
    )?;
    Ok((layout, reloc))
}

/// Writes a synthetic section's precomputed bytes into its slot.
fn write_section_bytes(
    image: &mut [u8],
    layout: &PeLayout,
    name: &[u8],
    bytes: &[u8],
) -> Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    let section = layout
        .sections
        .iter()
        .find(|s| trim_nul(&s.name) == name)
        .ok_or(Error::Format("synthetic section missing"))?;
    let slot = section_slot(
        image,
        section,
        bytes.len(),
        "synthetic section copy",
        "synthetic section slot",
    )?;
    slot.copy_from_slice(bytes);
    Ok(())
}

/// The image slot a section's raw data occupies.
///
/// `copy` and `slot` name the section in the two out-of-range diagnostics, so
/// each caller reports its own rather than a generic one.
fn section_slot<'a>(
    image: &'a mut [u8],
    section: &OutSection,
    len: usize,
    copy: &'static str,
    slot: &'static str,
) -> Result<&'a mut [u8]> {
    let off = usize::try_from(section.pointer_to_raw_data).unwrap_or(0);
    let end = off.checked_add(len).ok_or(Error::OutOfRange(copy))?;
    image.get_mut(off..end).ok_or(Error::OutOfRange(slot))
}

/// The RVA of an exported symbol (image-relative), or `None` if it is not
/// defined in the inputs.
fn export_rva(
    inputs: &[CoffFile<'_>],
    sym_addr: &[Vec<u64>],
    name: &[u8],
    layout: &PeLayout,
) -> Option<u32> {
    let va = global_symbol_addr(inputs, sym_addr, name)?;
    va.checked_sub(layout.image_base)
        .and_then(|r| u32::try_from(r).ok())
}

/// Ensures every declared export resolves to a defined symbol; a missing
/// definition would leave a zero RVA in the export table and a broken
/// `GetProcAddress`.
fn verify_exports(
    exports: &ExportPlan,
    sym_addr: &[Vec<u64>],
    inputs: &[CoffFile<'_>],
    layout: &PeLayout,
) -> Result<()> {
    for (name, internal) in exports.resolved_pairs() {
        if export_rva(inputs, sym_addr, internal, layout).is_some() {
            continue;
        }
        // An alias resolves a different symbol than it publishes, so name the
        // one that is actually missing.
        if internal == name {
            return Err(Error::UndefinedReference(format!(
                "export `{}` is not defined in any input",
                String::from_utf8_lossy(name)
            )));
        }
        return Err(Error::UndefinedReference(format!(
            "export `{}` names `{}`, which is not defined in any input",
            String::from_utf8_lossy(name),
            String::from_utf8_lossy(internal)
        )));
    }
    Ok(())
}

/// Collects the final image RVAs of every absolute (`IMAGE_REL_AMD64_ADDR64`)
/// relocation site across the linked sections, for the base-relocation table.
fn collect_reloc_sites(
    inputs: &[CoffFile<'_>],
    layout: &PeLayout,
    target: CoffTarget,
) -> Vec<u32> {
    use crate::reloc::coff_x86_64::IMAGE_REL_AMD64_ADDR64;
    if target != CoffTarget::X86_64 {
        return Vec::new();
    }
    let absolute = absolute_symbols(inputs);
    let mut sites = Vec::new();
    for (file, section_ordinal, section_rva, member_off) in
        build_section_map(layout)
    {
        let Some(input) = inputs.get(file) else {
            continue;
        };
        let Some(section) = input.section_at(section_ordinal) else {
            continue;
        };
        for reloc in &section.relocations {
            if u32::from(reloc.typ) != IMAGE_REL_AMD64_ADDR64 {
                continue;
            }
            // An absolute symbol is a constant, not an address, so the
            // loader must not add the base delta to it.
            if absolute.get(file).is_some_and(|indices| {
                indices.binary_search(&reloc.symbol_table_index).is_ok()
            }) {
                continue;
            }
            let rva = section_rva
                .wrapping_add(u32::try_from(member_off).unwrap_or(0))
                .wrapping_add(reloc.virtual_address);
            sites.push(rva);
        }
    }
    sites
}

/// The symbol-table index of every absolute symbol in each input, ascending
/// so a site can be tested with a binary search.
///
/// An `IMAGE_SYM_ABSOLUTE` symbol holds a constant rather than an address:
/// its value means the same wherever the image loads, so a `.reloc` entry
/// naming one has the loader add the base delta to a number that was never a
/// pointer. lld skips a `DefinedAbsolute` target when it collects base
/// relocations (`lld/COFF/Chunks.cpp`), and so does this.
fn absolute_symbols(inputs: &[CoffFile<'_>]) -> Vec<Vec<u32>> {
    inputs
        .iter()
        .map(|input| {
            let mut indices: Vec<u32> = input
                .symbols()
                .iter()
                .filter(|sym| sym.is_absolute())
                .map(|sym| sym.index)
                .collect();
            // `symbols()` yields ascending indices already; the sort states
            // the invariant the binary search depends on.
            indices.sort_unstable();
            indices
        })
        .collect()
}

/// The DLL the entry stub terminates through, plus its imported function.
const EXIT_PROCESS_DLL: &[u8] = b"kernel32.dll";
const EXIT_PROCESS_FUNC: &[u8] = b"ExitProcess";

/// Collects the import set from the inputs' `__imp_<func>` references.
///
/// clang `-msvc` emits a `dllimport` call as a reference to the undefined
/// external `__imp_<func>` (the IAT slot address); the function name is the
/// symbol name without the `__imp_` prefix. The owning DLL is not recorded in
/// the object, so [`dllmap`] supplies it from a well-known table. An executable
/// always imports `kernel32.dll!ExitProcess` for its entry stub; a DLL has no
/// stub, and imports exactly what its own objects reference -- including
/// `ExitProcess` itself when the DLL's code terminates the process.
///
/// The stub's seed and an object's reference to the same function name one
/// import row, so the two never describe the same IAT slot twice.
fn detect_imports(
    inputs: &[CoffFile<'_>],
    shared: bool,
) -> Result<Vec<ImportEntry>> {
    let mut entries = if shared {
        Vec::new()
    } else {
        vec![ImportEntry {
            dll: EXIT_PROCESS_DLL.to_vec(),
            func: EXIT_PROCESS_FUNC.to_vec(),
        }]
    };
    for input in inputs {
        for sym in input.symbols().iter() {
            if sym.storage_class != IMAGE_SYM_CLASS_EXTERNAL
                || sym.section_number != IMAGE_SYM_UNDEFINED
                || sym.value != 0
            {
                continue;
            }
            let Some(func) = sym.name.strip_prefix(IMP_PREFIX) else {
                continue;
            };
            if func.is_empty() || entries.iter().any(|e| e.func == func) {
                continue;
            }
            let dll = dllmap::resolve(func)
                .ok_or_else(|| Error::UndefinedReference(import_error(func)))?;
            entries.push(ImportEntry {
                dll: dll.to_vec(),
                func: func.to_vec(),
            });
        }
    }
    Ok(entries)
}

/// The diagnostic for an `__imp_<func>` reference whose DLL is unknown.
fn import_error(func: &[u8]) -> String {
    let name = String::from_utf8_lossy(func);
    format!(
        "__imp_{name}: no known source DLL \
         (not in the well-known table; supply an import library)"
    )
}

/// Derives the single link target from the inputs' machine type. Every input
/// must agree; a mix is a format error.
fn derive_target(inputs: &[CoffFile<'_>]) -> Result<CoffTarget> {
    let mut target: Option<CoffTarget> = None;
    for input in inputs {
        let current = CoffTarget::from_machine(input.machine())?;
        match target {
            None => target = Some(current),
            Some(existing) if existing == current => {}
            Some(_) => {
                return Err(Error::Format("inputs disagree on machine type"));
            }
        }
    }
    let target =
        target.ok_or(Error::Format("no COFF inputs to derive a target"))?;
    // The reader recognises more machines than the writer can produce an
    // image for. Accepting one it cannot write meant emitting a PE32+ image
    // -- 64-bit optional header, x86-64 image base, 8-byte thunks, an x86-64
    // entry stub -- under an i386 machine word, which Windows refuses. Saying
    // so here is the same refusal, from the place that knows why.
    if !target.is_writable() {
        return Err(Error::Format(
            "only x86-64 COFF objects can be linked: this linker has no PE32 \
             writer",
        ));
    }
    Ok(target)
}

/// Resolves the absolute virtual address of every symbol of every file, for
/// the relocation resolver. Section-defined symbols resolve to the image base
/// plus their laid-out RVA; undefined externals resolve through a global
/// name-to-address map across all files; an `__imp_<func>` undefined external
/// (a `dllimport` reference) resolves to its IAT slot address, which the loader
/// fills with the function pointer. The undefined `_tls_index` external
/// resolves to the synthesised index slot when the image carries TLS.
///
/// Returns [`SymbolAddrs`], both halves indexed by the *raw* symbol-table index
/// (counting auxiliary records), because COFF relocations address symbols by
/// that index. `sec_rvas` carries each symbol's owning output section RVA for
/// the `SECREL` section-offset fixups.
/// Resolved symbol addresses and their owning output section RVAs, each
/// indexed by file and then by raw symbol-table index.
type SymbolAddrs = (Vec<Vec<u64>>, Vec<Vec<u32>>);

fn resolve_symbol_addrs<'data>(
    inputs: &[CoffFile<'data>],
    layout: &PeLayout,
    imports: &ImportPlan,
    tls: Option<&TlsPlan>,
    commons: &CommonsPlan<'data>,
) -> Result<SymbolAddrs> {
    let sec_map = build_section_map(layout);
    let base = layout.image_base;
    let commons_va = commons_base(layout);
    let globals = collect_globals(inputs, &sec_map, commons, commons_va, base)?;
    let tls_index_va = tls.map_or(0, |p| p.index_va(base));
    let mut addrs_out = Vec::with_capacity(inputs.len());
    let mut rvas_out = Vec::with_capacity(inputs.len());
    for (file, input) in inputs.iter().enumerate() {
        let syms = input.symbols();
        let len = syms
            .iter()
            .map(|s| s.index)
            .max()
            .map_or(0, |i| i as usize + 1);
        let mut addrs = vec![0u64; len];
        let mut rvas = vec![0u32; len];
        for sym in syms.iter() {
            let pos = sym.index as usize;
            if pos >= addrs.len() {
                continue;
            }
            if sym.section_number > 0 {
                // A definition in a COMDAT copy that lost has no placement of
                // its own; the reference belongs to the copy that won.
                let (addr, rva) = section_symbol_addr(
                    &sec_map,
                    file,
                    sym.section_number,
                    sym.value,
                    base,
                )
                .unwrap_or_else(|| {
                    (resolve_undefined(&globals, sym.name, base, imports), 0)
                });
                addrs[pos] = addr;
                rvas[pos] = rva;
            } else if sym.section_number == IMAGE_SYM_ABSOLUTE {
                addrs[pos] = u64::from(sym.value);
            } else if sym.is_common() {
                addrs[pos] = commons
                    .offset_of(sym.name)
                    .map_or(0, |off| commons_va.wrapping_add(u64::from(off)));
            } else if sym.section_number == IMAGE_SYM_UNDEFINED
                && sym.value == 0
            {
                if sym.name == TLS_INDEX_NAME {
                    addrs[pos] = tls_index_va;
                } else if sym.name == IMAGE_BASE_NAME {
                    // The image base under its conventional name: MSVC's
                    // `__ImageBase` is how a program reads its own load
                    // address (`&__ImageBase`). The linker defines it the
                    // way lld's `addSynthetic` does with a null chunk
                    // (`lld/COFF/Driver.cpp`).
                    addrs[pos] = base;
                } else {
                    let mut addr =
                        resolve_undefined(&globals, sym.name, base, imports);
                    // A weak external falls back to the definition its
                    // auxiliary record names. Only when nothing defines the
                    // name for real, which the search above already asked.
                    if addr == 0
                        && let Some(tag) = sym.weak_tag_index()
                    {
                        addr = weak_fallback(
                            &sec_map, syms, file, tag, base, &globals, imports,
                        );
                    }
                    addrs[pos] = addr;
                }
            }
        }
        addrs_out.push(addrs);
        rvas_out.push(rvas);
    }
    Ok((addrs_out, rvas_out))
}

/// Every externally visible definition by name, with the input that defined it
/// so a clash can name both.
///
/// A map rather than a list: every global is looked up once when it is defined
/// and once per reference, and scanning a list for each made a link quadratic
/// in its symbol count. Nothing iterates it, so the hash order never reaches
/// the output. A common is a definition too: a reference from a file that does
/// not itself declare the tentative definition still resolves to its storage.
fn collect_globals<'data>(
    inputs: &[CoffFile<'data>],
    sec_map: &[(usize, u32, u32, u64)],
    commons: &CommonsPlan<'data>,
    commons_va: u64,
    base: u64,
) -> Result<FxHashMap<&'data [u8], (u64, usize)>> {
    let mut globals: FxHashMap<&'data [u8], (u64, usize)> =
        FxHashMap::default();
    for (file, input) in inputs.iter().enumerate() {
        for sym in input.symbols().iter() {
            if sym.storage_class != IMAGE_SYM_CLASS_EXTERNAL
                || sym.section_number <= 0
            {
                continue;
            }
            let Some((addr, _rva)) = section_symbol_addr(
                sec_map,
                file,
                sym.section_number,
                sym.value,
                base,
            ) else {
                continue;
            };
            // Two strong definitions of one name was a first-wins race
            // decided by the link line. COMDAT copies are already down to
            // one, so what is left is the error it always was.
            if let Some(&(_, first)) = globals.get(sym.name) {
                return Err(duplicate(sym.name, first, file));
            }
            globals.insert(sym.name, (addr, file));
        }
    }
    for (name, off) in commons.iter() {
        globals
            .entry(name)
            .or_insert_with(|| (commons_va.wrapping_add(u64::from(off)), 0));
    }
    Ok(globals)
}

/// The duplicate-definition error, naming the symbol and the two inputs.
///
/// The COFF pipeline parses bytes rather than paths, so the inputs are named
/// by their position on the link line, which is what the reader has to work
/// with anyway.
fn duplicate(name: &[u8], first: usize, second: usize) -> Error {
    Error::duplicate_symbol(
        String::from_utf8_lossy(name).into_owned(),
        format!("input {first}"),
        format!("input {second}"),
    )
}

/// The virtual address of the common block: the `.bss` RVA plus the block's
/// offset within it. Zero when the image has no commons and hence no block.
fn commons_base(layout: &PeLayout) -> u64 {
    layout::section_rva(layout, b".bss").map_or(0, |(rva, _)| {
        layout
            .image_base
            .wrapping_add(u64::from(rva))
            .wrapping_add(u64::from(layout.commons_off))
    })
}

/// Resolves a weak external through the default definition its auxiliary
/// record names.
///
/// The tag is a raw symbol-table index in the same file, and the default it
/// names may itself be a weak external pointing one hop further: the walk
/// follows the chain. The bound is the symbol table's own length, which no
/// cycle-free chain can exceed; a cycle falls off the end and answers zero,
/// which the caller treats as an undefined weak. A default that is a plain
/// undefined external is resolved by name, which is how an alias chain to an
/// import terminates.
fn weak_fallback(
    sec_map: &[(usize, u32, u32, u64)],
    syms: &CoffSymbolTable<'_>,
    file: usize,
    mut tag: u32,
    base: u64,
    globals: &FxHashMap<&[u8], (u64, usize)>,
    imports: &ImportPlan,
) -> u64 {
    for _ in 0..syms.len() {
        let Some(def) = syms.iter().find(|s| s.index == tag) else {
            return 0;
        };
        if def.section_number > 0 {
            return section_symbol_addr(
                sec_map,
                file,
                def.section_number,
                def.value,
                base,
            )
            .map_or(0, |(addr, _)| addr);
        }
        if def.is_weak()
            && let Some(next) = def.weak_tag_index()
        {
            tag = next;
            continue;
        }
        return resolve_undefined(globals, def.name, base, imports);
    }
    0
}

/// Resolves an undefined external: a global definition by name if one exists,
/// otherwise an imported function's IAT slot (for `__imp_<func>` references).
fn resolve_undefined(
    globals: &FxHashMap<&[u8], (u64, usize)>,
    name: &[u8],
    base: u64,
    imports: &ImportPlan,
) -> u64 {
    if let Some(&(addr, _)) = globals.get(name) {
        return addr;
    }
    let Some(func) = name.strip_prefix(IMP_PREFIX) else {
        return 0;
    };
    imports
        .iat_slot_rva(func)
        .map_or(0, |rva| base.wrapping_add(u64::from(rva)))
}

/// Builds `(file, 1-based section ordinal) -> (rva, member offset within the
/// output section)`.
fn build_section_map(layout: &PeLayout) -> Vec<(usize, u32, u32, u64)> {
    let mut out = Vec::new();
    for section in &layout.sections {
        for m in &section.members {
            out.push((m.file, m.section, section.virtual_address, m.offset));
        }
    }
    // The lookup below is a binary search; a member appears once, so the sort
    // is total and the order it imposes reaches nothing but that search.
    out.sort_unstable_by_key(|&(file, sec, _, _)| (file, sec));
    out
}

/// Resolves a section-defined symbol's address: image base + output section
/// RVA + the member's offset within it + the symbol's section-relative value.
fn section_symbol_addr(
    sec_map: &[(usize, u32, u32, u64)],
    file: usize,
    section_number: i32,
    value: u32,
    base: u64,
) -> Option<(u64, u32)> {
    let ord = u32::try_from(section_number).ok()?;
    // Sorted by `(file, ordinal)`, so this is a search rather than a walk of
    // every placed member: it runs once per symbol of every input.
    let at = sec_map
        .binary_search_by(|(f, s, _, _)| (*f, *s).cmp(&(file, ord)))
        .ok()?;
    sec_map.get(at).map(|(_, _, rva, off)| {
        (base + u64::from(*rva) + off + u64::from(value), *rva)
    })
}

// --- header serialisation -------------------------------------------------

/// Writes the DOS header, the PE signature, the file and optional headers, the
/// data directory and the section table.
fn write_headers(
    image: &mut [u8],
    layout: &PeLayout,
    target: CoffTarget,
    imports: &ImportPlan,
    exports: &ExportPlan,
    reloc: &RelocPlan,
    tls: Option<&TlsPlan>,
) {
    let reloc_dir = layout::section_rva(layout, b".reloc")
        .map(|(rva, _)| reloc.directory(rva));
    // The resource directory names the whole `.rsrc` section: the tree's
    // internal offsets are section-relative, so the loader starts its walk
    // at the section base and the directory's size is the section's.
    let resource_dir =
        layout::section_rva(layout, b".rsrc").map(|(rva, size)| {
            DataDirectory {
                virtual_address: U32::new(rva),
                size: U32::new(size),
            }
        });
    let tls_dir = tls.map(TlsPlan::directory);
    let pe_off = u32::try_from(IMAGE_SIZEOF_DOS_HEADER).unwrap_or(0x40);
    write_pod(image, 0, &DosHeader::minimal(pe_off));
    let mut off = write_pod(
        image,
        u64::from(pe_off),
        &NtFileHeader {
            signature: U32::new(IMAGE_NT_SIGNATURE),
            machine: U16::new(machine(target)),
            number_of_sections: U16::new(
                u16::try_from(layout.sections.len()).unwrap_or(0),
            ),
            time_date_stamp: U32::new(0),
            pointer_to_symbol_table: U32::new(0),
            number_of_symbols: U32::new(0),
            size_of_optional_header: U16::new(
                u16::try_from(
                    IMAGE_SIZEOF_OPTIONAL_HEADER64
                        + IMAGE_NUMBEROF_DIRECTORY_ENTRIES * 8,
                )
                .unwrap_or(240),
            ),
            characteristics: U16::new(file_characteristics(layout.is_dll)),
        },
    );
    off = write_optional_header(image, off, layout);
    off = write_data_directory(
        image,
        off,
        imports,
        exports,
        reloc_dir,
        resource_dir,
        tls_dir,
    );
    let _ = write_section_table(image, off, layout);
}

/// The `IMAGE_FILE_*` characteristics: an executable sets `EXECUTABLE_IMAGE`,
/// a DLL sets `DLL`. Both strip line numbers and locals and are large-address-
/// aware.
fn file_characteristics(is_dll: bool) -> u16 {
    let mode = if is_dll {
        IMAGE_FILE_DLL
    } else {
        IMAGE_FILE_EXECUTABLE_IMAGE
    };
    mode | IMAGE_FILE_LARGE_ADDRESS_AWARE
        | IMAGE_FILE_LINE_NUMS_STRIPPED
        | IMAGE_FILE_LOCAL_SYMS_STRIPPED
}

/// The `IMAGE_FILE_MACHINE_*` value for `target`.
fn machine(target: CoffTarget) -> u16 {
    match target {
        CoffTarget::X86_64 => IMAGE_FILE_MACHINE_AMD64,
        CoffTarget::I386 => IMAGE_FILE_MACHINE_I386,
    }
}

/// Writes `ImageOptionalHeader64` (without the trailing data directory).
fn write_optional_header(image: &mut [u8], off: u64, layout: &PeLayout) -> u64 {
    let (size_of_code, size_of_init) = section_sizes(layout);
    let opt = OptionalHeader64 {
        magic: U16::new(IMAGE_NT_OPTIONAL_HDR64_MAGIC),
        major_linker_version: 14,
        minor_linker_version: 0,
        size_of_code: U32::new(size_of_code),
        size_of_initialized_data: U32::new(size_of_init),
        size_of_uninitialized_data: U32::new(0),
        address_of_entry_point: U32::new(layout.address_of_entry_point),
        base_of_code: U32::new(layout.base_of_code),
        image_base: U64::new(layout.image_base),
        section_alignment: U32::new(SECTION_ALIGNMENT),
        file_alignment: U32::new(FILE_ALIGNMENT),
        major_operating_system_version: U16::new(6),
        minor_operating_system_version: U16::new(0),
        major_image_version: U16::new(0),
        minor_image_version: U16::new(0),
        major_subsystem_version: U16::new(6),
        minor_subsystem_version: U16::new(0),
        win32_version_value: U32::new(0),
        size_of_image: U32::new(layout.size_of_image),
        size_of_headers: U32::new(layout.size_of_headers),
        check_sum: U32::new(0),
        subsystem: U16::new(IMAGE_SUBSYSTEM_WINDOWS_CUI),
        dll_characteristics: U16::new(dll_characteristics(layout.is_dll)),
        size_of_stack_reserve: U64::new(0x10_0000),
        size_of_stack_commit: U64::new(0x1000),
        size_of_heap_reserve: U64::new(0x10_0000),
        size_of_heap_commit: U64::new(0x1000),
        loader_flags: U32::new(0),
        number_of_rva_and_sizes: U32::new(
            u32::try_from(IMAGE_NUMBEROF_DIRECTORY_ENTRIES).unwrap_or(16),
        ),
    };
    write_pod(image, off, &opt)
}

/// The `IMAGE_DLLCHARACTERISTICS_*` flags. An executable keeps a fixed image
/// base (ASLR disabled, no base-relocation table); a DLL opts into ASLR
/// (`DYNAMIC_BASE` with high-entropy VA) and is relocatable through `.reloc`.
fn dll_characteristics(is_dll: bool) -> u16 {
    if is_dll {
        IMAGE_DLLCHARACTERISTICS_NX_COMPAT
            | IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE
            | IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA
    } else {
        IMAGE_DLLCHARACTERISTICS_NX_COMPAT
    }
}

/// Writes the 16-entry data directory, slotting the IMPORT, IAT, EXPORT,
/// RESOURCE, BASERELOC and TLS entries from the import, export, reloc,
/// resource and TLS plans.
fn write_data_directory(
    image: &mut [u8],
    off: u64,
    imports: &ImportPlan,
    exports: &ExportPlan,
    reloc_dir: Option<DataDirectory>,
    resource_dir: Option<DataDirectory>,
    tls_dir: Option<DataDirectory>,
) -> u64 {
    let mut dirs = [DataDirectory::default(); IMAGE_NUMBEROF_DIRECTORY_ENTRIES];
    let [import_dir, iat_dir] = imports.directories();
    dirs[IMAGE_DIRECTORY_ENTRY_IMPORT] = import_dir;
    dirs[IMAGE_DIRECTORY_ENTRY_IAT] = iat_dir;
    dirs[IMAGE_DIRECTORY_ENTRY_EXPORT] = exports.directory();
    if let Some(reloc) = reloc_dir {
        dirs[IMAGE_DIRECTORY_ENTRY_BASERELOC] = reloc;
    }
    if let Some(resource) = resource_dir {
        dirs[IMAGE_DIRECTORY_ENTRY_RESOURCE] = resource;
    }
    if let Some(tls) = tls_dir {
        dirs[IMAGE_DIRECTORY_ENTRY_TLS] = tls;
    }
    let mut cursor = off;
    for d in &dirs {
        cursor = write_pod(image, cursor, d);
    }
    cursor
}

/// Writes the section table: one `ImageSectionHeader` per output section.
fn write_section_table(image: &mut [u8], off: u64, layout: &PeLayout) -> u64 {
    let mut cursor = off;
    for s in &layout.sections {
        let hdr = SectionHeader {
            name: s.name,
            virtual_size: U32::new(s.virtual_size),
            virtual_address: U32::new(s.virtual_address),
            size_of_raw_data: U32::new(s.size_of_raw_data),
            pointer_to_raw_data: U32::new(s.pointer_to_raw_data),
            pointer_to_relocations: U32::new(0),
            pointer_to_linenumbers: U32::new(0),
            number_of_relocations: U16::new(0),
            number_of_linenumbers: U16::new(0),
            characteristics: U32::new(s.characteristics),
        };
        cursor = write_pod(image, cursor, &hdr);
    }
    cursor
}

/// `(size_of_code, size_of_initialized_data)` summed across output sections.
fn section_sizes(layout: &PeLayout) -> (u32, u32) {
    let mut code = 0u32;
    let mut init = 0u32;
    for s in &layout.sections {
        let raw = s.size_of_raw_data;
        if s.name == *b".text\0\0\0" {
            code = code.wrapping_add(raw);
        } else if !s.zerofill {
            init = init.wrapping_add(raw);
        }
    }
    (code, init)
}

// --- import + entry-stub serialisation ------------------------------------

/// Writes the import-table bytes into `.idata`. A no-op when the plan imports
/// nothing (no `.idata` section was reserved).
fn write_imports(
    image: &mut [u8],
    layout: &PeLayout,
    imports: &ImportPlan,
) -> Result<()> {
    if imports.is_empty() {
        return Ok(());
    }
    let section = layout
        .sections
        .iter()
        .find(|s| s.name == *b".idata\0\0")
        .ok_or(Error::Format(".idata section missing"))?;
    let slot = section_slot(
        image,
        section,
        imports.bytes().len(),
        ".idata copy",
        ".idata image slot",
    )?;
    slot.copy_from_slice(imports.bytes());
    Ok(())
}

/// Writes the linker-generated entry stub at the start of `.text`. The stub
/// calls the user entry, moves the return value into `ecx`, and tail-calls
/// `ExitProcess` through its IAT slot (a RIP-relative indirect call).
///
/// The two displacements are computed directly from the final layout, since the
/// stub is linker-generated code (the user-section relocations go through the
/// COFF apply driver; this mirrors how the Mach-O writer fixes `LC_MAIN`
/// without a relocation).
fn write_entry_stub(
    image: &mut [u8],
    layout: &PeLayout,
    inputs: &[CoffFile<'_>],
    sym_addr: &[Vec<u64>],
    entry: &[u8],
    imports: &ImportPlan,
) -> Result<()> {
    let text = layout
        .sections
        .iter()
        .find(|s| s.name == *b".text\0\0\0")
        .ok_or(Error::Format(".text section missing"))?;
    let entry_va = global_symbol_addr(inputs, sym_addr, entry)
        .ok_or_else(|| Error::UndefinedReference(show(entry)))?;
    let exit_rva = imports
        .iat_slot_rva(EXIT_PROCESS_FUNC)
        .ok_or(Error::Format("ExitProcess import missing from the plan"))?;
    let stub_base_va = layout.image_base + u64::from(text.virtual_address);
    let exit_iat_va = layout.image_base + u64::from(exit_rva);

    // stub layout:
    //   0: 48 83 ec 28        sub rsp, 0x28
    //   4: e8 XX XX XX XX     call entry      (disp at 5..9)
    //   9: 89 c1              mov ecx, eax
    //  11: ff 15 XX XX XX XX  call [rip+disp] (disp at 13..17, -> ExitProcess
    // IAT)  17: 0f 0b              ud2
    let mut stub = [
        0x48, 0x83, 0xec, 0x28, 0xe8, 0, 0, 0, 0, 0x89, 0xc1, 0xff, 0x15, 0, 0,
        0, 0, 0x0f, 0x0b,
    ];
    let call_entry = i32::try_from(entry_va.wrapping_sub(stub_base_va + 9))
        .map_err(|_| Error::OutOfRange("entry call displacement"))?;
    stub[5..9].copy_from_slice(&call_entry.to_le_bytes());
    let call_exit = i32::try_from(exit_iat_va.wrapping_sub(stub_base_va + 17))
        .map_err(|_| Error::OutOfRange("ExitProcess call displacement"))?;
    stub[13..17].copy_from_slice(&call_exit.to_le_bytes());

    let slot = section_slot(
        image,
        text,
        stub.len(),
        "entry stub copy",
        "entry stub slot",
    )?;
    slot.copy_from_slice(&stub);
    Ok(())
}
