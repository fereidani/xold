//! Address resolution: turning every referenced symbol into the value its
//! relocations are applied against.
//!
//! Placement has already fixed where each input section lands, so this pass
//! reads that back and writes one row per symbol of every input file. Globals
//! are resolved once into a dense table indexed by [`SymbolId`]; locals are
//! resolved per file, which is why the files can be walked in parallel. A TLS
//! symbol resolves to its offset from the thread pointer, not to a virtual
//! address.

use rayon::prelude::*;
use rustc_hash::FxHashSet;

use super::{
    GotKey, GotKind, GotOwner, Layout, SymFlags, SymRows, UNPLACED,
    collect_exports, fill_got, got_slot, plt_slot,
};
use crate::{
    dynamic::LinkMode,
    elf::{
        Sym64,
        constants::{SHN_ABS, SHN_COMMON, SHN_UNDEF, STB_LOCAL, STT_TLS},
    },
    error::Result,
    input::InputFile,
    linker::Context,
    merge::MergePlan,
    reloc::Target,
    symbol::{DefSource, Definition, SymbolId, SymbolKind},
    tls::{TlsBlock, tpoff},
};

/// Resolves every referenced symbol's address and fills the GOT. A TLS symbol
/// (`STT_TLS`) resolves to its thread-pointer-relative offset (TPOFF), so the
/// local-exec relocation arithmetic (`S + A`) yields `TPOFF + A`; every other
/// symbol resolves to its runtime vaddr.
pub(super) fn resolve_addresses(
    ctx: &Context<'_>,
    target: Target,
    common_off: &[u64],
    sym_id: &[Vec<Option<SymbolId>>],
    layout: &mut Layout,
) -> Result<()> {
    let tls = TlsCtx {
        target,
        block: layout.tls_block(),
    };
    let mut global_addr = resolve_globals(ctx, tls, layout, common_off);
    // An indirect function's symbol names its resolver, which no reference
    // should reach: they all go through the stub whose slot the runtime fills
    // with what the resolver returned.
    let resolvers: Vec<u64> = layout
        .ifunc
        .iter()
        .map(|id| global_addr.get(id.0).copied().unwrap_or_default())
        .collect();
    for &id in &layout.ifunc {
        let stub = plt_slot(layout, GotKey::addr(id));
        if stub != 0
            && let Some(slot) = global_addr.get_mut(id.0)
        {
            *slot = stub;
        }
    }
    layout.ifunc_resolvers = resolvers;
    // The linker's own symbols name bounds of the layout, which only now
    // exist. Each one also reports the section its address belongs to, which
    // the output symbol tables need: a bound published `SHN_ABS` would keep
    // its link-time value in an image the loader placed elsewhere.
    let defsym_shndx = crate::defsym::resolve(ctx, layout, &mut global_addr);
    // Each file owns a disjoint row, so hand every thread its own and read the
    // rest of the layout (section addresses, GOT/PLT indices) immutably. The
    // rows come out of the layout for the duration and go straight back, which
    // keeps `Layout` free of interior mutability.
    let facts = global_facts(ctx, layout.mode());
    let slots = slot_cache(layout, global_addr.len(), ctx.files.len());
    let rc = ResolveCtx {
        tls,
        global_addr: &global_addr,
        sym_id,
        merge: &ctx.merge,
        facts: &facts,
        slots: &slots,
    };
    let mut rows = std::mem::take(&mut layout.sym);
    let result = rows
        .par_iter_mut()
        .zip(ctx.files.par_iter())
        .enumerate()
        .try_for_each(|(file, (row, input))| {
            resolve_file_symbols(rc, input, file, layout, row)
        });
    layout.sym = rows;
    result?;
    layout.tls_index = got_slot(layout, GotKey::MODULE);
    fill_got(&global_addr, layout);
    collect_exports(ctx, tls.block, &global_addr, &defsym_shndx, layout);
    Ok(())
}

/// The relaxation facts of one resolved global.
#[derive(Clone, Copy, Default)]
struct GlobalFacts {
    /// Whether another image may define this name instead of this one.
    preemptible: bool,
    /// Whether the symbol's value is a link-time constant rather than a place
    /// in the image, which is lld's `isAbsoluteValue`: an `SHN_ABS`
    /// definition, a reference nothing defines, or a thread-local, whose value
    /// is an offset from the thread pointer. The linker's own bounds (`_end`
    /// and friends) are recorded as absolute yet name real addresses, so they
    /// are reported as constants too -- which only ever declines a rewrite.
    absolute: bool,
    /// Whether the symbol is a weak reference that nothing defines, which is
    /// what redirects a PC-relative site to
    /// [`crate::reloc::Arch::undef_weak_pc`].
    undef_weak: bool,
}

/// The relaxation facts of every resolved global, indexed by [`SymbolId`].
///
/// Computed once for the whole link so the per-file pass answers by index
/// rather than re-deriving them per reference.
fn global_facts(ctx: &Context<'_>, mode: LinkMode) -> Vec<GlobalFacts> {
    let shared = mode == LinkMode::Shared;
    ctx.symbols
        .ids()
        .map(|id| {
            let Some(sym) = ctx.symbols.symbol(id) else {
                return GlobalFacts::default();
            };
            // A weak reference a dependency defines is not undefined: the
            // loader binds it, and its static-site value is the copy slot
            // or PLT entry resolution assigned. The reference must also be
            // able to reach the dynamic table -- a hidden one never gets a
            // .dynsym row, so the loader cannot bind it however many
            // dependencies define the name, and it stays an undefined weak
            // (the same pairing the scan's import classification requires).
            // Only then does the architecture's undefined-weak answer step
            // aside (lld draws the line by making a bindable dep-defined
            // name a SharedSymbol rather than an Undefined).
            let dep_defined = || {
                crate::symbol::exports_to_dynsym(sym.visibility)
                    && ctx.dep_exports.get(ctx.symbols.name(id)).is_some()
            };
            GlobalFacts {
                preemptible: sym.is_preemptible(shared),
                absolute: sym.is_absolute_value(),
                undef_weak: matches!(
                    sym.kind,
                    SymbolKind::Undefined { weak: true }
                ) && !dep_defined(),
            }
        })
        .collect()
}

/// The architecture and laid-out TLS block, bundled so they flow through
/// resolution as one value. `Copy` so it is passed by value without borrowing
/// the layout.
#[derive(Clone, Copy)]
struct TlsCtx {
    target: Target,
    block: Option<TlsBlock>,
}

/// The read-only tables the per-file resolution pass consults, bundled so they
/// travel as one argument.
#[derive(Clone, Copy)]
struct ResolveCtx<'a> {
    tls: TlsCtx,
    /// Resolved address of every global, indexed by [`SymbolId`].
    global_addr: &'a [u64],
    /// Per file, per input symbol: the resolved global id.
    sym_id: &'a [Vec<Option<SymbolId>>],
    /// Deduplicated `SHF_MERGE` content, for the sections whose symbols
    /// resolve to a pool base.
    merge: &'a MergePlan,
    /// The relaxation facts of each global, indexed by [`SymbolId`].
    facts: &'a [GlobalFacts],
    /// Every GOT and PLT slot address, spread by owner.
    slots: &'a SlotCache,
}

/// Every GOT and PLT slot address, spread out by owner: a dense array per
/// global, a short list per file for the file-local entries.
///
/// The per-file pass fills four slot columns for every symbol of every input,
/// and probing the `got_index`/`plt_index` hash maps for each was a few
/// million probes that mostly missed. The maps are inverted here once: a
/// global's slots are then array reads, and a file's local slots -- rare --
/// are applied from its own list after the loop, so the loop probes nothing.
struct SlotCache {
    /// By global id: the `GotKind::Addr` slot address, or 0.
    got: Vec<u64>,
    /// By global id: the PLT entry address, or 0.
    plt: Vec<u64>,
    /// By global id: the TLS slot address, or 0. `TlsOffset` is preferred
    /// over `TlsModule`, the same preference the per-symbol probe applied.
    tls: Vec<u64>,
    /// By file: the slots owned by that file's local symbols.
    local: Vec<Vec<LocalSlot>>,
}

/// One file-local slot: the symbol row it belongs to, and what it holds.
struct LocalSlot {
    sym_idx: u32,
    what: SlotUse,
    addr: u64,
}

/// Which column of the per-symbol rows a [`LocalSlot`] fills.
enum SlotUse {
    Got,
    Plt,
    TlsOffset,
    TlsModule,
}

/// Inverts the GOT and PLT indices into a [`SlotCache`].
///
/// The map iteration order is not an ordering; every write below is keyed by
/// `(owner, kind)`, which each map holds at most once, and the one place two
/// entries can meet -- a symbol with both TLS kinds -- is resolved by
/// preference (`TlsOffset` wins), not by arrival.
fn slot_cache(layout: &Layout, globals: usize, files: usize) -> SlotCache {
    let mut cache = SlotCache {
        got: vec![0; globals],
        plt: vec![0; globals],
        tls: vec![0; globals],
        local: (0..files).map(|_| Vec::new()).collect(),
    };
    if layout.got.size != 0 {
        for key in layout.got_index.keys() {
            let addr = got_slot(layout, *key);
            let what = match key.kind {
                GotKind::Addr => SlotUse::Got,
                GotKind::TlsOffset => SlotUse::TlsOffset,
                GotKind::TlsModule => SlotUse::TlsModule,
            };
            record_slot(&mut cache, key.owner, what, addr);
        }
    }
    if layout.plt.size != 0 {
        for key in layout.plt_index.keys() {
            let addr = plt_slot(layout, *key);
            record_slot(&mut cache, key.owner, SlotUse::Plt, addr);
        }
    }
    cache
}

/// Files one slot under its owner.
fn record_slot(
    cache: &mut SlotCache,
    owner: GotOwner,
    what: SlotUse,
    addr: u64,
) {
    match owner {
        GotOwner::Global(id) => {
            let column = match what {
                SlotUse::Got => &mut cache.got,
                SlotUse::Plt => &mut cache.plt,
                SlotUse::TlsOffset | SlotUse::TlsModule => &mut cache.tls,
            };
            if let Some(slot) = column.get_mut(id.0) {
                // The module half of a TLS pair yields to the offset slot,
                // whichever the map hands over first.
                if !matches!(what, SlotUse::TlsModule) || *slot == 0 {
                    *slot = addr;
                }
            }
        }
        GotOwner::Local(file, sym_idx) => {
            if let Some(list) = cache.local.get_mut(file) {
                list.push(LocalSlot {
                    sym_idx,
                    what,
                    addr,
                });
            }
        }
        GotOwner::Module => {}
    }
}

/// The thread-pointer-relative offset of a TLS symbol whose plain vaddr is
/// `sym_vaddr`. If no `PT_TLS` block exists (malformed input), the vaddr is
/// returned unchanged so a downstream range check, not a panic, surfaces the
/// error.
fn tls_value(tls: TlsCtx, sym_vaddr: u64) -> u64 {
    let Some(block) = tls.block else {
        return sym_vaddr;
    };
    let sym_off = sym_vaddr.wrapping_sub(block.vaddr);
    tpoff(tls.target, sym_off, &block)
}

/// Resolves the address of every global symbol (`Defined`, `Common`, etc.).
/// A TLS definition yields its TPOFF; other definitions yield their vaddr. A
/// symbol a copy relocation covers -- the one that selected the slot, or any
/// alias the dependency defines beside it -- resolves to that `.bss` slot.
fn resolve_globals(
    ctx: &Context<'_>,
    tls: TlsCtx,
    layout: &Layout,
    common_off: &[u64],
) -> Vec<u64> {
    let mut out = vec![0u64; ctx.symbols.len()];
    let mut copied: FxHashSet<SymbolId> = FxHashSet::default();
    for (id, addr) in layout.copy_bindings() {
        if id.0 < out.len() {
            out[id.0] = addr;
        }
        copied.insert(id);
    }
    for id in ctx.symbols.ids() {
        // Copy-relocated symbols already hold their slot address; do not
        // overwrite it with the undefined fallback (zero).
        if copied.contains(&id) {
            continue;
        }
        let Some(sym) = ctx.symbols.symbol(id) else {
            continue;
        };
        let addr = match &sym.kind {
            SymbolKind::Defined(def) => {
                let vaddr = definition_addr(&ctx.merge, layout, def);
                if def.sym_type == STT_TLS {
                    tls_value(tls, vaddr)
                } else {
                    vaddr
                }
            }
            SymbolKind::Common { .. } => layout.bss.vaddr.wrapping_add(
                common_off.get(id.0).copied().unwrap_or_default(),
            ),
            SymbolKind::Undefined { .. } => {
                // A canonical-PLT import resolves to its PLT entry address so
                // every absolute reference (e.g. a `.eh_frame` CIE personality
                // pointer) lands on the fixed-address stub the loader binds.
                if layout.canonical_plt.contains(&id) {
                    plt_slot(layout, GotKey::addr(id))
                } else {
                    0
                }
            }
        };
        out[id.0] = addr;
    }
    out
}

/// The runtime address of a resolved definition.
///
/// A definition inside a section the merge pass deduplicated is at the offset
/// that pass moved it to, not the one it was written at: `sec_vaddr` is the
/// pool's base, and the piece the symbol names may have landed anywhere in it
/// or been folded onto an identical copy from another file.
pub(super) fn definition_addr(
    merge: &MergePlan,
    layout: &Layout,
    def: &Definition,
) -> u64 {
    match def.source {
        DefSource::Section { index, offset } => {
            // A definition in a section placement dropped has no address to
            // give. The value stays what it always was -- zero plus the
            // offset -- because the writer is where a reference to it is
            // caught, and it needs the reference's own section to know
            // whether the answer is a diagnostic or a tombstone.
            let base = placed_base(layout, def.file, index);
            let moved = merge.remap(def.file, index, offset).unwrap_or(offset);
            base.wrapping_add(moved)
        }
        DefSource::Absolute { value } => value,
    }
}

/// The per-file tables symbol resolution reads alongside an input's own symbol
/// table. `Copy` so it travels as one argument.
#[derive(Clone, Copy)]
struct FileSyms<'a> {
    /// The input's index among the link's files.
    file: usize,
    /// The file's `sym_idx -> SymbolId` row, empty at a local symbol's index.
    sym_id: &'a [Option<SymbolId>],
    /// The file's row of placed section addresses.
    sec_vaddr: Option<&'a [u64]>,
}

/// Resolves every input symbol of one file to an address, for a link with
/// neither a GOT nor a PLT.
///
/// Every slot row keeps its initialised zero in that case, so only the address
/// row is touched. That is the common case for a static non-PIC link, and it
/// lets the loop zip the pre-sized row against the symbol slice with no
/// per-symbol bounds checks.
fn resolve_addr_row(
    rc: ResolveCtx<'_>,
    syms: FileSyms<'_>,
    input_syms: &[Sym64],
    addr: &mut [u64],
) {
    for (i, (slot, sym)) in addr.iter_mut().zip(input_syms).enumerate() {
        *slot = symbol_addr(rc, syms, i, sym);
    }
}

/// The resolved value of one input symbol: a local's own address, or the
/// precomputed address of the global it resolved to.
fn symbol_addr(
    rc: ResolveCtx<'_>,
    syms: FileSyms<'_>,
    i: usize,
    sym: &Sym64,
) -> u64 {
    if sym.bind() == STB_LOCAL {
        let merged = merged(rc.merge, syms.file, sym);
        local_addr(rc.tls, syms.sec_vaddr, sym, merged)
    } else {
        syms.sym_id
            .get(i)
            .copied()
            .flatten()
            .and_then(|id| rc.global_addr.get(id.0).copied())
            .unwrap_or_default()
    }
}

/// Fills the flag row for a link with no GOT or PLT, where the facts are all
/// that is asked of a symbol and no slot was ever allocated.
fn resolve_flags_row(
    rc: ResolveCtx<'_>,
    syms: FileSyms<'_>,
    input_syms: &[Sym64],
    flags: &mut [SymFlags],
) {
    for (i, (slot, sym)) in flags.iter_mut().zip(input_syms).enumerate() {
        let id = if sym.bind() == STB_LOCAL {
            None
        } else {
            syms.sym_id.get(i).copied().flatten()
        };
        *slot = sym_flags(rc.facts, id, sym, false);
    }
}

/// The relaxation facts for one input symbol. `id` is its resolved global id,
/// or `None` for a file-local symbol, which no other image can even name and
/// so can never be preempted; whether such a symbol is a constant is read off
/// the input entry, since nothing resolved it.
fn sym_flags(
    facts: &[GlobalFacts],
    id: Option<SymbolId>,
    sym: &Sym64,
    got_slot: bool,
) -> SymFlags {
    let global = id.and_then(|id| facts.get(id.0).copied());
    SymFlags {
        preemptible: global.is_some_and(|f| f.preemptible),
        absolute: global.map_or_else(|| local_is_absolute(sym), |f| f.absolute),
        // A file-local entry is a definition or a local reference; either
        // way it is not the global feature probe the flag exists for.
        undef_weak: global.is_some_and(|f| f.undef_weak),
        got_slot,
    }
}

/// Whether a file-local symbol's value is a link-time constant: an `SHN_ABS`
/// value, a thread-pointer offset, or the zero an undefined entry stands for.
fn local_is_absolute(sym: &Sym64) -> bool {
    matches!(sym.st_shndx.get(), SHN_ABS | SHN_UNDEF) || sym.type_() == STT_TLS
}

/// Resolves every input symbol of one file to an address and a GOT slot.
/// Local symbols come from [`local_addr`]; globals read the precomputed
/// [`resolve_globals`] table via the per-file `sym_id` cache (no per-symbol
/// name probe). A TLS symbol yields its TPOFF either way.
fn resolve_file_symbols(
    rc: ResolveCtx<'_>,
    input: &InputFile<'_>,
    file: usize,
    layout: &Layout,
    row: &mut SymRows,
) -> Result<()> {
    let ResolveCtx {
        sym_id,
        facts,
        slots,
        ..
    } = rc;
    let Some(symtab) = input.symbol_table()? else {
        return Ok(());
    };
    let syms = FileSyms {
        file,
        sym_id: sym_id.get(file).map(Vec::as_slice).unwrap_or_default(),
        sec_vaddr: layout.sec_vaddr.get(file).map(Vec::as_slice),
    };
    let have_got = layout.got.size != 0;
    let have_plt = layout.plt.size != 0;
    if !have_got && !have_plt {
        resolve_addr_row(rc, syms, symtab.syms, &mut row.addr);
        // A link with no tables still names its weak undefined references:
        // their PC-relative sites read `undef_weak` off this row.
        resolve_flags_row(rc, syms, symtab.syms, &mut row.flags);
        return Ok(());
    }
    for (i, sym) in symtab.syms.iter().enumerate() {
        let id_opt = if sym.bind() == STB_LOCAL {
            None
        } else {
            syms.sym_id.get(i).copied().flatten()
        };
        let addr = symbol_addr(rc, syms, i, sym);
        if let Some(slot) = row.addr.get_mut(i) {
            *slot = addr;
        }
        // A global's slots are dense-array reads; a local's stay zero here
        // and the file's own slot list fills them after the loop, so the
        // loop probes no map for either.
        let got =
            id_opt.map_or(0, |id| slots.got.get(id.0).copied().unwrap_or(0));
        if have_got && let Some(slot) = row.got.get_mut(i) {
            *slot = got;
        }
        if let Some(slot) = row.flags.get_mut(i) {
            *slot = sym_flags(facts, id_opt, sym, got != 0);
        }
        if have_plt
            && let Some(id) = id_opt
            && let Some(slot) = row.plt.get_mut(i)
        {
            *slot = slots.plt.get(id.0).copied().unwrap_or(0);
        }
        // Whichever TLS entry the scan made for this symbol, the writer
        // wants its address: only one of the two kinds is ever allocated for
        // a given symbol in a given link.
        if let Some(id) = id_opt
            && let Some(slot) = row.tls_got.get_mut(i)
        {
            *slot = slots.tls.get(id.0).copied().unwrap_or(0);
        }
    }
    // The file-local slots, applied over the zeros the loop left. The list
    // order is a map's and means nothing; each row is keyed at most once per
    // use, except the TLS pair, where the offset slot wins by rule.
    for s in slots.local.get(file).map_or(&[][..], Vec::as_slice) {
        let i = s.sym_idx as usize;
        match s.what {
            SlotUse::Got => {
                if let Some(slot) = row.got.get_mut(i) {
                    *slot = s.addr;
                }
                if let Some(flag) = row.flags.get_mut(i) {
                    flag.got_slot = true;
                }
            }
            SlotUse::Plt => {
                if let Some(slot) = row.plt.get_mut(i) {
                    *slot = s.addr;
                }
            }
            SlotUse::TlsOffset => {
                if let Some(slot) = row.tls_got.get_mut(i) {
                    *slot = s.addr;
                }
            }
            SlotUse::TlsModule => {
                if let Some(slot) = row.tls_got.get_mut(i)
                    && *slot == 0
                {
                    *slot = s.addr;
                }
            }
        }
    }
    Ok(())
}

/// The placed base address of an input section, or zero when placement
/// dropped it. See [`UNPLACED`].
fn placed_base(layout: &Layout, file: usize, index: u16) -> u64 {
    layout
        .sec_vaddr
        .get(file)
        .and_then(|f| f.get(usize::from(index)))
        .copied()
        .filter(|v| *v != UNPLACED)
        .unwrap_or_default()
}

/// Whether a symbol is defined in a section the merge pass deduplicated.
fn merged(merge: &MergePlan, file: usize, sym: &Sym64) -> bool {
    !merge.is_empty() && merge.is_merged(file, sym.st_shndx.get())
}

/// Address of a local (file-scoped) symbol. A local TLS symbol resolves to its
/// TPOFF like a global one. `sec_vaddr` is the file's own row of placed
/// section addresses.
fn local_addr(
    tls: TlsCtx,
    sec_vaddr: Option<&[u64]>,
    sym: &Sym64,
    merged: bool,
) -> u64 {
    match sym.st_shndx.get() {
        SHN_UNDEF | SHN_COMMON => 0,
        SHN_ABS => sym.st_value.get(),
        shndx => {
            // A symbol in a merged section resolves to the pool base. Its
            // value is an offset into content that moved when it was
            // deduplicated, so the writer remaps the whole offset into the
            // relocation's addend instead.
            let value = if merged { 0 } else { sym.st_value.get() };
            let vaddr = sec_vaddr
                .and_then(|f| f.get(usize::from(shndx)))
                .copied()
                .filter(|v| *v != UNPLACED)
                .unwrap_or_default()
                .wrapping_add(value);
            if sym.type_() == STT_TLS {
                tls_value(tls, vaddr)
            } else {
                vaddr
            }
        }
    }
}
