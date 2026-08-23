//! Linker-defined symbols: the names a C runtime refers to that no input
//! defines.
//!
//! A static glibc walks its constructor arrays between `__init_array_start`
//! and `__init_array_end`, and asks the linker where the image ends so it can
//! seed the heap. Nothing in the inputs defines those names; the linker is
//! expected to, at the bounds of the sections it laid out. Without them a
//! static link fails with `undefined reference to __fini_array_start`.
//!
//! Only a name some input actually referenced is defined, which is what
//! `PROVIDE` means in a linker script: an input that defines one of these for
//! itself keeps its own definition, and a program that never mentions them
//! gets no extra symbols.
//!
//! Each bound names one end of one region, so the two halves live apart:
//! [`define`] runs once the symbol table is final and marks the referenced
//! names as definitions the linker owns, and [`bound`] answers for each one
//! after the layout has placed the region it names.
//!
//! # A bound is a place, not a constant
//!
//! [`Bound`] carries a section index beside the address, and both come off the
//! one region so they cannot disagree. That is load-bearing rather than tidy:
//! `st_shndx` is what tells a loader whether to add the load base to
//! `st_value`, and glibc's `SYMBOL_ADDRESS` skips the base for an `SHN_ABS`
//! row. A bound published as absolute therefore keeps its link-time value in
//! an image the loader placed somewhere else -- `_end` in a shared object read
//! back as `0x10e8` rather than `base + 0x10e8`. lld defines every one of these
//! against an output section for the same reason
//! (`lld/ELF/Writer.cpp`, `setReservedSymbolSections`).
//!
//! A bound whose region is not in the image reports the image base, at both
//! ends. The runtime walks the half-open range between them, so an empty range
//! is what "no constructors" has to look like -- a start past its end would
//! send it off the end of the image -- and the base is the one address every
//! image has. lld anchors the same case to its `elfHeader` pseudo-section
//! (`lld/ELF/Writer.cpp`, "loaders expect equal `st_value`").
//!
//! # Visibility
//!
//! The bounds a C runtime uses to walk its own image are private to it, and
//! carry the visibility lld gives them: hidden for the array bounds and the
//! image bounds, protected for `__start_`/`__stop_` (lld's
//! `-z start-stop-visibility` default), and default for the three a program may
//! legitimately publish (`__bss_start`, `_edata`, `_end`). A hidden bound stays
//! out of `.dynsym`, which is what keeps a shared object from exporting the
//! placement of its own constructor array.
//!
//! The same two halves cover `__start_NAME`/`__stop_NAME`, the bounds of a
//! section whose name is a valid C identifier: the names are collected from
//! what the inputs reference, and the addresses come from the run
//! [`crate::startstop`] gathered. Those names are not in [`TABLE`] because
//! there is no fixed list of them -- the program picks them.

use rustc_hash::FxHashMap;

use crate::{
    dynamic::LinkMode,
    elf::constants::{STT_NOTYPE, STV_DEFAULT, STV_HIDDEN, STV_PROTECTED},
    error::Result,
    layout::{Layout, Sect},
    linker::Context,
    startstop::{START, STOP},
    symbol::SymbolId,
};

/// One linker-defined bound.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum DefSym {
    /// The image's load address, which is where its ELF header sits.
    ImageStart,
    /// First byte of `.preinit_array`, and one past its last.
    PreinitStart,
    PreinitEnd,
    /// First byte of `.init_array`, and one past its last.
    InitStart,
    InitEnd,
    /// First byte of `.fini_array`, and one past its last.
    FiniStart,
    FiniEnd,
    /// First byte of `.bss`.
    BssStart,
    /// One past the last byte of initialised data: the end of the last region
    /// the image backs with file bytes.
    DataEnd,
    /// One past the last byte of the image in memory: the break the runtime
    /// starts its heap from.
    ImageEnd,
    /// One past the last byte of program text: the end of the last executable
    /// region, which is `.plt` or `.fini` rather than `.text` when the image
    /// places either.
    TextEnd,
    /// The base every GOT-relative expression measures from, which is the
    /// start of `.got`. See the `_GLOBAL_OFFSET_TABLE_` row in [`TABLE`].
    GotBase,
    /// First `R_*_IRELATIVE` relocation, and one past the last. A static image
    /// has no loader, so its own startup walks this range and calls each
    /// indirect function's resolver before `main`.
    IrelStart,
    IrelEnd,
    /// First byte of the run of members named by one C identifier, and one
    /// past its last. The payload is the run's index in
    /// [`crate::startstop::StartStop`].
    RunStart(usize),
    RunEnd(usize),
}

/// Every name the linker supplies, with the bound it stands for and the
/// visibility it carries.
///
/// The underscore-prefixed spellings are what a C library refers to from its
/// own sources; the bare `edata` and `end` are the historical spellings that
/// a program may declare itself, and the system linker provides both.
///
/// `__bss_start`, `_edata`, `_end` and the `etext` pair are the ones a
/// program may publish, and lld leaves them at default visibility. The rest
/// describe where this linker put things, which is nothing another image may
/// bind to, so they are hidden as lld's are (`addOptionalRegular` defaults
/// `stOther` to `STV_HIDDEN`). `_GLOBAL_OFFSET_TABLE_` is here for the same
/// reason as the rest: an object may name it, and nothing else defines it.
/// Hand-written position-independent assembly reaches the GOT through it, and
/// so does any code built the way a 32-bit ABI does it.
///
/// Its address is the base `RelExpr::GotBase` and `RelExpr::GotOff` already
/// measure from, so the symbol and the expressions cannot disagree -- which is
/// the property that matters, and the one that was latently at risk while the
/// symbol did not exist. That base is `.got`, where lld's is `.got.plt`
/// (`lld/ELF/Arch/X86_64.cpp`). The difference is not observable
/// here: every GOT slot xold allocates lives in `.got`, so measuring from
/// `.got` is measuring from the table the slots are in, while lld's choice is
/// an x86-64 convention that matters to code indexing the lazy-binding table
/// directly. Settling on one base was the requirement; this settles on the one
/// the slots are in.
const TABLE: [(&[u8], DefSym, u8); 19] = [
    (b"_GLOBAL_OFFSET_TABLE_", DefSym::GotBase, STV_HIDDEN),
    (b"__executable_start", DefSym::ImageStart, STV_HIDDEN),
    (b"__ehdr_start", DefSym::ImageStart, STV_HIDDEN),
    // `__dso_handle` is what `__cxa_atexit` passes so the exit code can
    // tell which module a handler belongs to. Its value only has to differ
    // between modules, so lld anchors it at the ELF header, hidden
    // (`lld/ELF/Writer.cpp`); the image start is that anchor.
    (b"__dso_handle", DefSym::ImageStart, STV_HIDDEN),
    (b"__preinit_array_start", DefSym::PreinitStart, STV_HIDDEN),
    (b"__preinit_array_end", DefSym::PreinitEnd, STV_HIDDEN),
    (b"__init_array_start", DefSym::InitStart, STV_HIDDEN),
    (b"__init_array_end", DefSym::InitEnd, STV_HIDDEN),
    (b"__fini_array_start", DefSym::FiniStart, STV_HIDDEN),
    (b"__fini_array_end", DefSym::FiniEnd, STV_HIDDEN),
    (b"__bss_start", DefSym::BssStart, STV_DEFAULT),
    (b"_edata", DefSym::DataEnd, STV_DEFAULT),
    (b"edata", DefSym::DataEnd, STV_DEFAULT),
    (b"etext", DefSym::TextEnd, STV_DEFAULT),
    (b"_etext", DefSym::TextEnd, STV_DEFAULT),
    (b"__rela_iplt_start", DefSym::IrelStart, STV_HIDDEN),
    (b"__rela_iplt_end", DefSym::IrelEnd, STV_HIDDEN),
    (b"_end", DefSym::ImageEnd, STV_DEFAULT),
    (b"end", DefSym::ImageEnd, STV_DEFAULT),
];

/// Defines every listed name that an input referenced and none defined,
/// recording what each stands for so the layout can fill in its address.
///
/// Runs after archive extraction, so a member that defines one of these for
/// itself has already been pulled in and keeps the definition.
pub fn define(ctx: &mut Context<'_>) -> Result<()> {
    for (name, sym, visibility) in TABLE {
        let Some(id) = ctx.symbols.find(name) else {
            continue;
        };
        if !ctx.symbols.define_absolute(id, STT_NOTYPE, visibility) {
            continue;
        }
        ctx.defsyms.push((id, sym));
    }
    define_tls_module_base(ctx);
    define_runs(ctx);
    Ok(())
}

/// The name a TLS descriptor sequence uses to reach this module's own
/// thread-local block, which is the descriptor spelling of the local-dynamic
/// model.
///
/// The compiler emits `lea _TLS_MODULE_BASE_@tlsdesc(%rip), %rax` and reaches
/// each thread-local at a fixed offset from what that leaves in `%rax`. Nothing
/// defines the name: it is the linker's, and its value is the module base
/// measured from the thread pointer.
const TLS_MODULE_BASE: &[u8] = b"_TLS_MODULE_BASE_";

/// Defines [`TLS_MODULE_BASE`] as the constant zero when an input referenced
/// it.
///
/// It takes no entry in `defsyms` because it is not a bound: it names no region
/// and the layout has nothing to fill in for it. Zero is the whole definition,
/// and it is zero rather than the address of the thread-local block because of
/// what the sequence does with it. The offsets that follow the descriptor call
/// are `DTPOFF` relocations, which an executable resolves against the thread
/// pointer -- the same base the lowered local-dynamic sequence leaves behind --
/// so the base the descriptor supplies has to be the thread pointer too, which
/// is an offset of zero from itself. lld defines it the same way and for the
/// same reason (`getTlsTpOffset`, "on targets that support TLSDESC,
/// `_TLS_MODULE_BASE_@tpoff = 0`").
///
/// Hidden, like every other name here that describes this linker's own choices:
/// no other image may bind to it.
fn define_tls_module_base(ctx: &mut Context<'_>) {
    if let Some(id) = ctx.symbols.find(TLS_MODULE_BASE) {
        ctx.symbols.define_absolute(id, STT_NOTYPE, STV_HIDDEN);
    }
}

/// Collects the C-identifier section names the inputs ask to be bounded, then
/// defines `__start_NAME` and `__stop_NAME` for each on the same terms as the
/// fixed names above.
///
/// The collection has to happen here rather than in the section passes: the
/// names come from the symbol table, and this is the point at which no further
/// input can arrive to add one.
///
/// The pair is protected rather than hidden: a program may hand the run to
/// another image, so the names stay in `.dynsym`, but nothing outside may
/// replace this image's own view of where its run sits. That is lld's
/// `-z start-stop-visibility` default.
fn define_runs(ctx: &mut Context<'_>) {
    let Context {
        symbols,
        start_stop,
        defsyms,
        ..
    } = ctx;
    start_stop.collect(symbols);
    // One reused buffer rather than two allocations per run.
    let mut spelling: Vec<u8> = Vec::new();
    for (run, section) in start_stop.names().iter().enumerate() {
        for (prefix, bound) in
            [(START, DefSym::RunStart(run)), (STOP, DefSym::RunEnd(run))]
        {
            spelling.clear();
            spelling.extend_from_slice(prefix);
            spelling.extend_from_slice(section);
            let Some(id) = symbols.find(&spelling) else {
                continue;
            };
            if !symbols.define_absolute(id, STT_NOTYPE, STV_PROTECTED) {
                continue;
            }
            defsyms.push((id, bound));
        }
    }
}

/// Which end of a region a bound sits at.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
enum Edge {
    /// The region's first byte.
    Start,
    /// One past the region's last byte.
    End,
}

/// The place a linker-defined bound resolves to.
///
/// See the module docs for why the section index travels with the address
/// rather than being left `SHN_ABS`.
#[derive(Clone, Copy, Debug)]
pub struct Bound {
    /// The runtime address of the bound.
    pub addr: u64,
    /// The section header the address belongs to.
    pub shndx: u16,
}

/// Where one linker-defined symbol lands, once the layout has placed the
/// regions it bounds.
pub fn bound(layout: &Layout, sym: DefSym) -> Bound {
    match sym {
        // The ELF header sits at the load base, ahead of the first region, so
        // this is the one bound whose address is not a region's edge.
        DefSym::ImageStart => Bound {
            addr: layout.base(),
            shndx: header_shndx(layout),
        },
        DefSym::PreinitStart => edge(layout, Sect::PreinitArray, Edge::Start),
        DefSym::PreinitEnd => edge(layout, Sect::PreinitArray, Edge::End),
        DefSym::InitStart => edge(layout, Sect::InitArray, Edge::Start),
        DefSym::InitEnd => edge(layout, Sect::InitArray, Edge::End),
        DefSym::FiniStart => edge(layout, Sect::FiniArray, Edge::Start),
        DefSym::FiniEnd => edge(layout, Sect::FiniArray, Edge::End),
        DefSym::BssStart => bss_start(layout),
        DefSym::TextEnd => text_end(layout),
        DefSym::DataEnd => data_end(layout),
        DefSym::ImageEnd => image_end(layout),
        // Only a static image resolves its own indirect functions. A dynamic
        // image carries `IRELATIVE` entries in the same `.rela.plt`, but hands
        // the whole table to the loader through `DT_JMPREL`, which applies
        // them; a startup that walked this range as well would call every
        // resolver a second time. So the range is empty outside a static link,
        // as it is in lld, which does not define the pair at all in a
        // position-independent image.
        // Not `edge(Sect::Got, ..)`: an empty region reports the load base,
        // and the symbol has to be the value `RelExpr::GotBase` uses whether
        // or not this link allocated a slot. Reading it from the same place
        // the resolver does is what makes the two agree by construction.
        DefSym::GotBase => got_base(layout),
        DefSym::IrelStart => irel(layout, Edge::Start),
        DefSym::IrelEnd => irel(layout, Edge::End),
        DefSym::RunStart(run) => run_bound(layout, run, Edge::Start),
        DefSym::RunEnd(run) => run_bound(layout, run, Edge::End),
    }
}

/// The bound at one end of `sect`, or the empty bound when the region holds
/// nothing.
fn edge(layout: &Layout, sect: Sect, at: Edge) -> Bound {
    let region = layout.region(sect);
    if region.size == 0 {
        return empty(layout);
    }
    Bound {
        addr: match at {
            Edge::Start => region.vaddr,
            Edge::End => region.vaddr.wrapping_add(region.size),
        },
        shndx: layout.shndx(sect),
    }
}

/// `_GLOBAL_OFFSET_TABLE_`: the address every GOT-relative expression measures
/// from.
///
/// Read from [`Layout::got_base`] rather than from the region, because an
/// absent region reports the load base and the symbol has to be the value the
/// resolver uses whether or not this link allocated a slot. When there is no
/// `.got` the address is still meaningful -- `GOTPC32` computes from it, and a
/// program that reaches the table through the symbol simply finds it empty --
/// so the row is attributed to the section the address falls in rather than
/// left undefined.
fn got_base(layout: &Layout) -> Bound {
    let addr = layout.got_base();
    let shndx = if layout.region(Sect::Got).size == 0 {
        header_shndx(layout)
    } else {
        layout.shndx(Sect::Got)
    };
    Bound { addr, shndx }
}

/// The bound a pair reports when the region it names is not in the image: the
/// load base, which both ends answer so the range between them is empty.
fn empty(layout: &Layout) -> Bound {
    Bound {
        addr: layout.base(),
        shndx: header_shndx(layout),
    }
}

/// The section an address at the load base belongs to: the first one in the
/// image, which is the section the ELF header runs into.
fn header_shndx(layout: &Layout) -> u16 {
    layout.first_placed().map_or(0, |s| layout.shndx(s))
}

/// `__bss_start`: the first byte of `.bss`, and the end of initialised data
/// when the image has no zero-filled data to start.
fn bss_start(layout: &Layout) -> Bound {
    if layout.region(Sect::Bss).size == 0 {
        return data_end(layout);
    }
    edge(layout, Sect::Bss, Edge::Start)
}

/// `_edata`: one past the last byte the image backs with file bytes. This is
/// lld's "end of the last non-`SHT_NOBITS` mapped section"
/// (`lld/ELF/Writer.cpp`), which is not `.bss`'s start whenever
/// alignment padding separates the two.
fn data_end(layout: &Layout) -> Bound {
    layout
        .last_file_backed()
        .map_or_else(|| empty(layout), |s| edge(layout, s, Edge::End))
}

/// `etext`: one past the last byte of program text. The placement order puts
/// `.fini` and `.plt` after `.text`, so the text region's own end can leave
/// executable bytes above the bound; the end of the last executable region is
/// the end of program text whatever subset of the code regions is present.
/// GNU ld defines the pair after `.fini` and lld at the end of the
/// read-execute portion, which agree with this on the images each produces.
fn text_end(layout: &Layout) -> Bound {
    layout
        .last_executable()
        .map_or_else(|| empty(layout), |s| edge(layout, s, Edge::End))
}

/// `_end`: one past the last byte of the image in memory.
fn image_end(layout: &Layout) -> Bound {
    layout
        .last_placed()
        .map_or_else(|| empty(layout), |s| edge(layout, s, Edge::End))
}

/// The bounds of the `R_*_IRELATIVE` table a static image applies itself.
fn irel(layout: &Layout, at: Edge) -> Bound {
    if layout.mode() != LinkMode::Static {
        return empty(layout);
    }
    edge(layout, Sect::RelaPlt, at)
}

/// The bounds of one `__start_`/`__stop_` run.
///
/// Both ends name the section the run sits in, as lld's do: the stop address is
/// one past the run's last byte, which may fall in whatever follows it or past
/// the end of the image, so the start is what decides the section.
fn run_bound(layout: &Layout, run: usize, at: Edge) -> Bound {
    let bounds = layout.run_bounds(run);
    if bounds.start == bounds.stop {
        return empty(layout);
    }
    Bound {
        addr: match at {
            Edge::Start => bounds.start,
            Edge::End => bounds.stop,
        },
        shndx: layout.shndx_at(bounds.start),
    }
}

/// Fills in the address of every linker-defined symbol and reports the section
/// each one is anchored to, keyed by symbol id.
///
/// `addr` is the table address resolution built, indexed by [`SymbolId`]. The
/// section indices go to [`crate::layout::collect_exports`], which writes them
/// into the output symbol tables; both come from the one [`bound`] call, so the
/// address a reference resolves to and the section the tables publish cannot
/// describe different places.
pub fn resolve(
    ctx: &Context<'_>,
    layout: &Layout,
    addr: &mut [u64],
) -> FxHashMap<SymbolId, u16> {
    let mut shndx = FxHashMap::default();
    shndx.reserve(ctx.defsyms.len());
    for &(id, sym) in &ctx.defsyms {
        let place = bound(layout, sym);
        if let Some(slot) = addr.get_mut(id.0) {
            *slot = place.addr;
        }
        shndx.insert(id, place.shndx);
    }
    shndx
}
