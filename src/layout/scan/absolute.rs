//! The absolute references a dynamic executable treats specially.
//!
//! A GOT or PLT reference is position independent by construction, so the
//! scan filters both out before asking here. What is left is an absolute
//! reference, which an executable answers one of three ways: a data import
//! becomes a copy slot in `.bss`, a function import becomes a canonical PLT
//! entry, and a reference whose stored value no load base can shift forces the
//! image to a fixed base.
//!
//! The first two are last resorts. A slot the loader can be asked to fix up
//! takes neither: `.rela.dyn` names the import and the loader writes the
//! address in, which is both cheaper and what keeps the image relocatable.
//! Both leave the import at an address this link chose, so a slot the loader
//! will not fix up holds a link-time constant either way, and forces the fixed
//! base exactly as a reference to the image's own symbols does.

use super::{RelocSite, ScanWork, UndefFlags, cached_global_id, push_plt};
use crate::{
    elf::{
        Shdr64,
        constants::{SHF_TLS, STT_FUNC, STT_OBJECT, STT_TLS, STV_DEFAULT},
    },
    error::{Error, Result},
    layout::{CopyName, CopySlot, GotKey},
    linker::{Context, DepAddr},
    reloc::{RelExpr, Target, Write, WriteKind, spec_target},
    symbol::{SymbolId, SymbolKind, version_stem},
};

/// Classifies the absolute references a dynamic executable treats specially:
/// a data import it copy-relocates, a function import it reaches through a
/// canonical PLT entry, and a reference whose stored value is a link-time
/// address, which forces a fixed load base.
///
/// A GOT or PLT reference is none of these: a GOT access stays on `GLOB_DAT`
/// (the correct mechanism for position-independent access) and a call stays on
/// the PLT path, so the caller filters both out before asking.
///
/// The first two are what an executable falls back on when the slot cannot
/// carry a dynamic relocation, so [`carries_dyn_reloc`] rules them out first.
pub(super) fn classify_absolute(
    ctx: &Context<'_>,
    target: Target,
    site: &RelocSite<'_>,
    undef: &UndefFlags,
    r: &crate::elf::Rela64,
    work: &mut ScanWork<'_>,
) -> Result<()> {
    let sym_idx = r.sym();
    let id_opt = cached_global_id(site.sym_id_row, sym_idx);
    // A slot the loader can be handed leaves the address for it to store, and
    // needs neither mechanism; any other slot binds the import to storage this
    // link owns, and `bound` says whether one of them took it.
    let bound = if carries_dyn_reloc(target, site.writable, r.r_type()) {
        false
    } else {
        bind_shared_import(ctx, site, sym_idx, id_opt, work)?
    };
    // A non-PIC absolute reference (`R_X86_64_64`/`32`/`32S`/`16`) stores a
    // full virtual address. Under ASLR a PIE loads at a random base, so a
    // link-time absolute address is wrong; and a 32-bit slot (`32S`/`32`/`16`)
    // cannot hold a 64-bit load base, so the loader cannot relocate it at all.
    // The link therefore drops to a fixed base (`ET_EXEC`) and resolves the
    // reference at link time, matching `clang`/`clang -no-pie` for non-PIC
    // input.
    //
    // Two kinds of target make the stored address a link-time constant: a
    // symbol the executable itself defines (a local/section symbol or a
    // defined global), and an import the step above just bound to storage this
    // link chose -- a copy slot or a canonical PLT stub. An import still left
    // to the loader is neither: its slot is writable, so the loader stores the
    // address the moment it knows it, and [`crate::dynamic::rela`] refuses the
    // reference outright when it is not.
    if absolute_forces_exec(
        target,
        r.r_type(),
        site.writable,
        is_tls_ref(site.symtab, site.sections, sym_idx),
    ) && (bound || !undef.is_undefined(id_opt))
    {
        *work.has_abs = true;
    }
    Ok(())
}

/// Whether the loader can be handed this reference as a dynamic relocation,
/// which is what makes the two mechanisms in [`bind_shared_import`]
/// unnecessary.
///
/// This is lld's `canWrite` gate at the head of
/// `RelocationScanner::processAux`, which reaches the copy-relocation and
/// canonical-PLT block only when the slot cannot carry one.
///
/// xold has no `-z notext`, so `canWrite` is exactly `writable`. And lld's
/// `getDynRel` answers with a non-zero type only for the target's
/// pointer-width absolute relocation, so its `rel != 0` is exactly
/// `r_type == abs64`. That is also the one type
/// [`crate::dynamic::rela`] turns into a dynamic relocation, so the gate here
/// and the emitter agree on which references the loader is asked about.
///
/// A narrower or PC-relative reference falls through even in a writable
/// section, because no dynamic relocation can describe it: that is how
/// `crt1.o`'s `R_X86_64_PC32` to a libc function still reaches a canonical PLT
/// entry.
fn carries_dyn_reloc(target: Target, writable: bool, r_type: u32) -> bool {
    writable && r_type == target.dyn_relocs().abs64
}

/// Binds a reference to a shared dependency's symbol to storage this image
/// owns, which is what an executable falls back on when the slot cannot carry
/// a dynamic relocation: a copy slot in `.bss` for data, a canonical PLT entry
/// for a function.
///
/// Returns whether the reference now resolves to an address this link chose,
/// which the caller reads as "the value stored here is a link-time constant".
/// The answer is membership of the two sets rather than whether this call
/// added the entry: an earlier section of the same file may have bound the
/// very same import.
fn bind_shared_import(
    ctx: &Context<'_>,
    site: &RelocSite<'_>,
    sym_idx: u32,
    id_opt: Option<SymbolId>,
    work: &mut ScanWork<'_>,
) -> Result<bool> {
    let symtab = site.symtab;
    // A data symbol imported from a shared dependency is copy-relocated: the
    // executable then owns the storage (in `.bss`) and the loader copies the
    // initial bytes. Matching lld's `shouldCopyRel`, COPY applies to an
    // absolute reference to a shared `STT_OBJECT` in an executable.
    if let Some(id) = id_opt
        && !work.seen_copy.contains(&id)
        && let Some(slot) = copy_candidate(ctx, symtab, sym_idx, id)?
    {
        // Every name the slot defines is served by it, so a later reference
        // to an alias must not open a second one. Recording them all here is
        // also what keeps the alias walk off the per-relocation path.
        for name in &slot.names {
            if let Some(alias) = name.id {
                work.seen_copy.insert(alias);
            }
        }
        work.copy.push(slot);
    }
    // A reference to a shared-library function (for example the
    // `__gxx_personality_v0` pointer in a `.eh_frame` CIE) cannot store the
    // function's runtime address at link time and cannot emit a dynamic
    // relocation against a read-only section. Instead it allocates a canonical
    // PLT entry at a fixed address, resolves every reference to it, and pairs
    // it with a `JUMP_SLOT` the loader binds. Mirrors lld's `NEEDS_PLT` path
    // for an absolute reloc against a shared `STT_FUNC` in an executable.
    if let Some(id) = canonical_plt_candidate(ctx, symtab, sym_idx, id_opt)
        && work.seen_canonical.insert(id)
    {
        work.canonical_plt.push(id);
        // A canonical PLT entry needs the same PLT/GOT.PLT/JUMP_SLOT
        // plumbing as a call-driven import; reserve it here so the layout
        // sizes the stub before any address is known.
        push_plt(work, GotKey::addr(id));
    }
    Ok(id_opt.is_some_and(|id| {
        work.seen_copy.contains(&id) || work.seen_canonical.contains(&id)
    }))
}

/// Whether an absolute relocation at this site forces a fixed-base (`ET_EXEC`)
/// link under `DynExec`.
///
/// The classification comes from the relocation table rather than a per-target
/// type list: an absolute reference is [`RelExpr::Abs`], and the storage kind
/// says whether the loader could fix it up at load time. A pointer-width slot
/// (`Write::Bytes(W64)`) in a writable section becomes a `RELATIVE` dynamic
/// relocation the loader applies, so a constructor pointer in `.init_array`
/// stays position independent. The same slot in a read-only section cannot be
/// patched without `TEXTREL`, and a narrower slot cannot hold a 64-bit load
/// base anywhere; both force a fixed base.
///
/// An instruction immediate (`Write::Field`) forces one only when it encodes
/// address bits above the page offset. A load base is page aligned, so the low
/// 12 bits of an address survive it: that is exactly the `_LO12_` half of an
/// `AArch64` ADRP pair or a RISC-V AUIPC pair, whose other half is PC-relative.
/// A wider immediate (`R_AARCH64_MOVW_UABS_*`) does encode moving bits, and no
/// loader can patch bits inside an instruction, so it needs a fixed base.
///
/// `tls` excludes local-exec TLS, whose value is a thread-pointer-relative
/// offset rather than an address, and so is position independent. An
/// unclassifiable type never forces a fixed base: the apply pass reports it.
fn absolute_forces_exec(
    target: Target,
    r_type: u32,
    writable: bool,
    tls: bool,
) -> bool {
    if tls {
        return false;
    }
    let Ok(spec) = spec_target(target, r_type) else {
        return false;
    };
    if spec.expr != RelExpr::Abs {
        return false;
    }
    match spec.write {
        Write::Bytes(WriteKind::W64) => !writable,
        Write::Bytes(_) => true,
        Write::Field(f) => {
            u32::from(f.right_shift).saturating_add(u32::from(f.width))
                > PAGE_BITS
        }
    }
}

/// Bits of an address that a page-aligned load base leaves untouched.
const PAGE_BITS: u32 = 12;

/// Whether relocation symbol `sym_idx` names thread-local storage: either the
/// symbol is `STT_TLS` or it is a section symbol for an `SHF_TLS` section.
fn is_tls_ref(
    symtab: &crate::elf::SymbolTable<'_>,
    sections: &[Shdr64],
    sym_idx: u32,
) -> bool {
    symtab.syms.get(sym_idx as usize).is_some_and(|sym| {
        sym.type_() == STT_TLS
            || sections
                .get(usize::from(sym.st_shndx.get()))
                .is_some_and(|sh| sh.sh_flags.get() & SHF_TLS != 0)
    })
}

/// Whether relocation symbol `sym_idx` is a data symbol (`STT_OBJECT`) an
/// executable imports from a shared dependency, returning the copy-slot
/// descriptor (sized and aligned from the dependency's export) if so. Matches
/// lld's `shouldCopyRel` rule for an executable: a defined-in-shared-object
/// `STT_OBJECT` referenced by an absolute relocation. The caller has already
/// filtered to absolute (non-GOT, non-PLT) relocation kinds, and passes the
/// cached global `id` for `sym_idx`, so no name probe is needed.
///
/// The slot is sized and aligned from the export that named it, as lld's
/// `addCopyRelSymbol` does, and defines every alias of it besides.
fn copy_candidate(
    ctx: &Context<'_>,
    symtab: &crate::elf::SymbolTable<'_>,
    sym_idx: u32,
    id: SymbolId,
) -> Result<Option<CopySlot>> {
    let Some(sym) = symtab.syms.get(sym_idx as usize) else {
        return Ok(None);
    };
    // The reference may arrive in its versioned spelling (`bar@VER`); the
    // export it asks for is filed under the stem.
    let name = version_stem(symtab.name(sym));
    let Some(s) = ctx.symbols.symbol(id) else {
        return Ok(None);
    };
    if !matches!(s.kind, SymbolKind::Undefined { .. }) {
        return Ok(None);
    }
    let Some(exp) = ctx.dep_exports.get(name) else {
        return Ok(None);
    };
    // The dependency must have storage at `exp.at` for the loader to copy. An
    // `SHN_ABS` export has none: its value is a constant the dependency's own
    // references resolve against, and nothing sits there to copy from.
    // Only a default-visibility export may be taken over. A protected one is
    // the dependency's promise that its own references bind to its own
    // definition, so a copy would leave the two halves of the process reading
    // different storage for one name; a hidden one is not reachable from
    // outside the dependency at all.
    //
    // Neither is a case to skip quietly. Falling through leaves the reference
    // resolving to the address the dependency happened to be linked at, which
    // is not where it will be loaded, so the image is wrong with no
    // relocation to show for it. lld stops with "cannot preempt symbol", in
    // `canDefineSymbolInExecutable`, and so does this.
    if exp.sym_type == STT_OBJECT
        && exp.storage
        && exp.visibility != STV_DEFAULT
    {
        return Err(Error::UnpreemptableSymbol(
            String::from_utf8_lossy(name).into_owned(),
        ));
    }
    // A copy slot needs a length. An export with storage but no stated size
    // cannot be skipped quietly either: the reference would resolve to the
    // address the dependency was linked at, wrong wherever it is loaded.
    // lld stops in `addCopyRelSymbol` ("cannot create a copy relocation for
    // symbol"), and so does this.
    if exp.sym_type == STT_OBJECT && exp.storage && exp.size == 0 {
        return Err(Error::CopyZeroSized(
            String::from_utf8_lossy(name).into_owned(),
        ));
    }
    if exp.sym_type != STT_OBJECT || !exp.placed {
        return Ok(None);
    }
    let mut names = Vec::new();
    match ctx.dep_exports.aliases_at(exp.at) {
        Some(group) => copy_names(ctx, exp.at, group, &mut names),
        None => names.push(CopyName {
            name: name.to_vec(),
            size: exp.size,
            id: Some(id),
        }),
    }
    // The reference that opened the slot resolves to it: it named an export
    // the dependency places at `exp.at`, and it is undefined in this link, so
    // the walk above cannot have dropped it.
    debug_assert!(names.iter().any(|n| n.id == Some(id)));
    Ok(Some(CopySlot {
        id,
        size: exp.size,
        align: exp.align,
        addr: 0,
        origin: exp.at,
        names,
    }))
}

/// Collects into `out` every name of the object at `at`, in the dependency's
/// symbol table order, paired with the symbol this link resolved it to.
///
/// A dependency may give one object several names, and all of them have to
/// land on the one copy slot: with a slot each, the program would hold two
/// copies of an object the dependency has one of, so a write through one name
/// would be invisible through the other -- and glibc's own `setenv` writes
/// through `__environ`, which a program's separate `environ` copy would never
/// see. This is lld's `getSymbolsAt`.
///
/// `group` is what the dependency declares, so two kinds of name are dropped:
/// one an earlier dependency exported first, and one an input defines itself.
/// Neither reaches this object, so neither may be defined at its slot.
///
/// Like lld, a name no input references is kept: it costs a `.dynsym` entry
/// and nothing else, and it is how another image binds that spelling to the
/// copy. It pulls in no input of its own.
fn copy_names(
    ctx: &Context<'_>,
    at: DepAddr,
    group: &[&[u8]],
    out: &mut Vec<CopyName>,
) {
    for &name in group {
        let Some(exp) =
            ctx.dep_exports.get(name).filter(|e| e.placed && e.at == at)
        else {
            continue;
        };
        let id = ctx.symbols.find(name);
        if id.is_some_and(|i| !is_import(ctx, i)) {
            continue;
        }
        out.push(CopyName {
            name: name.to_vec(),
            size: exp.size,
            id,
        });
    }
}

/// Whether a resolved symbol is still an undefined reference, and so free to
/// be bound to a copy slot.
fn is_import(ctx: &Context<'_>, id: SymbolId) -> bool {
    ctx.symbols
        .symbol(id)
        .is_some_and(|s| matches!(s.kind, SymbolKind::Undefined { .. }))
}

/// Whether relocation symbol `sym_idx` is a function symbol (`STT_FUNC`) an
/// executable imports from a shared dependency, returning its global id if so.
/// Mirrors lld's canonical-PLT rule for an executable: an absolute reference
/// to a shared `STT_FUNC` allocates a PLT entry whose address stands in for
/// the function until the loader binds it. The caller has already filtered to
/// absolute (non-GOT, non-PLT) relocation kinds.
fn canonical_plt_candidate(
    ctx: &Context<'_>,
    symtab: &crate::elf::SymbolTable<'_>,
    sym_idx: u32,
    id_opt: Option<SymbolId>,
) -> Option<SymbolId> {
    let id = id_opt?;
    let sym = symtab.syms.get(sym_idx as usize)?;
    let name = version_stem(symtab.name(sym));
    let s = ctx.symbols.symbol(id)?;
    if !matches!(s.kind, SymbolKind::Undefined { .. }) {
        return None;
    }
    let exp = ctx.dep_exports.get(name)?;
    if exp.sym_type != STT_FUNC {
        return None;
    }
    Some(id)
}
