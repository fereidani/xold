//! The x86-64 relocation table: each raw `R_X86_64_*` type mapped to a
//! [`Spec`].
//!
//! This is the declarative table consumed by the arch-neutral drivers in the
//! parent module. Mapping a type once, here, is what lets scan and apply share
//! a single implementation. The set covers the relocations ordinary C code
//! emits. Local-exec TLS (`TPOFF32`, `TPOFF64`) is reduced to
//! [`RelExpr::Abs`]: the resolver returns the thread-pointer-relative offset
//! for a TLS symbol, so `S + A` is exactly the value the instruction stores.
//! Initial-exec TLS (`GOTTPOFF`) is [`RelExpr::TlsGotPc`], a PC-relative
//! reference to the GOT slot holding that offset. The local-dynamic pair
//! (`TLSLD`, `DTPOFF32`) is reduced too -- [`RelExpr::TlsIndexPc`] for the
//! reference to this module's own entry and [`RelExpr::DtpOff`] for the
//! per-variable offsets that follow it -- so a shared object can keep the
//! model. `DTPMOD64` is the one type still routed through
//! [`crate::reloc::RelExpr::Escape`] and rejected: the module id belongs to
//! the loader, and an input object has no business naming it.
//!
//! `TLSGD` is handled by relaxation rather than by a runtime `__tls_get_addr`
//! call, because in an executable there is no such helper to reach: it
//! rewrites the general-dynamic pair into the local-exec form when it places
//! the thread-local itself, and into the initial-exec form when a shared
//! object places it (see [`relax_tls_gd`]). Compilers emit that pair for every
//! `_Thread_local` unless told otherwise, so without this a plain C program
//! with a thread-local would not link. A shared object keeps both dynamic
//! models instead: its sequences call the runtime helper, which reads the
//! module and offset pair the reference names, so nothing is rewritten and
//! nothing is rejected there. See [`X86_64::relax_required`], which is the one
//! place that difference is stated.
//!
//! A lowered pair takes its closing `call __tls_get_addr` with it, and the
//! replacement ends with a 4-byte field exactly where that call's
//! displacement was, so the relocation on the displacement must not be
//! applied. That relocation is an ordinary `PLT32` on an ordinary
//! `call rel32`: nothing about its own type, symbol or bytes tells it apart
//! from any other call in the image. It is identified by the pair beside it
//! instead -- [`X86_64::relax_span`] reports the sixteen bytes the lowering
//! consumes, and the writer drops every relocation whose slot lands inside
//! them. [`relax_tls_ld`] reports its twelve the same way.
//!
//! A TLS descriptor (`-mtls-dialect=gnu2`, which is what Fedora's clang emits
//! by default) is the same story told with two instructions instead of a pair:
//! `lea x@tlsdesc(%rip), %rax` names a descriptor the loader fills in, and
//! `call *x@tlscall(%rax)` jumps to the resolver function it holds, leaving the
//! offset from the thread pointer in `%rax`. An executable resolves neither at
//! run time -- there is no descriptor and no resolver in an image xold writes
//! -- so both instructions are lowered, exactly as the general-dynamic pair is:
//! the `lea` becomes `mov $x@tpoff, %rax` when this image places the
//! thread-local and `mov x@gottpoff(%rip), %rax` when a shared object does, and
//! the `call` becomes a two-byte `nop` either way (see [`relax_tls_desc`] and
//! [`relax_tls_desc_call`], after lld's `relaxTlsDescToLe` and
//! `relaxTlsDescToIe`). Unlike the general-dynamic pair the two rewrites are
//! independent -- each instruction carries its own relocation and rewrites its
//! own bytes -- so nothing is swallowed and [`X86_64::relax_span`] has nothing
//! to say about them. A shared object cannot keep the model, because filling a
//! descriptor takes an `R_X86_64_TLSDESC` dynamic relocation this linker does
//! not emit; the scan refuses it there and names the thread-local.
//!
//! `GOTPCRELX`/`REX_GOTPCRELX` relaxation rewrites a GOT-indirect access of a
//! locally defined symbol into a direct PC-relative form, matching lld's
//! `relaxGotPcRelX`:
//!
//! - `mov sym@GOTPCREL(%rip), %reg` (`8b ..`) -> `lea sym(%rip), %reg` (`8d
//!   ..`): a one-byte opcode change, displacement `S + A - P`.
//! - `call/jmp *sym@GOTPCREL(%rip)` (`ff 15`/`ff 25`) -> a direct `call`/`jmp`
//!   to the symbol.
//!
//! The site must reference a symbol defined in the output (resolved to a
//! non-zero address) and the displacement must fit a signed 32-bit field, or
//! the site falls back to the GOT load unchanged.

use std::ops::Range;

use super::bytes_of as of;
use crate::{
    error::Result,
    reloc::{Arch, Needs, RelExpr, RelaxSite, Resolver, Spec, WriteKind},
    symbol::SymbolId,
};

// --- raw relocation type numbers (x86-64 psABI) --------------------------

/// No relocation; the entry is skipped.
pub const R_X86_64_NONE: u32 = 0;
/// Direct 64-bit: `S + A`.
pub const R_X86_64_64: u32 = 1;
/// PC-relative 32-bit: `S + A - P`.
pub const R_X86_64_PC32: u32 = 2;
/// PC-relative PLT 32-bit: `PLT[sym] + A - P`.
pub const R_X86_64_PLT32: u32 = 4;
/// 32-bit GOT slot offset: `G + A`, from the table's base.
pub const R_X86_64_GOT32: u32 = 3;
/// PC-relative GOT 32-bit: `GOT[sym] + A - P`.
pub const R_X86_64_GOTPCREL: u32 = 9;
/// Direct 32-bit zero-extended: `S + A`.
pub const R_X86_64_32: u32 = 10;
/// Direct 32-bit sign-extended: `S + A`.
pub const R_X86_64_32S: u32 = 11;
/// Direct 16-bit: `S + A`.
pub const R_X86_64_16: u32 = 12;
/// Direct 16-bit PC-relative: `S + A - P`.
pub const R_X86_64_PC16: u32 = 13;
/// Direct 8-bit: `S + A`.
pub const R_X86_64_8: u32 = 14;
/// Direct 8-bit PC-relative: `S + A - P`.
pub const R_X86_64_PC8: u32 = 15;
/// PC-relative 64-bit: `S + A - P`.
pub const R_X86_64_PC64: u32 = 24;
/// 64-bit GOT slot offset: `G + A`, the slot's offset from the GOT base.
///
/// The table's *offset*, not its address: the large code model reaches the
/// slot by adding this to a register already holding
/// `_GLOBAL_OFFSET_TABLE_`. lld folds both widths into one expression
/// (`R_GOTPLT`, `GOT[sym] + A - GOT`, `lld/ELF/Arch/X86_64.cpp`);
/// the base it subtracts is `.got.plt`, where xold's is the `.got` its own
/// GOT symbol names -- each measures from the table its code was told to
/// add.
pub const R_X86_64_GOT64: u32 = 27;
/// 64-bit PC-relative GOT slot address: `GOT[sym] + A - P`.
pub const R_X86_64_GOTPCREL64: u32 = 28;
/// 64-bit PC-relative GOT base: `GOT + A - P`.
pub const R_X86_64_GOTPC64: u32 = 29;
/// 64-bit GOT-relative PLT entry address: `PLT[sym] + A - GOT`.
pub const R_X86_64_PLTOFF64: u32 = 31;
/// 32-bit size of the symbol: `Z + A`.
pub const R_X86_64_SIZE32: u32 = 32;
/// 64-bit size of the symbol: `Z + A`.
pub const R_X86_64_SIZE64: u32 = 33;
/// APX relaxable PC-relative GOT 32-bit, four bytes of prefix.
pub const R_X86_64_CODE_4_GOTPCRELX: u32 = 43;
/// APX initial-exec GOT reference, four bytes of prefix.
pub const R_X86_64_CODE_4_GOTTPOFF: u32 = 44;
/// APX TLS descriptor reference, four bytes of prefix.
pub const R_X86_64_CODE_4_GOTPC32_TLSDESC: u32 = 45;
/// APX initial-exec GOT reference, six bytes of prefix.
pub const R_X86_64_CODE_6_GOTTPOFF: u32 = 46;
/// APX TLS descriptor reference, six bytes of prefix.
pub const R_X86_64_CODE_6_GOTPC32_TLSDESC: u32 = 47;
/// GOT-relative 64-bit: `S + A - GOT`.
pub const R_X86_64_GOTOFF64: u32 = 25;
/// PC-relative GOT base 32-bit: `GOT + A - P`.
pub const R_X86_64_GOTPC32: u32 = 26;
/// Relaxable PC-relative GOT 32-bit (same value as `R_X86_64_GOTPCREL`).
pub const R_X86_64_GOTPCRELX: u32 = 41;
/// REX relaxable PC-relative GOT 32-bit.
pub const R_X86_64_REX_GOTPCRELX: u32 = 42;

// --- TLS placeholders (recognised but not reduced; routed to Escape) ------

/// 64-bit ID of the module containing the thread-local symbol.
pub const R_X86_64_DTPMOD64: u32 = 16;
/// 64-bit offset of the thread-local symbol in its TLS block.
pub const R_X86_64_DTPOFF64: u32 = 17;
/// 64-bit offset of the thread-local symbol relative to the thread pointer.
pub const R_X86_64_TPOFF64: u32 = 18;
/// General-dynamic TLS 32-bit.
pub const R_X86_64_TLSGD: u32 = 19;
/// Local-dynamic TLS 32-bit.
pub const R_X86_64_TLSLD: u32 = 20;
/// 32-bit offset in a local-dynamic TLS block.
pub const R_X86_64_DTPOFF32: u32 = 21;
/// GOT offset for initial-exec TLS.
pub const R_X86_64_GOTTPOFF: u32 = 22;
/// 32-bit offset for initial-exec TLS.
pub const R_X86_64_TPOFF32: u32 = 23;
/// PC-relative reference to a thread-local's TLS descriptor.
pub const R_X86_64_GOTPC32_TLSDESC: u32 = 34;
/// Marks the indirect call through a TLS descriptor's resolver function.
pub const R_X86_64_TLSDESC_CALL: u32 = 35;
/// The descriptor pair itself, which only a loader ever fills in.
pub const R_X86_64_TLSDESC: u32 = 36;

// --- dynamic relocation types (emitted into `.rela.dyn`/`.rela.plt`) ------

/// Loader computes `B + A` (load base plus addend); used for internal
/// pointer slots so they follow the shared object at runtime.
pub const R_X86_64_RELATIVE: u32 = 8;
/// Loader fills a GOT slot with the resolved address of a dynamic symbol.
pub const R_X86_64_GLOB_DAT: u32 = 6;
/// Lazy-binding PLT entry; the loader patches the PLT slot on first call.
pub const R_X86_64_JUMP_SLOT: u32 = 7;
/// Loader copies a defined data symbol's bytes from a shared object into a
/// slot the executable reserved in `.bss`.
///
/// The symbol then resolves to that copy. Emitted only for executables that
/// import a data symbol.
pub const R_X86_64_COPY: u32 = 5;
/// Loader resolves an `STT_GNU_IFUNC` indirect function through its resolver.
pub const R_X86_64_IRELATIVE: u32 = 37;

/// The x86-64 architecture, for use as the `A` type parameter of the drivers.
pub struct X86_64;

impl Arch for X86_64 {
    fn spec(r_type: u32) -> Result<Spec> {
        Ok(match r_type {
            R_X86_64_NONE => of(RelExpr::None, WriteKind::W64),
            R_X86_64_64 | R_X86_64_TPOFF64 => of(RelExpr::Abs, WriteKind::W64),
            R_X86_64_PC32 => of(RelExpr::Pc, WriteKind::W32S),
            R_X86_64_PLT32 => of(RelExpr::PltPc, WriteKind::W32S),
            R_X86_64_GOTPCREL | R_X86_64_GOTPCRELX | R_X86_64_REX_GOTPCRELX => {
                of(RelExpr::GotPc, WriteKind::W32S)
            }
            R_X86_64_32 => of(RelExpr::Abs, WriteKind::W32),
            // `32S` and local-exec `TPOFF32` share the spec: both store
            // `S + A` as a signed 32-bit value. For `TPOFF32` the resolver
            // supplies the thread-pointer-relative offset as `S`.
            R_X86_64_32S | R_X86_64_TPOFF32 => {
                of(RelExpr::Abs, WriteKind::W32S)
            }
            // Signed or unsigned: a 16-bit absolute slot legitimately holds
            // a small negative constant, and `0xffff` is how it is spelled.
            // Accepting only the unsigned range refused a store lld takes
            // (`checkIntUInt`, `[-32768, 65535]`). mold agrees with the
            // narrower reading, so this follows the reference this project
            // measures itself against rather than splitting the difference --
            // and it only ever accepts more.
            R_X86_64_16 => of(RelExpr::Abs, WriteKind::W16SU),
            // The narrow widths. lld classifies and writes all of them; here
            // they were falling to `UnsupportedReloc`, so an object with a
            // byte-sized absolute constant refused to link at all.
            R_X86_64_8 => of(RelExpr::Abs, WriteKind::W8),
            R_X86_64_PC8 => of(RelExpr::Pc, WriteKind::W8),
            R_X86_64_PC16 => of(RelExpr::Pc, WriteKind::W16SU),
            // `sizeof` an object whose definition the compiler could not see:
            // the value is the symbol's `st_size`, not its address.
            R_X86_64_SIZE32 => of(RelExpr::Size, WriteKind::W32),
            R_X86_64_SIZE64 => of(RelExpr::Size, WriteKind::W64),
            // The large code model (`-mcmodel=large -fPIC`). Every one of
            // these is an existing expression at 64-bit width: the GOT slot's
            // offset from the base, the slot's own address, the base itself,
            // and a PLT entry measured from the base.
            // The APX forms differ from `GOTPCRELX` only in how many prefix
            // bytes precede the displacement, which matters to relaxation and
            // not to the value. They are classified here and left unrelaxed:
            // `relax_lead` reports no window for them, so the writer computes
            // the ordinary GOT-indirect value, which is always correct if
            // sometimes larger than it needs to be.
            R_X86_64_CODE_4_GOTPCRELX => of(RelExpr::GotPc, WriteKind::W32S),
            R_X86_64_CODE_4_GOTTPOFF | R_X86_64_CODE_6_GOTTPOFF => {
                of(RelExpr::TlsGotPc, WriteKind::W32S)
            }
            R_X86_64_CODE_4_GOTPC32_TLSDESC
            | R_X86_64_CODE_6_GOTPC32_TLSDESC => {
                of(RelExpr::Escape, WriteKind::W32S)
            }
            R_X86_64_GOT32 => of(RelExpr::GotOffset, WriteKind::W32),
            R_X86_64_GOT64 => of(RelExpr::GotOffset, WriteKind::W64),
            R_X86_64_GOTPCREL64 => of(RelExpr::GotPc, WriteKind::W64),
            R_X86_64_GOTPC64 => of(RelExpr::GotBase, WriteKind::W64),
            // `PLT[sym] - GOT`, which no existing expression spells: `Plt`
            // is the entry's address and `GotOff` measures the *symbol* from
            // the base. Left unimplemented rather than approximated -- a
            // wrong large-model call target is worse than a refusal, and no
            // compiler emits this without `-mcmodel=large`, which the rest of
            // this row set now covers.

            R_X86_64_PC64 => of(RelExpr::Pc, WriteKind::W64),
            R_X86_64_GOTOFF64 => of(RelExpr::GotOff, WriteKind::W64),
            R_X86_64_GOTPC32 => of(RelExpr::GotBase, WriteKind::W32S),
            // General dynamic: the slot is the 4-byte displacement of the
            // `lea` that opens the pair. Relaxation rewrites the whole pair
            // before the value is ever computed; the escape path rejects it
            // when relaxation did not apply.
            // General-dynamic: the reference names a pair of GOT slots that
            // `__tls_get_addr` reads, and shares its spec with initial-exec
            // below because both are a PC-relative reference to the symbol's
            // TLS entry. An executable never computes either -- the pair is
            // rewritten by `relax` first -- so this is what a shared object
            // stores.
            R_X86_64_TLSGD
            // Initial-exec: the offset from the thread pointer is loaded from
            // a GOT slot rather than folded into the instruction.
            | R_X86_64_GOTTPOFF
            // A descriptor reference an executable lowers to the initial-exec
            // form stores exactly this: a PC-relative reference to the slot
            // holding the offset. The local-exec form stores no GOT reference
            // at all, and `relax` writes that one itself.
            | R_X86_64_GOTPC32_TLSDESC => of(RelExpr::TlsGotPc, WriteKind::W32S),
            // The call a descriptor sequence closes with has no field of its
            // own: the two bytes the relocation names are the instruction, and
            // lowering replaces them with a `nop`. It reaches the escape path
            // only when that lowering did not run, which is not a site any
            // value computation can rescue.
            R_X86_64_TLSDESC_CALL => of(RelExpr::Escape, WriteKind::W16),
            // The descriptor pair is written by the loader from a dynamic
            // relocation, so no input object has cause to carry this type --
            // the same standing as the module id: both are the loader's to
            // supply, and an input object has no business naming either.
            R_X86_64_TLSDESC | R_X86_64_DTPMOD64 => {
                of(RelExpr::Escape, WriteKind::W64)
            }
            // Local-dynamic: one reference to the module's own GOT entry,
            // then a fixed offset per thread-local within that module.
            R_X86_64_TLSLD => of(RelExpr::TlsIndexPc, WriteKind::W32S),
            R_X86_64_DTPOFF32 => of(RelExpr::DtpOff, WriteKind::W32S),
            R_X86_64_DTPOFF64 => of(RelExpr::DtpOff, WriteKind::W64),
            other => return Err(crate::error::Error::UnsupportedReloc(other)),
        })
    }

    /// What a dynamic TLS reference allocates depends on the image being
    /// produced, not on the type: a shared object needs a module entry, an
    /// executable needs a single offset slot or nothing at all, because the
    /// sequence is rewritten. The scan makes that call, so the table asks for
    /// nothing here.
    #[inline]
    fn scan_needs(r_type: u32) -> Result<Needs> {
        if matches!(
            r_type,
            R_X86_64_TLSGD
                | R_X86_64_TLSLD
                | R_X86_64_GOTPC32_TLSDESC
                | R_X86_64_TLSDESC_CALL
        ) {
            return Ok(Needs::NONE);
        }
        Ok(Self::spec(r_type)?.expr.needs())
    }

    /// `GOTPCRELX`/`REX_GOTPCRELX` relaxation needs the two opcode/ModR/M bytes
    /// immediately preceding the 4-byte displacement slot; `TLSGD` needs the
    /// four that open its `lea` and `TLSLD` the three that open its own. Every
    /// other type is not relaxable, which includes `PLT32`: an ordinary call
    /// is never rewritten in place, so no call site pays for a window.
    fn relax_lead(r_type: u32) -> usize {
        match r_type {
            R_X86_64_GOTPCRELX | R_X86_64_REX_GOTPCRELX => 2,
            R_X86_64_TLSGD => TLS_GD_LEAD.len(),
            R_X86_64_TLSLD => TLS_LD_LEAD.len(),
            R_X86_64_GOTPC32_TLSDESC => TLS_DESC_LEAD,
            _ => 0,
        }
    }

    /// The `call` a descriptor sequence closes with is rewritten where it
    /// stands: the two bytes the relocation names are the whole instruction, so
    /// it declares neither a lead nor a trail and has to say separately that it
    /// is rewritten at all.
    fn relax_covers(r_type: u32) -> bool {
        r_type == R_X86_64_TLSDESC_CALL || Self::relax_lead(r_type) != 0
    }

    /// A `GOTPCRELX` site the writer will certainly rewrite reads no GOT
    /// slot, so the scan need not allocate one.
    ///
    /// This asks the half of [`relax_gotpcrelx`] that depends on the input
    /// alone: the canonical `-4` addend, and an instruction form the rewrite
    /// covers. The caller answers the rest -- preemptibility and whether the
    /// symbol's value is an address -- because those are facts about the
    /// symbol, not about these bytes.
    ///
    /// The one thing neither can settle at scan time is whether the relaxed
    /// displacement fits in 32 bits, which needs addresses the layout has not
    /// assigned. An image where it does not is one whose ordinary PC-relative
    /// references do not fit either, so it was never linkable; the writer
    /// ends it with a diagnostic rather than filling a slot nothing reserved.
    fn relax_drops_got(r_type: u32, addend: i64, lead: &[u8]) -> bool {
        if !matches!(r_type, R_X86_64_GOTPCRELX | R_X86_64_REX_GOTPCRELX) {
            return false;
        }
        if addend != -4 {
            return false;
        }
        let (Some(&op), Some(&modrm)) = (lead.first(), lead.get(1)) else {
            return false;
        };
        // `mov` becomes `lea`; an indirect `call`/`jmp` becomes a direct one.
        // Every other form keeps its GOT access, and so keeps its slot.
        op == 0x8b || (op == 0xff && (modrm == 0x15 || modrm == 0x25))
    }

    /// In an executable a dynamic TLS sequence has no fallback: there is no
    /// runtime `__tls_get_addr` to reach, so lowering `TLSGD` or `TLSLD` is the
    /// only way the site can be resolved at all, and it runs whether or not
    /// relaxation was requested. The writer holds this to its word -- a
    /// mandatory rewrite that declines ends the link rather than falling
    /// through to a value computation -- so the answer must be false wherever
    /// the fallback is real.
    ///
    /// A shared object is exactly that case. It keeps the general-dynamic and
    /// local-dynamic models: the sequence calls the runtime helper, which reads
    /// the module and offset pair the reference names, so the ordinary value
    /// computation is the right answer and neither lowering applies. Both
    /// predicates say so too ([`tls_gd_lowering`] and [`is_tls_ld_sequence`]
    /// open by declining anything that is not an executable), and this arm is
    /// gated on the same question so the three cannot disagree.
    ///
    /// The `GOTPCRELX` arm covers a reference to `__tls_get_addr` that the
    /// scan dropped without allocating a slot for. Inside a pair this image
    /// lowers, [`Self::relax_span`] already claims those bytes and the writer
    /// never reaches the site; the arm is what catches one that stands
    /// outside such a pair, where a GOT displacement would be measured from
    /// address zero. It is keyed on the symbol rather than on the bytes,
    /// which is why the resolver is in scope.
    fn relax_required<R: Resolver>(
        r_type: u32,
        sym: Option<SymbolId>,
        resolver: &R,
    ) -> bool {
        match r_type {
            R_X86_64_TLSGD
            | R_X86_64_TLSLD
            // A descriptor is filled in by the loader from a dynamic
            // relocation this linker does not emit, so an executable has no
            // descriptor to reach and no resolver to call: both halves of the
            // sequence must be lowered or the link is wrong.
            | R_X86_64_GOTPC32_TLSDESC
            | R_X86_64_TLSDESC_CALL => resolver.is_exec(),
            R_X86_64_GOTPCRELX | R_X86_64_REX_GOTPCRELX => {
                sym.is_some_and(|id| !resolver.has_got_slot(id))
            }
            _ => false,
        }
    }

    /// A dynamic TLS sequence is rewritten as a unit, so it reports the whole
    /// sequence: sixteen bytes for a general-dynamic pair, twelve for a
    /// local-dynamic one. Both are measured from the `lea` that opens them,
    /// which sits [`Self::relax_lead`] bytes before the slot.
    ///
    /// The range is reported only when [`relax_tls_gd`] or [`relax_tls_ld`]
    /// will accept the same window, because the decision is the same call:
    /// [`tls_gd_lowering`] and [`is_tls_ld_sequence`] are the predicates both
    /// paths run.
    /// The two TLS sequence openers. `TLSGD` opens a general-dynamic pair and
    /// `TLSLD` a local-dynamic one; both close on an instruction whose own
    /// relocation the lowering swallows.
    fn spans_type(r_type: u32) -> bool {
        matches!(r_type, R_X86_64_TLSGD | R_X86_64_TLSLD)
    }

    fn relax_span<R: Resolver>(
        site: RelaxSite<'_>,
        resolver: &R,
    ) -> Option<Range<u64>> {
        if !Self::spans_type(site.r_type) {
            return None;
        }
        // The same window the writer hands to `relax`: the opcode bytes that
        // open the sequence, its own slot, and the instruction that closes it.
        // The trail is read off the same bytes the writer reads it off, so the
        // range reported here and the window rewritten there are one thing.
        let lead = Self::relax_lead(site.r_type);
        let width = Self::spec(site.r_type).ok()?.write.width();
        let slot_end = usize::try_from(site.slot).ok()?.checked_add(width)?;
        let after = site.data.get(slot_end..).unwrap_or(&[]);
        let span = lead.checked_add(width)?.checked_add(Self::relax_trail(
            site.r_type,
            &[],
            after,
        ))?;
        let start = site.slot.checked_sub(u64::try_from(lead).ok()?)?;
        let end = start.checked_add(u64::try_from(span).ok()?)?;
        let window = site
            .data
            .get(usize::try_from(start).ok()?..usize::try_from(end).ok()?)?;
        let lowered = if site.r_type == R_X86_64_TLSGD {
            tls_gd_lowering(site.sym, site.addend, site.place, resolver, window)
                .is_some()
        } else {
            is_tls_ld_sequence(resolver, window)
        };
        lowered.then_some(start..end)
    }

    /// A `TLSGD` rewrite continues past its own slot: the `call` that closes
    /// the general-dynamic pair is replaced along with it. So does a `TLSLD`
    /// one, and there the call has two encodings of different lengths -- a
    /// direct `call` under the default ABI and a `call *...@GOTPCREL(%rip)`
    /// under `-fno-plt`, one byte wider -- so `after` decides which.
    ///
    /// Only the indirect opcode buys the extra byte. Anything else, including
    /// a slice too short to hold the opcode, takes the direct form's trail: a
    /// window is only ever claimed on bytes that are there, and
    /// [`is_tls_ld_sequence`] rejects a shape that then fails to match.
    ///
    /// The general-dynamic pair needs no such test, because both of its call
    /// encodings are four bytes wide.
    fn relax_trail(r_type: u32, _slot: &[u8], after: &[u8]) -> usize {
        match r_type {
            R_X86_64_TLSGD => TLS_GD_CALL.len() + DISP32,
            R_X86_64_TLSLD => {
                if after.get(..2) == Some(&TLS_LD_CALL_INDIRECT[..]) {
                    TLS_LD_CALL_INDIRECT.len() + DISP32
                } else {
                    TLS_LD_CALL.len() + DISP32
                }
            }
            _ => 0,
        }
    }

    fn relax<R: Resolver>(
        r_type: u32,
        sym: Option<SymbolId>,
        addend: i64,
        place: u64,
        resolver: &R,
        window: &mut [u8],
    ) -> Result<bool> {
        match r_type {
            R_X86_64_TLSGD => {
                Ok(relax_tls_gd(sym, addend, place, resolver, window))
            }
            R_X86_64_TLSLD => Ok(relax_tls_ld(resolver, window)),
            R_X86_64_GOTPC32_TLSDESC => {
                Ok(relax_tls_desc(sym, addend, place, resolver, window))
            }
            R_X86_64_TLSDESC_CALL => Ok(relax_tls_desc_call(resolver, window)),
            _ => Ok(relax_gotpcrelx(
                r_type, sym, addend, place, resolver, window,
            )),
        }
    }
}

/// The 4-byte displacement slot width shared by every relaxable form.
const DISP32: usize = 4;

/// The width of a general-dynamic TLS pair: `data16 lea sym@tlsgd(%rip),
/// %rdi` then `data16 data16 rex.W call __tls_get_addr`, and of either form
/// it is rewritten into.
const TLS_GD_PAIR: usize = 16;

/// The bytes that open a general-dynamic TLS pair, immediately before the
/// `TLSGD` slot: `data16 lea sym@tlsgd(%rip), %rdi`.
const TLS_GD_LEAD: [u8; 4] = [0x66, 0x48, 0x8d, 0x3d];

/// The bytes that follow the `TLSGD` slot and open the call that closes the
/// pair: `data16 data16 rex.W call __tls_get_addr`.
const TLS_GD_CALL: [u8; 4] = [0x66, 0x66, 0x48, 0xe8];

/// The same call under `-fno-plt`, which reaches the helper through the GOT
/// instead: `data16 rex.W call *__tls_get_addr@GOTPCREL(%rip)`. It is the same
/// eight bytes wide, so the pair is rewritten the same way; only the
/// relocation on its displacement differs (`GOTPCRELX`, not `PLT32`).
const TLS_GD_CALL_INDIRECT: [u8; 4] = [0x66, 0x48, 0xff, 0x15];

/// The bytes that open a local-dynamic sequence, immediately before the
/// `TLSLD` slot: `lea sym@tlsld(%rip), %rdi`.
const TLS_LD_LEAD: [u8; 3] = [0x48, 0x8d, 0x3d];

/// The byte that follows the `TLSLD` slot and opens the call that closes the
/// sequence: `call __tls_get_addr`.
const TLS_LD_CALL: [u8; 1] = [0xe8];

/// The same call under `-fno-plt`, which reaches the helper through the GOT:
/// `call *__tls_get_addr@GOTPCREL(%rip)`. It is one byte wider than the direct
/// form, which is the whole reason the local-dynamic window has two sizes.
/// Arch Linux ships `-fno-plt` in its default CFLAGS, so two file-local
/// `_Thread_local`s in one translation unit are enough to produce it.
const TLS_LD_CALL_INDIRECT: [u8; 2] = [0xff, 0x15];

/// The whole twelve bytes a lowered local-dynamic sequence becomes: three
/// `data16` prefixes padding out `mov %fs:0, %rax`. The thread pointer takes
/// the place of the module base, so the offsets that follow it are measured
/// from there instead.
const TLS_LD_LE: [u8; 12] = [
    0x66, 0x66, 0x66, 0x64, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00,
];

/// The thirteen bytes the `-fno-plt` form becomes: one more `data16` prefix in
/// front of the same `mov %fs:0, %rax`, so the sequence stays exactly as long
/// as it was. This is the psABI's "Table 11.9: LD -> LE Code Transition
/// (LP64)", and lld writes the same bytes in `relaxTlsLdToLe`.
const TLS_LD_LE_INDIRECT: [u8; 13] = [
    0x66, 0x66, 0x66, 0x66, 0x64, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00,
    0x00,
];

/// The initial-exec sequence a relaxed pair becomes when the loader owns the
/// offset, up to its displacement field: `mov %fs:0, %rax` followed by the
/// opcode of `add off(%rip), %rax`. Its displacement lands in the same place
/// as the local-exec form's offset.
const TLS_IE_PREFIX: [u8; 12] = [
    0x64, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, 0x48, 0x03, 0x05,
];

/// The local-exec sequence a relaxed pair becomes, up to its offset field:
/// `mov %fs:0, %rax` followed by the opcode of `lea off(%rax), %rax`. The
/// 4-byte offset that completes it lands exactly where the `call`'s
/// displacement was, which is why the relocation on that displacement is
/// dropped: [`X86_64::relax_span`] reports the pair's bytes and the writer
/// skips every relocation inside them.
const TLS_LE_PREFIX: [u8; 12] = [
    0x64, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, 0x48, 0x8d, 0x80,
];

/// Rewrites a general-dynamic TLS pair into a form an executable can resolve,
/// following lld's `relaxTlsGdToLe` and `relaxTlsGdToIe`.
///
/// The pair is 16 bytes -- `data16 lea sym@tlsgd(%rip), %rdi` then
/// `data16 data16 rex.W call __tls_get_addr` -- and both replacements are 16
/// bytes as well, so the rewrite is in place.
///
/// Which one applies depends on who owns the thread-local. One the image
/// itself defines lives in the main module's static TLS block at an offset
/// fixed by this link, so the pair becomes the local-exec form, `mov %fs:0,
/// %rax` then `lea sym@tpoff(%rax), %rax`, with the offset folded in. One a
/// shared object defines is placed by the loader, so the pair becomes the
/// initial-exec form, `mov %fs:0, %rax` then `add sym@gottpoff(%rip), %rax`,
/// which loads the offset from a GOT slot the loader filled.
///
/// Returns `false` unless [`tls_gd_lowering`] accepts the window, in which
/// case the site is left for the escape path to reject rather than rewritten
/// into something unverified.
fn relax_tls_gd<R: Resolver>(
    sym: Option<SymbolId>,
    addend: i64,
    place: u64,
    resolver: &R,
    window: &mut [u8],
) -> bool {
    let Some((prefix, value)) =
        tls_gd_lowering(sym, addend, place, resolver, window)
    else {
        return false;
    };
    let head = prefix.len();
    window[..head].copy_from_slice(prefix);
    window[head..].copy_from_slice(&value.to_le_bytes());
    true
}

/// Whether `window` is a general-dynamic pair this image lowers, and what it
/// lowers to: the twelve replacement bytes and the 4-byte field that
/// completes them.
///
/// This is the one predicate the rewrite and the writer's pre-pass both ask.
/// The pre-pass needs the answer before any relocation is applied, so it can
/// drop the one on the `call` the rewrite consumes; the rewrite needs it to
/// know what to write. Two spellings of "is this the canonical pair" would be
/// free to disagree, and a disagreement means either a displacement written
/// over a lowered sequence or a call left pointing at a helper that is not
/// there.
///
/// The bytes must be the canonical encoding, the addend the canonical `-4`,
/// and the image an executable: a shared object reaches its own thread-locals
/// through the runtime, at an offset that is not known here.
fn tls_gd_lowering<R: Resolver>(
    sym: Option<SymbolId>,
    addend: i64,
    place: u64,
    resolver: &R,
    window: &[u8],
) -> Option<(&'static [u8; 12], i32)> {
    if !resolver.is_exec() {
        return None;
    }
    // The canonical encoding: the displacement is relative to the end of its
    // own field, which the assembler spells as an addend of -4.
    if addend != -4 {
        return None;
    }
    let (Some(sym), TLS_GD_PAIR) = (sym, window.len()) else {
        return None;
    };
    if window[..4] != TLS_GD_LEAD
        || (window[8..12] != TLS_GD_CALL
            && window[8..12] != TLS_GD_CALL_INDIRECT)
    {
        return None;
    }
    let got = resolver.tls_got_addr(sym);
    let (prefix, value) = if got == 0 {
        (&TLS_LE_PREFIX, resolver.symbol_addr(sym).cast_signed())
    } else {
        // The displacement is measured from the end of the rewritten
        // sequence, which is the `place` of the pair's own slot plus the
        // twelve bytes that follow it.
        (
            &TLS_IE_PREFIX,
            got.wrapping_sub(place.wrapping_add(12)).cast_signed(),
        )
    };
    i32::try_from(value).ok().map(|value| (prefix, value))
}

/// The bytes that open a descriptor reference, immediately before the
/// `GOTPC32_TLSDESC` slot: `rex.W lea x@tlsdesc(%rip), %reg`. Only the opcode
/// is fixed -- the REX byte carries the destination register's high bit and the
/// ModR/M byte its low three -- so the encoding is checked by mask rather than
/// compared, exactly as lld's `relaxTlsDescToLe` checks it.
const TLS_DESC_LEAD: usize = 3;

/// The two bytes of the indirect call a descriptor sequence closes with:
/// `call *(%rax)`.
const TLS_DESC_CALL: [u8; 2] = [0xff, 0x10];

/// What that call becomes once the descriptor is gone: `xchg %ax, %ax`, the
/// two-byte `nop`.
const TLS_DESC_NOP: [u8; 2] = [0x66, 0x90];

/// Rewrites a TLS descriptor reference into a form an executable can resolve,
/// following lld's `relaxTlsDescToLe` and `relaxTlsDescToIe`.
///
/// The instruction is `lea x@tlsdesc(%rip), %reg`, seven bytes of which the
/// last four are the slot. Which replacement applies is the same question the
/// general-dynamic pair asks (see [`tls_gd_lowering`]): a thread-local this
/// image places has a fixed offset from the thread pointer, so the `lea`
/// becomes `mov $x@tpoff, %reg`; one a shared object places is the loader's to
/// position, so it becomes `mov x@gottpoff(%rip), %reg` and reads the offset
/// from a GOT slot. Both replacements are the same seven bytes, so the rewrite
/// is in place, and both keep the destination register the input chose.
///
/// Returns `false` for anything that is not the canonical encoding with the
/// canonical `-4` addend, leaving the site for the writer to reject rather than
/// rewriting bytes whose meaning was not established.
fn relax_tls_desc<R: Resolver>(
    sym: Option<SymbolId>,
    addend: i64,
    place: u64,
    resolver: &R,
    window: &mut [u8],
) -> bool {
    let (Some(sym), true) = (sym, resolver.is_exec()) else {
        return false;
    };
    // The canonical encoding: the displacement is relative to the end of its
    // own field, which the assembler spells as an addend of -4.
    if addend != -4 || window.len() != TLS_DESC_LEAD + DISP32 {
        return false;
    }
    // `rex.W` with any REX.R, opcode `lea`, and a RIP-relative ModR/M byte
    // (mod = 00, r/m = 101) naming any destination register.
    if window[0] & 0xfb != 0x48 || window[1] != 0x8d || window[2] & 0xc7 != 0x05
    {
        return false;
    }
    let got = resolver.tls_got_addr(sym);
    let value = if got == 0 {
        resolver.symbol_addr(sym).cast_signed()
    } else {
        // The same displacement the unlowered reference would have stored, and
        // computed the same way: the slot keeps its place in the instruction,
        // so only the opcode changes.
        got.wrapping_add(addend.cast_unsigned())
            .wrapping_sub(place)
            .cast_signed()
    };
    let Ok(value) = i32::try_from(value) else {
        return false;
    };
    if got == 0 {
        // `lea` -> `mov $imm32, %reg`: the register moves from the ModR/M
        // reg field to its r/m field, so REX.R moves to REX.B with it.
        window[0] = 0x48 | ((window[0] >> 2) & 1);
        window[1] = 0xc7;
        window[2] = 0xc0 | ((window[2] >> 3) & 7);
    } else {
        // `lea` -> `mov m64, %reg`: one opcode byte, the operands unchanged.
        window[1] = 0x8b;
    }
    window[TLS_DESC_LEAD..].copy_from_slice(&value.to_le_bytes());
    true
}

/// Replaces the indirect call that closes a descriptor sequence with a
/// two-byte `nop`, following lld.
///
/// The sequence's first instruction already left the offset from the thread
/// pointer in the destination register, so the call to the resolver has nothing
/// left to do -- and no descriptor to reach it through. Returns `false` for
/// anything but the canonical `call *(%rax)`, which the writer reports against
/// the thread-local rather than guessing at.
fn relax_tls_desc_call<R: Resolver>(resolver: &R, window: &mut [u8]) -> bool {
    if !resolver.is_exec() || window != TLS_DESC_CALL {
        return false;
    }
    window.copy_from_slice(&TLS_DESC_NOP);
    true
}

/// Rewrites a local-dynamic sequence to read the thread pointer directly,
/// following lld's `relaxTlsLdToLe`.
///
/// The sequence is `lea sym@tlsld(%rip), %rdi` then
/// `call __tls_get_addr`, twelve bytes that ask the runtime for this module's
/// thread-local block. An executable's block sits at a fixed offset from the
/// thread pointer, so the call is replaced by `mov %fs:0, %rax` and the
/// `DTPOFF` offsets that follow are resolved against the thread pointer
/// instead (see [`Resolver::tls_dtp_off`]).
///
/// Returns `false` unless [`is_tls_ld_sequence`] accepts the window, in which
/// case the reference is left for `compute` to reject rather than rewritten
/// into something unverified.
fn relax_tls_ld<R: Resolver>(resolver: &R, window: &mut [u8]) -> bool {
    if !is_tls_ld_sequence(resolver, window) {
        return false;
    }
    // Both replacements are as long as the sequence they replace, so the
    // rewrite is in place either way; the window's own length says which form
    // this is, because `is_tls_ld_sequence` has already matched the opcodes.
    if window.len() == TLS_LD_LE_INDIRECT.len() {
        window.copy_from_slice(&TLS_LD_LE_INDIRECT);
    } else {
        window.copy_from_slice(&TLS_LD_LE);
    }
    true
}

/// Whether `window` is a local-dynamic sequence this image lowers.
///
/// The counterpart of [`tls_gd_lowering`], and asked for the same reason: the
/// rewrite consumes the call that closes the sequence, so the relocation on
/// that call must be dropped, and the writer decides that from the sequence
/// rather than from the call. Only an executable lowers it.
///
/// Two shapes qualify, both opening with the same `lea sym@tlsld(%rip), %rdi`
/// and differing only in how they reach the helper: a direct `call` under the
/// default ABI, twelve bytes in all, and a `call *...@GOTPCREL(%rip)` under
/// `-fno-plt`, thirteen. `TLSLD` is mandatory in an executable, so a shape
/// this declines is not a missed optimisation but a failed link. lld matches
/// the same two in `relaxTlsLdToLe` and errors on anything else.
fn is_tls_ld_sequence<R: Resolver>(resolver: &R, window: &[u8]) -> bool {
    if !resolver.is_exec() || window.get(..3) != Some(&TLS_LD_LEAD[..]) {
        return false;
    }
    match window.len() {
        n if n == TLS_LD_LE.len() => window.get(7..8) == Some(&TLS_LD_CALL[..]),
        n if n == TLS_LD_LE_INDIRECT.len() => {
            window.get(7..9) == Some(&TLS_LD_CALL_INDIRECT[..])
        }
        _ => false,
    }
}

/// Relaxes a `GOTPCRELX`/`REX_GOTPCRELX` site, following lld's `relaxGot`.
/// Returns `true` if the instruction was rewritten, `false` to leave it as a
/// GOT access. Conservative: any form lld does not handle, or any precondition
/// that does not hold, falls back.
fn relax_gotpcrelx<R: Resolver>(
    r_type: u32,
    sym: Option<SymbolId>,
    addend: i64,
    place: u64,
    resolver: &R,
    window: &mut [u8],
) -> bool {
    if !matches!(r_type, R_X86_64_GOTPCRELX | R_X86_64_REX_GOTPCRELX) {
        return false;
    }
    // A missing GOT slot is not a reason to decline. The relaxed form reads
    // no slot: its displacement is `S + A - P`, measured to the symbol
    // itself. The scan declines to allocate precisely when it predicts this
    // rewrite, so declining here would refuse the sites the prediction was
    // made for. `relax_required` still reports the absence, so a site that
    // fails one of the tests below ends the link with a diagnostic rather
    // than leaving the input's placeholder bytes in the image.
    // lld requires addend == -4 (the canonical RIP-relative encoding). Any
    // other addend means the instruction does not load the full GOT entry, so
    // rewriting it would change semantics.
    if addend != -4 {
        return false;
    }
    // Need opcode + ModR/M + disp32.
    let lead = X86_64::relax_lead(r_type);
    if window.len() < lead + DISP32 {
        return false;
    }
    let Some(sym) = sym else {
        return false;
    };
    // Only relax a reference nothing can redirect. Folding the GOT load into a
    // direct `lea` binds the site to this image's definition for good, which
    // is wrong for a symbol the loader may resolve elsewhere: an import it
    // binds to a dependency, or a shared object's own default-visibility
    // definition that the executable loading it may interpose. lld gates
    // `relaxGot` on the same question.
    if resolver.is_preemptible(sym) {
        return false;
    }
    // And only one whose value is an address. The GOT slot holds the symbol's
    // value; a `lea` computes it from the program counter, so for a constant
    // -- an `SHN_ABS` definition, an undefined reference, a thread-pointer
    // offset -- the two differ by the load base. lld gates `adjustGotPcExpr`
    // on `isAbsoluteValue` for exactly this.
    if resolver.is_absolute(sym) {
        return false;
    }
    let sym_addr = resolver.symbol_addr(sym);
    // The PC-relative displacement the relaxed instruction stores, with the
    // canonical RIP-relative addend folded in: `S + A - P`. The original
    // 6-byte instruction ends at `P + 4`, so its displacement is
    // `sym - P - 4`, which is exactly this value.
    let val = sym_addr
        .wrapping_add(addend.cast_unsigned())
        .wrapping_sub(place);
    let op = window[0];
    let modrm = window[1];
    if op == 0x8b {
        // `mov sym@GOTPCREL(%rip), %reg` -> `lea sym(%rip), %reg`. Single
        // opcode byte change; ModR/M and the disp32 layout are unchanged.
        if !WriteKind::W32S.fits(val) {
            return false;
        }
        window[0] = 0x8d;
        write_disp32(window, lead, val);
        return true;
    }
    if op == 0xff && (modrm == 0x15 || modrm == 0x25) {
        return relax_indirect_call(window, modrm, val);
    }
    // Other forms (test/binop via `R_RELAX_GOT_PC_NOPIC`, REX2 variants) are
    // not relaxed; they keep the GOT access.
    false
}

/// Relaxes `call/jmp *sym@GOTPCREL(%rip)` to a direct `call`/`jmp`.
///
/// - `ff 15 disp32` (call) -> `67 e8 disp32` (an `addr32`-prefixed direct call,
///   so the encoding stays one instruction and the disp32 field does not move;
///   the displacement is the same `val = sym + A - P`).
/// - `ff 25 disp32` (jmp) -> `e9 disp32 90` (direct jmp plus a trailing `nop`).
///   The direct jmp is one byte shorter than the indirect form, so its
///   end-of-instruction origin shifts back by one byte and the displacement
///   becomes `val + 1`.
///
/// The form is chosen and its displacement range-checked before any byte is
/// written: a rejected relaxation must leave the GOT access exactly as it
/// found it, because the caller then applies the original relocation over
/// these same bytes. Rewriting the opcode first would leave a direct
/// `call`/`jmp` holding a GOT-relative displacement.
fn relax_indirect_call(window: &mut [u8], modrm: u8, val: u64) -> bool {
    let is_call = modrm == 0x15;
    // The direct jmp's end-of-instruction origin is one byte earlier than the
    // indirect form's, so its displacement is `val + 1`.
    let disp_val = if is_call { val } else { val.wrapping_add(1) };
    if !WriteKind::W32S.fits(disp_val) {
        return false;
    }
    if is_call {
        // call: prefix 0x67 at [0], opcode 0xe8 at [1], disp32 at [2..6].
        window[0] = 0x67; // addr32 prefix
        window[1] = 0xe8; // call
        write_disp32(window, 2, disp_val);
    } else {
        // jmp: opcode 0xe9 at [0], disp32 at [1..5], nop at [5].
        window[0] = 0xe9; // jmp
        write_disp32(window, 1, disp_val);
        // The original disp32 occupied window[2..6]; window[5] is now past
        // the direct jmp's disp32 and becomes a trailing nop.
        window[5] = 0x90;
    }
    true
}

/// Writes `value` as a little-endian signed 32-bit displacement at
/// `window[at..at + 4]`. The caller must have range-checked `value`.
fn write_disp32(window: &mut [u8], at: usize, value: u64) {
    // Truncation is sound: the caller checked `WriteKind::W32S.fits(value)`.
    #[allow(clippy::cast_possible_truncation)]
    let bytes = (value as i32).to_le_bytes();
    window[at..at + DISP32].copy_from_slice(&bytes);
}
