//! Symbol visibility and preemptibility.
//!
//! `st_other` decides three things at once, so they are checked together:
//!
//! - A hidden or internal definition is not part of a shared object's ABI, so
//!   it stays out of `.dynsym` while keeping its place (and its real
//!   `st_other`) in `.symtab`.
//! - A protected definition is exported, but nothing can preempt it, so a
//!   pointer to it takes an `R_X86_64_RELATIVE` rather than the symbol-based
//!   `R_X86_64_64` a default-visibility definition needs. A GOT slot answers
//!   the same question and gets `R_X86_64_GLOB_DAT` or `R_X86_64_RELATIVE`
//!   accordingly.
//! - A hidden *undefined* reference names something no loader can supply, so it
//!   is never emitted as an import: a weak one resolves to zero, and a strong
//!   one is an error.
//! - `--relax` may only fold a GOT load into a direct reference when the symbol
//!   cannot be preempted; the visibility that decides it is the merged one, so
//!   a hidden definition constrains a reference compiled against a
//!   default-visibility declaration.
//!
//! Gated on `clang`; if it is absent the tests print a note and return, so the
//! build never fails over a missing toolchain.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::{
        ObjectFile, Sym64,
        constants::{SHN_UNDEF, STV_DEFAULT, STV_HIDDEN, STV_PROTECTED},
    },
    icf::IcfMode,
    linker::link_shared,
};

mod common;

/// `R_X86_64_64`: the symbol-based absolute data relocation.
const R_X86_64_64: u32 = 1;
/// `R_X86_64_GLOB_DAT`: the loader resolves the named symbol and stores its
/// address in the slot.
const R_X86_64_GLOB_DAT: u32 = 6;
/// `R_X86_64_RELATIVE`: load base plus addend, no symbol named.
const R_X86_64_RELATIVE: u32 = 8;

/// One hidden and one default-visibility function, the hidden one reached only
/// from inside the object.
const HIDDEN_SRC: &[u8] = b"__attribute__((visibility(\"hidden\")))\n\
    int hidden_fn(int x) { return x + 1; }\n\
    int public_fn(int x) { return hidden_fn(x) * 2; }\n";

/// A protected and a default-visibility datum, each with a pointer to it so
/// both flavours of absolute data relocation appear in one object.
const PROTECTED_SRC: &[u8] = b"__attribute__((visibility(\"protected\")))\n\
    int pg = 5;\n\
    int dg = 7;\n\
    int *pp = &pg;\n\
    int *dp = &dg;\n";

/// The referencing translation unit. `g` is declared without an attribute, so
/// clang emits the GOT-indirect `R_X86_64_REX_GOTPCRELX` form and the merged
/// visibility is whatever the defining object says.
const USE_SRC: &[u8] = b"extern int g;\nint getg(void) { return g; }\n";

/// The two defining translation units `USE_SRC` is linked against.
const DEF_DEFAULT_SRC: &[u8] = b"int g = 5;\n";
const DEF_HIDDEN_SRC: &[u8] =
    b"__attribute__((visibility(\"hidden\"))) int g = 5;\n";

/// A weak reference to a hidden name nothing defines. It is a probe: no loader
/// can supply a hidden name, so the reference must resolve to zero and the
/// guard around it takes the other branch.
const WEAK_HIDDEN_SRC: &[u8] =
    b"extern void wh_fn(void) __attribute__((weak, visibility(\"hidden\")));\n\
      int call_wh(void) { if (wh_fn) { wh_fn(); return 1; } return 0; }\n";

/// The same reference without `weak`. Nothing can ever satisfy it, so the link
/// has to say so.
const STRONG_HIDDEN_SRC: &[u8] =
    b"extern void sh_fn(void) __attribute__((visibility(\"hidden\")));\n\
      int call_sh(void) { sh_fn(); return 0; }\n";

/// `mov off(%rip), %rax`: the GOT-indirect load, kept for a preemptible
/// symbol.
const REX_MOV_RIP: [u8; 3] = [0x48, 0x8b, 0x05];
/// `lea off(%rip), %rax`: what the relaxation folds that load into.
const REX_LEA_RIP: [u8; 3] = [0x48, 0x8d, 0x05];

/// A hidden definition is absent from `.dynsym` while a default-visibility one
/// is present, and `.symtab` keeps both with their real `st_other`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_hidden_definition_is_kept_out_of_dynsym() {
    let Some(dir) = workdir("hidden") else {
        return;
    };
    let obj = dir.join("v.o");
    let out = dir.join("libv.so");
    compile(HIDDEN_SRC, &obj).expect("host clang compiles -fPIC");
    link(&[obj], &out);

    let bytes = fs::read(&out).expect("read shared object");
    assert_eq!(
        visibility_in(&bytes, Table::Dyn, b"hidden_fn"),
        None,
        "a hidden definition is not part of the object's ABI"
    );
    assert_eq!(
        visibility_in(&bytes, Table::Dyn, b"public_fn"),
        Some(STV_DEFAULT),
        "a default-visibility definition is exported"
    );
    // The static symbol table is not the ABI, so it keeps both -- with the
    // visibility the input declared rather than a hard-coded zero.
    assert_eq!(
        visibility_in(&bytes, Table::Static, b"hidden_fn"),
        Some(STV_HIDDEN),
        "`.symtab` records the hidden definition as hidden"
    );
    assert_eq!(
        visibility_in(&bytes, Table::Static, b"public_fn"),
        Some(STV_DEFAULT)
    );
}

/// A reference's visibility constrains the resolved symbol even when the
/// definition that wins carries none: a hidden `extern` declaration in one
/// object hides a default-visibility definition in another.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_hidden_reference_hides_a_default_definition() {
    let Some(dir) = workdir("merge") else {
        return;
    };
    let def = dir.join("def.o");
    let refr = dir.join("ref.o");
    let out = dir.join("libmerge.so");
    compile(b"int g = 5;\nint getg(void) { return g; }\n", &def)
        .expect("host clang compiles the definition");
    compile(
        b"__attribute__((visibility(\"hidden\"))) extern int g;\n\
          int useg(void) { return g + 1; }\n",
        &refr,
    )
    .expect("host clang compiles the reference");
    link(&[def, refr], &out);

    let bytes = fs::read(&out).expect("read shared object");
    assert_eq!(
        visibility_in(&bytes, Table::Dyn, b"g"),
        None,
        "the hidden reference constrains the definition out of `.dynsym`"
    );
    assert_eq!(
        visibility_in(&bytes, Table::Static, b"g"),
        Some(STV_HIDDEN),
        "the merged visibility is the most constrained one observed"
    );
}

/// A protected definition is exported, but its address is fixed by this link,
/// so a pointer to it takes `R_X86_64_RELATIVE` where a pointer to a
/// default-visibility definition stays symbol-based.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_protected_definition_is_exported_but_not_preemptible() {
    let Some(dir) = workdir("protected") else {
        return;
    };
    let obj = dir.join("p.o");
    let out = dir.join("libp.so");
    compile(PROTECTED_SRC, &obj).expect("host clang compiles -fPIC");
    link(&[obj], &out);

    let bytes = fs::read(&out).expect("read shared object");
    assert_eq!(
        visibility_in(&bytes, Table::Dyn, b"pg"),
        Some(STV_PROTECTED),
        "a protected definition is still exported"
    );
    // `pp` holds `&pg` and `dp` holds `&dg`. Both are pointer slots in the
    // same section, so the only thing that can distinguish their relocations
    // is whether the target can be preempted.
    let relocs = rela_dyn(&bytes);
    let pp = value_addr(&bytes, b"pp").expect("`pp` is exported");
    let dp = value_addr(&bytes, b"dp").expect("`dp` is exported");
    let pg = value_addr(&bytes, b"pg").expect("`pg` is exported");
    let protected = reloc_at(&relocs, pp).expect("`pp` needs a dynamic reloc");
    let interposable =
        reloc_at(&relocs, dp).expect("`dp` needs a dynamic reloc");
    assert_eq!(
        protected.r_type, R_X86_64_RELATIVE,
        "a protected target's address is fixed by this link"
    );
    assert_eq!(
        protected.addend,
        pg.cast_signed(),
        "the RELATIVE addend is the protected symbol's own address"
    );
    assert_eq!(
        interposable.r_type, R_X86_64_64,
        "a default-visibility target may be interposed, so the loader \
         resolves it by name"
    );
    assert_ne!(
        interposable.sym, 0,
        "the symbol-based entry names its target"
    );
}

/// A GOT slot poses the loader the same question a pointer slot does, so the
/// same answer decides it: a preemptible definition's slot is bound by name,
/// a hidden one's holds an address this link fixed.
///
/// Both links reference `g` through the same `R_X86_64_REX_GOTPCRELX` site and
/// leave relaxation off, so the slot survives in both and the only difference
/// is the merged visibility of the definition.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_got_slot_is_bound_by_name_only_when_the_symbol_is_preemptible() {
    let Some(dir) = workdir("gotslot") else {
        return;
    };
    let use_o = dir.join("use.o");
    let def_default = dir.join("dd.o");
    let def_hidden = dir.join("dh.o");
    compile(USE_SRC, &use_o).expect("host clang compiles the reference");
    compile(DEF_DEFAULT_SRC, &def_default).expect("host clang compiles");
    compile(DEF_HIDDEN_SRC, &def_hidden).expect("host clang compiles");
    let default_so = dir.join("libgd.so");
    let hidden_so = dir.join("libgh.so");
    link(&[use_o.clone(), def_default], &default_so);
    link(&[use_o, def_hidden], &hidden_so);

    // Preemptible: the executable that loads this object may define `g` too,
    // so the slot must name `g` for the loader to fill in.
    let default_bytes = fs::read(&default_so).expect("read default output");
    let index = dynsym_index(&default_bytes, b"g").expect("`g` is exported");
    let glob = rela_dyn(&default_bytes)
        .into_iter()
        .find(|r| r.r_type == R_X86_64_GLOB_DAT)
        .expect("a preemptible symbol's GOT slot needs the loader");
    assert_eq!(glob.sym, index, "the GLOB_DAT entry must name `g`");
    assert_eq!(glob.addend, 0, "the loader supplies the whole address");

    // Hidden: nothing can replace this definition, so the link fixes the
    // address and the loader only shifts it by the load base.
    let hidden_bytes = fs::read(&hidden_so).expect("read hidden output");
    let g = find_symbol(&hidden_bytes, Table::Static, b"g")
        .expect("`g` is in `.symtab`")
        .st_value
        .get();
    let relocs = rela_dyn(&hidden_bytes);
    assert!(
        relocs.iter().all(|r| r.r_type != R_X86_64_GLOB_DAT),
        "a hidden definition gives the loader nothing to look up"
    );
    assert!(
        relocs
            .iter()
            .any(|r| r.r_type == R_X86_64_RELATIVE
                && r.addend == g.cast_signed()),
        "the hidden symbol's GOT slot takes a RELATIVE of its own address"
    );
}

/// A hidden undefined reference is never turned into an import.
///
/// Hidden means the name is private to this image, so a `.dynsym` row for one
/// is a name that can never bind. A weak reference is a probe and resolves to
/// zero; a strong one is an error, which is what lld reports as well.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_hidden_undefined_reference_is_never_imported() {
    let Some(dir) = workdir("undef") else {
        return;
    };
    let probe = dir.join("wh.o");
    let hard = dir.join("sh.o");
    compile(WEAK_HIDDEN_SRC, &probe).expect("host clang compiles the probe");
    compile(STRONG_HIDDEN_SRC, &hard).expect("host clang compiles");

    let probe_so = dir.join("libwh.so");
    link(&[probe], &probe_so);
    let bytes = fs::read(&probe_so).expect("read weak output");
    assert_eq!(
        visibility_in(&bytes, Table::Dyn, b"wh_fn"),
        None,
        "a hidden reference is not something a loader could resolve"
    );
    assert!(
        undefined_dynsym_names(&bytes).is_empty(),
        "the object imports nothing at all"
    );

    // The strong reference has no such escape: nothing in this image defines
    // it and nothing outside it ever could.
    let err = link_shared(
        &[hard],
        &dir.join("libsh.so"),
        None,
        false,
        IcfMode::None,
        false,
    )
    .expect_err("a strong hidden undefined reference must be refused");
    assert!(
        err.to_string().contains("sh_fn"),
        "the diagnostic must name the symbol, got {err}"
    );
}

/// `--relax` leaves a preemptible GOT load alone and folds a hidden one.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn relax_folds_only_what_cannot_be_preempted() {
    let Some(dir) = workdir("relax") else {
        return;
    };
    let use_o = dir.join("use.o");
    let def_default = dir.join("dd.o");
    let def_hidden = dir.join("dh.o");
    compile(USE_SRC, &use_o).expect("host clang compiles the reference");
    compile(DEF_DEFAULT_SRC, &def_default).expect("host clang compiles");
    compile(DEF_HIDDEN_SRC, &def_hidden).expect("host clang compiles");

    let default_so = dir.join("libdefault.so");
    let hidden_so = dir.join("libhidden.so");
    link_relaxed(&[use_o.clone(), def_default], &default_so);
    link_relaxed(&[use_o, def_hidden], &hidden_so);

    let default_bytes = fs::read(&default_so).expect("read default output");
    let hidden_bytes = fs::read(&hidden_so).expect("read hidden output");
    let default_text = text_of(&default_bytes);
    let hidden_text = text_of(&hidden_bytes);
    assert!(
        contains(&default_text, &REX_MOV_RIP)
            && !contains(&default_text, &REX_LEA_RIP),
        "a preemptible symbol keeps its GOT load"
    );
    assert!(
        contains(&hidden_text, &REX_LEA_RIP)
            && !contains(&hidden_text, &REX_MOV_RIP),
        "a hidden symbol's GOT load folds into a direct reference"
    );
    // Keeping the load is only worth anything if the loader has a name to
    // rebind, so the preemptible symbol stays in `.dynsym`.
    assert_eq!(
        visibility_in(&default_bytes, Table::Dyn, b"g"),
        Some(STV_DEFAULT),
        "the preemptible symbol is exported for the loader to rebind"
    );
    assert_eq!(
        visibility_in(&hidden_bytes, Table::Dyn, b"g"),
        None,
        "the hidden symbol is not exported at all"
    );
}

/// Which symbol table a lookup reads.
#[derive(Clone, Copy)]
enum Table {
    /// `.dynsym`: the image's ABI.
    Dyn,
    /// `.symtab`: every symbol the link kept, ABI or not.
    Static,
}

/// The visibility recorded for `name`, or `None` when the table has no such
/// symbol.
fn visibility_in(bytes: &[u8], table: Table, name: &[u8]) -> Option<u8> {
    find_symbol(bytes, table, name).map(|s| s.visibility())
}

/// The `st_value` of an exported symbol, read from `.dynsym`.
fn value_addr(bytes: &[u8], name: &[u8]) -> Option<u64> {
    find_symbol(bytes, Table::Dyn, name).map(|s| s.st_value.get())
}

/// The 1-based `.dynsym` index of `name`, or `None` when it is not exported.
/// This is the index a `GLOB_DAT` relocation names.
fn dynsym_index(bytes: &[u8], name: &[u8]) -> Option<u32> {
    let obj = ObjectFile::parse(bytes).expect("output must be valid ELF");
    let symtab = obj.dynamic_symbols().ok().flatten()?;
    let at = symtab.syms.iter().position(|s| symtab.name(s) == name)?;
    u32::try_from(at).ok()
}

/// The names of every undefined `.dynsym` entry: the imports the loader is
/// asked to resolve.
fn undefined_dynsym_names(bytes: &[u8]) -> Vec<Vec<u8>> {
    let obj = ObjectFile::parse(bytes).expect("output must be valid ELF");
    let Ok(Some(symtab)) = obj.dynamic_symbols() else {
        return Vec::new();
    };
    symtab
        .syms
        .iter()
        .filter(|s| s.st_shndx.get() == SHN_UNDEF && !symtab.name(s).is_empty())
        .map(|s| symtab.name(s).to_vec())
        .collect()
}

/// The entry named `name` in the requested table.
fn find_symbol(bytes: &[u8], table: Table, name: &[u8]) -> Option<Sym64> {
    let obj = ObjectFile::parse(bytes).expect("output must be valid ELF");
    let symtab = match table {
        Table::Dyn => obj.dynamic_symbols(),
        Table::Static => obj.symbol_table(),
    }
    .ok()
    .flatten()?;
    symtab.syms.iter().find(|s| symtab.name(s) == name).copied()
}

/// One decoded `.rela.dyn` entry.
struct DynReloc {
    offset: u64,
    sym: u32,
    r_type: u32,
    addend: i64,
}

/// Decodes `.rela.dyn`, or an empty list when the image has none.
fn rela_dyn(bytes: &[u8]) -> Vec<DynReloc> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(sec) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".rela.dyn")
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(sec) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for chunk in data.as_chunks::<24>().0 {
        let info = read_u64(chunk, 8);
        out.push(DynReloc {
            offset: read_u64(chunk, 0),
            sym: u32::try_from(info >> 32).unwrap_or(0),
            r_type: u32::try_from(info & 0xffff_ffff).unwrap_or(0),
            addend: read_u64(chunk, 16).cast_signed(),
        });
    }
    out
}

/// The relocation applying to the slot at `addr`.
fn reloc_at(relocs: &[DynReloc], addr: u64) -> Option<&DynReloc> {
    relocs.iter().find(|r| r.offset == addr)
}

/// The output's `.text` bytes.
fn text_of(bytes: &[u8]) -> Vec<u8> {
    let obj = ObjectFile::parse(bytes).expect("output must be valid ELF");
    let Some(sec) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".text")
    else {
        return Vec::new();
    };
    obj.section_data(sec).unwrap_or_default().to_vec()
}

/// Whether `haystack` contains `needle`.
fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

fn read_u64(bytes: &[u8], at: usize) -> u64 {
    let mut buf = [0u8; 8];
    if let Some(slot) = bytes.get(at..at + 8) {
        buf.copy_from_slice(slot);
    }
    u64::from_le_bytes(buf)
}

/// Compiles `src` to a `-fPIC` object with the host clang. `-O1` is what makes
/// clang emit the GOT-indirect load these tests are about.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-O1", "-fPIC", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// Links `objs` into a shared object with xold.
fn link(objs: &[PathBuf], out: &Path) {
    link_shared(objs, out, None, false, IcfMode::None, false)
        .expect("xold -shared link must succeed");
}

/// Links `objs` into a shared object with xold, with relaxation on.
fn link_relaxed(objs: &[PathBuf], out: &Path) {
    link_shared(objs, out, None, false, IcfMode::None, true)
        .expect("xold -shared --relax link must succeed");
}

/// A fresh per-test working directory, or `None` (after printing a note) when
/// clang is unavailable.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping visibility test {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("xold_visibility_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    Some(dir)
}
