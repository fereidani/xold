//! `.gnu.hash` (GNU symbol hash table) end-to-end tests.
//!
//! These compile a real `-fPIC` C source with several exports plus an
//! undefined import, link it with `xold -shared`, and check the resulting
//! shared object carries a well-formed `.gnu.hash`:
//!
//! - The `.gnu.hash` section (`SHT_GNU_HASH`) and the `DT_GNU_HASH` tag are
//!   present, and the classic `.hash`/`DT_HASH` remain alongside it.
//! - The on-disk table (header, bloom filter, buckets, chain) decodes and is
//!   internally consistent: every exported symbol's name passes the bloom
//!   filter, every bucket points at the first symbol of its chain, and each
//!   chain walk terminates (the LSB terminator is set on the last entry).
//! - `dlopen`/`dlsym` resolves every export and a lookup of a name that is
//!   absent returns `NULL` (the path that exercises the chain terminator).
//! - A dynamic executable produced by xold also carries `.gnu.hash` +
//!   `DT_GNU_HASH` and runs under the system `ld.so`.
//!
//! All tests are gated on `clang` (and `ld.so` for the executable run); if a
//! tool is absent they print a note and return, so the build never fails over
//! a missing toolchain.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::{ObjectFile, constants::*},
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
};

mod common;

/// A fixture with several exported functions plus a data export and one
/// undefined import. The exports are hashed (the defined tail of `.dynsym`);
/// `extern_sym` is the undefined import that forms the unhashed prefix.
const SHARED_SRC: &[u8] = b"int counter = 5;\n\
    int alpha(void) { return counter + 1; }\n\
    int beta(void) { return counter + 2; }\n\
    int gamma(void) { return counter + 3; }\n\
    int delta(void) { return counter + 4; }\n\
    extern int extern_sym(void);\n\
    int use_import(void) { return extern_sym() + counter; }\n";

/// The main source for the dynamic-executable run: calls `alpha` (imported
/// from the shared lib) and returns its value. Compiled `-fPIE`, the call is
/// `R_X86_64_PLT32`, routed through a PLT stub.
const MAIN_SRC: &[u8] =
    b"extern int alpha(void);\nint main(void) { return alpha() - 6; }\n";

/// A freestanding `_start` that calls `main` then exits with its return value
/// (syscall 60).
const START_SRC: &[u8] = b"    .text\n    .globl _start\n_start:\n    \
    call    main\n    movl    %eax, %edi\n    movl    $60, %eax\n    syscall\n";

/// Compiles `src` to a `-fPIC` (library) or `-fPIE` (main) object.
fn compile(src: &[u8], obj: &Path, pic: bool) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let mut cmd = Command::new(clang);
    cmd.arg("--target=x86_64-linux-gnu");
    cmd.arg(if pic { "-fPIC" } else { "-fPIE" });
    cmd.args(["-ffreestanding", "-c"]);
    cmd.arg(&src_path).arg("-o").arg(obj);
    let ok = cmd.status().ok()?.success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// Assembles a `.S` source into `obj`.
fn assemble(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("S");
    fs::write(&src_path, src).expect("write assembly");
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// A fresh per-test working directory under the system temp dir.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xold_gnuhash_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Links `obj` with `xold -shared` into `out`.
fn link_shared_with_xold(obj: &PathBuf, out: &Path, soname: &[u8]) {
    link_shared(
        std::slice::from_ref(obj),
        out,
        Some(soname),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold -shared link must succeed");
}

/// The `DT_*` tags of an ELF file, read via the xold reader.
fn dt_tags(path: &Path) -> Vec<i64> {
    let bytes = fs::read(path).expect("read ELF");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let dyn_shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynamic")
        .expect("has .dynamic");
    let data = obj.section_data(dyn_shdr).expect("dynamic bytes");
    let mut tags = Vec::new();
    for chunk in data.chunks(16) {
        if chunk.len() < 16 {
            break;
        }
        let tag = i64::from_le_bytes(chunk[..8].try_into().unwrap_or([0; 8]));
        tags.push(tag);
        if tag == DT_NULL {
            break;
        }
    }
    tags
}

/// Whether `path` carries a section named `name`.
fn has_section(path: &Path, name: &[u8]) -> bool {
    let bytes = fs::read(path).expect("read ELF");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    obj.sections().iter().any(|s| obj.section_name(s) == name)
}

/// The decoded GNU hash table of `path`, plus the exported (defined) symbol
/// names it ought to index. Returns `None` if the table is absent.
struct GnuHash {
    nbuckets: u32,
    symoffset: u32,
    mask_words: u32,
    bloom_shift: u32,
    bloom: Vec<u64>,
    buckets: Vec<u32>,
    chain: Vec<u32>,
    /// Defined (hashed) dynsym names in dynsym order from `symoffset`.
    hashed_names: Vec<Vec<u8>>,
}

/// Reads and decodes the `.gnu.hash` section of `path`.
fn decode_gnu_hash(path: &Path) -> Option<GnuHash> {
    let bytes = fs::read(path).expect("read ELF");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let gh = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".gnu.hash")?;
    let data = obj.section_data(gh).ok()?;
    let mut r = Reader::new(data);
    let nbuckets = r.u32();
    let symoffset = r.u32();
    let mask_words = r.u32();
    let bloom_shift = r.u32();
    let mut bloom = Vec::with_capacity(mask_words as usize);
    for _ in 0..mask_words {
        bloom.push(r.u64());
    }
    let mut buckets = Vec::with_capacity(nbuckets as usize);
    for _ in 0..nbuckets {
        buckets.push(r.u32());
    }
    // Collect defined dynsym names from symoffset onward.
    let dynsym = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynsym")
        .expect("has .dynsym");
    let dynstr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynstr")
        .expect("has .dynstr");
    let sym_bytes = obj.section_data(dynsym).expect("dynsym bytes");
    let strtab = obj.section_data(dynstr).expect("dynstr bytes");
    let nsyms = sym_bytes.len() / 24;
    let nchain = nsyms.saturating_sub(symoffset as usize);
    let mut chain = Vec::with_capacity(nchain);
    for _ in 0..nchain {
        chain.push(r.u32());
    }
    let mut hashed_names = Vec::new();
    for i in (symoffset as usize)..nsyms {
        let off = i * 24;
        if off + 24 > sym_bytes.len() {
            break;
        }
        let name_off = u32::from_le_bytes(
            sym_bytes[off..off + 4].try_into().unwrap_or([0; 4]),
        ) as usize;
        let end = strtab[name_off..]
            .iter()
            .position(|&b| b == 0)
            .map_or(strtab.len(), |p| name_off + p);
        hashed_names.push(strtab[name_off..end].to_vec());
    }
    Some(GnuHash {
        nbuckets,
        symoffset,
        mask_words,
        bloom_shift,
        bloom,
        buckets,
        chain,
        hashed_names,
    })
}

/// A minimal little-endian cursor over a byte slice.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(
            self.bytes[self.pos..self.pos + 4]
                .try_into()
                .unwrap_or([0; 4]),
        );
        self.pos += 4;
        v
    }

    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(
            self.bytes[self.pos..self.pos + 8]
                .try_into()
                .unwrap_or([0; 8]),
        );
        self.pos += 8;
        v
    }
}

/// The GNU hash (`dl_new_hash`).
#[allow(clippy::missing_const_for_fn)]
fn gnu_hash(name: &[u8]) -> u32 {
    let mut h: u32 = 5381;
    for &c in name {
        h = h.wrapping_shl(5).wrapping_add(h).wrapping_add(u32::from(c));
    }
    h
}

/// The structural test: the `.gnu.hash` section and `DT_GNU_HASH` are present,
/// `SHT_GNU_HASH` is the section type, and the classic `.hash`/`DT_HASH` are
/// kept alongside it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn shared_object_has_gnu_hash_section_and_tag() {
    let Some(_) = which("clang") else {
        eprintln!("skipping gnu.hash structural test: clang not found");
        return;
    };
    let dir = workdir("struct");
    let obj = dir.join("lib.o");
    let so = dir.join("libgh.so");
    compile(SHARED_SRC, &obj, true).expect("host clang compiles -fPIC");
    link_shared_with_xold(&obj, &so, b"libgh.so");

    let bytes = fs::read(&so).expect("read .so");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let gh = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".gnu.hash")
        .expect(".gnu.hash section present");
    assert_eq!(
        gh.sh_type.get(),
        SHT_GNU_HASH,
        "section type is SHT_GNU_HASH"
    );

    let tags = dt_tags(&so);
    assert!(tags.contains(&DT_GNU_HASH), "DT_GNU_HASH present");
    assert!(tags.contains(&DT_HASH), "classic DT_HASH retained");
    assert!(has_section(&so, b".hash"), ".hash section retained");
}

/// Decodes the `.gnu.hash` table and checks it is internally consistent:
/// symoffset sits past the undefined prefix, every hashed name passes the
/// bloom filter, each bucket points at the first symbol of its chain, and the
/// loader walk (chain entries, LSB terminator) locates each export.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn gnu_hash_table_decodes_and_is_consistent() {
    let Some(_) = which("clang") else {
        eprintln!("skipping gnu.hash decode test: clang not found");
        return;
    };
    let dir = workdir("decode");
    let obj = dir.join("lib.o");
    let so = dir.join("libgh.so");
    compile(SHARED_SRC, &obj, true).expect("host clang compiles -fPIC");
    link_shared_with_xold(&obj, &so, b"libgh.so");

    let gh = decode_gnu_hash(&so).expect(".gnu.hash decodes");
    assert!(
        gh.mask_words >= 1 && gh.mask_words.is_power_of_two(),
        "bloom size is a power-of-two word count, got {}",
        gh.mask_words
    );
    assert_eq!(gh.bloom.len(), gh.mask_words as usize);
    assert_eq!(gh.buckets.len(), gh.nbuckets as usize);
    assert!(gh.nbuckets >= 1, "at least one bucket");

    // Undefined imports form the unhashed prefix: symoffset must be past the
    // first dynsym entry (index 1, the first real symbol).
    assert!(
        gh.symoffset >= 1,
        "symoffset past the null entry, got {}",
        gh.symoffset
    );

    let word_bits = 64u32;
    let word_mask = u64::from(gh.mask_words).saturating_sub(1);
    for name in &gh.hashed_names {
        let h = gnu_hash(name);
        let word_idx = ((u64::from(h) / 64) & word_mask) as usize;
        let bit1 = u64::from(h % word_bits);
        let bit2 = u64::from((h >> gh.bloom_shift) % word_bits);
        let w = gh.bloom[word_idx];
        assert!(
            w & (1u64 << bit1) != 0 && w & (1u64 << bit2) != 0,
            "bloom filter sets both bits for {:?}",
            String::from_utf8_lossy(name)
        );

        // The loader walk: bucket -> chain[symidx - symoffset] until match or
        // terminator (LSB set). Must terminate and find the symbol's hash.
        let bucket = h % gh.nbuckets;
        let start = gh.buckets[bucket as usize];
        assert!(start >= gh.symoffset, "bucket points into the hashed tail");
        let mut found = false;
        for symidx in (start..).take(gh.chain.len()) {
            let ci = usize::try_from(symidx)
                .unwrap_or(0)
                .saturating_sub(gh.symoffset as usize);
            let chain_hash = *gh.chain.get(ci).expect("chain in range");
            if (chain_hash & !1) == (h & !1) {
                found = true;
                break;
            }
            if chain_hash & 1 != 0 {
                break;
            }
        }
        assert!(
            found,
            "chain walk locates {:?} by hash",
            String::from_utf8_lossy(name)
        );
    }

    // A fabricated name the bloom filter is unlikely to clear: confirm the
    // table does not silently accept everything (best-effort, not exhaustive).
    let bad = gnu_hash(b"__xold_definitely_not_exported_zzz__");
    let b1 = bad % word_bits;
    let b2 = (bad >> gh.bloom_shift) % word_bits;
    let wi = ((u64::from(bad) / 64) & word_mask) as usize;
    let bloom_pass =
        gh.bloom[wi] & (1u64 << b1) != 0 && gh.bloom[wi] & (1u64 << b2) != 0;
    // A bloom false positive is allowed (probabilistic); a true negative is
    // the common case. Either way the chain walk must NOT find it.
    let bucket = bad % gh.nbuckets;
    let start = gh.buckets[bucket as usize];
    let mut hit = false;
    if start >= gh.symoffset {
        for symidx in (start..).take(gh.chain.len()) {
            let ci = usize::try_from(symidx)
                .unwrap_or(0)
                .saturating_sub(gh.symoffset as usize);
            let Some(chain_hash) = gh.chain.get(ci) else {
                break;
            };
            if (chain_hash & !1) == (bad & !1) {
                hit = true;
                break;
            }
            if chain_hash & 1 != 0 {
                break;
            }
        }
    }
    assert!(!hit, "absent name is not found in the chain");
    let _ = bloom_pass;
}

/// The authoritative functional check: `dlopen` the library produced by xold,
/// resolve every export, call them, and verify a lookup of an absent name
/// returns `NULL` (the path that exercises the chain terminator).
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn dlopen_resolves_exports_and_a_missing_lookup_fails() {
    let Some(_) = which("clang") else {
        eprintln!("skipping dlopen test: clang not found");
        return;
    };
    let dir = workdir("dlopen");
    let obj = dir.join("lib.o");
    let so = dir.join("libgh.so");
    compile(SHARED_SRC, &obj, true).expect("host clang compiles -fPIC");
    link_shared_with_xold(&obj, &so, b"libgh.so");

    if !cfg!(target_os = "linux") {
        assert_loader_lookup_contract(&so);
        return;
    }

    let harness_src = b"#include <dlfcn.h>\n\
        int extern_sym(void) { return 11; }\n\
        int main(void) {\n\
            void *h = dlopen(\"./libgh.so\", RTLD_NOW);\n\
            if (!h) return 100;\n\
            int (*a)(void) = dlsym(h, \"alpha\");\n\
            int (*g)(void) = dlsym(h, \"gamma\");\n\
            int (*d)(void) = dlsym(h, \"delta\");\n\
            int *c = dlsym(h, \"counter\");\n\
            if (!a||!g||!d||!c) return 101;\n\
            dlerror();\n\
            void *nope = dlsym(h, \"__xold_missing_zzz\");\n\
            if (nope || !dlerror()) return 102;\n\
            int (*u)(void) = dlsym(h, \"use_import\");\n\
            if (!u) return 104;\n\
            int ok = (a()==6 && g()==8 && d()==9 && *c==5 && u()==16);\n\
            dlclose(h);\n\
            return ok ? 0 : 103;\n\
        }\n";
    let harness_c = dir.join("harness.c");
    let harness = dir.join("harness");
    fs::write(&harness_c, harness_src).expect("write harness");
    let built = Command::new(which("clang").unwrap())
        .args(["--target=x86_64-linux-gnu"])
        .arg(&harness_c)
        .arg("-o")
        .arg(&harness)
        // The library imports `extern_sym`; exporting the harness's own
        // symbols is what lets `RTLD_NOW` resolve it.
        .args(["-rdynamic", "-ldl"])
        .status()
        .expect("compile harness")
        .success();
    assert!(built, "harness compiles");

    let status = Command::new(&harness)
        .current_dir(&dir)
        .status()
        .expect("run harness");
    assert!(
        status.success(),
        "dlopen resolves exports and absent lookup fails; got {status}"
    );
}

/// Exercises the same bloom/bucket/chain/name comparisons as glibc's GNU-hash
/// lookup on hosts that cannot dlopen an ELF image.
fn assert_loader_lookup_contract(so: &Path) {
    let gh = decode_gnu_hash(so).expect(".gnu.hash decodes");
    for name in [
        b"alpha".as_slice(),
        b"beta",
        b"gamma",
        b"delta",
        b"counter",
        b"use_import",
    ] {
        assert!(
            gnu_lookup(&gh, name).is_some(),
            "loader lookup resolves {}",
            String::from_utf8_lossy(name)
        );
    }
    assert!(
        gnu_lookup(&gh, b"__xold_missing_zzz").is_none(),
        "an absent lookup reaches a chain terminator"
    );
    assert!(
        gnu_lookup(&gh, b"extern_sym").is_none(),
        "an undefined import is not in this object's defining hash tail"
    );

    let bytes = fs::read(so).expect("read shared object");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let dynsym = obj.dynamic_symbols().expect("read dynsym").expect("dynsym");
    let counter = dynsym
        .syms
        .iter()
        .find(|sym| dynsym.name(sym) == b"counter")
        .expect("counter export");
    let addr = counter.st_value.get();
    let value = obj.sections().iter().find_map(|sec| {
        let base = sec.sh_addr.get();
        if addr < base || addr.saturating_add(4) > base + sec.sh_size.get() {
            return None;
        }
        let at = usize::try_from(addr - base).ok()?;
        let data = obj.section_data(sec).ok()?;
        data.get(at..at + 4)
            .and_then(|cell| cell.try_into().ok())
            .map(u32::from_le_bytes)
    });
    assert_eq!(value, Some(5), "the resolved counter export starts at five");
}

fn gnu_lookup(table: &GnuHash, name: &[u8]) -> Option<u32> {
    let hash = gnu_hash(name);
    let word = table.bloom
        [((u64::from(hash) / 64) & u64::from(table.mask_words - 1)) as usize];
    let first = u64::from(hash % 64);
    let second = u64::from((hash >> table.bloom_shift) % 64);
    if word & (1 << first) == 0 || word & (1 << second) == 0 {
        return None;
    }
    let mut sym = *table.buckets.get((hash % table.nbuckets) as usize)?;
    if sym < table.symoffset {
        return None;
    }
    loop {
        let at = usize::try_from(sym - table.symoffset).ok()?;
        let chain = *table.chain.get(at)?;
        if chain & !1 == hash & !1
            && table
                .hashed_names
                .get(at)
                .is_some_and(|found| found == name)
        {
            return Some(sym);
        }
        if chain & 1 != 0 {
            return None;
        }
        sym = sym.checked_add(1)?;
    }
}

/// The dynamic-executable path: a `-fPIE` main linked against a shared object
/// by xold carries `.gnu.hash` + `DT_GNU_HASH` and runs under the system
/// `ld.so` (the loader resolves the imported call through the PLT).
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn dynamic_executable_carries_gnu_hash_and_runs() {
    let Some(_) = which("clang") else {
        eprintln!("skipping dynexec gnu.hash test: clang not found");
        return;
    };
    let Some(interp) = interpreter() else {
        eprintln!("skipping dynexec gnu.hash test: interpreter path unknown");
        return;
    };
    let dir = workdir("dynexec");
    let lib_obj = dir.join("lib.o");
    let lib_so = dir.join("libfoo.so");
    compile(SHARED_SRC, &lib_obj, true).expect("host clang compiles -fPIC");
    link_shared_with_xold(&lib_obj, &lib_so, b"libfoo.so");

    let main_o = dir.join("main.o");
    let start_o = dir.join("start.o");
    compile(MAIN_SRC, &main_o, false).expect("host clang compiles -fPIE");
    assemble(START_SRC, &start_o).expect("host clang assembles start.S");

    let prog = dir.join("prog");
    link_dyn_exec(
        &[main_o.clone(), start_o.clone(), lib_so.clone()],
        &prog,
        b"_start",
        &interp,
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dyn-exec link must succeed");

    // Structural: the executable carries the GNU hash tag and section.
    let bytes = fs::read(&prog).expect("read exe");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    assert!(
        obj.sections()
            .iter()
            .any(|s| obj.section_name(s) == b".gnu.hash"),
        "dynamic executable has .gnu.hash"
    );
    let tags = dt_tags(&prog);
    assert!(tags.contains(&DT_GNU_HASH), "DT_GNU_HASH present");

    // Runtime: run under the host loader. The imported `alpha` returns 6, so
    // main returns 0.
    let status = Command::new(&prog)
        .env("LD_LIBRARY_PATH", dir.to_str().unwrap_or("."))
        .status()
        .expect("run executable");
    assert!(
        status.success(),
        "dynamic executable runs under ld.so; got {status}"
    );
    let _ = (main_o, start_o, lib_so);
}

/// Compares the `.gnu.hash` shape xold emits against the system linker for the
/// same input: both decode, both cover the same exported name set, and the
/// symoffset semantics match (hashed tail = defined symbols).
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn gnu_hash_shape_matches_system_linker() {
    let Some(_) = which("clang") else {
        eprintln!("skipping system comparison: clang not found");
        return;
    };
    let dir = workdir("compare");
    let obj = dir.join("lib.o");
    let xold_so = dir.join("libgh_xold.so");
    let sys_so = dir.join("libgh_sys.so");
    compile(SHARED_SRC, &obj, true).expect("host clang compiles -fPIC");
    link_shared_with_xold(&obj, &xold_so, b"libgh.so");

    let sys_ok = Command::new(which("clang").unwrap())
        .args([
            "-shared",
            "-nostdlib",
            "-nodefaultlibs",
            "-Wl,-soname,libgh.so",
        ])
        .arg(&obj)
        .arg("-o")
        .arg(&sys_so)
        .status()
        .is_ok_and(|s| s.success());
    if !sys_ok {
        eprintln!("skipping system comparison: system linker unavailable");
        return;
    }

    let xold_gh = decode_gnu_hash(&xold_so).expect("xold .gnu.hash decodes");
    let sys_gh = decode_gnu_hash(&sys_so).expect("system .gnu.hash decodes");
    // Both must cover the same exported name set (the hashed tail). Undefined
    // imports are excluded; only defined exports are hashed.
    let xold_set: Vec<&Vec<u8>> = xold_gh.hashed_names.iter().collect();
    let sys_set: Vec<&Vec<u8>> = sys_gh.hashed_names.iter().collect();
    for name in &xold_set {
        assert!(
            sys_set.contains(name),
            "system .gnu.hash also hashes {:?}",
            String::from_utf8_lossy(name)
        );
    }
    assert_eq!(
        xold_gh.hashed_names.len(),
        sys_gh.hashed_names.len(),
        "same hashed symbol count"
    );
    // Bloom and bucket sizing must be power-of-two / load-factor sane in both.
    assert!(xold_gh.mask_words.is_power_of_two());
    assert!(sys_gh.mask_words.is_power_of_two());
    assert!(xold_gh.nbuckets >= 1 && sys_gh.nbuckets >= 1);
}

/// Probes the interpreter path from `/bin/true` so the test matches the host.
fn interpreter() -> Option<Vec<u8>> {
    let out = Command::new("readelf")
        .args(["-l", "/bin/true"])
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().find(|l| l.contains("interpreter:"))?;
    let start = line.find('/')?;
    let end = line.rfind(']').unwrap_or(line.len());
    Some(line.as_bytes()[start..end].to_vec())
}
