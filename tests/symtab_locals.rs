//! The file-scoped symbols the output symbol table carries.
//!
//! Symbol resolution interns only global and weak names, because a local is
//! file-scoped and there is nothing to merge. The output table was built from
//! that same set, so every `static` function and object in the program was
//! nameless: `gdb`, `perf`, `addr2line -f` and every backtrace symbolizer read
//! the table and found nothing between the globals. DWARF survived the link
//! all along, which is why this hid -- a `-g` build still symbolized, and a
//! stripped-of-debug-info release build did not.
//!
//! The rows are copied out of each input's own symbol table at write time,
//! following lld's `demoteAndCopyLocalSymbols`/`shouldKeepInSymtab`. What that
//! rule keeps and drops is what these tests pin:
//!
//! - **Present and correct**: a `static` function reaches `.symtab` as
//!   `STT_FUNC`/`STB_LOCAL`, sized, in the right output section, and at an
//!   address whose bytes really are that function's.
//! - **`sh_info`**: it names the first non-local entry. Off by one and every
//!   reader mis-labels a binding, so it is checked against the table's own
//!   contents rather than against a count kept alongside.
//! - **Dropped where they must be**: a local in a section `--gc-sections`
//!   collected, and the section symbols an input carries.
//! - **Ordering**: file order, then input symbol order, which is what keeps the
//!   table a function of the command line rather than of the schedule.
//!
//! Gated on `clang`, `gcc` (to locate the crt objects) and the system `ld.so`;
//! if absent the tests print a note and return, so the build never fails over
//! a missing toolchain.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{
    elf::{
        ObjectFile, Sym64,
        constants::{STB_LOCAL, STT_FILE, STT_FUNC, STT_SECTION},
    },
    icf::IcfMode,
    linker::link_dyn_exec,
};

mod common;

/// The fixture: two statics `main` reaches and one nothing reaches.
///
/// `-ffunction-sections` gives each its own input section, so `--gc-sections`
/// can take `orphan` alone and the surviving functions can be compared byte
/// for byte against their input sections.
const SRC: &[u8] = b"#include <stdio.h>\n\
     static char buf[8];\n\
     static __attribute__((noinline)) int helper(int x) { return x * 3 + 1; }\n\
     static __attribute__((used, noinline)) int orphan(void) { return 0x5b; }\n\
     __attribute__((constructor)) static void init(void) { buf[0] = 'o'; }\n\
     int main(void) { printf(\"%c %d\\n\", buf[0], helper(2)); return 0; }\n";

/// A second translation unit with a static of the same name, so the table has
/// to carry both and keep them in file order.
const SRC2: &[u8] = b"static __attribute__((noinline)) int helper(int x)\n\
     { return x - 1; }\n\
     int other(int x) { return helper(x); }\n";

/// A `static` function must reach `.symtab` with the right binding, type,
/// size, section and address.
///
/// The address is checked against the bytes it points at: the function has its
/// own input section under `-ffunction-sections`, so the image must hold that
/// section's exact content there. A row with a plausible-looking address that
/// names the wrong bytes is the failure this catches.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_static_function_reaches_the_symbol_table() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("static");
    let obj = dir.join("mainu.o");
    let Some(()) = compile(SRC, &obj) else {
        return;
    };
    let prog = dir.join("prog");
    h.link(std::slice::from_ref(&obj), &prog, false);
    let bytes = fs::read(&prog).expect("read output");

    let row = local(&bytes, b"helper").expect("the static must be named");
    assert_eq!(row.st_info >> 4, STB_LOCAL, "a static binds locally");
    assert_eq!(row.type_(), STT_FUNC, "and it is a function");
    assert_ne!(row.st_size.get(), 0, "its size comes from the input");
    assert_eq!(
        section_name_of(&bytes, row.st_shndx.get()),
        b".text".to_vec(),
        "it lives in the output .text"
    );
    let want = input_section_bytes(&obj, b".text.helper")
        .expect("the input carries the function in its own section");
    assert_eq!(
        image_bytes_at(&bytes, row.st_value.get(), want.len()),
        Some(want),
        "the address must name the function's own bytes"
    );

    // The program still runs: 'o' from the constructor, 7 from helper(2).
    let out = Command::new(&prog).output().expect("program runs");
    assert!(out.status.success(), "the program must exit 0");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "o 7\n");
    let _ = fs::remove_dir_all(&dir);
}

/// `sh_info` must be the index of the first non-local entry.
///
/// Checked against the table itself: every row below the boundary binds
/// locally and every row at or above it does not. An off-by-one either way
/// leaves a reader labelling one row wrongly, and is exactly what a count kept
/// beside the table drifts into.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn sh_info_names_the_first_global() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("shinfo");
    let obj = dir.join("mainu.o");
    let Some(()) = compile(SRC, &obj) else {
        return;
    };
    let prog = dir.join("prog");
    h.link(&[obj], &prog, false);
    let bytes = fs::read(&prog).expect("read output");

    let (syms, first_global) = symtab(&bytes).expect("the image has a symtab");
    assert!(first_global > 1, "the locals block must not be empty");
    assert!(
        first_global as usize <= syms.len(),
        "sh_info must be inside the table"
    );
    for (i, sym) in syms.iter().enumerate() {
        let is_local = sym.st_info >> 4 == STB_LOCAL;
        assert_eq!(
            is_local,
            i < first_global as usize,
            "row {i} of {} is on the wrong side of sh_info {first_global}",
            syms.len()
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

/// A local whose section `--gc-sections` collected must not be named, and no
/// section symbol may be copied out of an input.
///
/// A row naming collected bytes points into whatever the sweep left behind,
/// which is worse than no row at all; lld drops the same symbols by demoting
/// anything whose section is not live. `STT_SECTION` rows are dropped for a
/// different reason: they name an input section index that the output does not
/// have, and lld emits its own only under `--emit-relocs`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn collected_and_section_locals_are_dropped() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("gc");
    let obj = dir.join("mainu.o");
    let Some(()) = compile(SRC, &obj) else {
        return;
    };
    let plain = dir.join("plain");
    let collected = dir.join("collected");
    h.link(std::slice::from_ref(&obj), &plain, false);
    h.link(&[obj], &collected, true);

    let plain_bytes = fs::read(&plain).expect("read output");
    let gc_bytes = fs::read(&collected).expect("read gc output");
    assert!(
        local(&plain_bytes, b"orphan").is_some(),
        "the unreferenced static is named in a link that keeps it"
    );
    assert!(
        local(&gc_bytes, b"orphan").is_none(),
        "and must not be named once its section was collected"
    );
    assert!(
        local(&gc_bytes, b"helper").is_some(),
        "the live static survives the same link"
    );

    let (syms, _) = symtab(&plain_bytes).expect("the image has a symtab");
    assert!(
        !syms.iter().any(|s| s.type_() == STT_SECTION),
        "no STT_SECTION row may be copied out of an input"
    );
    // The input's `STT_FILE` row is kept, which is what makes `nm` group the
    // locals by translation unit; lld keeps it too.
    assert!(
        syms.iter().any(|s| s.type_() == STT_FILE),
        "the input's STT_FILE row is carried through"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Two files defining the same local name both reach the table, in the order
/// the files were given, and swapping the inputs swaps the rows.
///
/// The block is built per input file in parallel, so this is what proves the
/// concatenation follows the command line rather than whichever file finished
/// first.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn locals_follow_input_order() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("order");
    let a = dir.join("a.o");
    let b = dir.join("b.o");
    let Some(()) = compile(SRC, &a) else {
        return;
    };
    let Some(()) = compile(SRC2, &b) else {
        return;
    };

    let forward = dir.join("forward");
    let reverse = dir.join("reverse");
    h.link(&[a.clone(), b.clone()], &forward, false);
    h.link(&[b, a], &reverse, false);

    let names = |path: &Path| -> Vec<Vec<u8>> {
        let bytes = fs::read(path).expect("read output");
        let (syms, first_global) = symtab(&bytes).expect("symtab");
        syms.iter()
            .take(first_global as usize)
            .filter(|s| s.type_() == STT_FILE)
            .map(|s| name_of(&bytes, s.st_name.get()))
            .collect()
    };
    let fwd = names(&forward);
    let mut rev = names(&reverse);
    assert_eq!(fwd.len(), 2, "one STT_FILE row per input, got {fwd:?}");
    rev.reverse();
    assert_eq!(
        fwd, rev,
        "the locals of each file must appear in the order the files were given"
    );

    // Both `helper` statics are present: they are distinct symbols that only
    // share a spelling.
    let bytes = fs::read(&forward).expect("read output");
    let (syms, _) = symtab(&bytes).expect("symtab");
    let helpers = syms
        .iter()
        .filter(|s| {
            s.type_() == STT_FUNC
                && name_of(&bytes, s.st_name.get()) == b"helper"
        })
        .count();
    assert_eq!(helpers, 2, "each file's own static keeps its own row");
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// A fresh per-test working directory under the system temp dir.
fn workdir(prefix: &str) -> PathBuf {
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("xold_locals_{prefix}_{pid}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Compiles `src` into `obj` with per-symbol sections, or `None` (with a note)
/// when clang is unavailable.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args([
            "--target=x86_64-linux-gnu",
            "-c",
            "-O0",
            "-fno-pie",
            "-ffunction-sections",
            "-fdata-sections",
        ])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// The host toolchain these tests need.
struct Harness {
    crt1: PathBuf,
    crti: PathBuf,
    crtn: PathBuf,
    libc: PathBuf,
    interp: Vec<u8>,
}

impl Harness {
    /// Collects the harness, or returns `None` when a piece is missing.
    fn detect() -> Option<Self> {
        if which("clang").is_none() {
            eprintln!("skipping symtab-locals tests: clang unavailable");
            return None;
        }
        Some(Self {
            crt1: crt_file("crt1.o")?,
            crti: crt_file("crti.o")?,
            crtn: crt_file("crtn.o")?,
            libc: libc_so()?,
            interp: interpreter()?,
        })
    }

    /// Links `objs` (in the order given) plus the crt objects and libc.
    fn link(&self, objs: &[PathBuf], prog: &Path, gc: bool) {
        let mut inputs = objs.to_vec();
        inputs.extend([
            self.crti.clone(),
            self.crt1.clone(),
            self.crtn.clone(),
            self.libc.clone(),
        ]);
        link_dyn_exec(
            &inputs,
            prog,
            b"_start",
            &self.interp,
            gc,
            IcfMode::None,
            false,
        )
        .expect("xold link must succeed");
    }
}

// --- readers ---------------------------------------------------------------

/// The image's `.symtab` rows and its `sh_info`.
fn symtab(bytes: &[u8]) -> Option<(Vec<Sym64>, u32)> {
    let obj = ObjectFile::parse(bytes).expect("output must be valid ELF");
    let sec = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".symtab")?;
    let data = obj.section_data(sec).ok()?;
    let syms = bytemuck::try_cast_slice::<u8, Sym64>(data).ok()?.to_vec();
    Some((syms, sec.sh_info.get()))
}

/// The local row named `name`, or `None` when the table has none.
fn local(bytes: &[u8], name: &[u8]) -> Option<Sym64> {
    let (syms, first_global) = symtab(bytes)?;
    syms.iter()
        .take(first_global as usize)
        .find(|s| name_of(bytes, s.st_name.get()) == name)
        .copied()
}

/// The `.strtab` string at `offset`.
fn name_of(bytes: &[u8], offset: u32) -> Vec<u8> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(sec) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".strtab")
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(sec) else {
        return Vec::new();
    };
    let start = offset as usize;
    let Some(tail) = data.get(start..) else {
        return Vec::new();
    };
    let end = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
    tail[..end].to_vec()
}

/// The name of the output section at `shndx`.
fn section_name_of(bytes: &[u8], shndx: u16) -> Vec<u8> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    obj.sections()
        .get(usize::from(shndx))
        .map(|s| obj.section_name(s).to_vec())
        .unwrap_or_default()
}

/// `len` bytes of the image at virtual address `addr`, found through the
/// section that covers it.
fn image_bytes_at(bytes: &[u8], addr: u64, len: usize) -> Option<Vec<u8>> {
    let obj = ObjectFile::parse(bytes).ok()?;
    for sec in obj.sections() {
        let base = sec.sh_addr.get();
        if base == 0 || addr < base || addr >= base + sec.sh_size.get() {
            continue;
        }
        let data = obj.section_data(sec).ok()?;
        let at = usize::try_from(addr - base).ok()?;
        return data.get(at..at.checked_add(len)?).map(<[u8]>::to_vec);
    }
    None
}

/// The content of one named section of an input object.
fn input_section_bytes(path: &Path, name: &[u8]) -> Option<Vec<u8>> {
    let bytes = fs::read(path).ok()?;
    let obj = ObjectFile::parse(&bytes).ok()?;
    let sec = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == name)?;
    obj.section_data(sec).ok().map(<[u8]>::to_vec)
}
