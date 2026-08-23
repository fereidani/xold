//! PLT allocation order is a function of the input order.
//!
//! The relocation scan runs per input file in parallel and folds the per-file
//! results back together serially. That fold is what fixes the PLT entry
//! order, and the PLT entry order fixes `.rela.plt`, `.got.plt` and every stub
//! address in the image. Deduplicating with a hash container is fine -- it
//! decides only whether two references name the same import -- but the order
//! that survives has to be the order the serial walk saw, never the order a
//! table happened to iterate in.
//!
//! Each unit below references exactly one import of its own plus one import
//! all three share. The private imports therefore appear in `.rela.plt` in
//! file order whatever the compiler does inside a unit, and reversing the
//! input order must reverse them; the shared import must appear exactly once.
//!
//! Gated on `clang`; if it is missing the tests print a note and return, so
//! the build never fails over a missing toolchain.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::{ObjectFile, constants::SHF_INFO_LINK},
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
};

mod common;

/// The library every unit imports from: three functions used by one unit
/// apiece, and one used by all three.
const LIB_SRC: &str = "int p(void){return 1;}\nint q(void){return 2;}\n\
    int r(void){return 3;}\nint z(void){return 4;}\n";

/// The three units, each calling its own import and the shared one.
const UNIT_SRC: [(&str, &str); 3] = [
    (
        "ua",
        "int p(void);int z(void);int ua(void){return p()+z();}\n",
    ),
    (
        "ub",
        "int q(void);int z(void);int ub(void){return q()+z();}\n",
    ),
    (
        "uc",
        "int r(void);int z(void);int uc(void){return r()+z();}\n",
    ),
];

/// The entry unit. It calls the three units, which the link defines, so it
/// contributes no import of its own and cannot disturb the order under test.
const MAIN_SRC: &str = "int ua(void);int ub(void);int uc(void);\n\
    int _start(void){return ua()+ub()+uc();}\n";

/// A private working directory, named after the process so concurrent test
/// binaries cannot delete each other's files.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_plt_order_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Compiles `src` to `dir/name.o`, returning `None` without a host compiler.
fn compile(src: &str, name: &str, dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let source = dir.join(format!("{name}.c"));
    let obj = dir.join(format!("{name}.o"));
    fs::write(&source, src).ok()?;
    let ok = Command::new(clang)
        .args(["-fPIC", "-O1", "-c"])
        .arg(&source)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    ok.then_some(obj)
}

/// Builds the shared library and the four objects, or `None` without clang.
fn build(dir: &Path) -> Option<(PathBuf, PathBuf, Vec<PathBuf>)> {
    let lib_obj = compile(LIB_SRC, "plt_order_lib", dir)?;
    let lib = dir.join("libpltorder.so");
    link_shared(
        &[lib_obj],
        &lib,
        Some(b"libpltorder.so"),
        false,
        IcfMode::None,
        false,
    )
    .ok()?;
    let main = compile(MAIN_SRC, "plt_order_main", dir)?;
    let mut units = Vec::with_capacity(UNIT_SRC.len());
    for (name, src) in UNIT_SRC {
        units.push(compile(src, name, dir)?);
    }
    Some((lib, main, units))
}

/// The imported names of `.rela.plt`, in entry order.
fn plt_names(path: &Path) -> Vec<Vec<u8>> {
    let bytes = fs::read(path).expect("read the linked image");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let section = |want: &[u8]| {
        obj.sections()
            .iter()
            .find(|s| obj.section_name(s) == want)
            .and_then(|s| obj.section_data(s).ok())
    };
    let rela = section(b".rela.plt").expect("has .rela.plt");
    let dynsym = section(b".dynsym").expect("has .dynsym");
    let dynstr = section(b".dynstr").expect("has .dynstr");
    let mut out = Vec::new();
    for entry in rela.chunks(24) {
        if entry.len() < 24 {
            break;
        }
        let info =
            u64::from_le_bytes(entry[8..16].try_into().unwrap_or([0; 8]));
        let sym = usize::try_from(info >> 32).unwrap_or(0);
        let Some(row) = dynsym.get(sym * 24..sym * 24 + 24) else {
            continue;
        };
        let at = u32::from_le_bytes(row[..4].try_into().unwrap_or([0; 4]));
        let at = usize::try_from(at).unwrap_or(0);
        let Some(tail) = dynstr.get(at..) else {
            continue;
        };
        let end = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
        out.push(tail[..end].to_vec());
    }
    out
}

/// Links `inputs` into a dynamic executable and reports its PLT order.
///
/// The interpreter path is only the `.interp` string here: the image is
/// parsed, never run, so the test needs no host loader.
fn link_order(dir: &Path, tag: &str, inputs: &[PathBuf]) -> Vec<Vec<u8>> {
    let prog = dir.join(format!("prog_{tag}"));
    link_dyn_exec(
        inputs,
        &prog,
        b"_start",
        b"/lib64/ld-linux-x86-64.so.2",
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-exec link must succeed");
    plt_names(&prog)
}

/// The private imports must appear in the order their files were given, and
/// the shared import exactly once.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn plt_entries_follow_the_input_file_order() {
    let dir = workdir("forward");
    let Some((lib, main, units)) = build(&dir) else {
        eprintln!("skipping PLT order test: host clang unavailable");
        return;
    };
    let mut inputs = vec![main];
    inputs.extend(units);
    inputs.push(lib);
    let names = link_order(&dir, "forward", &inputs);

    let private: Vec<&[u8]> = names
        .iter()
        .map(Vec::as_slice)
        .filter(|n| matches!(*n, b"p" | b"q" | b"r"))
        .collect();
    assert_eq!(
        private,
        vec![&b"p"[..], &b"q"[..], &b"r"[..]],
        "PLT entries must follow first-seen input order"
    );
    assert_eq!(
        names.iter().filter(|n| n.as_slice() == b"z").count(),
        1,
        "an import referenced by three files takes one PLT entry"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `.rela.plt` names the section its entries patch.
///
/// A relocation section's `sh_info` is the index of the section it applies to,
/// and `SHF_INFO_LINK` says that the number is a section index rather than a
/// count. `.rela.plt` patches `.got.plt`, one entry per slot, so the pair is
/// what tells a reader which storage those `JUMP_SLOT` entries fill. lld sets
/// both in `RelocationBaseSection::finalizeContents`
/// (`lld/ELF/SyntheticSections.cpp`); xold left `sh_info` at zero,
/// which reads as "applies to the null section".
///
/// `.rela.dyn` is checked for the opposite: lld sets neither on it, because
/// its entries reach wherever their targets live and name no one section.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn rela_plt_points_at_the_got_plt_it_patches() {
    let dir = workdir("infolink");
    let Some((lib, main, units)) = build(&dir) else {
        eprintln!("skipping .rela.plt sh_info test: host clang unavailable");
        return;
    };
    let mut inputs = vec![main];
    inputs.extend(units);
    inputs.push(lib);
    let prog = dir.join("prog_infolink");
    link_dyn_exec(
        &inputs,
        &prog,
        b"_start",
        b"/lib64/ld-linux-x86-64.so.2",
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-exec link must succeed");
    let bytes = fs::read(&prog).expect("read the linked image");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");

    let got_plt = obj
        .sections()
        .iter()
        .position(|s| obj.section_name(s) == b".got.plt")
        .expect("a link with PLT entries has a .got.plt");
    let rela_plt = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".rela.plt")
        .expect("and a .rela.plt beside it");
    assert_eq!(
        rela_plt.sh_info.get() as usize,
        got_plt,
        ".rela.plt's sh_info must name the .got.plt its entries patch"
    );
    assert_ne!(
        rela_plt.sh_flags.get() & SHF_INFO_LINK,
        0,
        "and SHF_INFO_LINK must say that sh_info is a section index"
    );

    // `.rela.dyn` is emitted only when something needs it, so this half is
    // checked when it is there.
    if let Some(rela_dyn) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".rela.dyn")
    {
        assert_eq!(
            (
                rela_dyn.sh_info.get(),
                rela_dyn.sh_flags.get() & SHF_INFO_LINK
            ),
            (0, 0),
            ".rela.dyn applies to no single section, so it names none"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

/// Reversing the input order must reverse the entries. Without this the first
/// test would still pass on a linker that sorted the names.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn reversing_the_inputs_reverses_the_plt() {
    let dir = workdir("reverse");
    let Some((lib, main, mut units)) = build(&dir) else {
        eprintln!("skipping PLT order test: host clang unavailable");
        return;
    };
    units.reverse();
    let mut inputs = vec![main];
    inputs.extend(units);
    inputs.push(lib);
    let names = link_order(&dir, "reverse", &inputs);

    let private: Vec<&[u8]> = names
        .iter()
        .map(Vec::as_slice)
        .filter(|n| matches!(*n, b"p" | b"q" | b"r"))
        .collect();
    assert_eq!(
        private,
        vec![&b"r"[..], &b"q"[..], &b"p"[..]],
        "PLT order must derive from the input order and nothing else"
    );
    let _ = fs::remove_dir_all(&dir);
}
