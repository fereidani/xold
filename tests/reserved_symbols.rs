//! `etext`, `_etext` and `__dso_handle` resolve.
//!
//! A C runtime and its libc refer to all three: `atexit` and
//! `__cxa_atexit` pass `__dso_handle` so the exit code can tell which
//! module a handler belongs to, and `etext`/`_etext` mark the end of the
//! program text for profiling and for `mallopt` heuristics. None of them
//! is defined by any input; each is the linker's to supply, and a link
//! that does not fails with an undefined reference to a name every libc
//! knows. lld defines the pair against the last read-only segment and
//! `__dso_handle` against the ELF header, hidden
//! (`lld/ELF/Writer.cpp`).
//!
//! The tests link a program that references all three and read the rows
//! back out of `.symtab`: the pair names the end of the program text --
//! the last executable region, which is `.plt` in a dynamic image -- and
//! `__dso_handle` is a hidden symbol anchored at the image start.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::{ObjectFile, constants::SHF_EXECINSTR},
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared, link_to},
};

mod common;

const SRC: &str = r"
extern char etext[], _etext[], __dso_handle[];

void _start(void) { }
char *text_end(void) { return etext; }
char *text_end2(void) { return _etext; }
char *self(void) { return __dso_handle; }
";

/// The library the dynamic fixture imports from, which is what forces a
/// `.plt` into the image.
const LIB_SRC: &str = "int imported(void) { return 1; }\n";

/// The dynamic fixture: references the pair and calls an import, so the
/// image holds executable `.plt` bytes placed after `.text`.
const DYN_SRC: &str = r"
extern char etext[], _etext[];

int imported(void);
int _start(void) { return imported(); }
char *text_end(void) { return etext; }
char *text_end2(void) { return _etext; }
";

/// The program links, and `etext`/`_etext` name the end of the text.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn etext_marks_the_end_of_the_text() {
    let Some(dir) = workdir("etext") else {
        return;
    };
    let Some(obj) = compile(&dir) else {
        return;
    };
    let out = dir.join("prog");
    let res = link_to(&[obj], &out, b"_start", false, IcfMode::None, false);
    assert!(res.is_ok(), "the references must resolve: {:?}", res.err());
    let bytes = fs::read(&out).expect("read the image");
    let image = ObjectFile::parse(&bytes).expect("parse the image");
    let table = image.symbol_table().ok().flatten().expect(".symtab");
    let mut ends = Vec::new();
    for name in [&b"etext"[..], b"_etext"] {
        let sym = table
            .syms
            .iter()
            .find(|s| table.name(s) == name)
            .unwrap_or_else(|| {
                panic!("{} is defined", String::from_utf8_lossy(name))
            });
        assert_ne!(
            sym.st_shndx.get(),
            0,
            "{} names a place in the image",
            String::from_utf8_lossy(name)
        );
        let sect = image
            .sections()
            .get(usize::from(sym.st_shndx.get()))
            .expect("the named section exists");
        let end = sect.sh_addr.get() + sect.sh_size.get();
        assert_eq!(
            sym.st_value.get(),
            end,
            "{} is the end of the region it names",
            String::from_utf8_lossy(name)
        );
        ends.push(sym.st_value.get());
    }
    assert_eq!(ends[0], ends[1], "the two spellings are one bound");
    let _ = fs::remove_dir_all(&dir);
}

/// In an image with a PLT, `etext` covers every executable byte.
///
/// The placement order puts `.fini` and `.plt` after `.text`, so a bound
/// pinned to the text region's own end would leave executable bytes above
/// it. The pair must name the end of the last executable region: no
/// `SHF_EXECINSTR` section header may span an address above `etext`, and
/// the pair must still equal the end of the very section its `st_shndx`
/// names, so the address and the section index describe one place.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn etext_covers_every_executable_byte() {
    let Some(dir) = workdir("etext_plt") else {
        return;
    };
    let Some(prog) = build_dynamic(&dir) else {
        eprintln!("skipping reserved-symbols: cannot build the fixture");
        return;
    };
    let bytes = fs::read(&prog).expect("read the image");
    let image = ObjectFile::parse(&bytes).expect("parse the image");
    let table = image.symbol_table().ok().flatten().expect(".symtab");
    let code: Vec<(Vec<u8>, u64)> = image
        .sections()
        .iter()
        .filter(|s| s.sh_flags.get() & SHF_EXECINSTR != 0)
        .map(|s| {
            let name = image.section_name(s).to_vec();
            (name, s.sh_addr.get() + s.sh_size.get())
        })
        .collect();
    assert!(
        code.iter().any(|(name, _)| name == b".plt"),
        "the fixture must place a .plt for the bound to be exercised"
    );
    for name in [&b"etext"[..], b"_etext"] {
        let sym = table
            .syms
            .iter()
            .find(|s| table.name(s) == name)
            .unwrap_or_else(|| {
                panic!("{} is defined", String::from_utf8_lossy(name))
            });
        let sect = image
            .sections()
            .get(usize::from(sym.st_shndx.get()))
            .expect("the named section exists");
        assert_eq!(
            sym.st_value.get(),
            sect.sh_addr.get() + sect.sh_size.get(),
            "{} is the end of the region it names",
            String::from_utf8_lossy(name)
        );
        for (code_name, end) in &code {
            assert!(
                sym.st_value.get() >= *end,
                "{} at {:#x} leaves executable bytes of {} above it \
                 (section ends at {:#x})",
                String::from_utf8_lossy(name),
                sym.st_value.get(),
                String::from_utf8_lossy(code_name),
                end
            );
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

/// `__dso_handle` is hidden and anchored at the image start, so each
/// module's copy identifies the module the loader placed.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn dso_handle_is_hidden_at_the_image_start() {
    const HIDDEN: u8 = 2;
    let Some(dir) = workdir("dso") else {
        return;
    };
    let Some(obj) = compile(&dir) else {
        return;
    };
    let out = dir.join("prog");
    let res = link_to(&[obj], &out, b"_start", false, IcfMode::None, false);
    assert!(res.is_ok(), "the reference must resolve: {:?}", res.err());
    let bytes = fs::read(&out).expect("read the image");
    let image = ObjectFile::parse(&bytes).expect("parse the image");
    let table = image.symbol_table().ok().flatten().expect(".symtab");
    let sym = table
        .syms
        .iter()
        .find(|s| table.name(s) == b"__dso_handle")
        .expect("__dso_handle is defined");
    assert_eq!(sym.st_other, HIDDEN, "__dso_handle is private to the image");
    assert_ne!(
        sym.st_shndx.get(),
        0,
        "__dso_handle names a place in the image"
    );
    assert_eq!(
        sym.st_value.get(),
        first_load_vaddr(&bytes),
        "__dso_handle anchors at the start of the image"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping reserved-symbols {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_reserved_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// The lowest `PT_LOAD` virtual address, which is where the ELF header --
/// and so the image -- starts.
fn first_load_vaddr(bytes: &[u8]) -> u64 {
    let phoff = usize::try_from(read_u64(bytes, 0x20)).unwrap_or(0);
    let phnum = read_u16(bytes, 0x38);
    (0..phnum)
        .filter(|&i| {
            let at = phoff + i * read_u16(bytes, 0x36);
            read_u32(bytes, at) == 1
        })
        .map(|i| {
            let at = phoff + i * read_u16(bytes, 0x36);
            read_u64(bytes, at + 16)
        })
        .min()
        .unwrap_or(0)
}

fn read_u16(bytes: &[u8], at: usize) -> usize {
    bytes
        .get(at..at + 2)
        .and_then(|c| <[u8; 2]>::try_from(c).ok())
        .map_or(0, |c| usize::from(u16::from_le_bytes(c)))
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    bytes
        .get(at..at + 4)
        .and_then(|c| <[u8; 4]>::try_from(c).ok())
        .map_or(0, u32::from_le_bytes)
}

fn read_u64(bytes: &[u8], at: usize) -> u64 {
    bytes
        .get(at..at + 8)
        .and_then(|c| <[u8; 8]>::try_from(c).ok())
        .map_or(0, u64::from_le_bytes)
}

/// Compiles `src` position-independent to `dir/name.o`, or `None` without a
/// host compiler.
fn compile_pic(src: &str, name: &str, dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let source = dir.join(format!("{name}.c"));
    let obj = dir.join(format!("{name}.o"));
    fs::write(&source, src).ok()?;
    let built = Command::new(clang)
        .args(["-c", "-fPIC"])
        .arg("-o")
        .arg(&obj)
        .arg(&source)
        .status()
        .ok()?
        .success();
    built.then_some(obj)
}

/// Builds the shared library and links the dynamic fixture against it,
/// returning the linked program.
///
/// The interpreter path is only the `.interp` string: the image is parsed,
/// never run, so the test needs no host loader.
fn build_dynamic(dir: &Path) -> Option<PathBuf> {
    let lib_obj = compile_pic(LIB_SRC, "lib", dir)?;
    let lib = dir.join("libimported.so");
    let soname = Some(&b"libimported.so"[..]);
    link_shared(&[lib_obj], &lib, soname, false, IcfMode::None, false)
        .expect("xold shared link must succeed");
    let main = compile_pic(DYN_SRC, "main", dir)?;
    let prog = dir.join("prog");
    link_dyn_exec(
        &[main, lib],
        &prog,
        b"_start",
        b"/lib64/ld-linux-x86-64.so.2",
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-exec link must succeed");
    Some(prog)
}

/// Compiles the fixture to a static object.
fn compile(dir: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("prog.c");
    let obj = dir.join("prog.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args(["-c", "-fno-pie"])
        .arg("-o")
        .arg(&obj)
        .arg(&src)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping reserved-symbols: clang cannot build the fixture");
        return None;
    }
    Some(obj)
}
