//! Mach-O reader: parses real darwin objects produced by cross clang and
//! checks the section, symbol and relocation views.
//!
//! Each test is gated on `clang --target=*-apple-darwin` being available; if
//! the cross target is absent the test prints a note and returns, so the build
//! never fails over a missing toolchain. The committed ELF fixture backs the
//! format-detection and reader-rejection checks that must always run.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use common::which;
use xold::{
    input::Format,
    macho::{
        MachOFile,
        constants::{
            CPU_SUBTYPE_ARM64_ALL, CPU_SUBTYPE_X86_64_ALL, CPU_TYPE_ARM64,
            CPU_TYPE_X86_64, MH_OBJECT, N_EXT, N_SECT, N_TYPE, N_UNDF,
        },
    },
    mmap_file::MappedFile,
};

mod common;

/// The C source compiled for the darwin targets. `global` is a defined data
/// symbol, `global_ptr` is a data pointer to it (emits an `UNSIGNED` reloc in
/// `__data`), and `ext_var`/`ext_func` are undefined references resolved by
/// the linker.
const SRC: &[u8] = b"extern int ext_var;\n\
                    extern int ext_func(int);\n\
                    int global = 42;\n\
                    int *global_ptr = &global;\n\
                    static int local = 7;\n\
                    int call_ext(void) { return ext_func(ext_var); }\n\
                    int *get_global(void) { return &global; }\n\
                    int *get_local(void) { return &local; }\n";

/// Returns the clang binary if it can compile for `triple`, else `None`.
fn darwin_clang(triple: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let probe = std::env::temp_dir().join("xold_darwin_probe.o");
    let ok = Command::new(&clang)
        .args([triple, "-c", "-x", "c", "-", "-o"])
        .arg(&probe)
        .stdin(Stdio::null())
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&probe);
    ok.then_some(clang)
}

/// Compiles `SRC` for `triple` into `out`, returning whether it succeeded.
fn compile(clang: &Path, triple: &str, out: &Path) -> bool {
    let src = std::env::temp_dir().join("xold_darwin_src.c");
    let _ = fs::write(&src, SRC);
    Command::new(clang)
        .args([triple, "-c"])
        .arg(&src)
        .arg("-o")
        .arg(out)
        .status()
        .is_ok_and(|s| s.success())
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn parses_a_real_x86_64_darwin_object() {
    let Some(clang) = darwin_clang("--target=x86_64-apple-darwin") else {
        eprintln!(
            "skipping x86_64 Mach-O reader: clang darwin target not found"
        );
        return;
    };
    let path = std::env::temp_dir().join("xold_macho_x86_64.o");
    if !compile(&clang, "--target=x86_64-apple-darwin", &path) {
        eprintln!("skipping x86_64 Mach-O reader: clang compile failed");
        return;
    }
    let mapped = MappedFile::open(&path).expect("open darwin object");
    let obj = MachOFile::parse(mapped.bytes()).expect("parse darwin object");

    assert!(obj.is_object());
    assert_eq!(obj.filetype(), MH_OBJECT);
    assert!(obj.is_x86_64());
    assert_eq!(obj.cpu_type(), CPU_TYPE_X86_64);
    assert_eq!(obj.cpu_subtype(), CPU_SUBTYPE_X86_64_ALL);
    assert!(!obj.is_arm64());

    let sections = obj.sections();
    let text = find_section(&sections, b"__text").expect("__text section");
    assert_eq!(text.segname, b"__TEXT");
    assert_ne!(text.data.len(), 0);
    let data = find_section(&sections, b"__data").expect("__data section");
    assert_eq!(data.segname, b"__DATA");
    assert_ne!(data.data.len(), 0);

    // __text carries an X86_64_RELOC_BRANCH (the call) and
    // X86_64_RELOC_GOT_LOAD (the movq of ext_var); __data carries
    // X86_64_RELOC_UNSIGNED (&global).
    assert!(text.relocations.iter().any(|r| r.r_type == 2)); // BRANCH
    assert!(text.relocations.iter().any(|r| r.r_type == 3)); // GOT_LOAD
    assert!(data.relocations.iter().any(|r| r.r_type == 0)); // UNSIGNED

    let syms = obj.symbols();
    assert_ne!(syms.len(), 0);
    assert_named(&syms, b"_global");
    assert_named(&syms, b"_ext_var");
    assert_named(&syms, b"_ext_func");
    // `_global` is an external, section-relative definition; `ext_var` is
    // an undefined external reference.
    let global = find_symbol(&syms, b"_global").expect("_global present");
    assert!(global.is_external());
    assert!(global.is_section());
    assert!(global.n_type & N_EXT != 0);
    assert!(global.n_type & N_SECT != 0);
    let ext = find_symbol(&syms, b"_ext_var").expect("_ext_var present");
    assert!(ext.is_undefined());
    // `N_UNDF` is 0, so the undefined check is `n_type & N_TYPE == N_UNDF`.
    assert_eq!(ext.n_type & N_TYPE, N_UNDF);
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn parses_a_real_arm64_darwin_object() {
    let Some(clang) = darwin_clang("--target=arm64-apple-darwin") else {
        eprintln!(
            "skipping arm64 Mach-O reader: clang darwin target not found"
        );
        return;
    };
    let path = std::env::temp_dir().join("xold_macho_arm64.o");
    if !compile(&clang, "--target=arm64-apple-darwin", &path) {
        eprintln!("skipping arm64 Mach-O reader: clang compile failed");
        return;
    }
    let mapped = MappedFile::open(&path).expect("open darwin object");
    let obj = MachOFile::parse(mapped.bytes()).expect("parse darwin object");

    assert!(obj.is_object());
    assert!(obj.is_arm64());
    assert_eq!(obj.cpu_type(), CPU_TYPE_ARM64);
    assert_eq!(obj.cpu_subtype(), CPU_SUBTYPE_ARM64_ALL);
    assert!(!obj.is_x86_64());

    let sections = obj.sections();
    let text = find_section(&sections, b"__text").expect("__text section");
    // __text carries an ARM64_RELOC_BRANCH26 (the call), ARM64_RELOC_PAGE21 /
    // PAGEOFF12 pairs (for global/local), and the GOT_LOAD_PAGE21 /
    // GOT_LOAD_PAGEOFF12 pair (for ext_var).
    assert!(text.relocations.iter().any(|r| r.r_type == 2)); // BRANCH26
    assert!(text.relocations.iter().any(|r| r.r_type == 3)); // PAGE21
    assert!(text.relocations.iter().any(|r| r.r_type == 4)); // PAGEOFF12
    assert!(text.relocations.iter().any(|r| r.r_type == 5)); // GOT_LOAD_PAGE21
    assert!(text.relocations.iter().any(|r| r.r_type == 6)); // GOT_LOAD_PAGEOFF12

    // Section ordinals are 1-based and match n_sect.
    for (i, s) in sections.iter().enumerate() {
        assert_eq!(s.index, u32::try_from(i + 1).unwrap());
    }

    let syms = obj.symbols();
    assert_named(&syms, b"_global");
    assert_named(&syms, b"_ext_func");
    let global = find_symbol(&syms, b"_global").expect("_global present");
    assert!(global.is_section());
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn detects_elf_and_macho_format_by_magic() {
    // A committed ELF fixture lets this check run without clang.
    let elf = MappedFile::open(&fixture("min.o")).expect("open ELF fixture");
    assert_eq!(Format::detect(elf.bytes()), Some(Format::Elf));

    // A synthesised little-endian 64-bit Mach-O magic.
    let macho = [0xcf, 0xfa, 0xed, 0xfe, 0, 0, 0, 0];
    assert_eq!(Format::detect(&macho), Some(Format::MachO));
    // Fat container magic is recognised as the Mach-O family.
    let fat = [0xca, 0xfe, 0xba, 0xbe, 0, 0, 0, 0];
    assert_eq!(Format::detect(&fat), Some(Format::MachO));
    assert_eq!(Format::detect(&[0x00, 0x01, 0x02, 0x03]), None);
    assert_eq!(Format::detect(&[0x7f]), None);
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn reader_rejects_non_macho_and_non_64bit() {
    let elf = MappedFile::open(&fixture("min.o")).expect("open ELF fixture");
    assert!(MachOFile::parse(elf.bytes()).is_err());
    // A 32-bit Mach-O magic is rejected with a precise error.
    let macho32 = [0xce, 0xfa, 0xed, 0xfe, 0, 0, 0, 0];
    assert!(reject_reason(&macho32).contains("32-bit"));
    // A fat container is rejected by name. Its header is big-endian on disk
    // whatever it holds, so these are the bytes a real one starts with, and
    // the reader has to recognise the swapped spelling to say so. Asserting
    // only that the parse fails would pass on the generic "not a Mach-O"
    // answer and miss that.
    let fat = [0xca, 0xfe, 0xba, 0xbe, 0, 0, 0, 0];
    assert!(
        reject_reason(&fat).contains("fat"),
        "got: {}",
        reject_reason(&fat)
    );
    let fat64 = [0xca, 0xfe, 0xba, 0xbf, 0, 0, 0, 0];
    assert!(reject_reason(&fat64).contains("fat"));
    // Something that is no Mach-O at all still gets the generic answer.
    let junk = [0x11, 0x22, 0x33, 0x44, 0, 0, 0, 0];
    assert!(reject_reason(&junk).contains("not a Mach-O"));
}

/// The rendered error `MachOFile::parse` rejects `bytes` with.
fn reject_reason(bytes: &[u8]) -> String {
    match MachOFile::parse(bytes) {
        Ok(_) => panic!("these bytes must not parse as a Mach-O object"),
        Err(e) => e.to_string(),
    }
}

// --- helpers --------------------------------------------------------------

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    p
}

fn find_section<'d>(
    sections: &'d [xold::macho::MachSection<'d>],
    name: &[u8],
) -> Option<&'d xold::macho::MachSection<'d>> {
    sections.iter().find(|s| s.sectname == name)
}

fn find_symbol<'d>(
    syms: &xold::macho::MachSymbolTable<'d>,
    name: &[u8],
) -> Option<xold::macho::MachSymbol<'d>> {
    syms.iter().find(|s| s.name == name)
}

fn assert_named(syms: &xold::macho::MachSymbolTable<'_>, name: &[u8]) {
    assert!(
        find_symbol(syms, name).is_some(),
        "expected symbol {} in symtab",
        String::from_utf8_lossy(name)
    );
}
