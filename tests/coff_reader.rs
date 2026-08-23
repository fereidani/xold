//! COFF reader: parses real Windows objects produced by cross clang and
//! checks the section, symbol and relocation views.
//!
//! Each test is gated on `clang --target=*-pc-windows-*` being available; if
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
    coff::{
        CoffFile, SymbolAux,
        constants::{
            IMAGE_FILE_MACHINE_AMD64, IMAGE_FILE_MACHINE_I386,
            IMAGE_SCN_CNT_CODE, IMAGE_SCN_CNT_INITIALIZED_DATA,
            IMAGE_SCN_CNT_UNINITIALIZED_DATA, IMAGE_SCN_MEM_EXECUTE,
            IMAGE_SYM_CLASS_EXTERNAL, IMAGE_SYM_CLASS_STATIC,
        },
    },
    input::Format,
    mmap_file::MappedFile,
};

mod common;

/// The C source compiled for the Windows targets. `global` is a defined data
/// symbol, `get_global` returns its address (emits an `ADDR32`/`ADDR64` reloc
/// in `.data`), and `ext_var`/`ext_func` are undefined references resolved by
/// the linker (emit `REL32` relocs in `.text`).
const SRC: &[u8] = b"extern int ext_var;\n\
                    extern int ext_func(int);\n\
                    int global = 42;\n\
                    int *get_global(void) { return &global; }\n\
                    int call_ext(void) { return ext_func(ext_var); }\n";

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn parses_a_real_x86_64_msvc_object() {
    let Some(clang) = windows_clang("--target=x86_64-pc-windows-msvc") else {
        eprintln!("skipping x86_64 COFF reader: clang msvc target not found");
        return;
    };
    let path = std::env::temp_dir().join("xold_coff_x86_64_msvc.o");
    if !compile(&clang, "--target=x86_64-pc-windows-msvc", &path) {
        eprintln!("skipping x86_64 COFF reader: clang compile failed");
        return;
    }
    let mapped = MappedFile::open(&path).expect("open COFF object");
    let obj = CoffFile::parse(mapped.bytes()).expect("parse COFF object");

    assert_eq!(obj.machine(), IMAGE_FILE_MACHINE_AMD64);
    assert!(obj.is_x86_64());
    assert!(!obj.is_i386());

    let sections = obj.sections();
    assert_eq!(
        sections.len(),
        usize::try_from(obj.number_of_sections()).unwrap()
    );

    let text = find_section(sections, b".text").expect(".text section");
    assert_ne!(text.characteristics & IMAGE_SCN_CNT_CODE, 0);
    assert_ne!(text.characteristics & IMAGE_SCN_MEM_EXECUTE, 0);
    assert_ne!(text.data.len(), 0);
    let data = find_section(sections, b".data").expect(".data section");
    assert_ne!(data.characteristics & IMAGE_SCN_CNT_INITIALIZED_DATA, 0);
    let bss = find_section(sections, b".bss").expect(".bss section");
    assert_ne!(bss.characteristics & IMAGE_SCN_CNT_UNINITIALIZED_DATA, 0);
    assert_eq!(bss.data, []);

    // `.text` references `ext_func` (a call) and `ext_var`: at least one
    // PC-relative relocation is emitted.
    assert!(
        !text.relocations.is_empty(),
        ".text should carry relocations"
    );

    // Section ordinals are 1-based and match the symbol-table section numbers.
    for (i, s) in sections.iter().enumerate() {
        assert_eq!(s.index, u32::try_from(i + 1).unwrap());
    }

    let syms = obj.symbols();
    assert_ne!(syms.len(), 0);
    assert_named(syms, b"global");
    assert_named(syms, b"ext_var");
    assert_named(syms, b"ext_func");

    let global = find_symbol(syms, b"global").expect("global present");
    assert_eq!(global.storage_class, IMAGE_SYM_CLASS_EXTERNAL);
    assert!(global.section_number > 0);
    let ext = find_symbol(syms, b"ext_var").expect("ext_var present");
    assert!(ext.is_undefined());

    // Section symbols carry an auxiliary section record with COMDAT metadata.
    let text_sym = find_symbol(syms, b".text").expect(".text section symbol");
    assert_eq!(text_sym.storage_class, IMAGE_SYM_CLASS_STATIC);
    assert!(matches!(text_sym.aux, SymbolAux::Section(_)));
    if let SymbolAux::Section(aux) = text_sym.aux {
        assert_eq!(aux.number, text.index);
        assert_eq!(aux.length, text.size_of_raw_data);
    }
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn parses_a_real_x86_64_gnu_object() {
    let Some(clang) = windows_clang("--target=x86_64-pc-windows-gnu") else {
        eprintln!("skipping x86_64 COFF reader: clang gnu target not found");
        return;
    };
    let path = std::env::temp_dir().join("xold_coff_x86_64_gnu.o");
    if !compile(&clang, "--target=x86_64-pc-windows-gnu", &path) {
        eprintln!("skipping x86_64 gnu COFF reader: clang compile failed");
        return;
    }
    let mapped = MappedFile::open(&path).expect("open COFF object");
    let obj = CoffFile::parse(mapped.bytes()).expect("parse COFF object");

    assert_eq!(obj.machine(), IMAGE_FILE_MACHINE_AMD64);
    assert!(obj.is_x86_64());
    let syms = obj.symbols();
    assert_named(syms, b"global");
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn parses_a_real_i386_msvc_object() {
    let Some(clang) = windows_clang("--target=i686-pc-windows-msvc") else {
        eprintln!("skipping i386 COFF reader: clang msvc target not found");
        return;
    };
    let path = std::env::temp_dir().join("xold_coff_i386_msvc.o");
    if !compile(&clang, "--target=i686-pc-windows-msvc", &path) {
        eprintln!("skipping i386 COFF reader: clang compile failed");
        return;
    }
    let mapped = MappedFile::open(&path).expect("open COFF object");
    let obj = CoffFile::parse(mapped.bytes()).expect("parse COFF object");

    assert_eq!(obj.machine(), IMAGE_FILE_MACHINE_I386);
    assert!(obj.is_i386());
    assert!(!obj.is_x86_64());
    let syms = obj.symbols();
    // The i386 Windows cdecl ABI prefixes C symbols with `_`.
    assert_named(syms, b"_global");
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn detects_coff_pe_elf_and_macho_format_by_magic() {
    // A committed ELF fixture lets this check run without clang.
    let elf = MappedFile::open(&fixture("min.o")).expect("open ELF fixture");
    assert_eq!(Format::detect(elf.bytes()), Some(Format::Elf));

    // A bare COFF x86_64 object: machine field 0x8664 little-endian.
    let coff = [0x64, 0x86, 0, 0, 0, 0, 0, 0];
    assert_eq!(Format::detect(&coff), Some(Format::Coff));
    // A bare COFF i386 object: machine field 0x014c little-endian.
    let coff32 = [0x4c, 0x01, 0, 0, 0, 0, 0, 0];
    assert_eq!(Format::detect(&coff32), Some(Format::Coff));
    // A PE image: `MZ` DOS stub.
    let pe = [b'M', b'Z', 0, 0, 0, 0, 0, 0];
    assert_eq!(Format::detect(&pe), Some(Format::Pe));
    // A synthesised little-endian 64-bit Mach-O magic.
    let macho = [0xcf, 0xfa, 0xed, 0xfe, 0, 0, 0, 0];
    assert_eq!(Format::detect(&macho), Some(Format::MachO));
    assert_eq!(Format::detect(&[0x00, 0x01, 0x02, 0x03]), None);
    assert_eq!(Format::detect(&[0x7f]), None);
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn reader_rejects_non_coff_inputs() {
    let elf = MappedFile::open(&fixture("min.o")).expect("open ELF fixture");
    assert!(CoffFile::parse(elf.bytes()).is_err());
    // A 64-bit Mach-O magic is rejected.
    let macho = [0xcf, 0xfa, 0xed, 0xfe, 0, 0, 0, 0];
    assert!(CoffFile::parse(&macho).is_err());
    // A PE image (`MZ`) is rejected: the reader targets bare COFF objects.
    let pe = [b'M', b'Z', 0, 0, 0, 0, 0, 0];
    assert!(CoffFile::parse(&pe).is_err());
    // Random bytes are rejected.
    assert!(CoffFile::parse(&[0x00, 0x01, 0x02, 0x03]).is_err());
}

// --- helpers --------------------------------------------------------------

/// A scratch path under the temp directory, named so two test binaries sharing
/// a `TMPDIR` cannot collide: each helper below writes its file and then
/// removes it, so a shared name lets one run delete what another is still
/// using.
fn scratch(name: &str) -> PathBuf {
    let pid = std::process::id();
    std::env::temp_dir().join(format!("xold_coff_{pid}_{name}"))
}

/// Returns the clang binary if it can compile for `triple`, else `None`.
fn windows_clang(triple: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let probe = scratch("probe.o");
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
    let src = scratch("src.c");
    let _ = fs::write(&src, SRC);
    Command::new(clang)
        .args([triple, "-c"])
        .arg(&src)
        .arg("-o")
        .arg(out)
        .status()
        .is_ok_and(|s| s.success())
}

fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    p
}

fn find_section<'d>(
    sections: &'d [xold::coff::CoffSection<'d>],
    name: &[u8],
) -> Option<&'d xold::coff::CoffSection<'d>> {
    sections.iter().find(|s| s.name == name)
}

fn find_symbol<'d>(
    syms: &xold::coff::CoffSymbolTable<'d>,
    name: &[u8],
) -> Option<xold::coff::CoffSymbol<'d>> {
    syms.iter().copied().find(|s| s.name == name)
}

fn assert_named(syms: &xold::coff::CoffSymbolTable<'_>, name: &[u8]) {
    assert!(
        find_symbol(syms, name).is_some(),
        "expected symbol {} in symtab",
        String::from_utf8_lossy(name)
    );
}
