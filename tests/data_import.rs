//! Imports an image reaches only through an absolute data relocation.
//!
//! `long *p = &imported;` in a `-fPIE` object is an `R_X86_64_64` in a
//! writable section and nothing else: no GOT slot, no PLT entry. `.dynsym`
//! import rows used to come from the copy slots and the GOT/PLT keys alone, so
//! such a reference reached no row, the relocation naming it was dropped, and
//! the slot the sizing pass had reserved stayed zeroed -- an `R_X86_64_NONE`
//! the loader skips, leaving a null pointer where the address belongs. The
//! system linker emits `R_X86_64_64 <sym>` for the same input.
//!
//! The export has to be `STT_NOTYPE` with no size for the case to bite: a
//! sized `STT_OBJECT` import is copy-relocated instead, and the copy slot
//! supplies the `.dynsym` row by another route.
//!
//! Gated on `clang` and the host `ld.so`; if either is missing the test prints
//! a note and returns, so the build never fails over a missing toolchain.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{
    elf::{ObjectFile, constants::SHN_UNDEF},
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
    reloc::x86_64::R_X86_64_64,
};

mod common;

/// The dependency: a `.globl` with neither `.type` nor `.size`, which is what
/// hand-written assembly publishes and what reaches `.dynsym` as an unsized
/// `STT_NOTYPE` export.
const LIB_SRC: &[u8] = b"    .data\n    .globl notyped_datum\n\
    notyped_datum:\n    .quad 42\n";

/// Stores the address of that export in a writable slot and reports whether it
/// arrived: exit 0 when the loader filled the pointer, 3 when it is null.
const MAIN_SRC: &[u8] = b"extern long notyped_datum;\n\
    long *datump = &notyped_datum;\n\
    int main(void) { return datump != 0 ? 0 : 3; }\n";

/// A freestanding `_start` that calls `main` then exits with its return value
/// (syscall 60).
const START_SRC: &[u8] = b"    .text\n    .globl _start\n_start:\n    \
    call    main\n    movl    %eax, %edi\n    movl    $60, %eax\n    syscall\n";

/// The import needs both halves to be of any use: a name in `.dynsym` for the
/// loader to look up, and a relocation naming it so the pointer slot is
/// actually filled. Either one alone leaves the program with a null pointer.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_absolute_data_reference_imports_the_symbol_it_names() {
    let Some(dir) = workdir("data_import") else {
        return;
    };
    let Some(prog) = link_program(&dir) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");

    let (idx, shndx) = dynsym_row(&bytes, b"notyped_datum")
        .expect("the data-only import must reach .dynsym");
    assert_eq!(
        shndx, SHN_UNDEF,
        "the import stays undefined; the loader supplies it"
    );

    let rela = rela_rows(&bytes, b".rela.dyn");
    assert!(
        rela.iter().any(|r| r.1 == idx && r.2 == R_X86_64_64),
        "the pointer slot needs an R_X86_64_64 naming notyped_datum, got \
         {rela:#x?}"
    );
    // The reserved slot used to stay zeroed, which reads as R_X86_64_NONE.
    assert!(
        rela.iter().all(|r| r.2 != 0),
        "no dynamic relocation may be left as R_*_NONE filler, got {rela:#x?}"
    );

    let status = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(0),
        "the relocated pointer must be non-null at runtime"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs at all.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("xold_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the dependency with xold itself and links the fixture against it
/// into a dynamic executable, returning the executable's path. The shared
/// object's file name is its `DT_NEEDED` name, which is what lets
/// `LD_LIBRARY_PATH` point the loader at `dir`.
fn link_program(dir: &Path) -> Option<PathBuf> {
    let Some(interp) = interpreter() else {
        eprintln!("skipping data_import: interpreter path unknown");
        return None;
    };
    let lib_o = dir.join("lib.o");
    assemble(LIB_SRC, &lib_o)?;
    let lib = dir.join("libnotyped.so");
    link_shared(
        std::slice::from_ref(&lib_o),
        &lib,
        Some(b"libnotyped.so"),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold -shared link must succeed");

    let main_o = dir.join("main.o");
    let start_o = dir.join("start.o");
    compile(MAIN_SRC, &main_o)?;
    assemble(START_SRC, &start_o)?;

    let prog = dir.join("prog");
    let linked = link_dyn_exec(
        &[main_o, start_o, lib],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    );
    assert!(
        linked.is_ok(),
        "the reference to the dependency export must resolve: {:?}",
        linked.err()
    );
    Some(prog)
}

/// Compiles `src` with the host clang as freestanding `-fPIE`, so the pointer
/// slot the test cares about is an `R_X86_64_64` in a writable section.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIE", "-ffreestanding", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// Assembles `src` with the host clang.
fn assemble(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("S");
    fs::write(&src_path, src).expect("write assembly");
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIC", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

// --- readers ---------------------------------------------------------------

/// Decodes the named relocation section into `(r_offset, sym, r_type)` rows.
fn rela_rows(bytes: &[u8], section: &[u8]) -> Vec<(u64, u32, u32)> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(sec) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == section)
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(sec) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for chunk in data.chunks(24) {
        if chunk.len() < 24 {
            break;
        }
        let off = u64::from_le_bytes(chunk[..8].try_into().unwrap_or([0; 8]));
        let info =
            u64::from_le_bytes(chunk[8..16].try_into().unwrap_or([0; 8]));
        out.push((
            off,
            u32::try_from(info >> 32).unwrap_or(0),
            u32::try_from(info & 0xffff_ffff).unwrap_or(0),
        ));
    }
    out
}

/// Locates `name` in `.dynsym`, returning its index and `st_shndx`.
fn dynsym_row(bytes: &[u8], name: &[u8]) -> Option<(u32, u16)> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let sec = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynsym")?;
    let data = obj.section_data(sec).ok()?;
    for (i, chunk) in data.chunks(24).enumerate() {
        if chunk.len() < 24 {
            break;
        }
        let st_name =
            u32::from_le_bytes(chunk[..4].try_into().unwrap_or([0; 4]));
        let st_shndx =
            u16::from_le_bytes(chunk[6..8].try_into().unwrap_or([0; 2]));
        if dynstr_name(bytes, sec.sh_link.get(), st_name) == name {
            return Some((u32::try_from(i).ok()?, st_shndx));
        }
    }
    None
}

/// Reads the NUL-terminated string at `offset` within the section indexed by
/// `strtab_shndx`.
fn dynstr_name(bytes: &[u8], strtab_shndx: u32, offset: u32) -> &[u8] {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return &[];
    };
    let Some(strtab) = obj.sections().get(strtab_shndx as usize) else {
        return &[];
    };
    let Ok(data) = obj.section_data(strtab) else {
        return &[];
    };
    let start = offset as usize;
    if start >= data.len() {
        return &[];
    }
    let end = data[start..]
        .iter()
        .position(|&b| b == 0)
        .map_or(data.len(), |nul| start + nul);
    &data[start..end]
}
