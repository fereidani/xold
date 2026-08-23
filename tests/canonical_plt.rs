//! A canonical PLT entry is the definition, so one function has one address.
//!
//! When an executable takes the address of a shared-library function from a
//! slot that cannot carry a dynamic relocation -- a read-only slot, or a
//! narrower field than a pointer -- the link allocates a PLT stub, resolves
//! every reference to it, and pairs it with a `JUMP_SLOT` the loader binds.
//! That makes the stub the function's address *in this image*, and everything
//! else has to agree with it.
//!
//! xold agreed everywhere but one place: a pointer slot in `.data` also got a
//! symbol-based `.rela.dyn` entry naming the same import, so the writer stored
//! the stub and the loader then stored the real function over it. A pointer
//! taken through the stub and a pointer read out of `.data` compared unequal
//! -- two values for one function pointer in one image, which ISO C forbids.
//!
//! lld's mechanism, mirrored here: the address is fixed by this link, so the
//! slot needs no name resolved (`Relocations.cpp:1103-1110`), and the import's
//! `.dynsym` row stays `SHN_UNDEF` while carrying the stub in `st_value`
//! (`SyntheticSections.cpp:2276`) so another image binding the name lands on
//! the same stub. `SHN_UNDEF` is only how the row is *spelled*, though: lld
//! makes the symbol `Defined` at the stub before choosing the hash set
//! (`Relocations.cpp:1557`) and forces the section index back down only at
//! emission, so the row is hashed. It has to be -- a name no other image can
//! look up is a stub no other image can reach. The test checks that too.
//!
//! Gated on `clang` and a probeable interpreter; if either is missing the test
//! prints a note and returns, so the build never fails over a missing
//! toolchain.

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
    reloc::x86_64::{R_X86_64_64, R_X86_64_JUMP_SLOT},
};

mod common;

/// The dependency: one function, one datum it bumps.
const LIB_SRC: &[u8] = b"int counter = 5;\n\
    int bump(void) { counter += 1; return counter; }\n";

/// Takes the function's address through a PC-relative reference, which is a
/// slot no dynamic relocation can describe. That is what allocates the
/// canonical PLT entry; the pointer in `.data` below is what has to agree
/// with it.
const ADDR_SRC: &[u8] = b"    .text\n    .globl taken\n\
    .type taken,@function\ntaken:\n    leaq bump(%rip), %rax\n    retq\n\
    .size taken, .-taken\n";

/// The program: one pointer to the import in `.data`, compared against the
/// address the code takes. Exit 0 only when the two agree and the call
/// through the pointer reaches the real function.
const MAIN_SRC: &[u8] = b"extern int bump(void);\n\
    extern int (*taken(void))(void);\n\
    int (*fp)(void) = bump;\n\
    int main(void)\n\
    {\n\
        if (fp != taken()) { return 1; }\n\
        if (fp() != 6) { return 2; }\n\
        return 0;\n\
    }\n";

/// A freestanding `_start` that calls `main` then exits with its return value
/// (syscall 60).
const START_SRC: &[u8] = b"    .text\n    .globl _start\n_start:\n    \
    call    main\n    movl    %eax, %edi\n    movl    $60, %eax\n    syscall\n";

/// A third image that takes `&bump` without defining it, so its `GLOB_DAT` is
/// resolved by the loader against the whole global scope -- the executable
/// first. What it returns is whichever `bump` the loader found.
const PROBE_SRC: &[u8] = b"extern int bump(void);\n\
    int (*probe(void))(void) { return bump; }\n";

/// The program for the cross-image test: its own canonical stub against what
/// a separate library's address-of resolved to. Exit 0 only when the two are
/// one value.
const CROSS_SRC: &[u8] = b"extern int bump(void);\n\
    extern int (*taken(void))(void);\n\
    extern int (*probe(void))(void);\n\
    int main(void)\n\
    {\n\
        if (taken() != probe()) { return 1; }\n\
        if (taken()() != 6) { return 2; }\n\
        return 0;\n\
    }\n";

/// Bytes of the PLT header on x86-64, which `PLT[0]` (the resolver
/// trampoline) occupies before the first user entry.
const PLT_HEADER: u64 = 16;
/// Bytes of one x86-64 PLT entry.
const PLT_ENTRY: u64 = 16;

/// The whole of item 2 in one link: the import's row names the stub, no
/// symbol-based entry contradicts it, the hashed part of `.dynsym` is
/// untouched, and a function pointer compares equal at runtime.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_canonical_plt_symbol_resolves_to_its_stub_everywhere() {
    let Some(dir) = workdir("canon") else {
        return;
    };
    let Some(prog) = link_program(&dir) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");

    // The stub the link allocated, from the JUMP_SLOT that binds it.
    let idx = dynsym_index(&bytes, b"bump").expect("bump needs a .dynsym row");
    let plt = rela_rows(&bytes, b".rela.plt");
    let entry = plt
        .iter()
        .position(|r| r.1 == idx && r.2 == R_X86_64_JUMP_SLOT)
        .expect("the address-taken import needs a canonical PLT entry");
    let stub = section_addr(&bytes, b".plt")
        .expect("an allocated PLT entry needs a .plt section")
        + PLT_HEADER
        + PLT_ENTRY * u64::try_from(entry).unwrap_or(0);

    // The row stays undefined -- the executable does not define `bump` -- but
    // states the address every reference in this image resolved to.
    let row = dynsym_entry(&bytes, idx).expect("bump's row");
    assert_eq!(
        row.shndx, SHN_UNDEF,
        "the row must stay undefined, or this image's own JUMP_SLOT would \
         bind back to the stub it is meant to fill"
    );
    assert_eq!(
        row.value, stub,
        "st_value must be the PLT stub, so another image binding the name \
         reaches the same address this one uses"
    );
    assert_ne!(row.size, 0, "the extent comes from the dependency's export");

    // Nothing may overwrite the slot the writer filled with that address.
    let rela = rela_rows(&bytes, b".rela.dyn");
    assert!(
        !rela.iter().any(|r| r.1 == idx && r.2 == R_X86_64_64),
        "a symbol-based entry would have the loader store the real function \
         over the stub, got {rela:#x?}"
    );

    // The row is undefined but this image answers for the address, so it sits
    // in the hashed tail: the whole point of the stub is that another image
    // looking `bump` up here finds it, and `.gnu.hash` is the only table
    // glibc and musl consult.
    let symoffset =
        gnu_hash_symoffset(&bytes).expect("a dynamic image carries .gnu.hash");
    assert!(
        idx >= symoffset,
        "bump's row ({idx}) must be in the hashed tail ({symoffset}), or no \
         other image can find the stub by name and function-pointer equality \
         across images breaks"
    );

    // The property all of that exists for.
    assert_eq!(
        run(&prog, &dir),
        Some(0),
        "the pointer in .data and the address the code takes must be one \
         value, and calling through it must reach the dependency"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// What the stub is *for*: another image asking for `bump` must land on this
/// executable's stub, not on the dependency's own definition, or the same
/// function has two addresses in one process.
///
/// The loader reaches the row only through `.gnu.hash`, and that table
/// describes a contiguous tail of `.dynsym`, so a canonical row parked in the
/// unhashed prefix is invisible however correct its `st_value` is. glibc then
/// binds the probe's `GLOB_DAT` to the dependency's definition and the two
/// pointers compare unequal. (An `SHN_UNDEF` row is skipped only for
/// PLT-class relocations, which is what keeps this image's own `JUMP_SLOT`
/// from binding back to the stub it fills.)
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn another_image_binds_to_this_images_canonical_stub() {
    let Some(dir) = workdir("cross") else {
        return;
    };
    let Some(prog) = link_cross_program(&dir) else {
        return;
    };
    assert_eq!(
        run(&prog, &dir),
        Some(0),
        "a second library taking &bump must reach this executable's stub, or \
         function-pointer equality across images is broken"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs at all.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping canonical_plt {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("xold_canplt_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the dependency with `xold -shared`, then links the program against
/// it into a dynamic executable inside `dir`.
///
/// The dependency's file name is its `DT_NEEDED` name, which is what lets
/// `LD_LIBRARY_PATH` point the loader at `dir`.
fn link_program(dir: &Path) -> Option<PathBuf> {
    let Some(interp) = interpreter() else {
        eprintln!("skipping canonical_plt: interpreter path unknown");
        return None;
    };
    let lib_o = dir.join("lib.o");
    compile(LIB_SRC, &lib_o)?;
    let lib = dir.join("libcanplt.so");
    link_shared(
        std::slice::from_ref(&lib_o),
        &lib,
        Some(b"libcanplt.so"),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold -shared link must succeed");

    let main_o = dir.join("main.o");
    let addr_o = dir.join("addr.o");
    let start_o = dir.join("start.o");
    compile(MAIN_SRC, &main_o)?;
    assemble(ADDR_SRC, &addr_o)?;
    assemble(START_SRC, &start_o)?;

    let prog = dir.join("prog");
    let linked = link_dyn_exec(
        &[main_o, addr_o, start_o, lib],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    );
    assert!(
        linked.is_ok(),
        "xold dynamic-exec link must succeed: {:?}",
        linked.err()
    );
    Some(prog)
}

/// Builds the same dependency, a second library that takes `&bump` without
/// defining it, and a program that compares its own stub against what that
/// library resolved to.
///
/// The probe is built by the host toolchain: what is under test is whether
/// another image's loader-resolved address agrees with xold's stub, so the
/// other image should not come from xold too.
fn link_cross_program(dir: &Path) -> Option<PathBuf> {
    let Some(interp) = interpreter() else {
        eprintln!("skipping canonical_plt: interpreter path unknown");
        return None;
    };
    let lib_o = dir.join("lib.o");
    compile(LIB_SRC, &lib_o)?;
    let lib = dir.join("libcanplt.so");
    link_shared(
        std::slice::from_ref(&lib_o),
        &lib,
        Some(b"libcanplt.so"),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold -shared link must succeed");

    let probe = build_probe(dir, &lib)?;
    let main_o = dir.join("cross_main.o");
    let addr_o = dir.join("cross_addr.o");
    let start_o = dir.join("cross_start.o");
    compile(CROSS_SRC, &main_o)?;
    assemble(ADDR_SRC, &addr_o)?;
    assemble(START_SRC, &start_o)?;

    let prog = dir.join("cross_prog");
    let linked = link_dyn_exec(
        &[main_o, addr_o, start_o, probe, lib],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    );
    assert!(
        linked.is_ok(),
        "xold dynamic-exec link must succeed: {:?}",
        linked.err()
    );
    Some(prog)
}

/// Builds the address-taking probe library with the host toolchain, linked
/// against `dep` so its `bump` reference is an ordinary import.
fn build_probe(dir: &Path, dep: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let obj = dir.join("probe.o");
    compile(PROBE_SRC, &obj)?;
    let so = dir.join("libcanpltprobe.so");
    let ok = Command::new(clang)
        .arg("-shared")
        .arg("-o")
        .arg(&so)
        .arg(&obj)
        .arg(dep)
        .arg("-Wl,-soname,libcanpltprobe.so")
        .status()
        .ok()?
        .success();
    ok.then_some(so)
}

/// Compiles `src` with the host clang as freestanding position-independent
/// code.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIC", "-ffreestanding", "-c"])
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

/// Runs the linked program with the dependency reachable, returning its exit
/// status.
fn run(prog: &Path, dir: &Path) -> Option<i32> {
    Command::new(prog)
        .env("LD_LIBRARY_PATH", dir)
        .status()
        .expect("linked program must be runnable")
        .code()
}

// --- readers ---------------------------------------------------------------

/// The fields of one `.dynsym` row this test asserts on.
struct DynSymRow {
    shndx: u16,
    value: u64,
    size: u64,
}

/// The virtual address of the named section.
fn section_addr(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == name)
        .map(|s| s.sh_addr.get())
}

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

/// The `.dynsym` index of `name`, which is what a relocation row references.
fn dynsym_index(bytes: &[u8], name: &[u8]) -> Option<u32> {
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
        if dynstr_name(bytes, sec.sh_link.get(), st_name) == name {
            return u32::try_from(i).ok();
        }
    }
    None
}

/// The `.dynsym` row at `index`.
fn dynsym_entry(bytes: &[u8], index: u32) -> Option<DynSymRow> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let sec = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynsym")?;
    let data = obj.section_data(sec).ok()?;
    let chunk = data.chunks(24).nth(index as usize)?;
    if chunk.len() < 24 {
        return None;
    }
    Some(DynSymRow {
        shndx: u16::from_le_bytes(chunk[6..8].try_into().ok()?),
        value: u64::from_le_bytes(chunk[8..16].try_into().ok()?),
        size: u64::from_le_bytes(chunk[16..24].try_into().ok()?),
    })
}

/// The `symoffset` field of `.gnu.hash`: the `.dynsym` index of the first
/// hashed (defined) symbol.
fn gnu_hash_symoffset(bytes: &[u8]) -> Option<u32> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let sec = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".gnu.hash")?;
    let data = obj.section_data(sec).ok()?;
    Some(u32::from_le_bytes(data.get(4..8)?.try_into().ok()?))
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
