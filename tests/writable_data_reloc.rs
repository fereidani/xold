//! Where a pointer to a shared-library symbol is stored decides how it is
//! filled in.
//!
//! An executable has three mechanisms for a reference to a symbol a dependency
//! defines, and only one of them keeps the image relocatable:
//!
//! - The slot is in a writable section, so the loader can write it: `.rela.dyn`
//!   names the symbol and the loader stores its address there. The image keeps
//!   its random base.
//! - The slot is read-only, so the address has to be a link-time constant. For
//!   data that means a copy relocation -- the executable owns the storage in
//!   `.bss` and every reference resolves to it -- and the image drops to a
//!   fixed base, because the constant it just baked in is only valid there.
//! - The same, for a function: a canonical PLT entry supplies the constant. The
//!   fixed base follows from the read-only slot holding that constant, not from
//!   the stub existing: a stub only ever reached PC-relatively moves with the
//!   image and costs it nothing.
//!
//! xold used to reach for the last two on *any* absolute reference, without
//! asking whether the slot was writable, so one ordinary `-fPIE` translation
//! unit holding `int *p = &imported;` dropped the whole image to `ET_EXEC` and
//! cost the program its ASLR. lld picks the mechanism the other way round: the
//! copy-relocation and canonical-PLT block in `RelocationScanner::processAux`
//! is reached only once `canWrite` has ruled a dynamic relocation out.
//!
//! Gated on `clang` and the system `ld.so`; if absent the tests print a note
//! and return, so the build never fails over a missing toolchain.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{
    elf::{
        ObjectFile,
        constants::{ET_DYN, ET_EXEC, SHN_UNDEF},
    },
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
    reloc::x86_64::{R_X86_64_64, R_X86_64_COPY, R_X86_64_JUMP_SLOT},
};

mod common;

/// The dependency: an initialised datum and a function that bumps it.
const LIB_SRC: &[u8] = b"int counter = 5;\n\
    int bump(void) { counter += 1; return counter; }\n";

/// Stores the address of the imported datum in a writable slot
/// (`int *cp = &counter;` is an `R_X86_64_64` in `.data`), writes through it
/// and reads it back. Exits 6 when the slot reached the dependency's storage.
const WRITABLE_MAIN_SRC: &[u8] = b"extern int counter;\nint *cp = &counter;\n\
    int main(void) { *cp += 1; return *cp; }\n";

/// The same pointer, forced into a read-only section. `-fPIE` C has no
/// spelling for that -- the compiler puts a relocatable `const` pointer in
/// `.data.rel.ro`, which is writable -- so the slot is assembled by hand.
const READONLY_PTR_SRC: &[u8] = b"    .section .rodata,\"a\",@progbits\n    \
    .globl cp_ro\n    .p2align 3\ncp_ro:\n    .quad counter\n";

/// Reads the pointer out of that read-only slot and writes through it.
const READONLY_MAIN_SRC: &[u8] = b"extern int *const cp_ro;\n\
    int main(void) { *cp_ro += 1; return *cp_ro; }\n";

/// Stores the address of an imported *function* in a writable slot and calls
/// through it. The call is indirect, so nothing else asks for a PLT entry:
/// whether one exists is exactly the question under test.
const FN_PTR_MAIN_SRC: &[u8] = b"extern int bump(void);\n\
    int (*fp)(void) = bump;\nint main(void) { return fp(); }\n";

/// The same pointer to a function, in a read-only section. Assembled by hand
/// for the same reason the data one is; it is also the shape `.eh_frame`'s
/// personality pointer takes when the unwind tables are built without `-fPIC`.
const READONLY_FN_PTR_SRC: &[u8] =
    b"    .section .rodata,\"a\",@progbits\n    \
    .globl fp_ro\n    .p2align 3\nfp_ro:\n    .quad bump\n";

/// Reads that read-only pointer, compares it against the address the same
/// function has when taken the ordinary way (through the GOT, which is how
/// `-fPIE` code reaches an import), and calls through it. Exit 0 only when the
/// two agree and the call reaches the dependency.
const READONLY_FN_MAIN_SRC: &[u8] = b"extern int bump(void);\n\
    extern int (*const fp_ro)(void);\n\
    int main(void)\n\
    {\n\
        if (fp_ro != bump) { return 3; }\n\
        if (fp_ro() != 6) { return 4; }\n\
        return 0;\n\
    }\n";

/// A freestanding `_start` that calls `main` then exits with its return value
/// (syscall 60).
const START_SRC: &[u8] = b"    .text\n    .globl _start\n_start:\n    \
    call    main\n    movl    %eax, %edi\n    movl    $60, %eax\n    syscall\n";

/// The feature-probe idiom: a weak declaration of a name nothing in the link
/// defines, tested before use. Compiled `-fno-pie` the test compiles to an
/// absolute `R_X86_64_64` against `maybe_fn` in `.text`, which is read-only.
/// Exits 42 when the address came out zero and the absent branch ran, and 1
/// when the linker put something else there (which would then be called).
const WEAK_PROBE_SRC: &[u8] =
    b"extern void maybe_fn(void) __attribute__((weak));\n\
    int main(void) { if (maybe_fn) { maybe_fn(); return 1; } return 42; }\n";

/// The same idiom against a name the dependency does export, through a
/// writable pointer slot. The loader has a definition to bind, so the entry
/// naming it must survive: exits 5, the dependency's `counter`.
const WEAK_IMPORT_MAIN_SRC: &[u8] =
    b"extern int counter __attribute__((weak));\nint *cp = &counter;\n\
    int main(void) { return cp ? *cp : 0; }\n";

/// A weakly declared imported *function* whose address is taken in a read-only
/// slot. The `.weak` directive is on the assembly reference because that is
/// the only reference to the name: an ordinary one would make the symbol
/// strong however the C file declared it.
const WEAK_RO_FN_PTR_SRC: &[u8] = b"    .weak bump\n    \
    .section .rodata,\"a\",@progbits\n    \
    .globl wfp_ro\n    .p2align 3\nwfp_ro:\n    .quad bump\n";

/// Calls through that read-only slot, which can only work if the slot holds
/// the canonical PLT stub's address. Exits 6, the bumped counter.
const WEAK_RO_FN_MAIN_SRC: &[u8] = b"extern int (*const wfp_ro)(void);\n\
    int main(void) { return wfp_ro ? wfp_ro() : 0; }\n";

/// A pointer to imported data in a writable section is filled in by the
/// loader, so the image needs neither a copy slot nor a fixed base.
///
/// This is the case that cost every affected image its ASLR: the reference is
/// an ordinary `-fPIE` `int *p = &imported;`, and it used to open a copy slot,
/// which forced `ET_EXEC`. `clang -fuse-ld=lld` emits `R_X86_64_64
/// counter@...` for the same input and keeps the image `ET_DYN`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_writable_pointer_to_imported_data_stays_position_independent() {
    let Some(dir) = workdir("writable") else {
        return;
    };
    let Some(prog) = link_program(&dir, WRITABLE_MAIN_SRC, None) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");

    assert_eq!(
        e_type(&bytes),
        Some(ET_DYN),
        "a writable slot needs no link-time address, so the image keeps its \
         random base"
    );
    let idx = dynsym_index(&bytes, b"counter")
        .expect("the import needs a name the loader can look up");
    let rela = rela_rows(&bytes, b".rela.dyn");
    assert!(
        rela.iter().any(|r| r.1 == idx && r.2 == R_X86_64_64),
        "the slot is filled by an R_X86_64_64 naming counter, got {rela:#x?}"
    );
    assert!(
        !rela.iter().any(|r| r.2 == R_X86_64_COPY),
        "and by nothing else: no copy relocation, got {rela:#x?}"
    );

    assert_eq!(
        run(&prog, &dir),
        Some(6),
        "the write must reach the dependency's counter (5), making it 6"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The same reference from a read-only section still takes a copy relocation
/// and still forces a fixed base, so the fix narrows the rule rather than
/// removing it.
///
/// Nothing may write the slot at load time, so the address in it has to be one
/// this link chose: the copy slot in `.bss`. That address is only valid at a
/// fixed base, which is why the image gives up `ET_DYN` for it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_read_only_pointer_to_imported_data_is_still_copy_relocated() {
    let Some(dir) = workdir("readonly") else {
        return;
    };
    let Some(prog) =
        link_program(&dir, READONLY_MAIN_SRC, Some(READONLY_PTR_SRC))
    else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");

    assert_eq!(
        e_type(&bytes),
        Some(ET_EXEC),
        "the read-only slot holds a link-time address, which only a fixed \
         base makes true"
    );
    let rela = rela_rows(&bytes, b".rela.dyn");
    let copies: Vec<_> = rela.iter().filter(|r| r.2 == R_X86_64_COPY).collect();
    assert_eq!(copies.len(), 1, "one copy slot for counter, got {rela:#x?}");
    let idx = dynsym_index(&bytes, b"counter").expect("counter in .dynsym");
    assert_eq!(copies[0].1, idx, "the copy relocation must name counter");

    assert_eq!(
        run(&prog, &dir),
        Some(6),
        "the loader copies counter (5) into .bss and the write makes it 6"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A pointer to an imported *function* in a writable section takes the same
/// route: the loader stores the function's address, and no canonical PLT entry
/// stands in for it.
///
/// A canonical PLT entry exists to give such a reference a link-time constant.
/// A writable slot needs none, and allocating one would cost a stub, a
/// `.got.plt` slot and a `JUMP_SLOT` for nothing.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_writable_pointer_to_an_imported_function_needs_no_canonical_plt() {
    let Some(dir) = workdir("fnptr") else {
        return;
    };
    let Some(prog) = link_program(&dir, FN_PTR_MAIN_SRC, None) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");

    assert_eq!(
        e_type(&bytes),
        Some(ET_DYN),
        "no canonical PLT means no link-time address, so no fixed base"
    );
    let idx = dynsym_index(&bytes, b"bump").expect("bump in .dynsym");
    let rela = rela_rows(&bytes, b".rela.dyn");
    assert!(
        rela.iter().any(|r| r.1 == idx && r.2 == R_X86_64_64),
        "the slot is filled by an R_X86_64_64 naming bump, got {rela:#x?}"
    );
    let plt = rela_rows(&bytes, b".rela.plt");
    assert!(
        !plt.iter().any(|r| r.1 == idx && r.2 == R_X86_64_JUMP_SLOT),
        "no canonical PLT entry may be allocated for it, got {plt:#x?}"
    );

    assert_eq!(
        run(&prog, &dir),
        Some(6),
        "the call through the relocated pointer must reach bump"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A read-only pointer to an imported function links, drops the image to a
/// fixed base, and stores the canonical PLT stub the whole image agrees on.
///
/// Nothing may write this slot at load time, so the address in it has to be
/// one this link chose -- the stub -- and that address is only valid at a
/// fixed base. Until the scan counted the reference, the image stayed `ET_DYN`
/// and the link was refused instead: the stub was not a link-time constant,
/// and the read-only slot could carry no dynamic relocation either. GNU `ld`
/// and `ld.lld -no-pie` both produce a working `ET_EXEC` here; `ld.lld -pie`
/// links it and miscompiles it, storing the unbased stub address and dying on
/// the first call through the pointer, so the fixed base is the answer and its
/// PIE path is not a model to copy.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_read_only_pointer_to_an_imported_function_takes_a_canonical_plt() {
    let Some(dir) = workdir("rofn") else {
        return;
    };
    let Some(prog) =
        link_program(&dir, READONLY_FN_MAIN_SRC, Some(READONLY_FN_PTR_SRC))
    else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");

    assert_eq!(
        e_type(&bytes),
        Some(ET_EXEC),
        "the read-only slot holds the stub's address, which only a fixed base \
         makes true"
    );
    let idx = dynsym_index(&bytes, b"bump").expect("bump in .dynsym");
    let plt = rela_rows(&bytes, b".rela.plt");
    let entry = plt
        .iter()
        .position(|r| r.1 == idx && r.2 == R_X86_64_JUMP_SLOT)
        .expect("the reference needs a canonical PLT entry to resolve to");
    let stub = section_addr(&bytes, b".plt")
        .expect("an allocated PLT entry needs a .plt section")
        + PLT_HEADER
        + PLT_ENTRY * u64::try_from(entry).unwrap_or(0);
    let (shndx, value) = dynsym_row(&bytes, idx).expect("bump's row");
    assert_eq!(
        shndx, SHN_UNDEF,
        "the executable does not define bump; the stub only stands in for it"
    );
    assert_eq!(
        value, stub,
        "the row must name the stub, which is the address every reference in \
         this image resolved to"
    );
    let rela = rela_rows(&bytes, b".rela.dyn");
    assert!(
        !rela.iter().any(|r| r.1 == idx && r.2 == R_X86_64_64),
        "and nothing may be left to store the real function over it, got \
         {rela:#x?}"
    );

    assert_eq!(
        run(&prog, &dir),
        Some(0),
        "the pointer in .rodata must be the one address bump has in this \
         image, and calling through it must reach the dependency"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A weak undefined nothing can define needs no entry at all, and a read-only
/// slot holding one is not a reason to refuse the link.
///
/// `extern void f(void) __attribute__((weak)); if (f) f();` is how a program
/// asks whether a function is there. The answer is the ABI's: with no
/// definition in the link and none in any dependency, the address is zero, and
/// zero is what the writer already stored. The loader has nothing to add --
/// there is no name it could look the symbol up under and no storage it could
/// write -- so the reference needs no dynamic relocation, and the refusal that
/// guards read-only slots must not fire on one that asks for nothing.
///
/// Compiled `-fno-pie`, the probe lands in `.text`. xold used to classify it
/// symbol-based, notice `.text` is read-only, and report `reference to
/// 'maybe_fn' from a read-only section`. `ld.lld` links the same input and the
/// program takes the absent branch, which is what this asserts by running it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_weak_undefined_nothing_defines_resolves_to_zero_and_runs() {
    let Some(dir) = workdir("weakprobe") else {
        return;
    };
    let Some(prog) = link_program_pic(&dir, WEAK_PROBE_SRC, None, "-fno-pie")
    else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");

    // No entry names the probe: nothing in `.rela.dyn` could describe a value
    // the loader does not have.
    if let Some(idx) = dynsym_index(&bytes, b"maybe_fn") {
        let rela = rela_rows(&bytes, b".rela.dyn");
        assert!(
            !rela.iter().any(|r| r.1 == idx),
            "a name no image can supply needs no .rela.dyn entry, got {rela:#x?}"
        );
    }

    assert_eq!(
        run(&prog, &dir),
        Some(42),
        "the probe must see a null address and take the absent branch"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The same idiom against a name the dependency *does* export keeps its
/// symbol-based entry, so the loader still binds it.
///
/// This is the half the rule above must not swallow: weakness says what
/// happens when there is no definition, not that the reference is dead. The
/// slot is writable and the image position-independent, so the loader fills it
/// in, and the program reads the dependency's own storage through it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_weak_import_a_dependency_exports_is_still_bound_by_name() {
    let Some(dir) = workdir("weakimport") else {
        return;
    };
    let Some(prog) = link_program(&dir, WEAK_IMPORT_MAIN_SRC, None) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");

    let idx = dynsym_index(&bytes, b"counter")
        .expect("a weak import a dependency exports still needs its name");
    let rela = rela_rows(&bytes, b".rela.dyn");
    assert!(
        rela.iter().any(|r| r.1 == idx && r.2 == R_X86_64_64),
        "the slot is filled by an R_X86_64_64 naming counter, got {rela:#x?}"
    );

    assert_eq!(
        run(&prog, &dir),
        Some(5),
        "the pointer must reach the dependency's counter"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A weakly referenced imported *function* in a read-only slot still reaches a
/// canonical PLT stub.
///
/// The dependency defines it, so the address is not zero and the slot is not
/// final: it holds the stub, which is the one address the name has in this
/// image. Reading weakness as "resolves to zero" here would leave a null
/// pointer the program then declines to call.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_weak_read_only_function_pointer_still_takes_a_canonical_plt() {
    let Some(dir) = workdir("weakrofn") else {
        return;
    };
    let Some(prog) =
        link_program(&dir, WEAK_RO_FN_MAIN_SRC, Some(WEAK_RO_FN_PTR_SRC))
    else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");

    let idx = dynsym_index(&bytes, b"bump").expect("bump in .dynsym");
    let plt = rela_rows(&bytes, b".rela.plt");
    assert!(
        plt.iter().any(|r| r.1 == idx && r.2 == R_X86_64_JUMP_SLOT),
        "the reference needs a canonical PLT entry to resolve to, got {plt:#x?}"
    );

    assert_eq!(
        run(&prog, &dir),
        Some(6),
        "calling through the read-only slot must reach the dependency"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Bytes of the PLT header on x86-64, which `PLT[0]` (the resolver
/// trampoline) occupies before the first user entry.
const PLT_HEADER: u64 = 16;
/// Bytes of one x86-64 PLT entry.
const PLT_ENTRY: u64 = 16;

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs at all.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping writable-data-reloc {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("xold_wdreloc_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the dependency with `xold -shared` and links `main_src` (plus an
/// optional assembly unit holding the pointer slot) against it into a dynamic
/// executable inside `dir`, returning the executable's path. The main object is
/// compiled `-fPIE`, which is what an ordinary position-independent build gives
/// the linker; [`link_program_pic`] takes the flag for a test that needs
/// another.
///
/// The dependency's file name is its `DT_NEEDED` name, which is what lets
/// `LD_LIBRARY_PATH` point the loader at `dir`.
fn link_program(
    dir: &Path,
    main_src: &[u8],
    extra_asm: Option<&[u8]>,
) -> Option<PathBuf> {
    link_program_pic(dir, main_src, extra_asm, "-fPIE")
}

/// The same, compiling the main object under `pic`.
fn link_program_pic(
    dir: &Path,
    main_src: &[u8],
    extra_asm: Option<&[u8]>,
    pic: &str,
) -> Option<PathBuf> {
    let Some(interp) = interpreter() else {
        eprintln!("skipping writable-data-reloc: interpreter path unknown");
        return None;
    };
    let lib_o = dir.join("lib.o");
    compile(LIB_SRC, &lib_o, "-fPIC")?;
    let lib = dir.join("libwdreloc.so");
    link_shared(
        std::slice::from_ref(&lib_o),
        &lib,
        Some(b"libwdreloc.so"),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold -shared link must succeed");

    let main_o = dir.join("main.o");
    let start_o = dir.join("start.o");
    compile(main_src, &main_o, pic)?;
    assemble(START_SRC, &start_o)?;
    let mut inputs = vec![main_o, start_o];
    if let Some(src) = extra_asm {
        let ptr_o = dir.join("ptr.o");
        assemble(src, &ptr_o)?;
        inputs.push(ptr_o);
    }
    inputs.push(lib);

    let prog = dir.join("prog");
    let linked = link_dyn_exec(
        &inputs,
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

/// Compiles `src` with the host clang as freestanding code under `pic`.
fn compile(src: &[u8], obj: &Path, pic: &str) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", pic, "-ffreestanding", "-c"])
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

/// The `e_type` field of the linked image.
fn e_type(bytes: &[u8]) -> Option<u16> {
    Some(ObjectFile::parse(bytes).ok()?.header().e_type.get())
}

/// The virtual address of the named section.
fn section_addr(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == name)
        .map(|s| s.sh_addr.get())
}

/// The `(st_shndx, st_value)` of the `.dynsym` row at `index`, which is what
/// tells a canonical PLT stub apart from a definition.
fn dynsym_row(bytes: &[u8], index: u32) -> Option<(u16, u64)> {
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
    Some((
        u16::from_le_bytes(chunk[6..8].try_into().ok()?),
        u64::from_le_bytes(chunk[8..16].try_into().ok()?),
    ))
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
