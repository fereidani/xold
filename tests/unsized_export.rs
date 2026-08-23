//! Exports a shared object declares no size for.
//!
//! `st_size` is optional. An assembler emits it only for a symbol the source
//! gave a `.size` directive, so hand-written assembly routinely publishes
//! `.globl` names with `st_size == 0`: entry points, `STT_NOTYPE` markers, and
//! the `_end`/`__bss_start` style bounds a shared object defines for itself.
//!
//! A missing size says only that the dependency declared no extent, which
//! decides whether a copy relocation can size a slot from the name. It does
//! not decide whether the export exists. Reading it as absence hid every such
//! name from `ctx.dep_exports`, and an absolute reference to one then had no
//! export to resolve against and was reported as an undefined reference, on
//! input the system linker links and runs.
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

/// The dependency, in assembly so the two exports carry no `.size`: a function
/// returning 7 and a datum, both `.globl` and both reaching `.dynsym` with
/// `st_size == 0`. The datum has no `.type` either, so it is `STT_NOTYPE`.
const LIB_SRC: &[u8] = b"    .text\n    .globl unsized_fn\n    \
    .type unsized_fn,@function\nunsized_fn:\n    movl $7, %eax\n    retq\n\
    \n    .data\n    .globl unsized_datum\nunsized_datum:\n    .quad 42\n";

/// Stores the address of the unsized function in a writable slot and calls
/// through it, so the slot is an `R_X86_64_64` against an import rather than a
/// call the PLT would answer. Exits 7 when the call reaches the dependency.
const FN_MAIN_SRC: &[u8] = b"extern int unsized_fn(void);\n\
    int (*fnp)(void) = unsized_fn;\nint main(void) { return fnp(); }\n";

/// The same pointer, in a read-only section. C has no spelling for that -- a
/// relocatable `const` pointer goes to `.data.rel.ro`, which is writable -- so
/// the slot is assembled by hand.
const RO_PTR_SRC: &[u8] = b"    .section .rodata,\"a\",@progbits\n    \
    .globl fnp_ro\n    .p2align 3\nfnp_ro:\n    .quad unsized_fn\n";

/// Calls through the read-only slot. Compiled non-PIC, so its own references
/// are absolute and the image is fixed-base, which is what makes the link-time
/// address in that slot true.
const RO_MAIN_SRC: &[u8] = b"extern int (*const fnp_ro)(void);\n\
    int main(void) { return fnp_ro(); }\n";

/// Stores the address of the unsized `STT_NOTYPE` datum in a writable slot.
const DATA_MAIN_SRC: &[u8] = b"extern long unsized_datum;\n\
    long *datump = &unsized_datum;\nint main(void) { return datump != 0; }\n";

/// A freestanding `_start` that calls `main` then exits with its return value
/// (syscall 60).
const START_SRC: &[u8] = b"    .text\n    .globl _start\n_start:\n    \
    call    main\n    movl    %eax, %edi\n    movl    $60, %eax\n    syscall\n";

/// A link against a shared object exporting an unsized function must resolve
/// the import, give it a `.dynsym` row, and relocate the pointer slot by name.
///
/// This is what the system linker does with the same inputs. xold used to
/// report `undefined reference to unsized_fn` instead, because the export
/// never reached `ctx.dep_exports`.
///
/// The pointer slot is in `.data`, so the loader can write it: the name-based
/// relocation is the whole mechanism, and neither a canonical PLT entry nor a
/// copy relocation is allocated beside it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn unsized_function_export_takes_a_symbol_based_relocation() {
    let Some(dir) = workdir("fn") else {
        return;
    };
    let Some(prog) = link_program(&dir, FN_MAIN_SRC) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");

    // The import needs a name the loader can look up, or nothing can ever fill
    // the slot that holds its address.
    let idx = dynsym_index(&bytes, b"unsized_fn")
        .expect("the unsized import must reach .dynsym");
    assert_eq!(
        dynsym_shndx(&bytes, b"unsized_fn"),
        Some(SHN_UNDEF),
        "the import stays undefined; the loader supplies it"
    );

    // The pointer slot is relocated by name, not left as the zeroed filler an
    // unresolved symbol-based target leaves behind.
    let rela = rela_rows(&bytes, b".rela.dyn");
    assert!(
        rela.iter().any(|r| r.1 == idx && r.2 == R_X86_64_64),
        "the pointer slot needs an R_X86_64_64 naming unsized_fn, got \
         {rela:#x?}"
    );
    // The slot the address goes in is writable, so the relocation above is all
    // it takes: no canonical PLT entry stands in for the function. A canonical
    // PLT is the fallback for a slot the loader cannot write -- the address has
    // to be a link-time constant then, and the PLT stub supplies one. lld
    // reaches its `NEEDS_PLT` path only after `canWrite` has ruled a dynamic
    // relocation out, and on these exact inputs emits the one `R_X86_64_64`
    // above and an empty `.rela.plt`.
    let plt = rela_rows(&bytes, b".rela.plt");
    assert!(
        !plt.iter().any(|r| r.1 == idx && r.2 == R_X86_64_JUMP_SLOT),
        "a writable pointer slot needs no canonical PLT entry, got {plt:#x?}"
    );
    assert!(
        !rela.iter().any(|r| r.2 == R_X86_64_COPY),
        "and no copy relocation either, got {rela:#x?}"
    );
    assert_eq!(
        ObjectFile::parse(&bytes)
            .expect("valid ELF")
            .header()
            .e_type
            .get(),
        ET_DYN,
        "neither mechanism was needed, so the image keeps its random base"
    );

    let status = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(7),
        "the call through the relocated pointer must reach the dependency"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The same address, taken from a read-only slot, does need the canonical PLT
/// entry: nothing can write that slot at load time, so what the link stores in
/// it has to be an address the link already knows, and the PLT stub is the
/// only such address a shared function has. The `JUMP_SLOT` beside it is what
/// the loader binds to the real function.
///
/// This is the half of the rule the writable case above narrows rather than
/// removes, and it is what `ld.lld` emits for the same inputs.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_read_only_slot_still_takes_a_canonical_plt_entry() {
    let Some(dir) = workdir("fn_ro") else {
        return;
    };
    let Some(prog) = link_program_with(&dir, RO_MAIN_SRC, PicMode::NoPic, true)
    else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");

    let idx = dynsym_index(&bytes, b"unsized_fn")
        .expect("the unsized import must reach .dynsym");
    let plt = rela_rows(&bytes, b".rela.plt");
    assert!(
        plt.iter().any(|r| r.1 == idx && r.2 == R_X86_64_JUMP_SLOT),
        "the canonical PLT entry needs a JUMP_SLOT naming unsized_fn, got \
         {plt:#x?}"
    );
    assert_eq!(
        ObjectFile::parse(&bytes)
            .expect("valid ELF")
            .header()
            .e_type
            .get(),
        ET_EXEC,
        "the stub's address is a link-time one, so the base must be fixed"
    );

    let status = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(7),
        "the call through the stub must reach the dependency"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The same for an unsized `STT_NOTYPE` datum, the shape a `.globl` with
/// neither `.type` nor `.size` produces. Nothing downstream can size a copy
/// slot from it, so the link is what is under test here: the export is found,
/// rather than the reference being reported as undefined.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn unsized_data_export_is_not_an_undefined_reference() {
    let Some(dir) = workdir("data") else {
        return;
    };
    let Some(_prog) = link_program(&dir, DATA_MAIN_SRC) else {
        return;
    };
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs at all.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping unsized-export {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("xold_unsized_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Which position-independence mode the main object is compiled under.
#[derive(Clone, Copy)]
enum PicMode {
    /// `-fPIE`: the pointer slot the test cares about lands in `.data`.
    Pie,
    /// `-fno-pie -fno-pic`: the object's own references are absolute, so the
    /// image is fixed-base.
    NoPic,
}

impl PicMode {
    /// The clang flags this mode compiles under.
    const fn flags(self) -> &'static [&'static str] {
        match self {
            Self::Pie => &["-fPIE"],
            Self::NoPic => &["-fno-pie", "-fno-pic"],
        }
    }
}

/// Builds the dependency and links `main_src`, compiled `-fPIE`, against it.
fn link_program(dir: &Path, main_src: &[u8]) -> Option<PathBuf> {
    link_program_with(dir, main_src, PicMode::Pie, false)
}

/// Builds the dependency and links `main_src` against it into a dynamic
/// executable inside `dir`, returning the executable's path. `ro_ptr` adds the
/// hand-assembled read-only pointer slot beside the main object.
///
/// The dependency is built by xold itself, so the test needs no shared object
/// from the host. Its file name is its `DT_NEEDED` name, which is what lets
/// `LD_LIBRARY_PATH` point the loader at `dir`.
fn link_program_with(
    dir: &Path,
    main_src: &[u8],
    pic: PicMode,
    ro_ptr: bool,
) -> Option<PathBuf> {
    let Some(interp) = interpreter() else {
        eprintln!("skipping unsized-export: interpreter path unknown");
        return None;
    };
    let lib_o = dir.join("unsized_lib.o");
    assemble(LIB_SRC, &lib_o)?;
    let lib = dir.join("libunsized.so");
    link_shared(
        std::slice::from_ref(&lib_o),
        &lib,
        Some(b"libunsized.so"),
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
    if ro_ptr {
        let ptr_o = dir.join("ro_ptr.o");
        assemble(RO_PTR_SRC, &ptr_o)?;
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
        "a reference to an unsized dependency export must resolve: {:?}",
        linked.err()
    );
    Some(prog)
}

/// Compiles `src` with the host clang as freestanding code under `pic`.
fn compile(src: &[u8], obj: &Path, pic: PicMode) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-ffreestanding", "-c"])
        .args(pic.flags())
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

/// The `.dynsym` index of `name`, which is what a relocation row references.
fn dynsym_index(bytes: &[u8], name: &[u8]) -> Option<u32> {
    dynsym_row(bytes, name).map(|(i, _)| i)
}

/// The `st_shndx` of the `.dynsym` row named `name`.
fn dynsym_shndx(bytes: &[u8], name: &[u8]) -> Option<u16> {
    dynsym_row(bytes, name).map(|(_, shndx)| shndx)
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
