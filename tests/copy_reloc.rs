//! Copy relocation end-to-end tests.
//!
//! A dynamic executable that imports a *data* symbol from a shared object can
//! reference it two ways, and xold handles each differently:
//!
//! - **Absolute access** (`-fno-pie -fno-pic`, `R_X86_64_32`/`32S`/`64`): xold
//!   copy-relocates the symbol. It reserves a slot in `.bss`, emits an
//!   `R_X86_64_COPY` relocation in `.rela.dyn`, and resolves the absolute
//!   reference to the copy slot's address. Because the absolute address is
//!   fixed at link time, the executable drops to a fixed load base (`ET_EXEC`).
//!   At load time the loader copies the symbol's initial bytes from the shared
//!   object into the executable's slot; the executable then owns and mutates
//!   that storage.
//!
//! - **GOT access** (`-fPIE`, `R_X86_64_REX_GOTPCRELX`): xold emits an
//!   `R_X86_64_GLOB_DAT` and the GOT entry is filled at load time with the
//!   symbol's address from the shared object. No copy relocation is produced.
//!   This is the correct mechanism for position-independent access.
//!
//! Both paths must exit 6 when the program imports `counter` (initially 5),
//! increments it, and returns it.
//!
//! A dependency may also give one object several names. All of them must land
//! on the one copy slot: a slot each would leave the program holding two
//! copies of an object the dependency has one of, so a write through one name
//! would be invisible through the other. The alias cases below cover that.
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
    elf::{ObjectFile, constants::*},
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
    reloc::x86_64::{R_X86_64_COPY, R_X86_64_GLOB_DAT, R_X86_64_RELATIVE},
};

mod common;

/// The shared library source: exports an initialised global `counter`.
const LIB_SRC: &[u8] = b"int counter = 5;\n";

/// A shared library that exports the same four bytes under two names, the way
/// glibc exports `environ`, `__environ` and `_environ` for one object.
const ALIAS_LIB_SRC: &[u8] = b"int counter = 5;\n\
     extern int alt_counter __attribute__((alias(\"counter\")));\n";

/// The main source: imports `counter`, increments and returns it.
const MAIN_SRC: &[u8] =
    b"extern int counter;\nint main(void) { counter += 1; return counter; }\n";

/// Imports both names of the aliased object, writes through one and reads
/// through the other. Returns 6 when the two are one object and 5 when they
/// are two.
const ALIAS_MAIN_SRC: &[u8] = b"extern int counter;\nextern int alt_counter;\n\
     int main(void) { counter += 1; return alt_counter; }\n";

/// A non-PIC unit that takes the address of `counter` absolutely, which is
/// what selects the copy slot.
const ABS_ADDR_SRC: &[u8] =
    b"extern int counter;\nint *counter_addr(void) { return &counter; }\n";

/// A PIC unit that reads `counter` through the GOT, so the same symbol gets a
/// GOT slot beside its copy slot.
const GOT_READ_SRC: &[u8] =
    b"extern int counter;\nint read_counter(void) { return counter; }\n";

/// Writes through the absolute path and reads back through the GOT one. The
/// two agree only if the GOT slot resolves to the copy in `.bss`; a slot left
/// pointing at the dependency's storage reads the original 5.
const ABS_AND_GOT_MAIN_SRC: &[u8] = b"extern int *counter_addr(void);\n\
     extern int read_counter(void);\n\
     int main(void) { *counter_addr() += 1; return read_counter(); }\n";

/// A freestanding `_start` that calls `main` then exits with its return value
/// (syscall 60).
const START_SRC: &[u8] = b"    .text\n    .globl _start\n_start:\n    \
    call    main\n    movl    %eax, %edi\n    movl    $60, %eax\n    syscall\n";

/// How to compile the main object: PIC (shared lib), PIE (GOT access), or
/// non-PIC (absolute access).
#[derive(Clone, Copy)]
enum PicMode {
    /// `-fPIC`: position-independent code (for the shared library).
    Pic,
    /// `-fPIE`: position-independent executable (GOT access, Case A).
    Pie,
    /// `-fno-pie -fno-pic`: non-PIC code (absolute access, Case B).
    NoPic,
}

/// Compiles `src` with the host clang, writing the object to `obj`.
fn compile(src: &[u8], obj: &Path, mode: PicMode) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let mut cmd = Command::new(clang);
    cmd.arg("--target=x86_64-linux-gnu");
    match mode {
        PicMode::Pic => {
            cmd.arg("-fPIC");
        }
        PicMode::Pie => {
            cmd.arg("-fPIE");
        }
        PicMode::NoPic => {
            cmd.args(["-fno-pie", "-fno-pic"]);
        }
    }
    cmd.args(["-ffreestanding", "-c"]);
    cmd.arg(&src_path).arg("-o").arg(obj);
    let ok = cmd.status().ok()?.success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// Assembles `src` (assembly) with clang.
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

/// Creates a fresh per-test working directory under the system temp dir, so
/// parallel tests get disjoint namespaces.
///
/// The process id is part of the name: two test binaries sharing a `TMPDIR`
/// otherwise pick the same path, and the second `remove_dir_all` deletes the
/// executable the first is running (`ETXTBSY`, or a link against files that
/// vanished mid-run).
fn workdir(prefix: &str) -> PathBuf {
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("xold_copyrel_{prefix}_{pid}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Builds a shared library with xold (`-shared`) from `src` compiled `-fPIC`,
/// returning its path inside `dir`. `name` is both the file name and the
/// soname, because the loader searches by the `DT_NEEDED` name.
fn build_lib(dir: &Path, src: &[u8], name: &str) -> Option<PathBuf> {
    let obj = dir.join(name).with_extension("o");
    let so = dir.join(name);
    compile(src, &obj, PicMode::Pic)?;
    link_shared(
        std::slice::from_ref(&obj),
        &so,
        Some(name.as_bytes()),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold -shared link must succeed");
    let _ = fs::remove_file(&obj);
    Some(so)
}

/// Case B: a non-PIC executable that imports `counter` via absolute
/// references (`R_X86_64_32S`) gets a copy relocation. The loader copies
/// `counter` into the executable's `.bss` and the code mutates that copy,
/// exiting 6.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn absolute_data_import_is_copy_relocated_and_runs() {
    let Some(clang) = which("clang") else {
        eprintln!("skipping copy-reloc absolute: clang unavailable");
        return;
    };
    let _ = clang;
    let Some(interp) = interpreter() else {
        eprintln!("skipping copy-reloc absolute: interpreter path unknown");
        return;
    };
    let dir = workdir("abs");
    let Some(lib) = build_lib(&dir, LIB_SRC, "libcopy.so") else {
        eprintln!("skipping copy-reloc absolute: host clang unavailable");
        return;
    };
    let main_o = dir.join("copyrel_abs_main.o");
    let start_o = dir.join("copyrel_abs_start.o");
    compile(MAIN_SRC, &main_o, PicMode::NoPic)
        .expect("host clang compiles -fno-pie");
    assemble(START_SRC, &start_o).expect("host clang assembles start.S");

    let prog = dir.join("copyrel_abs_prog");
    link_dyn_exec(
        &[main_o.clone(), start_o.clone(), lib.clone()],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-exec link must succeed");

    // The `.rela.dyn` must carry an `R_X86_64_COPY` for `counter`.
    let bytes = fs::read(&prog).expect("read output");
    let rela = rela_dyn_rows(&bytes);
    assert!(
        rela.iter().any(|r| r.2 == R_X86_64_COPY),
        "non-PIC data import must produce an R_X86_64_COPY"
    );

    let status = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(6),
        "counter should be copied (5) then incremented to 6"
    );

    cleanup(&[&main_o, &start_o, &lib, &prog]);
}

/// Case A: a PIE executable that imports `counter` via GOT access
/// (`R_X86_64_REX_GOTPCRELX`) uses `GLOB_DAT`, not COPY. The loader resolves
/// `counter` from the shared object at load time; the program exits 6.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn pie_data_import_uses_glob_dat_not_copy() {
    let Some(clang) = which("clang") else {
        eprintln!("skipping copy-reloc PIE: clang unavailable");
        return;
    };
    let _ = clang;
    let Some(interp) = interpreter() else {
        eprintln!("skipping copy-reloc PIE: interpreter path unknown");
        return;
    };
    let dir = workdir("pie");
    let Some(lib) = build_lib(&dir, LIB_SRC, "libcopy.so") else {
        return;
    };
    let main_o = dir.join("copyrel_pie_main.o");
    let start_o = dir.join("copyrel_pie_start.o");
    compile(MAIN_SRC, &main_o, PicMode::Pie)
        .expect("host clang compiles -fPIE");
    assemble(START_SRC, &start_o).expect("host clang assembles start.S");

    let prog = dir.join("copyrel_pie_prog");
    link_dyn_exec(
        &[main_o.clone(), start_o.clone(), lib.clone()],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-exec link must succeed");

    // GOT access routes to GLOB_DAT, never COPY.
    let bytes = fs::read(&prog).expect("read output");
    let rela = rela_dyn_rows(&bytes);
    assert!(
        rela.iter().any(|r| r.2 == R_X86_64_GLOB_DAT),
        "PIE GOT data import must produce an R_X86_64_GLOB_DAT"
    );
    assert!(
        !rela.iter().any(|r| r.2 == R_X86_64_COPY),
        "PIE GOT data import must not produce a COPY relocation"
    );

    let status = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(6),
        "counter (5) incremented to 6 via the resolved GOT entry"
    );

    cleanup(&[&main_o, &start_o, &lib, &prog]);
}

/// A symbol reached both absolutely and through the GOT gets a copy slot and a
/// GOT slot, and the two must describe the same storage.
///
/// The copy slot is this image's own `.bss`, so the GOT slot holds an address
/// this link fixed rather than a name for the loader to resolve: no `GLOB_DAT`
/// may name it, or the loader would write the dependency's address over the
/// one this link chose. The emitter used to classify it as preemptible and
/// emit `GLOB_DAT` while the sizing pass counted it as `RELATIVE`, so the two
/// passes decided the same slot differently; both are one `Rela64`, which is
/// why the region size hid the disagreement.
///
/// A copy relocation forces a fixed base, and at a fixed base the `RELATIVE`
/// entry that used to carry the address is a no-op -- the loader would add a
/// zero base to the value the writer already stored -- so no dynamic
/// relocation is emitted for the slot at all and the assertion is on its
/// contents instead. `ld.lld` emits `GLOB_DAT` here, having computed
/// preemptibility before it created the copy; the loader then binds the name
/// to the executable's own copy and lands on the same address by a longer
/// route.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn copy_relocated_symbol_got_slot_holds_the_copy_address() {
    let Some(clang) = which("clang") else {
        eprintln!("skipping copy-reloc got slot: clang unavailable");
        return;
    };
    let _ = clang;
    let Some(interp) = interpreter() else {
        eprintln!("skipping copy-reloc got slot: interpreter path unknown");
        return;
    };
    let dir = workdir("absgot");
    let Some(lib) = build_lib(&dir, LIB_SRC, "libcopy.so") else {
        return;
    };
    let main_o = dir.join("absgot_main.o");
    let abs_o = dir.join("absgot_abs.o");
    let got_o = dir.join("absgot_got.o");
    let start_o = dir.join("absgot_start.o");
    compile(ABS_AND_GOT_MAIN_SRC, &main_o, PicMode::NoPic)
        .expect("host clang compiles -fno-pie");
    compile(ABS_ADDR_SRC, &abs_o, PicMode::NoPic)
        .expect("host clang compiles -fno-pie");
    compile(GOT_READ_SRC, &got_o, PicMode::Pic)
        .expect("host clang compiles -fPIC");
    assemble(START_SRC, &start_o).expect("host clang assembles start.S");

    let prog = dir.join("absgot_prog");
    link_dyn_exec(
        &[
            main_o.clone(),
            abs_o.clone(),
            got_o.clone(),
            start_o.clone(),
            lib.clone(),
        ],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-exec link must succeed");

    let bytes = fs::read(&prog).expect("read output");
    let rela = rela_dyn_rows(&bytes);
    let copies: Vec<_> = rela.iter().filter(|r| r.2 == R_X86_64_COPY).collect();
    assert_eq!(copies.len(), 1, "one copy slot for counter");
    let slot_addr = copies[0].0;

    let counter = dynsym_lookup(&bytes, b"counter")
        .expect("counter must have a .dynsym entry");
    assert_eq!(
        counter.value, slot_addr,
        "the dynsym entry must define counter at its copy slot"
    );

    // The GOT slot must carry the copy address, and carry it as bytes: the
    // image has a fixed base, so there is no load base for a `RELATIVE` entry
    // to add and the writer's value is final.
    assert!(
        got_holds(&bytes, slot_addr),
        "the GOT must hold the copy slot's address {slot_addr:#x}"
    );
    assert!(
        !rela.iter().any(|r| r.2 == R_X86_64_RELATIVE),
        "a fixed-base image adds nothing to that address, so no RELATIVE \
         entry describes it, got {rela:#x?}"
    );
    let counter_idx = dynsym_index(&bytes, b"counter").expect("counter index");
    assert!(
        !rela
            .iter()
            .any(|r| r.2 == R_X86_64_GLOB_DAT && r.1 == counter_idx),
        "no GLOB_DAT may name a copy-relocated symbol"
    );

    let status = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(6),
        "the absolute write and the GOT read must reach the same storage"
    );

    cleanup(&[&main_o, &abs_o, &got_o, &start_o, &lib, &prog]);
}

/// Structural checks for the non-PIC (absolute) case: `.rela.dyn` carries one
/// `R_X86_64_COPY` for `counter` whose offset lands in `.bss`; the dynamic
/// symbol table defines `counter` at that same `.bss` address (the executable
/// owns the copy); and the `.bss` slot is non-empty.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn copy_relocation_structure_matches_the_system_linker_layout() {
    let Some(clang) = which("clang") else {
        eprintln!("skipping copy-reloc structural test: clang unavailable");
        return;
    };
    let _ = clang;
    let Some(interp) = interpreter() else {
        eprintln!("skipping copy-reloc structural test: interpreter unknown");
        return;
    };
    let dir = workdir("struct");
    let Some(lib) = build_lib(&dir, LIB_SRC, "libcopy.so") else {
        return;
    };
    let main_o = dir.join("copyrel_struct_main.o");
    let start_o = dir.join("copyrel_struct_start.o");
    compile(MAIN_SRC, &main_o, PicMode::NoPic)
        .expect("host clang compiles -fno-pie");
    assemble(START_SRC, &start_o).expect("host clang assembles start.S");
    let prog = dir.join("copyrel_struct_prog");
    link_dyn_exec(
        &[main_o.clone(), start_o.clone(), lib.clone()],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-exec link must succeed");

    let bytes = fs::read(&prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    assert_eq!(
        obj.header().e_type.get(),
        ET_EXEC,
        "absolute import forces ET_EXEC"
    );

    // The `.bss` region holds the copy slot.
    let bss = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".bss")
        .expect("must have a .bss section");
    let bss_addr = bss.sh_addr.get();
    let bss_end = bss_addr.checked_add(bss.sh_size.get()).expect("bss end");
    assert!(bss.sh_size.get() > 0, ".bss must hold the copy slot");

    // `.rela.dyn` has exactly one COPY relocation, naming `counter`, landing
    // inside `.bss`.
    let rela = rela_dyn_rows(&bytes);
    let copies: Vec<_> = rela
        .iter()
        .copied()
        .filter(|r| r.2 == R_X86_64_COPY)
        .collect();
    assert_eq!(copies.len(), 1, "exactly one R_X86_64_COPY for counter");
    let (copy_off, _copy_sym, _, _) = copies[0];
    assert!(
        copy_off >= bss_addr && copy_off < bss_end,
        "COPY offset {copy_off:#x} must lie in .bss [{bss_addr:#x},{bss_end:#x})"
    );

    // `counter` is a defined dynamic symbol at the copy slot address.
    let dynsym = dynsym_lookup(&bytes, b"counter").expect("counter in .dynsym");
    assert_eq!(
        dynsym.shndx,
        bss_shndx(&bytes),
        "counter must be defined in .bss"
    );
    assert_eq!(dynsym.value, copy_off, "counter value is the copy slot");
    assert_eq!(dynsym.size, 4, "counter is a 4-byte int");
    assert_eq!(dynsym.sym_type, STT_OBJECT);

    cleanup(&[&main_o, &start_o, &lib, &prog]);
}

/// Two names for one dependency object, both referenced absolutely, share a
/// single copy slot: one `R_X86_64_COPY`, one `.bss` slot, and both names
/// defined at it. The program writes through one name and reads through the
/// other, so it exits 6 only if the two really are one object -- with a slot
/// each it would read the untouched second copy and exit 5.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn aliased_data_imports_share_one_copy_slot() {
    let Some(prog) = link_alias_program("alias", ALIAS_MAIN_SRC) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");

    // One slot, so one relocation: a second would tell the loader to copy the
    // dependency's bytes to a second address.
    let copies: Vec<_> = rela_dyn_rows(&bytes)
        .into_iter()
        .filter(|r| r.2 == R_X86_64_COPY)
        .collect();
    assert_eq!(
        copies.len(),
        1,
        "aliases of one object take one R_X86_64_COPY between them"
    );
    let (copy_off, ..) = copies[0];

    // Both names are defined at that one slot, so the loader binds every
    // reference -- this image's and any other's -- to the same storage.
    let bss = bss_shndx(&bytes);
    for name in ["counter", "alt_counter"] {
        let row = dynsym_lookup(&bytes, name.as_bytes())
            .unwrap_or_else(|| panic!("{name} must be in .dynsym"));
        assert_eq!(row.shndx, bss, "{name} must be defined in .bss");
        assert_eq!(
            row.value, copy_off,
            "{name} must resolve to the copy slot the relocation fills"
        );
        assert_eq!(row.size, 4, "{name} is a 4-byte int");
        assert_eq!(row.sym_type, STT_OBJECT);
    }

    let dir = prog.parent().expect("workdir").to_path_buf();
    let status = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(6),
        "a write through counter must be visible through alt_counter"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// An alias no input references is still defined at the copy slot. That is the
/// point of copying the whole alias set: another image resolving the object by
/// its other name has to reach this executable's copy, not the dependency's
/// original.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn unreferenced_alias_is_defined_at_the_copy_slot() {
    let Some(prog) = link_alias_program("alias_unref", MAIN_SRC) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");

    let copies: Vec<_> = rela_dyn_rows(&bytes)
        .into_iter()
        .filter(|r| r.2 == R_X86_64_COPY)
        .collect();
    assert_eq!(copies.len(), 1, "one referenced import, one R_X86_64_COPY");
    let (copy_off, ..) = copies[0];

    let referenced =
        dynsym_lookup(&bytes, b"counter").expect("counter in .dynsym");
    assert_eq!(referenced.value, copy_off, "counter is the copy slot");
    let alias = dynsym_lookup(&bytes, b"alt_counter")
        .expect("an unreferenced alias still reaches .dynsym");
    assert_eq!(
        alias.value, copy_off,
        "the alias must name the same storage, not a slot of its own"
    );
    assert_eq!(
        alias.shndx,
        bss_shndx(&bytes),
        "the alias is defined in .bss"
    );

    let dir = prog.parent().expect("workdir").to_path_buf();
    let status = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(status.code(), Some(6), "counter (5) incremented to 6");
    let _ = fs::remove_dir_all(&dir);
}

/// Links `main_src` (non-PIC, so its data references are absolute) against a
/// shared library exporting one object under two names, returning the
/// executable's path. `None` when the host toolchain cannot build the inputs,
/// which the caller reports as a skip.
fn link_alias_program(prefix: &str, main_src: &[u8]) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping copy-reloc {prefix}: clang unavailable");
        return None;
    }
    let Some(interp) = interpreter() else {
        eprintln!("skipping copy-reloc {prefix}: interpreter path unknown");
        return None;
    };
    let dir = workdir(prefix);
    let lib = build_lib(&dir, ALIAS_LIB_SRC, "libalias.so")?;
    let main_o = dir.join("main.o");
    let start_o = dir.join("start.o");
    compile(main_src, &main_o, PicMode::NoPic)
        .expect("host clang compiles -fno-pie");
    assemble(START_SRC, &start_o).expect("host clang assembles start.S");
    let prog = dir.join("prog");
    link_dyn_exec(
        &[main_o, start_o, lib],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-exec link must succeed");
    Some(prog)
}

/// System-linker comparison. The same non-PIC inputs linked with `clang
/// -no-pie` also emit `R_X86_64_COPY` and run to 6, confirming xold matches
/// the canonical mechanism for an absolute data import.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn copy_relocation_matches_the_system_linker() {
    let Some(clang) = which("clang") else {
        eprintln!("skipping copy-reloc comparison: clang unavailable");
        return;
    };
    let _ = clang;
    let Some(interp) = interpreter() else {
        eprintln!("skipping copy-reloc comparison: interpreter unknown");
        return;
    };
    let dir = workdir("cmp");
    let Some(lib) = build_lib(&dir, LIB_SRC, "libcopy.so") else {
        return;
    };
    let main_o = dir.join("copyrel_cmp_main.o");
    let start_o = dir.join("copyrel_cmp_start.o");
    compile(MAIN_SRC, &main_o, PicMode::NoPic)
        .expect("host clang compiles -fno-pie");
    assemble(START_SRC, &start_o).expect("host clang assembles start.S");

    let xold_prog = dir.join("copyrel_cmp_xold");
    link_dyn_exec(
        &[main_o.clone(), start_o.clone(), lib.clone()],
        &xold_prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-exec link must succeed");

    // System linker on the same non-PIC inputs.
    let sys_prog = dir.join("copyrel_cmp_sys");
    let sys_ok = Command::new("clang")
        .args([
            "-no-pie",
            "-nostdlib",
            "-nodefaultlibs",
            "-Wl,--dynamic-linker",
            "-Wl,/lib64/ld-linux-x86-64.so.2",
        ])
        .arg(&main_o)
        .arg(&start_o)
        .arg(&lib)
        .arg("-o")
        .arg(&sys_prog)
        .status()
        .is_ok_and(|s| s.success());
    if !sys_ok {
        eprintln!("skipping copy-reloc comparison: system linker unavailable");
        cleanup(&[&main_o, &start_o, &lib, &xold_prog]);
        return;
    }

    let xold_status = Command::new(&xold_prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("xold prog runnable");
    let sys_status = Command::new(&sys_prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("system prog runnable");
    assert_eq!(xold_status.code(), Some(6), "xold executable must reach 6");
    assert_eq!(sys_status.code(), Some(6), "system executable must reach 6");

    // Both linkers emit a COPY relocation for `counter`.
    let xold_rela = rela_dyn_rows(&fs::read(&xold_prog).expect("read xold"));
    let sys_rela = rela_dyn_rows(&fs::read(&sys_prog).expect("read sys"));
    assert!(
        xold_rela.iter().any(|r| r.2 == R_X86_64_COPY),
        "xold must emit R_X86_64_COPY for counter"
    );
    assert!(
        sys_rela.iter().any(|r| r.2 == R_X86_64_COPY),
        "system linker must emit R_X86_64_COPY for counter"
    );

    cleanup(&[&main_o, &start_o, &lib, &xold_prog, &sys_prog]);
}

// --- helpers ---------------------------------------------------------------

/// A decoded `.dynsym` row of interest.
struct DynRow {
    value: u64,
    size: u64,
    shndx: u16,
    sym_type: u8,
}

/// Decodes `.rela.dyn` into `(r_offset, sym, r_type, r_addend)` rows.
fn rela_dyn_rows(bytes: &[u8]) -> Vec<(u64, u32, u32, i64)> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(rela) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".rela.dyn")
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(rela) else {
        return Vec::new();
    };
    decode_rela(data)
}

/// Decodes a `.rela*` byte block into rows.
fn decode_rela(data: &[u8]) -> Vec<(u64, u32, u32, i64)> {
    let mut out = Vec::new();
    for chunk in data.chunks(24) {
        if chunk.len() < 24 {
            break;
        }
        let off = u64::from_le_bytes(chunk[..8].try_into().unwrap_or([0; 8]));
        let info =
            u64::from_le_bytes(chunk[8..16].try_into().unwrap_or([0; 8]));
        let add =
            i64::from_le_bytes(chunk[16..24].try_into().unwrap_or([0; 8]));
        out.push((
            off,
            u32::try_from(info >> 32).unwrap_or(0),
            u32::try_from(info & 0xffff_ffff).unwrap_or(0),
            add,
        ));
    }
    out
}

/// Looks up `name` in `.dynsym`, returning its decoded fields.
fn dynsym_lookup(bytes: &[u8], name: &[u8]) -> Option<DynRow> {
    let obj = ObjectFile::parse(bytes).ok()?;
    for sec in obj.sections() {
        if obj.section_name(sec) != b".dynsym" {
            continue;
        }
        let Ok(data) = obj.section_data(sec) else {
            return None;
        };
        for chunk in data.chunks(24) {
            if chunk.len() < 24 {
                break;
            }
            let st_name =
                u32::from_le_bytes(chunk[..4].try_into().unwrap_or([0; 4]));
            let st_info = chunk[4];
            let st_shndx =
                u16::from_le_bytes(chunk[6..8].try_into().unwrap_or([0; 2]));
            let st_value =
                u64::from_le_bytes(chunk[8..16].try_into().unwrap_or([0; 8]));
            let st_size =
                u64::from_le_bytes(chunk[16..24].try_into().unwrap_or([0; 8]));
            let sym_name = dynstr_name(bytes, sec.sh_link.get(), st_name);
            if sym_name == name {
                return Some(DynRow {
                    value: st_value,
                    size: st_size,
                    shndx: st_shndx,
                    sym_type: st_info & 0x0f,
                });
            }
        }
    }
    None
}

/// The `.dynsym` index of `name`, which is what a `.rela.dyn` row references.
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

/// Whether any 8-byte slot of `.got` holds `value`.
///
/// A fixed-base image resolves its GOT at link time, so the slot's contents
/// are the assertion; there is no dynamic relocation left to read.
fn got_holds(bytes: &[u8], value: u64) -> bool {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return false;
    };
    let Some(sec) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".got")
    else {
        return false;
    };
    let Ok(data) = obj.section_data(sec) else {
        return false;
    };
    data.chunks(8).any(|c| {
        c.try_into()
            .map(u64::from_le_bytes)
            .is_ok_and(|slot| slot == value)
    })
}

/// Reads the `.bss` section-header index.
fn bss_shndx(bytes: &[u8]) -> u16 {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return 0;
    };
    for (i, sec) in obj.sections().iter().enumerate() {
        if obj.section_name(sec) == b".bss" {
            return u16::try_from(i).unwrap_or(0);
        }
    }
    0
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

/// Removes `paths` if they exist (best effort; test artifacts under the temp
/// dir are not worth failing a run over).
fn cleanup(paths: &[&Path]) {
    for p in paths {
        let _ = fs::remove_file(p);
    }
}
