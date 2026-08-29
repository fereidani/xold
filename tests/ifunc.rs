//! Indirect functions (`STT_GNU_IFUNC`) in a dynamic image.
//!
//! An indirect function is not at the address its symbol names: that address
//! is a resolver the runtime calls once, and the function is whatever it
//! returns. The linker therefore gives the symbol a PLT stub, a `.got.plt`
//! slot for the stub to jump through, and an `R_X86_64_IRELATIVE` naming the
//! slot and carrying the resolver as its addend. A static image applies those
//! itself before `main`; a dynamic image hands them to the loader in
//! `.rela.plt`, beside the `JUMP_SLOT` entries for its imports.
//!
//! Only running the program proves any of it. Every way of getting this wrong
//! -- naming the implementation instead of the resolver, exporting the
//! resolver's address, leaving the type `STT_GNU_IFUNC` on a symbol that now
//! points at a stub -- produces a well-formed image whose calls land somewhere
//! plausible and return the wrong thing.
//!
//! Gated on `clang` and the host `ld.so`; if either is missing the tests print
//! a note and return, so the build never fails over a missing toolchain.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{
    elf::{ObjectFile, Sym64, constants::STT_FUNC},
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
    reloc::x86_64::{R_X86_64_IRELATIVE, R_X86_64_JUMP_SLOT},
};

mod common;

/// An indirect function and the pieces it is made of. `resolve` is global so
/// the tests can read its address out of the symbol table and check that the
/// `IRELATIVE` addend names it, rather than the implementation or the stub.
const IFUNC_DEF: &str = "int impl(int x) { return x * 2; }\n\
    void *resolve(void) { return (void *)impl; }\n\
    int doubler(int) __attribute__((ifunc(\"resolve\")));\n";

/// A freestanding `_start` that calls `main` and exits with what it returned
/// (syscall 60), so the test needs no C runtime.
const START_SRC: &str = "    .text\n    .globl _start\n_start:\n    \
    call    main\n    movl    %eax, %edi\n    movl    $60, %eax\n    syscall\n";

/// The shared library `bump` comes from: an ordinary exported function, so the
/// executable that calls it takes a `JUMP_SLOT` beside its own `IRELATIVE`.
const LIB_BUMP_SRC: &str =
    "int counter = 5;\nint bump(void) { counter += 1; return counter; }\n";

/// The shared object under test: an indirect function it exports, plus a
/// function of its own that calls it.
const LIB_IFUNC_SRC: &str = "int call_doubler(int x) { return doubler(x); }\n";

/// The soname of the shared object the caller binds against.
const IFUNC_SONAME: &[u8] = b"libxifunc.so";

/// Compiles `src` into `obj`. `pic` selects `-fPIC` (a shared object) over
/// `-fPIE` (an executable). `None` when clang is unavailable.
fn compile(src: &str, obj: &Path, pic: bool) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let model = if pic { "-fPIC" } else { "-fPIE" };
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", model, "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// Assembles the freestanding entry stub into `obj`.
fn assemble_start(obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("S");
    fs::write(&src_path, START_SRC).expect("write assembly");
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

/// A fresh per-test working directory, so tests running in parallel keep
/// their shared objects (which the loader finds by soname) apart.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xold_ifunc_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// One decoded relocation: `(r_offset, symbol index, type, addend)`.
type Reloc = (u64, u32, u32, i64);

/// Decodes the named `.rela*` section, or an empty list when it is absent.
fn rela_rows(bytes: &[u8], name: &[u8]) -> Vec<Reloc> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(sec) = obj.sections().iter().find(|s| obj.section_name(s) == name)
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(sec) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for chunk in data.as_chunks::<24>().0 {
        let info = read_u64(chunk, 8);
        out.push((
            read_u64(chunk, 0),
            u32::try_from(info >> 32).unwrap_or(0),
            u32::try_from(info & 0xffff_ffff).unwrap_or(0),
            read_u64(chunk, 16).cast_signed(),
        ));
    }
    out
}

/// The `(address, size)` of a named output section, or `None` when absent.
fn section_span(bytes: &[u8], name: &[u8]) -> Option<(u64, u64)> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let sec = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == name)?;
    Some((sec.sh_addr.get(), sec.sh_size.get()))
}

/// The `.symtab` entry named `name`, or `None` when the image has none.
fn static_symbol(bytes: &[u8], name: &[u8]) -> Option<Sym64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab.syms.iter().find(|s| symtab.name(s) == name).copied()
}

/// The `.dynsym` entry named `name`, or `None` when the image has none.
fn dynamic_symbol(bytes: &[u8], name: &[u8]) -> Option<Sym64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.dynamic_symbols().ok().flatten()?;
    symtab.syms.iter().find(|s| symtab.name(s) == name).copied()
}

fn read_u64(bytes: &[u8], at: usize) -> u64 {
    let mut buf = [0u8; 8];
    if let Some(slot) = bytes.get(at..at + 8) {
        buf.copy_from_slice(slot);
    }
    u64::from_le_bytes(buf)
}

/// Asserts that `rows` holds exactly one `IRELATIVE`, that it names a slot in
/// `.got.plt` and that its addend is the address of `resolve`.
fn assert_one_irelative(bytes: &[u8], rows: &[Reloc]) {
    let irel: Vec<&Reloc> =
        rows.iter().filter(|r| r.2 == R_X86_64_IRELATIVE).collect();
    assert_eq!(
        irel.len(),
        1,
        "one indirect function must produce exactly one IRELATIVE"
    );
    let &&(offset, sym, _, addend) = irel.first().expect("checked above");
    assert_eq!(sym, 0, "an IRELATIVE names no symbol");
    let (got_plt, size) =
        section_span(bytes, b".got.plt").expect(".got.plt must exist");
    assert!(
        offset >= got_plt && offset < got_plt + size,
        "IRELATIVE offset {offset:#x} must name a .got.plt slot \
         ({got_plt:#x}..{:#x})",
        got_plt + size
    );
    let resolve = static_symbol(bytes, b"resolve")
        .expect("the resolver must be in .symtab");
    assert_eq!(
        addend.cast_unsigned(),
        resolve.st_value.get(),
        "the IRELATIVE addend must be the resolver's address"
    );
}

/// The end-to-end proof for an executable: an indirect function called
/// directly and through a function pointer, in an image the loader relocates.
/// `doubler(16) + p(5)` is 42 only if both reach the implementation.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_indirect_function_runs_in_a_dynamic_executable() {
    let Some(interp) = interpreter() else {
        eprintln!("skipping ifunc exec test: interpreter path unknown");
        return;
    };
    let dir = workdir("exec");
    let main_o = dir.join("main.o");
    let start_o = dir.join("start.o");
    let src = format!(
        "{IFUNC_DEF}int main(void) {{ int (*p)(int) = doubler; \
         return doubler(16) + p(5); }}\n"
    );
    let (Some(()), Some(())) =
        (compile(&src, &main_o, false), assemble_start(&start_o))
    else {
        eprintln!("skipping ifunc exec test: clang unavailable");
        return;
    };

    let prog = dir.join("prog");
    link_dyn_exec(
        &[main_o, start_o],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    )
    .expect("a dynamic executable defining an ifunc must link");

    let bytes = fs::read(&prog).expect("read output");
    assert_one_irelative(&bytes, &rela_rows(&bytes, b".rela.plt"));

    let status = Command::new(&prog).status().expect("program must run");
    assert_eq!(
        status.code(),
        Some(42),
        "doubler(16) + p(5) must be 42: both calls reach the implementation"
    );
}

/// The two kinds of `.rela.plt` entry in one image: an `IRELATIVE` for the
/// indirect function the executable defines and a `JUMP_SLOT` for the function
/// it imports. The choice is per entry, so an image that needs both must carry
/// both. `doubler(18) + bump()` is 42.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_dynamic_executable_carries_irelative_beside_jump_slot() {
    let Some(interp) = interpreter() else {
        eprintln!("skipping mixed-rela test: interpreter path unknown");
        return;
    };
    let dir = workdir("mixed");
    let lib_o = dir.join("libbump.o");
    let main_o = dir.join("main.o");
    let start_o = dir.join("start.o");
    let src = format!(
        "{IFUNC_DEF}int bump(void);\n\
         int main(void) {{ return doubler(18) + bump(); }}\n"
    );
    let (Some(()), Some(()), Some(())) = (
        compile(LIB_BUMP_SRC, &lib_o, true),
        compile(&src, &main_o, false),
        assemble_start(&start_o),
    ) else {
        eprintln!("skipping mixed-rela test: clang unavailable");
        return;
    };
    let lib = dir.join("libbump.so");
    link_shared(
        std::slice::from_ref(&lib_o),
        &lib,
        Some(b"libbump.so"),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold -shared link must succeed");

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
    .expect("a dynamic executable with an ifunc and an import must link");

    let bytes = fs::read(&prog).expect("read output");
    let rows = rela_rows(&bytes, b".rela.plt");
    assert_one_irelative(&bytes, &rows);
    assert_eq!(
        rows.iter().filter(|r| r.2 == R_X86_64_JUMP_SLOT).count(),
        1,
        "the imported `bump` must keep its JUMP_SLOT: {rows:?}"
    );

    let status = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("program must run");
    assert_eq!(
        status.code(),
        Some(42),
        "doubler(18) + bump() must be 42: the ifunc and the import both bind"
    );
}

/// What a shared object publishes for an indirect function it exports: the
/// address of the stub, as a plain `STT_FUNC`.
///
/// The type is the load-bearing part. A loader binding another image to an
/// `STT_GNU_IFUNC` definition calls the definition's value and takes what
/// comes back for the address; leaving the type on a symbol that now points at
/// a stub would have it call the stub with no arguments and use the doubled
/// garbage as a function pointer.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_shared_object_publishes_an_indirect_function_as_its_stub() {
    let dir = workdir("export");
    let Some(lib) = build_ifunc_lib(&dir) else {
        eprintln!("skipping ifunc export test: clang unavailable");
        return;
    };
    let bytes = fs::read(&lib).expect("read output");
    assert_one_irelative(&bytes, &rela_rows(&bytes, b".rela.plt"));

    let doubler =
        dynamic_symbol(&bytes, b"doubler").expect("doubler must be exported");
    assert_eq!(
        doubler.type_(),
        STT_FUNC,
        "an exported indirect function is published as a plain function"
    );
    let (plt, size) = section_span(&bytes, b".plt").expect(".plt must exist");
    let value = doubler.st_value.get();
    assert!(
        value >= plt && value < plt + size,
        "doubler must be published at its stub ({plt:#x}..{:#x}), got \
         {value:#x}",
        plt + size
    );
    let resolve = dynamic_symbol(&bytes, b"resolve")
        .expect("the resolver is exported too");
    assert_ne!(
        value,
        resolve.st_value.get(),
        "the resolver's address must not be published as the function"
    );
}

/// The end-to-end proof for a shared object: a program linked against it by
/// the host toolchain calls the indirect function directly and through the
/// object's own wrapper. `doubler(16) + call_doubler(5)` is 42 only if the
/// outside call binds to something that doubles, and the inside call reaches
/// the implementation through the stub.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_caller_binds_to_a_shared_object_s_indirect_function() {
    let Some(clang) = which("clang") else {
        eprintln!("skipping ifunc caller test: clang unavailable");
        return;
    };
    let dir = workdir("caller");
    let Some(lib) = build_ifunc_lib(&dir) else {
        eprintln!("skipping ifunc caller test: clang unavailable");
        return;
    };
    let caller_src = "int doubler(int);\nint call_doubler(int);\n\
         int main(void) { return doubler(16) + call_doubler(5); }\n";
    if !cfg!(target_os = "linux") {
        assert_foreign_caller_contract(&dir, &lib, caller_src);
        return;
    }
    let caller_c = dir.join("caller.c");
    fs::write(&caller_c, caller_src).expect("write source");
    let caller = dir.join("caller");
    let linked = Command::new(clang)
        .arg(&caller_c)
        .arg("-o")
        .arg(&caller)
        .arg(&lib)
        .arg(format!("-Wl,-rpath,{}", dir.display()))
        .status()
        .expect("clang runs")
        .success();
    assert!(
        linked,
        "the host toolchain must link against the xold library"
    );

    let status = Command::new(&caller).status().expect("program must run");
    assert_eq!(
        status.code(),
        Some(42),
        "doubler(16) + call_doubler(5) must be 42: the caller binds to the \
         implementation, not to the resolver"
    );
}

/// Builds the same ELF caller with xold when the host toolchain cannot link or
/// execute ELF, then verifies both external calls bind through `JUMP_SLOT`
/// rows to the shared object's plain-function exports. The library's own call
/// keeps its IRELATIVE-backed stub.
fn assert_foreign_caller_contract(dir: &Path, lib: &Path, caller_src: &str) {
    let caller_o = dir.join("caller.o");
    let start_o = dir.join("start.o");
    compile(caller_src, &caller_o, false).expect("compile ELF caller");
    assemble_start(&start_o).expect("assemble ELF entry");
    let caller = dir.join("caller");
    link_dyn_exec(
        &[caller_o, start_o, lib.to_path_buf()],
        &caller,
        b"_start",
        b"/lib64/ld-linux-x86-64.so.2",
        false,
        IcfMode::None,
        false,
    )
    .expect("xold links the ELF caller against the xold library");

    let caller_bytes = fs::read(&caller).expect("read caller");
    let mut imports: Vec<Vec<u8>> = rela_rows(&caller_bytes, b".rela.plt")
        .iter()
        .filter(|row| row.2 == R_X86_64_JUMP_SLOT)
        .filter_map(|row| dynamic_symbol_name(&caller_bytes, row.1))
        .collect();
    imports.sort_unstable();
    assert_eq!(
        imports,
        vec![b"call_doubler".to_vec(), b"doubler".to_vec()],
        "the caller binds both external calls through PLT slots"
    );

    let lib_bytes = fs::read(lib).expect("read ifunc library");
    assert_one_irelative(&lib_bytes, &rela_rows(&lib_bytes, b".rela.plt"));
    for name in [b"doubler".as_slice(), b"call_doubler"] {
        let sym = dynamic_symbol(&lib_bytes, name).expect("library export");
        assert_eq!(
            sym.type_(),
            STT_FUNC,
            "{name:?} is a plain callable export"
        );
    }
    let doubler = dynamic_symbol(&lib_bytes, b"doubler").unwrap();
    let (plt, size) = section_span(&lib_bytes, b".plt").expect("library PLT");
    assert!(
        doubler.st_value.get() >= plt && doubler.st_value.get() < plt + size,
        "the caller binds doubler to its IRELATIVE-backed stub"
    );
}

fn dynamic_symbol_name(bytes: &[u8], index: u32) -> Option<Vec<u8>> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let table = obj.dynamic_symbols().ok()??;
    let sym = table.syms.get(usize::try_from(index).ok()?)?;
    Some(table.name(sym).to_vec())
}

/// Builds the shared object under test with `xold -shared`, returning its
/// path. `None` when clang cannot produce the input object.
fn build_ifunc_lib(dir: &Path) -> Option<PathBuf> {
    let obj = dir.join("libxifunc.o");
    let src = format!("{IFUNC_DEF}{LIB_IFUNC_SRC}");
    compile(&src, &obj, true)?;
    let lib = dir.join("libxifunc.so");
    link_shared(
        std::slice::from_ref(&obj),
        &lib,
        Some(IFUNC_SONAME),
        false,
        IcfMode::None,
        false,
    )
    .expect("a shared object defining an ifunc must link");
    Some(lib)
}
