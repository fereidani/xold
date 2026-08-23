//! What a dynamic executable puts in `.dynsym`.
//!
//! A program is not a library, so it does not publish its symbol table. But it
//! is not opaque either: a `DT_NEEDED` dependency may reference a name the
//! program defines, and the loader has to be able to bind that reference. A
//! library calling back into the program, a program interposing its own
//! `malloc`, a data object a library writes through -- all of them need a
//! defined `.dynsym` row in the executable, or the loader has nothing to bind
//! to and the reference stays unresolved.
//!
//! The rule is not "export everything": that is `--export-dynamic`, a separate
//! request. It is "export what a dependency actually references", which is how
//! lld reaches the same set -- reading a shared file it marks the symbol behind
//! every undefined `.dynsym` row as exported, and the writer emits a row for
//! any symbol so marked.
//!
//! The three tests below cover the rule and both its edges: a referenced
//! definition is exported at the address it lives at, a definition no
//! dependency names is not, and a hidden definition is not even when a
//! dependency does name it. The last one runs the program, because a shared
//! library calling back into the executable is the behaviour; the table is only
//! how it is spelled.
//!
//! Gated on `clang` and the host `ld.so`; if either is missing the tests print
//! a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{
    elf::{ObjectFile, Sym64, constants::SHN_UNDEF},
    icf::IcfMode,
    linker::link_dyn_exec,
};

mod common;

/// The dependency: it calls back into the program, so `prog_value` reaches its
/// `.dynsym` as an undefined row. `lib_read` is what the program calls to get
/// there.
const LIB_SRC: &[u8] = b"extern int prog_value(void);\n\
    int lib_read(void) { return prog_value(); }\n";

/// The same dependency, but reaching back for a name the program keeps hidden.
/// It still defines `lib_read`, so the program links against either one.
const HIDDEN_LIB_SRC: &[u8] = b"extern int hidden_value(void);\n\
    int lib_read(void) { return hidden_value(); }\n";

/// The program: one definition the dependency references, one nothing outside
/// the program names, and one the dependency does name but which is hidden.
/// It returns the sum of the callback and the private global.
const MAIN_SRC: &[u8] = b"extern int lib_read(void);\n\
    int prog_value(void) { return 42; }\n\
    int unreferenced_global = 3;\n\
    __attribute__((visibility(\"hidden\"))) int hidden_value(void) \
    { return 9; }\n\
    int main(void) { return lib_read() + unreferenced_global; }\n";

/// A freestanding `_start` that calls `main` then exits with its return value.
const START_SRC: &[u8] = b"    .text\n    .globl _start\n_start:\n    \
    call    main\n    movl    %eax, %edi\n    movl    $60, %eax\n    syscall\n";

/// What the program returns once the dependency's callback reaches the
/// program's own definition: 42 from `prog_value` plus 3 from the private
/// global.
const EXPECTED_EXIT: i32 = 45;

/// A definition a dependency references is exported, at the address it lives
/// at, and the two names beside it are not: one no dependency mentions, and
/// one the program keeps hidden.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_executable_exports_what_a_dependency_references() {
    let Some(dir) = workdir("refs") else {
        return;
    };
    let Some(prog) = link_program(&dir, LIB_SRC) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");

    let exported = dynsym(&bytes, b"prog_value")
        .expect("a referenced definition must be exported");
    assert_ne!(
        exported.st_shndx.get(),
        SHN_UNDEF,
        "the row must be defined; an undefined one binds nothing"
    );
    let defined = symtab(&bytes, b"prog_value")
        .expect("the definition is in .symtab either way");
    assert_eq!(
        (exported.st_value.get(), exported.st_shndx.get()),
        (defined.st_value.get(), defined.st_shndx.get()),
        "the exported row must name the address the definition sits at"
    );

    assert!(
        dynsym(&bytes, b"unreferenced_global").is_none(),
        "a definition no dependency references is not exported; that is what \
         keeps this from being --export-dynamic"
    );
    assert!(
        symtab(&bytes, b"unreferenced_global").is_some(),
        "it is still in .symtab, so the test is about the ABI and not about \
         the symbol surviving the link"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A hidden definition stays out of `.dynsym` even when a dependency names it.
/// Hidden visibility is a promise that the name is private to the image, so no
/// loader may bind another image to it; lld gives such a symbol `STB_LOCAL`
/// binding for the same reason.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_hidden_definition_is_not_exported_to_a_dependency() {
    let Some(dir) = workdir("hidden") else {
        return;
    };
    let Some(prog) = link_program(&dir, HIDDEN_LIB_SRC) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");
    assert!(
        dynsym(&bytes, b"hidden_value").is_none(),
        "a hidden definition is private to the image whatever a dependency \
         asks for"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The behaviour the table stands for: the dependency's call reaches the
/// program's definition at run time. Without the exported row the loader has
/// nothing to bind `prog_value` to and the program never gets its answer.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_dependency_binds_to_the_executables_definition() {
    let Some(dir) = workdir("run") else {
        return;
    };
    let Some(prog) = link_program(&dir, LIB_SRC) else {
        return;
    };
    let status = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(EXPECTED_EXIT),
        "the library's callback must reach the program's definition"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs at all.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping dep-exports {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_depexports_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds `lib_src` into `libdepcb.so` with the host linker, then links the
/// program against it with xold. Returns the executable's path.
///
/// The dependency is built by the host toolchain rather than by xold: what is
/// under test is how xold reads another linker's `.dynsym`, and the file name
/// is its `DT_NEEDED` name, which is what lets `LD_LIBRARY_PATH` point the
/// loader at `dir`.
fn link_program(dir: &Path, lib_src: &[u8]) -> Option<PathBuf> {
    let Some(interp) = interpreter() else {
        eprintln!("skipping dep-exports: interpreter path unknown");
        return None;
    };
    let lib = build_dependency(dir, lib_src)?;
    let main_o = dir.join("depexports_main.o");
    let start_o = dir.join("depexports_start.o");
    compile(MAIN_SRC, "c", &main_o, &["-fPIE"])?;
    compile(START_SRC, "S", &start_o, &[])?;

    let prog = dir.join("depexports_prog");
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
        "xold dynamic-exec link must succeed: {:?}",
        linked.err()
    );
    Some(prog)
}

/// Builds the shared dependency from `src` with the host toolchain. `None`
/// when the host cannot build it.
fn build_dependency(dir: &Path, src: &[u8]) -> Option<PathBuf> {
    let clang = which("clang")?;
    let obj = dir.join("depexports_lib.o");
    compile(src, "c", &obj, &["-fPIC"])?;
    let so = dir.join("libdepcb.so");
    let ok = Command::new(clang)
        .arg("-shared")
        .arg("-o")
        .arg(&so)
        .arg(&obj)
        .arg("-Wl,-soname,libdepcb.so")
        .arg("-Wl,--allow-shlib-undefined")
        .status()
        .ok()?
        .success();
    ok.then_some(so)
}

/// Compiles or assembles `src` (written beside `obj` with extension `ext`)
/// with the host clang, passing `args` through.
fn compile(src: &[u8], ext: &str, obj: &Path, args: &[&str]) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension(ext);
    fs::write(&src_path, src).ok()?;
    Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c"])
        .args(args)
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success()
        .then_some(())
}

/// The `.dynsym` row named `name`, or `None` when the image does not export it.
fn dynsym(bytes: &[u8], name: &[u8]) -> Option<Sym64> {
    find_symbol(bytes, name, true)
}

/// The `.symtab` row named `name`, or `None` when the link kept no such symbol.
fn symtab(bytes: &[u8], name: &[u8]) -> Option<Sym64> {
    find_symbol(bytes, name, false)
}

/// The row named `name` in `.dynsym` (when `dynamic`) or in `.symtab`.
fn find_symbol(bytes: &[u8], name: &[u8], dynamic: bool) -> Option<Sym64> {
    let obj = ObjectFile::parse(bytes).expect("output must be valid ELF");
    let table = if dynamic {
        obj.dynamic_symbols()
    } else {
        obj.symbol_table()
    }
    .ok()
    .flatten()?;
    table.syms.iter().find(|s| table.name(s) == name).copied()
}
