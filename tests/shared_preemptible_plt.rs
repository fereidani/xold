//! A shared object's call to its own exported function goes through the PLT.
//!
//! Every default-visibility definition a shared object exports is
//! preemptible: something loaded earlier -- another library, the executable,
//! an `LD_PRELOAD` -- may supply the same name, and then that definition is
//! the one the whole process must use, this library included.
//!
//! xold allocated a PLT stub only for an *undefined* symbol, so a library's
//! internal call to one of its own exports collapsed to a direct PC-relative
//! branch. The result is `-Bsymbolic-functions` semantics nobody asked for,
//! and it is asymmetric: an interposing definition replaced the function for
//! every other image but not for this one, while this image's *data* accesses
//! went through `GLOB_DAT` and did see the replacement. Two halves of one
//! library disagreeing about which definition is live is the kind of thing
//! that shows up as a bug in the program, not in the linker.
//!
//! lld gives every preemptible symbol `NEEDS_PLT` in `processAux`, which is
//! the same rule: a stub is needed exactly when this link cannot say what the
//! call will reach.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::ObjectFile, icf::IcfMode, linker::link_shared,
    reloc::x86_64::R_X86_64_JUMP_SLOT,
};

mod common;

/// The library: an exported function and an internal caller of it.
const LIB_SRC: &[u8] = b"int hookable(void) { return 1; }\n\
    int caller(void) { return hookable(); }\n";

/// The interposing definition, built with the host toolchain.
const HOOK_SRC: &[u8] = b"int hookable(void) { return 42; }\n";

/// The harness prints the internal call's answer and the external one's. They
/// have to agree.
const MAIN_SRC: &[u8] = b"#include <stdio.h>\n\
    extern int caller(void);\n\
    extern int hookable(void);\n\
    int main(void)\n\
    {\n\
        printf(\"%d %d\\n\", caller(), hookable());\n\
        return 0;\n\
    }\n";

/// The exported definition gets a `JUMP_SLOT`, because the call to it has to
/// go somewhere the loader can redirect.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_exported_definition_gets_a_jump_slot() {
    let Some(dir) = workdir("slot") else {
        return;
    };
    let Some(lib) = link_library(&dir) else {
        return;
    };
    let bytes = fs::read(&lib).expect("read shared object");
    let names = jump_slot_names(&bytes);
    assert!(
        names.iter().any(|n| n == b"hookable"),
        "a call to a preemptible definition needs a stub the loader binds; \
         the JUMP_SLOT rows are {names:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The behaviour behind the row: an interposing definition replaces the
/// function for the library's own call too.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_interposing_definition_reaches_the_internal_call() {
    let Some(dir) = workdir("interpose") else {
        return;
    };
    let Some(lib) = link_library(&dir) else {
        return;
    };
    let Some(prog) = build_harness(&dir, &lib) else {
        return;
    };
    let plain = run(&prog, None);
    assert_eq!(plain, "1 1\n", "without interposition both calls see 1");
    let hook = dir.join("libhook.so");
    let Some(()) = build_hook(&dir, &hook) else {
        return;
    };
    let preloaded = run(&prog, Some(&hook));
    assert_eq!(
        preloaded, "42 42\n",
        "both calls must reach the interposing definition; \"1 42\" is the \
         defect -- the library kept calling its own copy while everyone else \
         got the replacement"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping shared-preemptible-plt {prefix}: no clang");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_preempt_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the library and links it with `xold -shared`.
fn link_library(dir: &Path) -> Option<PathBuf> {
    let obj = compile(dir, LIB_SRC, "lib")?;
    let lib = dir.join("libpreempt.so");
    let res = link_shared(
        std::slice::from_ref(&obj),
        &lib,
        Some(b"libpreempt.so"),
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    Some(lib)
}

/// Compiles one position-independent object.
fn compile(dir: &Path, src: &[u8], stem: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src_path = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIC", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping shared-preemptible-plt: clang cannot build it");
        return None;
    }
    Some(obj)
}

/// Builds the harness against the produced library with the host toolchain.
fn build_harness(dir: &Path, lib: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src = dir.join("main.c");
    fs::write(&src, MAIN_SRC).ok()?;
    let bin = dir.join("harness");
    Command::new(clang)
        .arg("-fPIE")
        .arg(&src)
        .arg(lib)
        .arg("-o")
        .arg(&bin)
        .arg("-Wl,-rpath")
        .arg(dir)
        .status()
        .ok()?
        .success()
        .then_some(bin)
}

/// Builds the interposing library with the host toolchain.
fn build_hook(dir: &Path, out: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src = dir.join("hook.c");
    fs::write(&src, HOOK_SRC).ok()?;
    Command::new(clang)
        .args(["-fPIC", "-shared"])
        .arg(&src)
        .arg("-o")
        .arg(out)
        .status()
        .ok()?
        .success()
        .then_some(())
}

/// Runs the harness, optionally with a preloaded library, and returns stdout.
fn run(prog: &Path, preload: Option<&Path>) -> String {
    let mut cmd = Command::new(prog);
    if let Some(lib) = preload {
        cmd.env("LD_PRELOAD", lib);
    }
    let out = cmd.output().expect("harness must run");
    assert!(out.status.success(), "harness must exit 0");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

// --- readers ---------------------------------------------------------------

/// The `.dynstr` names of the symbols `.rela.plt`'s `JUMP_SLOT` rows name.
fn jump_slot_names(bytes: &[u8]) -> Vec<Vec<u8>> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Ok(Some(dynsym)) = obj.dynamic_symbols() else {
        return Vec::new();
    };
    let Some(shdr) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".rela.plt")
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(shdr) else {
        return Vec::new();
    };
    data.as_chunks::<24>()
        .0
        .iter()
        .filter_map(|c| {
            let info = u64::from_le_bytes(<[u8; 8]>::try_from(&c[8..16]).ok()?);
            #[allow(clippy::cast_possible_truncation)]
            if info as u32 != R_X86_64_JUMP_SLOT {
                return None;
            }
            let sym = usize::try_from(info >> 32).ok()?;
            dynsym.syms.get(sym).map(|s| dynsym.name(s).to_vec())
        })
        .collect()
}
