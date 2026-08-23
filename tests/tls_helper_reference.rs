//! A reference to `__tls_get_addr` that is not part of a TLS sequence.
//!
//! Lowering a general-dynamic or local-dynamic sequence consumes the
//! `call __tls_get_addr` that closes it, so the relocation on that call must
//! claim neither a PLT entry nor a GOT slot. xold decided that from the
//! symbol's *name* alone, which is not what distinguishes the two: the call
//! that closes a sequence is an ordinary call to an ordinary symbol, and only
//! the sequence beside it says it was swallowed.
//!
//! So every other reference to the helper was dropped too -- hand-written
//! assembly, a static libc that both defines and calls it, code taking its
//! address. Those sites kept the input's placeholder bytes: a `GOTPCREL`
//! whose displacement was computed from storage nothing allocated (`lea`
//! pointing at itself), and a `PLT32` call to address zero. The link
//! succeeded and the program jumped into the weeds.
//!
//! The question now goes to the sequence, as
//! `crate::reloc::Arch::relax_span` asks it in the writer: the reference is
//! consumed only if it sits inside the bytes the relocation before it would
//! swallow. lld draws the same line, consuming exactly the relocation that
//! follows a relaxed `TLSGD`/`TLSLD` (`getTlsGdRelaxSkip`) and giving a
//! standalone helper reference its normal GOT and PLT treatment.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{interpreter, which};
use xold::{
    elf::ObjectFile,
    icf::IcfMode,
    linker::link_dyn_exec,
    reloc::x86_64::{R_X86_64_GLOB_DAT, R_X86_64_JUMP_SLOT},
};

mod common;

/// Two standalone references to the helper: one takes its address through the
/// GOT, one calls it through the PLT. Neither closes a TLS sequence, so both
/// are ordinary references.
const STRAY_SRC: &[u8] = b"    .text\n\
    .globl take_helper\n\
    .type take_helper,@function\n\
take_helper:\n\
    leaq    __tls_get_addr@GOTPCREL(%rip), %rax\n\
    retq\n\
    .size take_helper, .-take_helper\n\
    .globl call_helper\n\
    .type call_helper,@function\n\
call_helper:\n\
    jmp     __tls_get_addr@PLT\n\
    .size call_helper, .-call_helper\n";

/// The program: the helper's address must be a real one, not this
/// instruction's own.
const MAIN_SRC: &[u8] = b"extern void *take_helper(void);\n\
    int main(void)\n\
    {\n\
        void *helper = take_helper();\n\
        /* The bug left the `lea` with a zero displacement, so the answer was\n\
           the address of the instruction after it -- inside `take_helper`\n\
           itself. */\n\
        if (helper == 0) { return 1; }\n\
        if (helper >= (void *)take_helper\n\
            && helper < (void *)((char *)take_helper + 16)) { return 2; }\n\
        return 0;\n\
    }\n";

/// A standalone reference keeps its GOT slot and its PLT entry.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_standalone_helper_reference_keeps_its_entries() {
    let Some(dir) = workdir("entries") else {
        return;
    };
    let Some(prog) = link(&dir) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");
    let dyn_names = relocated_names(&bytes, b".rela.dyn", R_X86_64_GLOB_DAT);
    assert!(
        dyn_names.iter().any(|n| n == b"__tls_get_addr"),
        "the address-of reference needs a GOT slot the loader fills; got \
         {dyn_names:?}"
    );
    let plt_names = relocated_names(&bytes, b".rela.plt", R_X86_64_JUMP_SLOT);
    assert!(
        plt_names.iter().any(|n| n == b"__tls_get_addr"),
        "the call needs a PLT entry the loader binds; got {plt_names:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The behaviour: the address the program reads is the helper's, not its own
/// instruction's.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_helpers_address_is_the_helpers() {
    let Some(dir) = workdir("run") else {
        return;
    };
    let Some(prog) = link(&dir) else {
        return;
    };
    let code = Command::new(&prog)
        .status()
        .expect("linked program must be runnable")
        .code();
    assert_eq!(
        code,
        Some(0),
        "1 means the reference resolved to zero, 2 means it resolved into the \
         instruction that reads it -- both are the placeholder bytes reaching \
         the image"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping tls-helper-reference {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_tlshelper_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the inputs and links them into a runnable dynamic executable
/// against the host libc and the loader, which is where `__tls_get_addr`
/// actually lives: since glibc 2.34 `libc.so.6` names it undefined and
/// reaches it through `ld.so` like everyone else.
fn link(dir: &Path) -> Option<PathBuf> {
    let Some(interp) = interpreter() else {
        eprintln!("skipping tls-helper-reference: interpreter path unknown");
        return None;
    };
    let stray = dir.join("stray.o");
    let main_o = dir.join("straymain.o");
    build(STRAY_SRC, "S", &stray)?;
    build(MAIN_SRC, "c", &main_o)?;
    let start = common::crt_file("Scrt1.o")?;
    let prologue = common::crt_file("crti.o")?;
    let epilogue = common::crt_file("crtn.o")?;
    let libc = common::libc_so()?;
    let loader = common::loader_so()?;

    let prog = dir.join("strayprog");
    let res = link_dyn_exec(
        &[start, prologue, main_o, stray, libc, loader, epilogue],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    );
    assert!(
        res.is_ok(),
        "a standalone helper reference must link: {:?}",
        res.err()
    );
    Some(prog)
}

/// Compiles or assembles `src` (written beside `obj` with extension `ext`).
fn build(src: &[u8], ext: &str, obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension(ext);
    fs::write(&src_path, src).ok()?;
    Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIE", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success()
        .then_some(())
}

// --- readers ---------------------------------------------------------------

/// The `.dynstr` names of the symbols `section`'s rows of type `r_type`
/// reference.
fn relocated_names(bytes: &[u8], section: &[u8], r_type: u32) -> Vec<Vec<u8>> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Ok(Some(dynsym)) = obj.dynamic_symbols() else {
        return Vec::new();
    };
    let Some(shdr) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == section)
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
            if info as u32 != r_type {
                return None;
            }
            let sym = usize::try_from(info >> 32).ok()?;
            dynsym.syms.get(sym).map(|s| dynsym.name(s).to_vec())
        })
        .collect()
}
