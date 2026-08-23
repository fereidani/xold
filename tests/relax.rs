//! End-to-end tests for x86-64 `GOTPCRELX`/`REX_GOTPCRELX` relaxation.
//!
//! A program references a locally-defined global through a real
//! `R_X86_64_REX_GOTPCRELX` site (the compiler emits the GOT-indirect form
//! only when the symbol's definition is in another translation unit, so the
//! definition lives in a separate object). The relaxed link must rewrite the
//! `mov sym@GOTPCREL(%rip), %reg` site into `lea sym(%rip), %reg` (opcode
//! `0x8b` -> `0x8d`), and both links must run identically.
//!
//! Gated on `clang`, `gcc` (for the crt objects), and the system `ld.so`; if
//! absent the tests print a note and return, so the build never fails over a
//! missing toolchain.
//!
//! The tests at the end of the file need no toolchain: they drive the
//! relaxation hook directly over a byte window and a stub resolver, which is
//! how the out-of-range reject path is reached without building a 2 GiB
//! image.

// `crt1`/`crti`/`crtn` are the canonical names of the crt objects; renaming
// them would obscure the test.
#![allow(clippy::similar_names)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{
    elf::ObjectFile,
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
    reloc::{
        Arch, Resolver,
        x86_64::{R_X86_64_GOTPCRELX, X86_64},
    },
    symbol::SymbolId,
};

mod common;

/// The referencing TU: `&target` is `R_X86_64_REX_GOTPCRELX` because the
/// definition is in another object.
const MAIN_SRC: &[u8] = b"extern int target;\n\
     int printf(const char *, ...);\n\
     int main(void){\n\
         int *p = &target;\n\
         printf(\"%d\\n\", *p);\n\
         return *p != 0x1234;\n\
     }\n";

/// The defining TU: `target` is a locally-defined global with a known value.
const DEF_SRC: &[u8] = b"int target = 0x1234;\n";

/// Compiles `src` (`-fPIE`) to `obj` with the host clang. Returns `None` if
/// clang is unavailable so callers can skip gracefully.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIE", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// A fresh per-test working directory under the system temp dir.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xold_relax_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Whether the host toolchain needed for these tests is present: clang, gcc,
/// the three crt objects, `libc.so.6`, and a probeable interpreter.
struct Harness {
    crt1: PathBuf,
    crti: PathBuf,
    crtn: PathBuf,
    libc: PathBuf,
    interp: Vec<u8>,
}

impl Harness {
    /// Collects the harness, or returns `None` (printing a note) when a piece
    /// is missing.
    fn detect() -> Option<Self> {
        if which("clang").is_none() {
            eprintln!("skipping relax tests: clang unavailable");
            return None;
        }
        let crt1 = crt_file("crt1.o")?;
        let crti = crt_file("crti.o")?;
        let crtn = crt_file("crtn.o")?;
        let libc = libc_so()?;
        let interp = interpreter()?;
        Some(Self {
            crt1,
            crti,
            crtn,
            libc,
            interp,
        })
    }

    /// Links `main_obj` + `def_obj` plus the crt objects and libc into `prog`
    /// with xold, selecting whether relaxation runs.
    fn link(&self, main_obj: &Path, def_obj: &Path, prog: &Path, relax: bool) {
        link_dyn_exec(
            &[
                main_obj.to_path_buf(),
                def_obj.to_path_buf(),
                self.crti.clone(),
                self.crt1.clone(),
                self.crtn.clone(),
                self.libc.clone(),
            ],
            prog,
            b"_start",
            &self.interp,
            false,
            IcfMode::None,
            relax,
        )
        .expect("xold relax link must succeed");
    }
}

/// Finds `<main>:` in the `objdump -d` output and returns it followed by the
/// next few disassembly lines, so the caller can scan the `mov`/`lea` form.
fn disasm_main(prog: &Path) -> String {
    let out = Command::new("objdump")
        .args(["-d", "--no-show-raw-insn"])
        .arg(prog)
        .output()
        .expect("objdump runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut lines = stdout.lines();
    let Some(start) = lines.by_ref().find(|l| l.contains("<main>:")) else {
        return String::new();
    };
    let mut out = String::from(start);
    out.push('\n');
    for l in lines.take(14) {
        if l.is_empty() {
            break;
        }
        out.push_str(l);
        out.push('\n');
    }
    out
}

/// The `.text` bytes of `prog` (the executable's first `PT_LOAD` code
/// segment).
fn text_bytes(prog: &Path) -> Vec<u8> {
    let bytes = fs::read(prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF output");
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".text")
        .expect("output has a .text");
    obj.section_data(shdr).expect("text bytes").to_vec()
}

/// Step 1: a real `REX_GOTPCRELX` site relaxes to `lea` (opcode `0x8d`) under
/// `--relax`, while the unrelaxed link keeps the `mov` (opcode `0x8b`). Both
/// run identically (print `4660`, exit 0).
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn gotpcrelx_relaxes_to_lea_and_runs_identically() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("mov");
    let main_o = dir.join("main.o");
    let def_o = dir.join("def.o");
    compile(MAIN_SRC, &main_o).expect("host clang compiles main");
    compile(DEF_SRC, &def_o).expect("host clang compiles def");

    let prog_relax = dir.join("prog_relax");
    let prog_norelax = dir.join("prog_norelax");
    h.link(&main_o, &def_o, &prog_relax, true);
    h.link(&main_o, &def_o, &prog_norelax, false);

    // Both programs must run identically: print 4660 (0x1234) and exit 0.
    let out_relax = Command::new(&prog_relax)
        .output()
        .expect("relaxed program runs");
    assert!(
        out_relax.status.success(),
        "relaxed program should exit 0 (exit {:?})",
        out_relax.status.code()
    );
    assert_eq!(
        String::from_utf8_lossy(&out_relax.stdout),
        "4660\n",
        "relaxed program prints the symbol's value"
    );
    let out_norelax = Command::new(&prog_norelax)
        .output()
        .expect("unrelaxed program runs");
    assert!(
        out_norelax.status.success(),
        "unrelaxed program should exit 0 (exit {:?})",
        out_norelax.status.code()
    );
    assert_eq!(out_relax.stdout, out_norelax.stdout);
    assert_eq!(out_relax.status.code(), out_norelax.status.code());

    // The relaxed main must contain a `lea` of `target` where the unrelaxed
    // has a `mov` from the GOT. `objdump -d` shows the difference directly.
    let disasm_relax = disasm_main(&prog_relax);
    let disasm_norelax = disasm_main(&prog_norelax);
    assert!(
        disasm_relax.contains("lea") && disasm_relax.contains("target"),
        "relaxed main must `lea target(...(%rip))`; got:\n{disasm_relax}"
    );
    assert!(
        disasm_norelax.contains("mov"),
        "unrelaxed main must still `mov` from the GOT; got:\n{disasm_norelax}"
    );

    // Structural byte check: the relaxed `.text` contains an `8d 05` (lea
    // r64, rip+disp32) sequence; the unrelaxed contains `8b 05` (mov r64,
    // rip+disp32). Both encodings share the ModR/M byte `0x05`.
    let text_relax = text_bytes(&prog_relax);
    let text_norelax = text_bytes(&prog_norelax);
    let lea_seq = [0x48u8, 0x8d, 0x05];
    let mov_seq = [0x48u8, 0x8b, 0x05];
    assert!(
        text_relax.windows(lea_seq.len()).any(|w| w == lea_seq),
        "relaxed .text must contain a REX lea rip+disp32 encoding"
    );
    assert!(
        text_norelax.windows(mov_seq.len()).any(|w| w == mov_seq),
        "unrelaxed .text must contain a REX mov rip+disp32 (GOT) encoding"
    );
    assert!(
        !text_relax.windows(mov_seq.len()).any(|w| w == mov_seq),
        "relaxed .text must not still contain the original REX mov rip+disp32"
    );
}

/// Step 2: the relaxed binary still carries the GOT section (allocated but
/// now unused by the relaxed site), matching the conservative layout choice
/// documented in the relax pass: the win is the relaxed instruction, not the
/// GOT-slot saving.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn relaxed_link_keeps_the_got_section_allocated() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("got");
    let main_o = dir.join("main.o");
    let def_o = dir.join("def.o");
    compile(MAIN_SRC, &main_o).expect("host clang compiles main");
    compile(DEF_SRC, &def_o).expect("host clang compiles def");

    let prog = dir.join("prog");
    h.link(&main_o, &def_o, &prog, true);

    let bytes = fs::read(&prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF output");
    let has_got = obj
        .sections()
        .iter()
        .any(|s| obj.section_name(s) == b".got");
    assert!(
        has_got,
        "relaxed link still allocates the (now unused) GOT section"
    );
}

/// Two GOT loads that differ only in what their target's value is: `abs_sym`
/// is `SHN_ABS` (a constant), `def_sym` lives in `.data` (a place). Both are
/// hidden, so preemptibility cannot tell them apart and only absoluteness can.
///
/// Written in assembly because no C spelling produces an `SHN_ABS` global.
const ABS_SRC: &[u8] = b"    .text\n\
    .globl get_abs\n\
get_abs:\n\
    movq abs_sym@GOTPCREL(%rip), %rax\n\
    ret\n\
    .globl get_def\n\
get_def:\n\
    movq def_sym@GOTPCREL(%rip), %rcx\n\
    ret\n\
    .data\n\
    .globl def_sym\n\
    .hidden def_sym\n\
def_sym:\n\
    .quad 7\n\
    .globl abs_sym\n\
    .hidden abs_sym\n\
    abs_sym = 0x1234\n";

/// Assembles `src` with the host clang, returning `None` when it is absent.
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

/// The GOT load of an absolute symbol must survive `--relax`.
///
/// The slot holds the symbol's value; a `lea` computes it from the program
/// counter, so in a position-independent image the two differ by the load
/// base. The `.data` symbol beside it is folded in the same link, which is
/// what makes this a test of the absolute guard rather than of relaxation
/// being off.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn relax_leaves_an_absolute_symbols_got_load_alone() {
    if which("clang").is_none() {
        eprintln!("skipping absolute-relax test: clang unavailable");
        return;
    }
    let dir = workdir("abs");
    let obj = dir.join("abs.o");
    let Some(()) = assemble(ABS_SRC, &obj) else {
        eprintln!("skipping absolute-relax test: host clang unavailable");
        return;
    };
    let out = dir.join("libabs.so");
    link_shared(
        std::slice::from_ref(&obj),
        &out,
        None,
        false,
        IcfMode::None,
        true,
    )
    .expect("xold -shared --relax link must succeed");

    // `%rax` is register 0 and `%rcx` register 1, so the two sites are told
    // apart by the ModR/M byte: `05` reads `abs_sym`, `0d` reads `def_sym`.
    let text = text_bytes(&out);
    let has = |seq: &[u8]| text.windows(seq.len()).any(|w| w == seq);
    assert!(
        has(&[0x48, 0x8b, 0x05]) && !has(&[0x48, 0x8d, 0x05]),
        "an absolute symbol's GOT load must stay a load"
    );
    assert!(
        has(&[0x48, 0x8d, 0x0d]) && !has(&[0x48, 0x8b, 0x0d]),
        "the section-backed symbol beside it must still fold to a lea"
    );
}

/// A resolver for a single symbol: defined at a fixed address, not
/// preemptible, not absolute, and given a GOT slot by the scan. Those are the
/// four answers the `GOTPCRELX` relaxation path consults; it never asks for a
/// GOT or PLT address, so those return zero.
struct FixedSym(u64);

impl Resolver for FixedSym {
    fn symbol_addr(&self, _: SymbolId) -> u64 {
        self.0
    }
    fn got_addr(&self, _: SymbolId) -> u64 {
        0
    }
    fn got_base(&self) -> u64 {
        0
    }
    fn plt_addr(&self, _: SymbolId) -> u64 {
        0
    }
    fn is_preemptible(&self, _: SymbolId) -> bool {
        false
    }
    fn is_absolute(&self, _: SymbolId) -> bool {
        false
    }
    fn has_got_slot(&self, _: SymbolId) -> bool {
        true
    }
}

/// Drives the relaxation hook over `window` for a `GOTPCRELX` site at `place`
/// whose symbol is defined at `sym_addr`, and reports whether it rewrote it.
/// `-4` is the canonical RIP-relative addend the hook requires.
fn relax_site(window: &mut [u8], sym_addr: u64, place: u64) -> bool {
    X86_64::relax(
        R_X86_64_GOTPCRELX,
        Some(SymbolId(0)),
        -4,
        place,
        &FixedSym(sym_addr),
        window,
    )
    .expect("the x86-64 relax hook reports by return value, not by error")
}

/// A `call`/`jmp *sym@GOTPCREL(%rip)` whose direct displacement overflows the
/// signed 32-bit field must be refused with the window byte-for-byte as it
/// was found.
///
/// The caller falls back to applying the original relocation over these same
/// bytes, so an opcode rewritten before the range check would leave a direct
/// `call`/`jmp` holding a GOT-relative displacement.
#[test]
fn out_of_range_indirect_call_leaves_the_window_untouched() {
    // 4 GiB from the place, so neither form's displacement fits.
    const FAR: u64 = 0x1_0000_0000;
    for modrm in [0x15u8, 0x25] {
        let original = [0xff, modrm, 0x11, 0x22, 0x33, 0x44];
        let mut window = original;
        assert!(
            !relax_site(&mut window, FAR, 0x1000),
            "a displacement overflowing disp32 must not relax (modrm {modrm:#x})"
        );
        assert_eq!(
            window, original,
            "a refused relaxation must not touch a byte (modrm {modrm:#x})"
        );
    }
}

/// At exactly `disp32` maximum the `call` form fits and the `jmp` form does
/// not, because the shorter direct jmp carries `val + 1`. The jmp must come
/// back unmodified.
#[test]
fn indirect_jmp_rejects_the_boundary_the_call_accepts() {
    // sym - 4 - place == 0x7fff_ffff, the largest displacement disp32 holds.
    const PLACE: u64 = 0x1000;
    const SYM: u64 = 0x8000_1003;

    let mut call = [0xffu8, 0x15, 0x00, 0x00, 0x00, 0x00];
    assert!(relax_site(&mut call, SYM, PLACE));
    assert_eq!(call, [0x67, 0xe8, 0xff, 0xff, 0xff, 0x7f]);

    let jmp_original = [0xffu8, 0x25, 0x00, 0x00, 0x00, 0x00];
    let mut jmp = jmp_original;
    assert!(
        !relax_site(&mut jmp, SYM, PLACE),
        "the jmp form's `val + 1` overflows disp32 here"
    );
    assert_eq!(jmp, jmp_original, "the refused jmp keeps the GOT access");
}

/// The bytes a successful relaxation writes, pinned: `ff 15` becomes an
/// `addr32`-prefixed direct call, `ff 25` a direct jmp plus a trailing nop
/// whose displacement is one higher.
#[test]
fn in_range_indirect_call_relaxes_to_the_direct_form() {
    // sym - 4 - place == 0xffc.
    const PLACE: u64 = 0x1000;
    const SYM: u64 = 0x2000;

    let mut call = [0xffu8, 0x15, 0x00, 0x00, 0x00, 0x00];
    assert!(relax_site(&mut call, SYM, PLACE));
    assert_eq!(call, [0x67, 0xe8, 0xfc, 0x0f, 0x00, 0x00]);

    let mut jmp = [0xffu8, 0x25, 0x00, 0x00, 0x00, 0x00];
    assert!(relax_site(&mut jmp, SYM, PLACE));
    assert_eq!(jmp, [0xe9, 0xfd, 0x0f, 0x00, 0x00, 0x90]);
}
