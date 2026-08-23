//! C++ exception-handling end-to-end tests.
//!
//! These prove the runtime fix for the `.eh_frame_hdr` / `PT_GNU_EH_FRAME`
//! gap: a program with `try/catch/throw` linked by xold must reach its catch
//! handler instead of aborting at `terminate`. Before the fix the unwinder
//! could not locate the FDE for the throw site (xold folded `.eh_frame` into
//! `.rodata` and emitted no header / segment), so every throw aborted with
//! `SIGABRT` (exit 134).
//!
//! - `cpp_throw_catch_runs`: the canonical repro -- `throw` inside `main`,
//!   caught by a `catch (const std::exception&)` handler. Links with the crt
//!   objects + `libc` + `libstdc++`, runs under the system `ld.so`, asserts
//!   "caught: boom" on stdout and exit 0.
//! - `cpp_throw_in_function_catch_in_caller_runs`: a `throw` inside a helper
//!   function, caught by the caller. Exercises an FDE whose coverage is a
//!   non-`main` function, with the throw frame below `main` on the stack.
//! - `cpp_eh_frame_hdr_segment_present`: structural check -- the linked binary
//!   carries a `PT_GNU_EH_FRAME` segment and a `.eh_frame_hdr` section.
//!
//! Gated on `clang++`, `gcc` (for the crt objects), and the system `ld.so`;
//! if absent the tests print a note and return, so the build never fails over
//! a missing toolchain.

#![allow(clippy::similar_names)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, which};
use xold::elf::ObjectFile;

mod common;

/// `PT_GNU_EH_FRAME`: locates `.eh_frame_hdr` for the runtime unwinder.
const PT_GNU_EH_FRAME: u32 = 0x6474_e550;

/// The canonical repro: `throw` inside `main`, caught in the same function.
const THROW_SRC: &[u8] = b"\
#include <cstdio>\n\
#include <stdexcept>\n\
int main(){\n\
    try { throw std::runtime_error(\"boom\"); }\n\
    catch (const std::exception& e){ printf(\"caught: %s\\n\", e.what()); return 0; }\n\
    return 2;\n\
}\n";

/// A throw inside a helper function, caught by the caller. Exercises an FDE
/// whose coverage is a non-`main` function with a real unwind frame between
/// the throw site and the handler.
const THROW_IN_FUNC_SRC: &[u8] = b"\
#include <cstdio>\n\
#include <stdexcept>\n\
static int detonate(){ throw std::runtime_error(\"deep\"); }\n\
int main(){\n\
    try { return detonate(); }\n\
    catch (const std::exception& e){ printf(\"caught: %s\\n\", e.what()); return 0; }\n\
    return 3;\n\
}\n";

/// A throwing program with two unreachable helpers. Compiled with
/// `-ffunction-sections`, each helper is its own section, so `--gc-sections`
/// collects them and their `.eh_frame` records describe nothing.
const THROW_WITH_DEAD_SRC: &[u8] = b"\
#include <cstdio>\n\
#include <stdexcept>\n\
int dead_a(){ throw std::runtime_error(\"a\"); }\n\
int dead_b(){ throw std::runtime_error(\"b\"); }\n\
int main(){\n\
    try { throw std::runtime_error(\"boom\"); }\n\
    catch (const std::exception& e){ printf(\"caught: %s\\n\", e.what()); return 0; }\n\
    return 2;\n\
}\n";

/// A program with one function compiled into `.init`. Unwind tables are on by
/// default on x86-64, so the function gets an FDE whose covered PC lands in
/// `.init` rather than `.text`. The probe sits ahead of `crti.o`'s `.init`
/// contribution, so `_init` still names the runtime's prologue and the program
/// runs as usual.
const INIT_FDE_SRC: &[u8] = b"\
#include <cstdio>\n\
__attribute__((section(\".init\"), used, noinline))\n\
void init_probe(int x){ printf(\"%d\\n\", x); }\n\
int main(){ printf(\"ok\\n\"); return 0; }\n";

/// Links `main_obj` plus the crt objects, libc and libstdc++ into `prog` with
/// xold. Panics on link failure so a regression reads as a test failure, not
/// a silent skip.
fn link(h: &Harness, main_obj: &Path, prog: &Path) {
    link_with(h, main_obj, prog, false);
}

/// Links as [`link`] does, with `--gc-sections` selected by `gc`.
fn link_with(h: &Harness, main_obj: &Path, prog: &Path, gc: bool) {
    link_order(
        h,
        &[
            main_obj.to_path_buf(),
            h.crti.clone(),
            h.crt1.clone(),
            h.crtn.clone(),
        ],
        prog,
        gc,
    );
}

/// Links `objs` in exactly that order, with libc and libstdc++ appended, into
/// `prog`. The object order is the whole point of
/// [`eh_frame_is_input_order_independent`], so it is the caller's to choose.
fn link_order(h: &Harness, objs: &[PathBuf], prog: &Path, gc: bool) {
    let mut inputs = objs.to_vec();
    inputs.push(h.libc.clone());
    inputs.push(h.libstdcxx.clone());
    // `_Unwind_Resume` lives in `libgcc_s`, which `libstdc++` names undefined;
    // without it the link is underlinked.
    inputs.push(h.libgcc_s.clone());
    xold::linker::link_dyn_exec(
        &inputs,
        prog,
        b"_start",
        &h.interp,
        gc,
        xold::icf::IcfMode::None,
        false,
    )
    .expect("xold C++ link must succeed");
}

/// Compiles `src` (`-fPIE`) to `obj` with the host `clang++`.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    compile_with(src, obj, &[])
}

/// Compiles `src` (`-fPIE`) to `obj` with `extra` appended to the flags.
fn compile_with(src: &[u8], obj: &Path, extra: &[&str]) -> Option<()> {
    let clangxx = which("clang++")?;
    let src_path = obj.with_extension("cpp");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clangxx)
        .args(["--target=x86_64-linux-gnu", "-fPIE", "-c"])
        .args(extra)
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// One-stop handle for the crt/libc/libstdc++ paths a C++ dynamic link needs.
struct Harness {
    crt1: PathBuf,
    crti: PathBuf,
    crtn: PathBuf,
    libc: PathBuf,
    libstdcxx: PathBuf,
    libgcc_s: PathBuf,
    interp: Vec<u8>,
}

impl Harness {
    /// Detects the harness; returns `None` if any required tool or file is
    /// missing so the caller can skip gracefully.
    fn detect() -> Option<Self> {
        if which("clang++").is_none() {
            eprintln!("skipping C++ exception tests: clang++ unavailable");
            return None;
        }
        if which("gcc").is_none() {
            eprintln!("skipping C++ exception tests: gcc unavailable (crt)");
            return None;
        }
        let crt1 = crt_file("crt1.o")?;
        let crti = crt_file("crti.o")?;
        let crtn = crt_file("crtn.o")?;
        let libc = find_lib("libc.so.6")?;
        let libstdcxx = find_lib("libstdc++.so.6")?;
        let libgcc_s = find_lib("libgcc_s.so.1")?;
        let interp = interpreter()?;
        Some(Self {
            crt1,
            crti,
            crtn,
            libc,
            libstdcxx,
            libgcc_s,
            interp,
        })
    }
}

/// The canonical throw/catch repro. Before the fix xold's output printed
/// "terminate called after throwing..." and aborted (exit 134); the fix lets
/// the runtime unwinder locate the FDE via `PT_GNU_EH_FRAME` so the catch
/// handler runs.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn cpp_throw_catch_runs() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("throw");
    let main_o = dir.join("throw.o");
    compile(THROW_SRC, &main_o).expect("host clang++ compiles throw.cpp");
    let prog = dir.join("throw_prog");
    link(&h, &main_o, &prog);

    let out = Command::new(&prog)
        .output()
        .expect("linked program must be runnable");
    assert!(
        out.status.success(),
        "throw program should exit 0, got {:?}",
        out.status
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "caught: boom\n",
        "catch handler should print"
    );
}

/// A throw inside a helper function, caught by the caller. Exercises an FDE
/// for a non-`main` function with a stack frame between throw and catch.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn cpp_throw_in_function_catch_in_caller_runs() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("deep");
    let main_o = dir.join("deep.o");
    compile(THROW_IN_FUNC_SRC, &main_o)
        .expect("host clang++ compiles deep.cpp");
    let prog = dir.join("deep_prog");
    link(&h, &main_o, &prog);

    let out = Command::new(&prog)
        .output()
        .expect("linked program must be runnable");
    assert!(
        out.status.success(),
        "deep throw program should exit 0, got {:?}",
        out.status
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "caught: deep\n",
        "catch handler should print"
    );
}

/// `--gc-sections` must not collect the unwind data. Nothing relocates into
/// `.eh_frame`, so plain reachability drops every one of its members, leaving
/// the runtime with no FDE for the throw site: the program then aborts at
/// `terminate` (exit 134) instead of reaching its handler. The collector keeps
/// `.eh_frame` as a root and ignores the FDE-to-function edges, so garbage is
/// still collected while unwinding keeps working.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn cpp_throw_catch_runs_with_gc_sections() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("throw_gc");
    let main_o = dir.join("throw.o");
    compile(THROW_SRC, &main_o).expect("host clang++ compiles throw.cpp");
    let prog = dir.join("throw_gc_prog");
    link_with(&h, &main_o, &prog, true);

    let bytes = fs::read(&prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let has_eh_frame = obj
        .sections()
        .iter()
        .any(|s| obj.section_name(s) == b".eh_frame");
    assert!(has_eh_frame, "--gc-sections must not collect .eh_frame");
    assert!(
        program_header_types(&bytes).any(|t| t == PT_GNU_EH_FRAME),
        "--gc-sections must keep PT_GNU_EH_FRAME"
    );

    let out = Command::new(&prog)
        .output()
        .expect("linked program must be runnable");
    assert!(
        out.status.success(),
        "throw program built with --gc-sections should exit 0, got {:?}",
        out.status
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "caught: boom\n",
        "catch handler should print"
    );
}

/// The records describing collected functions must go with them. Linking the
/// same object twice, once with `--gc-sections`, must leave the collected
/// version with a strictly smaller `.eh_frame` and a smaller FDE count, while
/// the surviving throw still reaches its handler: the FDE `CIE_pointer` is a
/// backwards distance, so moving records without rewriting it corrupts the
/// section.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn gc_sections_drops_records_for_collected_functions() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("throw_dead");
    let main_o = dir.join("dead.o");
    compile_with(THROW_WITH_DEAD_SRC, &main_o, &["-ffunction-sections"])
        .expect("host clang++ compiles dead.cpp");

    let kept = dir.join("dead_nogc");
    let collected = dir.join("dead_gc");
    link_with(&h, &main_o, &kept, false);
    link_with(&h, &main_o, &collected, true);

    let (kept_size, kept_fdes) = eh_frame_shape(&kept);
    let (gc_size, gc_fdes) = eh_frame_shape(&collected);
    assert!(
        gc_size < kept_size,
        ".eh_frame must shrink when functions are collected \
         ({gc_size} vs {kept_size})"
    );
    assert!(
        gc_fdes < kept_fdes,
        "the FDE count must shrink too ({gc_fdes} vs {kept_fdes})"
    );

    let out = Command::new(&collected)
        .output()
        .expect("linked program must be runnable");
    assert!(
        out.status.success(),
        "collected program should exit 0, got {:?}",
        out.status
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "caught: boom\n",
        "catch handler should print"
    );
}

/// The same objects in any input order must produce the same unwind data.
///
/// An `.eh_frame` is a run of records introduced by a length word, and a
/// length of zero ends the section. A member whose size is not a multiple of
/// its 8-byte alignment is followed by four bytes of inter-member padding when
/// anything comes after it, and those four zero bytes read as that terminator:
/// concatenating the inputs hides every record behind the first padded member
/// from `.eh_frame_hdr` construction and from the runtime unwinder alike.
/// `crt1.o` is such a member on a glibc host, so an order that put it ahead of
/// the program's own object dropped `fde_count` to 1 and aborted the throw at
/// `terminate`; the order that happened to place it last worked, which is what
/// hid the defect. The records are packed now, so every order agrees.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn eh_frame_is_input_order_independent() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("order");
    let main_o = dir.join("order.o");
    compile(THROW_SRC, &main_o).expect("host clang++ compiles order.cpp");
    assert!(
        padded_eh_frame(&h.crt1) || padded_eh_frame(&main_o),
        "this test only bites while some input has an .eh_frame whose size is \
         not a multiple of its alignment; neither {} nor {} does",
        h.crt1.display(),
        main_o.display()
    );

    // The last of the three is the order the rest of this file links in, and
    // the only one that ever worked: it puts the program's own object first,
    // so the padded crt member has nothing behind it to hide.
    let orders: [(&str, [PathBuf; 4]); 3] = [
        (
            "middle",
            [
                h.crt1.clone(),
                h.crti.clone(),
                main_o.clone(),
                h.crtn.clone(),
            ],
        ),
        (
            "last",
            [
                h.crt1.clone(),
                h.crti.clone(),
                h.crtn.clone(),
                main_o.clone(),
            ],
        ),
        (
            "first",
            [main_o, h.crti.clone(), h.crt1.clone(), h.crtn.clone()],
        ),
    ];

    let mut shapes: Vec<(&str, u32, usize)> = Vec::new();
    for (name, order) in &orders {
        let prog = dir.join(format!("order_{name}"));
        link_order(&h, order, &prog, false);
        let (_, hdr_fdes) = eh_frame_shape(&prog);
        let (records, fdes) = eh_frame_records(&prog);
        assert_eq!(
            u64::from(hdr_fdes),
            fdes,
            "{name}: every FDE in .eh_frame must have a search table row"
        );
        shapes.push((name, hdr_fdes, records));

        let out = Command::new(&prog)
            .output()
            .expect("linked program must be runnable");
        assert!(
            out.status.success(),
            "{name}: throw program should exit 0, got {:?}",
            out.status
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "caught: boom\n",
            "{name}: catch handler should print"
        );
    }
    assert!(
        shapes
            .windows(2)
            .all(|w| w[0].1 == w[1].1 && w[0].2 == w[1].2),
        "input order must not change the unwind data, got {shapes:?}"
    );
}

/// Whether `obj`'s `.eh_frame` would be padded by placement: its size is not a
/// multiple of its alignment, so a member behind it starts a few zero bytes
/// further on.
fn padded_eh_frame(obj: &Path) -> bool {
    let bytes = fs::read(obj).expect("read object");
    let parsed = ObjectFile::parse(&bytes).expect("valid ELF");
    parsed
        .sections()
        .iter()
        .filter(|s| parsed.section_name(s) == b".eh_frame")
        .any(|s| {
            let align = s.sh_addralign.get().max(1);
            s.sh_size.get() % align != 0
        })
}

/// The record count and FDE count of the output `.eh_frame`.
///
/// The walk also proves the section is one packed run: it must consume every
/// byte up to the four-byte terminator that closes it. A terminator anywhere
/// else, or padding between two members, ends the walk early and fails here --
/// which is exactly where the runtime unwinder would stop, only louder.
fn eh_frame_records(prog: &Path) -> (usize, u64) {
    let bytes = fs::read(prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".eh_frame")
        .expect("output must carry .eh_frame");
    let data = obj.section_data(shdr).expect("eh_frame data");
    let mut at = 0usize;
    let mut records = 0usize;
    let mut fdes = 0u64;
    while at + 8 <= data.len() {
        let len = u32::from_le_bytes(
            data[at..at + 4].try_into().expect("length word"),
        );
        if len == 0 {
            break;
        }
        let cie_ptr = u32::from_le_bytes(
            data[at + 4..at + 8].try_into().expect("CIE pointer"),
        );
        records += 1;
        fdes += u64::from(cie_ptr != 0);
        at += 4 + len as usize;
        assert!(at <= data.len(), "record at {records} runs past .eh_frame");
    }
    assert_eq!(
        at + 4,
        data.len(),
        ".eh_frame must be one packed run of records closed by a single \
         terminator; the walk stopped at {at:#x} of {:#x}",
        data.len()
    );
    (records, fdes)
}

/// An FDE covering code outside `.text` must reach the search table.
///
/// `init_probe` is compiled into `.init`, which is an output section of its
/// own, and carries unwind tables like any other function. `.eh_frame_hdr`
/// used to keep only the rows whose PC fell inside `.text`, so this FDE was
/// dropped from the binary-search table and the unwinder could not find it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn eh_frame_hdr_covers_code_outside_text() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("init_fde");
    let main_o = dir.join("init_fde.o");
    compile(INIT_FDE_SRC, &main_o).expect("host clang++ compiles init_fde.cpp");
    let prog = dir.join("init_fde_prog");
    link(&h, &main_o, &prog);

    let bytes = fs::read(&prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let init = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".init")
        .expect("output must carry .init");
    let start = init.sh_addr.get();
    let end = start + init.sh_size.get();
    assert!(start < end, ".init must hold the probe's bytes");

    let table = eh_frame_hdr_pcs(&obj);
    assert_ne!(table.len(), 0, ".eh_frame_hdr must carry a search table");
    assert!(
        table.iter().any(|&pc| pc >= start && pc < end),
        "an FDE covering .init ({start:#x}..{end:#x}) must reach the search \
         table, got {table:#x?}"
    );
    assert!(
        table.windows(2).all(|w| w[0] <= w[1]),
        "the search table must stay sorted by PC"
    );
}

/// The covered PC of every row of the `.eh_frame_hdr` binary-search table.
///
/// The table is `(initial_location, fde_address)` pairs, both encoded
/// `DW_EH_PE_datarel | DW_EH_PE_sdata4`, i.e. signed 32-bit offsets from the
/// `.eh_frame_hdr` load address.
fn eh_frame_hdr_pcs(obj: &ObjectFile<'_>) -> Vec<u64> {
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".eh_frame_hdr")
        .expect("output must carry .eh_frame_hdr");
    let base = shdr.sh_addr.get();
    let data = obj.section_data(shdr).expect("hdr data");
    let count = u32::from_le_bytes(
        data.get(8..12)
            .and_then(|b| b.try_into().ok())
            .expect("fde_count field"),
    );
    (0..count as usize)
        .map(|i| {
            let at = 12 + i * 8;
            let rel = i32::from_le_bytes(
                data.get(at..at + 4)
                    .and_then(|b| b.try_into().ok())
                    .expect("table entry"),
            );
            base.wrapping_add(i64::from(rel).cast_unsigned())
        })
        .collect()
}

/// The `.eh_frame` byte size and the `fde_count` field of `.eh_frame_hdr`.
fn eh_frame_shape(prog: &Path) -> (u64, u32) {
    let bytes = fs::read(prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let mut size = 0u64;
    let mut count = 0u32;
    for shdr in obj.sections() {
        match obj.section_name(shdr) {
            b".eh_frame" => size = shdr.sh_size.get(),
            b".eh_frame_hdr" => {
                let data = obj.section_data(shdr).expect("hdr data");
                let field: [u8; 4] = data
                    .get(8..12)
                    .and_then(|b| b.try_into().ok())
                    .expect("fde_count field");
                count = u32::from_le_bytes(field);
            }
            _ => {}
        }
    }
    assert!(size > 0, "output must carry .eh_frame");
    assert!(count > 0, "output must carry a non-empty search table");
    (size, count)
}

/// Structural check: the linked binary carries `PT_GNU_EH_FRAME` and a
/// `.eh_frame_hdr` section, the two pieces the runtime unwinder needs.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn cpp_eh_frame_hdr_segment_present() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("struct");
    let main_o = dir.join("struct.o");
    compile(THROW_SRC, &main_o).expect("host clang++ compiles struct.cpp");
    let prog = dir.join("struct_prog");
    link(&h, &main_o, &prog);

    let bytes = fs::read(&prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("output must be valid ELF");

    let has_hdr = obj
        .sections()
        .iter()
        .any(|s| obj.section_name(s) == b".eh_frame_hdr");
    assert!(has_hdr, "output must contain a .eh_frame_hdr section");

    let has_eh_frame = obj
        .sections()
        .iter()
        .any(|s| obj.section_name(s) == b".eh_frame");
    assert!(has_eh_frame, "output must contain a .eh_frame section");

    // The program-header table must contain a PT_GNU_EH_FRAME entry pointing
    // at `.eh_frame_hdr`; without it the loader never builds its unwind index.
    assert!(
        program_header_types(&bytes).any(|t| t == PT_GNU_EH_FRAME),
        "output must carry PT_GNU_EH_FRAME"
    );
}

/// Iterates the `p_type` of every program header in the ELF64 image `bytes`.
/// Reads the table directly so the test does not depend on a public phdr
/// accessor on `ObjectFile`.
fn program_header_types(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    let header = bytes.get(..64).expect("ELF header present");
    // ELF64 Ehdr: e_phoff at offset 32 (u64), e_phentsize at 54 (u16),
    // e_phnum at 56 (u16).
    let phoff =
        usize::try_from(u64::from_le_bytes(header[32..40].try_into().unwrap()))
            .unwrap();
    let phentsize =
        usize::from(u16::from_le_bytes(header[54..56].try_into().unwrap()));
    let phnum = u16::from_le_bytes(header[56..58].try_into().unwrap());
    (0..phnum).map(move |i| {
        let base = phoff + i as usize * phentsize;
        u32::from_le_bytes(bytes[base..base + 4].try_into().unwrap())
    })
}

// --- harness helpers ------------------------------------------------------

/// Locates a shared library by name in the linker search path.
fn find_lib(name: &str) -> Option<PathBuf> {
    for dir in lib_search_dirs() {
        let candidate = dir.join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// The directories `gcc` searches for shared libraries, parsed from
/// `gcc -print-search-dirs`.
fn lib_search_dirs() -> Vec<PathBuf> {
    let Some(gcc) = which("gcc") else {
        return Vec::new();
    };
    let Some(out) = Command::new(gcc).arg("-print-search-dirs").output().ok()
    else {
        return Vec::new();
    };
    let s = String::from_utf8_lossy(&out.stdout);
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("libraries: =") {
            return rest
                .split(':')
                .filter(|p| !p.is_empty())
                .map(PathBuf::from)
                .collect();
        }
    }
    Vec::new()
}

/// The dynamic linker path the kernel should hand the executable. Uses the
/// system `ld.so` discovered from the host `clang++` (its default interp).
fn interpreter() -> Option<Vec<u8>> {
    let clangxx = which("clang++")?;
    // Link a tiny probe executable with the system linker and read its
    // `.interp`; that is the path the kernel execs for a dynamically linked
    // x86-64 binary on this host.
    let dir = std::env::temp_dir().join("xold_interp_probe");
    fs::create_dir_all(&dir).ok()?;
    let src = dir.join("probe.cpp");
    fs::write(&src, b"int main(){return 0;}\n").ok()?;
    let prog = dir.join("probe");
    let ok = Command::new(clangxx)
        .arg(&src)
        .arg("-o")
        .arg(&prog)
        .status()
        .ok()?
        .success();
    if !ok {
        return None;
    }
    let bytes = fs::read(&prog).ok()?;
    let obj = ObjectFile::parse(&bytes).ok()?;
    let interp = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".interp")
        .and_then(|s| obj.section_data(s).ok())?;
    let _ = fs::remove_file(&src);
    let _ = fs::remove_file(&prog);
    // Trim a trailing NUL so the path is a clean C string.
    Some(interp.iter().take_while(|&&b| b != 0).copied().collect())
}

/// A unique working directory under `std::env::temp_dir()` keyed by `tag`.
fn workdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("xold-exceptions-tests").join(tag);
    fs::create_dir_all(&dir).expect("create workdir");
    dir
}
