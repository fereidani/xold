//! `.rela.dyn` around identical code folding, and the input folding used to
//! make interesting.
//!
//! This file was written to pin a real fix: the dynamic relocation passes walk
//! *input* sections, and layout stamps a folded section with the
//! representative's virtual address, so an address lookup could not tell the
//! two apart and every folded copy emitted a duplicate entry aimed at the
//! representative's slot. The fixture put a `.quad target` inside an `"ax"`
//! section, because that is a section `--icf` will fold.
//!
//! That fixture is no longer linkable, and the image it produced was never
//! loadable. A dynamic relocation into a section without `SHF_WRITE` needs the
//! loader to make the mapping writable first, which an image asks for with
//! `DT_TEXTREL`; xold emits no such tag, so the loader either skipped the
//! entry or died applying it. xold now refuses the input, as `ld.lld` does
//! ("relocation `R_X86_64_64` cannot be used against symbol 'target';
//! recompile with -fPIC"); GNU ld and mold instead accept it and emit
//! `DT_TEXTREL`.
//!
//! That leaves the duplicate-entry bug unreachable through `--icf`, which
//! folds only executable sections without `SHF_WRITE`: any dynamic relocation
//! in one of those is now refused, and refused on the representative, which
//! carries the very relocations that made the copy foldable. The exclusion in
//! `for_each_data_reloc_section` is kept as the walk's own invariant and
//! commented as such. What is still observable, and what the last test pins,
//! is that folding code leaves the surviving pointer slots described exactly
//! once each.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::ObjectFile,
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
};

mod common;

/// The refused shape: a pointer stored inside executable, read-only code.
/// Assembly is the only way to write it -- `-fPIC` and `-fPIE` compilers put
/// a relocatable pointer in `.data.rel.ro`, which is writable.
const IN_TEXT_SRC: &[u8] = b"    .text\n\
    .globl target\n    .type target,@function\n\
    .section .text.target,\"ax\",@progbits\ntarget:\n    movl $7, %eax\n    \
    retq\n    .size target, .-target\n\
    \n    .section .text.fa,\"ax\",@progbits\n    .globl fa\n\
    .type fa,@function\n    .p2align 4\nfa:\n    movl $1, %eax\n    retq\n    \
    .p2align 3\n    .quad target\n    .size fa, .-fa\n";

/// The same two pointers, in the writable section a position-independent
/// compiler would have put them in, beside two identical functions for
/// `--icf=all` to fold. Folding the code must not disturb the data.
const IN_DATA_SRC: &[u8] = b"    .section .text.target,\"ax\",@progbits\n\
    .globl target\n    .type target,@function\ntarget:\n    movl $7, %eax\n    \
    retq\n    .size target, .-target\n\
    \n    .section .text.fa,\"ax\",@progbits\n    .globl fa\n\
    .type fa,@function\n    .p2align 4\nfa:\n    movl $1, %eax\n    retq\n    \
    .size fa, .-fa\n\
    \n    .section .text.fb,\"ax\",@progbits\n    .globl fb\n\
    .type fb,@function\n    .p2align 4\nfb:\n    movl $1, %eax\n    retq\n    \
    .size fb, .-fb\n\
    \n    .data\n    .globl pa\n    .p2align 3\npa:\n    .quad target\n\
    .globl pb\n    .p2align 3\npb:\n    .quad target\n";

/// A `_start` that checks both pointers arrived at the one function they name
/// and exits with 0, or with 1 when they did not. Nothing else runs, so what
/// the loader did to `.data` is all the exit status can report.
const START_SRC: &[u8] = b"    .text\n    .globl _start\n_start:\n    \
    leaq    target(%rip), %rax\n    movq    pa(%rip), %rcx\n    \
    cmpq    %rax, %rcx\n    jne     1f\n    movq    pb(%rip), %rcx\n    \
    cmpq    %rax, %rcx\n    jne     1f\n    xorl    %edi, %edi\n    jmp     2f\n\
1:  movl    $1, %edi\n2:  movl    $60, %eax\n    syscall\n";

/// A relocation into a read-only section is refused, whatever folding is set
/// to, and the diagnostic names the symbol, the reason and the remedy.
///
/// Folding is what used to make this input interesting; it must not make it
/// linkable either, since the representative carries the same relocation.
///
/// The diagnostic is its own error, not a borrowed one: `target` is defined
/// right there in the input, and the refusal used to render as "undefined
/// reference to 'target' from a read-only section", which sends the reader
/// after a symbol that is not missing.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_pointer_inside_read_only_code_is_refused() {
    let Some(dir) = workdir("intext") else {
        return;
    };
    let obj = dir.join("t.o");
    if assemble(IN_TEXT_SRC, &obj).is_none() {
        eprintln!("skipping icf_rela_dyn: clang cannot assemble the fixture");
        return;
    }
    for icf in [IcfMode::None, IcfMode::All] {
        let out = dir.join("t.so");
        let err = link_shared(
            std::slice::from_ref(&obj),
            &out,
            None,
            false,
            icf,
            false,
        )
        .expect_err("a relocation the loader cannot apply must be refused");
        let text = err.to_string();
        assert!(
            text.contains("'target'"),
            "the diagnostic must name the symbol, got {text:?}"
        );
        assert!(
            text.contains("read-only section"),
            "and say where the relocation cannot go, got {text:?}"
        );
        assert!(
            text.contains("-fPIC"),
            "and the remedy that makes the input linkable, got {text:?}"
        );
        assert!(
            !text.contains("undefined"),
            "target is defined in this very input, so the diagnostic must not \
             call it undefined, got {text:?}"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

/// The position-independent form of the same program links, and the loader
/// fills both pointer slots with the one address they name.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_same_pointers_in_a_writable_section_link_and_run() {
    let Some(dir) = workdir("indata") else {
        return;
    };
    let obj = dir.join("d.o");
    let start = dir.join("start.o");
    if assemble(IN_DATA_SRC, &obj).is_none()
        || assemble(START_SRC, &start).is_none()
    {
        eprintln!("skipping icf_rela_dyn: clang cannot assemble the fixture");
        return;
    }
    let Some(interp) = common::interpreter() else {
        eprintln!("skipping icf_rela_dyn: interpreter path unknown");
        return;
    };
    let prog = dir.join("prog");
    link_dyn_exec(
        &[obj, start],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::All,
        false,
    )
    .expect("xold dynamic-exec link must succeed");

    assert_eq!(
        run(&prog),
        Some(0),
        "both slots must hold the address of target once the loader has \
         applied .rela.dyn"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Folding leaves one dynamic relocation per surviving slot.
///
/// The folded copy contributes no bytes, so it must contribute no entry
/// either; the slots it does not own must keep exactly the entries the
/// unfolded link gave them. Folding really happens here -- the two identical
/// functions collapse onto one address -- so this is the reachable remnant of
/// what this file was written for.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn folding_code_leaves_the_data_slots_relocated_exactly_once() {
    let Some(dir) = workdir("fold") else {
        return;
    };
    let obj = dir.join("d.o");
    if assemble(IN_DATA_SRC, &obj).is_none() {
        eprintln!("skipping icf_rela_dyn: clang cannot assemble the fixture");
        return;
    }
    let plain = dir.join("plain.so");
    let folded = dir.join("folded.so");
    for (out, icf) in [(&plain, IcfMode::None), (&folded, IcfMode::All)] {
        link_shared(std::slice::from_ref(&obj), out, None, false, icf, false)
            .expect("xold -shared link must succeed");
    }
    let plain_bytes = fs::read(&plain).expect("read unfolded output");
    let folded_bytes = fs::read(&folded).expect("read folded output");

    // Folding happened: the two identical functions now share one address,
    // and did not before.
    let fa = dynsym_value(&folded_bytes, b"fa").expect("fa in .dynsym");
    assert_eq!(
        dynsym_value(&folded_bytes, b"fb"),
        Some(fa),
        "--icf=all must fold fb onto fa"
    );
    assert_ne!(
        dynsym_value(&plain_bytes, b"fa"),
        dynsym_value(&plain_bytes, b"fb"),
        "and must be the only reason they share an address"
    );

    // The data slots are untouched by it: one entry each, naming the one
    // symbol, at the two addresses `.dynsym` gives them.
    let idx = dynsym_index(&folded_bytes, b"target").expect("target row");
    for (bytes, what) in [(&plain_bytes, "unfolded"), (&folded_bytes, "folded")]
    {
        let rows = rela_rows(bytes, b".rela.dyn");
        let mut slots: Vec<u64> =
            rows.iter().filter(|r| r.1 == idx).map(|r| r.0).collect();
        slots.sort_unstable();
        let mut want = vec![
            dynsym_value(bytes, b"pa").expect("pa in .dynsym"),
            dynsym_value(bytes, b"pb").expect("pb in .dynsym"),
        ];
        want.sort_unstable();
        assert_eq!(
            slots, want,
            "the {what} link must relocate each surviving slot once, got \
             {rows:#x?}"
        );
        assert!(
            rows.iter().all(|r| r.2 != 0),
            "a count that shrank only because the emitter skipped an entry \
             would leave the reserved slot zeroed, got {rows:#x?}"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the input at all.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping icf_rela_dyn {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir().join(format!("xold_icfrela_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
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

/// Runs the linked program, returning its exit status.
fn run(prog: &Path) -> Option<i32> {
    Command::new(prog)
        .status()
        .expect("linked program must be runnable")
        .code()
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

/// The `st_value` of the `.dynsym` row named `name`.
fn dynsym_value(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let sec = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynsym")?;
    let data = obj.section_data(sec).ok()?;
    let index = dynsym_index(bytes, name)? as usize;
    let chunk = data.chunks(24).nth(index)?;
    if chunk.len() < 24 {
        return None;
    }
    Some(u64::from_le_bytes(chunk[8..16].try_into().ok()?))
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
