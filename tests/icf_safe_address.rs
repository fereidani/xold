//! `--icf=safe` and the two ways a function's address escapes.
//!
//! Safe mode promises one thing: `&f == &g` stays false for two functions the
//! program can compare. xold decided address-taking from the relocation
//! expression alone, and counted `RelExpr::Pc` as a call. On x86-64 that is
//! backwards -- a direct call to a global is `R_X86_64_PLT32` even in non-PIC
//! code, while `R_X86_64_PC32` is what `lea f(%rip), %rax` emits to *take* a
//! function's address. So the plainest address-take there is, returning a
//! function pointer, folded two functions onto one address under a mode whose
//! whole purpose is to not do that.
//!
//! The second escape needs no relocation in this link at all. An exported
//! definition can have its address taken by another image in the process, and
//! nothing here can see that reference. lld marks the same set, over
//! `sym->isExported` in `findKeepUniqueSections` (`Driver.cpp:2583`).
//!
//! Reading `Pc` as address-taking has a consequence worth stating: `.eh_frame`
//! names every function in the image through a `PC_begin` that is exactly an
//! `R_X86_64_PC32`. Scanning it would mark everything and safe mode would fold
//! nothing, so the scan skips it -- lld never consults `.eh_frame` for this
//! either. `a_call_only_pair_still_folds` is what holds that line: it fails
//! just as loudly if the exclusion goes as if the fix does.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::{ObjectFile, constants::STB_GLOBAL},
    icf::IcfMode,
    linker::{link_shared, link_to},
};

mod common;

/// A freestanding program in which `g1` and `g2` have their address taken (by
/// `pick`, through `R_X86_64_PC32`) while `c1` and `c2` are only ever called
/// (through `R_X86_64_PLT32`).
///
/// The four are hidden so that taking an address does not go through the GOT:
/// a `GOTPCREL` was already read as address-taking, and it is the plain
/// PC-relative form that was not. Hidden visibility is also what a `static`
/// function or an `-fvisibility=hidden` build produces, so this is the shape a
/// real program folds wrongly in.
///
/// It reports what folding did rather than what the symbol table says: the
/// pointers the program compares are the addresses safe mode is about.
const ADDR_SRC: &[u8] = b"typedef int (*fp)(int);\n\
    #define HID __attribute__((visibility(\"hidden\")))\n\
    HID int g1(int x) { return x + 3; }\n\
    HID int g2(int x) { return x + 3; }\n\
    HID int c1(int x) { return x * 5; }\n\
    HID int c2(int x) { return x * 5; }\n\
    HID fp pick(int k) { return k ? g1 : g2; }\n\
    static void exit_with(int code)\n\
    {\n\
        __asm__ volatile(\"syscall\" :: \"a\"(60), \"D\"((long)code));\n\
        __builtin_unreachable();\n\
    }\n\
    void _start(void)\n\
    {\n\
        int rc = 0;\n\
        if (pick(1) == pick(0)) { rc |= 1; }\n\
        if (pick(1)(1) != 4) { rc |= 2; }\n\
        if (pick(0)(1) != 4) { rc |= 4; }\n\
        if (c1(2) + c2(2) != 20) { rc |= 8; }\n\
        exit_with(rc);\n\
    }\n";

/// A shared object with an exported identical pair and a hidden identical
/// pair. Only the exported pair can be compared from outside this link.
const EXPORT_SRC: &[u8] = b"int e1(int x) { return x + 3; }\n\
    int e2(int x) { return x + 3; }\n\
    __attribute__((visibility(\"hidden\"))) int h1(int x) { return x * 7; }\n\
    __attribute__((visibility(\"hidden\"))) int h2(int x) { return x * 7; }\n\
    int use_hidden(int x) { return h1(x) + h2(x); }\n";

/// A function pointer returned to the program keeps the two functions apart
/// under `--icf=safe`, and `--icf=all` folds them as it always did.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn safe_mode_keeps_a_pc_relative_address_take_unfolded() {
    let Some(dir) = workdir("pcrel") else {
        return;
    };
    let Some(obj) = compile(&dir, ADDR_SRC, "addr") else {
        return;
    };

    let safe = link_static(&dir, &obj, "addr_safe", IcfMode::Safe);
    let g1 = symbol_value(&safe, b"g1").expect("g1 present");
    let g2 = symbol_value(&safe, b"g2").expect("g2 present");
    assert_ne!(
        g1, g2,
        "the address of g1 and g2 reaches the program through pick(), so \
         --icf=safe must leave them at distinct addresses"
    );

    let all = link_static(&dir, &obj, "addr_all", IcfMode::All);
    let g1_all = symbol_value(&all, b"g1").expect("g1 present");
    let g2_all = symbol_value(&all, b"g2").expect("g2 present");
    assert_eq!(
        g1_all, g2_all,
        "--icf=all folds identical functions whatever their address does"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The behaviour behind the addresses: the program compares the two pointers
/// itself and still computes with both.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_program_sees_two_distinct_function_pointers() {
    let Some(dir) = workdir("run") else {
        return;
    };
    let Some(obj) = compile(&dir, ADDR_SRC, "addr") else {
        return;
    };
    let prog = dir.join("addr_run");
    link(&obj, &prog, IcfMode::Safe);
    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        assert_program_contract(&fs::read(&prog).expect("read linked image"));
        let _ = fs::remove_dir_all(&dir);
        return;
    }
    let code = Command::new(&prog)
        .status()
        .expect("linked program must be runnable")
        .code();
    assert_eq!(
        code,
        Some(0),
        "bit 0 means the program observed &g1 == &g2; bits 1, 2 and 3 mean a \
         fold changed what a call computes"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control, and the reason the scan skips `.eh_frame`: a pair reached only
/// by direct calls still folds under `--icf=safe`. Without that exclusion the
/// FDE naming every function would mark this pair too, and safe mode would
/// fold nothing at all.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_call_only_pair_still_folds() {
    let Some(dir) = workdir("callonly") else {
        return;
    };
    let Some(obj) = compile(&dir, ADDR_SRC, "addr") else {
        return;
    };
    let safe = link_static(&dir, &obj, "call_safe", IcfMode::Safe);
    let c1 = symbol_value(&safe, b"c1").expect("c1 present");
    let c2 = symbol_value(&safe, b"c2").expect("c2 present");
    assert_eq!(
        c1, c2,
        "nothing takes the address of c1 or c2, so --icf=safe must still fold \
         them -- a safe mode that folds nothing is not a safe mode"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// An exported definition is address-taken by definition: another image can
/// take its address without this link ever seeing a relocation for it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn safe_mode_keeps_exported_definitions_unfolded() {
    let Some(dir) = workdir("export") else {
        return;
    };
    let Some(obj) = compile(&dir, EXPORT_SRC, "export") else {
        return;
    };
    let lib = dir.join("libicfexport.so");
    let res = link_shared(
        std::slice::from_ref(&obj),
        &lib,
        Some(b"libicfexport.so"),
        false,
        IcfMode::Safe,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    let bytes = fs::read(&lib).expect("read shared object");

    let e1 = symbol_value(&bytes, b"e1").expect("e1 present");
    let e2 = symbol_value(&bytes, b"e2").expect("e2 present");
    assert_ne!(
        e1, e2,
        "e1 and e2 are in .dynsym, so a program that loads this object can \
         compare their addresses; --icf=safe must keep them apart"
    );

    let h1 = symbol_value(&bytes, b"h1").expect("h1 present");
    let h2 = symbol_value(&bytes, b"h2").expect("h2 present");
    assert_eq!(
        h1, h2,
        "the hidden pair is private to this image and only ever called, so it \
         must still fold -- the rule is export, not caution about everything"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping icf-safe-address {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_icfsafe_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles `src` with one section per function, which is what gives ICF
/// something to fold.
fn compile(dir: &Path, src: &[u8], stem: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src_path = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clang)
        .args([
            "--target=x86_64-linux-gnu",
            "-fPIC",
            "-ffunction-sections",
            "-fdata-sections",
            "-c",
        ])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping icf-safe-address: clang cannot build the fixture");
        return None;
    }
    Some(obj)
}

/// Links `obj` into `dir/name` and returns the image bytes.
fn link_static(dir: &Path, obj: &Path, name: &str, mode: IcfMode) -> Vec<u8> {
    let out = dir.join(name);
    link(obj, &out, mode);
    fs::read(&out).expect("read output")
}

/// Links `obj` as a static executable. The fixture calls `exit` by syscall, so
/// it needs no libc and the link stays a pure test of the ICF pass.
fn link(obj: &Path, out: &Path, mode: IcfMode) {
    let res = link_to(
        std::slice::from_ref(&obj.to_path_buf()),
        out,
        b"_start",
        false,
        mode,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
}

// --- readers ---------------------------------------------------------------

/// The `st_value` of a global symbol in the output's `.symtab`.
fn symbol_value(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab
        .iter()
        .find(|s| s.bind() == STB_GLOBAL && symtab.name(s) == name)
        .map(|s| s.st_value.get())
}

/// Structural equivalent of the exit-bit checks on a host that cannot run
/// x86_64 ELF: the address-taken pair remains distinct, `pick`'s two LEAs
/// resolve to exactly those addresses, and both arithmetic bodies retain the
/// constants the runtime checks.
fn assert_program_contract(bytes: &[u8]) {
    let g1 = symbol_value(bytes, b"g1").expect("g1 present");
    let g2 = symbol_value(bytes, b"g2").expect("g2 present");
    assert_ne!(g1, g2, "pick returns two distinct function pointers");
    let g1_body = image_bytes_at(bytes, g1, 15).expect("g1 body");
    let g2_body = image_bytes_at(bytes, g2, 15).expect("g2 body");
    assert_eq!(g1_body, g2_body, "both pointer targets compute alike");
    assert!(
        g1_body.windows(3).any(|w| w == [0x83, 0xc0, 0x03]),
        "g1 and g2 retain x + 3"
    );

    let c1 = symbol_value(bytes, b"c1").expect("c1 present");
    let c2 = symbol_value(bytes, b"c2").expect("c2 present");
    assert_eq!(c1, c2, "the call-only pair folds to one implementation");
    let call_body = image_bytes_at(bytes, c1, 13).expect("folded call body");
    assert!(
        call_body.windows(4).any(|w| w == [0x6b, 0x45, 0xfc, 0x05]),
        "the folded implementation retains x * 5"
    );

    let pick = symbol_value(bytes, b"pick").expect("pick present");
    let pick_body = image_bytes_at(bytes, pick, 33).expect("pick body");
    let mut targets = Vec::new();
    for at in 0..pick_body.len().saturating_sub(7) {
        if pick_body[at..at + 3] == [0x48, 0x8d, 0x05]
            || pick_body[at..at + 3] == [0x48, 0x8d, 0x0d]
        {
            let disp = i32::from_le_bytes(
                pick_body[at + 3..at + 7]
                    .try_into()
                    .expect("LEA displacement"),
            );
            let next = pick + u64::try_from(at + 7).unwrap();
            targets.push(next.wrapping_add(i64::from(disp).cast_unsigned()));
        }
    }
    targets.sort_unstable();
    let mut expected = vec![g1, g2];
    expected.sort_unstable();
    assert_eq!(targets, expected, "pick selects exactly g1 and g2");
}

fn image_bytes_at(bytes: &[u8], addr: u64, len: usize) -> Option<&[u8]> {
    let obj = ObjectFile::parse(bytes).ok()?;
    for sec in obj.sections() {
        let base = sec.sh_addr.get();
        if addr < base || addr >= base.saturating_add(sec.sh_size.get()) {
            continue;
        }
        let at = usize::try_from(addr - base).ok()?;
        let data = obj.section_data(sec).ok()?;
        return data.get(at..at.checked_add(len)?);
    }
    None
}
