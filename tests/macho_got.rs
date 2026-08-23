//! Mach-O `x86_64` GOT (`__got`) support: a static executable that accesses an
//! external data symbol through the GOT must link without overflow and carry a
//! well-formed `__DATA __got` non-lazy symbol-pointer table.
//!
//! Verification is structural: Linux has no Mach-O loader, so the linked image
//! is checked with `file`, re-parsed by xold's own `MachOFile` reader, and its
//! `__got` section header and slot contents are decoded against the darwin ABI.
//! Each test is gated on `clang --target=x86_64-apple-darwin` being available.

use std::{
    convert::TryFrom,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use common::{which, xold_bin};
use xold::macho::{
    MachOFile,
    constants::{
        CPU_SUBTYPE_X86_64_ALL, CPU_TYPE_X86_64, MH_EXECUTE,
        S_NON_LAZY_SYMBOL_POINTERS, SECTION_TYPE,
    },
};

mod common;

/// Conventional base of the `__TEXT` segment; the GOT is mapped above this.
const TEXT_BASE: u64 = 0x1_0000_0000;

// --- source programs ------------------------------------------------------

/// Defines `counter`, referenced as an extern from another object. A cross-file
/// extern data access is what clang lowers to `X86_64_RELOC_GOT_LOAD`
/// (`movq sym@GOTPCREL(%rip), %reg`), exercising the GOT path.
const DEF_SRC: &[u8] = b"int counter = 5;\n";
const USE_SRC: &[u8] =
    b"extern int counter;\nint main(void){ return counter; }\n";

/// Two extern globals, each accessed through its own GOT slot.
const DEFS_SRC: &[u8] = b"int a = 1;\nint b = 2;\n";
const USES_SRC: &[u8] =
    b"extern int a;\nextern int b;\nint main(void){ return a + b; }\n";

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_x86_64_extern_data_via_got() {
    let triple = "--target=x86_64-apple-darwin";
    let Some(clang) = darwin_clang(triple) else {
        eprintln!("skipping: clang darwin target not found");
        return;
    };
    let dir = std::env::temp_dir();
    let def = dir.join("xold_macho_got_def.o");
    let usef = dir.join("xold_macho_got_use.o");
    let exe = dir.join("xold_macho_got_single");
    if !compile(&clang, triple, DEF_SRC, &def)
        || !compile(&clang, triple, USE_SRC, &usef)
    {
        eprintln!("skipping: clang compile failed");
        return;
    }

    // Before the GOT fix this emitted `relocation overflow (type 3)`; the link
    // must now succeed and allocate a `__got` slot for `counter`.
    assert!(
        xold_link(&[&def, &usef], &exe),
        "xold link of GOT extern data failed"
    );

    let kind = file_type(&exe);
    eprintln!("file(x86_64 got): {kind}");
    assert!(kind.contains("Mach-O"), "expected Mach-O, got: {kind}");
    assert!(
        kind.contains("executable"),
        "expected executable, got: {kind}"
    );
    assert!(kind.contains("x86_64"), "expected x86_64, got: {kind}");

    let headers = objdump_headers(&exe);
    assert!(
        headers.contains("__got"),
        "__got section missing in objdump:\n{headers}"
    );
    assert!(
        headers.contains("S_NON_LAZY_SYMBOL_POINTERS"),
        "__got not a non-lazy pointer table:\n{headers}"
    );

    let bytes = fs::read(&exe).expect("read linked executable");
    let obj = MachOFile::parse(&bytes).expect("reader round-trips the output");
    assert_eq!(obj.cpu_type(), CPU_TYPE_X86_64);
    assert_eq!(obj.cpu_subtype(), CPU_SUBTYPE_X86_64_ALL);
    assert_eq!(obj.filetype(), MH_EXECUTE);

    let sections = obj.sections();
    let got = sections
        .iter()
        .find(|s| s.sectname == b"__got")
        .expect("__got section present");
    assert_eq!(
        got.flags & SECTION_TYPE,
        S_NON_LAZY_SYMBOL_POINTERS,
        "__got section type"
    );
    assert_eq!(got.size, 8, "one 8-byte GOT slot for counter");
    assert_eq!(got.align, 3, "__got is 8-byte aligned");
    assert_eq!(got.data.len(), 8, "__got payload is the slot bytes");
    assert!(
        got.addr >= TEXT_BASE,
        "GOT mapped above the TEXT base: {:#x}",
        got.addr
    );

    let data = sections
        .iter()
        .find(|s| s.sectname == b"__data")
        .expect("__data section present");
    // `counter` is the first (and only) int in `__data`, so its runtime address
    // is the `__data` base. The GOT slot holds that resolved address.
    let slot = read_u64_le(got.data);
    assert_eq!(
        slot, data.addr,
        "GOT slot holds counter's resolved address ({:#x} vs __data {:#x})",
        slot, data.addr
    );

    let text = sections
        .iter()
        .find(|s| s.sectname == b"__text")
        .expect("__text section present");
    let targets =
        assert_movq_targets_in_got(text.data, text.addr, got.addr, got.size);
    assert_eq!(
        targets,
        vec![got.addr],
        "one GOT_LOAD targeting the single counter slot"
    );
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn links_x86_64_multi_symbol_got() {
    let triple = "--target=x86_64-apple-darwin";
    let Some(clang) = darwin_clang(triple) else {
        eprintln!("skipping: clang darwin target not found");
        return;
    };
    let dir = std::env::temp_dir();
    let defs = dir.join("xold_macho_got_defs.o");
    let uses = dir.join("xold_macho_got_uses.o");
    let exe = dir.join("xold_macho_got_multi");
    if !compile(&clang, triple, DEFS_SRC, &defs)
        || !compile(&clang, triple, USES_SRC, &uses)
    {
        eprintln!("skipping: clang compile failed");
        return;
    }

    assert!(
        xold_link(&[&defs, &uses], &exe),
        "xold link of multi-symbol GOT failed"
    );

    let headers = objdump_headers(&exe);
    assert!(
        headers.contains("S_NON_LAZY_SYMBOL_POINTERS"),
        "__got not a non-lazy pointer table:\n{headers}"
    );

    let bytes = fs::read(&exe).expect("read linked executable");
    let obj = MachOFile::parse(&bytes).expect("reader round-trips the output");
    let sections = obj.sections();
    let got = sections
        .iter()
        .find(|s| s.sectname == b"__got")
        .expect("__got section present");
    assert_eq!(got.size, 16, "two 8-byte GOT slots (one per global)");

    let data = sections
        .iter()
        .find(|s| s.sectname == b"__data")
        .expect("__data section present");
    // `a` and `b` are the two leading ints in `__data`: `a` at +0, `b` at +4.
    // The scan order is the reloc table's file order, so compare as a set
    // rather than asserting a specific slot assignment.
    let mut want = [data.addr, data.addr + 4];
    want.sort_unstable();
    let mut got_vals: Vec<u64> = got
        .data
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_le_bytes(*c))
        .collect();
    got_vals.sort_unstable();
    assert_eq!(
        got_vals, want,
        "GOT slots hold a and b resolved addresses: got {got_vals:#x?}, want {want:#x?}"
    );

    let text = sections
        .iter()
        .find(|s| s.sectname == b"__text")
        .expect("__text section present");
    let targets =
        assert_movq_targets_in_got(text.data, text.addr, got.addr, got.size);
    // Two extern globals -> two GOT_LOAD fixups, each landing on its own slot.
    let mut distinct = targets;
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        2,
        "two distinct GOT slots addressed by __text: {distinct:#x?}"
    );
}

// --- structural checks ----------------------------------------------------

/// Locates every RIP-relative `movq sym@GOTPCREL(%rip), %reg` (the encoding
/// clang emits for a `GOT_LOAD`) in `__text`, decodes each 32-bit displacement,
/// and asserts every effective target lands on an 8-byte-aligned slot within
/// the `__got` extent `[got_addr, got_addr + got_size)`. A displacement that
/// overflowed would have failed the link; this confirms each resolved fixup
/// actually addresses a real GOT slot. Returns the list of targets.
fn assert_movq_targets_in_got(
    text: &[u8],
    text_addr: u64,
    got_addr: u64,
    got_size: u64,
) -> Vec<u64> {
    let mut targets = Vec::new();
    let mut i = 0;
    while i + 7 <= text.len() {
        if is_got_load_mov(&text[i..i + 3]) {
            let disp = i32::from_le_bytes(
                <[u8; 4]>::try_from(&text[i + 3..i + 7]).unwrap(),
            );
            // RIP points at the end of this 7-byte instruction.
            let rip = text_addr + u64::try_from(i).unwrap() + 7;
            let target = rip.wrapping_add(i64::from(disp).cast_unsigned());
            assert!(
                disp.unsigned_abs() < (1 << 30),
                "GOT displacement {disp} unexpectedly large"
            );
            assert!(
                target >= got_addr
                    && target < got_addr + got_size
                    && (target - got_addr).is_multiple_of(8),
                "`GOT_LOAD` target {:#x} is not an aligned slot within \
                 __got [{:#x}, {:#x})",
                target,
                got_addr,
                got_addr + got_size
            );
            targets.push(target);
        }
        i += 1;
    }
    assert!(
        !targets.is_empty(),
        "no RIP-relative movq (GOT_LOAD) found in __text"
    );
    targets
}

/// Whether `p` opens a `mov r64, [rip+disp32]`: a REX.W prefix (0x48..=0x4f),
/// opcode `0x8b`, and a ModR/M byte with mod=00, r/m=101 (RIP-relative).
fn is_got_load_mov(p: &[u8]) -> bool {
    (0x48..=0x4f).contains(&p[0]) && p[1] == 0x8b && (p[2] & 0xc7) == 0x05
}

/// Reads the first 8 bytes of `buf` as a little-endian `u64` (zero-padded).
fn read_u64_le(buf: &[u8]) -> u64 {
    let mut wide = [0u8; 8];
    let n = wide.len().min(buf.len());
    wide[..n].copy_from_slice(&buf[..n]);
    u64::from_le_bytes(wide)
}

// --- tool helpers ---------------------------------------------------------

/// Returns the clang binary if it can compile for `triple`, else `None`.
fn darwin_clang(triple: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let probe = std::env::temp_dir().join("xold_macho_got_probe.o");
    let ok = Command::new(clang.as_os_str())
        .args([triple, "-c", "-x", "c", "-", "-o"])
        .arg(&probe)
        .stdin(Stdio::null())
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&probe);
    ok.then_some(clang)
}

/// Compiles `src` for `triple` into `out`.
fn compile(clang: &Path, triple: &str, src: &[u8], out: &Path) -> bool {
    // Derive the temp source name from the output stem so parallel tests do not
    // clobber a shared source file mid-compile.
    let stem = out
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("xold_macho_got_src");
    let srcfile = std::env::temp_dir().join(format!("{stem}.c"));
    let _ = fs::write(&srcfile, src);
    Command::new(clang)
        .args([triple, "-c"])
        .arg(&srcfile)
        .arg("-o")
        .arg(out)
        .status()
        .is_ok_and(|s| s.success())
}

/// Links `objs` with xold into `out`, returning success.
fn xold_link(objs: &[&Path], out: &Path) -> bool {
    let mut cmd = Command::new(xold_bin());
    cmd.args(objs.iter().copied()).args(["-o"]).arg(out);
    cmd.status().is_ok_and(|s| s.success())
}

/// Runs `file` on `path` and returns its stdout.
fn file_type(path: &Path) -> String {
    String::from_utf8_lossy(
        &Command::new("file")
            .arg(path)
            .output()
            .map(|o| o.stdout)
            .unwrap_or_default(),
    )
    .into_owned()
}

/// Runs `llvm-objdump --macho --private-headers` on `path`, returning stdout.
fn objdump_headers(path: &Path) -> String {
    String::from_utf8_lossy(
        &Command::new("llvm-objdump")
            .args(["--macho", "--private-headers"])
            .arg(path)
            .output()
            .map(|o| o.stdout)
            .unwrap_or_default(),
    )
    .into_owned()
}
