//! A dependency without `DT_SONAME` contributes its path as raw bytes.
//!
//! The soname a `DT_NEEDED` carries is a byte string, and so is a Unix path.
//! The fallback used to route the path through `str::to_str` and drop a
//! non-UTF-8 one to the empty string, which put an unresolvable empty
//! `DT_NEEDED` in the image. lld carries the raw bytes here
//! (`withLOption ? path::filename(path) : path`), so the two agree.
//!
//! A path that is not valid UTF-8 is unusual but legal, and the command line
//! deliberately accepts one: `Inputs` keeps arguments as `OsString` precisely
//! so a path it cannot decode still reaches the linker.
//!
//! Gated on `clang` and the host `ld.so`; if either is missing the test prints
//! a note and returns.

use std::{
    ffi::OsString,
    fs,
    os::unix::ffi::{OsStrExt, OsStringExt},
    path::PathBuf,
    process::Command,
};

use common::{interpreter, which};
use xold::{icf::IcfMode, linker::link_dyn_exec};

mod common;

/// The dependency. Built without `-Wl,-soname`, so it advertises no
/// `DT_SONAME` and the linker has to fall back to the path it was given.
const LIB_SRC: &[u8] = b"int lib_read(void) { return 7; }\n";

/// The program, which needs the dependency for `lib_read`.
const MAIN_SRC: &[u8] = b"extern int lib_read(void);\n\
    int main(void) { return lib_read(); }\n";

/// A freestanding `_start` that calls `main` then exits with its return value.
const START_SRC: &[u8] = b"    .text\n    .globl _start\n_start:\n    \
    call    main\n    movl    %eax, %edi\n    movl    $60, %eax\n    syscall\n";

/// The `DT_NEEDED` string is the dependency's path, byte for byte, including
/// the byte that makes it invalid UTF-8.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_non_utf8_dependency_path_reaches_dt_needed_intact() {
    let Some(clang) = which("clang") else {
        println!("clang not found; skipping");
        return;
    };
    let Some(interp) = interpreter() else {
        println!("no host interpreter; skipping");
        return;
    };
    let dir = std::env::temp_dir().join("xold_dep_soname_bytes");
    if fs::create_dir_all(&dir).is_err() {
        println!("cannot create the work directory; skipping");
        return;
    }

    // 0xff is not valid UTF-8 in any position, so `to_str` on this path
    // fails and the old fallback produced an empty name.
    let mut raw = b"lib".to_vec();
    raw.push(0xff);
    raw.extend_from_slice(b".so");
    let lib = dir.join(PathBuf::from(OsString::from_vec(raw)));
    let lib_path = lib.clone();

    let lib_c = dir.join("lib.c");
    let main_c = dir.join("main.c");
    let start_s = dir.join("start.s");
    if fs::write(&lib_c, LIB_SRC).is_err()
        || fs::write(&main_c, MAIN_SRC).is_err()
        || fs::write(&start_s, START_SRC).is_err()
    {
        println!("cannot write the sources; skipping");
        return;
    }

    // No `-Wl,-soname`: the dependency must advertise no DT_SONAME for the
    // path fallback to be the thing under test.
    if !run(&clang, &["-fPIC", "-shared", "-o"], &lib, &lib_c) {
        println!("clang cannot build the dependency; skipping");
        return;
    }
    let main_o = dir.join("main.o");
    let start_o = dir.join("start.o");
    if !run(&clang, &["-c", "-fPIC", "-o"], &main_o, &main_c)
        || !run(&clang, &["-c", "-o"], &start_o, &start_s)
    {
        println!("clang cannot build the objects; skipping");
        return;
    }

    let prog = dir.join("prog.out");
    let linked = link_dyn_exec(
        &[main_o, start_o, lib],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    );
    assert!(linked.is_ok(), "the link must succeed: {:?}", linked.err());

    // Named directly rather than through `-l`, so the whole path is the
    // soname, and the 0xff byte has to survive it.
    let expected = lib_path.as_os_str().as_bytes().to_vec();
    let image = fs::read(&prog).expect("the image is readable");
    let needed = dt_needed_strings(&image);
    assert!(
        needed.iter().any(|n| n == &expected),
        "DT_NEEDED does not carry the dependency path verbatim: {needed:?}"
    );
    assert!(
        expected.contains(&0xff),
        "the fixture path lost the byte that makes it non-UTF-8"
    );
    assert!(
        !needed.iter().any(Vec::is_empty),
        "an empty DT_NEEDED reached the image: {needed:?}"
    );
}

/// Runs `clang args... out input`, reporting whether it succeeded.
fn run(
    clang: &std::path::Path,
    args: &[&str],
    out: &std::path::Path,
    input: &std::path::Path,
) -> bool {
    Command::new(clang)
        .args(args)
        .arg(out)
        .arg(input)
        .status()
        .is_ok_and(|s| s.success())
}

/// Every `DT_NEEDED` string in the image, read out of `.dynamic` and
/// `.dynstr`. The section headers are used rather than `DT_STRTAB`, which
/// carries an address and not a file offset.
fn dt_needed_strings(image: &[u8]) -> Vec<Vec<u8>> {
    const DT_NEEDED: u64 = 1;
    let mut out = Vec::new();
    let Some((dyn_off, dyn_size)) = section_range(image, b".dynamic") else {
        return out;
    };
    let Some((str_off, _)) = section_range(image, b".dynstr") else {
        return out;
    };
    let mut offs = Vec::new();
    let mut at = usize::try_from(dyn_off).unwrap_or(0);
    let end = at.saturating_add(usize::try_from(dyn_size).unwrap_or(0));
    while at + 16 <= end {
        let Some(tag) = u64_at(image, at) else { break };
        let Some(val) = u64_at(image, at + 8) else {
            break;
        };
        if tag == 0 {
            break;
        }
        if tag == DT_NEEDED {
            offs.push(val);
        }
        at += 16;
    }
    let base = usize::try_from(str_off).unwrap_or(0);
    for off in offs {
        let start = base.saturating_add(usize::try_from(off).unwrap_or(0));
        let Some(bytes) = image.get(start..) else {
            continue;
        };
        let Some(nul) = bytes.iter().position(|&b| b == 0) else {
            continue;
        };
        out.push(bytes[..nul].to_vec());
    }
    out
}

/// The `(offset, size)` of the section named `name`.
fn section_range(image: &[u8], name: &[u8]) -> Option<(u64, u64)> {
    let shoff = u64_at(image, 0x28)?;
    let shentsize = usize::from(u16_at(image, 0x3a)?);
    let shnum = usize::from(u16_at(image, 0x3c)?);
    let shstrndx = usize::from(u16_at(image, 0x3e)?);
    let base = usize::try_from(shoff).ok()?;
    let str_hdr = base.checked_add(shstrndx.checked_mul(shentsize)?)?;
    let str_off = usize::try_from(u64_at(image, str_hdr + 0x18)?).ok()?;
    for i in 0..shnum {
        let hdr = base.checked_add(i.checked_mul(shentsize)?)?;
        let name_off = usize::try_from(u32_at(image, hdr)?).ok()?;
        let start = str_off.checked_add(name_off)?;
        let bytes = image.get(start..)?;
        let end = bytes.iter().position(|&b| b == 0)?;
        if bytes.get(..end)? == name {
            return Some((
                u64_at(image, hdr + 0x18)?,
                u64_at(image, hdr + 0x20)?,
            ));
        }
    }
    None
}

fn u16_at(image: &[u8], off: usize) -> Option<u16> {
    let b = image.get(off..off.checked_add(2)?)?;
    Some(u16::from_le_bytes([b[0], b[1]]))
}

fn u32_at(image: &[u8], off: usize) -> Option<u32> {
    let b = image.get(off..off.checked_add(4)?)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn u64_at(image: &[u8], off: usize) -> Option<u64> {
    let b = image.get(off..off.checked_add(8)?)?;
    let mut v = [0u8; 8];
    v.copy_from_slice(b);
    Some(u64::from_le_bytes(v))
}
