//! Merge-pool offsets are checked, not clamped.
//!
//! A piece's offset within its pool and its length are both `u32`: that is
//! what the piece map and the remapped relocation addends carry. The
//! conversions swallowed their failures -- an offset past 4 GiB became
//! `u32::MAX` and a longer piece became length zero -- so every piece past the
//! boundary was mapped to the same wrong place and the addends naming them
//! were remapped to garbage, with nothing said. A `-g` link that deduplicates
//! more than 4 GiB of `.debug_str` reaches it. The pool and scope indices had
//! the same shape, and clamping either would have given two pools one identity
//! and merged their content.
//!
//! Stated plainly: the threshold cannot be reached in a test -- it needs a
//! four-gigabyte pool. What these tests guard is the path that was
//! restructured to make the checks possible: the deduplicating hit, the
//! inserting miss, and the addend remap that reads the result. They would
//! catch a mistake in the rewrite, not the overflow itself.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_to};

mod common;

/// Two translation units sharing three literals and holding one each of their
/// own, so the pool takes both the deduplicating and the inserting path, and
/// the program reads every one back.
const A_SRC: &[u8] = b"const char *a1(void) { return \"shared alpha\"; }\n\
    const char *a2(void) { return \"shared bravo\"; }\n\
    const char *a3(void) { return \"only in a\"; }\n";

const B_SRC: &[u8] = b"const char *b1(void) { return \"shared alpha\"; }\n\
    const char *b2(void) { return \"shared bravo\"; }\n\
    const char *b3(void) { return \"only in b\"; }\n";

const MAIN_SRC: &[u8] = b"const char *a1(void); const char *a2(void);\n\
    const char *a3(void); const char *b1(void);\n\
    const char *b2(void); const char *b3(void);\n\
    static int eq(const char *p, const char *q)\n\
    {\n\
        while (*p && *p == *q) { p++; q++; }\n\
        return *p == *q;\n\
    }\n\
    static void bye(int code)\n\
    {\n\
        __asm__ volatile(\"syscall\" :: \"a\"(60), \"D\"((long)code));\n\
        __builtin_unreachable();\n\
    }\n\
    void _start(void)\n\
    {\n\
        int rc = 0;\n\
        if (a1() != b1()) { rc |= 1; }\n\
        if (a2() != b2()) { rc |= 2; }\n\
        if (a3() == b3()) { rc |= 4; }\n\
        if (!eq(a1(), \"shared alpha\")) { rc |= 8; }\n\
        if (!eq(a2(), \"shared bravo\")) { rc |= 16; }\n\
        if (!eq(a3(), \"only in a\")) { rc |= 32; }\n\
        if (!eq(b3(), \"only in b\")) { rc |= 64; }\n\
        bye(rc);\n\
    }\n";

/// The program reads every literal back, shared and unshared alike.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn every_pooled_piece_maps_to_its_own_bytes() {
    let Some(dir) = workdir("run") else {
        return;
    };
    let Some(prog) = link(&dir) else {
        return;
    };
    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        assert_pool_contract(&fs::read(&prog).expect("read linked image"));
        let _ = fs::remove_dir_all(&dir);
        return;
    }
    let code = Command::new(&prog)
        .status()
        .expect("linked program must run")
        .code();
    assert_eq!(
        code,
        Some(0),
        "bits 0-2 mean two pieces that should share, or should not, got the \
         wrong identity; bits 3-6 mean a piece maps to the wrong bytes"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// And the pool really did deduplicate, so the hit path was taken.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_shared_literals_are_stored_once() {
    let Some(dir) = workdir("dedup") else {
        return;
    };
    let Some(prog) = link(&dir) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");
    let pool = rodata(&bytes).expect(".rodata present");
    assert_eq!(
        occurrences(&pool, b"shared alpha"),
        1,
        "a literal both inputs carry is stored once"
    );
    assert_eq!(
        occurrences(&pool, b"only in a"),
        1,
        "and one only a single input carries is stored once too"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping merge-pool-bounds {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_poolbounds_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the three objects and links them.
fn link(dir: &Path) -> Option<PathBuf> {
    let a = compile(dir, A_SRC, "a")?;
    let b = compile(dir, B_SRC, "b")?;
    let m = compile(dir, MAIN_SRC, "m")?;
    let out = dir.join("prog");
    let res = link_to(&[a, b, m], &out, b"_start", false, IcfMode::None, false);
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    Some(out)
}

/// Compiles one translation unit.
fn compile(dir: &Path, src: &[u8], stem: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src_path = dir.join(format!("{stem}.c"));
    let obj = dir.join(format!("{stem}.o"));
    fs::write(&src_path, src).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fno-pic", "-O1", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping merge-pool-bounds: clang cannot build it");
        return None;
    }
    Some(obj)
}

// --- readers ---------------------------------------------------------------

/// The bytes of `.rodata`, which is where the string pool lands.
fn rodata(bytes: &[u8]) -> Option<Vec<u8>> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".rodata")?;
    obj.section_data(shdr).ok().map(<[u8]>::to_vec)
}

/// How many times `needle` appears in `hay`.
fn occurrences(hay: &[u8], needle: &[u8]) -> usize {
    hay.windows(needle.len()).filter(|w| *w == needle).count()
}

/// Decodes the immediate pointer each accessor returns, reproducing every
/// identity and byte comparison in `_start` without executing foreign ELF.
fn assert_pool_contract(bytes: &[u8]) {
    let obj = ObjectFile::parse(bytes).expect("valid ELF");
    let symtab = obj.symbol_table().expect("read symtab").expect("symtab");
    let accessor = |name: &[u8]| {
        let addr = symtab
            .syms
            .iter()
            .find(|sym| symtab.name(sym) == name)
            .map(|sym| sym.st_value.get())
            .unwrap_or_else(|| {
                panic!("{} is defined", String::from_utf8_lossy(name))
            });
        let body = image_at(&obj, addr, 6).expect("accessor body");
        assert_eq!(body[0], 0xb8, "accessor returns an immediate pointer");
        u64::from(u32::from_le_bytes(body[1..5].try_into().unwrap()))
    };
    let a1 = accessor(b"a1");
    let a2 = accessor(b"a2");
    let a3 = accessor(b"a3");
    let b1 = accessor(b"b1");
    let b2 = accessor(b"b2");
    let b3 = accessor(b"b3");
    assert_eq!(a1, b1, "shared alpha has one identity");
    assert_eq!(a2, b2, "shared bravo has one identity");
    assert_ne!(a3, b3, "the private literals remain distinct");
    for (addr, want) in [
        (a1, b"shared alpha".as_slice()),
        (a2, b"shared bravo"),
        (a3, b"only in a"),
        (b3, b"only in b"),
    ] {
        assert_eq!(
            cstring_at(&obj, addr),
            Some(want),
            "piece maps to its bytes"
        );
    }
}

fn image_at<'a>(
    obj: &ObjectFile<'a>,
    addr: u64,
    len: usize,
) -> Option<&'a [u8]> {
    for sec in obj.sections() {
        let base = sec.sh_addr.get();
        if addr < base || addr >= base.saturating_add(sec.sh_size.get()) {
            continue;
        }
        let at = usize::try_from(addr - base).ok()?;
        return obj.section_data(sec).ok()?.get(at..at.checked_add(len)?);
    }
    None
}

fn cstring_at<'a>(obj: &ObjectFile<'a>, addr: u64) -> Option<&'a [u8]> {
    for sec in obj.sections() {
        let base = sec.sh_addr.get();
        if addr < base || addr >= base.saturating_add(sec.sh_size.get()) {
            continue;
        }
        let data = obj.section_data(sec).ok()?;
        let at = usize::try_from(addr - base).ok()?;
        let tail = data.get(at..)?;
        let end = tail.iter().position(|&byte| byte == 0)?;
        return Some(&tail[..end]);
    }
    None
}
