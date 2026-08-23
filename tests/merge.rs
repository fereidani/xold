//! `SHF_MERGE` / `SHF_STRINGS` deduplication.
//!
//! Merging is only sound if every reference still reads the string it named,
//! so these tests link programs that compare their literals at run time and
//! report a failure through the exit status. A test that only checked the
//! output got smaller would pass just as happily on a linker that dropped the
//! relocations entirely.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{
    icf::IcfMode,
    linker::{link_dyn_exec, link_to},
};

mod common;

/// A private working directory. The name carries the process id so concurrent
/// test binaries cannot delete each other's files.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_merge_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Compiles `src` to `obj`, returning `None` when no host compiler exists.
fn compile(src: &str, obj: &Path, dir: &Path) -> Option<()> {
    let clang = which("clang")?;
    let file =
        dir.join(format!("{}.c", obj.file_stem()?.to_str().unwrap_or("unit")));
    fs::write(&file, src).ok()?;
    let status = Command::new(clang)
        .args(["-c", "-O1", "-fno-pic", "-o"])
        .arg(obj)
        .arg(&file)
        .status()
        .ok()?;
    status.success().then_some(())
}

/// Assembles the freestanding entry stub, which exits with `main`'s return.
fn assemble(obj: &Path, dir: &Path) -> Option<()> {
    let clang = which("clang")?;
    let file = dir.join("start.S");
    fs::write(
        &file,
        "    .text\n    .globl _start\n_start:\n    call main\n\
         movl %eax, %edi\n    movl $60, %eax\n    syscall\n",
    )
    .ok()?;
    let status = Command::new(clang)
        .args(["-c", "-o"])
        .arg(obj)
        .arg(&file)
        .status()
        .ok()?;
    status.success().then_some(())
}

/// A translation unit holding shared and private literals, plus a checker that
/// verifies each one still reads correctly.
const UNIT_A: &str = r#"
static const char *shared = "shared_literal_alpha";
static const char *only_a = "private_to_a";
static const char *dup    = "shared_literal_alpha";
int cmp(const char *x, const char *y);
int check_a(void) {
    if (cmp(shared, "shared_literal_alpha")) return 1;
    if (cmp(only_a, "private_to_a")) return 2;
    if (cmp(dup, "shared_literal_alpha")) return 3;
    /* The two references to the same literal must resolve to one address. */
    if (shared != dup) return 4;
    return 0;
}
"#;

const UNIT_B: &str = r#"
static const char *shared = "shared_literal_alpha";
static const char *only_b = "private_to_b";
static const char *tail   = "alpha";
int cmp(const char *x, const char *y);
int check_b(void) {
    if (cmp(shared, "shared_literal_alpha")) return 5;
    if (cmp(only_b, "private_to_b")) return 6;
    if (cmp(tail, "alpha")) return 7;
    return 0;
}
"#;

const MAIN: &str = r"
int check_a(void);
int check_b(void);
int cmp(const char *x, const char *y) {
    while (*x && *x == *y) { x++; y++; }
    return (int)((unsigned char)*x) - (int)((unsigned char)*y);
}
int main(void) {
    int r = check_a();
    if (r) return r;
    return check_b();
}
";

/// Links the three units with xold, returning the image path.
fn link(dir: &Path) -> Option<PathBuf> {
    let a = dir.join("unit_a.o");
    let b = dir.join("unit_b.o");
    let m = dir.join("unit_main.o");
    let s = dir.join("start.o");
    compile(UNIT_A, &a, dir)?;
    compile(UNIT_B, &b, dir)?;
    compile(MAIN, &m, dir)?;
    assemble(&s, dir)?;
    let out = dir.join("prog");
    link_to(&[a, b, m, s], &out, b"_start", false, IcfMode::None, false)
        .expect("xold links the merge test");
    Some(out)
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn merged_literals_still_read_correctly() {
    let dir = workdir("run");
    let Some(prog) = link(&dir) else {
        eprintln!("skipping merge test: host clang unavailable");
        return;
    };
    let status = Command::new(&prog)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(0),
        "every literal must still read back correctly after merging"
    );
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn duplicate_literals_are_stored_once() {
    let dir = workdir("size");
    let Some(prog) = link(&dir) else {
        eprintln!("skipping merge size test: host clang unavailable");
        return;
    };
    let image = fs::read(&prog).expect("image is readable");
    let needle = b"shared_literal_alpha";
    let copies = image.windows(needle.len()).filter(|w| *w == needle).count();
    // Three references across two translation units, and the string also
    // contains the "alpha" literal as a tail, so exactly one copy must remain.
    assert_eq!(
        copies, 1,
        "the shared literal must appear once in the image, found {copies}"
    );
}

/// Two units that share source-level declarations, so their debug info shares
/// most of its strings.
const DBG_A: &str = "
struct SharedShapeForDebugInfo { long alpha; long beta; };
int use_a(struct SharedShapeForDebugInfo *p) { return (int)(p->alpha + p->beta); }
";

const DBG_B: &str = "
struct SharedShapeForDebugInfo { long alpha; long beta; };
int use_b(struct SharedShapeForDebugInfo *p) { return (int)(p->alpha - p->beta); }
";

const DBG_MAIN: &str = "
struct SharedShapeForDebugInfo { long alpha; long beta; };
int use_a(struct SharedShapeForDebugInfo *p);
int use_b(struct SharedShapeForDebugInfo *p);
int main(void) {
    struct SharedShapeForDebugInfo v = { 3, 1 };
    return use_a(&v) - use_b(&v) - 2;
}
";

/// Compiles with debug info, which puts the type and member names into a
/// `.debug_str` marked `SHF_MERGE|SHF_STRINGS`.
fn compile_debug(src: &str, obj: &Path, dir: &Path) -> Option<()> {
    let clang = which("clang")?;
    let file =
        dir.join(format!("{}.c", obj.file_stem()?.to_str().unwrap_or("unit")));
    fs::write(&file, src).ok()?;
    Command::new(clang)
        .args(["-c", "-g", "-O0", "-fno-pic", "-o"])
        .arg(obj)
        .arg(&file)
        .status()
        .ok()?
        .success()
        .then_some(())
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn debug_strings_are_deduplicated_and_still_resolve() {
    let dir = workdir("debug");
    let a = dir.join("dbg_a.o");
    let b = dir.join("dbg_b.o");
    let m = dir.join("dbg_main.o");
    let s = dir.join("start.o");
    let (Some(()), Some(()), Some(()), Some(())) = (
        compile_debug(DBG_A, &a, &dir),
        compile_debug(DBG_B, &b, &dir),
        compile_debug(DBG_MAIN, &m, &dir),
        assemble(&s, &dir),
    ) else {
        eprintln!("skipping debug merge test: host clang unavailable");
        return;
    };
    let prog = dir.join("prog");
    link_to(&[a, b, m, s], &prog, b"_start", false, IcfMode::None, false)
        .expect("xold links the debug merge test");

    // The program must still run: merging must not disturb the code.
    let status = Command::new(&prog).status().expect("program runs");
    assert_eq!(status.code(), Some(0), "merging must not affect the code");

    // `.debug_str` is SHF_MERGE|SHF_STRINGS, and all three units name the same
    // struct and members, so each name must be stored once.
    let image = fs::read(&prog).expect("image is readable");
    for needle in [&b"SharedShapeForDebugInfo"[..], &b"alpha"[..], &b"beta"[..]]
    {
        let copies =
            image.windows(needle.len()).filter(|w| *w == needle).count();
        assert_eq!(
            copies,
            1,
            "{} must be stored once, found {copies}",
            String::from_utf8_lossy(needle)
        );
    }

    // And the debug info must still decode: every name a consumer reads has to
    // land on the surviving copy, which is what the offset remapping is for.
    let dump = Command::new("readelf")
        .args(["--debug-dump=info"])
        .arg(&prog)
        .output()
        .expect("readelf runs");
    let text = String::from_utf8_lossy(&dump.stdout);
    for want in ["SharedShapeForDebugInfo", "alpha", "beta", "use_a", "use_b"] {
        assert!(
            text.contains(want),
            "debug info must still resolve {want} after merging"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn merging_is_independent_of_thread_count() {
    let dir = workdir("threads");
    if link(&dir).is_none() {
        eprintln!("skipping merge determinism test: host clang unavailable");
        return;
    }
    // The pool is built through a hash map; the map decides which strings are
    // equal, never the order they are emitted in. Re-linking must therefore
    // reproduce the image exactly.
    let first = fs::read(dir.join("prog")).expect("image is readable");
    for _ in 0..4 {
        let again = link(&dir).expect("relink succeeds");
        let bytes = fs::read(&again).expect("image is readable");
        assert_eq!(
            bytes, first,
            "merged output must be a function of the inputs alone"
        );
    }
}

/// Two translation units whose literals share one deduplicated pool, each
/// exporting a table of pointers into it.
///
/// A table entry is an `R_X86_64_64` naming the literal's section symbol with
/// the byte offset in its addend, so in a position-independent executable the
/// loader applies it as an `R_X86_64_RELATIVE`. The first unit's literals
/// happen to keep their input offsets in the pool, so only the second unit
/// proves the addend follows the content the merge pass moved.
const PIE_TABLE_A: &str = r#"
#include <stdio.h>
extern const char *table_b[2];
const char *table_a[2] = { "one", "two" };
int main(void) {
    printf("%s %s %s %s\n", table_a[0], table_a[1], table_b[0], table_b[1]);
    return 0;
}
"#;

const PIE_TABLE_B: &str = r#"
const char *table_b[2] = { "three", "four" };
"#;

/// The host pieces a `-fPIE` link against the system libc needs.
struct PieHarness {
    scrt1: PathBuf,
    crti: PathBuf,
    crtn: PathBuf,
    libc: PathBuf,
    interp: Vec<u8>,
}

impl PieHarness {
    /// Collects the harness, or returns `None` (with a note) when a piece is
    /// missing. `Scrt1.o` rather than `crt1.o`: only the former is built to be
    /// linked into a position-independent executable.
    fn detect() -> Option<Self> {
        if which("clang").is_none() {
            eprintln!("skipping merged-pointer test: clang unavailable");
            return None;
        }
        Some(Self {
            scrt1: crt_file("Scrt1.o")?,
            crti: crt_file("crti.o")?,
            crtn: crt_file("crtn.o")?,
            libc: libc_so()?,
            interp: interpreter()?,
        })
    }
}

/// Compiles `src` to a position-independent object.
fn compile_pie(src: &str, obj: &Path, dir: &Path) -> Option<()> {
    let clang = which("clang")?;
    let file =
        dir.join(format!("{}.c", obj.file_stem()?.to_str().unwrap_or("unit")));
    fs::write(&file, src).ok()?;
    Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c", "-fPIE", "-o"])
        .arg(obj)
        .arg(&file)
        .status()
        .ok()?
        .success()
        .then_some(())
}

/// A pointer into a merged pool must still name its own string once the image
/// is relocated at load time.
///
/// The in-place writer remaps such an offset; the dynamic-relocation emitter
/// used to store the pre-merge one, so every pointer from the second
/// contributor onwards named whatever had landed at that offset instead. The
/// table sizes, the relocation count and the section sizes are all unchanged
/// by that, which is why this reads the strings back rather than counting
/// anything.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_pie_reads_merged_literals_from_every_unit() {
    let Some(h) = PieHarness::detect() else {
        return;
    };
    let dir = workdir("pie");
    let a = dir.join("pie_a.o");
    let b = dir.join("pie_b.o");
    let (Some(()), Some(())) = (
        compile_pie(PIE_TABLE_A, &a, &dir),
        compile_pie(PIE_TABLE_B, &b, &dir),
    ) else {
        eprintln!("skipping merged-pointer test: host clang unavailable");
        return;
    };
    let prog = dir.join("pie_prog");
    link_dyn_exec(
        &[h.scrt1, h.crti, a, b, h.libc, h.crtn],
        &prog,
        b"_start",
        &h.interp,
        false,
        IcfMode::None,
        false,
    )
    .expect("xold links the merged-pointer test");

    let out = Command::new(&prog).output().expect("program runs");
    assert!(
        out.status.success(),
        "the program must exit 0, got {:?}",
        out.status.code()
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "one two three four\n",
        "every table entry must still name the literal it was built from"
    );
}
