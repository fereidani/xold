//! End-to-end tests for `-l<name>` resolving to a static archive.
//!
//! A `-lt` that resolves to `libt.a` must be mined lazily through
//! [`Context::pull_archives`]: only members that satisfy still-undefined
//! symbols are extracted, iterating to a fixpoint, so an unreferenced member
//! is absent from the output while referenced members (and their transitive
//! dependencies) are present.
//!
//! Both the `-l` path and a direct `.a` input classify by content in
//! `parse_direct` and flow through the same `archives` vec, so they produce
//! byte-identical output. These tests pin that contract: the program runs, the
//! unused symbol is absent, the used symbol (and what it transitively needs) is
//! present, and the `-l` archive binary is no larger than the direct `.a`
//! binary.
//!
//! Gated on `clang`, `ar`, and `gcc` (to locate the crt objects and libc); if
//! any are absent the tests print a note and return, so the build never fails
//! over a missing toolchain.

// `crt1`/`crti`/`crtn` are the canonical crt-object names; renaming them would
// obscure the test.
#![allow(clippy::similar_names)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{archive_tool, crt_file, interpreter, libc_so, which};
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_dyn_exec, search};

mod common;

/// `used` returns 1. A standalone archive member so lazy extraction can omit
/// the sibling `unused` member.
const USED_SRC: &[u8] = b"int used(void){ return 1; }\n";

/// `unused` returns 2. A separate member that nothing references, so lazy
/// extraction must leave it out of the output.
const UNUSED_SRC: &[u8] = b"int unused(void){ return 2; }\n";

/// `helper` returns 5. `used` calls it, so pulling `used` must also pull
/// `helper` on the next fixpoint iteration.
const HELPER_SRC: &[u8] = b"int helper(void){ return 5; }\n";

/// `used` calls `helper` (a transitive dependency within the archive).
const USED_TRANS_SRC: &[u8] = b"int helper(void);\n\
     int used(void){ return helper() + 1; }\n";

/// Main calls `used`; exits 0 when `used()` returns 1.
const MAIN_SRC: &[u8] = b"int used(void);\n\
     int main(void){ return used() - 1; }\n";

/// Main calls `used` which calls `helper`; exits 0 when `used()` returns 6.
const MAIN_TRANS_SRC: &[u8] = b"int used(void);\n\
     int main(void){ return used() - 6; }\n";

/// Main refers to `used` weakly and does nothing but report whether the
/// reference was bound: exits 0 when it was not, 7 when it was. The archive
/// member defining `used` must stay out, so the address is null and the branch
/// the program wrote for that case is the one it takes.
const MAIN_WEAK_SRC: &[u8] = b"extern int used(void) __attribute__((weak));\n\
     int main(void){ return used ? 7 : 0; }\n";

/// `late` returns 5, in a member of its own.
const LATE_SRC: &[u8] = b"int late(void){ return 5; }\n";

/// `used` calls `late` -- strongly. Pulling this member for `used` is what
/// turns a weak reference to `late` elsewhere into a strong one.
const USED_LATE_SRC: &[u8] = b"int late(void);\n\
     int used(void){ return late() + 1; }\n";

/// Main refers to `late` weakly and to `used` strongly, so `late` is not
/// extractable when extraction starts. Exits 0 when `late` was bound anyway
/// (through the member pulled for `used`) and its value is right.
const MAIN_FIXPOINT_SRC: &[u8] =
    b"extern int late(void) __attribute__((weak));\n\
     int used(void);\n\
     int main(void){ if (!late) return 1; return used() + late() - 11; }\n";

/// Compiles `src` to `obj` with the host clang (`-fPIE`). Returns `None` when
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

/// Builds `archive` from the listed object files with the host `ar`. Members
/// are listed in the order given, which controls the archive symbol index.
fn make_archive(archive: &Path, members: &[&Path]) -> Option<()> {
    let ar = archive_tool()?;
    let dir = archive.parent().expect("archive has a parent");
    let _ = fs::create_dir_all(dir);
    let status = Command::new(ar)
        .args(["rcs"])
        .arg(archive)
        .args(members.iter())
        .status()
        .ok()?;
    status.success().then_some(())
}

/// A fresh per-test working directory under the system temp dir.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xold_archivel_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Whether `name` has a placed definition, as opposed to the `SHN_UNDEF` row
/// an unresolved weak reference keeps (lld writes that row too). Reads the
/// primary `.symtab`, which is where xold records the archive-pulled
/// definitions.
fn has_defined_symbol(bytes: &[u8], name: &[u8]) -> bool {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return false;
    };
    let Ok(Some(symtab)) = obj.symbol_table() else {
        return false;
    };
    symtab
        .iter()
        .any(|sym| symtab.name(sym) == name && sym.st_shndx.get() != 0)
}

/// Whether `name` appears at all, defined or not.
fn has_symbol(bytes: &[u8], name: &[u8]) -> bool {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return false;
    };
    let Ok(Some(symtab)) = obj.symbol_table() else {
        return false;
    };
    symtab.iter().any(|sym| symtab.name(sym) == name)
}

/// The host toolchain needed to build and run these tests.
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
            eprintln!("skipping -l archive tests: clang unavailable");
            return None;
        }
        if archive_tool().is_none() {
            eprintln!("skipping -l archive tests: ar unavailable");
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

    /// Links `main_obj`, the crt objects, `extra` (an archive or shared dep),
    /// and libc into `prog` as a dynamic executable.
    fn link(&self, main_obj: &Path, extra: &Path, prog: &Path) {
        link_dyn_exec(
            &[
                main_obj.to_path_buf(),
                self.crti.clone(),
                self.crt1.clone(),
                self.crtn.clone(),
                extra.to_path_buf(),
                self.libc.clone(),
            ],
            prog,
            b"_start",
            &self.interp,
            false,
            IcfMode::None,
            false,
        )
        .expect("xold link must succeed");
    }
}

/// The repro: `-L dir -lt` where `libt.a` has a `used` member and an
/// `unused` member. Lazy extraction must pull only `used`, so `unused` is
/// absent from the output, `used` is present, and the program runs (exit 0).
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn l_archive_extracts_lazily_unused_absent() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("lazy");

    let used_o = dir.join("used.o");
    let unused_o = dir.join("unused.o");
    compile(USED_SRC, &used_o).expect("clang compiles used");
    compile(UNUSED_SRC, &unused_o).expect("clang compiles unused");

    // Place the archive in a subdirectory so `-L dir -lt` resolves it by
    // search, exactly like the system linker.
    let lib_dir = dir.join("ld");
    let archive = lib_dir.join("libt.a");
    make_archive(&archive, &[&used_o, &unused_o]).expect("ar builds libt.a");

    // `-lt` resolved against the `-L` directory, the way `main.rs::parse` does.
    let resolved =
        search::find_library("t", std::slice::from_ref(&lib_dir), None)
            .expect("find_library");
    assert_eq!(
        resolved.file_name(),
        Some(std::ffi::OsStr::new("libt.a")),
        "-lt must resolve to the static archive"
    );

    let main_o = dir.join("main.o");
    compile(MAIN_SRC, &main_o).expect("clang compiles main");
    let prog = dir.join("prog");
    h.link(&main_o, &resolved, &prog);

    let bytes = fs::read(&prog).expect("read output");
    assert!(has_symbol(&bytes, b"used"), "used must be present");
    assert!(!has_symbol(&bytes, b"unused"), "unused must be absent");

    let status = Command::new(&prog)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(status.code(), Some(0), "lazy-pull program should exit 0");
}

/// A multi-member archive where `used` transitively needs `helper`. Tests the
/// fixpoint iteration in `pull_archives`: pulling `used` surfaces an undefined
/// `helper`, which the next iteration pulls. `unused` is never referenced.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn l_archive_fixpoint_pulls_transitive_member() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("fixpoint");

    let helper_o = dir.join("helper.o");
    let used_o = dir.join("used.o");
    let unused_o = dir.join("unused.o");
    compile(HELPER_SRC, &helper_o).expect("clang compiles helper");
    compile(USED_TRANS_SRC, &used_o).expect("clang compiles used");
    compile(UNUSED_SRC, &unused_o).expect("clang compiles unused");

    let lib_dir = dir.join("ld");
    let archive = lib_dir.join("libt.a");
    make_archive(&archive, &[&helper_o, &used_o, &unused_o])
        .expect("ar builds libt.a");

    let resolved =
        search::find_library("t", std::slice::from_ref(&lib_dir), None)
            .expect("find_library");
    assert_eq!(
        resolved.file_name(),
        Some(std::ffi::OsStr::new("libt.a")),
        "-lt must resolve to the static archive"
    );

    let main_o = dir.join("main.o");
    compile(MAIN_TRANS_SRC, &main_o).expect("clang compiles main");
    let prog = dir.join("prog");
    h.link(&main_o, &resolved, &prog);

    let bytes = fs::read(&prog).expect("read output");
    assert!(has_symbol(&bytes, b"used"), "used must be present");
    assert!(
        has_symbol(&bytes, b"helper"),
        "helper must be pulled transitively"
    );
    assert!(!has_symbol(&bytes, b"unused"), "unused must be absent");

    let status = Command::new(&prog)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(0),
        "transitive lazy-pull program should exit 0"
    );
}

/// An undefined *weak* reference does not extract an archive member, while a
/// strong reference to the same name does.
///
/// This is the mechanism behind the feature probe: a program declares a
/// function weak, tests its address and does without when it is null. Binding
/// it from an archive answers a question the program asked in order to hear
/// "no", and drags the member (and everything it needs) into an image that
/// never calls it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_weak_reference_does_not_extract_a_member() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("weak");

    let used_o = dir.join("used.o");
    let unused_o = dir.join("unused.o");
    compile(USED_SRC, &used_o).expect("clang compiles used");
    compile(UNUSED_SRC, &unused_o).expect("clang compiles unused");
    let archive = dir.join("libt.a");
    make_archive(&archive, &[&used_o, &unused_o]).expect("ar builds libt.a");

    let weak_o = dir.join("weak_main.o");
    compile(MAIN_WEAK_SRC, &weak_o).expect("clang compiles the weak main");
    let weak_prog = dir.join("weak_prog");
    h.link(&weak_o, &archive, &weak_prog);

    let bytes = fs::read(&weak_prog).expect("read output");
    assert!(
        !has_defined_symbol(&bytes, b"used"),
        "a weak reference must not extract the member defining `used`"
    );
    let status = Command::new(&weak_prog)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(0),
        "the unbound weak reference must be null at run time"
    );

    // The same archive, the same name, referenced strongly: the member is
    // extracted, which is what makes the case above a decision and not an
    // accident of the archive index.
    let strong_o = dir.join("strong_main.o");
    compile(MAIN_SRC, &strong_o).expect("clang compiles the strong main");
    let strong_prog = dir.join("strong_prog");
    h.link(&strong_o, &archive, &strong_prog);

    let bytes = fs::read(&strong_prog).expect("read output");
    assert!(
        has_symbol(&bytes, b"used"),
        "a strong reference must still extract the member"
    );
    let status = Command::new(&strong_prog)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(status.code(), Some(0), "strong-pull program should exit 0");
}

/// A name only weakly referenced becomes extractable once a member pulled for
/// some other name refers to it strongly.
///
/// `main` refers to `late` weakly and to `used` strongly. Extraction starts
/// with `used` alone; the member it pulls refers to `late` strongly, which
/// outranks the weak reference in the fold, so the next round of the fixpoint
/// pulls the member defining `late`. Deciding extractability once, before the
/// loop, would leave `late` unbound.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_member_pulled_for_one_name_makes_a_weak_name_extractable() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("weakfixpoint");

    let late_o = dir.join("late.o");
    let used_o = dir.join("used.o");
    compile(LATE_SRC, &late_o).expect("clang compiles late");
    compile(USED_LATE_SRC, &used_o).expect("clang compiles used");
    let archive = dir.join("libt.a");
    make_archive(&archive, &[&late_o, &used_o]).expect("ar builds libt.a");

    let main_o = dir.join("main.o");
    compile(MAIN_FIXPOINT_SRC, &main_o).expect("clang compiles main");
    let prog = dir.join("prog");
    h.link(&main_o, &archive, &prog);

    let bytes = fs::read(&prog).expect("read output");
    assert!(has_symbol(&bytes, b"used"), "used must be present");
    assert!(
        has_symbol(&bytes, b"late"),
        "the strong reference from the pulled member must extract `late`"
    );

    let status = Command::new(&prog)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(0),
        "used() + late() must be 11: both members are in and bound"
    );
}

/// A `-l` archive and a direct `.a` input classify the same way (by content in
/// `parse_direct`) and flow through the same lazy path, so the two binaries are
/// byte-identical in size.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn l_archive_matches_direct_archive_size() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("size");

    let used_o = dir.join("used.o");
    let unused_o = dir.join("unused.o");
    compile(USED_SRC, &used_o).expect("clang compiles used");
    compile(UNUSED_SRC, &unused_o).expect("clang compiles unused");

    let lib_dir = dir.join("ld");
    let archive = lib_dir.join("libt.a");
    make_archive(&archive, &[&used_o, &unused_o]).expect("ar builds libt.a");
    // A direct copy for the direct-`.a` link.
    let direct = dir.join("libt_direct.a");
    fs::copy(&archive, &direct).expect("copy archive");

    let main_o = dir.join("main.o");
    compile(MAIN_SRC, &main_o).expect("clang compiles main");

    let resolved =
        search::find_library("t", std::slice::from_ref(&lib_dir), None)
            .expect("find_library");
    let l_prog = dir.join("l_prog");
    h.link(&main_o, &resolved, &l_prog);
    let direct_prog = dir.join("direct_prog");
    h.link(&main_o, &direct, &direct_prog);

    let l_size = fs::metadata(&l_prog).map_or(0, |m| m.len());
    let direct_size = fs::metadata(&direct_prog).map_or(0, |m| m.len());
    assert_eq!(
        l_size, direct_size,
        "-l archive (size {l_size}) must equal direct .a (size {direct_size})"
    );
}
