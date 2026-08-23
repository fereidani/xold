//! End-to-end tests for `__start_NAME` / `__stop_NAME`: the bounds a linker
//! supplies for a section whose name is a valid C identifier.
//!
//! A program keeps a table by putting each entry in its own translation unit
//! with `__attribute__((section("mysec")))` and walking the result between the
//! two symbols. Two things have to hold for that walk to be right, and neither
//! is free:
//!
//! - the entries from every input have to be laid out as one contiguous run,
//!   with nothing else between them, or the walk reads whatever the linker put
//!   in the gap;
//! - the two symbols have to be defined, at the first byte of that run and one
//!   past its last. Without them the link fails outright on a static image, and
//!   the program has no way to find its own table.
//!
//! The tests link two producer objects and one consumer, run the result, and
//! compare what it printed against the same program linked by the system
//! toolchain. Each producer contributes an ordinary global *before* its
//! entries, so on input the two contributions are separated by another
//! member of the same output section: the run can only come out contiguous if
//! the linker gathered it.
//!
//! Gated on `clang`, `gcc` (to locate the crt objects), and the system
//! `ld.so`; if absent the tests print a note and return, so the build never
//! fails over a missing toolchain.

#![allow(clippy::similar_names)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{
    elf::{ObjectFile, constants::SHN_ABS},
    icf::IcfMode,
    linker::link_dyn_exec,
};

mod common;

/// The table entry shape and the macro that files one in `mysec`. Sixteen
/// bytes with no relocation in them, so the walk reads the entries themselves
/// rather than anything they point at, and writable so the entries share an
/// output section with the ordinary globals around them.
const TBL_H: &[u8] = b"struct item { int value; char name[12]; };\n\
     #define ITEM(sym, v) \\\n\
         static struct item it_##sym \\\n\
             __attribute__((used, section(\"mysec\"))) = { (v), #sym }\n";

/// The first producer: an ordinary global, then two entries.
const A_SRC: &[u8] = b"#include \"tbl.h\"\n\
     int pad_a = 11;\n\
     ITEM(alpha, 1);\n\
     ITEM(bravo, 2);\n";

/// The second producer: another global, then the third entry. Its global sits
/// between the two contributions in input order.
const B_SRC: &[u8] = b"#include \"tbl.h\"\n\
     int pad_b = 21;\n\
     ITEM(charlie, 3);\n";

/// The consumer: walks the range and prints what it finds, then the globals,
/// so a run that swallowed a neighbour or lost an entry shows up in the
/// output.
const MAIN_SRC: &[u8] = b"#include <stdio.h>\n\
     #include \"tbl.h\"\n\
     extern char __start_mysec[], __stop_mysec[];\n\
     extern int pad_a, pad_b;\n\
     int main(void){\n\
         const struct item *p = (const struct item *)__start_mysec;\n\
         const struct item *e = (const struct item *)__stop_mysec;\n\
         printf(\"n=%ld\\n\", (long)(e - p));\n\
         for (; p < e; p++) printf(\"%s=%d\\n\", p->name, p->value);\n\
         printf(\"pads=%d %d\\n\", pad_a, pad_b);\n\
         return 0;\n\
     }\n";

/// What the program must print: three entries in input order, then the
/// globals. The same string the system toolchain produces.
const EXPECTED: &str = "n=3\nalpha=1\nbravo=2\ncharlie=3\npads=11 21\n";

/// Byte size of one `struct item`, and so of the whole run at three entries.
const ITEM_SIZE: u64 = 16;
const RUN_SIZE: u64 = 3 * ITEM_SIZE;

/// A fresh per-test working directory under the system temp dir.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xold_startstop_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// The host toolchain needed for these tests, or `None` (with a note) when a
/// piece is missing.
struct Harness {
    clang: PathBuf,
    crt1: PathBuf,
    crti: PathBuf,
    crtn: PathBuf,
    libc: PathBuf,
    interp: Vec<u8>,
}

impl Harness {
    /// Collects the harness, or returns `None` when a piece is missing.
    fn detect() -> Option<Self> {
        let Some(clang) = which("clang") else {
            eprintln!("skipping start/stop tests: clang unavailable");
            return None;
        };
        Some(Self {
            clang,
            crt1: crt_file("crt1.o")?,
            crti: crt_file("crti.o")?,
            crtn: crt_file("crtn.o")?,
            libc: libc_so()?,
            interp: interpreter()?,
        })
    }

    /// Compiles one source of `dir` into an object beside it.
    fn compile(&self, dir: &Path, name: &str, src: &[u8]) -> PathBuf {
        let src_path = dir.join(format!("{name}.c"));
        let obj = dir.join(format!("{name}.o"));
        fs::write(&src_path, src).expect("write source");
        let ok = Command::new(&self.clang)
            .args(["--target=x86_64-linux-gnu", "-fPIE", "-c"])
            .arg(&src_path)
            .arg("-o")
            .arg(&obj)
            .status()
            .expect("clang runs")
            .success();
        assert!(ok, "host clang must compile {name}.c");
        obj
    }

    /// Writes the header and the three sources into `dir` and compiles them,
    /// returning the object paths in link order.
    fn compile_all(&self, dir: &Path) -> Vec<PathBuf> {
        fs::write(dir.join("tbl.h"), TBL_H).expect("write header");
        [("m", MAIN_SRC), ("a", A_SRC), ("b", B_SRC)]
            .into_iter()
            .map(|(name, src)| self.compile(dir, name, src))
            .collect()
    }

    /// Links `objs` plus the crt objects and libc into `prog` with xold.
    fn link(&self, objs: &[PathBuf], prog: &Path) {
        let mut inputs = vec![self.crti.clone(), self.crt1.clone()];
        inputs.extend_from_slice(objs);
        inputs.push(self.crtn.clone());
        inputs.push(self.libc.clone());
        link_dyn_exec(
            &inputs,
            prog,
            b"_start",
            &self.interp,
            false,
            IcfMode::None,
            false,
        )
        .expect("xold link must succeed");
    }

    /// Links the same sources with the system toolchain, for comparison.
    fn reference(&self, dir: &Path, prog: &Path) -> bool {
        Command::new(&self.clang)
            .arg("-fPIE")
            .args([dir.join("m.c"), dir.join("a.c"), dir.join("b.c")])
            .arg("-o")
            .arg(prog)
            .status()
            .is_ok_and(|s| s.success())
    }
}

/// Runs `prog` and returns its stdout, asserting it exited cleanly.
fn run(prog: &Path) -> String {
    let out = Command::new(prog)
        .output()
        .expect("linked program must be runnable");
    assert!(
        out.status.success(),
        "{} exited {:?}",
        prog.display(),
        out.status
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The `(value, st_shndx)` of a `.symtab` entry, or `None` when the name is
/// not defined in the image.
fn symbol(bytes: &[u8], want: &[u8]) -> Option<(u64, u16)> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok()??;
    symtab
        .syms
        .iter()
        .find(|s| symtab.name(s) == want)
        .map(|s| (s.st_value.get(), s.st_shndx.get()))
}

/// The `st_size` of a `.symtab` entry, or `None` when the name is not defined
/// in the image.
fn symbol_size(bytes: &[u8], want: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok()??;
    symtab
        .syms
        .iter()
        .find(|s| symtab.name(s) == want)
        .map(|s| s.st_size.get())
}

/// The bounds the linker defined for `mysec`, as `(start, stop)`.
///
/// Both name the section the run sits in rather than being absolute. That is
/// what tells a loader the value moves with the load base; an `SHN_ABS` bound
/// keeps its link-time value in an image mapped anywhere else. See
/// `xold::defsym`.
fn bounds(bytes: &[u8]) -> (u64, u64) {
    let (start, start_shndx) =
        symbol(bytes, b"__start_mysec").expect("__start_mysec is defined");
    let (stop, stop_shndx) =
        symbol(bytes, b"__stop_mysec").expect("__stop_mysec is defined");
    assert_ne!(
        start_shndx, SHN_ABS,
        "a linker-supplied bound names the section its run sits in"
    );
    assert_eq!(
        start_shndx, stop_shndx,
        "both bounds of a run name the one section"
    );
    (start, stop)
}

/// The whole point: the two bounds are defined, the entries from both
/// producers land between them in input order, and the program reads the
/// values it wrote.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn start_stop_bounds_span_both_objects() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("walk");
    let objs = h.compile_all(&dir);
    let prog = dir.join("walk_prog");
    h.link(&objs, &prog);

    assert_eq!(
        run(&prog),
        EXPECTED,
        "the table walk read the wrong entries"
    );

    // Three entries, so the bounds are exactly `RUN_SIZE` apart. Together with
    // the values printed above that is contiguity: a run holding a
    // neighbouring member would be wider, one that lost a member narrower, and
    // either way an entry would read wrong.
    let bytes = fs::read(&prog).expect("read output");
    let (start, stop) = bounds(&bytes);
    assert_eq!(
        stop.checked_sub(start),
        Some(RUN_SIZE),
        "the bounds must span exactly the three contributed entries"
    );

    // The run lies inside one output section, wherever its flags sent it.
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    assert!(
        obj.sections().iter().any(|s| {
            let addr = s.sh_addr.get();
            addr != 0 && start >= addr && stop <= addr + s.sh_size.get()
        }),
        "the run must lie within one output section"
    );
}

/// Cross-check against the system toolchain: both links must print the same
/// thing, so the test cannot pass by agreeing with itself.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn start_stop_matches_the_system_linker() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("ref");
    let objs = h.compile_all(&dir);
    let prog = dir.join("ref_xold");
    h.link(&objs, &prog);
    let reference = dir.join("ref_system");
    if !h.reference(&dir, &reference) {
        eprintln!("skipping cross-check: the system link failed");
        return;
    }
    assert_eq!(
        run(&prog),
        run(&reference),
        "xold and the system linker must walk the same table"
    );
}

/// The bounds are `PROVIDE`-style: an input that defines `__start_mysec` for
/// itself keeps that definition, and the linker supplies only the other half.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_input_definition_wins_over_the_linker_bound() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("own");
    let mut objs = h.compile_all(&dir);
    objs.push(h.compile(&dir, "own", b"char __start_mysec[1] = { 7 };\n"));
    let prog = dir.join("own_prog");
    h.link(&objs, &prog);

    let bytes = fs::read(&prog).expect("read output");
    // The extent is what tells the two apart. Both bounds now name a section
    // -- a linker bound is a place in the image, not a constant -- so the
    // section index no longer distinguishes them, but only the input's
    // definition declares a size: it is a one-byte object, where a bound
    // describes an edge and has no length.
    assert_eq!(
        symbol_size(&bytes, b"__start_mysec"),
        Some(1),
        "the input's own definition must survive, with its own extent"
    );
    assert_eq!(
        symbol_size(&bytes, b"__stop_mysec"),
        Some(0),
        "the linker still supplies the half nothing defined"
    );
}
