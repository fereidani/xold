//! End-to-end tests for `DT_PREINIT_ARRAY`/`DT_INIT_ARRAY`/`DT_FINI_ARRAY`:
//! pre-initialisation functions, constructors and destructors linked against
//! the system libc.
//!
//! Each array is an allocated, writable section of its own type. Folding one
//! into `.data` and emitting no `DT_*_ARRAY` tag leaves the libc startup with
//! no array to walk, so its functions never run: these tests pin that down.
//! They link a C program whose `__attribute__((constructor))` sets a global
//! `main` later reads, run it, and assert the constructor ran. Both compile
//! flavours are covered:
//!
//! - default `clang -c` (non-PIC): the absolute references land in read-only
//!   `.text`, so the executable is `ET_EXEC` at a fixed base and the function
//!   pointers are resolved at link time.
//! - `clang -c -fPIE`: the constructor pointer in `.init_array` becomes an
//!   `R_X86_64_RELATIVE` dynamic relocation, so the executable is `ET_DYN` (a
//!   real PIE) and the loader patches the slot.
//!
//! A destructor test checks `DT_FINI_ARRAY` runs at exit. A structural test
//! verifies the tags exist and point at a real `.init_array` section of the
//! matching size, and cross-checks the tag presence against the system linker.
//!
//! `.preinit_array` gets the same treatment, plus the one thing that
//! distinguishes it: its functions must run before any `.init_array`
//! constructor, which the ordering test observes rather than infers.
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
    elf::{ObjectFile, constants::*},
    icf::IcfMode,
    linker::link_dyn_exec,
};

mod common;

/// A constructor that allocates a buffer and stamps it; `main` reads the
/// buffer, so the program only prints `buf=ok` and exits 0 if the constructor
/// ran before `main`.
const CTOR_SRC: &[u8] = b"#include <stdio.h>\n\
     #include <stdlib.h>\n\
     #include <string.h>\n\
     static char *buf;\n\
     __attribute__((constructor)) static void init(void){\n\
         buf = malloc(4); strcpy(buf, \"ok\");\n\
     }\n\
     int main(void){ printf(\"buf=%s\\n\", buf); return buf == NULL; }\n";

/// A destructor that writes a marker to stderr (unbuffered, so it survives
/// even when stdout flushing is skipped). Observed via the captured pipe.
const DTOR_SRC: &[u8] = b"#include <unistd.h>\n\
     __attribute__((destructor)) static void fini(void){\n\
         static const char m[] = \"dtor-ran\";\n\
         (void)!write(2, m, sizeof(m) - 1);\n\
     }\n\
     int main(void){ return 0; }\n";

/// A pre-initialisation function, a constructor and `main`, each appending its
/// own letter to a buffer in the order it ran. The runtime calls
/// `.preinit_array` before `.init_array`, so the only correct output is
/// `order=pim`: `p` missing means the array was never walked, and `ip` means it
/// was walked in the wrong place.
const PREINIT_SRC: &[u8] = b"#include <stdio.h>\n\
     static char order[8];\n\
     static int at;\n\
     static void preinit(int argc, char **argv, char **envp){\n\
         (void)argc; (void)argv; (void)envp;\n\
         if (at < 7) order[at++] = 'p';\n\
     }\n\
     __attribute__((section(\".preinit_array\"), used))\n\
     static void (*preinit_slot)(int, char **, char **) = preinit;\n\
     __attribute__((constructor)) static void ctor(void){\n\
         if (at < 7) order[at++] = 'i';\n\
     }\n\
     int main(void){\n\
         if (at < 7) order[at++] = 'm';\n\
         printf(\"order=%s\\n\", order); return 0;\n\
     }\n";

/// One half of the priority fixture: a constructor with the higher priority
/// number, so it must run *later*, plus one with no priority at all. Compiled
/// into its own object and named first on the command line, so link order and
/// priority order disagree.
const PRIO_HI_SRC: &[u8] = b"#include <stdio.h>\n\
     __attribute__((constructor(300))) static void c300(void){ puts(\"300\"); }\n\
     __attribute__((constructor)) static void hi(void){ puts(\"hi\"); }\n";

/// The other half: the lower priority number, which must run first, and a
/// second unprioritised constructor.
const PRIO_LO_SRC: &[u8] = b"#include <stdio.h>\n\
     __attribute__((constructor(101))) static void c101(void){ puts(\"101\"); }\n\
     __attribute__((constructor)) static void lo(void){ puts(\"lo\"); }\n";

/// The `main` of the priority fixture, in a third object.
const PRIO_MAIN_SRC: &[u8] =
    b"#include <stdio.h>\nint main(void){ puts(\"main\"); return 0; }\n";

/// What the priority fixture must print. The two prioritised constructors run
/// in priority order rather than link order (`101` before `300`, though `300`
/// was named first); the two unprioritised ones carry no priority, so they run
/// after both and in the order their objects appeared. GNU ld and lld produce
/// exactly this sequence.
const PRIO_EXPECTED: &str = "101\n300\nhi\nlo\nmain\n";

/// Compiles `src` with the host clang using `args` into `obj`. Returns `None`
/// when clang is unavailable so callers can skip gracefully.
fn compile(src: &[u8], obj: &Path, args: &[&str]) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args(args)
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
    let dir = std::env::temp_dir().join(format!("xold_ctor_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// The host toolchain needed for these tests, or `None` (with a note) when a
/// piece is missing.
struct Harness {
    crt1: PathBuf,
    crti: PathBuf,
    crtn: PathBuf,
    libc: PathBuf,
    interp: Vec<u8>,
}

impl Harness {
    /// Collects the harness, or returns `None` when a piece is missing.
    fn detect() -> Option<Self> {
        if which("clang").is_none() {
            eprintln!("skipping init_array tests: clang unavailable");
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

    /// Links `main_obj` plus the crt objects and libc into `prog` with xold.
    fn link(&self, main_obj: &Path, prog: &Path) {
        self.link_all(std::slice::from_ref(&main_obj.to_path_buf()), prog);
    }

    /// The same for several objects, which keep the order they are given in:
    /// that is the order the linker sees them on the command line, and what
    /// the init-priority ordering has to override.
    fn link_all(&self, objs: &[PathBuf], prog: &Path) {
        let mut inputs = objs.to_vec();
        inputs.extend([
            self.crti.clone(),
            self.crt1.clone(),
            self.crtn.clone(),
            self.libc.clone(),
        ]);
        link_dyn_exec(
            &inputs,
            prog,
            b"_start",
            &self.interp,
            false,
            IcfMode::None,
            false,
        )
        .expect("xold libc link must succeed");
    }
}

/// Default `clang -c` (non-PIC) constructor: the executable must be `ET_EXEC`
/// and the constructor must run before `main`, printing `buf=ok` exit 0.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn constructor_runs_non_pic_exec() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("np");
    let main_o = dir.join("np.o");
    compile(CTOR_SRC, &main_o, &["--target=x86_64-linux-gnu", "-c"])
        .expect("host clang compiles non-PIC ctor");
    let prog = dir.join("np_prog");
    h.link(&main_o, &prog);

    let bytes = fs::read(&prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    assert_eq!(
        obj.header().e_type.get(),
        ET_EXEC,
        "non-PIC must be ET_EXEC"
    );

    let out = Command::new(&prog)
        .output()
        .expect("linked program must be runnable");
    assert!(
        out.status.success(),
        "non-PIC ctor should exit 0, got {:?}",
        out.status
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "buf=ok\n",
        "constructor should have set buf before main"
    );
}

/// `clang -c -fPIE` constructor: the executable must be `ET_DYN` (a real PIE)
/// and the constructor must still run. The function-pointer slot is relocated
/// at load time via an `R_X86_64_RELATIVE` entry.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn constructor_runs_pic_dyn() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("pi");
    let main_o = dir.join("pi.o");
    compile(
        CTOR_SRC,
        &main_o,
        &["--target=x86_64-linux-gnu", "-fPIE", "-c"],
    )
    .expect("host clang compiles -fPIE ctor");
    let prog = dir.join("pi_prog");
    h.link(&main_o, &prog);

    let bytes = fs::read(&prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    assert_eq!(
        obj.header().e_type.get(),
        ET_DYN,
        "PIC must be ET_DYN (PIE)"
    );

    let out = Command::new(&prog)
        .output()
        .expect("linked program must be runnable");
    assert!(
        out.status.success(),
        "PIC ctor should exit 0, got {:?}",
        out.status
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "buf=ok\n",
        "constructor should have set buf before main"
    );
}

/// Destructor: `DT_FINI_ARRAY` must be wired so the runtime calls `fini` at
/// exit. Observed via the marker `fini` writes to stderr.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn destructor_runs_at_exit() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("dt");
    let main_o = dir.join("dt.o");
    compile(
        DTOR_SRC,
        &main_o,
        &["--target=x86_64-linux-gnu", "-fPIE", "-c"],
    )
    .expect("host clang compiles destructor");
    let prog = dir.join("dt_prog");
    h.link(&main_o, &prog);

    let out = Command::new(&prog)
        .output()
        .expect("linked program must be runnable");
    assert!(
        out.status.success(),
        "destructor program should exit 0, got {:?}",
        out.status
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("dtor-ran"),
        "destructor should have written its marker to stderr"
    );
}

/// Structural: `DT_INIT_ARRAY`/`DT_INIT_ARRAYSZ` exist for a constructor,
/// point at a real `.init_array` section, and the size matches that section's
/// byte length. Cross-checks tag presence against the system linker.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn init_array_dynamic_tags_match_section() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("st");
    let main_o = dir.join("st.o");
    compile(
        CTOR_SRC,
        &main_o,
        &["--target=x86_64-linux-gnu", "-fPIE", "-c"],
    )
    .expect("host clang compiles ctor");
    let prog = dir.join("st_prog");
    h.link(&main_o, &prog);

    let bytes = fs::read(&prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");

    // The `.init_array` section exists, is allocated+writable, and carries
    // exactly one function pointer (the user constructor).
    let init_sec = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".init_array")
        .expect(".init_array section must exist");
    assert_eq!(init_sec.sh_type.get(), SHT_INIT_ARRAY);
    assert_eq!(
        init_sec.sh_flags.get() & (SHF_ALLOC | SHF_WRITE),
        SHF_ALLOC | SHF_WRITE,
        ".init_array must be allocated and writable"
    );
    assert_eq!(
        init_sec.sh_size.get(),
        8,
        ".init_array holds one 8-byte function pointer"
    );

    // DT_INIT_ARRAY points at the section's address; DT_INIT_ARRAYSZ matches.
    let tags = dynamic_tags(&bytes);
    let init_addr = tags
        .iter()
        .find(|(t, _)| *t == DT_INIT_ARRAY)
        .map(|(_, v)| *v)
        .expect("DT_INIT_ARRAY tag must exist");
    let init_size = tags
        .iter()
        .find(|(t, _)| *t == DT_INIT_ARRAYSZ)
        .map(|(_, v)| *v)
        .expect("DT_INIT_ARRAYSZ tag must exist");
    assert_eq!(
        init_addr,
        init_sec.sh_addr.get(),
        "DT_INIT_ARRAY must name the .init_array address"
    );
    assert_eq!(
        init_size,
        init_sec.sh_size.get(),
        "DT_INIT_ARRAYSZ must match the .init_array byte size"
    );

    // The system linker must also emit DT_INIT_ARRAY for the same source.
    let ref_tag = system_linker_init_array(CTOR_SRC, &dir);
    if let Some(ref_addr) = ref_tag {
        assert_ne!(ref_addr, 0, "system linker DT_INIT_ARRAY must be non-zero");
    }
}

/// A `.preinit_array` entry must run, and must run before every `.init_array`
/// constructor. Observed through the order the three functions stamp into a
/// buffer, not inferred from the section headers.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn preinit_runs_before_init() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("pre");
    let main_o = dir.join("pre.o");
    compile(
        PREINIT_SRC,
        &main_o,
        &["--target=x86_64-linux-gnu", "-fPIE", "-c"],
    )
    .expect("host clang compiles preinit source");
    let prog = dir.join("pre_prog");
    h.link(&main_o, &prog);

    let out = Command::new(&prog)
        .output()
        .expect("linked program must be runnable");
    assert!(
        out.status.success(),
        "preinit program should exit 0, got {:?}",
        out.status
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "order=pim\n",
        "preinit must run before the constructor and before main"
    );
}

/// Structural: `.preinit_array` is a section of its own (not folded into
/// `.data`), `DT_PREINIT_ARRAY`/`DT_PREINIT_ARRAYSZ` describe it exactly, and
/// it is placed ahead of `.init_array`, the order the runtime walks them in.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn preinit_array_dynamic_tags_match_section() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("prest");
    let main_o = dir.join("prest.o");
    compile(
        PREINIT_SRC,
        &main_o,
        &["--target=x86_64-linux-gnu", "-fPIE", "-c"],
    )
    .expect("host clang compiles preinit source");
    let prog = dir.join("prest_prog");
    h.link(&main_o, &prog);

    let bytes = fs::read(&prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let section = |want: &[u8]| {
        obj.sections()
            .iter()
            .find(|s| obj.section_name(s) == want)
            .expect("section must exist")
    };
    let pre = section(b".preinit_array");
    assert_eq!(pre.sh_type.get(), SHT_PREINIT_ARRAY);
    assert_eq!(
        pre.sh_flags.get() & (SHF_ALLOC | SHF_WRITE),
        SHF_ALLOC | SHF_WRITE,
        ".preinit_array must be allocated and writable"
    );
    assert_eq!(
        pre.sh_size.get(),
        8,
        ".preinit_array holds one 8-byte function pointer"
    );
    assert!(
        pre.sh_addr.get() < section(b".init_array").sh_addr.get(),
        ".preinit_array must precede .init_array"
    );

    let tags = dynamic_tags(&bytes);
    let tag = |want: i64| {
        tags.iter()
            .find(|(t, _)| *t == want)
            .map(|(_, v)| *v)
            .expect("dynamic tag must exist")
    };
    assert_eq!(
        tag(DT_PREINIT_ARRAY),
        pre.sh_addr.get(),
        "DT_PREINIT_ARRAY must name the .preinit_array address"
    );
    assert_eq!(
        tag(DT_PREINIT_ARRAYSZ),
        pre.sh_size.get(),
        "DT_PREINIT_ARRAYSZ must match the .preinit_array byte size"
    );
}

/// `.init_array.N` members run in priority order, not link order, and members
/// of equal priority keep link order.
///
/// A compiler puts a prioritised constructor in `.init_array.NNNNN` and names
/// the priority in the section, so the runtime order is a property of the
/// section names rather than of the command line. The object holding priority
/// 300 is named first here, so a linker that lays the members out in input
/// order runs the constructors backwards. The two unprioritised constructors
/// pin the other half of the rule: they carry no priority, so nothing may
/// reorder them relative to each other.
///
/// Observed by running the program, since the ordering is what the runtime
/// does with the array and not how the section headers look.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn init_priority_orders_constructors_across_files() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("prio");
    let args = &["--target=x86_64-linux-gnu", "-c"];
    let hi = dir.join("prio_hi.o");
    let lo = dir.join("prio_lo.o");
    let main_o = dir.join("prio_main.o");
    compile(PRIO_HI_SRC, &hi, args).expect("host clang compiles hi");
    compile(PRIO_LO_SRC, &lo, args).expect("host clang compiles lo");
    compile(PRIO_MAIN_SRC, &main_o, args).expect("host clang compiles main");

    let prog = dir.join("prio_prog");
    h.link_all(&[hi.clone(), lo.clone(), main_o.clone()], &prog);
    let out = Command::new(&prog)
        .output()
        .expect("linked program must be runnable");
    assert!(
        out.status.success(),
        "priority program should exit 0, got {:?}",
        out.status
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        PRIO_EXPECTED,
        "constructors must run in init-priority order"
    );

    // Naming the two objects the other way round must not change a thing: the
    // priorities decide, and the unprioritised pair follows the new input
    // order, which is what a stable sort on the priority alone gives.
    let swapped = dir.join("prio_prog_swapped");
    h.link_all(&[lo, hi, main_o], &swapped);
    let out = Command::new(&swapped)
        .output()
        .expect("linked program must be runnable");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "101\n300\nlo\nhi\nmain\n",
        "priority order must hold whichever object comes first"
    );
}

/// Decodes the `.dynamic` section into `(d_tag, d_un)` rows.
fn dynamic_tags(bytes: &[u8]) -> Vec<(i64, u64)> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(dyn_sec) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynamic")
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(dyn_sec) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for chunk in data.chunks(16) {
        if chunk.len() < 16 {
            break;
        }
        let tag = i64::from_le_bytes(chunk[..8].try_into().unwrap_or([0; 8]));
        if tag == DT_NULL {
            break;
        }
        let val = u64::from_le_bytes(chunk[8..16].try_into().unwrap_or([0; 8]));
        out.push((tag, val));
    }
    out
}

/// Links `src` with the system clang driver and returns its `DT_INIT_ARRAY`
/// address, or `None` when the system toolchain is unavailable. The system
/// driver adds `crtbegin.o`/`crtend.o` frames, so its array may carry extra
/// entries; only the tag's presence is compared.
fn system_linker_init_array(src: &[u8], dir: &Path) -> Option<u64> {
    let clang = which("clang")?;
    let src_path = dir.join("ref.c");
    fs::write(&src_path, src).expect("write source");
    let ref_prog = dir.join("ref_prog");
    let ok = Command::new(clang)
        .arg(&src_path)
        .arg("-o")
        .arg(&ref_prog)
        .status()
        .is_ok();
    let _ = fs::remove_file(&src_path);
    if !ok {
        return None;
    }
    let bytes = fs::read(&ref_prog).expect("read reference");
    dynamic_tags(&bytes)
        .into_iter()
        .find(|(t, _)| *t == DT_INIT_ARRAY)
        .map(|(_, v)| v)
}
