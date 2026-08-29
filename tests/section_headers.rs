//! What the output section headers and the ELF header say about the image.
//!
//! These fields are not read by the loader, which drives off the program
//! headers, so a wrong value costs nothing at run time and everything to
//! anything that reads the image back: a debugger, `objcopy`, a second link.
//! Three of them were describing something other than what the linker did.
//!
//! - `sh_addralign` came off a constant table rather than the alignment
//!   placement honoured. It was stricter than the truth for `.text`, which the
//!   ELF spec forbids -- `sh_addr` must be congruent to zero modulo
//!   `sh_addralign` -- and looser for `.rodata` and `.data`, which understated
//!   what their members demand.
//! - `sh_entsize` was dropped from the aggregated debug sections while their
//!   `SHF_MERGE|SHF_STRINGS` flags were kept, leaving `.debug_str` describing
//!   entries of no length.
//! - `e_flags` was zero whatever the inputs carried, so a RISC-V image built
//!   from hard-float objects declared the soft-float ABI.
//!
//! Gated on `clang`; absent it the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::{
        ObjectFile, Shdr64,
        constants::{SHF_MERGE, SHF_STRINGS},
    },
    error::Error,
    icf::IcfMode,
    linker::link_to,
};

mod common;

fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_shdr_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Compiles `src` for the explicit target in `extra`, or `x86_64` ELF by
/// default, returning the object.
fn compile(
    dir: &Path,
    name: &str,
    src: &str,
    extra: &[&str],
) -> Option<PathBuf> {
    let clang = which("clang")?;
    let file = dir.join(format!("{name}.c"));
    let obj = dir.join(format!("{name}.o"));
    fs::write(&file, src).ok()?;
    let mut command = Command::new(clang);
    if !extra.iter().any(|arg| arg.starts_with("--target=")) {
        command.arg("--target=x86_64-linux-gnu");
    }
    command
        .args(["-c", "-O1"])
        .args(extra)
        .arg("-o")
        .arg(&obj)
        .arg(&file)
        .status()
        .ok()?
        .success()
        .then_some(obj)
}

/// Runs `f` over every section header of `image`.
fn for_each_section(image: &Path, mut f: impl FnMut(&[u8], &Shdr64)) {
    let bytes = fs::read(image).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("parse output");
    for shdr in obj.sections() {
        f(obj.section_name(shdr), shdr);
    }
}

/// The header of one output section, by name.
fn section(image: &Path, want: &[u8]) -> Option<Shdr64> {
    let mut found = None;
    for_each_section(image, |name, shdr| {
        if name == want {
            found = Some(*shdr);
        }
    });
    found
}

/// The ELF rule every header has to satisfy, over a program whose `.text`
/// alignment is 1 and whose data is over-aligned -- the two directions the
/// table constants got wrong.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn every_section_address_is_a_multiple_of_its_alignment() {
    let dir = workdir("align");
    let Some(obj) = compile(
        &dir,
        "aligned",
        r"
__attribute__((aligned(64))) const int ro[16] = { 1 };
__attribute__((aligned(64))) int rw[16] = { 1 };
__attribute__((aligned(1))) void _start(void) { }
",
        &["-fno-pie"],
    ) else {
        eprintln!("skipping section-header test: clang missing");
        return;
    };
    let out = dir.join("aligned");
    link_to(&[obj], &out, b"_start", false, IcfMode::None, false)
        .expect("the image must link");

    for_each_section(&out, |name, shdr| {
        let align = shdr.sh_addralign.get();
        if align == 0 {
            return;
        }
        assert_eq!(
            shdr.sh_addr.get() % align,
            0,
            "{}: address {:#x} is not a multiple of its alignment {align}",
            String::from_utf8_lossy(name),
            shdr.sh_addr.get()
        );
    });

    // And the alignment reported is the one the members asked for, not the
    // floor the table used to state.
    for name in [&b".rodata"[..], b".data"] {
        let shdr = section(&out, name).unwrap_or_else(|| {
            panic!("{} exists", String::from_utf8_lossy(name))
        });
        assert_eq!(
            shdr.sh_addralign.get(),
            64,
            "{} reports the strictest alignment its members declare",
            String::from_utf8_lossy(name)
        );
    }
}

/// A mergeable section states how large its entries are. Keeping the flags
/// while dropping the size describes entries of no length.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_mergeable_debug_section_states_its_entry_size() {
    let dir = workdir("entsize");
    let Some(obj) = compile(
        &dir,
        "dbg",
        "int _start_helper(void) { return 1; }\nvoid _start(void) { }\n",
        &["-g", "-fno-pie"],
    ) else {
        eprintln!("skipping section-header test: clang missing");
        return;
    };
    let out = dir.join("dbg");
    link_to(&[obj], &out, b"_start", false, IcfMode::None, false)
        .expect("the image must link");

    let Some(shdr) = section(&out, b".debug_str") else {
        eprintln!("skipping entsize test: this clang emits no .debug_str");
        return;
    };
    assert_ne!(
        shdr.sh_flags.get() & (SHF_MERGE | SHF_STRINGS),
        0,
        ".debug_str is mergeable strings"
    );
    assert_eq!(
        shdr.sh_entsize.get(),
        1,
        "a mergeable string section's entries are one byte wide"
    );
}

/// Whether `clang` can build for `triple`, which needs the target's builtins
/// to be installed rather than just recognised.
fn riscv_object(dir: &Path) -> Option<PathBuf> {
    compile(
        dir,
        "rv",
        "void _start(void) { }\n",
        &["--target=riscv64-unknown-elf", "-march=rv64gc"],
    )
}

/// The header's `e_flags` says which variant of the architecture the image is
/// built for. On RISC-V the loader and the kernel read the floating-point ABI
/// out of it, so an image that reports soft-float is refused against hard-float
/// libraries whatever its code does.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn riscv_carries_the_inputs_architecture_flags() {
    let dir = workdir("eflags");
    let Some(obj) = riscv_object(&dir) else {
        eprintln!("skipping e_flags test: clang cannot target riscv64");
        return;
    };
    let input = fs::read(&obj).expect("read input");
    let want = ObjectFile::parse(&input)
        .expect("parse input")
        .header()
        .e_flags
        .get();
    assert_ne!(
        want, 0,
        "the fixture must carry flags for this to be a test"
    );

    let out = dir.join("rv");
    link_to(&[obj], &out, b"_start", false, IcfMode::None, false)
        .expect("the image must link");
    let bytes = fs::read(&out).expect("read output");
    let got = ObjectFile::parse(&bytes)
        .expect("parse output")
        .header()
        .e_flags
        .get();
    assert_eq!(
        got, want,
        "the image carries the inputs' architecture flags"
    );
}

/// Two inputs that pass arguments differently cannot be one image, and the
/// header has no way to say they are both right. lld refuses the same pair.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_float_abi_mismatch_is_refused() {
    let dir = workdir("abi");
    let Some(hard) = riscv_object(&dir) else {
        eprintln!("skipping e_flags test: clang cannot target riscv64");
        return;
    };
    let Some(soft) = compile(
        &dir,
        "soft",
        "void other(void) { }\n",
        &[
            "--target=riscv64-unknown-elf",
            "-march=rv64imac",
            "-mabi=lp64",
        ],
    ) else {
        return;
    };
    let out = dir.join("mixed");
    let err =
        link_to(&[hard, soft], &out, b"_start", false, IcfMode::None, false)
            .expect_err("objects with different float ABIs must not link");
    assert!(
        matches!(
            &err,
            Error::IncompatibleInput(e) if e.what == "floating-point ABI"
        ),
        "the diagnostic names the ABI that disagreed, got {err}"
    );
}
