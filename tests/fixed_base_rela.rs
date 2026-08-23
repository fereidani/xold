//! A fixed-base dynamic executable still needs the loader to resolve names.
//!
//! `.rela.dyn` carries two kinds of entry for an absolute pointer slot, and
//! only one of them is about position independence. An `R_X86_64_RELATIVE`
//! asks the loader to add the load base to an address this link already
//! resolved, which a fixed-base (`ET_EXEC`) image has no use for: it has no
//! load base to add. A symbol-based `R_X86_64_64` asks the loader to *resolve
//! a name*, which no link-time base makes unnecessary -- the address lives in
//! a shared object that is not loaded yet.
//!
//! xold suppressed both halves for a fixed base. That was survivable only
//! while every absolute reference to a shared symbol became a copy relocation
//! or a canonical PLT entry; once those narrowed to references the loader
//! cannot fix up in place, a `char ***p = &environ;` in `.data` was left for
//! `.rela.dyn` to describe, and `.rela.dyn` described nothing. The slot kept
//! the zero the writer left, and the program dereferenced null.
//!
//! `ld.lld` on the same input emits `ET_EXEC` carrying `R_X86_64_64
//! environ@GLIBC_2.2.5` and `R_X86_64_64 atoi@GLIBC_2.2.5`, and runs.
//!
//! Gated on `clang`, the crt objects, `libc.so.6` and a probeable
//! interpreter; if any is missing the test prints a note and returns, so the
//! build never fails over a missing toolchain.

#![allow(clippy::similar_names, reason = "crt1/crti/crtn are the host names")]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{
    elf::{
        ObjectFile,
        constants::{ET_EXEC, SHN_UNDEF},
    },
    icf::IcfMode,
    linker::link_dyn_exec,
    reloc::x86_64::{R_X86_64_64, R_X86_64_COPY, R_X86_64_RELATIVE},
};

mod common;

/// Two pointers to libc, both in `.data`: one to a data export and one to a
/// function export. Compiled `-fno-pie -fno-pic`, so the 32-bit absolute
/// references its own code makes force the image to a fixed base, while the
/// two pointers stay `R_X86_64_64` in a writable section -- exactly the shape
/// only the loader can fill in.
///
/// The program checks both arrived: `environ` is non-null in any hosted
/// process, and `atoi` has to be the real one.
const MAIN_SRC: &[u8] = b"#include <stdlib.h>\n\
    extern char **environ;\n\
    char ***penv = &environ;\n\
    int (*sfn)(const char *) = atoi;\n\
    int main(void)\n\
    {\n\
        if (*penv == 0) { return 3; }\n\
        if (sfn(\"41\") != 41) { return 4; }\n\
        return 0;\n\
    }\n";

/// The headline case: the image is `ET_EXEC`, both pointers are described by
/// a symbol-based entry naming their import, and the program runs.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_fixed_base_executable_emits_symbol_based_data_relocations() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("symbased");
    let main_o = dir.join("main.o");
    if compile(MAIN_SRC, &main_o).is_none() {
        eprintln!("skipping fixed_base_rela: clang cannot build the fixture");
        return;
    }
    let prog = dir.join("prog");
    h.link(&main_o, &prog);
    let bytes = fs::read(&prog).expect("read output");

    assert_eq!(
        e_type(&bytes),
        Some(ET_EXEC),
        "the -fno-pic input fixes the base, which is the case under test"
    );
    let rela = rela_rows(&bytes, b".rela.dyn");
    for name in [b"environ".as_slice(), b"atoi".as_slice()] {
        let row = dynsym_row(&bytes, name)
            .unwrap_or_else(|| panic!("{} needs a .dynsym row", show(name)));
        assert_eq!(
            row.shndx,
            SHN_UNDEF,
            "{} is an import, so its row stays undefined",
            show(name)
        );
        assert!(
            rela.iter().any(|r| r.1 == row.index && r.2 == R_X86_64_64),
            "the slot holding {} needs an R_X86_64_64 naming it, got \
             {rela:#x?}",
            show(name)
        );
    }
    assert!(
        !rela.iter().any(|r| r.2 == R_X86_64_COPY),
        "a writable slot needs no copy of the storage, got {rela:#x?}"
    );

    // The whole point: the loader can act on what the image says.
    assert_eq!(
        run(&prog),
        Some(0),
        "both pointers must arrive: environ non-null and atoi callable"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The other half of the same rule: a fixed-base image emits no
/// `R_X86_64_RELATIVE` at all.
///
/// That entry asks the loader to add the load base to an address this link
/// already resolved. An `ET_EXEC` image is mapped at the addresses it names,
/// so the base is zero and the sum is the value the writer already stored:
/// the entry describes a fixup that changes nothing, and still costs a
/// `.rela.dyn` row and a `DT_RELACOUNT` the loader walks before `main`. The
/// absolute-data path has always dropped them; the GOT path did not, so a
/// local GOT slot left one behind. `ld.lld` emits none either -- `addGotEntry`
/// stores a constant once `!ctx.arg.isPic`.
///
/// The fixture is the same one above, which carries a GOT slot of its own
/// (`atoi` is reached through the PLT, and `-fno-pic` code still routes
/// `environ` through the GOT for the `.data` pointer's initialiser).
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_fixed_base_executable_emits_no_relative_entries() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("norelative");
    let main_o = dir.join("main.o");
    if compile(MAIN_SRC, &main_o).is_none() {
        eprintln!("skipping fixed_base_rela: clang cannot build the fixture");
        return;
    }
    let prog = dir.join("prog");
    h.link(&main_o, &prog);
    let bytes = fs::read(&prog).expect("read output");

    assert_eq!(e_type(&bytes), Some(ET_EXEC), "the case under test");
    let rela = rela_rows(&bytes, b".rela.dyn");
    assert!(
        !rela.iter().any(|r| r.2 == R_X86_64_RELATIVE),
        "a fixed-base image has no load base to add, so no RELATIVE entry \
         describes anything, got {rela:#x?}"
    );
    assert!(
        !dynamic_tags(&bytes)
            .iter()
            .any(|&(tag, _)| tag == DT_RELACOUNT),
        "and with no relative entries there is no prefix for DT_RELACOUNT to \
         cover"
    );

    assert_eq!(run(&prog), Some(0), "the image must still run");
    let _ = fs::remove_dir_all(&dir);
}

/// `DT_RELACOUNT`: the number of leading `R_*_RELATIVE` entries in
/// `.rela.dyn`.
const DT_RELACOUNT: i64 = 0x6fff_fff9;

/// Decodes the `.dynamic` section into `(d_tag, d_un)` rows.
fn dynamic_tags(bytes: &[u8]) -> Vec<(i64, u64)> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(sec) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynamic")
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(sec) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for chunk in data.chunks(16) {
        let (Ok(tag), Ok(val)) = (chunk[..8].try_into(), chunk[8..].try_into())
        else {
            break;
        };
        out.push((i64::from_le_bytes(tag), u64::from_le_bytes(val)));
    }
    out
}

// --- fixtures --------------------------------------------------------------

/// A fresh per-test working directory under the system temp dir.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xold_fbrela_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// The host pieces a runnable dynamic executable needs.
struct Harness {
    crt1: PathBuf,
    crti: PathBuf,
    crtn: PathBuf,
    libc: PathBuf,
    interp: Vec<u8>,
}

impl Harness {
    /// Collects them, or prints a note and returns `None`.
    fn detect() -> Option<Self> {
        if which("clang").is_none() {
            eprintln!("skipping fixed_base_rela: clang unavailable");
            return None;
        }
        // `crt1.o` rather than `Scrt1.o`: the non-PIE startup object is the
        // one a fixed-base link pairs with.
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

    /// Links `main_obj` against the crt objects and libc into `prog`.
    fn link(&self, main_obj: &Path, prog: &Path) {
        let inputs = [
            self.crt1.clone(),
            self.crti.clone(),
            main_obj.to_path_buf(),
            self.libc.clone(),
            self.crtn.clone(),
        ];
        link_dyn_exec(
            &inputs,
            prog,
            b"_start",
            &self.interp,
            false,
            IcfMode::None,
            false,
        )
        .expect("xold dynamic-exec link must succeed");
    }
}

/// Compiles `src` with the host clang, without position independence.
fn compile(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fno-pie", "-fno-pic", "-c"])
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

/// A symbol name as text, for assertion messages.
fn show(name: &[u8]) -> String {
    String::from_utf8_lossy(name).into_owned()
}

/// The `e_type` field of the linked image.
fn e_type(bytes: &[u8]) -> Option<u16> {
    Some(ObjectFile::parse(bytes).ok()?.header().e_type.get())
}

/// The fields of one `.dynsym` row a relocation can reference.
struct DynSymRow {
    /// The row's index, which is what a relocation names.
    index: u32,
    /// `st_shndx`.
    shndx: u16,
}

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

/// The `.dynsym` row named `name`, ignoring any `@version` suffix the table
/// does not carry (xold keeps the version in `.gnu.version`, not the name).
fn dynsym_row(bytes: &[u8], name: &[u8]) -> Option<DynSymRow> {
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
            return Some(DynSymRow {
                index: u32::try_from(i).unwrap_or(0),
                shndx: u16::from_le_bytes(
                    chunk[6..8].try_into().unwrap_or([0; 2]),
                ),
            });
        }
    }
    None
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
