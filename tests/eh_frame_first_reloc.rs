//! An `.eh_frame` record's first relocation is the lowest-offset one.
//!
//! Each FDE carries two relocations: the function it describes, then the LSDA
//! that function's handlers live in. Both `.eh_frame` passes want the first --
//! liveness asks whether the function survived, and `sections_with_lsda` marks
//! the function as carrying a handler so identical code folding leaves it
//! alone.
//!
//! "First" was read as first in the relocation table. That agrees with lowest
//! offset for a table an assembler wrote, and parts company after `ld -r`,
//! which may reorder `.rela.eh_frame`. Both passes then asked their question
//! about the LSDA's section instead of the function's: liveness turned on
//! whether the `.gcc_except_table` survived, and the exclusion marked that
//! table rather than the code -- so a function with a `catch` stayed eligible
//! for `--icf=safe` and folded onto an identical one with a different handler.
//! That is the exact miscompile the exclusion exists to prevent, and it is
//! silent.
//!
//! lld sorts the relocations by offset before reading `firstRelocation`, for
//! the same reason. Selecting the minimum is one comparison per relocation, so
//! nothing needs sorting here.
//!
//! Gated on `clang++` and a system `libstdc++`; without them the tests print a
//! note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_dyn_exec};

mod common;

/// Two byte-identical functions, each with its own `catch` and so its own
/// LSDA. Folding them would give one handler to both.
const SRC: &[u8] = b"struct Guard { ~Guard(); };\n\
    Guard::~Guard() {}\n\
    int h1(int x)\n\
    { Guard g; try { if (x < 0) { throw 1; } } catch (int e) { return e; }\n\
      return x + 5; }\n\
    int h2(int x)\n\
    { Guard g; try { if (x < 0) { throw 1; } } catch (int e) { return e; }\n\
      return x + 5; }\n\
    int main(void) { return h1(1) + h2(2) - 13; }\n";

/// With the relocation table reordered, the functions still keep their
/// handlers and so stay unfolded under `--icf=safe`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_reordered_reloc_table_still_finds_the_function() {
    let Some(dir) = workdir("reordered") else {
        return;
    };
    let Some(bytes) = link(&dir, true) else {
        return;
    };
    let h1 = symbol_value(&bytes, b"_Z2h1i").expect("h1 present");
    let h2 = symbol_value(&bytes, b"_Z2h2i").expect("h2 present");
    assert_ne!(
        h1, h2,
        "each function has its own LSDA, so --icf=safe must leave them apart; \
         reading the table's first entry instead of the lowest-offset one \
         marks the .gcc_except_table and lets the code fold, giving both \
         functions one handler"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: the assembler's own ordering gives the same answer, so the
/// reordering is what the fix is about and not the fixture.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_original_order_gives_the_same_answer() {
    let Some(dir) = workdir("original") else {
        return;
    };
    let Some(bytes) = link(&dir, false) else {
        return;
    };
    let h1 = symbol_value(&bytes, b"_Z2h1i").expect("h1 present");
    let h2 = symbol_value(&bytes, b"_Z2h2i").expect("h2 present");
    assert_ne!(h1, h2, "handler-bearing functions never fold under safe");
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang++").is_none() {
        eprintln!("skipping eh-frame-first-reloc {prefix}: no clang++");
        return None;
    }
    if cxx_runtime().is_none() {
        eprintln!("skipping eh-frame-first-reloc {prefix}: no libstdc++");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_ehfirst_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds the fixture, optionally reversing `.rela.eh_frame`, and links it
/// with `--icf=safe`.
fn link(dir: &Path, reorder: bool) -> Option<Vec<u8>> {
    let clangxx = which("clang++")?;
    let src = dir.join("l.cpp");
    let obj = dir.join("l.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clangxx)
        .args([
            "--target=x86_64-linux-gnu",
            "-O1",
            "-ffunction-sections",
            "-c",
        ])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping eh-frame-first-reloc: clang++ cannot build it");
        return None;
    }
    if reorder {
        reverse_eh_relocs(&obj)?;
    }
    let interp = common::interpreter()?;
    let paths = vec![
        common::crt_file("Scrt1.o")?,
        common::crt_file("crti.o")?,
        obj,
        cxx_runtime()?,
        // `_Unwind_Resume` lives in `libgcc_s`, which the C++ runtime names
        // undefined; without it the link is underlinked.
        common::libgcc_s_so()?,
        common::libc_so()?,
        common::crt_file("crtn.o")?,
    ];
    let out = dir.join("prog");
    let res = link_dyn_exec(
        &paths,
        &out,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::Safe,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    fs::read(&out).ok()
}

/// Reverses the entries of `.rela.eh_frame` in place.
///
/// This is the `ld -r` shape done by hand: the relocations are the same
/// relocations, describing the same records, in another order. Within each
/// FDE the LSDA's entry now precedes the function's.
fn reverse_eh_relocs(obj: &Path) -> Option<()> {
    let mut bytes = fs::read(obj).ok()?;
    let (off, size) = {
        let parsed = ObjectFile::parse(&bytes).ok()?;
        let shdr = parsed
            .sections()
            .iter()
            .find(|s| parsed.section_name(s) == b".rela.eh_frame")?;
        (
            usize::try_from(shdr.sh_offset.get()).ok()?,
            usize::try_from(shdr.sh_size.get()).ok()?,
        )
    };
    let rows: Vec<Vec<u8>> = bytes
        .get(off..off + size)?
        .as_chunks::<24>()
        .0
        .iter()
        .map(|c| c.to_vec())
        .rev()
        .collect();
    assert!(
        rows.len() > 2,
        "the fixture must have relocations to reorder"
    );
    let flat: Vec<u8> = rows.concat();
    bytes.get_mut(off..off + size)?.copy_from_slice(&flat);
    fs::write(obj, bytes).ok()
}

/// The C++ runtime as a real shared object.
///
/// The `libstdc++.so` beside the compiler is a GNU linker script on this
/// distribution. The driver expands those; this test drives the library, which
/// takes a settled input list, so the versioned object is named directly.
fn cxx_runtime() -> Option<PathBuf> {
    ["/usr/lib64/libstdc++.so.6", "/usr/lib/libstdc++.so.6"]
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
}

// --- readers ---------------------------------------------------------------

/// The `st_value` of a symbol in the output's `.symtab`.
fn symbol_value(bytes: &[u8], name: &[u8]) -> Option<u64> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let symtab = obj.symbol_table().ok().flatten()?;
    symtab
        .iter()
        .find(|s| symtab.name(s) == name)
        .map(|s| s.st_value.get())
}
