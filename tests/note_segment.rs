//! Allocated notes keep their type and get a `PT_NOTE`.
//!
//! A note is found by segment, never by section name: a reader walks
//! `PT_NOTE`'s extent and decodes the self-describing records inside it. A
//! `PT_NOTE` can only cover a section whose header says `SHT_NOTE`.
//!
//! `OutKind::from_shdr` had no note arm, so an allocated note fell through to
//! `.rodata`. Garbage collection roots notes, so the bytes survived -- under a
//! `SHT_PROGBITS` header, inside `.rodata`, with no segment pointing at them.
//! Every glibc link carries `.note.ABI-tag` and `.note.gnu.property`: after
//! this, `readelf -n` printed nothing, a build-id lookup by a core-dump
//! matcher or debuginfod found no id, and the CET and BTI properties the
//! loader acts on were dead weight in a read-only section. The bytes were
//! there and unreachable, which is the worst of the three possible states.
//!
//! lld emits one `PT_NOTE` per contiguous run of notes and keeps their
//! `sh_type`. There is one run here, because the notes are gathered into a
//! single `.note` region -- the records carry their own names, so nothing is
//! lost by not keeping the input section names apart.
//!
//! Gated on `clang` and the system crt objects; without them the tests print a
//! note and return.

use std::{fs, path::PathBuf, process::Command};

use common::{crt_file, interpreter, libc_so, which};
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_dyn_exec};

mod common;

const SRC: &[u8] = b"#include <stdio.h>\n\
    int main(void) { printf(\"note\\n\"); return 0; }\n";

/// `PT_NOTE` covers a non-empty run, and the section it covers is `SHT_NOTE`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_image_carries_a_pt_note_over_a_sht_note_section() {
    let Some(dir) = workdir("segment") else {
        return;
    };
    let Some(prog) = link(&dir) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");

    let note = note_segment(&bytes).expect(
        "a link against glibc's crt objects carries .note.ABI-tag and \
         .note.gnu.property, so the image must have a PT_NOTE; without one \
         the records are unreachable however well the bytes survived",
    );
    assert!(note.filesz > 0, "the segment must cover the records");
    assert_eq!(note.align, 4, "note records are 4-byte aligned");

    let (addr, size, sh_type) =
        note_section(&bytes).expect("a .note output section must exist");
    assert_eq!(sh_type, SHT_NOTE, "the section header must say SHT_NOTE");
    assert_eq!(addr, note.vaddr, "the segment must start at the section");
    assert_eq!(size, note.filesz, "and cover exactly it");
    let _ = fs::remove_dir_all(&dir);
}

/// The records inside decode, which is the whole point of keeping the type.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_notes_inside_are_readable_records() {
    let Some(dir) = workdir("records") else {
        return;
    };
    let Some(prog) = link(&dir) else {
        return;
    };
    let bytes = fs::read(&prog).expect("read output");
    let note = note_segment(&bytes).expect("PT_NOTE present");
    let start = usize::try_from(note.offset).expect("offset fits");
    let len = usize::try_from(note.filesz).expect("size fits");
    let data = bytes.get(start..start + len).expect("segment in the image");
    assert!(
        count_records(data) >= 2,
        "glibc's crt objects contribute at least .note.ABI-tag and \
         .note.gnu.property; walking the segment must find them"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// And the program still runs: the notes moved to the head of the image, in
/// front of the code.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_program_still_runs() {
    let Some(dir) = workdir("run") else {
        return;
    };
    let Some(prog) = link(&dir) else {
        return;
    };
    let out = Command::new(&prog).output().expect("must run");
    assert_eq!(out.status.code(), Some(0), "the program must exit 0");
    assert_eq!(out.stdout, b"note\n", "and print what it prints");
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping note-segment {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_note_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Builds and links a dynamic executable against the host crt objects, which
/// are where the notes come from.
fn link(dir: &std::path::Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let interp = interpreter()?;
    let src = dir.join("note.c");
    let obj = dir.join("note.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIE", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping note-segment: clang cannot build the fixture");
        return None;
    }
    let start = crt_file("Scrt1.o")?;
    let prologue = crt_file("crti.o")?;
    let epilogue = crt_file("crtn.o")?;
    let libc = libc_so()?;
    let prog = dir.join("noteprog");
    let res = link_dyn_exec(
        &[start, prologue, obj, libc, epilogue],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    );
    assert!(res.is_ok(), "the fixture must link: {:?}", res.err());
    Some(prog)
}

// --- readers ---------------------------------------------------------------

/// `SHT_NOTE`, spelled here so the test does not depend on the constant it is
/// checking.
const SHT_NOTE: u32 = 7;
/// `PT_NOTE`, likewise.
const PT_NOTE: u32 = 4;

/// The fields of `PT_NOTE` these tests reason about.
struct Segment {
    offset: u64,
    vaddr: u64,
    filesz: u64,
    align: u64,
}

/// Reads `PT_NOTE` out of the program header table.
fn note_segment(bytes: &[u8]) -> Option<Segment> {
    let phoff = usize::try_from(read_u64(bytes, 32)?).ok()?;
    let entsize = usize::from(read_u16(bytes, 54)?);
    let count = usize::from(read_u16(bytes, 56)?);
    (0..count).find_map(|i| {
        let at = phoff + i * entsize;
        if read_u32(bytes, at)? != PT_NOTE {
            return None;
        }
        Some(Segment {
            offset: read_u64(bytes, at + 8)?,
            vaddr: read_u64(bytes, at + 16)?,
            filesz: read_u64(bytes, at + 32)?,
            align: read_u64(bytes, at + 48)?,
        })
    })
}

/// The `(sh_addr, sh_size, sh_type)` of the output `.note` section.
fn note_section(bytes: &[u8]) -> Option<(u64, u64, u32)> {
    let obj = ObjectFile::parse(bytes).ok()?;
    obj.sections()
        .iter()
        .find(|s| obj.section_name(s) == b".note")
        .map(|s| (s.sh_addr.get(), s.sh_size.get(), s.sh_type.get()))
}

/// Walks a note run and counts the records in it. Each record is a 12-byte
/// header (`namesz`, `descsz`, `type`) followed by the 4-aligned name and
/// description.
fn count_records(mut data: &[u8]) -> usize {
    let mut seen = 0;
    while data.len() >= 12 {
        let Some(namesz) = read_u32(data, 0) else {
            break;
        };
        let Some(descsz) = read_u32(data, 4) else {
            break;
        };
        let name = (namesz as usize).next_multiple_of(4);
        let desc = (descsz as usize).next_multiple_of(4);
        let Some(total) =
            12usize.checked_add(name).and_then(|t| t.checked_add(desc))
        else {
            break;
        };
        if total > data.len() || total == 12 && namesz == 0 && descsz == 0 {
            break;
        }
        data = &data[total..];
        seen += 1;
    }
    seen
}

fn read_u16(bytes: &[u8], at: usize) -> Option<u16> {
    bytes
        .get(at..at + 2)
        .and_then(|c| <[u8; 2]>::try_from(c).ok())
        .map(u16::from_le_bytes)
}

fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .and_then(|c| <[u8; 4]>::try_from(c).ok())
        .map(u32::from_le_bytes)
}

fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
    bytes
        .get(at..at + 8)
        .and_then(|c| <[u8; 8]>::try_from(c).ok())
        .map(u64::from_le_bytes)
}
