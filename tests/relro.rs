//! `PT_GNU_RELRO`: the run of writable bytes the loader may re-protect
//! read-only once it has applied the image's relocations.
//!
//! Without it `.dynamic`, `.got` and the constructor arrays stay writable for
//! the life of the process, which is the classic GOT-overwrite target. The run
//! has to satisfy four things at once, and each is a way the feature can be
//! present and still protect nothing:
//!
//! - It must open the read-write segment. glibc mprotects one range, so
//!   anything before the run would be dragged in or the run split in two.
//! - It must end on a page boundary. `_dl_protect_relro` mprotects whole pages,
//!   so a run ending mid-page leaves that page writable.
//! - It must lie inside the read-write `PT_LOAD`. An mprotect over unmapped
//!   memory fails and glibc turns that into a fatal startup error.
//! - It must exclude `.got.plt`. xold binds lazily, so the loader stores a
//!   resolved address into a `.got.plt` slot the first time its stub is called
//!   -- long after the protection went on.
//!
//! The last test covers the other half of the change: the program header table
//! is counted before placement (the count sizes the header block every region
//! is placed after) and written afterwards, from two separate walks. A static
//! link exercises the path where most of the optional segments are absent.
//!
//! Gated on `clang`, `gcc` (to locate the crt objects), `libc.so.6` and the
//! system `ld.so`; if absent the tests print a note and return, so the build
//! never fails over a missing toolchain.

// `crt1.o`, `crti.o` and `crtn.o` are the host's own filenames; spelling them
// any other way would make the harness harder to read, not easier. The sibling
// libc link tests carry the same allowance.
#![expect(clippy::similar_names, reason = "crt object filenames")]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, libc_so, which};
use xold::{
    elf::ObjectFile,
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared, link_to},
};

mod common;

/// The segment types the assertions name. Spelled out rather than imported so
/// a test failure means the image is wrong, not that xold and the test share a
/// mistaken constant.
const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const PT_GNU_STACK: u32 = 0x6474_e551;
const PT_GNU_RELRO: u32 = 0x6474_e552;

/// `p_flags`: the protected run is read-only, and the writable `PT_LOAD` that
/// contains it is read-write.
const PF_W: u32 = 2;
const PF_R: u32 = 4;

/// The page granularity `_dl_protect_relro` rounds the run's end down to.
const PAGE: u64 = 0x1000;

/// Every section the RELRO run is allowed to contain, in placement order.
const PROTECTED: [&[u8]; 6] = [
    b".dynamic",
    b".got",
    b".data.rel.ro",
    b".preinit_array",
    b".init_array",
    b".fini_array",
];

/// A program with one section of every interesting kind: `table` and `names`
/// are `const` objects whose initialisers need a relocation, so the compiler
/// puts them in `.data.rel.ro`; `gv` is plain `.data`; the constructor gives
/// the image an `.init_array`; and the `printf` calls give it a PLT, and so a
/// `.got.plt` the loader writes to after the protection goes on.
const SRC: &[u8] = b"#include <stdio.h>\n\
     static void f0(void) { printf(\"f0\\n\"); }\n\
     static void f1(void) { printf(\"f1\\n\"); }\n\
     void (*const table[])(void) = { f0, f1 };\n\
     static const char *const names[] = { \"n0\", \"n1\" };\n\
     const char *const *pnames = names;\n\
     int gv = 3;\n\
     __attribute__((constructor)) static void ctor(void) { gv += 1; }\n\
     int main(void) {\n\
         table[0](); table[1]();\n\
         printf(\"%s %s %d\\n\", pnames[0], pnames[1], gv);\n\
         return 0;\n\
     }\n";

/// The same shapes without a `main`, for the `-shared` case.
const LIB_SRC: &[u8] = b"#include <stdio.h>\n\
     static void f0(void) { printf(\"f0\\n\"); }\n\
     void (*const lib_table[])(void) = { f0 };\n\
     int lib_gv = 7;\n\
     __attribute__((constructor)) static void lib_ctor(void) { lib_gv += 1; }\n\
     int lib_call(void) { lib_table[0](); return lib_gv; }\n";

/// A freestanding `_start` for the static link: it exits with 0 through the
/// `exit_group` syscall, so the image needs no libc.
const START_SRC: &[u8] = b"    .text\n    .globl _start\n_start:\n    \
    xorl    %edi, %edi\n    movl    $231, %eax\n    syscall\n";

/// A static-link translation unit that still fills the protected run: the
/// constructor gives it an `.init_array` and `counter` gives it a `.data`.
const STATIC_SRC: &[u8] = b"int counter;\n\
     __attribute__((constructor)) static void ctor(void){ counter++; }\n\
     int touch(void){ return counter; }\n";

/// An object whose only writable content is `SHT_NOBITS`, with a freestanding
/// entry so the image can be run. It has no `.data`, no GOT and no `.dynamic`,
/// so its writable segment has no file-backed region at all.
const BSS_ONLY_SRC: &[u8] = b"    .text\n    .globl _start\n_start:\n    \
    xorl    %edi, %edi\n    movl    $231, %eax\n    syscall\n\
    .bss\n    .globl zeroed\n    .type zeroed,@object\n    \
    .size zeroed, 64\nzeroed:\n    .zero 64\n";

// --- tests -----------------------------------------------------------------

/// A dynamic executable: the run exists, opens the writable segment, ends on a
/// page boundary, stays inside the writable `PT_LOAD`, and the program still
/// runs. A run covering something the loader writes after protecting it shows
/// up here as a fault during startup.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_dynamic_executable_protects_the_head_of_its_writable_segment() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("dynexec");
    let prog = dir.join("prog");
    let Some(obj) = compile(SRC, &dir.join("m.o"), &["-fPIE", "-c"]) else {
        return;
    };
    h.link_exec(&obj, &prog);
    let bytes = fs::read(&prog).expect("read output");

    check_relro(&bytes);

    let out = Command::new(&prog)
        .output()
        .expect("program must be runnable");
    assert!(
        out.status.success(),
        "the protected program must still start and run, got {:?}",
        out.status
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "f0\nf1\nn0 n1 4\n",
        "the constructor and the relocated const tables must all be intact"
    );
}

/// A shared object gets the same treatment, and a program that loads it still
/// runs: `ld.so` applies the same protection to every mapped object.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_shared_object_protects_the_head_of_its_writable_segment() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("shared");
    let Some(lib_o) = compile(LIB_SRC, &dir.join("lib.o"), &["-fPIC", "-c"])
    else {
        return;
    };
    let lib = dir.join("librelro.so");
    link_shared(
        &[lib_o, h.libc.clone()],
        &lib,
        Some(b"librelro.so"),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold -shared link must succeed");
    let bytes = fs::read(&lib).expect("read output");

    check_relro(&bytes);

    // Load it from a program and call through it: a protected range covering
    // something the loader writes later faults before `main` is reached.
    let user = b"#include <stdio.h>\nextern int lib_call(void);\n\
         int main(void){ printf(\"%d\\n\", lib_call()); return 0; }\n";
    let Some(user_o) = compile(user, &dir.join("u.o"), &["-fPIE", "-c"]) else {
        return;
    };
    let prog = dir.join("user");
    link_dyn_exec(
        &[
            user_o,
            h.crti.clone(),
            h.crt1.clone(),
            h.crtn.clone(),
            lib,
            h.libc.clone(),
        ],
        &prog,
        b"_start",
        &h.interp,
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-exec link must succeed");
    let out = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .output()
        .expect("program must be runnable");
    assert!(
        out.status.success(),
        "a program loading the protected object must run, got {:?}",
        out.status
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "f0\n8\n");
}

/// `.got` and `.dynamic` are inside the run; `.got.plt` is not.
///
/// This is the membership decision that cannot be relaxed. lld admits
/// `.got.plt` only under `-z now` (`isRelroSection`, ELF/Writer.cpp:627),
/// because with lazy binding the loader stores the resolved address into the
/// slot on the first call through its stub. Protecting it would fault at that
/// call rather than at startup, which is the worst kind of wrong.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn got_plt_stays_outside_the_protected_run() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("gotplt");
    let prog = dir.join("prog");
    let Some(obj) = compile(SRC, &dir.join("m.o"), &["-fPIE", "-c"]) else {
        return;
    };
    h.link_exec(&obj, &prog);
    let bytes = fs::read(&prog).expect("read output");
    let relro = find(&segments(&bytes), PT_GNU_RELRO).expect("PT_GNU_RELRO");

    for name in [b".got".as_slice(), b".dynamic"] {
        let s =
            section(&bytes, name).unwrap_or_else(|| panic!("{}", show(name)));
        assert!(
            covers(&relro, &s),
            "{} ({:#x}..{:#x}) must be inside the protected run \
             ({:#x}..{:#x})",
            show(name),
            s.addr,
            s.addr + s.size,
            relro.vaddr,
            relro.vaddr + relro.memsz
        );
    }

    let got_plt = section(&bytes, b".got.plt").expect(".got.plt");
    assert!(
        got_plt.addr >= relro.vaddr + relro.memsz,
        ".got.plt ({:#x}) must start at or after the end of the protected \
         run ({:#x}); the loader writes to it on the first lazy bind",
        got_plt.addr,
        relro.vaddr + relro.memsz
    );
}

/// A `.data.rel.ro` input section lands in the protected run and keeps out of
/// `.data`.
///
/// Compilers put a relocated `const` object there precisely so it can be made
/// read-only after startup. Folding it into `.data` -- which is what
/// `OutKind::from_shdr` did before there was a `.data.rel.ro` output section --
/// leaves it writable forever.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn data_rel_ro_lands_in_the_protected_run_and_not_in_data() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("relro_data");
    let prog = dir.join("prog");
    let Some(obj) = compile(SRC, &dir.join("m.o"), &["-fPIE", "-c"]) else {
        return;
    };
    // The fixture is only meaningful if the compiler actually emitted the
    // section; otherwise the test would pass on an image that never saw one.
    let input = fs::read(&obj).expect("read object");
    assert!(
        section(&input, b".data.rel.ro").is_some(),
        "the fixture object must carry a .data.rel.ro for the test to mean \
         anything"
    );

    h.link_exec(&obj, &prog);
    let bytes = fs::read(&prog).expect("read output");
    let relro = find(&segments(&bytes), PT_GNU_RELRO).expect("PT_GNU_RELRO");
    let rel_ro = section(&bytes, b".data.rel.ro").expect(".data.rel.ro");
    let data = section(&bytes, b".data").expect(".data");

    assert_ne!(rel_ro.size, 0, ".data.rel.ro must hold the const tables");
    assert!(
        covers(&relro, &rel_ro),
        ".data.rel.ro ({:#x}..{:#x}) must be inside the protected run \
         ({:#x}..{:#x})",
        rel_ro.addr,
        rel_ro.addr + rel_ro.size,
        relro.vaddr,
        relro.vaddr + relro.memsz
    );
    assert!(
        data.addr >= relro.vaddr + relro.memsz,
        ".data ({:#x}) must stay outside the protected run (ends {:#x})",
        data.addr,
        relro.vaddr + relro.memsz
    );
    assert!(
        rel_ro.addr + rel_ro.size <= data.addr
            || data.addr + data.size <= rel_ro.addr,
        ".data.rel.ro and .data must be separate output sections"
    );
}

/// A static link has no `.dynamic`, so most of the optional segments are
/// absent. The program header table it declares must still be exactly the one
/// it wrote.
///
/// The count is decided before placement (it sizes the header block every
/// region is then placed after) and the headers are emitted afterwards, by a
/// second walk. A count one too high leaves a zeroed slot inside the declared
/// table; one too low pushes the last header past the declared end and over
/// the first bytes of content. Both show up as: every declared slot names a
/// segment, and the last one is the `PT_GNU_STACK` marker the writer always
/// appends.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_static_link_emits_the_program_header_table_it_declares() {
    if which("clang").is_none() {
        eprintln!("skipping static relro test: clang unavailable");
        return;
    }
    let dir = workdir("static");
    let Some(main_o) = compile(STATIC_SRC, &dir.join("s.o"), &["-c"]) else {
        return;
    };
    let Some(start_o) = assemble(START_SRC, &dir.join("start.o")) else {
        return;
    };
    let prog = dir.join("static_prog");
    link_to(
        &[main_o, start_o],
        &prog,
        b"_start",
        false,
        IcfMode::None,
        false,
    )
    .expect("xold static link must succeed");
    let bytes = fs::read(&prog).expect("read output");

    let segs = segments(&bytes);
    assert_ne!(segs.len(), 0, "a static image still has program headers");
    check_phdr_table(&segs);
    assert!(
        find(&segs, PT_DYNAMIC).is_none(),
        "a static link has no .dynamic and so no PT_DYNAMIC"
    );

    // The static image obeys the same RELRO rules; it just has fewer regions
    // to protect.
    check_relro(&bytes);

    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        let status =
            Command::new(&prog).status().expect("static prog runnable");
        assert!(status.success(), "static image must run, got {status}");
    } else {
        assert_static_entry(&bytes);
    }
}

/// A writable segment whose only content is `SHT_NOBITS` reports an empty file
/// extent, not an underflowed one.
///
/// `p_filesz` is a byte count the kernel maps literally, and it is computed by
/// subtracting where the segment starts from where its last file-backed region
/// ends. Such a link has no file-backed writable region to end at, so the
/// subtraction has nothing to subtract from; getting it wrong yields a value
/// near `u64::MAX` and an image `execve` refuses.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_bss_only_writable_segment_reports_no_file_bytes() {
    if which("clang").is_none() {
        eprintln!("skipping bss-only segment test: clang unavailable");
        return;
    }
    let dir = workdir("bssonly");
    let Some(obj) = assemble(BSS_ONLY_SRC, &dir.join("bss_only.o")) else {
        return;
    };
    let prog = dir.join("bss_only");
    link_to(&[obj], &prog, b"_start", false, IcfMode::None, false)
        .expect("xold static link must succeed");
    let bytes = fs::read(&prog).expect("read output");

    let segs = segments(&bytes);
    check_phdr_table(&segs);
    check_load_extents(&bytes, &segs);
    let rw = segs
        .iter()
        .find(|s| s.p_type == PT_LOAD && s.flags & PF_W != 0)
        .expect("a .bss still needs a writable PT_LOAD");
    assert_eq!(
        rw.filesz, 0,
        "a writable segment holding only SHT_NOBITS content has no file bytes"
    );
    assert!(
        rw.memsz >= 64,
        "p_memsz {:#x} must cover the 64-byte object in .bss",
        rw.memsz
    );

    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        let status = Command::new(&prog)
            .status()
            .expect("bss-only prog runnable");
        assert!(status.success(), "bss-only image must run, got {status}");
    } else {
        assert_static_entry(&bytes);
    }
}

/// Pins the launch contract of a foreign static ELF image after its segment
/// layout has been checked: the entry is `_start`, which performs
/// `exit_group(0)`.
fn assert_static_entry(bytes: &[u8]) {
    let obj = ObjectFile::parse(bytes).expect("valid ELF");
    let symtab = obj.symbol_table().expect("read symtab").expect("symtab");
    let start = symtab
        .syms
        .iter()
        .find(|sym| symtab.name(sym) == b"_start")
        .expect("_start defined")
        .st_value
        .get();
    assert_eq!(obj.header().e_entry.get(), start, "ELF enters at _start");
    let text = obj
        .sections()
        .iter()
        .find(|sec| obj.section_name(sec) == b".text")
        .and_then(|sec| obj.section_data(sec).ok())
        .expect("text bytes");
    assert!(
        text.windows(9)
            .any(|w| w == [0x31, 0xff, 0xb8, 0xe7, 0, 0, 0, 0x0f, 0x05]),
        "_start performs exit_group(0)"
    );
}

// --- shared assertions -----------------------------------------------------

/// Every `PT_LOAD` describes a range the file actually has.
///
/// `p_filesz` and `p_memsz` are counts rather than offsets, so an arithmetic
/// slip does not shift a segment, it makes one the kernel cannot map. Checked
/// for every image these tests build, because the computation is shared.
fn check_load_extents(bytes: &[u8], segs: &[Segment]) {
    let len = u64::try_from(bytes.len()).expect("image size fits");
    for s in segs.iter().filter(|s| s.p_type == PT_LOAD) {
        assert!(
            s.filesz <= s.memsz,
            "p_filesz {:#x} exceeds p_memsz {:#x}: a segment cannot hold more \
             file bytes than the range it covers",
            s.filesz,
            s.memsz
        );
        // Only a segment that claims bytes has to name a range the file has.
        // A `p_filesz` of zero maps nothing -- the kernel's `elf_map` returns
        // before mmap when the page-aligned size is zero -- so its `p_offset`
        // is never read, and the identity map legitimately leaves it past the
        // end of a file whose writable content is all `SHT_NOBITS`.
        assert!(
            s.filesz == 0 || s.offset.saturating_add(s.filesz) <= len,
            "a segment claims file bytes {:#x}..{:#x} of a {len:#x}-byte image",
            s.offset,
            s.offset + s.filesz
        );
    }
}

/// Every invariant `PT_GNU_RELRO` has to hold, checked against the sections
/// the image actually has.
///
/// The segment is present exactly when some protected section is, it opens the
/// writable `PT_LOAD`, it ends on a page boundary, it stays inside that
/// `PT_LOAD`, and it contains no section it should not.
fn check_relro(bytes: &[u8]) {
    let segs = segments(bytes);
    check_phdr_table(&segs);
    check_load_extents(bytes, &segs);
    let present: Vec<&[u8]> = PROTECTED
        .into_iter()
        .filter(|n| section(bytes, n).is_some_and(|s| s.size != 0))
        .collect();
    let Some(relro) = find(&segs, PT_GNU_RELRO) else {
        assert!(
            present.is_empty(),
            "the image has protected sections ({:?}) but no PT_GNU_RELRO",
            present.iter().map(|n| show(n)).collect::<Vec<_>>()
        );
        return;
    };
    assert!(
        !present.is_empty(),
        "PT_GNU_RELRO must not be emitted for an empty run"
    );
    assert_eq!(relro.flags, PF_R, "the protected run is read-only");
    assert_eq!(
        relro.align, 1,
        "PT_GNU_RELRO names a range, it is not loaded"
    );

    let rw = segs
        .iter()
        .find(|s| s.p_type == PT_LOAD && s.flags & PF_W != 0)
        .copied()
        .expect("a writable PT_LOAD");
    assert_eq!(
        (relro.offset, relro.vaddr),
        (rw.offset, rw.vaddr),
        "the run must open the writable segment: glibc protects one range, so \
         anything ahead of it would have to be dragged in or the run split"
    );
    let end = relro.vaddr + relro.memsz;
    assert_eq!(
        end % PAGE,
        0,
        "the run must end on a page boundary ({end:#x}): _dl_protect_relro \
         rounds the end down, so a partial page is left writable"
    );
    assert!(
        end <= rw.vaddr + rw.memsz,
        "the run ({:#x}..{end:#x}) must stay inside the writable PT_LOAD \
         ({:#x}..{:#x}); mprotect over unmapped memory is a fatal startup \
         error",
        relro.vaddr,
        rw.vaddr,
        rw.vaddr + rw.memsz
    );
    assert!(
        relro.filesz <= relro.memsz,
        "the run's file content cannot exceed the range it covers"
    );
    for name in PROTECTED {
        if let Some(s) = section(bytes, name).filter(|s| s.size != 0) {
            assert!(
                covers(&relro, &s),
                "{} must be inside the protected run",
                show(name)
            );
        }
    }
}

/// `PT_GNU_STACK` describes a readable, writable, non-executable stack.
///
/// The segment covers no bytes: the kernel reads its flags and nothing else,
/// and only `PF_X` changes what it does with them. The other two are still
/// part of the answer, because the segment stands for the permissions of the
/// stack, and no loader has ever produced a stack that may be written but not
/// read. lld spells `PF_R | PF_W` out and GNU ld emits `RW` as well; xold
/// emitted a bare `W`, which every reader printed back.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_gnu_stack_marker_is_readable_and_writable() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let dir = workdir("gnustack");
    let prog = dir.join("prog");
    let Some(obj) = compile(SRC, &dir.join("m.o"), &["-fPIE", "-c"]) else {
        return;
    };
    h.link_exec(&obj, &prog);
    let bytes = fs::read(&prog).expect("read output");

    let stack = find(&segments(&bytes), PT_GNU_STACK)
        .expect("every image carries the PT_GNU_STACK marker");
    assert_eq!(
        stack.flags,
        PF_R | PF_W,
        "the stack is readable and writable, and must not be executable"
    );
    assert_eq!(
        (stack.filesz, stack.memsz),
        (0, 0),
        "the marker describes no bytes"
    );
}

/// The declared program header table is exactly the one the writer emitted.
///
/// The count is decided before placement (it sizes the header block every
/// region is then placed after) and the headers are emitted afterwards, by a
/// second walk. A count one too high leaves a zeroed slot inside the declared
/// table; one too low pushes the last header past the declared end, over the
/// first bytes of content. Both show up here: every declared slot names a
/// segment, and the last one is the `PT_GNU_STACK` marker the writer always
/// appends.
fn check_phdr_table(segs: &[Segment]) {
    for (i, s) in segs.iter().enumerate() {
        assert_ne!(
            s.p_type,
            0,
            "declared program header {i} of {} is empty: the header count \
             over-ran what the writer emitted",
            segs.len()
        );
    }
    assert_eq!(
        segs.last().map(|s| s.p_type),
        Some(PT_GNU_STACK),
        "the last declared header must be the PT_GNU_STACK marker the writer \
         appends; anything else means the count fell short of the headers"
    );
}

/// Whether a segment's memory range wholly contains a section's.
const fn covers(seg: &Segment, sec: &Section) -> bool {
    let (lo, hi) = (seg.vaddr, seg.vaddr + seg.memsz);
    sec.addr >= lo && sec.addr + sec.size <= hi
}

// --- image decoding --------------------------------------------------------

/// One program header, decoded from the image.
#[derive(Clone, Copy)]
struct Segment {
    p_type: u32,
    flags: u32,
    offset: u64,
    vaddr: u64,
    filesz: u64,
    memsz: u64,
    align: u64,
}

/// One section's placed address, size and file offset.
struct Section {
    addr: u64,
    size: u64,
}

/// `Ehdr64` offsets of the program-header location fields.
const E_PHOFF: usize = 32;
const E_PHENTSIZE: usize = 54;
const E_PHNUM: usize = 56;

/// Decodes every declared program header. Reading the table by `e_phnum`
/// rather than by what the writer emitted is the point: a mismatch between the
/// two is what the static test looks for.
fn segments(bytes: &[u8]) -> Vec<Segment> {
    let phoff = usize::try_from(read_u64(bytes, E_PHOFF)).unwrap_or(0);
    let entsize = read_u16(bytes, E_PHENTSIZE);
    let count = read_u16(bytes, E_PHNUM);
    (0..count)
        .map(|i| {
            let at = phoff + i * entsize;
            Segment {
                p_type: read_u32(bytes, at),
                flags: read_u32(bytes, at + 4),
                offset: read_u64(bytes, at + 8),
                vaddr: read_u64(bytes, at + 16),
                filesz: read_u64(bytes, at + 32),
                memsz: read_u64(bytes, at + 40),
                align: read_u64(bytes, at + 48),
            }
        })
        .collect()
}

/// The first segment of `p_type`, if the image has one.
fn find(segs: &[Segment], p_type: u32) -> Option<Segment> {
    segs.iter().find(|s| s.p_type == p_type).copied()
}

/// The placed address and size of the section named `name`.
fn section(bytes: &[u8], name: &[u8]) -> Option<Section> {
    let obj = ObjectFile::parse(bytes).ok()?;
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == name)?;
    Some(Section {
        addr: shdr.sh_addr.get(),
        size: shdr.sh_size.get(),
    })
}

/// A section name as text, for assertion messages.
fn show(name: &[u8]) -> String {
    String::from_utf8_lossy(name).into_owned()
}

fn read_u16(bytes: &[u8], at: usize) -> usize {
    let mut buf = [0u8; 2];
    if let Some(slot) = bytes.get(at..at + 2) {
        buf.copy_from_slice(slot);
    }
    usize::from(u16::from_le_bytes(buf))
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    let mut buf = [0u8; 4];
    if let Some(slot) = bytes.get(at..at + 4) {
        buf.copy_from_slice(slot);
    }
    u32::from_le_bytes(buf)
}

fn read_u64(bytes: &[u8], at: usize) -> u64 {
    let mut buf = [0u8; 8];
    if let Some(slot) = bytes.get(at..at + 8) {
        buf.copy_from_slice(slot);
    }
    u64::from_le_bytes(buf)
}

// --- toolchain -------------------------------------------------------------

/// A fresh per-test working directory, named after the running process so two
/// test binaries never share one.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_relro_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Compiles `src` into `obj` with the host clang, returning the object path or
/// `None` when clang is unavailable.
fn compile(src: &[u8], obj: &Path, args: &[&str]) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let ok = Command::new(clang)
        .arg("--target=x86_64-linux-gnu")
        .args(args)
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then(|| obj.to_path_buf())
}

/// Assembles `src` into `obj` with the host clang.
fn assemble(src: &[u8], obj: &Path) -> Option<PathBuf> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("S");
    fs::write(&src_path, src).expect("write assembly");
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then(|| obj.to_path_buf())
}

/// The host pieces a libc link needs, or `None` (with a note) when one is
/// missing.
struct Harness {
    crt1: PathBuf,
    crti: PathBuf,
    crtn: PathBuf,
    libc: PathBuf,
    interp: Vec<u8>,
}

impl Harness {
    fn detect() -> Option<Self> {
        if which("clang").is_none() {
            eprintln!("skipping relro tests: clang unavailable");
            return None;
        }
        let crt1 = crt_file("Scrt1.o").or_else(|| crt_file("crt1.o"))?;
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

    /// Links `obj` plus the crt objects and libc into a dynamic executable.
    fn link_exec(&self, obj: &Path, prog: &Path) {
        link_dyn_exec(
            &[
                self.crt1.clone(),
                self.crti.clone(),
                obj.to_path_buf(),
                self.libc.clone(),
                self.crtn.clone(),
            ],
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
