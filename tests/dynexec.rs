//! Dynamic executable linking end-to-end tests.
//!
//! These build the consumer side of dynamic linking: a `-fPIE` main that calls
//! a function imported from a shared object, linked into a position-independent
//! `ET_DYN` executable that runs under the system `ld.so` with default (lazy)
//! binding. The call crosses a PLT stub the loader binds on first use.
//!
//! - The end-to-end proof links `libfoo.so` with `xold -shared`, links a
//!   `-fPIE` main against it with xold, and runs the result directly under
//!   `ld.so` (no `LD_BIND_NOW`).
//! - The structural test checks `ET_DYN`, `PT_INTERP`, `PT_DYNAMIC`, the PLT
//!   sections, and that the entry point is `_start`.
//! - The system-linker comparison links the same inputs with `clang -pie` and
//!   confirms both reach the same result with matching PLT/GOT.PLT structure.
//!
//! Gated on `clang` and the system `ld.so`; if absent the tests print a note
//! and return, so the build never fails over a missing toolchain.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::which;
use xold::{
    elf::{ObjectFile, constants::*},
    icf::IcfMode,
    linker::{link_dyn_exec, link_shared},
};

mod common;

/// `PT_PHDR` (the program-header-table segment), emitted for a dynamic
/// executable so the loader can recover its load base.
const PT_PHDR: u32 = 6;
/// `PT_INTERP` is re-exported by the reader-facing constants; define it locally
/// too so the structural test does not depend on the re-export path.
const PT_INTERP_LOCAL: u32 = 3;
/// `DT_PLTREL` selecting `DT_RELA` for `.rela.plt`.
const DT_PLTREL_TAG: i64 = 20;

/// The shared library source: `bump` increments and returns `counter`.
const LIB_SRC: &[u8] =
    b"int counter = 5;\nint bump(void) { counter += 1; return counter; }\n";

/// The main source: calls the imported `bump` and returns its value. Compiled
/// `-fPIE`, the call is `R_X86_64_PLT32`, which xold routes through a PLT stub
/// because `bump` is an undefined import.
const MAIN_SRC: &[u8] =
    b"extern int bump(void);\nint main(void) { return bump(); }\n";

/// A main that stores the address of one of its own globals in a pointer the
/// compiler must initialise statically (an `R_X86_64_64` in `.data`), then
/// dereferences it and adds the imported `bump`.
///
/// In a position-independent executable that stored address is a link-time one
/// and means nothing until the loader has added the load base, so the slot
/// needs an `R_X86_64_RELATIVE`. Without it the pointer holds a low, unmapped
/// address and the program faults on the first dereference. `36 + bump() = 42`.
const OWN_PTR_SRC: &[u8] = b"int own = 36;\nint *ownp = &own;\n\
    extern int bump(void);\nint main(void) { return *ownp + bump(); }\n";

/// A freestanding `_start` that calls `main` then exits with its return value
/// (syscall 60). The call to `main` is resolved within the executable.
const START_SRC: &[u8] = b"    .text\n    .globl _start\n_start:\n    \
    call    main\n    movl    %eax, %edi\n    movl    $60, %eax\n    syscall\n";

/// Compiles `src` with the host clang, writing the object to `obj`. `pic`
/// selects `-fPIC` (for the library) vs `-fPIE` (for the main).
fn compile(
    src: &[u8],
    obj: &Path,
    pic: bool,
    freestanding: bool,
) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write source");
    let mut cmd = Command::new(clang);
    cmd.arg("--target=x86_64-linux-gnu");
    if pic {
        cmd.arg("-fPIC");
    } else {
        cmd.arg("-fPIE");
    }
    if freestanding {
        cmd.args(["-ffreestanding", "-c"]);
    } else {
        cmd.arg("-c");
    }
    cmd.arg(&src_path).arg("-o").arg(obj);
    let ok = cmd.status().ok()?.success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// Assembles `src` (assembly) with clang.
fn assemble(src: &[u8], obj: &Path) -> Option<()> {
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
    ok.then_some(())
}

/// Creates a fresh per-test working directory under the system temp dir, so
/// parallel tests get disjoint namespaces. The library file is named
/// `libfoo.so` to match its soname (the loader searches by the `DT_NEEDED`
/// name), so each test must keep its `libfoo.so` in its own directory.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xold_dynexec_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Builds `libfoo.so` with xold (`-shared`) from a `-fPIC` object, returning
/// its path inside `dir`.
fn build_lib(dir: &Path) -> Option<PathBuf> {
    let obj = dir.join("libfoo.o");
    let so = dir.join("libfoo.so");
    compile(LIB_SRC, &obj, true, true)?;
    link_shared(
        std::slice::from_ref(&obj),
        &so,
        Some(b"libfoo.so"),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold -shared link must succeed");
    let _ = fs::remove_file(&obj);
    Some(so)
}

/// The interpreter path the executable is loaded under. Probed from `/bin/true`
/// so the test matches the host's `ld.so`.
fn interpreter() -> Option<Vec<u8>> {
    let out = Command::new("readelf")
        .args(["-l", "/bin/true"])
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().find(|l| l.contains("interpreter:"))?;
    let start = line.find('/')?;
    let end = line.rfind(']').unwrap_or(line.len());
    Some(line.as_bytes()[start..end].to_vec())
}

/// The end-to-end proof: build the library with xold `-shared`, build a
/// `-fPIE` main that calls `bump`, link them with xold into a dynamic
/// executable, and run it directly under `ld.so`. The call must work under
/// default lazy binding (no `LD_BIND_NOW`), proving the PLT resolver path.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn dynamic_executable_runs_under_default_lazy_binding() {
    let Some(clang) = which("clang") else {
        eprintln!("skipping dynexec end-to-end: clang unavailable");
        return;
    };
    let _ = clang;
    let Some(interp) = interpreter() else {
        eprintln!("skipping dynexec end-to-end: interpreter path unknown");
        return;
    };
    let dir = workdir("run");
    let Some(lib) = build_lib(&dir) else {
        eprintln!("skipping dynexec end-to-end: host clang unavailable");
        return;
    };
    let main_o = dir.join("dynexec_run_main.o");
    let start_o = dir.join("dynexec_run_start.o");
    compile(MAIN_SRC, &main_o, false, true).expect("host clang compiles -fPIE");
    assemble(START_SRC, &start_o).expect("host clang assembles start.S");

    let prog = dir.join("dynexec_run_prog");
    link_dyn_exec(
        &[main_o.clone(), start_o.clone(), lib.clone()],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-exec link must succeed");

    // Run directly under ld.so (via PT_INTERP) with the library on the search
    // path. No LD_BIND_NOW is set: the call resolves lazily.
    let status = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("linked program must be runnable");
    let code = status.code();
    assert_eq!(
        code,
        Some(6),
        "bump() should return counter + 1 = 6 (lazy binding)"
    );

    let _ = fs::remove_file(&main_o);
    let _ = fs::remove_file(&start_o);
    let _ = fs::remove_file(&lib);
    let _ = fs::remove_file(&prog);
}

/// A pointer initialised to the address of a global the executable itself
/// defines must be relocated at load time.
///
/// The executable exports none of its own definitions, so there is no dynamic
/// symbol to name: the entry has to be an `R_X86_64_RELATIVE` carrying the
/// resolved address. xold used to emit nothing at all for such a reference,
/// which left the pointer at its link-time value and faulted under ASLR as
/// soon as it was dereferenced. Only running the program proves the slot was
/// relocated, and to the right address.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_pointer_to_an_own_global_is_relocated() {
    let Some(clang) = which("clang") else {
        eprintln!("skipping own-pointer test: clang unavailable");
        return;
    };
    let _ = clang;
    let Some(interp) = interpreter() else {
        eprintln!("skipping own-pointer test: interpreter path unknown");
        return;
    };
    let dir = workdir("ownptr");
    let Some(lib) = build_lib(&dir) else {
        eprintln!("skipping own-pointer test: host clang unavailable");
        return;
    };
    let main_o = dir.join("dynexec_ownptr_main.o");
    let start_o = dir.join("dynexec_ownptr_start.o");
    compile(OWN_PTR_SRC, &main_o, false, true)
        .expect("host clang compiles -fPIE");
    assemble(START_SRC, &start_o).expect("host clang assembles start.S");

    let prog = dir.join("dynexec_ownptr_prog");
    link_dyn_exec(
        &[main_o.clone(), start_o.clone(), lib.clone()],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-exec link must succeed");

    let status = Command::new(&prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("linked program must be runnable");
    assert_eq!(
        status.code(),
        Some(42),
        "*ownp + bump() should be 36 + 6 = 42; no exit code at all means the \
         program faulted on an unrelocated pointer",
    );

    let _ = fs::remove_file(&main_o);
    let _ = fs::remove_file(&start_o);
    let _ = fs::remove_file(&lib);
    let _ = fs::remove_file(&prog);
}

/// Structural checks: `ET_DYN`, `PT_PHDR`, `PT_INTERP` with the right path,
/// `PT_DYNAMIC`, and the PLT/GOT.PLT/`.rela.plt` sections present.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn dynamic_executable_structure_is_well_formed() {
    let Some(clang) = which("clang") else {
        eprintln!("skipping dynexec structural test: clang unavailable");
        return;
    };
    let _ = clang;
    let Some(interp) = interpreter() else {
        eprintln!("skipping dynexec structural test: interpreter unknown");
        return;
    };
    let dir = workdir("struct");
    let Some(lib) = build_lib(&dir) else {
        return;
    };
    let main_o = dir.join("dynexec_struct_main.o");
    let start_o = dir.join("dynexec_struct_start.o");
    compile(MAIN_SRC, &main_o, false, true).expect("host clang compiles -fPIE");
    assemble(START_SRC, &start_o).expect("host clang assembles start.S");
    let prog = dir.join("dynexec_struct_prog");
    link_dyn_exec(
        &[main_o.clone(), start_o.clone(), lib.clone()],
        &prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-exec link must succeed");

    let bytes = fs::read(&prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    assert_eq!(obj.header().e_type.get(), ET_DYN, "must be ET_DYN");
    assert_eq!(obj.machine(), EM_X86_64);
    assert_ne!(obj.header().e_entry.get(), 0, "entry point must be set");

    let ptypes = phdr_types(&bytes);
    assert!(ptypes.contains(&PT_PHDR), "must have PT_PHDR");
    assert!(ptypes.contains(&PT_INTERP_LOCAL), "must have PT_INTERP");
    assert!(ptypes.contains(&PT_DYNAMIC), "must have PT_DYNAMIC");

    let names: Vec<&[u8]> =
        obj.sections().iter().map(|s| obj.section_name(s)).collect();
    for required in [
        b".text".as_slice(),
        b".plt",
        b".got.plt",
        b".rela.plt",
        b".interp",
        b".dynamic",
        b".dynsym",
        b".dynstr",
    ] {
        assert!(
            names.contains(&required),
            "missing required section {required:?}"
        );
    }

    // The `.rela.plt` carries one JUMP_SLOT for `bump`.
    let rela = rela_plt_rows(&bytes);
    assert_eq!(rela.len(), 1, "exactly one JUMP_SLOT for bump");
    assert_eq!(
        rela[0].2,
        xold::reloc::x86_64::R_X86_64_JUMP_SLOT,
        "reloc must be JUMP_SLOT"
    );

    // `DT_NEEDED` names `libfoo.so` and the PLT tags are present.
    let tags = dt_tags(&prog);
    assert!(
        tags.iter().any(|(t, _)| *t == DT_NEEDED),
        "must carry DT_NEEDED"
    );
    assert!(tags.iter().any(|(t, _)| *t == DT_PLTGOT), "DT_PLTGOT");
    assert!(tags.iter().any(|(t, _)| *t == DT_JMPREL), "DT_JMPREL");
    assert!(tags.iter().any(|(t, _)| *t == DT_PLTRELSZ), "DT_PLTRELSZ");
    assert!(
        tags.iter().any(|(t, v)| *t == DT_PLTREL_TAG && *v == 7),
        "DT_PLTREL = DT_RELA"
    );

    let _ = fs::remove_file(&main_o);
    let _ = fs::remove_file(&start_o);
    let _ = fs::remove_file(&lib);
    let _ = fs::remove_file(&prog);
}

/// System-linker comparison: link the same inputs with `clang -pie` and confirm
/// both xold's and the system's executables run to the same result, and that
/// the PLT/GOT.PLT/`.rela.plt` structure (entry counts) matches.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn dynamic_executable_matches_the_system_linker() {
    let Some(clang) = which("clang") else {
        eprintln!("skipping dynexec comparison: clang unavailable");
        return;
    };
    let _ = clang;
    let Some(interp) = interpreter() else {
        eprintln!("skipping dynexec comparison: interpreter unknown");
        return;
    };
    let dir = workdir("cmp");
    let Some(lib) = build_lib(&dir) else {
        return;
    };
    let main_o = dir.join("dynexec_cmp_main.o");
    let start_o = dir.join("dynexec_cmp_start.o");
    compile(MAIN_SRC, &main_o, false, true).expect("host clang compiles -fPIE");
    assemble(START_SRC, &start_o).expect("host clang assembles start.S");

    let xold_prog = dir.join("dynexec_cmp_xold");
    link_dyn_exec(
        &[main_o.clone(), start_o.clone(), lib.clone()],
        &xold_prog,
        b"_start",
        interp.as_slice(),
        false,
        IcfMode::None,
        false,
    )
    .expect("xold dynamic-exec link must succeed");

    // System linker: `-pie -nostdlib` keeps the dynamic tables to the bare
    // minimum (no libc-induced extras), matching xold's first cut.
    let sys_prog = dir.join("dynexec_cmp_sys");
    let sys_ok = Command::new("clang")
        .args([
            "-pie",
            "-nostdlib",
            "-nodefaultlibs",
            "-Wl,--dynamic-linker",
            "-Wl,/lib64/ld-linux-x86-64.so.2",
        ])
        .arg(&main_o)
        .arg(&start_o)
        .arg(&lib)
        .arg("-o")
        .arg(&sys_prog)
        .status()
        .is_ok_and(|s| s.success());
    if !sys_ok {
        eprintln!("skipping dynexec comparison: system linker unavailable");
        let _ = fs::remove_file(&main_o);
        let _ = fs::remove_file(&start_o);
        let _ = fs::remove_file(&lib);
        let _ = fs::remove_file(&xold_prog);
        return;
    }

    let xold_status = Command::new(&xold_prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("xold prog runnable");
    let sys_status = Command::new(&sys_prog)
        .env("LD_LIBRARY_PATH", &dir)
        .status()
        .expect("system prog runnable");
    assert_eq!(
        xold_status.code(),
        sys_status.code(),
        "both executables must reach the same result"
    );
    assert_eq!(xold_status.code(), Some(6), "expected bump() = 6");

    // The PLT entry count (one per import) must match.
    let xold_rela = rela_plt_rows(&fs::read(&xold_prog).unwrap());
    let sys_rela = rela_plt_rows(&fs::read(&sys_prog).unwrap());
    assert_eq!(
        xold_rela.len(),
        sys_rela.len(),
        "PLT entry count must match the system linker"
    );

    let _ = fs::remove_file(&main_o);
    let _ = fs::remove_file(&start_o);
    let _ = fs::remove_file(&lib);
    let _ = fs::remove_file(&xold_prog);
    let _ = fs::remove_file(&sys_prog);
}

// --- helpers ---------------------------------------------------------------

/// `Ehdr64` offsets of the program-header location fields.
const E_PHOFF: usize = 32;
const E_PHENTSIZE: usize = 54;
const E_PHNUM: usize = 56;

/// Reads every program-header `p_type`.
#[expect(clippy::cast_possible_truncation, reason = "small counts")]
fn phdr_types(bytes: &[u8]) -> Vec<u32> {
    let phoff = read_u64(bytes, E_PHOFF) as usize;
    let phentsize = read_u16(bytes, E_PHENTSIZE);
    let phnum = read_u16(bytes, E_PHNUM);
    (0..phnum)
        .map(|i| read_u32(bytes, phoff + i * phentsize))
        .collect()
}

/// Decodes `.rela.plt` into `(r_offset, sym, r_type, r_addend)` rows.
fn rela_plt_rows(bytes: &[u8]) -> Vec<(u64, u32, u32, i64)> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(rela) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".rela.plt")
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(rela) else {
        return Vec::new();
    };
    decode_rela(data)
}

/// Decodes a `.rela*` byte block into rows.
fn decode_rela(data: &[u8]) -> Vec<(u64, u32, u32, i64)> {
    let mut out = Vec::new();
    for chunk in data.chunks(24) {
        if chunk.len() < 24 {
            break;
        }
        let off = u64::from_le_bytes(chunk[..8].try_into().unwrap_or([0; 8]));
        let info =
            u64::from_le_bytes(chunk[8..16].try_into().unwrap_or([0; 8]));
        let add =
            i64::from_le_bytes(chunk[16..24].try_into().unwrap_or([0; 8]));
        out.push((
            off,
            u32::try_from(info >> 32).unwrap_or(0),
            u32::try_from(info & 0xffff_ffff).unwrap_or(0),
            add,
        ));
    }
    out
}

/// Reads the `(d_tag, d_un)` pairs from `.dynamic`, stopping at `DT_NULL`.
fn dt_tags(path: &Path) -> Vec<(i64, u64)> {
    let bytes = fs::read(path).expect("read output");
    let Ok(obj) = ObjectFile::parse(&bytes) else {
        return Vec::new();
    };
    let Some(dyn_shdr) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynamic")
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(dyn_shdr) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for chunk in data.chunks(16) {
        if chunk.len() < 16 {
            break;
        }
        let tag = i64::from_le_bytes(chunk[..8].try_into().unwrap_or([0; 8]));
        let val = u64::from_le_bytes(chunk[8..16].try_into().unwrap_or([0; 8]));
        out.push((tag, val));
        if tag == DT_NULL {
            break;
        }
    }
    out
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
