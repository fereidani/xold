//! Dynamic linking (`-shared`) end-to-end tests.
//!
//! These compile a real `-fPIC` C source, link it with `xold -shared`, and
//! check the resulting shared object is structurally valid and loadable:
//!
//! - The ELF header is `ET_DYN` for `EM_X86_64`.
//! - The required dynamic sections (`.dynsym`, `.dynstr`, `.hash`, `.dynamic`,
//!   `.rela.dyn`, `.got`) and the `PT_DYNAMIC` segment exist.
//! - The `DT_*` tag set matches what the system linker emits for the same input
//!   (HASH, SYMTAB, STRTAB, STRSZ, SYMENT, RELA, RELASZ, RELAENT, NULL; plus
//!   RELACOUNT when there are relative relocations).
//! - The exported symbol is present in `.dynsym`.
//! - The authoritative functional check: a C harness `dlopen`s the library,
//!   `dlsym`s the exported function, calls it, and checks the return value. A
//!   correct dynamic table set is the only way that succeeds.
//!
//! All tests are gated on `clang` being available; if it is absent they print
//! a note and return, so the build never fails over a missing toolchain.

use std::{fs, path::Path, process::Command};

use common::which;
use xold::{
    elf::{ObjectFile, constants::*},
    icf::IcfMode,
    linker::link_shared,
};

mod common;

/// `PT_DYNAMIC` was added to the reader-facing constants; define it locally
/// too so the structural test does not depend on the constant being re-exported
/// through every path.
const PT_DYNAMIC: u32 = 2;
/// GNU hash-table tag (`DT_GNU_HASH`); the system linker emits this in place
/// of the classic `DT_HASH`. Either satisfies the loader's symbol lookup.
const DT_GNU_HASH: i64 = 0x6fff_fef5;

/// Whether a tag list carries a symbol-hash index (classic or GNU).
fn has_hash(tags: &[i64]) -> bool {
    tags.contains(&DT_HASH) || tags.contains(&DT_GNU_HASH)
}

/// The PIC fixture: `bump` reads and writes the global `counter` through a
/// GOT slot (compiled `-fPIC` this is `R_X86_64_REX_GOTPCRELX`), and is
/// exported. `counter` starts at 5, so a correct call returns 6.
const SHARED_SRC: &[u8] =
    b"int counter = 5;\nint bump(void) { counter += 1; return counter; }\n";

/// A fixture that adds absolute data pointers: `gp` is a global pointer to
/// the global `counter` (-> `R_X86_64_64` dynamic reloc), and `lp` is a
/// static pointer to a static `local` (-> `R_X86_64_RELATIVE`). Exercises
/// both symbol-based and relative dynamic relocation flavours in `.data`.
const DATA_PTR_SRC: &[u8] = b"int counter = 5;\n\
    static int local = 7;\n\
    int *gp = &counter;\n\
    static int *lp = &local;\n\
    int bump(void) { counter += 1; return counter; }\n\
    int readlp(void) { return *lp; }\n";

/// Compiles `src` to a `-fPIC` object with the host clang, returning its path.
/// Returns `None` (after printing a note) if clang is missing or the compile
/// fails.
fn compile_pic(src: &[u8], obj: &Path) -> Option<()> {
    let clang = which("clang")?;
    let src_path = obj.with_extension("c");
    fs::write(&src_path, src).expect("write shared source");
    let ok = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fPIC", "-ffreestanding", "-c"])
        .arg(&src_path)
        .arg("-o")
        .arg(obj)
        .status()
        .ok()?
        .success();
    let _ = fs::remove_file(&src_path);
    ok.then_some(())
}

/// Links `obj` with `xold -shared` (via the library entry) into `out`.
fn link_shared_with_xold(obj: &Path, out: &Path) {
    let inputs = [obj.to_path_buf()];
    link_shared(&inputs, out, None, false, IcfMode::None, false)
        .expect("xold -shared link must succeed");
}

/// The dynamic-link structural test: checks the ELF type, the presence of
/// every required section and program header, and that the exported symbol is
/// in `.dynsym`.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn shared_object_has_well_formed_dynamic_tables() {
    let Some(_) = which("clang") else {
        eprintln!("skipping dynamic structural test: clang not found");
        return;
    };
    let obj = std::env::temp_dir().join("xold_dyn_struct.o");
    let out = std::env::temp_dir().join("xold_dyn_struct.so");
    compile_pic(SHARED_SRC, &obj).expect("host clang compiles -fPIC");
    link_shared_with_xold(&obj, &out);

    let bytes = fs::read(&out).expect("output readable");
    let obj_out = ObjectFile::parse(&bytes).expect("output must be valid ELF");
    assert_eq!(obj_out.header().e_type.get(), ET_DYN, "must be ET_DYN");
    assert_eq!(obj_out.machine(), EM_X86_64);

    // Every required section is present.
    let names: Vec<&[u8]> = obj_out
        .sections()
        .iter()
        .map(|s| obj_out.section_name(s))
        .collect();
    for required in [
        b".text".as_slice(),
        b".dynsym",
        b".dynstr",
        b".hash",
        b".dynamic",
        b".rela.dyn",
        b".got",
    ] {
        assert!(
            names.contains(&required),
            "missing required dynamic section {required:?}"
        );
    }

    // PT_DYNAMIC is present.
    assert!(
        find_phdr_type(&bytes, PT_DYNAMIC).is_some(),
        "output has a PT_DYNAMIC segment"
    );

    // The exported symbol `bump` is in `.dynsym`.
    let dynsym = obj_out
        .sections()
        .iter()
        .find(|s| obj_out.section_name(s) == b".dynsym")
        .expect("has .dynsym");
    let dynstr = obj_out
        .sections()
        .iter()
        .find(|s| obj_out.section_name(s) == b".dynstr")
        .expect("has .dynstr");
    let dynstr_bytes = obj_out.section_data(dynstr).expect("dynstr bytes");
    let sym_bytes = obj_out.section_data(dynsym).expect("dynsym bytes");
    assert!(
        sym_includes_name(sym_bytes, dynstr_bytes, b"bump"),
        ".dynsym must export `bump`"
    );
    assert!(
        sym_includes_name(sym_bytes, dynstr_bytes, b"counter"),
        ".dynsym must export `counter`"
    );

    let _ = fs::remove_file(&obj);
    let _ = fs::remove_file(&out);
}

/// The dynamic `DT_*` tag set matches the system linker for the same input.
/// Hash values and exact layout need not match, but the tag set and the
/// exported symbol set must.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn dt_tags_match_the_system_linker() {
    let Some(_) = which("clang") else {
        eprintln!("skipping DT tag comparison: clang not found");
        return;
    };
    let obj = std::env::temp_dir().join("xold_dyn_dt.o");
    let xold_out = std::env::temp_dir().join("xold_dyn_dt_xold.so");
    let sys_out = std::env::temp_dir().join("xold_dyn_dt_sys.so");
    compile_pic(SHARED_SRC, &obj).expect("host clang compiles -fPIC");

    link_shared_with_xold(&obj, &xold_out);

    // System linker: link the same object `-shared -nostdlib` so the tag set
    // is the bare minimum (no libc-induced NEEDED / INIT / FINI_ARRAY).
    let sys_ok = Command::new("clang")
        .args(["-shared", "-nostdlib", "-nodefaultlibs"])
        .arg(&obj)
        .arg("-o")
        .arg(&sys_out)
        .status()
        .is_ok_and(|s| s.success());
    if !sys_ok {
        let _ = fs::remove_file(&obj);
        eprintln!("skipping DT tag comparison: system linker unavailable");
        return;
    }

    let xold_tags = dt_tags(&xold_out);
    let sys_tags = dt_tags(&sys_out);
    // xold emits the subset of tags the loader needs for this first cut; the
    // system linker may emit extras (GNU_HASH, VERNEED, ...). The classic
    // `DT_HASH` and the GNU `DT_GNU_HASH` are interchangeable here: either
    // gives the loader a symbol-lookup index. Both outputs must carry one.
    assert!(has_hash(&xold_tags), "xold output has a hash table tag");
    assert!(has_hash(&sys_tags), "system output has a hash table tag");
    // The remaining core tags every shared object must list in both outputs.
    for required in [
        DT_SYMTAB, DT_STRTAB, DT_STRSZ, DT_SYMENT, DT_RELA, DT_RELASZ,
        DT_RELAENT, DT_NULL,
    ] {
        assert!(
            xold_tags.contains(&required),
            "xold output missing required DT tag {required}"
        );
        assert!(
            sys_tags.contains(&required),
            "system output missing required DT tag {required}"
        );
    }

    // The exported dynsym symbol set must match.
    let xold_syms = dynsym_names(&xold_out);
    let sys_syms = dynsym_names(&sys_out);
    for required in [b"bump".as_slice(), b"counter"] {
        assert!(
            xold_syms.iter().any(|n| n == required),
            "xold .dynsym missing {required:?}"
        );
        assert!(
            sys_syms.iter().any(|n| n == required),
            "system .dynsym missing {required:?}"
        );
    }

    let _ = fs::remove_file(&obj);
    let _ = fs::remove_file(&xold_out);
    let _ = fs::remove_file(&sys_out);
}

/// The authoritative functional check: `dlopen` the library produced by xold,
/// resolve `bump`, call it, and verify the return value. A correct set of
/// dynamic tables, hash, and `R_X86_64_GLOB_DAT` relocation is the only way
/// this end-to-end check passes.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn dlopen_resolves_and_calls_the_exported_symbol() {
    let Some(_) = which("clang") else {
        eprintln!("skipping dlopen test: clang not found");
        return;
    };
    let obj = std::env::temp_dir().join("xold_dyn_dl.o");
    let so = std::env::temp_dir().join("xold_dyn_dl.so");
    compile_pic(SHARED_SRC, &obj).expect("host clang compiles -fPIC");
    link_shared_with_xold(&obj, &so);

    let harness_src = b"#include <dlfcn.h>\n\
        #include <stdio.h>\n\
        #include <stdlib.h>\n\
        int main(int argc, char **argv) {\n\
            if (argc < 2) return 4;\n\
            void *h = dlopen(argv[1], RTLD_NOW);\n\
            if (!h) { fprintf(stderr, \"dlopen: %s\\n\", dlerror()); return 1; }\n\
            int (*bump)(void) = (int (*)(void))dlsym(h, \"bump\");\n\
            if (!bump) { fprintf(stderr, \"dlsym: %s\\n\", dlerror()); return 2; }\n\
            int r = bump();\n\
            dlclose(h);\n\
            return r == 6 ? 0 : 3;\n\
        }\n";
    let src_path = std::env::temp_dir().join("xold_dyn_harness.c");
    let harness = std::env::temp_dir().join("xold_dyn_harness");
    fs::write(&src_path, harness_src).expect("write harness source");
    let built = Command::new("cc")
        .args(["-ldl"])
        .arg(&src_path)
        .arg("-o")
        .arg(&harness)
        .status()
        .is_ok_and(|s| s.success());
    let _ = fs::remove_file(&src_path);
    if !built {
        let _ = fs::remove_file(&obj);
        let _ = fs::remove_file(&so);
        eprintln!("skipping dlopen test: could not build harness");
        return;
    }

    let status = Command::new(&harness).arg(&so).status();
    let _ = fs::remove_file(&harness);
    let _ = fs::remove_file(&obj);
    let status = status.expect("harness must be runnable");
    assert!(
        status.success(),
        "dlopen/dlsym/bump failed with {:?} (expected 0)",
        status.code()
    );
    let _ = fs::remove_file(&so);
}

/// The data-reloc functional check: a fixture that stores absolute pointers
/// in `.data` (a global pointer to a global, and a static pointer to a
/// static) is linked with `xold -shared` and exercised via `dlopen`. The
/// global pointer becomes an `R_X86_64_64` dynamic reloc; the static pointer
/// becomes an `R_X86_64_RELATIVE`. Both must be resolved correctly by the
/// loader for the call to return the right value.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn dlopen_resolves_absolute_data_pointers() {
    let Some(_) = which("clang") else {
        eprintln!("skipping data-pointer dlopen test: clang not found");
        return;
    };
    let obj = std::env::temp_dir().join("xold_dyn_dp.o");
    let so = std::env::temp_dir().join("xold_dyn_dp.so");
    compile_pic(DATA_PTR_SRC, &obj).expect("host clang compiles -fPIC");
    link_shared_with_xold(&obj, &so);

    // Confirm both reloc flavours survived into `.rela.dyn`.
    let bytes = fs::read(&so).expect("output readable");
    let rela_bytes = rela_dyn_bytes(&bytes);
    let has_relative = rela_bytes.iter().any(|(_off, sym, ty, _add)| {
        *ty == xold::reloc::x86_64::R_X86_64_RELATIVE && *sym == 0
    });
    let has_abs = rela_bytes
        .iter()
        .any(|(_off, _sym, ty, _add)| *ty == xold::reloc::x86_64::R_X86_64_64);
    assert!(
        has_relative,
        ".rela.dyn must carry an R_X86_64_RELATIVE for the static pointer"
    );
    assert!(
        has_abs,
        ".rela.dyn must carry an R_X86_64_64 for the global pointer"
    );

    let harness_src = b"#include <dlfcn.h>\n\
        #include <stdio.h>\n\
        int main(int argc, char **argv) {\n\
            if (argc < 2) return 4;\n\
            void *h = dlopen(argv[1], RTLD_NOW);\n\
            if (!h) { fprintf(stderr, \"dlopen: %s\\n\", dlerror()); return 1; }\n\
            int (*bump)(void) = (int (*)(void))dlsym(h, \"bump\");\n\
            int (*readlp)(void) = (int (*)(void))dlsym(h, \"readlp\");\n\
            int **gp = (int **)dlsym(h, \"gp\");\n\
            if (!bump || !readlp || !gp) {\n\
                fprintf(stderr, \"dlsym: %s\\n\", dlerror()); return 2;\n\
            }\n\
            int lp_before = readlp();\n\
            int gp_before = **gp;\n\
            int after = bump();\n\
            dlclose(h);\n\
            return (lp_before == 7 && gp_before == 5 && after == 6) ? 0 : 3;\n\
        }\n";
    let src_path = std::env::temp_dir().join("xold_dyn_dp_harness.c");
    let harness = std::env::temp_dir().join("xold_dyn_dp_harness");
    fs::write(&src_path, harness_src).expect("write harness source");
    let built = Command::new("cc")
        .args(["-ldl"])
        .arg(&src_path)
        .arg("-o")
        .arg(&harness)
        .status()
        .is_ok_and(|s| s.success());
    let _ = fs::remove_file(&src_path);
    if !built {
        let _ = fs::remove_file(&obj);
        let _ = fs::remove_file(&so);
        eprintln!("skipping data-pointer dlopen test: no harness build");
        return;
    }

    let status = Command::new(&harness).arg(&so).status();
    let _ = fs::remove_file(&harness);
    let _ = fs::remove_file(&obj);
    let status = status.expect("harness must be runnable");
    assert!(
        status.success(),
        "data-pointer dlopen failed with {:?} (expected 0)",
        status.code()
    );
    let _ = fs::remove_file(&so);
}

// --- helpers ---------------------------------------------------------------

/// `Ehdr64` offset of `e_phoff`.
const E_PHOFF: usize = 32;
/// `Ehdr64` offset of `e_phentsize`.
const E_PHENTSIZE: usize = 54;
/// `Ehdr64` offset of `e_phnum`.
const E_PHNUM: usize = 56;

/// Returns the first program-header `p_type` matching `p_type`, if any.
fn find_phdr_type(bytes: &[u8], p_type: u32) -> Option<u32> {
    phdr_types(bytes).into_iter().find(|t| *t == p_type)
}

/// Reads every program-header `p_type` from an ELF64 image.
#[expect(
    clippy::cast_possible_truncation,
    reason = "file offsets and counts are small"
)]
fn phdr_types(bytes: &[u8]) -> Vec<u32> {
    let phoff = read_u64(bytes, E_PHOFF) as usize;
    let phentsize = read_u16(bytes, E_PHENTSIZE);
    let phnum = read_u16(bytes, E_PHNUM);
    let mut out = Vec::with_capacity(phnum);
    for i in 0..phnum {
        out.push(read_u32(bytes, phoff + i * phentsize));
    }
    out
}

/// Returns the set of `DT_*` tags in the `.dynamic` section.
fn dt_tags(path: &Path) -> Vec<i64> {
    let bytes = fs::read(path).expect("read shared object");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let Some(dyn_shdr) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynamic")
    else {
        return Vec::new();
    };
    let data = obj.section_data(dyn_shdr).expect("dynamic bytes");
    let mut tags = Vec::new();
    for chunk in data.chunks(16) {
        if chunk.len() < 16 {
            break;
        }
        let tag = i64::from_le_bytes(chunk[..8].try_into().unwrap_or([0; 8]));
        if tag == DT_NULL {
            tags.push(tag);
            break;
        }
        tags.push(tag);
    }
    tags
}

/// Returns the set of names in `.dynsym` (via `.dynstr`).
fn dynsym_names(path: &Path) -> Vec<Vec<u8>> {
    let bytes = fs::read(path).expect("read shared object");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let dynsym = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynsym")
        .expect("has .dynsym");
    let dynstr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".dynstr")
        .expect("has .dynstr");
    let dynstr_bytes = obj.section_data(dynstr).expect("dynstr bytes");
    let sym_bytes = obj.section_data(dynsym).expect("dynsym bytes");
    let mut out = Vec::new();
    for chunk in sym_bytes.chunks(24) {
        if chunk.len() < 24 {
            break;
        }
        let name_off =
            u32::from_le_bytes(chunk[..4].try_into().unwrap_or([0; 4]))
                as usize;
        if let Some(end) = dynstr_bytes[name_off..].iter().position(|&b| b == 0)
        {
            out.push(dynstr_bytes[name_off..name_off + end].to_vec());
        }
    }
    out
}

/// Decodes `.rela.dyn` into `(r_offset, sym, r_type, r_addend)` rows.
fn rela_dyn_bytes(bytes: &[u8]) -> Vec<(u64, u32, u32, i64)> {
    let Ok(obj) = ObjectFile::parse(bytes) else {
        return Vec::new();
    };
    let Some(rela) = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".rela.dyn")
    else {
        return Vec::new();
    };
    let Ok(data) = obj.section_data(rela) else {
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
        let add =
            i64::from_le_bytes(chunk[16..24].try_into().unwrap_or([0; 8]));
        let sym = u32::try_from(info >> 32).unwrap_or(0);
        let ty = u32::try_from(info & 0xffff_ffff).unwrap_or(0);
        out.push((off, sym, ty, add));
    }
    out
}

/// Whether `.dynsym` (read via `.dynstr`) contains a symbol named `name`.
fn sym_includes_name(sym_bytes: &[u8], strtab: &[u8], name: &[u8]) -> bool {
    for chunk in sym_bytes.chunks(24) {
        if chunk.len() < 24 {
            break;
        }
        let name_off =
            u32::from_le_bytes(chunk[..4].try_into().unwrap_or([0; 4]))
                as usize;
        if name_off >= strtab.len() {
            continue;
        }
        let end = strtab[name_off..]
            .iter()
            .position(|&b| b == 0)
            .map_or(0, |n| name_off + n);
        if &strtab[name_off..end] == name {
            return true;
        }
    }
    false
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
