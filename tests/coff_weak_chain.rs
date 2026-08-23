//! Weak externals resolve through their defaults, and an undefined weak
//! is a quiet zero, not a failed link.
//!
//! A COFF weak external names its default definition by raw
//! symbol-table index. Two gaps showed up against lld:
//!
//! * the default may itself be a weak external -- a chain -- and the resolver
//!   followed exactly one hop, so a chain ended at zero and the link died with
//!   "resolves to address 0". lld walks the chain.
//! * a weak *declaration* (clang's `__attribute__((weak))` extern) has an
//!   absolute-zero default: the whole point is that the reference binds to
//!   nothing when no definition arrives. xold treated the zero as an unresolved
//!   symbol and refused the link.
//!
//! The chain test hand-builds a COFF object (no compiler emits a
//! weak-to-weak chain today): `.text` with one ADDR64 relocation that
//! names `w1`, whose default is the weak `w2`, whose default is the
//! defined `def`. The slot must receive `def`'s address. The
//! declaration test compiles the clang shape and checks the link
//! succeeds.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use xold::{coff::PeImage, input::Input};

// --- the hand-built chain object -------------------------------------------

/// Machine type and the one relocation type the object uses.
const AMD64: u16 = 0x8664;
const ADDR64: u16 = 0x0001;

/// Section flags: code, executable, readable.
const TEXT_FLAGS: u32 = 0x6000_0020;

/// Builds the object: one `.text` section, one ADDR64 relocation naming
/// `w1`, and the symbol table
/// `w1 (weak, default w2), w2 (weak, default def), def (defined at +8)`.
fn chain_object() -> Vec<u8> {
    // The symbol table sits past the section header, the raw data and
    // the relocation record, on an 8-byte boundary.
    const DATA_OFF: u32 = 60;
    const REL_OFF: u32 = 76;
    const SYM_OFF: u32 = 88;
    let mut out = Vec::new();
    out.extend_from_slice(&AMD64.to_le_bytes()); // Machine
    out.extend_from_slice(&1u16.to_le_bytes()); // NumberOfSections
    out.extend_from_slice(&0u32.to_le_bytes()); // TimeDateStamp
    out.extend_from_slice(&SYM_OFF.to_le_bytes()); // PointerToSymbolTable
    out.extend_from_slice(&5u32.to_le_bytes()); // NumberOfSymbols
    out.extend_from_slice(&0u16.to_le_bytes()); // SizeOfOptionalHeader
    out.extend_from_slice(&0u16.to_le_bytes()); // Characteristics
    // Section header: `.text`, 16 raw bytes, one relocation.
    out.extend_from_slice(b".text\0\0\0");
    out.extend_from_slice(&16u32.to_le_bytes()); // VirtualSize
    out.extend_from_slice(&0u32.to_le_bytes()); // VirtualAddress
    out.extend_from_slice(&16u32.to_le_bytes()); // SizeOfRawData
    out.extend_from_slice(&DATA_OFF.to_le_bytes()); // PointerToRawData
    out.extend_from_slice(&REL_OFF.to_le_bytes()); // PointerToRelocations
    out.extend_from_slice(&0u32.to_le_bytes()); // PointerToLinenumbers
    out.extend_from_slice(&1u16.to_le_bytes()); // NumberOfRelocations
    out.extend_from_slice(&0u16.to_le_bytes()); // NumberOfLinenumbers
    out.extend_from_slice(&TEXT_FLAGS.to_le_bytes()); // Characteristics
    debug_assert_eq!(out.len(), DATA_OFF as usize);
    // Raw data: the ADDR64 slot at offset 0, then padding.
    out.extend_from_slice(&[0u8; 16]);
    debug_assert_eq!(out.len(), REL_OFF as usize);
    // Relocation: patch the quadword at 0 with symbol 0 (`w1`).
    out.extend_from_slice(&0u32.to_le_bytes()); // VirtualAddress
    out.extend_from_slice(&0u32.to_le_bytes()); // SymbolTableIndex
    out.extend_from_slice(&ADDR64.to_le_bytes()); // Type
    while out.len() < SYM_OFF as usize {
        out.push(0);
    }
    // Symbol records, 18 bytes each. Long names are string-table offsets.
    //   0: w1  -- weak external, default -> 2 (w2)
    //   1: aux weak (tag 2)
    //   2: w2  -- weak external, default -> 4 (def)
    //   3: aux weak (tag 4)
    //   4: def -- external, section 1, value 8
    symbol(&mut out, 4, 0, 0, 0x69, 1); // "w1"
    aux_weak(&mut out, 2);
    symbol(&mut out, 7, 0, 0, 0x69, 1); // "w2"
    aux_weak(&mut out, 4);
    symbol(&mut out, 10, 8, 1, 0x02, 0); // "def"
    // String table: total length then the names at their offsets.
    out.extend_from_slice(&14u32.to_le_bytes());
    out.extend_from_slice(b"w1\0");
    out.extend_from_slice(b"w2\0");
    out.extend_from_slice(b"def\0");
    out
}

/// Appends one primary symbol record: a string-table name, `value`,
/// 1-based `section` (0 undefined), storage `class`, `naux` auxiliary
/// records to follow.
fn symbol(
    out: &mut Vec<u8>,
    name_off: u32,
    value: u32,
    section: i16,
    class: u8,
    naux: u8,
) {
    out.extend_from_slice(&0u32.to_le_bytes()); // zero prefix: long name
    out.extend_from_slice(&name_off.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
    out.extend_from_slice(&section.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // Type
    out.push(class);
    out.push(naux);
}

/// Appends the auxiliary record of a weak external: the default's raw
/// symbol-table index, the search type, ten unused bytes.
fn aux_weak(out: &mut Vec<u8>, tag: u32) {
    out.extend_from_slice(&tag.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes()); // alias search
    out.extend_from_slice(&[0u8; 10]);
}

/// A weak-to-weak chain resolves to the definition at its end.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_weak_chain_reaches_the_defined_symbol() {
    let dir = std::env::temp_dir()
        .join(format!("xold_wkchain_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create the workdir");
    let obj = dir.join("chain.obj");
    fs::write(&obj, chain_object()).expect("write the object");
    // A DLL link: no entry stub, so `.text` starts at the member itself.
    let out = dir.join("chain.dll");
    let files = [Input::Path(&obj)];
    let res = xold::coff::link_coff(&files, &out, b"", true);
    assert!(res.is_ok(), "the chain must resolve: {:?}", res.err());
    let bytes = fs::read(&out).expect("read the image");
    let img = PeImage::parse(&bytes).expect("parse the image");
    let text = img
        .sections()
        .into_iter()
        .find(|s| s.name == b".text")
        .expect(".text must be placed");
    let at = usize::try_from(text.pointer_to_raw_data).unwrap_or(0);
    let slot: [u8; 8] = bytes[at..at + 8].try_into().expect("the quadword");
    let stored = u64::from_le_bytes(slot);
    let expected = img.image_base() + u64::from(text.virtual_address) + 8;
    assert_eq!(stored, expected, "w1 -> w2 -> def must reach def's address");
    let _ = fs::remove_dir_all(&dir);
}

// --- the clang weak declaration --------------------------------------------

/// A weak declaration with no definition: clang's absolute-zero default.
const SRC: &[u8] = b"extern int missing(void) __attribute__((weak));\n\
    int main(void)\n\
    {\n\
        return missing ? missing() : 0;\n\
    }\n";

/// An undefined weak external links: the reference binds to zero and the
/// program guards it. xold refused it as an unresolved symbol.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn an_undefined_weak_external_links() {
    let dir = std::env::temp_dir()
        .join(format!("xold_wkundef_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create the workdir");
    let obj = dir.join("weak.obj");
    if !compile_msvc(SRC, &obj) {
        eprintln!("skipping coff-weak-chain: no msvc target");
        let _ = fs::remove_dir_all(&dir);
        return;
    }
    let out = dir.join("weak.exe");
    let files = [Input::Path(&obj)];
    let res = xold::coff::link_coff(&files, &out, b"main", false);
    assert!(res.is_ok(), "an undefined weak must link: {:?}", res.err());
    let _ = fs::remove_dir_all(&dir);
}

/// Compiles `src` for the msvc target into `out`, or answers `false`.
fn compile_msvc(src: &[u8], out: &Path) -> bool {
    let Some(clang) = which("clang") else {
        return false;
    };
    let file = std::env::temp_dir().join("xold_coff_wkundef.c");
    if fs::write(&file, src).is_err() {
        return false;
    }
    let ok = Command::new(clang)
        .args(["--target=x86_64-pc-windows-msvc", "-c"])
        .arg(&file)
        .arg("-o")
        .arg(out)
        .stdin(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    let _ = fs::remove_file(&file);
    ok
}

/// `PATH` search for a tool, mirroring the common helper.
fn which(tool: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(tool))
        .find(|p| p.exists())
}
