//! `--gc-sections` must not revive a dead function through its own unwind
//! data.
//!
//! An `.eh_frame` FDE relocates twice: to the function it describes, and to
//! the LSDA in `.gcc_except_table` that says where its landing pads are. gcc
//! puts an `inline` function's `.text` and its `.gcc_except_table` in one
//! COMDAT group, so the two live or die together.
//!
//! Following the LSDA edge looked harmless -- keep the table, drop the code --
//! but the group edge runs the other way: marking the LSDA enqueues every
//! member of its group, the dead `.text` included. An unreferenced `inline`
//! function with a landing pad therefore survived `--gc-sections`, code and
//! all, because its own FDE vouched for it. lld refuses the same edge for
//! exactly this reason (`lld/ELF/MarkLive.cpp`): a grouped or
//! `SHF_LINK_ORDER` table is retained with its function when the function is
//! live, so following the relocation can only ever drag dead code in.
//!
//! Gated on `clang++` and the host C++ runtime pieces; if any is missing the
//! tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{crt_file, interpreter, which};
use xold::{elf::ObjectFile, icf::IcfMode, linker::link_dyn_exec};

mod common;

/// A dead `inline` function with a landing pad (the destructor is what forces
/// one), a live function, and a `main` that reaches only the live one.
const SRC: &[u8] = b"struct D { ~D(); };\n\
    D::~D() {}\n\
    __attribute__((used)) inline int dead_thrower(int v) {\n\
    D d;\n\
    if (v > 3) throw v;\n\
    return 1;\n\
    }\n\
    int live_fn(int v) { return v + 2; }\n\
    int main() { return live_fn(1); }\n";

/// The collected link drops the dead function, code and unwind data both.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_dead_grouped_function_is_not_revived_by_its_lsda() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let Some(dir) = workdir("gc") else {
        return;
    };
    let obj = compile(&dir).expect("workdir checked clang++");
    let out = dir.join("gc_prog");
    link(&h, &obj, &out, true);
    let image = fs::read(&out).expect("read the image");
    let obj_out = ObjectFile::parse(&image).expect("valid ELF");
    let symtab = obj_out.symbol_table().ok().flatten().expect(".symtab");
    assert!(
        !symtab
            .iter()
            .any(|s| symtab.name(s) == b"_Z12dead_throweri"),
        "an unreferenced function must not survive --gc-sections because \
         its own FDE relocates to its LSDA"
    );
    assert!(
        symtab.iter().any(|s| symtab.name(s) == b"_Z7live_fni"),
        "the reachable function stays"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: nothing collected, the function and its table are both there,
/// so the drop above is the collection's doing and not a fixture that never
/// emitted them.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_uncollected_link_keeps_it() {
    let Some(h) = Harness::detect() else {
        return;
    };
    let Some(dir) = workdir("plain") else {
        return;
    };
    let obj = compile(&dir).expect("workdir checked clang++");
    let out = dir.join("plain_prog");
    link(&h, &obj, &out, false);
    let image = fs::read(&out).expect("read the image");
    let obj_out = ObjectFile::parse(&image).expect("valid ELF");
    let symtab = obj_out.symbol_table().ok().flatten().expect(".symtab");
    assert!(
        symtab
            .iter()
            .any(|s| symtab.name(s) == b"_Z12dead_throweri"),
        "without collection the function is present, as compiled"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Locates a shared library by name in the linker search path.
fn find_lib(name: &str) -> Option<PathBuf> {
    for dir in lib_search_dirs() {
        let candidate = dir.join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// The directories `gcc` searches for shared libraries, parsed from
/// `gcc -print-search-dirs`.
fn lib_search_dirs() -> Vec<PathBuf> {
    let Some(gcc) = which("gcc") else {
        return Vec::new();
    };
    let Some(out) = Command::new(gcc).arg("-print-search-dirs").output().ok()
    else {
        return Vec::new();
    };
    let s = String::from_utf8_lossy(&out.stdout);
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("libraries: =") {
            return rest
                .split(':')
                .filter(|p| !p.is_empty())
                .map(PathBuf::from)
                .collect();
        }
    }
    Vec::new()
}

/// Creates a fresh per-test working directory.
fn workdir(prefix: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir()
        .join(format!("xold_gc_lsda_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the fixture with one section per function.
fn compile(dir: &Path) -> Option<PathBuf> {
    let clangxx = which("clang++")?;
    let src = dir.join("g.cpp");
    let obj = dir.join("g.o");
    fs::write(&src, SRC).expect("write source");
    let ok = Command::new(clangxx)
        .args([
            "--target=x86_64-linux-gnu",
            "-fPIE",
            "-ffunction-sections",
            "-fdata-sections",
            "-c",
        ])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("clang++ runs")
        .success();
    ok.then_some(obj)
}

/// The crt and runtime pieces a C++ dynamic link needs.
struct Harness {
    crti: PathBuf,
    scrt1: PathBuf,
    crtn: PathBuf,
    libc: PathBuf,
    libstdcxx: PathBuf,
    libgcc_s: PathBuf,
    interp: Vec<u8>,
}

impl Harness {
    /// Resolves every piece, or `None` when the host cannot run these links.
    fn detect() -> Option<Self> {
        let prologue = crt_file("crti.o")?;
        let scrt1 = crt_file("Scrt1.o")?;
        let epilogue = crt_file("crtn.o")?;
        let libc = find_lib("libc.so.6")?;
        let libstdcxx = find_lib("libstdc++.so.6")?;
        let libgcc_s = crt_file("libgcc_s.so.1")?;
        let interp = interpreter()?;
        Some(Self {
            crti: prologue,
            scrt1,
            crtn: epilogue,
            libc,
            libstdcxx,
            libgcc_s,
            interp,
        })
    }
}

/// Links the fixture with or without `--gc-sections`.
fn link(h: &Harness, obj: &Path, out: &Path, gc: bool) {
    link_dyn_exec(
        &[
            obj.to_path_buf(),
            h.crti.clone(),
            h.scrt1.clone(),
            h.crtn.clone(),
            h.libc.clone(),
            h.libstdcxx.clone(),
            h.libgcc_s.clone(),
        ],
        out,
        b"main",
        &h.interp,
        gc,
        IcfMode::None,
        false,
    )
    .expect("the fixture links");
}
