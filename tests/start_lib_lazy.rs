//! `--start-lib`/`--end-lib` extract their objects on demand.
//!
//! The markers turn the objects between them into one unnamed static
//! library: a member is linked only when it resolves an undefined symbol,
//! exactly as an archive member is. xold accepted both markers as no-ops,
//! so every object in the group was linked eagerly -- the marker pair
//! asked for laziness and got the opposite, which silently doubles image
//! size for the driver flag (`gcc -static` on some toolchains, `rustc`
//! and `clang -fno--...` builds) that writes it.
//!
//! lld builds the same unnamed library and runs its lazy extraction over
//! it (`lld/ELF/Driver.cpp`); GNU `ld` documents the pair
//! as "like an archive without a file".
//!
//! The group is packed into a real `ar` image in memory, so the archive
//! pass, the extraction fixpoint and the COMDAT laziness all apply to it
//! unchanged. The test links a group of two objects, one referenced and
//! one not, through the built binary.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{which, xold_bin};

mod common;

/// The referenced member: defines `group_used`, which `main` calls.
const USED_SRC: &[u8] = b"int group_used(void) { return 5; }\n";

/// The unreferenced member: defines `group_idle`, which nothing names.
/// It carries its own data so its section survives any dead-code pass
/// that is not the extraction itself.
const IDLE_SRC: &[u8] =
    b"int group_idle = 42;\nint group_idle_fn(void) { return group_idle; }\n";

const MAIN_SRC: &[u8] =
    b"extern int group_used(void);\nint main(void) { return group_used(); }\n";

/// Inside the markers, only the member that resolves a reference is
/// linked: `group_idle` is gone.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_start_lib_group_extracts_on_demand() {
    let Some(dir) = workdir("lazy") else {
        return;
    };
    let Some(files) = compile(&dir) else {
        return;
    };
    let ok = link(&dir, &files, true);
    assert!(ok, "the fixture links");
    let image = read_output(&dir);
    let symtab = symtab_of(&image);
    assert!(
        !symtab.iter().any(|s| symtab.name(s) == b"group_idle"),
        "an unreferenced member of a --start-lib group is not linked"
    );
    assert!(
        symtab.iter().any(|s| symtab.name(s) == b"group_used"),
        "the member that resolves a reference is linked"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// The control: the same objects outside the markers are direct inputs,
/// so both are linked and the drop above is the laziness doing it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn without_the_markers_both_are_linked() {
    let Some(dir) = workdir("plain") else {
        return;
    };
    let Some(files) = compile(&dir) else {
        return;
    };
    let ok = link(&dir, &files, false);
    assert!(ok, "the fixture links");
    let image = read_output(&dir);
    let symtab = symtab_of(&image);
    assert!(
        symtab.iter().any(|s| symtab.name(s) == b"group_idle"),
        "as direct inputs both objects are linked"
    );
    assert!(
        symtab.iter().any(|s| symtab.name(s) == b"group_used"),
        "and the referenced one of course"
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping start-lib-lazy {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_startlib_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Compiles the three objects.
fn compile(dir: &Path) -> Option<Vec<PathBuf>> {
    let clang = which("clang")?;
    let mut objects = Vec::new();
    for (name, src) in [
        ("used.c", USED_SRC),
        ("idle.c", IDLE_SRC),
        ("main.c", MAIN_SRC),
    ] {
        let src_path = dir.join(name);
        let obj = dir.join(name.replace(".c", ".o"));
        fs::write(&src_path, src).ok()?;
        let built = Command::new(&clang)
            .args([
                "--target=x86_64-linux-gnu",
                "-ffunction-sections",
                "-fdata-sections",
                "-c",
            ])
            .arg(&src_path)
            .arg("-o")
            .arg(&obj)
            .status()
            .ok()?
            .success();
        if !built {
            eprintln!("skipping start-lib-lazy: clang cannot build {name}");
            return None;
        }
        objects.push(obj);
    }
    Some(objects)
}

/// Links `main.o` with the two members, inside or outside the markers.
fn link(dir: &Path, objects: &[PathBuf], markers: bool) -> bool {
    let out = dir.join("prog");
    let mut cmd = Command::new(xold_bin());
    if markers {
        cmd.arg("--start-lib");
    }
    cmd.arg(&objects[0]).arg(&objects[1]);
    if markers {
        cmd.arg("--end-lib");
    }
    cmd.arg(&objects[2])
        .arg("--entry")
        .arg("main")
        .arg("-o")
        .arg(&out);
    cmd.status().expect("xold must run").success()
}

// --- readers ---------------------------------------------------------------

/// The linked image.
fn read_output(dir: &Path) -> Vec<u8> {
    fs::read(dir.join("prog")).expect("read the image")
}

/// The output's `.symtab`.
fn symtab_of(image: &[u8]) -> xold::elf::SymbolTable<'_> {
    let obj = xold::elf::ObjectFile::parse(image).expect("valid ELF");
    obj.symbol_table().ok().flatten().expect(".symtab")
}
