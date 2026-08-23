//! `.rsrc` resource sections become their own output section, published
//! through the RESOURCE data directory.
//!
//! cvtres emits a compiled resource script as `.rsrc$01` (the directory
//! tree) and `.rsrc$02` (the strings and data) input sections. xold
//! classified them as plain read-only data, so the tree's bytes landed
//! inside `.rdata` and no `IMAGE_DIRECTORY_ENTRY_RESOURCE` was ever
//! written: the loader saw an image with no resources, silently.
//!
//! The fixtures synthesise the cvtres shape with
//! `__attribute__((section))` -- no Windows resource compiler is needed,
//! the assertions are structural, not semantic. One object's tree is
//! placed whole into a `.rsrc` output section whose RVA and size the
//! RESOURCE directory publishes. A second object's tree is refused
//! rather than concatenated: each tree's internal offsets are relative
//! to its own base, so concatenating would corrupt both.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use xold::{coff::PeImage, input::Input};

/// The C source: a resource tree-shaped `.rsrc$01` section (header, one
/// entry, one data entry -- 34 bytes) and a `.rsrc$02` string member.
const SRC: &[u8] = concat!(
    "__attribute__((section(\".rsrc$01\")))\n",
    "const unsigned char tree[] = {\n",
    "  0xab,0,0,0, 0,0,0,0, 0,0, 0,0, 0,1, 0,0,\n",
    "  0,0,0,0, 0x80,0,0,0,\n",
    "  0,0,0,0, 4,0,0,0, 0,0,0,0, 0,0,0,0 };\n",
    "__attribute__((section(\".rsrc$02\")))\n",
    "const unsigned char strings[] = { 'h','i',0,0 };\n",
    "int main(void){ return 0; }\n",
)
.as_bytes();

/// Compiles `src` for the msvc target into `out`, or answers `false`. The
/// source file is named per `tag`: the two tests run in parallel and would
/// otherwise delete each other's source mid-compile.
fn compile_msvc(tag: &str, src: &[u8], out: &Path) -> bool {
    let Some(clang) = which("clang") else {
        return false;
    };
    let file = std::env::temp_dir().join(format!("xold_coff_rsrc_{tag}.c"));
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

/// The resource directory entry's index in the data directory.
const RESOURCE: usize = 2;

/// The second object's variant: distinct symbols, the same resource shape.
/// Without distinct names the duplicate-symbol check would fire first and
/// mask which refusal the test is about.
const SRC_B: &[u8] = concat!(
    "__attribute__((section(\".rsrc$01\")))\n",
    "const unsigned char tree_b[] = {\n",
    "  0xcd,0,0,0, 0,0,0,0, 0,0, 0,0, 0,1, 0,0,\n",
    "  0,0,0,0, 0x80,0,0,0,\n",
    "  0,0,0,0, 4,0,0,0, 0,0,0,0, 0,0,0,0 };\n",
    "__attribute__((section(\".rsrc$02\")))\n",
    "const unsigned char strings_b[] = { 'o','k',0,0 };\n",
    "int main_b(void){ return 0; }\n",
)
.as_bytes();

/// Links one `.rsrc`-carrying object and checks the image publishes it.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn resource_section_gets_its_own_section_and_directory() {
    let (work, obj) = workdir("one");
    if !compile_msvc("one", SRC, &obj) {
        eprintln!("skipping coff-rsrc: no msvc target");
        return;
    }
    let out = work.join("one.exe");
    let files = [Input::Path(&obj)];
    let res = xold::coff::link_coff(&files, &out, b"main", false);
    assert!(res.is_ok(), "the link must succeed: {:?}", res.err());
    let bytes = fs::read(&out).expect("read the image");
    let img = PeImage::parse(&bytes).expect("parse the image");
    let rsrc = img
        .sections()
        .into_iter()
        .find(|s| s.name.starts_with(b".rsrc"))
        .expect(".rsrc must be its own output section");
    // The tree sorts first among the resource members, so the output
    // section begins with it: the first byte is the marker byte.
    assert_eq!(&rsrc.data[..1], &[0xab], "the tree opens the section");
    let dir = img.data_directory(RESOURCE).expect("a RESOURCE directory");
    assert_eq!(dir.0, rsrc.virtual_address, "the directory names .rsrc");
    assert_eq!(dir.1, rsrc.virtual_size, "the directory sizes .rsrc");
    let _ = fs::remove_dir_all(&work);
}

/// Two objects each carrying a resource tree are refused: concatenating
/// them corrupts both trees.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_second_resource_tree_is_refused() {
    let (dir, obj) = workdir("two");
    let obj_b = dir.join("two_b.obj");
    if !compile_msvc("one", SRC, &obj) || !compile_msvc("two", SRC_B, &obj_b) {
        eprintln!("skipping coff-rsrc: no msvc target");
        return;
    }
    let out = dir.join("two.exe");
    let files = [Input::Path(&obj), Input::Path(&obj_b)];
    let res = xold::coff::link_coff(&files, &out, b"main", false);
    let Err(err) = res else {
        let _ = fs::remove_dir_all(&dir);
        panic!("two .rsrc trees must be refused, not concatenated");
    };
    let text = format!("{err}");
    assert!(
        text.contains(".rsrc"),
        "the refusal must name .rsrc: {text}"
    );
    assert!(!out.exists(), "nothing may be published on refusal");
    let _ = fs::remove_dir_all(&dir);
}

/// A fresh per-test directory and the object path inside it.
fn workdir(tag: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir()
        .join(format!("xold_coffrsrc_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create the workdir");
    let obj = dir.join(format!("{tag}.obj"));
    (dir, obj)
}

/// `PATH` search for a tool, mirroring the common helper.
fn which(tool: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(tool))
        .find(|p| p.exists())
}
