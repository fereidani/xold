//! The COFF link defines `__ImageBase`.
//!
//! MSVC programs read their own load address as `&__ImageBase`: the
//! symbol is undefined in every input and the linker defines it to the
//! image base, the way lld's `addSynthetic` does with a null chunk
//! (`lld/COFF/Driver.cpp`). xold left it undefined, so any
//! object that referenced it -- which clang-msvc emits for
//! `&__ImageBase` -- failed the link outright.
//!
//! The fixture holds `void *g = &__ImageBase;`, whose initializer a
//! linker-written `ADDR64` relocation fills with the image base. The
//! test links it and checks `.data`'s first quadword equals the
//! optional header's `ImageBase`. Gated on a msvc-capable clang, like
//! the other COFF tests.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use xold::{coff::PeImage, input::Input};

/// The C source: one initialised global holding the image base.
const SRC: &[u8] =
    b"extern char __ImageBase;\nvoid *g = &__ImageBase;\nint main(void){ return 0; }\n";

/// Compiles `SRC` for the msvc target into `out`, or answers `false`.
fn compile_msvc(out: &Path) -> bool {
    let Some(clang) = which("clang") else {
        return false;
    };
    let src = std::env::temp_dir().join("xold_coff_ib_src.c");
    if fs::write(&src, SRC).is_err() {
        return false;
    }
    let ok = Command::new(clang)
        .args(["--target=x86_64-pc-windows-msvc", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(out)
        .stdin(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    let _ = fs::remove_file(&src);
    ok
}

/// A reference to `__ImageBase` links and resolves to the image base.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_image_base_symbol_resolves() {
    let dir = std::env::temp_dir()
        .join(format!("xold_coffib_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create the workdir");
    let obj = dir.join("ib.obj");
    if !compile_msvc(&obj) {
        eprintln!("skipping coff-image-base: no msvc target");
        let _ = fs::remove_dir_all(&dir);
        return;
    }
    let out: PathBuf = dir.join("ib.exe");
    let files = [Input::Path(&obj)];
    let res = xold::coff::link_coff(&files, &out, b"main", false);
    assert!(res.is_ok(), "&__ImageBase must link: {:?}", res.err());
    let bytes = fs::read(&out).expect("read the image");
    let img = PeImage::parse(&bytes).expect("parse the image");
    let data = img
        .sections()
        .into_iter()
        .find(|s| s.name.starts_with(b".data"))
        .expect("the fixture initialises .data");
    let slot = &data.data[..8];
    let stored = u64::from_le_bytes(slot.try_into().expect("a quadword"));
    assert_eq!(
        stored,
        img.image_base(),
        "the initialiser must hold the image base"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `PATH` search for a tool, mirroring the common helper.
fn which(tool: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(tool))
        .find(|p| p.exists())
}
