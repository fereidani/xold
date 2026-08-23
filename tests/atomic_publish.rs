//! A failed output write leaves the previous image standing.
//!
//! The Mach-O and COFF writers used to write the destination file
//! directly. `std::fs::write` is not atomic: a write that runs out of
//! room lands its first bytes over the old output and then fails, so a
//! full disk replaced a working executable with a truncated one wearing
//! its name -- and a build watcher meanwhile saw the image appear long
//! before it was whole. The ELF writer already published through a
//! temporary sibling and a rename; the byte-image writers now do the
//! same, which also means a failure cleans up its own temporary.
//!
//! The test links a real darwin object under a file-size rlimit smaller
//! than the image, so the write fails part way through, and then checks
//! the two halves of the contract: the pre-existing output still holds
//! exactly its old bytes, and no temporary is left beside it. A control
//! link without the cap first proves the image really is larger than
//! the cap, so the failure under test cannot silently stop happening as
//! output sizes drift. Gated on a darwin-capable clang, like the other
//! Mach-O tests.

use std::{
    fs,
    path::PathBuf,
    process::{Command, Stdio},
};

use xold::{input::Input, macho::link_macho};

/// The bytes the previous output holds before the failing link.
const SENTINEL: &[u8] = b"the previous image, still complete\n";

/// The file-size cap forced on the failing link. The control link
/// asserts the real image is larger, so the cap keeps biting.
const CAP: u64 = 512;

/// Compiles `int main` for darwin into `out`, or answers `false`.
fn compile_darwin_main(out: &std::path::Path) -> bool {
    let Some(clang) = which("clang") else {
        return false;
    };
    let src = std::env::temp_dir().join("xold_atomic_pub_src.c");
    if fs::write(&src, b"int main(void){ return 0; }\n").is_err() {
        return false;
    }
    let ok = Command::new(clang)
        .args(["--target=x86_64-apple-darwin", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(out)
        .stdin(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    let _ = fs::remove_file(&src);
    ok
}

/// The first `size_t` of the `rlimit` pair.
fn rlimit_fsize() -> Option<(u64, u64)> {
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `lim` is a writable rlimit, which is what getrlimit fills.
    let rc = unsafe { libc::getrlimit(libc::RLIMIT_FSIZE, &raw mut lim) };
    (rc == 0).then_some((lim.rlim_cur, lim.rlim_max))
}

/// Caps the process's file-size limit, returning the previous pair.
fn cap_file_size(cur: u64) -> Option<(u64, u64)> {
    let was = rlimit_fsize()?;
    let lim = libc::rlimit {
        rlim_cur: cur,
        rlim_max: was.1.max(cur),
    };
    // SAFETY: `lim` holds a valid soft/hard pair derived from the read.
    let rc = unsafe { libc::setrlimit(libc::RLIMIT_FSIZE, &raw const lim) };
    (rc == 0).then_some(was)
}

/// Restores a previously read limit pair.
fn restore_file_size(was: (u64, u64)) {
    let lim = libc::rlimit {
        rlim_cur: was.0,
        rlim_max: was.1,
    };
    // SAFETY: the pair came from getrlimit in this process.
    unsafe {
        libc::setrlimit(libc::RLIMIT_FSIZE, &raw const lim);
    }
}

/// A write that fails part way leaves the old output and no temporary.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_failed_write_leaves_the_previous_output_standing() {
    let dir = std::env::temp_dir()
        .join(format!("xold_atomicpub_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create the workdir");
    let obj = dir.join("main.o");
    if !compile_darwin_main(&obj) {
        eprintln!("skipping atomic-publish: no darwin target");
        let _ = fs::remove_dir_all(&dir);
        return;
    }
    let out: PathBuf = dir.join("prog");
    fs::write(&out, SENTINEL).expect("write the previous output");
    let files = [Input::Path(&obj)];
    // Premise: an uncapped control link must produce an image larger
    // than the cap, or the capped link below would quietly succeed and
    // the test would stop testing anything.
    let control = dir.join("prog_control");
    link_macho(&files, &control, b"_main")
        .expect("the uncapped control link succeeds");
    let image_len = fs::metadata(&control)
        .expect("stat the control image")
        .len();
    assert!(
        image_len > CAP,
        "the {image_len}-byte image must exceed the {CAP}-byte cap"
    );
    // A partial write raises SIGXFSZ by default; ignoring it turns the
    // signal into an EFBIG return, which is the failure under test.
    // SAFETY: SIG_IGN for SIGXFSZ is valid in this single-threaded test.
    unsafe {
        libc::signal(libc::SIGXFSZ, libc::SIG_IGN);
    }
    let was = cap_file_size(CAP);
    let res = link_macho(&files, &out, b"_main");
    if let Some(was) = was {
        restore_file_size(was);
    }
    // SAFETY: restoring the default disposition.
    unsafe {
        libc::signal(libc::SIGXFSZ, libc::SIG_DFL);
    }
    assert!(
        res.is_err(),
        "a {CAP}-byte cap must fail the {image_len}-byte write"
    );
    let after = fs::read(&out).expect("the previous output is readable");
    assert_eq!(
        after, SENTINEL,
        "the failed write must not touch the previous image"
    );
    let leftovers: Vec<_> = fs::read_dir(&dir)
        .expect("read the workdir")
        .filter_map(std::result::Result::ok)
        .map(|e| e.file_name())
        .filter(|n| n.to_string_lossy().contains(".xold-"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "the failed write must clean up its temporary: {leftovers:?}"
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
