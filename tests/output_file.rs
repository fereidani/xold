//! How the linked image is published.
//!
//! The image is built under a temporary name and renamed into place. That is
//! not only faster; it is what makes two ordinary situations work at all.
//! Writing over a file that is currently executing fails with `ETXTBSY`, and
//! so does executing a file that some process still holds open for writing.
//! Renaming touches neither: the destination never carries a write reference,
//! and anything still running the previous image keeps its own inode.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::Duration,
};

use common::which;
use xold::{icf::IcfMode, linker::link_to};

mod common;

fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("xold_outfile_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// A program that sleeps briefly and exits 0, so the test can relink it while
/// it is still running.
const SRC: &str = "
static long nsec(void) { return 0; }
int main(void) {
    /* Spin long enough to still be running when the relink happens. */
    volatile long i = 0;
    while (i < 60000000L) { i = i + 1 + nsec(); }
    return 0;
}
";

/// Compiles and links the program with xold, returning its path.
fn build(dir: &Path, out: &str) -> Option<PathBuf> {
    let clang = which("clang")?;
    let c = dir.join("prog.c");
    fs::write(&c, SRC).ok()?;
    let obj = dir.join("prog.o");
    if !Command::new(&clang)
        .args(["-c", "-O0", "-fno-pic", "-o"])
        .arg(&obj)
        .arg(&c)
        .status()
        .ok()?
        .success()
    {
        return None;
    }
    let s = dir.join("start.S");
    fs::write(
        &s,
        "    .text\n    .globl _start\n_start:\n    call main\n\
         movl %eax, %edi\n    movl $60, %eax\n    syscall\n",
    )
    .ok()?;
    let start = dir.join("start.o");
    if !Command::new(&clang)
        .args(["-c", "-o"])
        .arg(&start)
        .arg(&s)
        .status()
        .ok()?
        .success()
    {
        return None;
    }
    let prog = dir.join(out);
    link_to(&[obj, start], &prog, b"_start", false, IcfMode::None, false)
        .expect("xold links the program");
    Some(prog)
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn relinking_a_running_program_succeeds() {
    let dir = workdir("running");
    let Some(prog) = build(&dir, "prog") else {
        eprintln!("skipping: host clang unavailable");
        return;
    };
    // Start it, and relink over it while it is still executing. Writing to the
    // destination in place would fail here with ETXTBSY.
    let mut child = Command::new(&prog).spawn().expect("program starts");
    thread::sleep(Duration::from_millis(20));
    let obj = dir.join("prog.o");
    let start = dir.join("start.o");
    link_to(&[obj, start], &prog, b"_start", false, IcfMode::None, false)
        .expect("relinking over a running program must succeed");
    let status = child.wait().expect("the running program finishes");
    assert_eq!(
        status.code(),
        Some(0),
        "the already-running image must keep running unaffected"
    );
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_failed_link_leaves_no_temporary_behind() {
    let dir = workdir("cleanup");
    if build(&dir, "prog").is_none() {
        eprintln!("skipping: host clang unavailable");
        return;
    }
    // A link into a directory that does not exist fails while the temporary
    // would be open; nothing should be left in the destination directory.
    let missing = dir.join("no_such_dir").join("out");
    let obj = dir.join("prog.o");
    let start = dir.join("start.o");
    let _ = link_to(
        &[obj, start],
        &missing,
        b"_start",
        false,
        IcfMode::None,
        false,
    );
    let strays: Vec<_> = fs::read_dir(&dir)
        .expect("work directory is readable")
        .filter_map(|e| e.ok().map(|e| e.file_name()))
        .filter(|n| n.to_string_lossy().contains(".xold-"))
        .collect();
    assert_eq!(strays, [""; 0], "left temporaries behind");
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_published_image_is_immediately_executable() {
    let dir = workdir("exec");
    let Some(prog) = build(&dir, "prog") else {
        eprintln!("skipping: host clang unavailable");
        return;
    };
    // No write reference may outlive the link: running the result must work
    // the instant it returns, with no retry.
    for _ in 0..20 {
        let obj = dir.join("prog.o");
        let start = dir.join("start.o");
        link_to(&[obj, start], &prog, b"_start", false, IcfMode::None, false)
            .expect("relink succeeds");
        let status = Command::new(&prog).status().expect("image runs at once");
        assert_eq!(status.code(), Some(0));
    }
}
