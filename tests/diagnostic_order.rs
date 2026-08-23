//! Which bad input gets reported does not depend on the thread schedule.
//!
//! Three per-input passes ran in parallel and reduced with
//! `collect::<Result<_>>`, which surfaces whichever failure the runtime
//! noticed first. A link with two malformed inputs therefore named one of them
//! on one run and the other on the next.
//!
//! Output bytes were never at stake -- a failed link writes nothing -- but a
//! diagnostic that changes between identical runs is the same defect as an
//! image that does, and it is the one a person actually reads. The reduction
//! is ordered now; the passes stay parallel.
//!
//! The cost is that a failing link no longer short-circuits, so every input is
//! parsed before the first error is reported. That is work spent only on links
//! that were going to fail.
//!
//! Gated on `clang`; if it is missing the tests print a note and return.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{which, xold_bin};

mod common;

const SRC: &[u8] = b"void _start(void) { }\n";

/// How many times to link the same command line. Each is a separate process,
/// so a scheduling difference has every chance to show; without the fix the
/// answer flips on roughly half of them.
const RUNS: usize = 30;

/// Junk inputs placed after the slow-failing one. The race needs work to
/// interleave: with a single pair the window is too narrow to observe, and
/// with a dozen the wrong answer wins about half the time.
const JUNK_COUNT: usize = 12;

/// The reported failure is the same every run, whatever the thread count.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_reported_failure_is_stable_across_runs() {
    let Some(dir) = workdir("stable") else {
        return;
    };
    let Some((exe, junk)) = inputs(&dir) else {
        return;
    };
    // The executable leads and fails late -- its header parses before the
    // `ET_REL` test rejects it -- while the junk inputs after it fail on the
    // magic. The slow failure is the one to report, and the fast ones must
    // never win the race.
    let mut paths: Vec<&Path> = vec![&exe];
    paths.extend(junk.iter().map(PathBuf::as_path));
    let seen = diagnostics(&dir, &paths);
    assert_eq!(
        seen.len(),
        1,
        "the same command line must give the same diagnostic every time; got \
         {seen:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// And it is the first bad input in command-line order, not merely a stable
/// arbitrary one.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn the_first_bad_input_is_the_one_reported() {
    let Some(dir) = workdir("first") else {
        return;
    };
    let Some((exe, junk)) = inputs(&dir) else {
        return;
    };
    let first_junk = junk.first().expect("a junk input");
    let junk_first = diagnostics(&dir, &[first_junk, &exe]);
    let mut exe_leading: Vec<&Path> = vec![&exe];
    exe_leading.extend(junk.iter().map(PathBuf::as_path));
    let exe_first = diagnostics(&dir, &exe_leading);
    assert_eq!(junk_first.len(), 1, "one answer per order: {junk_first:?}");
    assert_eq!(exe_first.len(), 1, "one answer per order: {exe_first:?}");
    assert!(
        junk_first[0].contains("invalid ELF header"),
        "the junk input leads, so its failure is the one to report; got {:?}",
        junk_first[0]
    );
    assert!(
        exe_first[0].contains("relocatable"),
        "and with the executable leading, its failure is; got {:?}",
        exe_first[0]
    );
    let _ = fs::remove_dir_all(&dir);
}

// --- fixtures --------------------------------------------------------------

/// Creates a fresh per-test working directory, or `None` (after printing a
/// note) when the host cannot build the inputs.
fn workdir(prefix: &str) -> Option<PathBuf> {
    if which("clang").is_none() {
        eprintln!("skipping diagnostic-order {prefix}: clang unavailable");
        return None;
    }
    let dir = std::env::temp_dir()
        .join(format!("xold_diagorder_{prefix}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Two inputs that fail for different reasons, so the message says which one
/// was reported: a finished executable, and bytes that only claim to be ELF.
fn inputs(dir: &Path) -> Option<(PathBuf, Vec<PathBuf>)> {
    let clang = which("clang")?;
    let src = dir.join("g.c");
    let obj = dir.join("g.o");
    fs::write(&src, SRC).ok()?;
    let built = Command::new(clang)
        .args(["--target=x86_64-linux-gnu", "-fno-pic", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping diagnostic-order: clang cannot build the fixture");
        return None;
    }
    let exe = dir.join("exe");
    let ok = Command::new(xold_bin())
        .arg("-o")
        .arg(&exe)
        .arg(&obj)
        .arg("--entry")
        .arg("_start")
        .status()
        .ok()?
        .success();
    assert!(ok, "the fixture executable must link");

    // An ELF magic followed by nothing that parses.
    let mut junk = Vec::with_capacity(JUNK_COUNT);
    for i in 0..JUNK_COUNT {
        let path = dir.join(format!("junk{i}.o"));
        let mut bytes = b"\x7fELF".to_vec();
        bytes.extend(std::iter::repeat_n(0x5au8, 300));
        fs::write(&path, bytes).ok()?;
        junk.push(path);
    }
    Some((exe, junk))
}

/// Runs the linker over `paths` at a spread of thread counts and returns the
/// distinct first lines it printed.
fn diagnostics(dir: &Path, paths: &[&Path]) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for _ in 0..RUNS {
        let out = Command::new(xold_bin())
            .env("RAYON_NUM_THREADS", "8")
            .arg("-o")
            .arg(dir.join("out"))
            .args(paths)
            .arg("--entry")
            .arg("_start")
            .output()
            .expect("xold must run");
        assert!(!out.status.success(), "the link must fail");
        let line = String::from_utf8_lossy(&out.stderr)
            .lines()
            .next()
            .unwrap_or_default()
            .to_string();
        if !seen.contains(&line) {
            seen.push(line);
        }
    }
    seen
}
