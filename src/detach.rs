//! Returning to the caller before teardown: the fork the CLI runs under.
//!
//! Unmapping several hundred megabytes of input and freeing a link's heap
//! takes tens of milliseconds, all of it after the image is already renamed
//! into place. A build system waiting on the linker is waiting for nothing
//! during that time. mold and wild both answer this by forking at startup:
//! the parent waits, the child links, and the moment the image is published
//! the child tells the parent to exit. The child then finishes its teardown
//! as an orphan, off the caller's clock.
//!
//! This module is that answer, and the only place the linker calls `fork`.
//! [`fork_child`] runs before any thread is spawned -- forking a threaded
//! process would hand the child a single thread and whatever locks the others
//! held -- and the caller-visible contract is exactly the sequential one:
//! the exit code is the link's, stderr arrives on the caller's stderr, and
//! the output file is complete and renamed into place before the parent
//! returns success. `--no-fork` skips all of it, which is also the right
//! mode for measuring the linker's full cost rather than its latency.
//!
//! The library never arms this on its own: [`published`] is a no-op unless
//! [`fork_child`] stored the pipe, so embedders and tests see plain
//! function-call behaviour.

use std::sync::atomic::{AtomicI32, Ordering};

/// The write end of the pipe the forked child reports through, or -1.
///
/// An `AtomicI32` rather than a `OnceLock`: the value is claimed back (and
/// reset to -1) by the one [`published`] call, so a second publish -- or one
/// in a process that never forked -- does nothing.
static PUBLISH_FD: AtomicI32 = AtomicI32::new(-1);

/// Forks; the parent waits for the link's outcome, the child returns and
/// links.
///
/// In the child this returns and the caller carries on into the link, with
/// the pipe stored for [`published`]. In the parent this never returns: it
/// blocks until the child either reports the image published (exit 0, while
/// the child tears down unobserved) or exits without reporting, in which
/// case the child's own exit status is repeated. If the pipe or the fork
/// cannot be had, the caller simply proceeds unforked; detaching is a
/// latency improvement, never a requirement.
pub fn fork_child() {
    let mut fds = [-1i32; 2];
    // SAFETY: `fds` is two writable ints, which is what `pipe` fills in.
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        return;
    }
    let [rd, wr] = fds;
    // SAFETY: the process is single-threaded here, so the child inherits a
    // coherent copy of the whole address space.
    match unsafe { libc::fork() } {
        -1 => {
            // SAFETY: both descriptors came from the `pipe` above.
            unsafe {
                libc::close(rd);
                libc::close(wr);
            }
        }
        0 => {
            // SAFETY: the read end belongs to the parent.
            unsafe { libc::close(rd) };
            PUBLISH_FD.store(wr, Ordering::Release);
        }
        child => {
            // SAFETY: the write end belongs to the child.
            unsafe { libc::close(wr) };
            wait_for_child(rd, child);
        }
    }
}

/// Reports the output image renamed into place, releasing the waiting
/// parent. A no-op in a process [`fork_child`] did not fork.
pub fn published() {
    let fd = PUBLISH_FD.swap(-1, Ordering::AcqRel);
    if fd < 0 {
        return;
    }
    let byte = [0u8];
    // SAFETY: `fd` is the pipe write end stored by `fork_child`, claimed
    // back atomically above so it is written and closed exactly once.
    unsafe {
        libc::write(fd, byte.as_ptr().cast(), 1);
        libc::close(fd);
    }
}

/// The parent's whole life: wait on the pipe, then exit.
///
/// One byte means the image is published and the exit code is 0. End of
/// file means the child exited without publishing -- a failed link, or a
/// crash -- and its status is collected and repeated, including the
/// conventional 128-plus-signal spelling for a killed child.
fn wait_for_child(rd: i32, child: libc::pid_t) -> ! {
    let mut byte = [0u8];
    loop {
        // SAFETY: `rd` is the open read end of the pipe made above.
        let got = unsafe { libc::read(rd, byte.as_mut_ptr().cast(), 1) };
        match got {
            1.. => std::process::exit(0),
            0 => break,
            _ if last_errno() == libc::EINTR => {}
            _ => break,
        }
    }
    let mut status = 0i32;
    loop {
        // SAFETY: `child` is this process's own forked child, and `status`
        // is a writable int.
        let got = unsafe { libc::waitpid(child, &raw mut status, 0) };
        if got == child {
            break;
        }
        if last_errno() != libc::EINTR {
            std::process::exit(1);
        }
    }
    if libc::WIFEXITED(status) {
        std::process::exit(libc::WEXITSTATUS(status));
    }
    if libc::WIFSIGNALED(status) {
        std::process::exit(128 + libc::WTERMSIG(status));
    }
    std::process::exit(1);
}

/// The calling thread's `errno`.
fn last_errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}
