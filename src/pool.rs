//! How many threads one link runs on.
//!
//! Every parallel pass fans out to rayon's global pool, which is one worker
//! per core. On a small link that pool costs far more than it earns: the
//! workers spin in the work-stealing loop with nothing to steal, which is
//! kernel time spent against no user work. A compiler driver's link of one
//! translation unit runs about 1.45 times slower in wall clock on the global
//! pool than on one thread, for 2.5 times the CPU.
//!
//! So a link sizes its own pool once, up front, from the workload it can see
//! after the inputs are open. Below the threshold it runs on a private
//! one-thread pool and the global pool is never built at all; above it nothing
//! is installed and the link uses the global pool exactly as before. Both
//! paths run the same code over the same data, so the image is the same bytes
//! either way; `tests/pool_threshold.rs` pins that.

use std::{
    env,
    ops::Range,
    sync::atomic::{AtomicBool, Ordering},
};

use rayon::ThreadPoolBuilder;

use crate::elf::constants::ET_DYN;

/// The variable rayon reads for the size of its global pool.
///
/// A caller who sets it has already answered this question, so the rule below
/// applies only when it is absent.
const THREAD_VAR: &str = "RAYON_NUM_THREADS";

/// What one input file is worth, in bytes of content, when estimating work.
///
/// A link's cost has a per-byte part (reading sections, copying them,
/// applying relocations) and a per-file part (opening, parsing the header,
/// walking the section table). Fitted timings put the per-file part at about
/// 26 us against about 3.8 us per KiB, so one input is worth roughly 6 KiB.
const FILE_WORK: u64 = 6 * 1024;

/// The estimated work at which the global pool starts paying for itself.
///
/// Below this the fan-out loses wall clock outright; above it the win grows
/// with the workload. Hosts disagree about where the crossover lies, so this
/// is the midpoint of the interval measurement leaves open rather than any
/// one machine's crossover.
const WORK_THRESHOLD: u64 = 768 * 1024;

/// The half-word an ELF header keeps `e_type` in. xold reads little-endian
/// ELF only, so it can be taken straight out of the mapped bytes without
/// parsing the file.
const E_TYPE: Range<usize> = 16..18;

/// The bytes of `input` that become link work.
///
/// A shared object contributes none: it is a `DT_NEEDED` dependency, so the
/// linker reads its dynamic symbol tables and never touches its sections. The
/// distinction is load-bearing, not a refinement -- a driver's link names a
/// 2.4 MiB `libc.so.6` against about 7 KiB of real input, so counting it puts
/// the very link this threshold exists for on the wrong side.
fn content_bytes(input: &[u8]) -> u64 {
    let shared = input.starts_with(b"\x7fELF")
        && input
            .get(E_TYPE)
            .and_then(|b| <[u8; 2]>::try_from(b).ok())
            .is_some_and(|b| u16::from_le_bytes(b) == ET_DYN);
    if shared {
        return 0;
    }
    u64::try_from(input.len()).unwrap_or(u64::MAX)
}

/// Whether `inputs` are worth fanning out over.
///
/// The estimate is the total [`content_bytes`] plus [`FILE_WORK`] per input,
/// compared against [`WORK_THRESHOLD`]. Both terms are needed; neither
/// predicts the crossover on its own. A count gate alone is wrong for a
/// single `ld -r` object carrying eighteen thousand sections, which the pool
/// links faster with no cross-file fan-out at all, since the parallel passes
/// iterate over input sections rather than files. A byte gate alone is wrong
/// for an object that is mostly section header table, which is cheap to walk
/// and prefers one thread despite outweighing workloads that must fan out.
///
/// Counting an archive's whole size is deliberate even though a link pulls
/// only some members: building the archive index and resolving against it is
/// itself work that fans out.
///
/// # Re-fitting the constants
///
/// The crossover moves with core count, memory bandwidth and allocator, and
/// two hosts already disagree about where it is by a third. Re-measure every
/// input shape rather than one: a sweep over object count alone finds a
/// threshold wrong by a factor of three for the others. The requirement is
/// that no link runs slower than it did with the pool always on, so compare
/// the rule-chosen wall clock against a forced multi-thread run at every
/// fixture size, and read CPU alongside it -- a row that is a wash in wall
/// clock but far worse in CPU is still worth taking serially, because under
/// `make -j` that CPU comes out of a concurrent compile.
pub fn fans_out(inputs: &[&[u8]]) -> bool {
    work_estimate(inputs) >= WORK_THRESHOLD
}

/// The estimated work in `inputs`: their [`content_bytes`] plus
/// [`FILE_WORK`] each. Two decisions read it, so it is computed once.
fn work_estimate(inputs: &[&[u8]]) -> u64 {
    let bytes = inputs
        .iter()
        .map(|b| content_bytes(b))
        .fold(0u64, u64::saturating_add);
    let files = u64::try_from(inputs.len()).unwrap_or(u64::MAX);
    bytes.saturating_add(files.saturating_mul(FILE_WORK))
}

/// The estimated work above which transparent huge pages stop paying.
///
/// mimalloc reserves a 1 GiB arena and marks all of it `MADV_HUGEPAGE`, so
/// every anonymous allocation a link makes faults as a 2 MiB page. Two costs
/// follow. A huge page is zeroed in full on first touch, and a link writes
/// its arena sparsely and once -- it streams through its inputs rather than
/// revisiting a working set -- so the TLB reuse that pays for huge pages
/// elsewhere never accrues. And under Linux's default `defrag=madvise` a
/// fault with no free 2 MiB block compacts *synchronously* in the faulting
/// thread, which is a stall no other thread overlaps. On the host this was
/// fitted on, 92% of the kernel's compaction attempts were already failing.
///
/// # What this was fitted to
///
/// 600 independent objects of about 0.77 MB each, linked as a dynamic
/// executable in prefixes; medians of 11 interleaved pairs per row, with
/// memory compacted before each row so no row inherits the previous one's
/// fragmentation. The ratio is huge-pages-off over huge-pages-on, so below
/// 1.0 turning them off wins.
///
/// | objects | estimate | ratio | disabled |
/// |---|---|---|---|
/// | 120 | 96 MB | 1.185 | no |
/// | 200 | 160 MB | 0.702 | no |
/// | 280 | 224 MB | 0.723 | no |
/// | 360 | 288 MB | 0.705 | yes |
/// | 440 | 353 MB | 0.680 | yes |
/// | 520 | 417 MB | 0.650 | yes |
/// | 600 | 481 MB | 0.602 | yes |
///
/// The real corpora agree at both ends: the 2864-object LLVM link (389 MB
/// estimated) measured 0.68x to 0.79x across four sessions and never once
/// preferred huge pages, while a 121-object C corpus (51 MB) measured 1.115x
/// and the same corpus against `libc.a` (81 MB) 1.091x.
///
/// # Why not at the crossover
///
/// The sweep puts the crossover between 96 MB and 160 MB, but two real
/// corpora sitting at 229 MB and 238 MB -- the same C++ objects linked shared
/// and dynamic -- changed sign between sessions, measuring 1.05x once and
/// 0.86x another time. Most of their bulk is DWARF, which is copied from one
/// mapping to another rather than held on the heap, so the byte estimate
/// overstates the anonymous footprint that actually faults. The threshold
/// therefore sits above them rather than at the crossover: every shape at or
/// above 256 MB won in every session it was measured in, and no shape below
/// it is put at risk. The cost of the choice is the 160 MB and 224 MB rows,
/// which give up a measured 0.70x.
///
/// # If this is re-fitted
///
/// The crossover moves with how fragmented the host's memory is, which is
/// what makes it worth a threshold rather than an unconditional call: on a
/// long-lived, fragmented machine turning huge pages off won every row of
/// the sweep, including the ones that prefer them here. Compact memory
/// (`/proc/sys/vm/compact_memory`) before each row, or the measurement reads
/// the previous row's fragmentation rather than the workload.
const HUGE_PAGE_THRESHOLD: u64 = 256 * 1024 * 1024;

/// Whether the process has allowed its huge-page policy to be changed.
///
/// Off unless [`allow_huge_page_tuning`] is called. The policy is
/// process-wide and permanent, which is the application's call to make and
/// not a library's: a host that links once and then runs for hours would
/// otherwise lose huge pages everywhere on the strength of that one link.
static TUNE_HUGE_PAGES: AtomicBool = AtomicBool::new(false);

/// Lets a link turn transparent huge pages off when the workload is large
/// enough to be hurt by them. See [`HUGE_PAGE_THRESHOLD`].
///
/// Intended for the `xold` binary, whose process exists only to link.
pub fn allow_huge_page_tuning() {
    TUNE_HUGE_PAGES.store(true, Ordering::Relaxed);
}

/// Opts the process out of transparent huge pages, if it allowed it.
#[cfg(target_os = "linux")]
fn disable_huge_pages() {
    if !TUNE_HUGE_PAGES.load(Ordering::Relaxed) {
        return;
    }
    // SAFETY: `prctl` is variadic; this option takes its four arguments by
    // value and reads no memory through them, so the call borrows nothing.
    // A kernel that does not know the option answers -1/EINVAL, which needs
    // no handling: huge pages stay on and the link is slower but correct.
    let rc = unsafe {
        libc::prctl(libc::PR_SET_THP_DISABLE, 1_u64, 0_u64, 0_u64, 0_u64)
    };
    debug_assert!(rc == 0 || rc == -1, "prctl answers 0 or -1");
    let _ = rc;
}

/// Runs `job` on a thread pool sized for `inputs`.
///
/// Three cases leave the ambient pool alone: an explicit [`THREAD_VAR`], a
/// pool the caller installed around this call, and a workload [`fans_out`]
/// judges big enough. Only a small link in an unconfigured process gets a
/// private one-thread pool.
pub(crate) fn run<T: Send>(
    inputs: &[&[u8]],
    job: impl FnOnce() -> T + Send,
) -> T {
    let work = work_estimate(inputs);
    // Taken before any parallel pass starts, so the arena is still mostly
    // untouched and the policy applies to the faults that matter.
    #[cfg(target_os = "linux")]
    if work >= HUGE_PAGE_THRESHOLD {
        disable_huge_pages();
    }
    if env::var_os(THREAD_VAR).is_some()
        || rayon::current_thread_index().is_some()
        || work >= WORK_THRESHOLD
    {
        return job();
    }
    match ThreadPoolBuilder::new().num_threads(1).build() {
        Ok(pool) => pool.install(job),
        // A pool that will not build is no reason to fail a link that would
        // otherwise succeed: fall back to the ambient one.
        Err(_) => job(),
    }
}
