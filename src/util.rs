//! Small helpers shared by the format backends and the layout passes.
//!
//! Every item here was duplicated across two or more modules before it landed
//! in this one. Nothing format-specific belongs here: these are the byte,
//! string-table and alignment primitives the ELF, Mach-O and COFF paths all
//! reach for.

use std::{
    path::Path,
    sync::{
        OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use bytemuck::Pod;

use crate::{
    error::{Error, Result},
    input::{Input, InputBytes},
};

/// An input list this long is opened in parallel; a shorter one is opened
/// on the calling thread.
///
/// Opening is dominated by kernel path resolution, which scales across
/// threads (the dentry cache is read under RCU), and a large link opens
/// thousands of files before any other work can start. Below the threshold
/// the fan-out costs more than it earns -- the same trade [`crate::pool`]
/// measures for the link passes -- and this runs before the pool there is
/// sized, so a small link must not be the thing that spins the global pool
/// up. At roughly 30us per open, a hundred files are three milliseconds:
/// the largest list still opened serially costs less than the noise floor
/// of the whole link.
const PARALLEL_OPEN: usize = 100;

/// Makes every input readable, in order: a path is mapped, caller-supplied
/// bytes are borrowed or realigned. See [`Input::open`].
///
/// The file opens fan out; the memory mappings do not. Every `mmap` takes
/// the process's address-space lock for writing, so mapping from many
/// threads buys spin contention rather than time -- measured once as 5% of
/// total CPU in `osq_lock` for a 1.5% wall-clock return. Opening scales,
/// mapping is cheap and serial: about 5us of the 30us an open-and-map pair
/// costs.
///
/// Errors keep input order: the parallel phase records per-input results
/// and the sequential fold below reports the first failure, so a bad input
/// names the same file whichever thread reached it first.
pub fn open_all<'data>(
    inputs: &[Input<'data>],
) -> Result<Vec<InputBytes<'data>>> {
    use rayon::prelude::*;
    if inputs.len() < PARALLEL_OPEN {
        return inputs.iter().map(Input::open).collect();
    }
    let opened: Vec<_> = inputs.par_iter().map(Input::open).collect();
    let mut out = Vec::with_capacity(opened.len());
    for entry in opened {
        out.push(entry?);
    }
    Ok(out)
}

/// Writes the linked image to `output` with executable permissions.
///
/// The image lands under a temporary sibling first and is renamed into
/// place only once every byte is on disk, the same contract the ELF
/// writer's `OutputFile` honours: a build watcher -- or a build system
/// globbing the output directory -- never sees a half-written
/// executable, and a full disk leaves the previous output standing
/// rather than a truncated one wearing its name.
pub fn write_output(output: &Path, image: &[u8]) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let temp = crate::mmap_file::temp_path(output);
    let result = (|| {
        std::fs::write(&temp, image)?;
        std::fs::set_permissions(
            &temp,
            std::fs::Permissions::from_mode(output_mode()),
        )?;
        std::fs::rename(&temp, output)
    })();
    if result.is_err() {
        // The rename either did not happen or failed; either way nothing
        // may be left beside the destination.
        let _ = std::fs::remove_file(&temp);
    }
    result?;
    // The image is complete: release a caller waiting on a forked link.
    crate::detach::published();
    Ok(())
}

/// The mode a produced executable is published with: everything, less what
/// the process's umask withholds.
///
/// A fixed `0o755` is not "executable permissions", it is a decision to
/// override the one setting whose entire job is to make that decision. Under
/// `umask 077` every other linker produces `0o700` and xold produced a
/// world-readable, world-executable image -- on a build of something the user
/// had asked the system to keep to themselves.
pub fn output_mode() -> u32 {
    0o777 & !umask()
}

/// The process umask, read once.
fn umask() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(read_umask)
}

/// Reads the umask from `/proc/self/status`, which Linux has reported since
/// 4.7.
///
/// There is no way to read it through libc without also setting it, and
/// setting it here -- even to put it straight back -- would race every other
/// thread in the process that creates a file, which for a linker used as a
/// library is not a race it is entitled to run. Reading is the only safe
/// answer.
///
/// The fallback when the line is absent is `0o022`, the near-universal
/// default; it is also what the fixed `0o755` assumed, so nothing regresses
/// where the file cannot be read.
fn read_umask() -> u32 {
    const DEFAULT: u32 = 0o022;
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return DEFAULT;
    };
    status
        .lines()
        .find_map(|line| line.strip_prefix("Umask:"))
        .and_then(|v| u32::from_str_radix(v.trim(), 8).ok())
        .unwrap_or(DEFAULT)
}

/// Writes one `Pod` struct into `image` at `off`.
///
/// Returns the offset one past the last byte written.
///
/// An out-of-range offset is recorded rather than ignored. Callers size the
/// image from the layout, so the branch cannot run for a correctly sized
/// image; should that ever stop holding, a silent truncation would be
/// indistinguishable from a clean link.
pub fn write_pod<T: Pod>(image: &mut [u8], off: u64, value: &T) -> u64 {
    let bytes = bytemuck::bytes_of(value);
    let start = usize::try_from(off).unwrap_or(usize::MAX);
    let end = start.saturating_add(bytes.len());
    if let Some(slot) = image.get_mut(start..end) {
        slot.copy_from_slice(bytes);
    } else {
        debug_assert!(
            false,
            "write_pod past the end of the image: {start}..{end} of {}",
            image.len()
        );
        record_truncation();
    }
    off.wrapping_add(u64::try_from(bytes.len()).unwrap_or(0))
}

/// Whether any [`write_pod`] fell outside the image this process has built.
///
/// A process-wide flag rather than a return value: the eighteen call sites
/// serialise headers and directory entries in sequence, and threading a
/// `Result` through each would say the same thing eighteen times over. The
/// writers ask once, at the end, and refuse to publish an image that any part
/// of failed to reach -- so a sizing bug ends the link instead of shipping a
/// truncated file.
static TRUNCATED: AtomicBool = AtomicBool::new(false);

/// Records that a write fell outside the image.
fn record_truncation() {
    TRUNCATED.store(true, Ordering::Relaxed);
}

/// Whether a write fell outside the image, clearing the flag for the next
/// link. Called once per image, after everything is serialised.
pub fn took_truncated_write() -> bool {
    TRUNCATED.swap(false, Ordering::Relaxed)
}

/// Aborts the link if any [`write_pod`] fell outside the image.
///
/// The format writers call this once, after serialising everything, instead of
/// threading the same `Result` through every header write. See
/// [`took_truncated_write`] for why the flag is process-wide.
pub fn check_truncated_write() -> Result<()> {
    if took_truncated_write() {
        return Err(Error::OutOfRange(
            "a serialiser wrote past the end of the image",
        ));
    }
    Ok(())
}

/// The page size every output format aligns segments and sections to.
pub const PAGE: u64 = 0x1000;

/// Rounds `value` up to a multiple of `align` (no effect for `align <= 1`).
///
/// Saturating rather than wrapping: a value within `align` of `u64::MAX` has
/// no next multiple, and answering with a small number would place a section
/// at the bottom of the address space. Every caller here is a placement cursor
/// that only ever moves forward, so saturating keeps that invariant while
/// wrapping would break it. Inputs that could reach the boundary are rejected
/// where they enter -- see [`check_align`] -- so this is the backstop rather
/// than the check.
pub fn align_up(value: u64, align: u64) -> u64 {
    let mask = align.max(1).wrapping_sub(1);
    value.saturating_add(mask) & !mask
}

/// Rejects an input section's `sh_addralign` that alignment arithmetic cannot
/// use.
///
/// A power of two is what the field means: the rounding is a mask, and a mask
/// only describes the alignment it was derived from when exactly one bit is
/// set. A non-power-of-two silently produced a mask for the next power below,
/// so a section declaring 24 was aligned to 8 with nothing said.
///
/// The bound is the largest alignment a linker has any use for. Nothing needs
/// more than a page-table entry's worth, and a value near `u64::MAX` is an
/// arithmetic hazard rather than a request: it exists in a malformed or
/// hostile input, not in an object a compiler wrote.
pub fn check_align(align: u64, what: &'static str) -> Result<()> {
    if align > MAX_ALIGN || (align != 0 && !align.is_power_of_two()) {
        return Err(Error::Format(what));
    }
    Ok(())
}

/// The largest `sh_addralign` this linker accepts: one gigabyte, which is far
/// past anything a target's ABI asks for and far short of where the rounding
/// arithmetic gets interesting.
const MAX_ALIGN: u64 = 1 << 30;

/// Rounds `value` up to a multiple of `align`, in the 32-bit widths the PE
/// section and file alignments use.
pub fn align_up_u32(value: u32, align: u32) -> u32 {
    let mask = align.max(1).wrapping_sub(1);
    value.wrapping_add(mask) & !mask
}

/// Returns the bytes of a NUL-terminated string starting at `offset`.
///
/// The string stops at the first NUL or the end of the table. A missing or
/// out-of-range offset yields an empty slice rather than an error, matching how
/// linkers tolerate stray string-table offsets.
pub fn cstr_at(table: &[u8], offset: u32) -> &[u8] {
    let start = offset as usize;
    if start >= table.len() {
        return &[];
    }
    let end = table[start..]
        .iter()
        .position(|&b| b == 0)
        .map_or(table.len(), |nul| start + nul);
    &table[start..end]
}

/// Trims a fixed-size name cell at the first NUL. Mach-O segment/section names
/// and PE section names are stored NUL-padded in a fixed-width field.
pub fn trim_nul(buf: &[u8]) -> &[u8] {
    buf.iter()
        .position(|&b| b == 0)
        .map_or(buf, |end| &buf[..end])
}

/// Pads `name` with NULs into an `N`-byte cell, truncating overlong names.
/// The inverse of [`trim_nul`]: PE uses `N = 8`, Mach-O `N = 16`.
pub fn pad_name<const N: usize>(name: &[u8]) -> [u8; N] {
    let mut cell = [0u8; N];
    let n = name.len().min(N);
    cell[..n].copy_from_slice(&name[..n]);
    cell
}

/// Appends `value` as a little-endian signed 32-bit integer. The relative
/// offsets within one image always fit a signed 32-bit slot; truncation only
/// discards the high bits the loader contract never uses.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn push_rel32(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&(value as i32).to_le_bytes());
}

/// Appends `value` as a little-endian unsigned 32-bit integer.
pub fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Whether `name` is a valid C identifier.
///
/// A section whose name is one is special: the linker is expected to bound it
/// with `__start_NAME` and `__stop_NAME` (see [`crate::startstop`]), so such a
/// section can be reached without any relocation naming it and must never be
/// folded onto another ([`crate::icf`]). A name containing a dot is never an
/// identifier, which is what makes the check cheap for the `.text.foo` shape
/// almost every input section has.
pub fn is_c_identifier(name: &[u8]) -> bool {
    let Some(&first) = name.first() else {
        return false;
    };
    if !first.is_ascii_alphabetic() && first != b'_' {
        return false;
    }
    name.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'_')
}

/// Renders symbol-table bytes for a diagnostic, replacing invalid UTF-8.
pub fn show(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}
