//! Memory-mapped files: read-only input views and the writable output image.
//!
//! This is the single place the linker touches the operating-system mapping
//! API. An input mapping is held alive by the [`MappedFile`] value and callers
//! borrow its bytes for the lifetime of the parse. The output image is mapped
//! writable by [`OutputFile`], so the writer serialises straight into the page
//! cache instead of filling a heap buffer that then has to be copied out.

use std::{
    fs::File,
    path::{Path, PathBuf},
};

use crate::error::{Error, Result};

/// Restates an input's I/O failure with the file it happened on.
///
/// `std::io::Error` carries no path, so "No such file or directory" on its own
/// leaves the reader to work out which of several hundred inputs it was. A
/// linker script makes that worse than tedious: the file it names never
/// appeared on the command line, so there is nothing to check the message
/// against.
fn on_path(path: &Path, err: &std::io::Error) -> Error {
    Error::Io(std::io::Error::new(
        err.kind(),
        format!("{}: {err}", path.display()),
    ))
}

/// A memory-mapped, read-only view of a file's contents.
pub struct MappedFile {
    // `Mmap` derefs to `&[u8]`; keeping it as a field avoids a self-
    // referential struct.
    map: memmap2::Mmap,
}

/// An input opened and measured, but not yet mapped: the two halves of
/// [`MappedFile::open`] meet over this.
pub struct OpenedFile {
    file: File,
    len: u64,
}

impl MappedFile {
    /// Opens and maps `path` for reading.
    ///
    /// The `unsafe` is confined to this wrapper. `memmap2::Mmap::map` requires
    /// `unsafe` because concurrent modification of the underlying file by
    /// another process is undefined behaviour; for static linker input this is
    /// the accepted, well-understood risk.
    pub fn open(path: &Path) -> Result<Self> {
        Self::map(path, &Self::open_file(path)?)
    }

    /// Opens `path` for reading, and reads its length, without mapping it.
    ///
    /// The open half of [`Self::open`], so a caller with many inputs can run
    /// the opens -- which scale across threads -- apart from the mappings,
    /// which serialise on the address-space lock. The length rides along
    /// because mapping would otherwise `fstat` the file itself, putting one
    /// serial stat (and its security hooks) back per input. `path` reappears
    /// in [`Self::map`] only to name the file in an error.
    pub fn open_file(path: &Path) -> Result<OpenedFile> {
        let file = File::open(path).map_err(|err| on_path(path, &err))?;
        let len = file.metadata().map_err(|err| on_path(path, &err))?.len();
        Ok(OpenedFile { file, len })
    }

    /// Maps an already-open `file`; `path` names it in any error.
    pub fn map(path: &Path, file: &OpenedFile) -> Result<Self> {
        let len = usize::try_from(file.len).map_err(|_| {
            Error::Format("input larger than the address space")
        })?;
        // SAFETY: the file is opened read-only and is treated as immutable
        // input for the duration of the link.
        let map =
            unsafe { memmap2::MmapOptions::new().len(len).map(&file.file) }
                .map_err(|err| on_path(path, &err))?;
        Ok(Self { map })
    }

    /// The mapped bytes, borrowed for the lifetime of this mapping.
    pub fn bytes(&self) -> &[u8] {
        &self.map
    }
}

/// The output image: a writable mapping over a temporary file that is renamed
/// into place once it is complete.
///
/// # Why a mapping
///
/// Building the image in a heap buffer means the kernel allocates and zeroes
/// anonymous pages as the writer touches them, and then `write` copies every
/// byte again into page-cache pages it also has to allocate. Writing through a
/// shared mapping does it once: the page-cache pages are the buffer, and the
/// kernel writes them back on its own schedule.
///
/// # Why a temporary
///
/// A file being written cannot be executed, and a file being executed cannot
/// be written. Both matter here.
///
/// Writing over the destination directly fails with `ETXTBSY` whenever the
/// previous image is still running -- relinking a program you are currently
/// executing is entirely normal. And a live shared write mapping raises the
/// destination's write count, so a caller that forks on another thread leaks
/// that reference into the child and its own attempt to run the result can
/// fail with `ETXTBSY` too.
///
/// Building into a temporary and renaming avoids both. The destination never
/// carries a write reference, the old inode is left untouched for anything
/// still executing it, and the replacement is atomic: a concurrent reader sees
/// either the whole old image or the whole new one.
///
/// # What this does not fix
///
/// One `ETXTBSY` window is not the linker's to close. A process that forks on
/// another thread while the image is being written hands the child a copy of
/// the write reference, which the child holds until it calls `exec`; running
/// the image inside that gap fails. That is inherent to writing an executable
/// at all, so a caller that links and immediately runs the result from a
/// multi-threaded, forking process should retry on `ETXTBSY`.
pub struct OutputFile {
    /// Taken by [`Self::finish`] so the mapping is dropped, and the write
    /// reference released, before the image is published.
    map: Option<memmap2::MmapMut>,
    file: Option<File>,
    /// The temporary being written. Cleared once it has been renamed, so the
    /// drop guard knows there is nothing to clean up.
    temp: PathBuf,
    dest: PathBuf,
}

impl Drop for OutputFile {
    fn drop(&mut self) {
        // Reached only when the link failed part way through: the image was
        // never published, so the temporary is just litter.
        if !self.temp.as_os_str().is_empty() {
            let _ = std::fs::remove_file(&self.temp);
        }
    }
}

/// On Linux, reserves the file's blocks up front, so a full disk is an error
/// rather than a signal.
///
/// `set_len` makes a sparse file: the blocks are allocated when the pages
/// fault in during writing, and a failed allocation there kills the process
/// with SIGBUS -- no message, and the temporary left behind. mold reserves for
/// this reason and lld's `FileOutputBuffer` reports the error.
///
/// A filesystem that cannot preallocate (`EOPNOTSUPP`, and `EINVAL` from the
/// ones that report it that way) keeps the sparse file it would have had, so
/// nothing that worked before stops working.
///
/// tmpfs is excluded on purpose. A disk filesystem satisfies `fallocate` by
/// reserving extents, which is nearly free; tmpfs has no extents, so it
/// allocates and zeroes every page of the image on the spot, on one thread,
/// before a single byte is written -- and the writer then writes each of
/// those pages again. Skipping the reservation there moves the page zeroing
/// to first touch, inside the writer's parallel copy pass, and costs only
/// the up-front `ENOSPC` report that the memory filesystem could not honour
/// meaningfully anyway (pages appear on write, not on reservation, whenever
/// the mapping outlives a concurrent consumer of the same memory).
#[cfg(target_os = "linux")]
fn reserve_blocks(file: &File, size: u64) -> Result<()> {
    use std::os::fd::AsRawFd;
    let Ok(len) = i64::try_from(size) else {
        return Ok(());
    };
    if len == 0 || on_tmpfs(file) {
        return Ok(());
    }
    // SAFETY: `fd` is open for writing for the duration of the call, and the
    // offset and length are non-negative.
    let rc = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, len) };
    match rc {
        0 | libc::EOPNOTSUPP | libc::EINVAL | libc::ENOSYS => Ok(()),
        // `posix_fallocate` reports through its return value, not `errno`.
        code => Err(std::io::Error::from_raw_os_error(code).into()),
    }
}

/// Whether `file` lives on tmpfs, where preallocation populates pages
/// instead of reserving extents. A failed `fstatfs` answers no, keeping the
/// reservation and its error report.
#[cfg(target_os = "linux")]
fn on_tmpfs(file: &File) -> bool {
    use std::{mem::MaybeUninit, os::fd::AsRawFd};
    let mut buf = MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: `fd` is a valid open descriptor and `buf` is sized for the
    // `statfs` the call fills in.
    let rc = unsafe { libc::fstatfs(file.as_raw_fd(), buf.as_mut_ptr()) };
    if rc != 0 {
        return false;
    }
    // SAFETY: `fstatfs` returned success, so `buf` is initialised.
    let statfs = unsafe { buf.assume_init() };
    statfs.f_type == libc::TMPFS_MAGIC
}

/// Filesystems this linker cannot preallocate on keep the sparse file.
// `Result` so both definitions present one signature to the caller.
#[allow(clippy::unnecessary_wraps)]
#[cfg(not(target_os = "linux"))]
fn reserve_blocks(_file: &File, _size: u64) -> Result<()> {
    Ok(())
}

impl OutputFile {
    /// Creates the temporary alongside `dest`, sizes it to `size` bytes and
    /// maps it writable.
    ///
    /// The temporary is created in the destination's own directory so the
    /// closing rename stays within one filesystem.
    pub fn create(dest: &Path, size: u64) -> Result<Self> {
        let temp = temp_path(dest);
        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp)?;
        file.set_len(size)?;
        reserve_blocks(&file, size)?;
        // A zero-length mapping is not valid; an empty image needs none.
        let map = if size == 0 {
            None
        } else {
            // SAFETY: the file was just created by this process under a name
            // carrying its process id, so nothing else is writing to it for
            // the duration of the link.
            let map = unsafe { memmap2::MmapMut::map_mut(&file)? };
            // On Linux, a writable mapping of a file makes it unexecutable
            // while it lives, and a fork inherits it. Without this, a caller
            // that links on one thread while spawning a process on another
            // hands the child a mapping it keeps until it execs, and any
            // attempt to run the image in that window fails with `ETXTBSY`.
            // Advice the kernel may decline is not worth failing the link over.
            #[cfg(target_os = "linux")]
            let _ = map.advise(memmap2::Advice::DontFork);
            Some(map)
        };
        // The mapping keeps the file alive on its own. Closing the descriptor
        // here means no fork can inherit one either, which is the other half
        // of the same race -- an inherited descriptor keeps the file's write
        // reference alive just as the mapping does.
        let file = (map.is_none()).then_some(file);
        Ok(Self {
            map,
            file,
            temp,
            dest: dest.to_path_buf(),
        })
    }

    /// The image bytes, for the writer to fill.
    pub fn bytes(&mut self) -> &mut [u8] {
        self.map.as_mut().map_or(&mut [], |m| m)
    }

    /// Publishes the finished image at its destination.
    ///
    /// The mapping is released and the file closed before the rename, so that
    /// once this returns nothing holds a write reference to the image and it
    /// can be executed immediately.
    ///
    /// The mapping is not synced. Dropping it makes every write visible to
    /// anything that opens or executes the file; forcing them to disk as well
    /// would mean waiting on the whole image for a durability guarantee a
    /// linker does not make.
    pub fn finish(mut self) -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        drop(self.map.take());
        drop(self.file.take());
        std::fs::set_permissions(
            &self.temp,
            std::fs::Permissions::from_mode(crate::util::output_mode()),
        )?;
        std::fs::rename(&self.temp, &self.dest)?;
        // Published: there is no longer a temporary to clean up, and a
        // caller waiting on a forked link can be released now -- everything
        // after this line is teardown it need not see.
        self.temp.clear();
        crate::detach::published();
        Ok(())
    }
}

/// The temporary path for an output: a sibling of the destination, marked with
/// this process's id so concurrent links never collide.
pub(crate) fn temp_path(dest: &Path) -> PathBuf {
    let name = dest.file_name().unwrap_or_default();
    let mut temp = name.to_os_string();
    temp.push(format!(".xold-{}", std::process::id()));
    dest.parent().unwrap_or_else(|| Path::new(".")).join(temp)
}
