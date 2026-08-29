//! Loading a shared library at run time, on either host.
//!
//! The LTO plugin is a shared object the linker opens rather than links
//! against, which is the whole reason LLVM stays out of this crate's build.
//! Opening one is the single piece of that arrangement with no portable
//! spelling: POSIX has `dlopen`, Windows has `LoadLibrary`, and they differ
//! in how a failure reports itself.
//!
//! Both are wrapped here so nothing above this module is written twice, and
//! so the raw handle has exactly one owner. [`Library`] closes its handle on
//! drop; a symbol resolved from it is only valid while it lives, which is why
//! [`Library::symbol`] hands back a bare address for the caller to transmute
//! under its own contract rather than a lifetime-erased function pointer.

use std::{ffi::c_void, path::Path};

/// A shared library held open for as long as its symbols are reachable.
pub struct Library {
    handle: *mut c_void,
}

// SAFETY: the handle is an opaque loader token, not a pointer into this
// process's Rust-owned memory. Both loaders permit use from any thread, and
// the only operations performed on it are symbol lookup and close.
unsafe impl Send for Library {}
// SAFETY: as above; symbol lookup does not mutate the handle.
unsafe impl Sync for Library {}

impl Library {
    /// Opens `path`, or reports why it could not be opened.
    ///
    /// # Errors
    ///
    /// Returns the loader's own message when the file is missing, is not a
    /// shared library for this host, or has a dependency that cannot itself
    /// be resolved.
    pub fn open(path: &Path) -> Result<Self, String> {
        let handle = imp::open(path)?;
        Ok(Self { handle })
    }

    /// The address of `name`, or `None` when the library does not export it.
    ///
    /// `name` must not contain an interior NUL.
    pub fn symbol(&self, name: &str) -> Option<*mut c_void> {
        imp::symbol(self.handle, name)
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        imp::close(self.handle);
    }
}

#[cfg(unix)]
mod imp {
    use std::{
        ffi::{CStr, CString, c_char, c_void},
        os::unix::ffi::OsStrExt as _,
        path::Path,
    };

    pub fn open(path: &Path) -> Result<*mut c_void, String> {
        let name = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| "path contains a NUL byte".to_string())?;
        // SAFETY: `name` is NUL-terminated and outlives the call. `RTLD_NOW`
        // reports an unresolved dependency here rather than at the first call
        // into the library.
        let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW) };
        if handle.is_null() {
            return Err(last_error());
        }
        Ok(handle)
    }

    pub fn symbol(handle: *mut c_void, name: &str) -> Option<*mut c_void> {
        let name = CString::new(name).ok()?;
        // SAFETY: `handle` came from a successful `dlopen` and is still open;
        // `name` is NUL-terminated.
        let sym = unsafe { libc::dlsym(handle, name.as_ptr()) };
        (!sym.is_null()).then_some(sym)
    }

    pub fn close(handle: *mut c_void) {
        // SAFETY: `handle` came from `dlopen` and is closed exactly once.
        unsafe { libc::dlclose(handle) };
    }

    /// The loader's message for the most recent failure.
    fn last_error() -> String {
        // SAFETY: `dlerror` returns null or a NUL-terminated string owned by
        // the loader; it is copied out before anything can overwrite it.
        let msg = unsafe { libc::dlerror() };
        if msg.is_null() {
            return "unknown error".to_string();
        }
        // SAFETY: non-null means a NUL-terminated string, as above.
        let text = unsafe { CStr::from_ptr(msg.cast::<c_char>()) };
        String::from_utf8_lossy(text.to_bytes()).into_owned()
    }
}

#[cfg(windows)]
mod imp {
    use std::{
        ffi::{CString, c_char, c_void},
        path::Path,
    };

    // The three loader entry points, declared rather than taken from a crate:
    // they are a stable part of every Windows install, and a dependency for
    // three declarations is a dependency that has to be justified for the
    // life of the project.
    unsafe extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(
            module: *mut c_void,
            name: *const c_char,
        ) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
        fn GetLastError() -> u32;
    }

    pub fn open(path: &Path) -> Result<*mut c_void, String> {
        let text = path
            .to_str()
            .ok_or_else(|| "path is not valid UTF-8".to_string())?;
        let name = CString::new(text)
            .map_err(|_| "path contains a NUL byte".to_string())?;
        // SAFETY: `name` is NUL-terminated and outlives the call.
        let handle = unsafe { LoadLibraryA(name.as_ptr()) };
        if handle.is_null() {
            // SAFETY: no arguments, no pointers; reads the calling thread's
            // own error slot, which the failed call above just set.
            let code = unsafe { GetLastError() };
            return Err(format!("LoadLibrary failed with error {code}"));
        }
        Ok(handle)
    }

    pub fn symbol(handle: *mut c_void, name: &str) -> Option<*mut c_void> {
        let name = CString::new(name).ok()?;
        // SAFETY: `handle` came from a successful `LoadLibraryA` and is still
        // loaded; `name` is NUL-terminated.
        let sym = unsafe { GetProcAddress(handle, name.as_ptr()) };
        (!sym.is_null()).then_some(sym)
    }

    pub fn close(handle: *mut c_void) {
        // SAFETY: `handle` came from `LoadLibraryA` and is freed exactly once.
        unsafe { FreeLibrary(handle) };
    }
}
