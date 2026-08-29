//! Offering inputs to the plugin.
//!
//! Every input is offered; the plugin answers whether it wants it. That is
//! the interface's design and it is why the linker needs no bitcode reader of
//! its own: `claim_file` sets its out-parameter to zero for anything it
//! cannot read, so an ELF object offered by mistake costs one call and
//! nothing else.
//!
//! A claimed file is the plugin's. Its symbols arrive through `add_symbols`
//! during the claim call, not from any reader here, and the bytes stay owned
//! by the session for the rest of the link because the plugin holds a view
//! into them until codegen is done.

use std::{ffi::CString, path::Path};

use crate::{
    error::{Error, Result},
    lto::{
        api::{InputFile, LDPS_OK},
        plugin::Plugin,
        session::{self, Claimed},
    },
};

/// Offers inputs to a loaded plugin.
pub struct Claimer<'p> {
    plugin: &'p Plugin,
}

/// What offering one input produced.
#[derive(Debug, Eq, PartialEq)]
pub enum Offer {
    /// The plugin took the file. Its symbols are in the session under the
    /// returned handle.
    Claimed { handle: usize, symbols: usize },
    /// The plugin cannot read this file, so the ordinary readers should.
    Declined,
}

impl<'p> Claimer<'p> {
    /// Starts offering inputs to `plugin`.
    pub const fn new(plugin: &'p Plugin) -> Self {
        Self { plugin }
    }

    /// Offers one input.
    ///
    /// `bytes` is moved into the session when the file is claimed: the plugin
    /// keeps a view into it until it has compiled, so the linker cannot be
    /// the one to decide when it goes away.
    ///
    /// # Errors
    ///
    /// Returns an error when the plugin reports one, which means it
    /// recognised the file as bitcode and then failed to read it -- a
    /// version mismatch between the bitcode and the plugin's LLVM is the
    /// usual cause.
    pub fn offer(&self, path: &Path, bytes: Vec<u8>) -> Result<Offer> {
        let Some(claim) = self.plugin.hooks().claim_file else {
            return Ok(Offer::Declined);
        };
        let name = CString::new(path.to_string_lossy().into_owned()).map_err(
            |_| {
                Error::CommandLine("input path contains a NUL byte".to_string())
            },
        )?;
        let size = i64::try_from(bytes.len())
            .map_err(|_| Error::OutOfRange("LTO input size"))?;
        let handle = register(path, bytes)?;

        let file = InputFile {
            name: name.as_ptr(),
            // The plugin reads through `get_view`, which this linker always
            // offers, so it never needs a descriptor. Handing it -1 rather
            // than a real one keeps the file table out of the interface.
            fd: -1,
            offset: 0,
            filesize: size,
            handle: std::ptr::without_provenance_mut(handle),
        };
        let mut claimed: std::ffi::c_int = 0;
        // SAFETY: `file` is a fully initialised `InputFile` that outlives the
        // call, `claimed` is a writable `int`, and the bytes `get_view` will
        // hand back were registered above and live in the session.
        let status = unsafe { claim(&raw const file, &raw mut claimed) };
        if status != LDPS_OK {
            forget(handle);
            return Err(Error::CommandLine(format!(
                "LTO plugin {} failed to read {}: the bitcode may have been \
                 written by a different LLVM version",
                self.plugin.path().display(),
                path.display()
            )));
        }
        if claimed == 0 {
            forget(handle);
            return Ok(Offer::Declined);
        }
        Ok(Offer::Claimed {
            handle,
            symbols: symbol_count(handle),
        })
    }
}

/// Records a file in the session so `get_view` can serve it during the claim.
fn register(path: &Path, bytes: Vec<u8>) -> Result<usize> {
    let mut state = session::session()
        .lock()
        .map_err(|_| Error::Format("LTO session lock was poisoned"))?;
    let handle = state.reserve();
    state.claimed.push(Claimed {
        handle,
        path: path.to_path_buf(),
        bytes,
        symbols: Vec::new(),
    });
    Ok(handle)
}

/// Drops a file the plugin did not take.
fn forget(handle: usize) {
    if let Ok(mut state) = session::session().lock() {
        state.claimed.retain(|file| file.handle != handle);
    }
}

/// How many symbols the plugin declared for a handle.
fn symbol_count(handle: usize) -> usize {
    session::session().lock().map_or(0, |state| {
        state
            .claimed
            .iter()
            .find(|file| file.handle == handle)
            .map_or(0, |file| file.symbols.len())
    })
}
