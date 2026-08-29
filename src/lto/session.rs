//! The linker side of the plugin conversation.
//!
//! The plugin calls back into the linker through bare `extern "C"` functions
//! that take no context argument: `plugin-api.h` identifies a file by an
//! opaque handle the linker chose, and nothing else. There is therefore
//! nowhere to hang per-link state but a global, and this module is the one
//! place that keeps one. A process performs one link, so the lifetime is not
//! in question; the lock is there because the plugin's `ThinLTO` backends are
//! threads.
//!
//! Every callback here is crossing an FFI boundary, so none of them may
//! unwind: a panic escaping into C is undefined behaviour. They are written
//! to have no panicking path -- no indexing, no `unwrap`, and a poisoned lock
//! is reported as [`api::LDPS_ERR`] rather than resumed.

use std::{
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

use crate::lto::api::{
    self, InputFile, LDPR_UNKNOWN, LDPS_ERR, LDPS_OK, Status, Symbol,
};

/// One symbol a plugin declared for a file it claimed.
///
/// The name is copied because the plugin's array is only valid for the
/// duration of the `add_symbols` call, while the linker needs the name for
/// the rest of the link.
pub struct PluginSymbol {
    pub name: Vec<u8>,
    /// `LDPK_*`: definition, weak definition, undefined, weak undefined or
    /// common.
    pub kind: c_uint,
    /// `LDPV_*`, mirroring `STV_*`.
    pub visibility: c_uint,
    pub size: u64,
    pub comdat: Option<Vec<u8>>,
    /// `LDPR_*`, filled by the linker before `get_symbols` is answered.
    pub resolution: c_int,
}

/// A file the plugin took ownership of.
pub struct Claimed {
    /// The handle the linker gave the plugin for this file.
    pub handle: usize,
    pub path: PathBuf,
    /// The bytes the file was claimed from, kept so `get_view` can hand the
    /// plugin a pointer that stays valid for the whole link.
    pub bytes: Vec<u8>,
    pub symbols: Vec<PluginSymbol>,
}

/// Everything the callbacks read and write.
#[derive(Default)]
pub struct State {
    pub claimed: Vec<Claimed>,
    /// The next handle to hand out. Handles name a file for the plugin's
    /// whole run, and the session outlives any one [`super::Claimer`], so the
    /// counter lives here: two claimers starting from one would give two
    /// files the same name.
    pub next_handle: usize,
    /// Native objects the plugin produced, in the order it announced them.
    pub added: Vec<PathBuf>,
    /// Extra library directories the plugin asked for.
    pub library_paths: Vec<PathBuf>,
}

impl State {
    /// Reserves a handle. Counting from one keeps every handle
    /// distinguishable from the null pointer a plugin checks against.
    pub fn reserve(&mut self) -> usize {
        self.next_handle = self.next_handle.saturating_add(1);
        self.next_handle
    }

    /// The claimed file a handle names.
    fn find(&mut self, handle: usize) -> Option<&mut Claimed> {
        self.claimed.iter_mut().find(|file| file.handle == handle)
    }
}

/// The single live session.
///
/// Interior mutability is the exception this interface forces: the callbacks
/// below are the plugin's only way back into the linker and carry no state of
/// their own.
static SESSION: OnceLock<Mutex<State>> = OnceLock::new();

/// The session, created on first use.
pub fn session() -> &'static Mutex<State> {
    SESSION.get_or_init(|| Mutex::new(State::default()))
}

/// Runs `body` against the session, reporting a poisoned lock as an error
/// rather than unwinding into the plugin.
fn with_state(body: impl FnOnce(&mut State) -> Status) -> Status {
    session()
        .lock()
        .map_or(LDPS_ERR, |mut state| body(&mut state))
}

/// Copies a NUL-terminated string the plugin owns.
fn copy_c_str(ptr: *const c_char) -> Option<Vec<u8>> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: the ABI states these are NUL-terminated strings valid for the
    // duration of the call; the bytes are copied out before returning.
    Some(unsafe { CStr::from_ptr(ptr) }.to_bytes().to_vec())
}

/// `LDPT_MESSAGE`: the plugin's diagnostic channel.
///
/// `plugin-api.h` declares this
/// `void (*)(int level, const char *format, ...)`, and a plugin does call it
/// with substitutions. They are discarded: walking a `va_list` from Rust
/// needs the format string parsed to know what types it holds, and a wrong
/// guess reads the wrong register. A plugin reporting a linker-interface
/// problem passes a plain string, so the format is reported as written.
/// Losing a substitution is better than the alternative, which is that a
/// plugin with no message channel calls a null pointer.
///
/// # Why this is declared without `...`
///
/// Defining a C-variadic function in Rust needs the unstable `c_variadic`
/// feature, which would make the whole linker nightly-only for the sake of
/// arguments this body never reads. Declaring it with just the two named
/// parameters is sound for every target xold builds for, because a callee
/// that touches only the named parameters cannot observe whether it was
/// declared variadic:
///
/// - x86-64 System V passes `level` and `format` in `rdi` and `rsi` either way.
///   A variadic call additionally sets `al` to the number of vector registers
///   used, which a non-variadic callee simply never reads.
/// - `AArch64` AAPCS64 passes the first eight integer arguments in `x0`-`x7`
///   under both forms.
/// - Apple's arm64 variant moves the *variadic* arguments to the stack, but the
///   named ones stay in `x0` and `x1`, which is all this reads.
///
/// The transfer vector erases the type to `*mut c_void` regardless, so the
/// declaration never has to match `ld_plugin_message` for the registration
/// to compile; it has to match only for the call to land correctly, which
/// is what the above establishes.
///
/// # Safety
///
/// Called only by the plugin, through the transfer vector, under the
/// `plugin-api.h` contract: every pointer argument is either null or valid
/// for the duration of the call, and any count describes that many elements.
pub unsafe extern "C" fn message(level: c_int, format: *const c_char) {
    let text = copy_c_str(format).map_or_else(
        || "(no message)".to_string(),
        |bytes| String::from_utf8_lossy(&bytes).into_owned(),
    );
    let severity = match level {
        api::LDPL_INFO => "info",
        api::LDPL_WARNING => "warning",
        api::LDPL_FATAL => "fatal",
        _ => "error",
    };
    eprintln!("xold: LTO plugin {severity}: {text}");
}

/// `LDPT_ADD_SYMBOLS`: the plugin declaring what a claimed file defines and
/// references.
///
/// # Safety
///
/// Called only by the plugin, through the transfer vector, under the
/// `plugin-api.h` contract: every pointer argument is either null or valid
/// for the duration of the call, and any count describes that many elements.
pub unsafe extern "C" fn add_symbols(
    handle: *mut c_void,
    count: c_int,
    syms: *const Symbol,
) -> Status {
    let Ok(count) = usize::try_from(count) else {
        return LDPS_ERR;
    };
    if syms.is_null() && count != 0 {
        return LDPS_ERR;
    }
    let handle = handle as usize;
    with_state(|state| {
        let mut collected = Vec::with_capacity(count);
        for index in 0..count {
            // SAFETY: the plugin promises `count` readable `Symbol` values at
            // `syms`, valid for this call.
            let sym = unsafe { &*syms.add(index) };
            let Some(name) = copy_c_str(sym.name) else {
                return LDPS_ERR;
            };
            collected.push(PluginSymbol {
                name,
                kind: c_uint::try_from(sym.def).unwrap_or(api::LDPK_UNDEF),
                visibility: sym.visibility,
                size: sym.size,
                comdat: copy_c_str(sym.comdat_key),
                resolution: LDPR_UNKNOWN.cast_signed(),
            });
        }
        let Some(file) = state.find(handle) else {
            return LDPS_ERR;
        };
        file.symbols = collected;
        LDPS_OK
    })
}

/// `LDPT_GET_SYMBOLS`: the plugin asking how each symbol resolved.
///
/// The resolutions were decided by the link before `all_symbols_read` ran;
/// this only copies them back into the plugin's array.
///
/// # Safety
///
/// Called only by the plugin, through the transfer vector, under the
/// `plugin-api.h` contract: every pointer argument is either null or valid
/// for the duration of the call, and any count describes that many elements.
pub unsafe extern "C" fn get_symbols(
    handle: *const c_void,
    count: c_int,
    syms: *mut Symbol,
) -> Status {
    let Ok(count) = usize::try_from(count) else {
        return LDPS_ERR;
    };
    if syms.is_null() && count != 0 {
        return LDPS_ERR;
    }
    let handle = handle as usize;
    with_state(|state| {
        let Some(file) = state.find(handle) else {
            return LDPS_ERR;
        };
        if file.symbols.len() != count {
            return LDPS_ERR;
        }
        for (index, sym) in file.symbols.iter().enumerate() {
            // SAFETY: the plugin promises `count` writable `Symbol` values at
            // `syms`, and `count` equals the number recorded for this file.
            unsafe { (*syms.add(index)).resolution = sym.resolution };
        }
        LDPS_OK
    })
}

/// `LDPT_ADD_INPUT_FILE`: a native object the plugin compiled.
///
/// # Safety
///
/// Called only by the plugin, through the transfer vector, under the
/// `plugin-api.h` contract: every pointer argument is either null or valid
/// for the duration of the call, and any count describes that many elements.
pub unsafe extern "C" fn add_input_file(path: *const c_char) -> Status {
    let Some(bytes) = copy_c_str(path) else {
        return LDPS_ERR;
    };
    with_state(|state| {
        state
            .added
            .push(PathBuf::from(String::from_utf8_lossy(&bytes).into_owned()));
        LDPS_OK
    })
}

/// `LDPT_SET_EXTRA_LIBRARY_PATH`: a directory the plugin's objects may need.
///
/// # Safety
///
/// Called only by the plugin, through the transfer vector, under the
/// `plugin-api.h` contract: every pointer argument is either null or valid
/// for the duration of the call, and any count describes that many elements.
pub unsafe extern "C" fn set_extra_library_path(path: *const c_char) -> Status {
    let Some(bytes) = copy_c_str(path) else {
        return LDPS_ERR;
    };
    with_state(|state| {
        state
            .library_paths
            .push(PathBuf::from(String::from_utf8_lossy(&bytes).into_owned()));
        LDPS_OK
    })
}

/// `LDPT_GET_INPUT_FILE`: the bytes behind a handle, as a descriptor-shaped
/// record.
///
/// The plugin reads the file through `get_view` in practice; this exists
/// because registering an all-symbols-read hook obliges the linker to offer
/// it, and a plugin that asks must get a coherent answer.
///
/// # Safety
///
/// Called only by the plugin, through the transfer vector, under the
/// `plugin-api.h` contract: every pointer argument is either null or valid
/// for the duration of the call, and any count describes that many elements.
pub unsafe extern "C" fn get_input_file(
    handle: *const c_void,
    file: *mut InputFile,
) -> Status {
    if file.is_null() {
        return LDPS_ERR;
    }
    let handle = handle as usize;
    with_state(|state| {
        let Some(claimed) = state.find(handle) else {
            return LDPS_ERR;
        };
        let Ok(size) = i64::try_from(claimed.bytes.len()) else {
            return LDPS_ERR;
        };
        // SAFETY: `file` is a writable `InputFile` the plugin supplied.
        unsafe {
            (*file).name = std::ptr::null();
            (*file).fd = -1;
            (*file).offset = 0;
            (*file).filesize = size;
            (*file).handle = handle as *mut c_void;
        }
        LDPS_OK
    })
}

/// `LDPT_RELEASE_INPUT_FILE`: the plugin is done reading a handle.
///
/// The bytes are owned by the session for the whole link, so there is nothing
/// to release; accepting the call is what the interface asks for.
///
/// # Safety
///
/// Called only by the plugin, through the transfer vector, under the
/// `plugin-api.h` contract: every pointer argument is either null or valid
/// for the duration of the call, and any count describes that many elements.
pub unsafe extern "C" fn release_input_file(_handle: *const c_void) -> Status {
    LDPS_OK
}

/// `LDPT_GET_VIEW`: a pointer to the claimed file's bytes.
///
/// # Safety
///
/// Called only by the plugin, through the transfer vector, under the
/// `plugin-api.h` contract: every pointer argument is either null or valid
/// for the duration of the call, and any count describes that many elements.
pub unsafe extern "C" fn get_view(
    handle: *const c_void,
    view: *mut *mut c_void,
) -> Status {
    if view.is_null() {
        return LDPS_ERR;
    }
    let handle = handle as usize;
    with_state(|state| {
        let Some(claimed) = state.find(handle) else {
            return LDPS_ERR;
        };
        let ptr = claimed.bytes.as_ptr().cast::<c_void>().cast_mut();
        // SAFETY: `view` is a writable out-parameter the plugin supplied. The
        // bytes it receives live in the session for the rest of the link.
        unsafe { *view = ptr };
        LDPS_OK
    })
}
