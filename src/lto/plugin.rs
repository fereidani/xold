//! Loading an LTO plugin and handing it the transfer vector.
//!
//! This is the only module that touches a raw pointer from the plugin ABI.
//! Everything it exposes is safe: a [`Plugin`] owns its `dlopen` handle for
//! as long as the link runs, and the hooks the plugin registered are plain
//! function pointers by the time a caller sees them.
//!
//! The plugin is resolved the way a compiler driver expects. `-plugin PATH`
//! is honoured exactly as written, because the bitcode a plugin can read is
//! tied to the LLVM it was built from: rustc's `-Clinker-plugin-lto` needs
//! rustc's own LLVM, which is routinely a different version from the system
//! one. Only when no `-plugin` was given does the conventional
//! `bfd-plugins` directory get searched.

use std::{
    ffi::{CString, OsStr, c_int, c_void},
    path::{Path, PathBuf},
    ptr,
};

use crate::{
    error::{Error, Result},
    lto::{
        api::{
            AllSymbolsRead, ClaimFile, Cleanup, LDPO_DYN, LDPO_EXEC, LDPO_PIE,
            LDPO_REL, LDPS_OK, LDPT_ADD_INPUT_FILE, LDPT_ADD_SYMBOLS,
            LDPT_API_VERSION, LDPT_GET_INPUT_FILE, LDPT_GET_SYMBOLS_V2,
            LDPT_GET_VIEW, LDPT_LINKER_OUTPUT, LDPT_MESSAGE, LDPT_NULL,
            LDPT_OPTION, LDPT_OUTPUT_NAME, LDPT_REGISTER_ALL_SYMBOLS_READ_HOOK,
            LDPT_REGISTER_CLAIM_FILE_HOOK, LDPT_REGISTER_CLEANUP_HOOK,
            LDPT_RELEASE_INPUT_FILE, LDPT_SET_EXTRA_LIBRARY_PATH, OnLoad,
            Status, Value,
        },
        dynlib::Library,
        session,
    },
};

/// What the link is producing, in the plugin's vocabulary.
///
/// The plugin needs it to decide whether a definition can be preempted at
/// load time, which is what licenses it to internalise one.
#[derive(Clone, Copy)]
pub enum Output {
    Relocatable,
    Executable,
    Pie,
    Shared,
}

impl Output {
    const fn tag(self) -> c_int {
        match self {
            Self::Relocatable => LDPO_REL,
            Self::Executable => LDPO_EXEC,
            Self::Pie => LDPO_PIE,
            Self::Shared => LDPO_DYN,
        }
    }
}

/// The directories searched for a plugin when the command line named none.
///
/// This is the location binutils established and every distribution follows;
/// a linker that ignored it would need `-plugin` on links where `ld` needs
/// nothing.
const PLUGIN_DIRS: &[&str] =
    &["/usr/lib64/bfd-plugins", "/usr/lib/bfd-plugins"];

/// The plugin file name LLVM installs.
const PLUGIN_NAME: &str = "LLVMgold.so";

/// The hooks a plugin registered during `onload`.
///
/// `claim_file` is the only one a plugin must provide; the others are absent
/// in a plugin that has nothing to do at those points, and calling none of
/// them is correct.
#[derive(Default, Clone, Copy)]
pub struct Hooks {
    pub claim_file: Option<ClaimFile>,
    pub all_symbols_read: Option<AllSymbolsRead>,
    pub cleanup: Option<Cleanup>,
}

/// Hooks a plugin registers into, before they are handed to a [`Plugin`].
///
/// The plugin calls the registration callbacks from inside `onload`, so the
/// destination has to exist before `onload` runs and be readable after it
/// returns. A process links once, and the callbacks the plugin is given are
/// bare `extern "C"` functions with no context argument, so there is nowhere
/// to put per-link state: this is a `static`, which is what the ABI forces.
static mut REGISTERED: Hooks = Hooks {
    claim_file: None,
    all_symbols_read: None,
    cleanup: None,
};

unsafe extern "C" fn register_claim_file(hook: ClaimFile) -> Status {
    // SAFETY: `onload` is called from `Plugin::load`, which holds `&mut self`
    // and is not reentrant, so no other reference to `REGISTERED` exists for
    // the duration of this call.
    unsafe { REGISTERED.claim_file = Some(hook) };
    LDPS_OK
}

unsafe extern "C" fn register_all_symbols_read(hook: AllSymbolsRead) -> Status {
    // SAFETY: as in `register_claim_file`.
    unsafe { REGISTERED.all_symbols_read = Some(hook) };
    LDPS_OK
}

unsafe extern "C" fn register_cleanup(hook: Cleanup) -> Status {
    // SAFETY: as in `register_claim_file`.
    unsafe { REGISTERED.cleanup = Some(hook) };
    LDPS_OK
}

/// A loaded LTO plugin.
///
/// The `dlopen` handle is held for the lifetime of the value: the hooks are
/// code inside the shared object, so closing it while they are reachable
/// would leave dangling function pointers.
pub struct Plugin {
    /// Held for the whole link: the hooks are code inside this library, so
    /// closing it while they are reachable leaves dangling function pointers.
    ///
    /// Never read after loading; holding it is the whole job.
    #[allow(dead_code)]
    library: Library,
    hooks: Hooks,
    path: PathBuf,
    /// The strings handed to the plugin through the transfer vector.
    ///
    /// They outlive `onload` because the plugin keeps the pointers rather
    /// than copying: `gold-plugin.cpp` pushes each `-plugin-opt=-...` value
    /// into a vector and only parses it when codegen runs, so a buffer freed
    /// when `onload` returned is read long after. Keeping them here for as
    /// long as the plugin is loaded is what makes that safe.
    ///
    /// Never read after `onload`: holding it is the whole job.
    #[allow(dead_code)]
    strings: Vec<CString>,
}

impl Plugin {
    /// Loads `path` and runs its `onload`, returning the hooks it registered.
    pub fn load(
        path: &Path,
        output: &Path,
        kind: Output,
        options: &[String],
    ) -> Result<Self> {
        let library = Library::open(path).map_err(|why| {
            Error::CommandLine(format!(
                "cannot load LTO plugin {}: {why}",
                path.display()
            ))
        })?;
        let onload = entry(&library, path)?;
        // The plugin names its temporary objects after the output, so it has
        // to be a string both sides can spell. A path this cannot render is a
        // diagnostic rather than a silently mangled name.
        let name = output.to_str().ok_or_else(|| {
            Error::CommandLine(
                "output path is not valid UTF-8, which the LTO plugin \
                 interface cannot carry"
                    .to_string(),
            )
        })?;
        let mut strings = Vec::with_capacity(options.len().saturating_add(1));
        strings.push(CString::new(name).map_err(|_| {
            Error::CommandLine("output path contains a NUL byte".to_string())
        })?);
        for opt in options {
            strings.push(CString::new(opt.as_str()).map_err(|_| {
                Error::CommandLine(format!(
                    "-plugin-opt={opt} contains a NUL byte"
                ))
            })?);
        }
        let hooks = call_onload(onload, path, kind, &strings)?;
        Ok(Self {
            library,
            hooks,
            path: path.to_path_buf(),
            strings,
        })
    }

    /// The hooks the plugin registered.
    pub const fn hooks(&self) -> Hooks {
        self.hooks
    }

    /// The path the plugin was loaded from, for diagnostics.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Resolves a plugin's single entry point.
fn entry(library: &Library, path: &Path) -> Result<OnLoad> {
    let sym = library.symbol("onload").ok_or_else(|| {
        Error::CommandLine(format!(
            "{} is not an LTO plugin: it exports no `onload`",
            path.display()
        ))
    })?;
    // SAFETY: a resolved `onload` is that function, and the plugin ABI fixes
    // its signature. The caller keeps the library loaded for as long as the
    // returned pointer is used.
    Ok(unsafe { std::mem::transmute::<*mut c_void, OnLoad>(sym) })
}

/// Builds the transfer vector and runs `onload`.
///
/// The vector is a stack array: the plugin reads it during the call and must
/// not retain it, which is what `plugin-api.h` specifies and what
/// `gold-plugin.cpp` does.
fn call_onload(
    onload: OnLoad,
    path: &Path,
    output: Output,
    strings: &[CString],
) -> Result<Hooks> {
    // `strings[0]` is the output name; the rest are `-plugin-opt=` values.
    // All of them are owned by the caller for as long as the plugin lives.
    let Some((output_name, opts)) = strings.split_first() else {
        return Err(Error::Format("LTO plugin strings were not built"));
    };
    let mut tv: Vec<Value> = vec![
        Value {
            tag: LDPT_API_VERSION,
            payload: ptr::without_provenance_mut(
                usize::try_from(crate::lto::api::API_VERSION).unwrap_or(1),
            ),
        },
        Value {
            tag: LDPT_REGISTER_CLAIM_FILE_HOOK,
            payload: register_claim_file as *mut c_void,
        },
        Value {
            tag: LDPT_REGISTER_ALL_SYMBOLS_READ_HOOK,
            payload: register_all_symbols_read as *mut c_void,
        },
        Value {
            tag: LDPT_REGISTER_CLEANUP_HOOK,
            payload: register_cleanup as *mut c_void,
        },
        // Registering an all-symbols-read hook obliges the linker to offer
        // the file-reading callbacks as well; a plugin given one without the
        // others reports the omission through `message`, so that channel is
        // not optional either.
        Value {
            tag: LDPT_ADD_SYMBOLS,
            payload: session::add_symbols as *mut c_void,
        },
        Value {
            tag: LDPT_GET_SYMBOLS_V2,
            payload: session::get_symbols as *mut c_void,
        },
        Value {
            tag: LDPT_ADD_INPUT_FILE,
            payload: session::add_input_file as *mut c_void,
        },
        Value {
            tag: LDPT_GET_INPUT_FILE,
            payload: session::get_input_file as *mut c_void,
        },
        Value {
            tag: LDPT_RELEASE_INPUT_FILE,
            payload: session::release_input_file as *mut c_void,
        },
        Value {
            tag: LDPT_GET_VIEW,
            payload: session::get_view as *mut c_void,
        },
        Value {
            tag: LDPT_SET_EXTRA_LIBRARY_PATH,
            payload: session::set_extra_library_path as *mut c_void,
        },
        Value {
            tag: LDPT_MESSAGE,
            payload: session::message as *mut c_void,
        },
        Value {
            tag: LDPT_LINKER_OUTPUT,
            payload: ptr::without_provenance_mut(
                usize::try_from(output.tag()).unwrap_or(0),
            ),
        },
        Value {
            tag: LDPT_OUTPUT_NAME,
            payload: output_name.as_ptr().cast::<c_void>().cast_mut(),
        },
    ];
    for opt in opts {
        tv.push(Value {
            tag: LDPT_OPTION,
            payload: opt.as_ptr().cast::<c_void>().cast_mut(),
        });
    }
    tv.push(Value {
        tag: LDPT_NULL,
        payload: ptr::null_mut(),
    });
    // SAFETY: no other reference to `REGISTERED` is live: `load` is the only
    // caller and holds `&mut self`.
    unsafe {
        REGISTERED = Hooks::default();
    }
    // SAFETY: `tv` is a `LDPT_NULL`-terminated array that outlives the call,
    // which is the whole contract `onload` is given.
    let status = unsafe { onload(tv.as_mut_ptr()) };
    if status != LDPS_OK {
        return Err(Error::CommandLine(format!(
            "LTO plugin {} refused to load (status {status})",
            path.display()
        )));
    }
    // SAFETY: `onload` has returned, so the plugin is no longer writing here.
    let hooks = unsafe { REGISTERED };
    if hooks.claim_file.is_none() {
        return Err(Error::CommandLine(format!(
            "LTO plugin {} registered no claim-file hook, so it can read no \
             bitcode",
            path.display()
        )));
    }
    Ok(hooks)
}

/// Where to load the plugin from: the path the command line named, else the
/// conventional directory, else `None` when the host has no plugin installed.
pub fn resolve(named: Option<&Path>) -> Option<PathBuf> {
    if let Some(path) = named {
        return Some(path.to_path_buf());
    }
    for dir in PLUGIN_DIRS {
        let candidate = Path::new(OsStr::new(dir)).join(PLUGIN_NAME);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Whether `status` is the plugin's success value.
pub const fn ok(status: c_int) -> bool {
    status == LDPS_OK
}
