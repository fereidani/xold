//! The GNU linker-plugin ABI, as `LLVMgold.so` consumes it.
//!
//! A linker that is not part of LLVM cannot call `lto::LTO` -- that is an
//! unstable C++ API. What it can do is what gold and `ld.bfd` do: load a
//! plugin that wraps LLVM and speak the C ABI binutils defined for the
//! purpose. This module is the transcription of that ABI and nothing else.
//! Every item here mirrors a declaration in binutils' `plugin-api.h`; the
//! consuming side is `llvm/tools/gold/gold-plugin.cpp`, which is what pins
//! down which tags a real plugin actually reads.
//!
//! The declarations are `repr(C)` layouts and function pointers, so they are
//! inherently unsafe to hand across the boundary. Nothing here dereferences
//! anything: the safe wrapper in [`super::plugin`] owns every raw pointer and
//! is the only place that may touch one.

use std::ffi::{c_char, c_int, c_uint, c_void};

/// Status returned by every plugin and linker callback.
pub type Status = c_int;

pub const LDPS_OK: Status = 0;
pub const LDPS_NO_SYMS: Status = 1;
pub const LDPS_ERR: Status = 2;

/// Transfer-vector tags. The linker hands the plugin an array of
/// [`Value`] terminated by [`LDPT_NULL`]; each entry is one capability.
///
/// Only the tags a plugin reads are listed. `gold-plugin.cpp` errors out
/// without `LDPT_REGISTER_CLAIM_FILE_HOOK` and `LDPT_ADD_SYMBOLS`; the rest
/// it takes when offered.
///
/// The values are positions in one `enum`, so a wrong one is not a missing
/// capability but a mismatched pair: the plugin calls the function it was
/// handed under a signature that belongs to another. Two of them are pinned
/// by `gold-plugin.cpp`'s own fallback defines -- `LDPT_GET_SYMBOLS_V3` is 28
/// and `LDPT_GET_WRAP_SYMBOLS` is 32 -- which is what fixes the rest of the
/// sequence.
pub const LDPT_NULL: c_int = 0;
pub const LDPT_API_VERSION: c_int = 1;
pub const LDPT_GOLD_VERSION: c_int = 2;
pub const LDPT_LINKER_OUTPUT: c_int = 3;
pub const LDPT_OPTION: c_int = 4;
pub const LDPT_REGISTER_CLAIM_FILE_HOOK: c_int = 5;
pub const LDPT_REGISTER_ALL_SYMBOLS_READ_HOOK: c_int = 6;
pub const LDPT_REGISTER_CLEANUP_HOOK: c_int = 7;
pub const LDPT_ADD_SYMBOLS: c_int = 8;
pub const LDPT_GET_SYMBOLS: c_int = 9;
pub const LDPT_ADD_INPUT_FILE: c_int = 10;
pub const LDPT_MESSAGE: c_int = 11;
pub const LDPT_GET_INPUT_FILE: c_int = 12;
pub const LDPT_RELEASE_INPUT_FILE: c_int = 13;
pub const LDPT_ADD_INPUT_LIBRARY: c_int = 14;
pub const LDPT_OUTPUT_NAME: c_int = 15;
pub const LDPT_SET_EXTRA_LIBRARY_PATH: c_int = 16;
pub const LDPT_GNU_LD_VERSION: c_int = 17;
pub const LDPT_GET_VIEW: c_int = 18;
// The section-inspection and ordering tags occupy 19 through 24; this linker
// offers none of them, and a plugin that wants one does without.
pub const LDPT_GET_SYMBOLS_V2: c_int = 25;
pub const LDPT_GET_SYMBOLS_V3: c_int = 28;

/// What the link is producing, passed under [`LDPT_LINKER_OUTPUT`]. The
/// plugin uses it to decide whether a definition can be preempted at load
/// time, which changes what it may internalise.
pub const LDPO_REL: c_int = 0;
pub const LDPO_EXEC: c_int = 1;
pub const LDPO_DYN: c_int = 2;
pub const LDPO_PIE: c_int = 3;

/// Severity for the [`LDPT_MESSAGE`] callback.
pub const LDPL_INFO: c_int = 0;
pub const LDPL_WARNING: c_int = 1;
pub const LDPL_ERROR: c_int = 2;
pub const LDPL_FATAL: c_int = 3;

/// How a symbol a plugin declared was resolved by the rest of the link.
///
/// This is the whole point of the interface: the plugin cannot know which
/// definition won, and the linker cannot know what the bitcode means. These
/// nine values are the entire vocabulary between them, and
/// `gold-plugin.cpp:769-800` shows the mapping into LLVM's own
/// `lto::SymbolResolution` bits -- getting one wrong lets LTO delete a
/// definition a regular object still calls.
pub const LDPR_UNKNOWN: c_uint = 0;
pub const LDPR_UNDEF: c_uint = 1;
/// This file's definition wins and a regular object can see it.
pub const LDPR_PREVAILING_DEF: c_uint = 2;
/// This file's definition wins, and only IR references it.
pub const LDPR_PREVAILING_DEF_IRONLY: c_uint = 3;
/// Preempted by a definition in a regular object.
pub const LDPR_PREEMPTED_REG: c_uint = 4;
/// Preempted by a definition in another bitcode file.
pub const LDPR_PREEMPTED_IR: c_uint = 5;
/// A reference resolved against a definition in bitcode.
pub const LDPR_RESOLVED_IR: c_uint = 6;
/// A reference resolved against a definition in a regular object.
pub const LDPR_RESOLVED_EXEC: c_uint = 7;
/// A reference resolved against a definition in a shared library.
pub const LDPR_RESOLVED_DYN: c_uint = 8;
/// Wins, IR-only, but must still appear in the dynamic symbol table.
pub const LDPR_PREVAILING_DEF_IRONLY_EXP: c_uint = 9;

/// Symbol kinds a plugin declares through `add_symbols`.
pub const LDPK_DEF: c_uint = 0;
pub const LDPK_WEAKDEF: c_uint = 1;
pub const LDPK_UNDEF: c_uint = 2;
pub const LDPK_WEAKUNDEF: c_uint = 3;
pub const LDPK_COMMON: c_uint = 4;

/// Symbol visibility, mirroring `STV_*`.
pub const LDPV_DEFAULT: c_uint = 0;
pub const LDPV_PROTECTED: c_uint = 1;
pub const LDPV_INTERNAL: c_uint = 2;
pub const LDPV_HIDDEN: c_uint = 3;

/// One symbol, as declared by the plugin and resolved by the linker.
///
/// The plugin fills every field but `resolution` and hands the array to
/// `add_symbols`; the linker fills `resolution` when the plugin asks for it
/// back through `get_symbols`. Both sides keep the same array, so the layout
/// must match `plugin-api.h` exactly.
#[repr(C)]
pub struct Symbol {
    pub name: *mut c_char,
    pub version: *mut c_char,
    /// `def` in `plugin-api.h`. Bitfields were added around it in later
    /// revisions; the field is read as a whole `int` by every version.
    pub def: c_int,
    pub visibility: c_uint,
    pub size: u64,
    pub comdat_key: *mut c_char,
    pub resolution: c_int,
}

/// An input file offered to the plugin for claiming.
#[repr(C)]
pub struct InputFile {
    pub name: *const c_char,
    pub fd: c_int,
    pub offset: i64,
    pub filesize: i64,
    pub handle: *mut c_void,
}

/// One entry of the transfer vector.
///
/// `plugin-api.h` spells the payload as a union of every callback type. A
/// union of function pointers all of which are pointer-sized is a
/// pointer-sized field, so this carries the payload as `*mut c_void` and the
/// constructors below cast into it. That keeps the unsafe cast in one place
/// rather than spread across a union with fourteen arms.
#[repr(C)]
pub struct Value {
    pub tag: c_int,
    pub payload: *mut c_void,
}

/// The plugin's entry point. `LLVMgold.so` exports exactly this one symbol.
pub type OnLoad = unsafe extern "C" fn(*mut Value) -> Status;

/// Hooks the plugin registers with the linker.
pub type ClaimFile =
    unsafe extern "C" fn(*const InputFile, *mut c_int) -> Status;
pub type AllSymbolsRead = unsafe extern "C" fn() -> Status;
pub type Cleanup = unsafe extern "C" fn() -> Status;

/// Callbacks the linker offers the plugin.
pub type RegisterClaimFile = unsafe extern "C" fn(ClaimFile) -> Status;
pub type RegisterAllSymbolsRead =
    unsafe extern "C" fn(AllSymbolsRead) -> Status;
pub type RegisterCleanup = unsafe extern "C" fn(Cleanup) -> Status;
pub type AddSymbols =
    unsafe extern "C" fn(*mut c_void, c_int, *const Symbol) -> Status;
pub type GetSymbols =
    unsafe extern "C" fn(*const c_void, c_int, *mut Symbol) -> Status;
pub type AddInputFile = unsafe extern "C" fn(*const c_char) -> Status;
pub type AddInputLibrary = unsafe extern "C" fn(*const c_char) -> Status;
pub type SetExtraLibraryPath = unsafe extern "C" fn(*const c_char) -> Status;
pub type GetInputFile =
    unsafe extern "C" fn(*const c_void, *mut InputFile) -> Status;
pub type ReleaseInputFile = unsafe extern "C" fn(*const c_void) -> Status;
pub type GetView =
    unsafe extern "C" fn(*const c_void, *mut *mut c_void) -> Status;
pub type Message = unsafe extern "C" fn(c_int, *const c_char, ...);

/// The API version this linker speaks, passed under [`LDPT_API_VERSION`].
pub const API_VERSION: c_int = 1;
