// The plugin host these exercise is compiled in only with the `lto`
// feature, so without it there is nothing here to test.
#![cfg(feature = "lto")]

//! The LTO plugin boundary: detection of bitcode, and loading a real plugin.
//!
//! Bitcode is the one input shape a compiler produces that none of the
//! readers here can consume, so the link has to recognise it before it can
//! say anything useful about it. `clang -flto -c` used to fail as an
//! unrecognised input, which tells the user nothing about what to do.
//!
//! The loading tests are gated on the host actually having `LLVMgold.so`
//! installed. Where it is absent they print a note and return, because a
//! missing plugin is a property of the machine and not of this linker.

use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::{Mutex, OnceLock},
};

use common::which;
use xold::{
    input::Format,
    lto::{Output, Plugin, plugin},
};

mod common;

/// Bare LLVM bitcode, as `clang -flto -c` writes it.
const BITCODE: &[u8] = b"BC\xc0\xde\0\0\0\0";
/// The bitcode wrapper header, `0x0b17c0de` little-endian.
const WRAPPED: &[u8] = &[0xde, 0xc0, 0x17, 0x0b, 0, 0, 0, 0];

#[test]
fn both_bitcode_spellings_are_recognised() {
    assert_eq!(Format::detect(BITCODE), Some(Format::Bitcode));
    assert_eq!(Format::detect(WRAPPED), Some(Format::Bitcode));
    // And the neighbours it must not be confused with.
    assert_eq!(
        Format::detect(b"\x7fELF\x02\x01\x01\0"),
        Some(Format::Elf),
        "an ELF object is still ELF"
    );
    assert_eq!(Format::detect(b"BC\xc0"), None, "a truncated magic is not");
}

/// A real `clang -flto` object is detected as bitcode.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_real_lto_object_is_detected() {
    let Some((dir, obj)) = compile_bitcode("detect") else {
        return;
    };
    let bytes = fs::read(&obj).expect("read the compiled object");
    assert_eq!(
        Format::detect(&bytes),
        Some(Format::Bitcode),
        "clang -flto -c emits bitcode, not an object"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A bitcode input is consumed by LTO, not refused as an unknown file.
///
/// Linked on its own with no runtime, it compiles and then fails the way any
/// object with no entry point fails. That the diagnostic is about `_start`
/// and not about an unreadable input is the whole point: it says the bitcode
/// reached the link.
#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn a_bitcode_input_is_consumed_not_refused() {
    if plugin::resolve(None).is_none() {
        eprintln!("skipping consume: no LLVMgold.so on this host");
        return;
    }
    let Some((dir, obj)) = compile_bitcode("consume") else {
        return;
    };
    let out = dir.join("prog");
    let run = Command::new(common::xold_bin())
        .arg(&obj)
        .arg("-o")
        .arg(&out)
        .output()
        .expect("run xold");
    assert!(
        !run.status.success(),
        "an image with no entry point cannot link"
    );
    let err = String::from_utf8_lossy(&run.stderr);
    assert!(
        err.contains("entry symbol"),
        "the link got past the bitcode and failed on the entry: {err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// `resolve` prefers the named path over the conventional directory, because
/// a plugin only reads the bitcode its own LLVM version wrote.
#[test]
fn a_named_plugin_wins_over_the_conventional_directory() {
    let named = PathBuf::from("/nowhere/mine/LLVMgold.so");
    assert_eq!(
        plugin::resolve(Some(&named)),
        Some(named.clone()),
        "-plugin is honoured exactly as written"
    );
}

/// The host's plugin loads and registers the hook it must register.
#[test]
#[cfg_attr(miri, ignore = "Miri cannot dlopen a host shared object")]
fn the_host_plugin_loads_and_registers_its_hooks() {
    let Some(loaded) = host_plugin() else {
        eprintln!("skipping plugin load: no LLVMgold.so on this host");
        return;
    };
    let hooks = loaded.hooks();
    assert!(
        hooks.claim_file.is_some(),
        "a plugin that reads bitcode must register claim_file"
    );
    assert!(
        hooks.all_symbols_read.is_some(),
        "and the hook that runs codegen once resolution is done"
    );
    assert!(loaded.path().is_file(), "it reports where it came from");
}

/// A file that is not a plugin is refused, naming the reason.
#[test]
#[cfg_attr(miri, ignore = "Miri cannot dlopen a host shared object")]
fn a_file_that_is_not_a_plugin_is_refused() {
    let out = std::env::temp_dir().join("xold-lto-probe");
    let missing = PathBuf::from("/nonexistent/xold-plugin.so");
    let loaded = Plugin::load(&missing, &out, Output::Executable, &[]);
    let Err(err) = loaded else {
        panic!("a missing plugin must be an error");
    };
    let text = format!("{err}");
    assert!(
        text.contains("cannot load LTO plugin"),
        "the diagnostic names the failure: {text}"
    );
}

/// The host plugin, loaded once for the whole test binary.
///
/// An LTO plugin keeps process-global state -- `onload` installs it and the
/// module list accumulates across claims -- so it is loaded once and shared,
/// which is also how a real link uses it. Loading it per test would have each
/// one re-entering `onload` over the previous test's state.
fn host_plugin() -> Option<&'static Plugin> {
    static PLUGIN: OnceLock<Option<Plugin>> = OnceLock::new();
    PLUGIN
        .get_or_init(|| {
            let path = plugin::resolve(None)?;
            let out = std::env::temp_dir().join("xold-lto-probe");
            Plugin::load(&path, &out, Output::Executable, &[]).ok()
        })
        .as_ref()
}

/// Serialises the claims. The plugin is not reentrant, and cargo runs tests
/// on several threads.
fn claim_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    // A failing test poisons the lock; the ones after it are still worth
    // running, and each asserts on its own handle, so the poison is stepped
    // over rather than turned into a second failure that hides the first.
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

// --- fixtures --------------------------------------------------------------

/// Compiles a tiny translation unit to bitcode, or `None` when the host
/// clang cannot.
fn compile_bitcode(tag: &str) -> Option<(PathBuf, PathBuf)> {
    let clang = which("clang")?;
    let dir = std::env::temp_dir()
        .join(format!("xold_lto_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).ok()?;
    let src = dir.join("t.c");
    let obj = dir.join("t.o");
    fs::write(
        &src,
        b"int helper(void){return 7;}\nint main(void){return helper();}\n",
    )
    .ok()?;
    let built = Command::new(clang)
        .args(["-flto", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .ok()?
        .success();
    if !built {
        eprintln!("skipping lto {tag}: clang cannot emit bitcode");
        let _ = fs::remove_dir_all(&dir);
        return None;
    }
    // A clang configured to emit fat objects gives an ELF file, which is a
    // different shape than the one under test.
    if !fs::read(&obj).is_ok_and(|b| b.starts_with(b"BC\xc0\xde")) {
        eprintln!("skipping lto {tag}: this clang emits fat LTO objects");
        let _ = fs::remove_dir_all(&dir);
        return None;
    }
    Some((dir, obj))
}

/// The plugin reads a real bitcode file and declares the symbols in it.
///
/// This is the whole claim half of an LTO link: if the names arrive, the
/// linker can resolve them against the rest of the link and answer the
/// plugin's `get_symbols` with a verdict per name.
#[test]
#[cfg_attr(miri, ignore = "needs a host plugin and toolchain")]
fn the_plugin_claims_bitcode_and_declares_its_symbols() {
    let Some(loaded) = host_plugin() else {
        eprintln!("skipping claim: no LLVMgold.so on this host");
        return;
    };
    let Some((dir, obj)) = compile_bitcode("claim") else {
        return;
    };
    let _guard = claim_lock();
    let claimer = xold::lto::Claimer::new(loaded);
    let bytes = fs::read(&obj).expect("read the bitcode");
    let offer = claimer.offer(&obj, bytes).expect("the offer must not fail");
    let xold::lto::Offer::Claimed { symbols, .. } = offer else {
        panic!("the plugin must claim a bitcode file, got {offer:?}");
    };
    assert!(
        symbols >= 2,
        "the fixture defines `main` and `helper`, got {symbols} symbol(s)"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// An ELF object offered to the plugin is declined, so the ordinary readers
/// keep it. This is what lets every input be offered without a reader here.
#[test]
#[cfg_attr(miri, ignore = "needs a host plugin and toolchain")]
fn a_native_object_is_declined() {
    let Some(loaded) = host_plugin() else {
        eprintln!("skipping decline: no LLVMgold.so on this host");
        return;
    };
    let Some(clang) = which("clang") else {
        return;
    };
    let dir = std::env::temp_dir()
        .join(format!("xold_lto_decline_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create the work directory");
    let src = dir.join("n.c");
    let obj = dir.join("n.o");
    fs::write(&src, b"int native(void){return 1;}\n").expect("write source");
    let built = Command::new(clang)
        .args(["-fno-lto", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .is_ok_and(|s| s.success());
    assert!(built, "clang builds a native object");

    let _guard = claim_lock();
    let claimer = xold::lto::Claimer::new(loaded);
    let bytes = fs::read(&obj).expect("read the object");
    assert_eq!(
        claimer.offer(&obj, bytes).expect("the offer must not fail"),
        xold::lto::Offer::Declined,
        "a plugin must decline what it cannot read"
    );
    let _ = fs::remove_dir_all(&dir);
}

/// Windows-targeted bitcode is claimed exactly as the native kind is.
///
/// Bitcode carries its own target triple, so the same plugin serves a COFF
/// link and an ELF one; nothing in the claim depends on which. Running the
/// image it will eventually produce needs `wine`, which the COFF tests
/// already use -- but codegen is not wired yet, so this checks the half that
/// is: that the plugin reads the file and declares its symbols.
#[test]
#[cfg_attr(miri, ignore = "needs a host plugin and toolchain")]
fn windows_targeted_bitcode_is_claimed_too() {
    let Some(loaded) = host_plugin() else {
        eprintln!("skipping windows claim: no LLVMgold.so on this host");
        return;
    };
    let Some(clang) = which("clang") else {
        return;
    };
    let dir = std::env::temp_dir()
        .join(format!("xold_lto_win_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create the work directory");
    let src = dir.join("w.c");
    let obj = dir.join("w.o");
    fs::write(
        &src,
        b"int helper(void){return 7;}\nint main(void){return helper();}\n",
    )
    .expect("write source");
    let built = Command::new(clang)
        .args(["--target=x86_64-pc-windows-gnu", "-flto", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .is_ok_and(|s| s.success());
    if !built {
        eprintln!("skipping windows claim: clang has no windows target");
        let _ = fs::remove_dir_all(&dir);
        return;
    }
    let _guard = claim_lock();
    let claimer = xold::lto::Claimer::new(loaded);
    let bytes = fs::read(&obj).expect("read the bitcode");
    let offer = claimer.offer(&obj, bytes).expect("the offer must not fail");
    let xold::lto::Offer::Claimed { symbols, .. } = offer else {
        panic!("windows-targeted bitcode must be claimed, got {offer:?}");
    };
    assert!(symbols >= 2, "`main` and `helper` are declared");
    let _ = fs::remove_dir_all(&dir);
}

/// A plugin calls `LDPT_MESSAGE` through a variadic pointer, so the callback
/// has to read its two named arguments correctly when extra ones are passed.
///
/// `session::message` is declared without `...`, which is what keeps the
/// crate buildable on stable Rust: defining a C-variadic function needs the
/// unstable `c_variadic` feature. That is only sound because a callee reading
/// just the named parameters cannot observe how it was declared. This calls
/// it exactly as a plugin does -- through the variadic type, with a string
/// and a float among the arguments, since a float is what makes an x86-64
/// caller set `al` -- and checks that both named arguments still arrive.
///
/// The call runs in a child so the parent can read what reached stderr.
#[test]
fn the_message_callback_reads_its_named_arguments() {
    use std::{env, ffi::c_char, mem, os::raw::c_int};

    const VAR: &str = "XOLD_TEST_PLUGIN_MESSAGE";
    type Message = unsafe extern "C" fn(c_int, *const c_char, ...);

    if env::var_os(VAR).is_some() {
        // SAFETY: the two types differ only in the trailing `...`. Every
        // target this builds for passes the named arguments in the same
        // places either way, and the callee reads nothing else; see the
        // `session::message` documentation for the per-ABI argument.
        let call: Message =
            unsafe { mem::transmute(xold::lto::session::message as *const ()) };
        // SAFETY: the format string is a live NUL-terminated literal, and
        // the variadic arguments are plain values passed by copy.
        unsafe {
            call(
                2, // LDPL_ERROR
                c"plugin reported %s at %d (%f)".as_ptr(),
                c"a-symbol".as_ptr(),
                42_i32,
                1.5_f64,
            );
        }
        return;
    }

    let exe = env::current_exe().expect("the test binary has a path");
    let out = Command::new(exe)
        .arg("the_message_callback_reads_its_named_arguments")
        .arg("--exact")
        .arg("--nocapture")
        .env(VAR, "1")
        .output()
        .expect("the child test binary runs");
    let err = String::from_utf8_lossy(&out.stderr);
    // `level` decides the severity and `format` the text: seeing both proves
    // each named argument arrived, rather than a register the varargs took.
    assert!(
        err.contains("plugin reported %s at %d (%f)"),
        "the format string must be reported as written, got: {err}"
    );
    assert!(
        err.contains("error"),
        "level 2 must be reported as an error, got: {err}"
    );
}
