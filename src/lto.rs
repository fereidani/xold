//! Link-time optimisation through the GNU linker-plugin interface.
//!
//! Bitcode is not a format this linker reads. `clang -flto` and
//! `gcc -flto` emit LLVM IR, and compiling it needs LLVM itself -- an
//! unstable C++ API that lld can call because lld ships inside LLVM. A
//! linker outside that tree has one supported route, and it is the one gold
//! and `ld.bfd` take: load a plugin that wraps LLVM and speak the C ABI
//! binutils defined for it.
//!
//! That choice buys three things worth the indirection. LLVM stays out of
//! this crate's build entirely -- one `dlopen`, no link-time dependency. The
//! interface carries a full per-symbol resolution rather than the
//! "must-preserve" name list the older `libLTO` C API takes, so a definition
//! preempted by another bitcode file is distinguishable from one preempted by
//! a regular object. And it is the interface compiler drivers already drive:
//! `-plugin` and `-plugin-opt=` are what `clang -flto` passes.
//!
//! The link splits in two around the plugin. Every input is offered to
//! `claim_file`; a claimed one is the plugin's, and its symbols arrive
//! through `add_symbols` rather than from any reader here. Once archive
//! extraction has reached its fixpoint the plugin is told so, asks for each
//! symbol's resolution through `get_symbols`, compiles, and hands back native
//! objects through `add_input_file`. Those objects then join the link as any
//! other object would.

pub mod api;
pub mod claim;
pub mod dynlib;
pub mod link;
pub mod plugin;
pub mod resolve;
pub mod session;

use std::path::{Path, PathBuf};

pub use claim::{Claimer, Offer};
pub use link::{Regular, resolve_claimed};
pub use plugin::{Hooks, Output, Plugin};
pub use resolve::{Facts, Winner, resolution};

use crate::error::{Error, Result};

/// Runs an LTO link: claim, resolve, compile.
///
/// Returns the native objects the plugin produced. They are ordinary
/// relocatable objects and join the link exactly as any other input does --
/// which is why nothing above this module needs to know bitcode existed.
///
/// The order of the three steps is not a choice. The plugin cannot decide
/// what to internalise until it knows how every name resolved, and the link
/// cannot know that until every input has been seen; so the claim covers all
/// inputs first, the verdicts are computed second, and only then is the
/// plugin told to compile.
///
/// # Errors
///
/// Reports a plugin that cannot be found or loaded, a bitcode file the plugin
/// recognised but could not read, and a codegen failure the plugin reports.
pub fn compile(job: &Job<'_>) -> Result<Compiled> {
    let Some(path) = plugin::resolve(job.named_plugin) else {
        return Err(Error::CommandLine(
            "an input is LLVM IR bitcode, but no LTO plugin was found: pass \
             -plugin /path/to/LLVMgold.so, or build with -fno-lto"
                .to_string(),
        ));
    };
    let plugin = Plugin::load(&path, job.output, job.kind, job.options)?;
    let claimer = Claimer::new(&plugin);
    let mut regular = Regular::default();
    let mut claimed = 0usize;
    for input in job.inputs {
        let bytes = std::fs::read(input).map_err(Error::Io)?;
        match claimer.offer(input, bytes)? {
            Offer::Claimed { .. } => claimed = claimed.saturating_add(1),
            // The plugin does not want it, so it is a regular input and its
            // names are the ones LTO must not assume it has seen every use
            // of. Re-reading is cheap next to compiling, and keeps the claim
            // loop from having to hold every input in memory at once.
            Offer::Declined => {
                let bytes = std::fs::read(input).map_err(Error::Io)?;
                regular.add_object(&bytes)?;
            }
        }
    }
    if claimed == 0 {
        return Ok(Compiled {
            plugin,
            objects: Vec::new(),
        });
    }
    for name in job.pinned {
        regular.pin(name);
    }
    resolve_claimed(&regular, job.export_all)?;
    run_codegen(&plugin)?;
    let objects = session::session()
        .lock()
        .map_err(|_| Error::Format("LTO session lock was poisoned"))
        .map(|state| state.added.clone())?;
    Ok(Compiled { plugin, objects })
}

/// The result of an LTO compile, holding the plugin open over it.
///
/// The objects are the plugin's temporary files. It deletes them in its
/// cleanup hook, so the plugin stays loaded until the link has read them and
/// [`Compiled::finish`] says so; dropping this without finishing leaves them
/// behind rather than removing files the link may still be reading.
pub struct Compiled {
    plugin: Plugin,
    /// The native objects to link, in the order the plugin announced them.
    pub objects: Vec<PathBuf>,
}

impl Compiled {
    /// Tells the plugin the link is done with its objects.
    ///
    /// # Errors
    ///
    /// Reports a cleanup the plugin refused, which leaves its temporary files
    /// on disk.
    pub fn finish(self) -> Result<()> {
        let Some(cleanup) = self.plugin.hooks().cleanup else {
            return Ok(());
        };
        // SAFETY: the hook takes no arguments, and every object it will
        // remove has been read by now -- that is what calling this means.
        let status = unsafe { cleanup() };
        if !plugin::ok(status) {
            return Err(Error::CommandLine(format!(
                "LTO plugin {} failed to clean up (status {status})",
                self.plugin.path().display()
            )));
        }
        Ok(())
    }
}

/// What an LTO link needs to know about the link around it.
pub struct Job<'a> {
    /// `-plugin`, honoured as written.
    pub named_plugin: Option<&'a Path>,
    /// `-plugin-opt=` values, passed through unchanged.
    pub options: &'a [String],
    pub output: &'a Path,
    pub kind: Output,
    /// Every input, in command-line order. Order decides which definition
    /// prevails, so it has to be the order the driver read them in.
    pub inputs: &'a [PathBuf],
    /// Names the command line pinned: `-u` and the entry symbol.
    pub pinned: &'a [&'a [u8]],
    /// Whether the image publishes its definitions.
    pub export_all: bool,
}

/// Tells the plugin that resolution is final, which is what makes it compile.
fn run_codegen(plugin: &Plugin) -> Result<()> {
    let Some(all_read) = plugin.hooks().all_symbols_read else {
        return Err(Error::CommandLine(format!(
            "LTO plugin {} registered no all-symbols-read hook, so it can \
             compile nothing",
            plugin.path().display()
        )));
    };
    // SAFETY: the hook takes no arguments. Every symbol the plugin declared
    // now carries a resolution, which is the state it reads through
    // `get_symbols` during this call.
    let status = unsafe { all_read() };
    if !plugin::ok(status) {
        return Err(Error::CommandLine(format!(
            "LTO plugin {} failed to compile the bitcode (status {status})",
            plugin.path().display()
        )));
    }
    Ok(())
}
