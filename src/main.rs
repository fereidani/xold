//! xold command-line entry point.
//!
//! A thin wrapper over the library: it collects the input paths, an optional
//! output path (`-o`, default `a.out`), an optional entry symbol (`--entry`,
//! default `_start`), and selects between a static executable (the default),
//! a shared object (`-shared`), and a dynamic executable (when a shared
//! object is given as a dependency, or `--dynamic-exec` is set).
//!
//! `-l NAME` resolves a shared dependency the way the system linker does: it
//! searches the `-L` directories and then the host defaults for `libNAME.so`,
//! falling back to the highest versioned `libNAME.so.M` and then to
//! `libNAME.a`. `--sysroot` re-roots those host defaults, which a link that
//! targets another architecture needs so that `-l` does not reach the build
//! machine's own libraries.
//!
//! What a name resolves to may be a GNU ld script rather than an object --
//! `-lc` on a glibc system finds `/usr/lib64/libc.so`, which is text naming
//! three files -- so the input list is expanded before anything is opened; see
//! [`InputList`] and [`xold::script`].

mod cli;

use std::{
    env,
    ffi::OsString,
    path::{Path, PathBuf},
    process::ExitCode,
};

use cli::{Options, Request, emulation, expand_response_files, parse};
// Its own statement rather than a member of the group above: a `use` group
// cannot carry an attribute on one of its names.
#[cfg(feature = "lto")]
use xold::lto;
use xold::{
    buildid::BuildId,
    coff::link_coff,
    dynamic::LinkMode,
    elf::constants::ET_DYN,
    error::{Error, Result},
    icf::IcfMode,
    input::{Format, Input},
    linker::{Link, link_image},
    macho::{Dylib, LinkOptions as MachOLinkOptions, link_macho_with_options},
    script::{self, Search},
    startlib,
    versionscript::{self, VersionScript},
};

/// A link allocates heavily across several threads (per-file parse results,
/// per-section work items, the growing symbol and image buffers), which is the
/// workload glibc's allocator handles worst. mimalloc is set here, on the
/// binary, rather than in the library: choosing a global allocator is the
/// application's call, not a library's.
///
/// Miri gets the default allocator instead. mimalloc is a C library, and Miri
/// interprets Rust rather than calling foreign code, so every allocation the
/// binary made would abort the run before it reached anything worth checking.
#[cfg(not(miri))]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> ExitCode {
    // This process exists only to link, so it is free to trade transparent
    // huge pages away on a link large enough to be hurt by them. The library
    // never decides this for a host that did not ask.
    xold::pool::allow_huge_page_tuning();
    let raw: Vec<OsString> = env::args_os().skip(1).collect();
    let args = match expand_response_files(&raw) {
        Ok(args) => args,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::FAILURE;
        }
    };
    match parse(&args) {
        Ok(Request::Print(text)) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Ok(Request::Link(opts)) => link_detached(&opts),
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::FAILURE
        }
    }
}

/// Runs the link, forked so the caller is released at publish.
///
/// The fork happens before any thread exists: the caller gets its prompt
/// back the moment the image is published, and the child unmaps and frees
/// off the caller's clock. mold and wild do the same by default;
/// `--no-fork` keeps the link in this process.
fn link_detached(opts: &Options) -> ExitCode {
    // A dynamic Mach-O link is ad-hoc signed after the writer publishes it.
    // Keep that final mutation in the foreground so clang never observes the
    // pre-signature image released by `write_output`.
    if !opts.no_fork && !opts.macho_dynamic {
        xold::detach::fork_child();
    }
    match with_threads(opts, || run(opts)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("xold: {err}");
            ExitCode::FAILURE
        }
    }
}
/// Runs `job` on a pool of the size `--threads` asked for.
///
/// Without the option the link sizes its own pool from the workload (see
/// `xold::pool`); with it the caller has already answered that question, and
/// the library leaves a pool the caller installed alone. A pool that will not
/// build is no reason to fail a link that would otherwise succeed, so the job
/// runs on the ambient one instead.
fn with_threads<T: Send>(opts: &Options, job: impl FnOnce() -> T + Send) -> T {
    let Some(threads) = opts.threads else {
        return job();
    };
    match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
        Ok(pool) => pool.install(job),
        Err(_) => job(),
    }
}

/// Links the inputs the command line named.
fn run(opts: &Options) -> Result<()> {
    let mut list = InputList::default();
    list.collect(opts)?;
    let inputs = list.scan;
    check_command_line(opts, inputs)?;
    if inputs.bitcode {
        return link_bitcode(opts, &list);
    }
    let files = list.views();
    // Mach-O inputs take a separate link path; the ELF linker cannot consume
    // them.
    if inputs.macho {
        return link_macho_image(opts, &list, &files);
    }
    // COFF inputs (Windows objects) take a separate link path that produces a
    // PE32+ image; the ELF linker cannot consume them. Windows C programs
    // conventionally enter at `main` (no underscore on x86_64). With `-shared`
    // the image is a DLL carrying an export directory.
    if inputs.coff {
        refuse_macho_output_options(opts, "COFF")?;
        refuse_elf_only(opts, "COFF")?;
        let entry = default_entry(opts, b"main");
        return link_coff(&files, &opts.output, entry, opts.shared);
    }
    link_elf(opts, &files, inputs.shared)
}

/// The ELF link itself, over an input list that is already settled.
fn link_elf(opts: &Options, files: &[Input<'_>], shared: bool) -> Result<()> {
    refuse_macho_output_options(opts, "ELF")?;
    // The version script is read here rather than during parsing: a link that
    // never reaches the ELF path has no use for it, and a read that fails
    // should report as a link error naming the file.
    let script = read_version_script(opts)?;
    let undefined: Vec<&[u8]> =
        opts.undefined.iter().map(String::as_bytes).collect();
    link_image(&Link {
        inputs: files,
        output: &opts.output,
        mode: mode_of(opts, shared),
        // A shared object has no entry point unless one is asked for, so the
        // default is not applied to it. `--entry` written on the command line
        // is honoured whatever the output kind, as it is in lld: accepting it
        // and dropping it is the "accepted but ignored" behaviour this
        // linker's unknown-option message promises never happens.
        entry: if opts.shared && !opts.entry_given {
            b""
        } else {
            opts.entry.as_bytes()
        },
        soname: opts.soname.as_deref().map(str::as_bytes),
        interpreter: (opts.dynamic_exec || shared)
            .then_some(opts.interpreter.as_bytes()),
        gc: opts.gc_sections,
        export_dynamic: opts.export_dynamic,
        icf: opts.icf,
        relax: opts.relax,
        no_undefined: opts.no_undefined,
        z: opts.z,
        pie: opts.pie,
        strip: opts.strip,
        hash_style: opts.hash_style,
        rpath: opts.rpath.as_bytes(),
        build_id: (opts.build_id != BuildId::None).then_some(&opts.build_id),
        undefined: &undefined,
        version_script: script.as_ref(),
        undefined_version: opts.undefined_version,
    })
}

/// Reads ld64's exact-symbol export allow-list.
///
/// The file format rustc writes is one symbol per line. ld64 also accepts
/// wildcard patterns, but treating one as a literal would silently hide or
/// publish the wrong API, so this implementation refuses that extension.
fn read_macho_exported_symbols(opts: &Options) -> Result<Option<Vec<String>>> {
    let Some(path) = opts.macho_exported_symbols.as_deref() else {
        return Ok(None);
    };
    let text = std::fs::read_to_string(path)?;
    let mut names = Vec::new();
    for line in text.lines() {
        let name = line.trim();
        if name.is_empty() {
            continue;
        }
        if name.bytes().any(|b| matches!(b, b'*' | b'?' | b'[')) {
            return Err(Error::CommandLine(format!(
                "wildcards in -exported_symbols_list are not implemented: \
                 `{name}`"
            )));
        }
        names.push(name.to_string());
    }
    names.sort_unstable();
    names.dedup();
    Ok(Some(names))
}

/// Reads and parses the `--version-script` file, when one was named.
fn read_version_script(opts: &Options) -> Result<Option<VersionScript>> {
    let Some(path) = opts.version_script.as_deref() else {
        return Ok(None);
    };
    let text = std::fs::read_to_string(path)?;
    versionscript::parse(path, &text).map(Some)
}

/// Checks the options that describe the link against what the inputs turned
/// out to be.
///
/// Both checks are made here rather than during parsing because neither can
/// be answered from the command line alone: `-m` names a machine, and
/// `-static` promises no shared object, and only the classified inputs say
/// what was really given.
fn check_command_line(opts: &Options, scan: InputScan) -> Result<()> {
    if opts.static_link && scan.shared {
        return Err(Error::CommandLine(
            "-static was given, but an input is a shared object: a static \
             image cannot depend on one"
                .into(),
        ));
    }
    if opts.pie == Some(true) && opts.static_link {
        return Err(Error::CommandLine(
            "-pie and -static ask for different images: a static image has \
             no loader to place it at a random base"
                .into(),
        ));
    }
    if opts.static_link && opts.shared {
        return Err(Error::CommandLine(
            "-static and -shared ask for different images".into(),
        ));
    }
    let Some(name) = opts.emulation.as_deref() else {
        return Ok(());
    };
    emulation::check(name, machine_of(scan)).map_err(Error::CommandLine)
}

/// What the classified inputs say the link's target is.
const fn machine_of(scan: InputScan) -> emulation::Machine {
    if scan.macho {
        emulation::Machine::MachO
    } else if scan.coff {
        emulation::Machine::Coff
    } else if let Some(machine) = scan.machine {
        emulation::Machine::Elf(machine)
    } else {
        emulation::Machine::Unknown
    }
}

/// Refuses the options the ELF path implements and `format` does not.
///
/// Accepting one and dropping it is the behaviour [`unknown_option`] promises
/// never happens: the image would differ from what the command line asked
/// for, quietly. Refusing says which option and which format.
fn refuse_elf_only(opts: &Options, format: &'static str) -> Result<()> {
    if opts.gc_sections {
        return Err(unsupported("--gc-sections".into(), format));
    }
    if opts.icf != IcfMode::None {
        return Err(unsupported("--icf".into(), format));
    }
    if opts.relax && opts.relax_named {
        return Err(unsupported("--relax".into(), format));
    }
    if opts.soname.is_some() {
        return Err(unsupported("-soname".into(), format));
    }
    if opts.pie.is_some() {
        return Err(unsupported("-pie/-no-pie".into(), format));
    }
    Ok(())
}

/// Links Mach-O inputs into an executable or a dylib, then signs the result.
///
/// Darwin executables conventionally enter at `_main`. A dylib has no entry
/// point and carries `LC_ID_DYLIB` instead, so it needs an install name: the
/// one `-install_name` gave, or the output path when the command line left it
/// to the linker.
fn link_macho_image(
    opts: &Options,
    list: &InputList,
    files: &[Input<'_>],
) -> Result<()> {
    refuse_elf_only(opts, "Mach-O")?;
    if opts.shared {
        return Err(unsupported("-shared".into(), "Mach-O"));
    }
    if opts.macho_install_name.is_some() && !opts.macho_dylib {
        return Err(unsupported("-install_name".into(), "MH_EXECUTE"));
    }
    if opts.macho_exported_symbols.is_some() && !opts.macho_dynamic {
        return Err(unsupported(
            "-exported_symbols_list".into(),
            "static Mach-O",
        ));
    }
    let entry = if opts.macho_dylib {
        b"".as_slice()
    } else {
        default_entry(opts, b"_main")
    };
    let install_name = if opts.macho_dylib {
        Some(opts.macho_install_name.as_deref().map_or_else(
            || {
                opts.output.to_str().ok_or_else(|| {
                    Error::CommandLine(
                        "Mach-O dylib output path is not valid UTF-8; use \
                         -install_name to give LC_ID_DYLIB a name"
                            .into(),
                    )
                })
            },
            Ok,
        )?)
    } else {
        None
    };
    let exported_symbols = read_macho_exported_symbols(opts)?;
    let dylibs: Vec<Dylib<'_>> = list
        .dylibs
        .iter()
        .map(|dylib| Dylib {
            install_name: &dylib.install_name,
            exports: &dylib.exports,
        })
        .collect();
    link_macho_with_options(
        files,
        &opts.output,
        entry,
        &MachOLinkOptions {
            arch: opts.macho_arch,
            dynamic: opts.macho_dynamic,
            dylib: opts.macho_dylib,
            install_name,
            exported_symbols: exported_symbols.as_deref(),
            platform: opts.macho_platform,
            dead_strip: opts.macho_dead_strip,
            dylibs: &dylibs,
        },
    )?;
    ad_hoc_sign(&opts.output)
}

/// The entry names this linker defaults to, one per output format: ELF
/// enters at `_start`, COFF at `main`, and Mach-O at `_main`.
#[cfg(feature = "lto")]
const DEFAULT_ENTRIES: [&[u8]; 3] = [b"_start", b"main", b"_main"];

/// Links a set of inputs that includes bitcode.
///
/// The plugin compiles the bitcode into ordinary relocatable objects, and
/// those take its place in the input list. Everything after that is the
/// normal ELF link: nothing downstream needs to know bitcode was ever here,
/// which is the property that keeps LTO out of the rest of the linker.
///
/// The objects go in where the first bitcode input was, rather than at the
/// end. Input order decides which archive member answers a reference, and the
/// code the plugin compiled came from that position on the command line.
/// Refuses bitcode in a build compiled without the `lto` feature.
///
/// The alternative is the reader's own "unrecognised input", which tells the
/// user nothing about what to do: the file is fine, this binary just has no
/// plugin host to compile it with.
#[cfg(not(feature = "lto"))]
const fn link_bitcode(_opts: &Options, _list: &InputList) -> Result<()> {
    Err(Error::Format(
        "input holds LLVM bitcode, and this build of xold was compiled \
         without the `lto` feature that loads a plugin to compile it: \
         rebuild xold with the feature, or build the input with -fno-lto",
    ))
}

#[cfg(feature = "lto")]
fn link_bitcode(opts: &Options, list: &InputList) -> Result<()> {
    let inputs: Vec<PathBuf> = list.paths().map(Path::to_path_buf).collect();
    let kind = if opts.shared {
        lto::Output::Shared
    } else if opts.pie == Some(true) {
        lto::Output::Pie
    } else {
        lto::Output::Executable
    };
    // The entry symbol and every `-u` name are pinned: nothing in any object
    // references them, and LTO deletes what nothing reaches.
    //
    // Which name that is depends on the output format, and the format comes
    // from the bitcode's own target triple -- which only the plugin can read,
    // and only while compiling. The entry cannot be known before the compile
    // that needs it pinned, so every default this linker would use is pinned
    // instead. Pinning a name no input defines costs nothing; failing to pin
    // the real one deletes the program, which is what an unpinned `main`
    // looked like: an object with no code in it at all.
    let mut pinned: Vec<&[u8]> =
        opts.undefined.iter().map(String::as_bytes).collect();
    if opts.entry_given {
        pinned.push(opts.entry.as_bytes());
    } else if !opts.shared {
        pinned.extend_from_slice(&DEFAULT_ENTRIES);
    }
    let groups: Vec<(&Path, &[u8])> = list.groups().collect();
    let compiled = lto::compile(&lto::Job {
        named_plugin: opts.lto_plugin.as_deref(),
        options: &opts.lto_plugin_opts,
        output: &opts.output,
        kind,
        inputs: &inputs,
        pinned: &pinned,
        groups: &groups,
        export_all: opts.shared || opts.export_dynamic,
    })?;
    let files = list.views_with_lto(&compiled.objects);
    let result = link_compiled(opts, &files, list, &compiled.objects);
    // The plugin's objects are temporary files it removes here. Cleanup runs
    // whether or not the link succeeded, and a link error outranks a cleanup
    // one: the first says why no image was written.
    let swept = compiled.finish();
    result.and(swept)
}

/// Links the objects LTO produced, on the path their own format selects.
///
/// Bitcode carries its target triple, so what comes back may be ELF, COFF or
/// Mach-O regardless of what the host is. The input scan could not have known
/// -- it saw only bitcode -- so the choice is made here, from the first
/// object the plugin returned.
#[cfg(feature = "lto")]
fn link_compiled(
    opts: &Options,
    files: &[Input<'_>],
    list: &InputList,
    objects: &[PathBuf],
) -> Result<()> {
    let mut head = [0u8; HEAD_LEN];
    let format = objects
        .first()
        .and_then(|path| read_head(path, &mut head))
        .and_then(|n| head.get(..n))
        .and_then(Format::detect);
    match format {
        Some(Format::Coff) => {
            refuse_macho_output_options(opts, "COFF")?;
            refuse_elf_only(opts, "COFF")?;
            let entry = default_entry(opts, b"main");
            link_coff(files, &opts.output, entry, opts.shared)
        }
        Some(Format::MachO) => link_macho_image(opts, list, files),
        // ELF, and anything the readers will reject by name themselves.
        _ => link_elf(opts, files, list.scan.shared),
    }
}

/// Refuses Mach-O output controls when another format's inputs selected its
/// writer. These options change the image and therefore cannot be ignored.
/// Refuses Mach-O output controls when another format's inputs selected its
/// writer. These options change the image and therefore cannot be ignored.
fn refuse_macho_output_options(
    opts: &Options,
    format: &'static str,
) -> Result<()> {
    let option = if opts.macho_dylib {
        Some("-dylib")
    } else if opts.macho_install_name.is_some() {
        Some("-install_name")
    } else if opts.macho_exported_symbols.is_some() {
        Some("-exported_symbols_list")
    } else {
        None
    };
    option.map_or(Ok(()), |name| Err(unsupported(name.into(), format)))
}

/// The error for an option this linker implements for ELF and not for
/// `format`.
fn unsupported(option: String, format: &'static str) -> Error {
    Error::unsupported_option(option, format)
}

/// Which image the request describes.
const fn mode_of(opts: &Options, has_shared_input: bool) -> LinkMode {
    if opts.static_link {
        LinkMode::Static
    } else if opts.shared {
        LinkMode::Shared
    } else if opts.dynamic_exec || has_shared_input {
        LinkMode::DynExec
    } else {
        LinkMode::Static
    }
}

/// Reads every member of a `--start-lib` group. The read is whole-file: the
/// group is serialised into one archive, so each member's bytes end up
/// resident exactly once, in that archive.
fn read_group(paths: &[PathBuf]) -> Result<Vec<(PathBuf, Vec<u8>)>> {
    paths
        .iter()
        .map(|path| Ok((path.clone(), std::fs::read(path)?)))
        .collect()
}

/// The entry symbol for a non-ELF link path: the caller's `--entry` unless it
/// is still the ELF default, in which case the platform convention applies.
fn default_entry<'a>(opts: &'a Options, platform: &'a [u8]) -> &'a [u8] {
    if opts.entry.is_empty() || opts.entry == "_start" {
        platform
    } else {
        opts.entry.as_bytes()
    }
}

/// What the input set contains, and therefore which link path runs.
/// Each field answers a separate question about the inputs, so grouping them
/// would only hide what the scan found; the tree allows the lint on the other
/// flag bags for the same reason.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Default)]
struct InputScan {
    /// Some input is a Mach-O object.
    macho: bool,
    /// Some input is a bare COFF object. PE images (`MZ`) are not link inputs.
    coff: bool,
    /// Some input is an `ET_DYN` shared object, which selects a dynamic
    /// executable link even without `--dynamic-exec`.
    shared: bool,
    /// The `e_machine` of the first ELF input, which is what `-m` is checked
    /// against. Inputs that disagree with each other are the link's own
    /// error, reported where the machine is derived.
    machine: Option<u16>,
    /// Some input is LLVM IR bitcode, which only the LTO plugin can read.
    bitcode: bool,
}

/// Bytes of each input read to classify it: enough for any format magic, for
/// the ELF `e_type` field at offset 16, and to tell script text from an
/// object.
const HEAD_LEN: usize = 64;

/// How deep a linker script may name another before this gives up.
///
/// A bound rather than a cycle check, for the reason [`RESPONSE_DEPTH`] is:
/// the depth a real toolchain reaches is one, and a script that names itself
/// is the only way past this.
const SCRIPT_DEPTH: usize = 8;

/// One input after `-l` resolution, linker-script expansion and
/// `--start-lib` grouping.
enum Entry {
    /// A file the link opens by path.
    File(Resolved),
    /// A `--start-lib` group, already packed into an archive in memory.
    Group { name: PathBuf, bytes: Vec<u8> },
}

/// One input after `-l` resolution and linker-script expansion.
struct Resolved {
    path: PathBuf,
    /// Whether a `-l` search found it, which decides the `DT_NEEDED` spelling
    /// of a library carrying no `DT_SONAME`.
    from_l: bool,
    /// Whether an `AS_NEEDED` list named it. See [`Input::AsNeeded`].
    as_needed: bool,
    /// Whether the file holds LLVM IR bitcode, from the same head bytes the
    /// format scan already read. Recorded rather than re-derived: the fold
    /// that replaces these with compiled objects would otherwise open every
    /// input a second time.
    bitcode: bool,
}

/// The settled input list the link runs on, with the classification that
/// chooses the link path.
///
/// Building the two together is what keeps each input to a single read: the
/// [`HEAD_LEN`] bytes that say whether a file is Mach-O, COFF, a shared object
/// or script text answer both questions at once, and a link over a few
/// thousand objects would otherwise open every one of them twice before
/// starting.
#[derive(Default)]
struct InputList {
    files: Vec<Entry>,
    scan: InputScan,
    /// Install names read from Darwin `.tbd` stubs, in command-line order.
    /// Aliases such as `libc.tbd` and `libm.tbd` commonly name libSystem, so
    /// duplicates are removed as they are collected.
    dylibs: Vec<DylibInput>,
}

/// One dynamic library collected from a TAPI stub or a direct `.dylib`.
struct DylibInput {
    install_name: String,
    exports: Vec<String>,
}

impl InputList {
    /// Resolves every command-line input, expanding the linker scripts among
    /// them.
    fn collect(&mut self, opts: &Options) -> Result<()> {
        let search = Search {
            paths: &opts.search,
            sysroot: opts.sysroot.as_deref(),
        };
        let mut i = 0;
        while let Some(path) = opts.inputs.get(i) {
            // A --start-lib group enters as one unit: the archive the group
            // packed into, placed where its first member stood.
            if let Some(&(_, end)) =
                opts.lib_groups.iter().find(|(start, _)| *start == i)
            {
                let members = read_group(&opts.inputs[i..end])?;
                let bytes = startlib::archive(&members)?;
                self.add_group(&members, &bytes);
                i = end;
                continue;
            }
            let file = Resolved {
                path: path.clone(),
                from_l: opts.from_l.get(i).copied().unwrap_or(false),
                as_needed: opts.as_needed.get(i).copied().unwrap_or(false),
                bitcode: false,
            };
            self.add(file, &search, SCRIPT_DEPTH)?;
            i += 1;
        }
        Ok(())
    }

    /// Every input with a path, in the order the command line named them.
    /// Order is what decides which definition prevails, so the plugin must be
    /// offered them in it.
    #[cfg(feature = "lto")]
    fn paths(&self) -> impl Iterator<Item = &Path> {
        self.files.iter().filter_map(|entry| match entry {
            Entry::File(file) => Some(&*file.path),
            // A `--start-lib` group has no path of its own; its members are
            // held in memory, and offering one to the plugin needs the group
            // reader that lazy bitcode extraction will bring.
            Entry::Group { .. } => None,
        })
    }

    /// The `--start-lib` groups, as archives the LTO sweep can search.
    #[cfg(feature = "lto")]
    fn groups(&self) -> impl Iterator<Item = (&Path, &[u8])> {
        self.files.iter().filter_map(|entry| match entry {
            Entry::Group { name, bytes } => {
                Some((name.as_path(), bytes.as_slice()))
            }
            Entry::File(_) => None,
        })
    }

    /// The list as the linker takes it, with every bitcode input replaced by
    /// the objects the LTO plugin compiled from it.
    ///
    /// The replacements are spliced in at the first bitcode input rather than
    /// appended. Input order decides which archive member answers a
    /// reference, and the compiled code came from that position.
    #[cfg(feature = "lto")]
    fn views_with_lto<'a>(&'a self, objects: &'a [PathBuf]) -> Vec<Input<'a>> {
        let mut out = Vec::with_capacity(self.files.len() + objects.len());
        let mut spliced = false;
        for entry in &self.files {
            match entry {
                Entry::File(file) if file.bitcode => {
                    if !spliced {
                        spliced = true;
                        out.extend(
                            objects.iter().map(|o| Input::Path(o.as_path())),
                        );
                    }
                }
                Entry::File(file) => out.push(Self::view_of(file)),
                Entry::Group { name, bytes } => {
                    out.push(Input::Memory { name, bytes });
                }
            }
        }
        out
    }

    /// One file entry as the linker takes it.
    #[cfg(feature = "lto")]
    fn view_of(file: &Resolved) -> Input<'_> {
        match (file.as_needed, file.from_l) {
            (true, from_l) => Input::AsNeeded {
                path: &file.path,
                from_l,
            },
            (false, true) => Input::Library(&file.path),
            (false, false) => Input::Path(&file.path),
        }
    }

    /// The list as the linker takes it, with the `-l`-found and `AS_NEEDED`
    /// inputs marked.
    fn views(&self) -> Vec<Input<'_>> {
        self.files
            .iter()
            .map(|f| match f {
                Entry::File(f) => match (f.as_needed, f.from_l) {
                    (true, from_l) => Input::AsNeeded {
                        path: &f.path,
                        from_l,
                    },
                    (false, true) => Input::Library(&f.path),
                    (false, false) => Input::Path(&f.path),
                },
                Entry::Group { name, bytes } => Input::Memory { name, bytes },
            })
            .collect()
    }

    /// Adds one input, replacing it with what it names when it turns out to be
    /// a linker script.
    ///
    /// An unreadable input is kept rather than reported here: the link path it
    /// selects opens it properly and says why it could not.
    fn add(
        &mut self,
        mut file: Resolved,
        search: &Search<'_>,
        depth: usize,
    ) -> Result<()> {
        if file.path.extension().and_then(|ext| ext.to_str()) == Some("dylib") {
            let install_name = search
                .sysroot
                .and_then(|root| file.path.strip_prefix(root).ok())
                .map_or_else(
                    || file.path.to_string_lossy().into_owned(),
                    |path| format!("/{}", path.display()),
                );
            if !self
                .dylibs
                .iter()
                .any(|dylib| dylib.install_name == install_name)
            {
                self.dylibs.push(DylibInput {
                    install_name,
                    exports: Vec::new(),
                });
            }
            return Ok(());
        }
        if file.path.extension().and_then(|ext| ext.to_str()) == Some("tbd") {
            let stub = xold::macho::tapi::parse(&std::fs::read_to_string(
                &file.path,
            )?)?;
            if let Some(existing) = self
                .dylibs
                .iter_mut()
                .find(|dylib| dylib.install_name == stub.install_name)
            {
                existing.exports.extend(stub.exports);
                existing.exports.sort_unstable();
                existing.exports.dedup();
            } else {
                self.dylibs.push(DylibInput {
                    install_name: stub.install_name,
                    exports: stub.exports,
                });
            }
            return Ok(());
        }
        let mut buf = [0u8; HEAD_LEN];
        let head = read_head(&file.path, &mut buf)
            .and_then(|n| buf.get(..n))
            .unwrap_or(&[]);
        match Format::detect(head) {
            Some(Format::MachO) => self.scan.macho = true,
            Some(Format::Coff) => self.scan.coff = true,
            Some(Format::Elf) => {
                self.scan.shared |= is_elf_shared(head);
                if self.scan.machine.is_none() {
                    self.scan.machine = elf_machine(head);
                }
            }
            Some(Format::Bitcode) => {
                self.scan.bitcode = true;
                file.bitcode = true;
            }
            // A PE image is not a link input; it falls through to the reader
            // that says so.
            Some(Format::Pe) => {}
            // Nothing identified it. Text is what is left, and a linker script
            // is what text means here; anything else falls through to the
            // reader that will reject it by name.
            None => {
                if depth > 0 && script::looks_like(head) {
                    return self.expand(&file, search, depth);
                }
            }
        }
        self.files.push(Entry::File(file));
        Ok(())
    }

    /// Adds a `--start-lib` group as one archive, classified from the bytes
    /// the group already read. An archive is none of the formats the scan
    /// marks, so only the file list grows here.
    fn add_group(&mut self, members: &[(PathBuf, Vec<u8>)], bytes: &[u8]) {
        // The group is an archive, which none of the scan's formats match;
        // only the file list grows here. Its name is the directory the first
        // member came from, so a diagnostic about it says where to look.
        let name = members.first().map_or_else(
            || PathBuf::from("--start-lib"),
            |(p, _)| {
                let mut dir = p.clone();
                dir.set_file_name("--start-lib group");
                dir
            },
        );
        self.files.push(Entry::Group {
            name,
            bytes: bytes.to_vec(),
        });
    }

    /// Replaces `file` with the inputs the script it names holds, in the order
    /// the script names them.
    fn expand(
        &mut self,
        file: &Resolved,
        search: &Search<'_>,
        depth: usize,
    ) -> Result<()> {
        let text = std::fs::read_to_string(&file.path)?;
        let mut names = Vec::new();
        script::parse(&file.path, &text, &mut names)?;
        for name in &names {
            let path = search
                .resolve(name, &file.path)
                .ok_or_else(|| script::not_found(&file.path, name))?;
            self.add(
                Resolved {
                    path,
                    from_l: name.library,
                    // A script reached through an `AS_NEEDED` list names
                    // as-needed libraries throughout: the enclosing directive
                    // applies to everything the script pulls in.
                    as_needed: name.as_needed || file.as_needed,
                    bitcode: false,
                },
                search,
                depth.saturating_sub(1),
            )?;
        }
        Ok(())
    }
}

/// Ad-hoc signs a dynamic Mach-O executable on macOS. Apple Silicon's kernel
/// validates every executable mapping; leaving the signature to a later tool
/// makes a successfully linked image die with SIGKILL before dyld can start.
// The `Result` is load-bearing on macOS, where this spawns `codesign` and
// reports its failure. Everywhere else the body is `Ok(())`, which is the
// only form clippy sees when it lints a non-darwin build.
#[cfg_attr(
    not(target_os = "macos"),
    allow(clippy::unnecessary_wraps, clippy::missing_const_for_fn)
)]
fn ad_hoc_sign(path: &Path) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let status = std::process::Command::new("/usr/bin/codesign")
            .args(["--force", "--sign", "-"])
            .arg(path)
            .status()?;
        if !status.success() {
            return Err(Error::CommandLine(format!(
                "ad-hoc codesigning failed for {}",
                path.display()
            )));
        }
    }
    let _ = path;
    Ok(())
}

/// Fills `head` from the start of `path`, returning how many bytes were read.
/// A short file is not an error: the classifier works on whatever is there.
fn read_head(path: &Path, head: &mut [u8; HEAD_LEN]) -> Option<usize> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut filled = 0;
    while filled < head.len() {
        match file.read(&mut head[filled..]) {
            Ok(0) => break,
            Ok(n) => filled = filled.saturating_add(n),
            Err(_) => return None,
        }
    }
    Some(filled)
}

/// The `e_machine` an ELF header carries: the little-endian half-word at
/// offset 18.
fn elf_machine(head: &[u8]) -> Option<u16> {
    head.get(18..20)
        .and_then(|b| <[u8; 2]>::try_from(b).ok())
        .map(u16::from_le_bytes)
}

/// Whether an ELF header names a shared object (`ET_DYN`). `e_type` is the
/// little-endian half-word at offset 16.
fn is_elf_shared(head: &[u8]) -> bool {
    head.get(16..18)
        .and_then(|b| <[u8; 2]>::try_from(b).ok())
        .is_some_and(|b| u16::from_le_bytes(b) == ET_DYN)
}
