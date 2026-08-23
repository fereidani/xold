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
    let files = list.views();
    // Mach-O inputs (darwin objects) take a separate link path that produces a
    // `MH_EXECUTE` image; the ELF linker cannot consume them. darwin programs
    // conventionally enter at `_main`.
    if inputs.macho {
        refuse_elf_only(opts, "Mach-O")?;
        if opts.shared {
            return Err(unsupported("-shared".into(), "Mach-O"));
        }
        let entry = default_entry(opts, b"_main");
        let dylibs: Vec<Dylib<'_>> = list
            .dylibs
            .iter()
            .map(|dylib| Dylib {
                install_name: &dylib.install_name,
                exports: &dylib.exports,
            })
            .collect();
        link_macho_with_options(
            &files,
            &opts.output,
            entry,
            &MachOLinkOptions {
                arch: opts.macho_arch,
                dynamic: opts.macho_dynamic,
                platform: opts.macho_platform,
                dead_strip: opts.macho_dead_strip,
                dylibs: &dylibs,
            },
        )?;
        return ad_hoc_sign(&opts.output);
    }
    // COFF inputs (Windows objects) take a separate link path that produces a
    // PE32+ image; the ELF linker cannot consume them. Windows C programs
    // conventionally enter at `main` (no underscore on x86_64). With `-shared`
    // the image is a DLL carrying an export directory.
    if inputs.coff {
        refuse_elf_only(opts, "COFF")?;
        let entry = default_entry(opts, b"main");
        return link_coff(&files, &opts.output, entry, opts.shared);
    }
    let shared = inputs.shared;
    // The version script is read here rather than during parsing: a link that
    // never reaches the ELF path has no use for it, and a read that fails
    // should report as a link error naming the file.
    let script = read_version_script(opts)?;
    let undefined: Vec<&[u8]> =
        opts.undefined.iter().map(String::as_bytes).collect();
    link_image(&Link {
        inputs: &files,
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
            };
            self.add(file, &search, SCRIPT_DEPTH)?;
            i += 1;
        }
        Ok(())
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
        file: Resolved,
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
