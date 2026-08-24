//! Command-line parsing: argv in, an [`Options`] the link runs from out.

pub mod emulation;
mod inputs;
mod opts;
mod response;
mod zopts;

use std::{
    ffi::{OsStr, OsString},
    path::PathBuf,
};

use inputs::{
    Inputs, has_syslibroot, parse_framework_search_path, parse_search_path,
    parse_sysroot,
};
use opts::{
    Attached, Parsed, attached, is_noop_flag, parse_build_id, parse_hash_style,
    parse_icf, parse_lib_marker, positional, raw_argument,
};
pub use response::expand_response_files;
use xold::{
    buildid::BuildId,
    dynamic::{HashStyle, Strip, ZOptions},
    icf::IcfMode,
    macho::{MachoTarget, PlatformVersion},
    search::LibKind,
};

/// The default interpreter a dynamic executable is loaded under, overridable
/// with `-dynamic-linker`.
const DEFAULT_INTERP: &str = "/lib64/ld-linux-x86-64.so.2";

/// Parsed command-line options.
#[allow(clippy::struct_excessive_bools)]
pub struct Options {
    pub inputs: Vec<PathBuf>,
    /// The index ranges of `inputs` each `--start-lib` group occupies.
    pub lib_groups: Vec<(usize, usize)>,
    /// Parallel to `inputs`: whether a `-l` search found each one. See
    /// [`Inputs`].
    pub from_l: Vec<bool>,
    /// Parallel to `inputs`: whether `--as-needed` was in force where each
    /// was written.
    pub as_needed: Vec<bool>,
    /// `-m`: the emulation the caller expects, checked against the inputs.
    pub emulation: Option<String>,
    /// `--threads=N`: how many threads the link may run on.
    pub threads: Option<usize>,
    /// `-static`: a static image, and no `-l` resolved to a shared object.
    pub static_link: bool,
    /// `-pie` or `-no-pie`, when one was written.
    pub pie: Option<bool>,
    /// `--export-dynamic`: put every definition of an executable into
    /// `.dynsym`.
    pub export_dynamic: bool,
    /// `--strip-debug` / `--strip-all`.
    pub strip: Strip,
    /// `--hash-style`: which symbol hash tables a dynamic image carries.
    pub hash_style: HashStyle,
    /// `-rpath`: the runtime search path, joined with `:`, empty when none
    /// was given.
    pub rpath: String,
    /// `--build-id`: which digest the image's build-id note carries.
    pub build_id: BuildId,
    /// `-u NAME`: names entered as undefined before the link starts.
    pub undefined: Vec<String>,
    /// `--version-script`: the file naming which globals stay exported.
    pub version_script: Option<String>,
    /// Whether a version script may name a symbol the link does not define.
    pub undefined_version: bool,
    /// The `-z` keywords that reach the image.
    pub z: ZOptions,
    /// `-z defs` / `--no-undefined`: refuse a link that leaves a strong
    /// reference unresolved.
    pub no_undefined: bool,
    /// The `-L` directories, kept past parsing because a linker script's own
    /// file list resolves against them too.
    pub search: Vec<PathBuf>,
    /// The `--sysroot` tree, kept for the same reason.
    pub sysroot: Option<PathBuf>,
    /// Whether `--entry` was written, as opposed to `entry` holding the
    /// default. A shared object has no entry point unless one is asked for,
    /// and honouring the default would give every one an `e_entry` of
    /// `_start`.
    pub entry_given: bool,
    pub output: PathBuf,
    pub entry: String,
    pub shared: bool,
    pub soname: Option<String>,
    pub dynamic_exec: bool,
    pub interpreter: String,
    pub gc_sections: bool,
    pub icf: IcfMode,
    pub relax: bool,
    pub relax_named: bool,
    /// Whether to run the link in the calling process rather than a forked
    /// child. See [`xold::detach`].
    pub no_fork: bool,
    /// `-dynamic`: produce a dyld-launched Mach-O executable.
    pub macho_dynamic: bool,
    /// `-dylib`: produce an `MH_DYLIB` image rather than an executable.
    pub macho_dylib: bool,
    /// `-install_name`: identity recorded in `LC_ID_DYLIB`.
    pub macho_install_name: Option<String>,
    /// `-exported_symbols_list`: exact names published through dyld's export
    /// trie. The file is read once the Mach-O path has been selected.
    pub macho_exported_symbols: Option<PathBuf>,
    /// `-arch`: the Mach-O target named by the compiler driver.
    pub macho_arch: Option<MachoTarget>,
    /// `-platform_version`: deployment target and SDK metadata.
    pub macho_platform: Option<PlatformVersion>,
    /// `-dead_strip`: enable the Mach-O section reachability pass.
    pub macho_dead_strip: bool,
}
/// What the command line asked for.
///
/// `--version` and `--help` are answered rather than linked: a build system
/// probes a linker with them to find out what it is before it uses it, and an
/// answer on standard output with a zero exit status is what it expects.
pub enum Request {
    /// Link the image these options describe.
    Link(Box<Options>),
    /// Print this text and stop, successfully.
    Print(String),
}

/// The name and version this linker reports to whoever asks.
///
/// The GNU spelling comes first because that is the shape every probe is
/// written against: `cmake`, `meson` and `rustc` all look for a known name at
/// the head of the first line.
fn version_text() -> String {
    format!("xold {}\n", env!("CARGO_PKG_VERSION"))
}

pub fn parse(args: &[OsString]) -> std::result::Result<Request, String> {
    let sysroot = parse_sysroot(args)?;
    let search = parse_search_path(args)?;
    let framework_search = parse_framework_search_path(args)?;
    let mut inputs = Inputs {
        search,
        framework_search,
        sysroot,
        macho_libraries: has_syslibroot(args),
        ..Default::default()
    };
    if let Some(text) = asked_about_itself(args) {
        return Ok(Request::Print(text));
    }
    let mut p = Parsed::default();
    let mut i = 0;
    while i < args.len() {
        i = step(args, i, &mut p, &mut inputs)?;
    }
    if inputs.in_lib {
        return Err("xold: --start-lib without --end-lib".into());
    }
    if inputs.paths.is_empty() {
        return Err(USAGE.into());
    }
    Ok(Request::Link(Box::new(p.into_options(inputs))))
}

/// Answers `--version` and `--help` before anything else is read.
///
/// Both are asked on their own, of a linker the caller has not decided to use
/// yet, so neither waits for the rest of the command line to make sense.
fn asked_about_itself(args: &[OsString]) -> Option<String> {
    for arg in args {
        // A non-UTF-8 argument is a path, and no question about the linker
        // is spelled in one; skip it rather than giving up on the scan.
        let Some(text) = arg.to_str() else {
            continue;
        };
        match text {
            "--version" | "-V" => return Some(version_text()),
            "--help" => {
                return Some(format!("{}\n{USAGE}\n", version_text().trim()));
            }
            // GNU `ld` prints its version for `-v` and links anyway, so this
            // answers only when there is nothing else to do. A build that
            // passes `-v` along with a real link gets the link.
            "-v" if args.len() == 1 => return Some(version_text()),
            _ => {}
        }
    }
    None
}

/// Reads the argument at `i`, returning the index the walk continues at.
///
/// One argument is one of four things, tried in this order: a path this
/// linker cannot spell an option in, an option carrying its value attached, an
/// option taking the argument after it, and everything else -- a switch, an
/// input file, or an option nobody implements.
fn step(
    args: &[OsString],
    i: usize,
    p: &mut Parsed,
    inputs: &mut Inputs,
) -> std::result::Result<usize, String> {
    let Some(arg) = args[i].to_str() else {
        raw_argument(inputs, &args[i])?;
        return Ok(i + 1);
    };
    if let Some((opt, value)) = attached(arg) {
        apply_attached(p, opt, value)?;
        return Ok(i + 1);
    }
    if let Some(next_i) = separated(args, i, arg, p, inputs)? {
        return Ok(next_i);
    }
    plain(arg, p, inputs)?;
    Ok(i + 1)
}

/// Applies an option written with its value attached, as `-ofile` and
/// `-Wl,-soname=x` arrive.
fn apply_attached(
    p: &mut Parsed,
    opt: Attached,
    value: &str,
) -> std::result::Result<(), String> {
    match opt {
        Attached::Output => p.output = PathBuf::from(value),
        Attached::Entry => {
            p.entry = value.to_string();
            p.entry_given = true;
        }
        Attached::Soname => p.soname = Some(value.to_string()),
        Attached::Interp => {
            p.interpreter = value.to_string();
            p.sw.dynamic_exec = true;
        }
        Attached::Icf => p.icf = parse_icf(value)?,
        Attached::HashStyle => p.hash_style = parse_hash_style(value)?,
        Attached::Rpath => p.rpath.push(value.to_string()),
        Attached::BuildId => p.build_id = parse_build_id(value)?,
        Attached::Undefined => p.undefined.push(value.to_string()),
        Attached::VersionScript => {
            p.version_script = Some(value.to_string());
        }
        Attached::Emulation => p.emulation = Some(value.to_string()),
        Attached::Threads => p.threads = Some(parse_threads(value)?),
        // `-O` tunes optimisations a linker may choose to run. xold's one
        // optional pass is `--relax`, which has its own flag and is on by
        // default, so every level describes the image this link already
        // produces. The value is still read, so that a level that is not a
        // number is an error rather than a silently accepted typo.
        Attached::OptLevel => parse_opt_level(value)?,
    }
    Ok(())
}

/// Applies an option whose value is the next argument, answering with the
/// index to continue at when `arg` was one.
///
/// `--sysroot` and `-L` are consumed here without being applied: both were
/// collected in a pass of their own before this walk, and are matched again
/// only so their value does not fall through to the input list.
fn separated(
    args: &[OsString],
    i: usize,
    arg: &str,
    p: &mut Parsed,
    inputs: &mut Inputs,
) -> std::result::Result<Option<usize>, String> {
    let at = i + 1;
    match arg {
        "-o" => p.output = PathBuf::from(next_os(args, at, "-o")?),
        "--entry" | "-e" => {
            p.entry = next(args, at, "--entry")?.to_string();
            p.entry_given = true;
        }
        "-soname" | "--soname" => {
            p.soname = Some(next(args, at, "-soname")?.to_string());
        }
        "-dynamic-linker" | "--dynamic-linker" => {
            p.interpreter = next(args, at, "-dynamic-linker")?.to_string();
            p.sw.dynamic_exec = true;
        }
        "-m" => p.emulation = Some(next(args, at, "-m")?.to_string()),
        // `-R` is the same option under ld's older spelling. GNU `ld` reads
        // it as `--just-symbols` when its argument is a file; that form is
        // not implemented here, so the directory reading is the only one.
        "-u" | "--undefined" => {
            p.undefined.push(next(args, at, "-u")?.to_string());
        }
        "-plugin" | "-plugin-opt" => {
            let _ = next(args, at, arg)?;
        }
        // ld64 driver plumbing that does not affect a non-LTO input link.
        "-lto_library" | "-mllvm" => {
            let _ = next(args, at, arg)?;
        }
        "-arch" => {
            p.macho_arch =
                Some(opts::parse_macho_arch(next(args, at, "-arch")?)?);
        }
        "-platform_version" => {
            let platform = next(args, at, "-platform_version")?;
            let min_os = next(args, at + 1, "-platform_version")?;
            let sdk = next(args, at + 2, "-platform_version")?;
            p.macho_platform =
                Some(opts::parse_platform_version(platform, min_os, sdk)?);
            return Ok(Some(at + 3));
        }
        "-install_name" => {
            p.macho_install_name =
                Some(next(args, at, "-install_name")?.to_string());
        }
        "-exported_symbols_list" => {
            p.macho_exported_symbols = Some(PathBuf::from(next_os(
                args,
                at,
                "-exported_symbols_list",
            )?));
        }
        "--version-script" | "-version-script" => {
            p.version_script = Some(next(args, at, "--version-script")?.into());
        }
        "-rpath" | "--rpath" | "-R" => {
            p.rpath.push(next(args, at, "-rpath")?.to_string());
        }
        // `-rpath-link` names where to look for a dependency's own
        // dependencies at link time. xold resolves those against the `-L`
        // path, so the directory is taken as one more place to look.
        "-rpath-link" | "--rpath-link" => {
            inputs.search.push(PathBuf::from(next_os(
                args,
                at,
                "-rpath-link",
            )?));
        }
        "--threads" => {
            p.threads = Some(parse_threads(next(args, at, "--threads")?)?);
        }
        "-z" => p.zkeyword(next(args, at, "-z")?)?,
        "-l" => inputs.library(next(args, at, "-l")?)?,
        "-framework" => inputs.framework(next(args, at, "-framework")?)?,
        "--sysroot" | "-syslibroot" | "-L" | "-F" => {}
        _ => return Ok(None),
    }
    Ok(Some(at + 1))
}

/// Applies a switch, an input path, or reports an option nobody implements.
fn plain(
    arg: &str,
    p: &mut Parsed,
    inputs: &mut Inputs,
) -> std::result::Result<(), String> {
    match arg {
        s if p.sw.parse(s) => {}
        s if positional(inputs, s)? => {}
        // `-static` says two things: link no shared object, and search for
        // archives only. The first is a property of the link, the second is
        // positional, so each half is recorded where it belongs.
        "-static" => {
            p.sw.static_link = true;
            inputs.kind = LibKind::ArchiveOnly;
        }
        "-dynamic" => p.macho_dynamic = true,
        "-dylib" => {
            p.macho_dynamic = true;
            p.macho_dylib = true;
        }
        "-dead_strip" => p.macho_dead_strip = true,
        "--no-threads" => p.threads = Some(1),
        s if s.starts_with("-z") && s.len() > 2 => p.zkeyword(&s[2..])?,
        // `--no-undefined` is the same request as `-z defs`, which is how GNU
        // `ld` spells it and how most build systems write it.
        "--no-undefined" => p.no_undefined = true,
        // `-v` alongside a real link prints the version and links anyway,
        // which is what GNU `ld` does; the printing happened before this
        // walk, in `asked_about_itself`.
        "-v" => println!("{}", version_text().trim()),
        // `gcc` passes its LTO plugin on every link, whether or not any
        // input needs one. The plugin exists to read LTO bytecode, and an
        // input that holds some is refused where it is read, by name -- so
        // for every other link the option describes work there is none of,
        // and accepting it produces exactly the image asked for.
        s if s.starts_with("-plugin-opt=") => {}
        // A bare `--build-id` leaves the style to the linker. lld answers
        // with its cheap digest and so does this: hashing a large image
        // cryptographically costs more than the rest of the link, and
        // nothing that reads a build id needs it to be one.
        "--build-id" => p.build_id = BuildId::Fast,
        "--no-undefined-version" => p.undefined_version = false,
        "--undefined-version" => p.undefined_version = true,
        "--no-build-id" => p.build_id = BuildId::None,
        s if is_noop_flag(s) => {}
        "--start-lib" | "--end-lib" => parse_lib_marker(inputs, arg)?,
        // Collected before this walk, so that a `-L` written after a `-l`
        // still serves it; matched here so the value does not fall through to
        // the input list.
        s if s.starts_with("--sysroot=") => {}
        s if s.starts_with("-L") && s.len() > 2 => {}
        s if s.starts_with("-F") && s.len() > 2 => {}
        s if s.starts_with("-l") && s.len() > 2 => inputs.library(&s[2..])?,
        // An argument that looks like an option and reached this far is one
        // this linker does not implement. Falling through to the input list is
        // what used to happen, and it reported a missing feature as `No such
        // file or directory` naming a path nobody wrote. A lone `-` is left
        // alone: it is a filename by convention, not an option.
        s if s.starts_with('-') && s.len() > 1 => {
            return Err(unknown_option(s));
        }
        s => inputs.file(OsStr::new(s)),
    }
    Ok(())
}
/// Every option this linker implements, in one place: the error for an empty
/// command line and the one for an option it does not know both print it.
pub const USAGE: &str = "usage: xold [-shared | --dynamic-exec | -static] \
     [-pie | -no-pie] [--strip-all | --strip-debug] \
     [--hash-style=sysv|gnu|both] [-rpath dir]... \
     [--build-id[=none|fast|md5|sha1|0xHEX]] [--version-script file] \
     [-u symbol]... [--export-dynamic] \
     [--gc-sections | --no-gc-sections] \
     [--relax | --no-relax] \
     [--icf=all|--icf=safe|--icf=none] [--eh-frame-hdr] [--no-fork] \
     [-m emulation] [-O level] [--threads N | --no-threads] [-z keyword]... \
     [-Bstatic | -Bdynamic] [--as-needed | --no-as-needed] \
     [--push-state | --pop-state] [-o output] [--entry symbol] [-soname name] \
     [-dynamic-linker path] [--sysroot dir] [-L dir]... [-l lib]... \
     <object files...>";

/// The error for an argument that looks like an option and is not one.
///
/// It says outright that the option is not implemented, rather than accepting
/// and ignoring it. An ignored option that would have changed the image --
/// `-pie`, `-z now`, `--version-script`, `--build-id` -- produces something
/// other than what the command line asked for, and does it quietly; a link
/// that stops is the answer that cannot mislead.
pub fn unknown_option(arg: &str) -> String {
    format!(
        "xold: unknown option `{arg}`: this linker implements a fixed set of \
         options and does not ignore the ones it lacks, since an ignored \
         option changes the image it was meant to describe\n{USAGE}"
    )
}

/// Returns the argument at `i`, or an error naming `flag` if it is missing.
pub fn next_os<'a>(
    args: &'a [OsString],
    i: usize,
    flag: &str,
) -> std::result::Result<&'a OsStr, String> {
    args.get(i)
        .map(OsString::as_os_str)
        .ok_or_else(|| format!("xold: {flag} requires an argument"))
}

/// Returns the argument at `i` as text, or an error naming `flag`.
///
/// The values this serves name things inside the image -- an entry symbol,
/// a SONAME, a library -- and those are text by nature. A byte string in
/// one of them is a command-line error, not a path to preserve.
pub fn next<'a>(
    args: &'a [OsString],
    i: usize,
    flag: &str,
) -> std::result::Result<&'a str, String> {
    next_os(args, i, flag)?
        .to_str()
        .ok_or_else(|| format!("xold: {flag} argument is not valid UTF-8"))
}

/// Reads a `--threads=N` value: a positive thread count.
fn parse_threads(value: &str) -> std::result::Result<usize, String> {
    match value.parse::<usize>() {
        Ok(n @ 1..) => Ok(n),
        _ => Err(format!(
            "xold: --threads value `{value}` is not a positive number"
        )),
    }
}

/// Reads an `-O` level, which must be a number even though no level changes
/// what this linker emits.
fn parse_opt_level(value: &str) -> std::result::Result<(), String> {
    if value.bytes().all(|b| b.is_ascii_digit()) {
        return Ok(());
    }
    Err(format!("xold: -O level `{value}` is not a number"))
}

/// Builds the error message for an unrecognised `--icf=` value.
pub fn unknown_icf(value: &str) -> String {
    format!("xold: unknown --icf value `{value}` (expected all|safe|none)")
}
