//! The self-contained switches and the options written with their value
//! attached, kept apart so the walk over argv in [`super::parse`] stays
//! readable.

use std::{ffi::OsString, path::PathBuf};

use xold::{
    buildid::{self, BuildId},
    dynamic::{HashStyle, Strip, ZOptions},
    icf::IcfMode,
    macho::{MachoTarget, PlatformVersion},
    search::LibKind,
};

use super::{
    DEFAULT_INTERP, Options, inputs::Inputs, unknown_icf, unknown_option,
};

/// An option written with its value attached to it.
#[derive(Clone, Copy)]
pub enum Attached {
    Output,
    Entry,
    Soname,
    Interp,
    Icf,
    /// `-rpath=`: a runtime search path.
    Rpath,
    /// `--build-id=`: which digest the build-id note carries.
    BuildId,
    /// `--version-script=`: the file naming which globals stay exported.
    VersionScript,
    /// `--undefined=`: a name to enter as undefined.
    Undefined,
    /// `--hash-style=`: which symbol hash tables the image carries.
    HashStyle,
    /// `-m`: the emulation the caller expects, checked against the inputs.
    Emulation,
    /// `--threads=`: how many threads the link may use.
    Threads,
    /// `-O`: the optimisation level, which this linker takes no setting from.
    OptLevel,
}

/// The attached spellings, longest prefix first so `--soname=` is not read as
/// `-s` and `--entry=` is not read as `-e=`.
///
/// These are the forms a compiler driver produces: `-Wl,-soname=x` arrives as
/// `-soname=x`, and `ld` has always taken an attached `-ofile`. Refusing them
/// while implementing the feature under another spelling is the same
/// unhelpful answer as refusing the feature.
pub const ATTACHED: [(&str, Attached); 19] = [
    ("--dynamic-linker=", Attached::Interp),
    ("--threads=", Attached::Threads),
    ("--hash-style=", Attached::HashStyle),
    ("--build-id=", Attached::BuildId),
    ("--undefined=", Attached::Undefined),
    ("--version-script=", Attached::VersionScript),
    ("-version-script=", Attached::VersionScript),
    ("--rpath=", Attached::Rpath),
    ("-rpath=", Attached::Rpath),
    ("-dynamic-linker=", Attached::Interp),
    ("--output=", Attached::Output),
    ("--soname=", Attached::Soname),
    ("--entry=", Attached::Entry),
    ("-soname=", Attached::Soname),
    ("--icf=", Attached::Icf),
    ("-e=", Attached::Entry),
    ("-o", Attached::Output),
    ("-m", Attached::Emulation),
    ("-O", Attached::OptLevel),
];

/// Splits an attached option into what it sets and the value it sets it to.
pub fn attached(arg: &str) -> Option<(Attached, &str)> {
    // ld64's `-mllvm OPTION` is a separated option. Do not let the GNU-ld
    // attached `-mEMULATION` spelling consume its name and strand OPTION.
    if arg == "-mllvm" {
        return None;
    }
    ATTACHED.iter().find_map(|&(prefix, opt)| {
        arg.strip_prefix(prefix)
            .filter(|rest| !rest.is_empty())
            .map(|rest| (opt, rest))
    })
}

/// Reads a `--build-id=` value.
///
/// `uuid` is refused rather than implemented: it names a random string, and
/// this linker produces the same image from the same inputs every time.
pub fn parse_build_id(value: &str) -> std::result::Result<BuildId, String> {
    match value {
        "none" => Ok(BuildId::None),
        "fast" => Ok(BuildId::Fast),
        "md5" => Ok(BuildId::Md5),
        "sha1" | "tree" => Ok(BuildId::Sha1),
        "uuid" => Err("xold: --build-id=uuid asks for a random identifier, \
                       which would make the same inputs produce a different \
                       image on every run; use fast, md5 or sha1"
            .into()),
        hex if hex.starts_with("0x") => buildid::parse_hex(&hex[2..])
            .map(BuildId::Hex)
            .ok_or_else(|| {
                format!(
                    "xold: --build-id={hex} is not an even-length string of \
                     hexadecimal digits"
                )
            }),
        other => Err(format!(
            "xold: unknown --build-id value `{other}` (expected \
             none|fast|md5|sha1|0xHEXSTRING)"
        )),
    }
}

/// Reads a `--hash-style=` value.
pub fn parse_hash_style(value: &str) -> std::result::Result<HashStyle, String> {
    match value {
        "both" => Ok(HashStyle::Both),
        "sysv" => Ok(HashStyle::Sysv),
        "gnu" => Ok(HashStyle::Gnu),
        other => Err(format!(
            "xold: unknown --hash-style value `{other}` (expected \
             sysv|gnu|both)"
        )),
    }
}

/// Reads an `--icf=` value.
pub fn parse_icf(value: &str) -> std::result::Result<IcfMode, String> {
    match value {
        "all" => Ok(IcfMode::All),
        "safe" => Ok(IcfMode::Safe),
        "none" => Ok(IcfMode::None),
        other => Err(unknown_icf(other)),
    }
}

/// Whether `arg` names behaviour the image already has, so honouring it means
/// doing nothing.
///
/// This is not the same as ignoring an option. Each of these describes what
/// xold does anyway, so accepting it produces exactly the image the command
/// line asked for -- which is the test [`unknown_option`] applies.
///
/// `--eh-frame-hdr` asks for a section xold always writes. The group markers
/// ask for archives to be reconsidered until nothing new is extracted, which
/// `pull_archives` does to a fixpoint for every archive whether they are
/// present or not -- a superset, since a member extracted from the last
/// archive can still pull one from the first. `gcc -static` writes the
/// group markers unconditionally. `--start-lib`/`--end-lib` are not here:
/// they change which objects are linked, and `parse` builds the group they
/// mark instead of accepting them silently.
pub fn is_noop_flag(arg: &str) -> bool {
    matches!(
        arg,
        "--eh-frame-hdr"
            | "--start-group"
            | "-("
            | "--end-group"
            | "-)"
            // A reference is resolved against the shared objects the command
            // line names and no others, which is what these ask for and what
            // every current linker defaults to. `--add-needed`, the opposite
            // request, is not implemented and so is not here.
            | "--no-add-needed"
            | "--no-copy-dt-needed-entries"
            // xold emits no warnings, so there is none to promote or demote.
            | "--fatal-warnings"
            | "--no-fatal-warnings"
            // ld64 enables these presentation/optimisation choices in its
            // driver invocation. They do not alter xold's linked semantics.
            | "-demangle"
            | "-no_deduplicate"
    )
}

/// Reads the two architecture names supported by the Mach-O backend.
pub fn parse_macho_arch(
    value: &str,
) -> std::result::Result<MachoTarget, String> {
    match value {
        "arm64" => Ok(MachoTarget::Arm64),
        "x86_64" => Ok(MachoTarget::X86_64),
        other => Err(format!(
            "xold: unsupported -arch value `{other}` (expected arm64|x86_64)"
        )),
    }
}

/// Reads ld64's `-platform_version PLATFORM MIN SDK` triple.
pub fn parse_platform_version(
    platform: &str,
    min_os: &str,
    sdk: &str,
) -> std::result::Result<PlatformVersion, String> {
    let platform = match platform {
        "macos" => 1,
        other => {
            return Err(format!(
                "xold: unsupported Mach-O platform `{other}` (expected macos)"
            ));
        }
    };
    Ok(PlatformVersion {
        platform,
        min_os: parse_macho_version(min_os)?,
        sdk: parse_macho_version(sdk)?,
    })
}

/// Packs `major[.minor[.patch]]` as Mach-O load commands do.
fn parse_macho_version(value: &str) -> std::result::Result<u32, String> {
    let mut parts = value.split('.');
    let parse = |part: Option<&str>| -> Option<u32> {
        match part {
            None => Some(0),
            Some("") => None,
            Some(n) => n.parse().ok(),
        }
    };
    let major = parse(parts.next());
    let minor = parse(parts.next());
    let patch = parse(parts.next());
    let valid = parts.next().is_none()
        && major.is_some_and(|n| n <= 0xffff)
        && minor.is_some_and(|n| n <= 0xff)
        && patch.is_some_and(|n| n <= 0xff);
    if !valid {
        return Err(format!(
            "xold: invalid Mach-O version `{value}` (expected major[.minor[.patch]])"
        ));
    }
    Ok((major.unwrap_or(0) << 16)
        | (minor.unwrap_or(0) << 8)
        | patch.unwrap_or(0))
}

/// Parses argv into options, or returns a usage message. `-l` and `-L` are
/// resolved to concrete paths here, expanding the input list in place.
/// The self-contained on/off switches of the command line.
///
/// Parsed apart from the options that carry values or interact with the
/// input list, so [`parse`] stays a readable walk over argv.
#[allow(clippy::struct_excessive_bools)]
#[derive(Default)]
pub struct Switches {
    pub shared: bool,
    pub dynamic_exec: bool,
    pub gc_sections: bool,
    /// Whether the user named `--relax`/`--no-relax` themselves. The default
    /// is on for ELF, and a format that cannot relax must not refuse a link
    /// over a default nobody asked for.
    pub relax_off: bool,
    pub relax_named: bool,
    pub no_fork: bool,
    /// `--export-dynamic`: put every definition of an executable into
    /// `.dynsym`.
    pub export_dynamic: bool,
    /// `--strip-debug` / `--strip-all`: how much non-loaded information the
    /// image keeps.
    pub strip: Strip,
    /// `-pie` or `-no-pie`: whether the executable is position-independent.
    /// `None` when the command line named neither, which leaves the choice to
    /// the link.
    pub pie: Option<bool>,
    /// `-static`: no dynamic linking at all. It forces a static image and
    /// narrows every following `-l` to archives, so a shared object among the
    /// inputs is a contradiction rather than a preference.
    pub static_link: bool,
}

impl Switches {
    /// Consumes `arg` if it is a switch, answering whether it was one.
    pub fn parse(&mut self, arg: &str) -> bool {
        match arg {
            "-shared" | "--shared" => self.shared = true,
            "--dynamic-exec" => self.dynamic_exec = true,
            "--gc-sections" => self.gc_sections = true,
            "--no-gc-sections" => self.gc_sections = false,
            "--no-fork" => self.no_fork = true,
            "--export-dynamic" | "-E" => self.export_dynamic = true,
            "--no-export-dynamic" => self.export_dynamic = false,
            // `-pie` asks for a dynamic executable as well as a
            // position-independent one: a fixed-base image is what
            // `-no-pie` describes, and a static image has no loader to
            // relocate it.
            "-pie" | "--pic-executable" => {
                self.pie = Some(true);
                self.dynamic_exec = true;
            }
            "-no-pie" | "--no-pic-executable" => self.pie = Some(false),
            // `-s` and `-S` are the one-letter spellings every driver uses.
            // The stronger request wins when both are written, as it does in
            // `ld`: `--strip-all` already drops what `--strip-debug` would.
            "--strip-all" | "-s" => self.strip = Strip::All,
            "--strip-debug" | "-S" => {
                if self.strip != Strip::All {
                    self.strip = Strip::Debug;
                }
            }
            "--relax" | "--no-relax" => {
                self.relax_off = arg == "--no-relax";
                self.relax_named = true;
            }
            _ => return false,
        }
        true
    }
}

/// Applies one option whose effect is positional: it describes the inputs
/// written after it and none of the ones before it.
///
/// `-Bstatic`/`-Bdynamic` choose what the next `-l` will accept, and
/// `--as-needed`/`--no-as-needed` choose whether the shared objects that
/// follow earn their `DT_NEEDED` by being bound to. `gcc` and `rustc` both
/// wrap a single library in a pair of these, so honouring them for the whole
/// command line rather than for the range they enclose would link something
/// other than what was asked for.
///
/// Answers whether `arg` was one of them.
pub fn positional(
    inputs: &mut Inputs,
    arg: &str,
) -> std::result::Result<bool, String> {
    match arg {
        "-Bstatic" | "-dn" | "-non_shared" => {
            inputs.kind = LibKind::ArchiveOnly;
        }
        "-Bdynamic" | "-dy" | "-call_shared" => inputs.kind = LibKind::Any,
        "--as-needed" => inputs.as_needed = true,
        "--no-as-needed" => inputs.as_needed = false,
        "--push-state" => inputs.push_state(),
        "--pop-state" => inputs.pop_state()?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// Applies one `--start-lib`/`--end-lib` marker, or reports why it cannot:
/// a group may not nest, and an end must have a start.
pub fn parse_lib_marker(
    inputs: &mut Inputs,
    arg: &str,
) -> std::result::Result<(), String> {
    if arg == "--start-lib" {
        if inputs.in_lib {
            return Err("xold: nested --start-lib".into());
        }
        inputs.begin_lib();
        return Ok(());
    }
    if !inputs.in_lib {
        return Err("xold: --end-lib without --start-lib".into());
    }
    inputs.end_lib();
    Ok(())
}

/// Routes one non-UTF-8 argument.
///
/// Every option this linker knows is ASCII, so such an argument is never
/// one: it is an input path, unless it carries a leading dash and so only
/// looks like an option nobody wrote.
pub fn raw_argument(
    inputs: &mut Inputs,
    arg: &OsString,
) -> std::result::Result<(), String> {
    let raw = arg.as_encoded_bytes();
    if raw.first() == Some(&b'-') && raw.len() > 1 {
        return Err(unknown_option(&arg.to_string_lossy()));
    }
    inputs.file(arg);
    Ok(())
}

/// The scalar settings accumulated while walking the command line. They are
/// filled in by one pass over the arguments and then handed to `into_options`
/// together with the collected inputs.
pub struct Parsed {
    pub output: PathBuf,
    /// `--build-id`: which digest the note carries, if any.
    pub build_id: BuildId,
    /// `-u NAME`: names entered as undefined before the link starts.
    pub undefined: Vec<String>,
    /// `--version-script`: the file named, read once the whole command line
    /// has been walked.
    pub version_script: Option<String>,
    /// Whether a version script may name a symbol the link does not define.
    pub undefined_version: bool,
    /// `-rpath`: every directory written, in order. Joined with `:` when the
    /// image records them, which is the form `DT_RUNPATH` takes.
    pub rpath: Vec<String>,
    /// `--hash-style`: which symbol hash tables a dynamic image carries.
    pub hash_style: HashStyle,
    /// The `-z` keywords that reach the image.
    pub z: ZOptions,
    /// `-z defs`: refuse a link that leaves a strong reference unresolved.
    pub no_undefined: bool,
    /// `-m`: the emulation named on the command line, checked against the
    /// inputs once their machine is known.
    pub emulation: Option<String>,
    /// `--threads=N`: the size of the pool the link runs on.
    pub threads: Option<usize>,
    pub entry: String,
    pub entry_given: bool,
    pub soname: Option<String>,
    pub interpreter: String,
    pub icf: IcfMode,
    pub sw: Switches,
    pub macho_dynamic: bool,
    pub macho_dylib: bool,
    pub macho_install_name: Option<String>,
    pub macho_exported_symbols: Option<PathBuf>,
    pub macho_arch: Option<MachoTarget>,
    pub macho_platform: Option<PlatformVersion>,
    pub macho_dead_strip: bool,
}

impl Default for Parsed {
    fn default() -> Self {
        Self {
            output: PathBuf::from("a.out"),
            build_id: BuildId::None,
            undefined: Vec::new(),
            version_script: None,
            undefined_version: true,
            rpath: Vec::new(),
            z: ZOptions::new(),
            no_undefined: false,
            hash_style: HashStyle::Both,
            emulation: None,
            threads: None,
            entry: String::from("_start"),
            entry_given: false,
            soname: None,
            interpreter: String::from(DEFAULT_INTERP),
            icf: IcfMode::None,
            sw: Switches::default(),
            macho_dynamic: false,
            macho_dylib: false,
            macho_install_name: None,
            macho_exported_symbols: None,
            macho_arch: None,
            macho_platform: None,
            macho_dead_strip: false,
        }
    }
}

impl Parsed {
    /// Applies one `-z` keyword. `defs` is the odd one out: it asks for a
    /// check at link time rather than a bit in the image, so it is recorded
    /// beside the keywords rather than among them.
    pub fn zkeyword(&mut self, kw: &str) -> std::result::Result<(), String> {
        if kw == "defs" {
            self.no_undefined = true;
            return Ok(());
        }
        if kw == "undefs" {
            self.no_undefined = false;
            return Ok(());
        }
        super::zopts::keyword(&mut self.z, kw)
    }

    /// Combines the parsed switches with the collected inputs. `relax` is
    /// stored inverted on the command line, as `--no-relax`.
    pub fn into_options(self, inputs: Inputs) -> Options {
        Options {
            lib_groups: inputs.lib_groups,
            pie: self.sw.pie,
            export_dynamic: self.sw.export_dynamic,
            strip: self.sw.strip,
            hash_style: self.hash_style,
            rpath: self.rpath.join(":"),
            build_id: self.build_id,
            undefined: self.undefined,
            version_script: self.version_script,
            undefined_version: self.undefined_version,
            z: self.z,
            no_undefined: self.no_undefined,
            as_needed: inputs.as_needed_flags,
            emulation: self.emulation,
            threads: self.threads,
            static_link: self.sw.static_link,
            from_l: inputs.from_l,
            search: inputs.search,
            sysroot: inputs.sysroot,
            entry_given: self.entry_given,
            inputs: inputs.paths,
            output: self.output,
            entry: self.entry,
            shared: self.sw.shared,
            soname: self.soname,
            dynamic_exec: self.sw.dynamic_exec,
            interpreter: self.interpreter,
            gc_sections: self.sw.gc_sections,
            icf: self.icf,
            relax: !self.sw.relax_off,
            relax_named: self.sw.relax_named,
            no_fork: self.sw.no_fork,
            macho_dynamic: self.macho_dynamic,
            macho_dylib: self.macho_dylib,
            macho_install_name: self.macho_install_name,
            macho_exported_symbols: self.macho_exported_symbols,
            macho_arch: self.macho_arch,
            macho_platform: self.macho_platform,
            macho_dead_strip: self.macho_dead_strip,
        }
    }
}
