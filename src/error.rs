//! Error handling for the linker.
//!
//! Every fallible operation returns [`Result`]. The data path never panics:
//! there is no `unwrap`, `expect`, `unreachable!` or array indexing without a
//! prior bounds check. Errors are cold and may allocate; the hot parsing and
//! layout paths borrow their data instead.

use std::fmt;

/// A type alias so call sites read as `Result<T>` rather than
/// `std::result::Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

/// Linker errors. Variants are intentionally small; rich diagnostic context is
/// added by the reporting layer, not carried through propagation.
#[derive(Debug)]
pub enum Error {
    /// An underlying system I/O failure (opening, mapping, writing).
    Io(std::io::Error),
    /// The input violates the expected binary format. The message names the
    /// specific check that failed.
    Format(&'static str),
    /// A byte range the file claims to point at is out of bounds.
    OutOfRange(&'static str),
    /// The same name was given two strong definitions across inputs. Carries
    /// the symbol name so the diagnostic is meaningful. Boxed: the payload is
    /// three strings, and an error this size would otherwise set the size
    /// every hot `Result` in the linker is moved at.
    DuplicateSymbol(Box<DuplicateSymbol>),
    /// A relocation references a symbol that stayed undefined after resolution.
    /// Carries the name so the diagnostic is meaningful.
    UndefinedReference(String),
    /// A relocation whose slot lies in an allocated section the loader cannot
    /// store into, so the dynamic relocation it needs would never be applied.
    /// Carries the target's name, empty for a local symbol, so the diagnostic
    /// names it.
    TextRelocation(String),
    /// A relocation type the architecture table does not handle. Carries the
    /// raw type number so the diagnostic names the offending relocation.
    UnsupportedReloc(u32),
    /// A computed relocation value does not fit its target field. Carries the
    /// raw type number.
    RelocOverflow(u32),
    /// A computed relocation value is not a multiple of the scale its target
    /// field encodes, so the low bits the field cannot hold are non-zero.
    /// Carries the raw type number.
    RelocMisaligned(u32),
    /// A thread-local reference the image being produced cannot resolve.
    /// Carries the symbol name and what the link cannot supply.
    UnresolvableTls(Box<UnresolvableTls>),
    /// Two inputs were built for incompatible variants of the one
    /// architecture, so no single ELF header flag word describes the image.
    /// Carries the input that disagreed and what it disagreed about.
    IncompatibleInput(Box<IncompatibleInput>),
    /// The entry symbol of an executable is not defined anywhere in the link.
    /// Carries the name that was looked for.
    ///
    /// The image would link clean and die on `exec` with `e_entry` at zero, so
    /// the link stops instead.
    UndefinedEntry(String),
    /// An executable references a dependency's protected or hidden data
    /// object. Carries the name.
    ///
    /// A copy relocation is how an executable takes over such a name, and
    /// neither visibility permits it: protected promises the dependency's own
    /// references bind to its own definition, and hidden is not reachable from
    /// outside it at all. Resolving the reference anyway would bind it to the
    /// address the dependency happened to be linked at.
    UnpreemptableSymbol(String),
    /// An executable references a dependency's data object that declares no
    /// size. Carries the name.
    ///
    /// The reference asks for a copy relocation, and a copy slot needs a
    /// length: a zero-sized slot would take over the name for the whole
    /// process with nothing behind it, leaving the dependency's real
    /// definition unreachable. lld refuses the same way, in
    /// `addCopyRelSymbol`.
    CopyZeroSized(String),
    /// A thread-local and a plain occurrence of one name were folded
    /// together. Carries the name.
    ///
    /// `st_value` of an `STT_TLS` symbol is an offset from the thread
    /// pointer, not an address, so a plain reference to the name would
    /// resolve against a number the code never meant. The link refuses
    /// rather than store it.
    TlsMismatch(String),
    /// A linker script this linker cannot follow: it names a directive that
    /// is not implemented, is malformed, or names a file that is not there.
    /// Carries the script and what stopped it.
    Script(Box<ScriptError>),
    /// An option this linker implements for one output format was written for
    /// another. Carries the option and the format.
    ///
    /// Accepting it and dropping it would produce an image other than the one
    /// the command line described, and do it quietly; the command line is
    /// refused instead.
    UnsupportedOption(Box<UnsupportedOption>),
    /// The command line and the inputs contradict each other: an option
    /// describes an image the inputs cannot produce. Carries the whole
    /// message, which names both halves.
    ///
    /// Separate from [`Self::UnsupportedOption`], which is about an option a
    /// format does not implement. Here the option is implemented and the
    /// inputs are what refuse it -- `-m elf_x86_64` over `AArch64` objects,
    /// `-static` over a shared object.
    CommandLine(String),
}

/// The payload of [`Error::DuplicateSymbol`].
#[derive(Debug)]
pub struct DuplicateSymbol {
    pub name: String,
    /// The two inputs that define it, as the caller named them. Both are
    /// in hand where the clash is found, and a message naming neither
    /// leaves the reader to search for them.
    pub first: String,
    pub second: String,
}

/// The payload of [`Error::UnresolvableTls`].
#[derive(Debug)]
pub struct UnresolvableTls {
    pub symbol: String,
    pub reason: &'static str,
}

/// The payload of [`Error::IncompatibleInput`].
#[derive(Debug)]
pub struct IncompatibleInput {
    pub input: String,
    pub what: &'static str,
}

/// The payload of [`Error::Script`].
#[derive(Debug)]
pub struct ScriptError {
    pub file: String,
    pub what: String,
}

/// The payload of [`Error::UnsupportedOption`].
#[derive(Debug)]
pub struct UnsupportedOption {
    pub option: String,
    pub format: &'static str,
}

impl Error {
    /// A [`Self::DuplicateSymbol`], boxed.
    pub fn duplicate_symbol(
        name: String,
        first: String,
        second: String,
    ) -> Self {
        Self::DuplicateSymbol(Box::new(DuplicateSymbol {
            name,
            first,
            second,
        }))
    }

    /// A [`Self::UnresolvableTls`], boxed.
    pub fn unresolvable_tls(symbol: String, reason: &'static str) -> Self {
        Self::UnresolvableTls(Box::new(UnresolvableTls { symbol, reason }))
    }

    /// An [`Self::IncompatibleInput`], boxed.
    pub fn incompatible_input(input: String, what: &'static str) -> Self {
        Self::IncompatibleInput(Box::new(IncompatibleInput { input, what }))
    }

    /// A [`Self::Script`], boxed.
    pub fn script(file: String, what: String) -> Self {
        Self::Script(Box::new(ScriptError { file, what }))
    }

    /// An [`Self::UnsupportedOption`], boxed.
    pub fn unsupported_option(option: String, format: &'static str) -> Self {
        Self::UnsupportedOption(Box::new(UnsupportedOption { option, format }))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Format(msg) => write!(f, "format error: {msg}"),
            Self::OutOfRange(msg) => write!(f, "out of range: {msg}"),
            Self::DuplicateSymbol(d) => {
                let DuplicateSymbol {
                    name,
                    first,
                    second,
                } = d.as_ref();
                write!(
                    f,
                    "duplicate symbol: {name}\n  defined in {first}\n  and \
                     in {second}"
                )
            }
            Self::UndefinedReference(name) => {
                write!(f, "undefined reference to {name}")
            }
            // The target is not missing: this link defines it, or a dependency
            // does and only the slot holding the reference is the problem. So
            // this must not read like the undefined-reference arm above.
            // Naming the target, "a local symbol" included, and the remedy
            // follow lld's own diagnostic.
            Self::TextRelocation(name) => {
                f.write_str("reference to ")?;
                if name.is_empty() {
                    f.write_str("a local symbol")?;
                } else {
                    write!(f, "'{name}'")?;
                }
                f.write_str(
                    " from a read-only section: a dynamic relocation cannot \
                     be applied there; recompile with -fPIC",
                )
            }
            Self::UnsupportedReloc(r_type) => {
                write!(f, "unsupported relocation type {r_type}")
            }
            Self::RelocOverflow(r_type) => {
                write!(f, "relocation overflow (type {r_type})")
            }
            Self::RelocMisaligned(r_type) => {
                write!(f, "improper alignment for relocation (type {r_type})")
            }
            Self::UnresolvableTls(e) => write!(
                f,
                "cannot resolve thread-local '{}': {}",
                e.symbol, e.reason
            ),
            Self::IncompatibleInput(e) => write!(
                f,
                "{} was built for a different {} from the earlier inputs",
                e.input, e.what
            ),
            Self::UndefinedEntry(name) => write!(
                f,
                "entry symbol '{name}' is not defined: an executable with no \
                 entry point links but cannot start"
            ),
            Self::UnpreemptableSymbol(name) => write!(
                f,
                "cannot preempt symbol '{name}': a dependency's protected or \
                 hidden data object cannot be copied into an executable"
            ),
            Self::CopyZeroSized(name) => write!(
                f,
                "cannot create a copy relocation for symbol '{name}': the \
                 dependency declares no size for it, so a slot would take \
                 over the name with nothing behind it"
            ),
            Self::TlsMismatch(name) => write!(
                f,
                "TLS attribute mismatch on symbol '{name}': one occurrence \
                 names a thread-local, another does not, and a plain \
                 reference would read the offset as an address"
            ),
            Self::Script(e) => {
                write!(f, "linker script {}: {}", e.file, e.what)
            }
            Self::CommandLine(msg) => write!(f, "{msg}"),
            Self::UnsupportedOption(e) => write!(
                f,
                "{} is not implemented for {} output, and this linker does \
                 not accept an option it would then ignore",
                e.option, e.format
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::UnpreemptableSymbol(_)
            | Self::CopyZeroSized(_)
            | Self::TlsMismatch(_)
            | Self::UndefinedEntry(_)
            | Self::Script(_)
            | Self::UnsupportedOption(_)
            | Self::CommandLine(_)
            | Self::Format(_)
            | Self::OutOfRange(_)
            | Self::DuplicateSymbol(_)
            | Self::UndefinedReference(_)
            | Self::TextRelocation(_)
            | Self::UnsupportedReloc(_)
            | Self::RelocOverflow(_)
            | Self::RelocMisaligned(_)
            | Self::UnresolvableTls(_)
            | Self::IncompatibleInput(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<bytemuck::PodCastError> for Error {
    fn from(_: bytemuck::PodCastError) -> Self {
        // Misalignment or size mismatch when viewing bytes as a structure.
        Self::Format("misaligned or wrongly sized structure view")
    }
}
