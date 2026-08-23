//! GNU ld linker scripts, to the extent an input list needs one.
//!
//! On a GNU system a library name often resolves to text rather than to an
//! object. `/usr/lib64/libc.so` is this:
//!
//! ```text
//! /* GNU ld script
//!    Use the shared library, but some functions are only in
//!    the static library, so try that secondarily.  */
//! OUTPUT_FORMAT(elf64-x86-64)
//! GROUP ( /lib64/libc.so.6 /usr/lib64/libc_nonshared.a
//!         AS_NEEDED ( /lib64/ld-linux-x86-64.so.2 ) )
//! ```
//!
//! A linker that cannot read that resolves `-lc` to `libc.so.6` alone, and so
//! has no definition of `atexit`, which lives only in the archive, nor of
//! `__tls_get_addr` and `__rseq_size`, which since glibc 2.34 live only in the
//! loader. The three files are one library; the script is how the system says
//! so.
//!
//! Only the directives that name inputs are implemented -- `INPUT`, `GROUP`
//! and `AS_NEEDED`, plus `OUTPUT_FORMAT`, which is read and discarded because
//! the target is derived from the input objects and not from a name in a
//! script. Every other directive is refused by name. `SECTIONS`, `MEMORY` and
//! `ENTRY` describe the image rather than its input list, and a script that
//! places sections and is quietly ignored produces an image nobody asked for
//! -- the same trade [`Error::UnsupportedOption`] exists to avoid.
//!
//! # Where this runs
//!
//! Expansion belongs where the command line becomes a list of files, before
//! anything is opened: `INPUT` is defined to mean what writing those names on
//! the command line would mean. The link itself takes a settled list, so no
//! later pass has to know a script was involved.

use std::path::{Path, PathBuf};

use crate::{
    archive::Archive,
    error::{Error, Result},
    search,
};

/// How deeply `AS_NEEDED` may nest inside one `INPUT` or `GROUP` list.
///
/// A bound rather than a cycle check: real scripts nest once, and a deeper one
/// is a malformed file rather than a shape worth supporting.
const MAX_NESTING: usize = 8;

/// One name a script's `INPUT` or `GROUP` list holds, before it is resolved to
/// a file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Name<'a> {
    /// The name as written, less any `-l` prefix.
    pub text: &'a str,
    /// Whether it was written `-lNAME`, and so resolves through the library
    /// search path rather than as a file name.
    pub library: bool,
    /// Whether an `AS_NEEDED` list held it.
    ///
    /// Such a library takes part in the link like any other; what it does not
    /// take is an unconditional `DT_NEEDED` entry. The loader is named this
    /// way by every glibc `libc.so`, and a program that never calls
    /// `__tls_get_addr` has no business depending on it.
    pub as_needed: bool,
}

impl<'a> Name<'a> {
    /// Reads one word of a file list, splitting the `-l` spelling from the
    /// plain one.
    fn new(word: &'a str, as_needed: bool) -> Self {
        word.strip_prefix("-l").map_or(
            Self {
                text: word,
                library: false,
                as_needed,
            },
            |name| Self {
                text: name,
                library: true,
                as_needed,
            },
        )
    }
}

/// Reads the input names `text` holds into `out`, in the order it names them.
///
/// `path` names the script in diagnostics only. The caller owns `out` so a
/// driver expanding several scripts reuses one buffer.
pub fn parse<'a>(
    path: &Path,
    text: &'a str,
    out: &mut Vec<Name<'a>>,
) -> Result<()> {
    out.clear();
    let mut lex = Lexer::new(text);
    // Terminates: every token the lexer hands back consumes input that it
    // never puts back, and the input is finite.
    while let Some(token) = lex.next_token(path)? {
        match token {
            Token::Word("INPUT" | "GROUP") => {
                names(path, &mut lex, false, 0, out)?;
            }
            Token::Word("OUTPUT_FORMAT") => skip_list(path, &mut lex)?,
            Token::Word(word) => return Err(unimplemented(path, word)),
            Token::Open | Token::Close => {
                return Err(refuse(
                    path,
                    "a parenthesis outside any directive".to_owned(),
                ));
            }
        }
    }
    Ok(())
}

/// Whether `head`, the leading bytes of a file, read as script text rather
/// than as an object.
///
/// This is the question a driver asks of an input no format magic identified.
/// Text is the answer by elimination, as it is for GNU ld and for lld, which
/// hands anything `identify_magic` does not recognise to its script parser.
///
/// Strict ASCII, and `0x7f` excluded with the other control bytes: that byte
/// opens the ELF magic, so a truncated object cannot pass for a script here.
/// An `ar` archive is ruled out by name rather than by elimination, since its
/// header really is printable text -- `!<arch>` followed by a member name and
/// a decimal timestamp -- and is the one binary format that would otherwise
/// read as a script.
pub fn looks_like(head: &[u8]) -> bool {
    !head.is_empty()
        && !Archive::is_archive(head)
        && head.iter().copied().all(is_text)
}

/// Whether `b` is a byte a plain-text file is made of.
const fn is_text(b: u8) -> bool {
    matches!(b, b'\t' | b'\n' | 0x0b | 0x0c | b'\r') || (b >= 0x20 && b < 0x7f)
}

/// Where the names a script holds are looked for.
#[derive(Clone, Copy)]
pub struct Search<'a> {
    /// The `-L` directories, in command-line order.
    pub paths: &'a [PathBuf],
    /// The `--sysroot` tree, when the link has one.
    pub sysroot: Option<&'a Path>,
}

impl Search<'_> {
    /// Resolves one of `script`'s names to a file, or `None` when nothing
    /// answers to it.
    ///
    /// The cases are GNU ld's, and lld spells them the same way in
    /// `ScriptParser::addFile` (`lld/ELF/ScriptParser.cpp`): a
    /// `-l` name searches the library path; an absolute name is re-rooted when
    /// the script itself came out of the sysroot and taken as written
    /// otherwise; a leading `=` asks for the sysroot outright; and a relative
    /// name is looked for beside the script, then in the working directory,
    /// then along the search path.
    ///
    /// An absolute name is returned without checking that it exists, so a
    /// script naming a file that is not there fails while naming it, rather
    /// than as "unable to find" against the name the script was reached by.
    pub fn resolve(&self, name: &Name<'_>, script: &Path) -> Option<PathBuf> {
        if name.library {
            return search::find_library(name.text, self.paths, self.sysroot);
        }
        if let Some(rest) = name.text.strip_prefix('=') {
            return Some(search::rooted(self.sysroot, rest));
        }
        let text = Path::new(name.text);
        if text.is_absolute() {
            return Some(self.rooted_absolute(name.text, script));
        }
        let beside =
            script.parent().unwrap_or_else(|| Path::new(".")).join(text);
        if beside.exists() {
            return Some(beside);
        }
        if text.exists() {
            return Some(text.to_path_buf());
        }
        search::find_in_paths(name.text, self.paths, self.sysroot)
    }

    /// An absolute name from a script that itself came out of the sysroot is
    /// taken relative to that tree: such a script was written against the
    /// target's filesystem, not the build machine's. One found anywhere else
    /// names itself.
    fn rooted_absolute(&self, text: &str, script: &Path) -> PathBuf {
        match self.sysroot {
            Some(root) if script.starts_with(root) => {
                search::rooted(Some(root), text)
            }
            _ => PathBuf::from(text),
        }
    }
}

/// The error for a name a script holds that resolves to no file.
pub fn not_found(path: &Path, name: &Name<'_>) -> Error {
    let spelling = if name.library { "-l" } else { "" };
    refuse(
        path,
        format!("nothing answers to `{spelling}{}`", name.text),
    )
}

/// Reads a parenthesised file list into `out`.
///
/// `as_needed` marks what an enclosing `AS_NEEDED` applies to, and `depth`
/// counts how many of those are open so a pathological nest ends the link
/// instead of the stack.
fn names<'a>(
    path: &Path,
    lex: &mut Lexer<'a>,
    as_needed: bool,
    depth: usize,
    out: &mut Vec<Name<'a>>,
) -> Result<()> {
    if depth > MAX_NESTING {
        return Err(refuse(
            path,
            format!("AS_NEEDED nested more than {MAX_NESTING} deep"),
        ));
    }
    expect_open(path, lex)?;
    while let Some(token) = lex.next_token(path)? {
        match token {
            Token::Close => return Ok(()),
            Token::Word("AS_NEEDED") => {
                names(path, lex, true, depth.saturating_add(1), out)?;
            }
            Token::Word(word) => out.push(Name::new(word, as_needed)),
            Token::Open => {
                return Err(refuse(
                    path,
                    "an unexpected ( inside a file list".to_owned(),
                ));
            }
        }
    }
    Err(refuse(path, "a file list with no closing )".to_owned()))
}

/// Consumes a parenthesised list without recording it, for a directive whose
/// value this linker does not read.
fn skip_list(path: &Path, lex: &mut Lexer<'_>) -> Result<()> {
    expect_open(path, lex)?;
    while let Some(token) = lex.next_token(path)? {
        if matches!(token, Token::Close) {
            return Ok(());
        }
    }
    Err(refuse(path, "a directive with no closing )".to_owned()))
}

/// Consumes the `(` a directive's list opens with.
fn expect_open(path: &Path, lex: &mut Lexer<'_>) -> Result<()> {
    match lex.next_token(path)? {
        Some(Token::Open) => Ok(()),
        _ => Err(refuse(path, "a directive with no ( after it".to_owned())),
    }
}

/// The error for a directive that describes the image rather than its inputs.
fn unimplemented(path: &Path, word: &str) -> Error {
    refuse(
        path,
        format!(
            "`{word}` is not implemented: this linker reads a script for the \
             inputs it names (INPUT, GROUP, AS_NEEDED) and refuses one that \
             describes the image, rather than ignoring it and producing \
             something else"
        ),
    )
}

/// Builds the error for a script this linker cannot follow.
fn refuse(path: &Path, what: String) -> Error {
    Error::script(path.display().to_string(), what)
}

/// What a file list is made of, once whitespace, separators and comments are
/// gone.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Token<'a> {
    /// `(`
    Open,
    /// `)`
    Close,
    /// A directive name or a file name, quoted or bare.
    Word(&'a str),
}

/// Splits script text into [`Token`]s.
struct Lexer<'a> {
    /// What is left to read.
    rest: &'a str,
}

impl<'a> Lexer<'a> {
    const fn new(text: &'a str) -> Self {
        Self { rest: text }
    }

    /// The next token, or `None` at the end of the script.
    fn next_token(&mut self, path: &Path) -> Result<Option<Token<'a>>> {
        self.skip_trivia(path)?;
        let Some(ch) = self.rest.chars().next() else {
            return Ok(None);
        };
        match ch {
            '(' => Ok(Some(self.punct(Token::Open))),
            ')' => Ok(Some(self.punct(Token::Close))),
            '"' => self.quoted(path).map(|word| Some(Token::Word(word))),
            _ => Ok(Some(Token::Word(self.word()))),
        }
    }

    /// Consumes a one-byte delimiter and returns the token it stands for.
    fn punct(&mut self, token: Token<'a>) -> Token<'a> {
        self.rest = self.rest.get(1..).unwrap_or("");
        token
    }

    /// Consumes whitespace, list separators and `/* */` comments.
    ///
    /// Terminates because every iteration either consumes at least the two
    /// bytes a comment opens with or leaves the loop, and `rest` is never
    /// extended.
    fn skip_trivia(&mut self, path: &Path) -> Result<()> {
        while !self.rest.is_empty() {
            let trimmed = self.rest.trim_start_matches(is_separator);
            let Some(body) = trimmed.strip_prefix("/*") else {
                self.rest = trimmed;
                return Ok(());
            };
            let Some(end) = body.find("*/") else {
                return Err(refuse(path, "a /* comment with no */".to_owned()));
            };
            self.rest = body.get(end.saturating_add(2)..).unwrap_or("");
        }
        Ok(())
    }

    /// The next bare word: everything up to a separator, a parenthesis or the
    /// start of a comment.
    ///
    /// A word is not restricted beyond that, because a file name is not: a
    /// script's list is full of `/`, `.`, `-` and `+`, and `/` in particular
    /// only ends a word when it opens a comment.
    fn word(&mut self) -> &'a str {
        let text = self.rest;
        let end = text
            .char_indices()
            .find(|&(i, ch)| {
                is_separator(ch)
                    || ch == '('
                    || ch == ')'
                    || text.get(i..).is_some_and(|t| t.starts_with("/*"))
            })
            .map_or(text.len(), |(i, _)| i);
        self.rest = text.get(end..).unwrap_or("");
        text.get(..end).unwrap_or(text)
    }

    /// The next `"`-quoted word, which may hold anything but a `"`.
    fn quoted(&mut self, path: &Path) -> Result<&'a str> {
        let body = self.rest.get(1..).unwrap_or("");
        let Some(end) = body.find('"') else {
            return Err(refuse(path, "an unterminated quoted name".to_owned()));
        };
        self.rest = body.get(end.saturating_add(1)..).unwrap_or("");
        Ok(body.get(..end).unwrap_or(""))
    }
}

/// Whether `ch` separates one name from the next. A comma and a semicolon are
/// punctuation a script may write between names and nothing more.
fn is_separator(ch: char) -> bool {
    ch.is_whitespace() || ch == ',' || ch == ';'
}
