//! `--version-script`: which of an image's global symbols stay visible to
//! whoever loads it.
//!
//! A shared object exports every default-visibility global it defines unless
//! it is told otherwise, and for a library with an intended interface that is
//! nearly always too much: the callers bind to names the author never meant to
//! promise, and the library can no longer change them. A version script is the
//! list of what is promised.
//!
//! The syntax this reads is the one build systems generate:
//!
//! ```text
//! {
//!   global:
//!     public_one;
//!     prefix_*;
//!   local:
//!     *;
//! };
//! ```
//!
//! A name matched by the `global` list is exported, a name matched only by the
//! `local` list is not, and a name matched by neither keeps the visibility its
//! definition gave it. `rustc` writes exactly this shape for a `cdylib`, with
//! one `global` entry per `#[no_mangle]` item and `local: *;` under it.
//!
//! # What is refused
//!
//! A named version node (`LIB_1.0 { ... };`) asks for `.gnu.version_d` records
//! that say which version each export belongs to, and this linker does not
//! emit that section. Accepting the script and dropping the versions would
//! produce a library whose exports are unversioned -- loadable, and wrong for
//! every consumer that binds to a versioned name. So the script is refused
//! with the reason, rather than half-applied.

use crate::error::{Error, Result};

/// One entry of a version script's `global` or `local` list.
///
/// The two forms are kept apart because they differ in precedence as well as
/// in cost: an exact name beats a wildcard whichever list each came from,
/// which is what lets `global: foo;` survive `local: *;`.
enum Pattern {
    /// A name written out in full.
    Exact(Vec<u8>),
    /// A glob: `*` matches any run of bytes, `?` matches one, `[...]` is a
    /// character class and `\` escapes the byte after it.
    Glob(Vec<u8>),
}

impl Pattern {
    /// Whether `name` matches.
    fn matches(&self, name: &[u8]) -> bool {
        match self {
            Self::Exact(text) => text == name,
            Self::Glob(pattern) => glob(pattern, name),
        }
    }

    /// Whether this is an exact name, which outranks every glob.
    const fn is_exact(&self) -> bool {
        matches!(self, Self::Exact(_))
    }
}

/// A parsed version script: what it promises and what it hides.
#[derive(Default)]
pub struct VersionScript {
    global: Vec<Pattern>,
    local: Vec<Pattern>,
}

/// What a version script says about one name.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// Exported, because a `global` entry names it.
    Global,
    /// Hidden, because only a `local` entry does.
    Local,
    /// Neither list mentions it, so its definition decides.
    Unmentioned,
}

impl VersionScript {
    /// What the script says about `name`.
    ///
    /// An exact entry wins over a glob in either list, which is the rule GNU
    /// `ld` and lld both apply: `local: *;` beside `global: foo;` hides
    /// everything but `foo`. Between two entries of the same kind, `global`
    /// wins, so a name in both lists is exported.
    pub fn visibility(&self, name: &[u8]) -> Visibility {
        let global = Self::rank(&self.global, name);
        let local = Self::rank(&self.local, name);
        match (global, local) {
            (None, None) => Visibility::Unmentioned,
            (None, Some(_)) => Visibility::Local,
            (Some(g), Some(l)) if l > g => Visibility::Local,
            (Some(_), _) => Visibility::Global,
        }
    }

    /// How strongly a list matches `name`: 1 for a glob, 2 for an exact name,
    /// `None` for no match at all.
    fn rank(list: &[Pattern], name: &[u8]) -> Option<u8> {
        let mut best = None;
        for pattern in list {
            if !pattern.matches(name) {
                continue;
            }
            let rank = u8::from(pattern.is_exact()) + 1;
            best = Some(best.map_or(rank, |b: u8| b.max(rank)));
        }
        best
    }

    /// Every name the script writes out in full, which is what
    /// `--no-undefined-version` checks the link against.
    pub fn exact_names(&self) -> impl Iterator<Item = &[u8]> {
        self.global
            .iter()
            .chain(self.local.iter())
            .filter_map(|p| match p {
                Pattern::Exact(name) => Some(name.as_slice()),
                Pattern::Glob(_) => None,
            })
    }

    /// Whether the script hides anything at all. A script with no `local`
    /// entries leaves every export where it was.
    pub fn hides_anything(&self) -> bool {
        !self.local.is_empty()
    }
}

/// Reads a version script.
///
/// `path` names the file only so a diagnostic can say which script was wrong.
pub fn parse(path: &str, text: &str) -> Result<VersionScript> {
    let mut script = VersionScript::default();
    let body = anonymous_body(path, text)?;
    let body = body.as_str();
    // Outside any `global:`/`local:` label, entries belong to the global
    // list: a script that lists names with no label at all is listing what it
    // exports.
    let mut local = false;
    for entry in body.split(';') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        match strip_label(entry) {
            Some(("global", rest)) => {
                local = false;
                push(path, &mut script, local, rest)?;
            }
            Some(("local", rest)) => {
                local = true;
                push(path, &mut script, local, rest)?;
            }
            Some((other, _)) => {
                return Err(script_error(
                    path,
                    format!("`{other}:` is not a version script label"),
                ));
            }
            None => push(path, &mut script, local, entry)?,
        }
    }
    Ok(script)
}

/// Splits a `global:`/`local:` label off the front of an entry, returning the
/// label and whatever name followed it on the same line.
fn strip_label(entry: &str) -> Option<(&str, &str)> {
    let (label, rest) = entry.split_once(':')?;
    Some((label.trim(), rest.trim()))
}

/// Adds one name to the list in force.
///
/// The token is classified while it still has its quotes and stored without
/// them, which is the order lld's `readSymbols` uses: it pushes
/// `{unquote(tok), hasWildcard(tok)}` (`lld/ELF/ScriptParser.cpp`). So
/// `"foo*"` is a glob whose quotes never reach the matcher, where treating
/// the token literally would leave a pattern no symbol can match.
fn push(
    path: &str,
    script: &mut VersionScript,
    local: bool,
    name: &str,
) -> Result<()> {
    if name.is_empty() {
        return Ok(());
    }
    let wildcard = has_wildcard(name);
    let name = unquote(name);
    let pattern = if wildcard {
        validate_glob(path, name)?;
        Pattern::Glob(name.as_bytes().to_vec())
    } else {
        Pattern::Exact(name.as_bytes().to_vec())
    };
    if local {
        script.local.push(pattern);
    } else {
        script.global.push(pattern);
    }
    Ok(())
}

/// Whether a version-script token is a pattern rather than a plain name.
///
/// The set is lld's `hasWildcard`, `find_first_of("?*[")`: a character class
/// makes a token a pattern just as much as a star does, and reading `[` as an
/// ordinary byte turns `api_[0-9]*` into a name nothing is called.
fn has_wildcard(token: &str) -> bool {
    token.bytes().any(|c| matches!(c, b'?' | b'*' | b'['))
}

/// Strips one pair of surrounding double quotes, after lld's `unquote`.
fn unquote(token: &str) -> &str {
    token
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .unwrap_or(token)
}

/// Rejects a glob that cannot be given a meaning: an unterminated character
/// class, or a range written backwards. `GlobPattern::create` reports the
/// same two, and refusing here keeps a silent non-match from hiding an
/// export the script meant to publish.
fn validate_glob(path: &str, pattern: &str) -> Result<()> {
    let bytes = pattern.as_bytes();
    let mut i = 0usize;
    // Every arm advances `i`, so the walk is bounded by the pattern length.
    while let Some(&c) = bytes.get(i) {
        match c {
            b'\\' => i = i.saturating_add(2),
            b'[' => {
                let end = class_end(bytes, i).ok_or_else(|| {
                    script_error(
                        path,
                        format!("unterminated `[` in pattern `{pattern}`"),
                    )
                })?;
                check_ranges(path, pattern, bytes, i, end)?;
                i = end;
            }
            _ => i = i.saturating_add(1),
        }
    }
    Ok(())
}

/// The index just past the `]` closing the class that opens at `open`.
///
/// The byte right after `[` is a member even when it is `]`, so the search
/// starts one past it; this is what `GlobPattern` does.
fn class_end(bytes: &[u8], open: usize) -> Option<usize> {
    let first = open.checked_add(1)?;
    let first = if matches!(bytes.get(first), Some(b'^' | b'!')) {
        first.checked_add(1)?
    } else {
        first
    };
    let from = first.checked_add(1)?;
    let at = bytes.get(from..)?.iter().position(|&c| c == b']')?;
    from.checked_add(at)?.checked_add(1)
}

/// Reports a `X-Y` range inside a class whose ends are the wrong way round.
fn check_ranges(
    path: &str,
    pattern: &str,
    bytes: &[u8],
    open: usize,
    end: usize,
) -> Result<()> {
    let mut i = open.saturating_add(1);
    // Bounded by the class extent, which the caller already located.
    while i < end.saturating_sub(1) {
        let (Some(&start), Some(&dash), Some(&last)) =
            (bytes.get(i), bytes.get(i + 1), bytes.get(i + 2))
        else {
            return Ok(());
        };
        if dash == b'-' && last != b']' {
            if start > last {
                return Err(script_error(
                    path,
                    format!("reversed range in pattern `{pattern}`"),
                ));
            }
            i = i.saturating_add(3);
        } else {
            i = i.saturating_add(1);
        }
    }
    Ok(())
}

/// Extracts the body of the one anonymous version node, refusing the forms
/// this linker does not implement.
fn anonymous_body(path: &str, text: &str) -> Result<String> {
    let text = strip_comments(text);
    let text = text.as_str();
    let open = text.find('{').ok_or_else(|| {
        script_error(path, "no version node: expected `{ ... };`".to_string())
    })?;
    let name = text.get(..open).unwrap_or("").trim();
    if !name.is_empty() {
        return Err(script_error(
            path,
            format!(
                "the named version node `{name}` needs `.gnu.version_d` \
                 records, which this linker does not emit; an anonymous node \
                 (`{{ global: ...; local: *; }};`) is what it implements"
            ),
        ));
    }
    let close = text.rfind('}').ok_or_else(|| {
        script_error(path, "the version node is not closed".to_string())
    })?;
    let body = text.get(open + 1..close).unwrap_or("");
    if body.contains('{') {
        return Err(script_error(
            path,
            "a nested block (`extern \"C++\" { ... }`) is not implemented"
                .to_string(),
        ));
    }
    if text.get(close + 1..).is_some_and(|rest| rest.contains('{')) {
        return Err(script_error(
            path,
            "more than one version node needs `.gnu.version_d` records, \
             which this linker does not emit"
                .to_string(),
        ));
    }
    Ok(body.to_string())
}

/// Drops `#` line comments and `/* ... */` block comments, both of which a
/// hand-written script uses.
///
/// A comment left in place would be read as a list of names, and every word
/// in it would become an export or a hidden symbol.
fn strip_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        let line = rest.find('#');
        let block = rest.find("/*");
        let Some(at) = line.into_iter().chain(block).min() else {
            out.push_str(rest);
            return out;
        };
        out.push_str(rest.get(..at).unwrap_or(""));
        // A comment stands in for whitespace, so the names on either side of
        // it do not run together.
        out.push(' ');
        let after = rest.get(at..).unwrap_or("");
        rest = if Some(at) == line {
            after.find('\n').map_or("", |end| &after[end..])
        } else {
            after.find("*/").map_or("", |end| &after[end + 2..])
        };
    }
}

/// Whether `pattern` matches `name`, with `*` for any run of bytes, `?` for
/// one, `[...]` for a character class and `\` to escape the byte after it.
///
/// Iterative rather than recursive, with a remembered star position, so the
/// worst case is bounded by the product of the two lengths and the stack
/// depth is one.
fn glob(pattern: &[u8], name: &[u8]) -> bool {
    let (mut p, mut n) = (0usize, 0usize);
    let (mut star, mut retry) = (None, 0usize);
    while n < name.len() {
        if pattern.get(p) == Some(&b'*') {
            star = Some(p);
            retry = n;
            p = p.saturating_add(1);
            continue;
        }
        match one(pattern, p, name[n]) {
            Some(next) => {
                p = next;
                n = n.saturating_add(1);
            }
            // Reconsider the last `*`, letting it swallow one more byte. Each
            // retry raises `retry`, so `n` reaches `name.len()` and the loop
            // ends whether or not a match is found.
            None => match star {
                Some(at) => {
                    p = at.saturating_add(1);
                    retry = retry.saturating_add(1);
                    n = retry;
                }
                None => return false,
            },
        }
    }
    pattern
        .get(p..)
        .is_some_and(|rest| rest.iter().all(|&c| c == b'*'))
}

/// Matches the single pattern element at `p` against `byte`, returning the
/// position just past that element.
///
/// `None` means the element did not match, which includes a malformed class:
/// `validate_glob` rejects those at parse time, so reaching one here can only
/// mean no match.
fn one(pattern: &[u8], p: usize, byte: u8) -> Option<usize> {
    match pattern.get(p)? {
        b'?' => p.checked_add(1),
        b'[' => {
            let end = class_end(pattern, p)?;
            in_class(pattern, p, end, byte).then_some(end)
        }
        b'\\' => {
            let literal = *pattern.get(p.checked_add(1)?)?;
            (literal == byte).then(|| p.saturating_add(2))
        }
        &c => (c == byte).then(|| p.saturating_add(1)),
    }
}

/// Whether `byte` is in the class spanning `open..end`.
///
/// A leading `^` or `!` inverts the set, `X-Y` names an inclusive range, and
/// a `]` in the first position is a member rather than the terminator. This
/// is the set LLVM's `GlobPattern` accepts, which is what lld matches
/// version-script patterns with.
fn in_class(pattern: &[u8], open: usize, end: usize, byte: u8) -> bool {
    let mut i = open.saturating_add(1);
    let invert = matches!(pattern.get(i), Some(b'^' | b'!'));
    if invert {
        i = i.saturating_add(1);
    }
    let mut hit = false;
    // `end` is one past the closing `]`, so the walk is bounded by the class.
    while i < end.saturating_sub(1) {
        let Some(&c) = pattern.get(i) else {
            break;
        };
        if pattern.get(i.saturating_add(1)) == Some(&b'-')
            && pattern
                .get(i.saturating_add(2))
                .is_some_and(|&last| last != b']')
        {
            let last = pattern.get(i.saturating_add(2)).copied().unwrap_or(c);
            if (c..=last).contains(&byte) {
                hit = true;
            }
            i = i.saturating_add(3);
            continue;
        }
        if c == byte {
            hit = true;
        }
        i = i.saturating_add(1);
    }
    hit != invert
}

/// A diagnostic naming the script and what was wrong with it.
fn script_error(path: &str, what: String) -> Error {
    Error::script(path.to_string(), what)
}
