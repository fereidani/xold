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
    /// A glob: `*` matches any run of bytes, `?` matches one.
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
                push(&mut script, local, rest);
            }
            Some(("local", rest)) => {
                local = true;
                push(&mut script, local, rest);
            }
            Some((other, _)) => {
                return Err(script_error(
                    path,
                    format!("`{other}:` is not a version script label"),
                ));
            }
            None => push(&mut script, local, entry),
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
fn push(script: &mut VersionScript, local: bool, name: &str) {
    if name.is_empty() {
        return;
    }
    let pattern = if name.contains('*') || name.contains('?') {
        Pattern::Glob(name.as_bytes().to_vec())
    } else {
        Pattern::Exact(name.as_bytes().to_vec())
    };
    if local {
        script.local.push(pattern);
    } else {
        script.global.push(pattern);
    }
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

/// Whether `pattern` matches `name`, with `*` for any run of bytes and `?`
/// for one.
///
/// Iterative rather than recursive, with a remembered star position, so the
/// worst case is bounded by the product of the two lengths and the stack
/// depth is one.
fn glob(pattern: &[u8], name: &[u8]) -> bool {
    let (mut p, mut n) = (0usize, 0usize);
    let (mut star, mut retry) = (None, 0usize);
    while n < name.len() {
        match pattern.get(p) {
            Some(b'*') => {
                star = Some(p);
                retry = n;
                p += 1;
            }
            Some(b'?') => {
                p += 1;
                n += 1;
            }
            Some(&c) if c == name[n] => {
                p += 1;
                n += 1;
            }
            _ => match star {
                Some(at) => {
                    p = at + 1;
                    retry += 1;
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

/// A diagnostic naming the script and what was wrong with it.
fn script_error(path: &str, what: String) -> Error {
    Error::script(path.to_string(), what)
}
