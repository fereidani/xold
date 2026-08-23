//! Response files: `@file` arguments expanded before the command line is
//! read.

use std::ffi::{OsStr, OsString};

/// How deep a response file may name another before this gives up.
///
/// A bound rather than a cycle check: the depth a real build reaches is one or
/// two, and a file that names itself is the only way to exceed this. Naming
/// the limit is a clearer answer than tracking visited paths, which a
/// symlinked tree can defeat anyway.
const RESPONSE_DEPTH: usize = 8;

/// Replaces every `@file` argument with the arguments that file holds.
///
/// A driver that would exceed `ARG_MAX` -- which any large link does, and
/// which is why gcc and cmake switch to this form -- writes the command line
/// to a file and passes its name. Without expansion `@/tmp/ccXXXX.rsp` reached
/// the input list as a path, and the link died reporting a format error about
/// a file nobody named.
///
/// The content is split on whitespace, honouring single and double quotes and
/// backslash escapes, which is the syntax GNU `ld` and `clang` both accept.
pub fn expand_response_files(
    args: &[OsString],
) -> std::result::Result<Vec<OsString>, String> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        expand_one(arg, RESPONSE_DEPTH, &mut out)?;
    }
    Ok(out)
}

/// Expands one argument into `out`, following `@file` up to `depth` levels.
fn expand_one(
    arg: &OsStr,
    depth: usize,
    out: &mut Vec<OsString>,
) -> std::result::Result<(), String> {
    // A response file is written by a build tool as text, so its name and
    // its contents are UTF-8; a non-UTF-8 argument is a path, not one.
    let Some(arg) = arg.to_str() else {
        out.push(arg.to_os_string());
        return Ok(());
    };
    let Some(path) = arg.strip_prefix('@') else {
        out.push(OsString::from(arg));
        return Ok(());
    };
    if depth == 0 {
        return Err(format!(
            "xold: response files nested more than {RESPONSE_DEPTH} deep at \
             `{path}`: a file that names itself is the usual cause"
        ));
    }
    let text = std::fs::read_to_string(path).map_err(|err| {
        format!("xold: cannot read response file `{path}`: {err}")
    })?;
    for word in split_response(&text) {
        expand_one(OsStr::new(&word), depth - 1, out)?;
    }
    Ok(())
}

/// Splits a response file into arguments.
///
/// Whitespace separates, single and double quotes group, and a backslash
/// escapes the next character. That is what GNU `ld` reads and what `clang`
/// writes, so a file either of them produced round-trips.
fn split_response(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    let mut quote: Option<char> = None;
    let mut escaped = false;
    for ch in text.chars() {
        if escaped {
            word.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if let Some(q) = quote {
            if ch == q {
                quote = None;
            } else {
                word.push(ch);
            }
        } else if ch == '\'' || ch == '"' {
            quote = Some(ch);
        } else if ch.is_whitespace() {
            if !word.is_empty() {
                out.push(std::mem::take(&mut word));
            }
        } else {
            word.push(ch);
        }
    }
    if !word.is_empty() {
        out.push(word);
    }
    out
}
