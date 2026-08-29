//! Minimal TAPI text-stub reader for Darwin SDK libraries.
//!
//! A `.tbd` is YAML, but the linker only needs two scalar/list fields from
//! it: the install name written to `LC_LOAD_DYLIB`, and the exported symbol
//! names used to classify unresolved references as imports. Apple SDK stubs
//! spell symbol lists as bracketed YAML sequences, often continued across
//! lines; parsing those directly keeps this path dependency-free while still
//! accepting every current TAPI v4 stub.

use crate::error::{Error, Result};

/// The parts of a TAPI stub needed by the Mach-O driver.
pub struct Stub {
    pub install_name: String,
    pub exports: Vec<String>,
}

/// Parses one `.tbd`, including all YAML documents concatenated within it.
///
/// Re-export documents are intentionally folded into the same symbol set:
/// an umbrella such as libSystem makes those names available through its own
/// install name.
pub fn parse(text: &str) -> Result<Stub> {
    let install_name = text
        .lines()
        .find_map(|line| scalar(line, "install-name"))
        .ok_or(Error::Format("Mach-O .tbd has no install-name"))?
        .to_owned();
    let mut exports = Vec::new();
    let mut list: Option<String> = None;
    for line in text.lines() {
        if let Some(buffer) = list.as_mut() {
            buffer.push(' ');
            buffer.push_str(line.trim());
            if buffer.contains(']') {
                parse_symbols(buffer, &mut exports);
                list = None;
            }
            continue;
        }
        let trimmed = line.trim();
        let Some((key, value)) = trimmed.split_once(':') else {
            continue;
        };
        if !is_symbol_key(key.trim()) || !value.contains('[') {
            continue;
        }
        if value.contains(']') {
            parse_symbols(value, &mut exports);
        } else {
            list = Some(value.to_owned());
        }
    }
    exports.sort_unstable();
    exports.dedup();
    Ok(Stub {
        install_name,
        exports,
    })
}

/// Reads a quoted or plain YAML scalar from `key:`.
fn scalar<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let value = line.trim().strip_prefix(key)?.strip_prefix(':')?.trim();
    let unquoted = value
        .strip_prefix('\'')
        .and_then(|v| v.strip_suffix('\''))
        .or_else(|| value.strip_prefix('"').and_then(|v| v.strip_suffix('"')))
        .unwrap_or(value);
    (!unquoted.is_empty()).then_some(unquoted)
}

/// TAPI list keys that name symbols visible to clients.
fn is_symbol_key(key: &str) -> bool {
    matches!(key, "symbols" | "weak-symbols" | "thread-local-symbols")
}

/// Extracts the comma-separated contents of one bracketed YAML list.
fn parse_symbols(list: &str, out: &mut Vec<String>) {
    let Some((_, body)) = list.split_once('[') else {
        return;
    };
    let body = body.split_once(']').map_or(body, |(before, _)| before);
    for item in body.split(',') {
        let symbol = item.trim().trim_matches(['\'', '"']);
        if !symbol.is_empty() {
            out.push(symbol.to_owned());
        }
    }
}
