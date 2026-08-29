//! The `--start-lib`/`--end-lib` group, packed into a real archive.
//!
//! The markers ask for the objects between them to be extracted on demand,
//! the way an archive's members are, without an archive file on disk to
//! hold them. Rather than teach every pass a second kind of lazy input,
//! the group is serialised into a complete GNU archive in memory --
//! symbol index and all -- and enters the link as one more library. The
//! archive pass, the extraction fixpoint and the COMDAT laziness then
//! apply to it unchanged. lld holds the same members in an unnamed
//! `InputFile` and runs the same extraction over them
//! (`lld/ELF/Driver.cpp`).
//!
//! The bytes this writes are what `ar rcs` would: `!<arch>` magic, a `/`
//! symbol index mapping every defined global to its member, a `//`
//! long-name table only when some member name needs it, and the members
//! themselves, each behind a 60-byte header and padded to an even offset.

use std::path::{Path, PathBuf};

use rustc_hash::FxHashSet;

use crate::{
    elf::{
        ObjectFile,
        constants::{SHN_UNDEF, STB_LOCAL},
    },
    error::{Error, Result},
};

/// The archive magic string, including its trailing newline.
const ARCHIVE_MAGIC: &[u8] = b"!<arch>\n";
/// Fixed size of every member header.
const HEADER_SIZE: usize = 60;
/// Width of the member header's name field.
const NAME_FIELD: usize = 16;
/// The longest member name the header's own field can hold: the name plus
/// its terminating slash.
const SHORT_NAME_MAX: usize = NAME_FIELD - 1;

/// Serialises `members` -- each a path to stand in the archive's name field
/// and the object's bytes -- into a GNU archive an
/// [`crate::archive::Archive`] can parse.
///
/// Every defined global of every member is indexed, first definition first,
/// which is the order extraction would see them in.
pub fn archive(members: &[(PathBuf, Vec<u8>)]) -> Result<Vec<u8>> {
    let index = symbol_index(members)?;
    // The name each member is filed under: its basename, or a `/N` reference
    // into the long-name table when the basename does not fit the field.
    let names: Vec<String> =
        members.iter().map(|(path, _)| member_name(path)).collect();
    let long_table = long_name_table(&names);
    // The layout: the index, then the long-name table when there is one,
    // then the members. Header offsets are the values the index records, so
    // they are settled before a single byte is written.
    let mut at = ARCHIVE_MAGIC.len();
    let index_size = index_size(&index);
    at = next_member(at, index_size);
    if let Some(table) = &long_table {
        at = next_member(at, table.len());
    }
    let mut member_at = Vec::with_capacity(members.len());
    for (_, bytes) in members {
        member_at.push(at);
        at = next_member(at, bytes.len());
    }
    let member_at: Vec<u32> = member_at
        .into_iter()
        .map(u32::try_from)
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| Error::OutOfRange("archive member offset"))?;

    let mut out = Vec::with_capacity(at);
    out.extend_from_slice(ARCHIVE_MAGIC);
    push_header(&mut out, "/", index_size)?;
    push_index(&mut out, &index, &member_at)?;
    pad(&mut out, index_size);
    if let Some(table) = &long_table {
        push_header(&mut out, "//", table.len())?;
        out.extend_from_slice(table);
        pad(&mut out, table.len());
    }
    for (name, (_, bytes)) in names.iter().zip(members) {
        match long_at_filter(long_table.as_deref(), name) {
            Some(offset) => push_long_ref(&mut out, offset, bytes.len())?,
            None => push_header(&mut out, name, bytes.len())?,
        }
        out.extend_from_slice(bytes);
        pad(&mut out, bytes.len());
    }
    Ok(out)
}

/// Every defined global of every member, as `(name, member ordinal)`, in
/// first-definition order. A name two members define is the archive
/// property extraction turns on: the first member wins, so only it is
/// indexed -- pulling the later one for the name would give the link two
/// strong definitions of one symbol.
fn symbol_index(members: &[(PathBuf, Vec<u8>)]) -> Result<Vec<(&[u8], usize)>> {
    let mut index: Vec<(&[u8], usize)> = Vec::new();
    let mut seen: FxHashSet<&[u8]> = FxHashSet::default();
    for (i, (_, bytes)) in members.iter().enumerate() {
        // A bitcode member carries no symbol table this can read: only the
        // LTO plugin knows what it defines. It is left out of the index and
        // picked up by the LTO pass instead, which reads it through the
        // plugin before the archive is ever searched by name.
        if crate::input::Format::detect(bytes)
            == Some(crate::input::Format::Bitcode)
        {
            continue;
        }
        let obj = ObjectFile::parse(bytes)?;
        if !obj.is_relocatable() {
            return Err(Error::Format(
                "a --start-lib member is not a relocatable object",
            ));
        }
        let Some(symtab) = obj.symbol_table()? else {
            continue;
        };
        for sym in symtab.iter() {
            if sym.bind() == STB_LOCAL || sym.st_shndx.get() == SHN_UNDEF {
                continue;
            }
            // Resolution asks the index for stemmed names, so a versioned
            // definition must be indexed under its stem to be extractable.
            let name = crate::symbol::version_stem(symtab.name(sym));
            if name.is_empty() || !seen.insert(name) {
                continue;
            }
            index.push((name, i));
        }
    }
    Ok(index)
}

/// The member's basename, as `ar` files it.
fn member_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.to_string_lossy().into_owned(),
        |n| n.to_string_lossy().into_owned(),
    )
}

/// The `//` table holding every name too long for the header field, each
/// terminated `/\n`; or `None` when no name needs it.
fn long_name_table(names: &[String]) -> Option<Vec<u8>> {
    let mut table = Vec::new();
    for name in names.iter().filter(|n| n.len() > SHORT_NAME_MAX) {
        table.extend_from_slice(name.as_bytes());
        table.extend_from_slice(b"/\n");
    }
    table.is_empty().then_some(table)
}

/// The byte offset `name` sits at inside the long-name table, if there is
/// one.
fn long_at_filter(table: Option<&[u8]>, name: &str) -> Option<usize> {
    let table = table?;
    if name.len() <= SHORT_NAME_MAX {
        return None;
    }
    let mut at = 0;
    for entry in table.split(|&b| b == b'\n') {
        // Every entry but a trailing empty one ends in `/`.
        let entry = entry.strip_suffix(b"/").unwrap_or(entry);
        if entry == name.as_bytes() {
            return Some(at);
        }
        at += entry.len().saturating_add(2);
    }
    None
}

/// The serialised size of the `/` symbol index.
fn index_size(index: &[(&[u8], usize)]) -> usize {
    4 + index.len() * 4 + index.iter().map(|(n, _)| n.len() + 1).sum::<usize>()
}

/// Where the member after one at `at` of `size` bytes starts.
fn next_member(at: usize, size: usize) -> usize {
    at.saturating_add(HEADER_SIZE)
        .saturating_add(size)
        .saturating_add(1)
        & !1
}

/// Writes one member header for `name` (short form) and `size`.
fn push_header(out: &mut Vec<u8>, name: &str, size: usize) -> Result<()> {
    let mut field = [b' '; NAME_FIELD];
    let bytes = name.as_bytes();
    let written = bytes
        .len()
        .saturating_add(usize::from(!name.ends_with('/')));
    if written > NAME_FIELD || bytes.contains(&b' ') {
        return Err(Error::Format("archive member name"));
    }
    field[..bytes.len()].copy_from_slice(bytes);
    if !name.ends_with('/') {
        field[bytes.len()] = b'/';
    }
    push_field(out, &field, size);
    Ok(())
}

/// Writes one member header naming offset `at` in the long-name table.
fn push_long_ref(out: &mut Vec<u8>, at: usize, size: usize) -> Result<()> {
    let name = format!("/{at}");
    let mut field = [b' '; NAME_FIELD];
    let bytes = name.as_bytes();
    if bytes.len() > NAME_FIELD {
        return Err(Error::Format("archive member name"));
    }
    field[..bytes.len()].copy_from_slice(bytes);
    push_field(out, &field, size);
    Ok(())
}

/// Writes the header fields every member shares: the name field, then the
/// timestamp, owner, group and mode `ar` records, then the size and the
/// magic terminator.
fn push_field(out: &mut Vec<u8>, name: &[u8], size: usize) {
    out.extend_from_slice(name);
    out.extend_from_slice(b"0           ");
    out.extend_from_slice(b"0     ");
    out.extend_from_slice(b"0     ");
    out.extend_from_slice(b"644     ");
    let size = format!("{size:>10}");
    out.extend_from_slice(size.as_bytes());
    out.extend_from_slice(b"`\n");
}

/// Writes the `/` symbol index member's data: the count, then each symbol's
/// member offset, then the names NUL-separated.
fn push_index(
    out: &mut Vec<u8>,
    index: &[(&[u8], usize)],
    at: &[u32],
) -> Result<()> {
    let count = u32::try_from(index.len())
        .map_err(|_| Error::OutOfRange("archive symbol index"))?;
    out.extend_from_slice(&count.to_be_bytes());
    for &(_, member) in index {
        let at = at.get(member).copied().unwrap_or(0);
        out.extend_from_slice(&at.to_be_bytes());
    }
    for (name, _) in index {
        out.extend_from_slice(name);
        out.push(0);
    }
    Ok(())
}

/// Pads the last member's data to an even offset, as the format requires.
fn pad(out: &mut Vec<u8>, size: usize) {
    if !size.is_multiple_of(2) {
        out.push(b'\n');
    }
}
