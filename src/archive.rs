//! Minimal `ar` archive reader, enough for lazy member extraction.
//!
//! Real linkers satisfy undefined references by pulling object members out of
//! static libraries on demand. This module parses the GNU symbol index (the `/`
//! member) of an archive into a name-to-member map and lets the driver fetch a
//! member's bytes by header offset. Both the fat layout and the thin one are
//! read. BSD `#1/` embedded names are also exposed by [`members`], which the
//! Mach-O driver uses for Rust's Darwin `.rlib` archives.

use std::{
    borrow::Cow,
    fs, io,
    ops::Range,
    path::{Path, PathBuf},
};

use rustc_hash::FxHashMap;

use crate::{
    elf::{
        ObjectFile,
        constants::{SHN_UNDEF, STB_LOCAL},
    },
    error::{Error, Result},
    input::AlignedBytes,
};

/// The archive magic string, including its trailing newline.
const ARCHIVE_MAGIC: &[u8] = b"!<arch>\n";
/// The thin-archive magic: the members are files beside the archive, and
/// their headers name them rather than hold them.
const THIN_MAGIC: &[u8] = b"!<thin>\n";
/// Fixed size of every member header.
const HEADER_SIZE: usize = 60;
/// Width of the member header's name field.
const NAME_FIELD: usize = 16;

/// A parsed static library: a symbol index over its mapped bytes.
pub struct Archive<'data> {
    bytes: &'data [u8],
    /// The names are borrowed from the archive for a `/` index and owned
    /// for a scanned one (see [`scan_members`]); `Cow` keys keep the
    /// indexed path allocation-free.
    index: FxHashMap<Cow<'data, [u8]>, u64>,
    /// Whether the magic was `!<thin>\n`, making every member a path to a
    /// file beside this one rather than a span of these bytes. The member
    /// walk needs this before anything else: a thin member's header counts
    /// bytes the archive does not hold, so the next header is the one this
    /// one ends at, not `size` bytes later.
    thin: bool,
    /// The directory the archive was named from, which is where a thin
    /// member's path is read from, as it is for `ld` and lld.
    dir: PathBuf,
    /// The `//` long-name table's span, when the archive carries one. GNU
    /// `ar` routes every name longer than the 16-byte field -- and, for a
    /// thin archive, typically every member path -- through it.
    long_names: Option<Range<usize>>,
    /// The library's position among the direct inputs, in input order. Set
    /// while the serial assembly fold walks the parsed inputs, because the
    /// parallel parse that builds the index has no order to read. The archive
    /// pass compares it against a dependency's position to decide which of
    /// the two was reached first; see `Context::extracts_member`.
    pos: u32,
}

/// One member's layout, derived from its header.
struct MemberHeader {
    data_offset: usize,
    size: usize,
    next: usize,
    kind: MemberKind,
}

/// How a member is classified by its 16-byte name field.
#[derive(Clone, Copy)]
enum MemberKind {
    /// The GNU symbol index (`/`), or its 64-bit form (`/SYM64/`). The flag is
    /// set for the 64-bit form, where offsets and counts are 8 bytes.
    SymIndex(bool),
    /// The long-name string table (`//`).
    LongNames,
    /// A regular object member.
    Regular,
}

impl<'data> Archive<'data> {
    /// Whether `bytes` begins with either archive magic, fat or thin.
    ///
    /// The caller uses this to route an input, so it answers for every
    /// shape [`Archive::parse`] reads rather than for one of them.
    pub fn is_archive(bytes: &[u8]) -> bool {
        bytes.starts_with(ARCHIVE_MAGIC) || bytes.starts_with(THIN_MAGIC)
    }

    /// Parses the symbol index of `bytes` as a GNU archive, fat or thin,
    /// named as `from`.
    ///
    /// An archive with no `/` index member is not an error: the index is
    /// only an accelerator over what the members themselves say, so one
    /// built without it (`ar rc` rather than `ar rcs`) or with a stale one
    /// is scanned instead, the way `ranlib` and a plain `ld` both read it.
    /// A thin archive is the exception, because a scan has nothing to read:
    /// the members are files beside it, and only the index says where their
    /// headers are.
    pub fn parse(bytes: &'data [u8], from: &Path) -> Result<Self> {
        let thin = bytes.starts_with(THIN_MAGIC);
        if !Self::is_archive(bytes) {
            return Err(Error::Format("not an ar archive"));
        }
        let dir = from.parent().unwrap_or_else(|| Path::new("")).to_path_buf();
        let mut index = FxHashMap::default();
        let mut long_names = None;
        let mut pos = ARCHIVE_MAGIC.len();
        let mut indexed = false;
        while let Some(hdr) = read_header(bytes, pos, thin)? {
            match hdr.kind {
                MemberKind::SymIndex(sym64) => {
                    parse_index(
                        bytes,
                        hdr.data_offset,
                        hdr.size,
                        sym64,
                        &mut index,
                    )?;
                    indexed = true;
                }
                // The span is recorded rather than parsed here because the
                // names in it are read per member, once the member's header
                // is known to point at one.
                MemberKind::LongNames => {
                    long_names =
                        Some(hdr.data_offset..hdr.data_offset + hdr.size);
                }
                MemberKind::Regular => {}
            }
            pos = hdr.next;
        }
        if !indexed && !thin {
            scan_members(bytes, &mut index)?;
        }
        Ok(Self {
            bytes,
            index,
            thin,
            dir,
            long_names,
            pos: u32::MAX,
        })
    }

    /// The header offset of the member defining `name`, if present.
    pub fn lookup(&self, name: &[u8]) -> Option<u64> {
        self.index.get(name).copied()
    }

    /// The library's position among the direct inputs (see [`Self::pos`]).
    pub const fn pos(&self) -> u32 {
        self.pos
    }

    /// Records the library's position among the direct inputs. Called by the
    /// serial assembly fold, the one pass that walks the inputs in order.
    pub fn set_pos(&mut self, pos: u32) {
        self.pos = pos;
    }

    /// The data bytes of the member whose header is at `header_offset`.
    ///
    /// A thin member's bytes are read from the file its header names, taken
    /// relative to the archive's own directory, which is the only place a
    /// thin archive keeps them.
    pub fn member(&self, header_offset: u64) -> Result<Cow<'data, [u8]>> {
        if !self.thin {
            let pos = to_usize(header_offset, "archive member offset")?;
            let hdr = read_header(self.bytes, pos, false)?
                .ok_or(Error::Format("archive member header missing"))?;
            return member_data(self.bytes, &hdr).map(Cow::Borrowed);
        }
        let path = self.member_path(header_offset)?;
        fs::read(&path).map(Cow::Owned).map_err(|err| {
            Error::Io(io::Error::new(
                err.kind(),
                format!("{}: {err}", path.display()),
            ))
        })
    }

    /// Best-effort name of the member at `header_offset`, as a display
    /// string. A thin member's name is its path.
    pub fn member_name(&self, header_offset: u64) -> Result<String> {
        if !self.thin {
            let field = self.name_field(header_offset)?;
            return Ok(short_name(field).map_or_else(String::new, |name| {
                String::from_utf8_lossy(name).into_owned()
            }));
        }
        self.member_path(header_offset)
            .map(|path| path.display().to_string())
    }

    /// The file a thin member's header names, resolved against the
    /// archive's directory.
    fn member_path(&self, header_offset: u64) -> Result<PathBuf> {
        let field = self.name_field(header_offset)?;
        let name = match long_name_offset(field) {
            Some(at) => self.long_name(at)?,
            None => short_name(field)
                .ok_or(Error::Format("archive member name missing"))?,
        };
        let path = path_from_bytes(name);
        if path.is_absolute() {
            return Ok(path);
        }
        Ok(self.dir.join(path))
    }

    /// The 16-byte name field of the member at `header_offset`.
    fn name_field(&self, header_offset: u64) -> Result<&[u8]> {
        let pos = to_usize(header_offset, "archive member offset")?;
        let end = pos
            .checked_add(NAME_FIELD)
            .ok_or(Error::OutOfRange("archive member offset"))?;
        self.bytes
            .get(pos..end)
            .ok_or(Error::OutOfRange("archive member name"))
    }

    /// The name a `/N` field points at in the `//` table, without the `//`
    /// table's `/` terminator and newline.
    fn long_name(&self, at: usize) -> Result<&[u8]> {
        let table = self
            .long_names
            .as_ref()
            .ok_or(Error::Format("archive long names missing"))?;
        let start = table
            .start
            .checked_add(at)
            .ok_or(Error::OutOfRange("archive long name offset"))?;
        let rest = self
            .bytes
            .get(start..table.end)
            .ok_or(Error::OutOfRange("archive long name offset"))?;
        let end = rest.iter().position(|&b| b == b'\n').unwrap_or(rest.len());
        let name = &rest[..end];
        Ok(name.strip_suffix(b"/").unwrap_or(name))
    }
}

/// Returns every regular member stored inside a fat archive, in archive
/// order. Metadata and symbol-index members are omitted; BSD `#1/N` names are
/// removed from the returned payload.
///
/// This is deliberately an eager enumeration primitive rather than archive
/// resolution policy. The ELF linker continues to use [`Archive::lookup`]
/// for lazy extraction; the Mach-O backend currently has no archive symbol
/// graph and uses this to admit Rust `.rlib` inputs at all.
pub fn members(bytes: &[u8]) -> Result<Vec<&[u8]>> {
    if bytes.starts_with(THIN_MAGIC) {
        return Err(Error::Format(
            "thin archives are not supported by the Mach-O linker",
        ));
    }
    if !bytes.starts_with(ARCHIVE_MAGIC) {
        return Err(Error::Format("not an ar archive"));
    }
    let mut out = Vec::new();
    let mut pos = ARCHIVE_MAGIC.len();
    while let Some(hdr) = read_header(bytes, pos, false)? {
        if matches!(hdr.kind, MemberKind::Regular) {
            let field = bytes
                .get(pos..pos + NAME_FIELD)
                .ok_or(Error::OutOfRange("archive member name"))?;
            let data = member_data(bytes, &hdr)?;
            let (name, payload) = bsd_member(field, data)?;
            let metadata = name.is_some_and(|name| {
                name.starts_with(b"__.SYMDEF")
                    || name == b"lib.rmeta"
                    || name == b"lib.rmeta-link"
            });
            if !metadata {
                out.push(payload);
            }
        }
        pos = hdr.next;
    }
    Ok(out)
}

/// Splits BSD's `#1/N` embedded filename from a member payload.
fn bsd_member<'a>(
    field: &'a [u8],
    data: &'a [u8],
) -> Result<(Option<&'a [u8]>, &'a [u8])> {
    let raw = trim_blank(field);
    let Some(digits) = raw.strip_prefix(b"#1/") else {
        return Ok((short_name(field), data));
    };
    let name_len = std::str::from_utf8(digits)
        .ok()
        .and_then(|n| n.parse::<usize>().ok())
        .ok_or(Error::Format("BSD archive member name length"))?;
    let name = data
        .get(..name_len)
        .ok_or(Error::OutOfRange("BSD archive member name"))?;
    let payload = data
        .get(name_len..)
        .ok_or(Error::OutOfRange("BSD archive member data"))?;
    let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
    Ok((Some(&name[..end]), payload))
}

/// Rebuilds the index from the members themselves, for an archive that
/// carries no `/` symbol index.
///
/// Every regular member that parses as a relocatable object contributes its
/// defined globals, first definition first -- the same rule the index would
/// have recorded. A member that does not parse is left out rather than
/// rejected: an archive may hold a text file beside its objects, and the
/// index this builds only ever refuses an extraction, never grants one.
fn scan_members(
    bytes: &[u8],
    index: &mut FxHashMap<Cow<'_, [u8]>, u64>,
) -> Result<()> {
    let mut pos = ARCHIVE_MAGIC.len();
    while let Some(hdr) = read_header(bytes, pos, false)? {
        if matches!(hdr.kind, MemberKind::Regular) {
            let data = member_data(bytes, &hdr)?;
            let offset = u64::try_from(pos)
                .map_err(|_| Error::OutOfRange("archive member offset"))?;
            // A member's bytes sit at an arbitrary offset inside the
            // archive, which no alignment the ELF records can rely on; the
            // scan copies them the way extraction itself does.
            let owned = AlignedBytes::new(data);
            if let Ok(obj) = ObjectFile::parse(owned.bytes())
                && let Ok(Some(symtab)) = obj.symbol_table()
            {
                for sym in symtab.iter() {
                    if sym.bind() == STB_LOCAL
                        || sym.st_shndx.get() == SHN_UNDEF
                    {
                        continue;
                    }
                    // Resolution asks for stemmed names, so a versioned
                    // definition is indexed under its stem.
                    let name = crate::symbol::version_stem(symtab.name(sym));
                    if !name.is_empty() {
                        index
                            .entry(Cow::Owned(name.to_vec()))
                            .or_insert(offset);
                    }
                }
            }
        }
        pos = hdr.next;
    }
    Ok(())
}

/// Returns one in-archive member's data span.
fn member_data<'a>(bytes: &'a [u8], hdr: &MemberHeader) -> Result<&'a [u8]> {
    let end = hdr
        .data_offset
        .checked_add(hdr.size)
        .ok_or(Error::OutOfRange("archive member data"))?;
    bytes
        .get(hdr.data_offset..end)
        .ok_or(Error::OutOfRange("archive member data"))
}

/// Reads a member header at `pos`, returning its layout or `None` at end.
///
/// `thin` picks where the next header sits for a regular member: its data
/// does not follow its header -- the size field counts an external file's
/// bytes -- so the next header is the byte this one ends at. Everything
/// else, the index included, carries its data in the archive as usual.
fn read_header(
    bytes: &[u8],
    pos: usize,
    thin: bool,
) -> Result<Option<MemberHeader>> {
    let header_end = pos
        .checked_add(HEADER_SIZE)
        .ok_or(Error::OutOfRange("archive header offset"))?;
    if header_end > bytes.len() {
        return Ok(None);
    }
    let name_field = &bytes[pos..pos + 16];
    let size = parse_decimal(&bytes[pos + 48..pos + 58])?;
    let data_offset = header_end;
    let kind = classify(name_field);
    // The index and the long-name table are the archive's own and sit in it
    // whether it is fat or thin; a regular member of a thin one is the only
    // header whose bytes are not there at all.
    let next = if thin && matches!(kind, MemberKind::Regular) {
        align_even(data_offset)
    } else {
        let data_end = data_offset
            .checked_add(size)
            .ok_or(Error::OutOfRange("archive member size"))?;
        if data_end > bytes.len() {
            return Err(Error::OutOfRange("archive member data"));
        }
        align_even(data_end)
    };
    Ok(Some(MemberHeader {
        data_offset,
        size,
        next,
        kind,
    }))
}

/// The name a fat member's field spells, up to the `/` or blank that ends
/// it, or `None` when the field holds nothing but terminators.
fn short_name(field: &[u8]) -> Option<&[u8]> {
    let end = field
        .iter()
        .position(|&b| b == b'/' || b == b' ')
        .unwrap_or(field.len());
    let name = &field[..end];
    (!name.is_empty()).then_some(name)
}

/// The `//`-table offset a `/N` name field holds, or `None` when the field
/// is not a long-name reference.
fn long_name_offset(field: &[u8]) -> Option<usize> {
    let digits = field.strip_prefix(b"/")?;
    let digits = trim_blank(digits);
    if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
        return None;
    }
    std::str::from_utf8(digits).ok()?.parse().ok()
}

/// A path from raw bytes, which need not be valid UTF-8.
fn path_from_bytes(name: &[u8]) -> PathBuf {
    use std::os::unix::ffi::OsStringExt;
    PathBuf::from(std::ffi::OsString::from_vec(name.to_vec()))
}

/// Parses the GNU symbol index into `index`.
fn parse_index<'a>(
    bytes: &'a [u8],
    data_offset: usize,
    size: usize,
    sym64: bool,
    index: &mut FxHashMap<Cow<'a, [u8]>, u64>,
) -> Result<()> {
    let end = data_offset
        .checked_add(size)
        .ok_or(Error::OutOfRange("archive index data"))?;
    let data = bytes
        .get(data_offset..end)
        .ok_or(Error::OutOfRange("archive index data"))?;
    let (cnt_size, off_size) = if sym64 { (8, 8) } else { (4, 4) };
    let count =
        to_usize(read_uint_be(data, 0, cnt_size)?, "archive index count")?;
    let offsets_start = cnt_size;
    // The symbol count is read from the file, so it bounds the loop only after
    // it is checked against the member's own size: the offset table alone needs
    // `count * off_size` bytes, and each symbol contributes at least a NUL to
    // the name table. A corrupt count that fails this would otherwise spin
    // through billions of empty names.
    let table_size = count
        .checked_mul(off_size)
        .ok_or(Error::OutOfRange("archive index table"))?;
    let mut name_pos = offsets_start
        .checked_add(table_size)
        .ok_or(Error::OutOfRange("archive index table"))?;
    let least = name_pos
        .checked_add(count)
        .ok_or(Error::OutOfRange("archive index table"))?;
    if least > data.len() {
        return Err(Error::Format("archive index count exceeds member size"));
    }
    for i in 0..count {
        let rest = data.get(name_pos..).unwrap_or(&[]);
        let name = cstr_slice(rest);
        name_pos = name_pos
            .checked_add(name.len())
            .and_then(|p| p.checked_add(1))
            .ok_or(Error::OutOfRange("archive index name"))?;
        let off_pos = i
            .checked_mul(off_size)
            .and_then(|p| p.checked_add(offsets_start))
            .ok_or(Error::OutOfRange("archive index offset"))?;
        let offset = read_uint_be(data, off_pos, off_size)?;
        // First definition wins, matching archive extraction order. The key
        // is the stem: resolution asks for stemmed names, and `ar` indexes a
        // versioned definition under its raw spelling.
        let key = crate::symbol::version_stem(name);
        index.entry(Cow::Borrowed(key)).or_insert(offset);
    }
    Ok(())
}

/// Classifies a member by its 16-byte name field.
fn classify(raw: &[u8]) -> MemberKind {
    let end = raw.iter().position(|&b| b == b' ').unwrap_or(raw.len());
    match &raw[..end] {
        b"/" => MemberKind::SymIndex(false),
        b"//" => MemberKind::LongNames,
        b"/SYM64/" => MemberKind::SymIndex(true),
        _ => MemberKind::Regular,
    }
}

/// Drops the blank padding a fixed-width field ends with.
fn trim_blank(field: &[u8]) -> &[u8] {
    let end = field.iter().position(|&b| b == b' ').unwrap_or(field.len());
    &field[..end]
}

/// Rounds `n` up to an even offset. Members are always padded to two bytes.
#[allow(clippy::arithmetic_side_effects)]
fn align_even(n: usize) -> usize {
    (n + 1) & !1
}

/// Parses a right-justified, space-padded decimal field.
fn parse_decimal(field: &[u8]) -> Result<usize> {
    let mut n = 0usize;
    for &c in field {
        if c == b' ' {
            continue;
        }
        if !c.is_ascii_digit() {
            return Err(Error::Format("archive member size"));
        }
        n = n
            .checked_mul(10)
            .ok_or(Error::OutOfRange("archive member size"))?
            .checked_add(usize::from(c - b'0'))
            .ok_or(Error::OutOfRange("archive member size"))?;
    }
    Ok(n)
}

/// Reads a big-endian unsigned integer of `size` bytes at `pos`.
fn read_uint_be(data: &[u8], pos: usize, size: usize) -> Result<u64> {
    let bytes = data
        .get(pos..pos + size)
        .ok_or(Error::OutOfRange("archive integer"))?;
    let mut v = 0u64;
    for &b in bytes {
        v = v
            .checked_shl(8)
            .ok_or(Error::OutOfRange("archive integer"))?
            | u64::from(b);
    }
    Ok(v)
}

/// Returns the bytes up to the first NUL, or the whole slice if there is none.
fn cstr_slice(data: &[u8]) -> &[u8] {
    let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
    &data[..end]
}

fn to_usize(value: u64, what: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::OutOfRange(what))
}
