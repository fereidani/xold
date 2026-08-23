//! PE export directory: turns `__declspec(dllexport)` into an export table.
//!
//! clang `-msvc` records a dllexport as a normal external definition plus a
//! linker directive in the object's `.drectve` section: `/EXPORT:name` (with an
//! optional `,DATA` for data, or `=internalname` for an alias). An explicit
//! `,@ordinal` and `,NONAME` are refused rather than ignored; see
//! [`EXPORT_ORDINAL_BASE`]. [`collect_directives`] scans every input's
//! `.drectve` and returns the exported names; the minimal xold exporter keeps
//! only the name, assigning ordinals densely from [`EXPORT_ORDINAL_BASE`].
//!
//! [`ExportPlan`] lays the export directory out in a `.edata` section: the
//! `ImageExportDirectory` header, the function-RVA, name-pointer and
//! name-ordinal tables, then the DLL name and the sorted export name strings.
//! Offsets within the section are fixed at construction;
//! [`ExportPlan::finalize`] stamps absolute RVAs once the layout places
//! `.edata`, and resolves each export's function RVA through a caller-supplied
//! lookup (the symbol's laid-out address). The EXPORT data-directory entry is
//! published for the writer.

use crate::{
    coff::{
        CoffFile,
        constants::IMAGE_SIZEOF_EXPORT_DIRECTORY,
        pe::{DataDirectory, ExportDirectory},
    },
    endian::{U16, U32},
    error::{Error, Result},
    util::trim_nul,
};

/// The ordinal the first export takes.
///
/// Ordinals are assigned densely from here: the i-th name, in sorted order,
/// gets `EXPORT_ORDINAL_BASE + i`. lld honours an explicit `,@ordinal`; this
/// linker cannot place one, so it refuses the request instead of quietly
/// exporting the name at a different ordinal.
pub const EXPORT_ORDINAL_BASE: u32 = 1;

/// One exported symbol.
///
/// `name` is what the DLL publishes and `internal` is what the linker
/// resolves. `/EXPORT:name` sets both to `name`; the alias form
/// `/EXPORT:name=internalname` publishes `name` but resolves `internalname`.
#[derive(Clone)]
pub struct ExportEntry {
    pub name: Vec<u8>,
    pub internal: Vec<u8>,
}

/// The export-table layout: the `.edata` byte content, the EXPORT
/// data-directory entry, and the sorted export names (for the writer's RVA
/// resolution).
///
/// Offsets within the section are fixed at construction; `finalize` adds the
/// section base RVA and stamps the absolute-RVA fields.
pub struct ExportPlan {
    bytes: Vec<u8>,
    names: Vec<Vec<u8>>,
    /// The symbol each entry in `names` resolves through, in the same order.
    /// Equal to `names` except where a `/EXPORT:name=internalname` alias
    /// redirects the lookup.
    internals: Vec<Vec<u8>>,
    directory: DataDirectory,
    header_off: u32,
    func_table_off: u32,
    name_table_off: u32,
    ordinal_table_off: u32,
    dll_name_off: u32,
    name_string_offs: Vec<u32>,
}

impl ExportPlan {
    /// Builds the plan for `entries` exported under `dll_name` (the DLL's own
    /// name string, e.g. `lib.dll`). Names are deduplicated and sorted so the
    /// on-disk order is reproducible across runs and input orders. Returns a
    /// plan even for an empty set; callers that export nothing pass an empty
    /// slice and get a zero-sized directory.
    pub fn new(entries: &[ExportEntry], dll_name: &[u8]) -> Self {
        // Sort by the exported name (then the target, so a duplicated name
        // resolves the same way regardless of input order) and keep one row
        // per exported name.
        let mut rows: Vec<(Vec<u8>, Vec<u8>)> = entries
            .iter()
            .map(|e| (e.name.clone(), e.internal.clone()))
            .collect();
        rows.sort();
        rows.dedup_by(|a, b| a.0 == b.0);
        let (names, internals): (Vec<Vec<u8>>, Vec<Vec<u8>>) =
            rows.into_iter().unzip();

        let n = u32::try_from(names.len()).unwrap_or(0);
        let header_off: u32 = 0;
        let header_len =
            u32::try_from(IMAGE_SIZEOF_EXPORT_DIRECTORY).unwrap_or(0);
        let func_table_off = header_off.wrapping_add(header_len);
        let name_table_off = func_table_off.wrapping_add(n.wrapping_mul(4));
        let ordinal_table_off = name_table_off.wrapping_add(n.wrapping_mul(4));
        let dll_name_off = ordinal_table_off.wrapping_add(n.wrapping_mul(2));
        let dll_name_len = string_len(dll_name);

        let mut name_string_offs = Vec::with_capacity(names.len());
        let mut cursor = dll_name_off.wrapping_add(dll_name_len);
        for name in &names {
            name_string_offs.push(cursor);
            cursor = cursor.wrapping_add(string_len(name));
        }

        let total = cursor;
        let mut bytes = vec![0u8; usize::try_from(total).unwrap_or(0)];
        // A trailing NUL relies on the buffer being zero-initialised.
        write_bytes(&mut bytes, dll_name_off, dll_name);
        for (i, name) in names.iter().enumerate() {
            let off = name_string_offs[i];
            write_bytes(&mut bytes, off, name);
        }
        // The name-ordinal table pairs each sorted name with its function-table
        // index. Names and functions share the sorted order, so entry i maps to
        // function i.
        for i in 0..n {
            let off = ordinal_table_off.wrapping_add(i.wrapping_mul(2));
            write_u16(&mut bytes, off, u16::try_from(i).unwrap_or(u16::MAX));
        }

        Self {
            bytes,
            names,
            internals,
            directory: DataDirectory::default(),
            header_off,
            func_table_off,
            name_table_off,
            ordinal_table_off,
            dll_name_off,
            name_string_offs,
        }
    }

    /// The on-disk byte size, for the layout to reserve `.edata` space.
    pub fn size(&self) -> u32 {
        u32::try_from(self.bytes.len()).unwrap_or(u32::MAX)
    }

    /// The sorted, deduplicated export names, as published in the DLL's name
    /// table.
    pub fn names(&self) -> &[Vec<u8>] {
        &self.names
    }

    /// Each export paired with the symbol whose address it publishes, for the
    /// writer to resolve every RVA before [`Self::finalize`]. The two differ
    /// only for a `/EXPORT:name=internalname` alias.
    pub fn resolved_pairs(&self) -> impl Iterator<Item = (&[u8], &[u8])> {
        self.names
            .iter()
            .zip(&self.internals)
            .map(|(n, i)| (n.as_slice(), i.as_slice()))
    }

    /// Stamps absolute RVAs: the directory header pointers, each function's RVA
    /// (looked up by name through `resolve`) and each name-pointer RVA. Also
    /// publishes the EXPORT data-directory entry. `base_rva` is the virtual
    /// address of the `.edata` section.
    pub fn finalize(
        &mut self,
        base_rva: u32,
        resolve: impl Fn(&[u8]) -> Option<u32>,
    ) {
        let n = u32::try_from(self.names.len()).unwrap_or(0);
        let header = ExportDirectory {
            characteristics: U32::new(0),
            time_date_stamp: U32::new(0),
            major_version: U16::new(0),
            minor_version: U16::new(0),
            name: U32::new(base_rva.wrapping_add(self.dll_name_off)),
            base: U32::new(EXPORT_ORDINAL_BASE),
            number_of_functions: U32::new(n),
            number_of_names: U32::new(n),
            address_of_functions: U32::new(
                base_rva.wrapping_add(self.func_table_off),
            ),
            address_of_names: U32::new(
                base_rva.wrapping_add(self.name_table_off),
            ),
            address_of_name_ordinals: U32::new(
                base_rva.wrapping_add(self.ordinal_table_off),
            ),
        };
        write_pod(&mut self.bytes, self.header_off, &header);
        for (i, name) in self.internals.iter().enumerate() {
            let i = u32::try_from(i).unwrap_or(0);
            let func_rva = resolve(name).unwrap_or(0);
            write_u32(
                &mut self.bytes,
                self.func_table_off.wrapping_add(i.wrapping_mul(4)),
                func_rva,
            );
            let name_rva = base_rva.wrapping_add(
                *self.name_string_offs.get(i as usize).unwrap_or(&0),
            );
            write_u32(
                &mut self.bytes,
                self.name_table_off.wrapping_add(i.wrapping_mul(4)),
                name_rva,
            );
        }
        let size = u32::try_from(self.bytes.len()).unwrap_or(0);
        self.directory = DataDirectory {
            virtual_address: U32::new(base_rva.wrapping_add(self.header_off)),
            size: U32::new(size),
        };
    }

    /// The EXPORT data-directory entry (set by `finalize`).
    pub fn directory(&self) -> DataDirectory {
        self.directory
    }

    /// The serialised bytes (finalised or not). Finalised bytes are written
    /// verbatim into `.edata`.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Collects the exported names declared by every input's `.drectve` linker
/// directives. Duplicate names across inputs are collapsed by
/// [`ExportPlan::new`].
pub fn collect_directives(inputs: &[CoffFile<'_>]) -> Result<Vec<ExportEntry>> {
    let mut out = Vec::new();
    for input in inputs {
        for section in input.sections() {
            if !is_drectve(section.name) {
                continue;
            }
            parse_directives(section.data, &mut out)?;
        }
    }
    Ok(out)
}

/// Whether `name` is the `.drectve` linker-directive section.
fn is_drectve(name: &[u8]) -> bool {
    trim_nul(name) == b".drectve"
}

/// Parses `.drectve` `data` for export directives, appending each name to
/// `out`.
///
/// The recognised forms are `/EXPORT:name` and `-export:name`. Directives are
/// whitespace-separated tokens, each prefixed with `/` or `-`; the option
/// keyword and its value are separated by `:` (or `=`). The export name ends
/// at the first unquoted comma (data, ordinal, NONAME markers), `=` (alias)
/// or whitespace; quotes are processed out of the name as on the Windows
/// command line (see [`export_name`]). Unknown directives are ignored.
pub fn parse_directives(data: &[u8], out: &mut Vec<ExportEntry>) -> Result<()> {
    for raw in split_tokens(data)? {
        let Some(token) = strip_option_prefix(raw) else {
            continue;
        };
        let Some((key, value)) = split_option_value(token) else {
            continue;
        };
        if !eq_ignore_case(key, b"export") {
            continue;
        }
        let (name, internal) = export_name(value)?;
        if name.is_empty() {
            return Err(Error::Format("empty name in /EXPORT directive"));
        }
        out.push(ExportEntry { name, internal });
    }
    Ok(())
}

/// Splits `data` into whitespace-separated tokens, keeping double-quoted runs
/// together.
///
/// The Windows command-line rules: a `"` opens a quoted run in which
/// whitespace is literal, and `""` inside one is an escaped quote. The quotes
/// stay in the token -- the reader of an option's value decides what they
/// mean there, and for `/EXPORT` they carry meaning of their own. lld reads
/// its directives through `cl::TokenizeWindowsCommandLineNoCopy`
/// (`lld/COFF/DriverUtils.cpp`), which answers the same question.
fn split_tokens(data: &[u8]) -> Result<Vec<&[u8]>> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    let mut quoted = false;
    let mut i = 0;
    while i < data.len() {
        let b = data[i];
        if quoted {
            // A doubled quote inside a quoted run is one literal quote, not
            // the end of the run; it is consumed as a pair.
            if b == b'"' {
                if data.get(i + 1) == Some(&b'"') {
                    i += 2;
                    continue;
                }
                quoted = false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            quoted = true;
            start = Some(start.unwrap_or(i));
            i += 1;
            continue;
        }
        if b.is_ascii_whitespace() {
            if let Some(s) = start.take() {
                out.push(&data[s..i]);
            }
            i += 1;
            continue;
        }
        start = Some(start.unwrap_or(i));
        i += 1;
    }
    if quoted {
        return Err(Error::Format("unterminated quote in .drectve"));
    }
    if let Some(s) = start {
        out.push(&data[s..]);
    }
    Ok(out)
}

/// Strips a leading `/` or `-` option prefix from `token`.
fn strip_option_prefix(token: &[u8]) -> Option<&[u8]> {
    token
        .strip_prefix(b"/")
        .or_else(|| token.strip_prefix(b"-"))
}

/// Splits an option (prefix already stripped) into `(keyword, value)` at the
/// first `:` or `=`. Returns `None` if there is no separator.
fn split_option_value(token: &[u8]) -> Option<(&[u8], &[u8])> {
    let pos = token.iter().position(|&b| b == b':' || b == b'=')?;
    Some((&token[..pos], &token[pos + 1..]))
}

/// Reads the exported name and the symbol it resolves to from a directive
/// value: everything up to the first unquoted `,` or end, split at the first
/// unquoted `=`.
///
/// `/EXPORT:name` publishes and resolves `name`. `/EXPORT:name=internalname`
/// publishes `name` and resolves `internalname`, matching lld's `extName`
/// rule. A right-hand side containing `.` is a re-export forwarder
/// (`name=other.dll.entry`), which this linker does not implement; it is
/// rejected rather than silently exported as a local symbol.
///
/// Quotes follow the Windows command-line rules lld applies before it reads
/// an export (`cl::TokenizeWindowsCommandLineNoCopy`): a `"` toggles quoting
/// and is dropped, and inside a quoted run `""` is one literal quote that
/// keeps the run open. `/EXPORT:"a""b"` therefore names `a"b`, and a
/// mid-token `foo"bar baz"` names `foobar baz`. Separators quoted into the
/// name stay part of it, so `/EXPORT:"a=b"` is one name and not an alias.
fn export_name(value: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut name = Vec::with_capacity(value.len());
    let mut internal: Option<Vec<u8>> = None;
    let mut quoted = false;
    let mut i = 0;
    while i < value.len() {
        let b = value[i];
        if b == b'"' {
            // Inside a quoted run a doubled quote is one literal quote and
            // does not close the run.
            if quoted && value.get(i + 1) == Some(&b'"') {
                push_name_byte(&mut name, internal.as_mut(), b'"');
                i += 2;
                continue;
            }
            quoted = !quoted;
            i += 1;
            continue;
        }
        if !quoted && b == b',' {
            check_export_params(&value[i + 1..])?;
            break;
        }
        // Only the first unquoted `=` splits; a later one belongs to the
        // internal name.
        if !quoted && b == b'=' && internal.is_none() {
            internal = Some(Vec::new());
            i += 1;
            continue;
        }
        push_name_byte(&mut name, internal.as_mut(), b);
        i += 1;
    }
    if quoted {
        // Defensive: `split_tokens` applies the same pairing rule, so a
        // token it produced cannot end inside a quoted run.
        return Err(Error::Format("unterminated quoted /EXPORT name"));
    }
    let Some(internal) = internal else {
        let plain = name.clone();
        return Ok((name, plain));
    };
    if internal.contains(&b'.') {
        return Err(Error::Format(
            "re-export forwarder in /EXPORT directive is not supported",
        ));
    }
    if internal.is_empty() {
        return Err(Error::Format("empty target in /EXPORT alias"));
    }
    Ok((name, internal))
}

/// Rejects the `/EXPORT:` parameters this linker cannot honour.
///
/// `,@N` asks for a specific ordinal and `,NONAME` asks for an ordinal-only
/// export. [`ExportPlan`] assigns ordinals densely from
/// [`EXPORT_ORDINAL_BASE`] in sorted-name order and has no way to place one,
/// so honouring either is out of scope. Ignoring them is worse than refusing:
/// the export still lands in the table, at whatever ordinal its name happened
/// to sort into, and a consumer importing this DLL by ordinal then binds the
/// wrong function with nothing to warn it. lld honours both
/// (`LinkerDriver::parseExport`), so this is a gap rather than a
/// disagreement.
///
/// `DATA`, `PRIVATE` and `CONSTANT` are accepted and ignored: none of them
/// changes which symbol an ordinal or a name reaches.
fn check_export_params(rest: &[u8]) -> Result<()> {
    for param in rest.split(|&b| b == b',') {
        let param = trim_ascii_space(param);
        if param.first() == Some(&b'@') {
            return Err(Error::Format(
                "an explicit ordinal in /EXPORT is not supported",
            ));
        }
        if eq_ignore_case(param, b"noname") {
            return Err(Error::Format("NONAME in /EXPORT is not supported"));
        }
    }
    Ok(())
}

/// `bytes` without leading or trailing ASCII spaces and tabs.
fn trim_ascii_space(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .map_or(start, |p| p + 1);
    &bytes[start..end]
}

/// Appends `b` to the alias target once a split has been seen, and to the
/// exported name before that.
fn push_name_byte(name: &mut Vec<u8>, internal: Option<&mut Vec<u8>>, b: u8) {
    match internal {
        Some(v) => v.push(b),
        None => name.push(b),
    }
}

/// Whether `a` equals `b` ignoring ASCII case.
fn eq_ignore_case(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| x.eq_ignore_ascii_case(y))
}

/// The byte length of a NUL-terminated copy of `s`.
fn string_len(s: &[u8]) -> u32 {
    u32::try_from(s.len()).unwrap_or(0).wrapping_add(1)
}

/// Writes `src` at `off`, leaving every other byte alone. The export table is
/// pre-sized from the same offsets, so an out-of-range write is a layout bug
/// and must not panic.
fn write_bytes(bytes: &mut [u8], off: u32, src: &[u8]) {
    let start = usize::try_from(off).unwrap_or(usize::MAX);
    let end = start.saturating_add(src.len());
    if let Some(slot) = bytes.get_mut(start..end) {
        slot.copy_from_slice(src);
    }
}

/// Writes a little-endian `u16` at `off`.
fn write_u16(bytes: &mut [u8], off: u32, value: u16) {
    write_bytes(bytes, off, &value.to_le_bytes());
}

/// Writes a little-endian `u32` at `off`.
fn write_u32(bytes: &mut [u8], off: u32, value: u32) {
    write_bytes(bytes, off, &value.to_le_bytes());
}

/// Writes a `Pod` value at `off`.
fn write_pod<T: bytemuck::Pod>(bytes: &mut [u8], off: u32, value: &T) {
    write_bytes(bytes, off, bytemuck::bytes_of(value));
}
