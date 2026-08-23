//! Unit tests for the COFF/PE synthetic-directory builders: the `.edata`
//! export plan, the `.reloc` base-relocation plan, and the TLS directory.
//!
//! These exercise the plan structures directly (no linking), which is why they
//! live apart from the end-to-end `coff_*` tests.

use xold::{
    coff::{
        basereloc::RelocPlan,
        exports::{ExportEntry, ExportPlan, parse_directives},
        layout::TLS_DIR_SIZE,
        tls::{TLS_STRUCT_SIZE, TlsPlan},
    },
    error::Error,
};

// --- `.drectve` export directives ----------------------------------------

/// The common clang-msvc shape: a leading-space `/EXPORT:name`.
#[test]
fn parses_slash_export_directive() {
    let mut out = Vec::new();
    parse_directives(b" /EXPORT:add", &mut out).expect("directive parses");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].name, b"add");
}

/// The `-export:` dash form and case-insensitive keyword.
#[test]
fn parses_dash_export_case_insensitive() {
    let mut out = Vec::new();
    parse_directives(b"-export:Sub", &mut out).expect("directive parses");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].name, b"Sub");
}

/// The `,DATA` marker (and any other comma clause) is stripped from the name;
/// only the symbol name is kept.
#[test]
fn strips_data_marker() {
    let mut out = Vec::new();
    parse_directives(b"/EXPORT:counter,DATA /EXPORT:blob,PRIVATE", &mut out)
        .expect("directives parse");
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].name, b"counter");
    assert_eq!(out[1].name, b"blob");
}

/// An explicit ordinal cannot be honoured -- ordinals are assigned densely in
/// sorted-name order -- so it is refused. Accepting and ignoring it would put
/// the export at some other ordinal and silently mis-bind every consumer that
/// imports this DLL by ordinal.
#[test]
fn an_explicit_ordinal_is_refused() {
    let mut out = Vec::new();
    let err = parse_directives(b"/EXPORT:fn,@8", &mut out)
        .expect_err("an explicit ordinal is not supported");
    assert!(matches!(err, Error::Format(_)), "unexpected error: {err:?}");
}

/// `NONAME` asks for an ordinal-only export, which needs the same ordinal
/// placement, so it is refused for the same reason.
#[test]
fn noname_is_refused() {
    let mut out = Vec::new();
    let err = parse_directives(b"/EXPORT:fn,@8,NONAME", &mut out)
        .expect_err("NONAME is not supported");
    assert!(matches!(err, Error::Format(_)), "unexpected error: {err:?}");
}

/// The alias form publishes the left name but resolves the right one, which
/// is lld's `extName` rule: `e.extName = x; e.name = y`.
#[test]
fn alias_export_publishes_one_name_and_resolves_another() {
    let mut out = Vec::new();
    parse_directives(b"/EXPORT:pub_name=internal_name", &mut out)
        .expect("directives parse");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].name, b"pub_name");
    assert_eq!(out[0].internal, b"internal_name");
}

/// A plain export resolves the name it publishes.
#[test]
fn plain_export_resolves_the_name_it_publishes() {
    let mut out = Vec::new();
    parse_directives(b"/EXPORT:add", &mut out).expect("directive parses");
    assert_eq!(out[0].name, b"add");
    assert_eq!(out[0].internal, b"add");
}

/// The alias target ends at the first unquoted comma, so the trailing
/// parameters are not swept into the resolved symbol name.
#[test]
fn alias_target_stops_at_the_parameter_comma() {
    let mut out = Vec::new();
    parse_directives(b"/EXPORT:counter=real_counter,DATA", &mut out)
        .expect("directives parse");
    assert_eq!(out[0].name, b"counter");
    assert_eq!(out[0].internal, b"real_counter");
}

/// A quoted `=` is part of the name, not an alias separator.
#[test]
fn quoted_equals_is_not_an_alias_separator() {
    let mut out = Vec::new();
    parse_directives(b"/EXPORT:\"a=b\"", &mut out).expect("directives parse");
    assert_eq!(out[0].name, b"a=b");
    assert_eq!(out[0].internal, b"a=b");
}

/// A dotted right-hand side is a re-export forwarder, which this linker does
/// not implement. It must be refused rather than exported as a local symbol.
#[test]
fn forwarder_export_is_refused() {
    let mut out = Vec::new();
    let err = parse_directives(b"/EXPORT:open=api.dll.open", &mut out)
        .expect_err("a forwarder is not supported");
    assert!(matches!(err, Error::Format(_)), "unexpected error: {err:?}");
}

/// An alias with nothing after the `=` names no symbol at all.
#[test]
fn alias_with_an_empty_target_is_an_error() {
    let mut out = Vec::new();
    let err = parse_directives(b"/EXPORT:name=", &mut out)
        .expect_err("an empty alias target is an error");
    assert!(matches!(err, Error::Format(_)), "unexpected error: {err:?}");
}

/// The export plan resolves each function RVA through the alias target while
/// the published name table keeps the exported spelling.
#[test]
fn plan_resolves_an_alias_through_its_internal_name() {
    let entries = vec![ExportEntry {
        name: b"pub_name".to_vec(),
        internal: b"internal_name".to_vec(),
    }];
    let mut plan = ExportPlan::new(&entries, b"lib.dll");
    assert_eq!(plan.names(), [b"pub_name".to_vec()]);
    // Only the internal name resolves, so a lookup keyed on the published
    // name would leave the function RVA at zero.
    plan.finalize(0x2000, |name| (name == b"internal_name").then_some(0x1000));
    // The function-RVA table starts right after the 40-byte header.
    assert_eq!(read_u32(plan.bytes(), 40), 0x1000);
}

/// Unknown directives are ignored, not errors.
#[test]
fn ignores_unknown_directives() {
    let mut out = Vec::new();
    parse_directives(b"/OPT:NOW /EXPORT:add /VERBOSE", &mut out)
        .expect("directives parse");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].name, b"add");
}

/// An export directive whose name trims to empty is a hard error, not a
/// silently dropped export.
#[test]
fn empty_export_name_is_an_error() {
    let mut out = Vec::new();
    let err = parse_directives(b"/EXPORT:", &mut out)
        .expect_err("an empty export name must be rejected");
    assert!(matches!(err, Error::Format(_)));
}

/// A quoted export name is taken whole: the space inside the quotes is part
/// of the name, and the tokens behind it are still read.
#[test]
fn quoted_name_keeps_its_spaces() {
    let mut out = Vec::new();
    parse_directives(b"/EXPORT:\"my func\" /EXPORT:plain", &mut out)
        .expect("directives parse");
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].name, b"my func");
    assert_eq!(out[1].name, b"plain");
}

/// Inside quotes the comma is part of the name, not the `,DATA` marker
/// separator.
#[test]
fn quoted_name_keeps_its_commas() {
    let mut out = Vec::new();
    parse_directives(b"/EXPORT:\"a,b\",DATA", &mut out)
        .expect("directives parse");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].name, b"a,b");
}

/// A doubled quote inside a quoted name is one literal quote in the export,
/// as lld and link.exe read it: `"a""b"` names `a"b`, not `a""b`.
#[test]
fn doubled_quote_inside_a_name_is_one_quote() {
    let mut out = Vec::new();
    parse_directives(b"/EXPORT:\"a\"\"b\" /EXPORT:next", &mut out)
        .expect("directives parse");
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].name, b"a\"b");
    assert_eq!(out[1].name, b"next");
}

/// A quote opening mid-token is removed and its run kept whole:
/// `foo"bar baz"` names `foobar baz`, and the next token still parses.
#[test]
fn mid_token_quote_is_removed_and_keeps_the_run() {
    let mut out = Vec::new();
    parse_directives(b"/EXPORT:foo\"bar baz\" /EXPORT:next", &mut out)
        .expect("directives parse");
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].name, b"foobar baz");
    assert_eq!(out[1].name, b"next");
}

/// A quote that never closes is a malformed directive, not a name that
/// quietly swallows the rest of the section.
#[test]
fn unterminated_quote_is_an_error() {
    let mut out = Vec::new();
    let err = parse_directives(b"/EXPORT:\"open /EXPORT:after", &mut out)
        .expect_err("an unterminated quote must be rejected");
    assert!(matches!(err, Error::Format(_)));
    assert_eq!(out.len(), 0, "nothing may be taken from a broken section");
}

/// Finalise stamps the header and resolves the function RVA: the directory
/// points at the `.edata` base, names are sorted, and the export for `add`
/// lands at the caller-supplied RVA.
#[test]
fn finalize_stamps_header_and_function_rva() {
    let entries = vec![
        ExportEntry {
            name: b"sub".to_vec(),
            internal: b"sub".to_vec(),
        },
        ExportEntry {
            name: b"add".to_vec(),
            internal: b"add".to_vec(),
        },
    ];
    let mut plan = ExportPlan::new(&entries, b"lib.dll");
    // Names sort: add (ordinal 1), sub (ordinal 2).
    assert_eq!(plan.names(), [b"add".to_vec(), b"sub".to_vec()]);
    plan.finalize(0x2000, |name| {
        if name == b"add" {
            Some(0x1000)
        } else {
            Some(0x1020)
        }
    });
    let dir = plan.directory();
    assert_eq!(dir.virtual_address.get(), 0x2000);
    assert!(dir.size.get() > 0);
    // The function-RVA table starts right after the 40-byte header.
    let func0 = read_u32(plan.bytes(), 40);
    assert_eq!(func0, 0x1000);
}

// --- `.reloc` base relocations --------------------------------------------

/// Two sites on the same page share a block; the block is padded to a 4-byte
/// boundary.
#[test]
fn groups_same_page_and_pads() {
    let plan = RelocPlan::new(&[0x1008, 0x1010]);
    let b = plan.bytes();
    // header (8) + 2 entries (4) = 12, even count so no pad.
    assert_eq!(b.len(), 12);
    assert_eq!(read_u32(b, 0), 0x1000);
    assert_eq!(read_u32(b, 4), 12);
}

/// Sites on different pages produce one block per page.
#[test]
fn splits_across_pages() {
    let plan = RelocPlan::new(&[0x1008, 0x2008]);
    // Two blocks of header (8) + 1 entry (2) + 1 pad (2) = 12 each.
    assert_eq!(plan.bytes().len(), 24);
}

/// No sites yield an empty table.
#[test]
fn empty_when_no_sites() {
    let plan = RelocPlan::new(&[]);
    assert_eq!(plan.bytes(), []);
    let dir = plan.directory(0x3000);
    assert_eq!(dir.virtual_address.get(), 0x3000);
    assert_eq!(dir.size.get(), 0);
}

// --- TLS directory ---------------------------------------------------------

/// An empty plan still serialises the trailer and reports a zero VA directory
/// before finalisation.
#[test]
fn empty_plan_has_trailer_bytes() {
    let plan = TlsPlan::new(&[], 0, 0);
    assert_eq!(plan.size(), TLS_DIR_SIZE);
    assert_eq!(
        plan.bytes().len(),
        usize::try_from(TLS_DIR_SIZE).unwrap_or(0)
    );
    let dir = plan.directory();
    assert_eq!(dir.virtual_address.get(), 0);
    assert_eq!(dir.size.get(), 0);
}

/// Finalise stamps the template range, the index VA and the callback VA
/// against the supplied section RVAs and image base, and reports the directory
/// size as the `ImageTlsDirectory64` struct extent (40 bytes).
#[test]
fn finalize_stamps_directory_fields() {
    let base: u64 = 0x0001_4000_0000;
    let mut plan = TlsPlan::new(&[], 4, 0);
    plan.finalize(0x5000, 0x6000, base);
    let dir = plan.directory();
    assert_eq!(dir.virtual_address.get(), 0x6000);
    assert_eq!(dir.size.get(), TLS_STRUCT_SIZE);
    let bytes = plan.bytes();
    assert_eq!(read_u64(bytes, 0), base + 0x5000);
    assert_eq!(read_u64(bytes, 8), base + 0x5000 + 4);
    assert_eq!(read_u64(bytes, 16), plan.index_va(base));
    assert_eq!(
        read_u64(bytes, 24),
        base + 0x6000 + u64::from(TLS_STRUCT_SIZE)
    );
}

// --- helpers ---------------------------------------------------------------

/// Reads a little-endian `u32` at `off`, failing the test if it is truncated.
fn read_u32(bytes: &[u8], off: usize) -> u32 {
    let slot: [u8; 4] = bytes
        .get(off..off + 4)
        .and_then(|s| s.try_into().ok())
        .expect("4 bytes in range");
    u32::from_le_bytes(slot)
}

/// Reads a little-endian `u64` at `off`, failing the test if it is truncated.
fn read_u64(bytes: &[u8], off: usize) -> u64 {
    let slot: [u8; 8] = bytes
        .get(off..off + 8)
        .and_then(|s| s.try_into().ok())
        .expect("8 bytes in range");
    u64::from_le_bytes(slot)
}
