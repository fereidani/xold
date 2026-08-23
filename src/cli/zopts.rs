//! The `-z` keywords.
//!
//! `-z` is where the ELF command line puts everything that describes the
//! image rather than its contents, and a compiler driver writes several of
//! them on every link: `clang` alone emits `-z relro`, `-z now` and
//! `-z noexecstack` for a hardened build.
//!
//! Each keyword lands in one of three places. Some set a bit the writer emits
//! ([`ZOptions`]). Some name what this linker already does -- the stack it
//! marks non-executable, the RELRO segment it always writes, the lazy binding
//! it defaults to -- and are accepted because the image the caller asked for
//! is the image they get. The rest ask for something else, and those are
//! refused by name rather than accepted and dropped, which is the rule the
//! whole command line follows: an option that is ignored changes the image it
//! was written to describe.

use xold::dynamic::ZOptions;

/// The keywords this linker already satisfies, with what makes each true.
///
/// Accepting one is not ignoring it. The test is whether the image the
/// keyword describes is the image this link produces, and for each of these
/// it is.
const SATISFIED: [(&str, &str); 6] = [
    (
        "noexecstack",
        "every image marks PT_GNU_STACK non-executable",
    ),
    (
        "relro",
        "every image with a dynamic section carries PT_GNU_RELRO",
    ),
    ("lazy", "PLT binding is lazy unless -z now is given"),
    (
        "notext",
        "a relocation into read-only text is refused outright",
    ),
    (
        "combreloc",
        "dynamic relocations are grouped, RELATIVE first, with DT_RELACOUNT",
    ),
    ("nodefaultlib", "no default library path is recorded"),
];

/// The keywords that describe an image this linker does not produce, with
/// what it produces instead.
const REFUSED: [(&str, &str); 8] = [
    ("execstack", "an executable stack is never emitted"),
    ("norelro", "the RELRO segment is not optional here"),
    (
        "text",
        "a relocation into read-only text is refused, not allowed",
    ),
    ("muldefs", "a duplicate strong definition is an error"),
    ("nocombreloc", "dynamic relocations are always grouped"),
    ("separate-code", "the segment layout is not configurable"),
    ("noseparate-code", "the segment layout is not configurable"),
    (
        "pack-relative-relocs",
        "RELR relocations are not implemented",
    ),
];

/// Applies one `-z` keyword.
pub fn keyword(z: &mut ZOptions, kw: &str) -> Result<(), String> {
    match kw {
        "now" => z.now = true,
        // `-z lazy` is the default, and is written to undo an earlier
        // `-z now` on the same command line.
        "lazy" => z.now = false,
        "origin" => z.origin = true,
        "nodelete" => z.nodelete = true,
        "nodlopen" => z.nodlopen = true,
        "initfirst" => z.initfirst = true,
        "interpose" => z.interpose = true,
        "global" => z.global = true,
        _ => return other(z, kw),
    }
    Ok(())
}

/// The keywords that are not a plain switch: the ones carrying a value, the
/// ones already satisfied, and the ones refused.
fn other(z: &mut ZOptions, kw: &str) -> Result<(), String> {
    if let Some(value) = kw.strip_prefix("stack-size=") {
        z.stack_size = Some(parse_size(value)?);
        return Ok(());
    }
    if SATISFIED.iter().any(|(name, _)| *name == kw) {
        return Ok(());
    }
    if let Some((_, instead)) = REFUSED.iter().find(|(name, _)| *name == kw) {
        return Err(format!(
            "xold: `-z {kw}` describes an image this linker does not \
             produce: {instead}"
        ));
    }
    Err(format!(
        "xold: unknown `-z {kw}`: the keywords this linker implements are \
         now, lazy, origin, nodelete, nodlopen, initfirst, interpose, global \
         and stack-size=N, and the ones it already satisfies are {}",
        SATISFIED
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Reads a byte count written plainly or in hexadecimal, as `ld` accepts it
/// for `-z stack-size`.
fn parse_size(value: &str) -> Result<u64, String> {
    let parsed = value.strip_prefix("0x").map_or_else(
        || value.parse::<u64>().ok(),
        |hex| u64::from_str_radix(hex, 16).ok(),
    );
    parsed.ok_or_else(|| {
        format!("xold: `-z stack-size={value}` is not a byte count")
    })
}
