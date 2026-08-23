//! COFF common (tentative) definitions and weak externals.
//!
//! A common is a symbol with section number `IMAGE_SYM_UNDEFINED` and a
//! non-zero value: the value is the byte size, not an address, and the linker
//! is expected to allocate the storage. Several inputs may declare the same
//! name; the largest wins and they all share one allocation. Nothing allocated
//! them, so `int g;` at file scope resolved to address zero and any store
//! through it faulted.
//!
//! A weak external is storage class `IMAGE_SYM_CLASS_WEAK_EXTERNAL` with an
//! auxiliary record naming a default symbol by table index. If some input
//! defines the name for real that definition wins; otherwise the reference
//! resolves to the default. Neither branch existed, so a weak external also
//! resolved to zero.
//!
//! The plan is built once from the inputs and consulted by the address
//! resolver. Allocation order is input order (file index, then symbol index
//! within the file), never the order a hash container happens to yield.

use crate::coff::CoffFile;

/// What a common allocation is aligned to at most. lld rounds a common's size
/// up to a power of two and clamps the result here, so a 4-byte int gets 4 and
/// nothing gets more than 32.
const MAX_COMMON_ALIGN: u32 = 32;

/// One allocated common: its name and its offset within the block.
struct Entry<'data> {
    name: &'data [u8],
    offset: u32,
    size: u32,
}

/// Storage assigned to the common definitions across all inputs.
///
/// The block is appended to `.bss`, which is uninitialised and zeroed by the
/// loader -- exactly the semantics a tentative definition asks for.
pub struct CommonsPlan<'data> {
    entries: Vec<Entry<'data>>,
    size: u32,
}

impl<'data> CommonsPlan<'data> {
    /// Allocates one slot per distinct common name across `inputs`.
    ///
    /// Names are visited in input order, and a repeated name keeps its first
    /// slot while growing it to the largest size any input declared -- the
    /// C tentative-definition rule, and what every COFF linker does.
    pub fn new(inputs: &[CoffFile<'data>]) -> Self {
        let mut entries: Vec<Entry<'data>> = Vec::new();
        for input in inputs {
            for sym in input.symbols().iter() {
                if !sym.is_common() {
                    continue;
                }
                match entries.iter_mut().find(|e| e.name == sym.name) {
                    Some(e) => e.size = e.size.max(sym.value),
                    None => entries.push(Entry {
                        name: sym.name,
                        offset: 0,
                        size: sym.value,
                    }),
                }
            }
        }
        let size = assign_offsets(&mut entries);
        Self { entries, size }
    }

    /// The byte size of the whole block, for the layout to reserve.
    pub const fn size(&self) -> u32 {
        self.size
    }

    /// The offset of `name`'s storage within the block.
    pub fn offset_of(&self, name: &[u8]) -> Option<u32> {
        self.entries
            .iter()
            .find(|e| e.name == name)
            .map(|e| e.offset)
    }

    /// Every allocated name with its offset, for the address resolver to seed
    /// its global map.
    pub fn iter(&self) -> impl Iterator<Item = (&'data [u8], u32)> + '_ {
        self.entries.iter().map(|e| (e.name, e.offset))
    }
}

/// Lays the entries out in order, each on its own alignment, and returns the
/// total size.
fn assign_offsets(entries: &mut [Entry<'_>]) -> u32 {
    let mut cursor: u32 = 0;
    for e in entries.iter_mut() {
        let align = common_align(e.size);
        cursor = cursor.next_multiple_of(align);
        e.offset = cursor;
        cursor = cursor.saturating_add(e.size);
    }
    cursor
}

/// The alignment lld gives a common of `size` bytes: the next power of two, at
/// least 1 and at most [`MAX_COMMON_ALIGN`].
fn common_align(size: u32) -> u32 {
    if size == 0 {
        return 1;
    }
    size.checked_next_power_of_two()
        .unwrap_or(MAX_COMMON_ALIGN)
        .min(MAX_COMMON_ALIGN)
}
