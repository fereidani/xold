//! PE import directory: a multi-DLL, multi-function import table.
//!
//! Builds the on-disk import directory from a set of `(DLL, function)` pairs
//! collected from the input objects' `__imp_<func>` references (mapped to a
//! DLL by [`super::dllmap`]). The directory is an array of
//! `ImageImportDescriptor` records, one per DLL, null-terminated; each
//! descriptor points at an import lookup table (ILT) and an import address
//! table (IAT) of 8-byte thunks, and each thunk points at an
//! `ImageImportByName` record (hint + NUL-terminated name). At load time the
//! loader overwrites each IAT thunk with the resolved function address.
//!
//! Layout within `.idata`:
//!
//! ```text
//! [ 0]  descriptors: (N + 1) * 20 bytes        (N DLLs, null-terminated)
//! [..]  ILT:          per DLL (count + 1) * 8   (null-terminated)
//! [..]  IAT:          per DLL (count + 1) * 8   (contiguous span)
//! [..]  hint/name:    2 + name + NUL, 2-aligned (one per function)
//! [..]  DLL names:    name + NUL                (one per DLL)
//! ```
//!
//! The IAT span is contiguous across DLLs so one data-directory entry covers
//! it. [`ImportPlan::finalize`] stamps absolute RVAs (descriptor pointers and
//! the ILT/IAT thunks) once the layout places `.idata`. The plan exposes each
//! function's IAT slot RVA, so a `call`/`mov` through an `__imp_<func>`
//! reference resolves to the slot address and the entry stub addresses
//! `ExitProcess` by name.

use crate::{
    coff::{
        constants::IMAGE_SIZEOF_IMPORT_DESCRIPTOR,
        pe::{DataDirectory, ImportDescriptor},
    },
    endian::U32,
    util::align_up_u32,
};

/// The prefix clang `-msvc` gives an imported function's IAT-slot symbol.
pub const IMP_PREFIX: &[u8] = b"__imp_";

/// One imported function: the DLL that exports it and its name.
#[derive(Clone)]
pub struct ImportEntry {
    pub dll: Vec<u8>,
    pub func: Vec<u8>,
}

/// The import-table layout: the `.idata` byte content, the IMPORT and IAT
/// data-directory entries, and each function's IAT slot RVA.
///
/// Offsets within the section are fixed at construction; `finalize` adds the
/// base RVA and stamps the pointer fields.
pub struct ImportPlan {
    bytes: Vec<u8>,
    /// `(function name, absolute RVA of its IAT slot)`, for the resolver and
    /// the entry stub.
    slots: Vec<(Vec<u8>, u32)>,
    groups: Vec<DllGroup>,
    descriptors_off: u32,
    /// Offset of the first IAT thunk; the IAT directory spans
    /// `[iat_off, iat_off + iat_len)`.
    iat_off: u32,
    iat_len: u32,
    import_dir_rva: u32,
    import_dir_size: u32,
    iat_rva: u32,
    iat_size: u32,
}

/// One DLL's laid-out position: its ILT, IAT and name-string offsets, and the
/// functions belonging to it (with their hint/name and IAT-slot offsets).
#[derive(Clone)]
struct DllGroup {
    name: Vec<u8>,
    name_off: u32,
    ilt_off: u32,
    iat_off: u32,
    funcs: Vec<FuncSlot>,
}

/// One function's laid-out position: its hint/name record offset and its IAT
/// thunk offset within `.idata`.
#[derive(Clone)]
struct FuncSlot {
    name: Vec<u8>,
    hint_off: u32,
    iat_slot_off: u32,
}

impl ImportPlan {
    /// Builds the plan for `entries`, which may span several DLLs and repeat
    /// names. The set is deduplicated, grouped by DLL and sorted (DLL name,
    /// then function name) for a reproducible layout. Returns a plan even for
    /// an empty set; callers that need no imports pass an empty slice and get
    /// zero-sized directories.
    pub fn new(entries: &[ImportEntry]) -> Self {
        let groups = build_groups(entries);
        let (bytes, groups, descriptors_off, iat_off, iat_len) =
            layout_bytes(groups);
        Self {
            bytes,
            slots: Vec::new(),
            groups,
            descriptors_off,
            iat_off,
            iat_len,
            import_dir_rva: 0,
            import_dir_size: 0,
            iat_rva: 0,
            iat_size: 0,
        }
    }

    /// The on-disk byte size, for the layout to reserve `.idata` space.
    pub fn size(&self) -> u32 {
        u32::try_from(self.bytes.len()).unwrap_or(u32::MAX)
    }

    /// Whether the plan imports nothing. An empty plan still serialises a
    /// null terminator descriptor, so callers that want no `.idata` section
    /// check this and pass `0` to the layout.
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Stamps absolute RVAs: the ILT/IAT thunks (pointing at each hint/name
    /// record), each descriptor (ILT, DLL name and IAT RVAs), and the
    /// data-directory entries. Also publishes the per-function IAT slot RVAs.
    /// `base_rva` is the virtual address of the `.idata` section.
    pub fn finalize(&mut self, base_rva: u32) {
        self.write_thunks(base_rva);
        self.write_descriptors(base_rva);
        self.import_dir_rva = base_rva.wrapping_add(self.descriptors_off);
        let n = u32::try_from(self.groups.len() + 1).unwrap_or(u32::MAX);
        let step = u32::try_from(IMAGE_SIZEOF_IMPORT_DESCRIPTOR).unwrap_or(0);
        self.import_dir_size = n.wrapping_mul(step);
        self.iat_rva = base_rva.wrapping_add(self.iat_off);
        self.iat_size = self.iat_len;
        self.slots.clear();
        for g in &self.groups {
            for f in &g.funcs {
                let rva = base_rva.wrapping_add(f.iat_slot_off);
                self.slots.push((f.name.clone(), rva));
            }
        }
    }

    /// Writes the ILT and IAT thunks: each points at the function's hint/name
    /// record RVA; each table ends with a null thunk (the buffer is
    /// zero-initialised, so the trailing null is already in place).
    fn write_thunks(&mut self, base_rva: u32) {
        for g in &self.groups {
            for (i, f) in g.funcs.iter().enumerate() {
                let hint_rva = base_rva.wrapping_add(f.hint_off);
                let ilt_slot =
                    g.ilt_off.wrapping_add(u32::try_from(i).unwrap_or(0) * 8);
                write_thunk(&mut self.bytes, ilt_slot, hint_rva);
                write_thunk(&mut self.bytes, f.iat_slot_off, hint_rva);
            }
        }
    }

    /// Writes one `ImageImportDescriptor` per DLL; the trailing null entry is
    /// already zero-filled in the byte buffer.
    fn write_descriptors(&mut self, base_rva: u32) {
        let step = u32::try_from(IMAGE_SIZEOF_IMPORT_DESCRIPTOR).unwrap_or(0);
        for (i, g) in self.groups.iter().enumerate() {
            let off = self
                .descriptors_off
                .wrapping_add(u32::try_from(i).unwrap_or(0) * step);
            let desc = ImportDescriptor {
                original_first_thunk: U32::new(
                    base_rva.wrapping_add(g.ilt_off),
                ),
                time_date_stamp: U32::new(0),
                forwarder_chain: U32::new(0),
                name: U32::new(base_rva.wrapping_add(g.name_off)),
                first_thunk: U32::new(base_rva.wrapping_add(g.iat_off)),
            };
            write_bytes(&mut self.bytes, off, bytemuck::bytes_of(&desc));
        }
    }

    /// The absolute RVA of `func`'s IAT slot, if it is imported.
    pub fn iat_slot_rva(&self, func: &[u8]) -> Option<u32> {
        self.slots
            .iter()
            .find(|(name, _)| name == func)
            .map(|(_, rva)| *rva)
    }

    /// The serialised bytes (finalised or not). Finalised bytes are written
    /// verbatim into `.idata`.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The IMPORT and IAT data-directory entries.
    pub fn directories(&self) -> [DataDirectory; 2] {
        [
            DataDirectory {
                virtual_address: U32::new(self.import_dir_rva),
                size: U32::new(self.import_dir_size),
            },
            DataDirectory {
                virtual_address: U32::new(self.iat_rva),
                size: U32::new(self.iat_size),
            },
        ]
    }
}

/// Groups `entries` by DLL (deduplicated), sorting DLLs by name and functions
/// within a DLL by name for a deterministic layout.
fn build_groups(entries: &[ImportEntry]) -> Vec<DllGroup> {
    let mut pairs: Vec<(&[u8], &[u8])> =
        entries.iter().map(|e| (&e.dll[..], &e.func[..])).collect();
    // Deduplicate exact (dll, func) pairs and sort by DLL then function so the
    // on-disk order is reproducible across runs and input orders.
    pairs.sort();
    pairs.dedup();
    let mut groups: Vec<DllGroup> = Vec::new();
    for (dll, func) in pairs {
        if func.is_empty() {
            continue;
        }
        if let Some(g) = groups.last_mut().filter(|g| g.name == dll) {
            g.funcs.push(FuncSlot {
                name: func.to_vec(),
                hint_off: 0,
                iat_slot_off: 0,
            });
            continue;
        }
        groups.push(DllGroup {
            name: dll.to_vec(),
            name_off: 0,
            ilt_off: 0,
            iat_off: 0,
            funcs: vec![FuncSlot {
                name: func.to_vec(),
                hint_off: 0,
                iat_slot_off: 0,
            }],
        });
    }
    groups
}

/// Computes the `.idata` layout, allocates the byte buffer and writes the
/// position-independent content (hint/name records and DLL name strings). The
/// thunk and descriptor fields hold RVAs and are stamped later by `finalize`.
/// Returns `(bytes, groups, descriptors_off, iat_off, iat_len)` with each
/// group's and function's offsets filled in.
#[allow(clippy::similar_names)] // ILT/IAT are standard PE import-table terms.
fn layout_bytes(
    mut groups: Vec<DllGroup>,
) -> (Vec<u8>, Vec<DllGroup>, u32, u32, u32) {
    let step = u32::try_from(IMAGE_SIZEOF_IMPORT_DESCRIPTOR).unwrap_or(0);
    let n = u32::try_from(groups.len()).unwrap_or(0);
    let descriptors_off = 0u32;
    let descriptors_len = n.wrapping_add(1).wrapping_mul(step);
    let ilt_len = groups
        .iter()
        .map(|g| u32::try_from(g.funcs.len() + 1).unwrap_or(0) * 8)
        .sum::<u32>();
    let ilt_off = descriptors_off.wrapping_add(descriptors_len);
    let iat_off = ilt_off.wrapping_add(ilt_len);
    let iat_len = ilt_len; // ILT and IAT are identical before binding.

    // The hint/name region follows the IAT; the DLL-name region follows that.
    let hint_region_len: u32 = groups
        .iter()
        .flat_map(|g| g.funcs.iter())
        .map(|f| hint_name_len(&f.name))
        .sum();
    let mut hint_cursor = iat_off.wrapping_add(iat_len);
    let mut name_cursor = hint_cursor.wrapping_add(hint_region_len);

    let mut ilt_cursor = ilt_off;
    let mut iat_cursor = iat_off;
    for g in &mut groups {
        g.ilt_off = ilt_cursor;
        g.iat_off = iat_cursor;
        for (i, f) in g.funcs.iter_mut().enumerate() {
            f.hint_off = hint_cursor;
            hint_cursor = hint_cursor.wrapping_add(hint_name_len(&f.name));
            f.iat_slot_off =
                iat_cursor.wrapping_add(u32::try_from(i).unwrap_or(0) * 8);
        }
        let advance = u32::try_from(g.funcs.len() + 1).unwrap_or(0) * 8;
        ilt_cursor = ilt_cursor.wrapping_add(advance);
        iat_cursor = iat_cursor.wrapping_add(advance);
        g.name_off = name_cursor;
        name_cursor = name_cursor.wrapping_add(name_len(&g.name));
    }

    let total = name_cursor;
    let mut bytes = vec![0u8; usize::try_from(total).unwrap_or(0)];
    for g in &groups {
        write_name(&mut bytes, g.name_off, &g.name);
        for f in &g.funcs {
            write_hint_name(&mut bytes, f.hint_off, &f.name);
        }
    }
    (bytes, groups, descriptors_off, iat_off, iat_len)
}

/// The byte length of a hint/name record for `func`: 2-byte hint plus the
/// NUL-terminated name, rounded up to a 2-byte boundary.
fn hint_name_len(func: &[u8]) -> u32 {
    let raw = 2u32
        .wrapping_add(u32::try_from(func.len()).unwrap_or(0))
        .wrapping_add(1);
    align_up_u32(raw, 2)
}

/// The byte length of a NUL-terminated DLL `name` string.
fn name_len(name: &[u8]) -> u32 {
    u32::try_from(name.len()).unwrap_or(0).wrapping_add(1)
}

/// Writes `src` at `off`, leaving every other byte alone. `.idata` is sized
/// from the same offsets, so an out-of-range write is a layout bug and must
/// not panic.
fn write_bytes(bytes: &mut [u8], off: u32, src: &[u8]) {
    let start = usize::try_from(off).unwrap_or(usize::MAX);
    let end = start.saturating_add(src.len());
    if let Some(slot) = bytes.get_mut(start..end) {
        slot.copy_from_slice(src);
    }
}

/// Writes the hint/name record at `off`: a zero 2-byte hint followed by the
/// NUL-terminated name. The trailing NUL and 2-byte padding rely on the buffer
/// being zero-initialised.
fn write_hint_name(bytes: &mut [u8], off: u32, func: &[u8]) {
    write_bytes(bytes, off.wrapping_add(2), func);
}

/// Writes the NUL-terminated DLL `name` at `off`. The trailing NUL relies on
/// the buffer being zero-initialised.
fn write_name(bytes: &mut [u8], off: u32, name: &[u8]) {
    write_bytes(bytes, off, name);
}

/// Writes a 64-bit import thunk (an RVA to `ImageImportByName`) at `off`.
fn write_thunk(bytes: &mut [u8], off: u32, rva: u32) {
    write_bytes(bytes, off, &u64::from(rva).to_le_bytes());
}
