//! Output sections: how allocated input sections are grouped for the image.
//!
//! xold emits a fixed set of output sections (`text`, `rodata`, `eh_frame`,
//! `data`, `data_rel_ro`, `bss`, the TLS pair `tdata`/`tbss` and the
//! `preinit_array`/`init_array`/`fini_array` trio) in canonical load order. An
//! input section is routed by its type and flags, with `.eh_frame`, `.init`,
//! `.fini` and the `.data.rel.ro` family matched by name; there is no
//! linker-script-driven section map.

use OutKind::{
    Bss, Data, DataRelRo, EhFrame, Fini, FiniArray, Init, InitArray, Note,
    PreinitArray, Rodata, Tbss, Tdata, Text,
};

use crate::elf::{
    Shdr64,
    constants::{
        SHF_EXECINSTR, SHF_TLS, SHF_WRITE, SHT_FINI_ARRAY, SHT_INIT_ARRAY,
        SHT_NOBITS, SHT_NOTE, SHT_PREINIT_ARRAY,
    },
};

/// The default output sections xold produces, in fixed slot order.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum OutKind {
    Text,
    Rodata,
    /// Allocated notes (`.note.ABI-tag`, `.note.gnu.property`,
    /// `.note.gnu.build-id`): `SHT_NOTE` with `SHF_ALLOC`.
    ///
    /// Kept out of `.rodata` because the type is the point. A note is only
    /// findable through a `PT_NOTE` segment, and a `PT_NOTE` can only cover a
    /// section the header says is `SHT_NOTE`. Folded into `.rodata` the bytes
    /// survive and nothing can read them: `readelf -n` shows nothing, a
    /// core-dump matcher or debuginfod finds no build id, and the CET and BTI
    /// properties the loader would act on are dead weight. lld emits one
    /// `PT_NOTE` per contiguous run of notes and keeps their `sh_type`.
    Note,
    /// The exception-handling frame section (`.eh_frame`): allocated,
    /// read-only. Kept as its own output section (rather than folded into
    /// `.rodata`) so `.eh_frame_hdr` can index a contiguous, well-addressed
    /// region and `PT_GNU_EH_FRAME` can point the runtime unwinder at it.
    EhFrame,
    Data,
    /// Relocated read-only data (`.data.rel.ro` and its `.local`/`.rel.ro.*`
    /// spellings): a `const` object whose initialiser needs a relocation, so
    /// the compiler cannot leave it in `.rodata`. It is writable only while
    /// the loader applies those relocations, which is what puts it at the head
    /// of the read-write segment inside `PT_GNU_RELRO`. Folded into `.data` it
    /// would stay writable for the life of the process.
    DataRelRo,
    Bss,
    /// Initialised thread-local storage (`.tdata`): `SHF_TLS` `SHT_PROGBITS`.
    Tdata,
    /// Zero-initialised thread-local storage (`.tbss`): `SHF_TLS`
    /// `SHT_NOBITS`.
    Tbss,
    /// The image's own init function (`.init`), assembled from fragments
    /// that must stay contiguous and in link order: the C runtime's prologue
    /// opens the function and its epilogue closes it, so anything placed
    /// between them would run as part of it.
    Init,
    /// The image's own fini function (`.fini`), assembled the same way.
    Fini,
    /// Constructor array (`.init_array`): an allocated, writable
    /// `SHT_INIT_ARRAY` section whose slots hold function pointers the
    /// runtime calls at startup.
    InitArray,
    /// Destructor array (`.fini_array`): an allocated, writable
    /// `SHT_FINI_ARRAY` section whose slots hold function pointers the
    /// runtime calls at exit.
    FiniArray,
    /// Pre-initialisation array (`.preinit_array`): an allocated, writable
    /// `SHT_PREINIT_ARRAY` section whose slots hold function pointers the
    /// runtime calls before any `.init_array` constructor. It is allocated and
    /// writable like the other two arrays, so without its own output section
    /// it would fold into `.data` and the runtime would never walk it.
    PreinitArray,
}

/// The names an output section is chosen by rather than by flags alone.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Named {
    /// `.eh_frame`, whose region `.eh_frame_hdr` indexes.
    EhFrame,
    /// `.init`, whose fragments form one function.
    Init,
    /// `.fini`, likewise.
    Fini,
    /// A member of the `.data.rel.ro` family, which rides in the protected run
    /// rather than in `.data`.
    DataRelRo,
    /// Any other name: the flags decide.
    Other,
}

impl OutKind {
    /// Conventional section name.
    pub const fn name(self) -> &'static str {
        match self {
            Text => ".text",
            Rodata => ".rodata",
            Note => ".note",
            EhFrame => ".eh_frame",
            Data => ".data",
            DataRelRo => ".data.rel.ro",
            Bss => ".bss",
            Tdata => ".tdata",
            Tbss => ".tbss",
            Init => ".init",
            Fini => ".fini",
            InitArray => ".init_array",
            FiniArray => ".fini_array",
            PreinitArray => ".preinit_array",
        }
    }

    /// The number of output sections xold emits.
    pub const COUNT: usize = 14;

    /// The file-backed output sections in placement order, which is ascending
    /// address order. `.bss` and `.tbss` are `SHT_NOBITS`: they take memory but
    /// no file bytes, so nothing is copied for them.
    ///
    /// This is not the slot order: `.data` is numbered before the init/fini
    /// arrays but placed after them (the arrays ride in the protected run and
    /// `.data` does not), so a walk that needs ascending offsets must follow
    /// this list rather than [`Self::index`].
    pub const FILE_ORDER: [Self; 12] = [
        Note,
        Init,
        Text,
        Fini,
        Rodata,
        EhFrame,
        DataRelRo,
        PreinitArray,
        InitArray,
        FiniArray,
        Data,
        Tdata,
    ];

    /// Fixed slot within the output-section array.
    pub const fn index(self) -> usize {
        match self {
            Text => 0,
            Rodata => 1,
            EhFrame => 2,
            Data => 3,
            Bss => 4,
            Tdata => 5,
            Tbss => 6,
            InitArray => 7,
            FiniArray => 8,
            Init => 9,
            Fini => 10,
            PreinitArray => 11,
            DataRelRo => 12,
            Note => 13,
        }
    }

    /// Picks the output section for an allocated input section from its flags
    /// and the handful of names that override them. `SHF_TLS` is checked first
    /// so a TLS section never lands in `.data` or `.bss` (TLS sections also
    /// carry `SHF_WRITE|SHF_ALLOC`). `.eh_frame` is matched by name so the
    /// unwind data gets its own output region (`.eh_frame_hdr` indexes a
    /// contiguous, well-addressed range), and the `.data.rel.ro` family so its
    /// relocated pointers can be re-protected after startup instead of staying
    /// writable in `.data`. The init/fini array types are matched next so their
    /// function-pointer slots get their own output region (and a `DT_*_ARRAY`
    /// tag) rather than folding into `.data`, which would leave the runtime
    /// with no array to walk. All three array types are matched,
    /// `.preinit_array` included: it is allocated and writable, so the
    /// `SHF_WRITE` arm below would otherwise swallow it.
    pub fn from_shdr(shdr: &Shdr64, named: Named) -> Self {
        match named {
            Named::EhFrame => return EhFrame,
            Named::Init => return Init,
            Named::Fini => return Fini,
            Named::DataRelRo => return DataRelRo,
            Named::Other => {}
        }
        if shdr.sh_flags.get() & SHF_TLS != 0 {
            return if shdr.sh_type.get() == SHT_NOBITS {
                Tbss
            } else {
                Tdata
            };
        }
        if shdr.sh_type.get() == SHT_NOTE {
            return Note;
        }
        if shdr.sh_type.get() == SHT_NOBITS {
            return Bss;
        }
        if shdr.sh_type.get() == SHT_INIT_ARRAY {
            return InitArray;
        }
        if shdr.sh_type.get() == SHT_FINI_ARRAY {
            return FiniArray;
        }
        if shdr.sh_type.get() == SHT_PREINIT_ARRAY {
            return PreinitArray;
        }
        if shdr.sh_flags.get() & SHF_EXECINSTR != 0 {
            return Text;
        }
        if shdr.sh_flags.get() & SHF_WRITE != 0 {
            return Data;
        }
        Rodata
    }
}

/// One input section placed in an output section.
#[derive(Copy, Clone, Debug)]
pub struct Member {
    pub file: usize,
    pub section: u16,
    /// The member's on-disk size (`sh_size`). `NOBITS` members still report
    /// their memory size here.
    pub size: u64,
    /// The member's alignment (`sh_addralign`, at least 1).
    pub align: u64,
    /// The initialisation priority its input section name encodes, or
    /// [`NO_PRIORITY`] for a name that carries none. Read only by
    /// [`OutputSections::sort_init_fini`]; every other output section ignores
    /// it. Carried on the member because the name is in hand exactly once, in
    /// the classifier, and re-deriving it later means re-reading the input's
    /// section header table per member.
    pub priority: i32,
}

/// The priority of a section name that encodes none.
///
/// One greater than the lowest real priority, so an unsuffixed `.init_array`
/// sorts after every `.init_array.N`. This is the value lld's
/// `elf::getPriority` returns for the same names
/// (`lld/ELF/OutputSections.cpp`).
pub const NO_PRIORITY: i32 = 65536;

/// The initialisation priority encoded in an input section name: `N` for a
/// name of the form `foo.N`, and [`NO_PRIORITY`] for anything else.
///
/// This is lld's `elf::getPriority` (`lld/ELF/OutputSections.cpp`), less its
/// `.ctors`/`.dtors` arm. Those two lists run in the
/// reverse of the order they are laid out in, which is what the `65535 - v`
/// inversion is for, but neither reaches a sorted output section here: both are
/// ordinary `SHF_ALLOC|SHF_WRITE` `SHT_PROGBITS` sections, so
/// [`OutKind::from_shdr`] routes them into `.data`, which keeps input order.
/// Spelling the inversion out would be a rule nothing can apply.
pub fn init_priority(name: &[u8]) -> i32 {
    let Some(dot) = name.iter().rposition(|&b| b == b'.') else {
        return NO_PRIORITY;
    };
    let Some(suffix) = name.get(dot + 1..).and_then(|s| str::from_utf8(s).ok())
    else {
        return NO_PRIORITY;
    };
    // lld parses the suffix with `to_integer(.., 10)`, which takes an optional
    // leading `-` and nothing else, and answers "no priority" for anything
    // that is not a whole base-10 number. `str::parse` also accepts a leading
    // `+`, so that one spelling is turned away here.
    if suffix.starts_with('+') {
        return NO_PRIORITY;
    }
    suffix.parse().unwrap_or(NO_PRIORITY)
}

/// A single output section aggregating input members.
#[derive(Debug)]
pub struct OutputSection {
    pub kind: OutKind,
    /// Maximum alignment requirement across members.
    pub align: u64,
    /// Sum of member sizes, ignoring alignment padding (refined in layout).
    pub size: u64,
    pub members: Vec<Member>,
}

/// The default output sections, in canonical load order.
pub struct OutputSections {
    sections: [OutputSection; OutKind::COUNT],
}

impl Default for OutputSections {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputSections {
    /// Empty output sections in slot order (see [`OutKind::index`]):
    /// `text, rodata, eh_frame, data, bss, tdata, tbss, init_array,
    /// fini_array, init, fini, preinit_array, data_rel_ro`.
    pub fn new() -> Self {
        fn empty(kind: OutKind) -> OutputSection {
            OutputSection {
                kind,
                align: 1,
                size: 0,
                members: Vec::new(),
            }
        }
        Self {
            sections: [
                empty(Text),
                empty(Rodata),
                empty(EhFrame),
                empty(Data),
                empty(Bss),
                empty(Tdata),
                empty(Tbss),
                empty(InitArray),
                empty(FiniArray),
                empty(Init),
                empty(Fini),
                empty(PreinitArray),
                empty(DataRelRo),
                empty(Note),
            ],
        }
    }

    /// Appends an allocated input section to the output section `kind` names.
    /// The caller classifies (see [`OutKind::from_shdr`]) and builds the
    /// member, which lets the per-file classification run off the serial path.
    pub fn add(&mut self, kind: OutKind, member: Member) {
        let idx = kind.index();
        debug_assert!(
            idx < self.sections.len(),
            "output section index in range"
        );
        let slot = &mut self.sections[idx];
        slot.size = slot.size.saturating_add(member.size);
        if member.align > slot.align {
            slot.align = member.align;
        }
        slot.members.push(member);
    }

    /// Orders the members of the three initialisation arrays by the priority
    /// their input section names encode, lowest first, so a constructor
    /// declared `__attribute__((constructor(101)))` runs before one declared
    /// `(300)` whatever order the two objects appeared on the command line.
    ///
    /// This is lld's `OutputSection::sortInitFini`
    /// (`lld/ELF/OutputSections.cpp`), which its writer calls for
    /// `.init_array` and `.fini_array`.
    ///
    /// The sort must stay stable, and is: members of equal priority keep the
    /// input order the classifier's fold gave them. A translation unit's own
    /// constructors carry no priority and must run in the order it wrote them,
    /// and the placement that reaches the image has to remain a function of
    /// input order. lld sorts stably on the same key for the same reason.
    ///
    /// `.preinit_array` is ordered with the other two. lld's `sortSection`
    /// does not name it, and neither does GNU ld's default script, but only
    /// because neither carries a `.preinit_array.*` pattern to sort: a
    /// priority suffix means the same thing on all three.
    pub fn sort_init_fini(&mut self) {
        for kind in [PreinitArray, InitArray, FiniArray] {
            let Some(out) = self.sections.get_mut(kind.index()) else {
                continue;
            };
            out.members.sort_by_key(|m| m.priority);
        }
    }

    /// Drops every member for which `keep` is false, then rebuilds each output
    /// section's aggregate size and alignment from the survivors.
    fn filter_members(&mut self, keep: impl Fn(&Member) -> bool) {
        for out in &mut self.sections {
            if out.members.is_empty() {
                continue;
            }
            out.members.retain(&keep);
            recompute_aggregate(out);
        }
    }

    /// Drops every member whose `(file, section)` is not in `live`. Used by
    /// `--gc-sections` after the mark/sweep to remove garbage-collected input
    /// sections before layout measures them.
    pub fn retain_live(&mut self, live: &rustc_hash::FxHashSet<(usize, u16)>) {
        self.filter_members(|m| live.contains(&(m.file, m.section)));
    }

    /// Drops every member whose `(file, section)` is in `folded`. Used by ICF
    /// to remove the folded duplicates after they have been aliased onto a
    /// representative, so layout places and the writer copies only the
    /// survivors.
    pub fn drop_members(
        &mut self,
        folded: &rustc_hash::FxHashSet<(usize, u16)>,
    ) {
        self.filter_members(|m| !folded.contains(&(m.file, m.section)));
    }

    /// Raises every fold representative's alignment to the strictest one in
    /// its group, so a folded section's own requirement survives the fold.
    ///
    /// `fold_of` names the representative a member was folded onto, or `None`
    /// for a member that stands on its own. Call before the folded members are
    /// dropped: their alignments are read from the member list itself, which is
    /// what layout would have honoured had they stayed.
    ///
    /// A section declaring `__attribute__((aligned(64)))` folded onto a
    /// byte-identical twin that asked for 16 would otherwise be given the
    /// representative's address, which need not be 64-byte aligned. The
    /// declaration is part of the contract with the code, not decoration: it is
    /// how a hot function is kept off a cache-line boundary and how an
    /// architecture's alignment requirement for a jump target is met. lld
    /// raises it in the same place, as it retires the folded section
    /// (`InputSection::replace`, `lld/ELF/InputSection.cpp`).
    ///
    /// The map decides identity only -- which representative a member belongs
    /// to -- and the value folded into each entry is a maximum, which does not
    /// depend on the order the members are visited in.
    pub fn absorb_alignments(
        &mut self,
        fold_of: impl Fn(usize, u16) -> Option<(usize, u16)>,
    ) {
        let mut needed: rustc_hash::FxHashMap<(usize, u16), u64> =
            rustc_hash::FxHashMap::default();
        for out in &self.sections {
            for m in &out.members {
                let Some(rep) = fold_of(m.file, m.section) else {
                    continue;
                };
                let slot = needed.entry(rep).or_insert(1);
                *slot = (*slot).max(m.align);
            }
        }
        if needed.is_empty() {
            return;
        }
        for out in &mut self.sections {
            let mut hit = false;
            for m in &mut out.members {
                if let Some(&align) = needed.get(&(m.file, m.section))
                    && align > m.align
                {
                    m.align = align;
                    hit = true;
                }
            }
            if hit {
                recompute_aggregate(out);
            }
        }
    }

    /// Resizes the members of `kind` in place, `sizes` in member order, then
    /// rebuilds the section's aggregate size once.
    ///
    /// Used by the `.eh_frame` splitter, which re-emits every member as a
    /// packed sequence of records and so resizes all of them at once; a
    /// per-member call would walk every member of every section per
    /// `.eh_frame` in the link. A `sizes` shorter than the member list leaves
    /// the remaining members alone.
    pub fn set_member_sizes(&mut self, kind: OutKind, sizes: &[u64]) {
        let Some(out) = self.sections.get_mut(kind.index()) else {
            return;
        };
        for (m, &size) in out.members.iter_mut().zip(sizes) {
            m.size = size;
        }
        recompute_aggregate(out);
    }

    /// Resizes one member, then rebuilds its output section's aggregate size.
    /// Used by the merge pass, which shrinks a pool carrier to the
    /// deduplicated content it now contributes.
    pub fn set_member_size(&mut self, file: usize, section: u16, size: u64) {
        for out in &mut self.sections {
            let mut hit = false;
            for m in &mut out.members {
                if m.file == file && m.section == section {
                    m.size = size;
                    hit = true;
                }
            }
            if hit {
                recompute_aggregate(out);
            }
        }
    }

    /// Gathers the members of each `__start_`/`__stop_` run into one
    /// contiguous block inside its output section.
    ///
    /// `run_of` names the run a member belongs to (see
    /// [`crate::startstop`]), or `None` for an ordinary member. Each run is
    /// collected at the position its first member already held; every other
    /// member keeps its place. So the members of one run stay in first-seen
    /// input order, as do the runs relative to each other and to the members
    /// around them, which is what makes the resulting bounds a function of
    /// input order alone.
    ///
    /// A link that bounds no section leaves every member exactly where it was.
    pub fn group_runs(&mut self, run_of: impl Fn(usize, u16) -> Option<usize>) {
        let mut key: Vec<usize> = Vec::new();
        let mut first: rustc_hash::FxHashMap<usize, usize> =
            rustc_hash::FxHashMap::default();
        for out in &mut self.sections {
            key.clear();
            first.clear();
            let mut bounded = false;
            for (at, m) in out.members.iter().enumerate() {
                let run = run_of(m.file, m.section);
                bounded |= run.is_some();
                key.push(run.map_or(at, |run| *first.entry(run).or_insert(at)));
            }
            if bounded {
                regroup(&mut out.members, &key);
            }
        }
    }

    /// The output section of `kind`, if it has any members.
    pub fn section(&self, kind: OutKind) -> Option<&OutputSection> {
        self.sections
            .get(kind.index())
            .filter(|s| !s.members.is_empty())
    }

    /// All output sections, in canonical order.
    pub fn iter(&self) -> impl Iterator<Item = &OutputSection> {
        self.sections.iter()
    }
}

/// Rewrites `members` into ascending `(key, original position)` order.
///
/// The pair is unique per member, so the permutation is total and the result
/// is decided by the keys alone. A key never exceeds its member's own
/// position, which is what pulls a run forward to where it started rather than
/// pushing the members around it back.
fn regroup(members: &mut [Member], key: &[usize]) {
    let mut order: Vec<usize> = (0..members.len()).collect();
    order.sort_unstable_by_key(|&at| (key.get(at).copied().unwrap_or(at), at));
    let sorted: Vec<Member> = order
        .iter()
        .filter_map(|&at| members.get(at).copied())
        .collect();
    if sorted.len() == members.len() {
        members.copy_from_slice(&sorted);
    }
}

/// Recomputes an output section's total size and alignment from its members.
fn recompute_aggregate(out: &mut OutputSection) {
    let mut size = 0u64;
    let mut align = 1u64;
    for m in &out.members {
        size = size.saturating_add(m.size);
        if m.align > align {
            align = m.align;
        }
    }
    out.size = size;
    out.align = align;
}
