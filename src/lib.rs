//! xold: a maintainable, multi-threaded linker for ELF, Mach-O and COFF.
//!
//! The crate is organised around three observations from studying lld and mold:
//!
//! - Each binary format lives behind its own reader/writer module (`elf`,
//!   `macho`, `coff`) and produces a common view of sections, symbols and
//!   relocations consumed by the format-neutral core.
//! - Per-architecture relocation handling is driven by a declarative table
//!   keyed on a shared [`RelExpr`] semantic enum, so the scan and apply phases
//!   are written once instead of copy-pasted per architecture.
//! - All on-disk structures are read zero-copy from a memory-mapped file via
//!   `bytemuck`, with no `unsafe` in the parsing layer.
//!
//! # Deterministic output (hard requirement)
//!
//! The linked image is a pure function of the inputs. The same object files
//! and the same command line produce the same bytes, on every run, on any
//! machine, at any thread count. Treat this as a correctness property with the
//! same standing as producing a runnable binary at all, because a build system
//! that caches or compares link outputs depends on it.
//!
//! Most of the linker runs in parallel, so the rule has teeth:
//!
//! - **Nothing scheduling-dependent may reach the output.** A concurrent or
//!   sharded structure may decide *identity* -- which name is which symbol,
//!   whether two strings are equal -- but never *order*. Every ordering that
//!   reaches the image (symbol table order, member placement, GOT and PLT slot
//!   assignment, merged string layout) derives from input order (file index,
//!   then index within that file) or from content.
//! - **Iteration order of a hash container is not an ordering.** It is stable
//!   for one insertion sequence and nothing more. Walk a dense index or an
//!   explicitly sorted list.
//! - **Verify it rather than assuming it.** Link the same inputs at
//!   `RAYON_NUM_THREADS` of 1, 2, 4, 8 and 16 and compare hashes; a change
//!   exercised only at the default thread count has not been tested.
//!
//! Two places show what this costs and why it is worth it.
//! [`symbol::SymbolTable`] assigns ids in first-seen input order, which is
//! inherently serial, so the parallel fold in `symbol::bulk` reproduces that
//! order rather than abandoning it. [`merge`] deduplicates through a hash map
//! but emits pieces in first-seen order, never map order.
//!
//! The practical consequence for changes: a refactor or an optimisation must
//! produce byte-identical output to the commit before it. That is a far
//! stricter regression test than the suite, and it catches ordering mistakes
//! the suite cannot. A change that is *meant* to alter the output should say
//! so in its commit message and explain why the new bytes are correct.
//!
//! # Running under Miri
//!
//! `cargo +nightly miri test` interprets the readers, the relocation
//! arithmetic and the plan builders, which is where a zero-copy parser would
//! hide undefined behaviour. `.cargo/config.toml` carries the two flags the
//! run needs and says why each is there.
//!
//! Miri neither spawns processes nor maps files, so the end-to-end tests --
//! which compile with the host toolchain and link through a mapping -- are
//! marked `ignore` under it and the harness reports them as such. What is left
//! is every test that builds its input in memory.
//!
//! Those in-memory fixtures must be aligned. A `Vec<u8>` or a `[u8; N]` asks
//! for alignment 1 and gets more only because a real allocator rounds up;
//! Miri grants exactly what was asked, so a fixture handed straight to a
//! reader is refused there and accepted everywhere else.
//! [`input::AlignedBytes`] is the storage that makes the request explicit, and
//! is what the tests use.

#![warn(clippy::pedantic, clippy::nursery)]
#![allow(
    // Namespaced module prefixes on public items are clearer than the stripped
    // form clippy suggests for a library surface.
    clippy::module_name_repetitions,
    // Doc strings are kept short; mandatory docs on every item is noise here.
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_const_for_fn,
    // The linker has many pure accessors; annotating each is noise without
    // much safety gain in this codebase.
    clippy::must_use_candidate
)]

pub mod archive;
pub mod buildid;
pub mod coff;
pub mod comdat;
pub mod debug;
pub mod defsym;
pub mod detach;
pub mod dynamic;
pub mod ehframe;
pub mod elf;
pub mod endian;
pub mod error;
pub mod gc;
pub mod icf;
pub mod input;
pub mod layout;
pub mod linker;
#[cfg(feature = "lto")]
pub mod lto;
pub mod macho;
pub mod merge;
pub mod mmap_file;
pub mod output;
pub mod plt;
pub mod pool;
pub mod reloc;
pub mod script;
pub mod search;
pub mod startlib;
pub mod startstop;
pub mod symbol;
pub mod tls;
pub mod util;
pub mod versionscript;
pub mod writer;

pub use error::{Error, Result};
