//! Deciding what to tell the plugin about each symbol it declared.
//!
//! This is the whole of the linker's half of LTO. The plugin knows what the
//! bitcode means and nothing about the rest of the link; the linker knows
//! which definition won and nothing about the bitcode. The nine `LDPR_*`
//! values are the entire vocabulary between them, and the plugin turns each
//! one into the flags that license it to inline, internalise or delete a
//! definition (`llvm/tools/gold/gold-plugin.cpp`, the switch that fills
//! `lto::SymbolResolution`).
//!
//! Two of those flags are worth stating plainly, because getting them wrong
//! is not a diagnostic:
//!
//! - **Prevailing** says this file's definition is the one the link chose. A
//!   file told it prevails when it does not gets its definition compiled into
//!   the image twice.
//! - **Visible to a regular object** says something outside the bitcode can
//!   reach this name. A definition wrongly reported as invisible is deleted by
//!   LTO while a real object still calls it, and the failure surfaces as a link
//!   error naming a symbol the user never wrote -- or, worse, as a call into
//!   whatever replaced it.
//!
//! So where the two can be confused the answer here is the conservative one:
//! a name a regular object touches is reported visible even when it is also
//! exported, because over-reporting visibility only costs an optimisation
//! while under-reporting it costs correctness.

use std::ffi::c_uint;

use crate::lto::api::{
    LDPR_PREEMPTED_IR, LDPR_PREEMPTED_REG, LDPR_PREVAILING_DEF,
    LDPR_PREVAILING_DEF_IRONLY, LDPR_PREVAILING_DEF_IRONLY_EXP,
    LDPR_RESOLVED_DYN, LDPR_RESOLVED_EXEC, LDPR_RESOLVED_IR, LDPR_UNDEF,
};

/// Where the definition that won a name lives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Winner {
    /// Nothing defines the name. For a reference this is an undefined
    /// symbol; the link reports it separately if that is an error.
    Nowhere,
    /// A relocatable object the linker read itself.
    Regular,
    /// A bitcode file, which may be the one being asked about.
    Bitcode,
    /// A shared library. A real definition always outranks one of these, so
    /// this is only ever the winner for a reference.
    Shared,
}

/// What became of one declaration in the file being asked about.
///
/// The three states were two bools, `defined` and `prevailing`, which between
/// them could spell a fourth that means nothing: a reference cannot prevail.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    /// A reference rather than a definition.
    Reference,
    /// A definition another file's copy outranked. Commons count as
    /// definitions here: they are ones the link may still replace.
    Preempted,
    /// The definition that won the name.
    Prevailing,
}

/// What the link knows about one symbol a plugin declared.
#[derive(Clone, Copy, Debug)]
pub struct Facts {
    /// What became of this file's declaration of the name.
    pub role: Role,
    /// Where the winning definition lives.
    pub winner: Winner,
    /// A regular object, a linker script, `-u`, or the entry symbol names
    /// this. Bitcode references do not count -- that is the point of the
    /// flag, and lld spells the same rule as `isUsedInRegularObj`.
    pub used_in_regular_obj: bool,
    /// The name reaches the dynamic symbol table, so a library the linker
    /// cannot see may interpose it.
    pub exported: bool,
}

/// The `LDPR_*` value describing `facts`.
#[must_use]
pub const fn resolution(facts: &Facts) -> c_uint {
    match facts.role {
        Role::Reference => match facts.winner {
            Winner::Nowhere => LDPR_UNDEF,
            Winner::Regular => LDPR_RESOLVED_EXEC,
            Winner::Bitcode => LDPR_RESOLVED_IR,
            Winner::Shared => LDPR_RESOLVED_DYN,
        },
        // Another definition won. Only a real definition can outrank one, so
        // the winner is an object or another bitcode file; a shared library
        // never preempts a definition at link time, and reporting it as a
        // regular preemption is the answer that keeps this file's copy out of
        // the image either way.
        Role::Preempted => match facts.winner {
            Winner::Bitcode => LDPR_PREEMPTED_IR,
            _ => LDPR_PREEMPTED_REG,
        },
        // This file's definition won. What remains is who can see it.
        Role::Prevailing => {
            if facts.used_in_regular_obj {
                LDPR_PREVAILING_DEF
            } else if facts.exported {
                LDPR_PREVAILING_DEF_IRONLY_EXP
            } else {
                LDPR_PREVAILING_DEF_IRONLY
            }
        }
    }
}

/// Whether a resolution says the plugin's copy of the definition is the one
/// that reaches the image.
///
/// The link uses this to know which names the compiled objects will define,
/// which is what lets it drop the placeholder it entered for them.
#[must_use]
pub const fn prevails(resolution: c_uint) -> bool {
    matches!(
        resolution,
        LDPR_PREVAILING_DEF
            | LDPR_PREVAILING_DEF_IRONLY
            | LDPR_PREVAILING_DEF_IRONLY_EXP
    )
}
