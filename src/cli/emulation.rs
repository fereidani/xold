//! The `-m` emulation name, checked against what the inputs actually are.
//!
//! A compiler driver writes `-m elf_x86_64` on every link, and gcc's own
//! cross drivers rely on it: the name is how a driver that can produce
//! several targets says which one this link is. xold reads its target out of
//! the inputs instead, so the option is a claim to check rather than a
//! setting to apply -- and checking it is worth doing, because a command line
//! whose `-m` disagrees with its objects is a build that has gone wrong
//! somewhere earlier, and the ELF header alone would not say so.

use xold::elf::constants::{EM_AARCH64, EM_RISCV, EM_X86_64};

/// The emulation names this linker answers to, with the machine each one
/// names.
///
/// GNU `ld` spells the `AArch64` one `aarch64linux` and lld accepts
/// `aarch64elf` and `aarch64linux` both, so all the spellings a driver on
/// these targets emits are here. The ILP32 names (`aarch64linux32`,
/// `elf32lriscv`, `elf_i386`) are deliberately absent: this linker produces
/// 64-bit images, and a link that asks for a 32-bit one has to hear so.
const ELF_EMULATIONS: [(&str, u16); 4] = [
    ("elf_x86_64", EM_X86_64),
    ("aarch64linux", EM_AARCH64),
    ("aarch64elf", EM_AARCH64),
    ("elf64lriscv", EM_RISCV),
];

/// The emulation names that mean a PE32+ image, which the COFF path produces.
const COFF_EMULATIONS: [&str; 1] = ["i386pep"];

/// What an input turned out to be, as far as `-m` is concerned.
#[derive(Clone, Copy)]
pub enum Machine {
    /// An ELF link over inputs carrying this `e_machine`.
    Elf(u16),
    /// A COFF link.
    Coff,
    /// A Mach-O link.
    MachO,
    /// Nothing among the inputs said, so there is nothing to check against.
    Unknown,
}

/// Checks `name` against the machine the inputs settled on.
///
/// A name this linker does not implement is refused whether or not it matches,
/// for the reason every unknown option is: a link that quietly ignores the
/// option describing its target is the one that produces the wrong image.
pub fn check(name: &str, machine: Machine) -> Result<(), String> {
    if let Some(&(_, want)) =
        ELF_EMULATIONS.iter().find(|(emu, _)| *emu == name)
    {
        return match machine {
            Machine::Elf(have) if have == want => Ok(()),
            Machine::Elf(have) => Err(mismatch(
                name,
                &format!(
                    "the inputs are ELF objects for {}, not the machine \
                     `{name}` names",
                    machine_name(have)
                ),
            )),
            Machine::Coff => Err(mismatch(name, "the inputs are COFF")),
            Machine::MachO => Err(mismatch(name, "the inputs are Mach-O")),
            Machine::Unknown => Ok(()),
        };
    }
    if COFF_EMULATIONS.contains(&name) {
        return match machine {
            Machine::Coff | Machine::Unknown => Ok(()),
            Machine::Elf(_) => Err(mismatch(name, "the inputs are ELF")),
            Machine::MachO => Err(mismatch(name, "the inputs are Mach-O")),
        };
    }
    Err(format!(
        "unknown emulation `{name}`: this linker produces 64-bit ELF \
         for x86-64, AArch64 and RISC-V, and PE32+ for x86-64 ({}, {})",
        ELF_EMULATIONS
            .iter()
            .map(|(emu, _)| *emu)
            .collect::<Vec<_>>()
            .join(", "),
        COFF_EMULATIONS.join(", ")
    ))
}

/// What to call an `e_machine` in a diagnostic: the architecture's name when
/// it is one this linker knows, and the raw number otherwise.
fn machine_name(machine: u16) -> String {
    match machine {
        EM_X86_64 => "x86-64".into(),
        EM_AARCH64 => "AArch64".into(),
        EM_RISCV => "RISC-V".into(),
        other => format!("machine {other}"),
    }
}

/// The error for an emulation the inputs contradict.
fn mismatch(name: &str, why: &str) -> String {
    format!("`-m {name}` does not describe this link: {why}")
}
