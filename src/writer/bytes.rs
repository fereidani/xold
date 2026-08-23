//! Synthetic-section byte emission: serialises the PLT and dynamic synthetic
//! sections at their placed offsets in the loaded image.
//!
//! Every section here is sized by a counting pass ([`crate::dynamic::sizes`]
//! and the PLT entry count) and serialised by a different one, so each write
//! goes through [`write_region`], which reports a disagreement between the two
//! rather than letting the surplus bytes overwrite the next section.

use super::write_region;
use crate::{
    dynamic::DynamicPlan,
    error::Result,
    layout::{Layout, Sect},
    plt::PltPlan,
};

/// Writes the PLT, GOT.PLT and `.rela.plt` bytes at their placed offsets.
pub(super) fn write_plt_bytes(
    plan: &PltPlan,
    layout: &Layout,
    image: &mut [u8],
) -> Result<()> {
    write_region(image, layout.region(Sect::Plt), &plan.plt, ".plt")?;
    write_region(
        image,
        layout.region(Sect::GotPlt),
        &plan.got_plt,
        ".got.plt",
    )?;
    write_region(
        image,
        layout.region(Sect::RelaPlt),
        bytemuck::cast_slice(&plan.rela_plt),
        ".rela.plt",
    )
}

/// Writes the dynamic synthetic section bytes at their placed offsets.
pub(super) fn write_dynamic_bytes(
    plan: &DynamicPlan,
    image: &mut [u8],
) -> Result<()> {
    let r = &plan.regions;
    write_region(image, r.gnu_hash, &plan.gnu_hash, ".gnu.hash")?;
    write_region(image, r.hash, &plan.hash, ".hash")?;
    write_region(
        image,
        r.dynsym,
        bytemuck::cast_slice(&plan.dynsym),
        ".dynsym",
    )?;
    write_region(image, r.dynstr, &plan.dynstr, ".dynstr")?;
    write_region(image, r.versym, &plan.versym, ".gnu.version")?;
    write_region(image, r.verneed, &plan.verneed, ".gnu.version_r")?;
    write_region(
        image,
        r.rela_dyn,
        bytemuck::cast_slice(&plan.rela_dyn),
        ".rela.dyn",
    )?;
    write_region(
        image,
        r.dynamic,
        bytemuck::cast_slice(&plan.dynamic),
        ".dynamic",
    )
}
