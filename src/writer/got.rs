//! GOT fill: writes the resolved global-offset-table entry values into the
//! `.got` region of the loaded image.

use crate::layout::{Layout, Sect};

/// Writes the resolved GOT entry values into the GOT region.
pub(super) fn fill_got(layout: &Layout, image: &mut [u8]) {
    let base =
        usize::try_from(layout.region(Sect::Got).offset).unwrap_or(usize::MAX);
    for (i, value) in layout.got_values.iter().enumerate() {
        let off = base.saturating_add(i * 8);
        if let Some(slot) = image.get_mut(off..off.saturating_add(8)) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
    }
}
