//! Little-endian integers for on-disk structures.
//!
//! Every container format xold reads and writes -- ELF for x86-64, `AArch64`
//! and RISC-V, PE/COFF, and Mach-O -- stores its integers little-endian. The
//! *host* running the linker need not be: xold is meant to build and run
//! anywhere.
//!
//! The on-disk structs are `#[repr(C)]` and viewed over mapped bytes with
//! `bytemuck`, which reinterprets bytes and never reorders them. Declaring
//! their fields as plain `u32`/`u64` would therefore read a file's
//! little-endian bytes as host-endian integers: correct on a little-endian
//! host by coincidence, silently wrong on a big-endian one, in both directions
//! -- garbage in from every input and an invalid image out.
//!
//! These types make the on-disk order explicit instead. Each is
//! `#[repr(transparent)]` over the primitive, so the struct layout, the
//! `bytemuck` casts and the bulk copies the writer relies on are all
//! unchanged; only reading and writing a field converts. On a little-endian
//! host `from_le`/`to_le` are the identity and the conversion compiles away
//! entirely, so this costs nothing where it is not needed.
//!
//! Big-endian *targets* are a separate matter: they would need their own
//! relocation tables and are not supported. This is about the host.

use bytemuck::{Pod, Zeroable};

/// Defines a little-endian newtype over a primitive integer.
macro_rules! little_endian {
    ($(#[$doc:meta])* $name:ident, $prim:ty) => {
        $(#[$doc])*
        #[repr(transparent)]
        #[derive(Clone, Copy, Default, Pod, Zeroable)]
        pub struct $name($prim);

        impl $name {
            /// The value in host order.
            pub const fn get(self) -> $prim {
                <$prim>::from_le(self.0)
            }

            /// Stores a host-order value.
            pub const fn new(value: $prim) -> Self {
                Self(value.to_le())
            }
        }

        impl From<$prim> for $name {
            fn from(value: $prim) -> Self {
                Self::new(value)
            }
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::Debug::fmt(&self.get(), f)
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl Eq for $name {}
    };
}

little_endian!(
    /// A little-endian `u16` as stored on disk.
    U16, u16
);
little_endian!(
    /// A little-endian `u32` as stored on disk.
    U32, u32
);
little_endian!(
    /// A little-endian `u64` as stored on disk.
    U64, u64
);
little_endian!(
    /// A little-endian `i64` as stored on disk, used for signed addends and
    /// the `.dynamic` tags.
    I64, i64
);
