<div align="center">

# xold

[![Crates.io][crates-badge]][crates-url]
[![Documentation][doc-badge]][doc-url]
[![MIT licensed][mit-badge]][mit-url]

[crates-badge]: https://img.shields.io/crates/v/xold.svg?style=for-the-badge
[crates-url]: https://crates.io/crates/xold
[mit-badge]: https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge
[mit-url]: https://github.com/fereidani/xold/blob/main/LICENSE
[doc-badge]: https://img.shields.io/docsrs/xold?style=for-the-badge
[doc-url]: https://docs.rs/xold

</div>

`xold` is an experimental, multi-threaded linker for ELF, Mach-O and COFF,
written in Rust. It is built around two goals that usually pull against each
other: link speed close to the fastest linkers available, and a codebase that
stays small enough to read, review and change.

This is experimental software. It links real C and C++ programs, but it is not
a drop-in replacement for a production linker and should not be used where a
wrong image is expensive.

## Usage

```bash
cargo install xold
```

Then invoke it the way you would invoke `ld`:

```bash
xold -o app main.o util.o -L/usr/lib64 -lc
```

Or hand it to a compiler driver:

```bash
clang -fuse-ld=/path/to/xold main.c -o app
```

Options:

```
xold [-shared | --dynamic-exec | -static] [-pie | -no-pie]
     [--strip-all | --strip-debug] [--hash-style=sysv|gnu|both]
     [--build-id[=none|fast|md5|sha1|0xHEX]] [--version-script file]
     [--gc-sections | --no-gc-sections] [--relax | --no-relax]
     [--icf=all|--icf=safe|--icf=none] [--eh-frame-hdr] [--no-fork]
     [-m emulation] [-O level] [--threads N | --no-threads] [-z keyword]...
     [-Bstatic | -Bdynamic] [--as-needed | --no-as-needed]
     [--push-state | --pop-state] [-rpath dir]... [-u symbol]...
     [--export-dynamic] [-o output] [--entry symbol] [-soname name]
     [-dynamic-linker path] [--sysroot dir] [-L dir]... [-l lib]...
     <object files...>
```

That is the set a compiler driver actually emits, and each option does what
its name says: `-m` is checked against the inputs rather than assumed,
`--as-needed` and `-Bstatic` apply from where they are written, `-z now`
reaches `DT_FLAGS`, and `--strip-debug` really drops the debug sections.
Anything outside the set is refused rather than ignored, because an ignored
option silently produces an image other than the one the command line
described -- a `-z execstack` that is accepted and dropped tells a build it
got a permission it did not get.

## Using xold from Cargo

`xold` is selected the way any alternative linker is, with `clang`'s
`--ld-path`. In `~/.cargo/config.toml` or the project's `.cargo/config.toml`:

```toml
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-Clink-arg=--ld-path=/path/to/xold"]
```

Or for a single build, without touching any config file:

```bash
RUSTFLAGS="-Clink-arg=--ld-path=/path/to/xold" cargo build
```

Binaries, `cdylib`s, `dylib`s and proc macros all link this way, in both the
`dev` and `release` profiles; `xold` builds itself with itself. If a
`RUSTFLAGS` environment variable is already set it overrides the `rustflags`
key in `config.toml`, so add the link argument to it rather than expecting the
config file to apply, and `cargo build -v` prints the `rustc` command line if
you need to check which linker is really being used.

## Using xold from CMake

CMake does not know `xold` by name, so tell it about the linker once in a
toolchain file and then select it with `CMAKE_LINKER_TYPE` (CMake 3.29 or
later). The variables have to be set before the compiler checks run, which is
what makes a toolchain file the place for them rather than `CMakeLists.txt`:

```cmake
# xold-toolchain.cmake
set(CMAKE_C_USING_LINKER_XOLD "-fuse-ld=/path/to/xold")
set(CMAKE_CXX_USING_LINKER_XOLD "-fuse-ld=/path/to/xold")
```

```bash
cmake -S . -B build \
      -DCMAKE_TOOLCHAIN_FILE=/path/to/xold-toolchain.cmake \
      -DCMAKE_LINKER_TYPE=XOLD
cmake --build build
```

On an older CMake, or when you would rather not add a toolchain file, pass the
flag straight to the link step:

```bash
cmake -S . -B build \
      -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=/path/to/xold" \
      -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=/path/to/xold"
```

`-fuse-ld=` with a path works with `clang` and with `gcc` 16 or later. For an
older `gcc`, which only accepts `bfd`, `gold` and `lld` by name, put a symlink
named `ld` in a directory of its own and point the driver at it with `-B`:

```bash
mkdir -p /tmp/xold-bin && ln -sf /path/to/xold /tmp/xold-bin/ld
export CFLAGS="${CFLAGS} -B/tmp/xold-bin"
export CXXFLAGS="${CXXFLAGS} -B/tmp/xold-bin"
export LDFLAGS="${LDFLAGS} -B/tmp/xold-bin"
```

The same two spellings serve autotools and meson, which read `LDFLAGS`:

```bash
export LDFLAGS="${LDFLAGS} -fuse-ld=/path/to/xold"
```

One caveat for every route: `xold` does not implement link-time optimisation.
An object built with `-flto` holds bytecode and no code, and linking it is
refused with that reason rather than half-done. Configure without it
(`-fno-lto`), or add `-ffat-lto-objects`, which makes the compiler emit real
code beside the bytecode; such an object links like any other.

## What it does

- **ELF**, static and dynamic, for x86-64, AArch64 and RISC-V: PLT/GOT, lazy
  binding, copy relocations, `DT_INIT_ARRAY`, `.eh_frame_hdr` for C++
  exceptions, COMDAT deduplication, DWARF passthrough, static TLS, `-shared`
  output, and GNU ld linker scripts (so plain `-lc` works).
- **COFF/PE**: PE32+ executables and DLLs with imports, exports, base
  relocations and `__declspec(thread)` TLS.
- **Mach-O**: static executables for x86-64 and arm64. Dynamic output is not
  implemented.
- **Link-time optimisation**, through the same plugin interface gold and
  `ld.bfd` use: `clang -flto` and `-flto=thin` inputs are compiled by
  `LLVMgold.so` and linked as ordinary objects, on any of the three output
  formats. `-plugin` and `-plugin-opt=` are honoured, bitcode is extracted
  from archives and `--start-lib` groups the way native objects are, and LLVM
  stays out of the build: the plugin is loaded at run time, not linked
  against.
- Size and speed passes: `--gc-sections`, `--icf=all|safe`, `--strip-all`,
  `--strip-debug`, and relaxation of GOT-relative sequences.
- The command line a driver writes: `-m`, `-pie`, `-z`, `--as-needed`,
  `-Bstatic`/`-Bdynamic`, `--hash-style`, `--build-id`, `--version-script`,
  `-rpath`, `-u`, `--export-dynamic` and the rest, each implemented rather
  than accepted and dropped.

## Performance

Median wall clock linking a 2868-member LLVM 21 corpus (373 MB in, 155 MB out)
on an 8-core AMD EPYC-Rome, warm cache, flags equalised so every linker does
the same work:

| linker | median |
|---|---|
| wild | 0.18s |
| mold | 0.34s |
| xold | 0.35s |
| lld  | 1.41s |
| ld   | 4.39s |

Numbers move with the corpus and the machine; reproduce them with
`bench/bench.py`.

## Design

- Each binary format lives behind its own reader and writer and presents one
  common view of sections, symbols and relocations to a format-neutral core.
- Relocation handling is a declarative per-architecture table keyed on a shared
  `RelExpr` semantic enum, so the scan and apply phases are written once
  instead of copy-pasted per target.
- On-disk structures are read zero-copy from a memory-mapped file through
  `bytemuck`, with no `unsafe` in the parsing layer.
- Output is deterministic: the same inputs and the same command line produce
  the same bytes on any machine at any thread count. Concurrency may decide
  identity, never order.

## License

`xold` is licensed under the MIT license. See the `LICENSE` file for more
information.
