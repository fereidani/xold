//! Library search: what `-l NAME` resolves to.
//!
//! The resolver has to accept a real library, an archive and a linker script
//! alike while refusing a file that is none of the three, pick the newest
//! versioned shared object, and -- for a link that targets another
//! architecture -- look under a sysroot rather than at the build machine's own
//! libraries.

use std::{
    fs,
    path::{Path, PathBuf},
};

use xold::search::find_library;

/// A fresh per-test working directory under the system temp dir, following
/// the convention of the neighbouring test files.
fn workdir(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xold_search_{prefix}"));
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Writes a file with the given leading bytes, creating its directory.
fn write(dir: &Path, name: &str, bytes: &[u8]) {
    fs::create_dir_all(dir).expect("create dir");
    fs::write(dir.join(name), bytes).expect("write fixture");
}

/// The four bytes an ELF object opens with, padded so a short-read guard
/// cannot pass it by accident.
const ELF: &[u8] = b"\x7fELF\x02\x01\x01\x00 and then some contents";

/// What an `ar` archive opens with.
const ARCHIVE: &[u8] = b"!<arch>\ndebian-binary   ";

/// What a GNU linker script that names the real library looks like. Both
/// `libNAME.so` and `libNAME.a` are spelled this way on real systems.
const SCRIPT: &[u8] =
    b"/* GNU ld script */\nGROUP ( libc.so.6 libc_nonshared.a )\n";

/// A file that is none of the three things a candidate may be: not an object,
/// not an archive, and not text.
const GARBAGE: &[u8] = b"\x00\x01\x02\x03\xff\xfe not a linker input";

/// A real-ELF bare name wins over a versioned one.
///
/// This is the order every reference linker uses, and the reason is ABI
/// selection rather than recency: a developer package installs `libfoo.so` as
/// a symlink to the version it was built against, and a runtime package may
/// leave a newer `libfoo.so.2` beside it. Preferring the versioned file links
/// against 2 where everything else links against 1, with nothing said.
#[test]
fn a_real_bare_name_wins_over_a_versioned_one() {
    let work = workdir("search_versioned");
    let dir = work.join("lib");
    write(&dir, "libt.so", ELF);
    write(&dir, "libt.so.6", ELF);
    let found = find_library("t", std::slice::from_ref(&dir), None)
        .expect("resolves libt");
    assert_eq!(found.file_name().and_then(|n| n.to_str()), Some("libt.so"));
}

/// A bare name that is a linker script still leads.
///
/// The script is how the system spells which files the library is made of --
/// glibc's `libc.so` names `libc.so.6`, `libc_nonshared.a` and the loader --
/// so following it is what resolves the library, and taking the versioned
/// object instead would silently drop two thirds of it.
#[test]
fn a_bare_name_that_is_a_script_wins_over_a_versioned_object() {
    let work = workdir("search_script_bare");
    let dir = work.join("lib");
    write(&dir, "libt.so", SCRIPT);
    write(&dir, "libt.so.6", ELF);
    let found = find_library("t", std::slice::from_ref(&dir), None)
        .expect("resolves libt");
    assert_eq!(found.file_name().and_then(|n| n.to_str()), Some("libt.so"));
}

/// A versioned candidate is read like every other: nothing about the suffix
/// makes a file an object, and a file that is neither object nor script is
/// stepped over rather than handed to a reader that cannot parse it.
#[test]
fn a_versioned_candidate_that_is_neither_is_skipped() {
    let work = workdir("search_script_versioned");
    let dir = work.join("lib");
    write(&dir, "libt.so.6", GARBAGE);
    write(&dir, "libt.a", ARCHIVE);
    let found = find_library("t", std::slice::from_ref(&dir), None)
        .expect("falls through to the archive");
    assert_eq!(found.file_name().and_then(|n| n.to_str()), Some("libt.a"));
}

/// `-l:filename` names a file outright, with no prefix or suffix added.
#[test]
fn a_colon_name_resolves_the_file_as_written() {
    let work = workdir("search_colon");
    let dir = work.join("lib");
    write(&dir, "oddly-named.so", ELF);
    write(&dir, "libt.so", ELF);
    let found =
        find_library(":oddly-named.so", std::slice::from_ref(&dir), None)
            .expect("resolves the exact name");
    assert_eq!(
        found.file_name().and_then(|n| n.to_str()),
        Some("oddly-named.so")
    );
    assert_eq!(
        find_library(":no-such-file.so", std::slice::from_ref(&dir), None),
        None,
        "a -l:name that names nothing resolves to nothing, rather than \
         falling back to a libNAME spelling"
    );
}

#[test]
fn the_newest_version_wins_numerically_not_lexically() {
    let work = workdir("search_newest");
    let dir = work.join("lib");
    for suffix in ["1", "9", "10", "10.2.1"] {
        write(&dir, &format!("libt.so.{suffix}"), ELF);
    }
    let found = find_library("t", std::slice::from_ref(&dir), None)
        .expect("resolves libt");
    // `.10.2.1` beats `.10`, which beats `.9`; a lexical compare would stop
    // at `.9`.
    assert_eq!(
        found.file_name().and_then(|n| n.to_str()),
        Some("libt.so.10.2.1")
    );
}

/// A script is a resolution on its own: the driver expands it into the files
/// it names, so there is nothing to fall back to and nothing to fall back for.
#[test]
fn a_bare_name_that_is_a_linker_script_is_taken() {
    let work = workdir("search_script_so");
    let dir = work.join("lib");
    write(&dir, "libt.so", SCRIPT);
    let found = find_library("t", std::slice::from_ref(&dir), None)
        .expect("resolves libt to its script");
    assert_eq!(found.file_name().and_then(|n| n.to_str()), Some("libt.so"));
}

/// The `.a` slot takes a script too. `libgcc_s.a` and `libc.a` are spelled
/// that way on some distributions, and the name says nothing about which.
#[test]
fn an_archive_name_that_is_a_linker_script_is_taken() {
    let work = workdir("search_script_a");
    let dir = work.join("lib");
    write(&dir, "libt.a", SCRIPT);
    let found = find_library("t", std::slice::from_ref(&dir), None)
        .expect("resolves libt to its script");
    assert_eq!(found.file_name().and_then(|n| n.to_str()), Some("libt.a"));
}

/// A candidate that is neither an object, an archive nor text resolves to
/// nothing, at every one of the three names.
#[test]
fn a_candidate_that_is_none_of_the_three_is_refused() {
    let work = workdir("search_garbage");
    let dir = work.join("lib");
    write(&dir, "libt.so", GARBAGE);
    write(&dir, "libt.so.6", GARBAGE);
    write(&dir, "libt.a", GARBAGE);
    assert_eq!(find_library("t", std::slice::from_ref(&dir), None), None);
}

#[test]
fn a_bare_name_that_is_a_real_object_is_taken() {
    let work = workdir("search_bare_elf");
    let dir = work.join("lib");
    write(&dir, "libt.so", ELF);
    let found = find_library("t", std::slice::from_ref(&dir), None)
        .expect("resolves libt");
    assert_eq!(found.file_name().and_then(|n| n.to_str()), Some("libt.so"));
}

#[test]
fn a_file_shorter_than_the_magic_is_refused() {
    let work = workdir("search_short");
    let dir = work.join("lib");
    // Three bytes: a prefix of the ELF magic, but not the magic. The probe
    // reads into a zeroed buffer, so a comparison that ignored the byte count
    // would still have to reject this.
    write(&dir, "libt.so", b"\x7fEL");
    assert_eq!(find_library("t", std::slice::from_ref(&dir), None), None);
    write(&dir, "libt.a", b"");
    assert_eq!(find_library("t", std::slice::from_ref(&dir), None), None);
}

#[test]
fn a_sysroot_redirects_the_default_directories() {
    let work = workdir("search_sysroot");
    // `/usr/lib64` is one of the compiled-in defaults, so placing a library
    // there inside the sysroot is what a cross toolchain looks like.
    let inside = work.join("sysroot/usr/lib64");
    write(&inside, "libt.so.1", ELF);

    // Without the sysroot the name is not on the host, so the search fails.
    assert_eq!(find_library("t", &[], None), None);

    let found = find_library("t", &[], Some(&work.join("sysroot")))
        .expect("resolves under the sysroot");
    assert_eq!(found, inside.join("libt.so.1"));
}

#[test]
fn a_search_path_is_sysroot_relative_only_when_it_says_so() {
    let work = workdir("search_dash_l_rooting");
    let root = work.join("sysroot");
    write(&root.join("opt/lib"), "libt.so.1", ELF);
    // A plain `-L` names itself, so it does not move under the sysroot and
    // finds nothing.
    let plain = PathBuf::from("/opt/lib");
    assert_eq!(
        find_library("t", std::slice::from_ref(&plain), Some(&root)),
        None
    );
    // The `=` form asks for the remainder to be taken relative to it.
    let marked = PathBuf::from("=/opt/lib");
    let found = find_library("t", std::slice::from_ref(&marked), Some(&root))
        .expect("resolves under the sysroot");
    assert_eq!(found, root.join("opt/lib/libt.so.1"));
}

#[test]
fn search_paths_are_tried_before_the_defaults() {
    let work = workdir("search_order");
    let dir = work.join("lib");
    // `c` resolves to the host libc through the defaults; a `-L` directory
    // carrying its own must win.
    write(&dir, "libc.so.99", ELF);
    let found = find_library("c", std::slice::from_ref(&dir), None)
        .expect("resolves libc");
    assert_eq!(found, dir.join("libc.so.99"));
}
