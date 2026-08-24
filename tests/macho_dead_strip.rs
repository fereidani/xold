//! `-dead_strip` is a real Mach-O section mark/sweep.
//!
//! The discarded object deliberately references a missing symbol.  A link
//! without collection must fail; with collection it succeeds and neither the
//! dead symbol nor its distinctive payload reaches the image.

use std::{fs, path::PathBuf, process::Command};

use common::which;
use xold::{
    input::Input,
    macho::{LinkOptions, link_macho_with_options},
};

mod common;

const MAIN: &[u8] = b"extern int live(void);\n\
    int main(void) { return live() == 7 ? 0 : 1; }\n";
const LIVE: &[u8] = b"int live(void) { return 7; }\n";
const DEAD: &[u8] = b"extern int deliberately_missing(void);\n\
    const char dead_payload[] = \"XOLD_DEAD_STRIP_SENTINEL\";\n\
    int dead(void) { return deliberately_missing() + dead_payload[0]; }\n";

#[test]
#[cfg_attr(miri, ignore = "needs clang, which Miri cannot spawn")]
fn dead_strip_drops_an_unreachable_object_section() {
    let Some(clang) = which("clang") else {
        eprintln!("skipping Mach-O dead strip: clang unavailable");
        return;
    };
    let dir = std::env::temp_dir()
        .join(format!("xold_macho_dead_strip_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let mut objects = Vec::new();
    for (name, source) in [("main", MAIN), ("live", LIVE), ("dead", DEAD)] {
        let src = dir.join(format!("{name}.c"));
        let obj = dir.join(format!("{name}.o"));
        fs::write(&src, source).unwrap();
        if !Command::new(&clang)
            .args(["--target=arm64-apple-darwin", "-c"])
            .arg(&src)
            .arg("-o")
            .arg(&obj)
            .status()
            .is_ok_and(|status| status.success())
        {
            eprintln!("skipping Mach-O dead strip: no arm64 Darwin target");
            let _ = fs::remove_dir_all(&dir);
            return;
        }
        objects.push(obj);
    }
    let inputs: Vec<Input<'_>> = objects
        .iter()
        .map(|path: &PathBuf| Input::Path(path))
        .collect();

    let unstripped = link_macho_with_options(
        &inputs,
        &dir.join("unstripped"),
        b"_main",
        &LinkOptions::default(),
    );
    assert!(
        unstripped.is_err(),
        "the control must see the dead section's undefined reference"
    );

    let stripped_path = dir.join("stripped");
    let stripped = link_macho_with_options(
        &inputs,
        &stripped_path,
        b"_main",
        &LinkOptions {
            dead_strip: true,
            ..LinkOptions::default()
        },
    );
    assert!(stripped.is_ok(), "dead-strip link failed: {stripped:?}");
    let image = fs::read(stripped_path).unwrap();
    assert!(
        !image
            .windows(b"XOLD_DEAD_STRIP_SENTINEL".len())
            .any(|window| window == b"XOLD_DEAD_STRIP_SENTINEL"),
        "discarded section bytes must not reach the image"
    );
    assert!(
        !image
            .windows(b"_dead".len())
            .any(|window| window == b"_dead"),
        "discarded section symbols must not reach the output table"
    );
    let _ = fs::remove_dir_all(dir);
}
