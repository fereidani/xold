//! Cross-member `.eh_frame` CIE deduplication.
//!
//! A CIE describes the calling convention its FDEs were compiled under, so
//! every translation unit built with the same flags emits an identical one.
//! xold used to keep all of them: an LLVM link carried 5268 CIEs in two
//! distinct shapes, 142 KiB of pure repetition. lld and mold emit one copy per
//! distinct CIE and point every FDE at it.
//!
//! - `identical_cies_are_emitted_once`: the output holds no two CIEs with the
//!   same bytes, while the distinct shapes the inputs contribute all survive.
//!   Fails without the fix, which emits one CIE per contributing member.
//! - `every_fde_reaches_a_cie`: each FDE's `CIE_pointer` -- the backwards
//!   distance to its CIE -- lands exactly on a CIE record. This is what the
//!   sharing puts at risk: a shared CIE lives in another member, so the
//!   distance spans a member boundary and no longer follows from the two
//!   records' offsets within one member.
//! - `throws_across_units_still_unwind`: the linked program throws out of
//!   several units that all share one CIE, and every throw reaches its handler.
//!
//! Gated on `clang++`, `gcc` (for the crt objects) and the system `ld.so`; if
//! any is missing the tests print a note and return.

use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use common::{
    crt_file, interpreter, libc_so, libgcc_s_so, libstdcxx_so, which,
};
use xold::elf::ObjectFile;

mod common;

/// Three units that throw and one that catches. Each throwing unit carries its
/// own copy of the same personality-bearing CIE, which is what there is to
/// share; the crt objects contribute a second, different shape, which there is
/// not.
const UNITS: [(&str, &str); 5] = [
    ("one", "int one(){ throw \"one\"; }\n"),
    ("two", "int two(){ throw \"two\"; }\n"),
    ("three", "int three(){ throw \"three\"; }\n"),
    (
        "main",
        "extern \"C\" int printf(const char *, ...);\n\
         int one(); int two(); int three();\n\
         int main(){\n\
             int n = 0;\n\
             try { one(); } catch (const char *e) { printf(\"caught: %s\\n\", e); ++n; }\n\
             try { two(); } catch (const char *e) { printf(\"caught: %s\\n\", e); ++n; }\n\
             try { three(); } catch (const char *e) { printf(\"caught: %s\\n\", e); ++n; }\n\
             return n == 3 ? 0 : 1;\n\
         }\n",
    ),
    // A non-personality CIE is the distinct shape that the Linux crt objects
    // otherwise contribute. Keeping it in the fixture makes the structural
    // dedup check meaningful on hosts without a Linux crt/sysroot.
    ("plain", "int plain(){ return 7; }\n"),
];

/// One `.eh_frame` record: whether it is an FDE, its offset, and its bytes.
struct Record {
    at: usize,
    fde: bool,
    bytes: Vec<u8>,
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn identical_cies_are_emitted_once() {
    let Some(prog) = link_fixture("dedup") else {
        return;
    };
    let records = eh_frame_records(&prog);
    let cies: Vec<&Record> = records.iter().filter(|r| !r.fde).collect();
    let distinct: HashSet<&[u8]> =
        cies.iter().map(|r| r.bytes.as_slice()).collect();
    assert!(
        cies.len() >= 2,
        "the fixture must contribute more than one CIE shape, got {}",
        cies.len()
    );
    assert_eq!(
        cies.len(),
        distinct.len(),
        "no two CIEs in the output may hold the same bytes: {} records for \
         {} distinct shapes",
        cies.len(),
        distinct.len()
    );
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn every_fde_reaches_a_cie() {
    let Some(prog) = link_fixture("reach") else {
        return;
    };
    let records = eh_frame_records(&prog);
    let cies: HashSet<usize> =
        records.iter().filter(|r| !r.fde).map(|r| r.at).collect();
    let mut fdes = 0usize;
    for record in records.iter().filter(|r| r.fde) {
        let field: [u8; 4] = record.bytes[4..8].try_into().expect("CIE ptr");
        let back = u32::from_le_bytes(field) as usize;
        let target = record
            .at
            .checked_add(4)
            .and_then(|f| f.checked_sub(back))
            .unwrap_or(usize::MAX);
        assert!(
            cies.contains(&target),
            "the FDE at {:#x} names {target:#x}, which is no CIE",
            record.at
        );
        fdes += 1;
    }
    assert!(fdes >= 3, "the fixture must contribute FDEs, got {fdes}");
}

#[test]
#[cfg_attr(miri, ignore = "needs the host toolchain, which Miri cannot spawn")]
fn throws_across_units_still_unwind() {
    let Some(prog) = link_fixture("unwind") else {
        return;
    };
    if !cfg!(target_os = "linux") {
        assert_exception_metadata_survives(&prog);
        return;
    }
    let out = Command::new(&prog)
        .output()
        .expect("linked program must be runnable");
    assert!(
        out.status.success(),
        "the program should exit 0, got {:?}",
        out.status
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "caught: one\ncaught: two\ncaught: three\n",
        "every throw must reach its handler"
    );
}

/// Compiles [`UNITS`] and links them into one executable under `tag`, or
/// `None` when the host lacks the toolchain to build it.
fn link_fixture(tag: &str) -> Option<PathBuf> {
    let clangxx = which("clang++")?;
    if cfg!(target_os = "linux") && which("gcc").is_none() {
        eprintln!("skipping CIE dedup tests: gcc unavailable (crt)");
        return None;
    }
    let dir = std::env::temp_dir().join("xold-cie-dedup").join(tag);
    fs::create_dir_all(&dir).expect("create workdir");
    let mut inputs = Vec::new();
    for (name, src) in UNITS {
        let obj = dir.join(format!("{name}.o"));
        let path = dir.join(format!("{name}.cpp"));
        fs::write(&path, src).expect("write source");
        let ok = Command::new(&clangxx)
            .args(["--target=x86_64-linux-gnu", "-fPIE", "-c"])
            .arg(&path)
            .arg("-o")
            .arg(&obj)
            .status()
            .ok()?
            .success();
        assert!(ok, "host clang++ must compile {name}.cpp");
        inputs.push(obj);
    }
    let prog = dir.join("prog");
    if cfg!(target_os = "linux") {
        inputs.push(crt_file("crti.o")?);
        inputs.push(crt_file("crt1.o")?);
        inputs.push(crt_file("crtn.o")?);
        inputs.push(libc_so()?);
        inputs.push(libstdcxx_so()?);
        // `_Unwind_Resume` lives in `libgcc_s`, which `libstdc++` names
        // undefined.
        inputs.push(libgcc_s_so()?);
        let interp = interpreter()?;
        xold::linker::link_dyn_exec(
            &inputs,
            &prog,
            b"_start",
            &interp,
            false,
            xold::icf::IcfMode::None,
            false,
        )
        .expect("xold C++ link must succeed");
    } else {
        xold::linker::link_shared(
            &inputs,
            &prog,
            None,
            false,
            xold::icf::IcfMode::None,
            false,
        )
        .expect("xold C++ shared link must succeed");
    }
    Some(prog)
}

/// Non-ELF hosts cannot execute the x86_64 image, so verify the exact unwind
/// contract the runtime consumes: every FDE reaches a retained CIE and the
/// language-specific handler table survived the link.
fn assert_exception_metadata_survives(prog: &Path) {
    let records = eh_frame_records(prog);
    let cies: HashSet<usize> =
        records.iter().filter(|r| !r.fde).map(|r| r.at).collect();
    let mut fdes = 0usize;
    for record in records.iter().filter(|r| r.fde) {
        let back = u32::from_le_bytes(
            record.bytes[4..8].try_into().expect("CIE pointer"),
        ) as usize;
        let target = record
            .at
            .checked_add(4)
            .and_then(|field| field.checked_sub(back))
            .expect("FDE points backward to its CIE");
        assert!(cies.contains(&target), "every FDE reaches a retained CIE");
        fdes += 1;
    }
    assert!(fdes >= 4, "throwers and catcher retain their FDEs");

    let lsda_cies: HashSet<usize> = records
        .iter()
        .filter(|r| !r.fde && cie_has_lsda(&r.bytes))
        .map(|r| r.at)
        .collect();
    assert!(!lsda_cies.is_empty(), "a catcher CIE advertises an LSDA");

    let bytes = fs::read(prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let eh = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".eh_frame")
        .expect("output carries .eh_frame");
    let rodata = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".rodata")
        .expect("the merged handler table is in .rodata");
    let ro_start = rodata.sh_addr.get();
    let ro_end = ro_start.saturating_add(rodata.sh_size.get());
    let mut lsda_fdes = 0usize;
    for record in records.iter().filter(|r| r.fde) {
        let back = u32::from_le_bytes(
            record.bytes[4..8].try_into().expect("CIE pointer"),
        ) as usize;
        let cie = record.at + 4 - back;
        if !lsda_cies.contains(&cie) {
            continue;
        }
        // This fixture's CIE advertises pcrel+sdata4 for the LSDA. After the
        // two four-byte PC fields, the FDE carries a one-byte augmentation
        // length followed by that signed displacement.
        assert_eq!(record.bytes.get(16), Some(&4), "one sdata4 LSDA field");
        let disp = i32::from_le_bytes(
            record.bytes[17..21].try_into().expect("LSDA displacement"),
        );
        let field = eh.sh_addr.get() + u64::try_from(record.at + 17).unwrap();
        let lsda = field.wrapping_add(i64::from(disp).cast_unsigned());
        assert!(
            lsda >= ro_start && lsda < ro_end,
            "the FDE's LSDA lands in the retained handler table"
        );
        lsda_fdes += 1;
    }
    assert!(lsda_fdes > 0, "a catcher FDE retains its LSDA pointer");
}

fn cie_has_lsda(bytes: &[u8]) -> bool {
    bytes
        .get(9..)
        .and_then(|tail| tail.split(|&byte| byte == 0).next())
        .is_some_and(|augmentation| augmentation.contains(&b'L'))
}

/// Walks the output `.eh_frame` into its records.
///
/// The walk must consume every byte up to the four-byte terminator that closes
/// the section: a length word of zero anywhere else ends it for the runtime
/// unwinder too, so stopping early here is the same defect, reported louder.
fn eh_frame_records(prog: &Path) -> Vec<Record> {
    let bytes = fs::read(prog).expect("read output");
    let obj = ObjectFile::parse(&bytes).expect("valid ELF");
    let shdr = obj
        .sections()
        .iter()
        .find(|s| obj.section_name(s) == b".eh_frame")
        .expect("output must carry .eh_frame");
    let data = obj.section_data(shdr).expect("eh_frame data");
    let mut at = 0usize;
    let mut records = Vec::new();
    while at + 8 <= data.len() {
        let len = u32::from_le_bytes(
            data[at..at + 4].try_into().expect("length word"),
        ) as usize;
        if len == 0 {
            break;
        }
        let end = at + 4 + len;
        assert!(end <= data.len(), "a record runs past .eh_frame");
        let cie_ptr = u32::from_le_bytes(
            data[at + 4..at + 8].try_into().expect("CIE pointer"),
        );
        records.push(Record {
            at,
            fde: cie_ptr != 0,
            bytes: data[at..end].to_vec(),
        });
        at = end;
    }
    assert_eq!(
        at + 4,
        data.len(),
        ".eh_frame must be one packed run closed by a single terminator; the \
         walk stopped at {at:#x} of {:#x}",
        data.len()
    );
    records
}
