"""Link scenarios and the per-linker flags that make them comparable.

A scenario is one link: an input list, the flags that select the image kind,
and whether the result is expected to run. Every linker gets the same objects
and semantically equivalent flags, so the timings measure the linker and not a
difference in the work requested.

A scenario whose corpus has not been generated, or whose toolchain pieces are
not installed, reports itself unavailable instead of failing the run.
"""

import glob
import os

import toolchain
from toolchain import Toolchain

CORPUS = toolchain.CORPUS_DIR
BENCH_DIR = toolchain.BENCH_DIR

# Libraries the C++ and LLVM links want, in link order. Ones the host does not
# have are dropped; a missing optional library shows up as an unresolved
# symbol, which the scenario reports as a failed link.
CPP_LIBS = ["stdc++", "m", "gcc_s", "c"]
LLVM_LIBS = ["stdc++", "m", "z", "zstd", "ffi", "edit", "xml2", "gcc_s", "c"]


def have_lib(name):
    """Whether -l<name> resolves to something installed."""
    return bool(toolchain.file_name("lib%s.so" % name)
                or toolchain.file_name("lib%s.a" % name))


def libs(names):
    return ["-l" + n for n in names if have_lib(n)]


def objects(*parts):
    return sorted(glob.glob(os.path.join(CORPUS, *parts)))


class Scenario:
    """One link, described once and driven by every linker."""

    def __init__(self, name, mode, objs, desc, pre=(), post=(), run=False,
                 interp=None, unavailable=None):
        self.name = name
        self.mode = mode
        self.objs = list(objs)
        self.desc = desc
        self.pre = list(pre)
        self.post = list(post)
        self.run = run
        self.interp = interp
        self.unavailable = unavailable

    def input_bytes(self):
        return sum(os.path.getsize(o) for o in self.objs if os.path.exists(o))

    def mode_flags(self, linker):
        """Flags that make `linker` build the same kind of image as the rest.

        xold always writes .eh_frame_hdr and never garbage-collects sections,
        so the others are told to match; otherwise they would be doing
        strictly less work. xold has no -static: a link with no shared
        dependency is already static for it.
        """
        if linker == "xold":
            if self.mode == "dynexec":
                return ["--dynamic-exec", "-dynamic-linker", self.interp]
            if self.mode == "shared":
                return ["-shared"]
            return []
        flags = ["--no-gc-sections", "--eh-frame-hdr"]
        if self.mode == "static":
            return flags + ["-static"]
        if self.mode == "dynexec":
            return flags + ["--dynamic-linker", self.interp]
        return flags + ["-shared"]

    def command(self, argv0, linker, out, extra=()):
        return (list(argv0) + self.mode_flags(linker) + list(extra)
                + self.pre + self.objs + self.post + ["-o", out])


def _static_wrap(tc):
    """Objects a static executable needs before and after the user objects."""
    pre = [tc.crt1, tc.crti, tc.crtbegin_static]
    post = [tc.libc_a, tc.libgcc_a, tc.libgcc_eh_a, tc.libc_a,
            tc.crtend, tc.crtn]
    return pre, [p for p in post if p]


def _dyn_wrap(tc, extra_libs):
    """Objects and libraries a dynamic executable needs around the objects."""
    pre = [tc.crt1, tc.crti, tc.crtbegin]
    post = ["-L" + tc.gcc_dir, "-L" + tc.libc_dir] + libs(extra_libs)
    return pre, post + [tc.crtend, tc.crtn]


def build(tc=None):
    """Every scenario, in a stable order, available or not."""
    tc = tc or Toolchain()
    out = []
    crt_ok = not tc.missing("crt1", "crti", "crtn", "crtbegin", "crtend",
                            "gcc_dir", "libc_dir")

    def add(name, mode, objs, desc, pre=(), post=(), run=False):
        why = None
        if not objs:
            why = "corpus not generated; run: python3 corpus.py"
        elif not crt_ok:
            why = "C runtime objects not found for %s" % tc.cc
        elif mode == "dynexec" and not tc.interp:
            why = "dynamic loader not found; set BENCH_INTERP"
        elif mode == "static" and not tc.libc_a:
            why = "no libc.a installed, so nothing to link statically against"
        out.append(Scenario(name, mode, objs, desc, pre, post, run,
                            tc.interp, why))

    c_objs = objects("c", "*.o")
    if crt_ok:
        s_pre, s_post = _static_wrap(tc)
        d_pre, d_post = _dyn_wrap(tc, ["c"])
    else:
        s_pre = s_post = d_pre = d_post = []

    add("c-static", "static", c_objs,
        "generated C corpus, static executable against libc.a",
        s_pre, s_post, run=True)
    add("c-dyn", "dynexec", c_objs,
        "generated C corpus, dynamic executable against libc.so",
        d_pre, d_post, run=True)

    cpp_objs = objects("cpp", "*.o")
    if crt_ok:
        cpp_pre, cpp_post = _dyn_wrap(tc, CPP_LIBS)
    else:
        cpp_pre = cpp_post = []
    add("cpp-dyn", "dynexec", cpp_objs,
        "C++ corpus: templates, COMDAT groups, eh_frame and DWARF, dynamic exe",
        cpp_pre, cpp_post, run=True)

    pic_objs = objects("pic", "*.o")
    add("shared", "shared", pic_objs,
        "same C++ corpus built -fPIC, shared object",
        [], (["-L" + tc.gcc_dir, "-L" + tc.libc_dir] + libs(CPP_LIBS)
             if crt_ok else []))

    llvm_objs = objects("llvm", "*", "*.o")
    if llvm_objs and crt_ok and tc.interp:
        llmain = toolchain.compile_object(os.path.join(BENCH_DIR, "llmain.c"))
        llvm_pre, llvm_post = _dyn_wrap(tc, LLVM_LIBS)
        llvm_objs = [llmain] + llvm_objs
    else:
        llvm_pre = llvm_post = []
    add("llvm", "dynexec", llvm_objs,
        "objects from the installed LLVM static archives, dynamic exe",
        llvm_pre, llvm_post)

    hello = None
    if crt_ok:
        hello = toolchain.compile_object(os.path.join(BENCH_DIR, "hello.c"))
    add("hello-static", "static", [hello] if hello else [],
        "one small C file against libc.a: archive resolution and latency",
        s_pre, s_post, run=True)
    add("hello-dyn", "dynexec", [hello] if hello else [],
        "one small C file against libc.so: the latency floor",
        d_pre, d_post, run=True)
    return out


def select(names, tc=None):
    """Scenarios whose name is in `names` (all of them when empty)."""
    all_scs = build(tc)
    if not names:
        return all_scs
    by_name = {s.name: s for s in all_scs}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise SystemExit("unknown scenario(s): %s\nknown: %s"
                         % (", ".join(unknown), ", ".join(by_name)))
    return [by_name[n] for n in names]
