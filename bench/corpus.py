#!/usr/bin/env python3
"""Generate the benchmark corpora used by bench.py.

Three corpora, all built from sources generated here or from what is already
installed on the machine, so a fresh checkout on any system can reproduce the
numbers:

  c     many C translation units with a large flat symbol table
  cpp   C++ translation units with templates, COMDAT groups, DWARF, eh_frame
        (built twice: position dependent and -fPIC for the shared-object link)
  llvm  real objects extracted from the LLVM static archives installed here

Usage:
    python3 corpus.py all
    python3 corpus.py c --files 120 --globals 1800
    python3 corpus.py llvm --libdir /path/to/llvm/lib
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import toolchain
from toolchain import CORPUS_DIR, Toolchain, ToolchainError

C_DIR = os.path.join(CORPUS_DIR, "c")
CPP_DIR = os.path.join(CORPUS_DIR, "cpp")
PIC_DIR = os.path.join(CORPUS_DIR, "pic")
LLVM_DIR = os.path.join(CORPUS_DIR, "llvm")

C_FLAGS = ["-O0", "-ffunction-sections", "-fdata-sections"]
CPP_FLAGS = ["-O1", "-g", "-std=c++17"]

# LLVM archives whose members reference libraries outside the LLVM install
# (gtest, fuzzer runtimes). Linking them would fail on undefined symbols on
# every linker, so they stay out of the corpus unless asked for.
LLVM_SKIP = ("Testing", "Fuzz")


def jobs_default():
    return os.cpu_count() or 4


def compile_all(cmds, jobs):
    """Run compile commands concurrently, aborting on the first failure."""
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for cmd, p in zip(cmds, pool.map(
                lambda c: subprocess.run(c, capture_output=True, text=True), cmds)):
            if p.returncode != 0:
                raise ToolchainError("%s failed:\n%s" % (" ".join(cmd), p.stderr))


def reset(path):
    """Start a corpus directory from scratch."""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


# --- C corpus ---------------------------------------------------------------


def write_c_sources(out, nfiles, per):
    """One master unit plus `nfiles` units of `per` global symbols each."""
    with open(os.path.join(out, "master.c"), "w") as f:
        for i in range(nfiles):
            f.write("int f%d_0(void);\n" % i)
        f.write("int main(void) {\n  volatile int s = 0;\n")
        for i in range(nfiles):
            f.write("  s += f%d_0();\n" % i)
        f.write("  return s & 0xff;\n}\n")
    half = max(per // 2, 1)
    for i in range(nfiles):
        with open(os.path.join(out, "file%d.c" % i), "w") as f:
            for j in range(half):
                f.write("int v%d_%d;\n" % (i, j))
            for j in range(half):
                f.write("int f%d_%d(void) { v%d_%d += %d; return v%d_%d; }\n"
                        % (i, j, i, j, j, i, j))


def build_c(args, tc):
    reset(C_DIR)
    write_c_sources(C_DIR, args.files, args.globals)
    srcs = sorted(glob.glob(os.path.join(C_DIR, "*.c")))
    flags = args.cflags.split() if args.cflags else C_FLAGS
    compile_all([[tc.cc, "-c", s, "-o", s[:-2] + ".o", *flags] for s in srcs],
                args.jobs)
    report("c", C_DIR)


# --- C++ corpus -------------------------------------------------------------

CPP_HEADER = r"""#pragma once
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

template <class K, class V>
struct Store {
    std::map<K, V> m;
    std::unordered_map<K, V> h;
    std::vector<V> order;
    void put(const K& k, const V& v) { m[k] = v; h[k] = v; order.push_back(v); }
    V get(const K& k) const {
        auto it = m.find(k);
        if (it == m.end()) throw std::runtime_error("missing");
        return it->second;
    }
    std::string dump() const {
        std::ostringstream os;
        for (const auto& p : m) os << p.first << '=' << p.second << ';';
        return os.str();
    }
};

struct Node {
    virtual ~Node() = default;
    virtual int eval() const = 0;
    virtual std::string name() const { return "node"; }
};

template <int Tag>
struct Leaf : Node {
    int v;
    explicit Leaf(int v) : v(v) {}
    int eval() const override { return v + Tag; }
    std::string name() const override { return "leaf" + std::to_string(Tag); }
};

template <class T>
T reduce(const std::vector<T>& xs, std::function<T(T, T)> f, T init) {
    for (const auto& x : xs) init = f(init, x);
    return init;
}
"""

CPP_UNIT = r"""#include "common.h"
int work%(i)d(int seed) {
    Store<std::string, int> si;
    Store<int, std::string> is;
    for (int k = 0; k < 8; ++k) {
        si.put("k" + std::to_string(k + %(i)d), k * seed);
        is.put(k, "v" + std::to_string(k));
    }
    std::vector<std::unique_ptr<Node>> ns;
    ns.push_back(std::make_unique<Leaf<%(i)d>>(seed));
    ns.push_back(std::make_unique<Leaf<%(j)d>>(seed + 1));
    int acc = 0;
    for (const auto& n : ns) acc += n->eval() + (int)n->name().size();
    std::vector<int> xs(16);
    for (int k = 0; k < 16; ++k) xs[k] = k ^ seed;
    std::sort(xs.begin(), xs.end());
    acc += reduce<int>(xs, [](int a, int b) { return a + b; }, 0);
    try {
        acc += si.get("k0");
    } catch (const std::exception& e) {
        acc -= (int)std::string(e.what()).size();
    }
    acc += (int)si.dump().size() + (int)is.dump().size();
    return acc;
}
"""


def write_cpp_sources(out, n):
    """`n` template-heavy units sharing one header, plus a driver."""
    with open(os.path.join(out, "common.h"), "w") as f:
        f.write(CPP_HEADER)
    for i in range(n):
        with open(os.path.join(out, "u%d.cpp" % i), "w") as f:
            f.write(CPP_UNIT % {"i": i, "j": (i * 7) % 97})
    with open(os.path.join(out, "main.cpp"), "w") as f:
        f.write("#include <cstdio>\n")
        for i in range(n):
            f.write("int work%d(int);\n" % i)
        f.write("int main() { long s = 0;\n")
        for i in range(n):
            f.write("  s += work%d(%d);\n" % (i, i))
        f.write('  std::printf("%ld\\n", s);\n  return 0;\n}\n')


def build_cpp(args, tc, pic):
    """Build the C++ corpus; `pic` selects the -fPIC shared-object variant."""
    out = PIC_DIR if pic else CPP_DIR
    reset(out)
    n = args.tus
    write_cpp_sources(out, n)
    flags = args.cxxflags.split() if args.cxxflags else CPP_FLAGS
    if pic:
        flags = flags + ["-fPIC"]
        # A shared object cannot carry the driver's main.
        os.remove(os.path.join(out, "main.cpp"))
    srcs = sorted(glob.glob(os.path.join(out, "*.cpp")))
    compile_all([[tc.cxx, "-c", s, "-o", s[:-4] + ".o", *flags] for s in srcs],
                args.jobs)
    report("pic" if pic else "cpp", out)


# --- LLVM corpus ------------------------------------------------------------


def llvm_libdir(explicit):
    """Directory holding the installed LLVM static archives, or None."""
    if explicit:
        return explicit
    for cfg in ("llvm-config", "llvm-config-21", "llvm-config-20", "llvm-config-19"):
        exe = toolchain.have(cfg)
        if not exe:
            continue
        p = subprocess.run([exe, "--libdir"], capture_output=True, text=True)
        if p.returncode == 0 and os.path.isdir(p.stdout.strip()):
            return p.stdout.strip()
    return None


def build_llvm(args, tc):
    """Extract members of the installed libLLVM*.a archives into the corpus."""
    libdir = llvm_libdir(args.libdir)
    if not libdir:
        print("llvm: no llvm-config found and no --libdir given, skipping")
        return
    ar = toolchain.have("ar")
    if not ar:
        print("llvm: 'ar' not found, skipping")
        return
    archives = sorted(glob.glob(os.path.join(libdir, "libLLVM*.a")))
    if not archives:
        print("llvm: no libLLVM*.a in %s, skipping" % libdir)
        return
    if not args.all_archives:
        archives = [a for a in archives
                    if not any(s in os.path.basename(a) for s in LLVM_SKIP)]
    if args.max_archives:
        archives = archives[:args.max_archives]
    reset(LLVM_DIR)
    for a in archives:
        d = os.path.join(LLVM_DIR, os.path.basename(a)[:-2])
        os.makedirs(d, exist_ok=True)
        p = subprocess.run([ar, "x", a], cwd=d, capture_output=True, text=True)
        if p.returncode != 0:
            raise ToolchainError("ar x %s failed:\n%s" % (a, p.stderr))
    print("llvm: extracted %d archives from %s" % (len(archives), libdir))
    report("llvm", LLVM_DIR, recursive=True)


# --- driver -----------------------------------------------------------------


def report(name, path, recursive=False):
    pat = os.path.join(path, "*", "*.o") if recursive else os.path.join(path, "*.o")
    objs = glob.glob(pat)
    total = sum(os.path.getsize(o) for o in objs)
    print("%s: %d objects, %.1f MB in %s" % (name, len(objs), total / 1e6, path))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", default=["all"],
                    choices=["all", "c", "cpp", "pic", "llvm"],
                    help="corpora to build (default: all)")
    ap.add_argument("--files", type=int, default=120,
                    help="C corpus: translation units (default 120)")
    ap.add_argument("--globals", type=int, default=1800,
                    help="C corpus: global symbols per unit (default 1800)")
    ap.add_argument("--tus", type=int, default=180,
                    help="C++ corpus: translation units (default 180)")
    ap.add_argument("--cflags", default="", help="override the C compile flags")
    ap.add_argument("--cxxflags", default="", help="override the C++ compile flags")
    ap.add_argument("--libdir", default="",
                    help="LLVM corpus: directory holding libLLVM*.a")
    ap.add_argument("--max-archives", type=int, default=0,
                    help="LLVM corpus: cap the number of archives extracted")
    ap.add_argument("--all-archives", action="store_true",
                    help="LLVM corpus: keep the archives that need gtest or a "
                         "fuzzer runtime (the link will not resolve)")
    ap.add_argument("--jobs", type=int, default=jobs_default(),
                    help="parallel compile jobs (default: one per CPU)")
    args = ap.parse_args()

    which = set(args.which)
    if "all" in which:
        which = {"c", "cpp", "pic", "llvm"}
    tc = Toolchain()
    os.makedirs(CORPUS_DIR, exist_ok=True)
    if "c" in which:
        build_c(args, tc)
    if "cpp" in which:
        build_cpp(args, tc, pic=False)
    if "pic" in which:
        build_cpp(args, tc, pic=True)
    if "llvm" in which:
        build_llvm(args, tc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
