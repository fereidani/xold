"""Toolchain, layout and linker discovery for the xold benchmarks.

Nothing here is tied to a distribution, an architecture or an absolute path.
The C runtime pieces are queried through the compiler driver, the dynamic
loader is read out of a probe executable's PT_INTERP, and every directory is
derived from the location of this file. Every setting can be overridden from
the environment, so an unusual toolchain can still be benchmarked.
"""

import os
import shutil
import struct
import subprocess
import sys

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BENCH_DIR)
CORPUS_DIR = os.environ.get("BENCH_CORPUS", os.path.join(BENCH_DIR, "corpus"))
BUILD_DIR = os.environ.get("BENCH_BUILD", os.path.join(BENCH_DIR, "build"))

CC = os.environ.get("CC", "cc")
CXX = os.environ.get("CXX", "c++")

# Linkers the comparison knows how to drive, and the executable to look for.
LINKER_EXE = {
    "xold": None,  # resolved separately, see xold_path()
    "ld": "ld",
    "mold": "mold",
    "wild": "wild",
    "lld": "ld.lld",
    "gold": "ld.gold",
}


class ToolchainError(SystemExit):
    """Raised with an actionable message when the host lacks a prerequisite."""


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def file_name(name):
    """Absolute path of a toolchain file, or None when it is not installed.

    The compiler driver echoes the name back unchanged when it cannot find it.
    """
    try:
        p = _run([CC, "-print-file-name=" + name])
    except OSError:
        return None
    if p.returncode != 0:
        return None
    path = p.stdout.strip()
    if not path or path == name or not os.path.exists(path):
        return None
    return os.path.abspath(path)


def have(cmd):
    """Path of an executable on PATH, or None."""
    return shutil.which(cmd)


def compile_probe():
    """Build a minimal dynamic executable and return its path."""
    os.makedirs(BUILD_DIR, exist_ok=True)
    src = os.path.join(BUILD_DIR, "probe.c")
    exe = os.path.join(BUILD_DIR, "probe")
    if not os.path.exists(exe):
        with open(src, "w") as f:
            f.write("int main(void) { return 0; }\n")
        p = _run([CC, src, "-o", exe])
        if p.returncode != 0:
            raise ToolchainError("cannot compile with %s:\n%s" % (CC, p.stderr))
    return exe


def read_interp(path):
    """Return the PT_INTERP string of an ELF file, or None if it has none."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 64 or data[:4] != b"\x7fELF":
        return None
    wide = data[4] == 2
    end = "<" if data[5] == 1 else ">"
    if wide:
        phoff = struct.unpack_from(end + "Q", data, 32)[0]
        phentsize = struct.unpack_from(end + "H", data, 54)[0]
        phnum = struct.unpack_from(end + "H", data, 56)[0]
        off_field, size_field = 8, 32
    else:
        phoff = struct.unpack_from(end + "I", data, 28)[0]
        phentsize = struct.unpack_from(end + "H", data, 42)[0]
        phnum = struct.unpack_from(end + "H", data, 44)[0]
        off_field, size_field = 4, 16
    word = end + ("Q" if wide else "I")
    for i in range(phnum):
        base = phoff + i * phentsize
        if base + phentsize > len(data):
            break
        if struct.unpack_from(end + "I", data, base)[0] != 3:  # PT_INTERP
            continue
        off = struct.unpack_from(word, data, base + off_field)[0]
        size = struct.unpack_from(word, data, base + size_field)[0]
        return data[off:off + size].split(b"\0")[0].decode()
    return None


def xold_path():
    """The linker under test: $XOLD, then this checkout, then PATH."""
    override = os.environ.get("XOLD")
    if override:
        return override
    built = os.path.join(REPO_DIR, "target", "release", "xold")
    if os.access(built, os.X_OK):
        return built
    return have("xold") or built


class Toolchain:
    """Everything a link command needs, discovered once per run."""

    def __init__(self):
        self.cc = CC
        self.cxx = CXX
        self.xold = xold_path()
        # Scrt1.o is the PIE variant some toolchains ship instead of crt1.o.
        self.crt1 = file_name("crt1.o") or file_name("Scrt1.o")
        self.crti = file_name("crti.o")
        self.crtn = file_name("crtn.o")
        self.crtbegin = file_name("crtbegin.o")
        self.crtend = file_name("crtend.o")
        self.crtbegin_static = file_name("crtbeginT.o") or self.crtbegin
        self.libc_a = file_name("libc.a")
        self.libgcc_a = file_name("libgcc.a")
        self.libgcc_eh_a = file_name("libgcc_eh.a")
        self.gcc_dir = os.path.dirname(self.crtbegin) if self.crtbegin else None
        self.libc_dir = os.path.dirname(self.crt1) if self.crt1 else None
        self.interp = os.environ.get("BENCH_INTERP") or self._find_interp()

    def _find_interp(self):
        try:
            return read_interp(compile_probe())
        except (OSError, ToolchainError):
            return None

    def require_xold(self):
        """Abort unless the linker under test is executable."""
        if not os.access(self.xold, os.X_OK):
            raise ToolchainError(
                "xold not found at %s; run 'cargo build --release' or set "
                "XOLD=/path/to/xold" % self.xold)
        return self.xold

    def missing(self, *names):
        """Names of the requested settings that discovery did not find."""
        return [n for n in names if not getattr(self, n)]

    def linkers(self):
        """Map of linker name to argv prefix, for the ones installed here."""
        found = {}
        for name, exe in LINKER_EXE.items():
            if name == "xold":
                if os.access(self.xold, os.X_OK):
                    found[name] = [self.xold]
            elif have(exe):
                found[name] = [exe]
        return found

    def describe(self):
        """One line per discovered setting, for --show-toolchain."""
        keys = ("cc", "cxx", "xold", "interp", "crt1", "crti", "crtn",
                "crtbegin", "crtend", "crtbegin_static", "libc_a", "libgcc_a",
                "libgcc_eh_a", "gcc_dir", "libc_dir")
        out = ["bench_dir=%s" % BENCH_DIR,
               "corpus_dir=%s" % CORPUS_DIR,
               "build_dir=%s" % BUILD_DIR]
        out += ["%s=%s" % (k, getattr(self, k)) for k in keys]
        out.append("linkers=%s" % ",".join(sorted(self.linkers())))
        return "\n".join(out)


def compile_object(src, obj=None, flags=()):
    """Compile one benchmark source into $BUILD_DIR, reusing a fresh object."""
    os.makedirs(BUILD_DIR, exist_ok=True)
    if obj is None:
        obj = os.path.join(BUILD_DIR, os.path.splitext(os.path.basename(src))[0] + ".o")
    if os.path.exists(obj) and os.path.getmtime(obj) >= os.path.getmtime(src):
        return obj
    p = _run([CC, "-c", src, "-o", obj, *flags])
    if p.returncode != 0:
        raise ToolchainError("failed to compile %s:\n%s" % (src, p.stderr))
    return obj


if __name__ == "__main__":
    sys.stdout.write(Toolchain().describe() + "\n")
