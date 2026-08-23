#!/usr/bin/env python3
"""Time xold, and the other linkers installed here, on identical links.

Examples:
    python3 bench.py                          # every scenario, every linker
    python3 bench.py --only c-static --runs 9
    python3 bench.py --linkers xold --threads 1,2,4,8
    python3 bench.py --ab ./old-xold ./new-xold --only llvm
    python3 bench.py --rss --only cpp-dyn
    python3 bench.py --list

Generate the inputs first with `python3 corpus.py`. Nothing here depends on a
particular distribution or path: see toolchain.py for what is discovered and
which environment variables override it.
"""

import argparse
import hashlib
import os
import statistics
import subprocess
import sys
import time

import scenarios
import toolchain
from toolchain import BUILD_DIR, Toolchain

# Flags that cap a linker's worker threads, so --threads compares like for
# like. xold takes the count through the environment instead.
THREAD_FLAG = {
    "mold": "--thread-count=%d",
    "wild": "--thread-count=%d",
    "lld": "--threads=%d",
    "gold": "--threads=%d",
}

# Linkers that fork a resident process by default; the fork hides the real
# peak memory and CPU from wait4(), so --rss turns it off.
NO_FORK = ("mold", "wild", "xold")


def thread_setup(linker, n):
    """Extra flags and environment that pin `linker` to `n` threads."""
    if n is None:
        return [], {}
    if linker == "xold":
        return [], {"RAYON_NUM_THREADS": str(n)}
    flag = THREAD_FLAG.get(linker)
    return ([flag % n] if flag else []), {}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_once(cmd, env):
    e = dict(os.environ)
    e.update(env)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, env=e)
    return time.perf_counter() - t0, p


def link_output(sc, tag):
    os.makedirs(BUILD_DIR, exist_ok=True)
    return os.path.join(BUILD_DIR, "%s_%s.out" % (sc.name, tag))


def measure(argv0, linker, sc, runs, threads=None, extra=(), tag=None):
    """Link `runs` times after a warm-up, returning timings and the result."""
    out = link_output(sc, tag or linker)
    tf, env = thread_setup(linker, threads)
    cmd = sc.command(argv0, linker, out, list(extra) + tf)
    # The warm-up doubles as the correctness gate: a linker that cannot build
    # this scenario is reported, not timed.
    _, p = run_once(cmd, env)
    if p.returncode != 0:
        err = (p.stderr or p.stdout).decode(errors="replace").strip()
        return {"linker": tag or linker, "ok": False,
                "err": err.splitlines()[:2] or ["exit %d" % p.returncode]}
    times = []
    for _ in range(runs):
        dt, p = run_once(cmd, env)
        if p.returncode != 0:
            return {"linker": tag or linker, "ok": False,
                    "err": ["failed on a repeat run"]}
        times.append(dt)
    times.sort()
    res = {"linker": tag or linker, "ok": True, "times": times,
           "median": statistics.median(times), "min": times[0],
           "max": times[-1], "size": os.path.getsize(out),
           "sha": sha256(out)[:16], "out": out}
    if sc.run:
        r = subprocess.run([out], capture_output=True)
        res["rc"] = r.returncode
        res["stdout"] = r.stdout.decode(errors="replace").strip()[:40]
    return res


def measure_ab(binaries, sc, runs):
    """Interleave two or more xold builds so drift hits both equally."""
    tags = ["A", "B", "C", "D"][:len(binaries)]
    cmds = {t: sc.command([b], "xold", link_output(sc, t))
            for t, b in zip(tags, binaries)}
    for cmd in cmds.values():
        _, p = run_once(cmd, {})
        if p.returncode != 0:
            err = (p.stderr or p.stdout).decode(errors="replace").strip()
            return [{"linker": "?", "ok": False, "err": err.splitlines()[:2]}]
    times = {t: [] for t in tags}
    for _ in range(runs):
        for t in tags:
            dt, p = run_once(cmds[t], {})
            if p.returncode != 0:
                return [{"linker": t, "ok": False, "err": ["failed mid-run"]}]
            times[t].append(dt)
    res = []
    for t, b in zip(tags, binaries):
        ts = sorted(times[t])
        out = link_output(sc, t)
        res.append({"linker": "%s %s" % (t, os.path.basename(b)), "ok": True,
                    "times": ts, "median": statistics.median(ts), "min": ts[0],
                    "max": ts[-1], "size": os.path.getsize(out),
                    "sha": sha256(out)[:16], "out": out})
    return res


def peak_usage(argv0, linker, sc):
    """Peak RSS in MB and CPU seconds of one link, or None if it failed.

    wait4() is used rather than getrusage(RUSAGE_CHILDREN) because the latter
    reports a high-water mark across every child this process has ever reaped,
    not the one link being measured. Linkers that fork a resident process are
    told not to, or the numbers would describe the wrong process.
    """
    extra = ["--no-fork"] if linker in NO_FORK else []
    cmd = sc.command(argv0, linker, link_output(sc, linker + "_rss"), extra)
    exe = cmd[0] if os.path.isabs(cmd[0]) else toolchain.have(cmd[0])
    if not exe:
        return None
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        pid = os.posix_spawn(exe, [exe] + cmd[1:], os.environ, file_actions=[
            (os.POSIX_SPAWN_DUP2, devnull, 1), (os.POSIX_SPAWN_DUP2, devnull, 2)])
        _, status, ru = os.wait4(pid, 0)
    finally:
        os.close(devnull)
    if status != 0:
        return None
    return ru.ru_maxrss / 1024.0, ru.ru_utime + ru.ru_stime


def fmt(res, baseline=None):
    if not res["ok"]:
        return "%-14s FAILED: %s" % (res["linker"], "; ".join(res["err"]))
    line = "%-14s median %8.3fs  min %8.3fs  size %10d  sha %s" % (
        res["linker"], res["median"], res["min"], res["size"], res["sha"])
    if baseline:
        line += "  %5.2fx" % (res["median"] / baseline)
    if "rc" in res:
        line += "  run rc=%d out=%r" % (res["rc"], res["stdout"])
    return line


def list_scenarios(scs, tc):
    print(tc.describe())
    print()
    for sc in scs:
        state = "unavailable: %s" % sc.unavailable if sc.unavailable else \
            "%d objects, %.1f MB" % (len(sc.objs), sc.input_bytes() / 1e6)
        print("%-13s %s\n              %s" % (sc.name, sc.desc, state))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=5,
                    help="timed runs per linker, after one warm-up (default 5)")
    ap.add_argument("--only", default="",
                    help="comma separated scenario names (default: all)")
    ap.add_argument("--linkers", default="",
                    help="comma separated linkers (default: every one found)")
    ap.add_argument("--threads", default="",
                    help="comma separated thread counts to sweep")
    ap.add_argument("--ab", nargs="+", metavar="XOLD",
                    help="interleave these xold builds instead of comparing "
                         "linkers")
    ap.add_argument("--rss", action="store_true",
                    help="also report peak memory and CPU of a single link")
    ap.add_argument("--list", action="store_true",
                    help="show the discovered toolchain and scenarios, then exit")
    a = ap.parse_args()

    tc = Toolchain()
    scs = scenarios.select([s for s in a.only.split(",") if s], tc)
    if a.list:
        list_scenarios(scs, tc)
        return 0

    available = tc.linkers()
    if a.ab:
        chosen = {}
    else:
        names = [n for n in a.linkers.split(",") if n] or list(available)
        if "xold" in names:
            tc.require_xold()
        missing = [n for n in names if n not in available]
        if missing:
            print("not installed, skipped: %s" % ", ".join(missing))
        chosen = {n: available[n] for n in names if n in available}
        if not chosen:
            raise SystemExit("no linker to run; build xold or name installed ones")

    threads = [int(x) for x in a.threads.split(",") if x] or [None]
    for sc in scs:
        if sc.unavailable:
            print("== %s: skipped (%s)\n" % (sc.name, sc.unavailable))
            continue
        print("== %s: %s" % (sc.name, sc.desc))
        print("   inputs: %d objects, %.1f MB"
              % (len(sc.objs), sc.input_bytes() / 1e6))
        if a.ab:
            for res in measure_ab(a.ab, sc, a.runs):
                print("   " + fmt(res))
            print()
            continue
        baseline = None
        for name, argv0 in chosen.items():
            for n in threads:
                res = measure(argv0, name, sc, a.runs, threads=n)
                if res["ok"] and baseline is None and n == threads[0]:
                    baseline = res["median"]
                prefix = "   " if n is None else "   T=%-3d" % n
                print(prefix + fmt(res, baseline if n is None else None))
                sys.stdout.flush()
        if a.rss:
            for name, argv0 in chosen.items():
                usage = peak_usage(argv0, name, sc)
                if usage:
                    print("   rss %-8s peak %7.1f MB  cpu %6.2fs"
                          % (name, usage[0], usage[1]))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
