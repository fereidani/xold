# xold benchmarks

Timing harness for xold against the other linkers installed on the machine.
It has no external dependencies: a C/C++ compiler and Python 3 are enough, and
every path it needs is discovered at run time, so the same commands work on any
distribution and any supported architecture.

## Quick start

```sh
cargo build --release          # from the repository root
cd bench
python3 corpus.py              # generate the link inputs (a few minutes)
python3 bench.py               # time every scenario on every linker found
```

`python3 bench.py --list` prints the discovered toolchain, which linkers were
found, and the state of each scenario.

## Files

| File | What it is |
| --- | --- |
| `toolchain.py` | Discovery: compiler, C runtime objects, dynamic loader, xold, installed linkers |
| `corpus.py` | Generates the benchmark inputs into `corpus/` |
| `scenarios.py` | The link scenarios and the per-linker flags that keep them comparable |
| `bench.py` | The runner: comparison, thread sweep, A/B, peak memory |
| `hello.c`, `llmain.c` | Tiny drivers used by the small and LLVM scenarios |

Generated data lives in `corpus/` (inputs) and `build/` (link outputs and
probes). Both are ignored by git and can be deleted at any time.

## Scenarios

| Name | Link |
| --- | --- |
| `c-static` | Generated C corpus, static executable against `libc.a` |
| `c-dyn` | Generated C corpus, dynamic executable against `libc.so` |
| `cpp-dyn` | C++ corpus with templates, COMDAT groups, eh_frame and DWARF |
| `shared` | Same C++ corpus built `-fPIC`, linked as a shared object |
| `llvm` | Objects extracted from the LLVM static archives installed here |
| `hello-static`, `hello-dyn` | One small object: archive resolution and the latency floor |

A scenario whose inputs are missing, or whose toolchain pieces are not
installed, is reported as skipped with the reason rather than failing the run.
The `llvm` scenario needs `llvm-config` (or `--libdir`) and `ar`; skip it with
`python3 corpus.py c cpp pic`.

## Running

```sh
python3 bench.py --only c-static --runs 9        # one scenario, more samples
python3 bench.py --linkers xold,mold             # only these linkers
python3 bench.py --linkers xold --threads 1,2,4,8   # thread scaling
python3 bench.py --ab ./xold-before ./xold-after    # interleaved A/B of two builds
python3 bench.py --rss --only cpp-dyn            # peak memory and CPU of one link
```

Each linker is warmed up once before the timed runs; the warm-up doubles as the
correctness gate, and scenarios marked runnable also execute the produced
binary. Output size and a SHA-256 prefix are printed so a change in link output
is visible next to the timing.

Every linker receives the same object list and semantically equivalent flags:
section garbage collection off everywhere, `.eh_frame_hdr` everywhere, the same
image kind, and the same thread count when one is requested. xold never
garbage-collects and always writes `.eh_frame_hdr`, so the others are told to
match rather than being allowed to do less work.

## Corpus options

```sh
python3 corpus.py c --files 200 --globals 3000   # bigger C corpus
python3 corpus.py cpp pic --tus 300              # bigger C++ corpus
python3 corpus.py llvm --libdir /usr/lib/llvm-18/lib --max-archives 50
python3 corpus.py c --cflags "-O2 -ffunction-sections"
```

## Environment overrides

| Variable | Meaning |
| --- | --- |
| `XOLD` | The linker under test (default: `target/release/xold` in this checkout, then `PATH`) |
| `CC`, `CXX` | Compilers used to build the corpora and to locate the C runtime |
| `BENCH_CORPUS`, `BENCH_BUILD` | Where inputs and outputs are written |
| `BENCH_INTERP` | Dynamic loader path, when it cannot be read from a probe binary |

## Comparing fairly

Timings vary with CPU frequency scaling and page cache state. For a
publishable number, pin the machine to a fixed governor, close other load, use
`--runs 9` or more and compare medians. When comparing two xold builds, prefer
`--ab`: it interleaves the two so any drift during the run hits both equally.
