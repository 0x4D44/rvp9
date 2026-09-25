# Unit 1425 — controlled old/current x86 build identity

Unit 1423 found that an integrated encoder was 12–14% slower than a retained
pre-Unit-1421 encoder in two paired screens, despite byte-identical output.
This unit rebuilt both sources on Crucible to test whether the old executable
carried different generated code or build flags. No codec source changed.

The clean old source was `5b6822c05c8d7ff3005902715d618f0ecbfbfd2c`;
the clean current source was `8759316a93e467c036d19a23e379fcc692f9f829`.
The only shipped-source difference is the FDCT16 AVX2 first pass in
`crates/rvp9-core/src/transforms/forward_sse2.rs`. Both new builds used
Rust 1.95.0, native x86 target, release opt-level 3, fat LTO, one codegen
unit, panic abort, debug info level 2, and no PGO. `build-flags.txt` records
their paths, hashes, effective rustc flags, and PE/PDB identities.

The retained-old and fresh-old executables have different whole-file hashes
but **identical `.text` bytes**: both contain a 1,612,800-byte raw `.text`
section with SHA-256
`330eb35734a45985a9ca24acc2dce7388fdb56abdbce7d6f3f107141603c7753`.
They also have different PDB GUIDs. The old-versus-fresh hash mismatch is
therefore not evidence of different generated instructions. The fresh-current
`.text` differs, as expected from the source change, but the previously
measured current executable was removed with its worktree and cannot be
compared byte-for-byte with this rebuild.

The two complete harness screens used the same three 60-frame 1080p clips,
CPU6, CBR 1260 kb/s, one thread, one warmup, three paired iterations, two
reversed rotations, and no load ceiling. All old, fresh-old, and fresh-current
IVF hashes matched for each clip and were stable within each arm. The manifests,
raw rows, and summaries are preserved beside this file.

| Screen | Test/anchor paired corpus | Why it cannot settle speed |
|---|---:|---|
| Retained-old / fresh-old | 0.843277× | Clip ratios span 0.551–1.114×; load reached 31.38/32 busy logical cores and individual encodes took up to 32 s. |
| Fresh-current / fresh-old | 1.008315× | Clip ratios span 0.680–1.496×; load reached 30.21/32 cores and individual encodes took up to 26 s. |

Crucible has two eight-core processor dies. A separate 60-encode spot check
used `CREATE_SUSPENDED` to set and verify process affinity before either
encoder executed its first instruction. It used one SMT sibling from each
core on die 0 (`0x5555`) or die 1 (`0x55550000`), with four alternating
old/current pairs per clip and die plus one warmup per arm. All 60 processes
exited successfully with their requested affinity; all 24 measured pairs
had byte-identical IVF output. Independent inspection of `affinity-check/raw.csv`
found one stable hash per clip and arm. Pre/post host load was 3.05/3.55
busy logical cores of 32.

| Die | Ducks current/old | In-to-tree current/old | Old-town current/old |
|---|---:|---:|---:|
| 0 | 1.423× | 0.408× | 1.066× |
| 1 | 0.908× | 1.136× | 0.680× |

These are paired geometric ratios, not gains or regressions. The raw rows
show isolated 5.8–9.1 s old runs amid roughly 1.1–1.2 s In-to-tree runs,
a 7.0 s current Ducks run amid roughly 2.3–3.5 s runs, and a 4.2 s old
Old-town run amid roughly 1.1–1.4 s runs. Thus same-die pinning did not
remove short stalls, and four pairs per cell cannot support a stable speed
estimate. As a descriptive check only, the fastest observed current/old
ratio in each clip/die cell ranges from 0.960× to 1.072×; it does not show
a uniform 12–14% slowdown. The exact runner settings, raw rows, load samples,
and its own assessment are in `affinity-check/`.

**Status:** These screens neither reproduce nor refute Unit 1423's historical
12–14% slowdown with enough precision to attribute it. The retained-old
`.text` is identical to a controlled fresh rebuild, so its whole-file hash
is not evidence of special generated code. A uniform FDCT16 regression is
not established. Short scheduling stalls and die placement are measurement
confounders; the pinned check removes only the latter. Do not roll back the
FDCT16 change or claim a new libvpx parity number from this evidence. A
subsequent optimization screen needs load observations short enough to catch
these stalls and enough pairs to test a small gain on each clip.
