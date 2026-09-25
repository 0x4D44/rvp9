# Unit 1428 — AVX2 8×8 variance prototype, not retained

The integrated Unit 1426 Crucible profile attributed 2.25% of In-to-tree
and 2.04% of Old-town encoder self samples to
`get_var_sse_8x8_avx2`. This bounded prototype replaced its generic
8×8 loop with four independent two-row AVX2 sum/SSE chains. It routed
both the 8×8 block and per-cell entry points through that kernel. Other
shapes retained the generic path. The exact rejected source change is
`unit1428-avx2-variance8.patch.b64` (base64-encoded; decoded SHA-256
`7e0d1aa1f5756267c861f4e1d458b9ff9f17f409e90b26c88ae9ddfe0ed6f2f4`).

The new test compared all three 8×8 entry points with the scalar oracle
over 1,024 deterministic random cases with independent padded strides.
Crucible's real AVX2 focused suite passed 10/10 tests. A deliberate
`+1` corruption of the new test's expected sum failed on its first case
(actual −585, expected −584; SSE 796729); restoring the assertion passed.
The AArch64 variance suite passed 35/35, the x86 test cross-check passed,
and rustfmt and `git diff --check` passed. No AVX2 execution was attempted
through Rosetta, which reported that feature unavailable.

Crucible built control and candidate with Rust 1.95.0, release
`CARGO_PROFILE_RELEASE_DEBUG=2`, native CPU, and no declared PGO.
The control executable SHA-256 was
`673bf75368c63cc9cb542c54034516fc893a3b114611ecfda34d44a5244bdb1a`;
the candidate was
`ca7a254badcc0c7abc82b6afb445953cf0602da2c7dbbdff7507cd23a93413b6`.
The restored candidate source SHA-256 was
`ebd60616f52c2122b4affe099d61691871715168330d20d8792e111b27a209d3`.

The paired screen used the three hash-pinned 60-frame 1080p clips on
Crucible's C: registered scratch, CPU6, CBR 1260 kb/s, one thread,
one warmup and seven AB/BA pairs per clip in each of two reversed
rotations. There was no load ceiling. Every one of the 96 IVF outputs
matched its clip's integrated-control hash, with only one hash per arm
per clip. The harness sampled 60-second mean host CPU at boundaries,
from 2.25 to 10.81 busy logical cores; it did not normalize timings
by load.

| Clip | Candidate/control paired wall | Separate median ratio |
| --- | ---: | ---: |
| Ducks | 1.081633× | 1.070046× |
| In-to-tree | 1.006767× | 0.997973× |
| Old-town-cross | 1.016538× | 1.024462× |
| Equal-clip corpus | **1.034453×** | **1.030400×** |

The only complete screen shows no gain and a roughly 3.4% paired
corpus regression. The Ducks result is consistently slower in the raw
pairs, but this one screen is not a claim of a stable 3.4% true
regression. It suffices to reject the candidate under the retention
rule. No second screen or pinned-libvpx comparison was run, and no
new production-parity estimate is claimed. The latest provisional
integrated RVP9/libvpx estimate remains Unit 1422's 1.400201×.

`screen-a-manifest.txt`, `screen-a-raw.csv`, and
`screen-a-summary.csv` preserve the binary/input hashes, exact
commands, all per-run wall times and output hashes, and the load
observations. The decoded source patch reverse-checks cleanly against
the tested candidate.
