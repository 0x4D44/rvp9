# Full-tree diagonal MV-rate bound: exact, not faster overall

The [CPU6 census](../2026-09-26-subpixel-diagonal-rate-bound-census/README.md)
found 297,344 Ducks diagonal probes whose nonnegative MV rate alone could
not beat the current winner. Two exact candidates were screened against the
integrated NSTEP production PGO binary on x86-64 Crucible. Neither yielded a
repeatable equal-clip process-CPU gain, so **neither code commit was integrated**.

| Candidate | Local commits | Ducks | Tree | Town | Equal-clip |
| --- | --- | ---: | ---: | ---: | ---: |
| Rate bound, duplicate rate calculation on misses | `3e05b7e8` | 1.060946× | 1.022452× | 1.003672× | 1.028749× |
| Reuse the calculated rate on misses | `3e05b7e8` + `9b895180` | 1.031287× | 0.996904× | 0.979038× | 1.002176× |

Ratios are candidate/control process CPU; below 1 is faster. Each screen used
one warm-up per arm and five alternating pairs per 60-frame clip. Every encode
matched its canonical IVF SHA-256. The `plain/` and `reuse/` directories hold
the raw rows, summaries, identities, PGO manifests, and exact wrapper scripts.
The 28-row PGO training plan and source were identical apart from the candidate
code. Base integrated commit was `684185fe`; docs-only commits since then did
not alter the encoder. Control EXE SHA-256 was
`3caee3f4416087ffe13069b4280f48fe92435bc32c9c0a41bebb7c187a73b3ee`;
candidate EXEs were
`f9e8ea11a15e3d1f3fc717cf2c754b1f01ffb67211c7d715c7731fd797d3cafe`
and `a8031c4fb1a341876f4ee2540e0433fc908bd0dca60b65b0e652e5f1b53ce608`.

The focused subpixel suite passed 21/21 on AArch64 and x86, and the encoder
check passed on both. The new tests compare the bounded full-tree winner with
the independent exact diagnostic across shapes, edges and rate weights. A
wrong-winner mutation made that comparison fail; changing `>=` to `>` made
the strict-tie test fail. For the reuse variant, replacing the reused rate
with zero also made the test fail. Both candidates kept the separate
diagnostic trace exact. `plain.patch.gz` and `reuse.patch.gz` archive the two
local code commits in order.

The bound saves real interpolation calls, but an early MV-rate calculation and
its extra branch/layout costs outweigh them on Ducks. Revisit only with a
different way to obtain that lower bound, and re-screen on production PGO.
