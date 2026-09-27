# Vertical-16 AVX2 shared sums: exact, no repeatable whole-encoder gain

Do not integrate the performance prototype. It replaces 14 independent 15-tap weighted sums in `crates/rvp9-core/src/dsp/x86_64/lpf_vertical_16_avx2.rs` with a rolling recurrence. The candidate is committed locally at `02a255ec`; the separate AVX2 mask correction is `bf4edd2f`. Neither code commit is on `origin/main`.

The new `wide15_shared_sums_match_scalar_weights` test compares all 14 outputs across eight lanes against scalar weights, then compares complete filters over several pitches. A deliberately corrupted recurrence failed its own assertion on Crucible (`case=0 output=10 lane=0`, 154 versus 167). The original full-filter test also exposed a pre-existing saturated-u8 mask defect at `blimit=255`; it failed at `case=4 pitch=16` before the correction and passed after. That separate defect is tracked as `RVP9-BUG-FLU-00034`. The x86 AVX2 module passed 9 tests and the AArch64 loop-filter suite passed 129 tests on the final local source.

All four paired screens used the same pinned integrated production-PGO control EXE (SHA-256 `3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad`) and the same integrated `encoder.profdata` (SHA-256 `3f02af95fbeca067d6d94b233a353681bce4ed25c76930a94f0564895ac10335`). Each screen alternated arms for 15 measured pairs per Ducks, Tree, and Town, plus warmups. All 384 output IVFs matched the canonical per-clip hashes. Ratios below are candidate/control process CPU, so lower is better.

| Candidate | Ducks | Tree | Town | Equal-clip geometric mean |
| --- | ---: | ---: | ---: | ---: |
| Rolling sums + always-wide mask, quiet lease | 1.010031 | 0.998955 | 0.999935 | 1.002961 |
| Rolling sums + narrow fast mask, relaxed load | 1.001714 | 0.990094 | 0.990234 | 0.993999 |
| Same narrow-mask EXE, relaxed load with mid-run CPU spike | 0.991359 | 1.008819 | 0.988324 | 0.996127 |
| Same narrow-mask EXE, quiet lease | 1.016018 | 1.009209 | 0.994271 | 1.006458 |

The first candidate EXE SHA-256 was `40ad84e9df48ad064ee5c78e24d1e3d2c8b9327b79274425b0ffb583e590ac07`; the narrow-mask EXE was `88194caab66933ae78b6eefbc6199573eb7f46c170feeddcdd3f1f8dc1da6245`. In the final quiet run, Crucible showed two cargo processes that predated the lease but no rustc process; three one-second whole-machine CPU samples immediately before the run were 9.2%, 9.3%, and 7.9%. The paired result reversed the apparent small gain. The integrated RVP9/libvpx ratio remains 1.057638×; this experiment changes no parity claim.

The checked-in CSVs contain arm identities, raw per-encode CPU and output hashes, and the per-clip summaries. The benchmark driver was `wrk_docs/evidence/2026-09-27-fresh-integrated-pgo-direct/direct-integrated-pgo.ps1`, with only worktree/binary paths and expected binary hashes replaced. Its temporary wrapper and bulky builds remain in the isolated Crucible worktree `D:/worktrees/rvp9-2/20260927-TSK-HUM-vertical16-shared-sums-candidate/.scratch`.
