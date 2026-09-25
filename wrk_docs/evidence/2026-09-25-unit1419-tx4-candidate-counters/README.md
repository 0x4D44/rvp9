# Unit 1419 — exact TX4 candidate counters

This is temporary diagnostic evidence for the current CPU6 emission-side TX4
selector. The source instrumentation is archived in
`unit1419-exact-instrumentation.patch.gz`; it is restored to the integrated source
after this evidence is written. No codec behavior change is retained.

The raw patch SHA-256 is
`59c360ffeddd44eaf8d3cf0e0a60493e70cbff01dccb1a353aa7d2fe59675b14`;
the gzip SHA-256 is
`5ffd667a3177bea5b18da91b6385945409a0f346857d986ca4af520db75d9123`.
`gzip -dc unit1419-exact-instrumentation.patch.gz | git apply --check`
passed on the restored baseline.

## Instrumentation

The diagnostic is compiled only with `runtime-profiling` and enabled only by
`RVP9_DIAG_TX4_COUNTERS=1`. It aggregates per-selector local counts into
process-wide relaxed atomics once per selector call, then prints per-frame
deltas as `[TX4_DIAG]` beside the existing profile records. It records eligible
selector calls, TX4 attempts, the TX8/TX16 incumbent before TX4, evaluated TX4
tiles, TX4 prunes, TX4 wins, and TX4 candidate nanoseconds.

The focused `tx4_diagnostic_counts_one_selector_call` test first ran red with an
intentional expected eligible-call count of `2` (observed `1`), then passed
after restoring the expected count to `1`.

## Build identity

- Remote host: Crucible, fresh detached worktree
  `D:\worktrees\rvp9-2\20260925-TSK-HUM-unit-1419-exact-tx4-candidate-counters`
- Base: `origin/main` `cd57ef3b543a44c35555814fb877a19ea91a9190`
- Baseline `lib.rs` blob: `a3e9e735470a08545068a1777146b1818a793a3d`
- Final instrumented `lib.rs` blob: `9f54f7a65475d61e11a44cbe198a40fa16c50c42`
- Build: `cargo build --locked --release -p rvp9-cli --bin rvp9enc --features rvp9-encoder/runtime-profiling`, with `CARGO_PROFILE_RELEASE_DEBUG=2`
- Feature executable: 1,920,512 bytes, SHA-256 `280b846fdc32123f38e15b20df20196224d352a0c7a7f91ce56c62906a7d3068`
- PDB: 16,158,720 bytes, SHA-256 `23ffa3baf56e0493f034a4de2d6c6e16ca25a44ac8b93a3b8f3da4495d97bdb2`

The local AArch64 host is `aarch64-apple-darwin` on Rust 1.95. The feature
test target compile passed with `cargo test -p rvp9-encoder --features
runtime-profiling --lib --no-run`; the focused feature test passed after the
intentional red run.

## Protocol and exact-output check

Each run used CPU-used 6, CBR 1260 kb/s, 30/1 fps, one thread, tile columns 0,
lag 0, AQ 0, noise 0, default tune, KF distance 60, IVF output, and
`--limit=60`, with `RVP9_EMIT_PROFILE=1 RVP9_RT_PROFILE=1
RVP9_DIAG_TX4_COUNTERS=1`. The input SHA-256 values were the Unit 1418 values:

| Clip | Input SHA-256 | IVF bytes | Diagnostic IVF SHA-256 | Unit 1418 baseline SHA-256 |
|---|---|---:|---|---|
| Ducks | `6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7` | 529,682 | `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089` | identical |
| In To Tree | `ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791` | 337,194 | `5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41` | identical |
| Old Town | `b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909` | 358,110 | `4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f` | identical |

All three logs contain 59 `RT_PROFILE`, 59 `EMIT_PROFILE`, and 59 `TX4_DIAG`
records. The raw TX4 lines are in `raw/tx4-counters.txt`.

## Aggregates and derived values

The timer fraction is summed TX4 candidate time divided by summed existing
`EMIT_PROFILE tx_select`; it is an instrumented diagnostic share, not a speed
claim. Tiles per attempt is evaluated TX4 tiles divided by TX4 attempts.

| Clip | Eligible / attempts | Incumbent TX8 | Incumbent TX16 | TX4 tiles | Prunes | Wins | TX4 time (ns) | Selector time (ms) | TX4 / selector | Tiles / attempt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ducks | 439,604 / 439,604 | 352,225 | 87,379 | 4,285,883 | 438,299 | 1,305 | 141,151,900 | 407.00 | 34.681% | 9.749 |
| In To Tree | 43,817 / 43,817 | 42,552 | 1,265 | 551,380 | 43,550 | 267 | 17,977,600 | 46.83 | 38.389% | 12.584 |
| Old Town | 41,550 / 41,550 | 40,492 | 1,058 | 549,128 | 40,595 | 955 | 17,875,400 | 44.18 | 40.460% | 13.216 |

The exact-output match shows that the diagnostic did not alter selection or
bitstream output. The counters show TX4 is pruned on most attempts, with a low
win count; this is evidence for the next design decision only, not a kernel
implementation approval.
