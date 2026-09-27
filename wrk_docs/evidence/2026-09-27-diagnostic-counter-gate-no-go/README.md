# Production diagnostic-counter gate: exact, no CPU gain

Do not integrate the code prototype on this evidence. Normal release builds no longer update proof-only atomics or copy/count trellis coefficients; debug tests and opt-in profiling/statistics retain the counters. The prototype is committed as `5040d545` on `task/20260927-TSK-HUM-compile-diagnostic-counters-out-of-produ`, but every equal-clip paired CPU screen was flat or slower than the integrated encoder.

## Why this was tested

In the exact integrated PGO EXE (SHA-256 `3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad`), RVA `0xf8aec` is a `lock incq` on the `PHASE2_NEWMV_REUSED` proof counter in `crates/rvp9-encoder/src/lib.rs`. The next IP, `0xf8af0`, has 142/44/42 direct samples in Ducks/Tree/Town (0.783%/0.537%/0.487%). This is an attribution lead, not a removable-CPU estimate: sampled IPs can skid. A whole-EXE lock census attributed nearly all lock-adjacent samples to this counter. The counter is read by a positive-control integration test and the opt-in runtime-profile report, not by encoding decisions.

The prototype gates the other proof-only writes in `crates/rvp9-encoder/src/lib.rs`, `rdo/inter_dispatch.rs`, and `bitstream_gates.rs`, plus the allocation-only proof counter in `crates/rvp9-core/src/frame.rs`. It leaves row-thread/queue synchronization and public counter statics untouched. `RVP9_P8_STATS` and `RVP9_C2B_STATS` continue to collect when explicitly requested. The C2B environment choice is cached on first use, matching normal CLI startup-env use; a later in-process environment change would not be observed.

## Correctness and build

On AArch64, default and release `cargo check -p rvp9-encoder -p rvp9-core`, `cargo fmt --all --check`, and `git diff --check` passed. The existing coefficient-probability positive control failed when the compile-time gate was deliberately forced off (`tables_updated=0`, `tables_suppressed_with_diff=0`), then passed after restoration. The partition proof and eight core scratch-pool tests passed. The NEWMV and ML proofs compiled but were ignored locally because their Y4M corpus was absent. No new self-referential gate test was kept.

On Crucible, the exact four modified source files were copied into an isolated worktree at `D:\worktrees\rvp9-2\20260927-TSK-HUM-test-release-diagnostic-counter-gating-o`. A frozen-profile PGO release build passed, with expected control-flow hash-mismatch warnings where functions changed. A second release binary used the repository's complete 28-row, 60-frame production PGO training matrix and built without mismatch warnings. All three paired screens matched the canonical IVF SHA-256 on every Ducks, Tree, and Town encode, including warmups (228 encodes total across both arms). The integrated control is the same exact EXE as the [60-pair direct libvpx comparison](../2026-09-27-integrated-pgo-direct-repeat/README.md).

| Candidate/control process CPU, geometric mean | Ducks | Tree | Town | Equal-clip |
| --- | ---: | ---: | ---: | ---: |
| Frozen profile, 5 pairs/clip | 1.000028 | 1.002910 | 1.003194 | **1.002043** |
| Frozen profile, 15 pairs/clip | 0.991662 | 1.007437 | 1.003867 | **1.000966** |
| Fresh candidate profile, 15 pairs/clip | 1.031743 | 0.998193 | 1.011026 | **1.013560** |

The frozen-profile candidate EXE SHA-256 is `2f7ae59b4416cf9e9d52fd8516865540a0ad4d376664f76801ee984f8f5151bf`. Its `encoder.profdata` was the integrated profile SHA-256 `3f02af95fbeca067d6d94b233a353681bce4ed25c76930a94f0564895ac10335`. The freshly trained candidate EXE SHA-256 is `74168ccdde659a08120de5f1ac422853c2999d63ced9faab337f21990ddb70d7`; its training manifest remains in the Crucible task worktree. The first two runs isolate the source change under a common frozen profile; the third checks whether retraining reverses the result. It did not. The different candidate profile means the third result also includes PGO variation, so it does not isolate the counters' local cost.

The raw per-encode CPU and hashes, identities, per-clip ratios, and exact benchmark wrapper are retained here. The wrapper was copied to and run from the remote task worktree's `.scratch`; its relative source path assumes that location. Bulky IVFs and PGO profiles remain in that same remote scratch. The integrated RVP9/libvpx equal-clip ratio remains **1.057638×**, above the **1.05×** goal; there is no new integrated encoder performance claim.
