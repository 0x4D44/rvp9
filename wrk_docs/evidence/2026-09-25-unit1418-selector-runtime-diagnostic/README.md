# Unit 1418 — selector runtime diagnostic

This bounded diagnostic measured the existing `runtime-profiling` feature
around the `lib.rs:11471` selector. It made no source or worktree changes.
The exact Unit 1416 profile directory and its ETL/PDB/executable were kept
untouched.

## Build and identity

The Crucible worktree was
`D:\worktrees\rvp9-2\20260924-TSK-HUM-fresh-x86-motion-search-attribution-unit`.
At the read it was clean at `5b6822c05c8d7ff3005902715d618f0ecbfbfd2c`
(`[behind 6]`). Its encoder source, manifest, and lockfile blobs matched
integrated `cf06dd816260efd246947c33385ffb7001729120`:

| Blob | SHA-1 |
|---|---|
| `crates/rvp9-encoder/src/lib.rs` | `a3e9e735470a08545068a1777146b1818a793a3d` |
| `crates/rvp9-encoder/Cargo.toml` | `72bf2dfa099f1dd8960a3e231dd7379de0f1c804` |
| `Cargo.lock` | `91d6e6b14b7b1a3ae9773a6a6984e6aecd4e3ad8` |

I built into the new remote target
`D:\worktrees\rvp9-2\20260924-TSK-HUM-fresh-x86-motion-search-attribution-unit\.scratch\unit1418-runtime-target`
with `cargo build --locked --release -p rvp9-cli --bin rvp9enc
--features rvp9-encoder/runtime-profiling`, `CARGO_PROFILE_RELEASE_DEBUG=2`,
and `CARGO_TARGET_DIR` set to that target. The build returned zero. The new
feature executable and PDB were:

| Artifact | Size | SHA-256 |
|---|---:|---|
| feature `rvp9enc.exe` | 1,919,488 | `6d695a341f671169372237a416e8104ca634bef8b9306cc1069df7f6a0da1334` |
| feature `rvp9enc.pdb` | 16,146,432 | `ee23f680088d7e4b6f6536da5df3fe00cf96e4d919ba9ad322b3255bf5ce169e` |

The retained Unit 1416 control executable was
`8d175c1bffba875f8381461a7be447335ad8ec77847ef83bb07b965474b29f6e`; its
PDB was `5dca6f9dca6cbc1ecbf924b090231b5fd86a149f590dec3985c3234d41014f2e`,
and its retained ETL was
`e7e4c331f99de00b532905e509f5c04b041baf3dbb70d72535ee544a86e9c0a8`.

## Protocol and output check

The three 1920×1080 60-frame Y4M inputs were run at CPU-used 6, CBR 1260
kb/s, 30/1 fps, one thread, tile columns 0, lag 0, AQ 0, noise 0, default
tune, and KF distance 60. Both binaries used one pass, real-time mode, IVF
output, and `--limit=60`. The control run had both profile variables removed;
the feature run used `RVP9_EMIT_PROFILE=1 RVP9_RT_PROFILE=1`. The raw outputs
and logs remain under
`D:\worktrees\rvp9-2\20260924-TSK-HUM-fresh-x86-motion-search-attribution-unit\.scratch\unit1418-runs`.

| Clip | Input SHA-256 | IVF bytes | Control SHA-256 | Runtime SHA-256 | Profile lines |
|---|---|---:|---|---|---:|
| Ducks | `6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7` | 529,682 | `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089` | `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089` | 59/60 |
| In To Tree | `ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791` | 337,194 | `5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41` | `5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41` | 59/60 |
| Old Town | `b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909` | 358,110 | `4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f` | `4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f` | 59/60 |

Each runtime log has 118 lines: 59 `RT_PROFILE` and 59 `EMIT_PROFILE`
records. The absent record is the first/key frame; these are inter-frame
diagnostics, not 60 frame-wide samples. All runtime values below are sums
over those 59 records:

| Clip | RT total (ms) | `tx_select` (ms) | `tx_select / RT total` | `tx_select / emit` |
|---|---:|---:|---:|---:|
| Ducks | 2,237.04 | 369.75 | 16.529% | 38.714% |
| In To Tree | 1,066.46 | 41.94 | 3.933% | 11.088% |
| Old Town | 1,084.74 | 43.83 | 4.041% | 10.031% |

These are instrumented per-frame timer sums. Enabling the profiling branches
can perturb execution, so they are not wall-clock speed measurements and do
not support a performance claim or code change.

## Inline attribution result

The earlier read-only DbgHelp pass used the exact Unit 1416 PDB and found that
all 1,525 direct sampled rows mapped to `crates/rvp9-encoder/src/lib.rs:11471`
had inline depth 0. The available inline-context APIs therefore split none
of those rows among TX4 transform, quantizer, `block_error`, `cost_coeffs`, or
the loop. The 3.72% sampled callsite remains an unsplit attribution ceiling,
not proof of duplicate selected-emission work.

No source edit, output mismatch, or timing result beyond these diagnostic
sums was found.
