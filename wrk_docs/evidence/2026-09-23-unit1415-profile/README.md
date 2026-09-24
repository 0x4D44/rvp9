# Unit 1415 — exact integrated x86 motion-search profile

Crucible profiled integrated `5e8426acab3e3a89e7afd7906e6f4090d23c397e` with
20 serial 60-frame 1920×1080 Ducks encodes: CPU-used 6, CBR 1260 kb/s,
one thread, tile 0, AQ0, noise 0, lag 0, KF60. The control was a plain
`cargo build --locked --release -p rvp9-cli --bin rvp9enc`, with native
target CPU from repo config and `CARGO_PROFILE_RELEASE_DEBUG=2` for PDB lines;
it was not a PGO build. All 20 IVF files had SHA-256
`8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`.

The exact control executable, PDB, input, ETL, and symbol report have SHA-256
values `a894a7a37b738aae11e3ab460a3ffcf4882aadf7885cb93b6f1dbccab54c6c70`,
`12027a7ff654edb5dd32081739ac67b56c57ebf05739a358d7a7045e2f2cadb1`,
`6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7`,
`bfc2c37e912699892a5a8f475d2bca8fe3e3d148d145f7605135b7e614b57dd4`,
and `4d386dd9a577f7120fbbb0e3c8ffee05c54135c12d9b11a3cfd360521e8321b9`
respectively. The matching artifacts were recorded under
`D:\worktrees\rvp9-2\20260923-TSK-HUM-x86-unit1415-profile\.scratch\profile\`.

Update, 2026-09-24: that Crucible profile directory is now absent. The hashes
above still identify the measured run, but the ETL, PDB, and executable cannot
currently be reprocessed for source-line attribution. The saved aggregate
results below remain the available evidence; obtaining new exact IP attribution
requires a fresh matched profile.

The symbol report gives 1,503,616 weighted self samples to
`sad4d_at_from_source` and 1,998,136 to its caller
`diamond_search_sad_from_step`; the former is about 3.3% of encoder symbol
weight. The AVX2 quantizer is still prominent at 3,041,410 weighted samples.
The ETL reports three lost events. The bounded raw-IP line extraction did not
complete, so this profile supports a specific function-level screen, not a
claim about which instructions inside the helper dominate.

## Four-reference bounds screen

The bounded candidate checked the union of four reference rectangles once,
then derived four in-plane slices for the existing SAD4d call. Clipped and
malformed-short-slice cases retained the original per-lane `block_at`
fallback. The differential covered distinct lanes, non-tight stride, clipped
references, invalid source, and short backing storage. Deliberately changing
one expected lane from 1,728 to 1,729 failed; restoring it passed. The local
AArch64 core suite passed 1,719 tests (two fixture-dependent tests ignored),
targeted clippy passed, and Crucible x86 passed all 20 full-pixel tests.

Two 1080p three-clip paired screens used the exact integrated control above,
one warmup, three pairs, and two clip-order rotations. Candidate and control
IVF hashes matched per clip and stayed stable. Screen B used the default
line-table-only candidate build and measured a 1.019292 paired corpus ratio.
Screen C rebuilt the candidate with the control's `CARGO_PROFILE_RELEASE_DEBUG=2`
and measured 1.023270 paired: Ducks 1.057438, in-to-tree 1.008883,
old-town-cross 1.004327. Both screens therefore favour the control. Their
complete raw rows, summaries, and manifests are adjacent to this file.
The candidate is rejected; the source change will not land.

## Fresh pinned-libvpx comparison

The unchanged integrated control was compared with the pinned libvpx
`12f3a2ac603e8f10742105519e0cd03c3b8f71dd` build (vpxenc SHA-256
`88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d`).
The same three clips and operating point used one warmup, three pairs, and
two clip-order rotations. Every arm produced one stable output hash per clip.
The paired RVP9/libvpx ratios were 1.352682 for Ducks, 1.225778 for
in-to-tree, and 1.124863 for old-town-cross: 1.230937 for the equal-clip
corpus. Full raw rows, summary, and manifest are `libvpx-*` beside this file.

The 60-second host-load samples ranged from 4.12 to 16.78 busy logical cores
of 32, with no load ceiling. The raw rows contain asymmetric stalls: one Ducks
RVP9 encode took 7.81 seconds against a 1.71-second reference; an in-to-tree
reference encode took 13.80 seconds; one old-town-cross RVP9 encode took
4.02 seconds. The paired median is therefore directional rather than a
precise trend estimate. It is close to Unit 1410's earlier 1.227734 ratio,
and remains well above the 1.05 parity target.
