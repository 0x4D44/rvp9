# Unit 1420 — exact 16-lane AVX2 quantizer screen (rejected)

The exact Crucible profile assigned 6.8478% of encoder self-sample weight to
`vp9_quantize_fp_avx2_inner`. We tested a 16-lane i16 quantization pass that
fused inverse-scan EOB tracking with coefficient production. The candidate kept
scalar DC handling and the existing i32 path for 4x4 blocks, invalid direct
parameters, and i32 extremes. This is distinct from Unit 1404's AC-tail removal
and Unit 1410's separate EOB reducer.

The complete source change is archived as `unit1420-candidate.patch.gz` (gzip
SHA-256 `fd5dc7e5072ebc3e0456c787490b5876951925be0c736eb934eb3ae9d4230b3e`;
decompressed patch SHA-256
`40b53f0b25c6ed1e91bd61253b70c8de84b0b92a068dd4f59e2bbbfb472b2110`).
The task branch has no codec change after reversing that exact patch.

## Exactness

- Crucible x86-64 release `cargo test --locked --release -p rvp9-core --lib quantize_fp_avx2 -- --nocapture` passed 8/8 tests. The suite covers scalar coefficient/EOB differentials, q-index and delta edges, DC replacement, skip output clearing, invalid parameters, and release-mode i32 extremes.
- Reversing the new DC/EOB condition made `i16_path_uses_scalar_dc_for_eob` fail with actual EOB 1 versus expected 0. Restoring it returned the same eight-test suite to green.
- Every candidate and control IVF in both paired screens had the same SHA-256, `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`. Each arm had one distinct hash.
- Local `cargo fmt --all -- --check` and `git diff --check` passed before the candidate source was reversed.

## Performance

Both screens used the pinned 60-frame 1920x1080 Ducks clip, native Crucible
release binaries, CPU6, CBR 1260 kbit/s, one thread, no AQ/noise sensitivity,
and two alternating rotations. The retained exact-profile control executable
SHA-256 was `8d175c1bffba875f8381461a7be447335ad8ec77847ef83bb07b965474b29f6e`;
the candidate SHA-256 was
`5248455504ff88e08e027e09358590f6d0adbeab838bc40bf06aa10ade4f6134`.
Ratios are candidate/control; below 1 is faster.

| Screen | Measured pairs | Separate ratio | Paired ratio | Decision |
| --- | ---: | ---: | ---: | --- |
| B | 6 | 1.660849x | 1.685294x | Inconclusive: raw pair times swung in both directions under contention. |
| C | 30 | 1.009047x | 1.012607x | No measurable gain; reject candidate. |

The shorter screen's striking summary was not a reliable slowdown: individual
pair timings ranged from candidate wins to large losses. The larger screen
instead put the candidate about 1% slower, with 39.968 versus 39.610 ms/frame
by separate medians. We do not claim a precise regression, only that this
prototype failed to show a reproducible gain. Its added packing, overflow
checks, and EOB work were not free enough to beat the current eight-lane path.

`screen-b/` and `screen-c/` contain each harness manifest, raw rows, and
summary. An earlier three-clip run was stopped because its initial progress
estimate was wrong; its partial output was never used for a performance claim.
