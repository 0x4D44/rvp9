# Unit 1417 — exact picker attribution

This is a read-only follow-up to the exact Unit 1416 Crucible profile. It
made no codec or benchmark-source change. The retained run used integrated
source `5b6822c05c8d7ff3005902715d618f0ecbfbfd2c`: twenty serial 60-frame
1920×1080 Ducks encodes, CPU-used 6, CBR 1260 kb/s, one thread, tile 0, AQ0,
noise 0, lag 0, and KF60. It was a native release build with
`CARGO_PROFILE_RELEASE_DEBUG=2`, not PGO. Every IVF output was 529,682 bytes
with SHA-256 `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`.

The exact input, executable, PDB, ETL, and xperf report identities are:

| Artifact | SHA-256 |
|---|---|
| Ducks input | `6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7` |
| `rvp9enc.exe` | `8d175c1bffba875f8381461a7be447335ad8ec77847ef83bb07b965474b29f6e` |
| `rvp9enc.pdb` | `5dca6f9dca6cbc1ecbf924b090231b5fd86a149f590dec3985c3234d41014f2e` |
| ETL | `e7e4c331f99de00b532905e509f5c04b041baf3dbb70d72535ee544a86e9c0a8` |
| xperf symbol report | `967a5ea885d6212122767f1fa33f2fd80456968dcc1b46e55d686a835ff0cb58` |

## Current self-weight ranking

The independently re-read remote report sums to 40,988,965 weighted
`rvp9enc.exe` self samples. These are distinct-symbol self weights, not
additive call-tree shares.

| Rank | Symbol | Self samples | Share |
|---:|---|---:|---:|
| 1 | `pick_and_simulate_inter_encode_partitions::closure$10` | 5,273,714 | 12.8662% |
| 2 | `quantize_fp_avx2::vp9_quantize_fp_avx2_inner` | 2,806,832 | 6.8478% |
| 3 | `sad4d_avx2::vpx_sad16x16x4d_avx2` | 2,080,005 | 5.0745% |
| 4 | `mode_decision::inter::pick_inter_mode_impl` | 1,865,459 | 4.5511% |
| 5 | `full_pixel::diamond_search_sad_from_step` | 1,756,889 | 4.2862% |
| 6 | `transforms::forward::fdct16x16` | 1,753,713 | 4.2785% |
| 7 | `variance_avx2::var_sum_sse_16x16_inner` | 1,719,943 | 4.1961% |
| 8 | `subpel_variance_avx2::vpx_sub_pixel_variance16x16_avx2` | 1,527,768 | 3.7273% |
| 9 | `transforms::forward::fdct8x8` | 1,509,356 | 3.6823% |
| 10 | `mode_decision::inter::evaluate_candidate_with_slots` | 1,449,906 | 3.5373% |

## Picker line attribution

The selected fresh candidate is the emission-side
`choose_inter_luma_tx_size` work charged to
`crates/rvp9-encoder/src/lib.rs:11471`. It is a design-study target, not a
claim that a cache or shortcut is correct. The remote source files were the
exact Unit 1416 executable/PDB under
`D:\worktrees\rvp9-2\20260924-TSK-HUM-fresh-x86-motion-search-attribution-unit\.scratch\profile\`.

DbgHelp loaded the PDB at image base `0x7ff7bcd10000`. The picker symbol began
at RVA `0x12def0`, had size `94,957` bytes, and ended at RVA `0x1451dd`.
Filtering `rvp9-cpu6-sampled-rvp9enc.txt` to sampled IPs in that range gave
5,274 direct rows and 1,189 unique IPs. `SymGetLineFromAddr64` mapped all
5,274 rows; 4,943 had nonzero source lines and 331 were line-zero/missing.

The leading mapped lines were:

| Source line | Direct rows | Share of picker rows |
|---:|---:|---:|
| `lib.rs:11471` | 1,525 | 28.915% |
| `lib.rs:11588` | 345 | 6.542% |
| `lib.rs:12115` | 254 | 4.816% |
| `lib.rs:10126` | 231 | 4.380% |
| `lib.rs:11563` | 223 | 4.228% |

The `:11471` rows are about 3.72% of the 40,989 direct encoder rows. That is
a directional sampled ceiling, not elapsed runtime or a measured speedup;
optimized PDB line tables can charge inlined work to this callsite. It makes
the chooser a plausible greater-than-1% sampled target only if a future
exact-output experiment removes a meaningful part of that region.

Unit 1412 saw the same callsite on older integrated source `cdbf7a6a`: 1,224
of 4,797 picker rows (25.52%) and 3.09% of 39,628 direct encoder rows. The
current 1,525-row result confirms persistence, not an improvement.

Do not reinterpret this callsite as proof of selected-emission duplication.
Units 585 and 586 in `wrk_journals/2026.07.30 - JRN - production parity
drive.md` tested copied and zero-copy selected-TX coefficient caches; both were
byte-exact but neutral within paired noise and were removed. A cache is
therefore a design fork, not the recommendation from this profile; it needs
new evidence of materially larger duplicate work before revival.

The latest direct pinned-libvpx comparison remains Unit 1415's equal-clip
paired ratio `1.230937x`, well above the `1.05x` target. Unit 1417 adds no
timing result and retains no source change.
