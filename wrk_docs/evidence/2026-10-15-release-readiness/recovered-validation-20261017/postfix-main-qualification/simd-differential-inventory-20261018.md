# SIMD differential inventory

**Scope:** source-only mapping of HLD-SIMD §3 to production kernels, shared dispatch paths, and existing scalar differential tests. Audited the current release-readiness worktree against product source `fb5c0c917038db6c640b3dbc7c2b84b567233333`; `git diff fb5c0c9..HEAD -- crates/rvp9-core/src notes/hld/02-simd-dispatch.md` was empty. No tests were run, so this is an inventory of test code and implementation shape, not pass evidence.

## Finding

The implemented SIMD families generally have source-level scalar oracles. The HLD §3 list is a design inventory, not an accurate inventory of the current shared dispatch table: several named entry points are absent, some families are scalar-only by design, and architecture support differs from the HLD's proposed SSE2/AVX2/NEON variants. A blanket claim that every HLD kernel has passed scalar differential tests on both architectures is unsupported.

## Implemented families with direct or closely mapped scalar oracles

| Family | Current implementation and representative oracle | Scope and limits |
| --- | --- | --- |
| SAD and SAD x4D | `dsp/x86_64/sad_avx2.rs` (`differential_random_all_sizes`, `differential_edge_content`) and `sad4d_avx2.rs` (`differential_random_all_sizes`, shared-plane-offset and edge tests); matching ARM tests in `dsp/aarch64/sad_neon.rs` and `sad4d_neon.rs`. | All 13 VP9 shapes; x86 AVX2 and ARM NEON/dotprod. HLD's separate SSE2 variants are absent. |
| Variance reductions, subpixel variance, 8×8 variance | `distortion.rs` scalar reductions; AVX2 and NEON tests in `variance_*` and `subpel_variance_*` compare sums/SSE or final variance/SSE against scalar. | Tests cover the production reduction signatures, not every named HLD wrapper. Subpixel tests also compare temporary bilinear bytes. Shape and ISA support is narrower than the HLD proposal. |
| SSE/MSE-related reductions | `sse_block_sse2.rs` tests all VP9 shapes/strides against a local scalar loop; `get_var_sse_8x8` has signed-sum/SSE checks. | Related arithmetic is covered, but distinct HLD entry points such as MSE variants, `get_mb_ss`, and `vpx_sse` are not all separate dispatch slots. `get4x4sse_cs` is scalar-only. |
| Residual subtraction and Hadamard | `subtract_sse2.rs` and `subtract_neon.rs` compare residual buffers and nonzero results with scalar. `mode_decision/hadamard.rs` compares SSE2/NEON Hadamard4/8/16 coefficient arrays and fused subtract results with scalar. | Production square residual helpers are covered; the HLD's arbitrary rectangular signature is not. SATD has no dedicated SIMD slot. |
| Quantization | `quant/forward.rs` scalar paths; `quantize_fp_avx2.rs` and `quantize_neon.rs` compare EOB, qcoeff and dqcoeff with scalar across their implemented sizes and inputs. | NEON has broad size/scan/qindex coverage. The primary AVX2 random-coefficient test is one deterministic vector at qindex 60, with additional sparse/EOB/reuse tests. 32×32 SIMD is NEON-only. |
| Forward transforms | ARM tests in `transforms/forward.rs` compare DCT4/8/16/32 and FHT4/8/16 outputs to scalar. x86 SSE2 tests cover DCT8/16, guard/fallback behavior and AVX2 first-pass equivalence. | Direct scalar comparisons exercise valid-residual domains. DCT4/32 x86 are scalar. DCT `_1` variants and DCT32 `_rd` are absent; FWHT4 is explicitly deferred. |
| Inverse transforms | AVX2/SSE2/NEON files under `dsp/x86_64` and `dsp/aarch64` have scalar comparisons for IDCT4/8/16/32 and IHT4/8/16 where implemented; see the detailed cases below. | Some tests are bounded-domain or indirect; do not claim universal equivalence or coverage of old, unselected backends. IWHT4 is scalar. |
| Intra predictors | AVX2 DC, V/H/TM and directional predictor files compare reconstructed pixels to scalar; ARM `intra_pred_neon.rs` covers its V/H/DC-full subset. | AVX2 mode coverage is broad. ARM directional/TM/DC-top/left/128 paths remain scalar. |
| Convolution and scaled filters | AVX2 copy/average, 8-tap horizontal/vertical/2D, average filters and scaled filters compare output buffers/pixels to scalar. ARM tests cover copy, 8-tap filters, and non-average scaled filters. | ARM average filters remain scalar. HLD scaled horizontal/vertical names are expressed by common scaled interfaces, not separate slots. |
| Loop filters | All twelve AVX2 single/dual × horizontal/vertical × 4/8/16 variants have scalar buffer comparisons; ARM `lpf16_neon.rs` covers the same shapes and boundary saturation. | Source-level direct oracle coverage exists on each architecture; execution still needs native-platform evidence. |
| Projection, vector variance, min/max | `int_pro_sse2.rs` and `int_pro_neon.rs` compare projection arrays and vector variance with scalar, including stride/edge cases. `minmax_sse2.rs` and `minmax_neon.rs` compare min/max over strides and offsets. | x86 column projection is fused rather than a separate named kernel. Fused avg-difference grid tests are not standalone HLD `avg4`/`avg8` tests. |
| Frame borders | `dispatch.rs` verifies `extend_frame_borders` remains a scalar adapter; HLD lists border extension/copy as scalar-only. | This is an intentional scalar slot, not a missing SIMD differential. Separate `copy_frame`/`copy_y` RTCD slots are absent. |

## HLD rows with no exact production SIMD slot or oracle

The source audit found no matching production/shared-dispatch implementation for SAD-skip, SAD-skip x4D, compound-average prediction, `sum_squares_2d_i16`, denoiser filtering, or full-frame `scale_and_extend_frame`. The HLD itself defers SIMD for SAD-average, subpixel-average variance, `filter_by_weight8/16`, diamond-search SAD and frame-border/copy operations; these are scalar-only decisions rather than failed SIMD tests. The exact HLD MSE/get-macro names are also not uniformly separate dispatch slots even where a related SSE/variance reduction exists. These distinctions require an owner decision if HLD §3 is treated as a release gate.

## Row-by-row crosswalk

Paths in this table are relative to `crates/rvp9-core/src/`; `D` means `dsp/`, `X` means `dsp/x86_64/`, `N` means `dsp/aarch64/`, and `T` means `transforms/`.

| HLD row | Current mapping and oracle source | Disposition |
| --- | --- | --- |
| 3.1 SAD, 13 shapes | `D/distortion.rs:211`; `X/sad_avx2.rs:287,309`; `N/sad_neon.rs:187,209`. Tests compare random shapes/strides and edge content with scalar. | Direct AVX2 and NEON/dotprod oracle; no HLD SSE2 variant. |
| 3.2 SAD-skip | No production function or shared slot found. | Missing implementation and oracle. |
| 3.3 SAD-average | No production function or shared slot found; HLD defers SIMD at `notes/hld/02-simd-dispatch.md:456`. | Scalar-only HLD decision. |
| 3.4 SAD x4D, all shapes | Scalar `D/distortion.rs:435`; x86 `X/sad4d_avx2.rs:316,355,393`; ARM `N/sad4d_neon.rs:279,314`. Tests compare four outputs against scalar SAD. | Direct AVX2 and NEON/dotprod oracle; no HLD SSE2 variant. |
| 3.5 SAD-skip x4D | No matching production implementation or slot found. | Missing implementation and oracle. |
| 3.6 Variance | Production reduction is `(sum,sse)`: `D/distortion.rs:127`, `D/dispatch.rs:3274`; scalar differential tests `X/variance_avx2.rs:440,462`, `N/variance_neon.rs:869,896`. | Direct oracle for the dispatched reduction, not separate HLD `(variance,sse)` wrappers. |
| 3.7 Subpixel variance | Scalar `D/distortion.rs:309`; AVX2 `X/subpel_variance_avx2.rs:616,658,686`; NEON `N/subpel_variance_neon.rs:870,916`; dispatch `D/dispatch.rs:3298`. | Direct tests cover supported shapes/offsets and intermediate bytes; support differs from HLD. |
| 3.8 Subpixel-average variance | No matching slot; HLD says scalar-only at `notes/hld/02-simd-dispatch.md:644`. | Scalar-only HLD decision. |
| 3.9 get8x8/get16x16 variance | `D/distortion.rs:160`, `D/dispatch.rs:3286`; AVX2 cell test `X/variance_avx2.rs:498`; ARM tests `N/variance_neon.rs:327,338,791,869`. | Covered by equivalent reductions; not all exact named C wrappers. |
| 3.9 MSE variants | No distinct MSE slots; `X/sse_block_sse2.rs:92,146,183` tests related pixel-SSE reductions against scalar. | Related arithmetic oracle only, not each HLD entry point. |
| 3.9 get4x4sse_cs | Scalar `D/distortion.rs:357`; scalar-focused tests at `:649,874`. | No dispatched SIMD implementation. |
| 3.9 get_mb_ss and vpx_sse | No exact named shared slot found; HLD describes `vpx_sse` as scalar-only at `notes/hld/02-simd-dispatch.md:2061`. | Missing named slots or scalar-only; no SIMD oracle to run. |
| 3.10 comp_avg_pred | No matching slot found. `convolve_avg` has different inputs and semantics. | Missing implementation and oracle. |
| 3.11 subtract_block | Scalar square helpers `D/accelerated.rs:23,40`; SSE2 `X/subtract_sse2.rs:89,123`; NEON `N/subtract_neon.rs:66,97`; tests x86 `:237,289`, ARM `:157,204`. | Direct oracle for square production helpers, not the arbitrary rectangular HLD signature. |
| 3.12 sum_squares_2d_i16 | No matching shared implementation found; byte-difference `sse_block_sse2` is a different operation. | Missing implementation and oracle. |
| 3.13 Hadamard and SATD | Hadamard shared path `mode_decision/hadamard.rs:198,218,340`; scalar implementations `:79,116,168`; direct tests `:531,574,626`. SATD reduction is scalar at `:366`. | Hadamard direct SSE2/NEON oracle; SATD has no dedicated SIMD slot. |
| 3.14 block_error variants | `mode_decision/block_yrd.rs:263` is scalar; fused quantization metrics use it as an oracle. | No standalone dispatched SIMD implementation. |
| 3.15 quantize variants | AVX2 `quant/forward.rs:489`, `X/quantize_fp_avx2.rs:400`; NEON tests `N/quantize_neon.rs:1201,1323,1400,1582,1635,1698`. | Direct EOB/qcoeff/dqcoeff comparisons for implemented paths; 32×32 SIMD is NEON-only; parameter breadth differs by ISA. |
| 3.16 forward DCT | Public paths `T/forward.rs:215,501,1096,1547`; ARM scalar comparisons at `:2352,2395,2600,2643`; x86 DCT8/16 comparisons in `T/forward_sse2.rs:429,612,658`. | Direct tests for implemented valid-residual domains; x86 DCT4/32 scalar. `_1` variants and DCT32 `_rd` absent. |
| 3.17 FHT and FWHT | FHT paths `T/forward.rs:294,653,1186`, ARM scalar comparisons `:2455,2509,2563`; `T/forward.rs:26` defers FWHT4. | ARM FHT direct oracle; x86 scalar; FWHT4 absent/deferred. |
| 3.18 inverse DCT and IWHT | See inverse-transform qualifications below for selected IDCT4/8/16/32 paths. IWHT4 is scalar in `T/inverse.rs`; its `_1` test compares scalar implementations. | Mixed direct, indirect, bounded-domain and scalar-only coverage. |
| 3.19 inverse hybrid transforms | AVX2 IHT4/8/16 tests in `X/iht4x4_avx2.rs:597`, `X/iht8x8_avx2.rs:668`, `X/iht16x16_avx2.rs:1006`; ARM IHT8 tests `N/idct8x8_neon.rs:501,530,562`. | Direct scalar output oracles for implemented variants; ARM IHT4/16 scalar. |
| 3.20 intra predictors | AVX2 differential tests in `X/dc_pred_4x4_avx2.rs`, `X/dc_predict_*_avx2.rs`, `X/*_predictor_avx2.rs`; ARM subset `N/intra_pred_neon.rs:161`. | Broad AVX2 direct coverage; ARM supports a narrower subset. |
| 3.21 convolution and scaled filters | AVX2 copy/average and 8-tap files under `X/`; ARM `N/convolve_copy_neon.rs:144`, `N/convolve8_neon.rs:885,923`, `N/convolve8_scaled_neon.rs:535,576`. | Direct scalar output checks for supplied variants; ARM average filters scalar; scaled axis variants are common-interface cases, not separate slots. |
| 3.22 loop filters | All single/dual H/V 4/8/16 AVX2 variants have scalar comparisons in `X/lpf_*_avx2*.rs`; ARM equivalents and boundary tests in `N/lpf16_neon.rs:1355,1444,1468,1557,1585,1602`. | Direct source oracles on AVX2 and NEON for all twelve shapes. |
| 3.23 projection and search | `X/int_pro_sse2.rs:420,472,555,601`; `N/int_pro_neon.rs:393,412,426,488,546`. Diamond-search SAD has no dedicated SIMD implementation and is scalar by HLD at `notes/hld/02-simd-dispatch.md:1166`. | Direct projection/vector-variance oracles; x86 column path is fused; diamond SAD scalar-only. |
| 3.24 avg4/avg8/minmax8 | Fused avg-difference grid tests `X/vbp_sse2.rs:154`, `N/vbp_neon.rs:122`; direct min/max tests `X/minmax_sse2.rs:90`, `N/minmax_neon.rs:96`. | Min/max direct; avg grid is only a related fused oracle, not standalone avg4/avg8 slots. |
| 3.25 denoiser_filter | No matching production implementation or slot found. | Missing implementation and oracle. |
| 3.26 scale_and_extend_frame | `inter/scale.rs` scales inter prediction; no matching full-frame SIMD operation found. | Different operation; HLD entry absent. |
| 3.27 filter_by_weight8/16 | No matching SIMD implementation; HLD permits scalar v1 at `notes/hld/02-simd-dispatch.md:1254`. | Scalar-only HLD decision. |
| 3.28 border, extend and copy | `D/dispatch.rs:3237,4358` keeps border extension scalar; HLD marks border/copy scalar-only at `notes/hld/02-simd-dispatch.md:1271`. | Intentional scalar coverage; separate copy-frame/copy-y slots absent. |
| 3.29 out-of-scope kernels | HLD excludes these from v1 at `notes/hld/02-simd-dispatch.md:1277`. | Excluded by design; not a failed differential test. |

## Inverse-transform qualifications

- Dense IDCT16 x86 selects SSE2. Its dense/sparse/EOB differential tests compare SSE2 to a retained AVX2 implementation; that AVX2 implementation has a separate scalar proptest. This is an indirect scalar oracle for the selected SSE2 path.
- IDCT32 i16-lane AVX2 kernels intentionally use saturating packing while scalar arithmetic wraps. Existing scalar comparisons establish equivalence only in bounded coefficient envelopes. The full-i16 test compares AVX2 with SSE2, not scalar.
- For IDCT32 EOB34 and EOB135, use the currently selected SSE2/AVX2 tests in `dispatch.rs` and the corresponding `idct32x32_*` files. Old similarly named AVX2 implementations retained in source are not evidence for the selected backend.
- IDCT/IWHT `_1` paths are scalar fast paths; an equivalence test between two scalar `_1`/dense implementations is not SIMD evidence.

## Backend and test-execution limits

- Runtime capability detection is in `dsp/caps.rs`; x86 and ARM modules are compiled and selected on different native architectures. Source presence or a test on one host does not establish execution on the other.
- Forward NEON transforms (`transforms/forward.rs`), NEON Hadamard (`mode_decision/hadamard.rs`) and NEON inverse16 (`transforms/inverse.rs`) dispatch at compile time outside the shared function table. All-false shared-table caps or `RVP9_FORCE_SCALAR` alone do not prove a fully scalar AArch64 run.
- A dispatch round-trip that only observes scalar-equivalent output is weaker backend-selection evidence than a pointer/counter assertion.
- The HLD's proposed ≥10,000 random inputs per kernel is not met uniformly by existing test source: several use 1,024 iterations, Hadamard uses nine stride jitters per size, and the primary AVX2 quantize random test uses one deterministic vector.
- Arm scalar routing remains blocked by FLU-00043. This source audit ran no tests. A separate [native Apple Silicon follow-up](mac-current-source-simd-differential-20261018/README.md) passed 35 `differential`-named and 33 `scalar`-named tests; that filtered run is partial evidence and does not execute every mapped test or prove scalar-only routing.

## Release-readiness consequence

Keep PLAN §3 SIMD status **open and unqualified**. The inventory improves the map of existing oracles, but does not satisfy the planned native x86-64 and Apple Silicon executions, backend-selection evidence, all-kernel coverage, uncovered-kernel disposition, or scalar-routing proof. Do not count source inspection as test execution, native NEON as scalar coverage, or bounded/indirect comparisons as universal direct equivalence.
