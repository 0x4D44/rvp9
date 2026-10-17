# Exact-candidate unsafe-boundary review — 2026-10-17

Candidate: `fb5c0c917038db6c640b3dbc7c2b84b567233333`. Each read-only source audit independently verified this exact HEAD and a clean tracked tree in `/Users/md/worktrees/rvp9/20261017-TSK-HUM-requalify-post-fix-candidate`. No source files changed and no tests or runtime probes were run for these audits. Results below are static source findings; they do not establish input reachability from a bitstream or prove release-profile runtime behavior.

## Shared core, decoder, and I/O disposition

All 49 production unsafe sites in the non-architecture core, decoder, and I/O scope were reviewed; no unreviewed remainder was reported. The source sites and disposition are:

| Source | Sites | Review disposition |
| --- | --- | --- |
| `rvp9-core/src/dsp/accelerated.rs` | 5: lines 102, 123, 145, 171, 190 | Unsafe functions document `# Safety` contracts. Callers must establish CPU support and buffer footprints; forwarding preserves the borrowed slices. Architecture-specific bodies are covered separately below. |
| `rvp9-core/src/entropy/detokenize.rs` | 12: lines 295, 306, 448, 451, 566, 571, 610, 613, 623, 628, 640, 647 | Every block has a `// SAFETY:` comment. Lines 295, 610, and 640 trust that scan length matches transform size, but only a debug assertion establishes that invariant. This is tracked as FLU-00051. Other bounds, context, and cache proofs hold for matching static scans. |
| `rvp9-core/src/loopfilter/apply.rs` | 15: lines 99, 102, 215, 217, 231, 233, 247, 249, 509, 617, 880, 913, 944, 979, 996 | Impls and blocks have `// SAFETY:` comments; unsafe functions have `# Safety` contracts. Current serial production use has whole-plane exclusivity and joins before deallocation. The public rectangle-only contracts at lines 913 and 979 permit calls that would form overlapping whole-plane mutable slices at lines 944 and 996; tracked as FLU-00052. |
| `rvp9-core/src/tile_threading/scope.rs` | 2: lines 111, 170 | Both blocks have `// SAFETY:` comments. Scope draining covers panic paths; job captures are consumed before completion notification. |
| `rvp9-core/src/tile_threading/lf_pool.rs` | 2: lines 206, 295 | The unsafe function documents the no-forget contract; the transmute has a `// SAFETY:` comment. Wait/drop consumes completion only after the borrowed closure is finished. |
| `rvp9-decoder/src/tile.rs` | 2: lines 280, 285 | Both blocks have `// SAFETY:` comments. The private partition helper receives bounded coordinates and masks its left index to eight entries. |
| `rvp9-decoder/src/tile_ctx.rs` | 5: lines 127, 149, 410, 458, 575 | Constructor has a `# Safety` contract; impl and blocks have `// SAFETY:` comments. Checked rectangle bounds, allocation lifetime, disjoint columns, sequential tile rows, and context borrows establish the reviewed obligations. |
| `rvp9-decoder/src/profile.rs` | 1: line 93 | `// SAFETY:` documents the x86-64-gated TSC intrinsic and supported-host assumption; there are no memory aliasing or bounds obligations. |
| `rvp9-decoder/src/lib.rs` | 5: lines 1947, 2215, 3115, 3434, 4009 | Every block has `// SAFETY:`. Parallel tile work is restricted to one tile row, inline tiles are sequential, storage survives the joined scope, and loop filtering uses one worker after decode/merge. |
| `rvp9-io` | 0 | No production unsafe sites found. |

The 49 sites comprise 33 blocks, 13 unsafe functions, and three unsafe impls: 36 sites in core, 13 in decoder, and none in I/O. The safe `decode_coefs` route can pair a legitimate 4×4 scan with Tx32x32 and `seg_eob = 17`; a continuing zero-token run then reaches an unchecked scan access beyond the 16-entry table. This is a source-derived residual size-validation gap in the scan trust boundary, distinct from constructing malformed scans. No runtime reproduction was performed.

## Architecture and encoder inventories

The x86 audit covered all 64 files under `rvp9-core/src/dsp/x86_64/`, plus `transforms/forward_sse2.rs` and `mode_decision/hadamard_sse2.rs`. Seven x86 files were test/doc-only; the 59 production files contained 302 unsafe functions, 87 explicit unsafe blocks, and no unsafe impls or traits. All 87 blocks had nearby `// SAFETY:` comments. The full production scope was dispositioned with no unreviewed remainder. Caller geometry and transform arithmetic equivalence were not independently proved.

The AArch64 NEON audit covered these 19 production files. Every explicit unsafe block had a `// SAFETY:` comment; there were no unsafe impls. Counts are source functions/blocks, before macro expansion.

| File under `rvp9-core/src/` | Unsafe functions | Unsafe blocks | Disposition |
| --- | ---: | ---: | --- |
| `dsp/aarch64/convolve8_neon.rs` | 21 | 8 | Reviewed; optional horizontal i8mm/dotprod loads exceed the validated slice by one byte (FLU-00039); unchecked stride-span arithmetic also belongs to FLU-00046. |
| `dsp/aarch64/convolve8_scaled_neon.rs` | 12 | 5 | Reviewed; signed Q4 coordinate overflow invalidates the span proof (FLU-00053). |
| `dsp/aarch64/convolve_copy_neon.rs` | 1 | 1 | Reviewed; unchecked footprint arithmetic is within FLU-00046. |
| `dsp/aarch64/idct8x8_neon.rs` | 13 | 3 | Reviewed; destination stride arithmetic is within FLU-00046. |
| `dsp/aarch64/int_pro_neon.rs` | 5 | 5 | Reviewed; unchecked row-kernel stride guards are within FLU-00046; fused projection spans are checked. |
| `dsp/aarch64/intra_pred_neon.rs` | 3 | 3 | Reviewed; checked layout establishes block, destination, and edge footprints. |
| `dsp/aarch64/lpf16_neon.rs` | 27 | 9 | Reviewed; mutable slice is exclusive; unchecked geometry arithmetic is within FLU-00046. |
| `dsp/aarch64/minmax_neon.rs` | 1 | 1 | Reviewed; representative release wraparound case is within FLU-00046. |
| `dsp/aarch64/mod.rs` | 0 | 0 | No unsafe site. |
| `dsp/aarch64/quantize_neon.rs` | 24 | 9 | Reviewed; fixed coefficient counts establish the vector spans. |
| `dsp/aarch64/sad4d_neon.rs` | 2 | 2 | Reviewed; unchecked footprint arithmetic is within FLU-00046. |
| `dsp/aarch64/sad_neon.rs` | 1 | 1 | Reviewed; unchecked footprint arithmetic is within FLU-00046. |
| `dsp/aarch64/subpel_variance_neon.rs` | 19 | 8 | Reviewed; checked local loads, but unchecked span guards are within FLU-00046. |
| `dsp/aarch64/subtract_neon.rs` | 1 | 2 | Reviewed; input span guards are within FLU-00046. |
| `dsp/aarch64/variance_neon.rs` | 10 | 6 | Reviewed; stride guards are within FLU-00046. |
| `dsp/aarch64/vbp_neon.rs` | 1 | 3 | Reviewed; actual 16-byte row loads fit the mathematical footprint, but span arithmetic is within FLU-00046. |
| `transforms/forward_neon.rs` | 49 | 8 | Reviewed; variable input-stride guards are within FLU-00046. |
| `transforms/inverse_neon.rs` | 10 | 1 | Reviewed; destination footprint arithmetic is within FLU-00046. |
| `mode_decision/hadamard_neon.rs` | 8 | 18 | Reviewed; source/prediction footprint guards are within FLU-00046. |

FLU-00039's wider optional-feature case is source-derived: width-eight i8mm/dotprod paths load `w + 8` bytes while validation requires `w + 7`; no separate runtime reproduction was run. FLU-00046 covers the same unchecked-stride arithmetic pattern across 15 NEON files, not a second record. For example, a 64-byte slice with stride `usize::MAX / 7` makes `7 * stride + 8` wrap to six on 64-bit release before a min/max row pointer leaves the allocation. Neither example establishes normal encoder geometry reachability.

The encoder audit found 32 production unsafe blocks and 21 unsafe functions, with no unsafe impls; test-only sites were excluded. Each block had adjacent `// SAFETY:` reasoning. File disposition:

| Encoder source | Unsafe functions | Unsafe blocks | Disposition |
| --- | ---: | ---: | --- |
| `loopfilter.rs` | 0 | 1 at 566 | Plane borrows are exclusive under the one-worker invariant. |
| `ml_partition/nn_predict.rs` | 0 | 1 at 146 | Stack-buffer lifetime and reborrow are valid; index/length stay within initialized scratch. |
| `scale_reference.rs` | 4 at 137, 160, 193, 227 | 8 at 112, 122, 143, 170, 206, 237, 241, 246 | Checked dimensions and row slices establish load/store spans. |
| `scale_source.rs` | 7 at 244, 435, 463, 486, 510, 541, 584 | 10 at 211, 266, 328, 394, 420, 469, 494, 524, 551, 596 | Validated dimensions and vector-loop bounds cover footprints. |
| `pre_analysis.rs` | 3 at 1361, 1448, 1458 | 5 at 1259, 1267, 1386, 1468, 1492 | Runtime AVX2 dispatch and bounded interior rows establish the reviewed loads. |
| `denoiser.rs` | 7 at 597, 610, 618, 664, 700, 812, 931 | 6 at 381, 628, 674, 703, 816, 937 | Validated plane windows and block comments cover pointer footprints. |
| `frame_loop.rs` | 0 | 1 at 5124 | Scoped borrow remains valid because `wait()` joins before reconstruction commits or drops. |

Private target-feature unsafe functions sometimes rely on caller-site comments rather than a standalone `# Safety` section; no exported unsafe API or actionable memory-safety defect was found in the encoder scope.

The CLI/benchmark/harness audit found 12 production `unsafe {}` blocks and three `unsafe extern` declaration blocks, with no unsafe functions or impls. All 12 operation blocks had `SAFETY` comments and their reviewed arguments/layouts were consistent. The three extern declaration blocks do not have adjacent `SAFETY` or `# Safety` text: `rvp9-cli/src/path_guard.rs:186`, `rvp9-bench/src/bin/soak.rs:284,293`. This is a comment-evidence gap, not a confirmed ABI or runtime defect. Other sites: CLI calls at `path_guard.rs:196,213`; benchmark calls at `soak.rs:307`, `decode-profile-tiles.rs:210`, and `dsp-microbench.rs:101,116,122,167,207,219,1165`; harness calls at `f02_block_oracle.rs:708,712`.

## New and expanded ledger findings

Seven new Open records were minted after source inspection; two existing records were expanded. Findings are source-only unless noted:

| Record | Finding and evidence limit |
| --- | --- |
| [FLU-00039](../../../../../bugs/RVP9-BUG-FLU-00039.md) — scope expanded | The existing width-four Arm convolution overread has a sibling width-eight i8mm/dotprod path that reads one byte beyond its validated source slice. Existing ASan evidence covers the width-four case; the width-eight observation is static only. |
| [FLU-00046](../../../../../bugs/RVP9-BUG-FLU-00046.md) — scope expanded | The unchecked AArch64 span arithmetic appears in 15 NEON source files. It remains one root-pattern record; no normal-geometry reachability is claimed. |
| [FLU-00048](../../../../../bugs/RVP9-BUG-FLU-00048.md) | Safe x86 SIMD dispatch getters expose AVX2 kernels whose buffer guards exist only as `debug_assert!`; the undersized-slice counterexample was not executed. |
| [FLU-00049](../../../../../bugs/RVP9-BUG-FLU-00049.md) | The SSE2 `fdct8x8` span check can wrap `7 * stride + 8` in release before unsafe input access; source-derived only. |
| [FLU-00050](../../../../../bugs/RVP9-BUG-FLU-00050.md) | 32×32 SSE2/AVX2 IDCT storage helpers form a pointer beyond one-past after the last row for a valid tight output slice; source-derived only. |
| [FLU-00051](../../../../../bugs/RVP9-BUG-FLU-00051.md) | Safe coefficient decode accepts a valid scan whose length is shorter than the independent transform size; a continuing run reaches unchecked scan access. No runtime reproduction. |
| [FLU-00052](../../../../../bugs/RVP9-BUG-FLU-00052.md) | Public loop-filter unsafe contracts omit the whole-plane exclusivity required by their raw mutable-slice reconstruction. Current production calls are serial; no runtime race was run. |
| [FLU-00053](../../../../../bugs/RVP9-BUG-FLU-00053.md) | Scaled NEON convolution overflows signed Q4 coordinate arithmetic before pointer validation; exact source-derived example, not executed. |
| [FLU-00054](../../../../../bugs/RVP9-BUG-FLU-00054.md) | Three Windows unsafe FFI declaration blocks lack adjacent safety reasoning required by repository guidance. This is a documentation gap; no ABI mismatch or runtime defect is established. |

The unsafe-site reviews do not establish SIMD/scalar bit-exactness, decoder throughput, sanitizer coverage, fuzz coverage, or memory safety of unreviewed inputs. They also do not constitute fixes or close any ledger item. No source changes or tests were made. The separate x86 audit explicitly excluded test/doc-only files; the NEON and encoder inventories likewise excluded test-only unsafe sites. No claim about the non-workspace fuzz crate follows from this production audit.
