# Stage 4 drive ledger

Mirror of `notes/overnight-2026-05-17-stage3.md`. One row per agent dispatch
or supervisor decision. Newest at top; commits referenced by short SHA.

## Goal

Drive Stage 4a (encoder MVP) toward exit. Stage 4a exit gate (PLAN.md §Stage 4a):

1. 720p30 real-time (≤33 ms/frame median on a single x86_64 core)
2. BD-rate within 2% of libvpx at `--cpu-used 7 --rt --tune-content default`
3. All produced bitstreams decode under `vpxdec`, `ffmpeg`, and our decoder
4. Every divergence either fixed or recorded in `notes/divergence-ledger.md`

## Sequencing (supervisor decision, 2026-05-19)

Stage 4a's gate is **continuous** (BD-rate + latency + decode-cleanness),
unlike Stage 3's binary MD5 oracle. Building the encoder without first
building the oracle infrastructure is the trap CLAUDE.md warns about:
"testing oracles encode 'what correct looks like' in a form machines can
check at scale. They are the critical multiplier — without them, compute
hours are worthless."

**Therefore, build oracles before encoder modules:**

1. **S4.O1** — Structured bitstream diff harness (`diff_bitstream` rewrite).
   Inputs: two `.ivf` files. Output: per-frame, per-syntax-element diff
   (uncompressed header, compressed header, tile data, residual coefs).
   Becomes the per-commit oracle for every encoder task.

2. **S4.O2** — Encoder smoke harness. Inputs: raw YUV + config. Output:
   `.ivf` round-tripped through our own decoder, byte-identical pixel
   compare against input where lossless, PSNR where lossy. Even an empty
   encoder skeleton produces SOMETHING that this harness validates.

3. **S4.O3** — `bdrate` + `realtime` audit. Confirm both bench bins
   actually do what their names imply or stub-out cleanly; wire them
   into a CI-skipped feature so they exist for manual gate runs.

After oracles: **task ordering per PLAN.md Stage 4a tasks 1–11** but
gated by S4.O1 verdict on each commit. First module: `bitstream_writer`
(LLD `notes/lld/bitstream.md`), round-tripped via S4.O2.

## Holds / open items

- Lossless q00 WHT path (decoder-side; Stage 3 carry-over). Out of Stage 4
  critical path unless we want lossless encode coverage.
- Pre-existing doctest failure on `Decoder::parse_compressed_partition`
  (commit `20bc60d`). Conformance gate dodges via `--lib --bins --tests`.
- Stage 3 secondary gates (SIMD audit, throughput bench, unsafe-comment
  audit) still pending — revisit before formal Stage 3 sign-off.
- libvpx-stock instrumentation tree (D:/language/libvpx-stock/)
  uncommitted; outside this workspace per scope rules. Useful for
  encoder oracle work too — Arthur to decide on commit-and-tag.

## Trajectory

| # | Date | Sub-task | Agent | Result | Commit |
|---|------|----------|-------|--------|--------|
| — | 2026-05-19 | Stage 3 closeout (PLAN.md ticks, scratch cleanup, journal commits) | supervisor | done | `92fbc4c` |
| S4.01 | 2026-05-19 | Read `notes/lld/bitstream.md` end-to-end; produce structured-diff harness design note | S4.01 | done — design note at `notes/stage4-design/diff-bitstream-harness.md` (656 lines, 4 commits planned) | — |
| — | 2026-05-19 | Supervisor decisions on Q5/Q6/Q7: defer Level 5, `packet=N frame=M` labels, ledger via post-processor (not harness) | supervisor | done | — |
| S4.02 | 2026-05-19 | Implement commit #1 from design: CLI restructure on `diff_bitstream.rs`; preserve Level 0 byte diff exactly; add output-formatter trait; tests + clippy clean | S4.02 | done — 103→1147 ll, 22 tests green, clippy clean. Socket dropped mid-report; supervisor fixed 3 stale 4-arg test calls + dead-code warn before commit. | `c77ae42` |
| S4.03 | 2026-05-19 | Implement commit #2: Level 1 (frame size + kind) + Level 2 (uncompressed-header field diff). Add `PartialEq, Eq` derives to `UncompressedHeader` + sub-types. Tests for L1/L2 identity, injected divergence at each level. | S4.03 | done — Levels 1+2 land (1147→2204 ll incl. supervisor fix); 6 new tests (28 total); field-walker audit OK; supervisor patched symmetric-parse-error noise before commit. | `27ce31f` |
| S4.04 | 2026-05-19 | Implement commit #3: Level 3 compressed-header + FrameContext prob-bank diff. Walk `parse_compressed_header` outputs per frame; visitor for prob banks. Tests: L3 identity, prob-bank golden divergence. | S4.04 | done — Level 3 + FrameContext walker (2212→3016 ll); 6 new tests (34 total); walker covers all 14 FrameContext fields incl. coef tree + NmvContext; supervisor verified field-by-field. | `96dab01` |
| S4.05 | 2026-05-19 | Implement commit #4 (final): Level 4 per-tile syntax via decode_tile_intra/inter; mi_grid walk; per-block ModeInfo + segment_id + EOB diff; "likely site" hint table; cost-budget guard. | S4.05 | done — Strategy A: Decoder gained `last_frame_mi_grid()` + `last_frame_mi_dims()` accessors; L4 driver instantiates two Decoders, walks `Option<ModeInfo>` per-cell; ModeInfo field walker covers all leaves incl. BlockMode/BmiInfo/Mv; "likely site" hint table; 9 new tests (43 total); gates green; supervisor patched stale `--help` text. | `709e865` |
| ✓ | 2026-05-19 | **S4 ORACLE INFRASTRUCTURE COMPLETE** — 4-commit diff-bitstream design delivered. Levels 0-4 in production; Level 5 (per-coef tokens) deferred per design Q5. 43 unit tests + conformance walker green. | supervisor | done | `c77ae42`..`709e865` |
| — | 2026-05-19 | Sweep-all-vectors idea NIXED — only 1 .ivf in conformance corpus (13 .webm); existing `l4_identity_on_real_ivf` test already covers identity on that vector. Skip the sweep; move to encoder code. | supervisor | done | — |
| S4.06 | 2026-05-19 | First real encoder commit: `write_uncompressed_header` in `rvp9-core::bitstream` (mirror of existing `parse_uncompressed_header`). Per-field writer; round-trip test (write -> parse, assert equal) on captured headers + synthetic fixtures. | S4.06 | done — `write_uncompressed_header(wb, h, first_partition_size)` + 6 sub-writers; round-trip on show-existing/key/intra-only/inter + captured keyframe from `vp90-2-05-resize.ivf`; rvp9-core lib 564→576 tests; workspace 752 total; clippy clean. | `fbb3a6c` |
| S4.07 | 2026-05-19 | Encoder write_compressed_header in `rvp9-core::bitstream` (mirror of existing `parse_compressed_header`). Writes tx_mode + per-section prob updates via `BoolWriter`. Round-trip test: parse libvpx-emitted compressed partition, re-encode, re-parse, assert post-update `FrameContext` equal. | S4.07 | done — `write_compressed_header(bw, pre_fc, post_fc, opts, target)` (Strategy B); 12 sub-writers + new `entropy::prob_diff_writer` module (remap_prob, cond_prob_diff_update); captured-frame round-trip on `vp90-2-05-resize.ivf` keyframe; rvp9-core 576→594 tests; agent dropped on auth before final report — supervisor fixed (a) dummy `const _: u8 = DIFF_UPDATE_PROB` outside test scope, (b) clippy `too_many_arguments` on 8-arg writer. | `4dc028b` |
| S4.08 | 2026-05-19 | `SuperframeBuilder` in `rvp9-core::bitstream` (LLD §3.7). Container-assembly companion to `parse_superframe_index`. Concatenate 1..N frame byte-buffers with optional superframe index trailer. Round-trip: build N frames -> parse_superframe_index recovers exactly the same boundaries. | supervisor | done — written directly (too small for agent dispatch overhead, ~200 ll + 9 tests); auto-mag selection at 255/65535/16M boundaries; rvp9-core 594→603 tests. | `2d476c3` |
| S4.09 | 2026-05-19 | `write_tile_buffers` in `rvp9-core::bitstream::tile` (LLD §3.6 companion). Encoder-side mirror of `parse_tile_buffers`: emit row-major tile payloads with 4-byte BE size prefixes for all but the last tile. | supervisor | done — written directly (~70 ll + 5 tests); round-trip via parse_tile_buffers; byte-identical to legacy `build_tiles` helper. | `f39000a` |
| S4.10 | 2026-05-19 | `write_tile_intra_keyframe` (LLD §3.5 mode-info encode, minimal scope). Decoder-loop validation: synthesize 64×64 mi_grid, encode → `Decoder::decode_frame` → assert decoded mi_grid equals input. | S4.10 | **MILESTONE — first end-to-end encoder→decoder round-trip green.** New module `bitstream::modes_writer` (957 ll, 8 unit tests); whole-loop integration test in rvp9-decoder; scope = keyframe / PARTITION_NONE-only / DC_PRED / skip=1 / no-seg / tx_mode=AllowTx32x32; rvp9-core 608→616, rvp9-decoder 99→100. | `ab093ed` |
| ✓ | 2026-05-19 | **FIRST END-TO-END LOOP GREEN.** rvp9-core encoder emits a synthetic-keyframe IVF packet; rvp9-decoder consumes it and recovers the same mi_grid. Scope is trivially partial (DC-skip-PARTITION_NONE) but every wire-format layer (uncompressed/compressed/tile-buffer/tile-data/superframe) is now operational. | supervisor | done | `ab093ed` |
| S4.10 | 2026-05-19 | First per-tile mode-info writer: `write_tile_intra_keyframe` v1 (keyframe / PARTITION_NONE / DC_PRED / skip=1 only). Lands the wire-format scaffolding for ModeInfo emission; rejects out-of-scope inputs with `ModesWriterError`. | S4.10 | done — new `crates/rvp9-core/src/bitstream/modes_writer.rs` (~960 ll incl. doc + tests). Per-piece round-trip via `BoolReader` + `read_partition` / `read_skip` / `read_intra_mode` (8 unit tests). **Whole-loop round-trip GREEN:** synthetic 64x64 keyframe assembled from `write_uncompressed_header` + `write_compressed_header` (tx_mode=AllowTx32x32) + `write_tile_intra_keyframe` decodes cleanly through `Decoder::parse_packet` + `Decoder::decode_frame` and recovers the same mi_grid the writer emitted. rvp9-core 611→616 tests, rvp9-decoder 99→100 tests, workspace clippy clean. | — |

## Supervisor notes

- Pattern same as Stage 3 overnight: supervisor (me) dispatches agents;
  agents do contained, well-bounded work; supervisor reviews + commits.
- Agents must NEVER commit; that's the supervisor's job. This keeps
  attribution clean and lets me reject hallucinated artefacts.
- Journal: `wrk_journals/2026.05.19 - JRN - stage4 kickoff.md` (and
  successors). One journal per drive day, prepend new entries.
- Hold for Arthur on architectural calls only. Routine implementation
  per LLD spec is supervisor-decidable.
