# Design: Structured diff-bitstream harness

**Status:** draft v0.1.0, Stage 4 oracle infrastructure (S4.O1).
**Author:** S4.01.
**Date:** 2026-05-19.
**Scope:** the diff oracle that gates every Stage 4a encoder commit
(`PLAN.md` Stage 4a tasks 1, 11; `notes/stage4-drive.md` §Sequencing).

This document specifies the **structured, syntax-element-aware**
rewrite of `crates/rvp9-harness/src/bin/diff_bitstream.rs`. The
existing 103-line binary (`D:/language/rvp9-2/crates/rvp9-harness/src/bin/diff_bitstream.rs`)
is a Stage 0 byte-level diff and stays callable as Level 0 of the new
harness. Stage 4a's exit gate is *continuous* (BD-rate, latency,
decode-cleanness), not the binary MD5 oracle Stage 3 used; this
harness becomes the per-commit substitute oracle that turns
"non-byte-identical" into an actionable, localized signal.

It is **design-only**. No code is written here. Implementation is
sliced into 4 commits (§9).

---

## 0. Document map

- [§1 Purpose](#1-purpose)
- [§2 CLI surface](#2-cli-surface)
- [§3 Layered diff levels](#3-layered-diff-levels)
- [§4 Reuse strategy](#4-reuse-strategy)
- [§5 Output format](#5-output-format)
- [§6 Test strategy for the harness itself](#6-test-strategy-for-the-harness-itself)
- [§7 Risks and open questions](#7-risks-and-open-questions)
- [§8 Non-goals](#8-non-goals)
- [§9 Implementation slicing](#9-implementation-slicing)
- [§10 References](#10-references)

---

## 1. Purpose

Gate every Stage 4a encoder commit by comparing rvp9's `.ivf`
against libvpx's `.ivf` at the **first syntax level at which they
disagree**. Returns "identical" or the precise frame / level /
syntax-element path of first divergence. The encoder task list
(`PLAN.md` §Stage 4a tasks 1, 11) treats this as the always-green
prerequisite: each encoder change is expected to maintain or
improve the deepest layer reached before divergence.

---

## 2. CLI surface

The binary lives at
`D:/language/rvp9-2/crates/rvp9-harness/src/bin/diff_bitstream.rs`
and the Cargo target name `diff-bitstream` is preserved
(`D:/language/rvp9-2/crates/rvp9-harness/Cargo.toml:23-25`). All
existing invocations keep working; new flags are additive.

### 2.1 Synopsis

```text
diff-bitstream
    --a <PATH>                  Reference IVF (typically libvpx output)
    --b <PATH>                  Candidate IVF (typically rvp9 output)
    [--level N | --level all]   Stop after level N (0..=5); default 4
    [--frame N | --frame N-M]   Restrict to frame index or inclusive range
    [--format text|json]        Output format; default text
    [--verbose summary|first|full]
                                summary   : counts only
                                first     : first divergence per frame (default)
                                full      : every divergence below level N
    [--max-frames N]            Cap frames processed (CI hint)
    [--no-color]                Disable ANSI colour in text output
    [--exit-on-diff]            Exit 1 on first divergence (CI gate)
```

### 2.2 Defaults and conventions

- `--level 4` (per-tile syntax) is the default — the deepest level
  reachable using purely decoder modules (Level 5 costs ~30× more).
- `--verbose first` matches Stage 0 behaviour: report once, fail fast.
- `--exit-on-diff` is **CI mode**: non-zero iff any divergence at or
  above `--level` is found.
- `--a` / `--b` retained from `diff_bitstream.rs:34-35`. By
  convention `--a` is libvpx and `--b` is rvp9; output uses the
  literal labels and CI scripts pin the convention.

---

## 3. Layered diff levels

Each level captures **all** invariants of the level above plus its
own. The harness walks down through the levels until it finds a
divergence; output mentions the highest level at which inputs
disagree. **Lower numbers run cheaper.**

### 3.1 Level 0 — IVF byte stream

Raw byte memcmp of the IVF container; if every byte matches, the
two files are bit-identical and we exit early. Implementation is
the existing `diff_frames` helper (`diff_bitstream.rs:41-66`),
preserving its "first byte / first frame / first byte-in-payload"
output. The 32-byte IVF file header is ignored beyond the `DKIF`
magic check (timestamps and the reserved field differ benignly
between encoders). Cost: O(n) memcmp.

### 3.2 Level 1 — Per-frame size + frame-kind

**Captures:** for every IVF chunk:
- IVF-declared frame size
- Superframe layout (single vs. multi; per-frame sub-sizes)
- `FrameKind` (KeyFrame / IntraOnly / InterFrame / ShowExistingFrame)
- `show_frame` bit
- `frame_context_idx` (which of the 4 banks)

**Skips:** loop-filter / quantization / segmentation / tile-info
sub-fields; compressed header; tile data. **Cost:** ~5 μs/frame.

A distinct level because size-mismatch-with-same-kind vs.
kind-mismatch-with-same-size are different bug classes (compressed
header vs. encoder control flow) and want different triage paths.

### 3.3 Level 2 — Uncompressed-header fields

**Captures:** every field of `rvp9_core::bitstream::UncompressedHeader`
(`D:/language/rvp9-2/crates/rvp9-core/src/bitstream/header.rs:59-99`)
including the `tail: Option<HeaderTail>` sub-fields
(`header.rs:105-126`). Recursively:

- `profile`, `show_frame`, `error_resilient`, `bit_depth`,
  `color_space`, `color_range`, `chroma`, `width`, `height`,
  `render_size`
- `kind` payload (per variant)
- `tail.refresh_frame_context`, `tail.frame_parallel_decoding_mode`,
  `tail.frame_context_idx`
- `tail.loop_filter` (every sub-field of `LoopFilterParams`)
- `tail.quantization` (every sub-field of `QuantizationParams`)
- `tail.segmentation` (every sub-field of `SegmentationParams`)
- `tail.tile_info` (`log2_tile_cols`, `log2_tile_rows`)
- `tail.first_partition_size`

**Skips:** compressed-header prob updates; per-tile bytes.
**Cost:** ~5 μs/frame (same parse output as L1, walked deeper).

Most Stage 4a `bitstream_writer` bugs surface here:
frame-context idx, tile-info unary-code arithmetic, render-size
gating, loop-filter delta update flags. Field-by-field diff makes
these one-shot debug.

**Impl note:** `UncompressedHeader` derives `Clone, Debug` but not
`PartialEq` today (`header.rs:58`). Commit 2 of §9 adds the
derive — zero-cost, local, also benefits the conformance harness.

### 3.4 Level 3 — Compressed-header fields

**Captures:** the two scalars returned by
`parse_compressed_header`
(`D:/language/rvp9-2/crates/rvp9-core/src/bitstream/compressed_header.rs:495`):
- `tx_mode`
- `reference_mode`

Plus the **post-update probability tables** in the
`FrameContext` bank
(`D:/language/rvp9-2/crates/rvp9-core/src/entropy/...` — `FrameContext`
type re-exported as `rvp9_core::entropy::FrameContext`):
- `tx_probs`, `coef_probs`, `skip_probs`
- (inter-only branch) `inter_mode_probs`, `switchable_interp_prob`,
  `intra_inter_prob`, `comp_inter_prob`, `single_ref_prob`,
  `comp_ref_prob`, `y_mode_prob`, `partition_prob`, `nmvc`

**Skips:** per-tile token coding. **Cost:** ~50-200 μs/frame.

Prob-update bugs are insidious: they don't crash, they silently
degrade compression. Post-update-bank parity is sufficient — if
the banks match, the diff-update bit-streams were identical, so we
don't separately compare the encoded update payload.

**Caveat: compound_allowed.** `parse_compressed_header` consumes a
`CompressedHeaderOptions::compound_allowed` bit
(`compressed_header.rs:67-87`). The decoder side currently passes
`false` conservatively for intra-only / keyframe (which is correct)
and falls back to `false` for inter frames in `parse_compressed_partition`
(`D:/language/rvp9-2/crates/rvp9-decoder/src/lib.rs:514-519` — comment
notes this is "conservative"). For the diff harness we need the
**correct** value, computed from the POC sign-bias of the active
refs. **This is open question Q3 (§7).**

### 3.5 Level 4 — Per-tile syntax elements

**Captures:** for every tile in every non-show-existing frame, the
sequence of per-block decode outputs produced by
`decode_tile_intra` / `decode_tile_inter`
(`D:/language/rvp9-2/crates/rvp9-decoder/src/tile.rs:341, 1391`):

- partition tree (recursive splits)
- per-block `ModeInfo` (mode, ref frames, MVs, tx_size, skip)
- per-block segment_id
- per-block coef-EOB (number of non-zero coefs, per plane / per tx)

**Skips:** non-zero coefficient *values* (Level 5).
**Cost:** ~3-10 ms/frame at 720p — a full per-block decode.

Deepest level reachable without re-implementing parsing.
Mode-decision divergences (different partition, MV, intra mode)
all surface here. EOB divergence with matching mode info implies
quantization divergence and feeds into Level 5.

**Impl note:** `decode_tile_intra` / `decode_tile_inter` populate a
`TileDecodeCtx` (`tile_ctx.rs:156-175`) holding the 2D `ModeInfo`
grid via `new_with_grid`. Harness walks two grids side-by-side, one
diff record per cell. Reconstructed pixels are allocated by the
call but ignored. Replaces today's `RVP9_TRACE_*` env-hook workflow
(`tile.rs:55+, lib.rs:99-106`) for the encoder-vs-libvpx case.

### 3.6 Level 5 — Per-block coef-token diff

**Captures:** for every block whose Level 4 mode info matched,
the full coefficient stream emitted by
`decode_block_tokens`
(`D:/language/rvp9-2/crates/rvp9-decoder/src/detokenize.rs:125`).
Each emitted coefficient (value, scan-position) is compared.

**Cost:** ~10-30 ms/frame at 720p; on-demand only (`--level 5`).
Catches quantization-rounding bugs that Level 4 (which sees only
EOB) misses.

**Open question Q5 (§7):** `decode_block_tokens` returns only the
EOB. Per-coef extraction requires either a side-channel sink
(invasive callback in the decoder hot path) or a duplicate parse
loop in the harness (clean but ~2× CPU). Escalated to supervisor.

### 3.7 Level table

| Level | What | Cost / frame (720p) | Reuses |
|------:|---|---|---|
| 0 | IVF byte memcmp | <1 ms | none |
| 1 | Frame size + kind | ~5 μs | `parse_superframe_index`, `parse_uncompressed_header` |
| 2 | Uncompressed-header fields | ~5 μs (same parse) | as L1 + sub-parsers |
| 3 | Compressed-header + post-update probs | ~200 μs | `parse_compressed_header` + `FrameContext` |
| 4 | Per-tile syntax (ModeInfo grid + EOB) | ~3-10 ms | `decode_tile_intra` / `decode_tile_inter` |
| 5 | Per-block coef tokens | ~10-30 ms | `decode_block_tokens` (probably with sink hook) |

---

## 4. Reuse strategy

**Hard rule:** the harness does not re-implement any parser. Every
syntax element it reads comes from a `pub` function in
`rvp9-decoder` or `rvp9-core`. If a divergence comes from rvp9's
parser being wrong, the harness sees the same wrong answer on both
inputs — that bug is owned by the decoder conformance gate
(`crates/rvp9-harness/src/bin/conformance.rs`), not us.

### 4.1 Module call graph

The harness binary lives in `rvp9-harness` (already depends on
`rvp9-core` and `rvp9-decoder` — see
`D:/language/rvp9-2/crates/rvp9-harness/Cargo.toml:34-36`). No new
crate deps. The call chain per level:

```text
read .ivf -> harness::container::read_packets  (existing,
                       D:/language/rvp9-2/crates/rvp9-harness/src/container.rs)

Level 0:  byte memcmp                  (current diff_bitstream.rs:41-66)

Level 1:  rvp9_core::bitstream::parse_superframe_index
                       (crates/rvp9-core/src/bitstream/superframe.rs)
          rvp9_core::bitstream::parse_uncompressed_header
                       (crates/rvp9-core/src/bitstream/header.rs:200+)

Level 2:  same parse output, walk struct fields.

Level 3:  rvp9_core::bitstream::parse_compressed_header
                       (crates/rvp9-core/src/bitstream/compressed_header.rs:495)
          using a fresh FrameContext per side
                       (rvp9_core::entropy::FrameContext::default()).

Level 4:  rvp9_decoder::Decoder::parse_packet
                       (crates/rvp9-decoder/src/lib.rs:365)
          rvp9_decoder::Decoder::parse_compressed_partition (lib.rs:533)
          rvp9_decoder::tile::decode_tile_intra | decode_tile_inter
                       (crates/rvp9-decoder/src/tile.rs:341, 1391)
          Then walk TileDecodeCtx.mi_at(row, col)
                       (crates/rvp9-decoder/src/tile_ctx.rs:156+).

Level 5:  rvp9_decoder::detokenize::decode_block_tokens
                       (crates/rvp9-decoder/src/detokenize.rs:125)
          via a sink hook — see §7 Q5.
```

### 4.2 One-decoder-instance-per-side

The harness instantiates **two** `Decoder`s
(`rvp9_decoder::Decoder::new`, `lib.rs:286`), one per input — full
per-side independence (own `frame_pool`, `ref_frame_map`,
`frame_contexts`). Feeding the same packet sequence into both
exercises the entire reference-frame plumbing, so a Level 4
"different ref slot wrote a different frame two frames ago" bug
surfaces here.

### 4.3 What the harness deliberately does NOT do

- Invoke libvpx. It parses libvpx's *output*, not link against it —
  no third-party tree dependency.
- Reconstruct pixels. That's `dump_frame_diff` territory
  (`D:/language/rvp9-2/crates/rvp9-harness/src/bin/dump_frame_diff.rs`).
- Re-implement parsing. All syntax extraction is via existing `pub`
  functions in `rvp9-decoder` / `rvp9-core` — bugs in those parsers
  are owned by the conformance gate, not us.

---

## 5. Output format

### 5.1 Format selection

- `--format text` (default): human-readable, ANSI-coloured.
- `--format json`: JSONL, one object per frame plus a trailing
  summary object — streams cleanly through `jq` / `grep`.

### 5.2 Text — example, Level 2 divergence

The canonical case to optimise for: the rvp9 encoder emitted the
wrong loop-filter sharpness on frame 3. Output:

```text
diff-bitstream  a=libvpx.ivf (12,847 B)  b=rvp9.ivf (12,851 B)

Level 0  bytes:  diverge at offset 287 (frame payload byte +5)
Level 1  frames: a=42  b=42  kinds match
Level 2  uncompressed-header:
  Frame 3  KeyFrame=false  size=1280x720
    tail.loop_filter.sharpness_level:  a=4   b=5
    tail.loop_filter.filter_level:     a=22  b=22   [OK]
  --- first divergence --- (use --verbose full to see more)

Result: divergence at Level 2, frame 3.
        Lower levels agreed; higher levels not inspected.
Exit:   1
```

### 5.3 Text — example, Level 4 divergence (abbreviated)

```text
Level 0..3  identical
Level 4  per-tile syntax:
  Frame 5  InterFrame  tiles=1
    Tile (0,0)  mi (3, 4)  partition=PARTITION_HORZ
                 a.tx_size=TX_16X16   b.tx_size=TX_8X8
Result: divergence at Level 4, frame 5, tile (0,0), mi (3,4).
        Likely site: tx_size selector in mode-decision.
Exit:   1
```

The "Likely site" hint is generated from a small static map of
"first-diverging field -> probable encoder module owner". Cheap to
maintain and high-value for triage.

### 5.4 JSON — one object per frame, trailing summary

```json
{"frame": 3, "level_0": "diverge_offset:287", "level_1": "ok",
 "level_2": {"first_diverging_field": "tail.loop_filter.sharpness_level",
             "a": 4, "b": 5,
             "context": {"kind":"InterFrame","width":1280,"height":720}},
 "result": "divergence", "first_divergence_level": 2}
```

Trailing summary object encodes `{ "frames_total": N,
"first_divergence_frame": K, ... }`.
`--verbose summary` emits only the summary object.

---

## 6. Test strategy for the harness itself

The harness's correctness is gated by these unit tests, all in
`crates/rvp9-harness/tests/diff_bitstream.rs`. The harness itself
must be **boring**: a parser-output comparator. If it has bugs they
must surface fast.

### 6.1 Identity test

Feed `--a same.ivf --b same.ivf` for a known conformance vector
(reuse a vector from `crates/rvp9-harness/tests/` or the conformance
corpus). All levels report "identical"; exit 0.

This is the "harness is not raising false positives" guard. It also
serves as the smoke test that the parsers used at each level
actually agree with themselves on the same bytes.

### 6.2 Injected-divergence test per level

For each of Levels 1..=4, programmatically construct two `.ivf`
files that differ at exactly that level and assert the harness
reports that level. Construction strategies:

- **Level 1:** truncate frame N in file b. The frame-size check
  fires before parsing.
- **Level 2:** decode an existing `.ivf` to extract header bytes,
  flip one bit in the `sharpness_level` field, re-emit. (Or
  use a synthetic encoder once `bitstream_writer` lands.)
- **Level 3:** harder — needs a hand-crafted compressed header.
  Until `bitstream_writer` lands, use a captured libvpx frame and
  flip a specific update-prob bit. Mark this test `#[ignore]`-until.
- **Level 4:** flip a mode info byte in a tile's bool-coded stream.
  Will cascade through subsequent blocks (the decoder is stateful)
  but the **first** divergence point should still match.

Each test asserts the harness exits with the right level *and* the
right frame number.

### 6.3 Container-parity test

`--a libvpx.ivf --b rvp9-pretending-to-be-libvpx.webm` should not
crash even if container types differ. (Both have to be IVF in v1;
the test asserts a graceful "unsupported container" error rather
than a panic.)

### 6.4 Cost-budget test

A perf regression guard: run all 4 levels on a 60-frame 720p
`.ivf` and assert wall-clock < 5 s. Catches "we accidentally
re-decoded each frame three times" bugs.

### 6.5 Property: lexicographic-monotonic levels

`proptest` property: for any pair of inputs, the level at which
divergence is reported is the lowest level at which any divergence
exists. I.e. if Level 2 reports identical, then Levels 0 and 1 also
reported identical. This is the harness's load-bearing invariant.

### 6.6 Snapshot test for JSON

One golden JSON output per level for a fixed synthetic case, stored
in `crates/rvp9-harness/tests/diff_bitstream_golden/`. Catches
accidental format breakage (we ARE going to want
post-processing scripts on top of this output).

---

## 7. Risks and open questions

Items below are **non-blocking working assumptions** unless
explicitly flagged BLOCKING. Each names the owner.

### 7.1 Working assumptions

- **Q1. `compound_allowed` resolution.** Decoder passes `false`
  conservatively (`lib.rs:514-519`). Harness computes the correct
  value itself via `rvp9_core::inter::mv_ref::sign_bias_from_header`
  (referenced from `lib.rs:53`) — no decoder-internal reach.
- **Q2. `UncompressedHeader: PartialEq + Eq`.** Adding the derive
  is zero-cost; bundled into commit 2 of §9. `HeaderTail` already
  has it (`header.rs:104`); sub-parser leaf types need a quick
  audit during implementation.
- **Q3. FrameContext walk cost.** `FrameContext` is a few KB; full
  diff would emit hundreds of records. Default to first-divergence
  only; `--verbose full` walks the rest. A small visit-driver macro
  keeps path strings ergonomic
  (`fc.coef_probs[TX_8X8][1][0][3][2][1]`).
- **Q4. Container support.** IVF-only in v1. WebM support exists in
  `container.rs:1-27` but the encoder emits IVF; reject WebM with a
  clear error.

### 7.2 Genuine open questions (need supervisor input)

- **Q5. Token-sink hook for Level 5 (BLOCKING for L5 only).**
  `decode_block_tokens` (`detokenize.rs:125`) returns only EOB; per-coef
  extraction needs either (a) an `Option<&mut TokenLog>` parameter
  added to the hot function (invasive, may hurt inlining), (b) a
  duplicate parse loop in the harness (clean, ~2× CPU), or (c)
  defer Level 5 until Stage 4a investigation demands it.
  **Recommend (c).** Awaiting supervisor decision.
- **Q6. Frame numbering.** `--frame N` numbers IVF *packets*; a
  superframe is 1 packet but K syntax frames. Working assumption:
  `--frame N` selects packet N; output labels `packet=N frame=M`
  to disambiguate. Awaiting supervisor concurrence.
- **Q7. Direct divergence-ledger output.** Should `--format ledger`
  emit `notes/divergence-ledger.md`-shaped records? Recommend leave
  to a post-processing script; keeps the harness a pure comparator.

### 7.3 LLD weaknesses noticed during this drive

These are observations about `notes/lld/bitstream.md` itself,
flagged for Arthur:

- **§8.5** says `TxMode` and `ReferenceMode` live in
  `rvp9-core::entropy`. They actually live in
  `rvp9_core::transforms` (`TxMode`) and
  `rvp9_core::bitstream::frame` (`ReferenceMode`). See
  `compressed_header.rs:61` and `bitstream/mod.rs:43-46`. Minor
  drift; harmless but worth a single-line update next time the LLD
  is touched.
- **§3.3 `FrameHeader`** in the LLD is the *encoder-shaped*
  envelope; the *decoder* surface is `UncompressedHeader +
  HeaderTail` (`header.rs:59, 105`). The two are not isomorphic
  in field names. This LLD's §3.3 anticipates the encoder side,
  which is fine; just noting that the diff harness uses the
  decoder's struct (since both inputs are decoded streams).
- **§9.1 unit-test table** lists tests under
  `crates/rvp9-core/src/bitstream/.../{writer,reader}.rs` — those
  files don't exist as of today (the current layout is
  `header.rs`, `compressed_header.rs`, `tile.rs`, etc.). The LLD
  was written before the implementation settled. Diff harness
  doesn't depend on these but the LLD's test-path references will
  go stale unless updated.

None of these block the harness. They are documentation hygiene.

---

## 8. Non-goals

Explicitly outside this design — listed so future contributors
don't try to grow the harness into them:

- **Pixel diff.** That's
  `D:/language/rvp9-2/crates/rvp9-harness/src/bin/dump_frame_diff.rs`
  and the `RVP9_DUMP_ALTREF` path
  (`D:/language/rvp9-2/crates/rvp9-decoder/src/lib.rs:67-93`).
- **BD-rate / PSNR.** That's `rvp9-bench`
  (referenced from `PLAN.md` §Stage 4a task 9).
- **libvpx invocation.** The harness consumes libvpx output. The
  encoder smoke harness (S4.O2 per `notes/stage4-drive.md`) may
  shell out to `vpxdec` / `ffmpeg`; this one does not.
- **Bitstream repair / canonicalization.** If the two inputs use
  legitimately different but equally valid encodings of the same
  picture (e.g. different valid `frame_context_idx` choices), the
  harness reports the divergence rather than masking it. That's a
  feature: divergence should always be visible, even when
  semantically harmless. The divergence-ledger
  (`notes/divergence-ledger.md`) is where "this is OK and we know
  why" rationalisations live.

---

## 9. Implementation slicing

Four commits, each independently mergeable and passing the
existing test suite. Tests for each level land in the same
commit that introduces the level.

### Commit 1 — Restructure CLI; preserve Level 0 byte diff

- Split `diff_bitstream.rs:26-37` into a `clap`- or
  `pico-args`-driven parser (the workspace already uses neither;
  follow the precedent of `conformance.rs` and stay hand-rolled).
- Add `--level`, `--frame`, `--format`, `--verbose`,
  `--exit-on-diff`, `--no-color`, `--max-frames`.
- Wire `Level 0` to existing `diff_frames`
  (`diff_bitstream.rs:41-66`); no behavioural change at level 0.
- Add an output-formatter trait (text vs JSON) with the text
  format identical to today's stdout text.
- Tests:
  - identity (`--a same --b same` exits 0)
  - flag parsing
  - smoke against an existing IVF from
    `crates/rvp9-harness/tests/`

**Risk:** trivial. Just refactor.

### Commit 2 — Level 1 + Level 2; add `PartialEq` to `UncompressedHeader`

- Add `PartialEq, Eq` derives to `UncompressedHeader` and all of
  its sub-types
  (`crates/rvp9-core/src/bitstream/header.rs:58` and
  `bitstream/sub_parsers.rs`).
- Implement Level 1 via `parse_superframe_index` +
  `parse_uncompressed_header`. Diff frame counts, frame kinds,
  superframe shape.
- Implement Level 2 via field-by-field walk of
  `UncompressedHeader`. Emit "first divergence" and "full"
  verbosity outputs.
- Tests:
  - injected-divergence at L1 (truncated frame)
  - injected-divergence at L2 (bit-flip on a captured header)
  - L1/L2 identity on a known IVF
- Resolve Q1 (compound_allowed helper call) so the
  infrastructure used at L3 is already in place.

**Risk:** low. All work is parser-output diff.

### Commit 3 — Level 3 compressed-header + FrameContext diff

- Instantiate two `FrameContext::default()`s, one per side.
- Drive each through `parse_compressed_header` per frame.
- Diff `(tx_mode, reference_mode)` plus the post-update prob
  banks. Walk the `FrameContext` field tree using a small visitor
  macro so paths come out as
  `fc.coef_probs[tx_size=8x8][plane=1][ref=0][band=3][ctx=2][prob=1]`.
- Tests:
  - L3 identity on a known IVF
  - prob-bank diff golden (capture two frame contexts, perturb
    one, assert harness reports the exact field path).
- Resolve Q3 (verbose-level for FrameContext walk).

**Risk:** moderate. The prob-bank walker is a chunk of
boilerplate; macro design needs care to keep path strings
readable.

### Commit 4 — Level 4 per-tile syntax via decode_tile_*

- Wire two `Decoder` instances; feed both sides synchronously.
- Call `Decoder::parse_compressed_partition` + `decode_tile_intra`
  / `decode_tile_inter` per tile per side.
- Walk the `TileDecodeCtx.mi_grid` via existing `mi_at` accessors
  (`tile_ctx.rs:156+`). Diff per-block `ModeInfo`, segment_id,
  EOB.
- Implement "likely site" hint (§5.3) using a static field-name
  -> module-owner table.
- Tests:
  - L4 identity on a known IVF
  - L4 injected-divergence (flip one mode-info byte in a tile;
    assert (frame, tile, mi) reported correctly)
  - cost-budget guard (60-frame 720p < 5 s)
  - property test for level monotonicity (§6.5)
- (Level 5 deferred per Q5 unless supervisor flips the decision.)

**Risk:** highest of the four. Decoding state machine has
subtle ref-pool plumbing; the harness has to drive both `Decoder`s
with identical inputs so internal state stays parallel. Probably
~400 LoC including tests.

### Total estimate

~1100-1500 LoC including tests, across 4 commits. Each commit
runs `cargo test -p rvp9-harness` green. The last commit's
property test gates monotonicity, so a refactor that quietly
breaks "lowest level wins" is caught immediately.

---

## 10. References

| Doc / file | Used for |
|---|---|
| `D:/language/rvp9-2/notes/lld/bitstream.md` | §3-§5 syntax structure; §9 test discipline |
| `D:/language/rvp9-2/notes/stage4-drive.md` | sequencing decision; oracle-before-encoder rule |
| `D:/language/rvp9-2/PLAN.md` lines 528-565 | Stage 4a tasks; exit criteria; what this gate gates |
| `D:/language/rvp9-2/crates/rvp9-harness/src/bin/diff_bitstream.rs` | current Stage 0 binary preserved as Level 0 |
| `D:/language/rvp9-2/crates/rvp9-harness/src/container.rs` | IVF reader the harness reuses |
| `D:/language/rvp9-2/crates/rvp9-core/src/bitstream/header.rs` | `UncompressedHeader`, `HeaderTail`, `FrameKind`, `parse_uncompressed_header` |
| `D:/language/rvp9-2/crates/rvp9-core/src/bitstream/compressed_header.rs` | `parse_compressed_header`, `CompressedHeaderResult`, `CompressedHeaderOptions` |
| `D:/language/rvp9-2/crates/rvp9-core/src/bitstream/superframe.rs` | `parse_superframe_index` |
| `D:/language/rvp9-2/crates/rvp9-core/src/bitstream/tile.rs` | `parse_tile_buffers`, per-tile `TileBuffer` slice |
| `D:/language/rvp9-2/crates/rvp9-decoder/src/lib.rs` | `Decoder`, `parse_packet`, `parse_compressed_partition`, `ParsedFrame` |
| `D:/language/rvp9-2/crates/rvp9-decoder/src/tile.rs` | `decode_tile_intra` / `decode_tile_inter`, `TileBounds`, `FrameDecodeParams` |
| `D:/language/rvp9-2/crates/rvp9-decoder/src/tile_ctx.rs` | `TileDecodeCtx`, ModeInfo grid accessors |
| `D:/language/rvp9-2/crates/rvp9-decoder/src/detokenize.rs` | `decode_block_tokens` (Level 5 sink-hook subject) |
| `D:/language/rvp9-2/notes/divergence-ledger.md` | downstream consumer; harness output feeds entries |

---

*Document version: 0.1.0. S4.01 design output. Companion to the
encoder LLDs that begin landing under §Stage 4a in PLAN.md.*
