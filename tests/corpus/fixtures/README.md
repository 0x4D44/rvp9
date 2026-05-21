# `tests/corpus/fixtures/` — committed VP9 fixtures

Small curated subset of the G8 real-world corpus. Per HLD V6 §G8
(`wrk_docs/2026.05.19 - HLD - decoder hardening.md` lines 1529-1534),
this directory holds a **committed** subset of encoded `.ivf` files
plus matching libvpx `.md5` ledgers so that CI always exercises at
least one real-world clip — no fetch step required.

Total size budget: **&lt; 10 MB** across all committed fixtures.

## Current state — G8.0 (this commit)

**Empty.** The G8 engineering deliverable (Md5Entry widening, the
per-packet sub-frame-count sanity oracle, this directory scaffolding,
the `real_world.rs` test harness, and `scripts/fetch-corpus.ps1`)
landed without binary fixtures because `vpxenc` is not on PATH in
the current sandbox.

`crates/rvp9-harness/tests/real_world.rs` skip-cleans when this
directory is empty; once fixtures arrive the test starts running
automatically without code changes.

## Coming in G8.1 — corpus acquisition + encoding

See `notes/g8-followups.md` for the full deliverable list. Briefly:

1. Encode 3 clips from the G8.1 source list to VP9 IVF via
   `vpxenc` (a mix of `--rt --cpu-used=4` and `--good --cpu-used=2`).
2. Generate the matching `<clip>.ivf.md5` ledger via
   `vpxdec --md5`.
3. Commit both files here. Total ≤ 10 MB.

Suggested initial trio (subject to encoder availability and final
file-size budget):

- A short low-motion clip (Derf collection): ~200 frames at 720p,
  `--good --cpu-used=2`. Exercises the slow-path encoder.
- A short Blender Open-Movie excerpt (Big Buck Bunny or Sintel):
  ~150 frames at 720p, `--rt --cpu-used=4`. Exercises the
  realtime encoder.
- A short screen-capture (self-recorded, no licensing question):
  ~100 frames at 1080p, `--good --cpu-used=2`. Exercises mixed
  high-frequency-text + low-motion content.

All three should produce non-trivial `show_frame=0` counts under
libvpx's default `--auto-alt-ref=1` (~25-30% of frames per HLD line
1520).

## Acceptance criteria

When fixtures land here, `crates/rvp9-harness/tests/real_world.rs`
must:

- Decode each `.ivf` end-to-end via `rvp9_decoder::Decoder`.
- Compare every shown-frame MD5 against the matching `.md5` row.
- Emit per-clip stats (total frames decoded, `show_frame=0` count,
  Profile 0 / Profile 1 split) to stdout.
- Hard-fail on any mismatch.

Aggregate across the in-repo subset, `show_frame=0` count must be
&gt; 0 (HLD line 1545).
