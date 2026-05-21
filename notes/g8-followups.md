# G8.1 follow-up catalog

Filed 2026-05-21 as part of the G8.0 commit (harness widening +
`Md5Entry` expansion + superframe-count sanity oracle + corpus
scaffolding). The supervisor split G8 in two per the same pattern G7
adopted: G8.0 lands the engineering deliverable (this commit); G8.1
lands the corpus acquisition + encoding scripts + reference MD5
generation once `vpxenc` is available on PATH.

HLD anchor: `wrk_docs/2026.05.19 - HLD - decoder hardening.md` §G8
(lines 1440-1554).

## What landed in G8.0 (this commit)

- `crates/rvp9-harness/src/md5file.rs` — `Md5Entry` widened to the
  HLD V5/V6 shape: `(packet_index, sub_frame_index, show_frame,
  md5_visible: Option<[u8; 16]>, width, height)`. Backward-
  compatible parse of the libvpx `.md5` format — see module docs.
- `crates/rvp9-harness/tests/conformance.rs` — per-packet sub-frame-
  count sanity oracle. Counts `ParsedFrame`s the decoder produces
  from `Decoder::parse_packet` and compares against an independent
  re-parse of the trailing superframe-index byte
  (`rvp9_core::bitstream::parse_superframe_index`). Hard-fails on
  mismatch with a packet-index-named diagnostic. All 14 AOM
  conformance vectors / 201 frames pass the oracle.
- `crates/rvp9-harness/tests/real_world.rs` — scans
  `tests/corpus/fixtures/` at test time. Skip-cleans when empty
  (G8.0 state); decodes + visible-MD5-compares + emits per-clip
  stats once fixtures land.
- `tests/corpus/fixtures/` — committed empty directory with README
  documenting the &lt; 10 MB curated-subset budget + acceptance
  criteria.
- `tests/corpus/sources/` — gitignored except for the README that
  points at the fetch script.
- `scripts/fetch-corpus.ps1` — PowerShell companion to
  `scripts/fetch-corpus.sh`. Fetches Derf-collection clips into
  `tests/corpus/sources/`. Supports `-WhatIf` dry-run.
- `.gitignore` — updated to keep `tests/corpus/fixtures/` tracked
  while still excluding `tests/corpus/sources/*` (with a README
  exception).

## G8.1 deliverables (deferred — blocked on `vpxenc`)

### Corpus acquisition

`scripts/fetch-corpus.{sh,ps1}` currently seed the Derf subset
(Stage 0 of HLD §G8 source list 1504-1516). G8.1 extends this with:

- The Blender Open-Movies subset (Big Buck Bunny, Sintel, Tears of
  Steel, Cosmos Laundromat — CC-BY licensed).
- Optional self-recorded segments (`ffmpeg -f gdigrab` screen capture
  on Windows; `x11grab` on Linux). Per HLD line 1524 explicitly NOT
  including YouTube extracts or WebRTC captures from Meet/Hangouts.

### Encoding script (`scripts/encode-corpus.ps1`)

Drives `vpxenc` over the fetched `.y4m` sources to produce `.ivf`
fixtures + matching `.md5` ledgers (`vpxdec --md5`):

- A mix of `--rt --cpu-used=4` (realtime presets — exercises the
  fast-path encoder) and `--good --cpu-used=2` (slower presets —
  exercises the BD-rate-tuned encoder).
- Default `--auto-alt-ref=1` to surface hidden altref frames
  (the load-bearing reason for the harness widening this commit
  delivered).
- Trim to short clips (~100-200 frames each) so the in-repo
  curated subset stays under the &lt; 10 MB total budget.

### Reference MD5 generation

`vpxdec --md5 --rawvideo` over each encoded `.ivf` emits the libvpx-
reference `.md5` ledger that `tests/real_world.rs` compares against.
This is the only point in the pipeline where we depend on a vpxdec
build.

### Curated subset selection + commit

Hand-pick 3 clips (one Derf, one Blender, one screen-capture) and
commit the `.ivf` + `.md5` pair into `tests/corpus/fixtures/`.
Acceptance gate (HLD V6 §G8 lines 1545-1546 + 1519-1520):

- Aggregate `show_frame=0` count &gt; 0 (the harness already gates
  on this — see `real_world.rs`).
- Across the **full** corpus (not just the in-repo subset), 30% of
  decoded frames are `show_frame=0` under libvpx's default
  `--auto-alt-ref=1`. The harness emits the per-clip enumeration
  to make this enforceable during G8.1 review.
- Profile breakdown (P0 vs P1) is reported per-clip — establishes
  the P1 surface area is non-zero once Profile 1 fixtures arrive
  alongside G7.1.

### Expected bugs to surface

HLD V6 §G8 line 1549 budgets +days for "1-3 bugs the AOM set
doesn't catch". When the in-repo subset starts running, expect at
least one rough edge in:

- Altref reordering for clips with heavy `--auto-alt-ref` use. The
  sub-frame-count sanity oracle this commit landed is the first
  line of defense.
- Real-content motion patterns that the synthetic AOM corpus
  doesn't probe (e.g. fast pans, scene changes, low-light noise).

## Cross-cutting

- `Md5Entry::md5_hex_visible()` is the migration shim for call
  sites that previously compared `e.md5_hex == got_hex`. It now
  returns `Option<String>` so future hidden-frame rows (a follow-
  up beyond G8.1 if/when we instrument libvpx) are statically
  representable.
- `Md5Entry::index` (the pre-G8 1-based filename-derived index) is
  gone. Call sites that need a 1-based displayed-frame label now
  derive it from `shown_idx + 1`. The libvpx `.md5` format always
  numbers rows sequentially per-displayed-frame so this is
  equivalent.
- The G8 sub-frame-count sanity oracle is layered ON TOP OF the
  existing K-5 `total_other_err` gate — they catch different
  classes of bug. K-5 catches silently-folded decode errors; G8
  catches silently-correct hidden-frame ordering bugs.
