# `tests/corpus/sources/` — fetched raw clips (gitignored)

Population target for the G8 real-world corpus source clips
(`.y4m` / `.mkv` etc.), prior to VP9 encoding into
`tests/corpus/fixtures/`.

This directory is **gitignored** (see top-level `.gitignore`); fetch
or re-fetch via the corpus fetch scripts:

- Windows: `scripts/fetch-corpus.ps1` (PowerShell 7+ recommended).
- POSIX:   `scripts/fetch-corpus.sh`.

Per HLD V6 §G8 the source list draws from (in priority order):

1. **Derf's collection** — `media.xiph.org/video/derf/`. Publicly
   downloadable; licence terms vary by clip and many state none, so
   check before redistributing. Several hundred MB per clip.
2. **Blender Open Movies** — Big Buck Bunny, Sintel, Tears of Steel,
   Cosmos Laundromat. CC-BY.
3. **Self-recorded** — screen captures (ffmpeg `gdigrab` /
   `x11grab`) and optional locally-captured WebRTC sessions.
   No third-party ToS concerns.

**Out of scope (explicitly excluded — see HLD lines 1524-1527):**

- No YouTube extracts.
- No WebRTC captures from Meet / Hangouts.

## Current state — G8.0 (this commit)

The G8 engineering deliverable landed without acquiring binaries
(no `vpxenc` on PATH in the current sandbox). See
`notes/g8-followups.md` for the G8.1 deliverable that closes this
gap.

## Workflow when G8.1 lands

```sh
# 1. Fetch raw sources (gitignored landing zone).
pwsh ./scripts/fetch-corpus.ps1
#   or:
./scripts/fetch-corpus.sh

# 2. Encode to VP9 IVF + emit matching libvpx .md5 (G8.1).
pwsh ./scripts/encode-corpus.ps1

# 3. Hand-pick the small curated subset and copy into
#    tests/corpus/fixtures/ (committed; <10 MB total budget).
```

## Compression-efficiency test clips

The Stage 4a/4b BD-rate and AQ.5 exit-gate harnesses
(`crates/rvp9-encoder/tests/bdrate_gate.rs`) probe for these y4m
sources at canonical paths. They are gitignored — fetch manually:

### `tests/corpus/foreman_cif.y4m`

**Format**: YUV4MPEG2 W352 H288 F30000:1001 Ip A128:117 (C420)  
**Size**: ~46 MB (300 frames)  
**Why**: Real-motion content (head turn, panning camera, construction
site detail) — the testsrc 720p clip is mostly flat colour bars and
saturates at ~75 kbps regardless of target bitrate. Foreman exposes
meaningful DCT/MV-search/MC working sets across the rate-distortion
curve. Classic Hannover H.263 test clip.

**Acquisition** (manual; download or sibling-tree copy):

```sh
# Option 1: Xiph derf collection.
curl -L -o tests/corpus/foreman_cif.y4m \
    https://media.xiph.org/video/derf/y4m/foreman_cif.y4m

# Option 2: copy from the legacy rvp9 sibling-tree fixture cache
# (auto-probed by `rvp9_harness::y4m::probe_foreman_cif`).
cp D:/language/rvp9/crates/rvp9/tests/data/cache/foreman_cif.y4m \
   tests/corpus/foreman_cif.y4m
```

The harness skips the real-content tests cleanly if the file is
absent — no panic, just an `[skip]` line on stderr.
