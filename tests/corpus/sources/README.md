# `tests/corpus/sources/` — fetched raw clips (gitignored)

Population target for the G8 real-world corpus source clips
(`.y4m` / `.mkv` etc.), prior to VP9 encoding into
`tests/corpus/fixtures/`.

This directory is **gitignored** (see top-level `.gitignore`); fetch
or re-fetch via the corpus fetch scripts:

- Windows: `scripts/fetch-corpus.ps1` (PowerShell 7+ recommended).
- POSIX:   `scripts/fetch-corpus.sh`.

Per HLD V6 §G8 the source list draws from (in priority order):

1. **Derf's collection** — `media.xiph.org/video/derf/`. Public,
   CC-licensed. Several hundred MB per clip.
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
