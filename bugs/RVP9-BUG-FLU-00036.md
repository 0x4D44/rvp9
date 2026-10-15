# RVP9-BUG-FLU-00036 — Corpus downloader fails on unavailable Parkrun source

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** scripts/fetch-corpus
- **Raised:** 2026-10-15T22:28:12Z
- **Discovery source:** Agent
- **Owner:** -
- **Owner role:** -
- **Owner run:** -
- **Owner host:** -
- **Owner branch:** -
- **Owner base:** -
- **Owner fingerprint:** -
- **Owner since:** -
- **Owner until:** -
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-10-15T22:28:12Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium)

## Observation

At candidate 1a84f60a24e8d82e8e6454743fbb80521280df0e on macOS arm64, running `deltic timeout 1800 bash scripts/fetch-corpus.sh </dev/null` downloaded the three canonical In To Tree/Ducks/Old Town clips but returned exit 1. Observation: `curl: (22) The requested URL returned error: 404` for `https://media.xiph.org/video/derf/y4m/parkrun_ter_720p50.y4m`; final summary `Corpus: 3 fetched, 0 already present, 1 failed.` A second invocation reproduced the same 404 and exit 1, with the three complete files skipped. Expected: the documented full-corpus setup command succeeds with valid sources, or explicitly documents a currently unavailable fixture. The three acceptance-matrix clips are present; no replacement clip or threshold was substituted. Raw first/retry logs are retained in the final-release-readiness evidence bundle. Fix ownership remains with the separate bug-fixing agent.

## Fix

<unfixed — raised only>

## Notes
