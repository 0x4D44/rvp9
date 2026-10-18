# RVP9-BUG-FLU-00036 — Corpus downloader fails on unavailable Parkrun source

- **State:** Closed
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
- **State history:** Open (2026-10-15T22:28:12Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium) -> Fixed (2026-10-18T20:29:22Z, deltic:auto role=fix run=fix-20261018T202454Z-f3cfdc4b branch=task/bug-RVP9-BUG-FLU-00036-run-fix-20261018T202454Z-f3cfdc4b code=4dfc49b0 gate=manual) -> Closed (2026-10-18T21:55:32Z, deltic:manual role=verify run=verify-20261018T214251Z-518f7803 verified_fix_run=fix-20261018T202454Z-f3cfdc4b verdict=close)

## Observation

At candidate 1a84f60a24e8d82e8e6454743fbb80521280df0e on macOS arm64, running `deltic timeout 1800 bash scripts/fetch-corpus.sh </dev/null` downloaded the three canonical In To Tree/Ducks/Old Town clips but returned exit 1. Observation: `curl: (22) The requested URL returned error: 404` for `https://media.xiph.org/video/derf/y4m/parkrun_ter_720p50.y4m`; final summary `Corpus: 3 fetched, 0 already present, 1 failed.` A second invocation reproduced the same 404 and exit 1, with the three complete files skipped. Expected: the documented full-corpus setup command succeeds with valid sources, or explicitly documents a currently unavailable fixture. The three acceptance-matrix clips are present; no replacement clip or threshold was substituted. Raw first/retry logs are retained in the final-release-readiness evidence bundle. Fix ownership remains with the separate bug-fixing agent.

## Fix

Both corpus fetch scripts now map Xiph's current source filename,
`720p50_parkrun_ter.y4m`, to the existing local name
`parkrun_ter_720p50.y4m`. The POSIX script's mocked regression run rejected
the old URL before the change and accepted all four current mappings after
it; PowerShell `-WhatIf` reports the corrected URL. The HLD source note now
documents the remote and local names. No replacement clip or quality
threshold was introduced.

### Verification summary (2026-10-18T21:55:32Z, deltic:auto run=verify-20261018T214251Z-518f7803 verified_fix_run=fix-20261018T202454Z-f3cfdc4b verdict=close)

Verifier note: Manual verify: Independent verification on 5b4b3f62: original deltic timeout 1800 bash scripts/fetch-corpus.sh command completed live downloads of all four canonical clips with Corpus: 4 fetched, 0 already present, 0 failed (exit 0). Parkrun response HTTP200, length696732659, established local filename retained. A mocked four-mapping regression rejects pre-fix script at exact obsolete Parkrun URL (3 fetched/1 failed, exit1), accepts current script (4 fetched/0 failed). PowerShell -WhatIf reports corrected remote/local mapping; bash -n passed. Reviewed both scripts and source note; no substitut...

## Notes
