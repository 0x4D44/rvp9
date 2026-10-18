# RVP9-BUG-FLUX-00016 — F.0.2 drops rvp9 worker records when threads exceed one

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/f02 oracle
- **Raised:** 2026-08-10T19:10:08Z
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
- **State history:** Open (2026-08-10T19:10:08Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) -> Fixed (2026-10-16T01:48:57Z, deltic:auto role=fix run=fix-20261016T013458Z-c870f86b branch=task/bug-RVP9-BUG-FLUX-00016-run-fix-20261016T013458Z-c870f86b code=8e015526ab40a877c2c6749160d0113c339b623d gate=manual) -> Closed (2026-10-18T17:21:19Z, deltic:manual role=verify run=verify-20261018T171218Z-5e1da460 verified_fix_run=fix-20261016T013458Z-c870f86b verdict=close)

## Observation

On the same 60-frame 1280x704 CPU7 CBR AQ3 old-town input, f02-block-oracle recorded only 16,275 rvp9 decisions with --threads 4 versus 55,215 with --threads 1, while the libvpx leg stayed at about 125,000. The harness advertises matched thread counts and should collect every encoded leaf regardless of worker placement; instead its thread-local rvp9 sink silently undercounts multithreaded runs and makes the differential report invalid. Reproduce with the Unit 1096 commands and compare the reported rvp9 decision counts.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T17:21:19Z, deltic:auto run=verify-20261018T171218Z-5e1da460 verified_fix_run=fix-20261016T013458Z-c870f86b verdict=close)

Verifier note: Manual verify: Fixed tile and compact F.0.2 worker record merging verified: both retained complete-record regressions and three capture/TLS invariants pass on native arm64. Removing each merge independently makes its corresponding pixel-coverage assertion fail. Exact source restored; scope uses generated three-frame fixtures, not the historical 60-frame corpus run.

## Notes
