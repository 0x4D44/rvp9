# RVP9-BUG-FLUX-00018 — README references a missing fetch-vectors PowerShell script

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** docs/setup
- **Raised:** 2026-08-12T13:59:40Z
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
- **State history:** Open (2026-08-12T13:59:40Z, raised via `deltic bugs new` model=gpt-5.6-sol@high)

## Observation

Observation #1, first reproduction: `ls scripts/fetch-vectors*` prints only `scripts/fetch-vectors.sh`, while `README.md:72` instructs Windows users to run `./scripts/fetch-vectors.ps1`. Second reproduction: `git ls-files scripts/fetch-vectors.ps1` returns no path. Expected: the documented bootstrap command names a tracked executable script. Actual: the command fails because the file does not exist.

## Fix

<unfixed — raised only>

## Notes
