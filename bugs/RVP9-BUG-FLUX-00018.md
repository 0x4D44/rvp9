# RVP9-BUG-FLUX-00018 — README references a missing fetch-vectors PowerShell script

- **State:** Closed
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
- **State history:** Open (2026-08-12T13:59:40Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-12T16:17:40Z, manual release-preparation fix code=aec4b9fcfc21181a5f15c04efb5bf18a603d4621 model=gpt-5.6-sol@high) -> Closed (2026-08-12T16:36:00Z, independent verification on origin/main code=aec4b9fcfc21181a5f15c04efb5bf18a603d4621 model=gpt-5.6-luna@max)

## Observation

Observation #1, first reproduction: `ls scripts/fetch-vectors*` prints only `scripts/fetch-vectors.sh`, while `README.md:72` instructs Windows users to run `./scripts/fetch-vectors.ps1`. Second reproduction: `git ls-files scripts/fetch-vectors.ps1` returns no path. Expected: the documented bootstrap command names a tracked executable script. Actual: the command fails because the file does not exist.

## Fix

Commit `aec4b9fcfc21181a5f15c04efb5bf18a603d4621` points the README,
packaged attribution files, conformance CLI, and live harness diagnostics at
the tracked `scripts/fetch-vectors.sh`. The README now tells Windows users to
run shell scripts through Git Bash, MSYS2, or WSL instead of naming a missing
PowerShell file.

Focused validation passed:

- `rg -n 'fetch-vectors\.ps1' README.md ATTRIBUTION.md crates --glob '!target/**'`
  returned no matches.
- `cargo test -p rvp9-harness --bin conformance --locked` passed 3/3 tests.
- `cargo fmt --all --check` and `git diff --check` passed.

## Notes

## Verification (2026-08-12T16:36:00Z, independent two-eyes closure)

An independent Luna-max packaging auditor verified the fix on current
`origin/main`; the auditor did not author the change.

- No shipped reference remains under `README.md`, `ATTRIBUTION.md`, `crates/`,
  or `scripts/` to the nonexistent `scripts/fetch-vectors.ps1`.
- `README.md` gives Windows users a workable Git Bash, MSYS2, or WSL route.
  The referenced `scripts/fetch-vectors.sh` exists, is executable, and passes
  `bash -n`.
- `cargo test -p rvp9-harness --test conformance --locked -- --nocapture`
  passed 5 tests; one data-dependent test was correctly ignored because the
  verifier checkout lacked fetched vectors.
- `cargo test -p rvp9-harness --bin conformance --locked -- --nocapture`
  passed 3/3 tests. `cargo fmt --all -- --check` and
  `git diff --check aec4b9fc^..origin/main` also passed.

Historical plans and engineering notes retain old command names as records;
they are not shipped setup instructions and do not reopen this product defect.
