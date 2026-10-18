# RVP9-BUG-FLU-00041 — Spatial quality summary emits unescaped bitrate lists into CSV columns

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** qualification/spatial-summary
- **Raised:** 2026-10-16T14:08:49Z
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
- **State history:** Open (2026-10-16T14:08:49Z, raised via `deltic bugs new`) -> Fixed (2026-10-18T20:10:15Z, deltic:auto role=fix run=fix-20261018T200522Z-71a33145 branch=task/bug-RVP9-BUG-FLU-00041-run-fix-20261018T200522Z-71a33145 code=c326b78713bc65befbc4392b521e637b8abcee41 gate=manual) -> Closed (2026-10-18T21:44:29Z, deltic:manual role=verify run=verify-20261018T214212Z-dcb96f95 verified_fix_run=fix-20261018T200522Z-71a33145 verdict=close)

## Observation

At release readiness spatial quality campaign sourced from 1a84f60a, Python csv.reader finds 13 header columns but 18 fields in every one of 240 data rows of quality/quality-summary.csv. Reproduced by reading the completed spatial-quality run under registered rvp9-final-release-acceptance-49ef1534 scratch; the header assigns layer/bdrate_pct/gate_pct/status after bitrates_kbps, but six comma-separated bitrate values are printed without CSV quoting by scripts/qualify-spatial-svc.sh run_quality_curve. For example the first row places 630 where a header-based parser expects layer and 840 where it expects bdrate_pct. Expected: each row conforms to the 13-column header and preserves the six-rate list as one field. Actual: all data rows have five extra fields. Source remains unchanged at merged candidate ebca64b0. Per-layer raw bdrate.csv and cross-decoder records remain separately available; this report does not claim codec or numeric quality failure. No fix performed; separate bug owner owns repair.

## Fix

The spatial quality summary now writes the comma-separated bitrate grid as a quoted CSV field. The tested `format_quality_summary_row` helper is shared by `run_quality_curve`, and the script's no-I/O self-test checks a representative row with the six-rate grid intact as one field.

Validation: before the formatter fix, the new self-test failed only the bitrate-field assertion and showed the grid split into extra comma-delimited fields. After the fix, `scripts/qualify-spatial-svc.sh --self-test`, `bash -n scripts/qualify-spatial-svc.sh`, and a Python `csv.reader` check for 13 columns with the full bitrate list in column 9 passed. `git diff --check` passed.

### Verification summary (2026-10-18T21:44:29Z, deltic:auto run=verify-20261018T214212Z-dcb96f95 verified_fix_run=fix-20261018T200522Z-71a33145 verdict=close)

Verifier note: Manual verify: Independent verification at 5b4b3f62: reviewed c326b787 formatter and production run_quality_curve call. Re-ran six-rate original CSV observation through production formatter: csv.reader yields exactly 13 columns and preserves all six rates at index 8. bash scripts/qualify-spatial-svc.sh --self-test and bash -n passed. Reversing only bitrate-field quoting selected the self-test and failed its quality-summary assertion; restored exact source and self-test passed. git diff --check clean. No codec or numeric-quality claim.

## Notes
