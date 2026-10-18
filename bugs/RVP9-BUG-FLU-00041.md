# RVP9-BUG-FLU-00041 — Spatial quality summary emits unescaped bitrate lists into CSV columns

- **State:** Open
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
- **State history:** Open (2026-10-16T14:08:49Z, raised via `deltic bugs new`)

## Observation

At release readiness spatial quality campaign sourced from 1a84f60a, Python csv.reader finds 13 header columns but 18 fields in every one of 240 data rows of quality/quality-summary.csv. Reproduced by reading the completed spatial-quality run under registered rvp9-final-release-acceptance-49ef1534 scratch; the header assigns layer/bdrate_pct/gate_pct/status after bitrates_kbps, but six comma-separated bitrate values are printed without CSV quoting by scripts/qualify-spatial-svc.sh run_quality_curve. For example the first row places 630 where a header-based parser expects layer and 840 where it expects bdrate_pct. Expected: each row conforms to the 13-column header and preserves the six-rate list as one field. Actual: all data rows have five extra fields. Source remains unchanged at merged candidate ebca64b0. Per-layer raw bdrate.csv and cross-decoder records remain separately available; this report does not claim codec or numeric quality failure. No fix performed; separate bug owner owns repair.

## Fix

<unfixed — raised only>

## Notes
