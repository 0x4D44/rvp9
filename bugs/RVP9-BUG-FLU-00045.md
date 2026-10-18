# RVP9-BUG-FLU-00045 — Cross-decoder gates accept nonnumeric decoded-size observations

- **State:** Open
- **Priority:** Must
- **Severity:** Medium
- **Area:** qualification/decoded-size
- **Raised:** 2026-10-17T05:18:22Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261018T155149Z-6a84ae55
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00045-run-fix-20261018T155149Z-6a84ae55
- **Owner base:** 3f4caaaace9bb3ac52e1bb182a2635bdf27504d2
- **Owner fingerprint:** -
- **Owner since:** 2026-10-18T15:51:49Z
- **Owner until:** 2026-10-18T19:51:49Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-10-17T05:18:22Z, raised via `deltic bugs new`)

## Observation

Current scripts/qualify-alpha-encoder.sh and qualify-reference-decoder.sh compare actual_bytes using if [ "$actual_bytes" -ne "$expected_bytes" ]; then return1; fi without validating numeric output. Fresh bounded diagnostics extract each exact unchanged condition block, supply actual_bytes=invalid, and both emit integer expression expected followed by ACCEPTED_INVALID_SIZE with exit0. This is a condition-block probe, not a full codec run. Existing preserved unadapted native Windows reverse24 evidence separately records48 integer-expression diagnostics,24 PASS manifest rows and command exit0: GNU stat -f prints filesystem data rather than BSD byte size. Both scripts are unchanged from that observation source. Expected: missing or nonnumeric byte measurement fails acceptance before PASS. Actual: arithmetic test error is false in an if condition, allowing acceptance to proceed. Documented process-local stat compatibility adapters and independent native size/hash checks can establish validity for their specific runs, but do not repair default script failure behavior; the old unadapted result remains explicitly invalid. Preserve numeric validation and portable stat behavior in the owner repair. Fresh probes/raw outputs and hashes in readiness recovered-validation-20261017/invalid-size-evidence.json. No source fix or acceptance weakening performed.

## Fix

Both qualifiers now use a shared, validated BSD/GNU stat adapter. They propagate measurement errors before PASS and compare canonical decimal strings, avoiding false acceptance from nonnumeric or oversized arithmetic inputs.

Regression: `python3 scripts/test_qualification_file_size.py` selected the production row checks and failed before the fix on six malformed-measurement acceptance subcases. After the fix, all six tests passed, covering BSD/GNU success, GNU filesystem-text fallback, empty/nonnumeric/failed output, failed stat with numeric stdout, leading zeros, mismatch and oversized values. Bash syntax checks and a native stat smoke with a space-containing filename passed.

Evidence: `wrk_journals/2026.10.18 - JRN - RVP9-BUG-FLU-00045 qualification size validation.md`. This repairs acceptance behavior; it does not revalidate prior invalid codec qualification evidence or claim a new full codec matrix run.

## Notes
