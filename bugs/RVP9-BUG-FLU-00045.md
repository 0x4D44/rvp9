# RVP9-BUG-FLU-00045 — Cross-decoder gates accept nonnumeric decoded-size observations

- **State:** Open
- **Priority:** Must
- **Severity:** Medium
- **Area:** qualification/decoded-size
- **Raised:** 2026-10-17T05:18:22Z
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
- **State history:** Open (2026-10-17T05:18:22Z, raised via `deltic bugs new`)

## Observation

Current scripts/qualify-alpha-encoder.sh and qualify-reference-decoder.sh compare actual_bytes using if [ "$actual_bytes" -ne "$expected_bytes" ]; then return1; fi without validating numeric output. Fresh bounded diagnostics extract each exact unchanged condition block, supply actual_bytes=invalid, and both emit integer expression expected followed by ACCEPTED_INVALID_SIZE with exit0. This is a condition-block probe, not a full codec run. Existing preserved unadapted native Windows reverse24 evidence separately records48 integer-expression diagnostics,24 PASS manifest rows and command exit0: GNU stat -f prints filesystem data rather than BSD byte size. Both scripts are unchanged from that observation source. Expected: missing or nonnumeric byte measurement fails acceptance before PASS. Actual: arithmetic test error is false in an if condition, allowing acceptance to proceed. Documented process-local stat compatibility adapters and independent native size/hash checks can establish validity for their specific runs, but do not repair default script failure behavior; the old unadapted result remains explicitly invalid. Preserve numeric validation and portable stat behavior in the owner repair. Fresh probes/raw outputs and hashes in readiness recovered-validation-20261017/invalid-size-evidence.json. No source fix or acceptance weakening performed.

## Fix

<unfixed — raised only>

## Notes
