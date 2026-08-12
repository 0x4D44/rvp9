# RVP9-BUG-FLUX-00022 — Decoder cannot decode the pinned spatial-SVC conformance vector

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** decoder/spatial SVC
- **Raised:** 2026-08-12T14:08:04Z
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
- **State history:** Open (2026-08-12T14:08:04Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-12T17:18:45Z, deltic:auto role=fix run=fix-20260812T163429Z-p42548-n323105000-c1 branch=task/bug-RVP9-BUG-FLUX-00022-run-fix-20260812T163429Z-p42548-n323105000-c1 code=d2be2f6dcc935270ea173d043275d178f8872687 gate=manual) -> Closed (2026-08-12T17:25:19Z, independently verified and closed by a two-eyes pass (Codex), code=d2be2f6dcc935270ea173d043275d178f8872687 model=codex@xhigh)

## Observation

The rvp9 decoder mismatches the first three displayed frames of vp90-2-22-svc_1280x720_3.ivf, then rejects packet 2 because a 1280x720 reference is outside the size envelope for a 320x180 frame. This is the pinned libvpx spatial-SVC vector. Reproduced in the 307-vector strict run and an isolated 7-vector rerun on 2026-08-12; spatial SVC encoding remains outside the alpha scope, but decoder conformance is incomplete.

## Fix

Commit `d2be2f6dcc935270ea173d043275d178f8872687` matches libvpx's mixed-reference
rule: `crates/rvp9-decoder/src/lib.rs:2983-3018` retains invalid per-reference
scale factors but rejects the frame only when all listed references are invalid.
`crates/rvp9-decoder/src/tile.rs:2749-2784` rejects a selected invalid scale before
inter prediction, preserving block-selection safety.

The packet-output changes decode every superframe component while exposing only the
final shown component, matching the libvpx output contract used by the official SVC
MD5 ledger.

## Notes

## Verification (2026-08-12T17:25:19Z, independent two-eyes closure)

Verifier: Codex, using verify run
`verify-20260812T172201Z-p80631-n983536000-c1`; the verifier is not the fixer of
this bug.

The focused decoder regressions passed with
`timeout 180 cargo test -p rvp9-decoder decode_inter_frame_ -- --nocapture </dev/null`:
5 passed, 0 failed. This covers mixed-list acceptance, all-invalid rejection,
selected-invalid rejection, and the existing empty-reference cases.

The official vector
`vp90-2-22-svc_1280x720_3.ivf` and its sibling `.md5` ledger passed the strict
one-vector oracle with
`timeout 240 cargo run -p rvp9-harness --bin conformance -- --vectors-dir /tmp/rvp9-svc-verify.O9gCjk --strict --verbose </dev/null`:
20 passed, 0 mismatched, 0 skipped, 20 expected. The harness decoded every
superframe component and compared the final shown 1280x720 component for each
packet. The packet-2 low-layer header lists `[0, 0, 2]` at 320x180; it decoded
successfully despite the unused 1280x720 ALTREF.

The prior pre-fix diagnostic recorded the original packet-2 invalid-reference-size
failure. The focused tests and official vector now pass on the integrated fix commit
`d2be2f6dcc935270ea173d043275d178f8872687`.
