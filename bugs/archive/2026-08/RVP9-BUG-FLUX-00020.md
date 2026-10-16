# RVP9-BUG-FLUX-00020 — Decoder rejects three official VP9 tile/skip vectors during entropy decoding

- **State:** Closed
- **Priority:** Must
- **Severity:** High
- **Area:** decoder/tile entropy
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
- **State history:** Open (2026-08-12T14:08:04Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-12T15:39:07Z, deltic:auto role=fix run=fix-20260812T150933Z-p46593-n297689000-c1 branch=task/bug-RVP9-BUG-FLUX-00020-run-fix-20260812T150933Z-p46593-n297689000-c1 code=83bd7566 gate=manual) -> Closed (2026-08-12T15:44:51Z, deltic:auto role=verify run=verify-20260812T154108Z-p87925-n351109000-c1 branch=task/bug-RVP9-BUG-FLUX-00020-run-verify-20260812T154108Z-p87925-n351109000-c1 code=83bd7566ab5adacc206bbc228ba38fcfaf239685 gate=manual model=codex@xhigh)

## Observation

The rvp9 decoder fails vp90-2-08-tile-4x1.webm and vp90-2-08-tile-4x4.webm on packet 1, plus vp90-2-19-skip-02.webm on packet 6, with tile payload ended during entropy decoding. These are SHA-1-verified pinned libvpx vectors with MD5 ledgers. Reproduced in the 307-vector strict run and an isolated 7-vector rerun on 2026-08-12.

## Fix

Commit `83bd7566ab5adacc206bbc228ba38fcfaf239685` preserves the previous
segmentation map across disabled inter frames. It rotates the current map into
`prev_segment_ids` only when segmentation is enabled, matching pinned libvpx's
`vp9_swap_current_and_last_seg_map` lifetime rule. A later enabled frame with
`update_map == false` therefore inherits the last enabled map instead of zeros.

## Notes

## Verification (2026-08-12T15:44:51Z, independent two-eyes closure)

Verifier: Codex, using run `verify-20260812T154108Z-p87925-n351109000-c1`; the
verifier is not the fixer of this bug.

The fetched official vector `vp90-2-19-skip-02.webm` hashes to SHA-1
`178f5bd239e38cc1cc2657a7a5e1a9f52ad2d3fe`; its 12-entry `.md5` ledger was
present and used. The two focused decoder state regressions passed:
`disabled_inter_segmentation_preserves_last_enabled_map` and
`enabled_inter_segmentation_replaces_previous_map` — 2 passed, 0 failed.
The named official regression `cargo test -p rvp9-decoder --test
conformance_coverage vec_skip_02 -- --nocapture` passed — 1 passed, 0 failed.

The exact pre-fix parent `f642e93e5c589eae4310a5509b5750a4473169b4` was built
and checked with the strict harness. The two tile vectors passed, but
`vp90-2-19-skip-02.webm` failed at packet 6 with `tile payload ended during
entropy decoding`; the harness exited 1. This confirms the regression fails
without the fix.

On the fixed tree, native strict conformance over the three recorded failure
vectors (`vp90-2-08-tile-4x1.webm`, `vp90-2-08-tile-4x4.webm`, and
`vp90-2-19-skip-02.webm`) passed 22/22 frames with 0 mismatches and 0 skips.
The same run with `RVP9_FORCE_SCALAR=1` also passed 22/22 with 0 mismatches and
0 skips.

For `vp90-2-19-skip-02.webm`, fixed `rvp9dec --i420 --threads 1` and clean
pinned libvpx `vpxdec` at revision
`12f3a2ac603e8f10742105519e0cd03c3b8f71dd` each emitted 539,136 bytes with
SHA-256 `b20c0cd323aa000969e799025cee6b0db94512415f7589f028c32ddfee55c85a`;
`cmp` reported `BYTE-EXACT`. The forced-scalar rvp9dec output was also 539,136
bytes with the same SHA-256 and matched the native output byte-for-byte.
