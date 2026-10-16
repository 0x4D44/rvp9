# RVP9-BUG-FLUX-00023 — Decoder output mismatches every frame at the official 226x210 edge size

- **State:** Closed
- **Priority:** Must
- **Severity:** High
- **Area:** decoder/odd frame dimensions
- **Raised:** 2026-08-12T14:15:51Z
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
- **State history:** Open (2026-08-12T14:15:51Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-12T15:06:43Z, deltic:auto role=fix run=fix-20260812T145724Z-p26247-n379818000-c1 branch=task/bug-RVP9-BUG-FLUX-00023-run-fix-20260812T145724Z-p26247-n379818000-c1 code=cf33cd3bf9855892084d1b65dae7c849d6acb4f1 gate=manual) -> Closed (2026-08-12T15:17:44Z, deltic:auto role=verify run=verify-20260812T151212Z-p52381-n813694000-c1 branch=task/bug-RVP9-BUG-FLUX-00023-run-verify-20260812T151212Z-p52381-n813694000-c1 code=cf33cd3bf9855892084d1b65dae7c849d6acb4f1 gate=manual model=codex@xhigh)

## Observation

The rvp9 decoder produces the wrong visible MD5 for all 10 frames of vp90-2-03-size-226x210.webm, while neighboring official size vectors pass. The stream decodes without an error, making this silent wrong output. Reproduced in two independent full 307-vector strict runs on 2026-08-12; pinned vpxdec and ffmpeg decode the SHA-1-verified stream successfully.

## Fix

Commit `cf33cd3bf9855892084d1b65dae7c849d6acb4f1` adds
`build_intra_predictors_with_block_clipping` and passes the parent coding
block's right/bottom edge flags from the decoder. This matches pinned libvpx's
`mb_to_right_edge` / `mb_to_bottom_edge` border-extension decision when a
transform fits inside padded storage but its parent crosses the visible frame.

## Notes

## Verification (2026-08-12T15:17:44Z, independent two-eyes closure)

Verifier: Codex, using run `verify-20260812T151212Z-p52381-n813694000-c1`; the
verifier is not the fixer of this bug.

The focused predictor regression passed on the verifier worktree:
`cargo test -p rvp9-core build_intra_predictors_d63_4x4_parent_block_right_clipped
-- --nocapture` — 1 passed, 0 failed. The exact pre-fix parent `2df4d1ce` was
checked with the same edge assertion through an external one-off harness; it
failed with exit 101 (`[232, 243, 254, 9]` versus expected `[221, 221, 221,
221]`).

The freshly built strict harness passed all adjacent vectors in both dispatch
modes. Native and `RVP9_FORCE_SCALAR=1` each reported 30 passed, 0 mismatched,
0 skipped, 30 total: all ten frames of 226x208, 226x210, and 226x224.

For the formerly failing 226x210 vector, the verifier's `rvp9dec --i420
--threads 1` and pinned libvpx `vpxdec` (libvpx revision
`12f3a2ac603e8f10742105519e0cd03c3b8f71dd`) each emitted 711,900 bytes with
SHA-256 `475f82ab66469cd569fdfb2fe9bd140af099007ce398381ab28156bbe283eee4`;
`cmp` reported `BYTE-EXACT`.
