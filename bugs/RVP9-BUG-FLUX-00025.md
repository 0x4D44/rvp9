# RVP9-BUG-FLUX-00025 — Buffer-pool golden rejects accepted unrestricted-MV output

- **State:** Closed
- **Priority:** Must
- **Severity:** Low
- **Area:** encoder/test infrastructure
- **Raised:** 2026-08-14T22:48:39Z
- **Discovery source:** Automation
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
- **State history:** Open (2026-08-14T22:48:39Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-14T22:53:16Z, deltic:auto role=fix run=fix-20260814T224910Z-p96012-n942909000-c1 branch=task/bug-RVP9-BUG-FLUX-00025-run-fix-20260814T224910Z-p96012-n942909000-c1 code=9105a73e249aeb3339258a2de5a2c33233c53fca gate=manual) -> Closed (2026-10-18T16:26:01Z, deltic:manual role=verify run=verify-20261018T161811Z-3d678f0f verified_fix_run=fix-20260814T224910Z-p96012-n942909000-c1 verdict=close)

## Observation

On integrated origin/main 5cd3eb39, cargo test -p rvp9-encoder --release --locked --test w_buffer_pool deterministically fails a10_byte_identity_pre_vs_post_refactor: the stream is 6,446 bytes with FNV-1a 0x20793807d02f6e58, while the test expects 7,078 bytes and 0x3d6d2810a35cd36d. The sibling a10_pool_stream_decodes_cleanly oracle passes. An isolated git bisect proves e5a57f0a98ff25d04e1986177f7383cbce20ef6a, which deliberately enabled libvpx-style unrestricted motion bounds at every v1 resolution, is the first changing commit; the golden should retain the buffer-pool invariant while recording that accepted coding-policy transition.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-18T16:26:01Z, deltic:auto run=verify-20261018T161811Z-3d678f0f verified_fix_run=fix-20260814T224910Z-p96012-n942909000-c1 verdict=close)

Verifier note: Manual verify: Independent: original release/locked w_buffer_pool suite 2/2. Reverting only golden to reachable fix parent makes exact byte test fail (6446 vs 7078); restored suite 2/2 at recorded 6446/FNV 0x20793807d02f6e58. Current UMV policy and sibling decode support accepted output. No historical quality/equivalence claim. Source SHA restored. Evidence: .scratch/flux25-proof.txt.

## Notes
