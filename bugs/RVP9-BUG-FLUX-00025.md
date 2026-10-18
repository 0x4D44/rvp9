# RVP9-BUG-FLUX-00025 — Buffer-pool golden rejects accepted unrestricted-MV output

- **State:** Fixed
- **Priority:** Must
- **Severity:** Low
- **Area:** encoder/test infrastructure
- **Raised:** 2026-08-14T22:48:39Z
- **Discovery source:** Automation
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T161811Z-3d678f0f
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00025-run-verify-20261018T161811Z-3d678f0f
- **Owner base:** 09fae7017606b27ed3d3b4b0eb9b05253f1500fe
- **Owner fingerprint:** sha256:4c428f7769c71d667ebc60288077bbae619cc547f5fadf145b2905563d68df8b
- **Owner since:** 2026-10-18T16:18:11Z
- **Owner until:** 2026-10-18T20:18:11Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-14T22:48:39Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-14T22:53:16Z, deltic:auto role=fix run=fix-20260814T224910Z-p96012-n942909000-c1 branch=task/bug-RVP9-BUG-FLUX-00025-run-fix-20260814T224910Z-p96012-n942909000-c1 code=9105a73e249aeb3339258a2de5a2c33233c53fca gate=manual)

## Observation

On integrated origin/main 5cd3eb39, cargo test -p rvp9-encoder --release --locked --test w_buffer_pool deterministically fails a10_byte_identity_pre_vs_post_refactor: the stream is 6,446 bytes with FNV-1a 0x20793807d02f6e58, while the test expects 7,078 bytes and 0x3d6d2810a35cd36d. The sibling a10_pool_stream_decodes_cleanly oracle passes. An isolated git bisect proves e5a57f0a98ff25d04e1986177f7383cbce20ef6a, which deliberately enabled libvpx-style unrestricted motion bounds at every v1 resolution, is the first changing commit; the golden should retain the buffer-pool invariant while recording that accepted coding-policy transition.

## Fix

<unfixed — raised only>

## Notes
