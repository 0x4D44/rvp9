# RVP9-BUG-KILN-00031 - Profile-1 AQ2 non-420 segmentation decode remains mismatched and untracked

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** decoder/profile1 segmentation
- **Raised:** 2026-07-07
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-decoder/` at `ceb57b6`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T163444Z-p94241-n760521000-c1 branch=task/bug-RVP9-BUG-KILN-00031-run-fix-20260730T163444Z-p94241-n760521000-c1 code=4de7766b06b78cc8e7b7ddb70a9d60252ec0cdfd gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`crates/rvp9-harness/tests/profile1_decode.rs:337` documents a real decoder bug in the Profile-1 AQ2 path: cyclic-refresh segmentation activates segment-id-based quantization and the non-420 decode path produces incorrect output. The comments record 9/12 mismatches on `foreman_c444_aq2.ivf` and 8/12 mismatches on `testsrc_c444_720p_aq2.ivf`.

Both follow-up tests remain ignored at `crates/rvp9-harness/tests/profile1_decode.rs:349` and `crates/rvp9-harness/tests/profile1_decode.rs:360`, and they still assert `mismatched > 0` rather than gating the expected zero-mismatch behavior.

`RVP9-BUG-KILN-00007` closed the kfonly Profile-1 ignored-test issue, but its notes explicitly leave the AQ2 portion open/unverified. Because that bug is Closed, the remaining AQ2 decoder mismatch is not in the Open bug queue.

## Fix

Open. Root-cause and fix the non-420 AQ2 segmentation decode mismatch, then convert both AQ2 follow-ups to zero-mismatch gates and move the fixtures into the normal Profile-1 vector set. Keep the bug Open until the ignored tests are either active or replaced by an equivalent default oracle.

## Notes

Confirmed by two skeptic passes in the 2026-07-07 decoder review; the third skeptic agreed the mismatch is real but considered the scratchpad note enough. It is filed because the fix loop drains `bugs/`, not scratchpad entries or Closed bug notes.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

A real root-cause fix, not a test relaxation. libvpx's `read_inter_segment_id` returns the
MINIMUM covered previous ID for reconstruction but calls `copy_segment_id` to preserve every
covered cell in the current map when `update_map == false`. rvp9 was splatting the minimum,
destroying per-cell variation and corrupting inheritance once a later frame used a finer
partition -- exactly the AQ2/cyclic-refresh symptom recorded here. The new branch mirrors libvpx
(`crates/rvp9-decoder/src/tile.rs:2303`, `:2853`); the `update_map == true` path still splats,
also correct.
`profile1_444_aq2_follow_up` and `profile1_720p_aq2_follow_up` had their `#[ignore]` REMOVED and
now assert `mismatched == 0`; both observed passing, as is `profile1_vectors_decode_bit_exact`,
which gates the two fixtures added to `PROFILE1_VECTORS`. `tile::tests::copy_segment_ids_preserves_variation_within_block`
also passes. The helper panics on a missing vector or ledger and asserts
`shown_idx == entries.len()`, so none of these can pass vacuously.
With conformance vectors fetched for this verification, `vec_aq2` also runs and passes.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
