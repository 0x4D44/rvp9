# RVP9-BUG-FLU-00052 — Loop-filter safety contracts omit whole-plane exclusivity

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** core/loopfilter safety
- **Raised:** 2026-10-17T20:46:58Z
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
- **State history:** Open (2026-10-17T20:46:58Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium) -> Fixed (2026-10-18T20:03:00Z, deltic:auto role=fix run=fix-20261018T195450Z-79f847db branch=task/bug-RVP9-BUG-FLU-00052-run-fix-20261018T195450Z-79f847db code=8f50c768e2b94ee5b4b68fd5c4e59f7e18e0cf0a gate=manual) -> Closed (2026-10-18T21:49:29Z, deltic:manual role=verify run=verify-20261018T214401Z-9404c950 verified_fix_run=fix-20261018T195450Z-79f847db verdict=close)

## Observation

The public unsafe loop_filter_sb_row and apply_one_sb functions in crates/rvp9-core/src/loopfilter/apply.rs document exclusion only for an SB rectangle and a cross-row write zone. Each then calls SharedFrame::y_slice/u_slice/v_slice, which constructs mutable slices spanning entire planes with from_raw_parts_mut (lines 944 and 996 for these entry points). Two concurrent calls on disjoint SB rectangles can satisfy the written per-rectangle obligations while creating overlapping whole-plane &mut slices, which violates Rust aliasing rules. Expected: each unsafe function documents full-plane exclusivity for the lifetime of reconstructed slices, or returns bounded disjoint plane views. Actual: its documented contract permits disjoint-region concurrency that is incompatible with the full-plane mutable references. The current encoder/decoder production path is serial, so this is an unsafe-contract defect rather than a demonstrated production race. Source-derived; no runtime reproduction was attempted.

## Fix

Documented that `SharedFrame` reconstructs full-plane mutable slices and requires exclusive access to each complete plane. Updated the slice-helper contracts and all three unsafe loop-filter entry points (`loop_filter_sb_row`, `apply_one_sb`, and `run_split_edge_lf_worker`) to state that row/SB progress does not permit overlapping full-plane slices. Corrected the decoder unsafe-call rationale and the cited HLD's contradictory aliasing claim; clarified that the safe tight-plane constructors also provide the raw storage.

Validation: `cargo fmt --all --check`, `cargo check -p rvp9-decoder --lib --locked`, and `cargo doc -p rvp9-core --no-deps --locked` passed. Rustdoc reports 79 warnings, including pre-existing private-item links in `apply.rs`; `git diff --check` passed. This is a contract-only fix, so no runtime regression test was added.

### Verification summary (2026-10-18T21:49:29Z, deltic:auto run=verify-20261018T214401Z-9404c950 verified_fix_run=fix-20261018T195450Z-79f847db verdict=close)

Verifier note: Manual verify: Independent source verification: re-derived complete-plane exclusivity from y[PATH] from_raw_parts_mut lengths, all three unsafe entrypoint contracts, joined decode and single-worker restriction. Current contracts require whole-plane exclusivity; pre-fix content fails each predicate. Correctness checked against implementation, not only text needles. cargo check -p rvp9-decoder --lib --locked and workspace rustdoc passed; fmt and diff checks pass. Documentation-only fix; no runtime behavior/regression claim.

## Notes
