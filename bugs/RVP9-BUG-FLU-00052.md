# RVP9-BUG-FLU-00052 — Loop-filter safety contracts omit whole-plane exclusivity

- **State:** Open
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
- **State history:** Open (2026-10-17T20:46:58Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium)

## Observation

The public unsafe loop_filter_sb_row and apply_one_sb functions in crates/rvp9-core/src/loopfilter/apply.rs document exclusion only for an SB rectangle and a cross-row write zone. Each then calls SharedFrame::y_slice/u_slice/v_slice, which constructs mutable slices spanning entire planes with from_raw_parts_mut (lines 944 and 996 for these entry points). Two concurrent calls on disjoint SB rectangles can satisfy the written per-rectangle obligations while creating overlapping whole-plane &mut slices, which violates Rust aliasing rules. Expected: each unsafe function documents full-plane exclusivity for the lifetime of reconstructed slices, or returns bounded disjoint plane views. Actual: its documented contract permits disjoint-region concurrency that is incompatible with the full-plane mutable references. The current encoder/decoder production path is serial, so this is an unsafe-contract defect rather than a demonstrated production race. Source-derived; no runtime reproduction was attempted.

## Fix

<unfixed — raised only>

## Notes
