# RVP9-BUG-KILN-00006 - Mid-tile decode error deadlocks the LF-overlap wavefront instead of returning Err (malformed-input liveness/DoS)

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** decoder/tile-threading + loopfilter LF-overlap
- **Raised:** 2026-06-19 by Claude overnight code review
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
- **State history:** 2026-06-19 Opened by Claude from the track2 decoder-fix harvest review pass. 2026-06-21 Fixed by Claude (task/fix-lf-abort): added an abort escape to the LF-overlap wavefront — `ColumnWavefrontLfState.abort`, checked by both wavefront spins in `run_split_edge_lf_worker`, plus an error-path `lf_abort.store(true)` before `handle.wait()` in both decode drivers. Shared root cause and fix with KILN-00012 (the panic/`Drop` manifestation). 2026-06-23 Closed by Codex after independent two-eyes verification on `daaafc1`: pre-fix parent `872f205^` lacks the abort/guard path and joins before returning `merge_err`; current focused tests `cargo test -p rvp9-core lf_worker_bails_promptly_when_aborted`, `cargo test -p rvp9-decoder lf_abort_guard`, and `cargo test -p rvp9-harness lf_overlap_pool_matches_serial` pass; 3/3 skeptic readers found no closure-blocking residual.

## Observation

G-C (`b245c53`) made `decode_inter_partition` / `decode_inter_block` fallible and
threaded `?` up into `decode_tile_inter`'s SB-row loop
(`crates/rvp9-decoder/src/tile.rs:1780`). The per-SB-row completion publish
`decode_progress[sb_row].fetch_add(..)` runs AFTER that `?`
(`crates/rvp9-decoder/src/tile.rs:1785-1793`). When a tile errors mid-frame — e.g.
the new `TileDecodeError::Sub8SegSkip` raised at
`crates/rvp9-decoder/src/tile.rs:2496-2497` on a malformed stream that activates
`SEG_LVL_SKIP` on a sub-8×8 inter block — the `?` short-circuits and the failed
tile's remaining SB-row publishes never run.

On the LF-overlap path the loop-filter worker spins in `spin_then_yield_wavefront`
until `decode_progress[gate_row] >= num_tile_cols`
(`crates/rvp9-core/src/loopfilter/apply.rs:455-462`, `310-321`) with **no
abort/poison escape**, so that threshold is never reached. The decode entry point
then calls `handle.wait()` to join the LF worker
(`crates/rvp9-decoder/src/lib.rs:3570-3571`, also `2191-2193`) **before** it
inspects `merge_err` (`crates/rvp9-decoder/src/lib.rs:3578-3580`), so
`handle.wait()` (a no-timeout `rx.recv()`) blocks forever and the decode API call
deadlocks instead of returning `Err`.

This is the **default** path on capable multithreaded hosts: `lf_overlap_pool`
defaults ON (`crates/rvp9-decoder/src/trace_flags.rs:189`), `row_mt` default
enabled, `lf_pool_size(n)` is `Some` on ≥4-core hosts
(`crates/rvp9-decoder/src/lib.rs:4198-4212`); eligibility further needs
`filter_level > 0`, `tile_rows == 1`, ≥2 tile columns
(`crates/rvp9-decoder/src/lib.rs:3027`), ≥2 SB rows
(`crates/rvp9-decoder/src/lib.rs:3235-3242`).

The deadlock mechanism is **pre-existing in class**: the
`BoolReader::new` → `TileDecodeError::InvalidTilePayload` `?` at
`crates/rvp9-decoder/src/tile.rs:1738` already returns before any publish and hangs
the same LF worker (present at the harvest base `14f58ad`). G-C adds a **new
in-loop fallible exit** that reaches the same deadlock. libvpx does not hang on
corrupt input — it returns `CORRUPT_FRAME` via `vpx_internal_error`/`longjmp` — so
the hang is a divergence, not matched behaviour.

## Notes

Confirmed by the reliability lens; survived adversarial verification 2-of-3 (the
dissenting skeptic agreed the deadlock is real but argued the attribution is
pre-existing and out-of-slice, recommending it be tracked as a standalone threading
defect — which is exactly what this entry does).

**Severity rationale:** a permanent hang (DoS) on attacker-controlled input on the
default multithreaded configuration. The verification panel hovered at Medium
because it does not affect the conformance / BD-rate release gates and the deadlock
class pre-exists; escalated to **High** here because the stated target
(WebRTC / livestream) decodes untrusted bitstreams, and a decoder hanging on
malformed input is a production robustness/DoS defect.

**Fix (the smallest correct one is at the LF-overlap-infra layer, not in G-C):** on
the error path of `decode_tile_inter`, before returning `Err`, release the LF
wavefront consumer — advance `decode_progress` to completion for the failed tile's
remaining SB rows (set each affected counter `>= num_tile_cols`), or add a
poison/abort flag the `spin_then_yield_wavefront` predicate also checks; then
`handle.wait()` returns and the existing `if let Some(e) = merge_err { return
Err(e) }` surfaces the error cleanly. The same fix also closes the pre-existing
`InvalidTilePayload` exit at `tile.rs:1738`. Add a regression test that decodes a
stream which errors mid-tile under the lf_overlap path and asserts the call returns
`Err` rather than hanging.

**Repro:** decode a malformed VP9 inter frame that activates `SEG_LVL_SKIP` on a
sub-8×8 inter block (forces `Sub8SegSkip` at `tile.rs:2496`) with default settings
on a host where `lf_pool_size(n)` is `Some` (≥4 cores), `filter_level > 0`,
`tile_rows == 1`, ≥2 tile columns, ≥2 SB rows. The decode call hangs instead of
returning `DecodeError::TileDecode`. `RVP9_DISABLE_LF_OVERLAP_POOL=1` makes it
return the error correctly, confirming the LF-overlap join is the blocking site.
