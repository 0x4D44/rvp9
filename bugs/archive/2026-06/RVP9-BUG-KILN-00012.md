# RVP9-BUG-KILN-00012 - Panic unwinding through the MT decode driver hangs forever in LfBatchHandle::Drop (no-timeout join + no-abort LF spin), swallowing the panic

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** decoder/tile-threading + loopfilter LF-overlap
- **Raised:** 2026-06-20 by Claude overnight code review
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
- **State history:** 2026-06-20 Opened by Claude from the decoder MT / LF-overlap wavefront liveness review pass (originally minted as KILN-00009 on a stale view; **renumbered to KILN-00012** per the ledger README "renumber on conflict" — trunk's KILN-00009 is an unrelated encoder entropy-reset bug minted concurrently on the same host). 2026-06-21 Fixed by Claude (task/fix-lf-abort) together with KILN-00006 (shared root cause). 2026-06-23 Closed by Codex after independent two-eyes verification on `daaafc1`: pre-fix parent `872f205^` lacks the abort/guard path and joins before returning `merge_err`; current focused tests `cargo test -p rvp9-core lf_worker_bails_promptly_when_aborted`, `cargo test -p rvp9-decoder lf_abort_guard`, and `cargo test -p rvp9-harness lf_overlap_pool_matches_serial` pass; 3/3 skeptic readers found no closure-blocking residual.

## Observation

On the **default** multithreaded decode path (`lf_overlap_pool` defaults ON,
`crates/rvp9-decoder/src/trace_flags.rs:189`; same eligibility as KILN-00006), the
decode driver dispatches the single LF-overlap worker (`LfWorkerPool`, N_LF=1) *before*
the decode scope and joins it after via `handle.wait()`. The LF worker spins in
`spin_then_yield_wavefront` (`apply.rs`, no abort) until
`decode_progress[gate_row] >= num_tile_cols`; tiles publish a row's counter only at the
*end* of a fully-decoded SB row, so a tile that exits early leaves rows unpublished and
the worker spinning forever.

KILN-00006 covers the **`Result::Err`** trigger (→ `merge_err` → `handle.wait()`
blocks). **This bug is the distinct `panic`-unwind trigger and a distinct hang site:**

1. A **panic** (not `Result::Err`) in a decode tile worker is collected and
   `resume_unwind`'d by `WorkerPool::scope` *after* its drain, as `scope()` returns
   (`crates/rvp9-core/src/tile_threading/pool.rs`) — so the panic unwinds *out of* the
   `self.worker_pool.scope(...)` call. Control therefore **never reaches** the explicit
   `handle.wait()`.
2. During unwinding the local `lf_overlap_handle` is dropped, invoking
   `LfBatchHandle::Drop` (`crates/rvp9-core/src/tile_threading/lf_pool.rs:349-358`),
   which runs `let _ = rx.recv()` **with no timeout**.
3. The panicking tile never published its remaining `decode_progress` rows, so the LF
   worker spins forever and never sends completion → `Drop` blocks **forever**, stalling
   the unwind and **swallowing the panic**.

Net effect: a panic on the default MT decode path becomes a **silent, permanent hang**
that also hides the panic — and is not reachable via KILN-00006's `Err`/`wait()`
mechanism nor fixable by its error-path-publish remedy alone.

## Notes

Confirmed by the 06.20 overnight review: framed at the infrastructure layer
(`LfBatchHandle::Drop` blocks unconditionally with no timeout + the LF spin has no
abort), it survived 3-of-3 adversarial skeptics with 0 refutations, all rating it
Critical and distinct from KILN-00006. Filed **High** to match KILN-00006's class:
release reachability via the current `decode_tile_intra` is narrow (it is written to
return `Err` on known malformed paths), but debug/test/fuzz reachability is clear
(`debug_assert!`, arithmetic overflow, index-OOB) and hangs the test suite / fuzzer.

**Fix (unified with KILN-00006, task/fix-lf-abort):**
- `ColumnWavefrontLfState` gains `abort: Option<&AtomicBool>`;
  `run_split_edge_lf_worker` checks it in **both** wavefront spins (L1 gate and the
  inner `cur_sb_col` sync-read) and `return`s early when set, before touching frame
  planes. Soundness preserved: abort only makes the worker finish sooner — the driver
  still joins it (temporal-exclusion edge intact), and abort is only set when output is
  discarded.
- The decode driver declares a per-dispatch `lf_abort: AtomicBool`, passes
  `abort: Some(&lf_abort)`, and signals it on **both** abnormal exits: the error path
  stores `true` before `handle.wait()` (KILN-00006), and an RAII guard
  `LfAbortOnPanic(&lf_abort)` — declared *after* the handle so it drops first on unwind —
  sets the flag in `Drop` iff `std::thread::panicking()` (this bug). Either way the
  worker bails, sends completion, and both `wait()` and `Drop`'s `recv` unblock.

**Tests (regression):**
- `crates/rvp9-core/src/loopfilter/apply.rs::tests::lf_worker_bails_promptly_when_aborted`
  — the worker, gated on an unsatisfiable `decode_progress`, must return within a
  deadline once `abort` is set (watchdog via `recv_timeout`; verified red-before-green by
  temporarily disabling the escape → 5 s timeout failure).
- `crates/rvp9-decoder/src/lib.rs::tests::{lf_abort_guard_sets_flag_on_panic_unwind,
  lf_abort_guard_noop_on_normal_drop}` — the panic guard sets the flag while unwinding
  and is inert on a normal drop.

A full end-to-end "decode malformed/​panicking stream without hanging" test is
host-dependent (needs ≥4 cores + a crafted bitstream that forces a tile panic under the
overlap path); the infra + guard unit tests are the deterministic, host-independent
substitute. The inner `cur_sb_col` abort escape is unreachable at N_LF=1 (a single
worker publishes its own `cur_sb_col`) so it is not separately unit-testable today; it
is in place for a uniform abort response and the future N_LF>1 phase.
