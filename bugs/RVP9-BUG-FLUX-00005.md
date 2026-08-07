# RVP9-BUG-FLUX-00005 — One-tile multithreaded encodes panic without loop-filter worker

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** encoder/loop-filter overlap
- **Raised:** 2026-08-07T10:54:24Z
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
- **State history:** Open (2026-08-07T10:54:24Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh) → Fixed (2026-08-07, `92c340999cf6c71a840445d7908edc856dccf40c`, fixed by Codex)

## Observation

On commit 4a07bca7, run: cargo test -p rvp9-cli --test cli_arg_coverage enc_y4m_input_encodes_successfully --locked -- --exact --nocapture. The spawned rvp9enc process panics at crates/rvp9-encoder/src/frame_loop.rs:3855 with 'multi-threaded encoder must own a loop-filter worker'. A configuration with max_threads > 1 and only one effective tile does not construct frame_worker_pool or lf_worker_pool, but the packet-overlap dispatch checks max_threads alone. Expected: the encoder falls back to the non-overlap loop-filter path and completes without panic.

## Fix

Commit `92c340999cf6c71a840445d7908edc856dccf40c` makes packet/loop-filter
overlap depend on an actually constructed loop-filter worker. One-tile
multithreaded encodes now use the non-overlap path.

Regression evidence:

- Before the fix,
  `cargo test -p rvp9-cli --test cli_arg_coverage enc_y4m_input_encodes_successfully --locked -- --exact --nocapture`
  reproduced the worker-ownership panic.
- After the fix, that exact CLI test passes.
- `cargo test -p rvp9-encoder overlap_requires_a_constructed_worker --locked`
  passes and pins the ownership predicate.

Awaiting independent verification before closure.

## Notes
