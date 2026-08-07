# RVP9-BUG-FLUX-00005 — One-tile multithreaded encodes panic without loop-filter worker

- **State:** Open
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
- **State history:** Open (2026-08-07T10:54:24Z, raised via `deltic bugs new` model=gpt-5.6-sol@xhigh)

## Observation

On commit 4a07bca7, run: cargo test -p rvp9-cli --test cli_arg_coverage enc_y4m_input_encodes_successfully --locked -- --exact --nocapture. The spawned rvp9enc process panics at crates/rvp9-encoder/src/frame_loop.rs:3855 with 'multi-threaded encoder must own a loop-filter worker'. A configuration with max_threads > 1 and only one effective tile does not construct frame_worker_pool or lf_worker_pool, but the packet-overlap dispatch checks max_threads alone. Expected: the encoder falls back to the non-overlap loop-filter path and completes without panic.

## Fix

<unfixed — raised only>

## Notes
