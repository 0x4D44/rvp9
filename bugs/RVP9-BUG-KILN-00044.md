# RVP9-BUG-KILN-00044 - WebM reader drops laced blocks and ignores track identity

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** io/webm packet reader
- **Raised:** 2026-07-07
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260730T173030Z-p2642-n978179000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00044-run-fix-20260730T173030Z-p2642-n978179000-c1
- **Owner base:** 39ead8ee70ec643c4500f4ef6014d3b38d4c0653
- **Owner fingerprint:** -
- **Owner since:** 2026-07-30T17:30:30Z
- **Owner until:** 2026-07-30T19:30:30Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-io/` at `6128772`)

## Observation

The WebM reader does not reliably return the VP9 video packet stream it advertises.

`crates/rvp9-io/src/container.rs:315`-`317` says the parser does not validate codec ID, track number, or metadata, while `read_webm_packets` documents returned ranges as video-track blocks at lines `380`-`381`. The walker accepts every `SimpleBlock` or `Block` inside a `Cluster` at lines `442`-`453`. `parse_block_frame` reads the track number into `_track` at line `498` and discards it, so audio, subtitle, or non-VP9 blocks can be handed to the VP9 decoder as packets.

Unsupported lacing goes the other direction and silently drops data. Lines `510`-`514` return `Ok(None)` for any non-zero lacing mode, and `crates/rvp9-io/tests/error_paths.rs:648`-`665` pins that as a no-error/no-packet outcome. `rvp9dec` reads the packet list at `crates/rvp9-cli/src/bin/rvp9dec.rs:399` and feeds each returned range into `parse_packet` at lines `432`-`436`, so both the CLI and harness users inherit the wrong packet set.

## Fix

Open. Parse `Tracks`/`CodecID` enough to identify the VP9 video track and filter to it, or document and enforce a single-track VP9-only subset. Reject laced blocks as unsupported unless lacing is implemented.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 IO review. This is adjacent to harness skip-accounting bugs, but not a duplicate: the parser itself can return the wrong packet stream.
