# RVP9-BUG-KIL-00085 — diff-bitstream frame selection skips required state priming

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/diff-bitstream
- **Raised:** 2026-08-22T13:20:52Z
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
- **State history:** Open (2026-08-22T13:20:52Z, raised via `deltic bugs new`)

## Observation

`diff-bitstream --frame N` skips every earlier packet before advancing the
state used by its deep comparisons. Level 3 skips at
`crates/rvp9-harness/src/bin/diff_bitstream.rs:1454-1455`, before parsing the
compressed header and updating its frame-context banks. Level 4 skips at
`crates/rvp9-harness/src/bin/diff_bitstream.rs:1917-1920`, before
`Decoder::parse_packet` and `decode_frame` can populate reference frames and
decoder contexts.

Selecting an interframe therefore analyzes it from fresh state. Symmetric
parse/decode errors are deliberately suppressed, so both legs can fail for the
same missing-history reason and produce a false negative. The selector should
restrict reported frames, not remove the prerequisite history needed to decode
them correctly.

## Fix

<unfixed — raised only>

## Notes

Prime both sides through the selected range start while suppressing diagnostics
for pre-range packets. Add an interframe regression that distinguishes a primed
run from a fresh-decoder run. Confirmed statically; no bitstream was executed.
