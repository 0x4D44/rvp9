# RVP9-BUG-KIL-00085 — diff-bitstream frame selection skips required state priming

- **State:** Closed
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
- **State history:** Open (2026-08-22T13:20:52Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T02:50:46Z, deltic:auto role=fix run=fix-20261016T023434Z-82979771 branch=task/bug-RVP9-BUG-KIL-00085-run-fix-20261016T023434Z-82979771 code=4aab47c03c0575b314f6fadcd1c3e290f96239b2 gate=manual) -> Closed (2026-10-18T17:46:44Z, deltic:manual role=verify run=verify-20261018T174326Z-83f3398a verified_fix_run=fix-20261016T023434Z-82979771 verdict=close)

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

### Verification summary (2026-10-18T17:46:44Z, deltic:auto run=verify-20261018T174326Z-83f3398a verified_fix_run=fix-20261016T023434Z-82979771 verdict=close)

Verifier note: Manual verify: Three frame-range priming tests and encoder-fixture calibration passed; removing L3 probability replay and L4 decoder replay each failed the retained packet-1 history assertion. Exact source restored and all four pass. Synthetic L3 and self-encoded 64x64 L4 fixture; no external corpus/libvpx run.

## Notes

Prime both sides through the selected range start while suppressing diagnostics
for pre-range packets. Add an interframe regression that distinguishes a primed
run from a fresh-decoder run. Confirmed statically; no bitstream was executed.
