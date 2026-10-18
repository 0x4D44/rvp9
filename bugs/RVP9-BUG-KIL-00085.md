# RVP9-BUG-KIL-00085 — diff-bitstream frame selection skips required state priming

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/diff-bitstream
- **Raised:** 2026-08-22T13:20:52Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T174326Z-83f3398a
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00085-run-verify-20261018T174326Z-83f3398a
- **Owner base:** 773503314d360c96f69746c032949076fbd91f42
- **Owner fingerprint:** sha256:e217f17c54b1717b0bd180e89028cc3a84ba9715abf23e6c57f64f7d5b2eb8ad
- **Owner since:** 2026-10-18T17:43:26Z
- **Owner until:** 2026-10-18T21:43:26Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T13:20:52Z, raised via `deltic bugs new`) -> Fixed (2026-10-16T02:50:46Z, deltic:auto role=fix run=fix-20261016T023434Z-82979771 branch=task/bug-RVP9-BUG-KIL-00085-run-fix-20261016T023434Z-82979771 code=4aab47c03c0575b314f6fadcd1c3e290f96239b2 gate=manual)

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
