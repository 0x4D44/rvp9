# RVP9-BUG-KILN-00035 - F.0.2 block oracle never compares the libvpx and rvp9 legs

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** harness/f02-block-oracle
- **Raised:** 2026-07-07
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-harness/` at `cfabd55`)

## Observation

`f02-block-oracle` is described as a paired libvpx-vs-rvp9 oracle, but the implementation does not compare the two legs.

`crates/rvp9-harness/src/bin/f02_block_oracle.rs:5` says the tool is a "libvpx-vs-rvp9 picker-decision oracle", and line `14` advertises raw paired records. The implementation loads or parses the libvpx sidecar at lines `128` and `321`, then only writes per-frame count summaries at line `156` and rvp9 JSONL at line `183`. Normal completion reaches `Ok(())` at line `197` with no pairwise record comparison and no mismatch exit.

The result is that a divergent libvpx oracle leg can be present, parsed, and still never checked against the rvp9 records the tool emits.

## Fix

Open. Align records by frame/block key, compare the fields that define the oracle contract, emit paired diagnostics, and exit nonzero on mismatches unless an explicit report-only mode is selected.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 harness review. No exact existing F.0.2 oracle bug was found.
