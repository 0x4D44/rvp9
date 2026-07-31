# RVP9-BUG-KILN-00065 — rvp9enc one-pass creates and header-writes the output file before encoding starts

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9enc
- **Raised:** 2026-07-31T10:22:26Z
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
- **State history:** Open (2026-07-31T10:22:26Z, raised via `deltic bugs new` model=claude-opus-5@high)

## Observation

Split from RVP9-BUG-KILN-00053 during its 2026-07-31 two-eyes verification.

KILN-00053's fix (ab172a3e) correctly converted rvp9enc's one-pass path to true streaming. A side effect nobody recorded: crates/rvp9-cli/src/bin/rvp9enc.rs:909 now constructs the IvfStreamWriter - creating the destination file and writing its 32-byte header - BEFORE the first frame is encoded. Any mid-encode failure now leaves a partial IVF (and clobbers a pre-existing file at that path), where the pre-fix buffered path wrote nothing until the whole sequence succeeded.

This is the exact inverse of what RVP9-BUG-KILN-00051 fixed for rvp9dec, whose accepted remedy was to defer opening the sink until a frame had survived decode. The two shipped binaries now have opposite failure semantics for the same class of event.

Some partial output is inherent to streaming, so the fix is not simply 'revert'. The question to settle is whether rvp9enc should write to a temp path and rename on success, or defer file creation until the first packet is ready, or document the partial-output contract.

Expected: a consistent, documented contract for what rvp9enc leaves behind when an encode fails part-way.
Actual: an undocumented behaviour change that contradicts KILN-00051's principle.

Confirmed by reading the current tree at 4abe0d8 + 3a7aba2.

## Fix

<unfixed — raised only>

## Notes
