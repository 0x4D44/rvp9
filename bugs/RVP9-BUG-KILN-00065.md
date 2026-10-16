# RVP9-BUG-KILN-00065 — rvp9enc one-pass creates and header-writes the output file before encoding starts

- **State:** Blocked
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
- **State history:** Open (2026-07-31T10:22:26Z, raised via `deltic bugs new` model=claude-opus-5@high) → Blocked (2026-10-16, Codex; Arthur decision required on one-pass output failure contract)

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

### Output contract decision required (2026-10-16)

Missing input: **Arthur** must choose whether one-pass encoding should defer destination creation until the first packet is ready (recommended, matching the accepted decoder contract), publish a temporary file only after the complete encode succeeds, or retain early creation with an explicit partial-output contract. The question has been presented; no response has been assumed.

Current source still creates the ordinary one-pass writer before `encode_frame_delayed` and the spatial one-pass writer before the first spatial access unit. The ordinary path may buffer source frames without emitting a packet, and it also writes packets from the final flush. A deferred-creation fix must cover those sites; a transactional fix additionally needs replacement, failure cleanup and cross-platform path semantics. These are different observable destination-preservation contracts, not interchangeable implementation details.

The archived KILN-00051 verification explicitly accepts partial decoder output after the first emitted frame. It establishes a useful precedent, but the original observation above explicitly leaves the encoder choice unsettled. No source change, codec execution or regression proof was performed for this decision-only pass. This bug is neither Fixed nor Closed. Reopen after Arthur supplies the contract decision.
