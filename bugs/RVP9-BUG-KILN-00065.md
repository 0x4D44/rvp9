# RVP9-BUG-KILN-00065 — rvp9enc one-pass creates and header-writes the output file before encoding starts

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9enc
- **Raised:** 2026-07-31T10:22:26Z
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T164417Z-37f59a3f
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00065-run-verify-20261018T164417Z-37f59a3f
- **Owner base:** 352e2be84ef18b5e5530349ffcec932a9e71430c
- **Owner fingerprint:** sha256:628fcca038d505ba03f35e5a9ee4b816fae3d5c3cac83fb98465629d37180657
- **Owner since:** 2026-10-18T16:44:17Z
- **Owner until:** 2026-10-18T20:44:17Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-31T10:22:26Z, raised via `deltic bugs new` model=claude-opus-5@high) → Blocked (2026-10-16, Codex; Arthur decision required on one-pass output failure contract) → Open (2026-10-17T05:14:39Z, deltic:manual host=flux verdict=reopen) → Fixed (2026-10-17T05:24:45Z, deltic:auto role=fix run=fix-20261017T051448Z-17f1d8d7 branch=task/bug-RVP9-BUG-KILN-00065-run-fix-20261017T051448Z-17f1d8d7 code=b74adaddb222c644d7532ff970b4da5f3133f5ac gate=manual)

## Observation

Split from RVP9-BUG-KILN-00053 during its 2026-07-31 two-eyes verification.

KILN-00053's fix (ab172a3e) correctly converted rvp9enc's one-pass path to true streaming. A side effect nobody recorded: crates/rvp9-cli/src/bin/rvp9enc.rs:909 now constructs the IvfStreamWriter - creating the destination file and writing its 32-byte header - BEFORE the first frame is encoded. Any mid-encode failure now leaves a partial IVF (and clobbers a pre-existing file at that path), where the pre-fix buffered path wrote nothing until the whole sequence succeeded.

This is the exact inverse of what RVP9-BUG-KILN-00051 fixed for rvp9dec, whose accepted remedy was to defer opening the sink until a frame had survived decode. The two shipped binaries now have opposite failure semantics for the same class of event.

Some partial output is inherent to streaming, so the fix is not simply 'revert'. The question to settle is whether rvp9enc should write to a temp path and rename on success, or defer file creation until the first packet is ready, or document the partial-output contract.

Expected: a consistent, documented contract for what rvp9enc leaves behind when an encode fails part-way.
Actual: an undocumented behaviour change that contradicts KILN-00051's principle.

Confirmed by reading the current tree at 4abe0d8 + 3a7aba2.

## Fix

Deferred writer creation until the first ready ordinary, flush or spatial packet. Before that point, errors preserve an existing destination or leave a missing path absent. Later failures may leave partial output. Empty input remains rejected before opening output. PathGuard protections and successful stream finalization remain in use.

## Notes

### Output contract decision required (2026-10-16)

Missing input: **Arthur** must choose whether one-pass encoding should defer destination creation until the first packet is ready (recommended, matching the accepted decoder contract), publish a temporary file only after the complete encode succeeds, or retain early creation with an explicit partial-output contract. The question has been presented; no response has been assumed.

Current source still creates the ordinary one-pass writer before `encode_frame_delayed` and the spatial one-pass writer before the first spatial access unit. The ordinary path may buffer source frames without emitting a packet, and it also writes packets from the final flush. A deferred-creation fix must cover those sites; a transactional fix additionally needs replacement, failure cleanup and cross-platform path semantics. These are different observable destination-preservation contracts, not interchangeable implementation details.

The archived KILN-00051 verification explicitly accepts partial decoder output after the first emitted frame. It establishes a useful precedent, but the original observation above explicitly leaves the encoder choice unsettled. No source change, codec execution or regression proof was performed for this decision-only pass. This bug is neither Fixed nor Closed. Reopen after Arthur supplies the contract decision.

### Reopen note (2026-10-17T05:14:39Z, deltic:manual host=flux verdict=reopen)

Reopen reason: Arthur approved deferring destination creation until the first encoded packet; preserve existing destination on startup failure and accept partial output after first packet, matching decoder contract.

### Arthur contract approved (2026-10-17)

Arthur selected deferred destination creation: open/truncate and write the
IVF header only once the first encoded packet is ready. Before that point,
a failure preserves an existing destination and leaves a missing path absent.
After the first packet, streaming failures may leave partial output, matching
the decoder contract. Cover ordinary, spatial, delayed and flush packet sites.

Regression evidence (2026-10-17): with old eager-open code, both lag2
pre-packet error tests ran and failed their destination-preservation
assertions (the old writer left a 32-byte IVF header). Restored code passed
all 19 CLI roundtrip tests, including ordinary/spatial success and later
partial-output errors, empty input, and first-packet creation during flush.
Omitting flush/spatial writes made their selected positive controls fail.
Input mutants also exercised the empty/error controls; all were restored.
Formatting, scoped CLI Clippy and independent static review passed. See
[task journal](../wrk_journals/2026.10.17%20-%20BUG%20-%20Defer%20output%20until%20first%20encoded%20packet.md).
