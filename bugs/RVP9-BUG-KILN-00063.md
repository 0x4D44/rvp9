# RVP9-BUG-KILN-00063 — Vector- and corpus-dependent tests still report green having asserted nothing

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** tests/false-green
- **Raised:** 2026-07-31T10:21:56Z
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
- **State history:** Open (2026-07-31T10:21:56Z, raised via `deltic bugs new` model=claude-opus-5@high)

## Observation

Split from RVP9-BUG-FLUX-00002, RVP9-BUG-KILN-00032 and RVP9-BUG-KILN-00033 during their 2026-07-31 two-eyes verification. Each of those bugs was fixed at the site it named, but the same false-green pattern survives one screen away in three places:

1. crates/rvp9-core/src/bitstream/header.rs:1462 (captured_header_round_trip) and crates/rvp9-core/src/bitstream/compressed_header.rs:2456 (write_compressed_header_captured_frame_round_trip) return early on io::ErrorKind::NotFound with NO marker. In a worktree without tests/conformance/vectors both tests report 'ok' having asserted nothing. This was FLUX-00002's chosen remedy, and it is the exact pattern KILN-00019 was raised to eliminate; the harness crate solved the same problem visibly, with a build.rs-driven #[cfg_attr(not(rvp9_conformance_vectors), ignore)]. rvp9-core has no build.rs, which explains but does not justify the divergence.

2. crates/rvp9-harness/tests/real_world.rs:77-84 still returns green with a printed note when zero fixtures are present - one screen above the ledger hardening that fixed KILN-00032.

3. crates/rvp9-harness/tests/conformance.rs:66-71 still silently returns green when the vectors dir is missing. Commit 465a6b52 added the rvp9_conformance_vectors cfg to this very crate but did not apply the ignore attribute to this file.

Expected: a test with no inputs is visibly ignored (or fails), never green.
Actual: three paths report success having compared nothing.

Verified present at trunk 4abe0d8 + 3a7aba2 by reading the current tree.

## Fix

<unfixed — raised only>

## Notes
