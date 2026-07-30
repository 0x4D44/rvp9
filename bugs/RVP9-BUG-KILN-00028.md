# RVP9-BUG-KILN-00028 - parse_packet commits stream metadata before frame decode succeeds

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** decoder/packet parse state
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-decoder/` at `ceb57b6`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T175756Z-p1788-n335904000-c1 branch=task/bug-RVP9-BUG-KILN-00028-run-fix-20260730T175756Z-p1788-n335904000-c1 code=35238a7a6bec9655f766f3d3a74818c03037a5d4 gate=manual)

## Observation

`Decoder::parse_packet` mutates stream history while it is only parsing headers. It overlays and updates `established_chroma` at `crates/rvp9-decoder/src/lib.rs:1063` and calls `refresh_ref_frame_sizes` at `crates/rvp9-decoder/src/lib.rs:1080`. That refresh updates the size cache for future `setup_frame_size_with_refs` header parsing.

The actual decoded reference-map commits happen later, only after frame decode succeeds: intra/keyframe refresh writes ref slots at `crates/rvp9-decoder/src/lib.rs:2420`, and inter refresh writes ref slots at `crates/rvp9-decoder/src/lib.rs:3876`.

A caller can parse a packet and drop it, or parse frames whose later decode fails, and still advance future header parsing to dimensions or chroma from frames that never landed in the reference map. Superframes make this sharper because all subframes are parsed before any decode commit.

## Fix

Open. Parse packet headers through local shadow copies of `ref_frame_sizes` and `established_chroma`. Commit those deltas only alongside the successful frame reference-map commit, or parse/decode superframes sequentially so parse state cannot outrun decode state.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 decoder review. Track Q already flags parse-time mutable state for future frame-parallel work, but this malformed/error-path state-commit bug was not in the bug ledger.
