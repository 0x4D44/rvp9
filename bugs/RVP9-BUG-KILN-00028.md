# RVP9-BUG-KILN-00028 - parse_packet commits stream metadata before frame decode succeeds

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** decoder/packet parse state
- **Raised:** 2026-07-07
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20260730T175756Z-p1788-n335904000-c1
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00028-run-fix-20260730T175756Z-p1788-n335904000-c1
- **Owner base:** ece9cbf238d0b0a139822139b224e036d5a4162e
- **Owner fingerprint:** -
- **Owner since:** 2026-07-30T17:57:56Z
- **Owner until:** 2026-07-30T19:57:56Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-decoder/` at `ceb57b6`)

## Observation

`Decoder::parse_packet` mutates stream history while it is only parsing headers. It overlays and updates `established_chroma` at `crates/rvp9-decoder/src/lib.rs:1063` and calls `refresh_ref_frame_sizes` at `crates/rvp9-decoder/src/lib.rs:1080`. That refresh updates the size cache for future `setup_frame_size_with_refs` header parsing.

The actual decoded reference-map commits happen later, only after frame decode succeeds: intra/keyframe refresh writes ref slots at `crates/rvp9-decoder/src/lib.rs:2420`, and inter refresh writes ref slots at `crates/rvp9-decoder/src/lib.rs:3876`.

A caller can parse a packet and drop it, or parse frames whose later decode fails, and still advance future header parsing to dimensions or chroma from frames that never landed in the reference map. Superframes make this sharper because all subframes are parsed before any decode commit.

## Fix

Open. Parse packet headers through local shadow copies of `ref_frame_sizes` and `established_chroma`. Commit those deltas only alongside the successful frame reference-map commit, or parse/decode superframes sequentially so parse state cannot outrun decode state.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 decoder review. Track Q already flags parse-time mutable state for future frame-parallel work, but this malformed/error-path state-commit bug was not in the bug ledger.
