# RVP9-BUG-KILN-00038 - e1_block_stats IVF reports can use a wrong frame denominator

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** harness/e1-block-stats
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

`e1_block_stats --rvp9-ivf --frames N` can decode the whole IVF file while reporting bitrate against the requested `N`.

In IVF mode, `crates/rvp9-harness/src/bin/e1_block_stats.rs:981` decodes the rvp9 and libvpx IVF inputs before parsing `--frames` at line `985`. `decode_ivf` loops over every packet at line `675`. The report then divides total bits by the caller-provided `n_frames` at line `704`, while the per-frame markdown table prints every decoded rvp9 frame at line `928`.

JSON output later asserts frame counts at line `1033`, but the markdown/stdout path can already contain a bitrate computed with a denominator that does not match the decoded data.

## Fix

Open. Parse `--frames` before decoding and pass the limit into `decode_ivf`, or treat `--frames` as an expected-count assertion and reject/report mismatches before computing bitrate or writing per-frame rows.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 harness review.
