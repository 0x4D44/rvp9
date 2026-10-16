# RVP9-BUG-KILN-00038 - e1_block_stats IVF reports can use a wrong frame denominator

- **State:** Closed
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-harness/` at `cfabd55`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T193024Z-p23523-n700722000-c1 branch=task/bug-RVP9-BUG-KILN-00038-run-fix-20260730T193024Z-p23523-n700722000-c1 code=a0db45c3762859996ffd2db7c9ae267f35c3e59d gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`e1_block_stats --rvp9-ivf --frames N` can decode the whole IVF file while reporting bitrate against the requested `N`.

In IVF mode, `crates/rvp9-harness/src/bin/e1_block_stats.rs:981` decodes the rvp9 and libvpx IVF inputs before parsing `--frames` at line `985`. `decode_ivf` loops over every packet at line `675`. The report then divides total bits by the caller-provided `n_frames` at line `704`, while the per-frame markdown table prints every decoded rvp9 frame at line `928`.

JSON output later asserts frame counts at line `1033`, but the markdown/stdout path can already contain a bitrate computed with a denominator that does not match the decoded data.

## Fix

Open. Parse `--frames` before decoding and pass the limit into `decode_ivf`, or treat `--frames` as an expected-count assertion and reject/report mismatches before computing bitrate or writing per-frame rows.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 harness review.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Root cause was the denominator being taken from `--frames` without checking it against the
decoded stream. The check now runs at `crates/rvp9-harness/src/bin/e1_block_stats.rs:1117`,
BEFORE `emit_report` (`:1135`), the markdown write (`:1140`) and the JSON path -- so the exact
reported outcome, a bitrate computed on a mismatched denominator, can no longer be produced.
This is the second option this record sanctioned (treat `--frames` as an expected-count
assertion).
`tests::report_frame_count_rejects_requested_count_mismatch`,
`::report_frame_count_rejects_paired_stream_mismatch` and
`::report_frame_count_accepts_matching_contract` all observed passing. Not debug-only: a plain
`Result` plus an abort in all profiles.
Residual noted, not blocking: the three tests call `report_frame_count` directly; none drives
`run_ivf_mode`, so the ordering at `:1117` is verified by reading, not by test.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
