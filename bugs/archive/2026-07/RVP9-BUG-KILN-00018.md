# RVP9-BUG-KILN-00018 - Public denoiser helpers trust malformed slices, strides, and dimensions

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** encoder/denoiser public API
- **Raised:** 2026-07-05
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
- **State history:** Open (2026-07-05, raised by Codex code-review pass over `crates/rvp9-encoder/` at `a86e58c260ce`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T194839Z-p44554-n241463000-c1 branch=task/bug-RVP9-BUG-KILN-00018-run-fix-20260730T194839Z-p44554-n241463000-c1 code=2f10e0f3b2e673297842e724cce39159f49ece59 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`crates/rvp9-encoder/src/lib.rs:109` exports `pub mod denoiser`. Public denoiser entry points trust caller-provided inputs:

- `vp9_denoiser_filter` accepts raw slices and strides at `crates/rvp9-encoder/src/denoiser.rs:248`, then indexes `sig`, `mc_avg`, and `avg` directly at `crates/rvp9-encoder/src/denoiser.rs:281`.
- `vp9_denoiser_denoise` accepts caller block coordinates and buffers at `crates/rvp9-encoder/src/denoiser.rs:774`, slices scratch state at `crates/rvp9-encoder/src/denoiser.rs:827`, and copies row windows at `crates/rvp9-encoder/src/denoiser.rs:845`.
- `DenoiserFrame::alloc` multiplies `width * height` and allocates at `crates/rvp9-encoder/src/denoiser.rs:379` without checked arithmetic or caps.

Malformed public inputs can panic the process or request huge allocations.

## Fix

Open. Add checked row-window validation for source, MC, and scratch buffers; validate block coordinates against frame dimensions; make allocation checked and capped to the encoder's supported frame limits; return typed errors instead of panicking.

## Notes

Confirmed by three skeptic passes in the 2026-07-05 encoder review.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

All three sub-observations are covered one-for-one: strided-window validation for
sig/mc_avg/avg (`crates/rvp9-encoder/src/denoiser.rs:368`, `:371-373`), checked block
coordinates plus scratch-state validation before any slicing (`:919`, `:1017`), and `alloc`
capped by `MAX_CODED_DIMENSION`/`MAX_CODED_LUMA_PIXELS` with `try_reserve_exact` (`:508`).
Four new tests, one per error variant, all observed passing:
`denoiser::tests::public_filter_rejects_short_and_invalid_strided_planes`,
`::public_frame_allocator_rejects_oversized_dimensions`,
`::public_denoise_rejects_out_of_frame_block`,
`::public_denoise_rejects_short_source_and_invalid_scratch_state`.
Two residuals noted, neither blocking. (1) The internal caller at `frame_loop.rs:2400` uses
`let Ok(_decision) = ... else { return; }`, so a validation failure silently aborts the frame's
denoise pass mid-way with no diagnostic; believed unreachable in practice but the wrong shape.
(2) The fix widened `sse_diff` to i64 and the bias-to-LAST comparison to u64
(`denoiser.rs:786`, `:836`) where libvpx uses `int`/`unsigned int`. CLAUDE.md requires
deliberate libvpx departures in `notes/divergence-ledger.md`; no entry exists. Values cannot
reach the divergent range for any real block size, so this is a paperwork gap, not a
correctness one -- recorded in `scratchpad.md`.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
