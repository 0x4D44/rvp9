# RVP9-BUG-KILN-00025 - One-axis scaled inter prediction uses a two-axis pre-shift contract

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** core/inter prediction scaling
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-core/` at `af4c7a2d7533`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T161823Z-p11038-n745797000-c1 branch=task/bug-RVP9-BUG-KILN-00025-run-fix-20260730T161823Z-p11038-n745797000-c1 code=f028253c3d297e2b10618da20ca2f051ce3f1a72 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`ScaleFactors::is_scaled()` is true if either axis is non-unit at `crates/rvp9-core/src/inter/scale.rs:121`, and `inter_predictor_with_dispatch` routes every scaled case to the full two-dimensional `convolve8_scaled` path at `crates/rvp9-core/src/inter/scale.rs:312`. The scaled convolution contract says callers pre-shift the source on the filtering axes at `crates/rvp9-core/src/inter/filter.rs:502`, and `convolve8_scaled` assumes the source is already shifted up by three rows at `crates/rvp9-core/src/inter/filter.rs:676`.

The decoder pre-shifts per axis only when that axis needs padding: `row_pre` and `col_pre` are chosen at `crates/rvp9-decoder/src/reconstruct_inter.rs:1105` and applied at `crates/rvp9-decoder/src/reconstruct_inter.rs:1148`. For width-only scaling, `y_pad` is false but the full two-dimensional path still runs vertical filtering; the identity filter has its 128 tap at index 3 at `crates/rvp9-core/src/inter/filter.rs:61`, so an unshifted source samples the wrong row. The scratch path has the same mismatch when it sizes only the unpadded dimension at `crates/rvp9-decoder/src/reconstruct_inter.rs:1194` and then calls the full two-dimensional predictor at `crates/rvp9-decoder/src/reconstruct_inter.rs:1253`.

## Fix

Open. Split one-axis scaled prediction into one-dimensional scaled kernels or make the caller always satisfy the two-dimensional pre-shift and scratch-footprint contract when using `convolve8_scaled`. Add differential regressions for width-only and height-only scaling against libvpx.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 core review.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

The cleanest fix in the batch. The dispatch predicate is now
`x_filtering = subpel_x_q4 != 0 || x_step_q4 != 16` (`crates/rvp9-core/src/inter/scale.rs:306-384`),
which is character-for-character the decoder's own `x_pad`/`y_pad` at
`crates/rvp9-decoder/src/reconstruct_inter.rs:1104-1105` -- so the kernel selected always
matches the per-axis 3-sample pre-shift the caller actually applied. It also reproduces libvpx's
`vp9_setup_scale_factors_for_frame` table exactly.
`inter::scale::tests::inter_predictor_width_only_scale_matches_libvpx_horiz` and
`::inter_predictor_height_only_scale_matches_libvpx_vert_avg` both observed passing; both fail
pre-fix, because the pre-fix 2-D path applies the identity filter's index-3 tap to an unshifted
axis and reads the wrong row/column.
Over-strict-guard check: the only new panic is an `unreachable!` on `(false, false, _)`, which
the outer `if` makes genuinely unreachable.
Residual noted, not blocking: the two new one-axis paths call the scalar
`convolve8_scaled_horiz/_vert` directly rather than through `SimdDispatch` -- a perf gap on
scaled content, not a correctness issue.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
