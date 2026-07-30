# RVP9-BUG-KILN-00025 - One-axis scaled inter prediction uses a two-axis pre-shift contract

- **State:** Fixed
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-core/` at `af4c7a2d7533`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T161823Z-p11038-n745797000-c1 branch=task/bug-RVP9-BUG-KILN-00025-run-fix-20260730T161823Z-p11038-n745797000-c1 code=f028253c3d297e2b10618da20ca2f051ce3f1a72 gate=manual)

## Observation

`ScaleFactors::is_scaled()` is true if either axis is non-unit at `crates/rvp9-core/src/inter/scale.rs:121`, and `inter_predictor_with_dispatch` routes every scaled case to the full two-dimensional `convolve8_scaled` path at `crates/rvp9-core/src/inter/scale.rs:312`. The scaled convolution contract says callers pre-shift the source on the filtering axes at `crates/rvp9-core/src/inter/filter.rs:502`, and `convolve8_scaled` assumes the source is already shifted up by three rows at `crates/rvp9-core/src/inter/filter.rs:676`.

The decoder pre-shifts per axis only when that axis needs padding: `row_pre` and `col_pre` are chosen at `crates/rvp9-decoder/src/reconstruct_inter.rs:1105` and applied at `crates/rvp9-decoder/src/reconstruct_inter.rs:1148`. For width-only scaling, `y_pad` is false but the full two-dimensional path still runs vertical filtering; the identity filter has its 128 tap at index 3 at `crates/rvp9-core/src/inter/filter.rs:61`, so an unshifted source samples the wrong row. The scratch path has the same mismatch when it sizes only the unpadded dimension at `crates/rvp9-decoder/src/reconstruct_inter.rs:1194` and then calls the full two-dimensional predictor at `crates/rvp9-decoder/src/reconstruct_inter.rs:1253`.

## Fix

Open. Split one-axis scaled prediction into one-dimensional scaled kernels or make the caller always satisfy the two-dimensional pre-shift and scratch-footprint contract when using `convolve8_scaled`. Add differential regressions for width-only and height-only scaling against libvpx.

## Notes

Confirmed by three skeptic passes in the 2026-07-07 core review.
