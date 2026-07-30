# RVP9-BUG-KILN-00018 - Public denoiser helpers trust malformed slices, strides, and dimensions

- **State:** Fixed
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
- **State history:** Open (2026-07-05, raised by Codex code-review pass over `crates/rvp9-encoder/` at `a86e58c260ce`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T194839Z-p44554-n241463000-c1 branch=task/bug-RVP9-BUG-KILN-00018-run-fix-20260730T194839Z-p44554-n241463000-c1 code=2f10e0f3b2e673297842e724cce39159f49ece59 gate=manual)

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
