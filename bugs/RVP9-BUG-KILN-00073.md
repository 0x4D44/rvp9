# RVP9-BUG-KILN-00073 — noise_sensitivity magnitudes all select medium denoising

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** encoder/denoiser configuration
- **Raised:** 2026-08-16T08:19:31Z
- **Discovery source:** Agent
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
- **State history:** Open (2026-08-16T08:19:31Z, raised via `deltic bugs new`)

## Observation

EncoderConfig documents noise_sensitivity 0 through 6 and says the magnitude maps to the libvpx denoiser level at D:\worktrees\rvp9-2\20260816-REV-RVP9-CDX@KILN-code-review-085355\crates\rvp9-encoder\src\frame_loop.rs:506. StreamEncoder does not reject values above 6, and Denoiser::alloc maps every nonzero value to DenoiserLevel::Medium at denoiser.rs:1093. No later assignment maps the configured magnitude. Values 1 through 6 therefore behave identically despite the public contract, while out-of-range values are silently accepted.

## Fix

Validate noise_sensitivity as 0..=6 when StreamEncoder is constructed, map each supported magnitude to the intended DenoiserLevel, and update the level where libvpx does. Add behavior tests that distinguish at least low and high sensitivity and reject 7.

## Notes
