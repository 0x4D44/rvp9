# RVP9-BUG-KILN-00073 — noise_sensitivity magnitudes all select medium denoising

- **State:** Blocked
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
- **State history:** Open (2026-08-16T08:19:31Z, raised via `deltic bugs new`) → Blocked (2026-10-16, Codex; Arthur decision required on temporal-denoiser contract)

## Observation

EncoderConfig documents noise_sensitivity 0 through 6 and says the magnitude maps to the libvpx denoiser level at D:\worktrees\rvp9-2\20260816-REV-RVP9-CDX@KILN-code-review-085355\crates\rvp9-encoder\src\frame_loop.rs:506. StreamEncoder does not reject values above 6, and Denoiser::alloc maps every nonzero value to DenoiserLevel::Medium at denoiser.rs:1093. No later assignment maps the configured magnitude. Values 1 through 6 therefore behave identically despite the public contract, while out-of-range values are silently accepted.

## Fix

Validate noise_sensitivity as 0..=6 when StreamEncoder is constructed, map each supported magnitude to the intended DenoiserLevel, and update the level where libvpx does. Add behavior tests that distinguish at least low and high sensitivity and reject 7.

## Notes

### Contract decision required (2026-10-16)

Missing input: **Arthur** must choose the temporal-denoiser contract before a complete fix can be implemented. Options presented: match pinned libvpx's adaptive temporal strength (recommended); explicitly retain fixed Medium and limit this repair to range/docs; or specify a new magnitude-to-strength mapping. No response or approval has been assumed.

Pinned libvpx v1.14.1 (`12f3a2ac603e8f10742105519e0cd03c3b8f71dd`) initializes every enabled temporal denoiser at Medium. Its setter consumes the **noise estimator's level**, not the configured sensitivity magnitude. The numeric strength switch is compiled only for the separate postprocessing path when temporal denoising is disabled. Consequently, the requested low/high **sensitivity** distinction is not an upstream temporal-denoising oracle. The original observation and proposed fix above are retained as reported, not silently reinterpreted.

The missing `0..=6` API validation is confirmed independently by the pinned interface range check. RVP9's missing adaptive connection is also confirmed, but a direct cast or invented six-to-four mapping would not fix it correctly. Adaptive integration must account for estimate publication, reset/reference lifecycle, estimator gates and the production filter's level checks. Neighboring KILN-00072 remains separate and untouched.

See [diagnosis and source links](../wrk_docs/2026.10.16%20-%20HLD%20-%20Denoiser%20sensitivity%20contract.md). No source change or test run was made for this pass. This record remains unresolved; it is neither Fixed nor Closed. Reopen after Arthur supplies the contract decision.
