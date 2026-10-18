# RVP9-BUG-KILN-00073 — noise_sensitivity magnitudes all select medium denoising

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** encoder/denoiser configuration
- **Raised:** 2026-08-16T08:19:31Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261018T175317Z-e1a34adb
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KILN-00073-run-verify-20261018T175317Z-e1a34adb
- **Owner base:** cb5f8c3daa9737e8a4e4835ce2b843f8d712b268
- **Owner fingerprint:** sha256:f1e66d936a6d5d304a1e061805737492e48416ccf257d95bb8ad78c8b071262f
- **Owner since:** 2026-10-18T17:53:17Z
- **Owner until:** 2026-10-18T21:53:17Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-16T08:19:31Z, raised via `deltic bugs new`) → Blocked (2026-10-16, Codex; Arthur decision required on temporal-denoiser contract) → Open (2026-10-17T05:13:55Z, deltic:manual host=flux verdict=reopen) → Fixed (2026-10-17T05:38:09Z, deltic:auto role=fix run=fix-20261017T051408Z-68ec08e1 branch=task/bug-RVP9-BUG-KILN-00073-run-fix-20261017T051408Z-68ec08e1 code=ae19661e3c230b600ad74257f8e3785f195a4ff4 gate=manual)

## Observation

EncoderConfig documents noise_sensitivity 0 through 6 and says the magnitude maps to the libvpx denoiser level at D:\worktrees\rvp9-2\20260816-REV-RVP9-CDX@KILN-code-review-085355\crates\rvp9-encoder\src\frame_loop.rs:506. StreamEncoder does not reject values above 6, and Denoiser::alloc maps every nonzero value to DenoiserLevel::Medium at denoiser.rs:1093. No later assignment maps the configured magnitude. Values 1 through 6 therefore behave identically despite the public contract, while out-of-range values are silently accepted.

## Fix

Arthur-approved implementation (2026-10-17): validate `0..=6`; 0 disables
and 1..=6 enables at Medium initially. Apply newly published noise estimates
to the production denoiser, preserve adaptive state across keyframes, and
mirror pinned estimator gates and raw-source history. LowLow disables
filtering and retains temporal references; re-enabling reseeds all references
from current raw source before filtering resumes. Keyframes use raw-source
seeding. Temporal-only SVC uses the actual reference selectors and refresh
masks. Decoder reconstruction references remain separate.

The adaptive inputs include the temporal minimum dimensions, sampling and
high-motion publication rules, skin exclusions, partial bottom MI rows with
edge replication, and the LAST/small-motion streak producer. These changes
are restricted to the existing supported denoising paths; spatial denoising
remains rejected.

Original proposed remedy, superseded by Arthur's explicit contract decision:

> Validate noise_sensitivity as 0..=6 when StreamEncoder is constructed, map each supported magnitude to the intended DenoiserLevel, and update the level where libvpx does. Add behavior tests that distinguish at least low and high sensitivity and reject 7.

## Notes

### Contract decision required (2026-10-16)

Missing input: **Arthur** must choose the temporal-denoiser contract before a complete fix can be implemented. Options presented: match pinned libvpx's adaptive temporal strength (recommended); explicitly retain fixed Medium and limit this repair to range/docs; or specify a new magnitude-to-strength mapping. No response or approval has been assumed.

Pinned libvpx v1.14.1 (`12f3a2ac603e8f10742105519e0cd03c3b8f71dd`) initializes every enabled temporal denoiser at Medium. Its setter consumes the **noise estimator's level**, not the configured sensitivity magnitude. The numeric strength switch is compiled only for the separate postprocessing path when temporal denoising is disabled. Consequently, the requested low/high **sensitivity** distinction is not an upstream temporal-denoising oracle. The original observation and proposed fix above are retained as reported, not silently reinterpreted.

The missing `0..=6` API validation is confirmed independently by the pinned interface range check. RVP9's missing adaptive connection is also confirmed, but a direct cast or invented six-to-four mapping would not fix it correctly. Adaptive integration must account for estimate publication, reset/reference lifecycle, estimator gates and the production filter's level checks. Neighboring KILN-00072 remains separate and untouched.

See [diagnosis and source links](../wrk_docs/2026.10.16%20-%20HLD%20-%20Denoiser%20sensitivity%20contract.md). No source change or test run was made for this pass. This record remains unresolved; it is neither Fixed nor Closed. Reopen after Arthur supplies the contract decision.

### Reopen note (2026-10-17T05:13:55Z, deltic:manual host=flux verdict=reopen)

Reopen reason: Arthur approved pinned libvpx adaptive temporal denoising: 0 disables, 1-6 enables; estimator controls strength; validate API range and production lifecycle.

### Arthur contract approved (2026-10-17)

Arthur selected pinned libvpx adaptive temporal semantics. Accepted scope:
validate the public 0..=6 range; treat 1..=6 as enable values rather than
strength levels; connect newly published noise estimates to production
filtering and preserve estimator/reference/reset ordering. The original
reported low/high configured-magnitude expectation is superseded by this
explicit decision. No partial range-only fix will be recorded as Fixed.

### Regression evidence (2026-10-17)

Observed assertion-red reversals cover production LowLow filtering, reset
consumption, raw keyframe seeding, mapped temporal references, partial-row
histogram behavior, small-motion streaks, publication wiring, range rejection,
keyframe estimator persistence, skin/high-motion gates, and an invalid
configured-magnitude mapping. The corrected temporal chronology test parses
and decodes real packets; forcing old LAST-only history updates fails on a
non-reference frame changing an untouched slot. Actual decoder output is
compared with encoder reconstruction through adaptive transitions.

The focused denoiser run selected 34 and passed 34, and independent source
review found no remaining blocker. Root independently reran the 34-test denoiser filter, then seven live GOP,
ALTREF and temporal-SVC integration tests; all passed. Four external-oracle
cases were ignored, not counted as passes. Four legacy estimator tests,
formatting and scoped Clippy also passed. Fixing provenance is recorded by
Deltic after landing. No new BD-rate or
performance result is claimed. See the
[implementation and proof journal](../wrk_journals/2026.10.17%20-%20BUG%20-%20Follow%20pinned%20adaptive%20temporal%20denoising.md).
