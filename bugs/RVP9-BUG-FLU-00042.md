# RVP9-BUG-FLU-00042 — BD-rate cubic solver loses accuracy on narrow realistic PSNR curves

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** harness/bdrate
- **Raised:** 2026-10-16T14:10:24Z
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
- **State history:** Open (2026-10-16T14:10:24Z, raised via `deltic bugs new`)

## Observation

On Apple Silicon at candidate ebca64b0, a standalone rustc diagnostic includes the unchanged production crates/rvp9-harness/src/bdrate.rs and calls its public bdrate function with six points on each curve. Anchor tuples (kbps,PSNR): [(1152.068,25.670945),(1174.084,25.723668),(1207.220,25.767877),(1253.696,25.822174),(1326.328,25.884390),(1569.352,26.307338)]. Test tuples: [(1151.356,25.697093),(1175.428,25.747508),(1207.796,25.794290),(1251.228,25.841255),(1326.504,25.908804),(1568.004,26.298732)]. Actual current Rust result -0.714401618768 percent. Python standard-library Decimal80 degree-three least-squares in the same original PSNR basis, using ln bitrate and the same exact overlap integral, gives -1.788791640314064 percent; centered Decimal80 agrees to over60 decimal places. Same-input error exceeds1.074 percentage points, material against a0.5percent release ceiling. The six-point data come from spatial CPU8/S2 1080p30 Ducks SL0. The full campaign recorded -3.383557percent using internal unrounded observations; do not equate that result to the rounded-input reproducer. Diagnostic all240 rounded layer curves still pass respective0.5/25percent ceilings; no codec quality failure or threshold crossing is claimed. Source polyfit3 builds unscaled normal equations from raw PSNR powers through6; current standalone benchmark has another raw-basis solver worth inspection. KIL-00107 singular-fit rejection is separate and already landed. Expected: trustworthy cubic-fit measurements or explicit rejection of ill-conditioned observations. Actual: finite inaccurate values accepted. Reproducer and Decimal script preserved under readiness evidence spatial-quality. No fix or acceptance workaround performed; bug owner must repair and qualify affected numerical tooling before expensive quality campaigns.

## Fix

<unfixed — raised only>

## Notes
