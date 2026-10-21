# RVP9-BUG-FLU-00058 — BD-rate treats IVF timestamp clocks as frame rates and reports false compression gains

- **State:** Fixed
- **Priority:** Must
- **Severity:** High
- **Area:** bench/bdrate
- **Raised:** 2026-10-21T07:01:25Z
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
- **State history:** Open (2026-10-21T07:01:25Z, raised via `deltic bugs new --land` model=gpt-6.1-sol@medium) -> Fixed (2026-10-21T07:23:00Z, deltic:auto role=fix run=fix-20261021T070244Z-d021ee07 branch=task/bug-RVP9-BUG-FLU-00058-run-fix-20261021T070244Z-d021ee07 code=d872b7216cb7c2ca8f23e494b1ade99b202e22e9 gate=manual)

## Observation

Observed twice on exact integrated beta candidate 15bf7162c9fb92a5165fab652460bd42da03a6e6, independently on native macOS and Windows. Verbatim tester observation: "nominal PASS rows show BD-rate≈-97%". In the CPU6/AQ3/CBR 720p30 Tree row, anchor420 is printed as26521.867kbps while the actual60-frame IVF has198914bytes over2seconds (~795.656kbps). The anchor IVF clock is1000/1 with PTS0,33,66,100,...; the candidate clock is30/1 with PTS0,1,2,3,... . Ducks independently reproduces the same33.333x anchor inflation. Both original CSVs and decoded header/PTS/byte/hash evidence are preserved in wrk_docs/evidence/2026-10-21-beta/ivf-clock-defect/. The apparent -97% PASS quality results are invalid and campaigns were stopped.

Confirmed in code: crates/rvp9-bench/src/bin/bdrate.rs::measured_bitrate_kbps divides frame count by PacketReader::framerate(), whose IVF result is the inverse timestamp timebase; this assumes one timestamp tick per frame. A valid IVF timebase need not equal displayed frame cadence. Prior closed KIL-00116 repaired source-Y4M-vs-output FPS but introduced/retained this separate timestamp-clock assumption. Expected: measure each actual encoded role's duration correctly for both frame-tick and millisecond-clock IVF, preserve fractional/effective-output FPS correctness, and reject invalid timing rather than publishing a misleading finite quality result. Add a selected regression that fails with current math, exercise actual pinned libvpx output, and rerun both beta quality campaigns using the repaired tool. Audit other bitrate acceptance paths for the same assumption; no codec compression gain is claimed from these invalid results.

Evidence fingerprint: `manual:v1:bd-rate-treats-ivf-timestamp-clocks-as-frame-ra-d599340cdb1d30c4`


## Fix

<unfixed — raised only>

## Notes
