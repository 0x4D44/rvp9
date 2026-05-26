# Scratchpad — RDO inter picker wiring (speed <= 4)

## Goal
At speed.unsigned_abs() <= 4, route per-leaf inter mode decision through
`rdo::pick_inter_sb::pick_inter_mode_sb` driven by the REAL
`handle_inter_mode_single_ref` (post 2f39eb7). Output flows into the same
emit TX walk (lib.rs:4082+) via the leaf descriptor fields
`(picked_mode, picked_mv, picked_ref_frame, picked_interp_filter)`.

## Key facts
- handle_inter RD is now real (transform-domain Y+chroma) since 2f39eb7.
- pick_inter_mode_sb already iterates {LAST,GOLDEN}x{NEAREST,NEAR,ZERO,NEW}
  + intra fallback, takes a per-ref HandleInterRunner closure +
  RefMvCandidates + IntraFallback.
- single_motion_search is real (diamond + subpel) -> NEWMV + rate_mv.
- Decode safety: writer + emit walk re-derive best_ref_mv from mi_grid,
  reading ONLY (mode, mv, ref, filter). Any in-bounds MV decodes. Same
  invariant the RT path relies on. So coef-source bridge is free.

## Glue (module rdo/inter_dispatch.rs)
1. HandleInterRunner factory per ref: closure (mode, mv, mode_rate, newmv_rate)
   -> handle_inter_mode_single_ref(HandleInterModeParams{...real tables}).
   Real inter_mode_rate computed by caller from inter_costs (not placeholder).
2. single_motion_search per ref -> RefMvCandidates.newmv + newmv_rate.
3. IntraFallback: keep simple/disabled (i64::MAX/2) -> inter always chosen
   for picker-supported leaves; intra-in-inter still reachable only via RT
   fallback path. DOCUMENTED deviation. (Driver's intra needs full intra
   search ~90 LOC; out of minimal scope.)
4. Map PickResult -> (mode, mv, ref, filter). filter from the winning
   FilterRd inside handle_inter result (need to surface best_filter).

## Scope / fallback to RT within speed<=4 path
- Only blocks in {16x16, 32x32} (handle_inter single-tile <=32x32). 8x8 and
  64x64 fall back to RT pick_inter_mode (documented).
- OOB nearest/near vetoed -> substitute ZERO (mirror RT mv_in_bounds veto).
- NEWMV clamped to mv_limits; if single_motion_search returns InvalidMv or
  OOB, drop NEWMV candidate.

## PROOF
- speed=3 vs speed=7 produce DIFFERENT bytes on a real-motion clip AND both
  decode through rvp9-decoder. Plus an invocation counter assert.

## Risks
- best_ref_mv mismatch: RDO uses same mvc as RT (same find_mv_refs on same
  mi_grid) so NEAREST/NEAR/ZERO deltas identical. NEWMV ref-mv = mvc.nearest
  (libvpx single_motion_search ref_mv = ref_mvs[ref][0]). Writer derives the
  same. OK.
