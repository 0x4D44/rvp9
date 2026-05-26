# W-compound — {ALTREF,LAST} compound inter prediction drive

## Verified facts (from reading code 2026-05-26)

- Compound avg rounding: `convolve_avg` etc. do `(dst + pred + 1) >> 1`
  (`crates/rvp9-core/src/inter/filter.rs:188` round_avg, libvpx vpx_convolve.c:59).
- Decoder recon order (`reconstruct_inter.rs:331`): `for r in 0..num_refs`
  reads `mvs[r]` + `refs[r]`; `is_compound_avg = r > 0`. So r=0 writes dst,
  r=1 averages. ref order = `ref_frames[r]`.
- `setup_compound_reference_mode([false,false,false,true])` =>
  comp_fixed=ALTREF, comp_var=[LAST, GOLDEN]. (last==golden branch.)
- For pair {ALTREF,LAST}: idx = sign_bias[ALTREF]=1 => ref_frames[1]=ALTREF,
  ref_frames[0]=LAST=comp_var_ref[0] => comp_ref bit = 0.
- **mi-grid cell: ref_frames=[LAST, ALTREF], mvs=[last_mv, altref_mv].**
- Core writer `write_inter_block_mode_info` (mode_info.rs:987) is FULLY
  compound-ready: derives is_compound from ref_frames[1].is_inter(), iterates
  num_refs MV residuals. No core changes needed.
- Gate: `altref_picker_live = enable_alt_ref && altref_slot_emitted`
  (frame_loop.rs:1843). Same gate used for ALTREF single-ref.
  When false: sign_bias=[false,false,false], reference_mode=Single, NO compound.

## Plan

1. inter_dispatch.rs: add compound {ALTREF,LAST} candidate. New fn
   `pick_inter_sb_leaf` extension: build compound RD via averaging two preds.
   Add counter RDO_DISPATCH_COMPOUND_PICKS. Return second ref+mv.
2. InterDispatchResult: add `ref_frame2: Option<RefFrame>` + `mv2: Mv`.
3. InterPartitionLeafDesc (lib.rs:2249): add `picked_ref_frame2` + `picked_mv2`.
4. Recon path (lib.rs:4632): when compound, build LAST pred then ALTREF avg pred.
5. mi-grid builder (frame_loop.rs:2894) + splat (lib.rs:2191): emit
   ref_frames=[LAST,ALTREF]+two MVs when compound.
6. Frame header (frame_loop.rs:3233): reference_mode=Select when altref live
   & compound enabled; compound_allowed=true.
7. PROOF test tests/w_compound_proof.rs (mirror w_altref_picker.rs).

## Scope decision
Joint-MV search: SCOPE DOWN to independent single-ref MVs reused for the pair
(search LAST + ALTREF separately, form compound pred from those two MVs).
Document deferral of libvpx joint_motion_search.

## Gate safety
enable_alt_ref=false => altref_picker_live=false => no compound enumerated,
reference_mode=Single, sign_bias all false. Sacred gate untouched.

## RESULTS (LANDED)
- show_frame_picker_selects_compound: OFF=0, ON=267 compound picks. PASS.
- compound_stream_byte_exact_vpxdec: 267 compound leaves, BYTE-EXACT vpxdec. PASS.
- Sacred cross_decoder_gate: 4/4 BYTE-EXACT (vpxdec + ffmpeg). UNCHANGED.
- w_altref_picker (both): still pass (798 single-ref ALTREF picks) BYTE-EXACT.
- w_altref_real_coding byte-exact: PASS on real corpus (Select reference_mode).
- rvp9-encoder lib: 424/424 pass.
- No new clippy warnings in changed files.

Key surprise: enabling reference_mode=Select also unblocked the ALTREF
byte-exact path that was previously emitting Single (the comp_inter bit is
now correctly signalled per-block). Single-ref + compound coexist in the
same Select frame, both byte-exact.

Joint-MV: SCOPED DOWN as planned (independent single-ref MVs reused). The
byte-exact vpxdec gate confirms the averaging + emit are libvpx-exact;
joint refinement would only tighten residuals further.
