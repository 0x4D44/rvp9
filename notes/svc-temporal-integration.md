# SVC temporal integration (A18 zone) — scratchpad

## Goal
Wire the ported-but-inert `crates/rvp9-encoder/src/svc.rs` scaffold into the
encode path for TEMPORAL SVC (frame-rate scalability). Keep single-layer gate
byte-identical (SVC opt-in). Prove SVC stream byte-exact through vpxdec.

## Key architectural finding (verified against libvpx)
VP9 **temporal SVC** (1 spatial layer):
- Each frame is a STANDALONE VP9 frame. Superframes pack only SPATIAL layers
  within an access unit — NOT temporal layers. (Task brief's "pack into a
  superframe" applies to spatial SVC. `vp9_svc_layercontext.c:451`
  `++cpi->svc.current_superframe` is per access unit; for 1 spatial layer that
  is one frame.)
- `temporal_layer_id` is NOT on the VP9 wire. It is encoder-side metadata
  (libvpx exposes via `vpx_codec_get_frame`/`vpx_svc_layer_id`). vpxdec decodes
  the stream as a normal frame sequence and plays all layers.
- What IS on the wire and IS the real SVC behaviour: the per-frame ref/refresh
  structure from the temporal setters (`set_flags_and_fb_idx_for_temporal_mode2/3`,
  `vp9_svc_layercontext.c:517-720`). Higher-temporal-layer frames are
  NON-REFERENCE (refresh nothing the base layer reads) → the stream is
  temporally scalable (drop the top layer → still decodes).

## rvp9 pool model mapping (1 spatial layer)
- LAST = ref_frame_map[0], rotated by refresh bit 0.
- tl0 frame: refresh_frame_flags = 0x01 (refresh LAST), reads LAST. (libvpx
  mode2 tl0 spatial_id==0: ext_refresh_last_frame, ref_frame_flags=LAST.)
- top-tl (enhancement) frame: refresh_frame_flags = 0x00 (refresh NOTHING),
  reads LAST. This is the non-reference property: the next tl0 frame skips over
  it. (libvpx mode2 tl1 spatial_id==0: ext_refresh_alt_ref but in rvp9's single-
  slot model the alt slot is never read by tl0, so "refresh nothing readable by
  base" == refresh 0x00. Honest simplification: the droppable property is
  preserved exactly.)

## Plan (minimal, gate-safe)
1. EncoderConfig: add `temporal_layers: u32` (1 = off / default).
2. StreamEncoder: hold `Svc` + assign temporal_layer_id per inter frame.
3. EncodedFrame: add `temporal_layer_id: u8` (metadata, default 0).
4. In encode_inter_internal, when svc active: run temporal setter from the
   svc scaffold; derive temporal_layer_id; override refresh_frame_flags
   (tl0=0x01, top-tl=0x00). Keep ref reads on LAST (slot 0) so picker path
   unchanged → byte-exact through vpxdec.
5. KF (frame 0) is tl0 always; refresh 0xFF unchanged.
6. CLI `--temporal-layers N`.
7. Disable golden/altref cadence when svc active (temporal pattern owns the
   refresh structure; mixing GF refresh would break the layer skip).

## Gates
- single-layer (temporal_layers=1): config gates everything off → byte-identical.
- SVC stream: encode w/ temporal_layers=2, decode via vpxdec byte-exact +
  assert per-frame temporal_layer_id pattern [0,1,0,1,...] + KF=0.

## Anti-G3
- temporal_layer_id pattern must come from the svc scaffold's setter
  (`apply_temporal_layering`), not a hardcoded `idx & 1` in the test. The test
  asserts the ENCODER tags the right pattern AND that refresh flags differ
  between tl0 (0x01) and top-tl (0x00) — a stub that left refresh at 0x01 for
  all frames would fail the non-reference assertion.
