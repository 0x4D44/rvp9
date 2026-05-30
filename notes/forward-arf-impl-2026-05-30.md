# Forward-ARF + Overlay + Compound architecture — implementation notes (2026-05-30)

## Ground truth (measured BEFORE changes, foreman_320x256_60f, gop=60)
- OFF (no ARF): Y-PSNR 38.27 dB @ 630.0 kbps
- ON (current BACKWARD ARF): Y-PSNR 35.96 dB @ 627.4 kbps  => -2.31 dB REGRESSION
- Cross-decoder byte-exact ON: md5 c80ebeacff9d3ad547b9aa00f7ba01a7 (vpxdec==rvp9dec) PASS
- Per-frame: clear post-ARF collapse frames 33-39 (~33 dB) after ARF at decode-order ~30/display 31.

## Decoder support (BLOCKER CHECK -- PASS)
- crates/rvp9-decoder/src/lib.rs:1260 decode_show_existing_frame -- fully implemented.
- crates/rvp9-core/src/bitstream/header.rs:234 parse path -- ShowExistingFrame{slot}.
- decode_frame dispatches FrameKind::ShowExistingFrame -> decode_show_existing_frame (lib.rs:1347).
- NOT a blocker. Decoder shows ref_frame_map[slot].

## Current architecture (frame_loop.rs)
- try_encode_one_popped pops ONE source frame per call in DISPLAY order, encodes immediately.
- ARF built from CURRENT popped frame (arf_src_offset=0) = BACKWARD ARF. This is the root cause.
- ARF emitted as superframe prefix [ARF, show_frame], non-show, refresh 0x04 (slot 2 ALTREF).
- gop.rs GfGroup.schedule(): arf_before fires at offset gf_interval+1 (=31 for gop=60).
- Overlay: gop.rs defines OverlayUpdate enum variant but NEVER emitted.

## KEY CONSTRAINT discovered
- lag_in_frames capped at MAX_LAG_BUFFERS=25 (lookahead.rs:40).
- gop=60 => gf_interval=30. A TRUE forward ARF needs peek(gf_interval-1)=peek(29) > depth 25.
- => forward ARF spanning the full half-GOP is INFEASIBLE with 25-frame lookahead.
- libvpx uses ADAPTIVE small gf_interval (4-16). Forward source at arf_src_offset=gf_interval
  must fit lag_in_frames. So design MUST cap gf_interval to lookahead depth.

## Design decision
Because the existing frame loop pops in strict display order and the show frames are encoded
immediately, building a TRUE forward ARF requires the encoder to, at a GF-group boundary, reach
forward into the lookahead to a FUTURE frame, encode the ARF from it (non-show), then continue
popping/encoding the coast frames (which now read forward ALTREF), and finally emit a
show_existing_frame overlay at the ARF's display position.

This is a large reorder. Implementing it fully byte-exact within the existing single-pop loop is
the hard part. Plan: gate everything behind RVP9_ALTREF env (OFF == byte-identical baseline).

## FINAL RESULTS (2026-05-30)

### What was built (all gated behind RVP9_ALTREF=1; OFF == byte-identical baseline)
- gop.rs: GfGroup::new_with_forward + ForwardArfSlot enum + forward_arf_slot(): forward ARF at
  GF-group start sourced gf_interval ahead, overlay at the ARF display position. gf_interval clamped
  to lookahead_depth-1 (peek happens post-pop) and capped at 16. 1-frame KF lead-in (REQUIRED).
- frame_loop.rs: try_build_forward_arf_packet (sources peek(gf_interval-1)), emit_overlay_packet
  (show_existing_frame slot 2), forward_slot dispatch in try_encode_one_popped (overlay early-return,
  GroupStart forward-ARF build), overlays_emitted/forward_arf_active accessors.
- rvp9enc.rs: RVP9_ALTREF_STATS dump (altref/compound picks, forward_arf, overlays).

### Positive controls (foreman_320x256, lag 9/interval 8): ALL FIRE
- forward_arf=true, overlays_emitted=7, compound_picks=1695, altref_single_picks=3310.

### Byte-exact (MANDATORY): YES on foreman, NO on larger synthetic
- foreman 250/500/1000k + lag9 + lag25: vpxdec MD5 == rvp9dec MD5 (e.g. cdcf3cb8..., 5b09a05b...).
- testsrc 1280x704 (multi-tile): vpxdec "Failed to decode tile data" at frame 62 -> NOT byte-exact.
  Remaining multi-tile/large-frame divergence (single-tile foreman is exact).

### TWO blockers found + FIXED to reach foreman byte-exactness:
1. ARF-immediately-after-KF: vpxdec rejects a non-show frame as decode-order frame 1. FIX: 1-frame
   KF lead-in (group 1 starts at GOP offset 2, not 1). Permanent + libvpx-faithful.
2. Multi-ARF FC bank-1 chain: encode_altref_internal adapted bank 1 from the POST-diff decoder_fc
   (mutated by build_inter_packet) instead of the PRE-header snapshot. Single ARF/GOP masked it; the
   2nd forward ARF read a diverged bank 1 -> its compressed header decoded wrong on vpxdec (overlay
   showed garbage). FIX: snapshot pre_header_decoder_fc_arf and adapt from it (mirrors show path).

### QUALITY: SEVERE NET REGRESSION (~-9 dB), despite byte-exact.
matched-bitrate foreman (FWD uses MORE bits too):
  target 250 : OFF 35.23 @274k  vs FWD 25.84 @345k  = -9.4 dB
  target 500 : OFF 37.88 @565k  vs FWD 29.77 @640k  = -8.1 dB
  target 1000: OFF 40.15 @1005k vs FWD 30.66 @1031k = -9.5 dB
  mean ~ -9.0 dB.

ROOT CAUSE of the regression (per-frame PSNR, NOREF case, foreman lag9):
- Coast/overlay frames are GOOD (~38 dB; overlay at display 9 = 38.4 dB).
- GROUP-START show frames (display 10,18,...) COLLAPSE to ~23 dB.
- Why: the show_existing_frame OVERLAY refreshes NOTHING (the header has no refresh field). So the
  group-start show frame predicts from LAST = the last real coast frame BEFORE the overlay, skipping
  over the displayed ARF -> a reference-chain GAP + content discontinuity at every group boundary.
- Referencing the forward ALTREF (compound on) makes it WORSE (-5 dB more): the RD picker over-trusts
  the temporally-filtered ARF (the original root cause, re-manifested forward).

### CONCLUSION
Architecture fully wired + byte-exact on foreman + all positive controls fire, BUT it is a large
quality REGRESSION. The show_existing_frame overlay cannot refresh a slot, so it creates a reference
gap that tanks group-start frames; libvpx instead uses a REAL coded overlay frame that refreshes
GOLDEN (or threads LAST across the ARF). Making forward ARF a WIN needs: (a) a real overlay frame
(coded, refresh GOLDEN) rather than show_existing_frame, OR LAST-chain management across the overlay;
(b) ARF Q boost + temporal-filter tuning; (c) RD trust calibration on the ARF. Plus the multi-tile
byte-exact divergence (testsrc frame 62) to chase. Verdict: regress (byte-exact on foreman only).
