# RVP9-BUG-KILN-00009 - SIG1 (decoder FrameContext keyframe reset) regressed encoder/decoder entropy parity: encoder never reset its FrameContext on keyframes

- **State:** Fixed
- **Severity:** High
- **Area:** encoder/keyframe entropy reset (frame_loop.rs)
- **Raised:** 2026-06-20 by Claude (green-the-build pass)
- **State history:** 2026-06-20 Opened by Claude from the green-the-build pass; bisected to commit 16801c8 (SIG1). 2026-06-21 Fixed upstream by `e24e98f` ("fix(enc): reset FC mirrors at keyframes") — the encoder-side P0b reset landed concurrently/independently while this pass was in flight; verified by Claude during integration (`frame_loop::tests::aq_cyclic_refresh_keyframes_reset_state` passes on `origin/main` @ `e24e98f`). Awaiting two-eyes closure.

## Resolution (2026-06-21)

`e24e98f` replaced the "P0b — do NOT reset" block at `frame_loop.rs:2154` with
`self.frame_context = FrameContext::new_default(); self.decoder_fc_state =
FrameContext::new_default(); self.decoder_fc_state_arf = FrameContext::new_default();`
— exactly the encoder-side reset this bug called for, restoring encoder/decoder
entropy parity across keyframes (both sides now reset, matching libvpx
`vp9_setup_past_independence`). The test asserts the correct invariant and now
passes. Closure left for a second reviewer (two-eyes).

## Observation

`frame_loop::tests::aq_cyclic_refresh_keyframes_reset_state` (rvp9-encoder lib)
is RED on `origin/main` (`0b62ecc`). It round-trips encode→decode and asserts the
inter-frame-4 CYCLIC_REFRESH `segment_id` grid of GOP1 equals that of GOP2 (a
keyframe must reset CR state identically across GOPs). Observed:
GOP1 grid = `[1,1,1,1,0,0,0,0, ...]`, GOP2 grid = all `0` (assertion at
`crates/rvp9-encoder/src/frame_loop.rs:7000`).

**This is a regression introduced by the track2 decoder-fix harvest.** Bisected
(build+run at each commit, stdin-closed, mdtimeout):
- `01f8f2b` (pre-harvest baseline): **PASS**
- `0a63721` (commit immediately before SIG1): **PASS**
- `16801c8` ("fix(decoder): reset all FrameContext banks on every keyframe (SIG1)"): **FAIL**
- `0b62ecc` (HEAD): **FAIL**

## Notes

Root cause: SIG1 correctly made the **decoder** reset all FrameContext banks on
every keyframe (libvpx `vp9_setup_past_independence` parity), but the **encoder**
still deliberately does NOT reset its FrameContext on keyframes — see the explicit
paired-divergence comment at `crates/rvp9-encoder/src/frame_loop.rs:2149-2154`
("P0b — do NOT reset the frame context on keyframes … keeps encoder and decoder in
sync … until a follow-up wires the decoder-side FC-bank reset"). SIG1 *was* that
decoder-side follow-up, but the **encoder half was never completed**. After the
2nd keyframe the encoder codes with stale adapted banks while the now-correct
decoder resets to defaults → entropy desync → the decoder mis-decodes GOP2's inter
frames and reads their CR segment maps back as all-zero. The test asserts the
*correct* invariant; it only passed before because of the matched
(encoder-buggy + decoder-buggy) sync.

Recommended fix (do NOT revert SIG1 — it is correct, libvpx-faithful behaviour):
complete the P0b follow-up by making the encoder also reset all FrameContext banks
on keyframes (mirror `vp9_setup_past_independence`), replacing the "do NOT reset"
block at `frame_loop.rs:2149-2154`. This changes encoder entropy state after every
keyframe → **bitstream-affecting**; it must be cross-validated against the
round-trip / cross-decoder gates, conformance, and BD-rate before landing, and
warrants a version bump. Left for the encoder owner (in-flight `vpxenc-parity`)
rather than fixed in the green-the-build pass.

Related: KILN-00007 (SIG1 multi-keyframe independence lacks a committed-vector
regression guard) — same SIG1 area, decoder side.
