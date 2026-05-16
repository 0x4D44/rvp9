# Divergence ledger

Every deliberate departure from libvpx's behaviour, with rationale and
expected impact on byte parity. See `PLAN.md` §3 D8 — byte parity is a
diagnostic tool, not a release gate, but every divergence must be conscious
and documented here.

Empty entries below are placeholders for the categories we expect to grow.

## Format

Each entry:

```
### <YYYY-MM-DD> — <short title>
**Where:** <crate::module::item> + libvpx file/line being departed from
**What libvpx does:** <brief>
**What rvp9 does:** <brief>
**Why:** <rationale: safety, Rust idiom, perf, bug we won't replicate, ...>
**Byte-parity impact:** <none / minor / structural>
**Compression-efficiency impact:** <expected delta in BD-rate, if any>
**Future plan:** <revisit when? never?>
```

## Entries

### 2026-05-16 — High-bit-depth (10/12-bit) deferred to v2
**Where:** `rvp9-core::frame::Pixel` (type alias `pub type Pixel = u8;`);
all SIMD kernels in `rvp9-core::simd::*`; transforms, quant, intra-pred,
loop filter, motion-estimation, denoiser, frame-buffers, predictor SIMD.
libvpx counterpart: every `#if CONFIG_VP9_HIGHBITDEPTH` branch and
parallel `vpx_highbd_*` / `vpx_*_high*` kernel.
**What libvpx does:** ships Profile 0 (8-bit 4:2:0), Profile 1 (8-bit
4:2:2/4:4:4), Profile 2 (10-bit), Profile 3 (12-bit) with parallel
`u8` and `u16` code paths through every pixel-touching subsystem.
**What rvp9 does:** Profile 0 only in v1. `Pixel = u8` is a concrete
type, not generic. No `u16` paths anywhere.
**Why:** ~2× scope multiplier on every pixel-touching path; SIMD
kernel count doubles. Profile 3 (12-bit) has near-zero real-time
demand. PLAN.md target use cases (WebRTC + livestream) are 8-bit.
**Byte-parity impact:** none for our Profile 0 scope. Cross-decoder
streams using Profile 2/3 will fail outright (decoder rejects).
**Compression-efficiency impact:** none within Profile 0; HBD content
is simply out of scope.
**Future plan:** Stage 6+ expansion. Generic-over-`Pixel` is the natural
v2 pattern; the v1 code structure leaves `Pixel = u8` as the only call
site so it can be lifted with a type-parameter pass. Decision: D-01.

### 2026-05-16 — AQ modes 1 and 2 not ported
**Where:** `rvp9-encoder::aq` ships **CYCLIC_REFRESH only**. libvpx
counterpart: `vp9/encoder/vp9_aq_variance.c` (mode 1),
`vp9/encoder/vp9_aq_complexity.c` (mode 2). The `aq_mode` enum in our
port has variants only for `Off` and `CyclicRefresh`.
**What libvpx does:** ships four AQ modes (variance, complexity,
cyclic-refresh, lookahead) but guards modes 1/2/4 behind
`!CONFIG_REALTIME_ONLY`. At cpu-used 5-8 (our target band) modes 1/2
are unreachable; they are dead code at our config.
**What rvp9 does:** drop modes 1 and 2 entirely. Mode 3
(cyclic-refresh) is the only AQ implementation shipped.
**Why:** modes 1/2 are dead at our config; building them is dead code
with zero call paths in v1. The lookahead mode (4) is altref-coupled
and similarly excluded by D-13.
**Byte-parity impact:** none — these modes are never reached in matched
libvpx configs.
**Compression-efficiency impact:** none — same matched configs.
**Future plan:** if a non-real-time preset is ever added, the modes
can be ported then. No earlier. Decision: D-05.

### 2026-05-16 — Alt-ref frames (altref) not supported
**Where:** `rvp9-encoder::ratectrl` and `rvp9-encoder::frame_loop`;
the encoder rejects `lag_in_frames > 0` at construction. libvpx
counterpart: `vp9/encoder/vp9_alt_ref_aq.c`, the `GF_GROUP` altref
slot logic in `vp9/encoder/vp9_ratectrl.c`, the lookahead queue in
`vp9/encoder/vp9_lookahead.c`.
**What libvpx does:** under VBR (and `--lag-in-frames >= 4`) inserts
synthetic alt-reference frames into the prediction structure for
quality gains via look-ahead.
**What rvp9 does:** forbid altref; `lag_in_frames = 0` is the only
supported value. No `GF_GROUP` altref slots. No lookahead queue.
**Why:** altref requires `lag_in_frames > 0`, which directly
contradicts real-time / WebRTC use cases. Latency floor for altref is
4-16 frames depending on `lag_in_frames`; WebRTC budget is sub-frame.
PLAN.md D1 commits to real-time priority.
**Byte-parity impact:** none in matched configs (we mirror libvpx's
behaviour when `lag_in_frames=0`).
**Compression-efficiency impact:** none in matched configs. Against
libvpx with `lag_in_frames > 0`, rvp9 produces lower quality at the
same bitrate — but that's by design (different operating point).
**Future plan:** revisit if a "near-real-time livestream" preset
with `lag_in_frames=4` and a few hundred ms of buffered latency is
ever added. Decision: D-13.

### 2026-05-16 — EXT_RATECTRL plug-in interface always available
**Where:** `rvp9-encoder::ratectrl::ext::ExtRateControl` trait,
always-on, no Cargo feature gate. libvpx counterpart:
`vp9/encoder/vp9_ext_ratectrl.c` behind `CONFIG_RATE_CTRL`.
**What libvpx does:** EXT_RATECTRL is a build-time configurable
feature (`CONFIG_RATE_CTRL=1`) used primarily by libwebrtc. Default
libvpx builds ship without it.
**What rvp9 does:** ships the trait surface unconditionally. The
default `ExtRateControl` impl is a no-op (delegate to the built-in
RC); consumers (libwebrtc) implement the trait for custom control.
**Why:** libwebrtc integration is a v1 success criterion (PLAN.md D3).
Shipping the trait always-on means the API surface doesn't
re-stabilise post-v1.
**Byte-parity impact:** none when the default no-op impl is used —
the built-in RC runs unchanged. Custom impls produce intentionally
different bitstreams.
**Compression-efficiency impact:** none with default impl.
**Future plan:** stable trait API by v1 release; expand callback set
in v1.x non-breakingly. Decision: R-06.

### 2026-05-16 — Trellis-quantization (`vp9_optimize_b`) not ported
**Where:** `rvp9-encoder::quant` ships no trellis-quant path. libvpx
counterpart: `vp9_optimize_b()` in `vp9/encoder/vp9_encodemb.c`.
**What libvpx does:** at slower presets (cpu-used 0-4), runs a
trellis-quantization optimization pass over each coded block to
re-pick token values for lower RD cost.
**What rvp9 does:** no trellis-quant; quant output is used as-is.
**Why:** at cpu-used 6-8 (our target) libvpx sets
`sf->optimize_coefficients=0` — trellis-quant is bypassed. It is
dead code at our config. PLAN.md D1 priorities are clear.
**Byte-parity impact:** none in matched configs.
**Compression-efficiency impact:** none in matched configs.
**Future plan:** revisit only if a quality preset (cpu-used ≤ 4) is
added in a future version. Decision: L-28.

### 2026-05-16 — Compound-mode reference search not ported
**Where:** `rvp9-encoder::me` ships no compound-pred refinement.
libvpx counterpart: `vp9_refining_search_8p_c` and surrounding
compound search code in `vp9/encoder/vp9_mcomp.c`.
**What libvpx does:** at RD-loop presets, searches compound-prediction
references (last+gold, etc.) with an 8-point refinement after the
initial best-MV.
**What rvp9 does:** non-RD path only; compound search never invoked.
**Why:** compound-mode selection is gated off in the non-RD path at
cpu-used 5-8 (`set_rt_speed_feature_framesize_*`). Dead at our config.
**Byte-parity impact:** none in matched configs.
**Compression-efficiency impact:** none in matched configs.
**Future plan:** required if compound modes become reachable in a
future preset. Decision: L-08, L-46.

### 2026-05-16 — Accurate sub-pel search variants stubbed
**Where:** `rvp9-encoder::me::subpel` ships only the cpu-used 6-8
sub-pel pipeline. libvpx counterpart: `vp9_find_best_sub_pixel_tree`,
`vp9_find_best_sub_pixel_iterative` and variants in
`vp9/encoder/vp9_mcomp.c`.
**What libvpx does:** dispatches between several sub-pel search
algorithms based on `sf->mv.subpel_search_method`. Slower variants
do more iterations / smaller step sizes.
**What rvp9 does:** ships only the "fast / 1-iteration / 2-step"
variant matching cpu-used 6-8. Other variants are not ported.
**Why:** the slower variants are gated off at our cpu-used. Dead at
our config. Mirrors L-05 / A3 §7 #1.
**Byte-parity impact:** none in matched configs.
**Compression-efficiency impact:** none in matched configs.
**Future plan:** add slower variants if cpu-used ≤ 5 preset is ever
needed. Decision: L-05.

### 2026-05-16 — REALTIME mode pinned at encoder construction
**Where:** `rvp9-encoder::Encoder::new` accepts no `mode` parameter;
`MODE = REALTIME` is implicit and fixed. libvpx counterpart: the
GOOD / BEST / REALTIME mode dispatch in `vp9/encoder/vp9_encoder.c`
keyed on `oxcf.mode`.
**What libvpx does:** runtime-selectable mode (`--good`, `--best`,
`--rt`) with significant per-mode behavioural divergence (speed
features, ref-frame management, scene-cut detection).
**What rvp9 does:** REALTIME only. GOOD and BEST mode dispatch are
absent; calls that would route there are compile-time impossible.
**Why:** PLAN.md D1 commits to real-time priority; GOOD/BEST are
out of v1 scope. Pinning at construction lets us delete entire
code paths rather than gate them at runtime.
**Byte-parity impact:** none when libvpx is matched to `--rt`.
**Compression-efficiency impact:** none in matched configs. Versus
libvpx `--good` / `--best`, rvp9 is intentionally lower quality at
the same bitrate — but that's a different operating point.
**Future plan:** quality presets are a possible v2 expansion;
GOOD/BEST mode dispatch would re-enter then. Decision: L-32.

### 2026-05-16 — `encode_with_recode_loop` dropped
**Where:** `rvp9-encoder::frame_loop` is single-pass only. libvpx
counterpart: `encode_with_recode_loop()` in
`vp9/encoder/vp9_encoder.c`.
**What libvpx does:** under VBR/two-pass, may re-encode a frame
multiple times with adjusted Q to hit a target rate within tolerance.
**What rvp9 does:** single encode per frame, period. No recode.
**Why:** recode requires either two-pass mode (out of scope, no
look-ahead) or significant per-frame compute (incompatible with
real-time at 1080p60). At cpu-used 6-8 libvpx itself runs single-pass
only — recode dead at our config.
**Byte-parity impact:** none in matched configs.
**Compression-efficiency impact:** none in matched configs.
**Future plan:** dependent on a quality-preset addition. Decision: L-34.

### 2026-05-16 — Cyclic-refresh `time_for_refresh` field dropped
**Where:** `rvp9-encoder::aq::cyclic_refresh` does not carry a
`time_for_refresh` counter. libvpx counterpart: `cr->time_for_refresh`
in `vp9/encoder/vp9_aq_cyclicrefresh.c`.
**What libvpx does:** declares the field but reads it only in dead
paths. A5 noted this as a libvpx-internal TODO (Q-RC-24).
**What rvp9 does:** drop the field from the struct entirely; the
state machine works without it.
**Why:** dead field; carries no state. API simplification.
**Byte-parity impact:** none (the field doesn't affect bitstream).
**Compression-efficiency impact:** none.
**Future plan:** none. Decision: L-03.

---

_More entries land as Stage 2 LLDs surface deliberate departures
(template at §1 above — every LLD §7 feeds this ledger)._

### Categories anticipated

- **Threading primitives** — D6 commits to mirroring libvpx tile-worker
  threading. Where Rust's aliasing rules force a different shape, note it
  here.
- **Integer overflow behaviour** — D7 commits to mirroring widths.
  `wrapping_*` site catalogue lands here.
- **Bug-for-bug compatibility** — places where libvpx has a known bug
  we **choose not** to replicate.
- **Heuristic improvements** — places where we make a different
  heuristic decision in the non-RD path (informed by Stage 1 analysis).
