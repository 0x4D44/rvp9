# LLD: rate-control

> **Module path:** `rvp9-encoder::ratectrl` (and its `rtc` / `ext`
> sub-modules per HLD-arch §8 and HLD-svc §7).
> **Owner-of-record:** A5 (`notes/vp9-analysis/05-rate-control.md`).
> **Scope:** single-pass CBR + VBR-RT, no altref, AQ-3 (cyclic refresh)
> Q-delta application, EXT_RATECTRL plug-in, frame-drop + buffer-model
> arithmetic, `vp9_encodedframe_overshoot` re-encode block.
> **Scope guardrails (locked by PLAN.md):**
> - **D-01:** 8-bit only — no HBD quant tables, no HBD bit-depth scaling
>   in `vp9_convert_qindex_to_q`.
> - **D-05:** AQ mode 3 (CYCLIC_REFRESH_AQ) only — modes 1/2/4 not ported.
>   This LLD wires AQ-3 into the RC loop; segment-id assignment lives in
>   `notes/lld/aq.md`.
> - **D-13:** CBR + VBR-RT, **no altref**. No `lag_in_frames > 0`, no
>   GF_GROUP altref slots, no `is_src_frame_alt_ref` branching, no
>   `use_altref_onepass` speed-feature paths.
> - **R-06:** EXT_RATECTRL plug-in is IN v1. The trait surface lives in
>   HLD-svc §7; this LLD wires it into the per-frame `pick_q` pipeline.

---

## 1. Purpose

`rvp9-encoder::ratectrl` owns every bit-budget decision the encoder makes:
which qindex each frame is encoded at, whether a frame is dropped, how
much each spatial+temporal layer is allowed to spend, and how the buffer
model is updated post-encode. It is the single subsystem that decides
*how many bits a frame gets*; the rest of the encoder decides *how to
spend them*.

The module exposes two surfaces:

1. **In-process** — the `RateControl` state machine consumed by
   `Encoder::encode`. Wired into the per-frame loop as the canonical
   sequence from `notes/vp9-analysis/05-rate-control.md` §3.1.
2. **Standalone (libwebrtc adapter)** — the `RateControlRtc` engine
   re-uses the same `RateControl` core but skips the encoder pipeline,
   per HLD-svc §7.1. WebRTC calls `compute_qp` / `get_qp` /
   `post_encode_update` from outside the encoder and the engine carries
   no buffer-pool / tile-thread state.

Both surfaces share the same `RateControl` type. The wire-up details (how
the in-process surface calls into `pick_q` from `Encoder::encode_layer`)
are in HLD-svc §7.3 and §2.3; this LLD owns the type's internal state
machine and the `pick_q` / `regulate_q` / `postencode_update` arithmetic.

Within the encoder, this module also owns:

- The **`EXT_RATECTRL` trait callback hook** (`ExternalRateCtrl`,
  HLD-svc §7.2) — consulted between the internal `pick_q` and the actual
  encode. The trait is declared in `rvp9-core::ratectrl_ext` (HLD-arch
  §8.4) so the decoder doesn't pull encoder dependencies; this LLD owns
  the **call-site** that consumes the trait.
- The **CBR frame-drop decision** (`vp9_test_drop`, `vp9_rc_drop_frame`).
  Post-encode drop is **off by default** (D-07 default = exclude
  `post_encode_drop_cbr` from v1) — gated by a feature flag in case
  libwebrtc requires it later.
- The **`vp9_encodedframe_overshoot` re-encode block** (L-19, L-74).
  This is the single most opaque state mutation in the entire encoder;
  it must port byte-exactly.

Out of scope for this LLD:

- AQ-3 *segment-id assignment* and *segmentation map* maintenance — that
  is `notes/lld/aq.md` (cyclic-refresh sweep, candidate selection,
  segment-map writes). This LLD owns the **Q-delta arithmetic** that
  consumes AQ-3's segment-id output via `vp9_cyclic_refresh_rc_bits_per_mb`
  and `vp9_cyclic_refresh_estimate_bits_at_q`.
- Denoiser internals — `notes/lld/denoiser.md`. RC reads
  `noise_estimate.level` from this LLD's perspective; the kernel itself
  lives elsewhere.
- SVC layer iteration — `notes/lld/svc.md` owns `start_layer` and
  `save_layer_context` / `restore_layer_context`. This LLD owns the
  **per-layer `RateControl` value-swap** and the cross-layer
  `force_max_q` propagation in `encodedframe_overshoot`.
- Two-pass anything. Per L-32 (HLD-arch), the encoder is pinned to
  `REALTIME` mode, so `pass == 2` branches in libvpx's
  `vp9_rc_pick_q_and_bounds`, `set_rate_correction_factor`, etc. are
  dead code in our port and not ported.

---

## 2. libvpx origin

| Module surface | libvpx source | A5 §ref |
|---|---|---|
| `RateControl` struct | `vp9/encoder/vp9_ratectrl.h:88-218` | §2.1 |
| Init + framerate setup | `vp9/encoder/vp9_ratectrl.c:360-449`, `vp9_new_framerate()` | §2.1, §4.7 |
| Bits-per-MB model | `vp9_ratectrl.c:185-205` | §4.1 |
| Qindex search (`regulate_q`) | `vp9_ratectrl.c:819-862` | §4.2 |
| `adjust_q_cbr` | `vp9_ratectrl.c:666-687` | §4.2 |
| RCF update | `vp9_ratectrl.c:737-817` | §4.3 |
| Active-worst-Q (CBR) | `vp9_ratectrl.c:931-1000` | §4.4 |
| Active-worst-Q (VBR) | `vp9_ratectrl.c:905-928` | §4.4 |
| `rc_pick_q_and_bounds_one_pass_cbr` | `vp9_ratectrl.c:1002-1100` | §4.4 |
| `rc_pick_q_and_bounds_one_pass_vbr` | `vp9_ratectrl.c:1138-1303` | §4.4 |
| Min-Q LUTs (3rd-order fit) | `vp9_ratectrl.c:102-154` | §4.5 |
| CBR P-frame target | `vp9_ratectrl.c:2145-2192` | §4.6 |
| CBR I-frame target | `vp9_ratectrl.c:2194-2221` | §4.6 |
| VBR P-frame target | `vp9_ratectrl.c:2019-2037` | §4.6 |
| VBR I-frame target | `vp9_ratectrl.c:2039-2048` | §4.6 |
| Buffer-level pre-encode | `vp9_ratectrl.c:250-256` | §4.7 |
| Buffer-level post-encode | `vp9_ratectrl.c:316-333` | §4.7 |
| Frame-drop pre-encode | `vp9_ratectrl.c:513-664` (`vp9_test_drop`, `vp9_rc_drop_frame`) | §4.8 |
| Post-encode CBR drop (D-07 default off) | `vp9_ratectrl.c:566-612` | §4.8 |
| Post-encode update | `vp9_ratectrl.c:1845-1996` | §3.1 |
| Drop bookkeeping | `vp9_ratectrl.c:1998-2017` | §4.8 |
| `vp9_encodedframe_overshoot` | `vp9_ratectrl.c:3248-3351` | §4 + Q-RC-20 + L-19/L-74 |
| Scene-detection one-pass | `vp9_ratectrl.c:3034-3243` | §3.1 |
| SVC pre-encode buffer pump | `vp9_ratectrl.c:263-294` | §3.4 |
| SVC post-encode buffer drain | `vp9_ratectrl.c:298-313` | §3.4 |
| Rate-correction reads (`get_rate_correction_factor`) | `vp9_ratectrl.c:689-710` | §4.3 |
| Rate-correction writes (`set_rate_correction_factor`) | `vp9_ratectrl.c:712-735` | §4.3 |
| `vbr_rate_correction` | `vp9_ratectrl.c:2679-2720` | §2.1 |
| `vp9_compute_qdelta_by_rate` | `vp9_ratectrl.c:2562-2584` | §3.3 |
| `VP9RateControlRTC` standalone | `vp9/ratectrl_rtc.{h,cc}` (whole file) | A5 §3.4 + HLD-svc §7.1 |
| EXT_RATECTRL plug-in | `vpx/vpx_ext_ratectrl.h:511-552` | A5 + HLD-svc §7.2 |

Stage 1 primary: `notes/vp9-analysis/05-rate-control.md` (especially §2.1,
§3.1, §4.1-4.8, §5.4, §7 Q-RC-1..Q-RC-30). Cross-cutting context:
`notes/vp9-analysis/00-overview.md` §3.1 (`cpi->rc` placement, ownership,
not-worker-write), §4.6 (CR ownership coupling), §5.1 dispositions for
A7 challenges #5 (L-19), #18 (L-69), #19 (no-altref), #21 (L-75), #22
(L-76), #25 (R-06), #42 (L-77).

---

## 3. Public API

All types and functions are declared in `rvp9-encoder::ratectrl` (or
sub-modules). Public re-exports through `rvp9_encoder::ratectrl::{...}`
match the HLD-arch §8.2 surface.

### 3.1 Configuration ingest

```rust
/// CBR | VBR(-RT) only — D-13 forbids CQ and constant-Q.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RcMode { Cbr, VbrRt }

/// 1-pass real-time. 2-pass / GOOD-quality are NOT in v1 (HLD-arch
/// §1.1, L-32). Pinning the type to `OnePassRealtime` lets the
/// type-checker reject 2-pass code paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RcPass { OnePassRealtime }

/// Encoder content hint. Affects scene-detect thresholds, buffer
/// clipping, and `vp9_encodedframe_overshoot` thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentMode { Default, Screen }

/// Real-time, no-lag config. Mirrors `VP9EncoderConfig` fields that
/// `RateControl::new` and `RateControl::reconfigure` consume.
#[derive(Debug, Clone)]
pub struct RcConfig {
    pub width:  u32,
    pub height: u32,
    pub target_bandwidth_bps: u32,
    pub framerate: f64,                 // libvpx stores as double
    pub rc_mode: RcMode,
    pub content: ContentMode,
    pub aq_mode: super::aq::AqMode,     // Off | CyclicRefresh (D-05)

    pub worst_allowed_q: u8,            // 0..=255
    pub best_allowed_q:  u8,            // 0..=255

    pub starting_buffer_level_ms: u32,
    pub optimal_buffer_level_ms:  u32,
    pub maximum_buffer_size_ms:   u32,

    pub under_shoot_pct: u8,            // 0..=100
    pub over_shoot_pct:  u8,            // 0..=200
    pub max_intra_bitrate_pct: u8,      // 0..=400 (libvpx unsigned)
    pub max_inter_bitrate_pct: u8,      // 0..=400

    pub key_freq:       u32,            // forced KF interval; 0 disables
    pub auto_key:       bool,
    pub drop_frames_water_mark: u8,     // 0..=100; 0 disables frame-drop

    /// L-73 reach: PLAN.md gates cpu-used 6/7/8; cpu-used 5 allowed
    /// without CI gate.
    pub cpu_used: u8,                   // 5..=8 (D-13 / D-05 / D-01 scope)

    /// L-76: VBR-RT *without* altref. We surface this as a *config
    /// invariant* and validate at `RateControl::new`: when
    /// `rc_mode == VbrRt`, this is forced to 0 and reject otherwise.
    pub lag_in_frames: u8,              // ALWAYS 0 in v1

    /// SVC counts. See `notes/lld/svc.md` for layer-context plumbing.
    pub spatial_layers:  u8,            // 1..=3
    pub temporal_layers: u8,            // 1..=3

    /// Per-layer min/max (only consumed when SVC enabled; mirrored
    /// here so the layer-context swap in `notes/lld/svc.md` can read
    /// without separate state).
    pub layer_min_q:        [u8; MAX_LAYERS],
    pub layer_max_q:        [u8; MAX_LAYERS],
    pub layer_target_kbps:  [u32; MAX_LAYERS],

    /// gf_cbr_boost_pct: 0..=500. RC P-frame target inflates by this
    /// fraction on golden-refresh frames. `vp9_calc_pframe_target_size_one_pass_cbr`
    /// branches on this.
    pub gf_cbr_boost_pct: u32,

    /// post-encode drop (D-07): default off in v1. Feature-gated to
    /// keep the libwebrtc adapter path open.
    pub ext_use_post_encode_drop: bool,

    /// CBR overshoot detection (sf.overshoot_detection_cbr_rt).
    /// `None` = disabled; `Some(ReEncodeMaxQ)` triggers the
    /// `vp9_encodedframe_overshoot` block on big overshoot.
    pub overshoot_detection_cbr_rt: OvershootDetection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OvershootDetection {
    Off,
    /// Re-encode with worst_quality if `frame_size > 8 × avg_bandwidth`
    /// (or 8× × 2 for screen).
    ReEncodeMaxQ,
    /// Scene-change driven (no size threshold); used by some screen
    /// configurations. Mirrors `FAST_DETECTION_MAXQ` (libvpx).
    FastDetectionMaxQ,
}

pub const MAX_SPATIAL_LAYERS:  usize = 3;     // HLD-svc §10.3 #2
pub const MAX_TEMPORAL_LAYERS: usize = 3;
pub const MAX_LAYERS: usize = MAX_SPATIAL_LAYERS * MAX_TEMPORAL_LAYERS;
```

### 3.2 Frame-level entry points

These mirror `vp9_ratectrl.h:241-261` ("First call per frame, then call
encode, then call post-encode-update"). One-pass-realtime only.

```rust
/// Per-frame inputs. Built by the encoder shell before calling
/// `RateControl::pick_q`. Fields originate either from `Encoder::encode`
/// args (e.g. `force_keyframe`) or from per-frame transient state
/// (`avg_source_sad`, `high_source_sad`) that the scene-detector
/// computed earlier in the frame loop (§5.2 step 1).
#[derive(Debug, Clone)]
pub struct FrameRcInputs {
    pub force_keyframe: bool,
    /// Per-spatial-layer; only the active layer is read. See
    /// HLD-svc §2.3 for the layer-iteration order.
    pub spatial_layer_id:  u8,
    pub temporal_layer_id: u8,
    /// Scene-detector output (always-on per L-23). The encoder shell
    /// owns this computation and passes it forward; this LLD reads
    /// `high_source_sad` from the inputs into `RateControl.high_source_sad`
    /// before `pick_q`.
    pub avg_source_sad:  u64,
    pub high_source_sad: bool,
    /// Noise-estimator level. Only consumed when AQ-3 is active and
    /// `cpi->oxcf.noise_sensitivity > 0`. See `notes/lld/denoiser.md`
    /// for the producer.
    pub noise_level: super::denoiser::NoiseLevel,
}

/// What `pick_q` returns to the encoder. The encoder shell then
/// (a) consults `ExternalRateCtrl` (§3.5), (b) calls `vp9_encode_frame`,
/// (c) optionally re-encodes via `encodedframe_overshoot` (§3.4),
/// (d) calls `post_encode_update`.
#[derive(Debug, Clone, Copy)]
pub struct QDecision {
    pub q_index:     u8,       // chosen qindex (cm->base_qindex)
    pub bottom_q:    u8,       // active_best_quality (for AQ-3 segment math)
    pub top_q:       u8,       // active_worst_quality
    /// True ⇒ the frame_type was forced to KEY by RC (auto_key, or
    /// `frames_to_key == 0`). The encoder shell mirrors this into
    /// `cm.frame_type`.
    pub forced_key: bool,
}

/// Drop verdict returned by `should_drop_pre_encode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropDecision { Encode, Drop }

impl RateControl {
    /// Fresh state machine. Mirrors `vp9_rc_init` (`vp9_ratectrl.c:360-449`)
    /// plus `vp9_new_framerate` buffer-sizing. Errors: see `RcConfigError`
    /// (§4.5), including L-76 `VbrRtWithLag` rejection.
    pub fn new(cfg: &RcConfig) -> Result<Self, RcConfigError>;

    /// Mid-stream config update. Mirrors `vp9_change_config` RC subset
    /// (`vp9_ratectrl.c:451-475`). Does NOT clear `rate_correction_factors[]`
    /// (L-25 / Q-RC-30 — `rcf_mult[]` scaling on read/write).
    pub fn reconfigure(&mut self, cfg: &RcConfig) -> Result<(), RcConfigError>;

    /// Pre-encode frame-drop check (`vp9_rc_drop_frame`/`vp9_test_drop`,
    /// `vp9_ratectrl.c:513-664`). Mutates `decimation_factor`/`_count`.
    /// Algorithm: §5.9.
    pub fn should_drop_pre_encode(
        &mut self,
        svc: Option<&super::svc::SvcSharedState>,
    ) -> DropDecision;

    /// Pick the qindex. Implements §5.2's canonical sequence. Callers
    /// must have already cleared `should_drop_pre_encode`. AQ-3 hook
    /// at `regulate_q` calls `cr.bits_per_mb` (in `notes/lld/aq.md`).
    pub fn pick_q(
        &mut self,
        inputs: &FrameRcInputs,
        cr: Option<&mut super::aq::CyclicRefresh>,
    ) -> QDecision;

    /// Post-encode update (§5.8 / `vp9_rc_postencode_update`,
    /// `vp9_ratectrl.c:1845-1996`). No altref state touched (D-13).
    pub fn post_encode_update(
        &mut self,
        encoded_bytes: u32,
        chosen_q: u8,
        cr: Option<&mut super::aq::CyclicRefresh>,
    );

    /// Drop bookkeeping (§5.9 / `vp9_rc_postencode_update_drop_frame`,
    /// `vp9_ratectrl.c:1998-2017`).
    pub fn post_encode_update_drop(&mut self, was_svc_full_drop: bool);

    /// Re-encode-at-maxQ entry. `true` ⇒ caller must discard tentative
    /// pack and re-encode at `q.q_index`. Mirrors
    /// `vp9_encodedframe_overshoot` (`vp9_ratectrl.c:3248-3351`).
    /// **Algorithm + L-19/L-74 mutation list in §5.7.**
    pub fn encodedframe_overshoot(
        &mut self,
        frame_size_bits: u32,
        q: &mut QDecision,
        cr: Option<&mut super::aq::CyclicRefresh>,
        svc: Option<&mut super::svc::SvcLayerRcs<'_>>,
        intra_usage_pct: Option<u32>,   // computed by encoder shell; see §5.7
    ) -> bool;
}
```

`RcConfigError` is a non-exhaustive enum per HLD-arch §6 (error model).

### 3.3 Frame target getters (read-only)

These let `notes/lld/svc.md` and `notes/lld/aq.md` read state without
re-computing it.

```rust
impl RateControl {
    /// Currently-set target. Set by `pick_q` before `regulate_q` runs.
    pub fn this_frame_target_bits(&self) -> u32;
    /// `avg_frame_qindex[FrameType]`. Used by AQ-3's screen-content
    /// `vp9_cyclic_refresh_limit_q` check (`vp9_aq_cyclicrefresh.c:695-702`).
    pub fn avg_frame_qindex(&self, ft: FrameType) -> u8;
    /// Buffer state — used by the SVC frame-drop chain (§5.6) and
    /// HLD-svc §6.4.
    pub fn buffer_level_bits(&self) -> i64;
    /// `optimal_buffer_level` after framerate setup.
    pub fn optimal_buffer_level_bits(&self) -> i64;
    /// `critical_level = optimal_buffer_level >> 3` per L-18.
    pub fn critical_level_bits(&self) -> i64 {
        self.optimal_buffer_level_bits() >> 3
    }
    /// `current_video_frame` — bumped by post_encode_update. Used by
    /// the active-worst-Q early-frame branch (`current_video_frame < 5*ntl`).
    pub fn current_video_frame(&self) -> u32;
    /// `frames_to_key`, used by AQ-3 wake-up logic.
    pub fn frames_to_key(&self) -> u32;
    pub fn frames_since_key(&self) -> u32;
    /// `force_max_q` set by post_encode_drop or SVC propagation.
    pub fn force_max_q(&self) -> bool;
    /// AQ-3 reads this to decide `qp_thresh` defaults.
    pub fn rate_correction_factor(&self, lvl: RateFactorLevel) -> f64;
}
```

### 3.4 EXT_RATECTRL plug-in surface

The plug-in trait lives in `rvp9-core::ratectrl_ext` per HLD-arch §8.4.
The encoder shell (not this module) **owns the `Option<Box<dyn
ExternalRateCtrl>>`** and consults it between `pick_q` and the actual
encode. This LLD owns only the **default-Q fallback path**:

```rust
/// Re-export from rvp9-core. See HLD-svc §7.2 for full trait shape.
pub use rvp9_core::ratectrl_ext::{
    ExternalRateCtrl, EncodeFrameInfo, EncodeFrameDecision, EncodeFrameResult,
    MaxFrameSize, RcType, RcStatus,
};

/// Build the EncodeFrameInfo from the current `RateControl` state +
/// the encoder shell's per-frame inputs. Called by `Encoder::encode_layer`
/// only when an `ExternalRateCtrl` is installed (HLD-svc §7.3).
pub fn build_encode_frame_info(
    rc: &RateControl,
    inputs: &FrameRcInputs,
    coding_index: u32,
    show_index:   u32,
    gop_index:    u32,
    gop_size:     u32,
    ref_frame_coding_indexes: [u32; 3],
    ref_frame_valid_list:     [bool; 3],
) -> EncodeFrameInfo;

/// Reconcile an external decision against the internal pick. Clamps
/// `q_index` to `[layer_min_q, layer_max_q]` per HLD-svc §10.3 #4 and
/// emits a `tracing::warn!` when clamped (out-of-range Q would break
/// the buffer model). `decision.max_frame_size = Bits(_)` is ignored
/// in v1 per HLD-svc §7.3 + divergence ledger entry §7.3 below.
pub fn reconcile_external_decision(
    rc: &RateControl,
    internal: QDecision,
    external: &EncodeFrameDecision,
    layer_min_q: u8,
    layer_max_q: u8,
) -> QDecision;
```

### 3.5 Standalone `RateControlRtc` (libwebrtc adapter)

Re-exports of HLD-svc §7.1; full surface defined there. The
implementation re-uses the *same* `RateControl` struct, exposed through
a thinner API that the WebRTC caller drives directly.

```rust
pub use crate::ratectrl::rtc::{
    RateControlRtc, VP9RateControlRtcConfig, VP9FrameParamsQpRtc,
    FrameDropDecision, RcFrameType, SegmentationData,
};

impl RateControlRtc {
    pub fn create(cfg: &VP9RateControlRtcConfig) -> Option<Self>;
    pub fn update_rate_control(&mut self, cfg: &VP9RateControlRtcConfig) -> bool;
    pub fn compute_qp(&mut self, params: &VP9FrameParamsQpRtc) -> FrameDropDecision;
    pub fn get_qp(&self) -> u8;
    pub fn get_loopfilter_level(&self) -> u8;
    pub fn get_segmentation_data(&self) -> Option<SegmentationData<'_>>;
    pub fn post_encode_update(&mut self, encoded_bits: u64, params: &VP9FrameParamsQpRtc);
}
```

Internal: `RateControlRtc` holds an `EncoderStateLite`, which is a
`RateControl` + `CyclicRefresh` + per-layer state. No buffer pool, no
workers; calls drop through to the same `pick_q` / `post_encode_update`
methods above.

---

## 4. Types

### 4.1 `RateControl` (the state machine)

Layout mirrors `RATE_CONTROL` (`vp9_ratectrl.h:88-218`) with no-altref
fields removed (D-13) and L-01 consolidation applied. Lives at
`rvp9-encoder::ratectrl::state`. Owned by `EncoderState`
(per-encoder, A5 §2.1; HLD-arch §3.5 row "rc"). Per-layer copies
live in `LayerContext.rc` (HLD-svc §2.2; §3.4 below).

```rust
/// `RATE_CONTROL` (`vp9_ratectrl.h:88`) — no-altref, no-2-pass slim.
///
/// **Send + !Sync** — main thread mutates between frames; workers see
/// only the read-only `cm.base_qindex` snapshot taken at frame start
/// (HLD-threading §5.2 row `cpi->rc`). No interior mutability.
#[derive(Debug, Clone)]
pub struct RateControl {
    // ───────── Config (cached from RcConfig at new/reconfigure) ─────────
    pub(crate) cfg: RcConfig,             // immutable per encode; reconfigure clones

    // ───────── Targeting ─────────
    /// `base_frame_target` — baseline before under/overshoot correction.
    pub(crate) base_frame_target:     u32,
    /// `this_frame_target` — after vbr_rate_correction adjustment.
    pub(crate) this_frame_target:     u32,
    /// Filled in by `post_encode_update` with `encoded_bytes << 3`.
    pub(crate) projected_frame_size:  u32,
    /// `sb64_target_rate` — `this_frame_target * (256 + sb_count/2) / sb_count`.
    pub(crate) sb64_target_rate:      u32,

    /// `last_q[KEY=0, INTER=1]` (`vp9_ratectrl.h:96`).
    pub(crate) last_q:                [u8; 2],
    /// Last Q used on a forced-KF or GF (used by KF Q-delta path).
    pub(crate) last_boosted_qindex:   u8,
    pub(crate) last_kf_qindex:        u8,

    /// `gfu_boost`, `last_boost`, `kf_boost`. Default GF/KF boost values
    /// per `DEFAULT_KF_BOOST=2000`, `DEFAULT_GF_BOOST=2000` (`vp9_ratectrl.h:30-31`).
    pub(crate) gfu_boost:             u32,
    pub(crate) last_boost:            u32,
    pub(crate) kf_boost:              u32,

    /// **The 5 per-rate-factor RCFs.** Indexed by `RateFactorLevel`.
    /// Clamped to `[MIN_BPB_FACTOR=0.005, MAX_BPB_FACTOR=50]`
    /// (`vp9_ratectrl.c:44-45`). L-19 #8: only `INTER_NORMAL` is mutated
    /// by `encodedframe_overshoot`.
    pub(crate) rate_correction_factors: [f64; RateFactorLevel::COUNT],

    /// **`damped_adjustment[5]`** — per-rf-level first-frame flag.
    /// 0 ⇒ next update uses `adjustment_limit = 1.0`; flips to 1 after.
    /// Never reset thereafter (Q-RC-7). Initialised to false in
    /// `RateControl::new` (`vp9_ratectrl.c:419-422`).
    pub(crate) damped_adjustment: [bool; RateFactorLevel::COUNT],

    /// `frames_since_golden` — bookkeeping for the GF interval. Reset
    /// to 0 on a golden refresh.
    pub(crate) frames_since_golden:        u32,
    pub(crate) frames_till_gf_update_due:  u32,
    pub(crate) min_gf_interval:            u32,   // `MIN_GF_INTERVAL=4`
    pub(crate) max_gf_interval:            u32,   // `MAX_GF_INTERVAL=16`
    pub(crate) static_scene_max_gf_interval: u32, // up to MAX_STATIC_GF_GROUP_LENGTH=250
    pub(crate) baseline_gf_interval:       u32,
    pub(crate) constrained_gf_group:       bool,
    pub(crate) frames_to_key:              u32,
    pub(crate) frames_since_key:           u32,
    pub(crate) this_key_frame_forced:      bool,
    pub(crate) next_key_frame_forced:      bool,

    // **Altref fields REMOVED** (D-13 / L-76, HLD-svc §9.4):
    //   ❌ source_alt_ref_pending
    //   ❌ source_alt_ref_active
    //   ❌ is_src_frame_alt_ref
    //   ❌ alt_ref_gf_group
    //   ❌ last_frame_is_src_altref

    /// `avg_frame_bandwidth = target_bandwidth_bps / framerate` (rounded).
    pub(crate) avg_frame_bandwidth: u32,
    /// `max(avg_frame_bandwidth >> 4, FRAME_OVERHEAD_BITS=200)` (per-frame floor).
    pub(crate) min_frame_bandwidth: u32,
    /// `min(avg_frame_bandwidth * MAX_MB_RATE=250, MAXRATE_1080P=4_000_000)` —
    /// per-frame ceiling for I/GF inflation.
    pub(crate) max_frame_bandwidth: u32,
    /// Stored from the **previous** frame's `avg_frame_bandwidth`. Used
    /// by `vp9_compute_qdelta_by_rate` and by the RCF normalisation.
    pub(crate) last_avg_frame_bandwidth: u32,

    /// `ni_av_qi`, `ni_tot_qi`, `ni_frames` — IIR over Q used by min-Q
    /// LUT path. Updated post-encode.
    pub(crate) ni_av_qi:    u8,
    pub(crate) ni_tot_qi:   u64,
    pub(crate) ni_frames:   u32,
    pub(crate) avg_frame_qindex: [u8; 2],     // [KEY=0, INTER=1]
    pub(crate) tot_q:       f64,
    pub(crate) avg_q:       f64,

    // ───────── Buffer model ─────────
    /// **L-01 RESOLVED:** consolidated to a single field.
    /// libvpx kept `buffer_level` and `bits_off_target` as aliases
    /// (TODO at `vp9_ratectrl.c:246-249`). We collapse to one. See
    /// §7.1 for the divergence-ledger entry.
    pub(crate) buffer_level: i64,
    /// VBR cumulative bit error (long-term, lines 1923-1931).
    pub(crate) vbr_bits_off_target:      i64,
    /// VBR fast-reclaim cumulative.
    pub(crate) vbr_bits_off_target_fast: i64,

    // ───────── Pre-encode buffer targets (derived from cfg ms × bandwidth) ─────────
    pub(crate) starting_buffer_level: i64,
    pub(crate) optimal_buffer_level:  i64,
    pub(crate) maximum_buffer_size:   i64,

    // ───────── Frame-drop ─────────
    /// `decimation_factor` ∈ {0, 1}. Capped at 1 (drop every other
    /// frame); `vp9_test_drop` never sets higher.
    pub(crate) decimation_factor: u8,
    pub(crate) decimation_count:  u8,

    // ───────── Rolling averages ─────────
    pub(crate) rolling_target_bits:      u32,
    pub(crate) rolling_actual_bits:      u32,
    pub(crate) long_rolling_target_bits: u32,
    pub(crate) long_rolling_actual_bits: u32,

    /// `rate_error_estimate` — exposed to `update_rate_correction_factors`
    /// for VBR adjustment.
    pub(crate) rate_error_estimate: i32,

    // ───────── Cumulative ─────────
    pub(crate) total_actual_bits:       u64,
    pub(crate) total_target_bits:       u64,
    pub(crate) total_target_vs_actual:  i64,

    // ───────── Q range ─────────
    pub(crate) worst_quality: u8,        // = cfg.worst_allowed_q
    pub(crate) best_quality:  u8,        // = cfg.best_allowed_q

    // ───────── Oscillation history ─────────
    /// rc_1/2_frame ∈ {-1, 0, +1}; +1 = undershoot, -1 = overshoot
    /// (sign inverted from intuition; see §5.4).
    pub(crate) rc_1_frame: i8,
    pub(crate) rc_2_frame: i8,
    pub(crate) q_1_frame:  u8,
    pub(crate) q_2_frame:  u8,

    // ───────── Auto frame-scaling ─────────
    pub(crate) frame_size_selector:        FrameScaleLevel,
    pub(crate) next_frame_size_selector:   FrameScaleLevel,
    pub(crate) frame_width:                [u32; FRAME_SCALE_STEPS],
    pub(crate) frame_height:               [u32; FRAME_SCALE_STEPS],
    /// `rf_level_maxq[5]` — per-rate-factor max Q observed. Used by
    /// the VBR `vbr_rate_correction` path; CBR ignores.
    pub(crate) rf_level_maxq: [u8; RateFactorLevel::COUNT],

    // ───────── VBR-RT active-worst-Q multipliers ─────────
    /// `fac_active_worst_inter` (default 150 → 1.5×) and
    /// `fac_active_worst_gf` (default 100). Both stored as integer
    /// percentages so the arithmetic matches libvpx.
    pub(crate) fac_active_worst_inter: u32,
    pub(crate) fac_active_worst_gf:    u32,

    // ───────── Scene-detect outputs ─────────
    /// Ring buffer; the dispatcher (`notes/lld/svc.md` scene_detect)
    /// fills index `current_video_frame % MAX_LAG_BUFFERS`.
    /// **MAX_LAG_BUFFERS = 25** in libvpx; we keep the same array
    /// dimension so the ring math matches.
    pub(crate) avg_source_sad:           [u64; MAX_LAG_BUFFERS],
    pub(crate) prev_avg_source_sad_lag:  u64,
    /// `high_source_sad_lagindex` is only meaningful when `lag_in_frames > 0`
    /// — left at 0 in v1 (D-13). Kept for traceability with libvpx.
    pub(crate) high_source_sad_lagindex: u32,
    pub(crate) high_num_blocks_with_motion: u32,
    pub(crate) high_source_sad:          bool,
    pub(crate) count_last_scene_change:  u32,
    pub(crate) hybrid_intra_scene_change: bool,
    pub(crate) re_encode_maxq_scene_change: bool,
    pub(crate) avg_frame_low_motion:     u32,
    pub(crate) af_ratio_onepass_vbr:     u32,   // default 10, [5..15]
    pub(crate) force_qpmin:              bool,
    pub(crate) reset_high_source_sad:    bool,

    pub(crate) force_max_q:              bool,
    pub(crate) last_post_encode_dropped_scene_change: bool,
    pub(crate) use_post_encode_drop:     bool,
    pub(crate) ext_use_post_encode_drop: bool,
    pub(crate) disable_overshoot_maxq_cbr: bool,

    // **Altref-coupled fields ELIDED** (D-13 / §7.6 divergence entry):
    //   ❌ perc_arf_usage
    //   ❌ arf_active_best_quality_adjustment_factor
    //   ❌ arf_increase_active_best_quality
    //   ❌ preserve_arf_as_gld / preserve_next_arf_as_gld / show_arf_as_gld

    pub(crate) constrain_gf_key_freq_onepass_vbr: bool,
    pub(crate) gop_global_index:          u32,
}
```

**Dead-field retention decision.** §8.1's recommendation is *elide* —
the dead altref fields above are removed from the active layout
(§7.6). The grep-equivalence loss is acceptable; LLDs are the audit
trail, not field names.

### 4.2 Enums and module-level constants

```rust
/// `RATE_FACTOR_LEVEL` (`vp9_ratectrl.h:48-55`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RateFactorLevel {
    InterNormal = 0,
    InterHigh   = 1,
    GfArfLow    = 2,
    GfArfStd    = 3,
    KfStd       = 4,
}
impl RateFactorLevel { pub const COUNT: usize = 5; }

/// `FRAME_SCALE_LEVEL` (`vp9_ratectrl.h:58-62`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameScaleLevel { Unscaled = 0, ScaleStep1 = 1 }

/// `FRAME_TYPE` (`vp9/common/vp9_enums.h:53`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameType { Key = 0, Inter = 1 }

// Constants from `vp9_ratectrl.{h,c}` (§5.4 of A5 / `notes/vp9-analysis/05-rate-control.md`).
pub const BPER_MB_NORMBITS:           u32 = 9;
pub const DEFAULT_KF_BOOST:           u32 = 2000;
pub const DEFAULT_GF_BOOST:           u32 = 2000;
pub const MIN_GF_INTERVAL:            u32 = 4;
pub const MAX_GF_INTERVAL:            u32 = 16;
pub const FIXED_GF_INTERVAL:          u32 = 8;
pub const FRAME_OVERHEAD_BITS:        u32 = 200;
pub const STATIC_KF_GROUP_THRESH:     u32 = 99;
pub const MAX_STATIC_GF_GROUP_LENGTH: u32 = 250;
pub const MAX_MB_RATE:                u32 = 250;
pub const MAXRATE_1080P:              u32 = 4_000_000;
pub const LIMIT_QRANGE_FOR_ALTREF_AND_KEY: bool = true;
pub const MIN_BPB_FACTOR:             f64 = 0.005;
pub const MAX_BPB_FACTOR:             f64 = 50.0;
pub const VBR_PCT_ADJUSTMENT_LIMIT:   u32 = 50;
pub const SMOOTH_PCT_MIN:             f64 = 0.1;
pub const SMOOTH_PCT_DIV:             f64 = 0.05;
pub const STATIC_MOTION_THRESH:       u32 = 95;

/// `MAX_LAG_BUFFERS = 25` (`vp9_lookahead.h:32`). Used here only to
/// dimension `avg_source_sad[]`; lookahead itself is 1-slot in v1
/// per L-30 (`lag_in_frames = 0`).
pub const MAX_LAG_BUFFERS: usize = 25;

pub const QINDEX_RANGE: usize = 256;
pub const MAXQ: u8 = 255;
pub const MINQ: u8 = 0;

pub const FRAME_SCALE_STEPS: usize = 2;
/// `frame_scale_factor[]` (`vp9_ratectrl.h:79`).
pub const FRAME_SCALE_FACTOR: [u8; FRAME_SCALE_STEPS] = [16, 24];
/// `rate_thresh_mult[]`.
pub const RATE_THRESH_MULT:   [f64; FRAME_SCALE_STEPS] = [1.0, 2.0];
/// `rcf_mult[]`. Applied to RCF on read & inverse on write (L-25).
pub const RCF_MULT:           [f64; FRAME_SCALE_STEPS] = [1.0, 2.0];

/// `rate_factor_deltas[]` (`vp9_ratectrl.c:1306-1312`).
pub const RATE_FACTOR_DELTAS: [f64; RateFactorLevel::COUNT] = [
    1.00, // InterNormal
    1.00, // InterHigh
    1.50, // GfArfLow
    1.75, // GfArfStd
    2.00, // KfStd
];

/// VBR fixed-pattern (FIXED_GF_INTERVAL=8) delta-rate weights. Only
/// reachable in VPX_Q mode (not used in CBR or VBR-RT default). Kept
/// for `notes/lld/svc.md` reference.
pub const VBR_DELTA_RATE: [f64; FIXED_GF_INTERVAL as usize] =
    [0.50, 1.0, 0.85, 1.0, 0.70, 1.0, 0.85, 1.0];
```

### 4.3 Min-Q LUTs (`init_minq_luts`)

Per A5 §4.5 these are *generated*, not transcribed. Six LUTs of 256
entries each; coefficients from `vp9_ratectrl.c:126-138`. **8-bit only
(D-01)** — we drop the HBD-conditional triple-instantiation.

```rust
/// One min-Q LUT: 256 entries, computed at `RateControl::new` from a
/// cubic polynomial (`vp9_ratectrl.c:104-124`). Stored as `[u8; 256]`
/// so the `regulate_q` inner loop reads are cache-line tight.
pub type MinqLut = [u8; QINDEX_RANGE];

/// All six LUTs in one block. Built once per encoder; rebuilt on
/// `reconfigure` only when `worst_allowed_q` or `best_allowed_q`
/// change (the LUT contents are bit-depth × index, and bit-depth is
/// fixed at 8 per D-01, so the *content* is encoder-lifetime
/// immutable, but the encoder may instantiate multiple LUT sets if we
/// later support a "Q range" change at runtime).
#[derive(Debug, Clone)]
pub struct MinqLuts {
    pub kf_low:    MinqLut,
    pub kf_high:   MinqLut,
    pub arfgf_low: MinqLut,    // non-AGGRESSIVE_VBR coefficients only (we don't
                               // ship AGGRESSIVE_VBR; D-13 + L-32 imply REALTIME)
    pub arfgf_high: MinqLut,
    pub inter:     MinqLut,
    pub rtc:       MinqLut,
}

impl MinqLuts {
    pub fn build() -> Self;       // 8-bit only; D-01
}
```

The coefficients applied at `MinqLuts::build`:

| LUT          | x3        | x2        | x1   |
|--------------|-----------|-----------|------|
| `kf_low`     | 0.000001  | -0.0004   | 0.150 |
| `kf_high`    | 0.0000021 | -0.00125  | 0.45  |
| `arfgf_low`  | 0.0000015 | -0.0009   | 0.30  |
| `arfgf_high` | 0.0000021 | -0.00125  | 0.55  |
| `inter`      | 0.00000271| -0.00113  | 0.70  |
| `rtc`        | 0.00000271| -0.00113  | 0.70  |

Same arithmetic as libvpx `init_minq_luts` per A5 §5.11:

```
maxq = ac_qlookup[i] / 4.0              // 8-bit
minqtarget = min(((x3*maxq + x2)*maxq + x1)*maxq, maxq)
if minqtarget <= 2.0: minqtarget = 0    // lossless special-case
LUT[i] = smallest j s.t. ac_qlookup[j] / 4.0 >= minqtarget
```

Float arithmetic ordering matches libvpx; see §5.10 for the determinism
note (L-04).

### 4.4 SVC layer-context plumbing (slim)

`notes/lld/svc.md` owns the full layer iteration. This LLD provides the
two operations it needs:

```rust
/// Per-layer RC slice. Owned by `LayerContext` (HLD-svc §2.2). A SVC
/// encoder owns `Vec<LayerContext>` of length `NSL * NTL`.
#[derive(Debug, Clone)]
pub struct LayerRc {
    pub rc: RateControl,                  // independent state machine
    /// `framedrop_thresh[spatial_layer_id]` — applied during
    /// `vp9_test_drop` for SVC.
    pub framedrop_thresh: u8,
    /// `consec_drop` counter — capped at `cfg.max_consec_drop`.
    pub consec_drop: u32,
}

impl RateControl {
    /// SVC layer-context swap (`vp9_save_layer_context`/`vp9_restore_layer_context`
    /// equivalent). The semantics in libvpx are *full memcpy of `cpi->rc`
    /// to/from `lc->rc`*. We do the same in Rust with `Clone` — the
    /// per-layer `LayerRc.rc` is an owned `RateControl`, no `Arc`.
    /// **Performance:** ~70 fields × 8 bytes ≈ 600 bytes/copy; one
    /// copy per frame per layer.
    ///
    /// L-58: ALL fields are swapped — no per-encoder field is excluded.
    /// This differs from libvpx where some `VP9_COMP`-level (not
    /// `cpi->rc`-level) fields stay attached to the encoder; in our
    /// design those fields live elsewhere (e.g. `EncoderState.svc`)
    /// rather than `RateControl`.
    pub fn save_to_layer(&self, layer: &mut LayerRc) {
        layer.rc = self.clone();
    }
    pub fn restore_from_layer(&mut self, layer: &LayerRc) {
        *self = layer.rc.clone();
    }
}
```

### 4.5 Error types

Per HLD-arch §6:

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RcConfigError {
    #[error("worst_allowed_q ({0}) < best_allowed_q ({1})")]
    InvertedQRange(u8, u8),
    #[error("target_bandwidth_bps must be > 0")]
    ZeroBandwidth,
    #[error("framerate must be > 0.0 and finite ({0})")]
    InvalidFramerate(f64),
    /// L-76: VBR-RT with lag is out of v1 scope.
    #[error("lag_in_frames > 0 with VBR-RT is not supported in v1 (D-13/L-76)")]
    VbrRtWithLag,
    #[error("spatial_layers ({0}) > MAX_SPATIAL_LAYERS={1}")]
    TooManySpatialLayers(u8, u8),
    #[error("temporal_layers ({0}) > MAX_TEMPORAL_LAYERS={1}")]
    TooManyTemporalLayers(u8, u8),
    #[error("optimal_buffer_level_ms ({0}) > maximum_buffer_size_ms ({1})")]
    BufferBoundsInverted(u32, u32),
}
```

---

## 5. Internal algorithm

This section steps through every flow, with libvpx file:line citations
on every non-obvious step. The §3.1 control-flow listing in A5 is the
template; this section instantiates it in our Rust port's order.

### 5.1 `RateControl::new` (init flow)

Mirrors `vp9_rc_init` (`vp9_ratectrl.c:360-449`) plus framerate-derived
buffer sizing (`vp9_new_framerate`).

```
1. Validate cfg → RcConfigError or fall through.
2. zero-initialise every field via Default; the rest are explicit:
3.  Q averages:
       last_q = [worst_q, worst_q]             // both intra and inter
       avg_frame_qindex = [(best_q + worst_q + 1) / 2, worst_q]
       last_kf_qindex   = 0
       last_boosted_qindex = (best_q + worst_q + 1) / 2
4.  Boosts:
       gfu_boost  = DEFAULT_GF_BOOST = 2000
       kf_boost   = DEFAULT_KF_BOOST = 2000
       last_boost = 0
5.  GF interval:
       frames_till_gf_update_due = 0
       baseline_gf_interval     = (MIN_GF_INTERVAL + MAX_GF_INTERVAL) / 2  // 10
       min_gf_interval          = MIN_GF_INTERVAL = 4
       max_gf_interval          = MAX_GF_INTERVAL = 16
       static_scene_max_gf_interval = MAX_GF_INTERVAL = 16   // 250 only if static
6.  RCFs:
       rate_correction_factors = [1.0; 5]
       damped_adjustment       = [false; 5]
7.  Bandwidth derived (`vp9_new_framerate` / `set_rc_buffer_sizes`):
       avg_frame_bandwidth = (target_bandwidth_bps as f64 / framerate)
                             .round() as u32
       min_frame_bandwidth = max(avg_frame_bandwidth >> 4, FRAME_OVERHEAD_BITS=200)
       max_frame_bandwidth = (avg_frame_bandwidth * MAX_MB_RATE=250)
                             .min(MAXRATE_1080P=4_000_000)
                             // — caps applied per `vp9_rc_clamp_pframe_target_size`
8.  Buffer levels (ms × bps):
       starting_buffer_level = (cfg.starting_buffer_level_ms as i64 * target_bps)
                               / 1000
       optimal_buffer_level  = (cfg.optimal_buffer_level_ms  as i64 * target_bps)
                               / 1000
       maximum_buffer_size   = (cfg.maximum_buffer_size_ms   as i64 * target_bps)
                               / 1000
       buffer_level          = starting_buffer_level
9.  fac_active_worst_inter = 150
    fac_active_worst_gf    = 100
    af_ratio_onepass_vbr   = 10  // [5..15] enforced on use
10. avg_source_sad = [0; MAX_LAG_BUFFERS]
11. frame_size_selector = next_frame_size_selector = Unscaled
12. last_avg_frame_bandwidth = avg_frame_bandwidth
13. ext_use_post_encode_drop = cfg.ext_use_post_encode_drop
    use_post_encode_drop     = ext_use_post_encode_drop  // libvpx sets equal
14. force_qpmin = false; force_max_q = false
15. constrain_gf_key_freq_onepass_vbr = (cfg.rc_mode == VbrRt)
        // libvpx default for 1-pass VBR
```

### 5.2 Per-frame `pick_q` pipeline

Mirrors §3.1 of A5 verbatim. Steps:

1. **Scene-detect ingest.** Caller has already run scene detect (always
   on per L-23) and populated `inputs.high_source_sad`,
   `inputs.avg_source_sad`. We copy these into
   `self.high_source_sad = inputs.high_source_sad;
    self.avg_source_sad[(self.current_video_frame as usize) % MAX_LAG_BUFFERS]
        = inputs.avg_source_sad;`.
   When `cfg.rc_mode == Cbr` *and* `self.avg_frame_qindex[Inter]` is at
   `best_quality` *and* `high_source_sad`, we set
   `self.reset_high_source_sad = true` (per `vp9_ratectrl.c:946-947`'s
   condition; this forces `active_worst_quality = worst_quality` for the
   next frame).

2. **Compute params** — dispatch on `cfg.rc_mode`:
   - `Cbr` → `get_one_pass_cbr_params` (`vp9_ratectrl.c:2493`); sets
     `frame_type`, GF refresh flags, base/this_frame_target,
     `vp9_update_buffer_level_preencode` (line 250-256).
   - `VbrRt` → `get_one_pass_vbr_params` (`vp9_ratectrl.c:2389-2491`);
     **no altref** branches taken (D-13). GF interval logic stays — VBR
     uses golden frames as periodic refreshes; only the `source_alt_ref_*`
     fields are skipped.

3. **AQ-3 hook** (when `cfg.aq_mode == CyclicRefresh`):
   - On the first frame of a GF group:
     `vp9_cyclic_refresh_set_golden_update(cpi)` — owner
     `notes/lld/aq.md`; this LLD invokes it via `cr.set_golden_update(&self)`.
   - `vp9_cyclic_refresh_update_parameters(cpi)` — owner
     `notes/lld/aq.md`; called every frame to refresh
     `cr.percent_refresh`, `cr.rate_ratio_qdelta`, etc.

4. **Forced-KF check.** `frames_to_key == 0` or `inputs.force_keyframe`:
   `frame_type = Key`; reset key-related counters. `auto_key` not used
   in v1 (CBR is always the keyframe driver; livestream callers force
   keyframes explicitly).

5. **Compute frame target.** Branch on (`rc_mode`, `frame_type`):
   - CBR I: `vp9_calc_iframe_target_size_one_pass_cbr` (§5.3 below).
   - CBR P: `vp9_calc_pframe_target_size_one_pass_cbr` (§5.3 below).
   - VBR I: `vp9_calc_iframe_target_size_one_pass_vbr` (`vp9_ratectrl.c:2039-2048`).
   - VBR P: `vp9_calc_pframe_target_size_one_pass_vbr` (`vp9_ratectrl.c:2019-2037`).

6. **`vp9_rc_set_frame_target`** (`vp9_ratectrl.c:1713-1737`):
   ```
   this_frame_target = clamp(target, min_frame_bandwidth, max_frame_bandwidth)
   if rc_mode == VbrRt: this_frame_target = vbr_rate_correction(this_frame_target)
   base_frame_target = this_frame_target
   sb64_target_rate  = compute_sb64_target_rate()   // for AQ-3 thresh_rate_sb
   ```

7. **Compute active-worst-Q.** Branches on (`rc_mode`, frame state):
   - CBR: `calc_active_worst_quality_one_pass_cbr` (§5.4).
   - VBR-RT: `calc_active_worst_quality_one_pass_vbr` (§5.4).

8. **Compute active-best-Q and bounds.** Branches:
   - CBR: `rc_pick_q_and_bounds_one_pass_cbr` (`vp9_ratectrl.c:1002-1100`).
   - VBR-RT: `rc_pick_q_and_bounds_one_pass_vbr` (`vp9_ratectrl.c:1138-1303`),
     with the altref branches *dropped* per D-13.

9. **Small-image-format Q adjustment (L-17).** At three sites
   (`vp9_ratectrl.c:1033-1035`, 1175-1178, 1368-1371):
   ```rust
   let mut q_adj_factor: f64 = 1.0;
   if (cfg.width * cfg.height) as u64 <= (352 * 288) {
       q_adj_factor -= 0.25;
   }
   let q_val = convert_qindex_to_q(active_best_quality);
   active_best_quality = active_best_quality
       .saturating_add_signed(compute_qdelta(q_val, q_val * q_adj_factor) as i16)
       .clamp(self.best_quality as u16, self.worst_quality as u16)
       as u8;
   ```
   See §5.10 (L-04 float-determinism) — `q_val * 0.75` is the load-bearing
   float operation here; results must match libvpx's `log10` LUTs to
   produce identical qindex deltas.

10. **`regulate_q`** (§5.5).

11. **`adjust_q_cbr`** (CBR only): clamp Q to last two frames' Q if
    oscillating (`vp9_ratectrl.c:666-687`); apply
    `vp9_cyclic_refresh_limit_q` for screen content (L-24); final clamp
    to `[best_quality, worst_quality]`.

12. **`forced_key` re-check.** If `frame_is_intra_only && this_key_frame_forced`,
    `q = last_boosted_qindex` (line 1081-1083).

13. **Return `QDecision { q_index: q, bottom_q: active_best_quality,
    top_q: active_worst_quality, forced_key }`.**

### 5.3 Frame target arithmetic (CBR)

**P-frame** (`vp9_calc_pframe_target_size_one_pass_cbr`,
`vp9_ratectrl.c:2145-2192`). Note: D-13 means no altref → no
`refresh_alt_ref_frame` branching; `refresh_golden_frame` still applies
for GF-refresh frames.

```rust
let diff:        i64 = (self.optimal_buffer_level - self.buffer_level).max(0);
let one_pct_bits: i64 = 1 + self.optimal_buffer_level / 100;
let min_target:  u32 = max(self.avg_frame_bandwidth >> 4, FRAME_OVERHEAD_BITS);

let mut target: u32 = if cfg.gf_cbr_boost_pct > 0 {
    let af = cfg.gf_cbr_boost_pct + 100;
    let gi = self.baseline_gf_interval;
    if refresh_golden {
        // (avg_bw * gf_interval * af) / (gf_interval*100 + af - 100)
        ((self.avg_frame_bandwidth as u64) * (gi as u64) * (af as u64)
         / (gi as u64 * 100 + af as u64 - 100)) as u32
    } else {
        ((self.avg_frame_bandwidth as u64) * (gi as u64) * 100
         / (gi as u64 * 100 + af as u64 - 100)) as u32
    }
} else {
    self.avg_frame_bandwidth
};

// SVC overrides target with layer.avg_frame_size (owner: notes/lld/svc.md)
if let Some(layer) = svc { target = layer.avg_frame_size; min_target = max(layer.avg_frame_size >> 4, FRAME_OVERHEAD_BITS); }

let buf_diff: i64 = self.optimal_buffer_level - self.buffer_level;
if buf_diff > 0 {
    let pct_low = (buf_diff / one_pct_bits).min(cfg.under_shoot_pct as i64);
    target = ((target as i64) - (target as i64 * pct_low / 200)) as u32;
} else if buf_diff < 0 {
    let pct_high = ((-buf_diff) / one_pct_bits).min(cfg.over_shoot_pct as i64);
    target = ((target as i64) + (target as i64 * pct_high / 200)) as u32;
}

if cfg.max_inter_bitrate_pct > 0 {
    let max_rate = self.avg_frame_bandwidth as u64 * cfg.max_inter_bitrate_pct as u64 / 100;
    target = (target as u64).min(max_rate) as u32;
}

target.max(min_target)
```

Asymmetry per A5 §4.6: `buf_diff > 0` ⇒ buffer below optimal ⇒ reduce
target to refill; `buf_diff < 0` ⇒ above optimal ⇒ spend more. `/200`
(not `/100`) caps single-frame swings to ~50% of pct.

**I-frame** (`vp9_calc_iframe_target_size_one_pass_cbr`,
`vp9_ratectrl.c:2194-2221`):

```rust
if self.current_video_frame == 0 {
    return (self.starting_buffer_level / 2) as u32;
}

let kf_boost: u32 = max(32, (2.0 * cfg.framerate) as u32 - 16);
let kf_boost = if self.frames_since_key < (cfg.framerate as u32 / 2) {
    (kf_boost * self.frames_since_key) / (cfg.framerate as u32 / 2)
} else { kf_boost };

let target: u32 = (((16 + kf_boost) as u64 * self.avg_frame_bandwidth as u64) >> 4) as u32;
clamp_iframe_target(target)     // `vp9_rc_clamp_iframe_target_size`
```

`clamp_iframe_target` caps to `cfg.max_intra_bitrate_pct * avg_bw / 100`
and to `max_frame_bandwidth`.

### 5.4 Active-worst-Q (CBR + VBR-RT)

**CBR** (`calc_active_worst_quality_one_pass_cbr`, `vp9_ratectrl.c:931-1000`).
The buffer-driven curve produces `active_worst_quality` from the buffer
level relative to `optimal` and the `critical_level = optimal >> 3`
threshold (L-18).

```rust
if frame_type == Key
   || self.reset_high_source_sad
   || self.force_max_q
{
    return self.worst_quality;
}

let num_frames_weight_key: u32 = 5 * (cfg.temporal_layers as u32);

let ambient_qp: u8 = if self.current_video_frame < num_frames_weight_key {
    min(self.avg_frame_qindex[Inter], self.avg_frame_qindex[Key])
} else {
    self.avg_frame_qindex[Inter]
};

let mut active_worst = min(
    self.worst_quality,
    ((ambient_qp as u32) * 5 / 4).min(255) as u8,
);

// SVC base-layer KF re-clamp:
if svc.spatial_layer_id == 0 && svc.layer_was_key {
    let ambient_qp_svc = min(ambient_qp, svc.key_layer_last_q);
    active_worst = min(self.worst_quality,
                       ((ambient_qp_svc as u32) * 9 / 8).min(255) as u8);
}

let critical_level: i64 = self.optimal_buffer_level >> 3;    // L-18

if self.buffer_level > self.optimal_buffer_level {
    // Above optimal — relax Q.
    let max_adj_down: u32 = if cfg.content == Screen {
        active_worst as u32 >> 3
    } else {
        active_worst as u32 / 3
    };
    if max_adj_down > 0 {
        let buff_step: i64 = (self.maximum_buffer_size - self.optimal_buffer_level)
                              / (max_adj_down as i64);
        if buff_step > 0 {
            let delta: i64 = (self.buffer_level - self.optimal_buffer_level) / buff_step;
            active_worst = active_worst.saturating_sub(delta as u8);
        }
    }
} else if self.buffer_level > critical_level {
    let buff_step: i64 = self.optimal_buffer_level - critical_level;
    if buff_step > 0 {
        let span: i64 = (self.worst_quality as i64 - ambient_qp as i64).max(0);
        let from_optimal: i64 = self.optimal_buffer_level - self.buffer_level;
        active_worst = (ambient_qp as i64
                       + (span * from_optimal) / buff_step).clamp(0, 255) as u8;
    }
} else {
    // Below critical (= optimal/8). Hard floor → worst-q.
    active_worst = self.worst_quality;
}
```

**Note the integer divisions** — these are not commutative and must run
in exactly this order to match libvpx (R-04). Lines 989-991 in libvpx
do `(self.worst_quality - ambient_qp) * (self.optimal_buffer_level -
self.buffer_level) / buff_step`; we reproduce in the same order.

**VBR-RT** (`calc_active_worst_quality_one_pass_vbr`,
`vp9_ratectrl.c:905-928`):

```rust
let last_q_key   = self.last_q[Key];
let last_q_inter = self.last_q[Inter];
let avg_q_inter  = self.avg_frame_qindex[Inter];
let curr_frame   = self.current_video_frame;

let worst = match frame_type {
    Key => {
        if curr_frame == 0 { self.worst_quality }
        else { ((last_q_key as u32) << 1).min(255) as u8 }
    }
    Inter if refresh_golden /* D-13: never refresh_alt */ => {
        if curr_frame == 1 { ((last_q_key as u32 * 5 / 4).min(255)) as u8 }
        else { ((last_q_inter as u32 * self.fac_active_worst_gf / 100).min(255)) as u8 }
    }
    Inter => {
        if curr_frame == 1 { ((last_q_key as u32) << 1).min(255) as u8 }
        else { ((avg_q_inter as u32 * self.fac_active_worst_inter / 100).min(255)) as u8 }
    }
};

min(worst, self.worst_quality)
```

### 5.5 `regulate_q` — qindex search

Mirrors `vp9_rc_regulate_q` (`vp9_ratectrl.c:819-862`) verbatim. **Q-RC-4
note:** the linear scan can be binary-searched but **must not change
which qindex is selected**. We keep the linear scan + add a debug-build
property test that brute-force-scans agrees with us.

```rust
fn regulate_q(
    &self,
    target_bits_per_frame: u32,
    active_best_quality:   u8,
    active_worst_quality:  u8,
    frame_type:            FrameType,
    mbs:                   u32,
    cr:                    Option<&CyclicRefresh>,
) -> u8 {
    let target_bpm: u32 = ((target_bits_per_frame as u64) << BPER_MB_NORMBITS) as u32
                          / mbs;
    let corr:  f64 = self.get_rate_correction_factor();
    let mut q: u8 = active_worst_quality;
    let mut last_error: i64 = i64::MAX;

    for i in active_best_quality..=active_worst_quality {
        let bpm_i: u32 = if let Some(cr) = cr {
            // AQ-3 weighted bits-per-MB (notes/lld/aq.md owns this kernel).
            cr.bits_per_mb(i, corr, frame_type)
        } else {
            self.bits_per_mb(frame_type, i, corr)
        };
        if bpm_i <= target_bpm {
            // Pick the index closer to the target in BPM space.
            q = if (target_bpm - bpm_i) as i64 <= last_error {
                i
            } else {
                i.saturating_sub(1)
            };
            break;
        } else {
            last_error = (bpm_i - target_bpm) as i64;
        }
    }

    if self.cfg.rc_mode == RcMode::Cbr {
        q = self.adjust_q_cbr(q, cr);
    }
    q
}
```

`bits_per_mb` is the BPM model verbatim from A5 §4.1 (file:line
`vp9_ratectrl.c:185-196`):

```rust
fn bits_per_mb(&self, frame_type: FrameType, qindex: u8, correction_factor: f64) -> u32 {
    let q: f64 = convert_qindex_to_q(qindex);     // 8-bit (D-01): ac_qlookup[qindex] / 4.0
    let mut enumerator: i64 = match frame_type {
        FrameType::Key   => 2_700_000,
        FrameType::Inter => 1_800_000,
    };
    enumerator += (enumerator * q as i64) >> 12;
    (enumerator as f64 * correction_factor / q) as u32
}
```

`adjust_q_cbr` is the oscillation clamp + screen-content `limit_q` per
A5 §4.2:

```rust
fn adjust_q_cbr(&self, mut q: u8, cr: Option<&CyclicRefresh>) -> u8 {
    if !self.reset_high_source_sad && !self.gf_cbr_boost_active() {
        if (self.rc_1_frame * self.rc_2_frame) == -1 && self.q_1_frame != self.q_2_frame {
            let lo = min(self.q_1_frame, self.q_2_frame);
            let hi = max(self.q_1_frame, self.q_2_frame);
            q = q.clamp(lo, hi);
            if self.rc_1_frame == -1 && q > hi {
                // Was last-frame overshoot AND q > hi: halve the gap toward hi.
                q = hi + (q - hi) / 2;
            }
        }
    }
    // L-24: screen-content cyclic refresh clamp.
    if self.cfg.content == ContentMode::Screen && let Some(cr) = cr {
        q = cr.limit_q(q, self.q_1_frame);   // notes/lld/aq.md owns body
    }
    q.clamp(self.best_quality, self.worst_quality)
}
```

### 5.6 `update_rate_correction_factors`

Mirrors `vp9_rc_update_rate_correction_factors` (`vp9_ratectrl.c:737-817`).
Called from `post_encode_update` after `projected_frame_size` is set.

```rust
fn update_rate_correction_factors(&mut self, cr: Option<&CyclicRefresh>) {
    let rf_lvl: RateFactorLevel = self.current_rate_factor_level();
    let mut rcf: f64 = self.get_rate_correction_factor();

    let projected_at_q: u32 = if let Some(cr) = cr {
        // AQ-3 path: weighted bits estimate (cr.estimate_bits_at_q in notes/lld/aq.md).
        cr.estimate_bits_at_q(rcf, self.current_frame_type())
    } else {
        self.estimate_bits_at_q(self.current_frame_type(), self.base_qindex, rcf)
    };

    let correction: i32 = if projected_at_q > FRAME_OVERHEAD_BITS {
        // libvpx: (100 * projected_frame_size) / projected_at_q. Stay in i64 to avoid overflow.
        ((100i64 * self.projected_frame_size as i64) / projected_at_q as i64) as i32
    } else {
        100
    };

    let adjustment_limit: f64 = if !self.damped_adjustment[rf_lvl as usize] {
        self.damped_adjustment[rf_lvl as usize] = true;
        1.0
    } else {
        // L-04: log10 here is the deterministic-libm pivot point.
        let term = (0.01f64 * correction as f64).abs();
        let log_term = if term > 0.0 { term.log10().abs() } else { 0.0 };
        0.25 + 0.5 * log_term.min(1.0)
    };

    self.q_2_frame = self.q_1_frame;
    self.q_1_frame = self.base_qindex;
    self.rc_2_frame = self.rc_1_frame;
    self.rc_1_frame = if correction > 110 { -1 }
                      else if correction < 90 { 1 }
                      else { 0 };
    // Sign inversion per A5 §4.3: +1 = undershoot, -1 = overshoot.
    // (`correction > 110` ⇒ encoder emitted MORE bits than projected at Q
    //  ⇒ BPM model under-predicted ⇒ overshoot.)

    if self.rc_1_frame == -1 && self.rc_2_frame == 1 && correction > 1000 {
        self.rc_2_frame = 0;
    }

    if correction > 102 {
        let corr_f = 100.0 + ((correction - 100) as f64) * adjustment_limit;
        rcf = (rcf * corr_f / 100.0).min(MAX_BPB_FACTOR);
    } else if correction < 99 {
        let corr_f = 100.0 - ((100 - correction) as f64) * adjustment_limit;
        rcf = (rcf * corr_f / 100.0).max(MIN_BPB_FACTOR);
    }

    self.set_rate_correction_factor(rcf);
}
```

`get_rate_correction_factor` / `set_rate_correction_factor` mirror lines
689-735 with the **`rcf_mult[frame_size_selector]` scaling on read /
inverse on write** (L-25). Pre-scale on read; post-scale (divide) on
write. This means a resize event preserves the underlying RCF "shape"
across resolution changes; the off-by-one trap is reading & writing
must use *the same `frame_size_selector`* — which they always will, in
our port, because resize happens at frame boundaries.

```rust
fn get_rate_correction_factor(&self) -> f64 {
    let rcf: f64 = match self.current_rate_factor_level() {
        RateFactorLevel::KfStd      => self.rate_correction_factors[KfStd as usize],
        RateFactorLevel::GfArfLow   => self.rate_correction_factors[GfArfLow as usize],
        RateFactorLevel::GfArfStd   => self.rate_correction_factors[GfArfStd as usize],
        RateFactorLevel::InterHigh  => self.rate_correction_factors[InterHigh as usize],
        RateFactorLevel::InterNormal => self.rate_correction_factors[InterNormal as usize],
    };
    (rcf * RCF_MULT[self.frame_size_selector as usize]).clamp(MIN_BPB_FACTOR, MAX_BPB_FACTOR)
}

fn set_rate_correction_factor(&mut self, factor: f64) {
    let factor = factor / RCF_MULT[self.frame_size_selector as usize];
    let factor = factor.clamp(MIN_BPB_FACTOR, MAX_BPB_FACTOR);
    let lvl: RateFactorLevel = self.current_rate_factor_level();
    // KF-on-intra-only goes to KF_STD; otherwise the current level.
    self.rate_correction_factors[lvl as usize] = factor;
}
```

`current_rate_factor_level`: returns `KfStd` on intra-only frames;
otherwise GF/ARF-conditional. **Per D-13 the `is_src_frame_alt_ref`
condition is gone**; the resulting level reduces to:

```rust
fn current_rate_factor_level(&self) -> RateFactorLevel {
    if self.frame_is_intra_only() {
        return RateFactorLevel::KfStd;
    }
    let is_gf_refresh = self.refresh_golden_frame;
    if is_gf_refresh && !self.cfg.use_svc()
       && (self.cfg.rc_mode != RcMode::Cbr || self.cfg.gf_cbr_boost_pct > 100)
    {
        RateFactorLevel::GfArfStd
    } else {
        RateFactorLevel::InterNormal
    }
}
```

### 5.7 `encodedframe_overshoot` (L-19 + L-74)

This is the single most opaque state mutation in the encoder. Mirrors
`vp9_encodedframe_overshoot` (`vp9_ratectrl.c:3248-3351`) byte-for-byte.
We **must** match every step; subtle reorderings will silently drift
BD-rate on scene-change recovery (per A8 §5.1 entry 5 / L-19).

L-74 catalogues five state mutations not covered by A5's original Q-RC-20
write-up; **all five are implemented below.**

```rust
pub fn encodedframe_overshoot(
    &mut self,
    frame_size_bits: u32,
    q: &mut QDecision,
    cr: Option<&mut CyclicRefresh>,
    mut svc: Option<&mut SvcLayerRcs<'_>>,
    intra_usage_pct: Option<u32>,
) -> bool {
    // Threshold setup (`vp9_ratectrl.c:3252-3257`).
    let mut thresh_qp:   u32 = 7 * (self.worst_quality as u32 >> 3);
    let     thresh_rate: u32 = self.avg_frame_bandwidth << 3;
    if self.cfg.content != ContentMode::Screen {
        thresh_qp = 3 * (self.worst_quality as u32 >> 2);
    }

    let trigger_fast = self.cfg.overshoot_detection_cbr_rt
                       == OvershootDetection::FastDetectionMaxQ;
    let trigger_size = frame_size_bits > thresh_rate;

    if !(trigger_fast || trigger_size) || (self.base_qindex as u32) >= thresh_qp {
        return false;                      // (b) early-out path, line 3348.
    }

    // === STATE MUTATION POINT — DO NOT REORDER WITHOUT BYTE-PARITY TEST ===

    // 1. Force max-Q.
    q.q_index = self.worst_quality;

    // 2. AQ-3 reset (L-19 #2 + L-74 #2).
    if let Some(cr) = cr.as_deref_mut() {
        cr.counter_encode_maxq_scene_change = 0;
    }

    // 3. Scene-change flag (L-19 #3).
    self.re_encode_maxq_scene_change = true;

    // 4. Hybrid-intra override (L-74 #2 + #5 confluence). Only when:
    //    - overshoot mode is RE_ENCODE_MAXQ (NOT FastDetectionMaxQ)
    //    - frame_size > 2 × thresh_rate (very big)
    //    - we are on the base spatial layer
    //    - intra usage in the tentative pack > 60%
    // `intra_usage_pct` is computed by the encoder shell using
    // `cm->mi_grid_visible` per libvpx lines 3283-3295.
    if self.cfg.overshoot_detection_cbr_rt == OvershootDetection::ReEncodeMaxQ
        && frame_size_bits > (thresh_rate << 1)
        && self.current_spatial_layer_id() == 0
    {
        if let Some(pct) = intra_usage_pct {
            if pct > 60 {
                self.hybrid_intra_scene_change = true;
            }
        }
    }

    // 5. avg_frame_qindex update (L-19 #5).
    self.avg_frame_qindex[FrameType::Inter as usize] = q.q_index;

    // 6. Buffer-level snap to optimal (L-19 #6 / L-01 collapse).
    self.buffer_level = self.optimal_buffer_level;
    // libvpx writes both buffer_level AND bits_off_target; under L-01
    // collapse we write the single field.

    // 7. Oscillation history reset (L-19 #7).
    self.rc_1_frame = 0;
    self.rc_2_frame = 0;

    // 8. RCF inverse-BPM recompute (L-19 #8 + L-74 #4).
    //    *** Mutates ONLY INTER_NORMAL, not all 5 RFLs. ***
    let target_size: u32  = self.avg_frame_bandwidth;
    let target_bpm:  u32  = (((target_size as u64) << BPER_MB_NORMBITS) / self.mbs as u64) as u32;
    let q2:          f64  = convert_qindex_to_q(q.q_index);
    let mut enumerator: i64 = 1_800_000;
    enumerator += (enumerator as f64 * q2) as i64 >> 12;
    let new_corr: f64 = (target_bpm as f64) * q2 / (enumerator as f64);

    let old_corr: f64 = self.rate_correction_factors[RateFactorLevel::InterNormal as usize];
    if new_corr > old_corr {
        let bumped = (2.0 * old_corr).min(new_corr).min(MAX_BPB_FACTOR);
        self.rate_correction_factors[RateFactorLevel::InterNormal as usize] = bumped;
    }
    // (libvpx: if `new_corr <= old_corr`, leave RCF unchanged.)

    // 9. SVC propagation (L-74 #3).
    //    Lines 3327-3346: for sl ∈ [0, max(1, first_spatial_layer_to_encode)),
    //                     for tl ∈ [0, NTL):
    //      mirror (5)..(8) into LAYER_CONTEXT.rc PLUS set lrc.force_max_q = 1.
    if let Some(svc) = svc.as_deref_mut() {
        let q_inter = q.q_index;
        let new_corr_used = self.rate_correction_factors[RateFactorLevel::InterNormal as usize];
        let first_sl: u8 = svc.first_spatial_layer_to_encode().max(1);
        let ntl: u8 = svc.number_temporal_layers();
        for sl in 0..first_sl {
            for tl in 0..ntl {
                let lc: &mut LayerRc = svc.layer_mut(sl, tl);
                lc.rc.avg_frame_qindex[FrameType::Inter as usize] = q_inter;
                lc.rc.buffer_level = lc.rc.optimal_buffer_level;
                lc.rc.rc_1_frame = 0;
                lc.rc.rc_2_frame = 0;
                lc.rc.rate_correction_factors[RateFactorLevel::InterNormal as usize]
                    = new_corr_used;
                lc.rc.force_max_q = true;
            }
        }
    }

    true
}
```

**Calling protocol (encoder shell side).** (1) pick Q normally; (2)
encode and tentatively pack to learn `frame_size_bits = tentative_bytes
<< 3`; (3) when `overshoot_detection_cbr_rt == ReEncodeMaxQ &&
frame_size_bits > 2*thresh_rate && spatial_layer_id == 0`, walk
`cm.mi_grid_visible` (lines 3284-3294) for `intra_usage_pct`, else
pass `None`; (4) call `encodedframe_overshoot` — `true` ⇒ discard the
tentative pack and re-encode at `q.q_index = worst_quality`. The L-19
byte-parity gate is §9.2.

### 5.8 `post_encode_update` (full path)

Mirrors `vp9_rc_postencode_update` (`vp9_ratectrl.c:1845-1996`). One
public entry; same path for CBR and VBR. Altref paths excised (D-13).

```rust
pub fn post_encode_update(
    &mut self,
    encoded_bytes: u32,
    chosen_q: u8,
    cr: Option<&mut CyclicRefresh>,
) {
    // 1. projected_frame_size.
    self.projected_frame_size = encoded_bytes << 3;
    self.base_qindex = chosen_q;

    // 2. RCF update (§5.6).
    self.update_rate_correction_factors(cr.as_deref());

    // 3. last_q + avg_frame_qindex EMA (3:1 IIR).
    let ft_idx = self.current_frame_type() as usize;
    self.last_q[ft_idx] = chosen_q;
    let prev = self.avg_frame_qindex[ft_idx] as u32;
    // ROUND_POWER_OF_TWO(3*prev + qindex, 2):
    self.avg_frame_qindex[ft_idx] =
        (((3 * prev + chosen_q as u32) + 2) >> 2).min(255) as u8;

    // 4. Buffer-level post-encode (line 316-333).
    self.update_buffer_level_postencode(encoded_bytes);

    // 5. Rolling target/actual (lines 1923-1931).
    //    rolling_target_bits = (3*prev + this_frame_target + 2) >> 2
    self.rolling_target_bits =
        ((3 * self.rolling_target_bits as u64 + self.this_frame_target as u64 + 2) >> 2) as u32;
    self.rolling_actual_bits =
        ((3 * self.rolling_actual_bits as u64 + self.projected_frame_size as u64 + 2) >> 2) as u32;
    // long_rolling: 31:1 IIR.
    self.long_rolling_target_bits =
        ((31 * self.long_rolling_target_bits as u64 + self.this_frame_target as u64 + 16) >> 5) as u32;
    self.long_rolling_actual_bits =
        ((31 * self.long_rolling_actual_bits as u64 + self.projected_frame_size as u64 + 16) >> 5) as u32;

    // 6. Cumulatives.
    self.total_actual_bits = self.total_actual_bits.saturating_add(self.projected_frame_size as u64);
    self.total_target_bits = self.total_target_bits.saturating_add(self.this_frame_target as u64);
    self.total_target_vs_actual = self.total_target_bits as i64 - self.total_actual_bits as i64;

    // 7. GF housekeeping.
    if !self.frame_is_intra_only() {
        if self.refresh_golden_frame {
            self.frames_since_golden = 0;
            self.gop_global_index += 1;
        } else {
            self.frames_since_golden = self.frames_since_golden.saturating_add(1);
        }
        // (D-13: no altref `update_alt_ref_frame_stats` call.)
    } else {
        self.frames_since_key = 0;
        self.last_kf_qindex = chosen_q;
        self.last_boosted_qindex = chosen_q;
    }

    // 8. Frame counters (show_frame only — for non-show frames, leave alone).
    if self.show_frame() {
        self.current_video_frame = self.current_video_frame.saturating_add(1);
        self.frames_since_key = self.frames_since_key.saturating_add(1);
        if self.frames_to_key > 0 {
            self.frames_to_key -= 1;
        }
    }

    // 9. Roll bandwidth.
    self.last_avg_frame_bandwidth = self.avg_frame_bandwidth;

    // 10. Force-flags clear (line ~1990).
    self.force_max_q = false;
    self.reset_high_source_sad = false;
    self.last_post_encode_dropped_scene_change = false;
}
```

`update_buffer_level_postencode` (`vp9_ratectrl.c:316-333`):

```rust
fn update_buffer_level_postencode(&mut self, encoded_bytes: u32) {
    let encoded_bits: i64 = (encoded_bytes as i64) << 3;
    self.buffer_level -= encoded_bits;
    self.buffer_level = self.buffer_level.min(self.maximum_buffer_size);
    // Screen content + drop-disabled: clamp lower bound at -max_buffer.
    if self.cfg.content == ContentMode::Screen && self.cfg.drop_frames_water_mark == 0 {
        self.buffer_level = self.buffer_level.max(-self.maximum_buffer_size);
    }
}
```

### 5.9 Frame-drop logic

Three drop paths, per A5 §4.8:

**Pre-encode (`should_drop_pre_encode`)** — implements
`vp9_rc_drop_frame` (`vp9_ratectrl.c:614-664`) delegating to
`vp9_test_drop` (513-564).

```rust
pub fn should_drop_pre_encode(&mut self, svc: Option<&SvcSharedState>) -> DropDecision {
    // (a) SVC dependent-layer cascade — when a prior layer was dropped
    //     in CONSTRAINED or FULL_SUPERFRAME_DROP modes, this layer drops
    //     too. (L-69 / L-75 — owned by notes/lld/svc.md; this is the hook.)
    if let Some(svc) = svc {
        if svc.must_drop_dependent(self.current_spatial_layer_id()) {
            return DropDecision::Drop;
        }
    }

    // (b) Disabled.
    if self.cfg.drop_frames_water_mark == 0 {
        return DropDecision::Encode;
    }

    // (c) Hard floor.
    if self.buffer_level < 0 {
        return DropDecision::Drop;
    }

    // (d) Watermark-driven decimation.
    let drop_mark: i64 =
        (self.cfg.drop_frames_water_mark as i64 * self.optimal_buffer_level) / 100;

    if self.buffer_level > drop_mark && self.decimation_factor > 0 {
        self.decimation_factor -= 1;        // recover one step
    } else if self.buffer_level <= drop_mark && self.decimation_factor == 0 {
        self.decimation_factor = 1;         // capped at 1 in libvpx
    }

    if self.decimation_factor > 0 {
        if self.decimation_count > 0 {
            self.decimation_count -= 1;
            return DropDecision::Encode;
        }
        self.decimation_count = self.decimation_factor;
        return DropDecision::Drop;
    }
    DropDecision::Encode
}
```

**Post-encode drop** (`post_encode_drop_cbr`, lines 566-612) — D-07
RESOLVED to **off by default**. Wired behind
`cfg.ext_use_post_encode_drop` so the API surface exists for libwebrtc
adapters. Implementation in `RateControl::should_drop_post_encode`,
called by the encoder shell after pack. Mutates `force_max_q`,
`avg_frame_qindex[Inter] = worst_quality`,
`last_post_encode_dropped_scene_change = true`. **Not byte-parity-tested
in v1**; documented divergence (§7.5).

**Drop bookkeeping** — `post_encode_update_drop`
(`vp9_ratectrl.c:1998-2017`):

```rust
pub fn post_encode_update_drop(&mut self, was_svc_full_drop: bool) {
    self.current_video_frame = self.current_video_frame.saturating_add(1);
    self.frames_since_key = self.frames_since_key.saturating_add(1);
    if self.frames_to_key > 0 { self.frames_to_key -= 1; }
    self.rc_1_frame = 0;
    self.rc_2_frame = 0;
    self.last_q[FrameType::Inter as usize] = self.base_qindex;
    // SVC: when whole superframe drops in non-LAYER_DROP mode, clip
    // buffer_level back to optimal to prevent upper-layer overgrowth.
    if was_svc_full_drop && self.buffer_level > self.optimal_buffer_level {
        self.buffer_level = self.optimal_buffer_level;
    }
}
```

### 5.10 Float-determinism note (L-04 / R-04)

Three float operations are load-bearing for byte parity:

1. `q_val * q_adj_factor` in §5.2 step 9 (small-image-format adjustment).
   `q_adj_factor ∈ {1.0, 0.75}` — exactly representable; the
   multiplication is bit-exact in IEEE-754.
2. `(0.01 * correction).abs().log10()` in §5.6 (RCF damping). **log10
   differs across libms.** Mitigation per R-04: use Rust's `f64::log10`
   on all platforms (currently delegated to `libm` on most targets);
   document the dependency and reference libm vendor in CI report.
   Differential test against `libvpx`-libm output on the same input
   sequence on the **same platform** (Linux glibc).
3. `convert_qindex_to_q(q) = ac_qlookup[q] as f64 / 4.0` — integer
   divided by 4 — bit-exact.

`MinqLuts::build` uses `pow`/`log` indirectly to find the smallest j;
the **table content** is built once at encoder init and frozen. We
ship the libvpx-derived 8-bit LUT *as a baked static array* in
`rvp9-encoder::ratectrl::tables::*` (six 256-entry `[u8; 256]`s) to
side-step the build-time `log` non-determinism. Divergence-ledger entry:
§7.4 below.

### 5.11 SVC pre-encode buffer pump (`vp9_update_buffer_level_svc_preencode`)

`vp9_ratectrl.c:263-294`. For each temporal layer above the active layer,
add `avg_frame_bandwidth` to that layer's `buffer_level`, clipped at
`maximum_buffer_size`. Called by `notes/lld/svc.md::start_layer` before
this layer's `pick_q`.

```rust
pub fn update_buffer_levels_svc_preencode(
    &mut self,
    layers: &mut SvcLayerRcs<'_>,
    active_tl: u8,
    ntl: u8,
) {
    for tl in (active_tl + 1)..ntl {
        let lc: &mut LayerRc = layers.layer_mut_in_current_spatial(tl);
        let inc: i64 = lc.rc.avg_frame_bandwidth as i64;
        lc.rc.buffer_level = (lc.rc.buffer_level + inc).min(lc.rc.maximum_buffer_size);
    }
}
```

`update_layer_buffer_level_postencode` (line 298-313) is symmetric:
subtract this layer's encoded bits from upper layers' `buffer_level`.

---

## 6. SIMD dispatch

**This module has zero SIMD kernels.** All RC arithmetic is scalar
integer + a handful of `f64` operations. The denoiser kernel
`vp9_denoiser_filter` is the only RTCD entry that A5 owns; it lives in
`notes/lld/denoiser.md`.

**Source-SAD computation** for `vp9_scene_detection_onepass` (which we
call before `pick_q`) reuses `cpi->fn_ptr[BLOCK_64X64].sdf` — the SAD
kernel owned by `notes/lld/motion-estimation.md` and exposed via the
shared SAD RTCD table in `rvp9-core::simd`. This LLD doesn't define
those kernels; it only **consumes** them through a `&[VarianceFns; 13]`
reference.

No per-arch dispatch decisions here.

---

## 7. Deliberate divergences from libvpx

Each entry feeds `notes/divergence-ledger.md`. Format per the ledger
spec (see `notes/divergence-ledger.md` §"Format").

### 7.1 `buffer_level` / `bits_off_target` consolidation (L-01)

**Where:** `RateControl.buffer_level` (single field) in
`rvp9-encoder::ratectrl::state`. libvpx counterpart: `RATE_CONTROL.buffer_level`
*and* `RATE_CONTROL.bits_off_target` (`vp9_ratectrl.h:107-108`).

**What libvpx does:** keeps two `int64_t` fields that are kept identical
everywhere (TODO at `vp9_ratectrl.c:246-249` notes they should be merged).

**What rvp9 does:** collapse to a single `buffer_level: i64`. Every
write site in libvpx writes both; every read site reads either; we
write the one and read the one.

**Why:** libvpx's own TODO. No algorithm ever reads the two
independently. Risk of divergence: zero.

**Byte-parity impact:** none.

**Compression-efficiency impact:** none.

**Future plan:** none — keep collapsed.

### 7.2 No altref / no `lag_in_frames > 0` (D-13 / L-76)

**Where:** every altref-conditional branch in `RateControl` /
`encodedframe_overshoot` / SVC fan-out. libvpx counterpart: every
`is_src_frame_alt_ref` / `refresh_alt_ref_frame` / `source_alt_ref_*`
branch in `vp9_ratectrl.c`.

**What libvpx does:** synthesizes an altref every `MAX_GF_INTERVAL`
frames when `lag_in_frames >= 4`; uses `LIMIT_QRANGE_FOR_ALTREF_AND_KEY`
to tighten Q on altref; uses
`source_alt_ref_pending`/`source_alt_ref_active` to drive frame-type
selection.

**What rvp9 does:** never. `RateControl::new` rejects
`(VbrRt, lag_in_frames > 0)` with `RcConfigError::VbrRtWithLag`. The
altref-conditional fields (`source_alt_ref_pending`, `is_src_frame_alt_ref`,
etc.) are absent from `RateControl`. `LIMIT_QRANGE_FOR_ALTREF_AND_KEY`
collapses to "tighten on key only" (the inter-GF tightening branch is
dead).

**Why:** PLAN.md D-13. Altref requires lookahead which directly
contradicts real-time / WebRTC. Stage 4d / Stage 4e do not need altref
for SVC.

**Byte-parity impact:** structural. Any libvpx bitstream produced with
altref+lag will not byte-match ours; our config never enables altref,
so when both encoders run our config they produce comparable streams.

**Compression-efficiency impact:** material **for offline content**
(no real-time deadline) — altref typically buys 5-15% BD-rate. For
real-time content the budget for altref doesn't exist, so the impact
is zero in our target use case.

**Future plan:** Stage 6+ — "near-real-time livestream" preset may add
altref + small lag. Out of v1 scope.

### 7.3 EXT_RATECTRL `max_frame_size = Bits(_)` ignored

**Where:** `reconcile_external_decision` in
`rvp9-encoder::ratectrl::ext`. libvpx counterpart:
`vpx_rc_encodeframe_decision_t::max_frame_size` consumed by the recode
loop in `encode_with_recode_loop` (which we do not port — L-34).

**What libvpx does:** `vp9_encoder.c` enters `encode_with_recode_loop`
on non-real-time configurations; the recode loop re-encodes until size
is under `max_frame_size`.

**What rvp9 does:** ignore `MaxFrameSize::Bits(_)`. `MaxFrameSize::NoLimit`
and `MaxFrameSize::VP9Default` are honoured but trivially (we don't
recode regardless).

**Why:** L-34: `encode_with_recode_loop` is dead at our cpu-used
6-8 setting. Real-time has no recode budget.

**Byte-parity impact:** none for in-scope (real-time) configs.

**Compression-efficiency impact:** none.

**Future plan:** revisit when "near-real-time livestream" preset lands
(may add recode loop for big keyframes).

### 7.4 Min-Q LUTs baked, not generated at init

**Where:** `rvp9-encoder::ratectrl::tables::{kf_low_minq, kf_high_minq,
arfgf_low_minq, arfgf_high_minq, inter_minq, rtc_minq}`. libvpx
counterpart: `vp9_init_minq_luts` at `vp9_ratectrl.c:142-154` which
calls `pow` and `log` to populate the LUTs at encoder boot.

**What libvpx does:** runs `init_minq_luts` once per `vpx_codec_enc_init`;
table contents depend on `pow`/`log` implementations.

**What rvp9 does:** baked static `[u8; 256]` arrays produced by a
build-time codegen script (`build.rs`) that **runs libvpx's exact
arithmetic** with libvpx's exact compiler. Output committed to the
repo so production builds are LUT-deterministic without depending on
the host's libm.

**Why:** L-04 / R-04 — float determinism. libvpx's `log10` calls in
`init_minq_luts` are the dominant non-determinism vector at encoder
init; baking eliminates it.

**Byte-parity impact:** none if codegen matches libvpx exactly (CI gate
verifies). Diagnostic note: differences could leak through if libvpx is
ever upgraded and we forget to regenerate.

**Compression-efficiency impact:** none.

**Future plan:** keep baked. Add a CI gate that re-runs `init_minq_luts`
under libvpx + compares.

### 7.5 `post_encode_drop_cbr` not byte-parity-tested in v1 (D-07)

**Where:** `RateControl::should_drop_post_encode`. libvpx counterpart:
`post_encode_drop_cbr` (`vp9_ratectrl.c:566-612`).

**What libvpx does:** ships the function; default-off (opt-in via
`ext_use_post_encode_drop`).

**What rvp9 does:** ships the function; default-off; **byte-parity
tested only as far as "default-off path is never taken"**; no
byte-parity gate when the feature is enabled.

**Why:** D-07 (kept off by default, matches libvpx); ensures we don't
spend Stage 4 cycles on a feature that no in-scope user enables.

**Byte-parity impact:** none for default config.

**Compression-efficiency impact:** none for default config; when
enabled by an external user, divergence from libvpx is possible.

**Future plan:** add byte-parity gate when libwebrtc integration
exercises the feature.

### 7.6 Dead altref fields elided from the on-disk `RateControl` layout

**Where:** `RateControl` struct in `rvp9-encoder::ratectrl::state`
omits `source_alt_ref_pending`, `source_alt_ref_active`,
`is_src_frame_alt_ref`, `alt_ref_gf_group`,
`last_frame_is_src_altref`. libvpx counterpart: present in
`RATE_CONTROL` (`vp9_ratectrl.h:104-118`).

**What libvpx does:** keeps them initialised to 0; reads only when
`lag_in_frames > 0`.

**What rvp9 does:** absent.

**Why:** L-32 / D-13. These fields are dead at our config.

**Byte-parity impact:** none for our config.

**Compression-efficiency impact:** none for our config.

**Future plan:** if Stage 6+ adds altref, re-add at the same struct
positions (libvpx's layout) to preserve byte-parity audits.

### 7.7 Linear qindex search retained (Q-RC-4)

**Where:** `regulate_q` in §5.5. libvpx counterpart: same.

**What libvpx does:** `O(QINDEX_RANGE)` linear scan from
`active_best_quality` to `active_worst_quality`.

**What rvp9 does:** same — linear scan. **Not** binary search.

**Why:** the off-by-one in `(target_bpm - bpm_i) <= last_error` (line
848-850) is subtle and we keep it identical. A binary search would
need to replicate the exact break/back-off semantics.

**Byte-parity impact:** none.

**Compression-efficiency impact:** none.

**Future plan:** re-evaluate if profiling shows `regulate_q` is a
bottleneck. It is called once per frame so cost is negligible.

### 7.8 Float determinism — single libm choice

**Where:** `update_rate_correction_factors` (`log10` call at
`vp9_ratectrl.c:778`); see §5.10. Min-Q LUTs are baked per §7.4 so they
don't count.

**What libvpx does:** calls C `log10` (platform libm).

**What rvp9 does:** calls Rust `f64::log10` which delegates to libm on
most targets. We document the dependency in `divergence-ledger.md` and
mark it as a known byte-parity caveat across (Linux glibc) vs
(Windows MSVC) vs (macOS).

**Why:** L-04 / R-04.

**Byte-parity impact:** minor — single-bit differences in
`adjustment_limit` propagating to RCF after many frames.

**Compression-efficiency impact:** sub-0.01% BD-rate drift (estimate).

**Future plan:** if Stage 5 CI shows cross-platform divergence > 0.05%
BD-rate, replace `log10` with a vendored libm shim.

---

## 8. Open questions

These are the things the LLD writer surfaces for Arthur or another LLD
writer to resolve.

### 8.1 Dead altref fields — elide or keep?

§7.6 elides them. Alternative: keep them as `pub(crate)` fields, never
read or written, for grep-equivalence with libvpx.

**Recommendation:** elide. Grep-equivalence isn't load-bearing once
LLDs are written, and dead fields are footguns.

**Decision needed by:** Stage 4a start.

### 8.2 `vp9_compute_qdelta_by_rate` location

Used by AQ-3 (`compute_deltaq`) and by KF Q-delta logic in this LLD
(small-image-format §5.2 step 9). Owner is ambiguous: AQ-3 owns the
caller, this LLD has the simpler use. **Recommendation:** put it in
this LLD as `RateControl::compute_qdelta_by_rate` (a method), exported
to `notes/lld/aq.md` via a shared visibility. Defer to `aq.md` for
sign-off.

### 8.3 Drop-during-overshoot interaction (Q-RC-13)

When `encodedframe_overshoot` returns `true` and the re-encode result
*still* overshoots, what happens? libvpx doesn't recurse; the
re-encode at max-Q is the final answer. But the CR state has been
mutated (`counter_encode_maxq_scene_change = 0`); if a second
overshoot happens on the *next* frame, it triggers again.
**Recommendation:** match libvpx exactly — no recursion; rely on the
next frame's normal path to recover.

### 8.4 `lag_in_frames = 0` enforcement site

The check is in `RateControl::new` per §4.5 — but the encoder API also
has a `LookaheadCtx` that the user could populate independently. **Risk:**
user constructs RC with `lag = 0` but then queues 4 frames in lookahead.
**Recommendation:** also validate at `Encoder::receive_raw_frame`
(`notes/lld/frame-buffers.md` LLD's responsibility) — reject if
queue exceeds 1 deep in real-time mode.

### 8.5 `current_video_frame` overflow (u32 wrap at ~16M frames @ 30fps ≈ 6.4 days)

libvpx uses `unsigned int`; we use `u32`. **Recommendation:** keep
`u32` and `saturating_add`. Wraparound at 6.4 days of continuous
encode is not a real-time concern. Document and move on.

### 8.6 `ext_use_post_encode_drop` API plumbing

D-07 RESOLVED off-by-default, but the libwebrtc adapter may need it
enabled. **Recommendation:** expose as a `RcConfig` flag (already in
§3.1); document that enabling it disables the byte-parity gate for
that config slice.

---

## 9. Test strategy

### 9.1 Property tests

- **`regulate_q` exhaustive** — for every (`frame_type`, `mbs ∈
  {20, 100, 1000, 8160}`, `rcf ∈ {0.005, 0.5, 1.0, 5.0, 50.0}`,
  `target_bits ∈ {200, 1000, 100_000, 1_000_000}`, `abest ∈ 0..=255`,
  `aworst ∈ abest..=255`), our `regulate_q` returns the same qindex as
  a brute-force scan implementation. ~10^8 cases; sampled in CI.
- **`bits_per_mb` brute-force** — for every (`frame_type`, `qindex ∈
  0..256`, `correction ∈ {0.005, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 50.0}`),
  our `bits_per_mb` matches a direct C transcription of
  `vp9_ratectrl.c:185-205` to within 0 ULP (the arithmetic is
  bit-exact integer + `f64`).
- **`buffer_level` invariants** — `0 ≤ buffer_level ≤ maximum_buffer_size`
  for default config (non-screen). For screen content with drop disabled:
  `-maximum_buffer_size ≤ buffer_level ≤ maximum_buffer_size`.
- **`damped_adjustment[]`** — for every `rf_lvl ∈ 0..5`, the first
  update sets `adjustment_limit = 1.0` and toggles the flag to `true`;
  subsequent updates use the damped formula.
- **L-04 small-image-format Q adjustment** — for every
  `(width, height)` ∈ {(176,144), (352,288), (640,360), (1280,720)},
  the q_adj_factor is 0.75 only for the first two.

### 9.2 Differential tests vs libvpx

- **Single-clip RC trace** — encode a 600-frame clip with libvpx CBR
  at the same config; capture every per-frame `(base_qindex,
  buffer_level, this_frame_target, projected_frame_size, rcf_inter_normal,
  high_source_sad)`. Feed the same `(encoded_bytes)` post-encode into
  our `RateControl`. Assert byte-for-byte field equality. The clip must
  include a scene change to exercise `encodedframe_overshoot`.
- **`encodedframe_overshoot` byte-parity** — the L-19 / L-74 byte-parity
  gate. Sample 5 scene-change traces from libvpx + the AOM scene-change
  corpus. For each, force the same trigger conditions in our RC, call
  `encodedframe_overshoot`, and assert every output mutation matches
  exactly.
- **Frame-drop trace** — for a buffer-stress clip (low bandwidth,
  high motion), capture every `vp9_test_drop` decision and
  `decimation_factor` from libvpx; assert ours match exactly.
- **VBR-RT trace** — encode a variable-rate clip with libvpx VBR-RT
  (CBR target with VBR adjust); assert RCF, target, and Q traces match.

### 9.3 SVC integration tests

- **Layer-context swap byte-parity** — for each (sl, tl), after
  `save_to_layer` then `restore_from_layer`, the `RateControl` is
  bit-equal to the original.
- **SVC overshoot propagation (L-74 #3)** — trigger
  `encodedframe_overshoot` with 3 spatial layers active; assert each
  layer's `rc.force_max_q`, `rc.buffer_level`, `rc.avg_frame_qindex[Inter]`,
  `rc.rate_correction_factors[InterNormal]` was mutated.
- **Cross-layer frame-drop chain (L-75)** — owned by
  `notes/lld/svc.md`; this LLD provides the `should_drop_pre_encode`
  hook tested there.

### 9.4 EXT_RATECTRL conformance

- **In-process trait fallback** — install an `impl ExternalRateCtrl`
  that returns `q_index = None` for every frame; assert the chosen Q
  equals what our internal `pick_q` produced (no override).
- **In-process trait override** — install an `impl ExternalRateCtrl`
  that returns `q_index = Some(64)`; assert every encoded frame uses
  `base_qindex = 64` (after clamp to `[layer_min_q, layer_max_q]`).
- **`RateControlRtc` byte-parity** — drive both
  `VP9RateControlRTC` (via FFI to libvpx) and our `RateControlRtc`
  through the same `compute_qp` / `post_encode_update` sequence.
  Assert byte-for-byte equality of QP / loopfilter level / drop
  verdict. **This is the libwebrtc contract test.**

### 9.5 Float-determinism CI

Per §5.10 / §7.8: a CI job on Linux glibc records the RCF sequence for
a fixed input clip; another CI job on Windows MSVC and macOS does the
same; the three sequences must match exactly **for the libvpx-libm
baseline** computation. Any divergence is logged but not gate-blocking
unless > 0.05% BD-rate drift on the test corpus.

### 9.6 Min-Q LUT codegen verification

A build-time step regenerates `kf_low_minq` etc. and verifies bit-equal
output against the committed `[u8; 256]` static. If different (libvpx
upgrade upstream changed `init_minq_luts`), CI fails with a "regenerate
min-Q LUTs" message. The codegen lives in
`crates/rvp9-encoder/build.rs` per HLD-arch §11.5.

### 9.7 Stage 4a end-to-end RC parity gate

PLAN.md Stage 4a exit: "BD-rate within 2% of libvpx at matched config".
This LLD's contribution: produce per-frame Q decisions and buffer-level
trajectories indistinguishable from libvpx on a 5-clip CBR corpus.
Concretely, the differential bitstream test in `tests/differential/diff-bitstream.rs`
should be byte-clean for the first 100 frames of each test clip (no
scene changes) and divergence-after-scene-change should match libvpx's
re-encode-maxq trigger frames exactly.

### 9.8 Stage 4b AQ-3 integration test

Per PLAN.md Stage 4b exit: "BD-rate gain from AQ-3 matches libvpx's gain
to within 0.3%". This LLD's responsibility: ensure
`vp9_cyclic_refresh_rc_bits_per_mb` is consumed correctly inside
`regulate_q` (§5.5) and `vp9_cyclic_refresh_estimate_bits_at_q` inside
`update_rate_correction_factors` (§5.6). The AQ-3 segment-id machinery
is tested in `notes/lld/aq.md`; this LLD tests only the RC-side
integration.

### 9.9 Property: rate-correction factor never escapes `[MIN_BPB_FACTOR,
MAX_BPB_FACTOR]`

After any sequence of `post_encode_update` calls, for every
`RateFactorLevel`,
`MIN_BPB_FACTOR ≤ rate_correction_factors[lvl] ≤ MAX_BPB_FACTOR` (per
Q-RC-6). Run for 10^4 random `(encoded_bytes, target, q)` sequences.

### 9.10 Determinism property

For every sequence of public API calls in any order valid per §3.2's
contract, executing the same sequence twice on two fresh `RateControl`s
produces bit-equal state at every step. Single-threaded by design
(HLD-threading §5.2); this property is mechanical but cheap to assert.

---

## 10. Cross-LLD dependencies

This LLD is consumed by:

- `notes/lld/svc.md` — calls `pick_q`, `post_encode_update`,
  `save_to_layer`, `restore_from_layer`, `update_buffer_levels_svc_preencode`,
  `encodedframe_overshoot`. Owns layer iteration (HLD-svc §2.3).
- `notes/lld/aq.md` — provides `CyclicRefresh::bits_per_mb`,
  `estimate_bits_at_q`, `limit_q`, `counter_encode_maxq_scene_change`
  field access. Consumed by `regulate_q`, `update_rate_correction_factors`,
  `adjust_q_cbr`, `encodedframe_overshoot`.
- `notes/lld/denoiser.md` — provides `NoiseLevel`; this LLD reads it
  via `FrameRcInputs.noise_level`.
- `notes/lld/mode-decision.md` — reads `RateControl::this_frame_target_bits`
  via the encoder shell.

This LLD consumes:

- `rvp9-core::ratectrl_ext` — the `ExternalRateCtrl` trait definition
  (HLD-arch §8.4).
- `rvp9-core::quant` — `ac_qlookup` (8-bit only) for
  `convert_qindex_to_q`. Per `notes/lld/quant.md`.
- `notes/lld/aq.md` — see above.
- `notes/lld/svc.md` for `SvcSharedState`, `SvcLayerRcs`, layer iteration.

---

## 11. Summary (one-line)

**Line count:** ~2160 lines (over the 1200-1800 band — RC is the
broadest surface in Stage 2; pseudocode in §5 carries the L-19/L-74
byte-parity gate and the CBR + VBR-RT regulate_q + RCF + buffer-model
arithmetic verbatim). **§7 entries:** 8 — L-01 buffer_level collapse;
D-13 no-altref; EXT_RATECTRL `MaxFrameSize::Bits(_)` ignored; baked
Min-Q LUTs; post-encode-drop not byte-parity-tested in v1 (D-07);
elided dead altref fields; linear qindex search retained; single libm
choice / float determinism (L-04). **§8 blocking questions:** none for
Stage 3 (this LLD blocks Stage 4a, not Stage 3; RC is encoder-only).
§8.1-§8.6 are recommendations with proposed answers pending Arthur
sign-off; none gate Stage 3 or Stage 4a entry.
