# Rate control + denoiser + adaptive quantization in libvpx VP9

Specialist: A5. Reference: `third_party/libvpx/` pinned at v1.14.1 commit
`12f3a2a`. All file:line citations are into that tree.

## 1. Scope

This document covers the three load-bearing subsystems that decide how many
bits a frame gets and how they are spent inside the frame:

- **1-pass rate control** (CBR and VBR), including the buffer model,
  active-Q range selection, the Q-index search, frame-drop logic and the
  scene-detection scaffolding around them. Source of truth:
  `vp9/encoder/vp9_ratectrl.{h,c}` (3351 lines).
- **Content-aware temporal denoiser**, including the noise estimator that
  drives it. Source of truth: `vp9/encoder/vp9_denoiser.{h,c}` (839 + 132
  lines) and `vp9/encoder/vp9_noise_estimate.{h,c}` (302 + 54 lines).
- **Adaptive quantization** modes 1/2/3 (and 4, 360-equirect): per-segment
  delta-Q assignment driven by source statistics. Source of truth:
  `vp9/encoder/vp9_aq_variance.{h,c}`, `vp9/encoder/vp9_aq_complexity.{h,c}`,
  `vp9/encoder/vp9_aq_cyclicrefresh.{h,c}`, `vp9/encoder/vp9_aq_360.{h,c}`.
  AQ modes 5–7 (PERCEPTUAL, PSNR, LOOKAHEAD) are 2-pass / experimental and
  out of scope for our real-time target (D1, D4); listed here only for
  completeness.

Out of scope (covered by other specialists): 2-pass RC + `vp9_firstpass.c`
(A1 / A8 will treat 2-pass as deferred); SVC layer rate-control mechanics
(A6); mode-decision integration with cyclic refresh's segment_id (A2);
top-level encode loop wiring (A1); the speed-features that toggle
denoiser/AQ paths (A6).

The interaction of rate control with mode decision and partition selection
is the trickiest part of this stack; it is the single most likely source
of silent BD-rate regression in our port. Open questions surfaced in §7
are unusually load-bearing.

## 2. Data structures

### 2.1 `RATE_CONTROL` (the godhead)

Definition: `vp9/encoder/vp9_ratectrl.h:88-218`. Single instance lives in
`VP9_COMP.rc`; per-SVC-layer instances live in `LAYER_CONTEXT.rc`. Lifetime
matches the encoder; zero-initialised then populated by `vp9_rc_init()`
(`vp9_ratectrl.c:360-449`). No internal locking — all access is from the
encoder thread that owns the layer.

```c
typedef struct {
  // Rate targeting variables
  int base_frame_target;  // A baseline frame target before adjustment
                          // for previous under or over shoot.
  int this_frame_target;  // Actual frame target after rc adjustment.
  int projected_frame_size;
  int sb64_target_rate;
  int last_q[FRAME_TYPES];  // Separate values for Intra/Inter
  int last_boosted_qindex;  // Last boosted GF/KF/ARF q
  int last_kf_qindex;       // Q index of the last key frame coded.

  int gfu_boost;
  int last_boost;
  int kf_boost;

  double rate_correction_factors[RATE_FACTOR_LEVELS];

  int frames_since_golden;
  int frames_till_gf_update_due;
  int min_gf_interval;
  int max_gf_interval;
  int static_scene_max_gf_interval;
  int baseline_gf_interval;
  int constrained_gf_group;
  int frames_to_key;
  int frames_since_key;
  int this_key_frame_forced;
  int next_key_frame_forced;
  int source_alt_ref_pending;
  int source_alt_ref_active;
  int is_src_frame_alt_ref;

  int avg_frame_bandwidth;  // Average frame size target for clip
  int min_frame_bandwidth;  // Minimum allocation used for any frame
  int max_frame_bandwidth;  // Maximum burst rate allowed for a frame.

  int ni_av_qi;
  int ni_tot_qi;
  int ni_frames;
  int avg_frame_qindex[FRAME_TYPES];
  double tot_q;
  double avg_q;

  int64_t buffer_level;
  int64_t bits_off_target;
  int64_t vbr_bits_off_target;
  int64_t vbr_bits_off_target_fast;

  int decimation_factor;
  int decimation_count;

  int rolling_target_bits;
  int rolling_actual_bits;

  int long_rolling_target_bits;
  int long_rolling_actual_bits;

  int rate_error_estimate;

  int64_t total_actual_bits;
  int64_t total_target_bits;
  int64_t total_target_vs_actual;

  int worst_quality;
  int best_quality;

  int64_t starting_buffer_level;
  int64_t optimal_buffer_level;
  int64_t maximum_buffer_size;

  // rate control history for last frame(1) and the frame before(2).
  // -1: undershot
  //  1: overshoot
  //  0: not initialized.
  int rc_1_frame;
  int rc_2_frame;
  int q_1_frame;
  int q_2_frame;
  // Keep track of the last target average frame bandwidth.
  int last_avg_frame_bandwidth;

  // Auto frame-scaling variables.
  FRAME_SCALE_LEVEL frame_size_selector;
  FRAME_SCALE_LEVEL next_frame_size_selector;
  int frame_width[FRAME_SCALE_STEPS];
  int frame_height[FRAME_SCALE_STEPS];
  int rf_level_maxq[RATE_FACTOR_LEVELS];

  int fac_active_worst_inter;
  int fac_active_worst_gf;
  uint64_t avg_source_sad[MAX_LAG_BUFFERS];
  uint64_t prev_avg_source_sad_lag;
  int high_source_sad_lagindex;
  int high_num_blocks_with_motion;
  int alt_ref_gf_group;
  int last_frame_is_src_altref;
  int high_source_sad;
  int count_last_scene_change;
  int hybrid_intra_scene_change;
  int re_encode_maxq_scene_change;
  int avg_frame_low_motion;
  int af_ratio_onepass_vbr;
  int force_qpmin;
  int reset_high_source_sad;
  double perc_arf_usage;
  int force_max_q;
  // Last frame was dropped post encode on scene change.
  int last_post_encode_dropped_scene_change;
  // Enable post encode frame dropping for screen content. Only enabled when
  // ext_use_post_encode_drop is enabled by user.
  int use_post_encode_drop;
  // External flag to enable post encode frame dropping, controlled by user.
  int ext_use_post_encode_drop;
  // Flag to disable CBR feature to increase Q on overshoot detection.
  int disable_overshoot_maxq_cbr;
  int damped_adjustment[RATE_FACTOR_LEVELS];
  double arf_active_best_quality_adjustment_factor;
  int arf_increase_active_best_quality;

  int preserve_arf_as_gld;
  int preserve_next_arf_as_gld;
  int show_arf_as_gld;

  // Flag to constrain golden frame interval on key frame frequency for 1 pass
  // VBR.
  int constrain_gf_key_freq_onepass_vbr;

  // The index of the current GOP. Start from zero.
  // When a key frame is inserted, it resets to zero.
  int gop_global_index;
} RATE_CONTROL;
```

Field-by-field semantics (the ones a Rust port needs to track precisely;
the rest follow naturally):

- **`base_frame_target` / `this_frame_target` / `projected_frame_size`** —
  the bit budget pipeline for the *current* frame. `base_frame_target` is
  set by `vp9_set_target_rate()` (`vp9_ratectrl.c:2722`) from the GF-group
  allocator; `this_frame_target` is what is *actually* fed to `regulate_q`
  after under/overshoot correction (`vp9_rc_set_frame_target()`,
  `vp9_ratectrl.c:1713-1737`); `projected_frame_size` is filled in by
  `vp9_rc_postencode_update()` (`vp9_ratectrl.c:1856`) and is the
  bits-per-frame actually emitted.
- **`last_q[FRAME_TYPES]` / `avg_frame_qindex[FRAME_TYPES]`** — exponential
  moving averages over qindex, separate intra / inter. Updated post-encode
  (`vp9_ratectrl.c:1862-1891`) with weight 3/4 old + 1/4 new
  (`ROUND_POWER_OF_TWO(3*avg + qindex, 2)`).
- **`rate_correction_factors[RATE_FACTOR_LEVELS]`** — per-rate-factor-level
  multiplier used by `vp9_rc_bits_per_mb()`. 5 levels:
  `INTER_NORMAL/INTER_HIGH/GF_ARF_LOW/GF_ARF_STD/KF_STD`
  (`vp9_ratectrl.h:48-55`). Initialised to 1.0
  (`vp9_ratectrl.c:419-422`); updated by
  `vp9_rc_update_rate_correction_factors()` after every encoded frame
  (`vp9_ratectrl.c:737-817`). Clamped to `[MIN_BPB_FACTOR=0.005,
  MAX_BPB_FACTOR=50]` (`vp9_ratectrl.c:44-45`).
- **`buffer_level` / `bits_off_target`** — leaky-bucket buffer state, in
  bits. **Both fields are kept in sync** and are essentially aliases (see
  TODO at `vp9_ratectrl.c:246-249`). Pre-encode: `+= avg_frame_bandwidth`
  then clipped to `maximum_buffer_size` (`vp9_ratectrl.c:250-256`).
  Post-encode: `-= projected_frame_size` then clipped (clipped to
  `[-maximum_buffer_size, maximum_buffer_size]` for screen content,
  `[-INT64_MIN, maximum_buffer_size]` otherwise — see lines 316-333).
- **`vbr_bits_off_target` / `vbr_bits_off_target_fast`** — VBR
  long/short-term cumulative bit error. The "fast" variant is used to
  reclaim huge undershoot in one or two frames (see `vbr_rate_correction`
  at `vp9_ratectrl.c:2679-2720`).
- **`decimation_factor` / `decimation_count`** — frame-drop pacing. When
  buffer is between 0 and `drop_mark`, drop every Nth frame
  (`vp9_test_drop()`, `vp9_ratectrl.c:513-564`). Currently capped at
  factor=1 (drop every other frame).
- **`rolling_target_bits` / `rolling_actual_bits`** — IIR average bit
  spending over ~4-frame window (weight 3:1, line 1923-1926).
  `long_rolling_*` are a longer-window variant (weight 31:1, line
  1927-1931).
- **`worst_quality` / `best_quality`** — qindex range from
  `oxcf.worst_allowed_q` / `oxcf.best_allowed_q`. These bound everything.
- **`starting_buffer_level` / `optimal_buffer_level` / `maximum_buffer_size`**
  — computed from `oxcf.starting_buffer_level_ms` etc. in
  `vp9_new_framerate()` / SVC setup (not in `vp9_ratectrl.c`). Drives the
  CBR active-worst-Q curve.
- **`rc_1_frame` / `rc_2_frame` / `q_1_frame` / `q_2_frame`** — history of
  whether each of the last two frames overshot (+1) or undershot (-1).
  Used by `adjust_q_cbr` (`vp9_ratectrl.c:666-687`) to detect oscillation
  and clamp Q between the last two values.
- **`avg_source_sad[MAX_LAG_BUFFERS]`** — temporal SAD between consecutive
  source frames at 64×64 granularity, decimated checker-board. Updated by
  `vp9_scene_detection_onepass()` (`vp9_ratectrl.c:3034-3243`). Drives
  scene-change detection and `high_source_sad` flag.
- **`damped_adjustment[RATE_FACTOR_LEVELS]`** — first-frame-of-type flag.
  When 0, RCF update uses adjustment_limit=1.0 (full correction); after
  the first frame, switches to damped formula
  `0.25 + 0.5 * min(1, |log10(0.01*corr)|)` (`vp9_ratectrl.c:771-779`).
- **`avg_frame_low_motion`** — IIR (weight 3:1) of % of MBs coded as
  LAST_FRAME with |MV|<16 in 1/8-pel units; updated by
  `vp9_compute_frame_low_motion()` (`vp9_ratectrl.c:1811-1843`).
- **`fac_active_worst_inter` / `fac_active_worst_gf`** — percentage
  multipliers (default 150 / 100, init at `vp9_ratectrl.c:399-400`) on
  `avg_frame_qindex[INTER_FRAME]` and `last_q[INTER_FRAME]` to compute
  `active_worst_quality` in 1-pass VBR.
- **`af_ratio_onepass_vbr`** — adaptive-filter ratio (default 10, capped
  [5,15] in `vp9_set_gf_update_one_pass_vbr`); divides VBR P-frame target
  between GF and non-GF frames (`vp9_calc_pframe_target_size_one_pass_vbr`,
  `vp9_ratectrl.c:2019-2037`).
- **`high_source_sad`** — scene-change flag, set by
  `vp9_scene_detection_onepass()` when current-frame average SAD exceeds
  a multiplier (default thresh=8.0 for CBR, 2.1 for VBR; see
  `vp9_ratectrl.c:3076-3110`).
- **`force_max_q` / `reset_high_source_sad`** — both force
  `active_worst_quality = rc->worst_quality` on the next frame
  (`vp9_ratectrl.c:946-947`). Set after a post-encode drop, or after the
  encoder detects it's stuck at lowest Q while content changes.
- **`frame_size_selector` / `next_frame_size_selector`** — auto-resize
  state. Indexes into `frame_scale_factor[FRAME_SCALE_STEPS]`
  (`vp9_ratectrl.h:79`).

### 2.2 `CYCLIC_REFRESH`

Definition: `vp9/encoder/vp9_aq_cyclicrefresh.h:32-74`. Lives in
`VP9_COMP.cyclic_refresh` as a `CYCLIC_REFRESH*`; allocated by
`vp9_cyclic_refresh_alloc()` (`vp9_aq_cyclicrefresh.c:32-53`).

```c
struct CYCLIC_REFRESH {
  // Percentage of blocks per frame that are targeted as candidates
  // for cyclic refresh.
  int percent_refresh;
  // Maximum q-delta as percentage of base q.
  int max_qdelta_perc;
  // Superblock starting index for cycling through the frame.
  int sb_index;
  // Controls how long block will need to wait to be refreshed again, in
  // excess of the cycle time, i.e., in the case of all zero motion, block
  // will be refreshed every (100/percent_refresh + time_for_refresh) frames.
  int time_for_refresh;
  // Target number of (8x8) blocks that are set for delta-q.
  int target_num_seg_blocks;
  // Actual number of (8x8) blocks that were applied delta-q.
  int actual_num_seg1_blocks;
  int actual_num_seg2_blocks;
  // RD mult. parameters for segment 1.
  int rdmult;
  // Cyclic refresh map.
  signed char *map;
  // Map of the last q a block was coded at.
  uint8_t *last_coded_q_map;
  // Thresholds applied to the projected rate/distortion of the coding block,
  // when deciding whether block should be refreshed.
  int64_t thresh_rate_sb;
  int64_t thresh_dist_sb;
  // Threshold applied to the motion vector (in units of 1/8 pel) of the
  // coding block, when deciding whether block should be refreshed.
  int16_t motion_thresh;
  // Rate target ratio to set q delta.
  double rate_ratio_qdelta;
  // Boost factor for rate target ratio, for segment CR_SEGMENT_ID_BOOST2.
  int rate_boost_fac;
  double low_content_avg;
  int qindex_delta[3];
  int reduce_refresh;
  double weight_segment;
  int apply_cyclic_refresh;
  int counter_encode_maxq_scene_change;
  int skip_flat_static_blocks;
  int content_mode;
};
```

Key invariants:

- **`map`** is signed-char per 8×8 mi. Value semantics:
  `1` = not a candidate; `0` = candidate for refresh; `-N` (negative) =
  was just refreshed, decremented towards 0 each frame
  (`vp9_aq_cyclicrefresh.c:202-212` and 442-444).
- **`last_coded_q_map`** is uint8 per 8×8 mi, holding the qindex this block
  was last coded at (clamped to `[0, MAXQ=255]`,
  `vp9_aq_cyclicrefresh.c:245-252`). Initialised to `MAXQ`
  (line 49).
- **`sb_index`** is the wrap-around cursor used by `cyclic_refresh_update_map`
  (`vp9_aq_cyclicrefresh.c:363-475`); refresh sweeps superblocks in raster
  order until `target_num_seg_blocks >= block_count` or a full loop has
  occurred.
- **`qindex_delta[3]`** stores the qindex offsets for segments
  `CR_SEGMENT_ID_BASE=0`, `CR_SEGMENT_ID_BOOST1=1`, `CR_SEGMENT_ID_BOOST2=2`
  (`vp9_aq_cyclicrefresh.h:25-27`). `qindex_delta[0]` is always 0 (base
  segment is the baseline Q); `[1]` and `[2]` are computed by
  `compute_deltaq()` in `vp9_cyclic_refresh_setup()` lines 656-672.
- **`weight_segment`** is the smoothed fraction of the frame that is
  expected to be in a non-base segment, used by `bits_per_mb` estimation
  in `vp9_cyclic_refresh_rc_bits_per_mb` (line 137-155).
- **`apply_cyclic_refresh`** is the global on/off switch decided in
  `vp9_cyclic_refresh_update_parameters` (line 478-505). Conditions that
  disable it: intra-only frames; non-base temporal SVC layer; lossless;
  `avg_frame_qindex[INTER_FRAME] < qp_thresh` (the encoder is already
  underspending); `avg_frame_qindex[INTER_FRAME] > 117*MAXQ/128`
  combined with `frames_since_key > 20` (encoder is starved); etc.

### 2.3 `VP9_DENOISER`

Definition: `vp9/encoder/vp9_denoiser.h:45-56`. Lives in `VP9_COMP.denoiser`.

```c
typedef enum vp9_denoiser_decision {
  COPY_BLOCK,
  FILTER_BLOCK,
  FILTER_ZEROMV_BLOCK
} VP9_DENOISER_DECISION;

typedef enum vp9_denoiser_level {
  kDenLowLow,
  kDenLow,
  kDenMedium,
  kDenHigh
} VP9_DENOISER_LEVEL;

typedef struct vp9_denoiser {
  YV12_BUFFER_CONFIG *running_avg_y;
  YV12_BUFFER_CONFIG *mc_running_avg_y;
  YV12_BUFFER_CONFIG last_source;
  int frame_buffer_initialized;
  int reset;
  int num_ref_frames;
  int num_layers;
  unsigned int current_denoiser_frame;
  VP9_DENOISER_LEVEL denoising_level;
  VP9_DENOISER_LEVEL prev_denoising_level;
} VP9_DENOISER;
```

Semantics:

- **`running_avg_y[]`** — an array of `num_ref_frames * num_layers`
  `YV12_BUFFER_CONFIG`s. For non-SVC, `num_ref_frames = NONSVC_REF_FRAMES
  = MAX_REF_FRAMES - 1` (i.e., MAX_REF_FRAMES is 4 → 3 here)
  (`vp9_denoiser.h:26`). For SVC, `num_ref_frames = SVC_REF_FRAMES = 9`
  (line 30) (one slot per `REF_FRAMES` plus one current). Each buffer
  holds the running denoised average for one reference slot.
- **`mc_running_avg_y[]`** — array of `num_layers` buffers; per-spatial-layer
  motion-compensated denoised frame, freshly computed each frame from
  `running_avg_y[best_ref]` using the block's MV.
- **`last_source`** — used by the noise estimator (only).
- **`denoising_level`** — current strength (`kDenLowLow…kDenHigh`). Driven
  by the noise estimator (`vp9_noise_estimate.c:282-296`) via
  `vp9_denoiser_set_noise_level()` (`vp9_denoiser.c:732-743`).
- **`reset`** — set when the level transitions from `kDenLowLow` back up,
  to force re-init of the running averages on the next frame.
- **`current_denoiser_frame`** — frame counter incremented in
  `vp9_denoiser_update_ref_frame()` (line 791).
- Aliasing: the denoiser shadows pointers in `MACROBLOCKD` during motion
  compensation (`vp9_denoiser.c:284-321`) and restores them; not
  thread-safe across blocks of the same frame without external
  serialisation, but libvpx encodes a frame sequentially within a tile.

### 2.4 `VP9_PICKMODE_CTX_DEN`

Definition: `vp9/encoder/vp9_denoiser.h:58-68`. A small staging struct used
by `vp9_pickmode.c` to re-evaluate ZEROMV after denoising changes the
prediction.

```c
typedef struct {
  int64_t zero_last_cost_orig;
  int *ref_frame_cost;
  int_mv (*frame_mv)[MAX_REF_FRAMES];
  int reuse_inter_pred;
  TX_SIZE best_tx_size;
  PREDICTION_MODE best_mode;
  MV_REFERENCE_FRAME best_ref_frame;
  INTERP_FILTER best_pred_filter;
  uint8_t best_mode_skip_txfm;
} VP9_PICKMODE_CTX_DEN;
```

### 2.5 `NOISE_ESTIMATE`

Definition: `vp9/encoder/vp9_noise_estimate.h:30-40`.

```c
typedef enum noise_level { kLowLow, kLow, kMedium, kHigh } NOISE_LEVEL;

typedef struct noise_estimate {
  int enabled;
  NOISE_LEVEL level;
  int value;
  int thresh;
  int adapt_thresh;
  int count;
  int last_w;
  int last_h;
  int num_frames_estimate;
} NOISE_ESTIMATE;
```

- **`thresh`** is initialised by resolution (`vp9_noise_estimate.c:38-47`):
  90 (<640×360), 115 (≥640×360), 140 (≥1280×720), 200 (≥1920×1080).
- **`adapt_thresh = (3*thresh) >> 1`**.
- **`value`** is the smoothed estimate `(3*old + max_bin*40) >> 2` of the
  variance histogram peak across low-motion 16×16 blocks
  (`vp9_noise_estimate.c:280`).
- Final level is one of:
  - `value > 2*thresh` → `kHigh`
  - `value > thresh` → `kMedium`
  - `value > thresh/2` → `kLow`
  - else → `kLowLow`
  (`vp9_noise_estimate.c:94-107`).

### 2.6 Enums and module-level constants

- `RATE_FACTOR_LEVEL`: 5 values (`vp9_ratectrl.h:48-55`).
- `FRAME_SCALE_LEVEL`: `{UNSCALED=0, SCALE_STEP1=1}` (`vp9_ratectrl.h:58-62`).
- `RESIZE_ACTION`: `{NO_RESIZE=0, DOWN_THREEFOUR=1, DOWN_ONEHALF=2,
  UP_THREEFOUR=-1, UP_ORIG=-2}` (`vp9_ratectrl.h:64-70`).
- `RESIZE_STATE`: `{ORIG=0, THREE_QUARTER=1, ONE_HALF=2}`
  (`vp9_ratectrl.h:72`).
- `AQ_MODE`: `{NO_AQ=0, VARIANCE_AQ=1, COMPLEXITY_AQ=2,
  CYCLIC_REFRESH_AQ=3, EQUATOR360_AQ=4, PERCEPTUAL_AQ=5, PSNR_AQ=6,
  LOOKAHEAD_AQ=7}` (`vp9_encoder.h:117-129`).
- `#define MAXQ 255`, `MINQ 0`, `QINDEX_RANGE 256`
  (`vp9_quant_common.h:21-23`). The encoder's tunable Q range is always
  `[best_quality, worst_quality]` inside this.
- `MAX_REF_FRAMES = 4` (LAST, GOLDEN, ALTREF, plus INTRA in some
  contexts).
- `MAX_LAG_BUFFERS = 25` (see `vp9_lookahead.h`); sizes
  `RATE_CONTROL.avg_source_sad[]`.

## 3. Control flow

### 3.1 Per-frame rate-control sequence (CBR / VBR, non-SVC)

This is the canonical sequence enforced for 1-pass real-time. Quoting
`vp9_ratectrl.h:241-261` verbatim:

> First call per frame, one of:
>   vp9_rc_get_one_pass_vbr_params()
>   vp9_rc_get_one_pass_cbr_params()
>   vp9_rc_get_svc_params()
>   vp9_rc_get_first_pass_params()
>   vp9_rc_get_second_pass_params()
>
> Then, call encode_frame_to_data_rate() to perform the actual encode.
> This function will in turn call encode_frame() one or more times,
> followed by one of:
>   vp9_rc_postencode_update()
>   vp9_rc_postencode_update_drop_frame()

Concretely for 1-pass CBR the flow is:

```
encode_one_frame()                                  [vp9_encoder.c]
 ├─ vp9_scene_detection_onepass()                   [vp9_ratectrl.c:3034]
 │    └─ updates rc->avg_source_sad[], rc->high_source_sad,
 │       and on CBR low-Q + sad-spike: rc->reset_high_source_sad
 ├─ vp9_rc_get_one_pass_cbr_params(cpi)             [vp9_ratectrl.c:2493]
 │    ├─ decide frame_type (KEY/INTER) from current_video_frame, frame_flags,
 │    │   rc->frames_to_key, oxcf.auto_key, oxcf.key_freq
 │    ├─ if frames_till_gf_update_due == 0:
 │    │     if aq_mode==CYCLIC_REFRESH_AQ:
 │    │        vp9_cyclic_refresh_set_golden_update()  [vp9_aq_cyclicrefresh.c:319]
 │    │     else: baseline_gf_interval = (min+max)/2
 │    │     frames_till_gf_update_due = baseline_gf_interval (clipped to frames_to_key)
 │    │     refresh_golden_frame = 1; gfu_boost = DEFAULT_GF_BOOST
 │    ├─ if aq_mode==CYCLIC_REFRESH_AQ:
 │    │     vp9_cyclic_refresh_update_parameters(cpi) [vp9_aq_cyclicrefresh.c:478]
 │    ├─ target = vp9_calc_iframe_target_size_one_pass_cbr(cpi)
 │    │           or vp9_calc_pframe_target_size_one_pass_cbr(cpi)
 │    ├─ vp9_rc_set_frame_target(cpi, target)       [vp9_ratectrl.c:1713]
 │    ├─ if show_frame: vp9_update_buffer_level_preencode(cpi) [line 250]
 │    └─ if resize_mode == RESIZE_DYNAMIC: cpi->resize_pending = vp9_resize_one_pass_cbr(cpi)
 ├─ vp9_rc_drop_frame(cpi)?                          [vp9_ratectrl.c:614]
 │    ├─ checks vp9_test_drop() [line 513]:
 │    │    if buffer_level < 0 → always drop
 │    │    else use drop_mark = water_mark * optimal_buffer_level/100
 │    │    and adjust decimation_factor (0 or 1)
 │    └─ if drop: vp9_rc_postencode_update_drop_frame(cpi) [line 1998]
 │
 ├─ vp9_rc_pick_q_and_bounds(cpi, &bottom, &top)    [vp9_ratectrl.c:1587]
 │    └─ dispatch on pass / rc_mode:
 │         rc_pick_q_and_bounds_one_pass_cbr        [line 1002]
 │         rc_pick_q_and_bounds_one_pass_vbr        [line 1138]
 │         rc_pick_q_and_bounds_two_pass            [line 1427]
 │
 ├─ AQ-mode frame setup (vp9_encoder.c:4209-4239):
 │    VARIANCE_AQ      → vp9_vaq_frame_setup        [vp9_aq_variance.c:47]
 │    EQUATOR360_AQ    → vp9_360aq_frame_setup      [vp9_aq_360.c:39]
 │    COMPLEXITY_AQ    → vp9_setup_in_frame_q_adj   [vp9_aq_complexity.c:47]
 │    CYCLIC_REFRESH_AQ→ vp9_cyclic_refresh_setup   [vp9_aq_cyclicrefresh.c:595]
 │
 ├─ vp9_encode_frame()                              [vp9_encodeframe.c]
 │    └─ per-SB encode_sb():
 │         per block:
 │           pickmode (nonrd or rd)
 │           if (aq_mode==CYCLIC_REFRESH_AQ && nonrd_pick):
 │             vp9_cyclic_refresh_update_segment()  [vp9_aq_cyclicrefresh.c:160]
 │           ...
 │           if (CONFIG_VP9_TEMPORAL_DENOISING && oxcf.noise_sensitivity>0 &&
 │               denoising_level>kDenLowLow):
 │              vp9_denoiser_update_frame_stats(mi, sse_y, mode, ctx)
 │                                                  [vp9_pickmode.c:2438]
 │           ...  (after the pickmode loop, per superblock)
 │           vp9_denoiser_denoise(cpi, x, mi_row, mi_col, bsize, ctx, &decision,
 │                                gf_temporal_ref)  [vp9_pickmode.c:2688]
 │
 ├─ (CBR_RT only) if sf.overshoot_detection_cbr_rt==RE_ENCODE_MAXQ &&
 │   (high_source_sad || svc.high_source_sad_superframe):
 │      pack_bitstream once to get size; if vp9_encodedframe_overshoot()
 │      [vp9_ratectrl.c:3248] decides yes → re-encode at max Q.
 │
 ├─ if aq_mode==CYCLIC_REFRESH_AQ && seg.enabled && !intra:
 │      vp9_cyclic_refresh_postencode(cpi)          [vp9_aq_cyclicrefresh.c:260]
 │
 ├─ pack_bitstream + (CBR) post_encode_drop_cbr(cpi, &size)
 │                                                  [vp9_ratectrl.c:566]
 │    if new buffer_level after this frame's bits would go below 0:
 │      drop frame, force max Q on next, set last_frame_dropped, etc.
 │
 └─ vp9_rc_postencode_update(cpi, bytes_used)       [vp9_ratectrl.c:1845]
      ├─ rc->projected_frame_size = bytes_used << 3
      ├─ vp9_rc_update_rate_correction_factors(cpi) [line 737]
      ├─ update last_q / avg_frame_qindex (3:1 IIR)
      ├─ update_buffer_level_postencode             [line 316]
      ├─ update rolling_target_bits / rolling_actual_bits (3:1 and 31:1)
      ├─ total_actual_bits += projected_frame_size
      ├─ if !intra: update_golden_frame_stats / update_alt_ref_frame_stats
      ├─ if intra: frames_since_key = 0
      ├─ if show_frame: ++frames_since_key, --frames_to_key
      └─ rc->last_avg_frame_bandwidth = rc->avg_frame_bandwidth
```

VBR differs only in the *params* and *target* calls; the post-encode and
regulate paths are shared.

### 3.2 Denoiser integration with the encode-SB path

Three touch-points:

1. **Per-mode stat collection.** `vp9_denoiser_update_frame_stats()`
   (`vp9_denoiser.c:533-548`) is called from inside the inter-mode loop in
   `vp9_pickmode.c:2438` for every (mode, ref_frame) the picker
   evaluates. It tracks:
   - `ctx->zeromv_sse` / `ctx->best_zeromv_reference_frame`: best
     zero-motion SSE seen and which ref attained it (including the special
     `zeromv_lastref_sse` for LAST_FRAME);
   - `ctx->newmv_sse` / `ctx->best_sse_mv` / `ctx->best_sse_inter_mode` /
     `ctx->best_reference_frame`: best non-zero-MV SSE seen.

   These are *not* the encoder's chosen mode — they are auxiliary stats
   used solely by the denoiser to decide whether motion compensation onto
   a denoised reference is likely to be cleaner than the source.

2. **Per-block denoise.** After mode decision, in `vp9_pickmode.c:2676-2695`
   inside `vp9_pick_inter_mode_sb`:
   - Guarded by `CONFIG_VP9_TEMPORAL_DENOISING`,
     `oxcf.noise_sensitivity > 0`, `resize_pending == 0`,
     `denoise_svc_pickmode`, `denoising_level > kDenLowLow`,
     `denoiser.reset == 0`.
   - `vp9_pickmode_ctx_den_update()` snapshots best mode/MV/ref/filter
     into a `VP9_PICKMODE_CTX_DEN`.
   - `vp9_denoiser_denoise()` (`vp9_denoiser.c:336-437`) runs the
     per-block filter (see §4.5). Outputs `decision ∈
     {COPY_BLOCK, FILTER_BLOCK, FILTER_ZEROMV_BLOCK}`.
   - If `denoise_recheck_zeromv` is on, `recheck_zeromv_after_denoising()`
     re-runs the ZEROMV cost computation on the denoised buffer and may
     swap the chosen mode.

3. **Post-encode update.** `vp9_denoiser_update_ref_frame()`
   (`vp9_denoiser.c:782-822`) is called after the frame is encoded. It:
   - Calls `vp9_denoiser_realloc_svc()` if SVC and any new refs need
     allocation.
   - Calls `vp9_denoiser_update_frame_info()` (line 464-524), which
     swaps/copies the current `running_avg_y[INTRA_FRAME]` into the
     ref-frame slot(s) that this frame is refreshing.
   - On KEY_FRAME or `denoiser->reset == 1`, all running_avg_y slots are
     overwritten from `src` (re-init).

The noise estimator is decoupled: `vp9_update_noise_estimate()`
(`vp9_noise_estimate.c:109-302`) is called early in the frame loop. Every
`frame_period = 8` frames it builds a 20-bin variance histogram of low-motion
16×16 blocks; finds the modal bin; the modal-bin index times 40 is folded
into `ne->value` with weight 3:1; once `num_frames_estimate` (15 initially,
30 thereafter) frames have accumulated, `ne->level` is updated and
`vp9_denoiser_set_noise_level()` is called.

### 3.3 AQ integration with rate control

Each AQ mode plugs in *between* `vp9_rc_pick_q_and_bounds` (which sets
`cm->base_qindex`) and `vp9_encode_frame`:

| AQ mode | Setup fn (frame) | Per-block | RC interaction |
|---|---|---|---|
| `NO_AQ=0` | — | — | — |
| `VARIANCE_AQ=1` | `vp9_vaq_frame_setup` (`vp9_aq_variance.c:47`) | `vp9_vaq_segment_id(energy)` chooses one of 6 segments based on log-variance bucket | Q deltas in segment table use `rate_ratio[]` × `vp9_compute_qdelta_by_rate` |
| `COMPLEXITY_AQ=2` | `vp9_setup_in_frame_q_adj` (`vp9_aq_complexity.c:47`) | `vp9_caq_select_segment` chooses segment by combining projected_rate vs target_rate × `aq_c_transitions[s][seg]` and `log_var < low_var_thresh + aq_c_var_thresholds[s][seg]` | Same `vp9_compute_qdelta_by_rate` machinery, strength `s = (base_quant>10) + (base_quant>25)` |
| `CYCLIC_REFRESH_AQ=3` | `vp9_cyclic_refresh_setup` (`vp9_aq_cyclicrefresh.c:595`) builds the segmentation map from the cyclic map; `vp9_cyclic_refresh_update_segment` updates per-block | Per-block decision driven by `candidate_refresh_aq()` (rate/dist/MV thresholds, intra rejection) | Deeply integrated: `vp9_rc_regulate_q` uses `vp9_cyclic_refresh_rc_bits_per_mb` (line 837-844); `vp9_rc_update_rate_correction_factors` uses `vp9_cyclic_refresh_estimate_bits_at_q` (line 757-759); `adjust_q_cbr` calls `vp9_cyclic_refresh_limit_q` for screen content (line 683-685) |
| `EQUATOR360_AQ=4` | `vp9_360aq_frame_setup` (`vp9_aq_360.c:39`) | `vp9_360aq_segment_id(mi_row, mi_rows)` — 0/1/2 by row band | Segment table with `rate_ratio[]` (lower for poles → lower Q) |

Important: `VARIANCE_AQ`, `COMPLEXITY_AQ`, `EQUATOR360_AQ` only apply
on **intra-only frames, error-resilient frames, alt-ref frames, or
golden-refresh non-altref frames** (see e.g. `vp9_aq_variance.c:52-54`).
On regular P frames the per-frame segmentation map is left undisturbed.
`CYCLIC_REFRESH_AQ` is the only mode that refreshes segmentation on every
inter frame, by design.

### 3.4 SVC interaction (just enough)

- `vp9_rc_get_svc_params()` (`vp9_ratectrl.c:2279-2491`) is the SVC
  entry point that replaces both `_cbr_params` and `_vbr_params`. It
  fans out target adjustments to all temporal layers, handles intra-only
  insertion, and updates per-layer buffer levels via
  `vp9_update_buffer_level_svc_preencode()` (line 263-294).
- Each spatial+temporal layer has its own `LAYER_CONTEXT.rc` (a full
  `RATE_CONTROL`). The encoder's top-level `cpi->rc` is *copied to/from*
  the active layer at frame start/end. Other agents (A6) cover this in
  detail.
- Denoiser SVC behaviour: only the top 1 or 2 spatial layers are
  denoised, controlled by `svc->first_layer_denoise`
  (`vp9_denoiser.c:625-631`).
- AQ-360 and AQ-variance are not designed for SVC and are not exercised
  in the real-time path. CYCLIC_REFRESH_AQ in SVC operates only on the
  base temporal layer (`vp9_aq_cyclicrefresh.c:491`).

## 4. Algorithms

### 4.1 The bits-per-MB model

Core formula, used everywhere Q→bits or bits→Q is needed
(`vp9_ratectrl.c:185-196`):

```
double q       = vp9_convert_qindex_to_q(qindex, bit_depth)
                 = ac_qlookup[qindex] / scale          // scale = 4 / 16 / 64 for 8/10/12-bit
int enumerator = (frame_type == KEY_FRAME) ? 2_700_000 : 1_800_000
enumerator    += (enumerator * (int)q) >> 12         // Q-dependent bump
bits_per_mb    = (int)(enumerator * correction_factor / q)
```

`correction_factor ∈ [0.005, 50]` from `rate_correction_factors[]`.

`vp9_estimate_bits_at_q(...)` (`vp9_ratectrl.c:198-205`):
```
bpm  = vp9_rc_bits_per_mb(...)
bits = max(FRAME_OVERHEAD_BITS=200, (bpm * mbs) >> BPER_MB_NORMBITS=9)
```

### 4.2 Q-index search (`vp9_rc_regulate_q`)

(`vp9_ratectrl.c:819-862`)

```
target_bits_per_mb = (target_bits_per_frame << 9) / cm->MBs    // 9 = BPER_MB_NORMBITS
corr               = get_rate_correction_factor(cpi)            // one of 5 RCFs

q = active_worst_quality
last_error = INT_MAX
for i in [active_best_quality .. active_worst_quality]:
  if (aq_mode == CYCLIC_REFRESH_AQ && cr->apply_cyclic_refresh
      && !(gf_cbr_boost_pct && refresh_golden)):
    bpm_i = vp9_cyclic_refresh_rc_bits_per_mb(cpi, i, corr)     // weighted base/segment
  else:
    bpm_i = vp9_rc_bits_per_mb(frame_type, i, corr, bit_depth)
  if bpm_i <= target_bits_per_mb:
    q = i if (target - bpm_i <= last_error) else (i - 1)
    break
  else:
    last_error = bpm_i - target_bits_per_mb
if rc_mode == VPX_CBR:
  q = adjust_q_cbr(cpi, q)
return q
```

`adjust_q_cbr` (`vp9_ratectrl.c:666-687`):
1. If neither `reset_high_source_sad` nor a golden-boost is active and the
   last two frames oscillated (`rc_1*rc_2 == -1`) and `q_1 != q_2`,
   clamp `q ∈ [min(q_1,q_2), max(q_1,q_2)]`; if last frame overshot and
   q > clamp, halve the gap.
2. For screen content + `CYCLIC_REFRESH_AQ`, call
   `vp9_cyclic_refresh_limit_q` (`vp9_aq_cyclicrefresh.c:695-702`): if
   `cr->percent_refresh > 0` and `q_1_frame - q > 8`, clamp
   `q := q_1_frame - 8` (cap frame-level Q decrease).
3. Final clamp to `[best_quality, worst_quality]`.

### 4.3 Rate-correction-factor update (`vp9_rc_update_rate_correction_factors`)

(`vp9_ratectrl.c:737-817`)

```
projected_size_at_q = (aq_mode==CYCLIC_REFRESH_AQ && seg.enabled)
   ? vp9_cyclic_refresh_estimate_bits_at_q(cpi, rcf)
   : vp9_estimate_bits_at_q(frame_type, cm->base_qindex, MBs, rcf, bit_depth)

correction = (projected_size_at_q > 200) ?
                 100 * projected_frame_size / projected_size_at_q : 100

if !damped_adjustment[rf_lvl]:
  adjustment_limit = 1.0
  damped_adjustment[rf_lvl] = 1
else:
  adjustment_limit = 0.25 + 0.5 * min(1.0, |log10(0.01 * correction)|)

q_2_frame = q_1_frame ; q_1_frame = base_qindex
rc_2_frame = rc_1_frame
rc_1_frame = (correction > 110) ? -1 : (correction < 90) ? +1 : 0
if rc_1_frame == -1 and rc_2_frame == +1 and correction > 1000: rc_2_frame = 0

if correction > 102:
  correction = 100 + (correction - 100) * adjustment_limit
  rcf = clamp(rcf * correction / 100, _, MAX_BPB_FACTOR)
elif correction < 99:
  correction = 100 - (100 - correction) * adjustment_limit
  rcf = clamp(rcf * correction / 100, MIN_BPB_FACTOR, _)

set_rate_correction_factor(cpi, rcf)
```

Note the sign of `rc_1_frame` is inverted from intuition: **+1 means
undershoot** (we used fewer bits than expected), **-1 means overshoot**.
This is because `correction > 110` means projected_size_at_q was much
smaller than what we *actually* emitted — i.e., the BPM model
under-predicted, so the encoder overshot.

### 4.4 Active min/max Q (`active_best_quality` / `active_worst_quality`)

**CBR (`calc_active_worst_quality_one_pass_cbr`,
`vp9_ratectrl.c:931-1000`)**:
```
if intra or reset_high_source_sad or force_max_q: return worst_quality

num_frames_weight_key = 5 * number_temporal_layers

ambient_qp = (current_video_frame < num_frames_weight_key)
             ? min(avg_frame_qindex[INTER_FRAME], avg_frame_qindex[KEY_FRAME])
             : avg_frame_qindex[INTER_FRAME]
active_worst = min(worst_quality, ambient_qp * 5/4)

if SVC layer 0 was a key: ambient_qp = min(ambient_qp, key_layer_last_q);
                          active_worst = min(worst_quality, ambient_qp * 9/8)

if buffer_level > optimal_buffer_level:
  max_adjustment_down = active_worst / 3
                       (or active_worst >> 3 for screen content)
  buff_step = (maximum_buffer_size - optimal_buffer_level) / max_adjustment_down
  active_worst -= (buffer_level - optimal_buffer_level) / buff_step
elif buffer_level > critical_level = optimal/8:
  buff_step = (optimal_buffer_level - critical_level)
  active_worst = ambient_qp + (worst_quality - ambient_qp)
                              * (optimal_buffer_level - buffer_level)
                              / buff_step
else:
  active_worst = worst_quality
return active_worst
```

**CBR `active_best_quality` (`rc_pick_q_and_bounds_one_pass_cbr`,
`vp9_ratectrl.c:1002-1100`):**
- KEY_FRAME first frame: `best_quality`. KEY_FRAME forced: derived from
  `last_boosted_qindex` using `vp9_compute_qdelta(rc, q, q*0.75, …)`.
  Otherwise `get_kf_active_quality(rc, avg_frame_qindex[KEY_FRAME])`
  using `kf_low/kf_high/kf_low_motion_minq/kf_high_motion_minq` interpolated
  by `kf_boost` (default 2000).
- GF/ARF with `gf_cbr_boost_pct`: `get_gf_active_quality`.
- Regular inter: `rtc_minq[min(avg_frame_qindex[INTER_FRAME], active_worst)]`.
- Small image (<=352×288): KF Q gets *additionally* lowered by
  `vp9_compute_qdelta(rc, q, q*0.75)`.

**VBR (`calc_active_worst_quality_one_pass_vbr`,
`vp9_ratectrl.c:905-928`):**
```
if KEY:
  worst = curr_frame==0 ? worst_quality : last_q[KEY_FRAME] << 1
elif !src_alt_ref && (refresh_golden || refresh_alt):
  worst = curr_frame==1 ? last_q[KEY]*5/4
                        : last_q[INTER] * fac_active_worst_gf / 100
else:
  worst = curr_frame==1 ? last_q[KEY] << 1
                        : avg_frame_qindex[INTER] * fac_active_worst_inter / 100
return min(worst, worst_quality)
```

**VBR `active_best_quality` (`rc_pick_q_and_bounds_one_pass_vbr`,
`vp9_ratectrl.c:1138-1303`):**
Branches as for CBR but uses `inter_minq[...]` (one of the LUTs in §5)
instead of `rtc_minq[...]`. Special-case CQ mode (VPX_CQ) and constant-Q
mode (VPX_Q) clamp to `cq_level` and to a per-frame delta-rate pattern,
respectively (see lines 1227-1233 for the
`{0.50,1.0,0.85,1.0,0.70,1.0,0.85,1.0}` pattern over a fixed 8-frame GF
group).

After picking, when `LIMIT_QRANGE_FOR_ALTREF_AND_KEY = 1` (default), the
top index is tightened by `vp9_compute_qdelta_by_rate(..., 2.0)` for KEY
frames and `..., 1.75` for GF/ARF (lines 1260-1279).

### 4.5 Active-min-Q LUTs (`init_minq_luts`, `get_minq_index`)

(`vp9_ratectrl.c:102-154`)

Each LUT is generated by a 3rd-order polynomial fit. For each qindex i,
let `maxq = ac_qlookup[i] / scale`:

```
minqtarget = min(((x3*maxq + x2)*maxq + x1)*maxq, maxq)
              // if <= 2.0, return 0 (lossless special case)
              // else return smallest j s.t. vp9_convert_qindex_to_q(j) >= minqtarget
```

Coefficients (`vp9_ratectrl.c:126-138`):

| LUT | x3 | x2 | x1 |
|---|---|---|---|
| `kf_low_m` | 0.000001 | -0.0004 | 0.150 |
| `kf_high_m` | 0.0000021 | -0.00125 | 0.45 |
| `arfgf_low` (non-AGGRESSIVE_VBR) | 0.0000015 | -0.0009 | 0.30 |
| `arfgf_low` (AGGRESSIVE_VBR) | 0.0000015 | -0.0009 | 0.275 |
| `arfgf_high` | 0.0000021 | -0.00125 | 0.55 |
| `inter` (non-AGGRESSIVE_VBR) | 0.00000271 | -0.00113 | 0.70 |
| `inter` (AGGRESSIVE_VBR) | 0.00000271 | -0.00113 | 0.80 |
| `rtc` | 0.00000271 | -0.00113 | 0.70 |

(The same formulae are used at all three bit depths; the difference is
the `vp9_convert_qindex_to_q` scale.)

`get_active_quality` (`vp9_ratectrl.c:864-877`) interpolates between the
low-motion and high-motion LUTs of the same family using `gfu_boost`
(or `kf_boost`) and the family's `low` / `high` thresholds:

```
if boost > high:        return low_motion_minq[q]
elif boost < low:       return high_motion_minq[q]
else:                   gap   = high - low
                        offset = high - boost
                        qdiff  = high_motion[q] - low_motion[q]
                        return low_motion[q] + (offset*qdiff + gap/2) / gap
```

Thresholds (non-AGGRESSIVE_VBR, line 95-100): `gf_high=2000, gf_low=400,
kf_high=4800, kf_low=300`.

### 4.6 Frame target sizes

**CBR P-frame (`vp9_calc_pframe_target_size_one_pass_cbr`,
`vp9_ratectrl.c:2145-2192`):**
```
diff           = optimal_buffer_level - buffer_level
one_pct_bits   = 1 + optimal_buffer_level / 100
min_target     = max(avg_frame_bandwidth >> 4, FRAME_OVERHEAD_BITS=200)

if oxcf.gf_cbr_boost_pct:
  af = gf_cbr_boost_pct + 100
  target = refresh_golden
             ? (avg_bw * gf_interval * af) / (gf_interval*100 + af - 100)
             : (avg_bw * gf_interval * 100) / (gf_interval*100 + af - 100)
else:
  target = avg_frame_bandwidth

if SVC: target = layer.avg_frame_size
        min_target = max(layer.avg_frame_size >> 4, FRAME_OVERHEAD_BITS)

if diff > 0:                          // buffer is below optimal → spend less
  pct_low = min(diff / one_pct_bits, oxcf.under_shoot_pct)
  target -= target * pct_low / 200
elif diff < 0:                        // buffer is above optimal → spend more
  pct_high = min(-diff / one_pct_bits, oxcf.over_shoot_pct)
  target += target * pct_high / 200

if oxcf.rc_max_inter_bitrate_pct:
  max_rate = avg_frame_bandwidth * rc_max_inter_bitrate_pct / 100
  target = min(target, max_rate)
return max(min_target, target)
```

Note the asymmetry of `diff`: positive means buffer is *below* optimal
(we underflowed), so we *reduce* target to refill. The `/200` (rather
than `/100`) keeps adjustments mild (max ~half of pct_low).

**CBR I-frame (`vp9_calc_iframe_target_size_one_pass_cbr`,
`vp9_ratectrl.c:2194-2221`):**
- Very first frame: `target = starting_buffer_level / 2`.
- Subsequent KFs: `kf_boost = max(32, 2*framerate - 16)`, decayed if
  `frames_since_key < framerate/2`. `target = ((16 + kf_boost) *
  avg_frame_bandwidth) >> 4`. Then clamped by
  `vp9_rc_clamp_iframe_target_size`.

**VBR P-frame (`vp9_calc_pframe_target_size_one_pass_vbr`,
`vp9_ratectrl.c:2019-2037`):**
```
af = af_ratio_onepass_vbr                              // default 10, [5..15]
target = (!is_src_frame_alt_ref && (refresh_golden || refresh_alt))
   ? (avg_bw * gf_interval * af) / (gf_interval + af - 1)
   : (avg_bw * gf_interval)      / (gf_interval + af - 1)
return clamp_pframe(target)
```
For SVC VBR, target falls back to CBR formula.

**VBR I-frame (`vp9_calc_iframe_target_size_one_pass_vbr`, line 2039-2048):**
`target = avg_frame_bandwidth * kf_ratio=25` (capped to INT_MAX).

### 4.7 Buffer model arithmetic

Per-frame: bits arrive at `avg_frame_bandwidth = target_bandwidth /
framerate`, bits drain at `projected_frame_size`. The "buffer level" is
the bits we have stored that we haven't sent yet (high = good).

Pre-encode (`vp9_update_buffer_level_preencode`, line 250-256):
```
bits_off_target += avg_frame_bandwidth
bits_off_target  = min(bits_off_target, maximum_buffer_size)
buffer_level     = bits_off_target
```

Post-encode (`update_buffer_level_postencode`, line 316-333):
```
bits_off_target -= encoded_frame_size
bits_off_target  = min(bits_off_target, maximum_buffer_size)
if (content == VP9E_CONTENT_SCREEN && drop_frames_water_mark == 0):
  bits_off_target = max(bits_off_target, -maximum_buffer_size)
buffer_level     = bits_off_target
```

For SVC, each upper temporal layer's `bits_off_target` also has the
encoded size subtracted (see `update_layer_buffer_level_postencode`,
line 298-313, and pre-encode pump for higher layers,
`vp9_update_buffer_level_svc_preencode`, line 263-294).

The drop threshold is `drop_mark = drop_frames_water_mark *
optimal_buffer_level / 100` (`vp9_ratectrl.c:541-542`). When the *current*
buffer level dips below this, `vp9_test_drop` arms decimation.

### 4.8 Frame-drop decision (`vp9_test_drop`, `vp9_rc_drop_frame`,
`post_encode_drop_cbr`)

Three drop paths:

1. **Pre-encode drop** (`vp9_rc_drop_frame`, `vp9_ratectrl.c:614-664`,
   delegating to `vp9_test_drop` at 513-564):
   - If SVC previous spatial layer dropped (in CONSTRAINED or
     FULL_SUPERFRAME_DROP modes), drop this layer too.
   - If `drop_frames_water_mark == 0`, never drop.
   - If `buffer_level < 0`, always drop.
   - Else compute `drop_mark = drop_frames_water_mark *
     optimal_buffer_level / 100`. If buffer rose back above drop_mark,
     decrement `decimation_factor`; if it dipped at or below drop_mark and
     factor is 0, set factor to 1. (Comment says "currently capped" — the
     code only ever sets it to 1, never higher.)
   - If `decimation_factor > 0`, drop every `(decimation_factor+1)`-th
     frame using `decimation_count`.

2. **Post-encode CBR drop** (`post_encode_drop_cbr`, line 566-612):
   Only enabled if `use_post_encode_drop` (which is in turn only set when
   `ext_use_post_encode_drop` is set by the API). After encoding,
   compute `new_buffer_level = buffer_level + avg_frame_bandwidth -
   encoded_size`. If < 0, drop, force `avg_frame_qindex[INTER_FRAME] =
   worst_quality`, `force_max_q = 1`, `last_frame_dropped = 1`. SVC
   propagates this force-maxQ to all layers.

3. **Drop bookkeeping** (`vp9_rc_postencode_update_drop_frame`,
   line 1998-2017): increment `current_video_frame`, `frames_since_key`;
   decrement `frames_to_key`; clear `rc_1_frame/rc_2_frame`;
   `last_q[INTER_FRAME] = base_qindex`. For SVC under non-LAYER_DROP, if
   `buffer_level > optimal`, clip back to optimal to prevent unbounded
   growth of upper-layer buffer when whole superframe is dropped.

### 4.9 Cyclic refresh delta-Q machinery

Two scratch values per frame (`vp9_cyclic_refresh_setup`,
`vp9_aq_cyclicrefresh.c:595-677`):
```
thresh_rate_sb = (sb64_target_rate << 8) << 2      // 1024× target → bits×256 units
thresh_dist_sb = (q*q) << 2                        // quadratic in real-Q
```

For segment 1 (`CR_SEGMENT_ID_BOOST1 = 1`):
```
qindex_delta[1] = compute_deltaq(cpi, base_qindex, rate_ratio_qdelta)
                = max(  vp9_compute_qdelta_by_rate(rate_ratio_qdelta),
                       -max_qdelta_perc * base_qindex / 100 )
```

For segment 2 (`CR_SEGMENT_ID_BOOST2 = 2`, more aggressive boost):
```
qindex_delta[2] = compute_deltaq(cpi, base_qindex,
                       min(CR_MAX_RATE_TARGET_RATIO=4.0,
                           0.1 * rate_boost_fac * rate_ratio_qdelta))
```

Default tuning (`vp9_cyclic_refresh_update_parameters`, line 478-592):
- `percent_refresh = 10` (or 5 if `reduce_refresh`).
- `max_qdelta_perc = 60` (delta capped at -60% of base Q).
- `motion_thresh = 32` (1/8 pel).
- `rate_boost_fac = 15` (10 disables seg-2).
- For first `4 * num_temporal_layers * (100/percent_refresh)` frames after
  KF, `rate_ratio_qdelta = 3.0`. Thereafter 2.0 (or 1.7 if noise level
  ≥ kMedium).
- Screen content: `rate_ratio_qdelta = 2.0`, `rate_boost_fac = 10`,
  `skip_flat_static_blocks = 1`.
- VBR: `percent_refresh = 10` (0 on golden-refresh frames),
  `rate_ratio_qdelta = 1.5`, `rate_boost_fac = 10`.
- Low-resolution (≤352×288): if `avg_frame_bandwidth < 3000`,
  `motion_thresh = 64`, `rate_boost_fac = 13`; else
  `max_qdelta_perc = 70`, `rate_ratio_qdelta = max(2.5, _)`.

Per-block decision (`candidate_refresh_aq`,
`vp9_aq_cyclicrefresh.c:67-87`):

```
if dist > thresh_dist_sb and (|mv|>motion_thresh or intra):
  return CR_SEGMENT_ID_BASE         // not refreshing
elif bsize >= BLOCK_16X16 and rate < thresh_rate_sb and zero-MV inter
     and rate_boost_fac > 10:
  return CR_SEGMENT_ID_BOOST2       // aggressive
else:
  return CR_SEGMENT_ID_BOOST1       // normal boost
```

(Plus skin-detection override: small blocks ≤16×16 that are skin get
forced to BOOST1 even if they otherwise wouldn't qualify — line 178-184.)

Sweep selection (`cyclic_refresh_update_map`, line 363-475): starting at
`sb_index`, iterate superblocks raster-order; a superblock is included if
*at least half* of its 8×8 sub-blocks have `map==0` (i.e., are candidates
for refresh). Flat-static blocks (zero variance and SAD to last source)
are excluded under screen-content mode if `skip_flat_static_blocks`.
Continue until `target_num_seg_blocks >= block_count = percent_refresh *
mi_rows * mi_cols / 100` or full wrap.

### 4.10 Variance AQ segment assignment

(`vp9_aq_variance.c`)

Frame setup (`vp9_vaq_frame_setup`, line 47-84) — enabled only on
KEY/altref/golden-refresh/error-resilient frames:

```
for i in 0..MAX_SEGMENTS:                   // MAX_SEGMENTS = 8
  if rate_ratio[i] == 1.0:
    skip                                    // segment uses baseline Q
  qd = vp9_compute_qdelta_by_rate(frame_type, base_qindex,
                                  rate_ratio[i], bit_depth)
  if base_qindex != 0 and base_qindex + qd == 0:
    qd = -base_qindex + 1                   // avoid lossless trap
  seg_set_data(i, SEG_LVL_ALT_Q, qd)
  seg_enable_feature(i, SEG_LVL_ALT_Q)
```

`rate_ratio[]` (line 31-32):
```
{ 2.5, 2.0, 1.5, 1.0, 0.75, 1.0, 1.0, 1.0 }
```
(`rate_ratio[i] > 1.0` ⇒ lower bits ⇒ higher Q for high-energy
segments.)

Per-block energy → segment:
```
energy_midpoint = 10.0   (DEFAULT_E_MIDPOINT)
energy          = log(block_var + 1.0) - energy_midpoint
energy          = clamp(round(energy), ENERGY_MIN=-4, ENERGY_MAX=+1)
segment_id      = segment_id[energy - ENERGY_MIN]
```
where `segment_id[ENERGY_SPAN=6] = {0, 1, 1, 2, 3, 4}` (`vp9_aq_variance.c:33`).

I.e., energies -4, -3, -2, -1, 0, +1 map to segments 0, 1, 1, 2, 3, 4
respectively.

### 4.11 Complexity AQ segment assignment

(`vp9_aq_complexity.c`)

5 segments, "default neutral" is segment 3
(`DEFAULT_AQ2_SEG=3`, line 23). Strength selection (line 41-45):
```
base_quant = ac_qlookup[base_qindex] / 4
s          = (base_quant > 10) + (base_quant > 25)   // ∈ {0, 1, 2}
```

Per-block (`vp9_caq_select_segment`, line 110-160):
```
target_rate = (sb64_target_rate * xmis * ymis * 256) / (8*8)
                                                     // ↑ blocks per 64×64
low_var_thresh = (pass==2) ? max(twopass.mb_av_energy, MIN_DEFAULT_LV_THRESH=8.0)
                           : DEFAULT_LV_THRESH=10.0
logvar = vp9_log_block_var(cpi, mb, bs)

segment = AQ_C_SEGMENTS - 1
for i in 0..AQ_C_SEGMENTS=5:
  if projected_rate < target_rate * aq_c_transitions[s][i]
     and logvar < low_var_thresh + aq_c_var_thresholds[s][i]:
    segment = i
    break
```

Tables (line 25-39):
```
aq_c_q_adj_factor[3][5] = { {1.75, 1.25, 1.05, 1.00, 0.90},
                            {2.00, 1.50, 1.15, 1.00, 0.85},
                            {2.50, 1.75, 1.25, 1.00, 0.80} }
aq_c_transitions[3][5]  = { {0.15, 0.30, 0.55, 2.00, 100.0},
                            {0.20, 0.40, 0.65, 2.00, 100.0},
                            {0.25, 0.50, 0.75, 2.00, 100.0} }
aq_c_var_thresholds[3][5] = { {-4.0, -3.0, -2.0, 100.00, 100.0},
                              {-3.5, -2.5, -1.5, 100.00, 100.0},
                              {-3.0, -2.0, -1.0, 100.00, 100.0} }
```

Frame setup is gated by `sb64_target_rate >= 256` (line 67-70) — under
that, AQ-COMPLEXITY is disabled because overhead exceeds benefit.

### 4.12 360-equirectangular AQ

(`vp9_aq_360.c`)

Row-band only. `vp9_360aq_segment_id(mi_row, mi_rows)`:
- `mi_row < mi_rows/8 || mi_row > 7*mi_rows/8` → segment 2 (polar)
- `mi_row < mi_rows/4 || mi_row > 3*mi_rows/4` → segment 1 (temperate)
- else → segment 0 (equatorial)

Setup uses (line 25-26):
```
rate_ratio[] = { 1.0, 0.75, 0.6, 0.5, 0.4, 0.3, 0.25 }
```
(Lower at poles → lower Q.) Same `compute_qdelta_by_rate` plumbing.

### 4.13 Denoiser scalar reference (`vp9_denoiser_filter_c`)

(`vp9_denoiser.c:68-180`)

Block-level filter taking `(sig, mc_avg) → avg`. Returns `FILTER_BLOCK`
or `COPY_BLOCK`.

```
adj_val[3] = {3, 4, 6}
if motion_magnitude <= MOTION_MAGNITUDE_THRESHOLD=24:
  shift_inc = increase_denoising ? 2 : 1
  adj_val[i] += shift_inc      // bumps to {4..5, 5..6, 7..8}

// First pass: per-pixel adjustment toward mc_avg
total_adj = 0
for each pixel (r,c) of the block:
  diff    = mc_avg[r,c] - sig[r,c]
  absdiff = |diff|
  if absdiff <= absdiff_thresh =3 (or 4 if increase_denoising):
     avg[r,c] = mc_avg[r,c]
     total_adj += diff
  else:
     adj = absdiff in 4..7   → adj_val[0]
           absdiff in 8..15  → adj_val[1]
           absdiff ≥ 16      → adj_val[2]
     avg[r,c] = clamp(sig + (diff>0 ? +adj : -adj), 0, 255)
     total_adj += (diff>0 ? +adj : -adj)

if |total_adj| <= total_adj_strong_thresh = num_pixels * (increase_denoising?3:2):
  return FILTER_BLOCK

// Second pass: dampen the filter
delta = ((|total_adj| - total_adj_strong_thresh) / num_pixels) + 1
if delta >= delta_thresh = 4:
  return COPY_BLOCK

for each pixel (r,c):
  diff = mc_avg[r,c] - sig[r,c]
  adj  = min(|diff|, delta)
  if diff > 0:
    avg[r,c] = max(0, avg[r,c] - adj); total_adj -= adj
  else:
    avg[r,c] = min(255, avg[r,c] + adj); total_adj += adj

if |total_adj| <= total_adj_weak_thresh = num_pixels * (increase_denoising?3:2):
  return FILTER_BLOCK
return COPY_BLOCK
```

(`total_adj_strong_thresh` and `total_adj_weak_thresh` have the same
formula — see `vp9_denoiser.h:107-111` and `vp9_denoiser.c:61-63`. The
"weak" threshold is applied after damping, so a block needs to fail it
*twice* to be rejected.)

### 4.14 Denoiser per-block decision (`vp9_denoiser_denoise` +
`perform_motion_compensation`)

(`vp9_denoiser.c:336-437` and `187-334`)

```
denoise(cpi, mb, mi_row, mi_col, bsize, ctx, decision, use_gf_temporal_ref):
  motion_mag = best_sse_mv.row² + best_sse_mv.col²

  if use_skin_detection && bsize ≤ 32×32 && level < kDenHigh:
    motion_level = (motion_mag < 16) ? 0 : 1
    consec_zeromv = min over all 8×8 sub-blocks of cpi->consec_zero_mv[]
    is_skin = vp9_compute_skin_block(...)
  if !is_skin && level==kDenHigh: increase_denoising = 1

  if last_is_reference && level ≥ kDenLow && !ctx->sb_skip_denoising:
    decision = perform_motion_compensation(...)
  else:
    decision = COPY_BLOCK

  if decision == FILTER_BLOCK:
    decision = vp9_denoiser_filter(...)        // RTCD dispatch

  if decision == FILTER_BLOCK:
    copy avg → src                              // commit denoise
  else:                                         // COPY_BLOCK
    copy src → avg                              // keep src in running_avg

  if decision == FILTER_BLOCK && zeromv_filter:
    decision = FILTER_ZEROMV_BLOCK
```

`perform_motion_compensation` (line 187-334):
1. Reject if skin and `(motion_mag > 0 || consec_zeromv < 4)`.
2. Reject (COPY_BLOCK) for the smallest blocks: BLOCK_8X8, 8X16, 16X8,
   and 16X16 if `width > 480 && level <= kDenLow`.
3. If the best-SSE reference is LAST_FRAME (or GOLDEN when not using gf-ref)
   *and* (zeromv_sse - newmv_sse) > `sse_diff_thresh = num_pixels << 4`
   (or `<< 2` when motion_mag is large and increase_denoising), use the
   non-zero-MV mode.
4. Else fall back to ZEROMV mode (bias toward LAST_FRAME).
5. Reject if `newmv_sse > sse_thresh = num_pixels * (increase_denoising?80:40)`
   (line 273-277).
6. Reject if `motion_mag > (noise_motion_thresh=625) << 3 = 5000` (line
   278-282).
7. Else run motion compensation onto `running_avg_y[best_ref]`, writing
   into `mc_running_avg_y[layer]`.

Thresholds, verbatim (`vp9_denoiser.c:28-63`):
```
absdiff_thresh(bs, inc)            = 3 + (inc ? 1 : 0)
delta_thresh(bs, inc)              = 4
noise_motion_thresh(bs, inc)       = 625
sse_thresh(bs, inc)                = num_pixels * (inc ? 80 : 40)
sse_diff_thresh(bs, inc, motion)   = motion > 625
                                       ? (inc ? num_pixels<<2 : 0)
                                       : num_pixels<<4
total_adj_weak_thresh(bs, inc)     = num_pixels * (inc ? 3 : 2)
total_adj_strong_thresh(bs, inc)   = num_pixels * (inc ? 3 : 2)
```
(`num_pixels = 1 << num_pels_log2_lookup[bs]`.)

### 4.15 Noise estimator

(`vp9_noise_estimate.c:109-302`)

Per-frame (every `frame_period = 8` frames):
1. Compute `num_low_motion` = number of 8×8 blocks with
   `consec_zero_mv[bl] > thresh_consec_zeromv` (6 by default; 2 when
   denoising enabled at HD).
2. `frame_low_motion = 1 if num_low_motion >= 3/8 * total_blocks`.
3. For every other 16×16 block at stride 4 (1/16 of the frame), where
   all four 8×8 sub-blocks are low-motion and *not* skin, compute the
   variance between current and last source and bin it into a 20-bin
   histogram (`bin_size=100`).
4. Adjust histogram for dark-scene roll-off (line 247-256).
5. 3-bin smoothed histogram → find `max_bin`.
6. `ne->value = (3*ne->value + max_bin * 40) >> 2`.
7. Bump `ne->count`. If `ne->value > adapt_thresh` and current
   `ne->level < kMedium`, snap `count` to `num_frames_estimate` to
   force a re-classification this frame.
8. After `num_frames_estimate` (15 first time, 30 thereafter) frames,
   classify using `vp9_noise_estimate_extract_level()` (§2.5) and call
   `vp9_denoiser_set_noise_level()`.

Forced overrides:
- `cm->use_highbitdepth` → noise estimate always disabled (line 53-55).
- `frame_counter > 60 && rc.avg_frame_low_motion < 60` (40 for HD) →
  force `level = kLowLow` (line 145-160).
- `screen content`, `SVC` (except top spatial layer), `resize_state !=
  ORIG`, `width*height < 640*360`, or non-CBR-CYCLIC_REFRESH path: noise
  estimation disabled entirely (line 52-74).

## 5. Tables and constants — verbatim

### 5.1 AC quantiser look-up (8-bit)

(`vp9/common/vp9_quant_common.c:89-110`)

```c
static const int16_t ac_qlookup[QINDEX_RANGE] = {
  4,    8,    9,    10,   11,   12,   13,   14,   15,   16,   17,   18,   19,
  20,   21,   22,   23,   24,   25,   26,   27,   28,   29,   30,   31,   32,
  33,   34,   35,   36,   37,   38,   39,   40,   41,   42,   43,   44,   45,
  46,   47,   48,   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,
  59,   60,   61,   62,   63,   64,   65,   66,   67,   68,   69,   70,   71,
  72,   73,   74,   75,   76,   77,   78,   79,   80,   81,   82,   83,   84,
  85,   86,   87,   88,   89,   90,   91,   92,   93,   94,   95,   96,   97,
  98,   99,   100,  101,  102,  104,  106,  108,  110,  112,  114,  116,  118,
  120,  122,  124,  126,  128,  130,  132,  134,  136,  138,  140,  142,  144,
  146,  148,  150,  152,  155,  158,  161,  164,  167,  170,  173,  176,  179,
  182,  185,  188,  191,  194,  197,  200,  203,  207,  211,  215,  219,  223,
  227,  231,  235,  239,  243,  247,  251,  255,  260,  265,  270,  275,  280,
  285,  290,  295,  300,  305,  311,  317,  323,  329,  335,  341,  347,  353,
  359,  366,  373,  380,  387,  394,  401,  408,  416,  424,  432,  440,  448,
  456,  465,  474,  483,  492,  501,  510,  520,  530,  540,  550,  560,  571,
  582,  593,  604,  615,  627,  639,  651,  663,  676,  689,  702,  715,  729,
  743,  757,  771,  786,  801,  816,  832,  848,  864,  881,  898,  915,  933,
  951,  969,  988,  1007, 1026, 1046, 1066, 1087, 1108, 1129, 1151, 1173, 1196,
  1219, 1243, 1267, 1292, 1317, 1343, 1369, 1396, 1423, 1451, 1479, 1508, 1537,
  1567, 1597, 1628, 1660, 1692, 1725, 1759, 1793, 1828,
};
```

### 5.2 DC quantiser look-up (8-bit)

(`vp9/common/vp9_quant_common.c:15-35`)

```c
static const int16_t dc_qlookup[QINDEX_RANGE] = {
  4,    8,    8,    9,    10,  11,  12,  12,  13,  14,  15,   16,   17,   18,
  19,   19,   20,   21,   22,  23,  24,  25,  26,  26,  27,   28,   29,   30,
  31,   32,   32,   33,   34,  35,  36,  37,  38,  38,  39,   40,   41,   42,
  43,   43,   44,   45,   46,  47,  48,  48,  49,  50,  51,   52,   53,   53,
  54,   55,   56,   57,   57,  58,  59,  60,  61,  62,  62,   63,   64,   65,
  66,   66,   67,   68,   69,  70,  70,  71,  72,  73,  74,   74,   75,   76,
  77,   78,   78,   79,   80,  81,  81,  82,  83,  84,  85,   85,   87,   88,
  90,   92,   93,   95,   96,  98,  99,  101, 102, 104, 105,  107,  108,  110,
  111,  113,  114,  116,  117, 118, 120, 121, 123, 125, 127,  129,  131,  134,
  136,  138,  140,  142,  144, 146, 148, 150, 152, 154, 156,  158,  161,  164,
  166,  169,  172,  174,  177, 180, 182, 185, 187, 190, 192,  195,  199,  202,
  205,  208,  211,  214,  217, 220, 223, 226, 230, 233, 237,  240,  243,  247,
  250,  253,  257,  261,  265, 269, 272, 276, 280, 284, 288,  292,  296,  300,
  304,  309,  313,  317,  322, 326, 330, 335, 340, 344, 349,  354,  359,  364,
  369,  374,  379,  384,  389, 395, 400, 406, 411, 417, 423,  429,  435,  441,
  447,  454,  461,  467,  475, 482, 489, 497, 505, 513, 522,  530,  539,  549,
  559,  569,  579,  590,  602, 614, 626, 640, 654, 668, 684,  700,  717,  736,
  755,  775,  796,  819,  843, 869, 896, 925, 955, 988, 1022, 1058, 1098, 1139,
  1184, 1232, 1282, 1336,
};
```

### 5.3 10-bit and 12-bit quantiser look-ups

`ac_qlookup_10`, `ac_qlookup_12`, `dc_qlookup_10`, `dc_qlookup_12` —
transcribed in `vp9/common/vp9_quant_common.c:38-86, 113-161`.
**Pulled into the corpus by A4 (bitstream/entropy/tables); not duplicated
here to avoid drift.** Our port must use these exact arrays if HBD is in
scope (Q1).

### 5.4 RC tuning constants in `vp9_ratectrl.{h,c}`

| Constant | Value | File:line |
|---|---|---|
| `BPER_MB_NORMBITS` | 9 | `vp9_ratectrl.h:28` |
| `DEFAULT_KF_BOOST` | 2000 | `vp9_ratectrl.h:30` |
| `DEFAULT_GF_BOOST` | 2000 | `vp9_ratectrl.h:31` |
| `MIN_GF_INTERVAL` | 4 | `vp9_ratectrl.h:33` |
| `MAX_GF_INTERVAL` | 16 | `vp9_ratectrl.h:34` |
| `FIXED_GF_INTERVAL` | 8 | `vp9_ratectrl.h:35` |
| `FRAME_OVERHEAD_BITS` | 200 | `vp9_ratectrl.h:38` |
| `STATIC_KF_GROUP_THRESH` | 99 | `vp9_ratectrl.h:43` |
| `MAX_STATIC_GF_GROUP_LENGTH` | 250 | `vp9_ratectrl.h:46` |
| `MAX_MB_RATE` | 250 | `vp9_ratectrl.c:39` |
| `MAXRATE_1080P` | 4_000_000 | `vp9_ratectrl.c:40` |
| `LIMIT_QRANGE_FOR_ALTREF_AND_KEY` | 1 | `vp9_ratectrl.c:42` |
| `MIN_BPB_FACTOR` | 0.005 | `vp9_ratectrl.c:44` |
| `MAX_BPB_FACTOR` | 50 | `vp9_ratectrl.c:45` |
| `gf_high` (default) | 2000 | `vp9_ratectrl.c:96` |
| `gf_low` | 400 | `vp9_ratectrl.c:97` |
| `kf_high` (default) | 4800 | `vp9_ratectrl.c:98` |
| `kf_low` (default) | 300 | `vp9_ratectrl.c:99` |
| `gf_high` (AGGRESSIVE_VBR) | 2400 | `vp9_ratectrl.c:91` |
| `kf_high` (AGGRESSIVE_VBR) | 4000 | `vp9_ratectrl.c:93` |
| `kf_low` (AGGRESSIVE_VBR) | 400 | `vp9_ratectrl.c:94` |
| `STATIC_MOTION_THRESH` | 95 | `vp9_ratectrl.c:1320` |
| `VBR_PCT_ADJUSTMENT_LIMIT` | 50 | `vp9_ratectrl.c:2677` |
| `SMOOTH_PCT_MIN` | 0.1 | `vp9_ratectrl.c:1115` |
| `SMOOTH_PCT_DIV` | 0.05 | `vp9_ratectrl.c:1116` |

Frame-scale constants (`vp9_ratectrl.h:79-86`):
```c
static const int    frame_scale_factor[FRAME_SCALE_STEPS] = { 16, 24 };
static const double rate_thresh_mult[FRAME_SCALE_STEPS]   = { 1.0, 2.0 };
static const double rcf_mult[FRAME_SCALE_STEPS]           = { 1.0, 2.0 };
```

VBR I-frame ratio (`vp9_ratectrl.c:2040`): `kf_ratio = 25`.

Rate-factor deltas (`vp9_ratectrl.c:1306-1312`):
```c
static const double rate_factor_deltas[RATE_FACTOR_LEVELS] = {
  1.00,  // INTER_NORMAL
  1.00,  // INTER_HIGH
  1.50,  // GF_ARF_LOW
  1.75,  // GF_ARF_STD
  2.00,  // KF_STD
};
```

VBR fixed-GF delta-rate pattern (VPX_Q mode, `vp9_ratectrl.c:1227-1228`):
```c
double delta_rate[FIXED_GF_INTERVAL] = { 0.50, 1.0, 0.85, 1.0,
                                         0.70, 1.0, 0.85, 1.0 };
```

Scene-detection thresholds (`vp9_ratectrl.c:3076-3081`):
```c
uint32_t min_thresh = 20000;    // ~5 * 64 * 64
float    thresh     = 8.0f;     // CBR (default)
uint32_t thresh_key = 140000;   // (240000 for speed <= 5)
if (content != SCREEN) min_thresh = 65000;
if (rc_mode == VPX_VBR) thresh = 2.1f;
```

VBR boost-lag SAD thresholds (`vp9_ratectrl.c:2895-2896`):
```c
uint32_t sad_thresh1 = 70000;
uint32_t sad_thresh2 = 120000;
```

VBR encode-overshoot detection (`vp9_ratectrl.c:3252-3253`):
```c
int thresh_qp   = 7 * (rc->worst_quality >> 3);       // 3/4 max for screen
                  (or 3 * (rc->worst_quality >> 2)    // 3/4 for video)
int thresh_rate = rc->avg_frame_bandwidth << 3;       // 8× per-frame avg
```

Resize thresholds (`vp9_ratectrl.c:2745-2769`):
```
avg_qp_thr1 = 70   (60 if denoiser on)
avg_qp_thr2 = 50   (40 if denoiser on)
min_width   = 320*4/3  ≈ 426     // resize floor 320×180
min_height  = 180*4/3  ≈ 240
HD-rate trigger: avg_bw < 300000/30 → DOWN_ONEHALF
                 avg_bw < 400000/30 → DOWN_THREEFOUR (or ONEHALFONLY)
```

### 5.5 CYCLIC_REFRESH tuning constants

(`vp9_aq_cyclicrefresh.{h,c}`)

| Constant | Value | File:line |
|---|---|---|
| `CR_SEGMENT_ID_BASE` | 0 | `vp9_aq_cyclicrefresh.h:25` |
| `CR_SEGMENT_ID_BOOST1` | 1 | `vp9_aq_cyclicrefresh.h:26` |
| `CR_SEGMENT_ID_BOOST2` | 2 | `vp9_aq_cyclicrefresh.h:27` |
| `CR_MAX_RATE_TARGET_RATIO` | 4.0 | `vp9_aq_cyclicrefresh.h:30` |

Defaults set in `vp9_cyclic_refresh_update_parameters`
(`vp9_aq_cyclicrefresh.c:478-592`):

| Field | Default | Conditional |
|---|---|---|
| `percent_refresh` | 10 | 5 if `reduce_refresh`; 5/15 for screen content with `skip_flat_static_blocks`; 0 on VBR golden refresh |
| `max_qdelta_perc` | 60 | 70 for low-resolution non-low-rate |
| `motion_thresh` | 32 | 64 for low-res low-rate |
| `rate_boost_fac` | 15 | 13 if noise≥kMedium or low-res; 10 for screen and VBR |
| `rate_ratio_qdelta` | 2.0 (or 3.0 for first ~4 refresh cycles after KF; 1.7 if noisy; 1.5 for VBR; 2.5+ for low-res non-low-rate) | |
| `time_for_refresh` | 0 | |
| `qp_thresh` | min(35 screen / 20 video, `best_quality<<1`) | |
| `qp_max_thresh` | `117 * MAXQ / 128` ≈ 233 | |
| `thresh_low_motion` | 20 | |

`VP9_VAR_OFFS` (flat-static reference block, line 24-30):
```c
static const uint8_t VP9_VAR_OFFS[64] = {
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128
};
```

### 5.6 Variance AQ tables (`vp9_aq_variance.c`)

```c
#define ENERGY_MIN (-4)
#define ENERGY_MAX (1)
#define ENERGY_SPAN (ENERGY_MAX - ENERGY_MIN + 1)        // = 6

static const double rate_ratio[MAX_SEGMENTS] = { 2.5,  2.0, 1.5, 1.0,
                                                 0.75, 1.0, 1.0, 1.0 };
static const int    segment_id[ENERGY_SPAN] = { 0, 1, 1, 2, 3, 4 };

#define DEFAULT_E_MIDPOINT 10.0
```

### 5.7 Complexity AQ tables (`vp9_aq_complexity.c`)

```c
#define AQ_C_SEGMENTS  5
#define DEFAULT_AQ2_SEG 3       // neutral segment
#define AQ_C_STRENGTHS 3

static const double aq_c_q_adj_factor[AQ_C_STRENGTHS][AQ_C_SEGMENTS] = {
  { 1.75, 1.25, 1.05, 1.00, 0.90 },
  { 2.00, 1.50, 1.15, 1.00, 0.85 },
  { 2.50, 1.75, 1.25, 1.00, 0.80 }
};
static const double aq_c_transitions[AQ_C_STRENGTHS][AQ_C_SEGMENTS] = {
  { 0.15, 0.30, 0.55, 2.00, 100.0 },
  { 0.20, 0.40, 0.65, 2.00, 100.0 },
  { 0.25, 0.50, 0.75, 2.00, 100.0 }
};
static const double aq_c_var_thresholds[AQ_C_STRENGTHS][AQ_C_SEGMENTS] = {
  { -4.0, -3.0, -2.0, 100.00, 100.0 },
  { -3.5, -2.5, -1.5, 100.00, 100.0 },
  { -3.0, -2.0, -1.0, 100.00, 100.0 }
};

#define DEFAULT_LV_THRESH 10.0
#define MIN_DEFAULT_LV_THRESH 8.0
```

### 5.8 360-AQ table (`vp9_aq_360.c:25-26`)

```c
static const double rate_ratio[MAX_SEGMENTS] = { 1.0, 0.75, 0.6, 0.5,
                                                 0.4, 0.3,  0.25 };
```

### 5.9 Denoiser thresholds (`vp9_denoiser.{h,c}`)

```c
#define MOTION_MAGNITUDE_THRESHOLD (8 * 3)   // = 24, vp9_denoiser.h:22
#define NONSVC_REF_FRAMES (MAX_REF_FRAMES - 1)
#define SVC_REF_FRAMES 9

absdiff_thresh(bs, inc)         = 3 + (inc ? 1 : 0)
delta_thresh(bs, inc)           = 4
noise_motion_thresh(bs, inc)    = 625
sse_thresh(bs, inc)             = num_pixels * (inc ? 80 : 40)
sse_diff_thresh(bs, inc, mag)   = (mag > 625) ? (inc ? 4*num_pixels : 0)
                                              : 16*num_pixels
total_adj_weak_thresh(bs, inc)  = num_pixels * (inc ? 3 : 2)
total_adj_strong_thresh(bs, inc)= num_pixels * (inc ? 3 : 2)

adj_val[3] = { 3, 4, 6 }     // per-pixel adjustment per absdiff bucket
```

### 5.10 Noise estimator constants (`vp9_noise_estimate.{h,c}`)

```c
#define MAX_VAR_HIST_BINS 20             // vp9_noise_estimate.h:26

ne->thresh: 90 default, 115 @≥640×360, 140 @≥1280×720, 200 @≥1920×1080
ne->adapt_thresh = (3 * thresh) >> 1
ne->num_frames_estimate = 15 (first) / 30 (later)
frame_period = 8
thresh_consec_zeromv = 6 (or 2 for 640<width≤1920 when denoising)
bin_size = 100
ne->value update: (3*old + max_bin*40) >> 2

level boundaries:  value > 2*thresh → kHigh
                   value > thresh   → kMedium
                   value > thresh/2 → kLow
                   else             → kLowLow
```

### 5.11 Active min-Q polynomial coefficients

(`vp9_ratectrl.c:126-138`, reproduced in §4.5 above)

| LUT | x3 | x2 | x1 |
|---|---|---|---|
| `kf_low_motion_minq` | 0.000001 | -0.0004 | 0.150 |
| `kf_high_motion_minq` | 0.0000021 | -0.00125 | 0.45 |
| `arfgf_low_motion_minq` | 0.0000015 | -0.0009 | 0.30 (0.275 AGG) |
| `arfgf_high_motion_minq` | 0.0000021 | -0.00125 | 0.55 |
| `inter_minq` | 0.00000271 | -0.00113 | 0.70 (0.80 AGG) |
| `rtc_minq` | 0.00000271 | -0.00113 | 0.70 |

These are *not* tables in libvpx — they are *generators*. The generated
tables are 256-entry int arrays per LUT per bit depth (so 6 LUTs × 3 bit
depths = 18 arrays of 256 ints when HBD is on; only the 8-bit set is
generated otherwise). Our port can either: (a) regenerate at init time
using `vp9_ac_quant`+`pow`+`log` matching the libvpx formulas exactly, or
(b) bake static tables. Option (a) is what libvpx does
(`vp9_rc_init_minq_luts`, `vp9_ratectrl.c:142-154`).

## 6. RTCD / SIMD surfaces

Search of `vp9/common/vp9_rtcd_defs.pl`:

```
# vp9_rtcd_defs.pl:122-126
# Denoiser
if (vpx_config("CONFIG_VP9_TEMPORAL_DENOISING") eq "yes") {
  add_proto qw/int vp9_denoiser_filter/,
            "const uint8_t *sig, int sig_stride, "
            "const uint8_t *mc_avg, int mc_avg_stride, "
            "uint8_t *avg, int avg_stride, int increase_denoising, "
            "BLOCK_SIZE bs, int motion_magnitude";
  specialize qw/vp9_denoiser_filter neon sse2/;
}
```

So the only RTCD entry directly owned by this subsystem is
`vp9_denoiser_filter`, with three implementations:

| Symbol | File | ISA | Notes |
|---|---|---|---|
| `vp9_denoiser_filter_c` | `vp9/encoder/vp9_denoiser.c:68-180` | scalar | The reference. Note the TODO at line 65-67 warning that `total_adj` may not be bit-exact across SSE2 and C when `increase_denoising` is true. |
| `vp9_denoiser_filter_sse2` | `vp9/encoder/x86/vp9_denoiser_sse2.c:308` | SSE2 | Dispatches to two block-size groups: `vp9_denoiser_NxM_sse2_big` for 16×16+/32×16/32×32/16×32/64×32/32×64/64×64/16×8, `vp9_denoiser_NxM_sse2_small` for the rest. |
| `vp9_denoiser_filter_neon` | `vp9/encoder/arm/neon/vp9_denoiser_neon.c:338` | NEON (AArch64 / AArch32) | Same block-size dispatch as SSE2. |

Function signature for our port (must match libvpx exactly to keep BD-rate
parity):

```
fn vp9_denoiser_filter(
    sig: *const u8, sig_stride: i32,
    mc_avg: *const u8, mc_avg_stride: i32,
    avg: *mut u8, avg_stride: i32,
    increase_denoising: i32,
    bs: BlockSize,
    motion_magnitude: i32,
) -> i32;     // 0 = COPY_BLOCK, 1 = FILTER_BLOCK
```

Complexity class for the SIMD port: **medium**. The kernel is essentially
a 4-bucket lookup followed by a clipped add (or sub), with two passes and
two threshold checks. SSE2 is unsigned-byte and saturating-arithmetic
heavy; NEON has direct equivalents. The two-pass damping is the
non-obvious part — it can't be expressed as a single horizontal sum and
needs an inter-pass branch on `|total_adj|`.

Other kernels this subsystem *depends on* but does not own (covered by
A3 / A4 specialists):

- `cpi->fn_ptr[bs].sdf` — SAD for `vp9_scene_detection_onepass` and
  `is_superblock_flat_static` (variance/SAD checks).
- `cpi->fn_ptr[bs].vf` — variance for AQ-variance and noise estimation.
- `vpx_convolve_copy` — used as memcpy-with-stride for the per-block
  src↔denoised commit (`vp9_denoiser.c:426-432`).
- `vp9_build_inter_predictors_sby` — used by `perform_motion_compensation`
  to populate `mc_running_avg_y` (`vp9_denoiser.c:323`).
- `vp9_compute_skin_block` — skin detection (`vp9_denoiser.c:399`,
  `vp9_aq_cyclicrefresh.c:181`).

## 7. Open questions for LLD

These are the things the LLD writer needs to resolve before we start
coding. Many are interactions, not isolated questions.

### Q-RC-1. Two-pass: skip entirely or stub?

D1 commits to cpu-used 6–8 which forbids 2-pass anyway. But `vp9_ratectrl.c`
threads 2-pass logic through every function that's also used by 1-pass
(see `pass == 0` checks everywhere). Cleanest design: factor the 1-pass
path into its own type. Risk: `vp9_rc_pick_q_and_bounds` and
`vp9_rc_update_rate_correction_factors` are *currently* used by 2-pass
too; if we narrow them we'll diverge.

Recommendation for LLD: define `RateControlOnePass` with its own
`pick_q`, `regulate_q`, `postencode_update` and forget the 2-pass branches.
Add a divergence-ledger entry citing D1.

### Q-RC-2. SVC layer rate-control fan-out

`vp9_update_buffer_level_svc_preencode`, `update_layer_buffer_level_postencode`,
and the per-layer `LAYER_CONTEXT.rc` mechanics make this subsystem the
core of multi-layer SVC. A6 (threading + SVC) needs to coordinate with us
on which struct owns what. Specifically: in libvpx the current layer's
state is *copied* in and out of `cpi->rc` each call; in Rust we likely
want `&mut LAYER_CONTEXT.rc` directly.

### Q-RC-3. `buffer_level` vs `bits_off_target`

These are kept synchronised everywhere (see TODO at line 246-249 — libvpx
itself notes they should be merged). Recommendation: collapse to one
field in our port and add a divergence-ledger entry. Risk to parity: low
— no algorithm reads them independently.

### Q-RC-4. Q-index search loop is `O(QINDEX_RANGE)` linear

`vp9_rc_regulate_q` walks qindex 0..255 looking for the largest q whose
projected bpm ≤ target_bpm. This is called once per frame, so the cost is
trivial in wall time, but the linear scan masks subtle off-by-one logic
(line 846-855: if the *previous* error was smaller than the current
exact-match error, return `q = i-1`, not `i`). A binary search would be
faster but **must not change which qindex is selected**. Recommendation:
match libvpx loop exactly; add a property-based test that for any
`(frame_type, mbs, rcf, target, abest, aworst)`, our search returns the
same q as a brute-force scan.

### Q-RC-5. Floating-point determinism

`vp9_ratectrl.c` is full of `vpx_clear_system_state()` calls
(e.g., line 751, 1263, 1531, 2625, 4288 in encoder.c). These clear x87
state on x86 to ensure consistent FPU mode. Our port using `f64` /
`libm.log10`/`log` won't have this problem on AArch64 / SSE2, but **the
exact result of `log` differs across libms on x86_64 Linux vs Windows
vs macOS**. This will cause spurious BD-rate divergence in the small
range where `correction_factor` is computed.

Recommendation: use `libm`-via-Rust-`std` and document the dependency.
For byte-parity diagnostic, use the same `log` implementation as libvpx
(this would mean linking against C `libm` from the platform — fine).

### Q-RC-6. `rate_correction_factor` clipping `[0.005, 50]`

This clipping range is **enormous** (4 orders of magnitude). In practice
the factor stays in [0.1, 5]. But adverse content (high-motion 1-pass
VBR opening at the wrong Q) can drive it to the edge. Our port needs to
preserve the clipping or BD-rate will diverge in degenerate cases.

### Q-RC-7. `damped_adjustment` is per-rate-factor-level

Five flags, one per rate factor level. They reset to 0 in `vp9_rc_init`
(line 419-422) but **are never reset thereafter**. This means a frame
type seen only once early in the clip gets undamped adjustment; if it
shows up again 1000 frames later, it gets the damped formula. Confirm
this is intended (it appears to be: the comment says "first frame of each
frame type"). LLD note: state lives on `RateControl`, not local.

### Q-RC-8. Denoiser: SVC ref-slot indexing

The denoiser uses an "INTRA_FRAME + shift" indexing scheme on
`running_avg_y[]` (`vp9_denoiser.c:344-353`) where for the second
spatial layer, the offset is `num_ref_frames`. This is a layered array
disguised as a flat one. Our Rust port should make this explicit:
`running_avg_y[layer_idx][ref_idx]`. Risk: aliasing analysis in libvpx
relies on never reading and writing the same slot in the same frame —
we need to validate.

### Q-RC-9. Denoiser: when does INTRA_FRAME slot get written?

`running_avg_y[INTRA_FRAME (=0) + shift]` is the **current denoised
output buffer**, not the previous intra frame. After
`vp9_denoiser_denoise()` writes the filtered (or copied) block to
`avg = running_avg_y[INTRA_FRAME + shift]`, the next
`vp9_denoiser_update_frame_info()` either swaps or copies it into the
ref-frame slot for whatever refresh flag is set. This dual use of slot 0
is **easy to get wrong** in a port. LLD: rename it
`current_denoised_frame` and store separately.

### Q-RC-10. Denoiser decision feedback to mode decision

When `vp9_denoiser_denoise` returns `FILTER_ZEROMV_BLOCK`, the encoder
*may* re-evaluate ZEROMV via `recheck_zeromv_after_denoising`. This is
controlled by the speed feature `denoise_recheck_zeromv` (A6's domain).
The interaction with cost accounting is delicate — see
`vp9_pickmode.c:1376-1450` and `recheck_zeromv_after_denoising`. **We
should mirror this exactly in v1; the temptation to "clean up" is what
silently breaks BD-rate.**

### Q-RC-11. Denoiser: skin detection coupling

`is_skin && (motion_mag > 0 || consec_zeromv < 4) → COPY_BLOCK`
(`vp9_denoiser.c:211`). This depends on `cpi->use_skin_detection` (speed
feature) and `cpi->consec_zero_mv[]` (a per-8×8 counter updated during
encode). The interaction across CR-AQ, denoiser, and skin-detection
forms a triangle that A2 / A6 should weigh in on.

### Q-RC-12. Cyclic refresh's `last_coded_q_map` is **not** stored
in segmentation map

`cr->last_coded_q_map` is a separate `uint8_t[mi_rows*mi_cols]` from the
segmentation map. It records the qindex each 8×8 block was last coded
at, **including segment delta**. Used to decide whether to refresh a
block now (compared against `qindex_thresh`,
`vp9_aq_cyclicrefresh.c:388-391`). This means cyclic refresh is *not*
purely a function of the segmentation map; the port must persist this
side-channel.

### Q-RC-13. CYCLIC_REFRESH_AQ + post-encode drop interaction

When a frame is dropped post-encode (`post_encode_drop_cbr` returns 1),
`force_max_q = 1` and `cyclic_refresh->counter_encode_maxq_scene_change`
is reset (`vp9_ratectrl.c:3276` — note this happens in the
*pre-drop* re-encode-overshoot path, not in `post_encode_drop_cbr`
itself; the post-encode-drop path doesn't touch CR). The interaction is
**uneven**: an overshoot re-encode and a post-encode drop have different
effects on the cyclic refresh state machine. Document this for the port.

### Q-RC-14. Variance-AQ activation conditions

`vp9_vaq_frame_setup` only runs the segment update on intra-only /
altref / golden-refresh frames (`vp9_aq_variance.c:52-54`). On regular
P frames the segment map is *frozen* from the last setup. This is
performance-sensitive but means the per-block `vp9_vaq_segment_id` is
applied against a possibly-stale segment delta table. Make sure the
port doesn't silently re-setup on every frame; that would change Q.

### Q-RC-15. Real-time disables Variance-AQ / 360-AQ / Complexity-AQ
in `#if !CONFIG_REALTIME_ONLY` block

(`vp9_encoder.c:4206-4239`) — VARIANCE_AQ, EQUATOR360_AQ, COMPLEXITY_AQ,
LOOKAHEAD_AQ are inside `#if !CONFIG_REALTIME_ONLY`. CYCLIC_REFRESH_AQ
is not. **This means under `--enable-realtime-only` builds, AQ modes 1/2/4
are unreachable.** D1 + D4 commit to AQ modes 1/2/3; for v1 our practical
deliverable is *CYCLIC_REFRESH_AQ* (mode 3). VARIANCE / COMPLEXITY land
post-MVP unless we drop the realtime-only gate. **Surface for Arthur**:
do AQ modes 1/2 really need to be in v1, or is mode 3 sufficient given
real-time targets?

### Q-RC-16. `vp9_compute_qdelta_by_rate` is also `O(QINDEX_RANGE)`

(`vp9_ratectrl.c:2562-2584`). Used by every AQ mode to convert a "rate
ratio" into a qindex offset. Same comment as Q-RC-4 — match libvpx, test
exhaustively.

### Q-RC-17. `frame_scale_factor[]` has only two entries

`{16, 24}` (`vp9_ratectrl.h:79`). UNSCALED=16/16=1, SCALE_STEP1=16/24=2/3.
The encoder cannot down-scale below 2/3 via this path; further down-scale
must go through `vp9_resize_one_pass_cbr` which uses 3/4 or 1/2 (a
*different* mechanism). The relationship between these two scaling
paths is confusing; LLD must document. (RESIZE_DYNAMIC is the user-API
mode, `frame_scale_factor` seems to be 2-pass only.)

### Q-RC-18. Active worst-Q "small image format" handling

`(cm->width * cm->height) <= (352 * 288)` triggers `q_adj_factor -= 0.25`
in three places (`vp9_ratectrl.c:1033-1035`, 1175-1178, 1368-1371). This
gives smaller frames a noticeably more aggressive low-Q. Make sure our
port honours this for cif-and-smaller real-time use cases.

### Q-RC-19. Buffer-level critical_level = optimal/8

`vp9_ratectrl.c:940`: `int64_t critical_level = rc->optimal_buffer_level >> 3`.
Below this, `active_worst = worst_quality` (line 996-997). This is the
*hard* lower edge of the buffer model. There's no separate parameter for
it. Document as a derived constant.

### Q-RC-20. `vp9_encodedframe_overshoot` rewrites RCF and buffer state

(`vp9_ratectrl.c:3248-3351`) When `overshoot_detection_cbr_rt ==
RE_ENCODE_MAXQ`, this function — called after a tentative pack — will:
- force `*q = worst_quality`
- reset `cyclic_refresh->counter_encode_maxq_scene_change = 0`
- set `rc->re_encode_maxq_scene_change = 1`
- *recompute* rate_correction_factors via the inverse of the BPM model
- *clobber* `avg_frame_qindex[INTER_FRAME] = *q`, `buffer_level =
  optimal_buffer_level`, `bits_off_target = optimal_buffer_level`,
  `rc_1_frame = rc_2_frame = 0`
- if SVC, propagate to all layers

This is a *huge* state mutation triggered by a heuristic. It is
critical for screen-content scene-change recovery but is also the
single most opaque step in the entire pipeline. **Make sure the port
matches it exactly.**

### Q-RC-21. Noise estimator: forced-low-motion override

(`vp9_noise_estimate.c:145-160`) When the frame counter exceeds 60 and
`rc.avg_frame_low_motion < 60` (40 for low-res), the noise level is
forced to `kLowLow` **and the denoiser is disabled**. This is an
asymmetric escape hatch — it can disable the denoiser, but the inverse
condition cannot re-enable it (only `num_frames_estimate` cycles of
re-estimation can). Document for LLD.

### Q-RC-22. AQ-cyclic refresh `vp9_cyclic_refresh_estimate_bits_at_q`
weighting is *non-conservative*

(`vp9_aq_cyclicrefresh.c:105-130`) The post-encode bit estimate uses the
**actual** seg1/seg2 block counts from the just-encoded frame. The
pre-encode `vp9_cyclic_refresh_rc_bits_per_mb` uses a smoothed
`weight_segment` that blends the *target* seg block count with the
**previous-frame actual**. These are computed slightly differently
(`weight_segment_target = target / num8x8bl` vs
`weight_segment = (target + actual_seg1 + actual_seg2) / 2 / num8x8bl`),
and there's an asymmetric clamp (`if weight_segment_target < 7/8 *
weight_segment` then `weight_segment = weight_segment_target`,
line 576-577). This is the kind of subtle code that **silently shifts
BD-rate by 0.1-0.3%** if you reorder operations.

### Q-RC-23. Denoiser bit-exactness across SSE2/NEON/C is **NOT**
guaranteed by libvpx

See the TODO at `vp9_denoiser.c:65-67`:
> If increase_denoising is enabled in the future, we might need to
> update the code for calculating 'total_adj' in case the C code is not
> bit-exact with corresponding sse2 code.

For our port, this means **we cannot assume bit-parity between scalar
and SIMD** even within libvpx itself for `increase_denoising = 1` paths.
Practical implication: differential testing for the denoiser must allow
small (1-2 pixel ULP) differences, **or** we pick a single canonical
implementation. Recommendation: scalar reference for tests; SSE2/NEON
get differential tests against a *known-good libvpx test vector* rather
than against our scalar.

### Q-RC-24. Cyclic refresh: `time_for_refresh` is always 0 in the
default config

(`vp9_aq_cyclicrefresh.c:509`). It exists in the struct but the only
write of a non-zero value would be from external code. The struct field
is therefore dead in libvpx but kept for API stability. In our port we
can either keep it (for API parity) or drop it (and document). Low risk.

### Q-RC-25. The whole "post_encode_drop_cbr" path is **opt-in**

(`vp9_ratectrl.c:566-612`) Only enabled when `rc.use_post_encode_drop`
is set, which requires `rc.ext_use_post_encode_drop` (set by user API).
Default off. So the standard CBR path **never** invokes it; the encoder
either drops pre-encode (decimation_factor) or doesn't drop at all.
**For v1 real-time, we should match this default.** Skipping
implementation of `post_encode_drop_cbr` in v1 is acceptable; add to
divergence ledger and Q3 followup.

### Q-RC-26. `vp9_scene_detection_onepass` runs **even when scene
detection is not requested**

It is always called from `vp9_encoder.c` before `get_one_pass_*_params`.
Its outputs (`rc.high_source_sad`, `rc.avg_source_sad[]`) gate dozens of
heuristics downstream. There is no way to disable it without re-wiring
the encoder. **Mirror this in the port; the SAD compute is cheap (~1/16
of the frame, decimated).**

### Q-RC-27. Variance-AQ uses `cpi->fn_ptr[bs].vf` (the
**unoptimised** variance)

See the TODO at `vp9_aq_variance.c:86-88`:
> The block_variance calls the unoptimized versions of variance() and
> highbd_8_variance(). It should not.

For our port we should just use the optimised variance and not preserve
this inefficiency. Add to divergence ledger.

### Q-RC-28. Active map / ROI interaction

(`vp9_encoder.c:4224-4232`) When ROI is enabled with background-skip, the
encoder applies cyclic refresh *and* ROI together for the first 20
frames after a key, then only ROI thereafter. This is a niche
production-mode interaction; v1 may defer ROI altogether (it's not in §2
success criteria of PLAN.md). **Decision needed**: is ROI in v1?

### Q-RC-29. `vp9_cyclic_refresh_limit_q` only affects screen content

(`vp9_ratectrl.c:683-685`, `vp9_aq_cyclicrefresh.c:695-702`) — invoked
only when `content == VP9E_CONTENT_SCREEN && aq_mode == CYCLIC_REFRESH_AQ`.
For non-screen CR mode, frame-level Q can drop freely. Document.

### Q-RC-30. `rate_correction_factors[]` is **not** reset on
resolution change

If the encoder resizes mid-stream (RESIZE_DYNAMIC), `rate_correction_factor`
is scaled by `rcf_mult[frame_size_selector]` on read and unscaled on write
(`vp9_ratectrl.c:708, 717`), but the underlying stored value is preserved.
Our port must do the same; this is one of the easier off-by-one
opportunities.

---

**End of A5 deep dive.** The 30 open questions above are sized to inform
the LLD writer; many will collapse together at Stage 2.
