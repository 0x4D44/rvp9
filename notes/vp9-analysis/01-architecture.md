# Architecture and Top-Level Control Flow in libvpx VP9 (Real-Time Encoder)

**Specialist:** A1
**Pinned commit:** libvpx v1.14.1 (`12f3a2a`), under `third_party/libvpx/`.
**Audience:** Stage 2 LLD writers. This document reconstructs the encoder
architecture and lifecycle from libvpx without requiring them to read libvpx.

All file/line citations are into `third_party/libvpx/`.

---

## 1. Scope

This document covers, for libvpx VP9 in real-time configurations
(`cpu-used 5..8`, `--rt`, `--end-usage=cbr|vbr`, optionally SVC):

- The full lifecycle of a `VP9_COMP` encoder instance:
  process-wide one-time init, codec-context creation, per-frame ingest,
  per-frame compress, teardown.
- The `VP9_COMP` "god struct" — verbatim definition, field-by-field semantics,
  ownership/lifetime/aliasing notes, and which subsystem owns each chunk.
- The `VP9EncoderConfig` (`oxcf`) — verbatim definition with semantics.
- The public encoder interface (`vp9/vp9_cx_iface.c`) that fronts `VP9_COMP`.
- The frame loop inside `vp9_get_compressed_data` →
  `encode_frame_to_data_rate` → `encode_without_recode_loop` →
  `vp9_encode_frame` → loop-filter → bitstream-pack → reference-frame update.
- Real-time frame dependency graph: keyframe / last / golden / altref usage;
  how the 8-slot reference pool (`ref_frame_map[8]`) is rotated; how SVC
  layers fit on top.
- Threading **overview only**: how `vp9_ethread.{h,c}` integrates with the
  frame loop. Internals of tile threading, row-mt synchronization, SVC layer
  threading, and `set_rt_speed_feature_*` dispatch are explicitly deferred
  to specialist **A6**.

**Out of scope** (deferred to other specialists, with explicit pointers):

- Encoder mode-decision internals (`vp9_pickmode.c`, `vp9_rdopt.c`,
  partition search): **A2**.
- Motion-estimation algorithms (`vp9_mcomp.c`, sub-pixel, MV prediction):
  **A3**.
- Bitstream/entropy coder, probability tables, header layout
  (`vp9_bitstream.c`, `vp9_entropymode.c`, `vp9_subexp.c`): **A4**.
- Rate control, buffer model, denoiser, AQ modes
  (`vp9_ratectrl.c`, `vp9_denoiser.c`, `vp9_aq_*.c`): **A5**.
- Threading internals + speed-feature dispatch: **A6**.

When this doc cites something owned by another specialist, it gives the
file:line and forwards to them; it does **not** describe the algorithm.

---

## 2. Data structures

### 2.1 `VP9EncoderConfig` — the configuration record (`oxcf`)

The whole encoder is configured by a `VP9EncoderConfig` value, generally
referred to in code as `oxcf` (legacy "On2 xcoder config"). The public
`vpx_codec_enc_cfg_t` from `vpx/vp8cx.h` is translated to `VP9EncoderConfig`
by `set_encoder_config()` at
`third_party/libvpx/vp9/vp9_cx_iface.c:500`. The translation reads
`cfg` (the user-visible `vpx_codec_enc_cfg_t`) and `extra_cfg`
(a private `vp9_extracfg` that carries fields with no public knob, such as
`cpu_used`, `aq_mode`, `noise_sensitivity`, `tile_columns`) and emits
`oxcf` (`third_party/libvpx/vp9/vp9_cx_iface.c:506-666`).

#### Verbatim definition

`third_party/libvpx/vp9/encoder/vp9_encoder.h:153-286`:

```c
typedef struct VP9EncoderConfig {
  BITSTREAM_PROFILE profile;
  vpx_bit_depth_t bit_depth;     // Codec bit-depth.
  int width;                     // width of data passed to the compressor
  int height;                    // height of data passed to the compressor
  unsigned int input_bit_depth;  // Input bit depth.
  double init_framerate;         // set to passed in framerate
  vpx_rational_t g_timebase;  // equivalent to g_timebase in vpx_codec_enc_cfg_t
  vpx_rational64_t g_timebase_in_ts;  // g_timebase * TICKS_PER_SEC

  int64_t target_bandwidth;  // bandwidth to be used in bits per second

  int noise_sensitivity;  // pre processing blur: recommendation 0
  int sharpness;          // sharpening output: recommendation 0:
  int speed;
  // maximum allowed bitrate for any intra frame in % of bitrate target.
  unsigned int rc_max_intra_bitrate_pct;
  // maximum allowed bitrate for any inter frame in % of bitrate target.
  unsigned int rc_max_inter_bitrate_pct;
  // percent of rate boost for golden frame in CBR mode.
  unsigned int gf_cbr_boost_pct;

  MODE mode;
  int pass;

  // Key Framing Operations
  int auto_key;  // autodetect cut scenes and set the keyframes
  int key_freq;  // maximum distance to key frame.

  int lag_in_frames;  // how many frames lag before we start encoding

  // ----------------------------------------------------------------
  // DATARATE CONTROL OPTIONS

  // vbr, cbr, constrained quality or constant quality
  enum vpx_rc_mode rc_mode;

  // buffer targeting aggressiveness
  int under_shoot_pct;
  int over_shoot_pct;

  // buffering parameters
  int64_t starting_buffer_level_ms;
  int64_t optimal_buffer_level_ms;
  int64_t maximum_buffer_size_ms;

  // Frame drop threshold.
  int drop_frames_water_mark;

  // controlling quality
  int fixed_q;
  int worst_allowed_q;
  int best_allowed_q;
  int cq_level;
  AQ_MODE aq_mode;  // Adaptive Quantization mode

  // Special handling of Adaptive Quantization for AltRef frames
  int alt_ref_aq;

  // Internal frame size scaling.
  RESIZE_TYPE resize_mode;
  int scaled_frame_width;
  int scaled_frame_height;

  // Enable feature to reduce the frame quantization every x frames.
  int frame_periodic_boost;

  // two pass datarate control
  int two_pass_vbrbias;  // two pass datarate control tweaks
  int two_pass_vbrmin_section;
  int two_pass_vbrmax_section;
  int vbr_corpus_complexity;  // 0 indicates corpus vbr disabled
  // END DATARATE CONTROL OPTIONS
  // ----------------------------------------------------------------

  // Spatial and temporal scalability.
  int ss_number_layers;  // Number of spatial layers.
  int ts_number_layers;  // Number of temporal layers.
  // Bitrate allocation for spatial layers.
  int layer_target_bitrate[VPX_MAX_LAYERS];
  int ss_target_bitrate[VPX_SS_MAX_LAYERS];
  int ss_enable_auto_arf[VPX_SS_MAX_LAYERS];
  // Bitrate allocation (CBR mode) and framerate factor, for temporal layers.
  int ts_rate_decimator[VPX_TS_MAX_LAYERS];

  int enable_auto_arf;

  int encode_breakout;  // early breakout : for video conf recommend 800

  /* Bitfield defining the error resiliency features to enable.
   * Can provide decodable frames after losses in previous
   * frames and decodable partitions after losses in the same frame.
   */
  unsigned int error_resilient_mode;

  /* Bitfield defining the parallel decoding mode where the
   * decoding in successive frames may be conducted in parallel
   * just by decoding the frame headers.
   */
  unsigned int frame_parallel_decoding_mode;

  int arnr_max_frames;
  int arnr_strength;

  int min_gf_interval;
  int max_gf_interval;

  int tile_columns;
  int tile_rows;

  int enable_tpl_model;

  int max_threads;

  unsigned int target_level;

  vpx_fixed_buf_t two_pass_stats_in;

  vp8e_tuning tuning;
  vp9e_tune_content content;
#if CONFIG_VP9_HIGHBITDEPTH
  int use_highbitdepth;
#endif
  vpx_color_space_t color_space;
  vpx_color_range_t color_range;
  int render_width;
  int render_height;
  VP9E_TEMPORAL_LAYERING_MODE temporal_layering_mode;

  int row_mt;
  unsigned int motion_vector_unit_test;
  int delta_q_uv;
  int use_simple_encode_api;  // Use SimpleEncode APIs or not
} VP9EncoderConfig;
```

#### Field-by-field semantics (real-time focus)

- `profile` (`BITSTREAM_PROFILE`): VP9 profile 0–3. Profile 0 is 8-bit 4:2:0;
  profile 2 is 10/12-bit 4:2:0. Real-time uses profile 0 almost exclusively.
- `bit_depth`, `input_bit_depth`: codec bit depth and input bit depth in bits.
- `width`, `height`: frame dimensions in pixels.
- `init_framerate`: source frame rate (computed from `g_timebase.num /
  g_timebase.den`); clamped to 30 if > 180 (`vp9_cx_iface.c:515-516`).
- `g_timebase`, `g_timebase_in_ts`: input/output timestamp scaling. Real-time
  uses `TICKS_PER_SEC = 10_000_000` (`vp9_encoder.h:67`).
- `target_bandwidth`: target bitrate in bits/sec. Derived from
  `cfg->rc_target_bitrate * 1000` and capped to `raw_target_rate` or 1 Gbps
  (`vp9_cx_iface.c:532-540`).
- `noise_sensitivity`: 0 = denoiser off; >0 enables `VP9_DENOISER`
  (subsystem **A5**).
- `sharpness`: post-deblocking strength (0–7). Influences loop filter.
- `speed`: `cpu-used` from `extra_cfg->cpu_used` (`vp9_cx_iface.c:582`).
  Range −9..9. Higher = faster, lower BD-rate gains. **Real-time gates
  cpu-used 5–8.** Maps to speed features via `vp9_speed_features.c`
  (specialist **A6**).
- `mode`: `MODE` enum (`vp9_encoder.h:95-109`): `GOOD`, `BEST`, `REALTIME`.
  Real-time uses `REALTIME` (set by `pick_quickcompress_mode()` when the
  deadline is `VPX_DL_REALTIME`, `vp9_cx_iface.c:1203-1206`).
- `pass`: 0 = one-pass, 1 = first pass (stats collection), 2 = second pass
  (consumes stats). Real-time is always one-pass.
- `auto_key`, `key_freq`: keyframe interval policy.
- `lag_in_frames`: lookahead depth. `MIN_LOOKAHEAD_FOR_ARFS = 4`
  (`vp9_encoder.h:1422`); `is_altref_enabled` requires lag ≥ 4
  (`vp9_encoder.h:1423-1427`). Real-time + CBR has altref disabled in
  `is_altref_enabled`.
- `rc_mode`: `VPX_VBR`, `VPX_CBR`, `VPX_CQ`, `VPX_Q`.
- `under_shoot_pct`, `over_shoot_pct`: VBR aggressiveness.
- `starting_buffer_level_ms`, `optimal_buffer_level_ms`,
  `maximum_buffer_size_ms`: CBR buffer model in milliseconds.
- `drop_frames_water_mark`: % of buffer below which frame dropping kicks in
  (CBR only; specialist **A5**).
- `fixed_q`, `worst_allowed_q`, `best_allowed_q`, `cq_level`: q-index range
  (`QINDEX_RANGE = 256`). Lossless is `worst == best == 0`
  (`vp9_encoder.h:288-290`).
- `aq_mode`: see `AQ_MODE` enum (`vp9_encoder.h:117-129`):
  `NO_AQ`, `VARIANCE_AQ` (1), `COMPLEXITY_AQ` (2), `CYCLIC_REFRESH_AQ` (3),
  `EQUATOR360_AQ` (4), `PERCEPTUAL_AQ` (5), `PSNR_AQ` (6), `LOOKAHEAD_AQ` (7).
  Real-time + CBR commonly uses `CYCLIC_REFRESH_AQ` (3). Specialist **A5**.
- `alt_ref_aq`: special AQ for altref frames.
- `resize_mode`: `RESIZE_NONE | RESIZE_FIXED | RESIZE_DYNAMIC`
  (`vp9_encoder.h:131-135`).
- `scaled_frame_width/height`, `resize_*`: dynamic resampling controls.
- `frame_periodic_boost`: occasional QP reduction.
- `two_pass_*`, `vbr_corpus_complexity`, `two_pass_stats_in`: 2-pass.
  Not used in real-time (`pass = 0`).
- `ss_number_layers`, `ts_number_layers`, `layer_target_bitrate`,
  `ss_target_bitrate`, `ss_enable_auto_arf`, `ts_rate_decimator`:
  spatial / temporal SVC config (specialist **A6** for the encode-time
  layer dispatch).
- `enable_auto_arf`: 0 = disable altref; ≥ 1 enable; ≥ 2 enables
  `multi_layer_arf` (`vp9_encoder.c:6394-6398`). Real-time CBR effectively
  disables altref via `is_altref_enabled` (`vp9_encoder.h:1423`).
- `encode_breakout`: SSE threshold for skipping mode evaluation. 800
  recommended for HD video conferencing (`vp9_encoder.h:846-848`).
- `error_resilient_mode`, `frame_parallel_decoding_mode`: bitstream bits
  for resilient streaming.
- `arnr_max_frames`, `arnr_strength`: alt-ref noise reduction (off in RT).
- `min_gf_interval`, `max_gf_interval`: golden-frame spacing bounds.
- `tile_columns`, `tile_rows`: log2 tile dimensions. **In libvpx, when
  `max_threads > 1` and `tile_columns > 0`, `tile_rows` is force-zeroed**
  to avoid row-dependency issues across tiles (`vp9_cx_iface.c:617-620`).
  Tile layout determined by `vp9_get_tile_n_bits()`; max columns capped by
  level (`vp9_encoder.h:1468-1480`).
- `enable_tpl_model`: temporal-propagation lookahead model (2-pass good
  quality only).
- `max_threads`: upper bound on `cpi->num_workers`.
- `target_level`: VP9 level (e.g. 4.1, 5.0). `LEVEL_AUTO` autoselects.
  Constants in `vp9_level_defs[]` (`vp9_encoder.h:450`).
- `two_pass_stats_in`: 2-pass first-pass stats input. NULL in real-time.
- `tuning`: `vp8e_tuning` enum (PSNR / SSIM).
- `content`: `vp9e_tune_content` enum (DEFAULT, SCREEN, FILM). Real-time
  WebRTC defaults `DEFAULT`; screen-share uses `VP9E_CONTENT_SCREEN`.
- `color_space`, `color_range`, `render_width`, `render_height`: bitstream
  metadata.
- `temporal_layering_mode`: SVC temporal pattern selector
  (`VP9E_TEMPORAL_LAYERING_MODE_*`, see `vpx/vp8cx.h`). `0212`, `0101`,
  `BYPASS`, `NOLAYERING`. Implemented in `vp9_svc_layercontext.c:819-830`.
- `row_mt`: row-based multi-threading flag (specialist **A6**).
- `motion_vector_unit_test`: testing knob.
- `delta_q_uv`: per-plane QP delta.
- `use_simple_encode_api`: alternative API used in some testing/rate-ctrl
  tooling (`CONFIG_RATE_CTRL`).

### 2.2 `MODE` enum

`third_party/libvpx/vp9/encoder/vp9_encoder.h:95-109`:

```c
typedef enum {
  // Good Quality Fast Encoding. The encoder balances quality with the amount of
  // time it takes to encode the output. Speed setting controls how fast.
  GOOD,

  // The encoder places priority on the quality of the output over encoding
  // speed. The output is compressed at the highest possible quality. This
  // option takes the longest amount of time to encode. Speed setting ignored.
  BEST,

  // Realtime/Live Encoding. This mode is optimized for realtime encoding (for
  // example, capturing a television signal or feed from a live camera). Speed
  // setting controls how fast.
  REALTIME
} MODE;
```

The dispatcher `pick_quickcompress_mode()` (`vp9_cx_iface.c:1170-1212`)
sets `oxcf->mode = REALTIME` when the deadline is `VPX_DL_REALTIME` and
calls `vp9_change_config()` on every transition.

### 2.3 `FRAMETYPE_FLAGS` and `AQ_MODE` enums

`third_party/libvpx/vp9/encoder/vp9_encoder.h:111-129`:

```c
typedef enum {
  FRAMEFLAGS_KEY = 1 << 0,
  FRAMEFLAGS_GOLDEN = 1 << 1,
  FRAMEFLAGS_ALTREF = 1 << 2,
} FRAMETYPE_FLAGS;

typedef enum {
  NO_AQ = 0,
  VARIANCE_AQ = 1,
  COMPLEXITY_AQ = 2,
  CYCLIC_REFRESH_AQ = 3,
  EQUATOR360_AQ = 4,
  PERCEPTUAL_AQ = 5,
  PSNR_AQ = 6,
  // AQ based on lookahead temporal
  // variance (only valid for altref frames)
  LOOKAHEAD_AQ = 7,
  AQ_MODE_COUNT  // This should always be the last member of the enum
} AQ_MODE;
```

`CONTENT_STATE_SB` (`vp9_encoder.h:137-145`) classifies per-superblock
content for CBR scene-change handling; consumed by **A5**.

### 2.4 `VP9_COMP` — the encoder "god struct"

`VP9_COMP` is the single state container the codec library hands around
internally. It is allocated 32-byte aligned with `vpx_memalign` in
`vp9_create_compressor()` (`vp9_encoder.c:2380-2383`) and zeroed via
`vp9_zero(*cpi)` (`vp9_encoder.c:2388`).

#### Verbatim definition

`third_party/libvpx/vp9/encoder/vp9_encoder.h:727-1069`:

```c
typedef struct VP9_COMP {
  FRAME_INFO frame_info;
  QUANTS quants;
  ThreadData td;
  MB_MODE_INFO_EXT *mbmi_ext_base;
  DECLARE_ALIGNED(16, int16_t, y_dequant[QINDEX_RANGE][8]);
  DECLARE_ALIGNED(16, int16_t, uv_dequant[QINDEX_RANGE][8]);
  VP9_COMMON common;
  VP9EncoderConfig oxcf;
  struct lookahead_ctx *lookahead;
  struct lookahead_entry *alt_ref_source;

  YV12_BUFFER_CONFIG *Source;
  YV12_BUFFER_CONFIG *Last_Source;  // NULL for first frame and alt_ref frames
  YV12_BUFFER_CONFIG *un_scaled_source;
  YV12_BUFFER_CONFIG scaled_source;
  YV12_BUFFER_CONFIG *unscaled_last_source;
  YV12_BUFFER_CONFIG scaled_last_source;
#ifdef ENABLE_KF_DENOISE
  YV12_BUFFER_CONFIG raw_unscaled_source;
  YV12_BUFFER_CONFIG raw_scaled_source;
#endif
  YV12_BUFFER_CONFIG *raw_source_frame;

  BLOCK_SIZE tpl_bsize;
  TplDepFrame tpl_stats[MAX_ARF_GOP_SIZE];
  // Used to store TPL stats before propagation
  VpxTplGopStats tpl_gop_stats;
  YV12_BUFFER_CONFIG *tpl_recon_frames[REF_FRAMES];
  EncFrameBuf enc_frame_buf[REF_FRAMES];
#if CONFIG_MULTITHREAD
  pthread_mutex_t kmeans_mutex;
#endif
  int kmeans_data_arr_alloc;
  KMEANS_DATA *kmeans_data_arr;
  int kmeans_data_size;
  int kmeans_data_stride;
  double kmeans_ctr_ls[MAX_KMEANS_GROUPS];
  double kmeans_boundary_ls[MAX_KMEANS_GROUPS];
  int kmeans_count_ls[MAX_KMEANS_GROUPS];
  int kmeans_ctr_num;
#if CONFIG_NON_GREEDY_MV
  MotionFieldInfo motion_field_info;
  int tpl_ready;
  int_mv *select_mv_arr;
#endif

  TileDataEnc *tile_data;
  int allocated_tiles;  // Keep track of memory allocated for tiles.

  int scaled_ref_idx[REFS_PER_FRAME];
  int lst_fb_idx;
  int gld_fb_idx;
  int alt_fb_idx;

  int ref_fb_idx[REF_FRAMES];

  int refresh_last_frame;
  int refresh_golden_frame;
  int refresh_alt_ref_frame;

  int ext_refresh_frame_flags_pending;
  int ext_refresh_last_frame;
  int ext_refresh_golden_frame;
  int ext_refresh_alt_ref_frame;

  int ext_refresh_frame_context_pending;
  int ext_refresh_frame_context;

  int64_t norm_wiener_variance;
  int64_t *mb_wiener_variance;
  int mb_wiener_var_rows;
  int mb_wiener_var_cols;
  double *mi_ssim_rdmult_scaling_factors;

  YV12_BUFFER_CONFIG last_frame_uf;

  TOKENEXTRA *tile_tok[4][1 << 6];
  TOKENLIST *tplist[4][1 << 6];

  // Ambient reconstruction err target for force key frames
  int64_t ambient_err;

  RD_CONTROL rd_ctrl;
  RD_OPT rd;

  CODING_CONTEXT coding_context;

  int *nmvcosts[2];
  int *nmvcosts_hp[2];
  int *nmvsadcosts[2];
  int *nmvsadcosts_hp[2];

  int64_t last_time_stamp_seen;
  int64_t last_end_time_stamp_seen;
  int64_t first_time_stamp_ever;

  RATE_CONTROL rc;
  double framerate;

  int interp_filter_selected[REF_FRAMES][SWITCHABLE];

  struct vpx_codec_pkt_list *output_pkt_list;

  MBGRAPH_FRAME_STATS mbgraph_stats[MAX_LAG_BUFFERS];
  int mbgraph_n_frames;  // number of frames filled in the above
  int static_mb_pct;     // % forced skip mbs by segmentation
  int ref_frame_flags;

  SPEED_FEATURES sf;

  uint32_t max_mv_magnitude;
  int mv_step_param;

  int allow_comp_inter_inter;

  // Default value is 1. From first pass stats, encode_breakout may be disabled.
  ENCODE_BREAKOUT_TYPE allow_encode_breakout;

  // Get threshold from external input. A suggested threshold is 800 for HD
  // clips, and 300 for < HD clips.
  int encode_breakout;

  uint8_t *segmentation_map;

  uint8_t *skin_map;

  // segment threshold for encode breakout
  int segment_encode_breakout[MAX_SEGMENTS];

  CYCLIC_REFRESH *cyclic_refresh;
  ActiveMap active_map;

  fractional_mv_step_fp *find_fractional_mv_step;
  struct scale_factors me_sf;
  vp9_diamond_search_fn_t diamond_search_sad;
  vp9_variance_fn_ptr_t fn_ptr[BLOCK_SIZES];
  uint64_t time_receive_data;
  uint64_t time_compress_data;
  uint64_t time_pick_lpf;
  uint64_t time_encode_sb_row;

  TWO_PASS twopass;

  // Force recalculation of segment_ids for each mode info
  uint8_t force_update_segmentation;

  YV12_BUFFER_CONFIG alt_ref_buffer;

  // class responsible for adaptive
  // quantization of altref frames
  struct ALT_REF_AQ *alt_ref_aq;

#if CONFIG_INTERNAL_STATS
  unsigned int mode_chosen_counts[MAX_MODES];

  int count;
  uint64_t total_sq_error;
  uint64_t total_samples;
  ImageStat psnr;

  uint64_t totalp_sq_error;
  uint64_t totalp_samples;
  ImageStat psnrp;

  double total_blockiness;
  double worst_blockiness;

  int bytes;
  double summed_quality;
  double summed_weights;
  double summedp_quality;
  double summedp_weights;
  unsigned int tot_recode_hits;
  double worst_ssim;

  ImageStat ssimg;
  ImageStat fastssim;
  ImageStat psnrhvs;

  int b_calculate_ssimg;
  int b_calculate_blockiness;

  int b_calculate_consistency;

  double total_inconsistency;
  double worst_consistency;
  Ssimv *ssim_vars;
  Metrics metrics;
#endif
  int b_calculate_psnr;

  int droppable;

  int initial_width;
  int initial_height;
  int initial_mbs;  // Number of MBs in the full-size frame; to be used to
                    // normalize the firstpass stats. This will differ from the
                    // number of MBs in the current frame when the frame is
                    // scaled.

  int last_coded_width;
  int last_coded_height;

  int use_svc;

  SVC svc;

  // Store frame variance info in SOURCE_VAR_BASED_PARTITION search type.
  Diff *source_diff_var;
  // The threshold used in SOURCE_VAR_BASED_PARTITION search type.
  unsigned int source_var_thresh;
  int frames_till_next_var_check;

  int frame_flags;

  search_site_config ss_cfg;

  int mbmode_cost[INTRA_MODES];
  unsigned int inter_mode_cost[INTER_MODE_CONTEXTS][INTER_MODES];
  int intra_uv_mode_cost[FRAME_TYPES][INTRA_MODES][INTRA_MODES];
  int y_mode_costs[INTRA_MODES][INTRA_MODES][INTRA_MODES];
  int switchable_interp_costs[SWITCHABLE_FILTER_CONTEXTS][SWITCHABLE_FILTERS];
  int partition_cost[PARTITION_CONTEXTS][PARTITION_TYPES];
  // Indices are:  max_tx_size-1,  tx_size_ctx,    tx_size
  int tx_size_cost[TX_SIZES - 1][TX_SIZE_CONTEXTS][TX_SIZES];

#if CONFIG_VP9_TEMPORAL_DENOISING
  VP9_DENOISER denoiser;
#endif

  int resize_pending;
  RESIZE_STATE resize_state;
  int external_resize;
  int resize_scale_num;
  int resize_scale_den;
  int resize_avg_qp;
  int resize_buffer_underflow;
  int resize_count;

  int use_skin_detection;

  int target_level;

  NOISE_ESTIMATE noise_estimate;

  // Count on how many consecutive times a block uses small/zeromv for encoding.
  uint8_t *consec_zero_mv;

  // VAR_BASED_PARTITION thresholds
  // 0 - threshold_64x64; 1 - threshold_32x32;
  // 2 - threshold_16x16; 3 - vbp_threshold_8x8;
  int64_t vbp_thresholds[4];
  int64_t vbp_threshold_minmax;
  int64_t vbp_threshold_sad;
  // Threshold used for partition copy
  int64_t vbp_threshold_copy;
  BLOCK_SIZE vbp_bsize_min;

  // Multi-threading
  int num_workers;
  VPxWorker *workers;
  struct EncWorkerData *tile_thr_data;
  VP9LfSync lf_row_sync;
  struct VP9BitstreamWorkerData *vp9_bitstream_worker_data;

  int keep_level_stats;
  Vp9LevelInfo level_info;
  MultiThreadHandle multi_thread_ctxt;
  void (*row_mt_sync_read_ptr)(VP9RowMTSync *const, int, int);
  void (*row_mt_sync_write_ptr)(VP9RowMTSync *const, int, int, const int);
  ARNRFilterData arnr_filter_data;

  int row_mt;
  unsigned int row_mt_bit_exact;

  // Previous Partition Info
  BLOCK_SIZE *prev_partition;
  int8_t *prev_segment_id;
  // Used to save the status of whether a block has a low variance in
  // choose_partitioning. 0 for 64x64, 1~2 for 64x32, 3~4 for 32x64, 5~8 for
  // 32x32, 9~24 for 16x16.
  // This is for the last frame and is copied to the current frame
  // when partition copy happens.
  uint8_t *prev_variance_low;
  uint8_t *copied_frame_cnt;
  uint8_t max_copied_frame;
  // If the last frame is dropped, we don't copy partition.
  uint8_t last_frame_dropped;

  // For each superblock: keeps track of the last time (in frame distance) the
  // the superblock did not have low source sad.
  uint8_t *content_state_sb_fd;

  int compute_source_sad_onepass;

  int compute_frame_low_motion_onepass;

  LevelConstraint level_constraint;

  uint8_t *count_arf_frame_usage;
  uint8_t *count_lastgolden_frame_usage;

  int multi_layer_arf;
  vpx_roi_map_t roi;

  LOOPFILTER_CONTROL loopfilter_ctrl;
#if CONFIG_RATE_CTRL
  ENCODE_COMMAND encode_command;
  PARTITION_INFO *partition_info;
  MOTION_VECTOR_INFO *motion_vector_info;
  MOTION_VECTOR_INFO *fp_motion_vector_info;
  TplDepStats *tpl_stats_info;

  RATE_QSTEP_MODEL rq_model[ENCODE_FRAME_TYPES];
#endif
  EXT_RATECTRL ext_ratectrl;

  int fixed_qp_onepass;

  // Flag to keep track of dynamic change in deadline mode
  // (good/best/realtime).
  MODE deadline_mode_previous_frame;

  // Flag to disable scene detection when rtc rate control library is used.
  int disable_scene_detection_rtc_ratectrl;

#if CONFIG_COLLECT_COMPONENT_TIMING
  /*!
   * component_time[] are initialized to zero while encoder starts.
   */
  uint64_t component_time[kTimingComponents];
  /*!
   * Stores timing for individual components between calls of start_timing()
   * and end_timing().
   */
  struct vpx_usec_timer component_timer[kTimingComponents];
  /*!
   * frame_component_time[] are initialized to zero at beginning of each frame.
   */
  uint64_t frame_component_time[kTimingComponents];
#endif
} VP9_COMP;
```

#### Field-by-field semantics

I group fields by owning subsystem because the struct is essentially a
shared bag of subsystem state.

**Frame geometry / common state (owner: `vp9/common`, used by all).**

- `frame_info` (`FRAME_INFO`, `vp9_onyxc_int.h:287-298`): cached
  width/height/mi_rows/mi_cols/MBs/bit_depth for the frame. Populated by
  `vp9_get_frame_info(oxcf)` in `vp9_create_compressor`
  (`vp9_encoder.c:2420`).
- `common` (`VP9_COMMON`, `vp9_onyxc_int.h:99-271`): the shared codec
  state — frame buffers, MI grid, entropy contexts, ref-frame map, tile
  log2, loop filter, frame contexts. Used by both encoder and decoder
  paths (the encoder owns and mutates it; the decoder has its own).
  Critical fields read by the frame loop: `cm->new_fb_idx`, `cm->frame_type`,
  `cm->show_frame`, `cm->show_existing_frame`, `cm->ref_frame_map[8]`,
  `cm->frame_refs[3]`, `cm->log2_tile_cols`, `cm->log2_tile_rows`,
  `cm->base_qindex`, `cm->cur_frame`, `cm->prev_frame`,
  `cm->current_video_frame`, `cm->current_frame_coding_index`,
  `cm->frame_context_idx`, `cm->refresh_frame_context`,
  `cm->reset_frame_context`, `cm->buffer_pool` (which holds
  `BufferPool::frame_bufs[FRAME_BUFFERS]`).

**Configuration mirror.**

- `oxcf` (`VP9EncoderConfig`, see §2.1): canonical config. Copied from the
  user-supplied `oxcf` at `init_config()` (`vp9_encoder.c:1483`) and again
  on every `vp9_change_config` (`vp9_encoder.c:2086`).

**Per-frame transient state (owner: encoder core, mostly **A2**/**A3**).**

- `quants` (`QUANTS`): forward quantization tables (specialist **A4**/**A5**;
  produced by `vp9_init_quantizer(cpi)` at `vp9_encoder.c:2735`).
- `y_dequant[QINDEX_RANGE][8]`, `uv_dequant[QINDEX_RANGE][8]`: 16-byte
  aligned dequant tables (`DECLARE_ALIGNED(16, ...)`). Specialist **A4**.
- `td` (`ThreadData`, `vp9_encoder.h:384-392`): main-thread macroblock
  state. Holds `MACROBLOCK mb`, `RD_COUNTS rd_counts`, `FRAME_COUNTS *counts`,
  `PICK_MODE_CONTEXT *leaf_tree`, `PC_TREE *pc_tree`, `PC_TREE *pc_root`.
  Worker threads each have their own `ThreadData` allocated in
  `create_enc_workers()` (`vp9_ethread.c:103-138`).
- `mbmi_ext_base`: extension array for `MB_MODE_INFO` records (motion-vector
  candidate cache); freed at teardown (`vp9_encoder.c:948-949`).

**Source / scaled / lookahead frame buffers.**

- `lookahead` (`struct lookahead_ctx *`): FIFO of input frames pushed by
  `vp9_receive_raw_frame` (`vp9_encoder.c:5977-5979`) and popped at the top
  of `vp9_get_compressed_data` (`vp9_encoder.c:6483-6487`). Definition in
  `vp9_lookahead.h:35-43`, capacity `MAX_LAG_BUFFERS = 25`
  (`vp9_lookahead.h:22`).
- `alt_ref_source`: pending lookahead entry pointed at by `arf_src_index`
  (`vp9_encoder.c:6435`). Only used in 2-pass / non-RT.
- `Source`: alias to the current source frame for this encode call
  (`vp9_encoder.c:6505-6507`).
- `Last_Source`: previous source frame (NULL on first frame and alt-ref).
- `un_scaled_source`: source pointer before optional downscaling.
- `scaled_source`, `scaled_last_source`: outputs of `vp9_scale_if_required`
  (`vp9_encoder.c:4013-4016, 4063-4065`). Hold SVC / dynamic-resize scaled
  copies.
- `unscaled_last_source`: previous source pointer.
- `raw_source_frame`: pre-spatial-denoise source used for PSNR calculation
  when KF denoise is on (`vp9_encoder.c:4025-4037`).

**TPL (Temporal Propagation Model) — 2-pass only.**

- `tpl_bsize`, `tpl_stats[MAX_ARF_GOP_SIZE]`, `tpl_gop_stats`,
  `tpl_recon_frames[REF_FRAMES]`, `enc_frame_buf[REF_FRAMES]`,
  `kmeans_*`. `MAX_ARF_GOP_SIZE = 2 * MAX_LAG_BUFFERS = 50`
  (`vp9_encoder.h:515`). Real-time path skips this entirely
  (`vp9_encoder.c:6617-6623`: only initialized when 2-pass and gf_group[1] is
  an ARF). Specialist **A5** for TPL semantics.

**Tile / partition state.**

- `tile_data` (`TileDataEnc *`), `allocated_tiles`: per-tile encode-time
  state, including `tile_info`, `thresh_freq_fact`, `mode_map`,
  `VP9RowMTSync row_mt_sync`. Allocated in `vp9_init_tile_data()` (called
  from inside the `vp9_encode_tiles_*` family). Definition in
  `vp9_encoder.h:331-346`.

**Reference-frame management.** This is the load-bearing piece of the
frame-dependency graph. Detailed in §3.5.

- `scaled_ref_idx[REFS_PER_FRAME]` (`REFS_PER_FRAME = 3`,
  `vp9_onyxc_int.h:35`): scaled-buffer-pool slot index used when the
  reference must be downscaled to match current frame size. Filled by
  `vp9_scale_references()` (`vp9_encoder.c:3454`).
- `lst_fb_idx`, `gld_fb_idx`, `alt_fb_idx`: indices **into**
  `cm->ref_frame_map[8]` — i.e. these tell you which of the 8 reference
  slots is currently "last", "golden", and "alt-ref" for this frame.
  Initialized in `init_buffer_indices()` to 0, 1, 2 (`vp9_encoder.c:1454-1463`).
- `ref_fb_idx[REF_FRAMES]` (`REF_FRAMES = 8`, `vp9_onyxc_int.h:38`):
  per-slot bookkeeping (SVC).
- `refresh_last_frame`, `refresh_golden_frame`, `refresh_alt_ref_frame`:
  three flags consumed by `update_ref_frames()` (`vp9_encoder.c:3292`) to
  decide which slots are overwritten by the just-encoded frame.
- `ext_refresh_*` family: user-provided overrides via
  `vp9_update_reference` (`vp9_encoder.c:2969-2974`). When
  `ext_refresh_frame_flags_pending` is set, the encoder uses these flags
  instead of its internal decision.
- `ext_refresh_frame_context_pending`, `ext_refresh_frame_context`:
  user override of `cm->refresh_frame_context`.

**Token buffers.**

- `tile_tok[4][1 << 6]` (i.e. `[4][64]`), `tplist[4][1 << 6]`: dimensioned
  for `MAX_NUM_TILE_ROWS = 4` and `MAX_NUM_TILE_COLS = 64`. They store
  per-tile token lists produced by the token-encoder pass. Specialist
  **A4** for the token pipeline. Allocation in
  `alloc_compressor_data()` (`vp9_encoder.c:1399-1411`).

**Rate-distortion / cost tables (specialist **A4**/**A2**).**

- `rd_ctrl` (`RD_CONTROL`), `rd` (`RD_OPT`): RD parameters and
  per-frame-type filter/prediction-mode thresholds. Initialized in
  `vp9_init_rd_parameters(cpi)` (`vp9_encoder.c:2423`).
- `coding_context` (`CODING_CONTEXT`, `vp9_encoder.h:69-84`): snapshot of
  costs / segmentation / frame context that `save_coding_context()` and
  `restore_coding_context()` flip around recode attempts and post-encode
  drops (`vp9_encoder.c:5727, 5798`).
- `nmvcosts[2]`, `nmvcosts_hp[2]`, `nmvsadcosts[2]`, `nmvsadcosts_hp[2]`:
  per-component MV cost LUTs (size `MV_VALS`). Allocated as
  `cpi->nmvcosts[i] = vpx_calloc(MV_VALS, ...)` and indexed at
  `&nmvcosts[i][MV_MAX]` to allow negative MV components
  (`vp9_encoder.c:2442-2533`).

**Timestamps.**

- `last_time_stamp_seen`, `last_end_time_stamp_seen`,
  `first_time_stamp_ever`: PTS bookkeeping (10 MHz tick units;
  `TICKS_PER_SEC = 10000000`, `vp9_encoder.h:67`).

**Rate control (specialist **A5**).**

- `rc` (`RATE_CONTROL`, `vp9_ratectrl.h:88-218`): the master RC state —
  buffer level, target rates, q-index history, scene-change flags,
  frames-to-key, GOP interval state.
- `framerate`: current target framerate (updated by `vp9_new_framerate`,
  `vp9_encoder.c:1416-1419`).

**Cost tables filled at frame-init time.**

- `mbmode_cost`, `inter_mode_cost`, `intra_uv_mode_cost`, `y_mode_costs`,
  `switchable_interp_costs`, `partition_cost`, `tx_size_cost`: per-frame
  mode-cost LUTs filled by `vp9_initialize_rd_consts()` (callers in **A2**).

**Speed features (specialist **A6**).**

- `sf` (`SPEED_FEATURES`, `vp9_speed_features.h:271-661`): big bag of
  algorithm-tuning flags driven by `cpu-used`. Programmed by
  `vp9_set_speed_features_framesize_independent()` and
  `vp9_set_speed_features_framesize_dependent()` (`vp9_encoder.c:2637-2638`
  for create, `:3750/3761` for change-config). For real-time the relevant
  call is the `set_rt_speed_feature_*` family (specialist **A6**).
- `find_fractional_mv_step`, `me_sf`, `diamond_search_sad`, `ss_cfg`:
  function-pointer / config knobs the motion-estimator (`vp9_mcomp.c`,
  specialist **A3**) uses. `fn_ptr[BLOCK_SIZES]`
  (`vp9_variance_fn_ptr_t fn_ptr[BLOCK_SIZES]`) holds SAD / variance /
  subpel-variance / SAD-skip / SAD-x4 function pointers per block size.
  Populated via the giant `BFP(...)` table at `vp9_encoder.c:2664-2724`.

**Two-pass (out of scope for real-time).**

- `twopass` (`TWO_PASS`): first-pass / second-pass state. RT path leaves it
  zero-valued.

**Altref / denoiser / AQ.**

- `alt_ref_buffer` (`YV12_BUFFER_CONFIG`): output of `vp9_temporal_filter`
  for altref creation (2-pass).
- `alt_ref_aq`: opaque pointer used in `LOOKAHEAD_AQ` mode.
- `cyclic_refresh` (`CYCLIC_REFRESH *`): state for `CYCLIC_REFRESH_AQ`
  (specialist **A5**).
- `active_map` (`ActiveMap`, `vp9_encoder.h:396-400`): user-provided
  block-level mask of "encode" vs "skip"; threaded via
  `vp9_set_active_map`.
- `denoiser` (`VP9_DENOISER`, only with `CONFIG_VP9_TEMPORAL_DENOISING`):
  the temporal denoiser state (specialist **A5**).
- `noise_estimate` (`NOISE_ESTIMATE`): block-level noise variance estimator
  feeding the denoiser (specialist **A5**).

**SVC.**

- `use_svc`: 0 = single-layer; 1 = SVC active. Default 0 set in
  `vp9_create_compressor` (`vp9_encoder.c:2408`). Switched on by
  `vp9_set_svc` (`vp9_encoder.c:`); checked by `is_one_pass_svc`
  (`vp9_encoder.h:1411-1413`).
- `svc` (`SVC`, `vp9_svc_layercontext.h:80-208`): layer counts, current
  spatial / temporal layer IDs, scaled frames, per-layer
  `LAYER_CONTEXT layer_context[VPX_MAX_LAYERS]`, drop flags, reference
  config, etc. Specialist **A6** for layer-driven dispatch
  (`vp9_one_pass_svc_start_layer`, `vp9_svc_update_ref_frame`, etc.).

**Source-variance partitioning (RT).**

- `source_diff_var` (`Diff *`), `source_var_thresh`,
  `frames_till_next_var_check`: state for the variance-based partition
  search path that real-time relies on (specialist **A2**).

**Per-superblock partition cache (RT).**

- `prev_partition`, `prev_segment_id`, `prev_variance_low`,
  `copied_frame_cnt`, `max_copied_frame`, `last_frame_dropped`: per-SB
  caches the variance-based partition path uses to copy partitions from
  previous frame (specialist **A2**).
- `content_state_sb_fd`: per-superblock content-state staleness counter.
- `compute_source_sad_onepass`: 1 when scene-detection / source-SAD is
  available for this frame (set in `encode_without_recode_loop` based on
  whether SVC, resize, or scaling has invalidated it,
  `vp9_encoder.c:3984-4070`).
- `compute_frame_low_motion_onepass`: drives `vp9_compute_frame_low_motion`
  (specialist **A5**).

**Resizing.**

- `resize_pending`, `resize_state` (`RESIZE_STATE`), `external_resize`,
  `resize_scale_num/_den`, `resize_avg_qp`, `resize_buffer_underflow`,
  `resize_count`: dynamic resize state (`RESIZE_DYNAMIC`).

**Levels.**

- `target_level`, `keep_level_stats`, `level_info` (`Vp9LevelInfo`),
  `level_constraint`: VP9 level tracking. The level definitions table
  `vp9_level_defs[VP9_LEVELS]` is declared at `vp9_encoder.h:450` and
  defined in `vp9_encoder.c` (consumed by **A5**).

**Multi-threading state (overview here; details **A6**).**

- `num_workers`: number of `VPxWorker` slots currently allocated.
- `workers` (`VPxWorker *`): array of length `num_workers`. The portable
  worker abstraction (`vpx_util/vpx_thread.h`) wraps pthreads or
  Win32 threads.
- `tile_thr_data` (`EncWorkerData *`): per-worker `EncWorkerData`
  (`vp9_ethread.h:25-31`), holding pointer back to `cpi`, pointer to that
  worker's `ThreadData`, start tile index, thread id, and per-tile
  completion flags.
- `lf_row_sync` (`VP9LfSync`): the loop-filter row synchronization
  primitives shared with the decoder (in `vp9_thread_common.h`).
- `vp9_bitstream_worker_data`: opaque worker data for the parallel
  bitstream pack path.
- `multi_thread_ctxt` (`MultiThreadHandle`, `vp9_encoder.h:361-376`): the
  row-mt job queue handles and per-tile-col `RowMTInfo`.
- `row_mt_sync_read_ptr` / `row_mt_sync_write_ptr`: function-pointer
  switches that flip between real sync (`vp9_row_mt_sync_read/write`) and
  dummy sync (`vp9_row_mt_sync_read/write_dummy`) depending on whether
  row-mt is enabled (`vp9_encodeframe.c:6181-6194`).
- `arnr_filter_data` (`ARNRFilterData`): scratch for the 2-pass temporal
  filter (not used in RT).
- `row_mt`, `row_mt_bit_exact`: row-mt enable flags.

**Stats / misc.**

- `count_arf_frame_usage`, `count_lastgolden_frame_usage`,
  `multi_layer_arf`, `roi` (`vpx_roi_map_t`), `loopfilter_ctrl`
  (`LOOPFILTER_CONTROL`, `vp9_encoder.h:147-151`), `ext_ratectrl`
  (`EXT_RATECTRL`), `fixed_qp_onepass`,
  `deadline_mode_previous_frame`, `disable_scene_detection_rtc_ratectrl`.

**Conditional / testing-only.**

- `encode_command`, `partition_info`, `motion_vector_info`,
  `fp_motion_vector_info`, `tpl_stats_info`, `rq_model[ENCODE_FRAME_TYPES]`
  (only `CONFIG_RATE_CTRL`).
- `component_time`, `component_timer`, `frame_component_time` (only
  `CONFIG_COLLECT_COMPONENT_TIMING`).

#### Ownership / lifetime / aliasing notes for `VP9_COMP`

- **Single `VP9_COMP` per encoder instance.** Allocated by
  `vp9_create_compressor` (`vp9_encoder.c:2380`) and freed by
  `vp9_remove_compressor` (`vp9_encoder.c:2772`). The pointer is held by
  the public `vpx_codec_alg_priv_t` (in `vp9_cx_iface.c`).
- **`oxcf` is the only allowed source of truth for config.** Subsystems
  must not stash copies of configuration values that may change across
  `vp9_change_config`.
- **`cpi->common` is also single-owner from the encoder's perspective.**
  The encoder writes to `common` freely. The decoder has its own
  `VP9_COMMON`. The shared types in `vp9_onyxc_int.h` are reused by
  copy-of-struct (entropy contexts, partition contexts, loop filter,
  segmentation).
- **`cpi->td` is the "tile-zero / main-thread" `ThreadData`.** When
  multi-threaded, worker threads `0 .. num_workers-2` each have their own
  heap-allocated `ThreadData` (`vp9_ethread.c:113-126`); worker
  `num_workers-1` (the main thread) uses `&cpi->td` directly
  (`vp9_ethread.c:131-135`). Worker `ThreadData` is copied **from**
  `cpi->td.mb` and `cpi->td.rd_counts` at the start of each frame
  (`vp9_ethread.c:209-211`), and `td->counts` is copied **back** at the end
  via `vp9_accumulate_frame_counts` (`vp9_ethread.c:244`). This is the
  primary multi-thread aliasing invariant: per-frame, each worker reads
  cpi-level config and writes only into its own `ThreadData`; cross-thread
  state is accumulated at frame teardown.
- **Reference frames live in `cpi->common.buffer_pool->frame_bufs[]`,
  not in `cpi` directly.** `cpi->scaled_ref_idx[]`, `cpi->lst_fb_idx`,
  `cpi->gld_fb_idx`, `cpi->alt_fb_idx`, `cm->ref_frame_map[8]`,
  `cm->new_fb_idx` are all indices into that pool. The pool is
  reference-counted: `ref_cnt_fb` increments and `--ref_count`
  decrements via `get_free_fb` (`vp9_onyxc_int.h:332-339` and several call
  sites in `vp9_encoder.c`). `vp9_get_compressed_data` releases the
  reference on the **previous** `new_fb_idx` and acquires a free one each
  call (`vp9_encoder.c:6549-6554`).
- **`cpi->lookahead` is mutated under the encoder's own discretion.** It
  is push-only (`vp9_lookahead_push`) and pop-only
  (`vp9_lookahead_pop` / `vp9_lookahead_peek`). Push happens from the
  caller's thread via `vp9_receive_raw_frame`
  (`vp9_encoder.c:5977-5979`); pop happens from inside
  `vp9_get_compressed_data` (`vp9_encoder.c:6483-6487`).
- **`setjmp(cm->error.jmp)` is used as a panic mechanism.** Both
  `vp9_create_compressor` (`vp9_encoder.c:2390-2394`) and
  `encoder_encode` (`vp9_cx_iface.c:1381-1386`) wrap their interior in a
  `setjmp` so any deep call site can `longjmp` out via
  `vpx_internal_error`. Real-time **never** longjmps under normal
  operation; this is for OOM / pathological inputs.
- **Worker-side allocations are short-lived to the worker structure.**
  `vp9_encode_free_mt_data` (`vp9_ethread.c:173-192`) frees per-worker
  `td->counts`, the `PC_TREE`, and the `ThreadData` itself.

### 2.5 `ThreadData`, `TileDataEnc`, `RowMTInfo`, `MultiThreadHandle`, `EncWorkerData`

`third_party/libvpx/vp9/encoder/vp9_encoder.h:331-376` and
`vp9_ethread.h:25-43`:

```c
typedef struct TileDataEnc {
  TileInfo tile_info;
  int thresh_freq_fact[BLOCK_SIZES][MAX_MODES];
  int thresh_freq_fact_prev[BLOCK_SIZES][MAX_MODES];
  int8_t mode_map[BLOCK_SIZES][MAX_MODES];
  FIRSTPASS_DATA fp_data;
  VP9RowMTSync row_mt_sync;

  // Used for adaptive_rd_thresh with row multithreading
  int *row_base_thresh_freq_fact;
  // The value of sb_rows when row_base_thresh_freq_fact is allocated.
  // The row_base_thresh_freq_fact array has sb_rows * BLOCK_SIZES * MAX_MODES
  // elements.
  int sb_rows;
  MV firstpass_top_mv;
} TileDataEnc;

typedef struct RowMTInfo {
  JobQueueHandle job_queue_hdl;
#if CONFIG_MULTITHREAD
  pthread_mutex_t job_mutex;
#endif
} RowMTInfo;

typedef struct MultiThreadHandle {
  int allocated_tile_rows;
  int allocated_tile_cols;
  int allocated_vert_unit_rows;

  // Frame level params
  int num_tile_vert_sbs[MAX_NUM_TILE_ROWS];

  // Job Queue structure and handles
  JobQueue *job_queue;

  int jobs_per_tile_col;

  RowMTInfo row_mt_info[MAX_NUM_TILE_COLS];
  int thread_id_to_tile_id[MAX_NUM_THREADS];  // Mapping of threads to tiles
} MultiThreadHandle;

typedef struct ThreadData {
  MACROBLOCK mb;
  RD_COUNTS rd_counts;
  FRAME_COUNTS *counts;

  PICK_MODE_CONTEXT *leaf_tree;
  PC_TREE *pc_tree;
  PC_TREE *pc_root;
} ThreadData;
```

`third_party/libvpx/vp9/encoder/vp9_ethread.h:18-43`:

```c
#define MAX_NUM_TILE_COLS (1 << 6)
#define MAX_NUM_TILE_ROWS 4
#define MAX_NUM_THREADS 80

struct VP9_COMP;
struct ThreadData;

typedef struct EncWorkerData {
  struct VP9_COMP *cpi;
  struct ThreadData *td;
  int start;
  int thread_id;
  int tile_completion_status[MAX_NUM_TILE_COLS];
} EncWorkerData;

// Encoder row synchronization
typedef struct VP9RowMTSyncData {
#if CONFIG_MULTITHREAD
  pthread_mutex_t *mutex;
  pthread_cond_t *cond;
#endif
  // Allocate memory to store the sb/mb block index in each row.
  int *cur_col;
  int sync_range;
  int rows;
} VP9RowMTSync;
```

Semantics (specialist **A6** owns the runtime details):

- `MAX_NUM_THREADS = 80`, `MAX_NUM_TILE_COLS = 64`, `MAX_NUM_TILE_ROWS = 4`
  are hard caps.
- One `EncWorkerData` per worker. `td` aliases either `&cpi->td` (main
  thread) or a heap-allocated `ThreadData` (workers `0..num_workers-2`).
- `VP9RowMTSync::cur_col[r]` is the column index up to which row `r` has
  finished encoding superblocks; row `r+1`'s SB at column `c` can proceed
  once `cur_col[r] >= c - sync_range + 1`. `sync_range` is 1 by default
  (`vp9_ethread.c:317-318`). The sync algorithm itself is in
  `vp9_row_mt_sync_read/write` (`vp9_ethread.c:348-404`); specialist **A6**.

### 2.6 `CODING_CONTEXT`

`third_party/libvpx/vp9/encoder/vp9_encoder.h:69-84`:

```c
typedef struct {
  int nmvjointcost[MV_JOINTS];
  int nmvcosts[2][MV_VALS];
  int nmvcosts_hp[2][MV_VALS];

  vpx_prob segment_pred_probs[PREDICTION_PROBS];

  unsigned char *last_frame_seg_map_copy;

  // 0 = Intra, Last, GF, ARF
  signed char last_ref_lf_deltas[MAX_REF_LF_DELTAS];
  // 0 = ZERO_MV, MV
  signed char last_mode_lf_deltas[MAX_MODE_LF_DELTAS];

  FRAME_CONTEXT fc;
} CODING_CONTEXT;
```

`save_coding_context()` (`vp9_encoder.c:5727`) snapshots and
`restore_coding_context()` restores this state. Two real-time paths use it:

1. **Post-encode drop.** `encode_frame_to_data_rate` saves before
   `vp9_pack_bitstream` if `rc.use_post_encode_drop`; if
   `post_encode_drop_cbr` chooses to drop, the context is restored
   (`vp9_encoder.c:5796-5800`).
2. **Recode loop.** Non-RT only.

### 2.7 `FRAME_INFO` (cited from common header)

`third_party/libvpx/vp9/common/vp9_onyxc_int.h:287-298`:

```c
typedef struct {
  int frame_width;
  int frame_height;
  int render_frame_width;
  int render_frame_height;
  int mi_rows;
  int mi_cols;
  int mb_rows;
  int mb_cols;
  int num_mbs;
  vpx_bit_depth_t bit_depth;
} FRAME_INFO;
```

### 2.8 `VP9_COMMON` (cited from common header)

`third_party/libvpx/vp9/common/vp9_onyxc_int.h:99-271` — full struct is too
long to re-paste here; verbatim, with the load-bearing fields for the
encoder frame loop:

```c
typedef struct VP9Common {
  struct vpx_internal_error_info error;
  vpx_color_space_t color_space;
  vpx_color_range_t color_range;
  int width;
  int height;
  int render_width;
  int render_height;
  int last_width;
  int last_height;

  int subsampling_x;
  int subsampling_y;

#if CONFIG_VP9_HIGHBITDEPTH
  int use_highbitdepth;
#endif

  YV12_BUFFER_CONFIG *frame_to_show;
  RefCntBuffer *prev_frame;
  RefCntBuffer *cur_frame;

  int ref_frame_map[REF_FRAMES]; /* maps fb_idx to reference slot */

  int next_ref_frame_map[REF_FRAMES];

  RefBuffer frame_refs[REFS_PER_FRAME];

  int new_fb_idx;

  int cur_show_frame_fb_idx;

  FRAME_TYPE last_frame_type;
  FRAME_TYPE frame_type;

  int show_frame;
  int last_show_frame;
  int show_existing_frame;

  uint8_t intra_only;
  uint8_t last_intra_only;

  int allow_high_precision_mv;

  int reset_frame_context;

  int MBs;
  int mb_rows, mi_rows;
  int mb_cols, mi_cols;
  int mi_stride;

  TX_MODE tx_mode;

  int base_qindex;
  int y_dc_delta_q;
  int uv_dc_delta_q;
  int uv_ac_delta_q;
  int16_t y_dequant[MAX_SEGMENTS][2];
  int16_t uv_dequant[MAX_SEGMENTS][2];

  int mi_alloc_size;
  MODE_INFO *mip;
  MODE_INFO *mi;
  MODE_INFO *prev_mip;
  MODE_INFO *prev_mi;

  int (*alloc_mi)(struct VP9Common *cm, int mi_size);
  void (*free_mi)(struct VP9Common *cm);
  void (*setup_mi)(struct VP9Common *cm);

  MODE_INFO **mi_grid_base;
  MODE_INFO **mi_grid_visible;
  MODE_INFO **prev_mi_grid_base;
  MODE_INFO **prev_mi_grid_visible;

  int use_prev_frame_mvs;

  int seg_map_idx;
  int prev_seg_map_idx;

  uint8_t *seg_map_array[NUM_PING_PONG_BUFFERS];
  uint8_t *last_frame_seg_map;
  uint8_t *current_frame_seg_map;
  int seg_map_alloc_size;

  INTERP_FILTER interp_filter;

  loop_filter_info_n lf_info;

  int refresh_frame_context;

  int ref_frame_sign_bias[MAX_REF_FRAMES];

  struct loopfilter lf;
  struct segmentation seg;

  MV_REFERENCE_FRAME comp_fixed_ref;
  MV_REFERENCE_FRAME comp_var_ref[2];
  REFERENCE_MODE reference_mode;

  FRAME_CONTEXT *fc;
  FRAME_CONTEXT *frame_contexts;
  unsigned int frame_context_idx;
  FRAME_COUNTS counts;

  unsigned int current_video_frame;
  int current_frame_coding_index;
  BITSTREAM_PROFILE profile;

  vpx_bit_depth_t bit_depth;
  vpx_bit_depth_t dequant_bit_depth;

  int error_resilient_mode;
  int frame_parallel_decoding_mode;

  int log2_tile_cols, log2_tile_rows;
  int byte_alignment;
  int skip_loop_filter;

  BufferPool *buffer_pool;

  PARTITION_CONTEXT *above_seg_context;
  ENTROPY_CONTEXT *above_context;
  int above_context_alloc_cols;

  int lf_row;
} VP9_COMMON;
```

Important constants for the reference pool:

`third_party/libvpx/vp9/common/vp9_onyxc_int.h:35-42`:

```c
#define REFS_PER_FRAME 3

#define REF_FRAMES_LOG2 3
#define REF_FRAMES (1 << REF_FRAMES_LOG2)

// 1 scratch frame for the new frame, REFS_PER_FRAME for scaled references on
// the encoder.
#define FRAME_BUFFERS (REF_FRAMES + 1 + REFS_PER_FRAME)
```

So the encoder side allocates **12 frame buffers** total in
`BufferPool::frame_bufs[FRAME_BUFFERS]`: 8 reference slots, 1 for the
current new frame, and 3 for scaled references.

`RefCntBuffer` is in `vp9_onyxc_int.h:66-84`:

```c
typedef struct {
  int ref_count;
  MV_REF *mvs;
  int mi_rows;
  int mi_cols;
  uint8_t released;

  int frame_index;         // Display order in the video
  int frame_coding_index;  // The coding order (starting from zero)
  vpx_codec_frame_buffer_t raw_frame_buffer;
  YV12_BUFFER_CONFIG buf;
} RefCntBuffer;

typedef struct BufferPool {
  void *cb_priv;

  vpx_get_frame_buffer_cb_fn_t get_fb_cb;
  vpx_release_frame_buffer_cb_fn_t release_fb_cb;

  RefCntBuffer frame_bufs[FRAME_BUFFERS];

  InternalFrameBufferList int_frame_buffers;
} BufferPool;
```

### 2.9 `lookahead_ctx` / `lookahead_entry`

`third_party/libvpx/vp9/encoder/vp9_lookahead.h:22-43`:

```c
#define MAX_LAG_BUFFERS 25

struct lookahead_entry {
  YV12_BUFFER_CONFIG img;
  int64_t ts_start;
  int64_t ts_end;
  int show_idx; /*The show_idx of this frame*/
  vpx_enc_frame_flags_t flags;
};

// The max of past frames we want to keep in the queue.
#define MAX_PRE_FRAMES 1

struct lookahead_ctx {
  int max_sz;        /* Absolute size of the queue */
  int sz;            /* Number of buffers currently in the queue */
  int read_idx;      /* Read index */
  int write_idx;     /* Write index */
  int next_show_idx; /* The show_idx that will be assigned to the next frame
                        being pushed in the queue*/
  struct lookahead_entry *buf; /* Buffer list */
};
```

Push/pop/peek API: `vp9_lookahead_push/pop/peek/full/depth/next_show_idx`
(`vp9_lookahead.h:50-121`). Real-time with `lag_in_frames = 0` keeps the
queue at depth 1; WebRTC sometimes runs with depth 0 (immediate drain).

### 2.10 `RATE_CONTROL` (cited; owned by **A5**)

`third_party/libvpx/vp9/encoder/vp9_ratectrl.h:88-218` — long. The fields
the architecture path touches at minimum: `this_frame_target`,
`projected_frame_size`, `buffer_level`, `frames_to_key`, `frames_since_key`,
`source_alt_ref_pending`, `source_alt_ref_active`, `is_src_frame_alt_ref`,
`frames_till_gf_update_due`, `high_source_sad`,
`re_encode_maxq_scene_change`, `use_post_encode_drop`,
`last_post_encode_dropped_scene_change`, `next_key_frame_forced`,
`worst_quality`, `best_quality`, `show_arf_as_gld`. Specialist **A5** owns
the semantics in detail.

### 2.11 `SVC` (cited; owned by **A6**)

`third_party/libvpx/vp9/encoder/vp9_svc_layercontext.h:80-208` — see §2.4
above for the verbatim. Fields the architecture path reads are
`spatial_layer_id`, `temporal_layer_id`, `number_spatial_layers`,
`number_temporal_layers`, `temporal_layering_mode`, `lst_fb_idx[]`,
`gld_fb_idx[]`, `alt_fb_idx[]`, `force_zero_mode_spatial_ref`,
`use_set_ref_frame_config`, `first_spatial_layer_to_encode`,
`skip_enhancement_layer`, `drop_spatial_layer[]`, `last_layer_dropped[]`,
`framedrop_thresh[]`. The flag-and-FB-idx setters are in
`vp9_one_pass_svc_start_layer` (`vp9_svc_layercontext.c:769`).

---

## 3. Control flow

### 3.1 Two-layer architecture: public iface and internal compressor

```
  +--------------------------------------------------------------+
  |                User: vpx_codec_encode(...)                   |
  +--------------------------------------------------------------+
                            v
  +--------------------------------------------------------------+
  | vp9/vp9_cx_iface.c    Public encoder interface               |
  |                                                              |
  |   vpx_codec_vp9_cx_algo (CODEC_INTERFACE)                    |
  |     ├─ encoder_init   ─► vp9_create_compressor               |
  |     ├─ encoder_encode ─► vp9_receive_raw_frame +             |
  |     │                    vp9_get_compressed_data (looped)    |
  |     ├─ encoder_destroy─► vp9_remove_compressor               |
  |     ├─ encoder_set_config                                    |
  |     └─ encoder_ctrl_maps[]   (per-knob setters)              |
  +--------------------------------------------------------------+
                            v
  +--------------------------------------------------------------+
  | vp9/encoder/vp9_encoder.{h,c}   Internal compressor          |
  |                                                              |
  |   VP9_COMP                                                   |
  |     ├─ Lookahead queue (vp9_lookahead.c)                     |
  |     ├─ Rate control state         (A5)                       |
  |     ├─ Speed features             (A6)                       |
  |     ├─ SVC layer pipeline         (A6)                       |
  |     ├─ Tile / row-mt threading    (A6)                       |
  |     └─ Per-frame encode pipeline:                            |
  |          encode_frame_to_data_rate                           |
  |            └─ encode_without_recode_loop  (real-time)        |
  |                 └─ vp9_encode_frame (in vp9_encodeframe.c)   |
  +--------------------------------------------------------------+
```

The public interface registration is at
`third_party/libvpx/vp9/vp9_cx_iface.c:2205-2234`:

```c
CODEC_INTERFACE(vpx_codec_vp9_cx) = {
  "WebM Project VP9 Encoder" VERSION_STRING,
  VPX_CODEC_INTERNAL_ABI_VERSION,
#if CONFIG_VP9_HIGHBITDEPTH
  VPX_CODEC_CAP_HIGHBITDEPTH |
#endif
      VPX_CODEC_CAP_ENCODER | VPX_CODEC_CAP_PSNR,
  encoder_init,                                    // vpx_codec_init_fn_t
  encoder_destroy,                                 // vpx_codec_destroy_fn_t
  encoder_ctrl_maps,                               // vpx_codec_ctrl_fn_map_t
  { NULL, NULL, NULL, NULL, NULL },                // decoder slots unused
  {
      1,
      encoder_usage_cfg_map,
      encoder_encode,
      encoder_get_cxdata,
      encoder_set_config,
      encoder_get_global_headers,
      encoder_get_preview,
      NULL
  }
};
```

### 3.2 Process-level one-time init

`vp9_initialize_enc()` is called once per process. It is invoked from
`encoder_init` at `vp9_cx_iface.c:1136`:

`third_party/libvpx/vp9/encoder/vp9_encoder.c:929-942`:

```c
static void initialize_enc(void) {
  vp9_rtcd();
  vpx_dsp_rtcd();
  vpx_scale_rtcd();
  vp9_init_intra_predictors();
  vp9_init_me_luts();
  vp9_rc_init_minq_luts();
  vp9_entropy_mv_init();
#if !CONFIG_REALTIME_ONLY
  vp9_temporal_filter_init();
#endif
}

void vp9_initialize_enc(void) { once(initialize_enc); }
```

What this does:

- `vp9_rtcd`, `vpx_dsp_rtcd`, `vpx_scale_rtcd` are the three runtime
  function-pointer dispatch tables that bind kernel pointers to the best
  available SIMD implementation for the current CPU. They are generated by
  the `rtcd.pl` script (`third_party/libvpx/build/make/rtcd.pl`); see §6.
  These calls populate dispatch tables in global static storage. They use
  `once()` internally as well so it is safe to call them more than once,
  but `vp9_initialize_enc` already wraps the whole block in `once()`.
- `vp9_init_intra_predictors`: populate intra-prediction dispatch table
  (**A4**).
- `vp9_init_me_luts`: populate MV-cost log lookup tables (**A3**).
- `vp9_rc_init_minq_luts`: populate rate-control min-q lookup tables
  (**A5**).
- `vp9_entropy_mv_init`: populate motion-vector entropy default tables
  (**A4**).
- `vp9_temporal_filter_init`: compiled out for `CONFIG_REALTIME_ONLY`.

This function is process-global; per-instance state is **not** touched.

### 3.3 Encoder lifecycle: init → encode loop → teardown

```
  encoder_init                                    cx_iface.c:1114
   ├─ vpx_calloc(vpx_codec_alg_priv_t)            :1120
   ├─ alloc BufferPool                            :1126
   ├─ priv->cfg = *ctx->config.enc                :1131
   ├─ vp9_initialize_enc()                        :1136
   ├─ validate_config(priv, &priv->cfg, ...)      :1138
   ├─ set_encoder_config(&priv->oxcf, &priv->cfg, &extra_cfg)
   │                                              :1147
   ├─ priv->cpi = vp9_create_compressor(&priv->oxcf,
   │                                    priv->buffer_pool)
   │                                              :1152
   └─ set_twopass_params_from_config              :1154
```

`vp9_create_compressor` (`vp9_encoder.c:2380-2762`) does, in order:

1. `vpx_memalign(32, sizeof(VP9_COMP))` and `vp9_zero(*cpi)`
   (`:2383-2388`).
2. `setjmp(cm->error.jmp)` for OOM handling (`:2390-2394`).
3. Wire `cm->{alloc_mi, free_mi, setup_mi} = vp9_enc_{alloc_mi, free_mi,
   setup_mi}` (`:2397-2399`) — the encoder uses different MI allocation
   than the decoder.
4. Allocate `cm->fc` (1 `FRAME_CONTEXT`) and `cm->frame_contexts`
   (`FRAME_CONTEXTS = 4` `FRAME_CONTEXT`s) — the entropy backup ring
   (`:2401-2405`). See `FRAME_CONTEXTS_LOG2 = 2` at
   `vp9_onyxc_int.h:44-45`.
5. Defaults: `compute_frame_low_motion_onepass = 1`, `use_svc = 0`,
   `resize_state = ORIG`, `external_resize = 0`,
   `force_update_segmentation = 0` (`:2407-2417`).
6. `cm->buffer_pool = pool`; `init_ref_frame_bufs(cm)` (`:2414-2415`) —
   sets `cm->ref_frame_map[i] = -1` for all 8 slots.
7. `init_config(cpi, oxcf)` (`:2419`): copy `*oxcf` into `cpi->oxcf`,
   propagate width/height/profile/bit_depth to `cm`, set initial SVC
   layer counts, set `cm->byte_alignment`, set `cpi->framerate`. Also calls
   `vp9_init_layer_context(cpi)` for SVC (specialist **A6**).
8. `cpi->frame_info = vp9_get_frame_info(oxcf)` (`:2420`).
9. `vp9_rc_init(&cpi->oxcf, oxcf->pass, &cpi->rc)` (`:2422`): initialize
   `RATE_CONTROL` (specialist **A5**).
10. `vp9_init_rd_parameters(cpi)` (`:2423`): initialize `RD_OPT` /
    `RD_CONTROL`.
11. `init_frame_indexes(cm)` (`:2425`): `current_video_frame = 0`,
    `current_frame_coding_index = 0`.
12. `realloc_segmentation_maps(cpi)` (`:2428`): allocate `seg_map_array`,
    `last_frame_seg_map`, `current_frame_seg_map`, and the encoder-only
    `cpi->segmentation_map`.
13. Allocate `skin_map`, `consec_zero_mv` (mi_rows × mi_cols bytes each)
    (`:2430-2440`).
14. Allocate MV cost tables `nmvcosts[2]`, `nmvcosts_hp[2]`,
    `nmvsadcosts[2]`, `nmvsadcosts_hp[2]` each of `MV_VALS` ints
    (`:2442-2457`).
15. Allocate `mbgraph_stats[i].mb_stats` for `i in [0, MAX_LAG_BUFFERS)`
    (`:2459-2464`). (Used in non-RT.)
16. `cpi->refresh_alt_ref_frame = 0`,
    `b_calculate_psnr = CONFIG_INTERNAL_STATS` (`:2466-2467`).
17. `init_level_info` and `init_level_constraint` (`:2469-2470`).
18. Initialize MV cost tables via `cal_nmvjointsadcost`,
    `cal_nmvsadcosts`, `cal_nmvsadcosts_hp` (`:2522-2533`). Important: the
    pointers `cpi->td.mb.nmvcost[i] = &cpi->nmvcosts[i][MV_MAX]` offset the
    base by `MV_MAX` so that negative MV components are valid indices.
19. `cpi->allow_encode_breakout = ENCODE_BREAKOUT_ENABLED` (`:2557`).
20. External rate-control init `vp9_extrc_init(&cpi->ext_ratectrl)`
    (`:2560`).
21. **First-pass / second-pass init**: only if `!CONFIG_REALTIME_ONLY` and
    `oxcf->pass != 0` (`:2566-2631`). Real-time skips this entirely.
22. `vp9_set_speed_features_framesize_independent(cpi, oxcf->speed)` and
    `vp9_set_speed_features_framesize_dependent(cpi, oxcf->speed)`
    (`:2637-2638`). For real-time these set up the non-RD pick mode
    (specialist **A2**), variance partition path, RT speed features
    (specialist **A6**).
23. Allocate `mi_ssim_rdmult_scaling_factors` (`:2640-2649`).
24. Init TPL placeholders (real-time skips TPL stats allocation;
    `tpl_ready = 0`, all `tpl_stats[i].tpl_stats_ptr = NULL`)
    (`:2651-2657`).
25. Allocate `source_diff_var = vpx_calloc(cm->MBs, ...)` for the
    variance-based partition search (`:2660-2663`).
26. **Bind variance / SAD function-pointer table** `cpi->fn_ptr[BLOCK_SIZES]`
    via the giant `BFP(...)` macro (`:2664-2724`). Each block size gets:
    - `sdf`: SAD function.
    - `sdsf`: SAD-skip function.
    - `sdaf`: SAD-avg (compound prediction).
    - `vf`: variance.
    - `svf`: subpel variance.
    - `svaf`: subpel-avg variance.
    - `sdx4df`: 4-MV SAD.
    - `sdsx4df`: 4-MV SAD-skip.
    These are the kernels specialist **A3** uses for ME. Their backing SIMD
    pointers are populated by the `vpx_dsp_rtcd()` call from §3.2.
27. Optional `highbd_set_var_fns(cpi)` for 10/12-bit (`:2726-2728`).
28. `vp9_init_quantizer(cpi)` (`:2735`): initialize `cpi->quants` and the
    `y_dequant` / `uv_dequant` tables. Specialist **A4**.
29. `vp9_loop_filter_init(cm)` (`:2737`): populate loop-filter tables.
30. `vp9_setup_scale_factors_for_frame(&cpi->me_sf, ...)` (`:2741/2745`):
    unit scaling factors for the motion search.
31. `cm->error.setjmp = 0` (`:2750`).
32. `CONFIG_RATE_CTRL` path: init `encode_command` and optionally allocate
    `partition_info`, `motion_vector_info`, `fp_motion_vector_info`,
    `tpl_stats_info` (`:2752-2760`).
33. Return `cpi`.

Note that `vp9_create_compressor` does **not** create worker threads. That
is deferred to the first call to `vp9_encode_tiles_mt` (or its row-mt /
fp-mt cousins), which calls `create_enc_workers(cpi, num_workers)` at
`vp9_ethread.c:80`.

#### Encode loop call sequence

```
  encoder_encode(image, pts, duration, flags, deadline)   cx_iface.c:1312
   ├─ validate_img(img)                                   :1333
   ├─ pick_quickcompress_mode(duration, deadline)         :1371
   │      └─ if deadline==VPX_DL_REALTIME: oxcf.mode=REALTIME
   │              vp9_change_config(cpi, &oxcf)
   ├─ vp9_apply_encoding_flags(cpi, flags)                :1389
   ├─ handle fixed-keyframe interval                      :1391
   ├─ image2yuvconfig(img, &sd)                           :1416
   ├─ vp9_receive_raw_frame(cpi, flags, &sd, pts, end_pts):1420
   │     ─► vp9_lookahead_push (vp9_encoder.c:5977-5979)
   └─ while (cx_data_sz >= half &&
             vp9_get_compressed_data(...) != -1)          :1480
         pack frame packet into ctx->pkt_list / pending
```

### 3.4 The frame loop: `vp9_get_compressed_data` in detail

`third_party/libvpx/vp9/encoder/vp9_encoder.c:6362-6678`.

```
  vp9_get_compressed_data(cpi, *frame_flags, *size, dest, dest_size,
                          *time_stamp, *time_end, flush,
                          *encode_frame_result)
  │
  ├─ if (is_one_pass_svc(cpi))                                :6382
  │     vp9_one_pass_svc_start_layer(cpi);          // A6
  │
  ├─ vp9_set_high_precision_mv(cpi, ALTREF_HIGH_PRECISION_MV) :6388
  ├─ multi_layer_arf = (pass==2 && !use_svc && enable_auto_arf>=2)
  │                                                           :6395-6398
  ├─ cm->reset_frame_context = 0;                             :6401
  │  cm->refresh_frame_context = 1;
  │  if (!is_one_pass_svc(cpi)) {
  │     cpi->refresh_last_frame = 1;
  │     cpi->refresh_golden_frame = 0;
  │     cpi->refresh_alt_ref_frame = 0;
  │  }
  │
  ├─ arf_src_index = get_arf_src_index(cpi)                   :6410
  ├─ scan forward in lookahead for forced KF; clamp           :6412-6424
  ├─ if (gf_group_index == 1) {                               :6427
  │     stack_init(twopass.gf_group.arf_index_stack, ...);
  │     stack_size = 0;
  │  }
  │
  ├─ if (arf_src_index) {                                     :6432
  │     source = vp9_lookahead_peek(arf_src_index);
  │     if (source != NULL) {
  │        cpi->alt_ref_source = source;
  │        (non-RT: vp9_temporal_filter(arf_src_index);)
  │        cm->show_frame = 0; cm->intra_only = 0;
  │        cpi->refresh_alt_ref_frame = 1;
  │        cpi->refresh_golden_frame = 0;
  │        cpi->refresh_last_frame  = 0;
  │     }
  │  }
  │
  ├─ if (!source) {                                           :6476
  │     // Get last frame source as a peek(-1)
  │     if (current_video_frame > 0)
  │        last_source = vp9_lookahead_peek(-1);
  │     if (use_svc || set_intra_only_frame)
  │        source = vp9_svc_lookahead_pop(cpi, lookahead, flush);
  │     else
  │        source = vp9_lookahead_pop(lookahead, flush);
  │     if (source) {
  │        cm->show_frame = 1; cm->intra_only = 0;
  │        if (source->flags & VPX_EFLAG_FORCE_KF) ...        :6495
  │        check_src_altref(cpi, source);                     :6501
  │     }
  │  }
  │
  ├─ if (source) {                                            :6505
  │     cpi->un_scaled_source = cpi->Source = &source->img;
  │     cpi->unscaled_last_source = last_source ? &last_source->img
  │                                             : NULL;
  │     *time_stamp = source->ts_start;
  │     *time_end   = source->ts_end;
  │     *frame_flags = (source->flags & VPX_EFLAG_FORCE_KF) ?
  │                    FRAMEFLAGS_KEY : 0;
  │  } else {
  │     *size = 0;
  │     return -1;                                            // drain done
  │  }
  │
  ├─ adjust_frame_rate(cpi, source) or vp9_svc_adjust_frame_rate(cpi)
  │                                                           :6534-6539
  ├─ if (is_one_pass_svc) vp9_update_temporal_layer_framerate +
  │                       vp9_restore_layer_context           :6542-6545
  │
  ├─ // Acquire new buffer-pool slot for cm->new_fb_idx       :6549-6554
  │  if (cm->new_fb_idx != INVALID_IDX)
  │     --pool->frame_bufs[cm->new_fb_idx].ref_count;
  │  cm->new_fb_idx = get_free_fb(cm);
  │  cm->cur_frame  = &pool->frame_bufs[cm->new_fb_idx];
  │
  ├─ *size = 0;                                               :6562
  ├─ cpi->frame_flags = *frame_flags;                         :6564
  │
  ├─ (non-RT only) get_second_pass_params or set_frame_size for pass==1
  │                                                           :6566-6578
  │
  ├─ if (oxcf->pass != 1 && level_constraint.level_index >= 0
  │      && !fail_flag)                                       :6580
  │     level_rc_framerate(cpi, arf_src_index);
  │
  ├─ if (oxcf->pass != 0 || use_svc || frame_is_intra_only)   :6584
  │     for i in REFS_PER_FRAME: scaled_ref_idx[i] = INVALID_IDX;
  │
  ├─ kmeans_data alloc (PERCEPTUAL_AQ; non-RT)               :6588-6599
  │  non_greedy_mv alloc (non-RT)                             :6601-6612
  │
  ├─ if (gf_group_index == 1 &&
  │      gf_group.update_type[gf_group_index] == ARF_UPDATE &&
  │      sf.enable_tpl_model) {
  │     vp9_init_tpl_buffer + vp9_estimate_qp_gop +
  │     vp9_setup_tpl_stats                                  :6617-6623
  │  }                                                       // non-RT
  │
  ├─ cpi->td.mb.fp_src_pred = 0;                              :6637
  │
  ├─ DISPATCH:                                                :6638-6678
  │   if CONFIG_REALTIME_ONLY:
  │      if (use_svc) SvcEncode(cpi, size, dest, dest_size, frame_flags);
  │      else         Pass0Encode(cpi, size, dest, dest_size, frame_flags);
  │   else:
  │      if (pass==1 && !use_svc) vp9_first_pass(cpi, source);
  │      else if (pass==2 && !use_svc) {
  │         Pass2Encode(cpi, size, dest, dest_size, frame_flags,
  │                     encode_frame_result);
  │         vp9_twopass_postencode_update(cpi);
  │      }
  │      else if (use_svc) SvcEncode(cpi, size, dest, dest_size, frame_flags);
  │      else              Pass0Encode(cpi, size, dest, dest_size, frame_flags);
  │
  ├─ if (cm->show_frame) cm->cur_show_frame_fb_idx = cm->new_fb_idx; :6680
  │  if (cm->refresh_frame_context)
  │     cm->frame_contexts[cm->frame_context_idx] = *cm->fc;
  │
  ├─ if ((*size == 0) && !frame_is_intra_only) release_scaled_references
  │                                                           :6686-6688
  ├─ if (*size > 0) cpi->droppable = !frame_is_reference(cpi); :6691
  │
  ├─ if (is_one_pass_svc || (pass==2 && multi-layer SVC))     :6695
  │     vp9_save_layer_context(cpi);
  │
  ├─ if (top spatial layer) cpi->fixed_qp_onepass = 0;        :6701-6702
  │
  ├─ cpi->time_compress_data += elapsed;                      :6704-6705
  ├─ if (keep_level_stats && pass!=1)                         :6707
  │     update_level_info(cpi, size, arf_src_index);
  │
  ├─ if (CONFIG_INTERNAL_STATS && pass!=1 && !last_frame_dropped)
  │     compute PSNR / SSIM / blockiness stats                :6712+
  │
  └─ return 0;
```

The dispatch in §6638-6678 is the heart of the frame loop:

- `Pass0Encode` (`vp9_encoder.c:5930-5939`) chooses CBR or VBR parameter
  computation (`vp9_rc_get_one_pass_cbr_params` or `..._vbr_params`,
  specialist **A5**) and calls `encode_frame_to_data_rate`.
- `SvcEncode` (`vp9_encoder.c:5923-5928`) calls `vp9_rc_get_svc_params` and
  then `encode_frame_to_data_rate`.
- `Pass2Encode` (`vp9_encoder.c:5942-5951`) is non-RT.

`encode_frame_to_data_rate` (`vp9_encoder.c:5537-5921`) is the per-frame
state machine. Real-time flow:

```
  encode_frame_to_data_rate(cpi, size, dest, dest_size,
                            frame_flags, encode_frame_result)
  │
  ├─ if (vp9_svc_check_skip_enhancement_layer(cpi)) return;      :5545
  ├─ set_ext_overrides(cpi);   // pull ext_refresh_* into refresh_*
  │                                                              :5547
  ├─ if (!cm->show_existing_frame) {                             :5555
  │     set_frame_index(cpi, cm);          // RefCntBuffer.frame_index
  │     set_ref_sign_bias(cpi);            // cm->ref_frame_sign_bias[]
  │  }
  │
  ├─ if (current_video_frame==0) deadline_mode_previous_frame = oxcf.mode;
  │                                                              :5565-5566
  │
  ├─ cm->lf.mode_ref_delta_update = 0;                           :5569
  │
  ├─ (non-RT path) recompute interp_filter_search_mask           :5571-5572
  │
  ├─ if (frame_is_intra_only(cm)) {                              :5575
  │     vp9_reset_segment_features(&cm->seg);
  │     if (seg->enabled) { seg->update_map=1; seg->update_data=1; }
  │     rc->source_alt_ref_active = 0;
  │     cm->error_resilient_mode      = oxcf->error_resilient_mode;
  │     cm->frame_parallel_decoding_mode =
  │        oxcf->frame_parallel_decoding_mode;
  │     if (cm->error_resilient_mode) {
  │        cm->frame_parallel_decoding_mode = 1;
  │        cm->reset_frame_context  = 0;
  │        cm->refresh_frame_context= 0;
  │     } else if (cm->intra_only) {
  │        cm->reset_frame_context  = 2;
  │     }
  │  }
  │
  ├─ if (oxcf->tuning == VP8_TUNE_SSIM) set_mb_ssim_rdmult_scaling :5602
  ├─ if (oxcf->aq_mode == PERCEPTUAL_AQ) {                       :5604
  │     init_mb_wiener_var_buffer + set_mb_wiener_variance;
  │  }                                                            // A5
  │
  ├─ save_encode_params(cpi);                                    :5616
  ├─ ext_ratectrl frame rdmult (if registered)                   :5617-5643
  │
  ├─ if (sf.recode_loop == DISALLOW_RECODE) {                    :5645
  │     // REAL-TIME PATH
  │     if (!encode_without_recode_loop(cpi, size, dest, dest_size))
  │        return;     // frame dropped
  │  } else {
  │     // non-RT path: encode_with_recode_loop  (multiple Q passes)
  │  }
  │
  ├─ if (cm->show_existing_frame) {                              :5668
  │     ref_cnt_fb(buffer_pool->frame_bufs, &cm->new_fb_idx,
  │                cm->ref_frame_map[cpi->alt_fb_idx]);
  │  }
  │
  ├─ (non-RT) LOOKAHEAD_AQ: vp9_try_disable_lookahead_aq         :5673-5677
  │
  ├─ // Force-KF pulse reduction: ambient_err = SSE(Source, new)
  │                                                              :5696-5707
  ├─ if (cm->frame_type == KEY_FRAME) cpi->refresh_last_frame = 1; :5710
  │
  ├─ cm->frame_to_show = get_frame_new_buffer(cm);               :5712
  │  cm->frame_to_show->color_space  = cm->color_space;
  │  cm->frame_to_show->color_range  = cm->color_range;
  │  cm->frame_to_show->render_width = cm->render_width;
  │  cm->frame_to_show->render_height= cm->render_height;
  │
  ├─ loopfilter_frame(cpi, cm);                                  :5722
  │    └─ vp9_pick_filter_level(...) + vp9_build_mask_frame +
  │       vp9_loop_filter_frame[_mt]   // A4 / A6
  │
  ├─ if (rc.use_post_encode_drop) save_coding_context(cpi);      :5727
  │
  ├─ vp9_pack_bitstream(cpi, dest, dest_size, size);             :5733     // A4
  │
  ├─ ext_ratectrl update_encodeframe_result (if registered)      :5738-5749
  │
  ├─ // Post-encode CBR drop:                                    :5796-5800
  │  if (rc.use_post_encode_drop && cm->base_qindex < rc.worst_quality
  │      && svc.spatial_layer_id == 0 && post_encode_drop_cbr(cpi, size)) {
  │     restore_coding_context(cpi);
  │     return;
  │  }
  │
  ├─ cpi->last_frame_dropped = 0;                                :5802
  │  svc.last_layer_dropped[svc.spatial_layer_id] = 0;
  │  if (svc.spatial_layer_id == svc.number_spatial_layers - 1)
  │     svc.num_encoded_top_layer++;
  │
  ├─ track fb_idx_upd_tl0 by refresh_* flags                     :5808-5816
  │
  ├─ if (cm->seg.update_map) update_reference_segmentation_map   :5818
  │
  ├─ if (!frame_is_intra_only(cm)) release_scaled_references;    :5820-5822
  ├─ vp9_update_reference_frames(cpi);                           :5823
  │    └─ update_ref_frames(cpi) + denoiser update + svc update
  │
  ├─ if (!cm->show_existing_frame) {                             :5825
  │     for (t=TX_4X4..TX_32X32) full_to_model_counts(...);
  │     if (!error_resilient && !frame_parallel_decoding_mode) {
  │        if (!intra) {
  │           vp9_adapt_mode_probs(cm);            // A4
  │           vp9_adapt_mv_probs(cm, allow_hp);    // A4
  │        }
  │        vp9_adapt_coef_probs(cm);               // A4
  │     }
  │  }
  │
  ├─ cpi->ext_refresh_frame_flags_pending = 0;                   :5840
  ├─ propagate refresh_golden/alt to cpi->frame_flags            :5842-5850
  ├─ cpi->ref_frame_flags = get_ref_frame_flags(cpi);            :5852
  ├─ cm->last_frame_type = cm->frame_type;                       :5854
  ├─ vp9_rc_postencode_update(cpi, *size);                       :5856
  │
  ├─ if (compute_frame_low_motion_onepass && oxcf.pass==0 &&
  │      !intra && (!use_svc || (use_svc && !is_key_frame &&
  │      top spatial layer))) {                                  :5858-5863
  │     vp9_compute_frame_low_motion(cpi);                       // A5
  │  }
  │
  ├─ *size = VPXMAX(1, *size);                                   :5867
  │
  ├─ if (cm->frame_type==KEY_FRAME) *frame_flags |= FRAMEFLAGS_KEY else &= ~
  │                                                              :5873-5878
  │
  ├─ cm->seg.update_map = 0; cm->seg.update_data = 0;
  │  cm->lf.mode_ref_delta_update = 0;                           :5882-5884
  │
  ├─ cm->last_width = cm->width; cm->last_height = cm->height;   :5887-5888
  │
  ├─ if (!cm->show_existing_frame) {                             :5891
  │     cm->last_show_frame = cm->show_frame;
  │     cm->prev_frame      = cm->cur_frame;
  │  }
  │
  ├─ if (cm->show_frame) {                                       :5896
  │     vp9_swap_mi_and_prev_mi(cm);
  │     if (cpi->use_svc) vp9_inc_frame_in_layer(cpi);
  │  }
  │  update_frame_indexes(cm, cm->show_frame);                   :5900
  │
  ├─ if (cpi->use_svc) {                                         :5902
  │     svc.layer_context[..].last_frame_type = cm->frame_type;
  │     svc.spatial_layer_sync[svc.spatial_layer_id] = 0;
  │  }
  │
  ├─ cpi->force_update_segmentation = 0;                         :5912
  │
  └─ svc.previous_frame_is_intra_only = cm->intra_only;          :5919
     svc.set_intra_only_frame = 0;
```

### 3.5 The real-time encode path: `encode_without_recode_loop`

`third_party/libvpx/vp9/encoder/vp9_encoder.c:3958-4290`. This is the
single function distinguishing real-time from good-quality.

```
  encode_without_recode_loop(cpi, size, dest, dest_size) → int (1=ok, 0=drop)
  │
  ├─ if (cm->show_existing_frame) { rc.this_frame_target=0;
  │     if (psnr_enabled) set_raw_source_frame(cpi); return 1; }  :3973-3977
  │
  ├─ svc.time_stamp_prev[layer] = svc.time_stamp_superframe;     :3979
  ├─ cpi->compute_source_sad_onepass = (oxcf.mode == REALTIME);  :3984
  │
  ├─ set_frame_size(cpi);                                        :3988
  │    └─ may update cm->width/height for dynamic resize;
  │       allocates / reallocates context buffers if size changed.
  │
  ├─ // Source frame scaling (SVC and dynamic resize):           :3990-4016
  │  if (1/4×1/4 svc downscale path) cpi->Source = vp9_svc_twostage_scale
  │  else if (1/2×1/2 reuse) cpi->Source = svc->scaled_temp
  │  else cpi->Source = vp9_scale_if_required(cm, un_scaled_source,
  │                                            &scaled_source,
  │                                            (oxcf.pass==0),
  │                                            filter_scaler, phase_scaler);
  │
  ├─ if (psnr_calc_enabled) cpi->raw_source_frame = cpi->Source; :4025-4037
  │
  ├─ // Invalidate per-SB content state on SVC/resize boundaries :4039-4050
  │  if (use_svc && not-top-layer || resize_pending || resize_state ||
  │      external_resize || resize_state != ORIG) {
  │      compute_source_sad_onepass = 0;
  │      memset(content_state_sb_fd, 0, ...);
  │  }
  │
  ├─ // Conditionally scale last source for source-SAD / nestd:  :4056-4070
  │  if (need Last_Source for various tests)
  │     cpi->Last_Source = vp9_scale_if_required(unscaled_last_source,
  │                                              scaled_last_source, ...);
  │  if (size mismatch) compute_source_sad_onepass = 0;
  │
  ├─ if (intra_only || resize_pending) memset(consec_zero_mv, 0, ...); :4072
  │
  ├─ (denoiser reset on first SVC frame)                         :4077-4080
  │
  ├─ // 1-pass scene detection                                   :4085-4093
  │  rc.high_source_sad = 0; rc.hybrid_intra_scene_change = 0;
  │  rc.re_encode_maxq_scene_change = 0;
  │  if (show_frame && mode==REALTIME && !disable_scene_detection_rtc &&
  │      (rc_mode==VBR || content==SCREEN || (speed>=5 && speed<8)))
  │      vp9_scene_detection_onepass(cpi);    // A5
  │
  ├─ // SVC superframe-level scene change + temporal-layer reset :4095-4111
  │
  ├─ vp9_update_noise_estimate(cpi);     // A5                    :4113
  │
  ├─ // 1-pass CBR frame drop                                    :4115-4127
  │  if (rc.high_source_sad || svc.high_source_sad_superframe ...)
  │     no_drop_scene_change = 1;
  │  if (oxcf.pass==0 && rc_mode==VPX_CBR && !intra && !no_drop &&
  │      !svc.superframe_has_layer_sync &&
  │      (!use_svc || !svc.layer_context[tl].is_key_frame)) {
  │     if (vp9_rc_drop_frame(cpi)) return 0;  // DROP
  │  }
  │
  ├─ if (!intra && (svc.single_layer_svc resize ||
  │                  !(svc force_zero_mode_spatial_ref))) {       :4134-4138
  │     vp9_scale_references(cpi);
  │  }
  │
  ├─ set_size_independent_vars(cpi);                              :4140
  │     // entropy mode init, lf level seed
  │  set_size_dependent_vars(cpi, &q, &bottom_index, &top_index); :4141
  │     // pick base qindex range, q-target etc.   (A5)
  │
  ├─ init_motion_estimation(cpi);                                 :4144
  │     // populate cpi->diamond_search_sad, ss_cfg etc.  (A3)
  │
  ├─ if (sf.copy_partition_flag) alloc_copy_partition_data(cpi);  :4146
  ├─ alloc svc->prev_partition_svc if needed (sf.svc_use_lowres_part)
  │                                                               :4148-4156
  │
  ├─ // Heuristic: enable skin detection                          :4158-4164
  │  if (bit_depth==8 && speed>=5 && pass==0 && rc_mode==VPX_CBR
  │      && content != SCREEN && aq_mode == CYCLIC_REFRESH_AQ)
  │      cpi->use_skin_detection = 1;
  │
  ├─ rc.use_post_encode_drop =                                    :4167-4170
  │     (rc.ext_use_post_encode_drop && rc_mode==VPX_CBR &&
  │      cm->frame_type != KEY_FRAME);
  │
  ├─ vp9_set_quantizer(cpi, q);                                   :4172
  │  vp9_set_variance_partition_thresholds(cpi, q, 0);            :4173
  │
  ├─ setup_frame(cpi);                                            :4175
  │    // Set frame_context_idx, reset_frame_context for KF.
  │
  ├─ suppress_active_map(cpi);                                    :4177
  │
  ├─ if (use_svc) {                                               :4179-4184
  │     if (spatial_layer_id > 0) vp9_svc_constrain_inter_layer_pred;
  │     vp9_svc_assert_constraints_pattern(cpi);
  │  }
  │
  ├─ // Scene change re-q                                         :4186-4204
  │  if (sf.overshoot_detection_cbr_rt == FAST_DETECTION_MAXQ &&
  │      (rc.high_source_sad || (use_svc && svc.high_source_sad_superframe))) {
  │     if (vp9_encodedframe_overshoot(cpi, -1, &q)) {
  │        vp9_set_quantizer(cpi, q);
  │        vp9_set_variance_partition_thresholds(cpi, q, 0);
  │     }
  │  }
  │
  ├─ // AQ mode setup                                             :4206-4239
  │  (non-RT branches: VARIANCE_AQ, EQUATOR360_AQ, COMPLEXITY_AQ, LOOKAHEAD_AQ)
  │  RT branch (or always for CYCLIC_REFRESH):
  │     if (roi.enabled && !intra) apply_roi_map / cyclic-refresh combo
  │     else if (aq_mode == CYCLIC_REFRESH_AQ) vp9_cyclic_refresh_setup;
  │
  ├─ apply_active_map(cpi);                                       :4241
  │
  ├─ vp9_encode_frame(cpi);    // dispatches to threading layer   :4243
  │
  ├─ // Optional re-encode at high Q on detected overshoot         :4249-4278
  │  if (sf.overshoot_detection_cbr_rt == RE_ENCODE_MAXQ &&
  │      (rc.high_source_sad || (use_svc && svc.high_source_sad_superframe))) {
  │     save_coding_context(cpi);
  │     vp9_pack_bitstream(cpi, dest, dest_size, size);   // size probe
  │     restore_coding_context(cpi);
  │     int frame_size = (int)(*size) << 3;
  │     if (vp9_encodedframe_overshoot(cpi, frame_size, &q)) {
  │        vp9_set_quantizer(cpi, q);
  │        vp9_set_variance_partition_thresholds(cpi, q, 0);
  │        suppress_active_map(cpi);
  │        if (aq_mode == CYCLIC_REFRESH_AQ) {
  │           memset(segmentation_map, 0, mi_rows*mi_cols);
  │           memset(cyclic_refresh.last_coded_q_map, MAXQ, ...);
  │           cyclic_refresh.sb_index = 0;
  │           vp9_disable_segmentation(&cm->seg);
  │        }
  │        apply_active_map(cpi);
  │        vp9_encode_frame(cpi);     // RE-ENCODE
  │     }
  │  }
  │
  ├─ if (aq_mode==CYCLIC_REFRESH_AQ && seg.enabled && !intra
  │      && cyclic_refresh.content_mode)
  │     vp9_cyclic_refresh_postencode(cpi);                       :4281-4283
  │
  └─ return 1;
```

Key invariants of the real-time path:

- **No recode loop.** `sf.recode_loop == DISALLOW_RECODE` is set by the
  real-time speed features. The single "re-encode-on-overshoot" branch
  inside `encode_without_recode_loop` is the only conditional re-encode;
  it triggers at most one extra pass and only on detected scene change.
- **No alt-ref frames.** `is_altref_enabled` returns 0 when
  `oxcf.mode == REALTIME && oxcf.rc_mode == VPX_CBR` (`vp9_encoder.h:1423`).
  RT VBR can use altref if `lag_in_frames >= 4`, but the typical
  WebRTC/livestream config is CBR with lag 0.
- **Single-pass.** `oxcf.pass == 0`. The `vp9_first_pass` / `Pass2Encode`
  branches in `vp9_get_compressed_data` are dead.
- **Frame dropping is enabled** (`vp9_rc_drop_frame` and post-encode
  `post_encode_drop_cbr`). Specialist **A5**.

### 3.6 `vp9_encode_frame` and the threading entry point

`third_party/libvpx/vp9/encoder/vp9_encodeframe.c:6272-6440`.

```
  vp9_encode_frame(cpi)
   ├─ restore_encode_params(cpi);                                :6275
   │     // copy thresh_freq_fact_prev → thresh_freq_fact in each tile
   ├─ if (!frame_is_intra_only) decide cpi->allow_comp_inter_inter
   │                                                              :6287-6294
   ├─ if (sf.frame_parameter_update) {                            :6296
   │     // single-pass-over-frame mode that infers reference_mode,
   │     // interp_filter, tx_mode from previous frame stats.
   │     pick prediction_type (SINGLE / COMPOUND / SELECT)        :6314-6324
   │     pick interp_filter from filter_thrs                      :6326-6327
   │     encode_frame_internal(cpi);                              :6332
   │     update prediction_type_threshes and filter_threshes
   │     post-process counts → may downgrade reference_mode
   │     post-process tx counts → may pin tx_mode
   │  } else {
   │     encode_frame_internal(cpi);
   │  }
```

`encode_frame_internal` (`vp9_encodeframe.c:6069-6207`) is where the
threading dispatch happens:

```
  encode_frame_internal(cpi)
   ├─ ThreadData *td = &cpi->td;
   │  MACROBLOCK *x = &td->mb;
   │  MACROBLOCKD *xd = &x->e_mbd;
   ├─ xd->mi = cm->mi_grid_visible; xd->mi[0] = cm->mi;          :6077-6078
   ├─ vp9_zero(*td->counts); vp9_zero(cpi->td.rd_counts);        :6079-6080
   ├─ xd->lossless = (base_qindex == 0 && all delta_q == 0);     :6082-6083
   ├─ choose fwd_txfm4x4 / inv_txfm_add for lossless vs not       :6086-6095
   ├─ x->optimize = sf.optimize_coefficients==1 && pass!=1;       :6096
   │  if (lossless) x->optimize = 0;
   ├─ x->sharpness = oxcf.sharpness;                              :6098
   ├─ x->adjust_rdmult_by_segment = (aq_mode == VARIANCE_AQ);     :6099
   ├─ cm->tx_mode = select_tx_mode(cpi, xd);                      :6101
   ├─ vp9_frame_init_quantizer(cpi);                              :6103   // A4
   ├─ vp9_initialize_rd_consts(cpi);                              :6105   // A2
   ├─ vp9_initialize_me_consts(cpi, x, cm->base_qindex);          :6106   // A3
   ├─ init_encode_frame_mb_context(cpi);                          :6107
   ├─ cm->use_prev_frame_mvs =                                    :6108-6110
   │     !error_resilient && width==last_width && height==last_height
   │     && !intra_only && last_show_frame;
   ├─ cm->prev_mi = use_prev_frame_mvs ? prev_mip + stride + 1 : NULL;
   │                                                              :6113-6114
   ├─ x->quant_fp = sf.use_quant_fp;                              :6116
   ├─ vp9_zero(x->skip_txfm);                                     :6117
   ├─ if (sf.use_nonrd_pick_mode) {                               :6118-6140
   │     // RT path. Wire pc_root->none plane buffers directly into
   │     // td->mb.plane[].coeff/qcoeff/eobs and xd->plane[].dqcoeff
   │     // so there is no buffer swap during non-RD pick.
   │     vp9_zero(x->zcoeff_blk);
   │     if (frame_type != KEY_FRAME && frames_since_golden==0 &&
   │         !(lag_in_frames>0 && rc_mode==VPX_VBR) && !use_svc)
   │        cpi->ref_frame_flags &= ~VP9_GOLD_FLAG;
   │     if (sf.partition_search_type == SOURCE_VAR_BASED_PARTITION)
   │        source_var_based_partition_search_method(cpi);
   │  } else if (gf_group_index && sf.enable_tpl_model) {
   │     // non-RT path: feed TPL stats into cpi->rd.r0
   │  }
   │
   ├─ // Invalidate references with bad scale factors             :6165-6172
   │  for ref_frame in LAST..ALTREF:
   │     if (ref_frame_flags & flag(ref) &&
   │         (frame_refs[ref-1].sf.x_scale_fp == REF_INVALID_SCALE ||
   │          frame_refs[ref-1].sf.y_scale_fp == REF_INVALID_SCALE))
   │        cpi->ref_frame_flags &= ~flag(ref);
   │
   ├─ if (aq_mode == PERCEPTUAL_AQ) build_kmeans_segmentation(cpi); :6175
   │
   └─ {                                                            :6178-6198
       struct vpx_usec_timer emr_timer;
       vpx_usec_timer_start(&emr_timer);
       if (!cpi->row_mt) {
          cpi->row_mt_sync_read_ptr  = vp9_row_mt_sync_read_dummy;
          cpi->row_mt_sync_write_ptr = vp9_row_mt_sync_write_dummy;
          if (VPXMIN(oxcf.max_threads, 1 << cm->log2_tile_cols) > 1)
             vp9_encode_tiles_mt(cpi);                // tile-parallel
          else
             encode_tiles(cpi);                       // single-thread
       } else {
          cpi->row_mt_sync_read_ptr  = vp9_row_mt_sync_read;
          cpi->row_mt_sync_write_ptr = vp9_row_mt_sync_write;
          vp9_encode_tiles_row_mt(cpi);                // row-mt
       }
       vpx_usec_timer_mark(&emr_timer);
       cpi->time_encode_sb_row += vpx_usec_timer_elapsed(&emr_timer);
      }
   │
   └─ sf.skip_encode_frame = sf.skip_encode_sb ?
                             get_skip_encode_frame(cm, td) : 0;   :6200-6201
```

This is the boundary between architecture (this doc) and tile / row-mt
threading (specialist **A6**). The three dispatch targets:

- `encode_tiles(cpi)` — single-threaded loop over tiles, calling
  `vp9_encode_tile(cpi, &cpi->td, tile_row, tile_col)` for each. Inside,
  the per-tile loop walks rows of superblocks via `vp9_encode_sb_row`
  (specialist **A2** for mode-decision; **A6** for the sb-row iteration).
- `vp9_encode_tiles_mt(cpi)` — parallel tile encode, one worker per tile
  (or interleaved when more tiles than workers). See §3.7.
- `vp9_encode_tiles_row_mt(cpi)` — row-mt path: each worker dequeues rows
  from a per-tile-column `JobQueue` and runs `vp9_encode_sb_row`. See §3.7.

### 3.7 Threading model **overview** (A6 owns the deep dive)

The encoder threading code lives in
`third_party/libvpx/vp9/encoder/vp9_ethread.{h,c}` plus
`vp9_multi_thread.{h,c}` for the row-mt job queue.

**Worker abstraction.** libvpx uses its own portable `VPxWorker` /
`VPxWorkerInterface` (`vpx_util/vpx_thread.h`). Each worker holds a
`hook`, two opaque `data1`/`data2` pointers, and condition variables for
launch and synchronization. Concrete platform backends: pthreads on POSIX
and Win32 threads on Windows.

**Worker lifecycle.** `create_enc_workers(cpi, num_workers)` is called
lazily on the first frame that needs more workers than currently
allocated. `vp9_ethread.c:80-138`:

- If `cpi->use_svc && !cpi->row_mt`, `num_workers` is clamped to
  `min(oxcf.max_threads, get_max_tile_cols(cpi))` (`:87-90`).
- If the requested worker count matches the existing pool, returns
  immediately (`:92`).
- Otherwise frees old workers (`vp9_loop_filter_dealloc`,
  `vp9_bitstream_encode_tiles_buffer_dealloc`, `vp9_encode_free_mt_data`)
  and reallocates `cpi->workers`, `cpi->tile_thr_data`.
- For workers `0 .. num_workers - 2`: allocate a `ThreadData` (32-byte
  aligned), zero it, set up `pc_tree` via `vp9_setup_pc_tree`, allocate
  `td->counts`, and `winterface->reset(worker)` to create the OS thread.
- Worker `num_workers - 1` (the main thread) uses `cpi->td` directly.
- `winterface->sync(worker)` to make sure each worker is idle before
  proceeding.

`vp9_encode_free_mt_data` (`vp9_ethread.c:173-192`) tears down workers:
`winterface->end(worker)` joins each OS thread; per-worker `td->counts`
and `pc_tree` are freed; `cpi->tile_thr_data` and `cpi->workers` are
freed; `cpi->num_workers = 0`.

**Three encode-time hooks.** Each is registered before launch via
`launch_enc_workers(cpi, hook, data2, num_workers)` (`:140-171`), which
sets `worker->hook`, `worker->data1`, `worker->data2`, then calls
`launch()` for workers 0..n-2 and `execute()` for the main thread
(blocking), then syncs all workers.

1. **`enc_worker_hook` (`vp9_ethread.c:40-59`).** Tile-parallel.
   Each worker iterates over tiles assigned to it by stride
   `num_workers`, calling `vp9_encode_tile(cpi, td, tile_row, tile_col)`.

2. **`enc_row_mt_worker_hook` (`vp9_ethread.c:586-618`).** Row-mt.
   Each worker repeatedly dequeues `JobNode` jobs from the
   `MultiThreadHandle.job_queue` via `vp9_enc_grp_get_next_job`; for each
   job, it calls `vp9_encode_sb_row(cpi, td, tile_row, tile_col, mi_row)`.
   When the per-tile queue is exhausted, it polls
   `vp9_get_tiles_proc_status` to migrate to another tile column.

3. **`first_pass_worker_hook` (`vp9_ethread.c:416-458`).** Non-RT
   first-pass row-mt.

Additionally `temporal_filter_worker_hook` (`vp9_ethread.c:507-545`)
parallelizes the 2-pass temporal filter.

**Synchronization.** Row-mt synchronization uses
`VP9RowMTSync::{mutex[], cond[], cur_col[]}` per row (`vp9_ethread.h:33-43`).
`vp9_row_mt_sync_read` waits until `cur_col[r-1] >= c - sync_range + 1`;
`vp9_row_mt_sync_write` updates `cur_col[r]` and signals
`cond[r]`. When row-mt is off, the `_dummy` variants are used as
no-ops via the `cpi->row_mt_sync_read_ptr` / `_write_ptr` function pointers
(`vp9_encodeframe.c:6181-6193`).

**Loop filter parallelism.** Independent of encode threading. `loopfilter_frame`
(`vp9_encoder.c:3392-3452`) chooses between `vp9_loop_filter_frame_mt` (when
`num_workers > 1`) and `vp9_loop_filter_frame` (single-threaded). The
multithreaded loop filter has its own row-sync structure
(`cpi->lf_row_sync`, type `VP9LfSync`).

**Bitstream packing parallelism.** Optional. When enabled, multiple
worker threads pack different tiles concurrently using
`cpi->vp9_bitstream_worker_data`. The default path (used by RT
WebRTC) is serial because tile packing must be cheap. Specialist **A4**.

Specialist **A6** has the deep dive on:

- The per-tile vs. per-row job assignment policy
  (`vp9_assign_tile_to_thread`, `vp9_prepare_job_queue`,
  `vp9_multi_thread_tile_init`).
- The exact memory model: which fields each worker reads vs. writes vs.
  shares.
- SVC layer-by-layer worker reuse.
- The `set_rt_speed_feature_*` dispatch table that drives RT mode
  decisions.

### 3.8 Encoder teardown

```
  encoder_destroy(ctx)                                            cx_iface.c:1161
   ├─ free(ctx->cx_data);
   ├─ free(ctx->global_headers.buf);
   ├─ vp9_remove_compressor(ctx->cpi);
   ├─ vpx_free(ctx->buffer_pool);
   └─ vpx_free(ctx);
```

`vp9_remove_compressor` (`vp9_encoder.c:2772-2945`):

1. (Internal stats) Free `cpi->ssim_vars` (`:2779`).
2. (Internal stats) If `current_video_frame > 0` and `pass != 1`, write
   `opsnr.stt` (`:2783-2860`).
3. (CONFIG_VP9_TEMPORAL_DENOISING) `vp9_denoiser_free(&cpi->denoiser)` (`:2874-2876`).
4. If `kmeans_data_arr_alloc`: destroy `kmeans_mutex` (multi-thread only)
   and free `kmeans_data_arr` (`:2878-2883`).
5. `vp9_free_tpl_buffer(cpi)` (`:2885`).
6. `vp9_loop_filter_dealloc(&cpi->lf_row_sync)` (`:2887`).
7. `vp9_bitstream_encode_tiles_buffer_dealloc(cpi)` (`:2888`).
8. `vp9_row_mt_mem_dealloc(cpi)` (`:2889`): row-mt job queue.
9. `vp9_encode_free_mt_data(cpi)` (`:2890`): worker threads joined, per-worker
   state freed.
10. (Non-RT) `vp9_alt_ref_aq_destroy(cpi->alt_ref_aq)` (`:2892-2894`).
11. `dealloc_compressor_data(cpi)` (`:2896`): frees `mbmi_ext_base`,
    `tile_data`, `segmentation_map`, MV cost LUTs, `skin_map`,
    `prev_partition`, `prev_segment_id`, `prev_variance_low`,
    `copied_frame_cnt`, `content_state_sb_fd`, and so on. See
    `:944-1000`.
12. Free `mbgraph_stats[].mb_stats` (`:2898-2901`).
13. `vp9_extrc_delete(&cpi->ext_ratectrl)` (`:2903`).
14. Scribble `cm->error.detail` with `'A'` so use-after-free is easy to
    spot (`:2906`).
15. `vp9_remove_common(cm)` (`:2909`): frees MI grid, segmentation
    pingpong arrays, frame contexts (`cm->fc`, `cm->frame_contexts`),
    loop-filter info.
16. `vp9_free_ref_frame_buffers(cm->buffer_pool)` (`:2910`): the 12
    `RefCntBuffer` payloads.
17. (Post-proc) `vp9_free_postproc_buffers(cm)` (`:2912`).
18. `vpx_free(cpi)` (`:2914`).
19. Close debug `OUTPUT_YUV_*` files (`:2917-2932`). All `#ifdef`'d
    out by default.

Note that the `BufferPool` itself is allocated by `encoder_init` in
`cx_iface.c:1126` and freed in `encoder_destroy` after `vp9_remove_compressor`.
That ordering matters: `vp9_remove_common` reads `cm->buffer_pool` to free
the frame buffers; the pool object itself is freed afterwards.

### 3.9 Frame-dependency graph

Real-time VP9 has a flat reference structure on top of an 8-slot ring of
reference buffers.

```
   cm->buffer_pool->frame_bufs[FRAME_BUFFERS]    (12 RefCntBuffer slots)
        │
        │   cm->ref_frame_map[REF_FRAMES]        (8 ref slots, values are
        │                                         indices into frame_bufs[])
        v
   ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
   │  0   │  1   │  2   │  3   │  4   │  5   │  6   │  7   │
   └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
      ↑       ↑       ↑                 ↑       ↑
      │       │       │                 │       │
      cpi->lst_fb_idx  cpi->alt_fb_idx   buffer_gf_temporal_ref[0..1]
      = 0  (init)      = 2  (init)       (SVC long-term ref slot 6, 7)
              ↑
              cpi->gld_fb_idx = 1 (init)
```

Initialization: `init_buffer_indices(cpi)` (`vp9_encoder.c:1454-1463`) sets
`cpi->ref_fb_idx[ref] = ref` for ref in `[0, REF_FRAMES)`, then
`lst_fb_idx = ref_fb_idx[LAST_FRAME - 1] = 0`,
`gld_fb_idx = ref_fb_idx[GOLDEN_FRAME - 1] = 1`,
`alt_fb_idx = ref_fb_idx[ALTREF_FRAME - 1] = 2`.

#### Refresh decisions

`encode_frame_to_data_rate` (RT, non-SVC) chooses `refresh_*` flags as
follows:

- **Keyframe**: `refresh_last_frame = 1`, `refresh_golden_frame = 1`,
  `refresh_alt_ref_frame = 1` set by `setup_frame()`
  (`vp9_encoder.c:851-854`) — KF refreshes everything. Also,
  `set_ext_overrides` overrides this with user `ext_refresh_*` if pending.
- **Default inter frame**: `refresh_last_frame = 1`,
  `refresh_golden_frame = 0`, `refresh_alt_ref_frame = 0` set at top of
  `vp9_get_compressed_data` (`vp9_encoder.c:6403-6407`).
- **Golden-frame update**: golden refresh decided by rate control's
  `frames_till_gf_update_due`; specialist **A5** owns the policy.
- **Alt-ref**: only when `is_altref_enabled(cpi)` is true
  (`vp9_encoder.h:1423-1427`). For RT CBR this is always false. For RT
  VBR with `lag_in_frames >= 4` and `enable_auto_arf`, it can be true and
  `arf_src_index = rc->frames_till_gf_update_due` (`:6058-6059`).

#### Reference-slot rotation after encode

`update_ref_frames(cpi)` (`vp9_encoder.c:3292-3380`) does the actual
`ref_cnt_fb` rotation:

- **`rc.show_arf_as_gld`**: swap `cpi->alt_fb_idx ↔ cpi->gld_fb_idx`
  (`:3297-3300`).
- **`cm->show_existing_frame`** (ARF promotion): pop the ARF index
  stack into `cpi->lst_fb_idx`, decrement `gf_group->stack_size`
  (`:3301-3307`).
- **Keyframe**: copy `cm->new_fb_idx` into `cm->ref_frame_map[gld_fb_idx]`
  *and* `cm->ref_frame_map[alt_fb_idx]` (`:3311-3315`). This is why on a
  KF, golden and alt-ref point at the same buffer.
- **`vp9_preserve_existing_gf` is true** (replace ARF with previous GF):
  swap ARF slot with new_fb_idx and swap `alt_fb_idx ↔ gld_fb_idx`
  (`:3316-3332`).
- **Other inter frames**:
  - If `refresh_alt_ref_frame`: push old `alt_fb_idx` onto the
    `arf_index_stack`, then write `new_fb_idx` into
    `cm->ref_frame_map[gf_group->top_arf_idx]`, and set
    `alt_fb_idx = top_arf_idx`. Also copy `interp_filter_selected[0]` into
    slot `ALTREF_FRAME` (`:3334-3350`).
  - If `refresh_golden_frame`: `cm->ref_frame_map[gld_fb_idx] = new_fb_idx`;
    copy `interp_filter_selected[0]` (or `[ALTREF_FRAME]` when overlay
    frame) into `[GOLDEN_FRAME]` (`:3352-3363`).
  - If `refresh_last_frame`: `cm->ref_frame_map[lst_fb_idx] = new_fb_idx`;
    copy `interp_filter_selected[0]` into `[LAST_FRAME]`
    (`:3366-3373`).
- **`MID_OVERLAY_UPDATE` gf_group type**: pop the ARF stack into
  `alt_fb_idx` (`:3375-3379`).

Then `vp9_update_reference_frames` (`vp9_encoder.c:3382-3390`) does:

1. `update_ref_frames(cpi)`.
2. `vp9_denoiser_update_ref_frame(cpi)` (if denoiser enabled).
3. If `is_one_pass_svc(cpi)`: `vp9_svc_update_ref_frame(cpi)` (specialist
   **A6**).

#### Real-time non-SVC frame graph (simplest case)

```
   frame N-1                       frame N                         frame N+1
   ┌──────────┐    refresh_last   ┌──────────┐    refresh_last   ┌──────────┐
   │  recon   │ ─────────────────►│  recon   │ ─────────────────►│  recon   │
   └────┬─────┘                   └────┬─────┘                   └────┬─────┘
        │                              │                              │
        ▼                              ▼                              ▼
   ref_frame_map[lst_fb_idx]      same slot                       same slot
   (single LAST slot, slot 0)     overwritten on each             rotated forward
                                  refresh_last
```

Plus, at the configurable GF interval (`rc->frames_till_gf_update_due`):

```
   frame K (golden)                                  next K + interval
   ┌──────────┐    refresh_golden  ┌──────────┐
   │  recon   │ ─────────────────► │ golden   │  (stays in ref_frame_map[gld_fb_idx]
   └──────────┘                    └──────────┘   until next golden refresh)
```

Golden is the bridge across the GOP; it persists for the duration of the
GF interval. The encoder uses golden as a second reference frame, e.g.
for objects that recur with motion (lighting, faces).

ALTREF is not present in RT CBR. In RT VBR or 2-pass it sits "in the
future" with `show_frame = 0` for the encoded ARF frame and an OVERLAY
frame later promoting it via `show_existing_frame`.

#### SVC frame graph

Specialist **A6** owns the SVC patterns. At the architecture level, the
following invariants apply:

- `cpi->svc.spatial_layer_id` and `cpi->svc.temporal_layer_id` are set
  before each `encode_frame_to_data_rate` call; on the same input frame,
  `encoder_encode` calls `vp9_get_compressed_data` repeatedly, once per
  spatial layer (the loop is `cx_iface.c:1480-1582`, terminating when
  the top spatial layer encoded successfully, `:1575-1578`).
- `vp9_one_pass_svc_start_layer` (`vp9_svc_layercontext.c:769-1010`)
  computes per-layer `lst_fb_idx`, `gld_fb_idx`, `alt_fb_idx` for the
  chosen `temporal_layering_mode` (`0212`, `0101`, `NOLAYERING`,
  `BYPASS`). Each layer effectively reuses the 8-slot reference pool with
  layer-specific assignments.
- `vp9_svc_update_ref_frame` (called from `vp9_update_reference_frames`,
  `vp9_encoder.c:3389`) does post-encode bookkeeping for SVC.
- For spatial SVC, `vp9_scale_references(cpi)` is what produces the
  current-layer-resolution scaled copy of the previous spatial layer's
  recon, stored in `cpi->scaled_ref_idx[]`.

#### Reference flags

`get_ref_frame_flags(cpi)` (`vp9_encoder.c:4292-`) collapses identical
slots:

```c
// Real flags: VP9_LAST_FLAG | VP9_GOLD_FLAG | VP9_ALT_FLAG initially.
if (gold_is_last) flags &= ~VP9_GOLD_FLAG;  // dedupe slot
if (frames_till_gf_update_due == INT_MAX && single-layer)
    flags &= ~VP9_GOLD_FLAG;
if (alt_is_last)  flags &= ~VP9_ALT_FLAG;
// (etc, ~10 lines)
```

This is the only place where the encoder might disable references at the
frame level for ME / mode decision: if two slots point at the same buffer,
the duplicate flag is cleared so the picker doesn't waste cycles
re-evaluating it.

`set_ref_sign_bias(cpi)` (`vp9_encoder.c:5034-5049`) computes
`cm->ref_frame_sign_bias[ref] = (cur_frame_index < ref_buf->frame_index)`
for each enabled reference — i.e. true when the reference is in the
future. ALTREF normally has sign_bias = 1; LAST and GOLDEN have 0. This
flows into entropy and MV-pred for the compound-prediction path.

---

## 4. Algorithms (pseudocode)

This section formalizes the loop-control algorithms specific to
architecture. Subsystem-specific algorithms (mode decision, ME, RC, AQ,
threading dispatch policy) are owned by other specialists.

### 4.1 Public encode call: deadline → mode dispatch

`third_party/libvpx/vp9/vp9_cx_iface.c:1170-1212`:

```
function pick_quickcompress_mode(ctx, duration, deadline):
    new_mode = BEST
    if CONFIG_REALTIME_ONLY:
        deadline = VPX_DL_REALTIME
    else:
        switch ctx.cfg.g_pass:
            case VPX_RC_ONE_PASS:
                if deadline > 0:
                    duration_us = duration * tsr.num /
                                  (tsr.den * (TICKS_PER_SEC / 1_000_000))
                    new_mode = (deadline > duration_us) ? GOOD : REALTIME
                else:
                    new_mode = BEST
            case VPX_RC_FIRST_PASS:
                # no-op
            case VPX_RC_LAST_PASS:
                new_mode = (deadline > 0) ? GOOD : BEST
    if deadline == VPX_DL_REALTIME:
        ctx.oxcf.pass = 0
        new_mode = REALTIME
    if ctx.oxcf.mode != new_mode:
        ctx.oxcf.mode = new_mode
        vp9_change_config(ctx.cpi, &ctx.oxcf)
```

### 4.2 Drain semantics

`encoder_encode` (`vp9_cx_iface.c:1480-1582`) calls
`vp9_get_compressed_data` in a loop, **until** the loop body breaks. The
loop condition is:

```
   while (cx_data_sz >= ctx->cx_data_sz / 2 &&
          vp9_get_compressed_data(...) != -1) {
      ...
   }
```

`-1` from `vp9_get_compressed_data` means "no source available and drain
requested" (`vp9_encoder.c:6520-6522`: `*size = 0; return -1`).

In SVC, the loop also breaks when the top spatial layer succeeded for
this superframe (`vp9_cx_iface.c:1575-1578`):

```c
if (is_one_pass_svc(cpi) && (cpi->svc.spatial_layer_id ==
                             cpi->svc.number_spatial_layers - 1)) {
   break;
}
```

This is how the encoder iterates `S` times over the same input image to
produce `S` spatial layers, each calling
`encode_frame_to_data_rate`.

### 4.3 Reference slot rotation after a frame

Distilled from §3.9 above. Pseudocode for `update_ref_frames(cpi)` in
the non-SVC case:

```
function update_ref_frames(cpi):
    pool = cm->buffer_pool
    gf_group = &cpi->twopass.gf_group

    if rc->show_arf_as_gld:
        swap(cpi->alt_fb_idx, cpi->gld_fb_idx)
    else if cm->show_existing_frame:
        cpi->lst_fb_idx = cpi->alt_fb_idx
        cpi->alt_fb_idx = stack_pop(gf_group->arf_index_stack,
                                    gf_group->stack_size)
        --gf_group->stack_size

    if cm->frame_type == KEY_FRAME:
        ref_cnt_fb(pool->frame_bufs, &cm->ref_frame_map[cpi->gld_fb_idx],
                   cm->new_fb_idx)
        ref_cnt_fb(pool->frame_bufs, &cm->ref_frame_map[cpi->alt_fb_idx],
                   cm->new_fb_idx)
    else if vp9_preserve_existing_gf(cpi):
        ref_cnt_fb(pool->frame_bufs, &cm->ref_frame_map[cpi->alt_fb_idx],
                   cm->new_fb_idx)
        swap(cpi->alt_fb_idx, cpi->gld_fb_idx)
    else:
        if cpi->refresh_alt_ref_frame:
            arf_idx = gf_group->top_arf_idx
            stack_push(gf_group->arf_index_stack,
                       cpi->alt_fb_idx, gf_group->stack_size)
            ++gf_group->stack_size
            ref_cnt_fb(pool->frame_bufs, &cm->ref_frame_map[arf_idx],
                       cm->new_fb_idx)
            memcpy(cpi->interp_filter_selected[ALTREF_FRAME],
                   cpi->interp_filter_selected[0], sizeof(...))
            cpi->alt_fb_idx = arf_idx
        if cpi->refresh_golden_frame:
            ref_cnt_fb(pool->frame_bufs, &cm->ref_frame_map[cpi->gld_fb_idx],
                       cm->new_fb_idx)
            copy_interp_filter(GOLDEN_FRAME,
                               rc->is_src_frame_alt_ref ? ALTREF_FRAME : 0)

    if cpi->refresh_last_frame:
        ref_cnt_fb(pool->frame_bufs, &cm->ref_frame_map[cpi->lst_fb_idx],
                   cm->new_fb_idx)
        if !rc->is_src_frame_alt_ref:
            memcpy(cpi->interp_filter_selected[LAST_FRAME],
                   cpi->interp_filter_selected[0], sizeof(...))

    if gf_group->update_type[gf_group->index] == MID_OVERLAY_UPDATE:
        cpi->alt_fb_idx = stack_pop(gf_group->arf_index_stack,
                                    gf_group->stack_size)
        --gf_group->stack_size
```

In RT (no altref, no overlay, no `show_existing_frame`, no
`show_arf_as_gld`), this collapses to:

```
if KEY_FRAME:
   map[gld_fb_idx] = new_fb_idx
   map[alt_fb_idx] = new_fb_idx
if refresh_golden_frame:
   map[gld_fb_idx] = new_fb_idx
if refresh_last_frame:
   map[lst_fb_idx] = new_fb_idx
```

`ref_cnt_fb` decrements the old slot's ref count and increments the new
one — exactly the operation that drives the `BufferPool` slot recycling
via `get_free_fb`.

### 4.4 `set_ext_overrides` interaction

When the application calls `vp9_update_reference(cpi, flags)`
(`vp9_encoder.c:2969-2974`) it sets `cpi->ext_refresh_*` and
`cpi->ext_refresh_frame_flags_pending = 1`. On the next encode, at the top
of `encode_frame_to_data_rate`, `set_ext_overrides(cpi)` copies these into
`cpi->refresh_last_frame / golden_frame / alt_ref_frame` and clears the
pending flag at the end of `encode_frame_to_data_rate`
(`vp9_encoder.c:5840`). This is how WebRTC SFUs that drive their own
SVC pattern control which slots are updated each frame.

### 4.5 Frame-pool slot acquisition

`vp9_get_compressed_data` at `vp9_encoder.c:6549-6554`:

```
function acquire_new_frame_slot(cpi):
    pool = cm->buffer_pool
    if cm->new_fb_idx != INVALID_IDX:
        --pool->frame_bufs[cm->new_fb_idx].ref_count
    cm->new_fb_idx = get_free_fb(cm)
    if cm->new_fb_idx == INVALID_IDX:
        return -1  // caller returns -1
    cm->cur_frame = &pool->frame_bufs[cm->new_fb_idx]
```

`get_free_fb` (`vp9_onyxc_int.h:332`) linearly scans `frame_bufs[]` for
the first slot with `ref_count == 0`, sets its `ref_count = 1`, returns
its index, or returns `INVALID_IDX` if all slots are full. With 12 slots
and at most 8 references plus the current frame plus up to 3 scaled
references, the pool can transiently run dry only under SVC + dynamic
resize stress.

### 4.6 Initial RTCD dispatch installation

Process-level. See §6.

---

## 5. Tables and constants

Most load-bearing tables (probability defaults, scan orders, quantizer
matrices, speed-feature dispatch) belong to other specialists. This
section indexes the few that the architecture path itself touches.

| Table | File:line | Purpose | Specialist |
|---|---|---|---|
| `vp9_level_defs[VP9_LEVELS]` | `vp9/encoder/vp9_encoder.h:450` (declaration) | VP9 level constraints | A5 (level RC) |
| `kVp9RefFlagList[4]` | `vp9/encoder/vp9_encoder.h:1295-1296` | Map `MV_REFERENCE_FRAME` → `VP9_*_FLAG` | A2 (inter selection) |
| `MAX_LAG_BUFFERS = 25` | `vp9/encoder/vp9_lookahead.h:22` | Lookahead queue depth | A5 |
| `MAX_PRE_FRAMES = 1` | `vp9/encoder/vp9_lookahead.h:33` | Past-frame retention | A5 |
| `REF_FRAMES = 8` | `vp9/common/vp9_onyxc_int.h:37-38` | Reference ring size | A4 (bitstream encodes 3-bit slot indices) |
| `REFS_PER_FRAME = 3` | `vp9/common/vp9_onyxc_int.h:35` | Per-frame ref count | A2/A3 |
| `FRAME_BUFFERS = 12` | `vp9/common/vp9_onyxc_int.h:42` | Encoder pool size | — (architecture) |
| `FRAME_CONTEXTS = 4` | `vp9/common/vp9_onyxc_int.h:44-45` | Entropy backup ring | A4 |
| `MAX_NUM_TILE_COLS = 64` | `vp9/encoder/vp9_ethread.h:18` | Tile-col cap | A6 |
| `MAX_NUM_TILE_ROWS = 4` | `vp9/encoder/vp9_ethread.h:19` | Tile-row cap | A6 |
| `MAX_NUM_THREADS = 80` | `vp9/encoder/vp9_ethread.h:20` | Worker cap | A6 |
| `TICKS_PER_SEC = 10_000_000` | `vp9/encoder/vp9_encoder.h:67` | Timebase tick | — |
| `MAX_ARF_GOP_SIZE = 2 * MAX_LAG_BUFFERS` | `vp9/encoder/vp9_encoder.h:515` | TPL GOP cap | A5 |
| `MAX_KMEANS_GROUPS = 8` | `vp9/encoder/vp9_encoder.h:516` | PERCEPTUAL_AQ groups | A5 |
| `MIN_LOOKAHEAD_FOR_ARFS = 4` | `vp9/encoder/vp9_encoder.h:1422` | Altref gating | A5 |
| BFP function-pointer table | `vp9/encoder/vp9_encoder.c:2664-2724` | `cpi->fn_ptr[BLOCK_SIZES]` SAD/variance dispatch | A3 (consumers); A4 (kernels) |

The BFP table is reproduced here for reference (this is **the** kernel
dispatch table the architecture wires up at encoder construction):

`third_party/libvpx/vp9/encoder/vp9_encoder.c:2664-2724`:

```c
#define BFP(BT, SDF, SDSF, SDAF, VF, SVF, SVAF, SDX4DF, SDSX4DF) \
  cpi->fn_ptr[BT].sdf = SDF;                                     \
  cpi->fn_ptr[BT].sdsf = SDSF;                                   \
  cpi->fn_ptr[BT].sdaf = SDAF;                                   \
  cpi->fn_ptr[BT].vf = VF;                                       \
  cpi->fn_ptr[BT].svf = SVF;                                     \
  cpi->fn_ptr[BT].svaf = SVAF;                                   \
  cpi->fn_ptr[BT].sdx4df = SDX4DF;                               \
  cpi->fn_ptr[BT].sdsx4df = SDSX4DF;

BFP(BLOCK_32X16, vpx_sad32x16, vpx_sad_skip_32x16, vpx_sad32x16_avg,
    vpx_variance32x16, vpx_sub_pixel_variance32x16,
    vpx_sub_pixel_avg_variance32x16, vpx_sad32x16x4d, vpx_sad_skip_32x16x4d)

BFP(BLOCK_16X32, vpx_sad16x32, vpx_sad_skip_16x32, vpx_sad16x32_avg,
    vpx_variance16x32, vpx_sub_pixel_variance16x32,
    vpx_sub_pixel_avg_variance16x32, vpx_sad16x32x4d, vpx_sad_skip_16x32x4d)

BFP(BLOCK_64X32, vpx_sad64x32, vpx_sad_skip_64x32, vpx_sad64x32_avg,
    vpx_variance64x32, vpx_sub_pixel_variance64x32,
    vpx_sub_pixel_avg_variance64x32, vpx_sad64x32x4d, vpx_sad_skip_64x32x4d)

BFP(BLOCK_32X64, vpx_sad32x64, vpx_sad_skip_32x64, vpx_sad32x64_avg,
    vpx_variance32x64, vpx_sub_pixel_variance32x64,
    vpx_sub_pixel_avg_variance32x64, vpx_sad32x64x4d, vpx_sad_skip_32x64x4d)

BFP(BLOCK_32X32, vpx_sad32x32, vpx_sad_skip_32x32, vpx_sad32x32_avg,
    vpx_variance32x32, vpx_sub_pixel_variance32x32,
    vpx_sub_pixel_avg_variance32x32, vpx_sad32x32x4d, vpx_sad_skip_32x32x4d)

BFP(BLOCK_64X64, vpx_sad64x64, vpx_sad_skip_64x64, vpx_sad64x64_avg,
    vpx_variance64x64, vpx_sub_pixel_variance64x64,
    vpx_sub_pixel_avg_variance64x64, vpx_sad64x64x4d, vpx_sad_skip_64x64x4d)

BFP(BLOCK_16X16, vpx_sad16x16, vpx_sad_skip_16x16, vpx_sad16x16_avg,
    vpx_variance16x16, vpx_sub_pixel_variance16x16,
    vpx_sub_pixel_avg_variance16x16, vpx_sad16x16x4d, vpx_sad_skip_16x16x4d)

BFP(BLOCK_16X8, vpx_sad16x8, vpx_sad_skip_16x8, vpx_sad16x8_avg,
    vpx_variance16x8, vpx_sub_pixel_variance16x8,
    vpx_sub_pixel_avg_variance16x8, vpx_sad16x8x4d, vpx_sad_skip_16x8x4d)

BFP(BLOCK_8X16, vpx_sad8x16, vpx_sad_skip_8x16, vpx_sad8x16_avg,
    vpx_variance8x16, vpx_sub_pixel_variance8x16,
    vpx_sub_pixel_avg_variance8x16, vpx_sad8x16x4d, vpx_sad_skip_8x16x4d)

BFP(BLOCK_8X8, vpx_sad8x8, vpx_sad_skip_8x8, vpx_sad8x8_avg, vpx_variance8x8,
    vpx_sub_pixel_variance8x8, vpx_sub_pixel_avg_variance8x8, vpx_sad8x8x4d,
    vpx_sad_skip_8x8x4d)

BFP(BLOCK_8X4, vpx_sad8x4, vpx_sad_skip_8x4, vpx_sad8x4_avg, vpx_variance8x4,
    vpx_sub_pixel_variance8x4, vpx_sub_pixel_avg_variance8x4, vpx_sad8x4x4d,
    vpx_sad_skip_8x4x4d)

BFP(BLOCK_4X8, vpx_sad4x8, vpx_sad_skip_4x8, vpx_sad4x8_avg, vpx_variance4x8,
    vpx_sub_pixel_variance4x8, vpx_sub_pixel_avg_variance4x8, vpx_sad4x8x4d,
    vpx_sad_skip_4x8x4d)

BFP(BLOCK_4X4, vpx_sad4x4, vpx_sad_skip_4x4, vpx_sad4x4_avg, vpx_variance4x4,
    vpx_sub_pixel_variance4x4, vpx_sub_pixel_avg_variance4x4, vpx_sad4x4x4d,
    vpx_sad_skip_4x4x4d)
```

Each `vpx_*` name on the right resolves at boot time, via the `vpx_dsp_rtcd`
dispatch, to one of:

- A scalar C implementation (always present).
- An SSE2 / SSSE3 / SSE4.1 / AVX2 implementation (x86_64).
- A NEON implementation (aarch64).

See §6 for the dispatch mechanism.

---

## 6. RTCD / SIMD surfaces at the architecture level

This document covers only how SIMD dispatch is **initialised at encoder
boot**; the per-kernel SIMD inventories live with their respective
specialists.

### 6.1 RTCD = "Runtime Code Dispatch"

libvpx uses a Perl-generated table of function pointers, one per kernel.
At process start, each table is initialised by a `setup_rtcd_internal()`
function generated from `third_party/libvpx/build/make/rtcd.pl`.

`vp9_initialize_enc` (`vp9_encoder.c:929-942`) calls three such tables in
sequence:

```c
static void initialize_enc(void) {
  vp9_rtcd();
  vpx_dsp_rtcd();
  vpx_scale_rtcd();
  vp9_init_intra_predictors();
  vp9_init_me_luts();
  vp9_rc_init_minq_luts();
  vp9_entropy_mv_init();
#if !CONFIG_REALTIME_ONLY
  vp9_temporal_filter_init();
#endif
}
```

- `vp9_rtcd()` (`vp9/common/vp9_rtcd_defs.pl`): VP9-specific kernels
  (forward DCT/ADST, encoder-only stuff).
- `vpx_dsp_rtcd()` (`vpx_dsp/vpx_dsp_rtcd_defs.pl`): the bulk of
  SIMD-able kernels — inverse transforms, intra predictors, subpel filters,
  SAD/variance/sub-pixel variance, sum-of-squares, convolution, loop
  filter, and so on. **The BFP table in §5 binds to symbols populated by
  this call.**
- `vpx_scale_rtcd()` (`vpx_scale/vpx_scale_rtcd.pl`): frame resampling
  kernels (used for SVC downscaling and dynamic resize).

Each of these is generated by `rtcd.pl`. On x86 the generated function is:

`third_party/libvpx/build/make/rtcd.pl:264-277`:

```perl
print <<EOF;
#ifdef RTCD_C
#include "vpx_ports/x86.h"
static void setup_rtcd_internal(void)
{
    int flags = x86_simd_caps();

    (void)flags;

EOF

  set_function_pointers("c", @ALL_ARCHS);

  print <<EOF;
}
#endif
EOF
```

`x86_simd_caps()` (in `vpx_ports/x86.h`) returns a bitmask of CPU
capabilities detected at runtime via CPUID. The generated body then
assigns:

```c
table_entry = best_available_impl(flags);
```

for each kernel, in order of preference (e.g. AVX2 → SSE4.1 → SSSE3 → SSE2
→ C scalar).

On aarch64, `setup_rtcd_internal` uses `arm_cpu_caps()` (`rtcd.pl:281-313`)
to select NEON vs. C scalar.

The `vpx_rtcd_h()` macro in `rtcd.pl` ensures the table is initialised
exactly once per process via the `once()` wrapper.

### 6.2 Implications for the Rust port (no decisions here — flagged in §7)

- The Rust port needs an equivalent dispatch mechanism. Decision Q2 in
  `PLAN.md` covers this.
- The architecture-layer requirement is small: one entry point that
  populates dispatch tables exactly once before the first encode call.
  Everything downstream (the per-block-size BFP table, the per-tx-size
  forward transforms in `cpi->td.mb.fwd_txfm4x4`, etc.) reads from those
  tables but does not care how they are populated.

### 6.3 Per-frame function-pointer wiring

Beyond process-level RTCD, the encoder programs additional function
pointers per frame:

- `cpi->row_mt_sync_read_ptr` / `_write_ptr` — flip between real and
  dummy at `vp9_encodeframe.c:6181-6193`.
- `cpi->td.mb.fwd_txfm4x4`, `cpi->td.mb.inv_txfm_add`,
  `cpi->td.mb.highbd_inv_txfm_add` — flip between lossless and lossy at
  `vp9_encodeframe.c:6086-6095` and again at `vp9_encoder.c:6649-6660`
  for the first-pass.
- `cpi->find_fractional_mv_step` — set per-frame by speed features (A3).
- `cpi->diamond_search_sad` — set per-frame by speed features (A3).

These are not architectural decisions; they are leaf-level wiring done by
the subsystems that own the kernel.

---

## 7. Open questions for the LLD writer

The architecture analysis surfaces the following decisions that must be
made (or deferred) in Stage 2. Each is flagged as either **arch-level**
(this doc's owner is the LLD writer for the encoder shell) or
**subsystem** (will be resolved by another specialist's LLD).

### 7.1 Frame-buffer pool ownership and reference counting (arch-level)

- libvpx uses `BufferPool` with 12 `RefCntBuffer` slots and explicit
  `ref_count` ints (`vp9_onyxc_int.h:66-97`). The pool is allocated
  externally to `VP9_COMP` (in `cx_iface.c:1126`) so that frame
  parallelism can share it. **Decision needed: do we expose the pool to
  the public API of `rvp9-encoder`, or hide it behind a single
  `Encoder` type?** For real-time only, the encoder owns the pool
  exclusively, so hiding it inside is simpler. But if we later add
  frame-parallel encode (PLAN.md Q4), we need the pool first.
- The pool's reference-counting semantics map cleanly to `Arc<FrameBuf>`
  or `Rc<FrameBuf>`. **Decision needed: `Arc` (`Send`) or `Rc`
  (single-thread)?** Workers do not retain reference-frame buffers; they
  read through `cpi->common.frame_refs[3]` and the per-tile shadow.
  `Arc` is the safe default; the per-frame "pin" of the new fb is what
  matters for the encode pipeline.

### 7.2 `VP9_COMP` god-struct decomposition (arch-level)

- `VP9_COMP` has ~120 fields spanning 8+ subsystems. A direct port
  would produce a 2000-line struct that's hostile to maintenance.
- **Decision needed:** decompose along subsystem boundaries
  (`Encoder { common: Common, config: Config, rc: RateControl,
  speed: SpeedFeatures, svc: Option<Svc>, threading: ThreadPool,
  ... }`) or keep a flat mega-struct? Decomposition complicates the
  worker-data copy step (libvpx does `thread_data->td->mb = cpi->td.mb`
  at `vp9_ethread.c:209-211`).
- Recommend: decompose by subsystem, with worker-shared state behind
  `Arc` and worker-private state owned. The shared/owned boundary is
  exactly the field-by-field semantics in §2.4.

### 7.3 Lookahead with `lag_in_frames = 0` (arch-level / A5)

- Real-time CBR commonly uses lag 0 (or 1). The `lookahead_ctx` still
  exists but has `max_sz = 1`. The push-then-pop pattern is
  unnecessary in this case.
- **Decision needed:** keep a thin lookahead even for lag 0 (mirrors
  libvpx and keeps SVC's `vp9_svc_lookahead_pop` path uniform) or
  bypass it (saves a memcpy of the YV12 source per frame)? Specialist
  **A5** has a stronger opinion here.

### 7.4 `setjmp`/`longjmp` panic path (arch-level)

- libvpx uses `setjmp(cm->error.jmp)` for OOM / pathological input
  paths. Rust has `Result<T, EncoderError>` and panics.
- **Decision needed:** map every `vpx_internal_error` call site to a
  `?`-propagating `Result` chain (the safe Rust path), or keep an
  internal `panic = "abort"` policy and treat memory allocation as
  infallible? This isn't free: there are ~30 `CHECK_MEM_ERROR` sites in
  `vp9_create_compressor` alone. We're in a sandbox per
  `CLAUDE.md`; we can use `try_reserve` / `Result` since memory failure
  recovery isn't a release goal — but it complicates the API.

### 7.5 Initial worker-pool sizing (arch-level / A6)

- libvpx creates workers lazily on first use of `vp9_encode_tiles_*`.
  The pool size can change across calls (SVC layer-by-layer with
  different `max_threads`).
- **Decision needed:** create workers eagerly at `Encoder::new()` or
  lazily? Eager is simpler and avoids the first-frame latency spike;
  lazy mirrors libvpx exactly. Real-time has a strict p99 budget so
  the first-frame init cost matters. Specialist **A6** likely owns
  this.

### 7.6 Mode dispatch (`GOOD` vs. `REALTIME` switch) (arch-level)

- libvpx switches between `GOOD` and `REALTIME` modes on every encode
  based on the user's deadline (`pick_quickcompress_mode`,
  `vp9_cx_iface.c:1170-1212`). The switch calls `vp9_change_config`,
  which can reallocate buffers if dimensions changed but mostly is
  cheap.
- **Decision needed:** support runtime mode switching in v1 or pin
  the encoder to `REALTIME` at construction? PLAN.md says cpu-used
  6-8 only, which is real-time exclusive. Pin to `REALTIME` and
  simplify the LLD.

### 7.7 External rate-control plug-in (`EXT_RATECTRL`) (subsystem A5)

- libvpx supports a user-supplied rate-controller via
  `cpi->ext_ratectrl` (`vp9_ext_ratectrl.h`). WebRTC uses this heavily.
- **Decision needed:** in scope for v1 or v1.1? PLAN.md doesn't mention
  it. Flag for **A5**.

### 7.8 `CONFIG_RATE_CTRL` / `CONFIG_NON_GREEDY_MV` / `CONFIG_RATE_CTRL` testing knobs (arch-level)

- libvpx has ~7 build-time `#ifdef`s that conditionally include large
  blocks of code (testing tools, optional internal stats, simple-encode
  API, non-greedy MV, KF denoise, etc.). Real-time disables most.
- **Decision needed:** drop all of these in the port (real-time only)
  or carry them under Cargo features? Most are dead code for our use
  case. Drop, but record in `divergence-ledger.md`.

### 7.9 Recode loop disposition (arch-level)

- The recode loop (`encode_with_recode_loop`) is the non-RT path. Its
  body retries the entire frame encode at adjusted Q on rate overshoot.
  Real-time uses only the single-pass `encode_without_recode_loop`.
- **Decision needed:** confirm v1 ships without `encode_with_recode_loop`.
  PLAN.md D1 says cpu-used 6-8 only ⇒ no recode loop needed. Confirmed.

### 7.10 Bitstream packing threading (subsystem A4)

- libvpx has an optional parallel bitstream-pack path
  (`cpi->vp9_bitstream_worker_data`). Real-time RTC defaults to the
  serial path because packing is cheap and parallelizing it adds sync
  overhead.
- **Decision needed:** scalar pack only in v1 (recommended) or
  parallel? Flag for **A4**.

### 7.11 Reference-slot management for SVC (subsystem A6)

- The 8-slot reference map is shared across spatial layers and
  temporal layers. libvpx's mapping in `vp9_one_pass_svc_start_layer`
  is the authoritative source. Specialist **A6** owns this.

### 7.12 `cm->buffer_pool` cross-thread safety (arch-level)

- During encode, the main thread mutates `cm->ref_frame_map[]` and
  `cm->new_fb_idx`; worker threads only **read** `cm->frame_refs[3]`
  (set up before launch). The aliasing invariant is "no worker modifies
  the reference map".
- **Decision needed:** model this as `&Common` (shared, immutable
  during encode) plus a small `&mut FramePoolWriter` held by the main
  thread, or use `UnsafeCell<Common>` and rely on the launch / join
  fence semantics? Specialist **A6** will need to resolve this with
  the synchronization decision (PLAN.md Q3).

---

## Cross-references to other specialists

| Cited subsystem | Owner | Key entry points |
|---|---|---|
| Mode decision (real-time non-RD) | **A2** | `vp9_pickmode.c`, `vp9_nonrd_pick_inter_mode_sb`, `set_rt_speed_feature_*` |
| Motion estimation | **A3** | `vp9_mcomp.c`, `vp9_find_best_sub_pixel_tree*`, `init_motion_estimation` |
| Bitstream / entropy / tables | **A4** | `vp9_bitstream.c`, `vp9_pack_bitstream`, `vp9_entropymode.c`, `vp9_subexp.c`, `vp9_adapt_*_probs` |
| Rate control / denoiser / AQ | **A5** | `vp9_ratectrl.c`, `vp9_rc_get_one_pass_cbr_params`, `vp9_rc_drop_frame`, `vp9_denoiser.c`, `vp9_aq_*.c`, `vp9_cyclic_refresh.c`, `vp9_scene_detection_onepass`, `vp9_compute_frame_low_motion` |
| Threading internals / SVC / speed features | **A6** | `vp9_ethread.c`, `vp9_multi_thread.c`, `vp9_thread_common.{h,c}`, `vp9_svc_layercontext.c`, `vp9_speed_features.c` |

---

*Document version: 0.1.0 — initial draft.*
