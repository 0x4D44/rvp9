# Threading, SVC, and Speed Features in libvpx VP9

Specialist A6, Stage 1. Reference: `third_party/libvpx/` at v1.14.1 commit `12f3a2a`. All `file:line` citations are relative to that submodule unless otherwise stated, and all paths are absolute under `D:\language\rvp9-2\` in the working tree.

## 1. Scope

This document is the deep-dive on the three interconnected subsystems that the rvp9 port treats as one layer:

1. **Tile-based threading model** in the VP9 encoder (`vp9_ethread.{h,c}`, `vp9_multi_thread.{h,c}`, `vpx_thread.{h,c}`, `vp9_thread_common.{h,c}`). Plus the decoder row-MT path (`vp9_decodeframe.c`) for completeness.
2. **Scalable Video Coding** — temporal and spatial — driven by `SVC` and `LAYER_CONTEXT` (`vp9_svc_layercontext.{h,c}`).
3. **Speed-features dispatch** — `vp9_speed_features.{h,c}` — specifically `set_rt_speed_feature_framesize_independent` / `set_rt_speed_feature_framesize_dependent`, which define what `--cpu-used 5..8` means.

Out of scope (cited where relevant, not described): top-level lifecycle (A1), `vp9_pickmode.c` heuristics (A2), motion-search internals (A3), bitstream layout (A4), RC algorithm internals (A5 — but per-layer RC plumbing is covered here).

A note on "frame-parallel encoding": libvpx VP9's **encoder is single-frame-in-flight**. The `cm->frame_parallel_decoding_mode` bit (`vp9/common/vp9_onyxc_int.h:257`, written in the uncompressed header at `vp9/encoder/vp9_bitstream.c:1304`) is a *decoder-side* hint that disables backward probability updates so independent decoders can decode frames in parallel; it does not run the encoder in parallel. `cpi->oxcf.lag_in_frames` provides a *look-ahead* buffer (for first-pass stats, ARF placement, source-SAD scene-cut detection) but the encode pipeline still processes one frame at a time. The locked design decision D6 ("mirror libvpx tile-based pthread model") therefore concerns the **intra-frame** parallelism: tile-MT and row-MT within a frame. Q4 ("frame-parallel encoding in v1 vs v2") remains a deferred question with no libvpx precedent on the encode side.

## 2. Data structures (verbatim)

### 2.1 `EncWorkerData` — per-worker thread state (`vp9/encoder/vp9_ethread.h:25`)

```c
typedef struct EncWorkerData {
  struct VP9_COMP *cpi;
  struct ThreadData *td;
  int start;
  int thread_id;
  int tile_completion_status[MAX_NUM_TILE_COLS];
} EncWorkerData;
```

Fields:

- `cpi` — back-pointer to the global encoder (`VP9_COMP`). **Read-mostly** during a frame; tile workers read it but only mutate fields documented as per-worker (`td->mb`, `td->counts`, `td->rd_counts`) and the worker's assigned slice of `cpi->tile_data[]`.
- `td` — pointer to per-worker `ThreadData` (see §2.4). For worker index `< num_workers - 1` this is a separately allocated 32-byte-aligned object; for the last worker (the main thread) it aliases `&cpi->td` (`vp9_ethread.c:131-135`).
- `start` — initial tile index for this worker in tile-MT mode. Worker `i` processes tiles `t = i, i + num_workers, i + 2*num_workers, ...` (`vp9_ethread.c:50-56`).
- `thread_id` — 0..num_workers-1, set in `vp9_prepare_job_queue` (`vp9_multi_thread.c:302`).
- `tile_completion_status` — per-tile-column completion flags used in row-MT job-stealing (`vp9_multi_thread.c:318`, `vp9_ethread.c:441-442,530-531,606-607`).

Limits: `MAX_NUM_TILE_COLS = 1 << 6 = 64`, `MAX_NUM_TILE_ROWS = 4`, `MAX_NUM_THREADS = 80` (`vp9_ethread.h:18-20`).

### 2.2 `VP9RowMTSync` — row-based sync (`vp9/encoder/vp9_ethread.h:34`)

```c
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

Per-row `mutex`/`cond` (one per SB row) protect `cur_col[r]`, the **furthest SB column** worker `r` has finished. Reader on row `r` (at SB column `c`) blocks until `cur_col[r-1] >= c - nsync + 1` (see `vp9_row_mt_sync_read`, `vp9_ethread.c:348-366`). `sync_range = 1` (hard-coded for encoder, `vp9_ethread.c:318`) — one column behind. `rows` is the number of jobs per tile column, equal to `mb_rows` for first-pass and `sb_rows` for encode (`vp9_multi_thread.c:243-251`).

Note: a single `VP9RowMTSync` object is allocated for the first tile column and **shared** by every tile row in that column (`vp9_multi_thread.c:113-119`). This is critical: it means workers across tile rows but same tile column share the same `cur_col[]` array.

### 2.3 `MultiThreadHandle` — frame-level row-MT context (`vp9/encoder/vp9_encoder.h:361`)

```c
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
```

- `job_queue` — flat array of `JobQueue` nodes, `total_jobs = jobs_per_tile_col * tile_cols`, 32-byte-aligned (`vp9_multi_thread.c:95-96`).
- `row_mt_info[tile_col]` — per-tile-column queue head + mutex (see §2.5).
- `thread_id_to_tile_id` — round-robin assignment: thread `i` is initially attached to tile column `i mod tile_cols` (`vp9_multi_thread.c:204-207`).

### 2.4 `RowMTInfo`, `JobNode`, `JobQueue`, `JobQueueHandle` (`vp9/encoder/vp9_encoder.h:348`, `vp9/encoder/vp9_job_queue.h:22-44`)

```c
typedef struct RowMTInfo {
  JobQueueHandle job_queue_hdl;
#if CONFIG_MULTITHREAD
  pthread_mutex_t job_mutex;
#endif
} RowMTInfo;

typedef enum {
  FIRST_PASS_JOB,
  ENCODE_JOB,
  ARNR_JOB,
  NUM_JOB_TYPES,
} JOB_TYPE;

// Encode job parameters
typedef struct {
  int vert_unit_row_num;  // Index of the vertical unit row
  int tile_col_id;        // tile col id within a tile
  int tile_row_id;        // tile col id within a tile
} JobNode;

// Job queue element parameters
typedef struct {
  // Pointer to the next link in the job queue
  void *next;

  // Job information context of the module
  JobNode job_info;
} JobQueue;

// Job queue handle
typedef struct {
  // Pointer to the next link in the job queue
  void *next;

  // Counter to store the number of jobs picked up for processing
  int num_jobs_acquired;
} JobQueueHandle;
```

`JobQueue` is a forward-linked list spliced over the flat `job_queue` array allocated in `MultiThreadHandle`. Each tile column has its own queue head (`row_mt_info[tile_col].job_queue_hdl`) protected by its own `job_mutex` (`vp9_multi_thread.c:18-52`). When a worker drains its assigned column it queries other columns for work (the work-stealing loop in `enc_row_mt_worker_hook`, `vp9_ethread.c:600-617`).

### 2.5 `ThreadData` — per-worker mutable encode state (`vp9/encoder/vp9_encoder.h:384`)

```c
typedef struct ThreadData {
  MACROBLOCK mb;
  RD_COUNTS rd_counts;
  FRAME_COUNTS *counts;

  PICK_MODE_CONTEXT *leaf_tree;
  PC_TREE *pc_tree;
  PC_TREE *pc_root;
} ThreadData;
```

This is the **thread-local arena** for tile-MT and row-MT. Allocated 32-byte-aligned (`vp9_ethread.c:113-115`). On entry to each frame, `cpi->td.mb` is *copy-assigned* into every worker's `td->mb` (`vp9_ethread.c:209-212`, `vp9_ethread.c:649-652`) — that is, MACROBLOCK is value-copied; subsequent in-frame mutations are confined to the worker's copy. Same for `rd_counts`. For `counts`, the global `cm->counts` is memcpy'd into each worker's `td->counts`; on frame exit the deltas are merged back via `vp9_accumulate_frame_counts(&cm->counts, thread_data->td->counts, 0)` (`vp9_ethread.c:243-247`, `vp9_ethread.c:680-689`).

`pc_tree` / `leaf_tree` are heap-allocated partition-tree scratch (allocated in `vp9_setup_pc_tree` called from `create_enc_workers`, `vp9_ethread.c:121`).

The last worker's `td` points to `&cpi->td` itself (`vp9_ethread.c:134`), so the main thread's `cpi->td` doubles as worker N-1's data. This is the *aliasing contract*: every other worker's `ThreadData` is owned by `cpi->tile_thr_data[i].td` (heap-allocated, distinct from `cpi->td`), but worker N-1 shares storage with `cpi->td`. Counts and rd_counts accumulation in `vp9_encode_tiles_mt` skips index N-1 because that worker already mutated `cpi->td` in place (`vp9_ethread.c:243`).

### 2.6 `VPxWorker` and the worker interface (`vpx_util/vpx_thread.h:387-420`)

```c
// State of the worker thread object
typedef enum {
  NOT_OK = 0,  // object is unusable
  OK,          // ready to work
  WORK         // busy finishing the current task
} VPxWorkerStatus;

// Function to be called by the worker thread.
typedef int (*VPxWorkerHook)(void *, void *);

// Platform-dependent implementation details for the worker.
typedef struct VPxWorkerImpl VPxWorkerImpl;

// Synchronization object used to launch job in the worker thread
typedef struct {
  VPxWorkerImpl *impl_;
  VPxWorkerStatus status_;
  VPxWorkerHook hook;  // hook to call
  void *data1;         // first argument passed to 'hook'
  void *data2;         // second argument passed to 'hook'
  int had_error;       // return value of the last call to 'hook'
} VPxWorker;

// The interface for all thread-worker related functions.
typedef struct {
  void (*init)(VPxWorker *const worker);
  int (*reset)(VPxWorker *const worker);
  int (*sync)(VPxWorker *const worker);
  void (*launch)(VPxWorker *const worker);
  void (*execute)(VPxWorker *const worker);
  void (*end)(VPxWorker *const worker);
} VPxWorkerInterface;
```

`VPxWorkerImpl` is defined in `vpx_util/vpx_thread.c:22`:

```c
struct VPxWorkerImpl {
  pthread_mutex_t mutex_;
  pthread_cond_t condition_;
  pthread_t thread_;
};
```

The worker model is a **persistent thread + state-machine** (NOT_OK → OK → WORK → OK …) signalled with a mutex+cond pair (`vpx_util/vpx_thread.c:32-73`). The thread spins in `thread_loop`: lock mutex, wait on cond while `status_ == OK`, on `WORK` call `worker->hook(data1, data2)`, transition to OK, signal cond, unlock. The main thread launches by setting status to WORK and signalling; sync waits for OK.

`reset()` creates the thread once and stores it in `impl_->thread_`. Subsequent frame encodes just toggle the worker through OK → WORK → OK without re-spawning. The Windows pthread shim is in `vpx_util/vpx_thread.h:30-213` (uses `_beginthreadex` or `CreateThread`, and either `CONDITION_VARIABLE` for Vista+ or a semaphore-based emulation for XP).

### 2.7 `VP9LfSync` — loopfilter row sync (`vp9/common/vp9_thread_common.h:25`)

```c
// Loopfilter row synchronization
typedef struct VP9LfSyncData {
#if CONFIG_MULTITHREAD
  pthread_mutex_t *mutex;
  pthread_cond_t *cond;
#endif
  // Allocate memory to store the loop-filtered superblock index in each row.
  int *cur_sb_col;
  // The optimal sync_range for different resolution and platform should be
  // determined by testing. Currently, it is chosen to be a power-of-2 number.
  int sync_range;
  int rows;

  // Row-based parallel loopfilter data
  LFWorkerData *lfdata;
  int num_workers;         // number of allocated workers.
  int num_active_workers;  // number of scheduled workers.

#if CONFIG_MULTITHREAD
  pthread_mutex_t *lf_mutex;
  pthread_mutex_t *recon_done_mutex;
  pthread_cond_t *recon_done_cond;
#endif
  int *num_tiles_done;
  int corrupted;
} VP9LfSync;
```

Used by both encoder loopfilter (post-encode) and decoder loopfilter. Decoder row-MT additionally uses `recon_done_mutex/cond` so that loopfilter jobs are queued only once every tile of a given SB row has finished recon (`vp9/decoder/vp9_decodeframe.c:1731-1750`).

### 2.8 `SVC` — top-level scalable-coding state (`vp9/encoder/vp9_svc_layercontext.h:80`)

```c
typedef struct SVC {
  int spatial_layer_id;
  int temporal_layer_id;
  int number_spatial_layers;
  int number_temporal_layers;

  int spatial_layer_to_encode;

  // Workaround for multiple frame contexts
  enum { ENCODED = 0, ENCODING, NEED_TO_ENCODE } encode_empty_frame_state;
  struct lookahead_entry empty_frame;
  int encode_intra_empty_frame;

  // Store scaled source frames to be used for temporal filter to generate
  // a alt ref frame.
  YV12_BUFFER_CONFIG scaled_frames[MAX_LAG_BUFFERS];
  // Temp buffer used for 2-stage down-sampling, for real-time mode.
  YV12_BUFFER_CONFIG scaled_temp;
  int scaled_one_half;
  int scaled_temp_is_alloc;

  // Layer context used for rate control in one pass temporal CBR mode or
  // two pass spatial mode.
  LAYER_CONTEXT layer_context[VPX_MAX_LAYERS];
  // Indicates what sort of temporal layering is used.
  // Currently, this only works for CBR mode.
  VP9E_TEMPORAL_LAYERING_MODE temporal_layering_mode;
  // Frame flags and buffer indexes for each spatial layer, set by the
  // application (external settings).
  int ext_frame_flags[VPX_MAX_LAYERS];
  int lst_fb_idx[VPX_MAX_LAYERS];
  int gld_fb_idx[VPX_MAX_LAYERS];
  int alt_fb_idx[VPX_MAX_LAYERS];
  int force_zero_mode_spatial_ref;
  // Sequence level flag to enable second (long term) temporal reference.
  int use_gf_temporal_ref;
  // Frame level flag to enable second (long term) temporal reference.
  int use_gf_temporal_ref_current_layer;
  // Allow second reference for at most 2 top highest resolution layers.
  BUFFER_LONGTERM_REF buffer_gf_temporal_ref[2];
  int current_superframe;
  int non_reference_frame;
  int use_base_mv;
  int use_partition_reuse;
  // Used to control the downscaling filter for source scaling, for 1 pass CBR.
  INTERP_FILTER downsample_filter_type[VPX_SS_MAX_LAYERS];
  int downsample_filter_phase[VPX_SS_MAX_LAYERS];

  BLOCK_SIZE *prev_partition_svc;
  int mi_stride[VPX_MAX_LAYERS];
  int mi_rows[VPX_MAX_LAYERS];
  int mi_cols[VPX_MAX_LAYERS];

  int first_layer_denoise;

  int skip_enhancement_layer;

  int lower_layer_qindex;

  int last_layer_dropped[VPX_MAX_LAYERS];
  int drop_spatial_layer[VPX_MAX_LAYERS];
  int framedrop_thresh[VPX_MAX_LAYERS];
  int drop_count[VPX_MAX_LAYERS];
  int force_drop_constrained_from_above[VPX_MAX_LAYERS];
  int max_consec_drop;
  SVC_LAYER_DROP_MODE framedrop_mode;

  INTER_LAYER_PRED disable_inter_layer_pred;

  int high_source_sad_superframe;
  int high_num_blocks_with_motion;

  // Flags used to get SVC pattern info.
  int update_buffer_slot[VPX_SS_MAX_LAYERS];
  uint8_t reference_last[VPX_SS_MAX_LAYERS];
  uint8_t reference_golden[VPX_SS_MAX_LAYERS];
  uint8_t reference_altref[VPX_SS_MAX_LAYERS];
  // TODO(jianj): Remove these last 3, deprecated.
  uint8_t update_last[VPX_SS_MAX_LAYERS];
  uint8_t update_golden[VPX_SS_MAX_LAYERS];
  uint8_t update_altref[VPX_SS_MAX_LAYERS];

  // Keep track of the frame buffer index updated/refreshed on the base
  // temporal superframe.
  int fb_idx_upd_tl0[VPX_SS_MAX_LAYERS];

  // Keep track of the spatial and temporal layer id of the frame that last
  // updated the frame buffer index.
  uint8_t fb_idx_spatial_layer_id[REF_FRAMES];
  uint8_t fb_idx_temporal_layer_id[REF_FRAMES];

  int spatial_layer_sync[VPX_SS_MAX_LAYERS];
  // Quantizer for each spatial layer.
  int base_qindex[VPX_SS_MAX_LAYERS];
  uint8_t set_intra_only_frame;
  uint8_t previous_frame_is_intra_only;
  uint8_t superframe_has_layer_sync;

  uint8_t fb_idx_base[REF_FRAMES];

  int use_set_ref_frame_config;

  int temporal_layer_id_per_spatial[VPX_SS_MAX_LAYERS];

  int first_spatial_layer_to_encode;

  // Parameters for allowing framerate per spatial layer, and buffer
  // update based on timestamps.
  int64_t duration[VPX_SS_MAX_LAYERS];
  int64_t timebase_fac;
  int64_t time_stamp_superframe;
  int64_t time_stamp_prev[VPX_SS_MAX_LAYERS];

  int num_encoded_top_layer;

  // Every spatial layer on a superframe whose base is key is key too.
  int simulcast_mode;

  // Flag to indicate SVC is dynamically switched to a single layer.
  int single_layer_svc;
  int resize_set;
} SVC;
```

Layer-array layout: `layer_context[sl * number_temporal_layers + tl]`. Macro `LAYER_IDS_TO_IDX(sl, tl, num_tl)` in `vp9/encoder/vp9_encoder.h:1495` is `(sl) * (num_tl) + (tl)`. `VPX_MAX_LAYERS = 12`, `VPX_SS_MAX_LAYERS = 5`, `VPX_TS_MAX_LAYERS = 5` (`vpx/vpx_encoder.h:42-48`).

### 2.9 `LAYER_CONTEXT` — per-layer encoder state (`vp9/encoder/vp9_svc_layercontext.h:40`)

```c
typedef struct {
  RATE_CONTROL rc;
  int target_bandwidth;
  int spatial_layer_target_bandwidth;  // Target for the spatial layer.
  double framerate;
  int avg_frame_size;
  int max_q;
  int min_q;
  int scaling_factor_num;
  int scaling_factor_den;
  // Scaling factors used for internal resize scaling for single layer SVC.
  int scaling_factor_num_resize;
  int scaling_factor_den_resize;
  TWO_PASS twopass;
  vpx_fixed_buf_t rc_twopass_stats_in;
  unsigned int current_video_frame_in_layer;
  int is_key_frame;
  int frames_from_key_frame;
  FRAME_TYPE last_frame_type;
  struct lookahead_entry *alt_ref_source;
  int alt_ref_idx;
  int gold_ref_idx;
  int has_alt_frame;
  size_t layer_size;
  // Cyclic refresh parameters (aq-mode=3), that need to be updated per-frame.
  // TODO(jianj/marpan): Is it better to use the full cyclic refresh struct.
  int sb_index;
  signed char *map;
  uint8_t *last_coded_q_map;
  uint8_t *consec_zero_mv;
  int actual_num_seg1_blocks;
  int actual_num_seg2_blocks;
  int counter_encode_maxq_scene_change;
  int qindex_delta[3];
  uint8_t speed;
  int loopfilter_ctrl;
  int frame_qp;
  int MBs;
} LAYER_CONTEXT;
```

The `LAYER_CONTEXT` is essentially a snapshot of all the `VP9_COMP` per-frame state that varies *per layer*. Each spatial layer encode begins with `vp9_restore_layer_context` (`vp9_svc_layercontext.c:352-389`) which swaps `cpi->rc = lc->rc`, `cpi->twopass = lc->twopass`, plus cyclic-refresh maps; and ends with `vp9_save_layer_context` (`vp9_svc_layercontext.c:391-424`) which writes back. (A5's territory describes the contents of `RATE_CONTROL`; this doc covers the layered-restore mechanic.)

Lifetime: the cyclic-refresh maps (`map`, `last_coded_q_map`, `consec_zero_mv`) are allocated only when `ss_number_layers > 1 && tl == 0` (`vp9_svc_layercontext.c:158-179`) — i.e. one set per spatial layer's base temporal layer.

### 2.10 `BUFFER_LONGTERM_REF`, `INTER_LAYER_PRED`, `VP9E_TEMPORAL_LAYERING_MODE`

```c
// vp9/encoder/vp9_svc_layercontext.h:22
typedef enum {
  INTER_LAYER_PRED_ON,
  INTER_LAYER_PRED_OFF,
  INTER_LAYER_PRED_OFF_NONKEY,
  INTER_LAYER_PRED_ON_CONSTRAINED
} INTER_LAYER_PRED;

typedef struct BUFFER_LONGTERM_REF {
  int idx;
  int is_used;
} BUFFER_LONGTERM_REF;

// vpx/vp8cx.h:790
typedef enum vp9e_temporal_layering_mode {
  VP9E_TEMPORAL_LAYERING_MODE_NOLAYERING = 0,
  VP9E_TEMPORAL_LAYERING_MODE_BYPASS = 1,
  VP9E_TEMPORAL_LAYERING_MODE_0101 = 2,
  VP9E_TEMPORAL_LAYERING_MODE_0212 = 3
} VP9E_TEMPORAL_LAYERING_MODE;
```

Default initial settings in `vp9_init_layer_context` (`vp9_svc_layercontext.c:43-86`): `disable_inter_layer_pred = INTER_LAYER_PRED_ON`, `framedrop_mode = CONSTRAINED_LAYER_DROP`, golden-temporal-ref buffer slots `6` and `7`. The default `lst_fb_idx[sl]=0, gld_fb_idx[sl]=1, alt_fb_idx[sl]=2` and `fb_idx_upd_tl0[sl] = -1` are stamped per-spatial-layer.

### 2.11 `SPEED_FEATURES` (`vp9/encoder/vp9_speed_features.h:271`)

Sub-structs first:

```c
// vp9/encoder/vp9_speed_features.h:181
typedef struct MV_SPEED_FEATURES {
  SEARCH_METHODS search_method;
  int reduce_first_step_size;
  int auto_mv_step_size;
  SUBPEL_SEARCH_METHODS subpel_search_method;
  int subpel_search_level;
  SUBPEL_FORCE_STOP subpel_force_stop;
  int enable_adaptive_subpel_force_stop;
  ADAPT_SUBPEL_FORCE_STOP adapt_subpel_force_stop;
  int fullpel_search_step_param;
  int use_downsampled_sad;
} MV_SPEED_FEATURES;

// vp9/encoder/vp9_speed_features.h:170
typedef struct ADAPT_SUBPEL_FORCE_STOP {
  int mv_thresh;
  SUBPEL_FORCE_STOP force_stop_below;
  SUBPEL_FORCE_STOP force_stop_above;
} ADAPT_SUBPEL_FORCE_STOP;

// vp9/encoder/vp9_speed_features.h:219
typedef struct PARTITION_SEARCH_BREAKOUT_THR {
  int64_t dist;
  int rate;
} PARTITION_SEARCH_BREAKOUT_THR;

#define MAX_MESH_STEP 4
typedef struct MESH_PATTERN {
  int range;
  int interval;
} MESH_PATTERN;

// vp9/encoder/vp9_speed_features.h:266
typedef struct TRELLIS_OPT_CONTROL {
  ENABLE_TRELLIS_OPT_METHOD method;
  double thresh;
} TRELLIS_OPT_CONTROL;
```

Enums (selected; see header for full list):

```c
// vp9/encoder/vp9_speed_features.h:53
typedef enum {
  DIAMOND = 0, NSTEP = 1, HEX = 2, BIGDIA = 3, SQUARE = 4,
  FAST_HEX = 5, FAST_DIAMOND = 6, MESH = 7
} SEARCH_METHODS;

// vp9/encoder/vp9_speed_features.h:77
typedef enum {
  SUBPEL_TREE = 0,
  SUBPEL_TREE_PRUNED = 1,
  SUBPEL_TREE_PRUNED_MORE = 2,
  SUBPEL_TREE_PRUNED_EVENMORE = 3,
} SUBPEL_SEARCH_METHODS;

// vp9/encoder/vp9_speed_features.h:168
typedef enum { EIGHTH_PEL, QUARTER_PEL, HALF_PEL, FULL_PEL } SUBPEL_FORCE_STOP;

// vp9/encoder/vp9_speed_features.h:90
typedef enum {
  USE_FULL_RD = 0, USE_LARGESTALL, USE_TX_8X8
} TX_SIZE_SEARCH_METHOD;

// vp9/encoder/vp9_speed_features.h:138
typedef enum {
  SEARCH_PARTITION, FIXED_PARTITION, REFERENCE_PARTITION,
  VAR_BASED_PARTITION, SOURCE_VAR_BASED_PARTITION, ML_BASED_PARTITION
} PARTITION_SEARCH_TYPE;

// vp9/encoder/vp9_speed_features.h:102
typedef enum {
  LPF_PICK_FROM_FULL_IMAGE, LPF_PICK_FROM_SUBIMAGE,
  LPF_PICK_FROM_Q, LPF_PICK_MINIMAL_LPF
} LPF_PICK_METHOD;
```

And the full struct (`vp9/encoder/vp9_speed_features.h:271-661`):

```c
typedef struct SPEED_FEATURES {
  MV_SPEED_FEATURES mv;

  // Frame level coding parameter update
  int frame_parameter_update;

  RECODE_LOOP_TYPE recode_loop;

  // Trellis (dynamic programming) optimization of quantized values (+1, 0).
  int optimize_coefficients;

  int static_segmentation;

  int comp_inter_joint_search_iter_level;

  int adaptive_rd_thresh;

  // Flag to use adaptive_rd_thresh when row-mt it enabled, only for non-rd
  // pickmode.
  int adaptive_rd_thresh_row_mt;

  int skip_encode_sb;
  int skip_encode_frame;
  int allow_skip_recode;

  int coeff_prob_appx_step;

  TRELLIS_OPT_CONTROL trellis_opt_tx_rd;

  int allow_acl;

  int enable_tpl_model;

  int allow_txfm_domain_distortion;
  double tx_domain_thresh;

  MOTION_THRESHOLD lf_motion_threshold;

  TX_SIZE_SEARCH_METHOD tx_size_search_method;

  int tx_size_search_depth;

  int use_lp32x32fdct;

  int mode_skip_start;

  int reference_masking;

  PARTITION_SEARCH_TYPE partition_search_type;
  BLOCK_SIZE always_this_block_size;

  int less_rectangular_check;

  int use_square_partition_only;
  BLOCK_SIZE use_square_only_thresh_high;
  BLOCK_SIZE use_square_only_thresh_low;

  int prune_ref_frame_for_rect_partitions;

  AUTO_MIN_MAX_MODE auto_min_max_partition_size;
  BLOCK_SIZE rd_auto_partition_min_limit;

  BLOCK_SIZE default_min_partition_size;
  BLOCK_SIZE default_max_partition_size;

  int adjust_partitioning_from_last_frame;

  int last_partitioning_redo_frequency;

  int disable_split_mask;

  int adaptive_motion_search;
  int enhanced_full_pixel_motion_search;

  int exhaustive_searches_thresh;
  MESH_PATTERN mesh_patterns[MAX_MESH_STEP];

  int schedule_mode_search;

  int adaptive_pred_interp_filter;

  int adaptive_mode_search;

  int prune_single_mode_based_on_mv_diff_mode_rate;

  int cb_pred_filter_search;

  int early_term_interp_search_plane_rd;

  int cb_partition_search;

  int motion_field_mode_search;

  int alt_ref_search_fp;

  int use_quant_fp;

  int force_frame_boost;

  int max_delta_qindex;

  unsigned int mode_search_skip_flags;

  unsigned int disable_filter_search_var_thresh;

  int intra_y_mode_mask[TX_SIZES];
  int intra_uv_mode_mask[TX_SIZES];

  int intra_y_mode_bsize_mask[BLOCK_SIZES];

  int use_rd_breakout;

  int use_uv_intra_rd_estimate;

  LPF_PICK_METHOD lpf_pick;

  FAST_COEFF_UPDATE use_fast_coef_updates;

  int use_nonrd_pick_mode;

  int inter_mode_mask[BLOCK_SIZES];

  int use_fast_coef_costing;

  int recode_tolerance_low;
  int recode_tolerance_high;

  BLOCK_SIZE max_intra_bsize;

  int search_type_check_frequency;

  int reuse_inter_pred_sby;

  int encode_breakout_thresh;

  INTERP_FILTER default_interp_filter;

  int tx_size_search_breakout;

  int adaptive_interp_filter_search;

  INTERP_FILTER_MASK interp_filter_search_mask;

  PARTITION_SEARCH_BREAKOUT_THR partition_search_breakout_thr;

  struct {
    int search_breakout;
    float search_breakout_thresh[3];
    int search_early_termination;
    int var_pruning;
    int prune_rect_thresh[4];
  } rd_ml_partition;

  int simple_model_rd_from_var;

  int short_circuit_flat_blocks;

  int short_circuit_low_temp_var;

  int limit_newmv_early_exit;

  int bias_golden;

  int base_mv_aggressive;

  int copy_partition_flag;

  int use_source_sad;

  int use_simple_block_yrd;

  int adapt_partition_source_sad;
  int adapt_partition_thresh;

  int use_altref_onepass;

  int use_compound_nonrd_pickmode;

  int nonrd_keyframe;

  int svc_use_lowres_part;

  OVERSHOOT_DETECTION_CBR_RT overshoot_detection_cbr_rt;

  int disable_16x16part_nonkey;

  int disable_golden_ref;

  SUBPEL_SEARCH_TYPE use_accurate_subpel_search;

  SEARCH_METHODS temporal_filter_search_method;

  int nonrd_use_ml_partition;

  int variance_part_thresh_mult;

  int force_smooth_interpol;

  int rt_intra_dc_only_low_content;

  int allow_skip_txfm_ac_dc;
} SPEED_FEATURES;
```

(Comments stripped for brevity; full doc-comments are in the header. The exhaustive ordering of fields here matters for memory layout if rvp9 mirrors libvpx's `vp9_zero` initialization.)

## 3. Control flow

### 3.1 Encoder threading dispatch decision

In `encode_frame_to_data_rate` → `encode_with_recode_loop` → (`encode_encode_frame` →) `encode_frame_internal` (`vp9/encoder/vp9_encodeframe.c:6177-6198`):

```text
if (!cpi->row_mt) {
    row_mt_sync_read_ptr = vp9_row_mt_sync_read_dummy
    row_mt_sync_write_ptr = vp9_row_mt_sync_write_dummy
    if (max_threads >= 2 && tile_cols > 1)
        vp9_encode_tiles_mt(cpi)        // one thread per tile column
    else
        encode_tiles(cpi)               // single-threaded fallback
} else {
    row_mt_sync_read_ptr = vp9_row_mt_sync_read
    row_mt_sync_write_ptr = vp9_row_mt_sync_write
    vp9_encode_tiles_row_mt(cpi)        // row-MT with work-stealing job queue
}
```

The `row_mt_sync_read_ptr` / `row_mt_sync_write_ptr` function pointers are invoked from the partition loop at the start and end of each superblock-column step within an SB row (`vp9_encodeframe.c:4526-4527, 4599-4600` for the RD path; `vp9_encodeframe.c:5556-5557, 5703-5704` for the non-RD/RT path; `vp9/encoder/vp9_firstpass.c:962, 1361` for first-pass). In single-thread / tile-MT modes they expand to the dummy no-ops, so the same partition loop runs unmodified.

### 3.2 Thread lifecycle (`create_enc_workers`, `vp9_ethread.c:80-138`)

```text
on first encode with target num_workers != cpi->num_workers:
    vp9_loop_filter_dealloc(...)
    vp9_bitstream_encode_tiles_buffer_dealloc(...)
    vp9_encode_free_mt_data(...)
    allocate cpi->workers[num_workers] (VPxWorker array)
    allocate cpi->tile_thr_data[num_workers] (EncWorkerData array)
    for i in 0..num_workers-1:
        winterface->init(&workers[i])
        if i < num_workers - 1:
            allocate td (32-byte aligned)
            vp9_zero(*td)
            vp9_setup_pc_tree(cm, td)
            allocate td->counts
            winterface->reset(&workers[i])   // spawns pthread, status->OK
        else:
            // last worker is main thread; td aliases cpi->td
            thread_data->td = &cpi->td
        winterface->sync(&workers[i])
```

Once created, threads persist for the lifetime of `cpi`. A change in `num_workers` (driven by SVC resolution change or user reconfig) tears down and re-spawns.

For SVC + non-row-MT, the number of workers is computed from the **highest-resolution layer's tile count** (`vp9_ethread.c:84-90`):

```text
if cpi->use_svc && !cpi->row_mt:
    num_workers = min(max_threads, get_max_tile_cols(cpi))
```

This ensures the worker pool can saturate the top spatial layer's tile parallelism without re-creating threads when the encoder shrinks to a smaller layer.

### 3.3 Tile-MT launch (`vp9_encode_tiles_mt`, `vp9_ethread.c:194-248`)

```text
vp9_init_tile_data(cpi)                        // §3.5
create_enc_workers(cpi, num_workers=min(max_threads, tile_cols))
for each worker i:
    if td != &cpi->td:
        td->mb = cpi->td.mb                    // value-copy MACROBLOCK
        td->rd_counts = cpi->td.rd_counts
    if td->counts != &cm->counts:
        memcpy(td->counts, &cm->counts, sizeof(counts))
    if sf.use_nonrd_pick_mode:
        re-wire MACROBLOCK plane coeff/qcoeff/dqcoeff pointers to ctx->coeff_pbuf[j][0]
launch_enc_workers(cpi, enc_worker_hook, NULL, num_workers)
    for each worker i:
        worker->hook = enc_worker_hook
        worker->data1 = &tile_thr_data[i]
        thread_data->start = i
        if i == num_workers-1: winterface->execute(worker)   // run inline
        else: winterface->launch(worker)
    for each worker: winterface->sync(worker)
for each non-last worker:
    vp9_accumulate_frame_counts(&cm->counts, td->counts, 0)
    accumulate_rd_opt(&cpi->td, td)
```

`enc_worker_hook` (`vp9_ethread.c:40-59`) is the tile-column striding loop:

```c
for (t = thread_data->start; t < tile_rows * tile_cols; t += cpi->num_workers) {
    int tile_row = t / tile_cols;
    int tile_col = t % tile_cols;
    vp9_encode_tile(cpi, thread_data->td, tile_row, tile_col);
}
```

So worker `i` claims tiles `i, i+N, i+2N, ...` (where `N = num_workers`). Since `tile_rows == 1` in real-time mode, this reduces to disjoint tile columns.

### 3.4 Row-MT launch (`vp9_encode_tiles_row_mt`, `vp9_ethread.c:620-690`)

```text
if MultiThreadHandle allocation insufficient: vp9_row_mt_mem_dealloc; vp9_init_tile_data; vp9_row_mt_mem_alloc
else: vp9_init_tile_data
create_enc_workers(cpi, max(max_threads, 1))
vp9_assign_tile_to_thread(multi_thread_ctxt, tile_cols, num_workers)
    // round-robin: thread_id_to_tile_id[i] = i % tile_cols
vp9_prepare_job_queue(cpi, ENCODE_JOB)
    // builds linked list of JobNodes per tile column, jobs_per_tile_col = sb_rows
    // ENCODE_JOB also advances tile_row when num_tile_vert_sbs[tile_row] consumed
vp9_multi_thread_tile_init(cpi)
    // reset cur_col[*] = -1, zero fp_data
for each worker: copy td->mb = cpi->td.mb; td->rd_counts = ...; memcpy(td->counts, ...)
launch_enc_workers(cpi, enc_row_mt_worker_hook, multi_thread_ctxt, num_workers)
for each non-last worker: vp9_accumulate_frame_counts; accumulate_rd_opt
```

`enc_row_mt_worker_hook` (`vp9_ethread.c:586-618`) is the work-stealing loop:

```c
while (!end_of_frame) {
    proc_job = vp9_enc_grp_get_next_job(multi_thread_ctxt, cur_tile_id);
    if (proc_job == NULL) {
        // current tile's queue is empty; find another tile with most work
        end_of_frame = vp9_get_tiles_proc_status(
            multi_thread_ctxt, thread_data->tile_completion_status,
            &cur_tile_id, tile_cols);
    } else {
        tile_col = proc_job->tile_col_id;
        tile_row = proc_job->tile_row_id;
        mi_row   = proc_job->vert_unit_row_num * MI_BLOCK_SIZE;
        vp9_encode_sb_row(cpi, thread_data->td, tile_row, tile_col, mi_row);
    }
}
```

`vp9_enc_grp_get_next_job` (`vp9_multi_thread.c:18-52`) pops the head of `row_mt_info[tile_col].job_queue_hdl` under `job_mutex`.

`vp9_get_tiles_proc_status` (`vp9_multi_thread.c:309-341`) marks the current tile complete in the worker's `tile_completion_status[]`, scans all tiles for the one with the most remaining jobs, returns that as the new `cur_tile_id`, or returns end-of-frame if every tile is done.

### 3.5 `vp9_init_tile_data` and tile bookkeeping

`vp9_init_tile_data` (`vp9_encodeframe.c:5844-5904`):

```text
if !cpi->tile_data or allocated_tiles < tile_cols*tile_rows:
    vp9_row_mt_mem_dealloc; free tile_data
    alloc tile_data[tile_cols * tile_rows]
    for each (row, col):
        thresh_freq_fact[i][j] = RD_THRESH_INIT_FACT       // for adaptive_rd_thresh
        mode_map[i][j] = j
        row_base_thresh_freq_fact = NULL
for each (row, col):
    vp9_tile_init(&tile_info, cm, row, col)
        // mi_row_start/end and mi_col_start/end via get_tile_offset
        // tile width = ((idx+1) * sb_cols >> log2) << MI_BLOCK_SIZE_LOG2 - prev
    cpi->tile_tok[row][col] = pre_tok + tile_tok          // token storage
    cpi->tplist[row][col]   = tplist + tplist_count       // token-list bookkeeping
```

Tile-column boundaries: `get_tile_offset` (`vp9/common/vp9_tile_common.c:18-22`) rounds to SB boundary:

```c
static int get_tile_offset(int idx, int mis, int log2) {
  const int sb_cols = mi_cols_aligned_to_sb(mis) >> MI_BLOCK_SIZE_LOG2;
  const int offset = ((idx * sb_cols) >> log2) << MI_BLOCK_SIZE_LOG2;
  return VPXMIN(offset, mis);
}
```

So tile boundaries are SB64-aligned. `MIN_TILE_WIDTH_B64 = 4` (256 px) and `MAX_TILE_WIDTH_B64 = 64` (4096 px) constrain the legal `log2_tile_cols` values (`vp9/common/vp9_tile_common.c:15-16, 39-49`).

### 3.6 Row-MT sync primitives (`vp9_ethread.c:289-413`)

```c
// Read: worker at (row r, col c) waits for row r-1 to have advanced
// at least to column c - nsync + 1 (nsync == sync_range == 1 for encoder).
void vp9_row_mt_sync_read(VP9RowMTSync *const row_mt_sync, int r, int c) {
  const int nsync = row_mt_sync->sync_range;
  if (r && !(c & (nsync - 1))) {                       // nsync==1 → !(c & 0) == true
    pthread_mutex_t *const mutex = &row_mt_sync->mutex[r - 1];
    pthread_mutex_lock(mutex);
    while (c > row_mt_sync->cur_col[r - 1] - nsync + 1)
      pthread_cond_wait(&row_mt_sync->cond[r - 1], mutex);
    pthread_mutex_unlock(mutex);
  }
}

// Write: signals row r reaches column c. When c reaches cols-1, write
// cols + nsync to mark "row done" for any subsequent waiters.
void vp9_row_mt_sync_write(VP9RowMTSync *const row_mt_sync, int r, int c,
                           const int cols) {
  const int nsync = row_mt_sync->sync_range;
  int cur, sig = 1;
  if (c < cols - 1) {
    cur = c;
    if (c % nsync != nsync - 1) sig = 0;
  } else {
    cur = cols + nsync;
  }
  if (sig) {
    pthread_mutex_lock(&row_mt_sync->mutex[r]);
    row_mt_sync->cur_col[r] = cur;
    pthread_cond_signal(&row_mt_sync->cond[r]);
    pthread_mutex_unlock(&row_mt_sync->mutex[r]);
  }
}
```

The contract: before encoding the SB at `(r, c)` a worker waits until the SB at `(r-1, c)` is complete (because intra prediction and the `above_context` ENTROPY_CONTEXT bytes for the current SB row are written by the SB above it). After encoding, it advances `cur_col[r]` and wakes anyone waiting.

The dummy variants (`vp9_row_mt_sync_read_dummy`, `vp9_row_mt_sync_write_dummy`) are no-ops; they are installed when row-MT is disabled (`vp9_encodeframe.c:6182-6183`).

### 3.7 Frame-level encode flow for SVC (one-pass CBR / RT path)

In `vp9_encode` (around the inner loop in `vp9_encoder.c:6383`):

```text
on every input frame (superframe):
    for sl in first_spatial_layer_to_encode..number_spatial_layers-1:
        cpi->svc.spatial_layer_to_encode = sl
        vp9_one_pass_svc_start_layer(cpi)
            // sets svc->simulcast_mode based on disable_inter_layer_pred
            // calls set_flags_and_fb_idx_for_temporal_mode{2,3,_noLayering,bypass}
            // sets cpi->lst_fb_idx, gld_fb_idx, alt_fb_idx, ref_frame_flags
            //   based on temporal_layering_mode
            // checks long-term GF temporal-ref enable
            // updates resolution via vp9_set_size_literal
        vp9_restore_layer_context(cpi)
            // cpi->rc = lc->rc; cpi->twopass = lc->twopass
            // swap cyclic_refresh maps if ss>1 && tl==0
        encode_frame_to_data_rate(cpi)
            // includes set_size_independent_vars → vp9_set_speed_features_*
        vp9_save_layer_context(cpi)
        vp9_inc_frame_in_layer(cpi)
            // ++lc.current_video_frame_in_layer
            // ++svc.current_superframe iff sl == number_spatial_layers - 1
    if (show_frame):
        ++spatial_layer_to_encode mod number_spatial_layers   // for next call
```

The outermost spatial-layer loop is driven by the calling application or the `simple_encode` shim; rate control and frame allocation are layered on top by A5.

### 3.8 Inter-layer prediction control (`vp9_svc_constrain_inter_layer_pred`, `vp9_svc_layercontext.c:1076-1147`)

For modes `INTER_LAYER_PRED_OFF`, `INTER_LAYER_PRED_OFF_NONKEY`, or when the previous spatial layer was dropped:

```text
for each reference frame ref ∈ {LAST, GOLDEN, ALTREF}:
    if scale_factors(ref) is scaled (cross-resolution → inter-layer):
        clear ref_frame_flags & ref-flag
        if not simulcast_mode:
            redirect GOLDEN/ALTREF buffer index to lst_fb_idx
```

For `INTER_LAYER_PRED_ON_CONSTRAINED`: only allow inter-layer references that point to a buffer that was refreshed by the immediately-previous spatial layer in the same superframe (checked via `svc->lst_fb_idx[sl-1]`, `gld_fb_idx[sl-1]`, `alt_fb_idx[sl-1]` AND a corresponding bit in `update_buffer_slot[sl-1]`).

### 3.9 Speed-feature dispatch on encoder init / re-init

`set_size_independent_vars` and `set_size_dependent_vars` (`vp9/encoder/vp9_encoder.c:3749-3774`) are called twice per encoded frame in the recode path (`vp9_encoder.c:4140-4141` and `4521-4536`):

```text
set_size_independent_vars(cpi):
    vp9_set_speed_features_framesize_independent(cpi, oxcf->speed)
    vp9_set_rd_speed_thresholds(cpi)
    vp9_set_rd_speed_thresholds_sub8x8(cpi)
    cm->interp_filter = sf.default_interp_filter

set_size_dependent_vars(cpi, &q, &btm, &top):
    vp9_set_speed_features_framesize_dependent(cpi, oxcf->speed)
    *q = vp9_rc_pick_q_and_bounds(cpi, &btm, &top)
    if VPX_CBR && rc.force_max_q: *q = worst_quality
    if use_svc: svc.base_qindex[spatial_layer_id] = *q
```

Both speed-feature setters (`vp9_speed_features.c:869-913` and `915-1093`) first zero the default ("best quality") values, then call either `set_good_*` or `set_rt_*` depending on `oxcf->mode`. The RT setters are the ones we care about (D1: cpu-used 6-8).

Importantly: speed features are recomputed **every frame** based on the current frame's `cm->frame_type`, `cm->base_qindex`, source pixel statistics, and the SVC state. This means setting cpu-used to 6 does not "lock in" a fixed `SPEED_FEATURES`; flags toggle within bounded ranges per frame.

## 4. Algorithms

### 4.1 Tile-MT worker sync protocol

Two concurrent constraints govern tile-MT encoding:

1. **Tile boundary aliasing.** VP9 tiles are decodable independently *only* for entropy and certain prediction state: each tile clears `xd->left_context` and `xd->left_seg_context` at the start of every SB row (`vp9_encodeframe.c:4508-4509, 5541-5542`). Tiles are SB-column-aligned, so different tile workers write disjoint slices of `cm->above_context[]` and `cm->above_seg_context[]` (allocated as a frame-level flat buffer in `vp9_alloc_context_buffers`). No cross-tile-column synchronization is needed for above-context.

2. **Loop filter at tile boundaries.** VP9 loopfilter crosses tile boundaries (unlike VP8's tile-decodable boundary). The encoder defers loopfilter to a post-encode stage with its own MT path (`VP9LfSync`, `vp9/common/vp9_thread_common.c`), which operates row-by-row and synchronizes on `cur_sb_col[]` like row-MT.

The tile-MT protocol therefore reduces to: launch N workers, each runs `enc_worker_hook` until its strided set of `(tile_row, tile_col)` is exhausted, then `sync()` from the main thread. The shared writes are:

- `cm->above_context` / `cm->above_seg_context` — disjoint per tile column.
- `cm->mi_grid_visible[mi_row * mi_stride + mi_col]` (the MODE_INFO grid) — disjoint per tile column.
- `cm->seg.last_frame_seg_map` — written at the end of frame encoding, not during tile encode.
- `cm->buffer_pool->frame_bufs[...].buf.y_buffer/u_buffer/v_buffer` — disjoint per tile column for the new frame; reference frames are read-only during tile encode.

Per-worker (no aliasing):

- `td->mb` (MACROBLOCK and nested MACROBLOCKD).
- `td->rd_counts`.
- `td->counts` (memcpy'd in, accumulated out).
- `td->pc_tree`, `td->leaf_tree`, `td->pc_root`.
- `tile_thr_data[i].tile_completion_status[]`.
- `tile_data[tile_row * tile_cols + tile_col]` slice (each worker only touches its own tile's `thresh_freq_fact`, `mode_map`, `fp_data`, `row_mt_sync`, `row_base_thresh_freq_fact`).

Shared-but-read-only during the frame: `cpi->oxcf`, `cpi->sf`, `cm->frame_refs[]`, reference YV12 buffers, probability tables (`cm->fc`), the `cpi->Source` pointer.

### 4.2 Row-MT worker sync protocol

Row-MT introduces a 2D dependency: every SB at `(mi_row, mi_col)` depends on the SB above `(mi_row - MI_BLOCK_SIZE, mi_col)` having finished writing the `above_context` slice covering its width. This is enforced by `vp9_row_mt_sync_read`/`_write` wrapping the partition call (`vp9_encodeframe.c:5556, 5703` for the RT non-RD path).

Sequence per SB encode at `(sb_row, sb_col_in_tile)`:

```text
vp9_row_mt_sync_read(&row_mt_sync, sb_row, sb_col_in_tile)
    // block until cur_col[sb_row - 1] >= sb_col_in_tile (because nsync == 1)
encode SB
    // reads above_context[mi_col_global..]
    // writes above_context[mi_col_global..] for next row
    // writes mi_grid_visible[mi_row..]
    // writes recon pixels in new_fb
vp9_row_mt_sync_write(&row_mt_sync, sb_row, sb_col_in_tile, num_sb_cols)
    // cur_col[sb_row] = sb_col_in_tile; signal cond[sb_row]
```

For multi-tile-row frames, the same `row_mt_sync` struct is shared across all tile rows in the same tile column (`vp9_multi_thread.c:113-119`), so tile-row N's first SB row blocks on tile-row N-1's last SB row.

Job ordering: `vp9_prepare_job_queue` for `ENCODE_JOB` (`vp9_multi_thread.c:262-297`) walks rows in scan order within a tile column. For multi-tile-row frames, after `num_tile_vert_sbs[tile_row] - 1` jobs in the current tile row, it bumps `tile_row++` and resets the per-tile row counter, so each per-column queue contains all rows of all tile rows for that column, in order. Work-stealing across columns is handled by `vp9_get_tiles_proc_status`.

### 4.3 Decoder threading (for completeness)

Three decode modes (`vp9/decoder/vp9_decodeframe.c:3022-3043`):

1. `decode_tiles_row_wise_mt` — row-MT with `recon_map` + parse/recon job queue (`vp9_decodeframe.c:2325-2440`).
2. `decode_tiles_mt` — one thread per tile column (`vp9_decodeframe.c:2442+`), parse-and-recon in sequence, runs `tile_worker_hook`.
3. `decode_tiles` — single-threaded fallback.

`tile_worker_hook` (`vp9_decodeframe.c:2182-2270`) per-tile loop:

```text
setjmp(error_info.jmp)                    // bitstream error recovery
do {
    vp9_tile_init(tile, cm, 0, buf->col)
    setup_token_decoder(...)
    vp9_init_macroblockd(...)
    for mi_row in tile.mi_row_start..mi_row_end step MI_BLOCK_SIZE:
        zero xd.left_context, xd.left_seg_context
        for mi_col in tile.mi_col_start..mi_col_end step MI_BLOCK_SIZE:
            decode_partition(tile_data, pbi, mi_row, mi_col, BLOCK_64X64, 4)
        if lpf_mt_opt: vp9_set_row(lf_sync, tile_cols, sb_row, is_last_row, corrupted)
} while (++n <= buf_end && !corrupted)
if (lpf_mt_opt): vp9_loopfilter_rows(lf_data, lf_sync)
```

Row-MT decoder (`row_decode_worker_hook`, `vp9_decodeframe.c:1894+`) splits work into PARSE_JOB / RECON_JOB / LPF_JOB pushed onto `JobQueueRowMt`. PARSE_JOB writes coefficients into `RowMTWorkerData.eob[plane]` and `dqcoeff[plane]` arrays of size `sb_rows * sb_cols * EOBS_PER_SB` / `DQCOEFFS_PER_SB`. RECON_JOB reads those and writes pixels; it waits for the **above SB to be recon'd** via `map_read(recon_map, ...)` (`vp9_decodeframe.c:1713-1729`). `map_write` (`vp9_decodeframe.c:1699-1711`) signals completion.

### 4.4 SVC inter-layer prediction flow (per superframe)

For temporal mode 0212 (3-layer):

```text
frame_num_within_temporal_struct = current_video_frame_in_layer % 4
temporal_id = (frame_num & 1) ? 2 : (frame_num >> 1)
    // sequence: 0 → tl=0; 1 → tl=2; 2 → tl=1; 3 → tl=2; (repeat)
```

For temporal mode 0101 (2-layer): `temporal_id = current_video_frame_in_layer & 1`.

For each spatial layer `sl ∈ 0..number_spatial_layers-1`, encode in order. Within a superframe (`vp9_svc_layercontext.c:517-684` for the three patterns):

- **tl == 0** (base temporal):
  - `lst_fb_idx = sl` → refresh slot `sl` with this frame.
  - `gld_fb_idx = sl - 1` if `sl > 0` (predict from lower spatial layer base) or `0` if `sl == 0`.
  - `alt_fb_idx = 0`.
  - On non-`sl=0` non-key: refresh LAST only, reference from LAST + GOLDEN.
  - On `sl > 0` and base spatial is key: refresh GOLDEN (the upper-layer key prediction).
- **tl == 1** (mid-temporal, in 0212 mode):
  - `lst_fb_idx = sl`.
  - `gld_fb_idx = number_spatial_layers + sl - 1`.
  - `alt_fb_idx = number_spatial_layers + sl` — refresh ALTREF.
- **tl == 2** (top-temporal, non-reference for the next TL=0 frame):
  - First TL2 picture: `lst_fb_idx = sl`, `gld_fb_idx = NSL+sl-1`, `alt_fb_idx = NSL+sl`; top spatial layer doesn't refresh.
  - Second TL2 picture: `lst_fb_idx = NSL+sl` (the just-refreshed ALTREF), `gld_fb_idx = NSL+sl-1`, `alt_fb_idx = NSL+sl`.

Where `NSL = number_spatial_layers`. The buffer-pool slot layout for SVC 3-spatial × 3-temporal therefore lives in slots `[0, 1, ..., 2*NSL - 1]`, plus slots `[6, 7]` reserved for long-term GF temporal references (`vp9_svc_layercontext.c:83-86`).

Bitstream metadata: each layer's frame is written as an independent VP9 frame in a superframe (the superframe index marker is in `vp9_bitstream.c`; A4's territory). The layer IDs `(spatial_layer_id, temporal_layer_id)` are not in the VP9 bitstream proper; they are extracted by the application via the `vpx_svc_ref_frame_config_t` API and the `cpi->common.uniform_quad_tree` / `cpi->common.frame_parallel_decoding_mode` are flagged appropriately. SVC layer headers (when used) are an external metadata convention, not a syntax element.

### 4.5 Speed-feature dispatch table — Real-time, cpu-used 5..=8

The full table is below in §5. The setup proceeds in two phases:

1. `vp9_set_speed_features_framesize_independent` (`vp9_speed_features.c:915-1093`) — defaults plus speed-stage modifiers; dispatches via `oxcf->mode == REALTIME ⇒ set_rt_speed_feature_framesize_independent`.
2. `vp9_set_speed_features_framesize_dependent` (`vp9_speed_features.c:869-913`) — frame-size overrides; dispatches via `oxcf->mode == REALTIME ⇒ set_rt_speed_feature_framesize_dependent`.

`set_rt_speed_feature_framesize_independent` (`vp9_speed_features.c:452-867`) is structured as a cumulative cascade `speed >= 1`, `speed >= 2`, ..., `speed >= 9`. Each block builds on the previous. Speed 5 first enables `use_nonrd_pick_mode = 1`; this is the qualitative split between RD-loop and the real-time non-RD path.

## 5. Tables and constants — VERBATIM

### 5.1 Tile-MT and row-MT constants

```c
// vp9/encoder/vp9_ethread.h:18
#define MAX_NUM_TILE_COLS (1 << 6)
#define MAX_NUM_TILE_ROWS 4
#define MAX_NUM_THREADS 80

// vpx_util/vpx_thread.h:26
#define MAX_DECODE_THREADS 8

// vp9/common/vp9_tile_common.c:15
#define MIN_TILE_WIDTH_B64 4    // 4 SB64s = 256 px
#define MAX_TILE_WIDTH_B64 64   // 64 SB64s = 4096 px

// vp9/encoder/vp9_ethread.c:318
row_mt_sync->sync_range = 1;    // 1 SB column behind, encoder
```

Decoder/loopfilter LF sync_range (`vp9/common/vp9_thread_common.c:265-276`):

```c
static INLINE int get_sync_range(int width) {
  // nsync numbers are picked by testing. For example, for 4k
  // video, using 4 gives best performance.
  if (width < 640)        return 1;
  else if (width <= 1280) return 2;
  else if (width <= 4096) return 4;
  else                    return 8;
}
```

### 5.2 SVC limits (`vpx/vpx_encoder.h:42-48`)

```c
#define VPX_TS_MAX_LAYERS 5
#define VPX_TS_MAX_PERIODICITY 16
#define VPX_MAX_LAYERS 12  // 3 temporal + 4 spatial layers are allowed.
#define VPX_SS_MAX_LAYERS 5
```

```c
// vp9/encoder/vp9_encoder.h:1495
#define LAYER_IDS_TO_IDX(sl, tl, num_tl) ((sl) * (num_tl) + (tl))
```

### 5.3 SVC temporal layering modes (verbatim)

```c
// vpx/vp8cx.h:790
typedef enum vp9e_temporal_layering_mode {
  VP9E_TEMPORAL_LAYERING_MODE_NOLAYERING = 0,
  VP9E_TEMPORAL_LAYERING_MODE_BYPASS = 1,
  VP9E_TEMPORAL_LAYERING_MODE_0101 = 2,
  VP9E_TEMPORAL_LAYERING_MODE_0212 = 3
} VP9E_TEMPORAL_LAYERING_MODE;
```

Mapping in `vp9_one_pass_svc_start_layer` (`vp9_svc_layercontext.c:819-830`):

| `temporal_layering_mode` | Dispatcher |
|---|---|
| `MODE_0212` (3) | `set_flags_and_fb_idx_for_temporal_mode3` (`vp9_svc_layercontext.c:517-621`) |
| `MODE_NOLAYERING` (0) | `set_flags_and_fb_idx_for_temporal_mode_noLayering` (`vp9_svc_layercontext.c:687-720`) |
| `MODE_0101` (2) | `set_flags_and_fb_idx_for_temporal_mode2` (`vp9_svc_layercontext.c:626-684`) |
| `MODE_BYPASS` (1) + `use_set_ref_frame_config` | `set_flags_and_fb_idx_bypass_via_set_ref_frame_config` (`vp9_svc_layercontext.c:722-738`) |

### 5.4 Temporal layering reference-structure tables

Below: for each (`temporal_id`, `spatial_id`) cell, the columns are `(ref_frame_flags, ext_refresh_last, ext_refresh_golden, ext_refresh_alt_ref, lst_fb_idx, gld_fb_idx, alt_fb_idx)` per `set_flags_and_fb_idx_for_temporal_mode2` and `_mode3`. Citations point to `vp9_svc_layercontext.c`. `NSL = number_spatial_layers`. `K` = "base spatial is key" predicate.

#### 5.4.1 `MODE_0101` (2-layer temporal, period 2)

`frame_num_within_temporal_struct = current_video_frame_in_layer & 1`
`temporal_id = frame_num_within_temporal_struct`
Pattern: `tl = 0, 1, 0, 1, ...`

| tl | sl | ref_flags | refresh_last | refresh_golden | refresh_alt | lst_fb_idx | gld_fb_idx | alt_fb_idx |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | `LAST` | 1 | 0 | 0 | 0 | 0 | 0 |
| 0 | sl>0, K | `LAST` | 0 | 1 | 0 | sl-1 | sl | 0 |
| 0 | sl>0, !K | `LAST\|GOLD` | 1 | 0 | 0 | sl | sl-1 | 0 |
| 1 | 0 | `LAST` | 0 | 0 | 1 | 0 | NSL-1 | NSL |
| 1 | 0<sl<NSL-1 | `LAST\|GOLD` | 0 | 0 | 1 | sl | NSL+sl-1 | NSL+sl |
| 1 | sl==NSL-1 | `LAST\|GOLD` | 0 | 0 | 0 (top, non-ref) | sl | NSL+sl-1 | NSL+sl |

Source: `vp9_svc_layercontext.c:626-684`.

#### 5.4.2 `MODE_0212` (3-layer temporal, period 4)

`frame_num_within_temporal_struct = current_video_frame_in_layer % 4`
`temporal_id = (frame_num & 1) ? 2 : (frame_num >> 1)`
Pattern: `tl = 0, 2, 1, 2, 0, 2, 1, 2, ...`

| tl | sl | ref_flags | refresh_last | refresh_golden | refresh_alt | lst_fb_idx | gld_fb_idx | alt_fb_idx |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | `LAST` | 1 | 0 | 0 | 0 | 0 | 0 |
| 0 | sl>0, K | `LAST` | 0 | 1 | 0 | sl-1 | sl | 0 |
| 0 | sl>0, !K | `LAST\|GOLD` | 1 | 0 | 0 | sl | sl-1 | 0 |
| 1 | 0 | `LAST` | 0 | 0 | 1 | 0 | NSL-1 | NSL |
| 1 | sl>0 | `LAST\|GOLD` | 0 | 0 | 1 | sl | NSL+sl-1 | NSL+sl |
| 2 (first) | sl==NSL-1 | `LAST` (sl==0) / `LAST\|GOLD` | 0 | 0 | 0 (top, non-ref) | NSL+sl | NSL+sl-1 | NSL+sl |
| 2 (first) | 0 | `LAST` | 0 | 0 | 1 | sl | NSL-1 | NSL |
| 2 (first) | 0<sl<NSL-1 | `LAST\|GOLD` | 0 | 0 | 1 | sl | NSL+sl-1 | NSL+sl |
| 2 (second) | sl==NSL-1 | `LAST` (sl==0) / `LAST\|GOLD` | 0 | 0 | 0 (top, non-ref) | NSL+sl | NSL+sl-1 | NSL+sl |
| 2 (second) | 0 | `LAST` | 0 | 0 | 1 | NSL+sl | NSL+sl-1 | NSL+sl |
| 2 (second) | 0<sl<NSL-1 | `LAST\|GOLD` | 0 | 0 | 1 | NSL+sl | NSL+sl-1 | NSL+sl |

Source: `vp9_svc_layercontext.c:517-621`. "first" / "second" tl2 distinguishes `frame_num_within_temporal_struct == 1` from `== 3`.

#### 5.4.3 `MODE_NOLAYERING` (spatial-only, no temporal hierarchy)

| sl | ref_flags | refresh_last | refresh_golden | lst_fb_idx | gld_fb_idx |
|---|---|---|---|---|---|
| 0 | `LAST` | 1 | 0 | 0 | 0 |
| sl>0, K | `LAST` | 0 | 1 | sl-1 | sl |
| sl>0, !K | `LAST\|GOLD` | 1 | 0 | sl | sl-1 |

`alt_fb_idx` is unused (zeroed via `ext_refresh_alt_ref_frame = 0`). Source: `vp9_svc_layercontext.c:687-720`.

#### 5.4.4 Long-term GF temporal reference (slots 6, 7)

`buffer_gf_temporal_ref[0].idx = 6`, `buffer_gf_temporal_ref[1].idx = 7` (`vp9_svc_layercontext.c:83-86`).

`use_gf_temporal_ref_current_layer` is enabled when (`vp9_svc_layercontext.c:851-860`):
- `use_gf_temporal_ref == 1` (sequence-level),
- both `buffer_gf_temporal_ref[*].is_used` are 0 (slots free),
- `temporal_layering_mode != BYPASS`,
- `disable_inter_layer_pred != INTER_LAYER_PRED_ON`,
- `number_spatial_layers <= 3 && number_temporal_layers <= 3`,
- `spatial_layer_id >= number_spatial_layers - 2` (i.e. top or second-from-top spatial layer).

### 5.5 Spatial-layer scaling defaults (`vp9_init_layer_context`, `vp9_svc_layercontext.c:73-74`)

```c
svc->downsample_filter_type[sl] = BILINEAR;
svc->downsample_filter_phase[sl] = 8;   // centered averaging
```

Adjusted in `vp9_one_pass_svc_start_layer` (`vp9_svc_layercontext.c:928-934`):

- `width * height <= 320 * 240` → `downsample_filter_type[sl] = EIGHTTAP_SMOOTH`.
- `scaling_factor_num > (3 * scaling_factor_den) >> 2` (i.e. > 0.75x) → `downsample_filter_phase[sl] = 0`.

`use_base_mv` and `use_partition_reuse` (cross-layer mode/partition reuse on 2x downscale) are disabled unless every spatial layer's scale ratio is exactly 1/2 or unless the 3-spatial-layer ratio is 1/4-1/2-1 (`vp9_svc_layercontext.c:940-961`).

### 5.6 Speed-feature framesize-INDEPENDENT — best-quality defaults

From `vp9_set_speed_features_framesize_independent` (`vp9_speed_features.c:925-1036`), the **defaults applied before any RT or GOOD overrides**:

| Field | Default value |
|---|---|
| `frame_parameter_update` | 1 |
| `mv.search_method` | NSTEP |
| `recode_loop` | ALLOW_RECODE_FIRST |
| `mv.subpel_search_method` | SUBPEL_TREE |
| `mv.subpel_search_level` | 2 |
| `mv.subpel_force_stop` | EIGHTH_PEL |
| `optimize_coefficients` | `!is_lossless_requested(&cpi->oxcf)` |
| `mv.reduce_first_step_size` | 0 |
| `coeff_prob_appx_step` | 1 |
| `mv.auto_mv_step_size` | 0 |
| `mv.fullpel_search_step_param` | 6 |
| `mv.use_downsampled_sad` | 0 |
| `comp_inter_joint_search_iter_level` | 0 |
| `tx_size_search_method` | USE_FULL_RD |
| `use_lp32x32fdct` | 0 |
| `adaptive_motion_search` | 0 |
| `enhanced_full_pixel_motion_search` | 1 |
| `adaptive_pred_interp_filter` | 0 |
| `adaptive_mode_search` | 0 |
| `prune_single_mode_based_on_mv_diff_mode_rate` | 0 |
| `cb_pred_filter_search` | 0 |
| `early_term_interp_search_plane_rd` | 0 |
| `cb_partition_search` | 0 |
| `motion_field_mode_search` | 0 |
| `alt_ref_search_fp` | 0 |
| `use_quant_fp` | 0 |
| `reference_masking` | 0 |
| `partition_search_type` | SEARCH_PARTITION |
| `less_rectangular_check` | 0 |
| `use_square_partition_only` | 0 |
| `use_square_only_thresh_high` | BLOCK_SIZES |
| `use_square_only_thresh_low` | BLOCK_4X4 |
| `auto_min_max_partition_size` | NOT_IN_USE |
| `rd_auto_partition_min_limit` | BLOCK_4X4 |
| `default_max_partition_size` | BLOCK_64X64 |
| `default_min_partition_size` | BLOCK_4X4 |
| `adjust_partitioning_from_last_frame` | 0 |
| `last_partitioning_redo_frequency` | 4 |
| `disable_split_mask` | 0 |
| `mode_search_skip_flags` | 0 |
| `force_frame_boost` | 0 |
| `max_delta_qindex` | 0 |
| `disable_filter_search_var_thresh` | 0 |
| `adaptive_interp_filter_search` | 0 |
| `allow_txfm_domain_distortion` | 0 |
| `tx_domain_thresh` | 99.0 |
| `trellis_opt_tx_rd.method` | `optimize_coefficients ? ENABLE_TRELLIS_OPT : DISABLE_TRELLIS_OPT` |
| `trellis_opt_tx_rd.thresh` | 99.0 |
| `allow_acl` | 1 |
| `enable_tpl_model` | `oxcf->enable_tpl_model` |
| `prune_ref_frame_for_rect_partitions` | 0 |
| `temporal_filter_search_method` | MESH |
| `allow_skip_txfm_ac_dc` | 0 |
| `intra_y_mode_mask[i] for i in 0..TX_SIZES-1` | INTRA_ALL |
| `intra_uv_mode_mask[i] for i in 0..TX_SIZES-1` | INTRA_ALL |
| `use_rd_breakout` | 0 |
| `skip_encode_sb` | 0 |
| `use_uv_intra_rd_estimate` | 0 |
| `allow_skip_recode` | 0 |
| `lpf_pick` | LPF_PICK_FROM_FULL_IMAGE |
| `use_fast_coef_updates` | TWO_LOOP |
| `use_fast_coef_costing` | 0 |
| `mode_skip_start` | MAX_MODES |
| `schedule_mode_search` | 0 |
| `use_nonrd_pick_mode` | 0 |
| `inter_mode_mask[i] for i in 0..BLOCK_SIZES-1` | INTER_ALL |
| `max_intra_bsize` | BLOCK_64X64 |
| `reuse_inter_pred_sby` | 0 |
| `always_this_block_size` | BLOCK_16X16 |
| `search_type_check_frequency` | 50 |
| `encode_breakout_thresh` | 0 |
| `recode_tolerance_low` | 12 |
| `recode_tolerance_high` | 25 |
| `default_interp_filter` | SWITCHABLE |
| `simple_model_rd_from_var` | 0 |
| `short_circuit_flat_blocks` | 0 |
| `short_circuit_low_temp_var` | 0 |
| `limit_newmv_early_exit` | 0 |
| `bias_golden` | 0 |
| `base_mv_aggressive` | 0 |
| `rd_ml_partition.prune_rect_thresh[0..3]` | -1, -1, -1, -1 |
| `rd_ml_partition.var_pruning` | 0 |
| `use_accurate_subpel_search` | USE_8_TAPS |
| `adaptive_rd_thresh` | 1 |
| `tx_size_search_breakout` | 1 |
| `tx_size_search_depth` | 2 |
| `exhaustive_searches_thresh` | `(fr_content_type == GRAPHICS_ANIMATION) ? (1<<20) : INT_MAX` |
| `mesh_patterns[*]` | from `best_quality_mesh_pattern[mesh_density_level]` |

`mesh_density_level = (fr_content_type == FC_GRAPHICS_ANIMATION) ? 0 : 1`.

```c
// vp9_speed_features.c:21
static MESH_PATTERN best_quality_mesh_pattern[2][MAX_MESH_STEP] = {
  { { 64, 4 }, { 28, 2 }, { 15, 1 }, { 7, 1 } },     // mesh_density_level 0
  { { 64, 8 }, { 28, 4 }, { 15, 1 }, { 7, 1 } },     // mesh_density_level 1
};
```

### 5.7 Speed-feature framesize-INDEPENDENT — REALTIME mode entry (`set_rt_speed_feature_framesize_independent`, `vp9_speed_features.c:452-484`)

Applied **after** the defaults of §5.6, **before** the speed-cascade. These overrides apply at **all** RT speeds 0..9:

| Field | Value (RT entry) |
|---|---|
| `static_segmentation` | 0 |
| `adaptive_rd_thresh` | 1 |
| `adaptive_rd_thresh_row_mt` | 0 |
| `use_fast_coef_costing` | 1 |
| `exhaustive_searches_thresh` | INT_MAX |
| `allow_acl` | 0 |
| `copy_partition_flag` | 0 |
| `use_source_sad` | 0 |
| `use_simple_block_yrd` | 0 |
| `adapt_partition_source_sad` | 0 |
| `use_altref_onepass` | 0 |
| `use_compound_nonrd_pickmode` | 0 |
| `nonrd_keyframe` | 0 |
| `svc_use_lowres_part` | 0 |
| `overshoot_detection_cbr_rt` | NO_DETECTION |
| `disable_16x16part_nonkey` | 0 |
| `disable_golden_ref` | 0 |
| `enable_tpl_model` | 0 |
| `enhanced_full_pixel_motion_search` | 0 |
| `use_accurate_subpel_search` | USE_2_TAPS |
| `nonrd_use_ml_partition` | 0 |
| `variance_part_thresh_mult` | 1 |
| `cb_pred_filter_search` | 0 |
| `force_smooth_interpol` | 0 |
| `rt_intra_dc_only_low_content` | 0 |
| `mv.enable_adaptive_subpel_force_stop` | 0 |

### 5.8 RT cascade — `speed >= 1` (`vp9_speed_features.c:485-504`)

```c
sf->allow_txfm_domain_distortion = 1;
sf->tx_domain_thresh             = 0.0;
sf->trellis_opt_tx_rd.method     = DISABLE_TRELLIS_OPT;
sf->trellis_opt_tx_rd.thresh     = 0.0;
sf->use_square_partition_only    = !frame_is_intra_only(cm);
sf->less_rectangular_check       = 1;
sf->tx_size_search_method        = frame_is_intra_only(cm) ? USE_FULL_RD : USE_LARGESTALL;
sf->use_rd_breakout              = 1;
sf->adaptive_motion_search       = 1;
sf->adaptive_pred_interp_filter  = 1;
sf->mv.auto_mv_step_size         = 1;
sf->adaptive_rd_thresh           = 2;
sf->intra_y_mode_mask[TX_32X32]  = INTRA_DC_H_V;
sf->intra_uv_mode_mask[TX_32X32] = INTRA_DC_H_V;
sf->intra_uv_mode_mask[TX_16X16] = INTRA_DC_H_V;
```

### 5.9 RT cascade — `speed >= 2` (`vp9_speed_features.c:506-542`)

```c
sf->mode_search_skip_flags =
    (cm->frame_type == KEY_FRAME) ? 0
      : FLAG_SKIP_INTRA_DIRMISMATCH | FLAG_SKIP_INTRA_BESTINTER |
        FLAG_SKIP_COMP_BESTINTRA   | FLAG_SKIP_INTRA_LOWVAR;
sf->adaptive_pred_interp_filter         = 2;
sf->reference_masking                    = (svc->number_spatial_layers == 1);
// + dynamic resize check that clears reference_masking if any scaled ref:
if (sf->reference_masking == 1 &&
    (cpi->external_resize == 1 || cpi->oxcf.resize_mode == RESIZE_DYNAMIC)) {
  for ref in LAST..ALTREF: if scaled ref => sf->reference_masking = 0;
}
sf->disable_filter_search_var_thresh    = 50;
sf->comp_inter_joint_search_iter_level  = 2;
sf->auto_min_max_partition_size         = RELAXED_NEIGHBORING_MIN_MAX;
sf->lf_motion_threshold                  = LOW_MOTION_THRESHOLD;   // = 7
sf->adjust_partitioning_from_last_frame  = 1;
sf->last_partitioning_redo_frequency     = 3;
sf->use_lp32x32fdct                       = 1;
sf->mode_skip_start                      = 11;
sf->intra_y_mode_mask[TX_16X16]          = INTRA_DC_H_V;
```

### 5.10 RT cascade — `speed >= 3` (`vp9_speed_features.c:544-556`)

```c
sf->use_square_partition_only         = 1;
sf->disable_filter_search_var_thresh  = 100;
sf->use_uv_intra_rd_estimate          = 1;
sf->skip_encode_sb                     = 1;
sf->mv.subpel_search_level             = 0;
sf->adaptive_rd_thresh                 = 4;
sf->mode_skip_start                    = 6;
sf->allow_skip_recode                  = 0;
sf->optimize_coefficients              = 0;
sf->disable_split_mask                 = DISABLE_ALL_SPLIT;
sf->lpf_pick                           = LPF_PICK_FROM_Q;
```

### 5.11 RT cascade — `speed >= 4` (`vp9_speed_features.c:558-583`)

```c
if (oxcf->rc_mode == VPX_VBR && oxcf->lag_in_frames > 0)
    sf->use_altref_onepass = 1;
sf->mv.subpel_force_stop              = QUARTER_PEL;
for i in 0..TX_SIZES-1:
  sf->intra_y_mode_mask[i]            = INTRA_DC_H_V;
  sf->intra_uv_mode_mask[i]           = INTRA_DC;
sf->intra_y_mode_mask[TX_32X32]       = INTRA_DC;
sf->frame_parameter_update             = 0;
sf->mv.search_method                   = FAST_HEX;
sf->allow_skip_recode                  = 0;
sf->max_intra_bsize                    = BLOCK_32X32;
sf->use_fast_coef_costing              = 0;
sf->use_quant_fp                       = !is_keyframe;
sf->inter_mode_mask[BLOCK_32X32]       = INTER_NEAREST_NEW_ZERO;
sf->inter_mode_mask[BLOCK_32X64]       = INTER_NEAREST_NEW_ZERO;
sf->inter_mode_mask[BLOCK_64X32]       = INTER_NEAREST_NEW_ZERO;
sf->inter_mode_mask[BLOCK_64X64]       = INTER_NEAREST_NEW_ZERO;
sf->adaptive_rd_thresh                 = 2;
sf->use_fast_coef_updates              = is_keyframe ? TWO_LOOP : ONE_LOOP_REDUCED;
sf->mode_search_skip_flags             = FLAG_SKIP_INTRA_DIRMISMATCH;
sf->tx_size_search_method              = is_keyframe ? USE_LARGESTALL : USE_TX_8X8;
sf->partition_search_type              = VAR_BASED_PARTITION;
```

### 5.12 RT cascade — `speed >= 5` (the RT-baseline; first use of non-RD pickmode) (`vp9_speed_features.c:585-660`)

```c
sf->use_altref_onepass                 = 0;
sf->use_quant_fp                       = !is_keyframe;
sf->auto_min_max_partition_size        = is_keyframe ? RELAXED_NEIGHBORING_MIN_MAX
                                                     : STRICT_NEIGHBORING_MIN_MAX;
sf->default_max_partition_size         = BLOCK_32X32;
sf->default_min_partition_size         = BLOCK_8X8;
sf->force_frame_boost                  = is_keyframe ||
    (frames_since_key % (sf->last_partitioning_redo_frequency << 1) == 1);
sf->max_delta_qindex                   = is_keyframe ? 20 : 15;
sf->partition_search_type              = REFERENCE_PARTITION;
if (oxcf->rc_mode == VPX_VBR && oxcf->lag_in_frames > 0 && rc.is_src_frame_alt_ref)
    sf->partition_search_type          = VAR_BASED_PARTITION;
sf->use_nonrd_pick_mode                = 1;                       // ← qualitative split
sf->allow_skip_recode                  = 0;
sf->inter_mode_mask[BLOCK_32X32]       = INTER_NEAREST_NEW_ZERO;
sf->inter_mode_mask[BLOCK_32X64]       = INTER_NEAREST_NEW_ZERO;
sf->inter_mode_mask[BLOCK_64X32]       = INTER_NEAREST_NEW_ZERO;
sf->inter_mode_mask[BLOCK_64X64]       = INTER_NEAREST_NEW_ZERO;
sf->adaptive_rd_thresh                 = 2;
sf->reuse_inter_pred_sby               = 1;                       // partition-fixed reuse
sf->coeff_prob_appx_step               = 4;
sf->use_fast_coef_updates              = is_keyframe ? TWO_LOOP : ONE_LOOP_REDUCED;
sf->mode_search_skip_flags             = FLAG_SKIP_INTRA_DIRMISMATCH;
sf->tx_size_search_method              = is_keyframe ? USE_LARGESTALL : USE_TX_8X8;
sf->simple_model_rd_from_var           = 1;
if (oxcf->rc_mode == VPX_VBR)
    sf->mv.search_method               = NSTEP;

// intra_y_mode_bsize_mask: content-dependent
if (!is_keyframe) {
  if (content == VP9E_CONTENT_SCREEN) {
    for i in 0..BLOCK_SIZES-1:
      sf->intra_y_mode_bsize_mask[i] = (i >= BLOCK_32X32) ? INTRA_DC_H_V : INTRA_DC_TM_H_V;
  } else {
    for i in 0..BLOCK_SIZES-1:
      sf->intra_y_mode_bsize_mask[i] = (i > BLOCK_16X16) ? INTRA_DC : INTRA_DC_H_V;
  }
}

if (content == VP9E_CONTENT_SCREEN) sf->short_circuit_flat_blocks = 1;

// CBR-only tweaks
if (rc_mode == VPX_CBR && content != VP9E_CONTENT_SCREEN) {
  sf->limit_newmv_early_exit = 1;
  if (!cpi->use_svc) sf->bias_golden = 1;
}

if (cpi->use_svc && svc->spatial_layer_id > 0) sf->nonrd_keyframe = 1;

// CBR overshoot detection on slide/scene change
if (cm->frame_type != KEY_FRAME && cpi->resize_state == ORIG &&
    rc_mode == VPX_CBR && !rc.disable_overshoot_maxq_cbr) {
  if (cm->width*cm->height <= 352*288 && !cpi->use_svc && content != VP9E_CONTENT_SCREEN)
    sf->overshoot_detection_cbr_rt = RE_ENCODE_MAXQ;
  else
    sf->overshoot_detection_cbr_rt = FAST_DETECTION_MAXQ;
}

// VBR + lag enables 1-pass alt-ref + compound nonrd at <=720p
if (rc_mode == VPX_VBR && lag_in_frames > 0 && cm->width <= 1280 && cm->height <= 720) {
  sf->use_altref_onepass            = 1;
  sf->use_compound_nonrd_pickmode   = 1;
}

if (cm->width*cm->height > 1280*720) sf->cb_pred_filter_search = 2;
if (!cpi->external_resize)            sf->use_source_sad         = 1;
```

### 5.13 RT cascade — `speed >= 6` (cpu-used 6 — primary RT target) (`vp9_speed_features.c:662-697`)

```c
if (rc_mode == VPX_VBR && lag_in_frames > 0) {
  sf->use_altref_onepass          = 1;
  sf->use_compound_nonrd_pickmode = 1;
}
sf->partition_search_type    = VAR_BASED_PARTITION;            // ← speed-6 change
sf->mv.search_method         = NSTEP;
sf->mv.reduce_first_step_size = 1;
sf->skip_encode_sb           = 0;

if (sf->use_source_sad) {
  sf->adapt_partition_source_sad = 1;
  sf->adapt_partition_thresh =
      (cm->width * cm->height <= 640 * 360) ? 40000 : 60000;
  if (cpi->content_state_sb_fd == NULL &&
      (!use_svc || svc->spatial_layer_id == svc->number_spatial_layers - 1))
    alloc content_state_sb_fd[((mi_stride>>3) * ((mi_rows>>3)+1))]
}

if (rc_mode == VPX_CBR && content != VP9E_CONTENT_SCREEN)
  sf->short_circuit_low_temp_var = 1;

// SVC tweaks for non-base temporal layers
if (svc->temporal_layer_id > 0) {
  sf->adaptive_rd_thresh    = 4;
  sf->limit_newmv_early_exit = 0;
  sf->base_mv_aggressive     = 1;
}
if (cm->frame_type != KEY_FRAME && resize_state == ORIG &&
    rc_mode == VPX_CBR && !rc.disable_overshoot_maxq_cbr)
  sf->overshoot_detection_cbr_rt = FAST_DETECTION_MAXQ;
```

### 5.14 RT cascade — `speed >= 7` (cpu-used 7 — secondary RT target) (`vp9_speed_features.c:699-749`)

```c
sf->adapt_partition_source_sad = 0;       // ← undoes speed-6 adapt
sf->adaptive_rd_thresh         = 3;
sf->mv.search_method           = FAST_DIAMOND;
sf->mv.fullpel_search_step_param = 10;

// SVC: better mv search on base temporal layer, base spatial low-res
if (svc->number_temporal_layers > 2 && svc->temporal_layer_id == 0 &&
    (svc->spatial_layer_id == 0 ||
     oxcf->width * oxcf->height <= 640 * 360)) {
  sf->mv.search_method            = NSTEP;
  sf->mv.fullpel_search_step_param = 6;
}

// SVC: fast block-yrd on enhancement layers; prune subpel on non-ref
if (svc->temporal_layer_id > 0 || svc->spatial_layer_id > 1) {
  sf->use_simple_block_yrd = 1;
  if (svc->non_reference_frame)
    sf->mv.subpel_search_method = SUBPEL_TREE_PRUNED_EVENMORE;
}

if (use_svc && cpi->row_mt && oxcf->max_threads > 1)
  sf->adaptive_rd_thresh_row_mt = 1;

// Partition copy (single layer or top spatial)
cpi->max_copied_frame = 0;
if (!cpi->last_frame_dropped && resize_state == ORIG &&
    !cpi->external_resize &&
    (!use_svc ||
     (svc->spatial_layer_id == svc->number_spatial_layers - 1 &&
      !svc->last_layer_dropped[svc->number_spatial_layers - 1]))) {
  sf->copy_partition_flag = 1;
  cpi->max_copied_frame    = 2;
  if (svc->number_temporal_layers > 1 &&
      svc->temporal_layer_id == svc->number_temporal_layers - 1)
    cpi->max_copied_frame = 255;       // top temporal: non-ref, copy forever
}

// SVC 3-spatial: lower-res partition reuse
if (use_svc && svc->use_partition_reuse &&
    svc->number_spatial_layers == 3 && svc->temporal_layer_id > 0 &&
    oxcf->width * oxcf->height > 640 * 480)
  sf->svc_use_lowres_part = 1;

// SVC + GF temporal ref: only on base temporal
if (use_svc && svc->use_gf_temporal_ref_current_layer &&
    svc->temporal_layer_id > 0)
  cpi->ref_frame_flags &= (~VP9_GOLD_FLAG);

if (cm->width * cm->height > 640 * 480) sf->cb_pred_filter_search = 2;
```

### 5.15 RT cascade — `speed >= 8` (cpu-used 8 — fastest stable RT) (`vp9_speed_features.c:751-793`)

```c
sf->adaptive_rd_thresh = 4;
sf->skip_encode_sb     = 1;

// SVC keyframe handling
if (svc->number_spatial_layers > 1 && !svc->simulcast_mode)
  sf->nonrd_keyframe = 0;
else
  sf->nonrd_keyframe = 1;

if (!use_svc) cpi->max_copied_frame = 4;

if (cpi->row_mt && oxcf->max_threads > 1)
  sf->adaptive_rd_thresh_row_mt = 1;

// ML partition for QQVGA
if (!frame_is_intra_only(cm) && cm->width * cm->height <= 352 * 288) {
  sf->nonrd_use_ml_partition = 1;
}
// HBD turns ML off
#if CONFIG_VP9_HIGHBITDEPTH
if (cpi->Source->flags & YV12_FLAG_HIGHBITDEPTH)
  sf->nonrd_use_ml_partition = 0;
#endif

if (content == VP9E_CONTENT_SCREEN) sf->mv.subpel_force_stop = HALF_PEL;
sf->rt_intra_dc_only_low_content = 1;

// CBR-only short_circuit_low_temp_var aggressiveness
if (!use_svc && rc_mode == VPX_CBR && content != VP9E_CONTENT_SCREEN) {
  sf->short_circuit_low_temp_var = 3;
  if (cpi->noise_estimate.enabled && cm->width >= 1280 && cm->height >= 720) {
    NOISE_LEVEL nl = vp9_noise_estimate_extract_level(&cpi->noise_estimate);
    if (nl >= kMedium) sf->short_circuit_low_temp_var = 2;
  }
  // adaptive_rd_thresh inversely scaled
  if (cm->width * cm->height > 352 * 288) sf->adaptive_rd_thresh = 1;
  else                                     sf->adaptive_rd_thresh = 2;
}

sf->limit_newmv_early_exit = 0;
sf->use_simple_block_yrd    = 1;
if (cm->width * cm->height > 352 * 288) sf->cb_pred_filter_search = 2;
```

### 5.16 RT cascade — `speed >= 9` (cpu-used 9 — experimental, below D1 floor) (`vp9_speed_features.c:795-814`)

```c
if (!is_keyframe)
  for i in 0..BLOCK_SIZES-1: sf->intra_y_mode_bsize_mask[i] = INTRA_DC;
sf->cb_pred_filter_search                    = 2;
sf->mv.enable_adaptive_subpel_force_stop     = 1;
sf->mv.adapt_subpel_force_stop.mv_thresh     = 1;
sf->mv.adapt_subpel_force_stop.force_stop_below = QUARTER_PEL;
sf->mv.adapt_subpel_force_stop.force_stop_above = HALF_PEL;
if (cm->frame_type != KEY_FRAME && cm->width >= 320 && cm->height >= 240)
  sf->disable_16x16part_nonkey = 1;
if (rc_mode == VPX_CBR) sf->disable_golden_ref = 1;
if (cpi->rc.avg_frame_low_motion < 70) sf->default_interp_filter = BILINEAR;
if (cm->width * cm->height >= 640 * 360) sf->variance_part_thresh_mult = 2;
```

cpu-used 9 is **outside D1** (real-time targets cpu-used 6-8). Documenting for completeness only; rvp9 v1 does not need to mirror it.

### 5.17 RT cascade — post-cascade overrides (apply to all speeds) (`vp9_speed_features.c:816-867`)

```c
// Disable split to 8x8 for low-res at very high Q (VAR_BASED_PARTITION only)
if (frame_type != KEY_FRAME && cm->width * cm->height <= 320 * 240 &&
    sf->partition_search_type == VAR_BASED_PARTITION &&
    rc.avg_frame_qindex[INTER_FRAME] > 208 &&
    cm->current_video_frame > 8)
  sf->disable_16x16part_nonkey = 1;

if (sf->nonrd_use_ml_partition)
  sf->partition_search_type = ML_BASED_PARTITION;

// use_altref_onepass: re-set fixed 64x64 on the alt-ref source frame
if (sf->use_altref_onepass) {
  if (rc.is_src_frame_alt_ref && frame_type != KEY_FRAME) {
    sf->partition_search_type = FIXED_PARTITION;
    sf->always_this_block_size = BLOCK_64X64;
  }
  alloc cpi->count_arf_frame_usage / count_lastgolden_frame_usage if NULL
}

// SVC intra-only follow-up: force FIXED 64x64
if (svc->previous_frame_is_intra_only) {
  sf->partition_search_type = FIXED_PARTITION;
  sf->always_this_block_size = BLOCK_64X64;
}

// Screen content + high motion: NSTEP + small step_param
if (content == VP9E_CONTENT_SCREEN && oxcf->speed >= 5 &&
    (svc->high_num_blocks_with_motion || svc->last_layer_dropped[0])) {
  sf->mv.search_method = NSTEP;
  sf->mv.fullpel_search_step_param = 2;
}

// Disable aq_mode=3 (cyclic refresh) for speed <= 3
if (speed <= 3 && oxcf->aq_mode == CYCLIC_REFRESH_AQ) oxcf->aq_mode = 0;

// Deadline mode change forces nonrd_keyframe
if (oxcf->mode != cpi->deadline_mode_previous_frame)
  sf->nonrd_keyframe = 1;
```

### 5.18 RT framesize-DEPENDENT cascade (`set_rt_speed_feature_framesize_dependent`, `vp9_speed_features.c:414-450`)

| speed | width/height ≥ 720 | width/height < 720 |
|---|---|---|
| `>= 1` | `disable_split_mask = show_frame ? DISABLE_ALL_SPLIT : DISABLE_ALL_INTER_SPLIT` | `disable_split_mask = DISABLE_COMPOUND_SPLIT` |
| `>= 2` | `disable_split_mask = show_frame ? DISABLE_ALL_SPLIT : DISABLE_ALL_INTER_SPLIT` | `disable_split_mask = LAST_AND_INTRA_SPLIT_ONLY` |
| `>= 5` | `partition_search_breakout_thr.dist = 1 << 25; .rate = 200` | `partition_search_breakout_thr.dist = 1 << 23; .rate = 200` |
| `>= 7` | `encode_breakout_thresh = 800` | `encode_breakout_thresh = 300` |

Plus the "best-quality defaults" `partition_search_breakout_thr.dist = 1 << 19; .rate = 80` and `rd_ml_partition.search_early_termination = search_breakout = 0` set just before the dispatch (`vp9_speed_features.c:876-880`).

Followed by a `disable_split_mask == DISABLE_ALL_SPLIT ⇒ adaptive_pred_interp_filter = 0` clamp (`vp9_speed_features.c:889-891`), the `encode_breakout` user override (`vp9_speed_features.c:893-896`), and the `disable_split_mask` → `rd->thresh_mult_sub8x8` propagation (`vp9_speed_features.c:898-903`).

### 5.19 Cumulative speed-5..8 summary (deltas only)

Per-field deltas between speeds 5, 6, 7, 8. Empty cell = same as previous column.

| Field | sp=5 | sp=6 | sp=7 | sp=8 |
|---|---|---|---|---|
| `partition_search_type` | REFERENCE_PARTITION (or VAR if VBR+lag+arf) | VAR_BASED_PARTITION | | |
| `mv.search_method` | (VBR? NSTEP : inherits FAST_HEX from sp4) | NSTEP | FAST_DIAMOND (or NSTEP for SVC base-tl0 of low-res) | |
| `mv.reduce_first_step_size` | 0 | 1 | | |
| `mv.fullpel_search_step_param` | 6 (inherited) | | 10 (or 6 for SVC base-tl0 low-res) | |
| `mv.subpel_force_stop` | QUARTER_PEL (inh. sp4) | | | HALF_PEL (screen only) |
| `mv.subpel_search_method` | SUBPEL_TREE_PRUNED_MORE (inh. sp4) | | SUBPEL_TREE_PRUNED_EVENMORE (SVC non-ref enh) | |
| `mv.enable_adaptive_subpel_force_stop` | 0 | | | |
| `skip_encode_sb` | 1 (inh. sp3) | 0 | | 1 |
| `default_max_partition_size` | BLOCK_32X32 | | | |
| `default_min_partition_size` | BLOCK_8X8 | | | |
| `auto_min_max_partition_size` | KF? RELAXED : STRICT | | | |
| `max_delta_qindex` | KF?20:15 | | | |
| `force_frame_boost` | KF \| (frames_since_key%(lpf_rdf<<1)==1) | | | |
| `use_nonrd_pick_mode` | 1 | | | |
| `reuse_inter_pred_sby` | 1 | | | |
| `coeff_prob_appx_step` | 4 | | | |
| `simple_model_rd_from_var` | 1 | | | |
| `use_source_sad` | !external_resize ? 1 : 0 | | | |
| `adapt_partition_source_sad` | 0 | (use_source_sad ? 1 : 0) | 0 | |
| `adapt_partition_thresh` | 0 | (`<=640*360 ? 40000 : 60000`) | | |
| `short_circuit_low_temp_var` | 0 | CBR+non-screen → 1 | | CBR+non-screen → 3 (or 2 noisy 720p+) |
| `short_circuit_flat_blocks` | screen?1:0 | | | |
| `limit_newmv_early_exit` | CBR+non-screen → 1 | (SVC enh: 0) | | 0 |
| `bias_golden` | CBR+non-screen+!SVC → 1 | | | |
| `base_mv_aggressive` | 0 | SVC enh → 1 | | |
| `nonrd_keyframe` | SVC enh → 1 | | | SVC>1+!simul → 0, else 1 |
| `use_simple_block_yrd` | 0 | | SVC enh → 1 | 1 |
| `use_compound_nonrd_pickmode` | VBR+lag+<=720p → 1 | VBR+lag → 1 | | |
| `use_altref_onepass` | VBR+lag+<=720p → 1 | VBR+lag → 1 | | |
| `overshoot_detection_cbr_rt` | CBR non-KF → RE_ENCODE_MAXQ (small) or FAST | CBR → FAST_DETECTION_MAXQ | | |
| `copy_partition_flag` / `max_copied_frame` | 0 / 0 | | (cond.) 1 / 2 or 255 | (!SVC) max_copied=4 |
| `svc_use_lowres_part` | 0 | | SVC-3sl+enh-tl+>640*480 → 1 | |
| `adaptive_rd_thresh` | 2 | (SVC tl>0 ? 4 : 2) | 3 | 4 (CBR non-screen variant: 1 or 2) |
| `adaptive_rd_thresh_row_mt` | 0 | | (use_svc&&row_mt&&maxT>1) → 1 | (row_mt&&maxT>1) → 1 |
| `cb_pred_filter_search` | (>1280*720 → 2) | | (>640*480 → 2) | (>352*288 → 2) |
| `rt_intra_dc_only_low_content` | 0 | | | 1 |
| `mv.fullpel_search_step_param` | 6 | | 10 / 6 | |
| `nonrd_use_ml_partition` | 0 | | | low-res non-KF → 1, HBD → 0 |
| `intra_y_mode_bsize_mask` | (set per content; see §5.12) | | | |
| `inter_mode_mask[BLOCK_{32X32,32X64,64X32,64X64}]` | INTER_NEAREST_NEW_ZERO | | | |
| `tx_size_search_method` | KF? USE_LARGESTALL : USE_TX_8X8 | | | |
| `use_fast_coef_updates` | KF? TWO_LOOP : ONE_LOOP_REDUCED | | | |
| `mode_search_skip_flags` | FLAG_SKIP_INTRA_DIRMISMATCH | | | |
| `disable_split_mask` (framesize-dep, sp≥2) | (≥720 → DISABLE_ALL_SPLIT or DISABLE_ALL_INTER_SPLIT; <720 → LAST_AND_INTRA_SPLIT_ONLY) | | | |
| `encode_breakout_thresh` (framesize-dep, sp≥7) | 0 | 0 | ≥720 → 800; <720 → 300 | |

### 5.20 Mode/intra masks (`vp9/encoder/vp9_speed_features.h:20-51`)

```c
enum {
  INTRA_ALL       = (1<<DC_PRED) | (1<<V_PRED) | (1<<H_PRED) | (1<<D45_PRED) |
                    (1<<D135_PRED) | (1<<D117_PRED) | (1<<D153_PRED) |
                    (1<<D207_PRED) | (1<<D63_PRED) | (1<<TM_PRED),
  INTRA_DC        = (1<<DC_PRED),
  INTRA_DC_TM     = (1<<DC_PRED) | (1<<TM_PRED),
  INTRA_DC_H_V    = (1<<DC_PRED) | (1<<V_PRED) | (1<<H_PRED),
  INTRA_DC_TM_H_V = (1<<DC_PRED) | (1<<TM_PRED) | (1<<V_PRED) | (1<<H_PRED)
};

enum {
  INTER_ALL              = (1<<NEARESTMV) | (1<<NEARMV) | (1<<ZEROMV) | (1<<NEWMV),
  INTER_NEAREST          = (1<<NEARESTMV),
  INTER_NEAREST_NEW      = (1<<NEARESTMV) | (1<<NEWMV),
  INTER_NEAREST_ZERO     = (1<<NEARESTMV) | (1<<ZEROMV),
  INTER_NEAREST_NEW_ZERO = (1<<NEARESTMV) | (1<<ZEROMV) | (1<<NEWMV),
  INTER_NEAREST_NEAR_NEW = (1<<NEARESTMV) | (1<<NEARMV) | (1<<NEWMV),
  INTER_NEAREST_NEAR_ZERO= (1<<NEARESTMV) | (1<<NEARMV) | (1<<ZEROMV),
};

enum {
  DISABLE_ALL_INTER_SPLIT    = (1<<THR_COMP_GA) | (1<<THR_COMP_LA) |
                               (1<<THR_ALTR) | (1<<THR_GOLD) | (1<<THR_LAST),
  DISABLE_ALL_SPLIT          = (1<<THR_INTRA) | DISABLE_ALL_INTER_SPLIT,
  DISABLE_COMPOUND_SPLIT     = (1<<THR_COMP_GA) | (1<<THR_COMP_LA),
  LAST_AND_INTRA_SPLIT_ONLY  = (1<<THR_COMP_GA) | (1<<THR_COMP_LA) |
                               (1<<THR_ALTR) | (1<<THR_GOLD),
};
```

### 5.21 Motion / filter thresholds (`vp9_speed_features.h:85-136`)

```c
typedef enum { NO_MOTION_THRESHOLD = 0, LOW_MOTION_THRESHOLD = 7 } MOTION_THRESHOLD;
typedef enum { NOT_IN_USE = 0, RELAXED_NEIGHBORING_MIN_MAX = 1,
               STRICT_NEIGHBORING_MIN_MAX = 2 } AUTO_MIN_MAX_MODE;
typedef enum {
  FLAG_EARLY_TERMINATE         = 1<<0,
  FLAG_SKIP_COMP_BESTINTRA     = 1<<1,
  FLAG_SKIP_INTRA_BESTINTER    = 1<<3,
  FLAG_SKIP_INTRA_DIRMISMATCH  = 1<<4,
  FLAG_SKIP_INTRA_LOWVAR       = 1<<5,
} MODE_SEARCH_SKIP_LOGIC;
typedef enum {
  FLAG_SKIP_EIGHTTAP        = 1<<EIGHTTAP,
  FLAG_SKIP_EIGHTTAP_SMOOTH = 1<<EIGHTTAP_SMOOTH,
  FLAG_SKIP_EIGHTTAP_SHARP  = 1<<EIGHTTAP_SHARP,
} INTERP_FILTER_MASK;
```

### 5.22 RT mesh patterns

Not used in real-time mode: `set_rt_speed_feature_framesize_independent` does not write `sf->mesh_patterns[]`. Only the GOOD path overrides them. RT uses the `best_quality_mesh_pattern[1]` row by default (mesh_density_level 1, `vp9_speed_features.c:1027-1035`). In practice `exhaustive_searches_thresh = INT_MAX` makes mesh search unreachable in RT.

## 6. RTCD / SIMD surfaces

Threading is not an RTCD surface itself, but threading-adjacent components are:

| Function / surface | SIMD coverage | Used by |
|---|---|---|
| `vp9_accumulate_frame_counts` (`vp9/common/vp9_thread_common.c:488`) | scalar only | tile-MT counts merge, decoder counts merge |
| `vp9_loop_filter_alloc` / loop-filter row jobs | calls `vpx_lpf_*` kernels (SSE2/AVX2/NEON) | encoder post-encode LF, decoder LF |
| `vp9_temporal_filter_*` (`vp9/encoder/vp9_temporal_filter.c`) | calls `vpx_highbd_subpel_variance` / pel-variance kernels | `vp9_temporal_filter_row_mt` |
| Down-sampling for SVC (`scaled_frames`, `scaled_temp`) | calls `vpx_scaled_*` and `vp9_resize_*` kernels (SSE2 in `vp9_resize.c`) | `vp9_one_pass_svc_start_layer` resolution change |

SIMD initialization is done globally in `vpx_dsp_rtcd()` (called from `vpx_codec_enc_init` paths before any worker exists). No per-thread RTCD state. Workers read function pointers from the global RTCD table; rvp9 must ensure the equivalent dispatch table is fully populated before spawning any worker, otherwise we hit a data race on first-time RTCD selection (libvpx solves this by detecting features once and storing in a fixed table).

The denoiser and AQ kernels (A5's territory) have SIMD that runs on tile workers; the tile-MT split assumes those kernels are reentrant — which they are, because they only mutate per-MACROBLOCK / per-segment-map state that lives in `td->mb` or the per-tile slice.

## 7. Open questions for LLD

These will be resolved in Stage 2 LLDs (`notes/lld/tile-threading.md`, `notes/lld/svc.md`, plus the HLD `notes/hld/01-threading.md`).

### 7.1 `SPEED_FEATURES` representation in Rust

libvpx's `SPEED_FEATURES` is a 600+ line POD struct accessed concurrently:
- The main thread mutates it before each frame (in `vp9_set_speed_features_*` from `set_size_independent_vars` / `set_size_dependent_vars`).
- Every tile worker **reads** it during partition / mode-decision (via `cpi->sf.field`).

The contract: `SPEED_FEATURES` is **read-only during a frame encode**. Workers see a consistent snapshot because the main thread finishes both setters before `vp9_encode_tiles_*` launches workers.

Rust options:
- (A) Owned by `VP9Comp`, accessed by workers via `&SpeedFeatures` (immutable borrow lasting the launch_workers scope). Requires scoped threads or `Arc<SpeedFeatures>` rebuilt each frame.
- (B) `UnsafeCell<SpeedFeatures>` inside the encoder struct, with documented invariant "no writes between launch and sync". Closer to libvpx, but requires unsafe at every read.
- (C) Snapshot the struct into each worker's `ThreadData` at frame start (memcpy). Costs ~600 bytes × N workers per frame; eliminates aliasing entirely.

**Recommendation:** start with (C) — it is the most Rust-idiomatic and the cost is negligible (≤80 × 600 B = 48 KB per frame). Revisit if profiling shows it dominates.

### 7.2 `ThreadData::mb` (`MACROBLOCK`) aliasing

The libvpx pattern is `td->mb = cpi->td.mb` — value-copy a struct that contains many *pointers* into `cpi`/`cm`/buffer pool/tile data. After the copy:
- Worker writes to its `td->mb.e_mbd.{plane[i].dst, plane[i].pre[], plane[i].above_context, plane[i].left_context}` etc.
- Some `mb` pointers (e.g. `mb.plane[i].coeff/qcoeff/eobs`) get re-wired to `ctx->coeff_pbuf[j][0]` for nonrd-pick-mode (`vp9_ethread.c:227-232, 668-673`).

In Rust this is the hardest piece: `MACROBLOCK` contains raw pointers into shared buffers that the workers will write to (e.g. `above_context[mi_col_global..]`). Even though tile workers write **disjoint** slices, both raw pointers point into the **same allocation**, which Rust's aliasing rules disallow for `&mut`.

Two viable approaches:
- (A) **Raw-pointer arena**: hold `cm->above_context` and similar as `UnsafeCell<[Atomic*]>` or `*mut [T]` with documented "tile worker `t` may only mutate the slice spanning its tile columns" invariant. Workers carry `*mut T` plus integer bounds, with `unsafe { *p }` accesses behind `#[inline]` helpers that assert the bounds in debug.
- (B) **Slice-splitting**: at the start of each frame, split `above_context` into N disjoint `&mut [ENTROPY_CONTEXT]` slices keyed by tile, hand one to each worker. Requires either scoped threads or `std::thread::scope` (stable since 1.63). Worker can never see another tile's slice. This is the safest path.

**Recommendation for the LLD:** option (B) — split-by-tile up front, with a `TileScratch<'a>` struct holding `&'a mut`s; pass into scoped worker threads. UnsafeCell only where the shared structures genuinely cannot be split (e.g. `cm->mi_grid_visible` — but this too is partitioned by tile column for the duration of encode).

For **row-MT** the splitting is more subtle: `above_context` slices need to be sub-divided by SB row in addition to tile column, but rows are processed in order within a column so each row's `above_context` write is consumed by the next row's read. The `VP9RowMTSync` happens-before edge (mutex/cond) gives us the synchronization the borrow-checker can't see. This will require `UnsafeCell` + raw pointers, with safety justified by the `vp9_row_mt_sync_read`/`_write` contract.

### 7.3 `VPxWorker` mapping

Options:
- (A) `std::thread::Builder::spawn` + a `Mutex<Status>` + `Condvar` mirroring libvpx exactly. The pattern is identical to `thread_loop` in `vpx_util/vpx_thread.c`.
- (B) `rayon::ThreadPool` with `pool.scope_fifo`. Wide ecosystem, hides the work-stealing.
- (C) `std::thread::scope` + a hand-rolled task queue. Stable, simple, no external dep.

Q3 in PLAN.md is the place this gets decided. The work-stealing in `enc_row_mt_worker_hook` makes a custom queue (option A or C) more natural than rayon's join model.

**Recommendation:** option (C). `std::thread::scope` gives us scoped lifetimes (perfect for the tile-split arrays from §7.2), and a hand-rolled `(Mutex<JobQueue>, Condvar)` per tile column matches libvpx 1:1.

### 7.4 Counts accumulation

libvpx accumulates `td->counts` into `cm->counts` AFTER all workers `sync()` (`vp9_ethread.c:243-247`). This is fine because no worker is running during accumulation. In Rust, `cm->counts` is `&mut` while the workers hold `&` (immutable) — the scoped-threads + immutable borrow model handles this cleanly. **No invariant work needed beyond §7.1.**

### 7.5 SVC layer-context swap ordering

`vp9_restore_layer_context` is called *before* each spatial layer's encode; `vp9_save_layer_context` is called *after*. The cyclic-refresh map swap (`vp9_svc_layercontext.c:380-388, 410-419`) is a pointer swap — in Rust, this is `std::mem::swap(&mut cpi.cyclic_refresh.map, &mut lc.map)` and matches the C semantics exactly. No aliasing concerns because this is single-threaded (between frame-encode invocations).

### 7.6 Per-thread `FRAME_COUNTS` allocation

libvpx allocates `td->counts` per non-last worker via `vpx_calloc(1, sizeof(*counts))` (`vp9_ethread.c:124-125`). `FRAME_COUNTS` is large (8-10 KB). In Rust, a `Box<FrameCounts>` per worker is the obvious mapping; the question is whether to keep it owned by the worker's `ThreadData` or held by the pool. Decision: keep it in `ThreadData`; the pool struct lives until encoder teardown.

### 7.7 `cpi->row_mt` and `cpi->oxcf.max_threads` modeling

`cpi->row_mt` is a runtime flag. The current libvpx default (in real-time) is `cpi->row_mt = 1` when both `oxcf.row_mt && oxcf.max_threads > 1`. The interaction with `adaptive_rd_thresh_row_mt` is subtle:

- `adaptive_rd_thresh` is gated to 0 when `!sf->adaptive_rd_thresh_row_mt && cpi->row_mt_bit_exact && max_threads > 1` (`vp9_speed_features.c:910-912, 1090-1092`). I.e., **bit-exact row-MT** disables the per-frame adaptive RD threshold; **non-bit-exact** row-MT keeps it (workers race-condition on `tile_data->thresh_freq_fact[][]`).

For rvp9: choosing bit-exact-by-default is the natural Rust answer (`adaptive_rd_thresh_row_mt = 0` ⇒ deterministic). This costs some BD-rate but eliminates a major source of non-determinism. Worth flagging in the divergence ledger.

### 7.8 Decoder threading: parse / recon split

libvpx decoder row-MT splits parse and recon into separate jobs (PARSE_JOB, RECON_JOB, LPF_JOB) consuming a shared queue (`vp9_decodeframe.c:1894+`). PARSE writes coefficients into `RowMTWorkerData.eob[plane]` / `dqcoeff[plane]` arrays sized `sb_rows * sb_cols * (EOBS_PER_SB | DQCOEFFS_PER_SB)`. RECON reads them. Sync uses `recon_map[sb_row*sb_cols + sb_col]` flag bytes + per-row mutex/cond pairs.

In Rust this is two arenas — parse-output and recon-output — each split by `(sb_row, sb_col)`. The natural mapping is a `Vec<UnsafeCell<SbCoeffSlot>>` of length `sb_rows * sb_cols`, with synchronization on `recon_map`. The `recon_map` element is plain `bool` / `i8` but protected by the row mutex+cond. **A `crossbeam::channel` or a futures `mpsc` does NOT fit** — the libvpx pattern is point-to-point: a recon-job for `(r, c)` waits specifically for `(r-1, c)` to be recon-done, not for an arbitrary message.

### 7.9 Frame-parallel encoding (Q4)

libvpx VP9 has **no** frame-parallel encoding. Q4 in PLAN.md may resolve to "deferred to v2". This document confirms that the libvpx threading machinery does not give a template for multiple frames in flight at once — the worker pool is sized per-frame and torn down/rebuilt on `num_workers` change, and `cpi` is a single shared encoder state. Adding frame-parallel-encode is a v2 redesign, not a v1 mirror.

### 7.10 SIMD / RTCD initialization ordering

The global `vpx_dsp_rtcd()` (in `vpx_dsp/vpx_dsp_rtcd.c`, generated from `vpx_dsp_rtcd_defs.pl`) must run before any worker spawns. libvpx ensures this by calling `once_init` at `vpx_codec_enc_init`. rvp9 mirror: one-time `std::sync::Once` in encoder init, performed before `create_enc_workers` equivalent runs. **Q2 in PLAN.md** (hand-written vs generated dispatch table) does not affect threading — both styles work fine here.

### 7.11 SVC max-tile-cols sizing under thread pool

`create_enc_workers` (`vp9_ethread.c:84-90`) sizes the worker pool to the *highest-resolution* spatial layer's `get_max_tile_cols`, not the current layer's. This means when encoding spatial layer 0 (e.g. 360p) with 4 tiles, but spatial layer 2 (1080p) supports 8 tiles, the pool has 8 workers but only 4 are launched per layer-0 frame. rvp9 mirror: size the pool by `max_tile_cols(top_spatial_layer)` once at init and re-launch only `min(max_threads, current_layer_tile_cols)` per frame. Worker thread overhead is a single mutex+cond+heap object per dormant worker (~200 bytes), negligible.

### 7.12 Tile-row support

Real-time encoding effectively never uses tile rows (`log2_tile_rows == 0` is asserted in the decoder row-MT path, `vp9_decodeframe.c:2343`). The encoder allows `log2_tile_rows > 0` (max `MAX_NUM_TILE_ROWS = 4`) but the row-MT job queue handles it by advancing `tile_row` within the per-column queue (`vp9_multi_thread.c:282-288`).

For rvp9 v1 with D1 (real-time priority), it's safe to **assume `log2_tile_rows == 0`** and panic on `log2_tile_rows > 0` until v2. Document in the divergence ledger.

### 7.13 Worker thread priority

libvpx Windows shim sets `SetThreadPriority(*thread, THREAD_PRIORITY_ABOVE_NORMAL)` (`vpx_util/vpx_thread.h:90`). pthreads on Linux relies on scheduler defaults. rvp9: not a portable concept on stable Rust; either skip (libvpx-on-Linux behaviour, which is the more common deployment), or wrap with a `#[cfg(windows)]` `SetThreadPriority` call using `windows-sys`. **Recommendation:** skip in v1, file as enhancement.

### 7.14 Cumulative speed-feature equivalence test

A concrete deliverable for Stage 4a: a **golden file** (e.g. JSON or RON serialization of `SPEED_FEATURES`) captured from libvpx via a small printf patch, one file per `(speed, frame_type, width, height, content_type, rc_mode, use_svc, lag, ...)` tuple of interest. rvp9's speed-feature setter is asserted byte-equal against the golden. This is the only way to ensure the cpu-used 6/7/8 cascades stay in lock-step with libvpx across the dozen-plus conditional toggles (resize, source-sad, lag, SVC, screen content, noise level, etc.).

This is fundamentally what D8 ("compression-efficiency parity is the gating goal; byte parity is a debugging technique") points to — but the speed-feature table is where byte-parity *is* tractable and where any divergence will silently cost BD-rate. We will gate on `SPEED_FEATURES` byte-parity even if downstream encode output is not bit-exact.
