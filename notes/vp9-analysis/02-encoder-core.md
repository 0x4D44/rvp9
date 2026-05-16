# Encoder Core (non-RD path) in libvpx VP9

Specialist: A2.
Sources cited against `third_party/libvpx/` at the v1.14.1 tag
(commit `12f3a2a`). Line numbers are absolute.

---

## 1. Scope

This doc covers the **real-time / non-RD mode-decision and partition path**
that libvpx VP9 uses at `--cpu-used 5..=8` (i.e. when
`sf->use_nonrd_pick_mode == 1` is set by
`vp9_speed_features.c:585+`). Specifically:

- The home file `vp9/encoder/vp9_pickmode.c` (2992 lines), top to bottom.
- The non-RD entry points called from `vp9/encoder/vp9_encodeframe.c`:
  `encode_nonrd_sb_row`, `choose_partitioning`, `nonrd_pick_sb_modes`,
  `nonrd_pick_partition`, `nonrd_select_partition`, `nonrd_use_partition`,
  `encode_sb_rt`, `encode_b_rt`, `update_state_rt`.
- Intra mode selection in non-RD mode (`vp9_pick_intra_mode`,
  `hybrid_intra_mode_search`, `compute_intra_yprediction`).
- Inter mode selection in non-RD mode (`vp9_pick_inter_mode`,
  `vp9_pick_inter_mode_sub8x8`, the 12-entry `ref_mode_set` loop,
  early-exit logic, mode bias).
- Variance-based partition decision (`choose_partitioning`,
  `set_vt_partitioning`, `set_vbp_thresholds`,
  `vp9_set_variance_partition_thresholds`, `set_low_temp_var_flag`,
  `chroma_check`).
- Skip detection: `encode_breakout_test`,
  `SKIP_TXFM_{NONE,AC_DC,AC_ONLY}` semantics,
  `model_rd_for_sb_y[_large]` skipping tests.
- Non-RD cost approximations: `block_yrd` (Hadamard + SATD path),
  `model_rd_for_sb_y`, `model_rd_for_sb_y_large`,
  `model_rd_for_sb_uv`, `vp9_model_rd_from_var_lapndz` (the Laplacian
  R-D model that replaces full RDO at speed 5+).
- The 12-entry inter-mode test list and 4-mode intra list (`ref_mode_set`,
  `ref_mode_set_svc`, `intra_mode_list`).
- The adaptive RD-threshold + frequency-fact update used to skip modes.

**NOT covered here** (owned by other specialists):

- Motion search internals — `combined_motion_search` calls
  `vp9_full_pixel_search` and `cpi->find_fractional_mv_step`; the inner
  algorithms (NSTEP, FAST_DIAMOND, sub-pel tree, etc.) and
  `vp9_int_pro_motion_estimation` are **A3**.
- Reference-frame selection logic past the `ref_mode_set` loop, MV-pred
  details — A3.
- Bitstream/entropy/probability tables (e.g. `vp9_inter_mode_tree`,
  `vp9_intra_mode_tree`, `vp9_kf_y_mode_prob`) — A4.
- Rate control, AQ modes (cyclic refresh, complexity, 360, variance),
  denoiser, noise estimate — A5. (We *do* describe how the non-RD path
  *reads* those signals, but not how they are computed.)
- The `set_rt_speed_feature_framesize_independent` master dispatch table
  — A6. (We extract only the entries that gate non-RD-path behaviour.)
- Top-level encoder lifecycle, `VP9_COMP` god-struct — A1.

---

## 2. Data structures

### 2.1 `BEST_PICKMODE`

The per-superblock "current best" container used by `vp9_pick_inter_mode`.

```
vp9/encoder/vp9_pickmode.c:44
typedef struct {
  PRED_BUFFER *best_pred;
  PREDICTION_MODE best_mode;
  TX_SIZE best_tx_size;
  TX_SIZE best_intra_tx_size;
  MV_REFERENCE_FRAME best_ref_frame;
  MV_REFERENCE_FRAME best_second_ref_frame;
  uint8_t best_mode_skip_txfm;
  INTERP_FILTER best_pred_filter;
} BEST_PICKMODE;
```

Stack-allocated in `vp9_pick_inter_mode` (line 1695). Initialised by
`init_best_pickmode` (line 1674) to:
- `best_mode = ZEROMV`
- `best_ref_frame = LAST_FRAME`
- `best_tx_size = TX_SIZES` (sentinel)
- `best_intra_tx_size = TX_SIZES`
- `best_pred_filter = EIGHTTAP`
- `best_mode_skip_txfm = SKIP_TXFM_NONE`

### 2.2 `PRED_BUFFER`

Pool of luma prediction buffers reused across modes when
`sf.reuse_inter_pred_sby` is set (enabled at speed ≥ 5).

```
vp9/encoder/vp9_pickmode.c:38
typedef struct {
  uint8_t *data;
  int stride;
  int in_use;
} PRED_BUFFER;
```

A 4-element array `PRED_BUFFER tmp[4]` is declared on the stack of
`vp9_pick_inter_mode` (line 1728). `tmp[0..2]` point into a local
`uint8_t pred_buf[3*64*64]` aligned to 16 bytes (line 1729); `tmp[3]`
shadows the actual destination plane. `get_pred_buffer`/`free_pred_buffer`
(lines 916, 928) maintain `in_use`.

### 2.3 `REF_MODE`, `ref_mode_set`, `ref_mode_set_svc`

The fixed ordering of (ref_frame, prediction_mode) pairs that the non-RD
inter-mode loop walks through:

```
vp9/encoder/vp9_pickmode.c:1232
typedef struct {
  MV_REFERENCE_FRAME ref_frame;
  PREDICTION_MODE pred_mode;
} REF_MODE;

#define RT_INTER_MODES 12
static const REF_MODE ref_mode_set[RT_INTER_MODES] = {
  { LAST_FRAME, ZEROMV },   { LAST_FRAME, NEARESTMV },
  { GOLDEN_FRAME, ZEROMV }, { LAST_FRAME, NEARMV },
  { LAST_FRAME, NEWMV },    { GOLDEN_FRAME, NEARESTMV },
  { GOLDEN_FRAME, NEARMV }, { GOLDEN_FRAME, NEWMV },
  { ALTREF_FRAME, ZEROMV }, { ALTREF_FRAME, NEARESTMV },
  { ALTREF_FRAME, NEARMV }, { ALTREF_FRAME, NEWMV }
};

#define RT_INTER_MODES_SVC 8
static const REF_MODE ref_mode_set_svc[RT_INTER_MODES_SVC] = {
  { LAST_FRAME, ZEROMV },      { LAST_FRAME, NEARESTMV },
  { LAST_FRAME, NEARMV },      { GOLDEN_FRAME, ZEROMV },
  { GOLDEN_FRAME, NEARESTMV }, { GOLDEN_FRAME, NEARMV },
  { LAST_FRAME, NEWMV },       { GOLDEN_FRAME, NEWMV }
};
```

The loop counter `idx` walks 0..`num_inter_modes + comp_modes - 1` at
`vp9_pickmode.c:2039`; `num_inter_modes` is set to one of the two
constants above at line 1760 depending on `cpi->use_svc`.

### 2.4 `mode_idx` and `intra_mode_list`

Maps a `(ref_frame, mode_offset)` pair to a `THR_MODES` slot index into
`cpi->rd.threshes[seg][bsize][...]`.

```
vp9/encoder/vp9_pickmode.c:1088
static const THR_MODES mode_idx[MAX_REF_FRAMES][4] = {
  { THR_DC, THR_V_PRED, THR_H_PRED, THR_TM },
  { THR_NEARESTMV, THR_NEARMV, THR_ZEROMV, THR_NEWMV },
  { THR_NEARESTG, THR_NEARG, THR_ZEROG, THR_NEWG },
  { THR_NEARESTA, THR_NEARA, THR_ZEROA, THR_NEWA },
};

static const PREDICTION_MODE intra_mode_list[] = { DC_PRED, V_PRED, H_PRED,
                                                   TM_PRED };
```

`mode_offset` (line 1098) maps a `PREDICTION_MODE` to a column index in
`mode_idx`: `INTER_OFFSET(mode)` for inter modes (i.e. `mode-NEARESTMV`),
or explicit cases 0..3 for `DC_PRED`/`V_PRED`/`H_PRED`/`TM_PRED`.

The full `THR_MODES` enum is at `vp9/encoder/vp9_rd.h:53` (transcribed
verbatim in §5.5).

### 2.5 `pos_shift_16x16`

Lookup used in `get_force_skip_low_temp_var` to index into the 25-entry
`variance_low[]` flag array.

```
vp9/encoder/vp9_pickmode.c:55
static const int pos_shift_16x16[4][4] = {
  { 9, 10, 13, 14 }, { 11, 12, 15, 16 }, { 17, 18, 21, 22 }, { 19, 20, 23, 24 }
};
```

### 2.6 Variance tree (`v4x4`/`v8x8`/`v16x16`/`v32x32`/`v64x64`)

The recursive variance-tree node types used by `choose_partitioning`.

```
vp9/encoder/vp9_encodeframe.c:344
typedef struct {
  uint32_t sum_square_error;
  int32_t sum_error;
  int log2_count;
  int variance;
} Var;

typedef struct {
  Var none;
  Var horz[2];
  Var vert[2];
} partition_variance;

typedef struct {
  partition_variance part_variances;
  Var split[4];
} v4x4;

typedef struct {
  partition_variance part_variances;
  v4x4 split[4];
} v8x8;

typedef struct {
  partition_variance part_variances;
  v8x8 split[4];
} v16x16;

typedef struct {
  partition_variance part_variances;
  v16x16 split[4];
} v32x32;

typedef struct {
  partition_variance part_variances;
  v32x32 split[4];
} v64x64;

typedef struct {
  partition_variance *part_variances;
  Var *split[4];
} variance_node;

typedef enum {
  V16X16,
  V32X32,
  V64X64,
} TREE_LEVEL;
```

The whole `v64x64 vt` lives on the stack of `choose_partitioning`
(`vp9_encodeframe.c:1266`); a separate `v16x16 *vt2` is heap-allocated
(via `vpx_calloc`, line 1556) only when low-resolution 4x4-avg
downsampling is required.

`fill_variance` (line 440) writes a single `Var`, `get_variance`
(line 446) finalises `variance = (256*(sum_square_error -
sum_error²/2^log2_count)) >> log2_count`, and `sum_2_variances`
(line 454) combines two children into a parent (asserting both have the
same `log2_count`). `fill_variance_tree` (line 460) does the full
4-children combine into `none`/`horz[2]`/`vert[2]`.

### 2.7 `Diff`

For source-variance-based partition (speed feature
`SOURCE_VAR_BASED_PARTITION`). Defined in `vp9_encoder.h`; used in
`set_source_var_based_partition` at `vp9_encodeframe.c:2423` and
`source_var_based_partition_search_method` at line 5804. Not
load-bearing for the cpu-used 6-8 path (which uses
`VAR_BASED_PARTITION`/`REFERENCE_PARTITION`); included for completeness:

```
vp9/encoder/vp9_encoder.h  (typedef Diff)
  unsigned int sse;
  int sum;
  unsigned int var;
```
(See A1 for the `VP9_COMP` cross-reference; `cpi->source_diff_var` is
the array of `Diff` records.)

### 2.8 `PICK_MODE_CONTEXT` / `PC_TREE`

Per-partition node held by the `pc_root` tree. Lives across the whole
SB partition decision. Verbatim from
`vp9/encoder/vp9_context_tree.h:26-97`:

```
typedef struct {
  MODE_INFO mic;
  MB_MODE_INFO_EXT mbmi_ext;
  uint8_t *zcoeff_blk;
  tran_low_t *coeff[MAX_MB_PLANE][3];
  tran_low_t *qcoeff[MAX_MB_PLANE][3];
  tran_low_t *dqcoeff[MAX_MB_PLANE][3];
  uint16_t *eobs[MAX_MB_PLANE][3];

  // dual buffer pointers, 0: in use, 1: best in store
  tran_low_t *coeff_pbuf[MAX_MB_PLANE][3];
  tran_low_t *qcoeff_pbuf[MAX_MB_PLANE][3];
  tran_low_t *dqcoeff_pbuf[MAX_MB_PLANE][3];
  uint16_t *eobs_pbuf[MAX_MB_PLANE][3];

  int is_coded;
  int num_4x4_blk;
  int skip;
  int pred_pixel_ready;
  int skippable;
  uint8_t skip_txfm[MAX_MB_PLANE << 2];
  int best_mode_index;
  int hybrid_pred_diff;
  int comp_pred_diff;
  int single_pred_diff;
  int64_t best_filter_diff[SWITCHABLE_FILTER_CONTEXTS];

  int rate;
  int64_t dist;
  int64_t rdcost;

#if CONFIG_VP9_TEMPORAL_DENOISING
  unsigned int newmv_sse;
  unsigned int zeromv_sse;
  unsigned int zeromv_lastref_sse;
  PREDICTION_MODE best_sse_inter_mode;
  int_mv best_sse_mv;
  MV_REFERENCE_FRAME best_reference_frame;
  MV_REFERENCE_FRAME best_zeromv_reference_frame;
  int sb_skip_denoising;
#endif

  MV pred_mv[MAX_REF_FRAMES];
  INTERP_FILTER pred_interp_filter;

  int32_t sum_y_eobs;
  uint8_t skip_ref_frame_mask;
} PICK_MODE_CONTEXT;

typedef struct PC_TREE {
  int index;
  PARTITION_TYPE partitioning;
  BLOCK_SIZE block_size;
  PICK_MODE_CONTEXT none;
  PICK_MODE_CONTEXT horizontal[2];
  PICK_MODE_CONTEXT vertical[2];
  union {
    struct PC_TREE *split[4];
    PICK_MODE_CONTEXT *leaf_split[4];
  };
  MV mv;
} PC_TREE;
```

The non-RD writes that matter most: in `nonrd_pick_partition` it stamps
`mic = *xd->mi[0]`, `mbmi_ext = *x->mbmi_ext`, `skip_txfm[0] =
x->skip_txfm[0]`, `skip = x->skip`, and `pred_pixel_ready = 1` for the
chosen partition (see `vp9_encodeframe.c:5003-5006`, `5076-5083`,
`5120-5126`).

### 2.9 `MACROBLOCK` fields read or written by the non-RD path

(Field-by-field semantics; types from `vp9/encoder/vp9_block.h`.)

```
vp9/encoder/vp9_block.h:159
  uint8_t skip_txfm[MAX_MB_PLANE << 2];
   // [0] is luma; SKIP_TXFM_NONE=0, SKIP_TXFM_AC_DC=1, SKIP_TXFM_AC_ONLY=2

vp9/encoder/vp9_block.h:177
  uint8_t color_sensitivity[2];     // 0=U, 1=V; set by chroma_check

vp9/encoder/vp9_block.h:179
  uint8_t sb_is_skin;               // skin_sb_split result

vp9/encoder/vp9_block.h:181
  uint8_t skip_low_source_sad;      // set when content_state_sb in {1,2}

vp9/encoder/vp9_block.h:183
  uint8_t lowvar_highsumdiff;       // content_state_sb == 5

vp9/encoder/vp9_block.h:185
  uint8_t last_sb_high_content;     // 1-byte counter from content_state_sb_fd

vp9/encoder/vp9_block.h:187
  int sb_use_mv_part;               // mi->mv from int_pro is reusable
  int sb_mvcol_part;                // captured mv col (1/8 pel)
  int sb_mvrow_part;
  int sb_pickmode_part;             // ML/reference partition was used

vp9/encoder/vp9_block.h:195
  int zero_temp_sad_source;         // SAD(src, last_src) == 0
  uint8_t content_state_sb;         // CONTENT_STATE_SB enum value
  uint8_t variance_low[25];         // bit flags for low-temp-var blocks

vp9/encoder/vp9_block.h:206
  uint8_t arf_frame_usage;
  uint8_t lastgolden_frame_usage;
  // fn ptrs:
  void (*fwd_txfm4x4)(...);
  void (*inv_txfm_add)(...);
  DECLARE_ALIGNED(16, uint8_t, est_pred[64 * 64]);
```

`variance_low[25]` semantics, per comment at `vp9_block.h:202-203`:
- `[0]` — 64×64
- `[1..2]` — 64×32 (top, bottom)
- `[3..4]` — 32×64 (left, right)
- `[5..8]` — 32×32 (TL, TR, BL, BR)
- `[9..24]` — 16×16 in row-major order through the 4×4 grid of 16×16
  blocks within the 64×64 SB. Indexing is via `pos_shift_16x16[i][j]`
  in `get_force_skip_low_temp_var` (line 1472).

### 2.10 `CONTENT_STATE_SB`

```
vp9/encoder/vp9_encoder.h:137
typedef enum {
  kInvalid = 0,
  kLowSadLowSumdiff = 1,
  kLowSadHighSumdiff = 2,
  kHighSadLowSumdiff = 3,
  kHighSadHighSumdiff = 4,
  kLowVarHighSumdiff = 5,
  kVeryHighSad = 6,
} CONTENT_STATE_SB;
```

Assigned by `avg_source_sad` (`vp9_encodeframe.c:1209`), see §4.6.

---

## 3. Control flow

### 3.1 Entry point — `encode_nonrd_sb_row`

`vp9_encode_sb_row` (`vp9_encodeframe.c:5906`) dispatches to either
`encode_rd_sb_row` (non-RT path, not in scope) or `encode_nonrd_sb_row`
(`vp9_encodeframe.c:5524`) based on `sf->use_nonrd_pick_mode`.
For each SB column at this `mi_row`:

```
encode_nonrd_sb_row(cpi, td, tile_data, mi_row, tp):
  for mi_col = mi_col_start; mi_col < mi_col_end; mi_col += MI_BLOCK_SIZE:
    row_mt_sync_read()
    if cpi->use_skin_detection: vp9_compute_skin_sb(BLOCK_16X16, mi_row, mi_col)
    reset x->{source_variance, pred_mv[*], color_sensitivity[*],
              sb_is_skin, skip_low_source_sad, lowvar_highsumdiff,
              content_state_sb, zero_temp_sad_source, sb_use_mv_part,
              sb_mv{col,row}_part, sb_pickmode_part, arf_frame_usage,
              lastgolden_frame_usage}
    if cpi->compute_source_sad_onepass && sf.use_source_sad:
      avg_source_sad()                              // sets content_state_sb
      maybe switch to REFERENCE_PARTITION
    if seg->enabled && SEG_LVL_SKIP:
      partition_search_type = FIXED_PARTITION
    switch (partition_search_type):
      VAR_BASED_PARTITION:
        choose_partitioning(...)
        nonrd_use_partition(BLOCK_64X64, do_recon=1)
      ML_BASED_PARTITION:
        get_estimated_pred(...)
        x->{max,min}_partition_size = BLOCK_{64X64, 8X8}
        x->sb_pickmode_part = 1
        nonrd_pick_partition(BLOCK_64X64, do_recon=1)
      SOURCE_VAR_BASED_PARTITION:
        set_source_var_based_partition(...)
        nonrd_use_partition(BLOCK_64X64, do_recon=1)
      FIXED_PARTITION:
        set_fixed_partitioning(sf->always_this_block_size)
        nonrd_use_partition(BLOCK_64X64, do_recon=1)
      REFERENCE_PARTITION (default):
        x->sb_pickmode_part = 1
        if (VBR && high_source_sad && speed < 6 && (refresh_gf|alt)):
          x->{max,min}_partition_size = BLOCK_{64X64 or 32X32, 8X8}
          nonrd_pick_partition(BLOCK_64X64, do_recon=1)
        else:
          choose_partitioning(...)
          if (intra_only): nonrd_use_partition(BLOCK_64X64, 1, ...)
          else:            nonrd_select_partition(BLOCK_64X64, 1, ...)
    if ARF-tracking is on: cpi->count_{arf,lastgolden}_frame_usage[sboffset] = ...
    row_mt_sync_write()
```

Citations: outer loop `vp9_encodeframe.c:5545-5705`; partition-type
switch `5627-5690`.

### 3.2 `partition_search_type` by cpu-used (key dispatch only)

(For full speed-feature dispatch see A6. The values here are the ones
that gate non-RD-path control flow.)

| cpu-used | `use_nonrd_pick_mode` | `partition_search_type` (typical) | Source |
|---|---|---|---|
| 5 | 1 | `REFERENCE_PARTITION` (unless VBR+ARF group → `VAR_BASED_PARTITION`) | `vp9_speed_features.c:596-600` |
| 6 | 1 | `VAR_BASED_PARTITION` (default) — VBR+lag also sets `use_compound_nonrd_pickmode` | `vp9_speed_features.c:667` |
| 7 | 1 | `VAR_BASED_PARTITION` (inherits from 6) | (inherited) |
| 8 | 1 | `VAR_BASED_PARTITION` (8) or `ML_BASED_PARTITION` only at `<=352x288` non-key | `vp9_speed_features.c:751-768` |

At speed 6 and `sf.use_source_sad`, the per-SB
`adapt_partition_source_sad` (see `vp9_encodeframe.c:5586-5592`) may
**dynamically switch to `REFERENCE_PARTITION`** when SAD exceeds
`adapt_partition_thresh`. Threshold (line 674) is `40000` for ≤VGA,
`60000` for higher.

### 3.3 Partition selection: `VAR_BASED_PARTITION` path

```
choose_partitioning(cpi, tile, x, mi_row, mi_col)         // sb-types written
nonrd_use_partition(cpi, td, tile_data, mi, tp, mi_row, mi_col,
                    BLOCK_64X64, output_enabled=1, &dummy_rdc, pc_root)
```

`nonrd_use_partition` (`vp9_encodeframe.c:5317`) reads partition from
`mi[0]->sb_type` and recurses. At each leaf it calls
`nonrd_pick_sb_modes` followed by `encode_b_rt`. Note this is **always
do-recon**: no rate/distortion comparison happens above the
mode-decision level — `choose_partitioning`'s decision is final.

### 3.4 Partition selection: `REFERENCE_PARTITION` path

```
choose_partitioning(...)
nonrd_select_partition(BLOCK_64X64, output_enabled=1, ...)
```

`nonrd_select_partition` (`vp9_encodeframe.c:5183`) differs from
`use_partition` by:
- For 32×32 with no split chosen by `choose_partitioning`, it
  **expands** with `nonrd_pick_partition` setting `max_partition_size =
  BLOCK_32X32, min_partition_size = BLOCK_16X16` (line 5207-5210).
- For 32×32 with split + `subsize >= subsize_ref` (16x16 unless
  `adapt_partition_source_sad`), `min_partition_size = BLOCK_8X8`
  (line 5211-5216).
- For 16×16 with split, `max=16, min=8` (line 5217-5221).
- All leaves call `nonrd_pick_sb_modes`; encode is deferred to
  `encode_sb_rt` at the 64×64 level.

### 3.5 Partition selection: `ML_BASED_PARTITION` path (speed 8 / low-res only)

```
get_estimated_pred(BLOCK_64X64)                  // fills x->est_pred[64*64]
x->{max,min}_partition_size = BLOCK_{64X64, 8X8}
nonrd_pick_partition(BLOCK_64X64, do_recon=1)
```

`nonrd_pick_partition` (`vp9_encodeframe.c:4922`) is the recursive
4-way RD-style partition search reused for non-RD: it tries
PARTITION_NONE, _SPLIT, _HORZ, _VERT and picks the lowest-RD-cost
combination. ML prediction (line 4984-4992) calls
`ml_predict_var_partitioning` which uses three NN configs
(`vp9_var_part_nnconfig_64`, `_32`, `_16`) declared in
`vp9_partition_models.h` to force NONE/SPLIT/uncertain — features used
are `logf(dc_q*dc_q/256+1)`, `logf(var+1)`, and 4 quarter-variance
ratios (lines 4879-4910).

`do_rect = 0 if speed >= 5` (line 4957) so the non-RD pick partition
only tries NONE and SPLIT at cpu-used 5+.

### 3.6 `nonrd_pick_sb_modes` — the mode-decision dispatcher

```
vp9/encoder/vp9_encodeframe.c:4711
nonrd_pick_sb_modes(cpi, tile_data, x, mi_row, mi_col, rd_cost, bsize, ctx):
  set_offsets(cpi, tile, x, mi_row, mi_col, bsize)
  set_segment_index(cpi, x, mi_row, mi_col, bsize, 0)
  x->skip_recode = 0
  mi->sb_type = bsize
  // save left/above contexts...
  if (aq_mode == CYCLIC_REFRESH_AQ && CR-boosted): x->rdmult = cr_rdmult
  if frame_is_intra_only(cm):
    hybrid_intra_mode_search(...)
  else if svc.is_key_frame at this temporal layer:
    hybrid_search_svc_baseiskey(...)
  else if SEG_LVL_SKIP:
    set_mode_info_seg_skip(...)
  else if bsize >= BLOCK_8X8:
    if rc.hybrid_intra_scene_change: hybrid_search_scene_change(...)
    else:                            vp9_pick_inter_mode(...)
  else (bsize < 8x8):
    vp9_pick_inter_mode_sub8x8(...)
  // restore contexts; record rate/dist into ctx
```

`hybrid_intra_mode_search` (`vp9_encodeframe.c:4672`) routes to
`vp9_rd_pick_intra_mode_sb` (RDO) for small blocks unless
`sf.nonrd_keyframe`; otherwise to `vp9_pick_intra_mode`.

`hybrid_search_svc_baseiskey` and `hybrid_search_scene_change`
(`vp9_encodeframe.c:4681`, `4699`) similarly fall back to full-RDO for
small blocks unless `sf.nonrd_keyframe == 1`.

### 3.7 `vp9_pick_inter_mode` — the heart of the non-RD path

Call graph for one bsize ≥ 8×8 inter block:

```
vp9_pick_inter_mode(cpi, x, tile_data, mi_row, mi_col, rd_cost, bsize, ctx):
  init_best_pickmode(&best_pickmode)
  read x->encode_breakout, source_variance (if needed), reset best_rdc
  init_ref_frame_cost(cm, xd, ref_frame_cost)
  for ref_frame in [LAST_FRAME .. usable_ref_frame]:
    if !skip_ref_find_pred[ref_frame]:
      find_predictors(...)              // populates frame_mv[*][ref], yv12_mb,
                                        //  computes pred_mv_sad via vp9_mv_pred
  large_block, use_model_yrd_large = ...
  for idx in [0 .. num_inter_modes + comp_modes):
    (ref_frame, this_mode) = ref_mode_set[idx] (or _svc)
    apply all the skip rules (see §4.2)
    if this_mode == NEWMV:
      search_new_mv(...)                // returns -1 to skip
    apply duplicate-MV skip
    if (NEWMV or filter==SWITCHABLE) and sub-pel mv:
      search_filter_ref(...)            // tries EIGHTTAP{,_SMOOTH}, picks best
    else:
      vp9_build_inter_predictors_sby()
      if large_block: model_rd_for_sb_y_large(...) (may set early_term)
      else:            model_rd_for_sb_y(...)
    if !early_term:
      block_yrd(...)                    // Hadamard+SATD model (see §4.3)
      adjust rate by skip prob
    if color_sensitivity: model_rd_for_sb_uv(...)
    add rate_mv + inter_mode_cost + ref_frame_cost
    apply vp9_NEWMV_diff_bias if CBR && speed>=5 && !screen
    encode_breakout_test(...)           // may set x->skip = 1
    update best if this_rdc.rdcost < best_rdc.rdcost
    break if x->skip; break if best_early_term && idx > 0

  apply intra-search gate (see §4.4):
    if best_rdc.rdcost > intra_threshold && perform_intra_pred:
      for mode in {DC_PRED, V_PRED, H_PRED, TM_PRED}:
        skip per intra_y_mode_bsize_mask, rt_intra_dc_only_low_content,
             short_circuit_flat_blocks, adaptive_rd_thresh
        compute_intra_yprediction(...)
        model_rd_for_sb_y(... is_intra=1)
        block_yrd(... is_intra=1)
        if better, update best_pickmode

  finalise mi-> fields from best_pickmode
  if reuse_inter_pred && best is inter: copy buffer to orig_dst
  CONFIG_VP9_TEMPORAL_DENOISING: denoise + recheck_zeromv_after_denoising
  update arf_frame_usage / lastgolden_frame_usage
  if sf.adaptive_rd_thresh: update_thresh_freq_fact[_row_mt] (§4.5)
  *rd_cost = best_rdc
```

Citations: outer function `vp9_pickmode.c:1685-2742`; main loop body
`2039-2478`; intra search block `2516-2645`; finalisation `2647-2740`.

### 3.8 `vp9_pick_inter_mode_sub8x8` — sub-8×8 path (key when `bsize < 8x8`)

```
vp9/encoder/vp9_pickmode.c:2744
vp9_pick_inter_mode_sub8x8(cpi, x, mi_row, mi_col, rd_cost, bsize, ctx):
  for ref_frame in [LAST_FRAME, GOLDEN_FRAME]:
    if usable: vp9_find_mv_refs(); vp9_find_best_ref_mvs()
    else:      ref_frame_skip_mask |= (1<<ref_frame)
  set mi->{sb_type, tx_size=TX_4X4, uv_mode=DC_PRED, ref_frame[0]=LAST,
           interp_filter = (SWITCHABLE? EIGHTTAP : cm->interp_filter)}
  for ref_frame in [LAST, GOLDEN] (skipping disallowed/scaled):
    for each 4x4 sub-position i in {0,1,2,3} within the 8x8:
      b_mv[ZEROMV] = 0; b_mv[NEWMV] = INVALID; vp9_append_sub8x8_mvs_for_idx()
      for this_mode in [NEARESTMV..NEWMV]:
        if NEWMV:
          vp9_set_mv_search_range(...)
          vp9_full_pixel_search(sadperbit4)            // A3
          if rate budget exceeded for nearestmv: continue
          cpi->find_fractional_mv_step(...)            // A3
        else:
          b_rate += inter_mode_cost[mode_ctx][INTER_OFFSET(this_mode)]
        build inter predictor for this 4x4 sub-block at interp_filter
        model_rd_for_sb_y(...)                         // simple model
        this_rdc.rate += b_rate
        update b_best_rd per-sub-block
      this_rd += b_best_rd
    if this_rd < best_rd: best_ref_frame = ref_frame
  set mi->tx_size = TX_4X4, mi->ref_frame[0] = best_ref_frame
  copy bsi[best_ref_frame][*] into xd->mi[0]->bmi[*]
  mi->mode = xd->mi[0]->bmi[3].as_mode       // mode of last sub-block
  rd_cost->rdcost = best_rd
```

Key simplifications vs `vp9_pick_inter_mode`:
- No intra check, no ALTREF, no compound, no encode_breakout, no
  filter search (always uses cm->interp_filter / EIGHTTAP), no
  skip-detection / SKIP_TXFM tracking, no denoiser.
- The chosen mode is the **last** sub-block's mode in raster order
  (line 2985: `mi->mode = xd->mi[0]->bmi[3].as_mode`). That's the
  per-MI mode that becomes part of the bitstream; per-sub-block MVs
  travel in `bmi[]`.

### 3.9 `encode_b_rt` / `encode_sb_rt` / `update_state_rt`

After mode decision, encoding is mechanical recursion over the
partition tree. `encode_b_rt` (`vp9_encodeframe.c:2579`) does:

```
set_offsets(cpi, tile, x, mi_row, mi_col, bsize)
update_state_rt(cpi, td, ctx, mi_row, mi_col, bsize)   // copies mic, mbmi_ext;
                                                      //  updates mv_count,
                                                      //  cur_frame mvs grid
encode_superblock(cpi, td, tp, output_enabled, ...)
update_stats(cm, td)
**tp = EOSB_TOKEN; (*tp)++
```

`encode_sb_rt` (`vp9_encodeframe.c:2594`) reads `mi[0]->sb_type` and
the `partition_lookup` to recurse with one of NONE/HORZ/VERT/SPLIT,
calling `encode_b_rt` at each leaf and finally
`update_partition_context`.

### 3.10 Flowchart — partition decision for a 64×64 SB at speed 6/7 (default `VAR_BASED_PARTITION`)

```
choose_partitioning(SB at (mi_row, mi_col)):
  ┌── set_offsets, set_segment_index
  ├── memset x->variance_low = 0
  ├── if (use_source_sad && !is_key_frame):
  │     content_state = x->content_state_sb     // from avg_source_sad
  │     skip_low_source_sad = content_state in {1,2}
  │     lowvar_highsumdiff  = content_state == 5
  │     if (svc_use_lowres_part): scale_partitioning_svc() → maybe return 0
  │     if (skip_low_source_sad && copy_partition_flag && !force_64_split):
  │       copy_partitioning() succeeds → sb_use_mv_part=1; return 0
  │
  ├── set_vbp_thresholds(thresholds, q, content_state)
  │
  ├── if (!is_key_frame):
  │     bsize = BLOCK_32X32 + (mi_col+4<mi_cols)*2 + (mi_row+4<mi_rows)
  │     y_sad_g (only if speed<8): SAD(src, golden_pred)
  │     if (speed>=8 && !low_res && state!=kVeryHighSad):
  │       y_sad = SAD(src, last_pred at zero MV)
  │     else:
  │       y_sad = vp9_int_pro_motion_estimation(LAST, BLOCK_64x64)
  │       x->sb_use_mv_part=1; capture mv as sb_mv{col,row}_part
  │     y_sad_thr = short_circuit_low_temp_var? (y_sad*7)>>3 : y_sad
  │     if (y_sad_g < y_sad_thr):
  │       ref = GOLDEN; mi->mv = 0
  │     else:
  │       ref = LAST; x->pred_mv[LAST] = mi->mv
  │     vp9_build_inter_predictors_sb()
  │     if (use_skin_detection): skin_sb_split → maybe force_split = 1
  │     d = xd->plane[0].dst.buf      // reconstructed prediction
  │     if (CR_seg_id==base && y_sad < vbp_threshold_sad):
  │       set 64×64 partition; x->variance_low[0]=1; chroma_check; return 0
  │     if (copy_partition_flag && y_sad_last < vbp_threshold_copy &&
  │         copy_partitioning()):
  │       chroma_check; return 0
  │
  ├── else:    // is_key_frame
  │     d = VP9_VAR_OFFS[64] = {128, 128, ..., 128}   // intra ref is "128"
  │
  ├── for i in 0..3:                       // each 32×32
  │     for j in 0..3:                     // each 16×16
  │       fill_variance_8x8avg → 4 × (s_avg - d_avg) squared at 8x8 granularity
  │       fill_variance_tree(BLOCK_16X16)
  │       get_variance(16x16.none)
  │       if (16x16.variance > thresholds[2]):
  │         force_split[5+i*4+j]=1; force_split[i+1]=1; force_split[0]=1
  │       elif (compute_minmax_variance && 16x16.variance > thresholds[1] &&
  │             !CR_boosted):
  │         minmax = compute_minmax_8x8()  // max-min of 8x8 averages
  │         thresh_minmax = vbp_threshold_minmax (<<1 if kVeryHighSad)
  │         if (minmax > thresh_minmax): force_split everything
  │     if (is_key_frame || (low_res && 16x16.variance > threshold_4x4avg)):
  │       force_split[16x16] = 0
  │       variance4x4downsample = 1; fill_variance_4x4avg over 4 sub-8x8
  │
  ├── for i in 0..3:                       // each 32×32 (combine)
  │     if (variance4x4downsample): fill_variance_tree(8x8) for each sub, then
  │       fill_variance_tree(16x16)
  │       if (variance > thresholds[2]): force_split everything
  │     fill_variance_tree(BLOCK_32X32)
  │     if (!force_split[i+1]):
  │       get_variance(32x32.none)
  │       if (variance > thresholds[1] ||
  │           (!is_key && variance > thresholds[1]>>1 &&
  │                       variance > avg_16x16[i]>>1)):
  │         force_split[i+1]=1; force_split[0]=1
  │       elif (!is_key && noise<kLow && height<=360 &&
  │             max16-min16 > thresholds[1]>>1 && max16 > thresholds[1]):
  │         force_split[i+1]=1; force_split[0]=1
  │
  ├── if (!force_split[0]):
  │     fill_variance_tree(BLOCK_64X64)
  │     get_variance(64x64.none)
  │     if (!is_key && noise>=kMedium && variance > (9*avg_32x32)>>5):
  │       force_split[0]=1
  │     elif (!is_key && noise<kMedium &&
  │           max_var_32x32 - min_var_32x32 > 3*(thresholds[0]>>3) &&
  │           max_var_32x32 > thresholds[0]>>1):
  │       force_split[0]=1
  │
  ├── set_vt_partitioning recursive walk:
  │     at 64x64, threshold = thresholds[0], min_bsize = BLOCK_16X16
  │     at 32x32, threshold = thresholds[1], min_bsize = BLOCK_16X16
  │     at 16x16, threshold = thresholds[2], min_bsize = vbp_bsize_min
  │     at 8x8 (only is_key && use_4x4_partition),
  │              threshold = thresholds[3], min_bsize = BLOCK_8X8
  │   set_vt_partitioning logic (§4.7) picks PARTITION_NONE / horz / vert /
  │   recurses to split.
  │
  ├── if (copy_partition_flag && !is_key): update_prev_partition(...)
  ├── if (svc_use_lowres_part && spatial_layer == N-2): update_partition_svc()
  ├── if (short_circuit_low_temp_var):
  │     set_low_temp_var_flag() — sets x->variance_low[*] bits
  ├── chroma_check(y_sad, ...)
  └── return 0
```

Citations: function `vp9_encodeframe.c:1261-1766`; key thresholds set by
`set_vbp_thresholds` at `573-635`.

---

## 4. Algorithms

### 4.1 Variance-based partition thresholds (`set_vbp_thresholds`)

The 4-entry `thresholds[0..3]` (64/32/16/8) array drives both
`force_split` decisions and the leaf-acceptance test in
`set_vt_partitioning`. Computed at `vp9_encodeframe.c:573`:

```
threshold_multiplier = is_key_frame ? 20 : sf.variance_part_thresh_mult
threshold_base       = threshold_multiplier * cpi->y_dequant[q][1]

if is_key_frame:
  thresholds = [tb, tb>>2, tb>>2, tb<<2]
else:
  if (noise_estimate.enabled && cm->width>=640 && cm->height>=480):
    tb *= 3 if noise==kHigh
    tb <<= 1 if noise==kMedium
    tb = (7*tb)>>3 if noise<kLow
  if (denoiser && speed>5 && denoising_level>=kDenLow):
    tb = vp9_scale_part_thresh(tb, level, content_state, temporal_layer)
  else:
    tb = scale_part_thresh_sumdiff(tb, speed, w, h, content_state)
  thresholds[0] = tb
  thresholds[2] = tb << speed
  if (w>=1280 && h>=720 && speed<7):
    thresholds[2] <<= 1
  if (w<=352 && h<=288):
    thresholds[0] = tb>>3
    thresholds[1] = tb>>1
    thresholds[2] = tb<<3
    if avg_frame_qindex[INTER]>220: thresholds[2] <<= 2
    elif avg_frame_qindex[INTER]>200: thresholds[2] <<= 1
  elif (w<1280 && h<720):
    thresholds[1] = (5*tb)>>2
  elif (w<1920 && h<1080):
    thresholds[1] = tb<<1
  else:
    thresholds[1] = (5*tb)>>1
  if (sf.disable_16x16part_nonkey): thresholds[2] = INT64_MAX
```

Note `thresholds[3]` (the 8×8 threshold, used only when descending to
4×4 on key frames) is `threshold_base<<2` from the key-frame branch and
never reset for inter frames.

`scale_part_thresh_sumdiff` (line 549):
```
if speed>=8:
  if w<=640 && h<=480: return (5*tb)>>2
  elif content_state in {kLowSadLowSumdiff, kHighSadLowSumdiff,
                         kLowVarHighSumdiff}: return (5*tb)>>2
elif speed==7:
  if content_state in same set: return (5*tb)>>2
return tb
```

`vp9_set_variance_partition_thresholds` (line 637) wraps
`set_vbp_thresholds` and additionally sets:
- `cpi->vbp_threshold_sad`: 10 for ≤QCIF; else `max(y_dequant[q][1]<<1, 1000)`
- `cpi->vbp_bsize_min`: `BLOCK_8X8` (key) or `BLOCK_16X16` (inter)
- `cpi->vbp_threshold_copy`: 4000 (≤QCIF), 8000 (≤nHD), else
  `max(y_dequant[q][1]<<3, 8000)`
- `cpi->vbp_threshold_minmax`: `15 + (q>>3)`
- If `rc.high_source_sad || (svc && high_source_sad_superframe)`:
  `vbp_threshold_sad = vbp_threshold_copy = 0` (force split work).

### 4.2 `vp9_pick_inter_mode` — mode-decision algorithm (pseudo)

```
init:
  best_pickmode := {ZEROMV, LAST_FRAME, EIGHTTAP, SKIP_TXFM_NONE, ...}
  intra_cost_penalty := vp9_get_intra_cost_penalty(bsize, qindex, qdelta)
                       = (20 * vp9_dc_quant(q, qdelta, BITS_8)) >> reduction
                         reduction = 4 if bsize<=8x8 else 2 if <=16x16 else 0
                         (forced to 0 if noise_estimate.level == kHigh)
  inter_mode_thresh := RDCOST(intra_cost_penalty, 0)
  rd_threshes := cpi->rd.threshes[seg_id][bsize][]          // 30-entry
  rd_thresh_freq_fact := row_mt vs per-tile based on sf
  init_ref_frame_cost(ref_frame_cost[INTRA, LAST, GOLDEN, ALTREF])
  reuse_inter_pred := sf.reuse_inter_pred_sby && ctx->pred_pixel_ready

  if (sf.short_circuit_flat_blocks || sf.limit_newmv_early_exit):
    x->source_variance := vp9_get_sby_perpixel_variance(src, bsize)
    if (screen && CR-AQ && seg_id>0 && zero_temp_sad_source &&
        source_variance==0): demote to seg 0; re-init quantizers

usable_ref_frame:
  if (frames_since_golden==0 && gf_temporal_ref && !alt_ref_gf_group &&
      !last_frame_is_src_altref): usable_ref_frame := LAST_FRAME
  else: usable_ref_frame := GOLDEN_FRAME
  if (VBR && lag_in_frames>0):
    if (alt_ref_gf_group || is_src_frame_alt_ref): usable_ref_frame := ALTREF
    if (is_src_frame_alt_ref): skip LAST and GOLDEN
    if (!show_frame && frames_since_key==1): usable := LAST; skip G/A
  if (svc && force_zero_mode_spatial_ref && spatial>0 && !gf_temporal_ref):
    set svc_force_zero_mode[*] for scaled refs
  if (sf.short_circuit_low_temp_var):
    force_skip_low_temp_var := get_force_skip_low_temp_var(variance_low, ...)
    if (mode in {1,3} && force_skip_low_temp_var): usable := LAST
  if (sf.disable_golden_ref && state!=kVeryHighSad && avg_frame_low_motion<60):
    usable := LAST
  if (speed>=8 && !svc && (frames_since_golden+1 < x->last_sb_high_content ||
                           x->last_sb_high_content > 40 ||
                           frames_since_golden > 120)):
    usable := LAST
  if (cm->reference_mode==REFERENCE_MODE_SELECT &&
      sf.use_compound_nonrd_pickmode && usable==ALTREF): comp_modes := 2
  if (seg has SEG_LVL_REF_FRAME == GOLDEN): usable := GOLDEN; thresh_skip_golden := 0

for ref in [LAST..usable]:
  skip_ref_find_pred[ref] = !(cpi->ref_frame_flags & ref_flag(ref))
  if !skip: find_predictors(ref, force_skip_low_temp_var, comp_modes>0)

large_block := (state==kVeryHighSad || (sb_is_skin && avg_frame_low_motion>70)
                || speed<7) ? bsize > BLOCK_32X32 : bsize >= BLOCK_32X32
use_model_yrd_large := (rc_mode==VPX_CBR && large_block &&
                        !CR_boosted && base_qindex)

main loop:
for idx in [0 .. num_inter_modes + comp_modes):
  unpack (this_mode, ref_frame) := ref_mode_set[idx] (or _svc)
  if idx >= num_inter_modes:                     // compound (0,0) on LAST or G
     this_mode = ZEROMV; ref_frame = LAST or GOLDEN; second_ref = ALTREF;
     comp_pred = 1
  skip-rules cascade (any continues):
    if (ref > usable) continue
    if (skip_ref_find_pred[ref]) continue
    if (previous_frame_is_intra_only && (ref!=LAST || mv!=0)) continue
    if (SEG_LVL_REF_FRAME && seg ref != ref_frame) continue
    if (flag_svc_subpel && ref==inter_layer_ref):
       enforce NEWMV=(svc_mv_col,row); skip non-NEWMV if mv != that
    if (comp_pred):
       if !allow_comp_inter_inter continue
       if !(ref_frame_flags & second_ref) continue
       if SEG_LVL_REF_FRAME continue
    if (ref==GOLDEN && rc_mode==VPX_CBR &&
        ((svc && sse_zeromv_normalized < thresh_svc_skip_golden) ||
         (!svc && sse_zeromv_normalized < thresh_skip_golden))) continue
    if !(ref_frame_flags & ref_flag(ref)) continue
    screen-content stationary-skip rules (lines 2125-2137)
    if !(sf.inter_mode_mask[bsize] & (1<<this_mode)) continue
    VBR-lag alt_ref_gf_group rules (lines 2141-2163)
    if (const_motion[ref] && this_mode==NEARMV) continue
    if (force_skip_low_temp_var && ref==GOLDEN && mv!=0) continue
    if (state!=kVeryHighSad &&
        ((sf.short_circuit_low_temp_var>=2) ||
         (sf.short_circuit_low_temp_var==1 && bsize==BLOCK_64X64)) &&
        force_skip_low_temp_var && ref==LAST && this_mode==NEWMV) continue
    if (svc && !force_mv_inter_layer && svc_force_zero_mode[ref] && mv!=0) continue
    if (sf.reference_masking && !(mv==0 && ref==LAST)):
       reference_masking heuristics (lines 2197-2214)
    if (ref_frame_skip_mask & (1<<ref)) continue

  set xd->plane[*].pre to yv12_mb[ref] (and second ref for comp)
  mi->ref_frame[0]=ref; mi->ref_frame[1]=second_ref; set_ref_ptrs

  mode_index := mode_idx[ref_frame][INTER_OFFSET(this_mode)]
  mode_rd_thresh := best_pickmode.best_mode_skip_txfm
                    ? rd_threshes[mode_index] << 1
                    : rd_threshes[mode_index]
  if (sf.bias_golden && ref==GOLDEN && frames_since_golden>4):
    mode_rd_thresh <<= 3
  if rd_less_than_thresh(best_rdc.rdcost, mode_rd_thresh, freq_fact[mode_index]):
    if (frame_mv[this_mode][ref].as_int != 0): continue

  if (this_mode == NEWMV && !force_mv_inter_layer):
    if search_new_mv(...) returns -1: continue            // calls A3

  duplicate-mv skip: if any earlier mode at this ref tested mv==0 same as this
  if (use_golden_nonzeromv && this_mode==NEWMV && ref==LAST && newmv valid):
    best_pred_sad := SAD(src, last_pred at newmv)
    x->pred_mv_sad[LAST] := best_pred_sad
  if (this_mode != NEARESTMV && !comp_pred && mv == nearest_mv): continue

  set mi->mode=this_mode; mi->mv[0] = frame_mv[this_mode][ref]; mi->mv[1]=0

  if (reuse_inter_pred):
    allocate this_mode_pred from tmp[] pool; redirect pd->dst to it

  if ((this_mode==NEWMV || filter_ref==SWITCHABLE) && pred_filter_search &&
      (ref==LAST || (ref==GOLDEN && !force_mv_inter_layer && (svc||vbr))) &&
      mv has sub-pel bits):
    rd_computed = 1
    search_filter_ref(...)   // tries EIGHTTAP{, _SMOOTH}, picks best
  else:
    mi->interp_filter := (filter_ref==SWITCHABLE? EIGHTTAP : filter_ref)
    (svc golden zero-mv overrides: filter_gf_svc)
    vp9_build_inter_predictors_sby()
    if use_model_yrd_large:
      model_rd_for_sb_y_large(...)        // 8x8-block variance test + skip
      may set early_term = 1 and return 0/0
    else:
      model_rd_for_sb_y(...)              // single-block variance, simpler
    if (ref==LAST && mv==0): sse_zeromv_normalized := sse_y >> (bw+bh)
    if (sse_y < best_sse_sofar): best_sse_sofar := sse_y

  if !early_term:
    block_yrd(this_rdc, &is_skippable, &this_sse,
              bsize, min(mi->tx_size, TX_16X16),
              rd_computed=rd_computed, is_intra=0)
    x->skip_txfm[0] = is_skippable
    if is_skippable: this_rdc.rate := cost_bit(skip_prob, 1)
    else:
      if RDCOST(this_rdc.rate, this_rdc.dist) < RDCOST(0, this_sse):
        this_rdc.rate += cost_bit(skip_prob, 0)
      else:
        this_rdc.rate := cost_bit(skip_prob, 1)
        this_rdc.dist := this_sse
        x->skip_txfm[0] := SKIP_TXFM_AC_DC
    if (cm->interp_filter==SWITCHABLE && mv sub-pel):
      this_rdc.rate += vp9_get_switchable_rate()
  else:                                            // early_term branch
    if (cm->interp_filter==SWITCHABLE && mv sub-pel):
      this_rdc.rate += vp9_get_switchable_rate()
    this_rdc.rate += cost_bit(skip_prob, 1)

  if !early_term && (color_sensitivity[0] || color_sensitivity[1]):
    build chroma predictors if not already done; model_rd_for_sb_uv()
    this_rdc.{rate,dist} += rdc_uv.{rate,dist}

  this_rdc.rate += rate_mv
  this_rdc.rate += cpi->inter_mode_cost[mode_ctx[ref]][INTER_OFFSET(this_mode)]
  this_rdc.rate += ref_frame_cost[ref]
  this_rdc.rdcost := RDCOST(rate, dist)

  if (rc_mode==VPX_CBR && speed>=5 && !screen):
    vp9_NEWMV_diff_bias(noise_estimate, xd, this_mode, &this_rdc, bsize,
                        mv.row, mv.col, ref==LAST, lowvar_highsumdiff, sb_is_skin)
                                                  // see §4.8

  if (allow_encode_breakout && !lossless && !scene_change &&
      !svc.high_num_blocks_with_motion):
    encode_breakout_test(...)                     // may set x->skip = 1
    if x->skip: re-cost as cost_bit(skip,1); recompute rdcost

  // screen-content extra penalty against zero-last on flat blocks at scene change
  if (screen && (scene_change||high_motion) && ref==LAST && mv==0 &&
      spatial==0 && source_variance==0 && sse_y>0):
    this_rdc.rdcost <<= 2

  CONFIG_VP9_TEMPORAL_DENOISING:
    update denoiser stats; if (ref==LAST && mv==0): zero_last_cost_orig := rdcost

  mode_checked[this_mode][ref] := 1
  if (this_rdc.rdcost < best_rdc.rdcost || x->skip):
    best_rdc := this_rdc
    best_early_term := this_early_term
    best_pickmode := {this_mode, mi->interp_filter, mi->tx_size, ref,
                      x->skip_txfm[0], second_ref}
    if (reuse_inter_pred): swap best_pred buffer
  else if (reuse_inter_pred): free this_mode_pred

  if x->skip && (!force_test_gf_zeromv || mode_checked[ZEROMV][GOLDEN]): break
  if (best_early_term && idx > 0 && !scene_change &&
      (!force_test_gf_zeromv || mode_checked[ZEROMV][GOLDEN])):
    x->skip := 1; break

// commit best inter pick
mi-> = best_pickmode.*

// intra eligibility — see §4.4
```

Citations: function body `1685-2742`; main mode-loop `2039-2478`;
encode-breakout `2414-2424`; NEWMV bias `2403-2410`; SKIP_TXFM
decision `2347-2375`.

### 4.3 `block_yrd` — Hadamard + SATD non-RD cost approximation

This is the **non-RD cost function** for luma. Replaces full forward
DCT + quantize + bit count + inverse DCT + SSE that the full-RDO path
does. Verbatim shape from `vp9_pickmode.c:718-844`:

```
block_yrd(cpi, x, this_rdc, &skippable, &sse, bsize, tx_size,
          rd_computed, is_intra):

  // Fast exit: at speed>=7 (use_simple_block_yrd) for non-key, small blocks,
  // skip Hadamard entirely and use model_rd_for_sb_y output.
  if (sf.use_simple_block_yrd && cm->frame_type != KEY_FRAME &&
      (bsize < BLOCK_32X32 ||
       (use_svc && (bsize < BLOCK_32X32 || svc.temporal_layer_id > 0)))):
    if (!rd_computed): model_rd_for_sb_y(...)
    *sse = INT_MAX
    *skippable = 0
    return                                       // rate, dist from model_rd

  assert(tx_size != TX_32X32)                   // max tx is TX_16X16 here

  vpx_subtract_block(bh, bw, src_diff, bw, src, src_stride, dst, dst_stride)
  *skippable = 1
  for r in 0..max_blocks_high step block_step:
    for c in 0..num_4x4_w step block_step:
      if c < max_blocks_wide:
        scan_order := vp9_default_scan_orders[tx_size]
        src_diff_ptr := src_diff + ((r*bw + c) << 2)
        switch tx_size:
          TX_16X16: vpx_hadamard_16x16(src_diff_ptr, bw, coeff)
                    vp9_quantize_fp(coeff, 256, p, qcoeff, dqcoeff,
                                     pd->dequant, eob, scan_order)
          TX_8X8:   vpx_hadamard_8x8(...);   vp9_quantize_fp(coeff, 64, ...)
          TX_4X4:   x->fwd_txfm4x4(src_diff_ptr, coeff, bw);
                    vp9_quantize_fp(coeff, 16, ...)
        *skippable &= (*eob == 0)
        eob_cost += 1
      block += step

  this_rdc->rate = 0
  if *sse < INT64_MAX:
    *sse = (*sse << 6) >> 2
    if *skippable: this_rdc->dist = *sse; return

  this_rdc->dist = 0
  for each block:
    eob = p->eobs[block]
    if eob == 1: this_rdc->rate += abs(qcoeff[0])
    elif eob > 1: this_rdc->rate += vpx_satd(qcoeff, step<<4)
    this_rdc->dist += vp9_block_error_fp(coeff, dqcoeff, step<<4) >> 2

  this_rdc->rate <<= (2 + VP9_PROB_COST_SHIFT)        // VP9_PROB_COST_SHIFT=9
  this_rdc->rate += (eob_cost << VP9_PROB_COST_SHIFT)
```

Notes:
- `vpx_hadamard_*` is the **Walsh–Hadamard** transform, not the actual
  DCT — choice driven by simpler SIMD and good-enough correlation with
  bit count.
- `vpx_satd` returns sum of absolute (Hadamard) transform coefficients
  — used as a rate proxy via the `<< (2 + VP9_PROB_COST_SHIFT)`
  scaling.
- `vp9_block_error_fp` returns `sum((coeff - dqcoeff)^2)` — used as
  distortion proxy, shifted right by 2 to match scale.
- `tx_size` is capped to `TX_16X16` by the caller (e.g.
  `vp9_pickmode.c:2350` `min(mi->tx_size, TX_16X16)`).

### 4.4 `model_rd_for_sb_y[_large]` — variance-based rate-distortion model

The smaller version (`vp9_pickmode.c:635-716`) is the default; the
`_large` version (`429-633`) is selected by `use_model_yrd_large` (CBR
+ large block + not CR-boosted + non-zero qindex) and additionally:
- partitions luma into 8×8 blocks via `block_variance()` (line 263),
- computes 16×16 and 32×32 aggregate variances via `calculate_variance`
  (line 316),
- runs a **per-tx_size skip test**: AC-skip iff every block's variance
  `<= ac_thr` OR overall variance is 0; DC-skip iff every block's
  `sse-var <= dc_thr` OR `sse==var`.
- May also test U/V plane variances and short-circuit `early_term=1` if
  all planes are skip-able (line 599-602).

The Laplacian R-D model itself lives in
`vp9/encoder/vp9_rd.c:518-539`:

```
vp9_model_rd_from_var_lapndz(var, n_log2, qstep, *rate, *dist):
  if var == 0: rate=0; dist=0; return
  xsq_q10_64 := ((qstep*qstep << (n_log2+10)) + var/2) / var
  xsq_q10    := min(xsq_q10_64, MAX_XSQ_Q10=245727)
  model_rd_norm(xsq_q10, &r_q10, &d_q10)         // table lookup (§5.4)
  *rate := round_pow2(r_q10 << n_log2, 10 - VP9_PROB_COST_SHIFT)
  *dist := (var * d_q10 + 512) >> 10
```

The model is parameterised by:
- `var` — measured block luma variance
- `n_log2` — `num_pels_log2_lookup[bsize]` (12 for 64×64, 6 for 8×8)
- `qstep` — `dequant >> 3` (8-bit) or `dequant >> (xd->bd - 5)` (HBD).

The `ac_thr` / `dc_thr` thresholds used in the model_rd skip tests in
the `_large` variant:
```
dc_thr := (dc_quant * dc_quant) >> 6
ac_thr := (ac_quant * ac_quant) >> 6 * ac_thr_factor(speed, w, h, |sum|>>(bw+bh))
```
where `ac_thr_factor` (line 342):
```
if speed>=8 && norm_sum<5:
  return 4 if w<=640 && h<=480 else 2
else: return 1
```
With denoiser on at speed>5 the `ac_thr` is overridden by
`vp9_scale_acskip_thresh(...)` (A5 owns).

For the smaller version (`model_rd_for_sb_y`):
```
dc_thr := p->quant_thred[0] >> 6
ac_thr := p->quant_thred[1] >> 6
var, sse := cpi->fn_ptr[bsize].vf(src, dst)              // variance fn
xd->mi[0]->tx_size := calculate_tx_size(...)             // see §4.10
num_blk_log2 := (bw - log2(unit_w)) + (bh - log2(unit_h))
sse_tx := sse >> num_blk_log2; var_tx := var >> num_blk_log2
if (var_tx < ac_thr || var==0):
  x->skip_txfm[0] = SKIP_TXFM_AC_ONLY
  if (sse_tx - var_tx < dc_thr || sse==var):
    x->skip_txfm[0] = SKIP_TXFM_AC_DC
else if (sse_tx - var_tx < dc_thr || sse==var):
  skip_dc = 1
... then call vp9_model_rd_from_var_lapndz twice (DC + AC) and combine.
```

### 4.5 Intra search inside `vp9_pick_inter_mode`

Re-uses prediction buffers if best inter already lives in `pred_buf`.
Gate (line 2516):
```
if (best_rdc.rdcost == INT64_MAX) ||
   (screen && source_variance==0) ||
   (scene_change && perform_intra_pred) ||
   ((!force_skip_low_temp_var || bsize<BLOCK_32X32 ||
     state==kVeryHighSad) &&
    perform_intra_pred && !x->skip &&
    best_rdc.rdcost > inter_mode_thresh &&
    bsize <= sf.max_intra_bsize &&
    !x->skip_low_source_sad && !x->lowvar_highsumdiff):
  // run intra
```

Where `inter_mode_thresh = RDCOST(intra_cost_penalty, 0)` (line 1709)
and `perform_intra_pred` is set to 0 if SVC enhancement layer not on
base, or if alt-ref reference is in use (lines 2494-2512).

Per-mode in `[DC_PRED, V_PRED, H_PRED, TM_PRED]`:
```
mode_rd_thresh := rd_threshes[mode_idx[INTRA_FRAME][i]]
if (short_circuit_flat_blocks && source_variance==0 &&
    ((zero_temp_sad_source && mode != DC_PRED) || i > 2)): skip
if !((1<<mode) & sf.intra_y_mode_bsize_mask[bsize]): skip
if (sf.rt_intra_dc_only_low_content && mode != DC_PRED &&
    state != kVeryHighSad): skip
if rd_less_than_thresh(best_rdc.rdcost, mode_rd_thresh, freq_fact[]):
  if (!screen || svc>0 || (!scene_change && !high_motion)): skip
mi->mode = this_mode; mi->ref_frame[0] = INTRA_FRAME
compute_intra_yprediction(this_mode, bsize, x, xd)        // §4.11
model_rd_for_sb_y(... is_intra=1)
block_yrd(... rd_computed=1, is_intra=1)
if skippable: skip_txfm=AC_DC; rate = cost_bit(skip,1)
else:         skip_txfm=NONE;  rate += cost_bit(skip,0)
if screen content && color_sensitivity[i]:
  vp9_foreach_transformed_block_in_plane(plane i, estimate_block_intra, &args)
rate += cpi->mbmode_cost[mode]
rate += ref_frame_cost[INTRA_FRAME]
rate += intra_cost_penalty
rdcost := RDCOST(rate, dist)
if better: best_pickmode := {mode, INTRA_FRAME, intra_tx_size, ...}
```

Citations: `vp9_pickmode.c:2516-2645`.

### 4.6 Adaptive RD threshold + frequency-fact update

`rd_less_than_thresh` (in `vp9_rd.c`, signature in `vp9_rd.h`) and
`rd_less_than_thresh_row_mt` (`vp9_pickmode.c:1112`):

```
rd_less_than_thresh_row_mt(best_rd, thresh, *thresh_fact):
  return best_rd < (thresh * (*thresh_fact)) >> 5  ||  thresh == INT_MAX
```

`update_thresh_freq_fact[_row_mt]` (`vp9_pickmode.c:1120, 1138`):

```
thr_idx := mode_idx[ref_frame][mode_offset(mode)]
freq_fact := &thresh_freq_fact[bsize][thr_idx]   (or row_mt array)
if (thr_idx == best_mode_idx):
  *freq_fact -= (*freq_fact >> 4)                  // decay toward 0
elif (sf.limit_newmv_early_exit && mode==NEWMV && ref==LAST &&
      source_variance < 5):
  *freq_fact := min(*freq_fact + 1, 32)
else:
  *freq_fact := min(*freq_fact + 1, sf.adaptive_rd_thresh * 64)
```

`RD_THRESH_INC=1`, `RD_THRESH_MAX_FACT=64` from `vp9_rd.h:45-46`.
`sf.adaptive_rd_thresh` is 2 (speed 5), 4 (speed 6 with
temporal_layer>0), 3 (speed 7), 4 (speed 8), but for speed 8 + CBR +
non-screen + non-low-res, it's clamped back to 1 (line 786) when
short-circuit-low-temp-var is in play.

### 4.7 `set_vt_partitioning` — leaf decision in variance tree

`vp9_encodeframe.c:472-547` — given a tree node at level `bsize`, the
function decides whether to keep `bsize`, split horizontally,
vertically, or recurse further.

```
set_vt_partitioning(bsize, mi_row, mi_col, threshold, bsize_min, force_split):
  if force_split == 1: return 0     // caller must descend

  if bsize == bsize_min:
    if intra-only: get_variance(none)
    if (block fully inside frame && none.variance < threshold):
      set_block_size(bsize); return 1
    return 0

  elif bsize > bsize_min:
    if intra-only: get_variance(none)
    if intra-only && (bsize>BLOCK_32X32 || none.variance > threshold<<4):
      return 0                       // always descend at key on large
    if (block fully inside && none.variance < threshold):
      set_block_size(bsize); return 1
    // try vertical 2-partition
    if (mi_row + half < mi_rows):
      get_variance(vert[0..1])
      if both < threshold && uv-block-size valid:
        set_block_size(bsize_vert) for both halves; return 1
    // try horizontal 2-partition
    if (mi_col + half < mi_cols):
      get_variance(horz[0..1])
      if both < threshold && uv-block-size valid:
        set_block_size(bsize_horz) for both halves; return 1
    return 0
  return 0
```

Return value semantics:
- `1` means "block accepted at this size (or as horz/vert split)" —
  caller doesn't recurse.
- `0` means "force descent into 4-split".

Drives the recursion in `choose_partitioning`'s bottom block
(`vp9_encodeframe.c:1701-1748`).

### 4.8 `vp9_NEWMV_diff_bias` — RD-cost biasing

`vp9_pickmode.c:1298-1355`. Adds bias to penalize unusual NEWMV and
favour zero-MV LAST on noisy content.

```
if (this_mode == NEWMV):
  Find above/left MVs (any in xd->above_mi, xd->left_mi).
  al_mv_avg := average of valid neighbours (or 0/0 if none valid)
  row_diff := al_mv_avg.row - mv.row
  col_diff := al_mv_avg.col - mv.col
  if (|row_diff| > 48 || |col_diff| > 48):
    if (bsize > BLOCK_32X32): rdcost <<= 1
    else:                     rdcost := 3 * rdcost >> 1     // *1.5
if (noise_enabled && noise.level >= kMedium && bsize >= BLOCK_32X32 &&
    ref==LAST && |mv.row|<8 && |mv.col|<8):
  rdcost := 7 * (rdcost >> 3)                                // *0.875
elif (lowvar_highsumdiff && !skin && bsize >= BLOCK_16X16 &&
      ref==LAST && |mv.row|<16 && |mv.col|<16):
  rdcost := 7 * (rdcost >> 3)
```

### 4.9 `encode_breakout_test` — early skip detection in mode-decision

`vp9_pickmode.c:932-1035`. Tests whether prediction is good enough that
the block can be encoded as `skip=1` (no residual).

```
if (svc && ref==GOLDEN): return        // not allowed
motion_low := !(|mv.row|>64 || |mv.col|>64)
if (encode_breakout > 0 && motion_low):
  max_thresh := 36000
  min_thresh := min(encode_breakout << 4, max_thresh)
  thresh_ac := (dequant[1]^2) >> 3
  thresh_ac := clamp(thresh_ac, min_thresh, max_thresh)
  thresh_ac >>= 8 - (log2(bw) + log2(bh))    // scale by partition size
  thresh_dc := (dequant[0]^2) >> 6
else:
  thresh_ac = thresh_dc = 0

if (var <= thresh_ac && (sse - var) <= thresh_dc):
  thresh_ac_uv := thresh_ac
  thresh_dc_uv := thresh_dc
  if (sb_is_skin): thresh_ac_uv = thresh_dc_uv = 0

  build UV predictors if not already
  var_u, sse_u := variance(src_u, pred_u)
  if (var_u<<2 <= thresh_ac_uv && sse_u-var_u <= thresh_dc_uv):
    var_v, sse_v := variance(src_v, pred_v)
    if (var_v<<2 <= thresh_ac_uv && sse_v-var_v <= thresh_dc_uv):
      x->skip := 1
      *rate := cpi->inter_mode_cost[mode_ctx[ref]][INTER_OFFSET(this_mode)]
      *dist := sse << 4
```

`encode_breakout` itself is set per-segment from
`cpi->segment_encode_breakout[seg_id]` (`vp9_pickmode.c:1776-1778`).

### 4.10 `calculate_tx_size` — pick the per-block tx size

`vp9_pickmode.c:353-384`:

```
var_thresh := is_intra ? (unsigned)ac_thr : 1
limit_tx := 1
if (aq_mode==CYCLIC_REFRESH_AQ && (source_variance==0 || var<var_thresh)):
  limit_tx := 0
if (cm->tx_mode == TX_MODE_SELECT):
  if (sse > (var << 2)):
    tx_size := min(max_txsize_lookup[bsize],
                    tx_mode_to_biggest_tx_size[cm->tx_mode])
  else:
    tx_size := TX_8X8
  if (CR_AQ && limit_tx && CR-boosted):       tx_size := TX_8X8
  elif (tx_size > TX_16X16 && limit_tx):       tx_size := TX_16X16
  // screen-content force 4x4 for high-variance small blocks
  if (screen && tx_size==TX_8X8 && bsize<=BLOCK_16X16 &&
      ((var >> 5) > ac_thr)):                 tx_size := TX_4X4
else:
  tx_size := min(max_txsize_lookup[bsize],
                 tx_mode_to_biggest_tx_size[cm->tx_mode])
return tx_size
```

Used in both `model_rd_for_sb_y_large` (line 491) and
`model_rd_for_sb_y` (line 658).

### 4.11 `compute_intra_yprediction` — strided intra prediction

`vp9_pickmode.c:386-427`. Computes the full intra prediction for the
partition without using the standard transform-block walker (which
would also dequantize and reconstruct). Used inside the intra branch
of `vp9_pick_inter_mode`.

```
tx_size := max_txsize_lookup[bsize]
num_4x4_w := num_4x4_blocks_wide_lookup[bsize]
num_4x4_h := num_4x4_blocks_high_lookup[bsize]
max_blocks_wide := clipped to UMV
max_blocks_high := clipped to UMV
for row in 0..max_blocks_high step (1 << tx_size):
  for col in 0..max_blocks_wide step (1 << tx_size):
    p->src.buf := src_buf_base + 4*(row*src_stride + col)
    pd->dst.buf := dst_buf_base + 4*(row*dst_stride + col)
    vp9_predict_intra_block(xd, log2(bw), tx_size, mode,
                            skip_encode? p->src.buf : pd->dst.buf,
                            skip_encode? src_stride: dst_stride,
                            pd->dst.buf, dst_stride, col, row, plane=0)
p->src.buf := src_buf_base; pd->dst.buf := dst_buf_base
```

### 4.12 `model_rd_for_sb_uv` — UV plane rate-distortion model

`vp9_pickmode.c:846-914`. For each of U, V (if `color_sensitivity[i-1]`
is set):
```
var, sse := cpi->fn_ptr[uv_bsize].vf(src, pred)
tot_var += var; tot_sse += sse
vp9_model_rd_from_var_lapndz(sse-var, n_log2, dc_quant>>3, &rate, &dist)
this_rdc->rate += rate>>1
this_rdc->dist += dist<<3
vp9_model_rd_from_var_lapndz(var, n_log2, ac_quant>>3, &rate, &dist)
this_rdc->rate += rate
this_rdc->dist += dist<<4
```
Final `*var_y, *sse_y` are `tot_var, tot_sse` (clamped to UINT32_MAX).

### 4.13 `find_predictors` — per-ref candidate generation

`vp9_pickmode.c:1255-1296`. For one ref frame, populate
`frame_mv[NEW/NEAREST/NEAR/ZERO][ref]`, `yv12_mb[ref][*]`,
`x->mbmi_ext->ref_mvs[ref][*]`, and `x->mbmi_ext->mode_context[ref]`.

- `frame_mv[NEWMV][ref] := INVALID_MV` (gets overwritten by motion
  search)
- `frame_mv[ZEROMV][ref] := 0`
- If `use_prev_frame_mvs` or `comp_pred_allowed`: full `vp9_find_mv_refs`
- Else: `mv_refs_rt` (line 59) — the **lightweight** real-time MV-refs
  routine that walks `mv_ref_blocks[mi->sb_type]` neighbours, finds at
  most 2 candidates from the same ref frame (no separate ZEROMV path),
  falls back to other ref frames if needed, and optionally pulls a
  base-layer MV (`use_base_mv`) for SVC enhancement layers.
- `vp9_find_best_ref_mvs` populates `frame_mv[NEARESTMV]` and `[NEARMV]`.
- If non-scaled and `bsize >= BLOCK_8X8` and not (force_skip+GOLDEN),
  call `vp9_mv_pred` to fill `x->pred_mv_sad[ref]` (used later for
  reference masking).

If ref is not available, set `ref_frame_skip_mask |= (1 << ref)`.

### 4.14 `mv_refs_rt` — lightweight MV-refs

`vp9_pickmode.c:59-152`. Returns `const_motion` flag (1 if neighbours
agree on motion). Walks `mv_ref_blocks[sb_type]` (8 positions) and
collects up to `MAX_MV_REF_CANDIDATES` (= 2) MVs from the same ref
frame; if none found, scans different-ref-frame candidates and applies
`IF_DIFF_REF_FRAME_ADD_MV` (sign-bias flip). Optional base-layer MV
import for SVC (line 127). Always clamps the result via `clamp_mv_ref`.

### 4.15 `get_force_skip_low_temp_var` — low-temporal-variance mask lookup

`vp9_pickmode.c:1441-1486`. Indexes into `x->variance_low[25]`
according to the SB-relative position `(mi_row & 0x7, mi_col & 0x7)`:

```
i := (mi_row & 0x7) >> 1               // {0,1,2,3}
j := (mi_col & 0x7) >> 1               // {0,1,2,3}
switch sb_type:
  BLOCK_64X64: return variance_low[0]
  BLOCK_64X32: return [1] if (col==0 && row==0)
                      [2] if (col==0 && row!=0)
                      else 0
  BLOCK_32X64: return [3] if (col==0 && row==0)
                      [4] if (col!=0 && row==0)
                      else 0
  BLOCK_32X32: dispatch on (col==0, row==0) → variance_low[5..8]
  BLOCK_16X16: return variance_low[pos_shift_16x16[i][j]]
  BLOCK_32X16: AND of the 2 variance_low entries for the two 16x16's
  BLOCK_16X32: AND of the 2 (vertical pair)
  default:     return 0
```

The flag is set during partition-time `set_low_temp_var_flag`
(`vp9_encodeframe.c:829-893`), which writes 1 to entries where the
variance is below a fraction of the partition threshold and the
`int_pro` MV is below `mv_thr = (cm->width>640 ? 8 : 4)`.

### 4.16 `set_low_temp_var_flag` thresholds (verbatim)

`vp9_encodeframe.c:829-893`:

```
mv_thr := cm->width > 640 ? 8 : 4
if (ref==LAST &&
    (short_circuit_low_temp_var==1 ||
     |mv.col|<mv_thr && |mv.row|<mv_thr)):
  // 64x64
  if sb_type==BLOCK_64X64 && vt.none.var < thresholds[0]>>1:
    variance_low[0] := 1
  // 64x32
  elif BLOCK_64X32: for i in 0..1: if vt.horz[i].var < thresholds[0]>>2:
    variance_low[i+1] := 1
  // 32x64
  elif BLOCK_32X64: similar with thresholds[0]>>2 → [i+3]
  // 32x32 (per quadrant)
  else: for i in 0..3:
    if mi[i*32-cluster] is 32x32:
      threshold_32x32 := (short_circuit_low_temp_var in {1,3})
                          ? (5*thresholds[1])>>3
                          : thresholds[1]>>1
      if vt.split[i].none.var < threshold_32x32:
        variance_low[i+5] := 1
    elif short_circuit_low_temp_var >= 2:
      // 16x16 and 32x16/16x32 → check each 16x16
      for j in 0..3:
        if vt.split[i].split[j].none.var < thresholds[2]>>8:
          variance_low[(i<<2) + j + 9] := 1
```

Effect cascade (`short_circuit_low_temp_var` value):
- 1: Skip all golden modes and ALL INTRA for `bsize >= 32x32`.
- 2: Skip golden non-zero-mv and newmv-last for `bsize >= 16x16`,
  skip ALL INTRA for `bsize >= 32x32` and vert/horz INTRA for
  16x16/16x32/32x16.
- 3: Same as 2 but additionally skips golden zeromv.

(Source: comment block at `vp9_speed_features.h:577-586`.)

### 4.17 `chroma_check` — set per-plane color sensitivity

`vp9_encodeframe.c:1173-1207`:

```
if is_key_frame: return
shift := 2
if speed>8 && y_sad>thresholds[1] && noise.level<kMedium: return
if (screen && scene_change_detected): shift := 5
for i in 1..2:
  bs := get_plane_block_size(bsize, pd)
  if bs != BLOCK_INVALID:
    uv_sad := SAD(src_uv, pred_uv)
  else:
    uv_sad := UINT_MAX
  x->color_sensitivity[i-1] := (uv_sad > (y_sad >> shift))
```

`color_sensitivity` drives the `model_rd_for_sb_uv` invocations inside
`vp9_pick_inter_mode` (line 2378-2392) and the per-plane intra checks
on screen content (line 2612-2618).

### 4.18 `avg_source_sad` — content classification (called once per SB before partition)

`vp9_encodeframe.c:1209-1257`:

```
tmp_sad := SAD(cur_source, last_source) for the 64x64 at this position
tmp_variance := vpx_variance64x64(cur, last, &tmp_sse)
                // tmp_sse - tmp_variance == (sum*sum) >> 12  (i.e. sum^2/4096)

if tmp_sad < 10000:
  state := (tmp_sse - tmp_variance) < 25 ? kLowSadLowSumdiff
                                          : kLowSadHighSumdiff
else:
  state := (tmp_sse - tmp_variance) < 25 ? kHighSadLowSumdiff
                                          : kHighSadHighSumdiff

if (!screen && CBR && tmp_variance < tmp_sse>>3 && (tmp_sse-tmp_variance)>10000):
  state := kLowVarHighSumdiff
elif tmp_sad > 20000:
  state := kVeryHighSad

if cpi->content_state_sb_fd != NULL:
  if tmp_sad < 12000:
    sb_fd[off] := min(sb_fd[off]+1, 255)
  else:
    sb_fd[off] := 0

if tmp_sad == 0: x->zero_temp_sad_source := 1
return tmp_sad
```

The constants 10000, 12000, 20000, 25 are **hardcoded** with no
content-type or resolution gating.

---

## 5. Tables and constants

### 5.1 `pos_shift_16x16` (vp9_pickmode.c:55)

```
static const int pos_shift_16x16[4][4] = {
  { 9, 10, 13, 14 }, { 11, 12, 15, 16 }, { 17, 18, 21, 22 }, { 19, 20, 23, 24 }
};
```

### 5.2 `mode_idx`, `intra_mode_list`, `ref_mode_set`, `ref_mode_set_svc`

(All transcribed verbatim in §2.3 and §2.4.)

### 5.3 `VP9_VAR_OFFS` — intra-frame reference for variance (vp9_encodeframe.c:70)

```
static const uint8_t VP9_VAR_OFFS[64] = {
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128
};
```

10-bit and 12-bit variants `VP9_HIGH_VAR_OFFS_10` (all 512),
`VP9_HIGH_VAR_OFFS_12` (all 2048) at lines 87-109.

### 5.4 `vp9_model_rd_from_var_lapndz` lookup tables (vp9_rd.c)

> **Status:** Patched verbatim by stage-2 prep work resolving
> **R-07 Blocker** (async A7 #2). Tables sourced from
> `vp9/encoder/vp9_rd.c:462-503`.

`MAX_XSQ_Q10 = 245727` (`vp9_rd.c:516`). The model uses three
table-lookup arrays `rate_tab_q10[100]`, `dist_tab_q10[100]`, and
`xsq_iq_q10[105]`, all indexed by `xsq_q10`'s top bits inside
`model_rd_norm` (`vp9_rd.c:505-514`). These are load-bearing for every
mode-decision rate/distortion estimate via `vp9_model_rd_from_var_lapndz`.

The closed-form derivation (from the comment block at `vp9_rd.c:452-461`):
the functions are sampled at the four most significant bits of
`x^2 + 8 / 256`. Normalized rate is `Rn(x) = H(sqrt(r)) +
sqrt(r) * [1 + H(r)/(1 - r)]` where `r = exp(-sqrt(2) * x)` and
`x = qpstep / sqrt(variance)`. Normalized distortion is
`Dn(x) = 1 - 1/sqrt(2) * x / sinh(x / sqrt(2))`. Actual distortion is
`Dn * variance`.

**`rate_tab_q10[100]`** (`vp9_rd.c:462-471`):

```c
static const int rate_tab_q10[] = {
  65536, 6086, 5574, 5275, 5063, 4899, 4764, 4651, 4553, 4389, 4255, 4142, 4044,
  3958,  3881, 3811, 3748, 3635, 3538, 3453, 3376, 3307, 3244, 3186, 3133, 3037,
  2952,  2877, 2809, 2747, 2690, 2638, 2589, 2501, 2423, 2353, 2290, 2232, 2179,
  2130,  2084, 2001, 1928, 1862, 1802, 1748, 1698, 1651, 1608, 1530, 1460, 1398,
  1342,  1290, 1243, 1199, 1159, 1086, 1021, 963,  911,  864,  821,  781,  745,
  680,   623,  574,  530,  490,  455,  424,  395,  345,  304,  269,  239,  213,
  190,   171,  154,  126,  104,  87,   73,   61,   52,   44,   38,   28,   21,
  16,    12,   10,   8,    6,    5,    3,    2,    1,    1,    1,    0,    0,
};
```

**`dist_tab_q10[100]`** (`vp9_rd.c:480-489`):

```c
static const int dist_tab_q10[] = {
  0,    0,    1,    1,    1,    2,    2,    2,    3,    3,    4,    5,    5,
  6,    7,    7,    8,    9,    11,   12,   13,   15,   16,   17,   18,   21,
  24,   26,   29,   31,   34,   36,   39,   44,   49,   54,   59,   64,   69,
  73,   78,   88,   97,   106,  115,  124,  133,  142,  151,  167,  184,  200,
  215,  231,  245,  260,  274,  301,  327,  351,  375,  397,  418,  439,  458,
  495,  528,  559,  587,  613,  637,  659,  680,  717,  749,  777,  801,  823,
  842,  859,  874,  899,  919,  936,  949,  960,  969,  977,  983,  994,  1001,
  1006, 1010, 1013, 1015, 1017, 1018, 1020, 1022, 1022, 1023, 1023, 1023, 1024,
};
```

**`xsq_iq_q10[105]`** (`vp9_rd.c:490-503`):

```c
static const int xsq_iq_q10[] = {
  0,      4,      8,      12,     16,     20,     24,     28,     32,
  40,     48,     56,     64,     72,     80,     88,     96,     112,
  128,    144,    160,    176,    192,    208,    224,    256,    288,
  320,    352,    384,    416,    448,    480,    544,    608,    672,
  736,    800,    864,    928,    992,    1120,   1248,   1376,   1504,
  1632,   1760,   1888,   2016,   2272,   2528,   2784,   3040,   3296,
  3552,   3808,   4064,   4576,   5088,   5600,   6112,   6624,   7136,
  7648,   8160,   9184,   10208,  11232,  12256,  13280,  14304,  15328,
  16352,  18400,  20448,  22496,  24544,  26592,  28640,  30688,  32736,
  36832,  40928,  45024,  49120,  53216,  57312,  61408,  65504,  73696,
  81888,  90080,  98272,  106464, 114656, 122848, 131040, 147424, 163808,
  180192, 196576, 212960, 229344, 245728,
};
```

**Consumer (`vp9_rd.c:505-514`)** — for context, the interpolation kernel
that turns `xsq_q10` into `(r_q10, d_q10)`:

```c
static void model_rd_norm(int xsq_q10, int *r_q10, int *d_q10) {
  const int tmp = (xsq_q10 >> 2) + 8;
  const int k = get_msb(tmp) - 3;
  const int xq = (k << 3) + ((tmp >> k) & 0x7);
  const int one_q10 = 1 << 10;
  const int a_q10 = ((xsq_q10 - xsq_iq_q10[xq]) << 10) >> (2 + k);
  const int b_q10 = one_q10 - a_q10;
  *r_q10 = (rate_tab_q10[xq] * b_q10 + rate_tab_q10[xq + 1] * a_q10) >> 10;
  *d_q10 = (dist_tab_q10[xq] * b_q10 + dist_tab_q10[xq + 1] * a_q10) >> 10;
}
```

**Byte-parity trap:** the `(rate_tab_q10[xq] * b_q10 + ...) >> 10`
arithmetic is signed int; on 32-bit targets the intermediate product
fits in i32 (max `65536 * 1024 ≈ 6.7e7`). Rust port: keep `i32`
throughout; do **not** widen to `i64` "for safety" — that changes the
shift-right rounding behaviour silently. Use `wrapping_*` per D7 just
in case, but the math is safely within i32.

**Rust port:** three `pub static` arrays of `i32` in
`rvp9_encoder::rd::laplacian`. `MAX_XSQ_Q10` as a `pub const`.
`model_rd_norm` as a free function with the same arithmetic shape.

### 5.5 `THR_MODES` enum (vp9_rd.h:53-93)

```
typedef enum {
  THR_NEARESTMV,
  THR_NEARESTA,
  THR_NEARESTG,

  THR_DC,

  THR_NEWMV,
  THR_NEWA,
  THR_NEWG,

  THR_NEARMV,
  THR_NEARA,
  THR_NEARG,

  THR_ZEROMV,
  THR_ZEROG,
  THR_ZEROA,

  THR_COMP_NEARESTLA,
  THR_COMP_NEARESTGA,

  THR_TM,

  THR_COMP_NEARLA,
  THR_COMP_NEWLA,
  THR_COMP_NEARGA,
  THR_COMP_NEWGA,

  THR_COMP_ZEROLA,
  THR_COMP_ZEROGA,

  THR_H_PRED,
  THR_V_PRED,
  THR_D135_PRED,
  THR_D207_PRED,
  THR_D153_PRED,
  THR_D63_PRED,
  THR_D117_PRED,
  THR_D45_PRED,
} THR_MODES;
```

(Maximum index = 29; `MAX_MODES` defined in `vp9_blockd.h` is 30.)

### 5.6 Intra-mode bitmask enums (vp9_speed_features.h:20-29)

```
enum {
  INTRA_ALL = (1 << DC_PRED) | (1 << V_PRED) | (1 << H_PRED) | (1 << D45_PRED) |
              (1 << D135_PRED) | (1 << D117_PRED) | (1 << D153_PRED) |
              (1 << D207_PRED) | (1 << D63_PRED) | (1 << TM_PRED),
  INTRA_DC = (1 << DC_PRED),
  INTRA_DC_TM = (1 << DC_PRED) | (1 << TM_PRED),
  INTRA_DC_H_V = (1 << DC_PRED) | (1 << V_PRED) | (1 << H_PRED),
  INTRA_DC_TM_H_V =
      (1 << DC_PRED) | (1 << TM_PRED) | (1 << V_PRED) | (1 << H_PRED)
};
```

### 5.7 Inter-mode bitmask enums (vp9_speed_features.h:31-39)

```
enum {
  INTER_ALL = (1 << NEARESTMV) | (1 << NEARMV) | (1 << ZEROMV) | (1 << NEWMV),
  INTER_NEAREST = (1 << NEARESTMV),
  INTER_NEAREST_NEW = (1 << NEARESTMV) | (1 << NEWMV),
  INTER_NEAREST_ZERO = (1 << NEARESTMV) | (1 << ZEROMV),
  INTER_NEAREST_NEW_ZERO = (1 << NEARESTMV) | (1 << ZEROMV) | (1 << NEWMV),
  INTER_NEAREST_NEAR_NEW = (1 << NEARESTMV) | (1 << NEARMV) | (1 << NEWMV),
  INTER_NEAREST_NEAR_ZERO = (1 << NEARESTMV) | (1 << NEARMV) | (1 << ZEROMV),
};
```

At speed ≥ 5 the SF dispatch sets
`sf->inter_mode_mask[BLOCK_32X32..64X64] = INTER_NEAREST_NEW_ZERO`
(`vp9_speed_features.c:603-606`); meaning **NEARMV is excluded** for
all 32×32/32×64/64×32/64×64 blocks at cpu-used ≥ 5.

### 5.8 `PARTITION_SEARCH_TYPE` enum (vp9_speed_features.h:138-156)

```
typedef enum {
  SEARCH_PARTITION,
  FIXED_PARTITION,
  REFERENCE_PARTITION,
  VAR_BASED_PARTITION,
  SOURCE_VAR_BASED_PARTITION,
  ML_BASED_PARTITION
} PARTITION_SEARCH_TYPE;
```

### 5.9 `CONTENT_STATE_SB` enum (vp9_encoder.h:137-145)

```
typedef enum {
  kInvalid = 0,
  kLowSadLowSumdiff = 1,
  kLowSadHighSumdiff = 2,
  kHighSadLowSumdiff = 3,
  kHighSadHighSumdiff = 4,
  kLowVarHighSumdiff = 5,
  kVeryHighSad = 6,
} CONTENT_STATE_SB;
```

### 5.10 `SKIP_TXFM` constants (vp9_block.h:159-165)

```
uint8_t skip_txfm[MAX_MB_PLANE << 2];
#define SKIP_TXFM_NONE 0
#define SKIP_TXFM_AC_DC 1
#define SKIP_TXFM_AC_ONLY 2
```

### 5.11 Key non-RD-path constants

| Constant | Value | File:line |
|---|---|---|
| `RT_INTER_MODES` | 12 | `vp9_pickmode.c:1237` |
| `RT_INTER_MODES_SVC` | 8 | `vp9_pickmode.c:1247` |
| `RD_THRESH_INC` | 1 | `vp9_rd.h:46` |
| `RD_THRESH_MAX_FACT` | 64 | `vp9_rd.h:45` |
| `QIDX_SKIP_THRESH` | 115 | `vp9_rd.h:36` |
| `MV_COST_WEIGHT` | 108 | `vp9_rd.h:38` |
| `VP9_PROB_COST_SHIFT` | 9 | `vp9_cost.h:24` |
| `VP9_DIST_SCALE_LOG2` | 4 | `vp9_rd.h:48` |
| `MAX_XSQ_Q10` | 245727 | `vp9_rd.c:516` |
| `avg_source_sad_threshold` | 10000 | `vp9_encodeframe.c:1219` |
| `avg_source_sad_threshold2` | 12000 | `vp9_encodeframe.c:1220` |
| `kVeryHighSad` threshold | `2*10000 = 20000` | `vp9_encodeframe.c:1243` |
| `(tmp_sse - tmp_variance) < 25` threshold | 25 | `vp9_encodeframe.c:1232/1235` |
| `mv_thr` for low-temp-var | `width>640 ? 8 : 4` | `vp9_encodeframe.c:835` |
| NEWMV neighbour-difference threshold | 48 | `vp9_pickmode.c:1339` |
| Noise+ref==LAST low-motion threshold | ±8 | `vp9_pickmode.c:1349` |
| `encode_breakout_test` motion clip | ±64 | `vp9_pickmode.c:948` |
| `encode_breakout_test` max_thresh | 36000 | `vp9_pickmode.c:955` |
| `vbp_threshold_minmax` | `15 + (q>>3)` | `vp9_encodeframe.c:674` |
| `vbp_threshold_sad` (≤QCIF) | 10 | `vp9_encodeframe.c:654` |
| `vbp_threshold_sad` (other) | `max(y_dequant[q][1]<<1, 1000)` | `vp9_encodeframe.c:656-658` |
| `vbp_threshold_copy` (≤QCIF) | 4000 | `vp9_encodeframe.c:661` |
| `vbp_threshold_copy` (≤nHD) | 8000 | `vp9_encodeframe.c:663` |
| `vbp_threshold_copy` (other) | `max(y_dequant[q][1]<<3, 8000)` | `vp9_encodeframe.c:665-667` |
| `thresh_skip_golden` (default) | 500 | `vp9_pickmode.c:1768` |
| `thresh_svc_skip_golden` (default) | 500 | `vp9_pickmode.c:1767` |
| `large_block` content-state threshold | 70 | `vp9_pickmode.c:2030` |
| Intra-cost-penalty coefficient | 20 | `vp9_rd.c:794` |
| Speed-8 cb_pred_filter_search threshold | 640×480 | `vp9_speed_features.c:792` |

### 5.12 `rd_thresh_block_size_factor[BLOCK_SIZES]` (vp9_rd.c:88-90)

```
static const uint8_t rd_thresh_block_size_factor[BLOCK_SIZES] = {
  2, 3, 3, 4, 6, 6, 8, 12, 12, 16, 24, 24, 32
};
```

Indexed by `BLOCK_4X4=0` through `BLOCK_64X64=12`. Factors are **<<2**
so 2 means "0.5", 32 means "8" (per comment line 88).

### 5.13 Speed-feature dispatch (relevant to non-RD path) — verbatim slices

From `vp9_speed_features.c:585-660` (speed 5+):

```
sf->use_altref_onepass = 0;
sf->use_quant_fp = !is_keyframe;
sf->auto_min_max_partition_size =
    is_keyframe ? RELAXED_NEIGHBORING_MIN_MAX : STRICT_NEIGHBORING_MIN_MAX;
sf->default_max_partition_size = BLOCK_32X32;
sf->default_min_partition_size = BLOCK_8X8;
sf->partition_search_type = REFERENCE_PARTITION;
if (rc_mode==VBR && lag>0 && is_src_frame_alt_ref):
  sf->partition_search_type = VAR_BASED_PARTITION;
sf->use_nonrd_pick_mode = 1;
sf->allow_skip_recode = 0;
sf->inter_mode_mask[BLOCK_32X32] = INTER_NEAREST_NEW_ZERO;
sf->inter_mode_mask[BLOCK_32X64] = INTER_NEAREST_NEW_ZERO;
sf->inter_mode_mask[BLOCK_64X32] = INTER_NEAREST_NEW_ZERO;
sf->inter_mode_mask[BLOCK_64X64] = INTER_NEAREST_NEW_ZERO;
sf->adaptive_rd_thresh = 2;
sf->reuse_inter_pred_sby = 1;
sf->coeff_prob_appx_step = 4;
sf->use_fast_coef_updates = is_keyframe ? TWO_LOOP : ONE_LOOP_REDUCED;
sf->mode_search_skip_flags = FLAG_SKIP_INTRA_DIRMISMATCH;
sf->tx_size_search_method = is_keyframe ? USE_LARGESTALL : USE_TX_8X8;
sf->simple_model_rd_from_var = 1;
if (rc_mode==VBR) sf->mv.search_method = NSTEP;

// per-bsize intra mask
if (!is_keyframe):
  if content == VP9E_CONTENT_SCREEN:
    for bsize >= 32x32: intra_y_mode_bsize_mask = INTRA_DC_H_V
    for bsize <  32x32: intra_y_mode_bsize_mask = INTRA_DC_TM_H_V
  else:
    for bsize >  16x16: intra_y_mode_bsize_mask = INTRA_DC
    for bsize <= 16x16: intra_y_mode_bsize_mask = INTRA_DC_H_V

if content == VP9E_CONTENT_SCREEN: sf->short_circuit_flat_blocks = 1
if (CBR && !screen):
  sf->limit_newmv_early_exit = 1
  if !svc: sf->bias_golden = 1
if (svc && spatial_layer>0): sf->nonrd_keyframe = 1
if (CBR && !screen) → overshoot_detection_cbr_rt
if (VBR && lag>0 && w<=1280 && h<=720):
  sf->use_altref_onepass = 1
  sf->use_compound_nonrd_pickmode = 1
if (w*h > 1280*720): sf->cb_pred_filter_search = 2
if (!external_resize): sf->use_source_sad = 1
```

Speed 6 (lines 662-697):
```
if (VBR && lag>0): use_altref_onepass=1; use_compound_nonrd_pickmode=1
sf->partition_search_type = VAR_BASED_PARTITION
sf->mv.search_method = NSTEP
sf->mv.reduce_first_step_size = 1
sf->skip_encode_sb = 0
if sf->use_source_sad:
  sf->adapt_partition_source_sad = 1
  sf->adapt_partition_thresh = (w*h<=640*360) ? 40000 : 60000
  allocate content_state_sb_fd
if (CBR && !screen): sf->short_circuit_low_temp_var = 1
if temporal_layer>0:
  sf->adaptive_rd_thresh = 4
  sf->limit_newmv_early_exit = 0
  sf->base_mv_aggressive = 1
if (CBR && !key && resize==ORIG && !disable_overshoot):
  overshoot_detection_cbr_rt = FAST_DETECTION_MAXQ
```

Speed 7 (lines 699-749):
```
sf->adapt_partition_source_sad = 0
sf->adaptive_rd_thresh = 3
sf->mv.search_method = FAST_DIAMOND
sf->mv.fullpel_search_step_param = 10
// svc base TL gets stronger ME
if (temporal>0 || spatial>1):
  sf->use_simple_block_yrd = 1
  if (non_reference_frame): mv.subpel_search_method = SUBPEL_TREE_PRUNED_EVENMORE
if (svc && row_mt && max_threads>1): sf->adaptive_rd_thresh_row_mt = 1
if (!last_frame_dropped && resize==ORIG && !external_resize):
  sf->copy_partition_flag = 1
  cpi->max_copied_frame = 2 (or 255 for top TL of multi-TL svc)
if (svc && use_partition_reuse && spatial==3 && temporal>0 && w*h>VGA):
  sf->svc_use_lowres_part = 1
if (svc && use_gf_temporal_ref_current_layer && temporal>0):
  cpi->ref_frame_flags &= ~VP9_GOLD_FLAG
if (w*h > 640*480): sf->cb_pred_filter_search = 2
```

Speed 8 (lines 751-793):
```
sf->adaptive_rd_thresh = 4
sf->skip_encode_sb = 1
sf->nonrd_keyframe = (svc.spatial>1 && !simulcast) ? 0 : 1
if (!svc) cpi->max_copied_frame = 4
if (row_mt && max_threads>1): sf->adaptive_rd_thresh_row_mt = 1
if (!is_intra_only && w*h <= 352*288):
  sf->nonrd_use_ml_partition = 1
if highbd: nonrd_use_ml_partition = 0
if (screen): sf->mv.subpel_force_stop = HALF_PEL
sf->rt_intra_dc_only_low_content = 1
if (!svc && CBR && !screen):
  sf->short_circuit_low_temp_var = 3
  if (noise_estimate && w>=1280 && h>=720 && noise>=kMedium):
    sf->short_circuit_low_temp_var = 2
  if (w*h > 352*288): sf->adaptive_rd_thresh = 1
  else:                sf->adaptive_rd_thresh = 2
sf->limit_newmv_early_exit = 0
sf->use_simple_block_yrd = 1
if (w*h > 352*288): sf->cb_pred_filter_search = 2
```

### 5.14 `coord_lookup` (32x32 → 16x16 sub-positions in 64x64 SB)

Used by `set_source_var_based_partition`:

```
vp9/encoder/vp9_encodeframe.c:2397
static const struct {
  int row;
  int col;
} coord_lookup[16] = {
  // 32x32 index = 0
  { 0, 0 }, { 0, 2 }, { 2, 0 }, { 2, 2 },
  // 32x32 index = 1
  { 0, 4 }, { 0, 6 }, { 2, 4 }, { 2, 6 },
  // 32x32 index = 2
  { 4, 0 }, { 4, 2 }, { 6, 0 }, { 6, 2 },
  // 32x32 index = 3
  { 4, 4 }, { 4, 6 }, { 6, 4 }, { 6, 6 },
};
```

### 5.15 Inter-mode prob tree (vp9_entropymode.c:257-260)

(Bitstream/entropy table; A4 owns the broader bitstream side, but the
cost generation that feeds `cpi->inter_mode_cost[ctx][INTER_OFFSET]`
walks this tree.)

```
const vpx_tree_index vp9_inter_mode_tree[TREE_SIZE(INTER_MODES)] = {
  -INTER_OFFSET(ZEROMV), 2, -INTER_OFFSET(NEARESTMV), 4, -INTER_OFFSET(NEARMV),
  -INTER_OFFSET(NEWMV)
};
```

`INTER_OFFSET(mode) := (mode) - NEARESTMV` (`vp9_entropymode.h:27`).

### 5.16 Partition prob tree (vp9_entropymode.c:262-264)

```
const vpx_tree_index vp9_partition_tree[TREE_SIZE(PARTITION_TYPES)] = {
  -PARTITION_NONE, 2, -PARTITION_HORZ, 4, -PARTITION_VERT, -PARTITION_SPLIT
};
```

---

## 6. RTCD / SIMD surfaces

All paths below are exercised by the non-RD path. ISAs listed are the
ones libvpx provides specialized implementations for in 1.14.1 (from
`vpx_dsp/vpx_dsp_rtcd_defs.pl` and `vp9/common/vp9_rtcd_defs.pl`).

### 6.1 Variance kernels — `vpx_variance*` (used by `cpi->fn_ptr[bsize].vf`)

| Kernel | Signature (return = uint32_t variance) | ISAs |
|---|---|---|
| `vpx_variance64x64` | `(src, sstride, ref, rstride, *sse) → var` | sse2 avx2 neon neon_dotprod msa mmi vsx lsx |
| `vpx_variance64x32` | same | sse2 avx2 neon neon_dotprod msa mmi vsx |
| `vpx_variance32x64` | same | sse2 avx2 neon neon_dotprod msa mmi vsx |
| `vpx_variance32x32` | same | sse2 avx2 neon neon_dotprod msa mmi vsx lsx |
| `vpx_variance32x16` | same | sse2 avx2 neon neon_dotprod msa mmi vsx |
| `vpx_variance16x32` | same | sse2 avx2 neon neon_dotprod msa mmi vsx |
| `vpx_variance16x16` | same | sse2 avx2 neon neon_dotprod msa mmi vsx lsx |
| `vpx_variance16x8`  | same | sse2 avx2 neon neon_dotprod msa mmi vsx |
| `vpx_variance8x16`  | same | sse2 avx2 neon neon_dotprod msa mmi vsx |
| `vpx_variance8x8`   | same | sse2 avx2 neon neon_dotprod msa mmi vsx lsx |
| `vpx_variance8x4`   | same | sse2 avx2 neon neon_dotprod msa mmi vsx |
| `vpx_variance4x8`   | same | sse2 neon neon_dotprod msa mmi vsx |
| `vpx_variance4x4`   | same | sse2 neon neon_dotprod msa mmi vsx |

Citations: `vpx_dsp/vpx_dsp_rtcd_defs.pl:1263-1299`.
Complexity for our Rust port: **medium** — straightforward SoA → SIMD,
but exact bit pattern matters (rounding inside `(sum*sum)>>n` must
match libvpx).

### 6.2 8×8 sub-variance — `vpx_get8x8var` (used by `block_variance`)

```
add_proto qw/void vpx_get8x8var/, "const uint8_t *src_ptr, int src_stride,
              const uint8_t *ref_ptr, int ref_stride,
              unsigned int *sse, int *sum";
specialize qw/vpx_get8x8var sse2 neon neon_dotprod msa vsx/;
```

Citation: `vpx_dsp/vpx_dsp_rtcd_defs.pl:1308`. **Required** —
`model_rd_for_sb_y_large` (`vp9_pickmode.c:465-470`) calls
`block_variance` which calls `vpx_get8x8var` for every 8×8 block in the
partition.

### 6.3 SAD kernels — `vpx_sad*` (used in `vp9_int_pro_motion_estimation`, partition copy, etc.)

All 13 block-size variants (64×64 down to 4×4), all with sse2 + avx2 +
neon variants; also `_skip_*` versions for sad-skip. The non-RD path
also uses the `*_avg` (compound-pred SAD) and `*x4d` (4-way SAD)
variants — listed at `vpx_dsp/vpx_dsp_rtcd_defs.pl:891-1009`.

Complexity: **low**. SADs are the simplest SIMD kernel; map directly to
`_mm_sad_epu8` / `vabd_u8`.

### 6.4 Averaging kernels — `vpx_avg_4x4`, `vpx_avg_8x8`

```
add_proto qw/unsigned int vpx_avg_8x8/, "const uint8_t *, int p";
specialize qw/vpx_avg_8x8 sse2 neon msa/;

add_proto qw/unsigned int vpx_avg_4x4/, "const uint8_t *, int p";
specialize qw/vpx_avg_4x4 sse2 neon msa/;
```

Citation: `vpx_dsp/vpx_dsp_rtcd_defs.pl:835-839`. Used by
`fill_variance_8x8avg` / `fill_variance_4x4avg` in
`choose_partitioning`. Complexity: **low**.

### 6.5 `vpx_minmax_8x8` (used in `compute_minmax_8x8`)

```
add_proto qw/void vpx_minmax_8x8/, "const uint8_t *s, int p,
              const uint8_t *d, int dp, int *min, int *max";
specialize qw/vpx_minmax_8x8 sse2 neon msa/;
```

Citation: `vpx_dsp/vpx_dsp_rtcd_defs.pl:841-842`. Used in
`choose_partitioning` to detect texture inside a 16×16 that has
moderate variance but high min-max spread → force split.

### 6.6 Hadamard transforms (used in `block_yrd`)

```
add_proto qw/void vpx_hadamard_8x8/, "const int16_t *src_diff,
              ptrdiff_t src_stride, tran_low_t *coeff";
specialize qw/vpx_hadamard_8x8 sse2 neon vsx lsx/, "$ssse3_x86_64";

add_proto qw/void vpx_hadamard_16x16/, ...; specialize ... avx2 sse2 neon vsx lsx
add_proto qw/void vpx_hadamard_32x32/, ...; specialize ... sse2 avx2 neon
```

Citation: `vpx_dsp/vpx_dsp_rtcd_defs.pl:845-876` (the 8×8 has a special
ssse3_x86_64 variant). For 8-bit (non-HBD) builds, the `tran_low_t` is
`int16_t` (line 869+ branch). Complexity: **medium-high** — Walsh-Hadamard
is structured but needs careful saturation in some variants.

### 6.7 `vpx_satd` (used as rate proxy in `block_yrd`)

```
add_proto qw/int vpx_satd/, "const tran_low_t *coeff, int length";
specialize qw/vpx_satd avx2 sse2 neon/;        // HBD branch
specialize qw/vpx_satd avx2 sse2 neon msa/;    // 8-bit branch
```

Citation: `vpx_dsp/vpx_dsp_rtcd_defs.pl:863-879`. **Single-array sum of
abs**. Complexity: **low**.

### 6.8 `vp9_block_error_fp` (distortion in `block_yrd`)

```
add_proto qw/int64_t vp9_block_error_fp/, "const tran_low_t *coeff,
              const tran_low_t *dqcoeff, int block_size";
specialize qw/vp9_block_error_fp neon avx2 sse2/;
```

Citation: `vp9/common/vp9_rtcd_defs.pl:131-132`. **SSD of two int16
arrays** → wide multiply-accumulate. Complexity: **low-medium**.

### 6.9 `vp9_quantize_fp` / `vp9_quantize_fp_32x32`

```
add_proto qw/void vp9_quantize_fp/, "..."
specialize qw/vp9_quantize_fp neon sse2 ssse3 avx2 vsx/;

add_proto qw/void vp9_quantize_fp_32x32/, "..."
specialize qw/vp9_quantize_fp_32x32 neon ssse3 avx2 vsx/;
```

Citation: `vp9/common/vp9_rtcd_defs.pl:134-138`. Called inside
`block_yrd`. Complexity: **medium** — the EOB tracking requires
horizontal max-index reduction.

### 6.10 `vpx_subtract_block`

```
add_proto qw/void vpx_subtract_block/, "int rows, int cols, int16_t *diff,
                                         ptrdiff_t diff_stride, const uint8_t *src,
                                         ptrdiff_t src_stride, const uint8_t *pred,
                                         ptrdiff_t pred_stride";
specialize qw/vpx_subtract_block neon msa mmi sse2 avx2 vsx lsx/;
```

Citation: `vpx_dsp/vpx_dsp_rtcd_defs.pl:744-745`. Drives the
`src - pred` residual into `p->src_diff` before Hadamard. Complexity:
**low**.

### 6.11 `vpx_int_pro_row`, `vpx_int_pro_col` (used by `vp9_int_pro_motion_estimation`)

```
add_proto qw/void vpx_int_pro_row/, "int16_t hbuf[16], const uint8_t *ref,
                                       const int ref_stride, const int height";
specialize qw/vpx_int_pro_row neon sse2 msa/;

add_proto qw/int16_t vpx_int_pro_col/, "const uint8_t *ref, const int width";
specialize qw/vpx_int_pro_col neon sse2 msa/;
```

Citation: `vpx_dsp/vpx_dsp_rtcd_defs.pl:882-885`. Used inside
`choose_partitioning`'s motion estimate for the 64×64 (A3 owns the
ME wrapper; A2 ownership is just "we call this from the partition
decision").

### 6.12 Sub-pixel variance (used by `cpi->find_fractional_mv_step`)

`vpx_sub_pixel_variance{64x64..4x4}` (13 sizes), with SIMD across sse2,
ssse3, neon, msa, mmi, avx2 (for 64x64 and 32x32). Citation:
`vpx_dsp/vpx_dsp_rtcd_defs.pl:1335-1371`. Strictly speaking owned by
A3 (it's the sub-pel kernel) but the non-RD mode loop calls
`cpi->find_fractional_mv_step` from `combined_motion_search` and
`search_filter_ref`.

### 6.13 Inter-predictor & convolve kernels

`vp9_build_inter_predictors_sby` (in
`vp9/common/vp9_reconinter.c`) calls into the `vpx_convolve8*` family
(`vpx_dsp/vpx_dsp_rtcd_defs.pl:300+`). These are the **biggest single
SIMD surface** the non-RD path uses, but they're shared with the
decoder and **owned in the shared-kernel module** (Stage 3); A2's only
ownership is identifying that `search_filter_ref` calls it twice (once
per filter) and `model_rd_for_sb_y[_large]` runs immediately after.

---

## 7. Open questions for LLD

1. **Variance-tree representation.** libvpx uses nested structs
   (`v64x64 → v32x32 → v16x16 → v8x8 → v4x4`) with per-level
   `partition_variance` (none, horz[2], vert[2]) plus `split[4]`.
   This is ~5KB on the stack of `choose_partitioning`. Do we want to
   mirror exactly (single big stack-allocated struct, `Var` arrays of
   length 4/16/64/256 in row-major), or a flat indexed layout? A flat
   layout makes SIMD-friendly traversal easier but breaks the
   tree-walk recursion in `set_vt_partitioning`. **Recommendation:**
   mirror for Stage 4a; flatten only if Stage 5 profiling shows
   benefit.

2. **`x->variance_low[25]` indexing.** This is a hand-coded
   non-uniform packed layout (1 64×64 + 2 64×32 + 2 32×64 + 4 32×32 +
   16 16×16). `get_force_skip_low_temp_var` (`vp9_pickmode.c:1441`)
   has a 9-case `switch` to index correctly. For Rust we should
   probably wrap this in a typed enum-indexed accessor; but byte
   parity requires the bit-set logic match exactly.

3. **Mode-loop control flow.** `vp9_pick_inter_mode` has ~30
   continue/break exits. A naive Rust port can preserve them as
   `continue;`/`break;` inside a single big `for`, but readability
   would benefit from splitting into pre-screen → screen → mode-cost
   phases. **Decision needed in mode-decision LLD:** preserve the C
   structure for byte parity, or refactor and run differential tests.

4. **`block_yrd` fast-path bypass.** At speed ≥ 7 (with
   `use_simple_block_yrd`), the function bails out without calling
   Hadamard at all — just uses `model_rd_for_sb_y` numbers and
   reports `*sse = INT_MAX, *skippable = 0`. The caller then handles
   it as "not skippable, use model rate/dist". For Rust, this is two
   different return types (skip-or-not) — best modelled as
   `Option<BlockYRD>` or a dedicated `BlockYrdResult` enum.

5. **`vp9_model_rd_from_var_lapndz` Laplacian R-D model.** ~~The
   `rate_tab_q10` and `dist_tab_q10` lookup tables aren't transcribed
   here~~ **RESOLVED 2026-05-16: tables transcribed verbatim in §5.4
   (Stage 2 prep patch resolving R-07 Blocker per async A7 #2). Three
   tables (rate_tab_q10[100], dist_tab_q10[100], xsq_iq_q10[105]),
   model_rd_norm consumer, and a byte-parity note on the i32-vs-i64
   intermediate are all in §5.4.**

6. **NEWMV neighbour-difference bias** (`vp9_NEWMV_diff_bias`): The
   "above + left averaging" path falls through specific cases when one
   or both neighbours are unavailable. The magic numbers `48`, `8`,
   `16` are hardcoded — should we make them tunable in the Rust port,
   or commit to byte parity? **Recommendation:** byte parity (D8 in
   PLAN.md).

7. **`set_low_temp_var_flag` 32×32 threshold expression.**
   `(5*thresholds[1])>>3` vs `thresholds[1]>>1` is gated by
   `short_circuit_low_temp_var` being 1 or 3. For SVC + speed 6 the
   value is 1 (`vp9_speed_features.c:687`); at speed 8 + low-noise it's
   3 (line 774). The non-RD path's reading of `variance_low[i+5]` is
   sensitive to this threshold — LLD must make the speed-to-flag
   mapping explicit, not hidden inside the speed-feature dispatch.

8. **`scale_part_thresh_sumdiff` behaviour at speed 6.** Reading the
   code, speed 6 falls through both `if speed >= 8` and `elif speed
   == 7` checks → returns `threshold_base` unchanged. **Confirm**
   this is intended (i.e., speed 6 doesn't get the content-state
   scaling that speed 7 and 8 get); if so, document in LLD.

9. **Compound mode handling.** `vp9_pick_inter_mode`'s `comp_modes`
   is at most 2, and only when `sf.use_compound_nonrd_pickmode` AND
   `usable_ref_frame == ALTREF_FRAME`. The two compound modes are
   forced to ZEROMV (LAST,ALTREF) and ZEROMV (GOLDEN,ALTREF). For
   v1 PLAN says VBR+lag is supported via VBR rate control but
   compound-pred is optional. **Decide in Stage 2:** is compound-pred
   non-RD mode part of v1, or post-v1? Currently the path is
   ~20 lines (with `comp_pred = 1` flagging) but it cascades into
   `vp9_build_inter_predictors_sby` second-ref handling.

10. **`sub8x8` path picks last sub-block's mode.** `mi->mode = bmi[3].as_mode`
    at `vp9_pickmode.c:2985` — this is the mode that ends up
    in the bitstream's `mode_info_ext`, but each sub-block can have a
    different mode and MV (`bmi[i].as_mode` / `bmi[i].as_mv`). The
    decoder reads `mi->mode` for syntax disambiguation. This is
    correct per VP9 syntax, but it's a counter-intuitive piece of
    state that must survive the port.

11. **Filter-ref selection in `search_filter_ref`.** The non-RD path
    tries `EIGHTTAP` and `EIGHTTAP_SMOOTH` (lines 1512-1514) — not
    `EIGHTTAP_SHARP` — and `force_smooth_filter` (from
    `sf.force_smooth_interpol`) drops `EIGHTTAP`. There's no
    fall-back to switchable here; the picked filter is committed to
    `mi->interp_filter`. LLD: confirm we don't need
    `EIGHTTAP_SHARP` for any cpu-used 6-8 path.

12. **Threshold-freq-fact storage.** Two-axis decision:
    - per-tile (`tile_data->thresh_freq_fact[BLOCK_SIZES][MAX_MODES]`),
      or
    - per-row (`tile_data->row_base_thresh_freq_fact[sb_row * BLOCK_SIZES *
      MAX_MODES + ...]`)

    selected by `sf.adaptive_rd_thresh_row_mt` (speed-feature; on at
    speed 7+ when row_mt is on). The Rust LLD must commit to a
    backing-store representation and access pattern early — both code
    paths in `vp9_pickmode.c` are 4 lines apart but read different
    `&` accessors.

13. **`reuse_inter_pred_sby` buffer pool.** 4-entry `PRED_BUFFER tmp[]`
    on the stack with manual `in_use` flag plus `get_pred_buffer` /
    `free_pred_buffer`. This is one of the few places where libvpx
    uses an explicit refcount-ish pool inside a single function. In
    Rust this could be a stack-allocated `[Option<PredBuf>; 4]` or a
    proper RAII `PredBufHandle`. Decision needed re. allocation
    pattern that survives byte parity.

14. **Are `force_split[21]` and `variance_low[25]` overlapping
    semantics?** They're both per-block flag arrays inside the SB
    but with different indexing schemes (`force_split` is
    `[64x64=0, 4×32x32=1..4, 16×16x16=5..20]`; `variance_low` is
    `[64x64=0, 2×64x32=1..2, 2×32x64=3..4, 4×32x32=5..8,
    16×16x16=9..24]`). One controls partition descent, the other
    controls mode-skip. In the LLD it's worth unifying terminology
    (e.g. `partition_force_split` vs `mode_skip_low_temp_var`) to
    avoid confusion.

15. **`encode_breakout_test` chroma path conditional.** The code
    references `flag_preduv_computed` to decide whether to rebuild U/V
    predictors. In the Rust port, the inter-pred call is currently
    side-effecting on `pd->dst.buf` — preserving this with Rust's
    aliasing rules is non-trivial. Decision: model `pd->dst` as
    a `&mut [u8]` slice with an owner, or as a raw pointer with
    safety comments.

16. **`vp9_NEWMV_diff_bias` runs after `encode_breakout_test`.** The
    test order matters: encode_breakout reads `var, sse_y` from the
    just-computed model; NEWMV-bias modifies `this_rdc.rdcost` after
    that. For byte parity we must preserve the exact ordering.

17. **`set_vt_partitioning` `block_height == block_width`
    assertion.** Implicitly only square BLOCK_64X64/32X32/16X16/8X8
    are supported as `bsize` arguments. The code does emit
    horz/vert 64x32, 32x64, etc., but only as terminal
    `set_block_size` decisions inside the function. LLD: this is
    fine; just match.

18. **What does the partition decision do at frame edges?**
    `mi_col + 8 > cm->mi_cols || mi_row + 8 > cm->mi_rows` forces
    descent below 64×64 (line 1701). `set_vt_partitioning` then
    refuses to set partition sizes whose right/bottom halves don't
    fit (line 493-494, 510-511, 518, 531). At the leaves
    `set_block_size` only writes if the block stays inside the
    frame (line 338). LLD: the edge handling must produce the same
    `partial_b64x64` pattern as `set_partial_b64x64_partition`
    (`vp9_encodeframe.c:2344`) for `FIXED_PARTITION`.

19. **Where does `cpi->vbp_thresholds[4]` live?** It's set once per
    frame by `vp9_set_variance_partition_thresholds` (`vp9_encodeframe.c:637`),
    then read both by `choose_partitioning` (line 1289-1290) and by
    `chroma_check` (`vp9_encodeframe.c:1184`). LLD: lives on the per-frame
    state struct, not per-thread.

20. **Should the Rust port keep the 12-entry `ref_mode_set` as a
    `const` slice, or generate it at runtime keyed by `cpi->use_svc`?**
    The C uses two static tables and selects at line 2061-2063. In
    Rust an enum-indexed slice keyed on `if use_svc` is the most
    natural; for byte parity preserve order exactly.
