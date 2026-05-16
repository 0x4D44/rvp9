# Motion estimation in libvpx VP9 (real-time / non-RD path)

Specialist: A3
Source root: `third_party/libvpx/` at v1.14.1 (commit `12f3a2a`).
All file:line citations are into that subtree.

---

## 1. Scope

This document covers the libvpx VP9 motion-estimation engine **as exercised by
the real-time encoder (`vp9_pick_inter_mode` in `vp9_pickmode.c`) at
`--cpu-used` 6–8**. It includes:

- The motion-vector primitives (`MV`, `int_mv`, `MvLimits`,
  `search_site_config`, `vp9_variance_fn_ptr_t`, `vp9_sad_fn_ptr_t`).
- The reference-MV candidate machinery (`vp9_find_mv_refs`, `mv_refs_rt`,
  `vp9_find_best_ref_mvs`, `vp9_mv_pred`, the `mv_ref_blocks[]` neighbor table).
- The full-pixel search dispatcher `vp9_full_pixel_search` and every search
  variant it can dispatch to: `FAST_DIAMOND`, `FAST_HEX`, `HEX`, `BIGDIA`,
  `SQUARE`, `NSTEP`, `MESH` (plus the underlying `vp9_pattern_search` /
  `vp9_pattern_search_sad` engines, `vp9_diamond_search_sad`,
  `full_pixel_diamond`, `full_pixel_exhaustive`, `exhaustive_mesh_search`,
  `vp9_refining_search_sad`).
- The integer-projection ("int-pro") fast ME used for golden-frame NEWMV on
  CBR (`vp9_int_pro_motion_estimation`, `vector_match`).
- Sub-pixel refinement: `vp9_find_best_sub_pixel_tree`,
  `vp9_find_best_sub_pixel_tree_pruned`,
  `vp9_find_best_sub_pixel_tree_pruned_more`,
  `vp9_find_best_sub_pixel_tree_pruned_evenmore`, `vp9_skip_sub_pixel_tree`,
  plus the `FIRST_LEVEL_CHECKS` / `SECOND_LEVEL_CHECKS` / `CHECK_BETTER`
  scaffolding.
- Reference-frame selection in `vp9_pick_inter_mode` (which of LAST / GOLDEN /
  ALTREF are tested, in what order, with what early-termination).
- Real-time-specific ME shortcuts (NEWMV vs NEAREST/NEAR pruning,
  reference_masking, `limit_newmv_early_exit`, low-temp-var skipping,
  `pred_mv_sad`-based ref selection, `bias_golden`).
- The RTCD / SIMD surfaces invoked from the ME hot loop (SAD, SADx4D, variance,
  sub-pixel variance, integral projection rows/cols, hadamard, satd).

**Not covered here** (delegated to other specialists):

- Mode-decision scaffolding around the ME hot loop — partition selection,
  intra-vs-inter, skip detection, transform/quantize coupling: A2.
- Bitstream coding of MVs (`vp9_encodemv`, MV probability tables, MV joint
  trees): A4.
- Rate-control's QP/MV interactions and frame dropping: A5.
- Threading of ME workers (tile-row barriers, row_mt synchronization): A6.

---

## 2. Data structures

### 2.1 The motion vector

`vp9/common/vp9_mv.h:22-37`:

```
#define INVALID_MV 0x80008000

typedef struct mv {
  int16_t row;
  int16_t col;
} MV;

typedef union int_mv {
  uint32_t as_int;
  MV as_mv;
} int_mv; /* facilitates faster equality tests and copies */

typedef struct mv32 {
  int32_t row;
  int32_t col;
} MV32;
```

- `MV` stores **sub-pel** vectors in 1/8-pel units (Q3). The integer-pel
  vector is `row >> 3`, with `row & 7` the sub-pel offset. `int_mv` aliases
  the two 16-bit components into a single 32-bit slot so equality and
  zero-checks are one load.
- `INVALID_MV` is the sentinel `0x80008000` (both components = `-32768`);
  used in `frame_mv[NEWMV][ref]` before any search runs
  (`vp9_pickmode.c:1268`).

### 2.2 MV ranges & limits

`vp9/common/vp9_entropymv.h:79-81`:

```
#define MV_IN_USE_BITS 14
#define MV_UPP ((1 << MV_IN_USE_BITS) - 1)
#define MV_LOW (-(1 << MV_IN_USE_BITS))
```

So the encoded MV range is `[-16384, 16383]` in 1/8-pel units, i.e.
`[-2048, 2048]` integer pels.

`vp9/encoder/vp9_block.h:58-63`:

```
typedef struct {
  int col_min;
  int col_max;
  int row_min;
  int row_max;
} MvLimits;
```

Stored at `MACROBLOCK::mv_limits` (`vp9_block.h:129`). Set per-block via
`vp9_set_mv_search_range` (`vp9_mcomp.c:32-49`), which intersects the
unit-pixel motion-vector ("UMV") window with the integer-pel MV range and
the `MAX_FULL_PEL_VAL` bound.

`vp9/encoder/vp9_mcomp.h:26-34`:

```
// The maximum number of steps in a step search given the largest
// allowed initial step
#define MAX_MVSEARCH_STEPS 11
// Max full pel mv specified in the unit of full pixel
// Enable the use of motion vector in range [-1023, 1023].
#define MAX_FULL_PEL_VAL ((1 << (MAX_MVSEARCH_STEPS - 1)) - 1)
// Maximum size of the first step in full pel units
#define MAX_FIRST_STEP (1 << (MAX_MVSEARCH_STEPS - 1))
// Allowed motion vector pixel distance outside image border
// for Block_16x16
#define BORDER_MV_PIXELS_B16 (16 + VP9_INTERP_EXTEND)
```

So `MAX_FIRST_STEP = 1024`, `MAX_FULL_PEL_VAL = 1023`, and the integer-pel
diamond/N-step search expands at most 1024 -> 512 -> ... -> 1 (11 steps).

Sub-pel range is computed by `vp9_set_subpel_mv_search_range`
(`vp9_mcomp.c:51-67`); after clamping it always lies in `[MV_LOW+1, MV_UPP-1]`.

### 2.3 Reference-frame identifiers

`vp9/common/vp9_blockd.h:57-65`:

```
#define NO_REF_FRAME (-1)
#define INTRA_FRAME 0
#define LAST_FRAME 1
#define GOLDEN_FRAME 2
#define ALTREF_FRAME 3
#define MAX_REF_FRAMES 4
#define MAX_INTER_REF_FRAMES 3

typedef int8_t MV_REFERENCE_FRAME;
```

And `vp9/common/vp9_enums.h:135`: `#define MAX_MV_REF_CANDIDATES 2` — the
external candidate list per ref frame is always two MVs (nearest, near).

`MODE_INFO::ref_frame[2]` (`vp9_blockd.h:91`) and `MODE_INFO::mv[2]`
(`vp9_blockd.h:94`) hold up to two refs and two MVs (compound prediction).
The real-time non-RD path uses compound only for `(LAST, ALTREF)` and
`(GOLDEN, ALTREF)` ZEROMV pairs (`vp9_pickmode.c:2064-2070`).

### 2.4 The ref-MV neighbor search pattern

`vp9/common/vp9_mvref_common.h:24-26`:

```
#define MVREF_NEIGHBOURS 8

typedef struct position {
  int row;
  int col;
} POSITION;
```

Per-block-size neighbor offsets, `vp9_mvref_common.h:89-207` — **transcribed
verbatim**:

```
static const POSITION mv_ref_blocks[BLOCK_SIZES][MVREF_NEIGHBOURS] = {
  // 4X4
  { { -1, 0 }, { 0, -1 }, { -1, -1 }, { -2, 0 },
    { 0, -2 }, { -2, -1 }, { -1, -2 }, { -2, -2 } },
  // 4X8
  { { -1, 0 }, { 0, -1 }, { -1, -1 }, { -2, 0 },
    { 0, -2 }, { -2, -1 }, { -1, -2 }, { -2, -2 } },
  // 8X4
  { { -1, 0 }, { 0, -1 }, { -1, -1 }, { -2, 0 },
    { 0, -2 }, { -2, -1 }, { -1, -2 }, { -2, -2 } },
  // 8X8
  { { -1, 0 }, { 0, -1 }, { -1, -1 }, { -2, 0 },
    { 0, -2 }, { -2, -1 }, { -1, -2 }, { -2, -2 } },
  // 8X16
  { { 0, -1 }, { -1, 0 }, { 1, -1 }, { -1, -1 },
    { 0, -2 }, { -2, 0 }, { -2, -1 }, { -1, -2 } },
  // 16X8
  { { -1, 0 }, { 0, -1 }, { -1, 1 }, { -1, -1 },
    { -2, 0 }, { 0, -2 }, { -1, -2 }, { -2, -1 } },
  // 16X16
  { { -1, 0 }, { 0, -1 }, { -1, 1 }, { 1, -1 },
    { -1, -1 }, { -3, 0 }, { 0, -3 }, { -3, -3 } },
  // 16X32
  { { 0, -1 }, { -1, 0 }, { 2, -1 }, { -1, -1 },
    { -1, 1 }, { 0, -3 }, { -3, 0 }, { -3, -3 } },
  // 32X16
  { { -1, 0 }, { 0, -1 }, { -1, 2 }, { -1, -1 },
    { 1, -1 }, { -3, 0 }, { 0, -3 }, { -3, -3 } },
  // 32X32
  { { -1, 1 }, { 1, -1 }, { -1, 2 }, { 2, -1 },
    { -1, -1 }, { -3, 0 }, { 0, -3 }, { -3, -3 } },
  // 32X64
  { { 0, -1 }, { -1, 0 }, { 4, -1 }, { -1, 2 },
    { -1, -1 }, { 0, -3 }, { -3, 0 }, { 2, -1 } },
  // 64X32
  { { -1, 0 }, { 0, -1 }, { -1, 4 }, { 2, -1 },
    { -1, -1 }, { -3, 0 }, { 0, -3 }, { -1, 2 } },
  // 64X64
  { { -1, 3 }, { 3, -1 }, { -1, 4 }, { 4, -1 },
    { -1, -1 }, { -1, 0 }, { 0, -1 }, { -1, 6 } }
};
```

This is **load-bearing** — the order is exactly what determines which MVs
become NEAREST / NEAR candidates, and is the same table the decoder uses
when reading bitstreams.

`vp9_mvref_common.h:209-211`:

```
static const int idx_n_column_to_subblock[4][2] = {
  { 1, 2 }, { 1, 3 }, { 3, 2 }, { 3, 3 }
};
```

— picks the right `bmi[]` entry when the neighbor is split into sub-8x8s.

### 2.5 Mode-counter context tables

`vp9_mvref_common.h:31-87` — **verbatim**:

```
typedef enum {
  BOTH_ZERO = 0,
  ZERO_PLUS_PREDICTED = 1,
  BOTH_PREDICTED = 2,
  NEW_PLUS_NON_INTRA = 3,
  BOTH_NEW = 4,
  INTRA_PLUS_NON_INTRA = 5,
  BOTH_INTRA = 6,
  INVALID_CASE = 9
} motion_vector_context;

static const int mode_2_counter[MB_MODE_COUNT] = {
  9,  // DC_PRED
  9,  // V_PRED
  9,  // H_PRED
  9,  // D45_PRED
  9,  // D135_PRED
  9,  // D117_PRED
  9,  // D153_PRED
  9,  // D207_PRED
  9,  // D63_PRED
  9,  // TM_PRED
  0,  // NEARESTMV
  0,  // NEARMV
  3,  // ZEROMV
  1,  // NEWMV
};

static const int counter_to_context[19] = {
  BOTH_PREDICTED,        // 0
  NEW_PLUS_NON_INTRA,    // 1
  BOTH_NEW,              // 2
  ZERO_PLUS_PREDICTED,   // 3
  NEW_PLUS_NON_INTRA,    // 4
  INVALID_CASE,          // 5
  BOTH_ZERO,             // 6
  INVALID_CASE,          // 7
  INVALID_CASE,          // 8
  INTRA_PLUS_NON_INTRA,  // 9
  INTRA_PLUS_NON_INTRA,  // 10
  INVALID_CASE,          // 11
  INTRA_PLUS_NON_INTRA,  // 12
  INVALID_CASE,          // 13
  INVALID_CASE,          // 14
  INVALID_CASE,          // 15
  INVALID_CASE,          // 16
  INVALID_CASE,          // 17
  BOTH_INTRA             // 18
};
```

Used to derive the inter-mode context: the encoder examines the two nearest
neighbors and accumulates `mode_2_counter[mode]` into `context_counter`,
then indexes `counter_to_context` to get the context written to
`mbmi_ext->mode_context[ref_frame]`. The bitstream coder uses this to pick
the `inter_mode_probs` row for entropy.

### 2.6 Search-site configuration

`vp9/encoder/vp9_mcomp.h:36-42`:

```
typedef struct search_site_config {
  // motion search sites
  MV ss_mv[8 * MAX_MVSEARCH_STEPS];        // Motion vector
  intptr_t ss_os[8 * MAX_MVSEARCH_STEPS];  // Offset
  int searches_per_step;
  int total_steps;
} search_site_config;
```

- `ss_mv[i]`: the MV offset of site `i` (rows, cols in integer pels).
- `ss_os[i]`: the corresponding byte offset
  (`ss_mv[i].row * pre_stride + ss_mv[i].col`), pre-baked once at
  encoder init so the inner SAD loop can do `best_address + ss_os[i]`
  with a single add.
- `searches_per_step`: 4 (2-step diamond) or 8 (NSTEP 8-site diamond).
- `total_steps = ss_count / searches_per_step` = 11.

Initialized once per encoder by either `vp9_init_dsmotion_compensation`
(4-site 2-step diamond, `vp9_mcomp.c:98-114`) or `vp9_init3smotion_compensation`
(8-site NSTEP diamond, `vp9_mcomp.c:116-134`). The real-time path uses the
8-site init (`cpi->ss_cfg`, fed to `vp9_diamond_search_sad` and
`full_pixel_diamond`).

`vp9_mcomp.c:116-134` — the NSTEP 8-site pattern (verbatim):

```
void vp9_init3smotion_compensation(search_site_config *cfg, int stride) {
  int len;
  int ss_count = 0;

  for (len = MAX_FIRST_STEP; len > 0; len /= 2) {
    // Generate offsets for 8 search sites per step.
    const MV ss_mvs[8] = { { -len, 0 },   { len, 0 },     { 0, -len },
                           { 0, len },    { -len, -len }, { -len, len },
                           { len, -len }, { len, len } };
    int i;
    for (i = 0; i < 8; ++i, ++ss_count) {
      cfg->ss_mv[ss_count] = ss_mvs[i];
      cfg->ss_os[ss_count] = ss_mvs[i].row * stride + ss_mvs[i].col;
    }
  }

  cfg->searches_per_step = 8;
  cfg->total_steps = ss_count / cfg->searches_per_step;
}
```

So for each step `len ∈ {1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1}`, the
8 candidate offsets are `{(-len,0), (len,0), (0,-len), (0,len), (-len,-len),
(-len,len), (len,-len), (len,len)}` — N, S, W, E, NW, SW, NE, SE. The
`search_param` passed to `vp9_diamond_search_sad` is the *index of the
starting step* (0 = `len=1024`, 1 = `len=512`, ...). The full-pixel-step
"reduce_first_step_size" speed feature is `sf->mv.fullpel_search_step_param`
and is set to 6 (start at `len=16`) or 10 (start at `len=1`) for cpu-used
6 / 7-8 (see §5.5).

### 2.7 SAD function-pointer table

`vp9/encoder/vp9_mcomp.h:44-47`:

```
typedef struct vp9_sad_table {
  vpx_sad_fn_t sdf;
  vpx_sad_multi_d_fn_t sdx4df;
} vp9_sad_fn_ptr_t;
```

Used inside the diamond search hot loop so it can swap between full-rate
and skip-rows-by-2 SAD without re-dispatching. Set up in `full_pixel_diamond`
(`vp9_mcomp.c:2501,2517-2518,2527-2528`).

### 2.8 Variance function-pointer table

`vpx_dsp/variance.h:69-82`:

```
#if CONFIG_VP9
typedef struct vp9_variance_vtable {
  vpx_sad_fn_t sdf;
  // Same as normal sad, but downsample the rows by a factor of 2.
  vpx_sad_fn_t sdsf;
  vpx_sad_avg_fn_t sdaf;
  vpx_variance_fn_t vf;
  vpx_subpixvariance_fn_t svf;
  vpx_subp_avg_variance_fn_t svaf;
  vpx_sad_multi_d_fn_t sdx4df;
  // Same as sadx4, but downsample the rows by a factor of 2.
  vpx_sad_multi_d_fn_t sdsx4df;
} vp9_variance_fn_ptr_t;
#endif  // CONFIG_VP9
```

Stored at `VP9_COMP::fn_ptr[BLOCK_SIZES]` (`vp9_encoder.h:863`) — one entry
per block size, populated at init in `vp9_encoder.c` from the RTCD-dispatched
`vpx_sad*`, `vpx_variance*`, `vpx_sub_pixel_variance*` symbols. Function
signatures, `vpx_dsp/variance.h:25-55`:

```
typedef unsigned int (*vpx_sad_fn_t)(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *ref_ptr, int ref_stride);

typedef unsigned int (*vpx_sad_avg_fn_t)(const uint8_t *src_ptr, int src_stride,
                                         const uint8_t *ref_ptr, int ref_stride,
                                         const uint8_t *second_pred);

typedef void (*vpx_sad_multi_d_fn_t)(const uint8_t *src_ptr, int src_stride,
                                     const uint8_t *const b_array[],
                                     int ref_stride, unsigned int *sad_array);

typedef unsigned int (*vpx_variance_fn_t)(const uint8_t *src_ptr,
                                          int src_stride,
                                          const uint8_t *ref_ptr,
                                          int ref_stride, unsigned int *sse);

typedef unsigned int (*vpx_subpixvariance_fn_t)(
    const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset,
    const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);

typedef unsigned int (*vpx_subp_avg_variance_fn_t)(
    const uint8_t *src_ptr, int src_stride, int x_offset, int y_offset,
    const uint8_t *ref_ptr, int ref_stride, unsigned int *sse,
    const uint8_t *second_pred);
```

### 2.9 MACROBLOCK fields relevant to ME

`vp9/encoder/vp9_block.h:85-132` (selected fields):

- `errorperbit` — line 87 — `rdmult` scaled into MV-cost units.
- `sadperbit16`, `sadperbit4` — lines 90/93 — SAD-domain cost per bit, used
  by `mvsad_err_cost`.
- `mv_best_ref_index[MAX_REF_FRAMES]` — line 105 — index (0/1/2) of the
  best of `{ref_mvs[ref][0], ref_mvs[ref][1], pred_mv[ref]}` selected by
  `vp9_mv_pred`.
- `pred_sse[MAX_REF_FRAMES]` — line 108 — SSE between source and predicted
  block at the best MV, used by skip-detection.
- `pred_mv_sad[MAX_REF_FRAMES]` — line 109 — best SAD found during
  `vp9_mv_pred`; **drives `reference_masking`**, see §5.6.
- `nmvjointcost[MV_JOINTS]`, `nmvcost[2]`, `nmvcost_hp[2]`, `mvcost` —
  lines 111-114 — costs in `(rate << VP9_PROB_COST_SHIFT)` units for the
  4 joint outcomes (ZERO/HNZVZ/HZVNZ/HNZVNZ) and per-component magnitudes,
  built from the current frame's `nmv_context` probs.
- `nmvjointsadcost[MV_JOINTS]`, `nmvsadcost[2]`, `nmvsadcost_hp[2]`,
  `mvsadcost` — lines 116-119 — the **SAD-domain** equivalents, derived
  from a uniform-cost approximation (see `cal_nmvjointsadcost` in
  `vp9_encoder.c:2230`).
- `mv_limits` — line 129 — current `MvLimits`.
- `pred_mv[MAX_REF_FRAMES]` — line 173 — last frame's MV at the same
  position, used as the third predictor candidate.

### 2.10 MB_MODE_INFO_EXT

`vp9/encoder/vp9_block.h:50-56` (paraphrased; the relevant fields):

```
MB_MODE_INFO_EXT {
  int_mv ref_mvs[MAX_REF_FRAMES][MAX_MV_REF_CANDIDATES];
  uint8_t mode_context[MAX_REF_FRAMES];
}
```

`ref_mvs[ref][0]` is NEAREST, `ref_mvs[ref][1]` is NEAR. Filled by
`vp9_find_mv_refs` + `vp9_find_best_ref_mvs`.

---

## 3. Control flow

### 3.1 Top-level ME entry from real-time mode decision

`vp9_pick_inter_mode` (`vp9_pickmode.c:1685`) is the entry point for one
coding block (one superblock-leaf in the var-based partition). For each
reference frame it does:

1. `find_predictors` (`vp9_pickmode.c:1255`) — for each enabled ref:
   - `vp9_setup_pred_block` — point predictor planes at the right ref
     buffer.
   - **If `cm->use_prev_frame_mvs || comp_pred_allowed`:** call the full
     `vp9_find_mv_refs` from `vp9_mvref_common.c:133`.
   - **Else (the common RT path):** call the trimmed `mv_refs_rt`
     (`vp9_pickmode.c:59-152`), which walks the same neighbor table but
     only considers `ref_frame[0]` matches and exits at the first match.
   - `vp9_find_best_ref_mvs` — clamp + 1/8-pel-round the two candidates
     into `frame_mv[NEARESTMV][ref]`, `frame_mv[NEARMV][ref]`.
   - If non-scaled and `bsize >= BLOCK_8X8` and not in the
     `force_skip_low_temp_var && ref == GOLDEN_FRAME` shortcut: call
     `vp9_mv_pred` (`vp9_rd.c:588-639`). This does a small SAD-search over
     `{NEAREST, NEAR, x->pred_mv}` to set `x->mv_best_ref_index[ref]` and
     `x->pred_mv_sad[ref]`.

2. Iterate over a fixed mode-ordering table (`ref_mode_set` /
   `ref_mode_set_svc`, see §5.4). For each `(ref_frame, mode)` pair:
   - Apply early-out filters (segment-feature constraints, ref-frame
     skip mask, SVC zero-mode constraints, golden-frame SAD threshold,
     `force_skip_low_temp_var`, mode-RD-thresh, `pred_mv_sad` ratio test).
   - If `mode == NEWMV` and not `force_mv_inter_layer`: call
     `search_new_mv` (`vp9_pickmode.c:1575-1672`).
   - Otherwise the MV is just `frame_mv[mode][ref]` (NEAREST/NEAR/ZERO,
     no search), and code path goes to `vp9_build_inter_predictors_sby` +
     `model_rd_for_sb_y[_large]` + `block_yrd`.

3. `search_new_mv` (`vp9_pickmode.c:1575`):
   - For **GOLDEN/ALTREF** in CBR with `gf_temporal_ref` and `bsize >= 16x16`:
     try `vp9_int_pro_motion_estimation` (the **integer projection ME**,
     `vp9_mcomp.c:2264-2400`) first, and early-out if its SAD beats the
     LAST-frame predicted SAD by enough — saves a full diamond search for
     GOLDEN-NEWMV.
   - For SVC `use_base_mv && spatial_layer_id > 0`: call
     `combined_motion_search` with `use_base_mv=1` if base-layer SSE is
     promising, else fall back to base-layer-free search.
   - Otherwise: `combined_motion_search(..., use_base_mv=0)` —
     `vp9_pickmode.c:154-261`. This sets up the search state, dispatches
     `vp9_full_pixel_search` (full-pel) then `cpi->find_fractional_mv_step`
     (sub-pel), and finally calls `vp9_mv_bit_cost` to compute the actual
     rate.

### 3.2 `vp9_full_pixel_search` call graph

`vp9/encoder/vp9_mcomp.c:2875-2952` — dispatcher:

```
vp9_full_pixel_search(cpi, x, bsize, mvp_full, step_param, search_method,
                     error_per_bit, cost_list, ref_mv, tmp_mv, var_max, rd)
  switch (search_method):
    FAST_DIAMOND : fast_dia_search   -> bigdia_search   -> vp9_pattern_search_sad
    FAST_HEX     : fast_hex_search   -> hex_search      -> vp9_pattern_search
    HEX          : hex_search                          -> vp9_pattern_search
    SQUARE       : square_search                       -> vp9_pattern_search
    BIGDIA       : bigdia_search                       -> vp9_pattern_search_sad
    NSTEP, MESH  : full_pixel_diamond
                     -> cpi->diamond_search_sad (= vp9_diamond_search_sad
                                                  or vp9_diamond_search_sad_neon)
                     -> vp9_refining_search_sad (final 1-away refine if do_refine)
  // for NSTEP and MESH only:
  if (var > exhaustive_thr or method == MESH):
    full_pixel_exhaustive
      -> exhaustive_mesh_search   (multi-stage)
```

- `fast_dia_search` / `fast_hex_search` (`vp9_mcomp.c:1614-1631`) cap
  `search_param` to `MAX(MAX_MVSEARCH_STEPS-2, search_param) = MAX(9, sp)`,
  so they start at step 9 (`len=2`) or smaller — only the two innermost
  scales get searched. The cheapest search variants.
- `bigdia_search`, `hex_search`, `square_search`, `fast_*`: all share
  the same `vp9_pattern_search[_sad]` engine; they differ only in the
  candidate-offset tables (transcribed in §5.1) and in
  `num_candidates[]`.
- `full_pixel_diamond` (`vp9_mcomp.c:2486-2605`) drives `n-step` searches
  using `cpi->diamond_search_sad` — by default the C version
  `vp9_diamond_search_sad_c` (`vp9_mcomp.c:2055-2190`); on AArch64 NEON the
  RTCD-dispatched `vp9_diamond_search_sad_neon` replaces it
  (`vp9/common/vp9_rtcd_defs.pl:176-177`).
- `vp9_refining_search_sad` (`vp9_mcomp.c:2749-2821`) is a 4-neighbor
  refinement loop; runs after `full_pixel_diamond` if `do_refine=1` and
  also can be invoked stand-alone after exhaustive search.

### 3.3 Sub-pixel refinement call graph

`vp9_pickmode.c:246-251,1608-1615,2903-2910` — call site:

```
cpi->find_fractional_mv_step(x, bestmv, ref_mv, allow_hp,
                             errorperbit, fn_ptr, subpel_force_stop,
                             subpel_search_level, cost_list, mvjcost, mvcost,
                             &dis, &pred_sse[ref], second_pred,
                             w, h, use_accurate_subpel_search)
```

`cpi->find_fractional_mv_step` is selected once at `vp9_set_speed_features_*`
time (`vp9_speed_features.c:1057-1068`):

```
if (sf->mv.subpel_force_stop == FULL_PEL)
  cpi->find_fractional_mv_step = vp9_skip_sub_pixel_tree;
else if (sf->mv.subpel_search_method == SUBPEL_TREE)
  cpi->find_fractional_mv_step = vp9_find_best_sub_pixel_tree;
else if (sf->mv.subpel_search_method == SUBPEL_TREE_PRUNED)
  cpi->find_fractional_mv_step = vp9_find_best_sub_pixel_tree_pruned;
else if (sf->mv.subpel_search_method == SUBPEL_TREE_PRUNED_MORE)
  cpi->find_fractional_mv_step = vp9_find_best_sub_pixel_tree_pruned_more;
else if (sf->mv.subpel_search_method == SUBPEL_TREE_PRUNED_EVENMORE)
  cpi->find_fractional_mv_step = vp9_find_best_sub_pixel_tree_pruned_evenmore;
```

For real-time speeds 0-8 the default is `SUBPEL_TREE`
(`vp9_speed_features.c:928`); cpu-used 8 with
`svc->non_reference_frame` overrides to `SUBPEL_TREE_PRUNED_EVENMORE`
(`vp9_speed_features.c:715`). All five subpel variants live in
`vp9_mcomp.c:354-925`. See §4.4 for each algorithm.

### 3.4 Reference-frame selection flow in `vp9_pick_inter_mode`

`vp9/encoder/vp9_pickmode.c:1907-2001`:

1. **`usable_ref_frame` selection** (lines 1907-1929):
   - Default: `usable_ref_frame = GOLDEN_FRAME` (so iteration goes
     `LAST_FRAME, GOLDEN_FRAME`).
   - If `cpi->rc.frames_since_golden == 0 && gf_temporal_ref &&
      !rc.alt_ref_gf_group && !rc.last_frame_is_src_altref`:
     `usable_ref_frame = LAST_FRAME` (skip golden when golden was just
     refreshed and is a duplicate of last).
   - If VBR with `lag_in_frames > 0`: extend to `ALTREF_FRAME`. For
     `is_src_frame_alt_ref` set `skip_ref_find_pred[LAST]` and
     `skip_ref_find_pred[GOLDEN]` so only ALTREF is searched.

2. **SVC constraints** (lines 1931-1949): if any ref is scaled and we're on
   a higher spatial layer with `force_zero_mode_spatial_ref`, set
   `svc_force_zero_mode[ref-1]=1` — only ZEROMV is later allowed for that
   ref.

3. **`short_circuit_low_temp_var`** (cpu-used >= 6, lines 1951-1961): if the
   variance-based partition flagged this block as low-temporal-variance, set
   `usable_ref_frame = LAST_FRAME` (no golden) for short-circuit modes 1/3.

4. **`disable_golden_ref`** (cpu-used >= 9 CBR, lines 1963-1965): if neither
   high-sad nor low-motion content, drop golden.

5. **Speed >= 8 heuristic** (lines 1971-1974): if a recent run of high-content
   SBs is over the `last_sb_high_content` threshold or
   `frames_since_golden > 120`, drop golden.

6. **`find_predictors` loop** (lines 1991-2001): iterate `ref_frame` from
   `LAST_FRAME` to `usable_ref_frame`; for each, if it's in
   `cpi->ref_frame_flags`, populate its NEAREST/NEAR candidates and
   `pred_mv_sad`. The "search" itself is **only triggered later** when the
   mode loop hits a `NEWMV` for that ref.

7. **Mode loop** (lines 2039-2478): iterates `ref_mode_set[]` (or
   `ref_mode_set_svc[]`). The order is **(ref_frame outer, mode inner)**
   interleaved — `LAST_ZEROMV, LAST_NEAREST, GOLDEN_ZEROMV, LAST_NEAR,
   LAST_NEW, GOLDEN_NEAREST, GOLDEN_NEAR, GOLDEN_NEW, ALTREF_ZEROMV, ...`
   (see §5.4). This ordering means cheap candidates (ZEROMV, NEAREST) are
   tested first across **all** refs before NEWMV is searched on **any** ref,
   so early-termination via `best_early_term` / `x->skip` works without
   wasting search cycles on later refs.

### 3.5 Sub-block (sub-8x8) ME

For BLOCK_4X4/4X8/8X4 the encoder calls `vp9_pick_inter_mode_sub8x8`
(`vp9_pickmode.c:2750-2992`). Same dispatch but `find_mv_refs` uses
`vp9_find_mv_refs` directly (full version, not the `_rt` variant), and the
NEWMV inner loop calls `vp9_full_pixel_search` with `sadperbit4` instead of
`sadperbit16` (`vp9_pickmode.c:2884-2887`).

---

## 4. Algorithms

### 4.1 `vp9_diamond_search_sad_c` (NSTEP 8-point diamond)

`vp9_mcomp.c:2055-2190`.

```
inputs:  start MV (full-pel), start_mv_sad, search_param (start step idx)
output:  best MV (full-pel), num00 (# of "center stayed best" steps)

ss_mv  = &cfg->ss_mv[search_param * 8]
ss_os  = &cfg->ss_os[search_param * 8]
tot_steps = cfg->total_steps - search_param         // 11 - search_param

best_mv = ref_mv (clamped)
best_address = in_what + ref_row*stride + ref_col
bestsad = start_mv_sad
i = 0
last_site = -1
for step in 0..tot_steps-1:
  all_in = (the 4 axis-aligned candidates at indices i, i+1, i+2, i+3
            stay inside mv_limits)
  if all_in:
    # Process 8 sites in two SADx4D batches.
    for j in 0..7 step 4:
      sdx4df(what, what_stride,
             {best_address + ss_os[i+0..i+3]}, in_what_stride,
             sad_array)
      for t in 0..3:
        if sad_array[t] < bestsad:
          this_mv = best_mv + ss_mv[i]
          sad_array[t] += mvsad_err_cost(this_mv, fcenter_mv, sad_per_bit)
          if sad_array[t] < bestsad:
            bestsad = sad_array[t]; best_site = i
        i++
  else:
    # Per-site bounds check, scalar SAD.
    for j in 0..7:
      this_mv = best_mv + ss_mv[i]
      if is_mv_in(mv_limits, this_mv):
        thissad = sdf(what, what_stride, best_address + ss_os[i],
                      in_what_stride)
        if thissad < bestsad:
          thissad += mvsad_err_cost(...)
          if thissad < bestsad:
            bestsad = thissad; best_site = i
      i++

  if best_site != last_site:
    best_mv += ss_mv[best_site]; best_address += ss_os[best_site]
    last_site = best_site
  elif best_address == in_what:
    # center stayed best from the very start
    num00++
return bestsad
```

The `all_in` check at lines 2104-2107 looks at sites `i`, `i+1`, `i+2`,
`i+3` (north, south, west, east axis-aligned), which suffices because
those are the outermost-reaching of the 8 candidates for any step. The
"single-best-site step" model means each step uses **at most 8 SADs**, and
all 8 are issued as **two 4-way batched SADs** in the fast path. Across the
11 steps that's worst-case 88 SADs at 8 sites/step + 4 SADs for the
1-away refine = ~92 SAD computations per ME, against a brute-force
`(2*1024+1)^2 ≈ 4.2M`.

The `num00` accumulates "center-stayed-best" trailing steps; if the next
n-step search would only refine within `num00` steps of where we already
converged, `full_pixel_diamond` skips it.

### 4.2 `full_pixel_diamond` (multi-step NSTEP wrapper)

`vp9_mcomp.c:2486-2605`.

```
clamp mvp_full to mv_limits
start_mv_sad_even = sdsf(src_even, pred_even)       # downsampled SAD even rows
start_mv_sad_odd  = sdsf(src_odd,  pred_odd)
start_mv_sad      = avg(even, odd) + mvsad_err_cost(...)

# Decide whether to use downsampled SAD in the search itself
sad_fn_ptr.{sdf, sdx4df} = fn_ptr->{sdf, sdx4df}
if use_downsampled_sad and bsize tall enough and even-vs-odd diff is small:
  sad_fn_ptr.{sdf, sdx4df} = fn_ptr->{sdsf, sdsx4df}

bestsme = cpi->diamond_search_sad(x, ss_cfg, mvp_full, start_mv_sad,
                                  &temp_mv, step_param, sadpb, &n,
                                  &sad_fn_ptr, ref_mv)
if bestsme < INT_MAX:
  bestsme = vp9_get_mvpred_var(x, temp_mv, ref_mv, fn_ptr, 1)
*dst_mv = temp_mv

# Further n-step passes at finer starting step (step_param+1, +2, ...)
# Each pass refines into the converged region of the previous one.
while n < further_steps:
  n++
  if num00 > 0: num00--
  else:
    thissme = cpi->diamond_search_sad(... step_param + n ...)
    if thissme < bestsme: bestsme = thissme; dst_mv = temp_mv

if do_refine:
  thissme = vp9_refining_search_sad(x, dst_mv, sadpb, 8, sad_fn_ptr, ref_mv)
  if better: take it

# Quality check: if downsampled-SAD was used, verify final result against
# full SAD. If they disagree by more than 10%, redo the entire search
# with use_downsampled_sad=0.
if sad_fn_ptr.sdf != fn_ptr->sdf:
  sad      = fn_ptr->sdf (best_addr)
  skip_sad = fn_ptr->sdsf(best_addr)
  if sad > kSADThresh and |skip - sad| * 10 >= sad * 9:
    return full_pixel_diamond(..., use_downsampled_sad=0, ...)  // recurse
```

The recursive fallback at `vp9_mcomp.c:2591-2598` is a real-time correctness
guard against the `sdsf` (downsample-rows) variant making a bad pick.

### 4.3 `vp9_pattern_search` (hex / bigdia / square engine)

`vp9_mcomp.c:1007-1171`. Generic engine driving HEX, BIGDIA, SQUARE,
FAST_HEX, FAST_DIAMOND. Two-stage:

```
Stage 1 — initial scan:
  s = search_param_to_steps[search_param]     // see §5.3, maps 0..10 to 10..0
  best_init_s = -1
  for t in 0..s:                              // try each scale 0..s
    for i in 0..num_candidates[t]-1:
      this_mv = (br, bc) + candidates[t][i]
      if in_bounds: thissad = sdf(...)
      CHECK_BETTER                            // mv-cost-augmented update
    if best_site found at this t: best_init_s = t; k = best_site
  if best_init_s >= 0:
    (br, bc) += candidates[best_init_s][k]

Stage 2 — refinement starting at best_init_s, descending scales:
  s = best_init_s
  do:
    if !do_init_search or s != best_init_s:
      # Full scan at this scale.
      for i in 0..num_candidates[s]-1: CHECK_BETTER, update (br, bc), k
    # Tight refinement: check only the 3 candidates adjacent to k.
    do:
      next_chkpts = { (k-1) mod n, k, (k+1) mod n }   // PATTERN_CANDIDATES_REF
      for each: CHECK_BETTER
    while moved
  while s-- > 0
```

`PATTERN_CANDIDATES_REF = 3` (`vp9_mcomp.c:957`). The trick is that **at the
top of stage 2 the previous `k` indexes into the candidate ring**, so the
"refine" pass only re-checks the previous best and its two ring neighbors —
not the full ring. This makes hex / bigdia converge with O(n_scales * 3)
SADs after the initial scale-selection pass.

`vp9_pattern_search_sad` (`vp9_mcomp.c:1177-1452`) is a specialization for
the case where the smallest-scale (`s=0`) candidates are 4 1-away neighbors
*and* the caller wants a populated `cost_list[5]` (needed for the
"check-cost-list-well-behaved" prune in `vp9_find_best_sub_pixel_tree_pruned*`).
It does the same two-stage thing but caches the 4 final-scale SADs into
`cost_list[1..4]`.

`CHECK_BETTER` macro (`vp9_mcomp.c:943-953`):

```
#define CHECK_BETTER                                                      \
  {                                                                       \
    if (thissad < bestsad) {                                              \
      if (use_mvcost)                                                     \
        thissad += mvsad_err_cost(x, &this_mv, &fcenter_mv, sad_per_bit); \
      if (thissad < bestsad) {                                            \
        bestsad = thissad;                                                \
        best_site = i;                                                    \
      }                                                                   \
    }                                                                     \
  }
```

Note the two-step check — first prune by raw SAD vs `bestsad` (which already
contains MV cost), only then add MV cost and re-compare. Saves the cost
multiplication on >95% of sites.

### 4.4 Sub-pixel refinement variants

#### Common scaffolding

The four "tree" variants share `FIRST_LEVEL_CHECKS` and
`SECOND_LEVEL_CHECKS` macros (`vp9_mcomp.c:202-248`):

```
#define FIRST_LEVEL_CHECKS                                       \
  do {                                                           \
    unsigned int left, right, up, down, diag;                    \
    CHECK_BETTER(left, tr, tc - hstep);                          \
    CHECK_BETTER(right, tr, tc + hstep);                         \
    CHECK_BETTER(up, tr - hstep, tc);                            \
    CHECK_BETTER(down, tr + hstep, tc);                          \
    whichdir = (left < right ? 0 : 1) + (up < down ? 0 : 2);     \
    switch (whichdir) {                                          \
      case 0: CHECK_BETTER(diag, tr - hstep, tc - hstep); break; \
      case 1: CHECK_BETTER(diag, tr - hstep, tc + hstep); break; \
      case 2: CHECK_BETTER(diag, tr + hstep, tc - hstep); break; \
      case 3: CHECK_BETTER(diag, tr + hstep, tc + hstep); break; \
    }                                                            \
  } while (0)

#define SECOND_LEVEL_CHECKS                                       \
  do {                                                            \
    int kr, kc;                                                   \
    unsigned int second;                                          \
    if (tr != br && tc != bc) {                                   \
      kr = br - tr; kc = bc - tc;                                 \
      CHECK_BETTER(second, tr + kr, tc + 2 * kc);                 \
      CHECK_BETTER(second, tr + 2 * kr, tc + kc);                 \
    } else if (tr == br && tc != bc) {                            \
      kc = bc - tc;                                               \
      CHECK_BETTER(second, tr + hstep, tc + 2 * kc);              \
      CHECK_BETTER(second, tr - hstep, tc + 2 * kc);              \
      switch (whichdir) {                                         \
        case 0: case 1: CHECK_BETTER(second, tr + hstep, tc + kc); break; \
        case 2: case 3: CHECK_BETTER(second, tr - hstep, tc + kc); break; \
      }                                                           \
    } else if (tr != br && tc == bc) {                            \
      kr = br - tr;                                               \
      CHECK_BETTER(second, tr + 2 * kr, tc + hstep);              \
      CHECK_BETTER(second, tr + 2 * kr, tc - hstep);              \
      switch (whichdir) {                                         \
        case 0: case 2: CHECK_BETTER(second, tr + kr, tc + hstep); break; \
        case 1: case 3: CHECK_BETTER(second, tr + kr, tc - hstep); break; \
      }                                                           \
    }                                                             \
  } while (0)
```

(Inside subpel `CHECK_BETTER` (`vp9_mcomp.c:176-199`) calls
`vfp->svf(...)` or `vfp->svaf(...)` — sub-pixel variance — not SAD.)

The starting state at the top of each tree (`SETUP_SUBPEL_SEARCH`,
`vp9_mcomp.c:250-282`) takes the integer-pel `bestmv` from full-pixel
search, scales to 1/8-pel units (`br = bestmv->row * 8`), sets `hstep = 4`
(half-pel), and calls `setup_center_error` to seed `besterr` with the
variance of the integer-pel candidate.

`forced_stop` semantics (from `SUBPEL_FORCE_STOP` enum,
`vp9_speed_features.h:168`):

```
typedef enum { EIGHTH_PEL, QUARTER_PEL, HALF_PEL, FULL_PEL } SUBPEL_FORCE_STOP;
```

`forced_stop = 0` → 1/8-pel allowed, `1` → quarter-pel only, `2` → half-pel
only, `3` → full-pel (skip subpel entirely, route to `vp9_skip_sub_pixel_tree`).

#### `vp9_find_best_sub_pixel_tree` (default for cpu-used 6-8)

`vp9_mcomp.c:721-925`. Uses a fixed `search_step_table` (12-entry, 3 rounds
of 4 cardinal steps at hstep=4, 2, 1):

`vp9_mcomp.c:599-606`:

```
/* clang-format off */
static const MV search_step_table[12] = {
  // left, right, up, down
  { 0, -4 }, { 0, 4 }, { -4, 0 }, { 4, 0 },
  { 0, -2 }, { 0, 2 }, { -2, 0 }, { 2, 0 },
  { 0, -1 }, { 0, 1 }, { -1, 0 }, { 1, 0 }
};
/* clang-format on */
```

Algorithm:

```
besterr = vfp->vf(y + offset, src, src_stride, &sse1)       // variance at integer-pel
         + mv_err_cost(bestmv, ref_mv, ...)
round = 3 - forced_stop                                     // 3, 2, 1, or 0
if not (allow_hp and use_mv_hp(ref_mv)): round = min(round, 2)
br = bestmv->row * 8; bc = bestmv->col * 8
hstep = 4
search_step = &search_step_table[0]

for iter in 0..round-1:
  # 4 cardinal sub-pel samples
  for idx in 0..3:
    tr = br + search_step[idx].row
    tc = bc + search_step[idx].col
    if in subpel bounds:
      if use_accurate_subpel_search:
        thismse = accurate_sub_pel_search(...)              // build 4-tap/8-tap pred, then vf
      else:
        thismse = vfp->svf(y + (tr>>3)*y_stride + (tc>>3), ..., sp(tc), sp(tr), src, ...)
      cost_array[idx] = thismse + mv_err_cost(this_mv, ref_mv, ...)
      if cost_array[idx] < besterr: best_idx = idx; besterr = ...

  # 1 diagonal sample at the inferred best diagonal
  kc = (cost_array[0] <= cost_array[1] ? -hstep : hstep)    // L vs R
  kr = (cost_array[2] <= cost_array[3] ? -hstep : hstep)    // U vs D
  test (br+kr, bc+kc); cost_array[4] = ...
  if better: best_idx = 4

  if best_idx < 4: br += search_step[best_idx].row; bc += search_step[best_idx].col
  elif best_idx == 4: br = tr; bc = tc

  # iters_per_step extra second-level checks
  if iters_per_step > 0 and best_idx != -1: ... (CHECK_BETTER on adjacent)

  search_step += 4
  hstep >>= 1
  best_idx = -1
```

Total samples per iter: 5 (4 cardinal + 1 diagonal). With `round=2`
(half-pel + quarter-pel), that's 10 sub-pel-variance evaluations per ME.
With `subpel_search_level >= 1` extra second-level checks add 1-3 more
each iter.

#### `vp9_find_best_sub_pixel_tree_pruned`

`vp9_mcomp.c:515-597`. Same first level but **uses the integer-pel
`cost_list[5]`** populated by `calc_int_cost_list` (or `vp9_pattern_search_sad`)
to skip 2 of the 4 cardinal half-pel checks: it figures out the
"hottest quadrant" from cost_list and only tests the 3 sites toward that
quadrant.

```
if cost_list is well-formed:
  whichdir = (cost_list[1] < cost_list[3] ? 0 : 1)          // left vs right
           + (cost_list[2] < cost_list[4] ? 0 : 2)          // bottom vs top
  switch (whichdir):
    case 0 (L,B): test {left, down, diag(down-left)}
    case 1 (R,B): test {right, down, diag(down-right)}
    case 2 (L,T): test {left, up, diag(up-left)}
    case 3 (R,T): test {right, up, diag(up-right)}
else:
  FIRST_LEVEL_CHECKS; if halfiters > 1: SECOND_LEVEL_CHECKS
```

Then quarter-pel + eighth-pel are identical to plain `tree`.

#### `vp9_find_best_sub_pixel_tree_pruned_more`

`vp9_mcomp.c:453-513`. Replaces the 5-direction cross at the half-pel
level with a **single point** derived from the integer-pel cost surface
quadratic fit:

```
get_cost_surf_min(cost_list, &ir, &ic, bits=1)              // see §5.7
if (ir != 0 or ic != 0):
  CHECK_BETTER(minpt, tr + ir*hstep, tc + ic*hstep)
```

Falls through to the same `FIRST_LEVEL_CHECKS / SECOND_LEVEL_CHECKS` if
the cost surface is ill-conditioned.

#### `vp9_find_best_sub_pixel_tree_pruned_evenmore`

`vp9_mcomp.c:387-451`. Even more aggressive: at the half-pel level, only
the quadratic-fit minimum is tested (using **2-bit precision** instead of
1-bit), and quarter-pel iteration is gated by `forced_stop != 2`. Used by
cpu-used 7-8 SVC non-reference frames.

#### `vp9_skip_sub_pixel_tree`

`vp9_mcomp.c:354-385`. Skips sub-pel entirely; just returns the integer-pel
position with its variance. Selected when `sf->mv.subpel_force_stop = FULL_PEL`
(cpu-used 8 screen content, `vp9_speed_features.c:769`).

### 4.5 `exhaustive_mesh_search` and `full_pixel_exhaustive`

`vp9_mcomp.c:1637-1721, 2609-2662`. Brute-force rectangular scan in steps
of `step` (with 4-way SAD batching when `step == 1`). Three-phase
mesh refinement: each pass narrows `range` and shrinks `step`, breaking
when `step == 1`.

In the real-time path `sf->exhaustive_searches_thresh = INT_MAX`
(`vp9_speed_features.c:462`), so `full_pixel_exhaustive` is **never
triggered** unless explicitly forced by `method == MESH`. The real-time
search-method dispatch never picks `MESH`. Effectively dead code for
cpu-used 6-8 — kept here for completeness.

### 4.6 `vp9_refining_search_sad`

`vp9_mcomp.c:2749-2821`. 4-neighbor refinement loop:

```
neighbors = { (-1,0), (0,-1), (0,1), (1,0) }
best_address = ref_mv buffer pointer
best_sad = sdf(src, best_address) + mvsad_err_cost(ref_mv, fcenter_mv, error_per_bit)
for i in 0..search_range-1:
  best_site = -1
  if all_in (1-pel margin from mv_limits):
    sdx4df(src, {best_addr - stride, -1, +1, +stride}, sads)
    for j in 0..3:
      if sads[j] < best_sad:
        sads[j] += mvsad_err_cost(...)
        if sads[j] < best_sad: best_sad = sads[j]; best_site = j
  else:
    scalar per-site bounds check + sdf
  if best_site == -1: break
  ref_mv += neighbors[best_site]; best_address = get_buf_from_mv(in_what, ref_mv)
return best_sad
```

Called by `full_pixel_diamond` with `search_range=8`. Terminates as soon
as the center is the best.

### 4.7 `vp9_int_pro_motion_estimation` (integer projection)

`vp9_mcomp.c:2264-2400`. Fast ME by 1-D projection — used for
GOLDEN/ALTREF NEWMV in CBR (`vp9_pickmode.c:1586-1615`) when the LAST-frame
SAD is good enough that a full diamond on golden is wasteful.

```
bw, bh = block dims
search_width  = bw << 1, search_height = bh << 1
norm_factor = 3 + (bw >> 5)

# Build 1-D horizontal projection of reference (search_width samples)
for idx in 0..search_width step 16:
  vpx_int_pro_row(&hbuf[idx], ref_buf + idx, ref_stride, bh)

# Build 1-D vertical projection of reference (search_height samples)
for idx in 0..search_height-1:
  vbuf[idx] = vpx_int_pro_col(ref_buf, bw) >> norm_factor

# Same for source.
... vpx_int_pro_row(src_hbuf, src, ...) for bw step 16 ...
... src_vbuf[idx] = vpx_int_pro_col(src, bw) >> norm_factor ...

# 1-D log searches.
tmp_mv.col = vector_match(hbuf, src_hbuf, bwl)
tmp_mv.row = vector_match(vbuf, src_vbuf, bhl)

# Refine: check the 4-neighbors of (tmp_mv.row, tmp_mv.col) with SAD.
best_sad = sdf(src, ref + tmp_mv*stride)
sdx4df(src, {ref - stride, ref - 1, ref + 1, ref + stride}, this_sad)
pick best of {original, 4 neighbors}; possibly try one diagonal too.

tmp_mv *= 8  // back to 1/8-pel
clamp to subpel_mv_limits
return best_sad
```

`vector_match` (`vp9_mcomp.c:2192-2255`) is itself a log-step search at
strides 16, 8, 4, 2, 1 over the 1-D buffers, using `vpx_vector_var` (an L1
distance on int16 arrays, SIMD-dispatched). Sees roughly 5+5+5+5+3 ≈ 23
1-D variance evaluations per `vector_match`, plus the 5-9 2-D SADs to
refine. **An order of magnitude cheaper** than even `FAST_DIAMOND`.

The catch: if `tmp_sad > x->pred_mv_sad[LAST_FRAME]` or
`tmp_sad + bsize_factor > best_pred_sad`, `search_new_mv` returns `-1`
(skip NEWMV on golden entirely) (`vp9_pickmode.c:1598-1599`).

### 4.8 `vp9_find_mv_refs` (full neighbor scan)

`vp9_mvref_common.c:16-131,133-139`. Walks `mv_ref_blocks[bsize][0..7]`:

1. For the first 2 entries: count modes into `context_counter` and try to
   match `ref_frame[0]==ref` or `ref_frame[1]==ref`; add MV to list.
2. For entries 2..7: same matching, no counting.
3. If `cm->use_prev_frame_mvs`: check the co-located MV from the previous
   frame.
4. If list still has slots: re-scan all 8 neighbors looking for any
   *different* ref frame's MV, scaled by `ref_sign_bias`.
5. If still missing: prev-frame fallback again, scaled.
6. Set `mode_context[ref_frame] = counter_to_context[context_counter]`.
7. Clamp each candidate via `clamp_mv_ref` (`vp9_mvref_common.h:216-220`,
   which clamps to `xd->mb_to_*_edge ± MV_BORDER` where
   `MV_BORDER = (16 << 3) = 128`).

`vp9_find_best_ref_mvs` (`vp9_mvref_common.c:141-151`) then applies
`lower_mv_precision` (round to even if not allow_hp) and `clamp_mv2` to
both candidates and emits them as `nearest_mv` and `near_mv`.

### 4.9 `mv_refs_rt` (real-time fast variant)

`vp9_pickmode.c:59-152`. The non-RD path's neighbor scan. Differences from
the full version:

- Only considers `candidate_mi->ref_frame[0] == ref_frame` matches (skips
  the `ref_frame[1]` second-ref check).
- The loop after the first 2 neighbors **exits at the first match**
  (`for (; i < MVREF_NEIGHBOURS && !refmv_count; ++i)` at line 99) — so
  it doesn't fill 2 candidates from the spatial neighborhood, just 1.
- Adds a base-layer-MV fast path for SVC: if non-keyframe and
  `ref_frame == LAST_FRAME`, grab the co-located MV from the lower spatial
  layer (`prev_frame->mvs[(mi_col>>1) + (mi_row>>1) * (mi_cols>>1)]`),
  scaled ×2, into `base_mv` (lines 127-141).
- Returns `const_motion` (1 if the first nearest match was found in the
  first 2 neighbors), used as a flag to skip NEARMV later
  (`vp9_pickmode.c:2165`: `if (const_motion[ref_frame] && this_mode == NEARMV) continue`).

### 4.10 `vp9_mv_pred` (best-of-3 candidate selection)

`vp9_rd.c:588-639`. Picks the best starting MV among
`{ref_mvs[ref][0], ref_mvs[ref][1], x->pred_mv[ref]}` by raw SAD (no MV cost):

```
num_mv_refs = MAX_MV_REF_CANDIDATES + (block_size < x->max_partition_size)
              # i.e. 2 or 3
pred_mv[0..2] = { ref_mvs[ref][0].as_mv, ref_mvs[ref][1].as_mv, x->pred_mv[ref] }
near_same_nearest = (ref_mvs[ref][0] == ref_mvs[ref][1])
zero_seen = false
best_sad = INT_MAX; best_index = 0
for i in 0..num_mv_refs-1:
  if pred_mv[i].row == INT16_MAX or .col == INT16_MAX: continue
  if i == 1 and near_same_nearest: continue
  fp_row = (row + 3 + (row >= 0)) >> 3                       # round to full-pel
  fp_col = ...
  if fp_row == 0 and fp_col == 0 and zero_seen: continue
  zero_seen |= (fp == 0)
  max_mv = max(max_mv, max(|row|, |col|) >> 3)
  this_sad = sdf(src, ref + fp_row*stride + fp_col, ...)
  if this_sad < best_sad: best_sad = this_sad; best_index = i

x->mv_best_ref_index[ref_frame] = best_index
x->max_mv_context[ref_frame]    = max_mv
x->pred_mv_sad[ref_frame]       = best_sad
```

`pred_mv_sad` is then the "we know this much is achievable" lower bound
that drives `reference_masking` (see §5.6) and golden-skip thresholds.

`combined_motion_search` picks the starting point at
`vp9_pickmode.c:193-200`:

```
assert(x->mv_best_ref_index[ref] <= 2);
if (x->mv_best_ref_index[ref] < 2)
  mvp_full = x->mbmi_ext->ref_mvs[ref][x->mv_best_ref_index[ref]].as_mv;
else
  mvp_full = x->pred_mv[ref];
mvp_full.col >>= 3; mvp_full.row >>= 3;        // 1/8-pel -> full-pel
```

So the diamond/N-step starts at whichever of NEAREST/NEAR/PRED_MV had the
lowest SAD.

### 4.11 MV-cost computation

`vp9_mcomp.c:80-97`:

```
int vp9_mv_bit_cost(const MV *mv, const MV *ref, const int *mvjcost,
                    int *mvcost[2], int weight) {
  const MV diff = { mv->row - ref->row, mv->col - ref->col };
  return ROUND_POWER_OF_TWO(mv_cost(&diff, mvjcost, mvcost) * weight, 7);
}

#define PIXEL_TRANSFORM_ERROR_SCALE 4
static int mv_err_cost(const MV *mv, const MV *ref, const int *mvjcost,
                       int *mvcost[2], int error_per_bit) {
  if (mvcost) {
    const MV diff = { mv->row - ref->row, mv->col - ref->col };
    return (int)ROUND64_POWER_OF_TWO(
        (int64_t)mv_cost(&diff, mvjcost, mvcost) * error_per_bit,
        RDDIV_BITS + VP9_PROB_COST_SHIFT - RD_EPB_SHIFT +
            PIXEL_TRANSFORM_ERROR_SCALE);
  }
  return 0;
}
```

And `vp9_encoder.h:1510-1525`:

```
static INLINE int mv_cost(const MV *mv, const int *joint_cost,
                          int *const comp_cost[2]) {
  assert(mv->row >= -MV_MAX && mv->row < MV_MAX);
  assert(mv->col >= -MV_MAX && mv->col < MV_MAX);
  return joint_cost[vp9_get_mv_joint(mv)] + comp_cost[0][mv->row] +
         comp_cost[1][mv->col];
}

static INLINE int mvsad_err_cost(const MACROBLOCK *x, const MV *mv,
                                 const MV *ref, int sad_per_bit) {
  MV diff;
  diff.row = mv->row - ref->row;
  diff.col = mv->col - ref->col;
  return ROUND_POWER_OF_TWO(
      (unsigned)mv_cost(&diff, x->nmvjointsadcost, x->nmvsadcost) * sad_per_bit,
      VP9_PROB_COST_SHIFT);
}
```

Three flavors of cost depending on caller:

| Caller | Cost source | Scaling |
|---|---|---|
| Full-pel SAD search | `nmvjointsadcost`, `nmvsadcost` (uniform-approx) | `sad_per_bit`, shifted by `VP9_PROB_COST_SHIFT = 9` |
| Sub-pel variance search | `nmvjointcost`, `mvcost` (real probabilities) | `error_per_bit`, shifted by `RDDIV_BITS + VP9_PROB_COST_SHIFT - RD_EPB_SHIFT + PIXEL_TRANSFORM_ERROR_SCALE` |
| Mode-decision bit-rate accounting | `nmvjointcost`, `mvcost` | `MV_COST_WEIGHT = 108`, shifted by 7 |

`MV_COST_WEIGHT = 108` and `MV_COST_WEIGHT_SUB = 120` come from
`vp9_rd.h:38-39`. `VP9_PROB_COST_SHIFT = 9` from `vp9_cost.h:24`.
`PIXEL_TRANSFORM_ERROR_SCALE = 4` from `vp9_mcomp.c:86`.

---

## 5. Tables and constants

### 5.1 Search patterns

**HEX** — `vp9_mcomp.c:1502-1525` (verbatim):

```
static const int hex_num_candidates[MAX_PATTERN_SCALES] = { 8, 6, 6, 6, 6, 6,
                                                            6, 6, 6, 6, 6 };
/* clang-format off */
static const MV hex_candidates[MAX_PATTERN_SCALES][MAX_PATTERN_CANDIDATES] = {
  { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 },
    { -1, 0 } },
  { { -1, -2 }, { 1, -2 }, { 2, 0 }, { 1, 2 }, { -1, 2 }, { -2, 0 } },
  { { -2, -4 }, { 2, -4 }, { 4, 0 }, { 2, 4 }, { -2, 4 }, { -4, 0 } },
  { { -4, -8 }, { 4, -8 }, { 8, 0 }, { 4, 8 }, { -4, 8 }, { -8, 0 } },
  { { -8, -16 }, { 8, -16 }, { 16, 0 }, { 8, 16 }, { -8, 16 }, { -16, 0 } },
  { { -16, -32 }, { 16, -32 }, { 32, 0 }, { 16, 32 }, { -16, 32 },
    { -32, 0 } },
  { { -32, -64 }, { 32, -64 }, { 64, 0 }, { 32, 64 }, { -32, 64 },
    { -64, 0 } },
  { { -64, -128 }, { 64, -128 }, { 128, 0 }, { 64, 128 }, { -64, 128 },
    { -128, 0 } },
  { { -128, -256 }, { 128, -256 }, { 256, 0 }, { 128, 256 }, { -128, 256 },
    { -256, 0 } },
  { { -256, -512 }, { 256, -512 }, { 512, 0 }, { 256, 512 }, { -256, 512 },
    { -512, 0 } },
  { { -512, -1024 }, { 512, -1024 }, { 1024, 0 }, { 512, 1024 },
    { -512, 1024 }, { -1024, 0 } }
};
/* clang-format on */
```

Note: scale-0 has 8 candidates (full 1-away ring), all other scales have
the 6-point hex.

**BIGDIA** — `vp9_mcomp.c:1538-1566` (verbatim):

```
static const int bigdia_num_candidates[MAX_PATTERN_SCALES] = {
  4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
};
/* clang-format off */
static const MV
    bigdia_candidates[MAX_PATTERN_SCALES][MAX_PATTERN_CANDIDATES] = {
      { { 0, -1 }, { 1, 0 }, { 0, 1 }, { -1, 0 } },
      { { -1, -1 }, { 0, -2 }, { 1, -1 }, { 2, 0 }, { 1, 1 }, { 0, 2 },
        { -1, 1 }, { -2, 0 } },
      { { -2, -2 }, { 0, -4 }, { 2, -2 }, { 4, 0 }, { 2, 2 }, { 0, 4 },
        { -2, 2 }, { -4, 0 } },
      { { -4, -4 }, { 0, -8 }, { 4, -4 }, { 8, 0 }, { 4, 4 }, { 0, 8 },
        { -4, 4 }, { -8, 0 } },
      { { -8, -8 }, { 0, -16 }, { 8, -8 }, { 16, 0 }, { 8, 8 }, { 0, 16 },
        { -8, 8 }, { -16, 0 } },
      { { -16, -16 }, { 0, -32 }, { 16, -16 }, { 32, 0 }, { 16, 16 },
        { 0, 32 }, { -16, 16 }, { -32, 0 } },
      { { -32, -32 }, { 0, -64 }, { 32, -32 }, { 64, 0 }, { 32, 32 },
        { 0, 64 }, { -32, 32 }, { -64, 0 } },
      { { -64, -64 }, { 0, -128 }, { 64, -64 }, { 128, 0 }, { 64, 64 },
        { 0, 128 }, { -64, 64 }, { -128, 0 } },
      { { -128, -128 }, { 0, -256 }, { 128, -128 }, { 256, 0 }, { 128, 128 },
        { 0, 256 }, { -128, 128 }, { -256, 0 } },
      { { -256, -256 }, { 0, -512 }, { 256, -256 }, { 512, 0 }, { 256, 256 },
        { 0, 512 }, { -256, 256 }, { -512, 0 } },
      { { -512, -512 }, { 0, -1024 }, { 512, -512 }, { 1024, 0 },
        { 512, 512 }, { 0, 1024 }, { -512, 512 }, { -1024, 0 } }
    };
/* clang-format on */
```

Scale-0 here is 4-point (cardinal); all other scales are 8-point diamonds.

**SQUARE** — `vp9_mcomp.c:1578-1607` (verbatim):

```
static const int square_num_candidates[MAX_PATTERN_SCALES] = {
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
};
/* clang-format off */
static const MV
    square_candidates[MAX_PATTERN_SCALES][MAX_PATTERN_CANDIDATES] = {
      { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 },
        { -1, 1 }, { -1, 0 } },
      { { -2, -2 }, { 0, -2 }, { 2, -2 }, { 2, 0 }, { 2, 2 }, { 0, 2 },
        { -2, 2 }, { -2, 0 } },
      { { -4, -4 }, { 0, -4 }, { 4, -4 }, { 4, 0 }, { 4, 4 }, { 0, 4 },
        { -4, 4 }, { -4, 0 } },
      { { -8, -8 }, { 0, -8 }, { 8, -8 }, { 8, 0 }, { 8, 8 }, { 0, 8 },
        { -8, 8 }, { -8, 0 } },
      { { -16, -16 }, { 0, -16 }, { 16, -16 }, { 16, 0 }, { 16, 16 },
        { 0, 16 }, { -16, 16 }, { -16, 0 } },
      { { -32, -32 }, { 0, -32 }, { 32, -32 }, { 32, 0 }, { 32, 32 },
        { 0, 32 }, { -32, 32 }, { -32, 0 } },
      { { -64, -64 }, { 0, -64 }, { 64, -64 }, { 64, 0 }, { 64, 64 },
        { 0, 64 }, { -64, 64 }, { -64, 0 } },
      { { -128, -128 }, { 0, -128 }, { 128, -128 }, { 128, 0 }, { 128, 128 },
        { 0, 128 }, { -128, 128 }, { -128, 0 } },
      { { -256, -256 }, { 0, -256 }, { 256, -256 }, { 256, 0 }, { 256, 256 },
        { 0, 256 }, { -256, 256 }, { -256, 0 } },
      { { -512, -512 }, { 0, -512 }, { 512, -512 }, { 512, 0 }, { 512, 512 },
        { 0, 512 }, { -512, 512 }, { -512, 0 } },
      { { -1024, -1024 }, { 0, -1024 }, { 1024, -1024 }, { 1024, 0 },
        { 1024, 1024 }, { 0, 1024 }, { -1024, 1024 }, { -1024, 0 } }
    };
/* clang-format on */
```

**Diamond NSTEP (vp9_init3smotion_compensation)** — see §2.6 above; the
8-site offsets at step `len` are
`{(-len,0), (len,0), (0,-len), (0,len), (-len,-len), (-len,len), (len,-len), (len,len)}`
generated programmatically from `MAX_FIRST_STEP=1024` down to 1.

### 5.2 Sub-pixel search step table

`vp9_mcomp.c:599-606`:

```
/* clang-format off */
static const MV search_step_table[12] = {
  // left, right, up, down
  { 0, -4 }, { 0, 4 }, { -4, 0 }, { 4, 0 },
  { 0, -2 }, { 0, 2 }, { -2, 0 }, { 2, 0 },
  { 0, -1 }, { 0, 1 }, { -1, 0 }, { 1, 0 }
};
/* clang-format on */
```

Three rounds × 4 cardinal directions: half-pel (±4 in 1/8-pel units),
quarter-pel (±2), eighth-pel (±1).

### 5.3 search_param-to-step mapping

`vp9_mcomp.c:1014-1016` and `vp9_mcomp.c:1184-1186`:

```
static const int search_param_to_steps[MAX_MVSEARCH_STEPS] = {
  10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
};
```

`search_param` indexes into this. So `search_param=0` → start at the
finest scale (`s=10`), `search_param=10` → start at the coarsest scale
(`s=0`). For NSTEP search via `cfg->ss_mv`, the semantics are different
and inverted: `search_param=0` selects step 0 (largest, `len=1024`) and
the search descends. Be careful — these are two different "search_param"
namespaces.

### 5.4 Real-time inter-mode iteration order

`vp9_pickmode.c:1237-1253`:

```
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

Key insight: the **non-SVC table** orders `LAST_NEW` before *both* GOLDEN
NEAREST/NEAR (positions 4 vs 5,6) — a deliberate choice so that if LAST
NEWMV produces a great match, the early-term/`x->skip` paths kick in and
we never search GOLDEN at all. The **SVC table** does the opposite: tests
*all* of LAST's cheap modes and *all* of GOLDEN's cheap modes (positions
0-5) before any NEWMV search (positions 6-7), because SVC has a strong
inter-layer-ZEROMV signal.

`MB_MODE_COUNT` indexing: ZEROMV=12, NEARESTMV=10, NEARMV=11, NEWMV=13
(from `vp9_blockd.h` PREDICTION_MODE enum). `INTER_OFFSET(NEARESTMV)=0`,
`INTER_OFFSET(NEARMV)=1`, `INTER_OFFSET(ZEROMV)=2`, `INTER_OFFSET(NEWMV)=3`
(used for `inter_mode_cost[]` indexing).

### 5.5 Speed-feature settings affecting ME for cpu-used 6, 7, 8

Filtered subset of `set_rt_speed_feature_framesize_independent` at
`vp9_speed_features.c:452-793`. The "MV" prefix means
`SPEED_FEATURES::mv.*`.

| Field | Speed 5 | Speed 6 | Speed 7 | Speed 8 |
|---|---|---|---|---|
| `mv.search_method` | `HEX` | `NSTEP` | `FAST_DIAMOND` (or `NSTEP` for SVC base) | `FAST_DIAMOND` (inherits) |
| `mv.fullpel_search_step_param` | 6 (default) | 6 (default) | **10** (or 6 SVC base) | 10 |
| `mv.subpel_search_method` | `SUBPEL_TREE` | `SUBPEL_TREE` | `SUBPEL_TREE` (or `SUBPEL_TREE_PRUNED_EVENMORE` for SVC non-ref) | inherits |
| `mv.subpel_force_stop` | `EIGHTH_PEL` | `EIGHTH_PEL` | inherits | `HALF_PEL` for screen content; else inherits |
| `mv.reduce_first_step_size` | 1 | 1 | inherits | inherits |
| `mv.auto_mv_step_size` | 1 (default) | inherits | inherits | inherits |
| `mv.use_downsampled_sad` | 1 (default) | inherits | inherits | inherits |
| `mv.enable_adaptive_subpel_force_stop` | 0 | 0 | 0 | 0 (1 at speed 9) |
| `use_accurate_subpel_search` | `USE_2_TAPS` | inherits | inherits | inherits |
| `short_circuit_low_temp_var` | 0 | 1 (CBR non-screen) | inherits | **2 or 3** |
| `limit_newmv_early_exit` | 1 (CBR non-screen) | 1 | 0 (SVC temp>0) | 0 |
| `bias_golden` | 1 (CBR non-screen, non-SVC) | inherits | inherits | inherits |
| `base_mv_aggressive` | 0 | 1 (SVC temp>0) | inherits | inherits |
| `reference_masking` | 1 (1-layer, no scale) | inherits | inherits | inherits |
| `disable_golden_ref` | 0 | 0 | 0 | 0 (1 at speed 9 CBR) |

Sources for the table:

- `mv.search_method`: `vp9_speed_features.c:569` (speed 4 sets FAST_HEX,
  speed 5 sets HEX at line 615 / set_good); for **rt** path, speed 5
  doesn't override and stays at the rt-default `NSTEP` (line 926); speed 6
  sets `NSTEP` at line 668; speed 7 sets `FAST_DIAMOND` at line 702 with a
  special-case at line 709 for SVC base.
- `mv.fullpel_search_step_param`: default 6 (line 935); speed 7 sets 10
  (line 703) with SVC override to 6 (line 710).
- `mv.subpel_search_method`: default `SUBPEL_TREE` (line 928); SVC non-ref
  at speed 7 sets `SUBPEL_TREE_PRUNED_EVENMORE` (line 715).
- `mv.subpel_force_stop`: default not set explicitly for rt 6-8 except
  speed 8 screen → `HALF_PEL` (line 769).
- `mv.reduce_first_step_size`: speed 6 sets 1 (line 669); speed 5 in good
  sets 1 (line 408).
- `mv.use_downsampled_sad`: default 1 (line 234, set in good-path; in rt
  it's inherited from the default-init at line 935).

The default rt initialization is at `vp9_speed_features.c:925-940`:

```
sf->mv.search_method = NSTEP;
sf->mv.reduce_first_step_size = 0;
sf->mv.subpel_search_method = SUBPEL_TREE;
sf->mv.subpel_force_stop = EIGHTH_PEL;
sf->mv.subpel_search_level = 1;        // adjusted to 0 at speed 3 (good only)
sf->mv.fullpel_search_step_param = 6;
sf->mv.auto_mv_step_size = 0;
```

(Then the speed-N blocks override.)

### 5.6 reference_masking heuristic

`vp9_pickmode.c:2194-2216`:

```
if (sf->reference_masking && !(this_mv == 0 && ref == LAST)) {
  if (usable_ref_frame < ALTREF):
    if !force_skip_low_temp_var and usable_ref_frame > LAST:
      i = (ref == LAST) ? GOLDEN : LAST
      if (ref_frame_flags & flag(i)):
        if x->pred_mv_sad[ref] > (x->pred_mv_sad[i] << 1):
          ref_frame_skip_mask |= (1 << ref)
  else if not is_src_frame_alt_ref and not (this_mv == 0 && ref == ALTREF):
    ref1 = (ref == GOLDEN) ? LAST : GOLDEN
    ref2 = (ref == ALTREF) ? LAST : ALTREF
    if pred_mv_sad[ref] > pred_mv_sad[ref1] << 1
       or pred_mv_sad[ref] > pred_mv_sad[ref2] << 1:
      ref_frame_skip_mask |= (1 << ref)
}
if ref_frame_skip_mask & (1 << ref): continue
```

So if my SAD on the current ref is more than 2× the SAD on the other ref,
I skip all non-ZERO modes on this ref. This is the main reason GOLDEN
search is often elided in stationary content.

### 5.7 `get_cost_surf_min` quadratic fit

`vp9_mcomp.c:336-352`:

```
// Returns surface minima estimate at given precision in 1/2^n bits.
// Assume a model for the cost surface: S = A(x - x0)^2 + B(y - y0)^2 + C
// For a given set of costs S0, S1, S2, S3, S4 at points
// (y, x) = (0, 0), (0, -1), (1, 0), (0, 1) and (-1, 0) respectively,
// the solution for the location of the minima (x0, y0) is given by:
// x0 = 1/2 (S1 - S3)/(S1 + S3 - 2*S0),
// y0 = 1/2 (S4 - S2)/(S4 + S2 - 2*S0).
// The code below is an integerized version of that.
static void get_cost_surf_min(int *cost_list, int *ir, int *ic, int bits) {
  const int64_t x0 = (int64_t)cost_list[1] - cost_list[3];
  const int64_t y0 = cost_list[1] - 2 * (int64_t)cost_list[0] + cost_list[3];
  const int64_t x1 = (int64_t)cost_list[4] - cost_list[2];
  const int64_t y1 = cost_list[4] - 2 * (int64_t)cost_list[0] + cost_list[2];
  const int b = 1 << (bits - 1);
  *ic = (int)divide_and_round(x0 * b, y0);
  *ir = (int)divide_and_round(x1 * b, y1);
}
```

Layout of `cost_list[5]`:
- `[0]` = center
- `[1]` = left (0, -1)
- `[2]` = bottom (1, 0)
- `[3]` = right (0, 1)
- `[4]` = top (-1, 0)

This matches the comments at `vp9_mcomp.c:1161-1166`.

### 5.8 Other ME-relevant constants

| Constant | Value | File:line |
|---|---|---|
| `MAX_MVSEARCH_STEPS` | 11 | `vp9_mcomp.h:26` |
| `MAX_FULL_PEL_VAL` | 1023 | `vp9_mcomp.h:29` |
| `MAX_FIRST_STEP` | 1024 | `vp9_mcomp.h:31` |
| `BORDER_MV_PIXELS_B16` | `16 + VP9_INTERP_EXTEND` | `vp9_mcomp.h:34` |
| `MAX_PATTERN_SCALES` | 11 | `vp9_mcomp.c:955` |
| `MAX_PATTERN_CANDIDATES` | 8 | `vp9_mcomp.c:956` |
| `PATTERN_CANDIDATES_REF` | 3 | `vp9_mcomp.c:957` |
| `MIN_RANGE` | 7 | `vp9_mcomp.c:1723` |
| `MAX_RANGE` | 256 | `vp9_mcomp.c:1724` |
| `MIN_INTERVAL` | 1 | `vp9_mcomp.c:1725` |
| `MAX_MESH_STEP` | 4 | `vp9_speed_features.h:224` |
| `MVREF_NEIGHBOURS` | 8 | `vp9_mvref_common.h:24` |
| `MAX_MV_REF_CANDIDATES` | 2 | `vp9_enums.h:135` |
| `MV_BORDER` | `(16 << 3) = 128` (1/8-pel units) | `vp9_mvref_common.h:214` |
| `LEFT_TOP_MARGIN`, `RIGHT_BOTTOM_MARGIN` | `(VP9_ENC_BORDER_IN_PIXELS - VP9_INTERP_EXTEND) << 3` | `vp9_mvref_common.h:20-22` |
| `MV_COST_WEIGHT` | 108 | `vp9_rd.h:38` |
| `MV_COST_WEIGHT_SUB` | 120 | `vp9_rd.h:39` |
| `VP9_PROB_COST_SHIFT` | 9 | `vp9_cost.h:24` |
| `PIXEL_TRANSFORM_ERROR_SCALE` | 4 | `vp9_mcomp.c:86` |
| `MV_LOW` | -16384 | `vp9_entropymv.h:81` |
| `MV_UPP` | 16383 | `vp9_entropymv.h:80` |
| `MV_IN_USE_BITS` | 14 | `vp9_entropymv.h:79` |
| `INVALID_MV` | `0x80008000` | `vp9_mv.h:22` |
| `INTER_LAYER_REF` flag-bit defaults | various | `vp9_speed_features.c:769-806` |
| `kMvRefThresh` (use_mv_hp threshold) | 64 | `vp9_entropymv.h:31` |

`use_mv_hp(ref)`: returns true iff both `|ref->row| < 64` and `|ref->col| < 64`
(1/8-pel units, i.e. less than 8 integer pels). Controls whether 1/8-pel
search is allowed.

### 5.9 Adaptive subpel force-stop (speed >= 9 only)

`vp9_speed_features.c:803-806`:

```
sf->mv.enable_adaptive_subpel_force_stop = 1;
sf->mv.adapt_subpel_force_stop.mv_thresh = 1;
sf->mv.adapt_subpel_force_stop.force_stop_below = QUARTER_PEL;
sf->mv.adapt_subpel_force_stop.force_stop_above = HALF_PEL;
```

`combined_motion_search` checks at `vp9_pickmode.c:238-245`:

```
if (cpi->sf.mv.enable_adaptive_subpel_force_stop) {
  const int mv_thresh = cpi->sf.mv.adapt_subpel_force_stop.mv_thresh;
  if (abs(tmp_mv->as_mv.row) >= mv_thresh ||
      abs(tmp_mv->as_mv.col) >= mv_thresh)
    subpel_force_stop = sf->mv.adapt_subpel_force_stop.force_stop_above;
  else
    subpel_force_stop = sf->mv.adapt_subpel_force_stop.force_stop_below;
}
```

So at cpu-used 9 we degrade to half-pel for any non-tiny MV. Outside our
scope (D1 only covers 6-8), noted for completeness.

---

## 6. RTCD / SIMD surfaces

The ME hot loop is dominated by SAD, SADx4D, sub-pixel-variance, and the
two integral-projection kernels. **All of these are RTCD-dispatched** in
libvpx via `vpx_dsp/vpx_dsp_rtcd_defs.pl`.

### 6.1 SAD (single-source single-destination)

`vpx_dsp/vpx_dsp_rtcd_defs.pl:753-790` — one `vpx_sad{w}x{h}` per block
size from 64×64 down to 4×4 (13 sizes total in 8-bit, double in HBD).

Signature: `unsigned int(const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride)`.

ISA coverage:

| Size | sse2 | avx2 | neon | neon_dotprod | msa | vsx | mmi | lsx |
|---|---|---|---|---|---|---|---|---|
| 64x64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 64x32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| 32x64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| 32x32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 32x16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| 16x32 | ✓ | | ✓ | ✓ | ✓ | ✓ | ✓ | |
| 16x16 | ✓ | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 16x8  | ✓ | | ✓ | ✓ | ✓ | ✓ | ✓ | |
| 8x16  | ✓ | | ✓ | | ✓ | ✓ | ✓ | |
| 8x8   | ✓ | | ✓ | | ✓ | ✓ | ✓ | ✓ |
| 8x4   | ✓ | | ✓ | | ✓ | ✓ | ✓ | |
| 4x8   | ✓ | | ✓ | | ✓ | | ✓ | |
| 4x4   | ✓ | | ✓ | | ✓ | | ✓ | |

For the **scope of our Rust port (target x86_64 + aarch64)** the relevant
ISAs are: scalar fallback, sse2, avx2, neon, neon_dotprod (ARMv8.4+).

### 6.2 SAD skip (downsampled rows)

`vpx_dsp/vpx_dsp_rtcd_defs.pl:792-829`. `vpx_sad_skip_{w}x{h}` — same signature as
SAD but only reads even rows (factor-of-2 vertical downsample). Used by
`fn_ptr->sdsf`. ISA coverage trimmed (no msa/vsx/mmi/lsx, sse2 + avx2 +
neon + neon_dotprod).

### 6.3 SAD-avg (compound reference)

`vpx_dsp/vpx_dsp_rtcd_defs.pl:891-928`. `vpx_sad{w}x{h}_avg`:
`unsigned int(... const uint8_t *second_pred)`. Used by `fn_ptr->sdaf`,
which is called only from `vp9_refining_search_8p_c` for joint-search
during compound NEWMV. Not used in the rt non-RD path.

### 6.4 SAD x4D (4 references, single source)

`vpx_dsp/vpx_dsp_rtcd_defs.pl:933-970`. `vpx_sad{w}x{h}x4d`:

```
void(const uint8_t *src, int src_stride,
     const uint8_t *const ref_array[4], int ref_stride,
     uint32_t sad_array[4])
```

This is the **single most performance-critical SIMD entry point** in ME,
because the NSTEP diamond search is structured to issue 4 SADs at a time
(8 sites = 2 batches). ISA coverage: avx512(only 64×64), avx2(64x64,
32x32), sse2 (all), neon / neon_dotprod (all), msa / vsx / mmi / lsx.

### 6.5 SAD skip x4D

`vpx_dsp/vpx_dsp_rtcd_defs.pl:972-1009`. `vpx_sad_skip_{w}x{h}x4d`. Used by
`sad_fn_ptr->sdsx4df` when `use_downsampled_sad` is on and the
even-vs-odd-rows discrepancy is small.

### 6.6 Variance

`vpx_dsp/vpx_dsp_rtcd_defs.pl:1262-1299`. `vpx_variance{w}x{h}`:

```
unsigned int(const uint8_t *src, int src_stride,
             const uint8_t *ref, int ref_stride,
             unsigned int *sse)
```

13 sizes. ISA: sse2 (all), avx2 (≥8x4), neon (all), neon_dotprod (all),
msa / mmi / vsx / lsx (most). Used at the *seed* of every sub-pel
search (the "center error") and for the final `vp9_get_mvpred_var`
re-evaluation.

### 6.7 Sub-pixel variance

`vpx_dsp/vpx_dsp_rtcd_defs.pl:1334-1371`. `vpx_sub_pixel_variance{w}x{h}`:

```
uint32_t(const uint8_t *src, int src_stride,
         int x_offset, int y_offset,
         const uint8_t *ref, int ref_stride,
         uint32_t *sse)
```

`x_offset, y_offset ∈ [0..7]` are the 1/8-pel sub-pel phases. Internally
the implementation applies the 8-tap (or 4-tap if `USE_4_TAPS`, or bilinear
if `use_accurate_subpel_search == 0`) sub-pel filter to the reference
then computes variance against the source. **This is the per-iteration
core of every sub-pel-tree variant.** ISA coverage: sse2 + ssse3 + avx2 (64x64,
32x32 only), neon, msa, mmi.

For the rt path with the default `use_accurate_subpel_search = USE_2_TAPS`,
the encoder uses **bilinear** sub-pel interpolation — the `vpx_sub_pixel_variance*`
SIMD already does the bilinear-filtered variance in one fused kernel.

### 6.8 Sub-pixel-avg variance

`vpx_dsp/vpx_dsp_rtcd_defs.pl:1373-1410`. `vpx_sub_pixel_avg_variance{w}x{h}`:

```
uint32_t(... const uint8_t *second_pred)
```

Used by `fn_ptr->svaf` when sub-pel searching compound-mode predictions.
Not used in the rt non-RD path beyond ZEROMV-compound, where MV refinement
is skipped.

### 6.9 Compound average prediction

`vpx_dsp/vpx_dsp_rtcd_defs.pl:1328-1329`. `vpx_comp_avg_pred`:

```
void(uint8_t *comp_pred, const uint8_t *pred, int width, int height,
     const uint8_t *ref, int ref_stride)
```

Used by `setup_center_error` in sub-pel search when compound (`second_pred != NULL`),
and by `accurate_sub_pel_search`. ISA: neon, sse2, avx2, vsx, lsx.

### 6.10 Integral projection

`vpx_dsp/vpx_dsp_rtcd_defs.pl:882-888`:

```
add_proto qw/void vpx_int_pro_row/, "int16_t hbuf[16], const uint8_t *ref, const int ref_stride, const int height";
specialize qw/vpx_int_pro_row neon sse2 msa/;
add_proto qw/int16_t vpx_int_pro_col/, "const uint8_t *ref, const int width";
specialize qw/vpx_int_pro_col neon sse2 msa/;

add_proto qw/int vpx_vector_var/, "const int16_t *ref, const int16_t *src, const int bwl";
specialize qw/vpx_vector_var neon sse2 msa/;
```

The three kernels used by `vp9_int_pro_motion_estimation` (§4.7).

### 6.11 Hadamard + SATD

`vpx_dsp/vpx_dsp_rtcd_defs.pl:845-879`:

```
add_proto qw/void vpx_hadamard_8x8/,   "const int16_t *src_diff, ptrdiff_t src_stride, [tran_low_t|int16_t] *coeff";
specialize qw/vpx_hadamard_8x8   sse2 neon vsx lsx [msa]/, "$ssse3_x86_64";
add_proto qw/void vpx_hadamard_16x16/, "const int16_t *src_diff, ptrdiff_t src_stride, [tran_low_t|int16_t] *coeff";
specialize qw/vpx_hadamard_16x16 avx2 sse2 neon [msa] vsx lsx/;
add_proto qw/void vpx_hadamard_32x32/, "const int16_t *src_diff, ptrdiff_t src_stride, [tran_low_t|int16_t] *coeff";
specialize qw/vpx_hadamard_32x32 sse2 avx2 neon/;

add_proto qw/int vpx_satd/, "const [tran_low_t|int16_t] *coeff, int length";
specialize qw/vpx_satd avx2 sse2 neon [msa]/;
```

Hadamard transforms + SATD (sum-of-absolute-transformed-differences). Used
by the **mode-decision** RD model in pickmode (`block_yrd` /
`model_rd_for_sb_y_large`) rather than in motion search itself, but
because they're invoked once per candidate MV per block size they appear
in any ME profile of the rt encoder. Documented here so the LLD writer
knows to schedule them alongside the SAD kernels.

### 6.12 Single VP9-specific ME-SIMD entry: vp9_diamond_search_sad

`vp9/common/vp9_rtcd_defs.pl:176-177`:

```
add_proto qw/int vp9_diamond_search_sad/, "const struct macroblock *x, const struct search_site_config *cfg,  struct mv *ref_mv, uint32_t start_mv_sad, struct mv *best_mv, int search_param, int sad_per_bit, int *num00, const struct vp9_sad_table *sad_fn_ptr, const struct mv *center_mv";
specialize qw/vp9_diamond_search_sad neon/;
```

This is the only ME-flavored kernel that has a vp9-specific SIMD override
(rather than just dispatching through `vp9_variance_fn_ptr_t`). The NEON
version (`vp9/encoder/arm/neon/vp9_diamond_search_sad_neon.c`) is a
hand-tuned re-implementation of the **entire** `vp9_diamond_search_sad_c`
loop, exploiting the symmetry of `nmvsadcost` (`mvsadcost[0][i] ==
mvsadcost[1][i]`, even in `i`) noted in the file's header comment
(`vp9_diamond_search_sad_neon.c:32-46`).

For our Rust port: do **not** plan to port this as a separate kernel;
the scalar implementation of `vp9_diamond_search_sad_c` plus a fast
`vpx_sad{w}x{h}x4d_avx2` is competitive. The NEON-specific routine in
libvpx exists because NEON's lack of efficient horizontal reductions
makes it advantageous to fuse the cost-vs-bound logic with the SAD batch
issue.

---

## 7. Open questions for the LLD writer

1. **Search-site-config initialization.** Do we keep `ss_mv` and `ss_os`
   as separate parallel arrays, or pack into a single
   `Vec<(MV, isize)>`? `ss_os` depends on stride, which depends on the
   reference frame buffer's stride, which depends on width — so it has to
   be re-initialized when resolution changes. libvpx does this lazily in
   `init_motion_estimation` at frame start. Resolve when sizing
   reference-frame structs.

2. **Diamond-search SIMD policy.** libvpx only ships AArch64-NEON for
   `vp9_diamond_search_sad`. For x86_64 it falls through to the C version,
   relying on the `sdx4df` AVX2 SAD-x4D batching to do the heavy lifting.
   We should do the same — *no* AVX2 port of the full diamond-search
   driver — but the LLD should explicitly state this so a future
   contributor doesn't waste a week on it.

3. **`use_downsampled_sad` quality-check recursion.** `full_pixel_diamond`
   may recurse into itself with `use_downsampled_sad=0` if the
   even/odd-rows SAD disagrees with full-rows SAD by ≥10%
   (`vp9_mcomp.c:2589-2598`). Plain `if/while` won't translate cleanly to
   Rust — needs a parameter or two-arm helper. Trivial but worth noting.

4. **Sub-pel-search filter dispatch.** `accurate_sub_pel_search` selects
   `BILINEAR / FOURTAP / EIGHTTAP / EIGHTTAP_SHARP` based on
   `use_accurate_subpel_search` (`vp9_mcomp.c:756-763`). For our rt path
   (cpu-used 6-8) the value is `USE_2_TAPS` (bilinear), so the
   `accurate_sub_pel_search` branch is **never taken** — the inline
   `vfp->svf` path runs the fused bilinear sub-pel-variance kernel. LLD
   should decide whether to wire `accurate_sub_pel_search` at all in v1,
   or stub it. Probably stub: it's only used for cpu-used ≤ 3 (good-quality
   modes) which D1 puts out of scope.

5. **NEWMV vs NEAREST/NEAR de-duplication.** The pickmode loop has two
   distinct de-dup tests:
   - lines 2255-2267: skip a mode if a *previously checked* mode for the
     same ref had the same `as_int` MV and the MV was zero.
   - lines 2285-2288: skip a mode (not NEARESTMV) if its MV equals the
     NEARESTMV.
   Both are correctness-required for matching libvpx output. Worth a
   dedicated LLD comment because the order of checks matters: after the
   first dedup, NEAREST is always "ahead" of NEAR/ZERO/NEW for the same
   ref.

6. **`reduce_first_step_size` vs `fullpel_search_step_param`.** Both are
   `MV_SPEED_FEATURES` fields. In the rt non-RD path the search starts
   from `fullpel_search_step_param`; `reduce_first_step_size` is used by
   `vp9_mbgraph.c`, `vp9_temporal_filter.c`, `vp9_tpl_model.c` (all of
   which are *not* in the rt encode hot path). LLD should clarify which
   knobs are reachable from a real-time encode session.

7. **`pred_mv_sad`-based reference masking thresholds.** The `<< 1`
   (2× threshold) at `vp9_pickmode.c:2201, 2210-2213` is hard-coded; some
   downstream forks have made this adaptive. Decide whether to expose it
   as a tunable in our LLD or freeze at 2×.

8. **`vp9_int_pro_motion_estimation` for non-CBR.** Currently gated on
   `gf_temporal_ref && cpi->oxcf.rc_mode == VPX_CBR && bsize >= BLOCK_16X16`
   (`vp9_pickmode.c:1586-1587, 1592`). If our rt path eventually supports
   VBR, the LLD needs to settle whether to extend this fast-path or
   restrict it to CBR.

9. **Compound-mode NEWMV joint search.** `vp9_refining_search_8p_c`
   (`vp9_mcomp.c:2825-2873`) is the iterative-refinement engine used by
   compound-pred mode decision. In the rt path compound is restricted to
   ZEROMV pairs (`vp9_pickmode.c:2064-2070`), so `vp9_refining_search_8p_c`
   is **not reached** in the rt non-RD path. LLD can stub this out for v1
   and revisit if/when full compound-NEWMV is enabled.

10. **Sub-pel cost-list contract.** The four sub-pel-pruned variants
    (`_pruned`, `_pruned_more`, `_pruned_evenmore`) all require the
    `cost_list[5]` to be populated by the full-pel search (either by
    `vp9_pattern_search_sad` directly or `calc_int_cost_list`). The skip
    variants (`vp9_skip_sub_pixel_tree`, `vp9_find_best_sub_pixel_tree`)
    don't need it. **If the cost_list is uninitialized (all INT_MAX) the
    pruned variants fall back to FIRST_LEVEL_CHECKS.** LLD should be
    explicit about who fills `cost_list[]` and when, because it's an
    invariant that can silently degrade quality if broken.

11. **`auto_mv_step_size` adaptive step.** Disabled by default in rt
    (`vp9_speed_features.c:935`). Enabled at speed 1+ in good-quality
    (line 233). Not in our scope, but worth noting that if a future rt
    config enables it the `step_param` plumbing through
    `combined_motion_search` would need to absorb a per-frame adjustment.

12. **`vp9_mv_pred` 3rd-candidate condition.** The "third candidate"
    `pred_mv[2] = x->pred_mv[ref_frame]` is only used if
    `block_size < x->max_partition_size` (`vp9_rd.c:599-600`). For our
    var-based partition path, `max_partition_size` is normally
    `BLOCK_64X64`, so the third candidate is used for everything except
    full-SB blocks. LLD should re-derive this when settling the partition
    type.
