# LLD: loopfilter

## 1. Purpose

`rvp9-core::loopfilter` owns the VP9 in-loop deblocking filter — the
post-reconstruction stage that smooths 8×8/16×16 transform-block edges,
4×4 internal edges, and prediction-block boundaries. It is **shared
between encoder and decoder** because libvpx's loop filter sits in
`vp9/common/` (`third_party/libvpx/vp9/common/vp9_loopfilter.c:1-1633`);
both pipelines pass reconstructed pixels through the same kernels using
the same per-frame parameter tables.

The module is responsible for:

1. The per-frame **parameter setup** (`loop_filter_info_n`): build the
   `[seg_id][ref_frame][mode]` filter-level cube from a base
   `filter_level`, mode/ref deltas, sharpness, and per-segment LF deltas
   (`vp9_loopfilter.c:212-295`).
2. The per-superblock **mask machinery** (`LOOP_FILTER_MASK`): for each
   64×64 region, compute `left_y[TX_SIZES]`, `above_y[TX_SIZES]`,
   `int_4x4_y`, the equivalent UV masks, and a flat `lfl_y[64]` byte array
   recording the per-8×8 `filter_level` (`vp9_loopfilter.c:39-203,
   555-1010`).
3. The **encoder-side per-frame level pick**, `vp9_pick_filter_level`,
   which is the load-bearing call from `loopfilter_frame` in the encoder
   shell. For our cpu-used 6-8 target it is **always** the
   `LPF_PICK_FROM_Q` linear-fit path; the RDO bypass is the entire story
   (see §5.4). Source: `vp9/encoder/vp9_picklpf.c:159-203`.
4. The **per-plane filter pass** — vertical edges first then horizontal,
   with `_ss00`/`_ss11`/`_non420` variants — that invokes the
   `vpx_lpf_{vertical,horizontal}_{4,8,16}[_dual]` SIMD kernels
   (`vp9_loopfilter.c:1078-1467`).
5. The **shared row-sync primitive** for the multi-tile / multi-thread
   filter pass (`VP9LfSync` in `vp9_thread_common.h:25-49`,
   `vp9_thread_common.c:38-217`). On the encoder side the LF pass is a
   post-encode stage decoupled from the encode-MT pool but reusing the
   same workers (HLD-threading §4 / §5 `lf_row_sync` row).

This LLD ports the 8-bit (`Pixel = u8`) deblocker only. The
`vpx_highbd_lpf_*` siblings are deferred to Stage 6+ per D-01
(`notes/divergence-ledger.md:27-47`). The `LF_PATH_SLOW` (non-420)
fall-through is kept in the design for decoder profile-1 parsing
support; the encoder side is 4:2:0 only, so this path is decoder-only
in v1.

## 2. libvpx origin

| Concern | libvpx file:line | Stage 1 anchor |
|---|---|---|
| Public LF API (encoder-shell entry) | `vp9_encoder.c:3392-3452` (`loopfilter_frame`) | `01-architecture.md:1875-1877`, step 47 |
| Per-frame level pick | `vp9_picklpf.c:159-203` (`vp9_pick_filter_level`); `:46-76` (`try_filter_frame`); `:78-157` (`search_filter_level`) | (Stage 1 gap — see §8) |
| Speed-feature `lpf_pick` | `vp9_speed_features.h:102-111`, `:493`; `vp9_speed_features.c:555` (speed≥3 → `LPF_PICK_FROM_Q`); `:988` (default `LPF_PICK_FROM_FULL_IMAGE`) | A6 (speed-feature dispatch) |
| `LOOPFILTER_CONTROL` per-frame on/off | `vp9_encoder.h:147-151`; `vp9_encoder.c:3408-3413` | `01-architecture.md:961`, `:668` |
| `struct loopfilter` (LF state in `VP9_COMMON`) | `vp9_loopfilter.h:70-90` | `01-architecture.md:1276` |
| `loop_filter_info_n` / `loop_filter_thresh` (parameter tables) | `vp9_loopfilter.h:40-49` | (Stage 1 gap) |
| `LOOP_FILTER_MASK` per-SB mask record | `vp9_loopfilter.h:60-68` | (Stage 1 gap) |
| Sharpness derivation (`mblim`/`lim`/`hev_thr`) | `vp9_loopfilter.c:212-231` (`update_sharpness`) + `:238-250` (`vp9_loop_filter_init`) | (Stage 1 gap) |
| Per-frame level-cube fill | `vp9_loopfilter.c:252-295` (`vp9_loop_filter_frame_init`) | — |
| 64×64 mask builder | `vp9_loopfilter.c:555-724` (`build_masks` + helpers); `:766-878` (`vp9_adjust_mask` including frame-edge handling); `:880-1010` (`vp9_setup_mask`); `:1488-1512` (`vp9_build_mask_frame`); `:1514-1606` (incremental `vp9_build_mask`) | — |
| Vertical-then-horizontal pass (Y, 4:2:0) | `vp9_loopfilter.c:297-372` (`filter_selectively_vert_row2`); `:453-543` (`filter_selectively_horiz`); `:1012-1040` (`filter_selectively_vert`); `:1239-1322` (`vp9_filter_block_plane_ss00`); `:1324-1420` (`vp9_filter_block_plane_ss11`); `:1078-1237` (`vp9_filter_block_plane_non420`) | — |
| Frame iteration | `vp9_loopfilter.c:1422-1483` (`loop_filter_rows`, `vp9_loop_filter_frame`) | — |
| Row-sync data | `vp9_thread_common.h:25-49` (`VP9LfSync`); `vp9_thread_common.c:38-87` (`sync_read`/`sync_write`); `:89-217` (`thread_loop_filter_rows`, `loop_filter_rows_mt`); `:219-262` (`vp9_loop_filter_frame_mt`, `vp9_lpf_mt_init`); `:264-339` (`vp9_loop_filter_alloc`, `get_sync_range`); `:342-387` (`vp9_loop_filter_dealloc`) | A6 §2.7 (`VP9LfSync`); `01-architecture.md:2303-2307` |
| Scalar reference kernels | `vpx_dsp/loopfilter.c:1-743` (`vpx_lpf_horizontal_4_c`, `..._8_c`, `..._16_c`, `..._vertical_4_c`, etc.) | A4 (passing) |
| RTCD signatures (12 entries) | `vpx_dsp/vpx_dsp_rtcd_defs.pl:449-483` | HLD-simd `02-simd-dispatch.md:1083-1129` |

Stage 1 coverage: thin. `01-architecture.md` mentions the call site
(step 47 / `:1875-1877`) and threading hand-off (`:2303-2307`) but no
specialist deep-dive transcribed the parameter tables, mask layouts,
sharpness math, edge-handling rules in `vp9_adjust_mask`, or the
`search_filter_level` golden-search algorithm. This LLD fills those gaps
directly from libvpx source. See §8 for the Stage 1.5 re-dispatch
verdict.

## 3. Public API

All items live in `rvp9_core::loopfilter`. The module is `pub(crate)`
to `rvp9-core` consumers and `pub` for `rvp9-encoder` and
`rvp9-decoder` via re-export. Encoder/decoder integrate via:

- The encoder shell (`rvp9-encoder::frame_loop::loopfilter_frame`)
  imports `vp9_pick_filter_level`, `build_mask_frame`, and
  `loop_filter_frame{,_mt}`.
- The decoder (`rvp9-decoder`) imports `loop_filter_frame{,_mt}` only;
  the decoder uses the bitstream-parsed `filter_level` directly without
  picking.

```rust
//
// 3.1 Per-frame level pick (encoder only).
//

/// Speed-feature selector for the loop-filter level search.
/// Mirrors libvpx `LPF_PICK_METHOD` (`vp9_speed_features.h:102-111`).
///
/// At cpu-used 6-8 (our target band) the speed-feature table pins this
/// to `FromQ` (`vp9_speed_features.c:555`). The encoder shell still
/// dispatches on the variant for parity with libvpx and for the future
/// where a quality preset re-enables RDO search.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LpfPickMethod {
    /// Linear regression from base qindex (the only RT path).
    FromQ,
    /// Full-image RDO golden-section search. Deferred (§7 L-LPF-1).
    FromFullImage,
    /// Sub-image RDO search (`partial_frame=1`). Deferred.
    FromSubImage,
    /// Force `filter_level = 0` if previously enabled. Used by speed≥9.
    MinimalLpf,
}

/// Computed the per-frame `filter_level` (0..=MAX_LOOP_FILTER=63) and
/// writes it into `lf.filter_level`. Also resets `lf.sharpness_level = 0`
/// (libvpx forces this; `vp9_picklpf.c:164`).
///
/// Reads:
/// - `lf.last_filt_level` (the previous-frame level, used by golden search)
/// - `common.base_qindex`, `common.frame_type`, `common.seg.enabled`
/// - `oxcf.pass`, `oxcf.rc_mode`, `oxcf.aq_mode`, `oxcf.content`
///
/// For `LpfPickMethod::FromQ` (the RT path), no source frame is read;
/// the `src` parameter is unused (mirrors libvpx, where
/// `LPF_PICK_FROM_Q` ignores `sd`).
///
/// Panics: never; clamping is total.
/// Unsafety: none.
pub fn pick_filter_level(
    src: Option<&YuvFrame>,
    lf: &mut LoopFilterState,
    common: &CommonState,
    method: LpfPickMethod,
);

//
// 3.2 Per-frame parameter rebuild (level cube).
//

/// Build `loop_filter_info_n` (the parameter tables) from
/// `default_filt_lvl`. Walks the 8 segments × 4 ref frames × 2 modes
/// cube and clamps the per-cell level via mode/ref deltas
/// (`vp9_loopfilter.c:252-295`). Also re-runs `update_sharpness` if
/// `sharpness_level` changed (`vp9_loopfilter.c:262-266`).
///
/// Called by:
/// - Encoder: from `build_mask_frame` (`vp9_loopfilter.c:1502`).
/// - Decoder: from its own frame-header parse step.
///
/// `default_filt_lvl` must be in `0..=MAX_LOOP_FILTER`.
pub fn loop_filter_frame_init(
    lf_info: &mut LoopFilterInfoN,
    lf: &LoopFilterState,
    seg: &Segmentation,
    default_filt_lvl: u8,
);

/// Initialise the static portion of `lf_info` (sharpness + HEV tables).
/// Idempotent. Called once at `Encoder::new` and on each sharpness
/// change. Mirrors `vp9_loop_filter_init` (`vp9_loopfilter.c:238-250`).
pub fn loop_filter_init(lf_info: &mut LoopFilterInfoN, sharpness_level: u8);

//
// 3.3 Mask machinery (encoder pre-pass).
//

/// Walk the frame's MI grid and populate every superblock's
/// `LOOP_FILTER_MASK` (`vp9_loopfilter.c:1488-1512`).
///
/// In libvpx this is encoder-only; the decoder builds masks
/// **incrementally as it decodes blocks** via `vp9_build_mask`
/// (`vp9_loopfilter.c:1526-1606`). Our decoder will mirror that
/// decoder pattern (see LLD-decoder for placement).
///
/// `partial_frame=true` filters only the centre strip (used by
/// `LpfPickMethod::FromSubImage`; **dead in RT**, kept for API parity).
///
/// Pre: `lf.lfm` is allocated and `lf.lfm_stride` is set.
/// Pre: `lf.filter_level > 0` (callers gate; if 0, this is a no-op).
/// Pre: `common.mi_grid_visible` is fully populated for the frame.
pub fn build_mask_frame(
    common: &mut CommonState,
    lf: &mut LoopFilterState,
    lf_info: &LoopFilterInfoN,
    frame_filter_level: u8,
    partial_frame: bool,
);

/// Compute one superblock mask. Used incrementally by the decoder per
/// 8×8/16×16/…/64×64 block (`vp9_build_mask`; `vp9_loopfilter.c:1526`).
pub fn build_mask_sb(
    common: &mut CommonState,
    lf_info: &LoopFilterInfoN,
    mi: &ModeInfo,
    mi_row: u32, mi_col: u32, bw: u32, bh: u32,
);

/// Final adjustment pass on a 64×64 mask: merge TX_32X32 into TX_16X16,
/// promote 4×4 border edges to 8×8, mask off frame edges, clear left
/// edge on column 0 (`vp9_loopfilter.c:766-878`).
pub fn adjust_mask(
    common: &CommonState,
    mi_row: u32, mi_col: u32,
    lfm: &mut LoopFilterMask,
);

/// Reset every superblock mask to zero (`vp9_loopfilter.c:1619-1625`).
/// Called between frames when `lf.filter_level > 0`; cheap memset.
pub fn reset_lfm(lf: &mut LoopFilterState, mi_rows: u32);

//
// 3.4 Single-threaded filter (decoder; encoder fallback when num_workers==1).
//

/// Apply the deblocker to `frame`. Vertical-then-horizontal pass per
/// 64×64 SB row, all planes (or Y only if `y_only=true`).
///
/// `y_only=true` is used by `try_filter_frame` during full-image RDO
/// search (`vp9_picklpf.c:54-60`) — **dead in RT** but exposed.
///
/// `partial_frame=true` selects the centre-strip subset
/// (`vp9_loopfilter.c:1476-1480`) — also dead in RT.
///
/// Pre: `frame_filter_level > 0` (otherwise no-op).
/// Pre: every superblock in the visible MI grid has a mask in `lf.lfm`
/// (caller invoked `build_mask_frame` or built incrementally per-SB).
///
/// # Safety
/// Internally calls into the SIMD dispatch table; the dispatch
/// functions are `unsafe extern "C" fn` from libvpx semantics. The
/// public function is safe because it owns its writes (no shared
/// pixel buffers, `frame` is `&mut`).
pub fn loop_filter_frame(
    frame: &mut YuvFrame,
    common: &CommonState,
    lf: &LoopFilterState,
    lf_info: &LoopFilterInfoN,
    planes: &[PlaneDesc; MAX_MB_PLANE],
    frame_filter_level: u8,
    y_only: bool,
    partial_frame: bool,
);

//
// 3.5 Multi-threaded filter (encoder when num_workers > 1, decoder always).
//

/// Multi-threaded post-encode loop filter. Dispatches the persistent
/// worker pool from HLD-threading §3 over SB rows (`vp9_thread_common.c:
/// 157-217`). Worker count is clamped to
/// `min(num_workers, min(num_tile_cols, sb_rows))` (`:170`).
///
/// Synchronisation: row-pipeline. Each worker's SB at `(r, c)` must wait
/// until the worker on row `r-1` has processed at least
/// `c + sync_range` SBs (`vp9_thread_common.c:38-56`). `sync_range`
/// scales with width: 1 (<640) / 2 (≤1280) / 4 (≤4096) / 8 (>4096)
/// (`:265-276`).
///
/// # Safety
/// Workers write disjoint pixel rows (workers fan out by `i *
/// MI_BLOCK_SIZE` rows then stride by `num_active_workers *
/// MI_BLOCK_SIZE`; `vp9_thread_common.c:110-111, 201`). Aliasing is
/// row-disjoint within the SB-row pipeline; the `RowSync`-style
/// happens-before edge guarantees no two workers touch overlapping
/// strict-edge pixels concurrently. See §4 `LfSync` and HLD-threading
/// §5 row `lf_row_sync` (strategy C).
pub fn loop_filter_frame_mt(
    frame: &mut YuvFrame,
    common: &CommonState,
    lf: &LoopFilterState,
    lf_info: &LoopFilterInfoN,
    planes: &[PlaneDesc; MAX_MB_PLANE],
    frame_filter_level: u8,
    y_only: bool,
    partial_frame: bool,
    workers: &WorkerPool,
    lf_sync: &mut LfSync,
);

/// Allocate `lf_sync` if shape changed (rows, num_workers). Idempotent
/// fast-path returns without work when shape matches.
/// Mirrors `vp9_loop_filter_alloc` (`vp9_thread_common.c:279-339`).
pub fn loop_filter_alloc(
    lf_sync: &mut LfSync,
    sb_rows: u32, width: u32, num_workers: u32,
);

/// Drop synchronisation primitives + LFWorkerData. Safe to call on a
/// zero-initialised `LfSync`. (`vp9_thread_common.c:342-387`.)
pub fn loop_filter_dealloc(lf_sync: &mut LfSync);
```

The signatures intentionally avoid `Option<bool>` for `y_only` /
`partial_frame` — both have exactly two meaningful values and the
libvpx call sites pass `1`/`0` literals.

## 4. Types

### 4.1 Compile-time constants

```rust
/// `MAX_LOOP_FILTER` — `vp9_loopfilter.h:24`.
pub const MAX_LOOP_FILTER: u8 = 63;

/// `MAX_SHARPNESS` — `vp9_loopfilter.h:25`. Wire-format bound; `oxcf.sharpness`
/// is clamped to this in `vp9_cx_iface.c:532-540`.
pub const MAX_SHARPNESS: u8 = 7;

/// `SIMD_WIDTH` — `vp9_loopfilter.h:27`. Width of the broadcast
/// blimit/limit/thresh byte arrays. The SIMD kernels expect 16-byte
/// aligned, 16-byte-wide repeated values.
pub const SIMD_WIDTH: usize = 16;

/// libvpx mode-delta indices (`vp9_loopfilter.h:30`).
pub const MAX_MODE_LF_DELTAS: usize = 2;   // 0 = ZEROMV, 1 = MV (non-zero)
pub const MAX_REF_LF_DELTAS: usize = 4;    // 0 = INTRA, 1 = LAST, 2 = GOLDEN, 3 = ALTREF

/// MI-block size: 8 8×8 MIs across a 64×64 superblock.
pub const MI_BLOCK_SIZE: u32 = 8;
pub const MI_BLOCK_SIZE_LOG2: u32 = 3;
```

### 4.2 Per-level threshold record (`loop_filter_thresh`)

`vp9_loopfilter.h:40-44`. Three 16-byte-wide replicated values for the
SIMD lanes — `mblim` (edge threshold), `lim` (interior threshold), and
`hev_thr` (high-edge-variance threshold).

```rust
/// Aligned to 16 bytes so the whole struct loads into one XMM/NEON
/// register pair per field.
#[repr(C, align(16))]
#[derive(Copy, Clone)]
pub struct LoopFilterThresh {
    pub mblim:   [u8; SIMD_WIDTH],
    pub lim:     [u8; SIMD_WIDTH],
    pub hev_thr: [u8; SIMD_WIDTH],
}
```

`Copy` is safe (it's POD); `Send + Sync` are auto-derived (no interior
mutability).

### 4.3 Frame-wide parameter cube (`loop_filter_info_n`)

`vp9_loopfilter.h:46-49`. One threshold record per (0..=63) filter
level, plus a 3D `[seg][ref][mode]` byte cube of filter levels for the
get-level fast path (`vp9_loopfilter.c:233-236`).

```rust
pub const MAX_SEGMENTS: usize = 8;
pub const MAX_REF_FRAMES: usize = 4;  // INTRA, LAST, GOLDEN, ALTREF

pub struct LoopFilterInfoN {
    /// 64 thresh records (levels 0..=MAX_LOOP_FILTER).
    pub lfthr: [LoopFilterThresh; (MAX_LOOP_FILTER as usize) + 1],
    /// Per-(segment, ref, mode) clamped filter level.
    /// `mode` is the 2-bit reduction `mode_lf_lut[mi->mode]`
    /// (`vp9_loopfilter.c:207-210`).
    pub lvl: [[[u8; MAX_MODE_LF_DELTAS]; MAX_REF_FRAMES]; MAX_SEGMENTS],
}
```

### 4.4 Per-superblock mask record (`LOOP_FILTER_MASK`)

`vp9_loopfilter.h:60-68`. One per 64×64 SB. Stored in a flat
`Box<[LoopFilterMask]>` owned by `LoopFilterState.lfm`.

```rust
/// Per-64×64-SB filter-edge bitmask + per-8×8 level map.
///
/// Each Y mask is a 64-bit word indexing an 8×8 grid of 8×8 subblocks:
/// bit `(row*8 + col)` is set when an edge runs along the
/// left/above of that subblock. Each UV mask is a 16-bit word indexing
/// a 4×4 grid (subsampled 4:2:0).
///
/// Indexed by `TX_SIZES` so the mask machinery can intersect the
/// "filter would apply at this TX size" set against the "the block
/// chose this TX" set (`vp9_loopfilter.c:39-203`).
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct LoopFilterMask {
    /// Edges along the left border of each 8×8 (Y), per TX size.
    pub left_y:    [u64; TX_SIZES],
    /// Edges along the above border of each 8×8 (Y), per TX size.
    pub above_y:   [u64; TX_SIZES],
    /// Internal 4×4 edges within each 8×8 (Y).
    pub int_4x4_y: u64,
    /// UV equivalents — 4×4 mask of subsampled 8×8s.
    pub left_uv:    [u16; TX_SIZES],
    pub above_uv:   [u16; TX_SIZES],
    pub int_4x4_uv: u16,
    /// Per-8×8 filter level (the `get_filter_level()` byte). Row-major,
    /// 8 cols × 8 rows. (`vp9_loopfilter.c:67`.)
    pub lfl_y: [u8; 64],
}
pub const TX_SIZES: usize = 4;  // TX_4X4, TX_8X8, TX_16X16, TX_32X32
```

### 4.5 Frame-level LF state (`struct loopfilter`)

`vp9_loopfilter.h:70-90`. Owned by `CommonState` (the rvp9 equivalent
of `VP9_COMMON`). Persists across frames; some fields are reset every
frame.

```rust
pub struct LoopFilterState {
    /// Current frame's level, 0..=MAX_LOOP_FILTER. 0 disables LF entirely.
    pub filter_level: u8,
    /// Previous frame's level — search seed for next frame's RDO pick.
    pub last_filt_level: u8,

    /// 0..=MAX_SHARPNESS. Wire-format value mapped from `oxcf.sharpness`.
    pub sharpness_level: u8,
    /// Set after `loop_filter_init`; trigger to re-run `update_sharpness`
    /// when changed.
    pub last_sharpness_level: u8,

    /// Bitstream bit — true when ref/mode deltas are sent.
    pub mode_ref_delta_enabled: bool,
    /// Bitstream bit — true if the encoder wrote new deltas this frame.
    /// Cleared post-pack (`vp9_encoder.c:5884`).
    pub mode_ref_delta_update: bool,

    /// Signed deltas applied per ref frame, scaled by
    /// `1 << (default_filt_lvl >> 5)` in `loop_filter_frame_init`
    /// (`vp9_loopfilter.c:257`). Range [-MAX_LOOP_FILTER, MAX_LOOP_FILTER].
    pub ref_deltas:  [i8; MAX_REF_LF_DELTAS],
    pub last_ref_deltas:  [i8; MAX_REF_LF_DELTAS],

    pub mode_deltas: [i8; MAX_MODE_LF_DELTAS],
    pub last_mode_deltas: [i8; MAX_MODE_LF_DELTAS],

    /// Per-SB mask records. Length = `lfm_stride * sb_rows`. Owned by
    /// `CommonState` for memory locality; rebuilt every frame.
    pub lfm: Box<[LoopFilterMask]>,
    /// Stride in SBs (= `mi_cols_aligned_to_sb(mi_cols) >> 3`).
    pub lfm_stride: u32,
}
```

`lfm` ownership note: in libvpx the array lives off
`cm->lf.lfm` (raw pointer, allocated by `vp9_alloc_loop_filter` —
implicit, see the call from `vp9_alloc_compressor_data`). In rvp9 the
allocation is owned by `LoopFilterState` itself; `CommonState` holds the
`LoopFilterState`. This keeps the encoder shell's `Drop` ordering
trivial and is consistent with the `Box<[…]>` policy in
`00-architecture.md:160-180`.

### 4.6 Encoder-side per-frame control (`LOOPFILTER_CONTROL`)

`vp9_encoder.h:147-151`. Lives on the encoder side (`EncoderConfig`),
not in `CommonState`.

```rust
/// Per-frame loop-filter on/off control. Used by the encoder to skip
/// LF on non-reference frames (low-latency drop frames).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LoopfilterControl {
    All = 0,        // Always run LF (default).
    Reference = 1,  // Skip LF on non-reference frames.
    None = 2,       // Always skip LF.
}
```

### 4.7 Multi-thread row-sync (`VP9LfSync`)

`vp9_thread_common.h:25-49`. Reused by **encoder and decoder**. The
encoder owns a `LfSync` instance in its `WorkerPool` neighbours
(HLD-threading §5 row `lf_row_sync`). The decoder owns its own.

```rust
use std::sync::{Mutex, Condvar};

/// Per-row sync record. The Mutex protects `cur_sb_col[r]` (and
/// indirectly, the corruption flag in the decoder).
struct LfRowSync {
    state: Mutex<LfRowState>,
    cond: Condvar,
}

struct LfRowState {
    /// Highest SB column the row's worker has finished filtering.
    /// Initialised to -1; the writer increments by `sync_range` chunks.
    cur_sb_col: i32,
}

/// Worker-local LF job data. `vp9_loopfilter.h:140-148`.
pub struct LfWorkerData {
    /// We use *mut here only inside an `UnsafeCell` shell at the
    /// worker dispatch boundary; see `LfSync` SAFETY notes below.
    pub start_mi_row: u32,
    pub stop_mi_row:  u32,
    pub y_only: bool,
    // Pointers to frame + common state are stored as raw pointers
    // inside the worker hook closure; the lifetime is bounded by
    // `loop_filter_frame_mt`'s caller-held `&mut YuvFrame`.
}

/// Row-pipeline synchroniser. Mirrors `VP9LfSync` minus the
/// decoder-only `recon_done_*` fields (which the encoder doesn't need).
///
/// Field ownership (HLD-threading §5 row `lf_row_sync`):
/// - Allocated once on first use; reallocated when `(sb_rows, width,
///   num_workers)` triple changes (`vp9_thread_common.c:173-177`).
/// - Main thread initialises before dispatching workers.
/// - Workers read/write `rows[]` under their per-row `Mutex` /
///   `Condvar`.
pub struct LfSync {
    /// One per SB row.
    pub rows: Box<[LfRowSync]>,
    /// One per worker (start/stop range).
    pub lfdata: Box<[LfWorkerData]>,
    /// nsync — column-look-ahead size before signalling next row.
    /// Per `get_sync_range` (`vp9_thread_common.c:265-276`):
    ///   width<640 -> 1; ≤1280 -> 2; ≤4096 -> 4; else 8.
    pub sync_range: u32,
    /// Number of allocated rows (matches `rows.len()`).
    pub sb_rows: u32,
    /// Number of allocated workers (matches `lfdata.len()`).
    pub num_workers: u32,
    /// Workers scheduled this frame ≤ num_workers.
    pub num_active_workers: u32,
    // Decoder-only fields (`num_tiles_done`, `recon_done_mutex`,
    // `recon_done_cond`, `corrupted`) live in a separate
    // `LfSyncDecoderExtra` struct in `rvp9-decoder` — not in
    // `rvp9-core::loopfilter`. Encoder pass is post-encode so it
    // has no recon-done dependency.
}

// SAFETY: All shared state across workers is behind Mutex/Condvar in
// `LfRowSync`; pixel writes are row-disjoint per worker (see §5.6).
unsafe impl Send for LfSync {}
unsafe impl Sync for LfSync {}
```

### 4.8 Plane descriptor (helper)

For §3 signatures. Mirrors libvpx `struct macroblockd_plane` but only
the fields the LF reads.

```rust
pub struct PlaneDesc {
    /// Subsampling on x (0 for Y, 1 for U/V at 4:2:0).
    pub subsampling_x: u8,
    /// Subsampling on y (0 for Y, 1 for U/V at 4:2:0).
    pub subsampling_y: u8,
    /// Stride (set per-frame by `vp9_setup_dst_planes`).
    pub stride: i32,
}
```

The `buf: *mut u8` pointer is **not** stored on `PlaneDesc` — at the
call site we derive it from `&mut YuvFrame` for the current
`(mi_row, mi_col)` window using a `vp9_setup_dst_planes`-equivalent
helper. This keeps `&mut`-aliasing checkable.

## 5. Internal algorithm

### 5.1 `loop_filter_init` — sharpness → thresh table

Mirrors `vp9_loopfilter.c:212-249`.

```text
fn loop_filter_init(lf_info, sharpness_level):
    for lvl in 0..=MAX_LOOP_FILTER:
        # block_inside_limit shrinks with sharpness:
        # sharpness 0          -> shift 0  (block_inside_limit = lvl)
        # sharpness 1..=4      -> shift 1  (block_inside_limit = lvl >> 1)
        # sharpness 5..=7      -> shift 2  (block_inside_limit = lvl >> 2)
        block_inside_limit = lvl >> ((sharpness > 0) + (sharpness > 4))

        # Hard cap at (9 - sharpness) when sharpness > 0.
        if sharpness > 0 and block_inside_limit > (9 - sharpness):
            block_inside_limit = 9 - sharpness

        # Floor at 1.
        block_inside_limit = max(block_inside_limit, 1)

        lfthr[lvl].lim   = [block_inside_limit; 16]
        lfthr[lvl].mblim = [2*(lvl+2) + block_inside_limit; 16]
        # HEV threshold = lvl >> 4 (so >=16 -> 1, >=32 -> 2, >=48 -> 3).
        lfthr[lvl].hev_thr = [lvl >> 4; 16]
```

Citation: `vp9_loopfilter.c:212-231` (`update_sharpness`), `:238-250`
(`vp9_loop_filter_init` hev-thr loop).

For real-time, libvpx **forces** `sharpness_level = 0`
(`vp9_picklpf.c:164`). Our `pick_filter_level` does the same. The
sharpness table is therefore typically static across a session, but
we still run the update on `last_sharpness_level != sharpness_level`
to support runtime sharpness change via `oxcf.sharpness` reconfigure.

### 5.2 `loop_filter_frame_init` — segment-aware level cube

Mirrors `vp9_loopfilter.c:252-295`.

```text
fn loop_filter_frame_init(lf_info, lf, seg, default_filt_lvl):
    # 1-bit shift when level ≥ 32 (so deltas have ~equal weight at
    # high and low levels).
    let scale = 1 << (default_filt_lvl >> 5);

    if last_sharpness_level != sharpness_level:
        update_sharpness(lf_info, sharpness_level)
        last_sharpness_level = sharpness_level

    for seg_id in 0..MAX_SEGMENTS:
        let mut lvl_seg = default_filt_lvl
        if segfeature_active(seg, seg_id, SEG_LVL_ALT_LF):
            let data = get_segdata(seg, seg_id, SEG_LVL_ALT_LF)
            lvl_seg = if seg.abs_delta == ABS:
                          clamp(data, 0, MAX_LOOP_FILTER)
                      else:
                          clamp(default_filt_lvl + data, 0, MAX_LOOP_FILTER)

        if !mode_ref_delta_enabled:
            # All cells share `lvl_seg`.
            lvl[seg_id] = [[lvl_seg; 2]; MAX_REF_FRAMES]
        else:
            # Intra (ref=0): mode index forced to 0.
            lvl[seg_id][INTRA][0] =
                clamp(lvl_seg + ref_deltas[INTRA] * scale, 0, MAX_LOOP_FILTER)
            for ref in LAST..ALTREF:
                for mode in 0..MAX_MODE_LF_DELTAS:
                    lvl[seg_id][ref][mode] =
                        clamp(lvl_seg
                              + ref_deltas[ref] * scale
                              + mode_deltas[mode] * scale,
                              0, MAX_LOOP_FILTER)
```

The encoder leaves `mode_ref_delta_enabled = true` in steady-state
(libvpx comment `vp9_loopfilter.c:277-280`: "encoder always uses deltas").
Our encoder mirrors this — see §7 ledger entry. The flat-fill path
exists so the decoder can still parse streams that turn deltas off.

### 5.3 The mask machinery — `vp9_setup_mask` family

The mask-build pass populates one `LoopFilterMask` per 64×64 SB. The
fundamental invariant is that for each transform-size class `tx ∈
{TX_4X4, TX_8X8, TX_16X16, TX_32X32}`:

- `left_y[tx]` bit `(r*8 + c)` set ⟺ a `tx`-size filter must run along
  the left edge of the 8×8 at row `r`, column `c` of this SB.
- `above_y[tx]` similarly for the top edge.
- `int_4x4_y` is the internal-4×4 mask (one bit per 8×8 that contains
  TX_4×4 transform internal edges).

The per-block walker `build_masks` (static, `vp9_loopfilter.c:649-724`)
performs:

1. Compute `filter_level = get_filter_level(lfi_n, mi)` (the
   `[seg][ref[0]][mode_lf_lut[mode]]` lookup,
   `vp9_loopfilter.c:233-236`). If 0, **return early** — block
   contributes no edges.
2. Memset `lfl_y[index..index+w]` for each of the block's 8×8 rows to
   `filter_level`. This is the per-pixel-edge level look-up consumed
   by `filter_selectively_*`.
3. If non-first-block-in-16×16 (only relevant for sub-16 blocks), skip
   UV mask updates. The decoder/encoder marks UV edges only at the
   first 8×8 of a 16×16, controlled by `first_block_in_16x16[8][8]`
   (`vp9_loopfilter.c:1516-1521`).
4. `above_y |= above_prediction_mask[block_size] << shift_y`
   (`:692-693`). Pulls in the block's outer edges.
5. `left_y |= left_prediction_mask[block_size] << shift_y` (`:694-695`).
6. UV equivalents using `above_prediction_mask_uv` / `left_prediction_mask_uv`.
7. **Inter-block skip** (`mi.skip && is_inter_block(mi)`): early return,
   skipping the *transform-size* mask additions (intra blocks always
   contribute TX-size edges; inter blocks with no coded coeffs do not).
8. `above_y |= (size_mask[bs] & above_64x64_txform_mask[tx_y]) << shift_y`
   (`:705-707`) — the TX-size internal edges within the prediction
   block.
9. Symmetric for `left_y`, UV.
10. If `tx_y == TX_4X4`: `int_4x4_y |= size_mask[bs] << shift_y`. Same
    for `int_4x4_uv` on UV.

Citation for the constant tables — verbatim transcription:

```text
left_64x64_txform_mask[TX_SIZES]   = {0xFFFFFFFFFFFFFFFF,
                                      0xFFFFFFFFFFFFFFFF,
                                      0x5555555555555555,
                                      0x1111111111111111};
above_64x64_txform_mask[TX_SIZES]  = {0xFFFFFFFFFFFFFFFF,
                                      0xFFFFFFFFFFFFFFFF,
                                      0x00FF00FF00FF00FF,
                                      0x000000FF000000FF};
left_prediction_mask[BLOCK_SIZES]  = {0x1, 0x1, 0x1, 0x1,
                                      0x101, 0x1, 0x101,
                                      0x1010101, 0x101,
                                      0x1010101, 0x101010101010101,
                                      0x1010101, 0x101010101010101};
above_prediction_mask[BLOCK_SIZES] = {0x1, 0x1, 0x1, 0x1, 0x1,
                                      0x3, 0x3, 0x3, 0xF, 0xF, 0xF,
                                      0xFF, 0xFF};
size_mask[BLOCK_SIZES]             = {0x1, 0x1, 0x1, 0x1, 0x101,
                                      0x3, 0x303, 0x3030303,
                                      0xF0F, 0xF0F0F0F,
                                      0xF0F0F0F0F0F0F0F,
                                      0xFFFFFFFF,
                                      0xFFFFFFFFFFFFFFFF};
left_border  = 0x1111111111111111
above_border = 0x000000FF000000FF
```

(`vp9_loopfilter.c:39-138`.) These appear as `pub(crate) const`
arrays in `loopfilter/tables.rs` in our port — never modified, always
the same byte sequence as libvpx. A unit test asserts each constant
matches.

UV equivalents: `left_64x64_txform_mask_uv = [0xFFFF, 0xFFFF, 0x5555,
0x1111]`, `above_64x64_txform_mask_uv = [0xFFFF, 0xFFFF, 0x0F0F,
0x000F]`, plus the analogous prediction / size masks
(`vp9_loopfilter.c:141-203`).

The 4-quadrant walker (`vp9_setup_mask`, `:880-1010`) walks each 64×64
in nested 32/16/8 levels, switching on `mip[0]->sb_type` and using the
shift tables:

```text
offset_32 = [4, (stride<<2) - 4, 4, -(stride<<2) - 4]
offset_16 = [2, (stride<<1) - 2, 2, -(stride<<1) - 2]
offset    = [1, stride - 1,      1, -stride - 1]
shift_32_y  = [0, 4, 32, 36]
shift_16_y  = [0, 2, 16, 18]
shift_8_y   = [0, 1, 8, 9]
shift_32_uv = [0, 2, 8, 10]
shift_16_uv = [0, 1, 4, 5]
```

These walk the Morton-order layout of the four quadrants of each
32×32/16×16/8×8 step, accumulating bits into the SB-level mask. (Same
structure as the variance-tree partition walker in LLD-partition.)

### 5.4 `vp9_adjust_mask` — frame-edge and merging

`vp9_loopfilter.c:766-878`. Three responsibilities:

1. **Merge 32×32 mask into 16×16** (no 32-tap filter exists; the kernel
   palette is 4/8/16-tap):
   ```text
   left_y[TX_16X16]   |= left_y[TX_32X32];
   above_y[TX_16X16]  |= above_y[TX_32X32];
   left_uv[TX_16X16]  |= left_uv[TX_32X32];
   above_uv[TX_16X16] |= above_uv[TX_32X32];
   ```
2. **Promote border-4×4 edges to 8×8** so SB boundaries always get the
   wider filter:
   ```text
   left_y[TX_8X8] |= left_y[TX_4X4] & left_border;     // 0x1111…
   left_y[TX_4X4] &= !left_border;
   above_y[TX_8X8] |= above_y[TX_4X4] & above_border;  // 0x000000FF000000FF
   above_y[TX_4X4] &= !above_border;
   ```
3. **Frame-edge masking** — mask off rows past `cm->mi_rows`, columns
   past `cm->mi_cols`, and the very-first column (the implicit-frame
   border itself is never filtered). When the last UV row is exactly 1
   or 5 8×8s tall (`rows == 1` / `rows == 5`), downgrade the 16×16
   filter to 8×8 (because the wider filter would read pixels past the
   frame; same for columns at 1 or 5). The exact byte mask for the
   16×16 column case is `0xCCCC` (top-half-of-each-nibble),
   `vp9_loopfilter.c:847-849`.

The asserts at the end (`:862-877`) verify that no two TX-size masks
overlap — once a bit is set at, e.g., `left_y[TX_16X16]`, it must
**not** be set at `left_y[TX_8X8]` or `left_y[TX_4X4]`. In rvp9 these
become `debug_assert!` (cheap in release; checked in our differential
tests).

### 5.5 The per-plane filter pass

`vp9_filter_block_plane_ss00` (`vp9_loopfilter.c:1239-1322`) is the
4:4:4-or-Y path. For each SB row of an SB column, it:

1. **Vertical pass first**, two MI-rows at a time
   (`filter_selectively_vert_row2` taking the *left* masks):
   ```text
   for r in (0, 2, 4, 6):
       call filter_selectively_vert_row2(plane.subsampling_x, dst,
           stride,
           low16(left_y[TX_16X16]), low16(left_y[TX_8X8]),
           low16(left_y[TX_4X4]),    low16(int_4x4_y),
           &lf_info.lfthr, &lfm.lfl_y[r*8..])
       dst += 16 * stride
       left_y[*] >>= 16; int_4x4_y >>= 16
   ```
2. **Horizontal pass second**, one MI-row at a time
   (`filter_selectively_horiz` taking the *above* masks):
   ```text
   for r in 0..8:
       if r == 0 and mi_row == 0:
           zero the 16/8/4 masks (no above-frame edge)
       else:
           take low byte of above_y[16],[8],[4] and int_4x4_y
       call filter_selectively_horiz(dst, stride, …)
       dst += 8 * stride
       above_y[*] >>= 8; int_4x4_y >>= 8
   ```

The choice of vertical-then-horizontal **must not be reversed** — the
horizontal filter reads pixels above the row, which would otherwise be
mid-vertical-filter. This is a hard correctness rule
(`vp9_loopfilter.c:1252` comment).

`filter_selectively_vert_row2` (`vp9_loopfilter.c:297-372`) is the
"dual" optimisation: it processes two 8×8 rows of an SB row at a time
and tries to call the `_dual` SIMD variants when neighbouring blocks'
filter levels match. The control flow per-bit:

```text
dual_mask_cutoff = ssx ? 0xFF : 0xFFFF
lfl_forward      = ssx ? 4 : 8
dual_one         = 1 | (1 << lfl_forward)

let mask = (m16 | m8 | m4 | m4int) & dual_mask_cutoff;
while mask != 0:
    if mask & dual_one:
        lfis[0] = &lfthr[*lfl]
        lfis[1] = &lfthr[*(lfl + lfl_forward)]
        ss[1] = ss[0] + 8 * pitch

        # mask_16x16: prefer dual variant when both bits set
        if m16 & dual_one == dual_one:
            vpx_lpf_vertical_16_dual(ss[0], …)
        elif m16 & 1:
            vpx_lpf_vertical_16(ss[0], …)
        elif m16 & dual_one:
            vpx_lpf_vertical_16(ss[1], …)
        # similar for 8 and 4 (with respective _dual variants)
        # mask_4x4_int adds an internal-4-tap pass at +4 cols
    ss[0] += 8; lfl += 1
    mask_*   >>= 1
```

The non-dual horizontal pass (`filter_selectively_horiz`,
`vp9_loopfilter.c:453-543`) similarly dispatches `_dual` when the
adjacent column also has the same TX-size edge. Critical detail:
mask priority is `16x16 > 8x8 > 4x4` per column — a column with
16×16 set takes the 16-tap path regardless of any 8/4 bits at that
column. The mask-build pass guarantees these are mutually exclusive
(see asserts at the end of `vp9_adjust_mask`).

The UV variant `vp9_filter_block_plane_ss11`
(`vp9_loopfilter.c:1324-1420`) does the same pattern with subsampling
`ss_x = ss_y = 1`; the per-8×8 UV level array is **derived** from the
Y array (each UV 4×4 inherits the level of its Y co-located 8×8):

```text
for r in (0, 4):                              # two UV "rows" per SB
    for c in 0..4:
        lfl_uv[r*2 + c]       = lfm.lfl_y[r*8 + c*2]
        lfl_uv[(r+2)*2 + c]   = lfm.lfl_y[(r+2)*8 + c*2]
```

(`vp9_loopfilter.c:1342-1346`.) `LF_PATH_SLOW` (decoder, 4:2:2 /
4:4:4) goes through `vp9_filter_block_plane_non420`
(`vp9_loopfilter.c:1078-1237`) which **rebuilds** masks locally from
the MI grid every plane and SB rather than from the pre-computed
`LOOP_FILTER_MASK` (slower; only used for non-420 chroma subsampling).

### 5.6 Frame iteration

`loop_filter_rows` (`vp9_loopfilter.c:1422-1467`) walks
`mi_row in start..stop step MI_BLOCK_SIZE` (8 MIs = 64 pixels at a
time). For each, walks `mi_col in 0..cm.mi_cols step MI_BLOCK_SIZE`:

```text
fn loop_filter_rows(frame, common, planes, start, stop, y_only):
    path = if y_only or 4:4:4 then LF_PATH_444
           else if 4:2:0      then LF_PATH_420
           else                    LF_PATH_SLOW
    for mi_row in (start..stop).step(MI_BLOCK_SIZE):
        let mi  = &common.mi_grid_visible[mi_row * mi_stride ..]
        let lfm = get_lfm(&common.lf, mi_row, 0)   # ptr into lfm[]
        for mi_col in (0..common.mi_cols).step(MI_BLOCK_SIZE):
            setup_dst_planes(planes, frame, mi_row, mi_col)
            adjust_mask(common, mi_row, mi_col, lfm)  # in-place
            filter_block_plane_ss00(common, &planes[0], mi_row, lfm)
            for plane in 1..num_planes:
                match path:
                    LF_PATH_420  -> filter_block_plane_ss11(common, plane, mi_row, lfm)
                    LF_PATH_444  -> filter_block_plane_ss00(common, plane, mi_row, lfm)
                    LF_PATH_SLOW -> filter_block_plane_non420(common, plane, …)
            lfm += 1
```

`vp9_loop_filter_frame` (`vp9_loopfilter.c:1469-1483`) wraps this with
the partial-frame range computation (centre strip for sub-image RDO,
dead in RT) and the `frame_filter_level > 0` short-circuit.

The fast 4:2:0 encoder path always takes `LF_PATH_420` because
`PlaneDesc { subsampling_x: 1, subsampling_y: 1 }` is hard-coded at
profile 0 (D-01 / 8-bit / 4:2:0).

### 5.7 `pick_filter_level` (encoder-side)

At cpu-used ≥ 3, `sf->lpf_pick = LPF_PICK_FROM_Q`
(`vp9_speed_features.c:555`). At cpu-used 6-8 this is the **only**
path. The algorithm (`vp9_picklpf.c:159-203`):

```text
fn pick_filter_level(src, lf, common, method):
    lf.sharpness_level = 0       # forced for RT

    if method == MinimalLpf:
        if lf.filter_level > 0:
            lf.filter_level = 0
        return

    if method >= FromQ:
        let max = MAX_LOOP_FILTER     # pass == 0 always
        let q = ac_quant(common.base_qindex, 0, bit_depth)  # 8-bit only
        let mut filt_guess =
            round_power_of_two(q * 20723 + 1_015_158, 18)

        # AQ-3 + CBR + post-warmup + non-screen + non-key adjustment:
        if pass == 0 && rc_mode == CBR && aq_mode == CyclicRefresh
           && seg.enabled
           && (base_qindex < 200 || width*height > 320*240)
           && content != SCREEN
           && frame_type != KEY_FRAME:
            filt_guess = (5 * filt_guess) >> 3   # 0.625x

        if frame_type == KEY_FRAME:
            filt_guess -= 4

        lf.filter_level = clamp(filt_guess, 0, max)
        return

    # FromFullImage / FromSubImage — RDO golden search.
    lf.filter_level = search_filter_level(src, common, lf,
        partial_frame: method == FromSubImage)
```

The linear fit `filt = q * 0.316206 + 3.87252` is in libvpx
fixed-point as `q * 20723 + 1015158, >> 18`. Test-bench validation:
`q = 0  → ROUND(1_015_158, 18) = round(3.87…) = 4`, `q = 255 →
ROUND(255*20723 + 1015158, 18) = ROUND(6299523, 18) ≈ 24`. Wire-format
range matches.

**`search_filter_level` (RDO, dead in RT)** (`vp9_picklpf.c:78-157`):
golden-section search seeded at `last_filt_level`, range
`[0, max_filter_level]`, initial step `4` (if seed < 16) or
`seed / 4`. Per iteration, encode-filter at `filt_low` and `filt_high`,
compute SSE via `vpx_get_y_sse`, bias against raising the level (the
classic libvpx visual-quality trim), halve step if no improvement.
Each `try_filter_frame` (`vp9_picklpf.c:46-76`):

1. Save the unfiltered Y plane to `cpi->last_frame_uf` (copy).
2. `vp9_build_mask_frame` + `vp9_loop_filter_frame[_mt]`.
3. `vpx_get_y_sse(src, frame_to_show)` — that's the search oracle.
4. Restore unfiltered Y from `last_frame_uf`.

The `last_frame_uf` buffer is encoder-internal state, allocated once
(`vp9_alloc_compressor_data`). For the RT path it is never used and
need not be allocated (see §7 ledger entry).

### 5.8 `loopfilter_frame` encoder shell

`vp9_encoder.c:3392-3452`. The control flow this LLD must implement
on the encoder side:

```text
fn loopfilter_frame(state):
    is_ref =  state.frame_type == KEY_FRAME
          || state.refresh_last_frame
          || state.refresh_golden_frame
          || state.refresh_alt_ref_frame
    if svc.bypass_temporal_layering:
        is_ref = !svc.non_reference_frame

    # 1. show_existing_frame → skip LF and return.
    if state.common.show_existing_frame:
        state.common.lf.filter_level = 0
        return

    # 2. LOOPFILTER_CONTROL.
    if state.loopfilter_ctrl == LoopfilterControl::None
       || (!is_ref && state.loopfilter_ctrl == LoopfilterControl::Reference):
        state.common.lf.filter_level = 0
        extend_frame_inner_borders(state.frame_to_show)
        return

    # 3. lossless mode → no LF.
    if state.xd.lossless:
        state.common.lf.filter_level    = 0
        state.common.lf.last_filt_level = 0
    else:
        # Forced KF: nuke the seed so the search starts from scratch.
        if state.common.frame_type == KEY_FRAME
           && !state.rc.this_key_frame_forced:
            state.common.lf.last_filt_level = 0
        pick_filter_level(state.source.as_ref(), &mut state.common.lf,
                          &state.common, state.sf.lpf_pick)
        state.common.lf.last_filt_level = state.common.lf.filter_level

    if state.common.lf.filter_level > 0 && is_ref:
        build_mask_frame(&mut state.common, &mut state.common.lf,
                         &state.lf_info, state.common.lf.filter_level,
                         partial_frame: false)
        if state.num_workers > 1:
            loop_filter_frame_mt(…)
        else:
            loop_filter_frame(…)

    extend_frame_inner_borders(state.frame_to_show)
```

The **time-pick-lpf timer** in libvpx (`vp9_encoder.c:3422`,
`:3437-3438`) is a profiling-only counter (`cpi->time_pick_lpf`). We
expose it via the optional `EncodeProfiler` trait HLD-arch §9; not in
the LLD shape.

### 5.9 Multi-threaded pass: `loop_filter_frame_mt`

`vp9_thread_common.c:157-217, 219-240`. Pseudocode:

```text
fn loop_filter_frame_mt(frame, common, lf, lf_info, planes,
                        level, y_only, partial_frame,
                        workers, lf_sync):
    if level == 0: return
    let (start, stop) = frame_range(common, partial_frame)
    loop_filter_frame_init(lf_info, lf, &common.seg, level)

    let sb_rows     = (common.mi_rows + 7) >> 3
    let num_tile_cols = 1 << common.log2_tile_cols
    let workers_used = min(workers.num_workers,
                           min(num_tile_cols, sb_rows))

    if lf_sync.shape_differs(sb_rows, common.width, workers_used):
        loop_filter_dealloc(lf_sync)
        loop_filter_alloc(lf_sync, sb_rows, common.width, workers_used)
    lf_sync.num_active_workers = workers_used

    # Reset every row's cur_sb_col to -1.
    for r in 0..sb_rows: lf_sync.rows[r].state.lock().cur_sb_col = -1

    # Dispatch each worker. Worker i starts at SB row start + i*8 and
    # strides by workers_used*8 (`vp9_thread_common.c:201-203`).
    for i in 0..workers_used:
        let lfd = &mut lf_sync.lfdata[i]
        lfd.start_mi_row = start + i * MI_BLOCK_SIZE
        lfd.stop_mi_row  = stop
        lfd.y_only       = y_only
        workers.dispatch(i, /* fn = */ loop_filter_row_worker, /* args */ lfd)

    workers.join_all()
```

Worker loop (`thread_loop_filter_rows`, `vp9_thread_common.c:90-145`):

```text
fn loop_filter_row_worker(lf_sync, lfd, frame, common, planes, sb_cols):
    let path = path_for_planes(planes, lfd.y_only)
    let n = lf_sync.num_active_workers
    let mi_row = lfd.start_mi_row
    while mi_row < lfd.stop_mi_row:
        for mi_col in (0..common.mi_cols).step(MI_BLOCK_SIZE):
            let r = mi_row >> 3
            let c = mi_col >> 3
            sync_read(lf_sync, r, c)            # wait for row r-1
            setup_dst_planes(planes, frame, mi_row, mi_col)
            adjust_mask(common, mi_row, mi_col,
                        get_lfm(&common.lf, mi_row, mi_col))
            filter_block_plane_ss00(common, &mut planes[0], mi_row, lfm)
            for p in 1..num_planes(lfd.y_only):
                match path: …
            sync_write(lf_sync, r, c, sb_cols)  # tell row r+1 they can run
        mi_row += n * MI_BLOCK_SIZE
```

`sync_read` (`vp9_thread_common.c:38-56`): wait on row `r-1`'s
condition variable until `cur_sb_col[r-1] >= c - sync_range + 1`. The
check is gated to `c & (sync_range - 1) == 0` so only every
`sync_range`-th SB column does an actual lock.

`sync_write` (`vp9_thread_common.c:58-87`): updates `cur_sb_col[r] = c`
(or `sb_cols + sync_range` at end-of-row) and signals only on
`sync_range` boundaries.

Worker-fanout invariant: workers `0..n` filter rows `0, n, 2n, …` and
`1, n+1, 2n+1, …` and so on. Two workers can be on row `r` and
`r + n*8` at any moment, but **no two workers are on the same row**;
the row-sync only mediates `(r, r+1)` dependencies — not `(r, r+n*8)`.
This is sound because workers stride large enough that worker `i+1`
on row `r` is always **n** SB-rows behind worker `i` on row `r + n*8`,
and the SBs are written disjointly.

### 5.10 Pre-condition: `lf_row_sync` allocation

`loop_filter_alloc` (`vp9_thread_common.c:279-339`):

```text
fn loop_filter_alloc(lf_sync, sb_rows, width, num_workers):
    lf_sync.sb_rows     = sb_rows
    lf_sync.num_workers = num_workers
    lf_sync.num_active_workers = num_workers
    lf_sync.sync_range  = get_sync_range(width)
    lf_sync.rows        = (0..sb_rows).map(|_|
                              LfRowSync { state: Mutex::new(LfRowState{ cur_sb_col: -1 }),
                                          cond:  Condvar::new() }
                          ).collect::<Vec<_>>().into_boxed_slice()
    lf_sync.lfdata      = (0..num_workers).map(|_| LfWorkerData::zeroed())
                                          .collect::<Vec<_>>().into_boxed_slice()
```

The `get_sync_range(width)` step (`vp9_thread_common.c:265-276`)
matches the table in §4.7.

Deallocation is automatic via `Drop` on `Box<[LfRowSync]>`;
`loop_filter_dealloc` is exposed as a no-op-equivalent for API parity
(the encoder shell calls it on reconfigure to force a re-alloc on the
next pass).

### 5.11 Why this order matters

A summary of why we cannot reorder the pipeline:

1. Mask machinery runs **before** filter pass. The filter pass reads
   `lfl_y` (per-8×8 level), `left_y[tx]`, `above_y[tx]`, `int_4x4_y`
   plus UV equivalents per SB. Building masks during the filter would
   stall vector throughput.
2. Vertical pass runs **before** horizontal pass per-SB. The
   horizontal filter at row `r` reads pixels at rows `r-3..r+3`; if
   the vertical filter ran second on rows `r-1` and `r+1`, the
   horizontal filter would have read stale pixels along the horizontal
   edge.
3. Row sync runs `r-1 -> r` only, in the **horizontal** SB direction
   (column position). The vertical pass on row `r` is allowed to start
   immediately after row `r-1` has filtered far enough horizontally —
   no full-row barrier.

## 6. SIMD dispatch

The 12 kernels are owned by `rvp9-core::simd::LpfDispatch`, populated
once at boot from per-arch tables. Source: `02-simd-dispatch.md:1083-1129`.

```rust
#[repr(C)]
pub struct LpfDispatch {
    pub vertical_16:        LpfFn,       pub vertical_16_dual: LpfDualFn,
    pub vertical_8:         LpfFn,       pub vertical_8_dual:  LpfDualFn,
    pub vertical_4:         LpfFn,       pub vertical_4_dual:  LpfDualFn,
    pub horizontal_16:      LpfFn,       pub horizontal_16_dual: LpfDualFn,
    pub horizontal_8:       LpfFn,       pub horizontal_8_dual:  LpfDualFn,
    pub horizontal_4:       LpfFn,       pub horizontal_4_dual:  LpfDualFn,
}

pub type LpfFn = unsafe extern "C" fn(
    s: *mut u8, pitch: i32,
    blimit: *const u8, limit: *const u8, thresh: *const u8,
);
pub type LpfDualFn = unsafe extern "C" fn(
    s: *mut u8, pitch: i32,
    blimit0: *const u8, limit0: *const u8, thresh0: *const u8,
    blimit1: *const u8, limit1: *const u8, thresh1: *const u8,
);
```

Per-arch availability (verbatim from
`vpx_dsp/vpx_dsp_rtcd_defs.pl:449-483`):

| Kernel | ISAs (8-bit) |
|---|---|
| `vpx_lpf_vertical_16`        | scalar, sse2, neon |
| `vpx_lpf_vertical_16_dual`   | scalar, sse2, neon |
| `vpx_lpf_vertical_8`         | scalar, sse2, neon |
| `vpx_lpf_vertical_8_dual`    | scalar, sse2, neon |
| `vpx_lpf_vertical_4`         | scalar, sse2, neon |
| `vpx_lpf_vertical_4_dual`    | scalar, sse2, neon |
| `vpx_lpf_horizontal_16`      | scalar, sse2, **avx2**, neon |
| `vpx_lpf_horizontal_16_dual` | scalar, sse2, **avx2**, neon |
| `vpx_lpf_horizontal_8`       | scalar, sse2, neon |
| `vpx_lpf_horizontal_8_dual`  | scalar, sse2, neon |
| `vpx_lpf_horizontal_4`       | scalar, sse2, neon |
| `vpx_lpf_horizontal_4_dual`  | scalar, sse2, neon |

For v1: scalar + sse2 + avx2 (the two horizontal-16 variants only) +
neon. dspr2 / msa / lsx are out (no MIPS / no LoongArch targets in v1).

The two AVX2 variants are **horizontal-16 only** because that's the
only kernel where AVX2 (32-byte SIMD) yields a clear win — the 16-tap
filter naturally vectorises over two 8-block columns. The other
kernels have no AVX2 in libvpx; our `LpfDispatch` consequently has
only six AVX2 entries (the two horizontal-16s; the rest fall back to
sse2 on AVX2 CPUs).

**Differential test plan** (per HLD-simd §3.22):

1. **Vector reference**: each pair `(scalar, simd_variant)` must
   produce **bit-exact** outputs for the same `s, pitch, blimit, limit,
   thresh` input. This is a unit test in `crates/rvp9-core/tests/
   loopfilter_simd.rs`.
2. **Fixed vectors** transcribed from `third_party/libvpx/test/
   lpf_test.cc:Loop8Test6Param` (which we port to Rust at Stage 3).
3. **Property test (proptest)**: random pixel buffer (`[u8; 96]`,
   pitch ∈ {16, 32, 64, 96, 128}), random `blimit/limit/thresh` in
   `[1, 80]` × 16-byte broadcast → bit-exact equivalence across
   `(scalar, sse2, avx2, neon)`.
4. **Edge cases**:
   - All pixels equal → no filtering (no pixel write).
   - Input exactly on each threshold (off-by-one detection).
   - Stride negative (legal in the `loop_filter_rows` call site? — no.
     The kernels assume positive pitch. Test pinned to positive.)
   - Pixel-at-boundary `s[0]` / `s[-1]` exercising the `0x80` XOR in
     `filter4` (`vpx_dsp/loopfilter.c:80-110`).

**Call-site bridge in the LLD:** `filter_selectively_horiz` and
`filter_selectively_vert{,_row2}` in our port become inline Rust
functions in `loopfilter::dispatch.rs`. They take an
`&LpfDispatch` (read once per frame from a thread-local snapshot of
`SimdDispatch`) and call `dispatch.vertical_16(…)` etc. — never the
`vpx_*` symbols directly.

## 7. Deliberate divergences from libvpx

Entries below feed `notes/divergence-ledger.md`. Format per the ledger
template.

### L-LPF-1 — `LpfPickMethod::FromFullImage` / `FromSubImage` deferred

**Where:** `rvp9-core::loopfilter::pick_filter_level`,
`search_filter_level` (not ported in v1). libvpx counterpart:
`vp9_picklpf.c:78-157` (`search_filter_level`); `:46-76`
(`try_filter_frame`).
**What libvpx does:** at speed < 3 (RD-loop), performs an SSE-based
golden-section search across `[0, MAX_LOOP_FILTER]` to pick the optimal
per-frame filter level. Re-encodes the loop filter multiple times per
frame (mean ~3-4 trials).
**What rvp9 does:** ship `FromQ` and `MinimalLpf` only. `pick_filter_level`
panics with a clear message if invoked with `FromFullImage` /
`FromSubImage`. The `last_frame_uf` save/restore buffer (needed by
`try_filter_frame`) is not allocated.
**Why:** at cpu-used 6-8 (our target), `sf->lpf_pick = LPF_PICK_FROM_Q`
(`vp9_speed_features.c:555`). `search_filter_level` is dead code at our
config. Removing it saves ~150 LoC and one full-frame buffer.
**Byte-parity impact:** none in matched RT configs (libvpx itself
takes `FromQ` at our speeds).
**Compression-efficiency impact:** none in matched configs; against
libvpx --good (which uses `FromFullImage`), rvp9 RT mode is
intentionally lower quality (different operating point).
**Future plan:** re-port if a quality preset (cpu-used ≤ 4) is added.
Decision: L-LPF-1 (new).

### L-LPF-2 — `LF_PATH_SLOW` (non-4:2:0) decoder-only

**Where:** `rvp9-core::loopfilter::loop_filter_frame` accepts
`PlaneDesc { subsampling_x, subsampling_y }` and dispatches on the
ss00/ss11/non420 path internally — same as libvpx. But the
`rvp9-encoder` crate **never** invokes the non420 path (encoder is
Profile 0 / 4:2:0 only per D-01).
**What libvpx does:** runs the slow path for 4:2:2 / 4:4:4 streams via
`vp9_filter_block_plane_non420`.
**What rvp9 does:** the encoder API forbids profile ≥ 1 at config-time
(`Encoder::new` rejects `bit_depth != 8` or `subsampling != 4:2:0`).
The decoder retains the non-4:2:0 path because Profile 1 streams are
legal at decode (different scope).
**Byte-parity impact:** none — encoder never goes there.
**Compression-efficiency impact:** none.
**Future plan:** re-port encoder support if Profile 1 enters scope.
Decision: D-01 already covers this. Listed for completeness.

### L-LPF-3 — `loopfilter_ctrl` field always present on encoder API

**Where:** `EncoderConfig::loopfilter_ctrl: LoopfilterControl` always
shipped (no Cargo feature gate). libvpx counterpart: `cpi->loopfilter_ctrl`
(`vp9_encoder.h:1033`), always present.
**What libvpx does:** uses `VP9E_SET_LOOPFILTER_CONTROL` control to
override; default = `LOOPFILTER_ALL` (always run LF).
**What rvp9 does:** identical default; trait method
`Encoder::set_loopfilter_control(ctrl: LoopfilterControl)` provides
runtime change.
**Byte-parity impact:** none.
**Compression-efficiency impact:** none.
**Future plan:** stable. Decision: L-LPF-3 (informational; not a
ledger entry — same behaviour as libvpx).

### L-LPF-4 — `lfm` array owned by `LoopFilterState`

**Where:** `LoopFilterState::lfm: Box<[LoopFilterMask]>` (§4.5).
libvpx counterpart: `cm->lf.lfm` (raw pointer), allocation hidden
inside `vp9_alloc_loop_filter` (called by `vp9_alloc_compressor_data`).
**What libvpx does:** `cm->lf.lfm` is a raw `LOOP_FILTER_MASK *`
manually `vpx_calloc`'d at compressor-alloc time, with `lfm_stride`
also field-stored.
**What rvp9 does:** `Box<[LoopFilterMask]>` owned by
`LoopFilterState`; growth on resolution change handled in
`LoopFilterState::resize`. `lfm_stride` retained because the indexing
math `(mi_col >> 3) + (mi_row >> 3) * lfm_stride` is on the hot path.
**Byte-parity impact:** none (memory layout is identical post-init).
**Compression-efficiency impact:** none.
**Future plan:** stable. Decision: L-LPF-4 (new).

### L-LPF-5 — `lf->mode_ref_delta_update = 0` post-pack

**Where:** `rvp9-encoder::frame_loop::post_pack_cleanup`. libvpx
counterpart: `vp9_encoder.c:5884` (resets after `vp9_pack_bitstream`).
**What libvpx does:** clears `cm->lf.mode_ref_delta_update` so the
next frame's bitstream doesn't re-send unchanged deltas.
**What rvp9 does:** identical; mentioned here so the encoder shell's
post-pack section keeps it.
**Byte-parity impact:** none (this is the correct libvpx behaviour
mirrored).
**Compression-efficiency impact:** none.
**Future plan:** stable. Not a divergence; recorded for traceability.

### L-LPF-6 — `LfSync` decoder-extra fields moved to `rvp9-decoder`

**Where:** `LfSync` (§4.7) drops `num_tiles_done`, `recon_done_mutex`,
`recon_done_cond`, `corrupted`. libvpx counterpart:
`vp9_thread_common.h:42-49` (all four fields live in `VP9LfSync`).
**What libvpx does:** unifies decoder + encoder LfSync structs because
the decoder needs recon-completion barriers (LF starts before tile
recon finishes; row N waits on tile recon row N+1 to be done before
filtering row N). The encoder uses `LfSync` post-encode where this is
irrelevant.
**What rvp9 does:** split — `rvp9-core::loopfilter::LfSync` has only
the encoder-relevant fields; `rvp9-decoder::lf_sync_extra::DecoderLfExtra`
adds the recon-done plumbing. Both reference the same `LfRowSync`
mutex/condvar pairs (shared `Box<[LfRowSync]>`).
**Why:** keeps `rvp9-core` independent of any tile-recon completion
machinery. The encoder doesn't need it; embedding it in core would
forge an unnecessary core ↔ decoder coupling.
**Byte-parity impact:** none (same byte output regardless of struct
layout).
**Compression-efficiency impact:** none.
**Future plan:** stable. Decision: L-LPF-6 (new).

### L-LPF-7 — `last_frame_uf` buffer not allocated in RT

**Where:** `rvp9-encoder::Encoder` carries no `last_frame_uf` field.
libvpx counterpart: `cpi->last_frame_uf` (`vp9_encoder.h`, allocated in
`vp9_alloc_compressor_data`).
**What libvpx does:** allocates a full YV12 frame to hold the
pre-loopfilter Y plane so `search_filter_level` can restore the
unfiltered frame between trials.
**What rvp9 does:** L-LPF-1 drops `search_filter_level`, so this
buffer is also dropped. Saves ~1.5 MB at 1080p, ~0.7 MB at 720p.
**Byte-parity impact:** none.
**Compression-efficiency impact:** none.
**Future plan:** restore alongside L-LPF-1 if quality preset is added.
Decision: L-LPF-7 (new; downstream of L-LPF-1).

## 8. Open questions

### 8.1 Stage 1.5 re-dispatch verdict

**Verdict: not required.** Stage 1's coverage on loopfilter was thin
(`01-architecture.md:1875-1877` + `:2303-2307` are the only structured
references), but the libvpx source is small (`vp9_loopfilter.c` is
1,633 lines, mostly mask-build boilerplate; `vp9_thread_common.c` for
LF is 596; `vp9_picklpf.c` is 203). Reading it directly during this
LLD's authoring produced complete coverage of every load-bearing
algorithm — sharpness math, mask layout, vertical-then-horizontal pass
ordering, row-sync semantics, encoder-side RDO bypass.

What a Stage 1.5 specialist would have added:

- Verbatim transcription of every constant table (largely done in §5.3
  here, ~80 LoC).
- Per-kernel scalar reference walk-through (covered well enough by
  HLD-simd §3.22's pointer to `vpx_dsp/loopfilter.c`).
- Devil's-advocate check on `vp9_adjust_mask` edge handling — the
  `rows == 1` / `rows == 5` / `columns == 1` / `columns == 5` special
  cases are subtle. A Stage 1.5 read would have flagged these for
  thorough differential testing; this LLD does so explicitly in §5.4
  and §9.

If Arthur disagrees and would like a Stage 1.5 dispatch, the prompt
would be: "Read `vp9/common/vp9_loopfilter.c`, `vp9_thread_common.c`,
and `vp9/encoder/vp9_picklpf.c` end-to-end. Transcribe every constant
table, document the edge-handling rules verbatim, identify each SIMD
kernel's mathematical operation, and write `08-loopfilter.md` matching
the format of `01-architecture.md` for the LF subsystem." That document
would be ~600 lines.

### 8.2 Allocation lifecycle for `LfSync`

The encoder's worker pool is built once and resized only when
`num_workers` changes (HLD-threading §3). But `LfSync` reallocates
whenever `(sb_rows, width, num_workers)` changes
(`vp9_thread_common.c:173-177`). For dynamic resolution changes
(spatial SVC, see HLD-svc §4) this triggers per-resolution-switch
heap traffic. **Question:** is this acceptable, or should `LfSync`
allocate to a peak size at `Encoder::new`?

**My recommendation:** ship the reallocate-on-shape-change model
(libvpx parity, simpler). At runtime in real-time the shape only
changes on `set_size`/`reconfigure` calls — never per frame. If we
later see allocator spike profiles, we can switch to peak-size
allocation as a non-breaking optimisation.

### 8.3 Inline mask build during encode (TODO at `vp9_loopfilter.c:1486`)

libvpx has a TODO from slavarnway: "Do the encoder the same way the
decoder does it and build the masks in line as part of the encode
process." Currently the encoder does a separate `build_mask_frame`
pass post-encode (one full extra MI walk). The decoder builds masks
incrementally as it decodes blocks (`vp9_build_mask` per-block).

**Question:** should rvp9 unify these into the in-flight encode (each
worker calls `build_mask_sb` after each SB pick)? Saves one MI-grid
walk per frame.

**My recommendation:** start with the post-encode separate pass
(libvpx parity, lower risk). If profiling shows the extra walk on the
hot path, fold it into the encode-MT path. Document the TODO; track
as L-LPF-8 if/when it lands.

### 8.4 `set_row` / `recon_done_cond` placement

These exist in `vp9_thread_common.c:389-437` (`get_next_row`) for the
decoder's row-MT integration where LF and recon overlap. We've split
them to `rvp9-decoder` (L-LPF-6). The actual LLD-decoder will need to
re-introduce these as `DecoderLfExtra`. **No question for this LLD**;
flagged for cross-LLD coordination with LLD-tile-threading.

## 9. Test strategy

### 9.1 Unit tests (`crates/rvp9-core/src/loopfilter/tests.rs`)

- `loop_filter_init`: every (sharpness, lvl) pair produces the right
  `lim`/`mblim`/`hev_thr` byte. Spot-check sharpness 0/3/7 against the
  table at `vp9_loopfilter.c:212-231`.
- `loop_filter_frame_init`: with `mode_ref_delta_enabled=false`, every
  cell equals `default_filt_lvl`. With deltas enabled, every cell
  computed correctly against a hand-rolled scalar reference. Property
  test on `(default_filt_lvl, ref_deltas, mode_deltas, seg_data)`.
- `pick_filter_level(FromQ, …)`: golden-value test at
  `q = 0, 32, 64, 128, 200, 255`; CBR + AQ-3 + non-screen + non-KF
  scaling case; KF -4 adjustment.
- Constant-table identity test: every byte of
  `left_64x64_txform_mask`, `above_64x64_txform_mask`,
  `left_prediction_mask`, `above_prediction_mask`, `size_mask`,
  `left_border`, `above_border`, and the UV equivalents matches
  libvpx exactly. (Constants are transcribed; this test prevents
  copy-paste regressions.)

### 9.2 Mask machinery property tests

- `build_mask_sb` + `adjust_mask`: random MI grid (varied partition
  trees, varied skip flags, varied TX sizes) → mask. Compare against
  the libvpx C reference via a wrapping bench harness in
  `crates/rvp9-harness::lpf_diff` (Stage 3).
- `adjust_mask` edge cases: frame heights of `MI_BLOCK_SIZE*k + r` for
  `r ∈ {1, 2, 3, 4, 5, 6, 7}` and `k ∈ {1, 2, 32}`. Same for widths.
  Both `rows == 1` and `rows == 5` special-case branches must hit.

### 9.3 SIMD differential (per HLD-simd §3.22)

- All 12 kernel symbols (scalar, sse2, [avx2], neon): bit-exact pairs
  under randomised inputs (`proptest` 1k cases per pair).
- Fixed vectors from `test/lpf_test.cc:Loop8Test6Param` ported to
  Rust as a `#[test]` table.

### 9.4 Frame-level integration

- Round-trip: `loop_filter_frame(encoded_frame, …)` followed by our
  decoder applying `loop_filter_frame` to the same parsed
  `filter_level` produces byte-identical pixel output. (Sanity gate:
  encoder-LF and decoder-LF share code; this catches accidental
  divergence in `loop_filter_rows`-driven enumeration.)
- Multi-threaded determinism: `loop_filter_frame_mt` with
  `num_workers ∈ {2, 4, 8}` must produce **identical** output to
  `loop_filter_frame` with `num_workers = 1`. Random seed × frame
  shapes from the test corpus (`tests/conformance/runner.rs`).
- Differential vs. libvpx on the test corpus: at matched config,
  every frame's post-LF Y/U/V plane is byte-identical to
  `vpxdec`-decoded reference. Lands in Stage 3 once the encoder shell
  exists.

### 9.5 Threading soundness

- Miri run of single-threaded `loop_filter_frame` over a fuzzed
  partition grid (catches OOB indexes in the mask machinery).
- `loom` model of `LfRowSync`: two SB rows, three column positions,
  `sync_range = 1` and `sync_range = 2`. Verify no panic, no
  data race on `cur_sb_col`, no missed wakeups
  (`vp9_thread_common.c:38-87` is the spec).
- ThreadSanitizer over `cargo test loop_filter_frame_mt` (when
  building under sanitizers; not on every CI run).

### 9.6 Conformance

The decoder conformance suite (Stage 3 exit criterion) implicitly
covers the LF subsystem — any LF correctness bug surfaces as a
conformance vector mismatch. Specifically:

- Vectors with `filter_level == 0` exercise the early-exit fast path.
- Vectors with non-zero `mode_ref_delta_enabled` exercise the level
  cube.
- Vectors at non-multiple-of-64 resolutions exercise `adjust_mask`'s
  edge handling (`vp9_loopfilter.c:790-851`).
- Vectors with multiple tile columns exercise `LfSync` if we choose to
  multi-thread the decoder LF (we will, per HLD-threading §4.5).

---

**One-line summary:** 1507 lines; §7 carries five new divergence-ledger entries (L-LPF-1 dropped `search_filter_level`, L-LPF-2 encoder-side 4:2:0 only, L-LPF-4 `lfm` owned by `LoopFilterState`, L-LPF-6 decoder LfSync-extra split, L-LPF-7 dropped `last_frame_uf`) plus two informational ones (L-LPF-3, L-LPF-5); §8 contains three minor open questions (allocation lifecycle, inline-build TODO, decoder-LF placement) — no blocking questions; Stage 1.5 re-dispatch flag **lowered** in §8.1 (libvpx source is small enough that direct read filled the gap during LLD authoring).
