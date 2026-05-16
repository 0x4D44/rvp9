# LLD: motion-estimation

**Module:** `rvp9-encoder/src/motion_est/`
**Stage 1 primary:** `notes/vp9-analysis/03-motion-estimation.md`
**Stage 2 anchors:** `notes/hld/00-architecture.md` §3.3 (motion vectors),
§3.2 (frame buffers); `notes/hld/01-threading.md` §5.x (worker aliasing);
`notes/hld/02-simd-dispatch.md` §3.1-§3.10, §3.23 (SAD / variance / sub-pel
variance / int-pro kernels).
**libvpx pin:** v1.14.1 (`12f3a2a`); every `vp9/...` citation resolves
under `third_party/libvpx/`.
**Scope:** cpu-used 6-8, non-RD path, 8-bit only, no altref, no compound
NEWMV. Single-pass CBR + VBR-RT.

---

## 1. Purpose

`motion_est` is the motion-search engine that the non-RD mode picker
(LLD-mode-decision) invokes for every block × reference-frame × inter-mode
candidate. It owns:

- The motion-vector data types (`Mv`, `MvLimits`, ref-MV candidate lists).
- The full-pel search dispatcher (`vp9_full_pixel_search` equivalent) for
  the three search methods reachable from cpu-used 6-8 — `NSTEP`,
  `FAST_DIAMOND`, plus the `vp9_int_pro_motion_estimation` fast-ME used
  for golden NEWMV.
- The sub-pel refinement pipeline (`SUBPEL_TREE` variant only — the
  pruned / pruned_more / pruned_evenmore variants are L-05-stubbed).
- The MV-cost arithmetic (`mvsad_err_cost`, `mv_err_cost`,
  `vp9_mv_bit_cost` equivalents) and the four cost LUTs
  (`nmvjointcost`, `mvcost`, `nmvjointsadcost`, `nmvsadcost`).
- The reference-frame skip mask (`reference_masking` per L-06).
- The `search_site_config` (8-site NSTEP diamond) initialised per
  reference-buffer stride at frame start.
- The `BlockFnPtrs` (BFP) table — the per-block-size collection of SAD,
  SAD-skip, SAD-x4D, variance, and sub-pel-variance function pointers
  (`cpi->fn_ptr[BLOCK_SIZES]` equivalent) — populated once at encoder
  construction from `rvp9-core::simd::SimdDispatch`.

It does **not** own:

- Partition selection (LLD-partition).
- Intra / inter mode-decision scaffolding, skip detection, transform
  selection, R-D modelling (LLD-mode-decision).
- MV bitstream coding or MV probability tables (LLD-bitstream,
  LLD-entropy).
- MV-cost LUTs as **data** (the joint / component cost tables); those
  derive from `Common::fc.nmv_context` and live in
  `rvp9-encoder::mode_decision::cost_tables`. `motion_est` borrows them
  via the `MeCosts` view documented in §4.6.

---

## 2. libvpx origin

| Function (libvpx) | File:line | Owner (this LLD §) |
|---|---|---|
| `vp9_full_pixel_search` (dispatcher) | `vp9/encoder/vp9_mcomp.c:2875-2952` | §3.2, §5.1 |
| `full_pixel_diamond` | `vp9/encoder/vp9_mcomp.c:2486-2605` | §5.2 |
| `vp9_diamond_search_sad_c` | `vp9/encoder/vp9_mcomp.c:2055-2190` | §5.3 |
| `vp9_refining_search_sad` | `vp9/encoder/vp9_mcomp.c:2749-2821` | §5.4 |
| `vp9_pattern_search` + `_sad` | `vp9/encoder/vp9_mcomp.c:1007-1452` | §5.5 (used only via `FAST_DIAMOND` → `bigdia_search`) |
| `vp9_int_pro_motion_estimation` | `vp9/encoder/vp9_mcomp.c:2264-2400` | §5.6 |
| `vector_match` | `vp9/encoder/vp9_mcomp.c:2192-2255` | §5.6 |
| `vp9_find_best_sub_pixel_tree` | `vp9/encoder/vp9_mcomp.c:721-925` | §5.7 |
| `calc_int_cost_list` | `vp9/encoder/vp9_mcomp.c:960-1000` | §5.8 (L-07) |
| `vp9_find_mv_refs` (full) | `vp9/common/vp9_mvref_common.c:133-139` | §5.9 |
| `mv_refs_rt` (RT fast variant) | `vp9/encoder/vp9_pickmode.c:59-152` | §5.9 |
| `vp9_find_best_ref_mvs` | `vp9/common/vp9_mvref_common.c:141-151` | §5.9 |
| `vp9_mv_pred` | `vp9/encoder/vp9_rd.c:588-639` | §5.10 |
| `combined_motion_search` | `vp9/encoder/vp9_pickmode.c:154-261` | §5.0 (top-level glue) |
| `search_new_mv` | `vp9/encoder/vp9_pickmode.c:1575-1672` | §5.0 |
| `vp9_set_mv_search_range` | `vp9/encoder/vp9_mcomp.c:32-49` | §5.11 |
| `vp9_set_subpel_mv_search_range` | `vp9/encoder/vp9_mcomp.c:51-67` | §5.11 |
| `vp9_init3smotion_compensation` | `vp9/encoder/vp9_mcomp.c:116-134` | §5.12 |
| `vp9_init_dsmotion_compensation` | `vp9/encoder/vp9_mcomp.c:98-114` | §5.12 (unused in v1 — 4-site init is for non-rt) |
| `mv_cost`, `mv_err_cost`, `mvsad_err_cost` | `vp9/encoder/vp9_encoder.h:1510-1525`, `vp9/encoder/vp9_mcomp.c:80-97` | §5.13 |
| Reference-masking heuristic | `vp9/encoder/vp9_pickmode.c:2194-2216` | §5.14 (L-06) |
| `vp9_NEWMV_diff_bias` | `vp9/encoder/vp9_pickmode.c:1298-1355` (consumer: `vp9_pickmode.c:1029-1031`) | §5.15 (L-43) |

**Out of scope** (cited only to be explicit about non-port):

| Function | File:line | Why excluded |
|---|---|---|
| `vp9_refining_search_8p_c` (compound joint search) | `vp9_mcomp.c:2825-2873` | L-08: compound NEWMV not reachable at cpu-used 6-8 (`vp9_pickmode.c:2064-2070` only allows ZEROMV compound). |
| `accurate_sub_pel_search` | `vp9_mcomp.c:629-720` (called from `vp9_mcomp.c:756-763`, etc.) | L-05: `use_accurate_subpel_search == USE_2_TAPS` (bilinear) at cpu-used 6-8 — the fused `vfp->svf` path is taken, this branch is dead. |
| `vp9_find_best_sub_pixel_tree_pruned[_more,_evenmore]` | `vp9_mcomp.c:387-597` | Speed-feature gated: `SUBPEL_TREE` is default for rt 0-8 (`vp9_speed_features.c:928`); `_evenmore` only on SVC non-ref at speed 7 (`vp9_speed_features.c:715`) — out of v1's locked-SF surface. |
| `vp9_skip_sub_pixel_tree` | `vp9_mcomp.c:354-385` | Requires `subpel_force_stop == FULL_PEL` (cpu-used 8 screen content only, `vp9_speed_features.c:769`). Stub for v1; revisit when screen-content tune lands. |
| `exhaustive_mesh_search` / `full_pixel_exhaustive` | `vp9_mcomp.c:1637-1721, 2609-2662` | `sf->exhaustive_searches_thresh = INT_MAX` in rt (`vp9_speed_features.c:462`); dispatcher (`vp9_mcomp.c:2924-2935`) never sets `run_exhaustive_search`. Dead code at our config. |
| `vp9_diamond_search_sad_neon` | `vp9/encoder/arm/neon/vp9_diamond_search_sad_neon.c` | HLD-simd §3.23: shipped scalar-only, AVX2 `sdx4df` covers x86_64 (A3 §6.12). Already in divergence ledger as L-RTCD-03 placeholder. |

---

## 3. Public API

The `motion_est` module exposes one entry point used by
LLD-mode-decision, plus a small handful of construction helpers.

### 3.1 Module surface (`rvp9-encoder/src/motion_est/mod.rs`)

```rust
pub use bfp::{Bfp, BlockFnPtrs};
pub use config::SearchSiteConfig;
pub use cost::{MeCosts, MeCostsRef};
pub use limits::{MvLimits, set_mv_search_range, set_subpel_mv_search_range};
pub use refmv::{
    find_mv_refs_rt, find_best_ref_mvs, mv_pred, mv_refs_rt_full,
    RefMvCandidates, MvBestRef,
};
pub use reference_mask::reference_skip_mask;
pub use search::{
    combined_motion_search, int_pro_motion_estimation,
    FullPixelSearchOut, IntProResult, SearchMethod,
};
pub use subpel::find_best_sub_pixel_tree;

pub mod types;       // §4.1-§4.4
pub mod bfp;         // §4.5
pub mod config;      // §4.6
pub mod cost;        // §4.6
pub mod limits;      // §4.7
pub mod refmv;       // §4.8
pub mod reference_mask;
pub mod search;      // §4.9 full-pel
pub mod subpel;      // §4.10
mod tables;          // const search-pattern + step tables
```

### 3.2 Top-level entry — combined full-pel + sub-pel search

```rust
/// Drives the full sequence: dispatches `vp9_full_pixel_search` then
/// `find_best_sub_pixel_tree`, returning the refined sub-pel MV plus
/// the per-step intermediates needed by mode-decision (variance, SSE).
///
/// Mirrors `combined_motion_search` (`vp9_pickmode.c:154-261`) for the
/// non-base-MV path only; SVC `use_base_mv` is handled by the
/// `_with_base_mv` wrapper in §3.3.
///
/// Preconditions:
///   - `block.mv_limits` already clamped (call `set_mv_search_range`
///     before invocation; libvpx does this inside `combined_motion_search`
///     but we hoist it to the call site so that the limits can be reused
///     across the multiple ME calls in one mode loop).
///   - `block.fn_ptr[bsize]` (= `&bfp.get(bsize)`) is the correct BFP.
///   - `mvp_full` is in full-pel units (call site has already done
///     `mvp_full.col >>= 3; mvp_full.row >>= 3`).
///
/// Postconditions:
///   - `out.mv` is in 1/8-pel units, clamped to subpel mv-limits.
///   - `out.full_pel_variance` is what `vp9_full_pixel_search` returned
///     (the variance/SAD at the integer-pel resolution).
///   - `out.sse_after_subpel` is filled from the sub-pel variance call
///     used during refinement (= libvpx's `pred_sse[ref]` per A3 §3.3).
///
/// Panics: never. Out-of-bounds MV-limits clamp at construction.
pub fn combined_motion_search(
    block:        &MotionBlock<'_>,
    ref_frame:    RefFrameId,
    bsize:        BlockSize,
    mvp_full:     Mv,
    ref_mv:       Mv,
    speed:        &MeSpeedFeatures,
    sf_subpel:    SubPelForceStop,
    costs:        &MeCostsRef<'_>,
    bfp:          &BlockFnPtrs,
    ss_cfg:       &SearchSiteConfig,
) -> FullPixelSearchOut;
```

```rust
/// The high-level result returned to LLD-mode-decision. Field ordering
/// mirrors what the caller reads in `vp9_pickmode.c:248-255` so that
/// the consuming code in mode-decision lines up 1:1.
pub struct FullPixelSearchOut {
    pub mv:                     Mv,        // 1/8-pel units, sub-pel refined
    pub full_pel_variance:      u32,       // var at integer-pel (mvpred_var)
    pub sse_after_subpel:       u32,       // = pred_sse[ref] in libvpx
    pub mv_rate_bits:           i32,       // vp9_mv_bit_cost output
}
```

### 3.3 Reference-frame predictor setup

```rust
/// Real-time variant of `vp9_find_mv_refs` per A3 §4.9
/// (`vp9_pickmode.c:59-152`). Fills the 2-entry nearest/near candidate
/// list for `ref_frame`, plus mode-context for entropy coding.
///
/// Returns:
///   - `const_motion`: 1 if the first match was found within the first 2
///     neighbours; used by mode-decision to skip NEARMV later
///     (`vp9_pickmode.c:2165`).
///   - `cands.nearest_mv`, `cands.near_mv` — both in 1/8-pel units,
///     `lower_mv_precision`-rounded and `clamp_mv2`-clamped per
///     `vp9_find_best_ref_mvs` (`vp9_mvref_common.c:141-151`).
///
/// Side effects: mutates `block.mbmi_ext.mode_context[ref_frame]`
/// according to the `counter_to_context` table (A3 §2.5).
pub fn find_mv_refs_rt(
    block:        &mut MotionBlock<'_>,
    common:       &Common,
    ref_frame:    RefFrameId,
    mi_row:       i32,
    mi_col:       i32,
) -> RefMvFindResult;

pub struct RefMvFindResult {
    pub candidates:    RefMvCandidates, // [nearest, near] in 1/8-pel
    pub const_motion:  bool,
    pub base_mv:       Option<Mv>,      // SVC inter-layer fast path
}
```

```rust
/// Best-of-3 starting-MV selection by raw SAD over
/// {NEAREST, NEAR, prev_pred_mv}. Mirrors `vp9_mv_pred`
/// (`vp9_rd.c:588-639`). Sets `block.mv_best_ref_index[ref]`,
/// `block.pred_mv_sad[ref]`, `block.max_mv_context[ref]`.
///
/// The 3rd candidate (`block.pred_mv[ref]`) is consulted only if
/// `bsize < block.max_partition_size`; for the var-based partition
/// path `max_partition_size == BLOCK_64X64`, so the 3rd candidate is
/// active for every block size except full-SB (per A3 §7 #12).
pub fn mv_pred(
    block:        &mut MotionBlock<'_>,
    ref_frame:    RefFrameId,
    bsize:        BlockSize,
    bfp:          &BlockFnPtrs,
);
```

### 3.4 Reference-skip-mask computation (L-06)

```rust
/// Computes the bitmask of references that should be skipped for
/// non-ZERO modes based on `pred_mv_sad` ratios. Mirrors
/// `vp9_pickmode.c:2194-2216`.
///
/// Threshold is hard-coded to `<< 1` (2×) per L-06 RESOLVED — see §5.14
/// for rationale on not exposing it.
///
/// `usable_ref_frame` is the highest ref enabled by
/// `set_usable_ref_frame` (`vp9_pickmode.c:1907-1974`); the mask only
/// considers refs ≤ this.
///
/// `force_skip_low_temp_var` carries the low-temp-variance flag from
/// partition (LLD-partition); when set, the inter-ref masking below
/// ALTREF is bypassed (see §5.14).
pub fn reference_skip_mask(
    pred_mv_sad:              &[u32; MAX_REF_FRAMES],
    ref_frame_flags:          RefFrameFlags,
    usable_ref_frame:         RefFrameId,
    force_skip_low_temp_var:  bool,
    is_src_frame_alt_ref:     bool,
) -> u8;   // bit i set => skip ref_frame i
```

### 3.5 Integer-projection ME (golden NEWMV fast path)

```rust
/// `vp9_int_pro_motion_estimation` (`vp9_mcomp.c:2264-2400`). Used for
/// GOLDEN/ALTREF NEWMV in CBR when LAST-frame SAD is small enough that
/// a full diamond search would be wasted. Returns None when the
/// projection-ME's SAD exceeds the early-exit thresholds at
/// `vp9_pickmode.c:1598-1599` (caller skips NEWMV on golden entirely).
///
/// Preconditions:
///   - CBR mode (caller already checked `oxcf.rc_mode == VPX_CBR`).
///   - `bsize >= BLOCK_16X16`.
///   - `gf_temporal_ref` is true.
///
/// Out-of-scope until rt path supports VBR fast-path (A3 §7 #8); the
/// dispatcher in `search_new_mv` keeps the same predicate.
pub fn int_pro_motion_estimation(
    block:         &MotionBlock<'_>,
    bsize:         BlockSize,
    ref_frame:     RefFrameId,
    pred_mv_sad_last:  u32,    // x->pred_mv_sad[LAST_FRAME]
    best_pred_sad:     u32,    // running best across refs
    bfp:           &BlockFnPtrs,
    int_pro:       &IntProKernels, // = SimdDispatch::int_pro_row/col/vector_var
) -> Option<IntProResult>;

pub struct IntProResult {
    pub mv:              Mv,    // 1/8-pel units, clamped to subpel mv-limits
    pub sad:             u32,
}
```

### 3.6 Sub-pel-only refinement (used by mode-decision directly)

For completeness; `combined_motion_search` calls this internally. The
direct entry exists because `vp9_int_pro_motion_estimation`'s result is
followed by a `vp9_find_best_sub_pixel_tree` call independent of the
full-pel dispatcher (`vp9_pickmode.c:1608-1615`).

```rust
/// `vp9_find_best_sub_pixel_tree` (`vp9_mcomp.c:721-925`). The
/// SUBPEL_TREE variant. Other variants are L-05-stubbed (see §7).
///
/// Preconditions:
///   - `bestmv` is in 1/8-pel units (call sites multiply integer-pel
///     output of full-pel search by 8 before calling).
///   - `cost_list` is the 5-entry array produced by `calc_int_cost_list`
///     or `vp9_pattern_search_sad`; unused by the SUBPEL_TREE variant
///     itself but must be valid for differential testing against the
///     pruned variants (L-07).
///
/// Returns: refined `Mv` in 1/8-pel units, with the final `sse` of the
/// sub-pel-variance check exposed via the `out.sse` field.
pub fn find_best_sub_pixel_tree(
    block:                  &MotionBlock<'_>,
    bestmv:                 Mv,
    ref_mv:                 Mv,
    allow_hp:               bool,
    error_per_bit:          i32,
    bfp:                    &BlockFnPtrs,
    forced_stop:            SubPelForceStop,
    iters_per_step:         i32,        // = sf.mv.subpel_search_level
    cost_list:              &[i32; 5],  // L-07
    costs:                  &MeCostsRef<'_>,
    use_accurate_subpel_search:  AccurateSubPelSearch,  // always USE_2_TAPS in v1
) -> SubPixelOut;

pub struct SubPixelOut {
    pub mv:    Mv,
    pub var:   u32,
    pub sse:   u32,
}
```

### 3.7 Constructors

```rust
impl SearchSiteConfig {
    /// `vp9_init3smotion_compensation` (`vp9_mcomp.c:116-134`). Built
    /// per reference-buffer stride at `init_motion_estimation` time
    /// (Step 24 in 00-overview §2). Recomputed by the frame loop on
    /// resolution change.
    pub fn nstep_8point(stride: i32) -> Self;
}

impl Bfp {
    /// Snapshot the per-block-size function pointers out of the shared
    /// `SimdDispatch`. Called once at `Encoder::new`. See HLD-simd §4.3
    /// for the data layout.
    pub fn new(dispatch: &rvp9_core::simd::SimdDispatch) -> Self;
    pub fn get(&self, bsize: BlockSize) -> &BlockFnPtrs;
}
```

### 3.8 Unsafety surface

Everything in §3 is safe Rust at its boundary. The unsafe blocks are
internal:

- The SAD / variance / sub-pixel-variance trampoline — every call to
  `unsafe extern "C" fn` from `BlockFnPtrs`. Wrapped by the §4.5 thin
  trait `Sad`, `Variance`, etc. that names the safety obligations once
  per kernel family.
- Pointer arithmetic into the reference frame buffer (`get_buf_from_mv`
  equivalent) — wrapped by the `RefPlane::offset` accessor in §4.3.
- Raw construction of `MeCostsRef<'_>` from `&mut Common::fc`-derived
  cost LUT slices — proof obligation on the borrow lifetime (LUTs are
  built per-frame on the main thread before worker spawn, then read-only
  during encode; see HLD-threading §5.x and 00-overview §3.1).

No `Pin`, no `UnsafeCell` in this module. The `MotionBlock<'_>` borrow
form is just a `&mut` to the per-worker `ThreadData::mb` snapshot.

---

## 4. Types

### 4.1 Motion vector

```rust
// rvp9-core/src/geom/mv.rs (exposed re-export at rvp9_core::geom::Mv).
// HLD-arch §3.3 commits to (row: i16, col: i16). All MV arithmetic in
// motion_est goes through this type.

#[repr(C)]
#[derive(Clone, Copy, Default, Eq, PartialEq, Hash)]
pub struct Mv {
    pub row: i16,
    pub col: i16,
}

impl Mv {
    /// `INVALID_MV = 0x80008000` per `vp9/common/vp9_mv.h:22-37`.
    /// Used as sentinel in `frame_mv[NEWMV][ref]` before any search.
    pub const INVALID: Mv = Mv { row: i16::MIN, col: i16::MIN };

    pub const fn as_u32(self) -> u32 {
        ((self.row as u16 as u32) << 16) | (self.col as u16 as u32)
    }

    pub const fn is_zero(self) -> bool { self.row == 0 && self.col == 0 }

    /// Round 1/8-pel MV down to integer-pel. `vp9_pickmode.c:194-195`.
    pub const fn to_full_pel(self) -> Mv {
        Mv { row: self.row >> 3, col: self.col >> 3 }
    }

    /// Lift integer-pel MV to 1/8-pel. `vp9_mcomp.c:285-286`.
    pub const fn to_subpel(self) -> Mv {
        Mv { row: self.row * 8, col: self.col * 8 }
    }
}
```

Encoded MV range constants (`vp9_entropymv.h:79-81`):

```rust
pub const MV_IN_USE_BITS: u32 = 14;
pub const MV_UPP: i16 = (1 << MV_IN_USE_BITS) - 1;       // 16383
pub const MV_LOW: i16 = -(1 << MV_IN_USE_BITS);          // -16384
pub const MV_MAX: i32 = (1 << MV_IN_USE_BITS) - 1;       // for indexing mvcost (component cost LUT)
pub const MV_VALS: i32 = (1 << MV_IN_USE_BITS) * 2 + 1;  // total component vals
pub const USE_MV_HP_THRESH: i32 = 64;                    // kMvRefThresh, vp9_entropymv.h:31

/// `use_mv_hp(ref)`: 1/8-pel allowed only when ref is near zero
/// (`|row| < 64 && |col| < 64`). `vp9_mcomp.c:51-67`, A3 §5.8.
#[inline]
pub fn use_mv_hp(reference: Mv) -> bool {
    reference.row.unsigned_abs() < USE_MV_HP_THRESH as u16
        && reference.col.unsigned_abs() < USE_MV_HP_THRESH as u16
}
```

### 4.2 MV-limits

```rust
// rvp9-encoder/src/motion_est/limits.rs

#[derive(Clone, Copy, Debug, Default)]
pub struct MvLimits {
    pub col_min: i32,
    pub col_max: i32,
    pub row_min: i32,
    pub row_max: i32,
}
```

Field order mirrors `vp9/encoder/vp9_block.h:58-63`. Constants:

```rust
pub const MAX_MVSEARCH_STEPS: u32 = 11;
pub const MAX_FIRST_STEP: i32 = 1 << (MAX_MVSEARCH_STEPS - 1);   // 1024
pub const MAX_FULL_PEL_VAL: i32 = MAX_FIRST_STEP - 1;            // 1023
// vp9_mcomp.h:34 — BORDER_MV_PIXELS_B16 = 16 + VP9_INTERP_EXTEND
pub const BORDER_MV_PIXELS_B16: i32 = 16 + 4;                    // 4 = VP9_INTERP_EXTEND
```

`MV_BORDER = 16 << 3 = 128` (1/8-pel units, `vp9_mvref_common.h:214`).

### 4.3 Reference-buffer view

The worker hot loop indexes into a reference frame's Y plane (and U/V
for compound predictions, which we don't reach). HLD-arch §3.2 puts
frame buffers behind `Arc<FrameBuffer>` and grants tile workers
read-only slices via `&RefFrames`. For ME the relevant view is:

```rust
/// `&'a` borrow of one reference plane (Y for the non-RD path).
/// Built per-frame by the encoder's frame-loop before worker spawn from
/// `Common::frame_refs[ref-1].buf.y_buffer / y_stride` (A3 §2.6,
/// `vp9/common/vp9_blockd.h:57-65`).
#[derive(Clone, Copy)]
pub struct RefPlane<'a> {
    pub data:   *const u8,    // base pointer; raw because workers index
                              // through SAFETY-documented bounds checks
    pub stride: i32,
    _marker: PhantomData<&'a [u8]>,
}

impl<'a> RefPlane<'a> {
    /// `get_buf_from_mv` (`vp9_mcomp.c:75-78` equivalent). Returns the
    /// pointer at `base + row*stride + col` in **full-pel** units.
    ///
    /// # Safety
    /// Caller must have validated `mv` lies within `MvLimits`. The
    /// resulting pointer is valid for the block extent (bsize) provided
    /// `mv` is in the clamped range — see §5.1 §all_in.
    #[inline]
    pub unsafe fn offset(&self, mv: Mv) -> *const u8 {
        self.data.offset(mv.row as isize * self.stride as isize + mv.col as isize)
    }
}

unsafe impl<'a> Send for RefPlane<'a> {}
unsafe impl<'a> Sync for RefPlane<'a> {}
```

`Send + Sync` is justified by HLD-threading §5.x: reference frames are
read-only during a frame encode; the `Arc<FrameBuffer>` they descend
from is mutated only between frames by the main thread.

### 4.4 Block context

`MotionBlock<'_>` is the subset of `MACROBLOCK` (`vp9_block.h:85-132`)
that motion-search reads or writes. Per L-29 we do not preserve a flat
`VP9_COMP`-equivalent; mode-decision LLD owns the full `MotionBlock`
type, and `motion_est` consumes the slice it needs through this view.

```rust
/// View into per-block state needed by motion search. Owned by
/// `ThreadData::mb` (LLD-mode-decision §4.x). The borrow is `&mut`
/// because ME writes to `mv_best_ref_index`, `pred_mv_sad`,
/// `pred_sse`, `mv_limits`.
pub struct MotionBlock<'a> {
    // Source plane (current block being encoded).
    pub src:                   SrcPlane<'a>,

    // Reference plane for the ref currently being searched.
    pub pre:                   RefPlane<'a>,

    // ME-search bounds (`vp9_block.h:129`).
    pub mv_limits:             MvLimits,

    // SAD-domain cost-per-bit (`vp9_block.h:90, 93`).
    pub sadperbit:             i32,        // 16 for ≥8x8 blocks; 4 for sub-8x8

    // Variance-domain cost-per-bit (`vp9_block.h:87`).
    pub errorperbit:           i32,

    // Best-of-3 ref-MV index (selected by mv_pred, `vp9_block.h:105`).
    pub mv_best_ref_index:     [u8; MAX_REF_FRAMES as usize],

    // SSE between source and best-MV prediction (`vp9_block.h:108`).
    pub pred_sse:              [u32; MAX_REF_FRAMES as usize],

    // SAD at best MV from vp9_mv_pred (`vp9_block.h:109`).
    // Drives reference_masking (L-06) — see §5.14.
    pub pred_mv_sad:           [u32; MAX_REF_FRAMES as usize],

    // Last-frame MV at the same position (3rd mv_pred candidate)
    // (`vp9_block.h:173`).
    pub pred_mv:               [Mv; MAX_REF_FRAMES as usize],

    // Whether 1/8-pel is allowed for the current frame
    // (= cm->allow_high_precision_mv).
    pub allow_hp:              bool,

    // ME-cost LUT view — see §4.6.
    // 'a outlives the worker; built by main thread.
    pub costs:                 MeCostsRef<'a>,

    // Two-entry NEAREST/NEAR candidate list per ref
    // (`vp9_block.h:50-56` MB_MODE_INFO_EXT.ref_mvs).
    pub ref_mvs:               [[Mv; MAX_MV_REF_CANDIDATES as usize]; MAX_REF_FRAMES as usize],
}

pub const MAX_REF_FRAMES: u8 = 4;          // INTRA + LAST + GOLDEN + ALTREF
pub const MAX_MV_REF_CANDIDATES: u8 = 2;
pub const MAX_INTER_REF_FRAMES: u8 = 3;
```

`SrcPlane<'a>` is symmetric to `RefPlane<'a>` (defined in LLD-mode-decision).
Both are `unsafe` raw-pointer views into the worker-disjoint slices.

### 4.5 Block function-pointer table (BFP)

The libvpx `cpi->fn_ptr[BLOCK_SIZES]` per A3 §2.8. HLD-simd §4.3
already specifies the struct; this LLD re-exports through `bfp::Bfp`.

```rust
// rvp9-encoder/src/motion_est/bfp.rs
pub use rvp9_core::simd::BlockFnPtrs;  // same struct, no wrapping

pub struct Bfp(pub(crate) [BlockFnPtrs; 13]);

impl Bfp {
    pub fn new(d: &rvp9_core::simd::SimdDispatch) -> Self {
        // HLD-simd §4.3 build helper.
        Self(rvp9_core::simd::BlockFnPtrs::build_all(d))
    }
    #[inline]
    pub fn get(&self, bsize: BlockSize) -> &BlockFnPtrs {
        &self.0[bsize as usize]
    }
}
```

### 4.6 MV-cost LUT borrow

The four MV-cost LUTs (`nmvjointcost[4]`, `mvcost[2][MV_VALS]`,
`nmvjointsadcost[4]`, `nmvsadcost[2][MV_VALS]`) are derived per-frame
by the main thread inside `vp9_initialize_rd_consts` (architecture
A1 §3.4) from `Common::fc.nmv_context`. They are stored in
`rvp9-encoder::mode_decision::cost_tables::FrameCosts` and shared
read-only across all tile workers. `motion_est` consumes a thin view:

```rust
/// Borrow of the four MV-cost LUTs, all pointing into one
/// `FrameCosts` allocation. Workers read only.
///
/// SAFETY (Send/Sync): the `FrameCosts` owner ensures the LUTs are
/// constructed before worker spawn and dropped after worker join
/// (HLD-threading §5.x).
#[derive(Clone, Copy)]
pub struct MeCostsRef<'a> {
    pub nmvjointcost:     &'a [i32; 4],
    pub mvcost:           [&'a [i32; MV_VALS as usize]; 2],

    pub nmvjointsadcost:  &'a [i32; 4],
    pub mvsadcost:        [&'a [i32; MV_VALS as usize]; 2],
}
```

The `mvcost` and `mvsadcost` outer arrays are offset-by-MV_MAX views
that allow indexing by signed component values `[-MV_MAX, MV_MAX]`.
The owner constructor builds them as `&lut[MV_MAX..]`. Mirrors the
libvpx convention noted in A1 §3.3 step 18.

### 4.7 Speed-feature subset

ME consumes the `MV_SPEED_FEATURES`-equivalent slice of
`SPEED_FEATURES`. Per A3 §5.5:

```rust
#[derive(Clone, Copy, Debug)]
pub struct MeSpeedFeatures {
    pub search_method:               SearchMethod,           // NSTEP (cpu6) / FAST_DIAMOND (cpu7-8)
    pub fullpel_search_step_param:   i32,                    // 6 (cpu6) / 10 (cpu7-8); A3 §5.5
    pub subpel_search_method:        SubPelSearchMethod,     // SUBPEL_TREE only in v1
    pub subpel_force_stop:           SubPelForceStop,        // EIGHTH_PEL default
    pub subpel_search_level:         i32,                    // = iters_per_step inside subpel
    pub use_downsampled_sad:         bool,                   // true at rt default
    pub reduce_first_step_size:      i32,                    // 1 (cpu6+); used only by non-rt callers
    pub reference_masking:           bool,                   // true at rt single-layer
    pub limit_newmv_early_exit:      bool,                   // true (cpu6 CBR non-screen)
    pub bias_golden:                 bool,                   // true (cpu6 CBR non-screen non-SVC)
    pub use_accurate_subpel_search:  AccurateSubPelSearch,   // always USE_2_TAPS in v1
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchMethod { Hex, FastHex, FastDiamond, Bigdia, Square, Nstep }

/// `SUBPEL_FORCE_STOP`, `vp9_speed_features.h:168`. Values mirror libvpx
/// so the bitstream-affecting integer comparisons in
/// `vp9_find_best_sub_pixel_tree` (e.g. `forced_stop = 3 - round`) stay
/// byte-equivalent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum SubPelForceStop { EighthPel = 0, QuarterPel = 1, HalfPel = 2, FullPel = 3 }

/// Variants per `vp9_speed_features.h`. `_pruned*` and `Skip` are L-05
/// dead code in v1 — we keep the enum to mirror libvpx and to make a
/// future re-enabling additive rather than invasive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubPelSearchMethod { Tree, TreePruned, TreePrunedMore, TreePrunedEvenmore, Skip }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccurateSubPelSearch { Use2Taps, Use4Taps, Use8Taps, Use8TapsSharp }
```

### 4.8 RefMvCandidates / RefFrame enums

```rust
// rvp9-core/src/geom/refframe.rs — exposed at crate root.

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(i8)]
pub enum RefFrameId {
    /// Sentinel — never appears as a "currently searched" ref.
    None    = -1,
    Intra   = 0,
    Last    = 1,
    Golden  = 2,
    Altref  = 3,    // Reachable only via VBR-RT lag-in-frames (D-13 RESOLVED);
                    // v1 path is altref-disabled.
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RefFrameFlags(pub u8);  // bit i set => ref i enabled

#[derive(Clone, Copy, Debug, Default)]
pub struct RefMvCandidates {
    pub nearest: Mv,
    pub near:    Mv,
}
```

Constants:

```rust
pub const MVREF_NEIGHBOURS: u8 = 8;     // vp9_mvref_common.h:24
pub const RT_INTER_MODES: u8 = 12;      // ref_mode_set length, vp9_pickmode.c:1237-1253
pub const RT_INTER_MODES_SVC: u8 = 8;
```

### 4.9 SearchSiteConfig

```rust
// rvp9-encoder/src/motion_est/config.rs

pub const SS_COUNT: usize = 8 * MAX_MVSEARCH_STEPS as usize;  // 88

#[derive(Clone, Copy, Debug)]
pub struct SearchSiteConfig {
    pub ss_mv: [Mv; SS_COUNT],
    /// Pre-baked byte offsets: ss_os[i] = ss_mv[i].row * stride + ss_mv[i].col.
    /// `vp9_mcomp.h:36-42`. Must be recomputed when reference-buffer stride
    /// changes (frame resize); the encoder frame-loop owns the lifecycle.
    pub ss_os:           [isize; SS_COUNT],
    pub searches_per_step: u32,    // 8 for NSTEP
    pub total_steps:       u32,    // SS_COUNT / searches_per_step = 11
}
```

We keep `ss_mv` and `ss_os` as parallel arrays per A3 §7 #1 — packing
into `Vec<(Mv, isize)>` makes the inner-loop `best_address + ss_os[i]`
arithmetic less direct.

### 4.10 Mode-counter context tables

These belong in `rvp9-core::geom` because the decoder needs them too,
but `motion_est` ports them by reference:

```rust
// rvp9-core/src/geom/refmv_tables.rs

/// `mv_ref_blocks[BLOCK_SIZES][MVREF_NEIGHBOURS]` — vp9_mvref_common.h:89-207.
/// Verbatim transcription; load-bearing because it's the same table the
/// decoder reads when parsing inter-mode bitstreams.
pub const MV_REF_BLOCKS: [[Position; MVREF_NEIGHBOURS as usize]; BLOCK_SIZES] = [
    // ... see vp9-analysis/03-motion-estimation.md §2.4
];

pub const IDX_N_COLUMN_TO_SUBBLOCK: [[u8; 2]; 4] =
    [[1, 2], [1, 3], [3, 2], [3, 3]];

pub const MODE_2_COUNTER: [u8; MB_MODE_COUNT] =
    [9,9,9,9,9,9,9,9,9,9, 0, 0, 3, 1];  // vp9_mvref_common.h:31-87

pub const COUNTER_TO_CONTEXT: [u8; 19] = [
    2, 1, 4, 3, 1, 9, 0, 9, 9, 5, 5, 9, 5, 9, 9, 9, 9, 9, 6,
    // BOTH_PREDICTED=2, NEW_PLUS_NON_INTRA=1, BOTH_NEW=4,
    // ZERO_PLUS_PREDICTED=3, ..., BOTH_INTRA=6 — see A3 §2.5
];
```

Numeric values are taken verbatim from `vp9_mvref_common.h:31-87`.

### 4.11 `ref_mode_set` (consumed here, owned by mode-decision)

L-55 — the 12-entry mode-iteration table is a `const` array of
`(RefFrameId, InterMode)` (`vp9_pickmode.c:1237-1253`). Owned by
LLD-mode-decision §4.x but cited here because ME's call order is
keyed off it.

---

## 5. Internal algorithm

This section walks the pipeline in execution order: from the
mode-decision call site down through full-pel → sub-pel → cost.

### 5.0 Top-level orchestration

`combined_motion_search` (`vp9_pickmode.c:154-261`). Flow:

1. **MV-limits setup** — `vp9_set_mv_search_range(&mv_limits, ref_mv)`
   (`vp9_mcomp.c:32-49`). Clamps to UMV window × `MAX_FULL_PEL_VAL`.
2. **Starting-MV selection** — pick whichever of
   `{nearest, near, pred_mv}` had the lowest SAD per `mv_pred`
   (`vp9_pickmode.c:193-200`):
   ```
   if mv_best_ref_index[ref] < 2:
     mvp_full = ref_mvs[ref][mv_best_ref_index[ref]]
   else:
     mvp_full = pred_mv[ref]
   mvp_full.col >>= 3; mvp_full.row >>= 3       // to full-pel
   ```
3. **Step-param adjustment** — `step_param` starts from
   `sf.mv.fullpel_search_step_param`; if
   `sf.mv.auto_mv_step_size` is on (off at rt; see §5.5 cpu6-8 row),
   `vp9_init_search_range` may bump it lower. In v1 this is a constant.
4. **Full-pel search** — dispatch by `sf.mv.search_method`. For cpu6
   that's `NSTEP`; for cpu7-8 that's `FAST_DIAMOND`. Outputs `tmp_mv`
   (integer-pel) and the variance.
5. **Sub-pel refinement** — multiply `tmp_mv` by 8, call
   `vp9_find_best_sub_pixel_tree`. Output: refined 1/8-pel MV plus
   `pred_sse[ref]`.
6. **Rate computation** — `vp9_mv_bit_cost(&mv, &ref_mv, ...)`
   (`vp9_mcomp.c:80-97`); result reported as `mv_rate_bits` to caller.

### 5.1 `vp9_full_pixel_search` dispatcher

`vp9_mcomp.c:2875-2952`. Per A3 §3.2:

```rust
pub fn full_pixel_search(
    block:          &MotionBlock<'_>,
    bsize:          BlockSize,
    mvp_full:       Mv,
    step_param:     i32,
    method:         SearchMethod,
    error_per_bit:  i32,
    ref_mv:         Mv,
    var_max:        i32,
    rd_flag:        bool,
    bfp:            &BlockFnPtrs,
    ss_cfg:         &SearchSiteConfig,
) -> FullPelSearchInternal {
    let mut cost_list = [i32::MAX; 5];   // L-07 — see §5.8 invariant
    let mut tmp_mv = Mv::default();
    let var = match method {
        SearchMethod::FastDiamond => fast_dia_search(...),
        SearchMethod::Nstep       => full_pixel_diamond(...),
        // NB: HEX, FAST_HEX, BIGDIA, SQUARE, MESH reachable from libvpx but not
        // from cpu-used 6-8 with default speed features (A3 §5.5). We
        // implement only the two we exercise; others fall through to a
        // debug_assert!(false).
        _ => unreachable_speed_feature("ME search_method"),
    };

    // libvpx does an exhaustive_search bump-up here when var > threshold;
    // dead code at our config because sf->exhaustive_searches_thresh = INT_MAX
    // (vp9_speed_features.c:462 rt path).

    if method != SearchMethod::Nstep && rd_flag && var < var_max as u32 {
        // Re-run variance to get the canonical mvpred_var.
        return vp9_get_mvpred_var(...);
    }
    FullPelSearchInternal { tmp_mv, var, cost_list }
}
```

The `cost_list = [INT_MAX; 5]` initialisation at the top
(`vp9_mcomp.c:2885-2891`) is the L-07 invariant: the sub-pel-pruned
variants fall back to FIRST_LEVEL_CHECKS when `cost_list[0] == INT_MAX`.
We always populate to `INT_MAX` first, then `calc_int_cost_list` or
`vp9_pattern_search_sad` may fill it.

### 5.2 `full_pixel_diamond` (NSTEP wrapper)

`vp9_mcomp.c:2486-2605`. Per A3 §4.2. Signature:

```rust
fn full_pixel_diamond(
    block:           &MotionBlock<'_>,
    bsize:           BlockSize,
    mvp_full:        Mv,
    step_param:      i32,
    error_per_bit:   i32,
    further_steps:   i32,           // MAX_MVSEARCH_STEPS-1-step_param
    do_refine:       bool,          // always true at our config
    use_downsampled: bool,          // sf.mv.use_downsampled_sad
    bfp:             &BlockFnPtrs,
    ref_mv:          Mv,
    cost_list:       &mut [i32; 5],
    ss_cfg:          &SearchSiteConfig,
) -> FullPelSearchInternal;
```

Algorithm shape:

1. Clamp `mvp_full` to `mv_limits`.
2. Choose SAD pair: full-rate (`sdf`, `sdx4df`) or downsampled
   (`sdsf`, `sdsx4df`) per `use_downsampled` plus the even-vs-odd-rows
   discrepancy test at `vp9_mcomp.c:2517-2528`.
3. Compute `start_mv_sad` via `sdsf` on the seed MV.
4. Call `diamond_search_sad` once with `step_param`; re-evaluate the
   result with `vp9_get_mvpred_var`.
5. For `n in 1..=further_steps`: if `num00 > 0` decrement and skip;
   else call `diamond_search_sad(step_param + n)` and update best.
6. If `do_refine`: call `refining_search_sad(search_range=8)`.
7. **L-07** Populate `cost_list[5]` via `calc_int_cost_list`. The
   array was zero-initialised to `INT_MAX` at the
   `vp9_full_pixel_search` dispatcher level (§5.1).
8. **Recursion guard (A3 §7 #3)** — `vp9_mcomp.c:2589-2598`: if the
   downsampled-SAD path was used and the chosen MV's full SAD
   disagrees with its downsampled SAD by ≥10%, return a tail-call
   into `full_pixel_diamond` with `use_downsampled=false`. Statically
   one-deep (the recursive call passes `false`, so cannot recurse
   again).

### 5.3 `vp9_diamond_search_sad_c` (8-site NSTEP)

`vp9_mcomp.c:2055-2190`. Per A3 §4.1. Signature:

```rust
fn diamond_search_sad(
    block:        &MotionBlock<'_>,
    cfg:          &SearchSiteConfig,
    mvp_full:     Mv,                       // pre-clamped
    start_mv_sad: u32,
    search_param: i32,                      // index of starting step
    sad_per_bit:  i32,
    sad_fn:       &(SadFn, SadX4dFn),
    center_mv:    Mv,
) -> (Mv, u32, i32 /* num00 */);
```

Algorithm in shape (verbatim source: `vp9_mcomp.c:2055-2190`):

1. `fcenter_mv = center_mv >> 3` (round-down to full-pel for cost).
2. `best_mv = clamp(mvp_full, mv_limits)`,
   `best_address = pre.offset(best_mv)`, `bestsad = start_mv_sad`,
   `last_site = -1`, `num00 = 0`.
3. For each of `total_steps - search_param` steps:
   - `i = search_param * 8`. Test `all_in` against `ss_mv[i..i+4]`
     (the 4 axis-aligned sites — they reach the furthest, so if all 4
     stay inside `mv_limits` then so do the diagonals).
   - **`all_in` fast path:** issue 2 batches of 4-way SAD via
     `sad_fn.1` (= `sdx4df`). For each of the 4 returned sads, first
     compare against `bestsad`; only on improvement, compute
     `mvsad_err_cost` and re-compare. Two-step pattern preserved.
   - **fallback:** per-site `is_mv_in` check + scalar `sad_fn.0`
     (= `sdf`); same two-step CHECK_BETTER.
   - If `best_site != -1 && best_site != last_site`:
     `best_mv += ss_mv[best_site]`,
     `best_address = best_address.offset(ss_os[best_site])`,
     `last_site = best_site`.
   - elif `best_address == initial_address` (center stayed best from
     the very start): `num00 += 1`.
4. Return `(best_mv, bestsad, num00)`.

Argument-order invariant: `mvsad_err_cost(this_mv, fcenter_mv,
sad_per_bit)` takes `diff = this_mv - fcenter_mv` per
`vp9_encoder.h:1523-1525`; we mirror exactly.

Safety: every `block.pre.offset(...)` and `best_address.offset(...)`
inside the `all_in` branch is justified by the bound check at the top
of the iteration. Inside the fallback branch every offset is preceded
by `is_mv_in`. Both invariants captured in a `// SAFETY:` block at
the `unsafe` site.

### 5.4 `vp9_refining_search_sad`

`vp9_mcomp.c:2749-2821`. 4-neighbour single-step refinement; called
after `full_pixel_diamond` with `search_range=8`. Algorithm exactly
matches A3 §4.6; same `all_in` SADx4D fast-path / scalar fallback
pattern.

### 5.5 `vp9_pattern_search` family (FAST_DIAMOND)

`vp9_mcomp.c:1007-1452`. For our scope only the **FAST_DIAMOND** entry
matters; it dispatches through `fast_dia_search` →
`bigdia_search` → `vp9_pattern_search_sad`. Algorithm A3 §4.3.

Key point: `fast_dia_search` (`vp9_mcomp.c:1614-1631`) caps the search
param so the search starts at the inner 2 scales only (scale ≤ 1,
i.e. `len ≤ 2`). At cpu-used 7-8 with
`fullpel_search_step_param = 10` (A3 §5.5), the search starts at
`len = 1` — the very smallest scale. That makes FAST_DIAMOND a tiny
search: only the 1-away ring plus a 2-away refinement.

`vp9_pattern_search_sad` is what populates `cost_list[5]` directly
(`vp9_mcomp.c:1168`). For cpu6 (which uses NSTEP) the cost list comes
from `calc_int_cost_list` instead (called inside `full_pixel_diamond`
at `vp9_mcomp.c:2602`). Both populate the same 5-slot array — see §5.8.

The `CHECK_BETTER` two-step pattern (`vp9_mcomp.c:943-953`) — first
compare raw SAD against `bestsad`, then re-compare with MV cost added —
is byte-load-bearing. We preserve the two-step:

```rust
#[inline]
fn check_better(thissad: u32, best: &mut Bestsad, this_mv: Mv,
                fcenter: Mv, sadpb: i32, costs: &MeCostsRef, idx: i32)
{
    if thissad < best.sad {
        let cost = mvsad_err_cost(this_mv, fcenter, costs, sadpb);
        let augmented = thissad + cost;
        if augmented < best.sad {
            best.sad = augmented;
            best.site = idx;
        }
    }
}
```

### 5.6 `vp9_int_pro_motion_estimation`

`vp9_mcomp.c:2264-2400`. A3 §4.7. Used for golden/altref NEWMV in CBR
when `bsize >= BLOCK_16X16` and `gf_temporal_ref` (per A3 §3.1.3).

Algorithm:

1. Compute 1-D horizontal projection of reference into `hbuf[search_width]`
   in 16-pixel chunks via `vpx_int_pro_row` (HLD-simd §3.23
   `IntProRowFn`).
2. Compute 1-D vertical projection into `vbuf[search_height]` one column
   at a time via `vpx_int_pro_col` (`IntProColFn`).
3. Same for source.
4. 1-D log-search:
   - `tmp_mv.col = vector_match(hbuf, src_hbuf, bwl)`
   - `tmp_mv.row = vector_match(vbuf, src_vbuf, bhl)`
5. Refine: SAD-x4D at the 4 neighbours; possibly try one diagonal.
6. Multiply MV by 8, clamp to subpel_mv_limits, return.

Early-exit per `vp9_pickmode.c:1598-1599`:

```text
if tmp_sad > x->pred_mv_sad[LAST_FRAME] || tmp_sad + bsize_factor > best_pred_sad:
    return None   // caller skips NEWMV on golden
```

The Rust `int_pro_motion_estimation` returns `Option<IntProResult>`;
caller branches accordingly.

### 5.7 `vp9_find_best_sub_pixel_tree` (sub-pel refinement)

`vp9_mcomp.c:721-925`. A3 §4.4 (first subsection). The default-and-only
variant for cpu-used 6-8 v1.

Algorithm shape:

1. **Seed:** `besterr = vf(pre + offset(bestmv), src) + mv_err_cost(bestmv, ref_mv)`.
2. **Round count:** `round = 3 - forced_stop` (3 for `EighthPel`, 2
   for `QuarterPel`, etc.). Clamp `round = min(round, 2)` if
   `!(allow_hp && use_mv_hp(ref_mv))`.
3. **Subpel state:** `br = bestmv.row * 8`, `bc = bestmv.col * 8`,
   `hstep = 4`, `search_step = &SEARCH_STEP_TABLE[0]`.
4. **For iter in 0..round:**
   - For idx in 0..4 (cardinal: L/R/U/D from `search_step[0..4]`):
     compute `(tr, tc)` candidate; if in subpel bounds, call
     `svf(pre + offset, x_phase(tc), y_phase(tr), src)` (the fused
     bilinear sub-pel-variance kernel — `use_accurate_subpel_search ==
     USE_2_TAPS`). Augment with `mv_err_cost`, write
     `cost_array[idx]`, update `best_idx`.
   - **Diagonal:** infer quadrant from cardinals (`kc = (cost[0] <=
     cost[1]) ? -hstep : +hstep`; `kr = (cost[2] <= cost[3]) ? -hstep
     : +hstep`). One diagonal sample; update `best_idx` to 4 if
     better.
   - **Commit:** `br/bc += search_step[best_idx]` if `best_idx < 4`;
     else `br = tr, bc = tc`.
   - **Optional SECOND_LEVEL_CHECKS:** if `iters_per_step > 0 &&
     best_idx != -1`, run the extra adjacent-site checks from
     `vp9_mcomp.c:202-248`.
   - **Descend:** `search_step += 4`, `hstep >>= 1`, `best_idx = -1`.
5. Return `(Mv { br as i16, bc as i16 }, besterr, sse)`.

Total subpel-variance samples per iter: 5 (4 cardinal + 1 diagonal)
plus 1-3 from SECOND_LEVEL_CHECKS. Round=2 (half + quarter-pel) yields
≤10 svf calls per ME.

`SEARCH_STEP_TABLE` is the 12-entry const from `vp9_mcomp.c:599-606`,
A3 §5.2:

```rust
const SEARCH_STEP_TABLE: [Mv; 12] = [
    Mv {row:  0, col: -4}, Mv {row:  0, col:  4},
    Mv {row: -4, col:  0}, Mv {row:  4, col:  0},
    Mv {row:  0, col: -2}, Mv {row:  0, col:  2},
    Mv {row: -2, col:  0}, Mv {row:  2, col:  0},
    Mv {row:  0, col: -1}, Mv {row:  0, col:  1},
    Mv {row: -1, col:  0}, Mv {row:  1, col:  0},
];
```

`use_accurate_subpel_search == USE_2_TAPS` (v1's value) takes the
`vfp->svf` path; the `accurate_sub_pel_search` branch is stubbed per
L-05 (see §7). The `BILINEAR / FOURTAP / EIGHTTAP / EIGHTTAP_SHARP`
filter dispatch inside `accurate_sub_pel_search` (`vp9_mcomp.c:756-763`)
is dead at our config.

`forced_stop = 3 - round` semantics — `EIGHTH_PEL=0` → round=3,
`HALF_PEL=2` → round=1. The `allow_hp` clamp is the `use_mv_hp(ref_mv)`
test from §4.1.

### 5.8 `calc_int_cost_list` and the 5-entry cost list (L-07)

`vp9_mcomp.c:960-1000`. Populates `cost_list[5]` after full-pel
search per the layout in A3 §5.7:

```
cost_list[0] = center (best_mv)
cost_list[1] = left     (row,  col-1)
cost_list[2] = bottom   (row+1, col)
cost_list[3] = right    (row,  col+1)
cost_list[4] = top      (row-1, col)
```

All are **variance + MV-cost** (not SAD). `cost_list[0]` uses
`mvsad_err_cost`; `cost_list[1..4]` use `mv_err_cost` (the variance-domain
cost). This inconsistency is preserved exactly (libvpx does it at
`vp9_mcomp.c:977-998`).

If any of the 4 neighbours falls outside `mv_limits`, its cost slot is
set to `INT_MAX` (`vp9_mcomp.c:987-991`). The pruned sub-pel variants
test for this and fall back to FIRST_LEVEL_CHECKS.

**L-07 invariant** (Rust phrasing):

```text
After full_pixel_search returns:
  cost_list[0] is either INT_MAX (no full-pel search ran -- shouldn't
                                  happen in v1) or the variance-plus-cost
                                  at the chosen full-pel MV.
  cost_list[1..4] are either INT_MAX (neighbour out of bounds) or the
                  variance-plus-cost at the 4-connected neighbours.

Callers MUST initialise the array to [INT_MAX; 5] before passing it in
(this matches vp9_full_pixel_search:2885-2891).

SUBPEL_TREE (v1's only variant) does NOT consult cost_list, but we
populate it anyway so that:
  - the pruned variants remain trivially re-enableable
  - the differential test suite (§9) can verify the cost-list layout
    against libvpx's output
```

### 5.9 `vp9_find_mv_refs` / `mv_refs_rt` / `vp9_find_best_ref_mvs`

`mv_refs_rt` (`vp9_pickmode.c:59-152`) is the real-time fast variant.
Differences from `vp9_find_mv_refs` (A3 §4.9):

- Only `candidate_mi->ref_frame[0] == ref_frame` matches considered
  (skip the `ref_frame[1]` second-ref check).
- Loop after first 2 neighbours **exits at first match**.
- Adds SVC base-layer-MV fast path: for non-keyframe + `ref == LAST`,
  pulls co-located MV from lower spatial layer (scaled ×2).
- Returns `const_motion` flag — 1 if the first nearest match was within
  the first 2 neighbours; used by mode-decision to skip NEARMV.

`vp9_find_best_ref_mvs` (`vp9_mvref_common.c:141-151`) post-processes
the candidate list:

1. `lower_mv_precision` — round to even if `!allow_hp`.
2. `clamp_mv2` — clamp to `[mb_to_left_edge - MV_BORDER,
   mb_to_right_edge + MV_BORDER]` etc., where `MV_BORDER = 128` (1/8-pel
   units, `vp9_mvref_common.h:214`).

Output: `RefMvCandidates { nearest, near }` in 1/8-pel units.

### 5.10 `vp9_mv_pred` — best-of-3 starting-MV selection

`vp9_rd.c:588-639`. A3 §4.10. Picks the best of
`{ref_mvs[ref][0], ref_mvs[ref][1], pred_mv[ref]}` by raw SAD (no MV
cost). Side effects on `MotionBlock`:

- `mv_best_ref_index[ref] = best_index` (0 / 1 / 2).
- `pred_mv_sad[ref] = best_sad` — drives `reference_masking` (§5.14).
- `max_mv_context[ref] = max(max_abs_components >> 3)`.

The "3rd candidate" gate (`block_size < x->max_partition_size`,
`vp9_rd.c:599-600`) is active for every block size except the full
superblock under the var-based partition path (A3 §7 #12); the LLD
exposes a `use_third_candidate` boolean derived in the caller, but the
default is `bsize < BLOCK_64X64`.

The deduplication tests (`near_same_nearest`, `zero_seen`) preserve
libvpx's behaviour exactly (A3 §4.10, `vp9_rd.c:614-623`).

### 5.11 MV-limit setup

```rust
/// `vp9_set_mv_search_range` (`vp9_mcomp.c:32-49`). Run once per block
/// before any full-pel call. Clamps to UMV window × MAX_FULL_PEL_VAL.
pub fn set_mv_search_range(limits: &mut MvLimits, mv: Mv);

/// `vp9_set_subpel_mv_search_range` (`vp9_mcomp.c:51-67`). Run once
/// before sub-pel; result is always in [MV_LOW+1, MV_UPP-1].
pub fn set_subpel_mv_search_range(
    sub_limits:    &mut MvLimits,
    full_limits:   &MvLimits,
    ref_mv:        Mv,
);
```

Pseudocode mirrors `vp9_mcomp.c` byte-for-byte. The `<< 3` (multiplied
by 8) factor for the subpel range is preserved.

### 5.12 `search_site_config` initialisation

`vp9_init3smotion_compensation` (`vp9_mcomp.c:116-134`). Built once per
encoder lifetime, recomputed when `init_motion_estimation` runs at
resolution change (Step 24 in 00-overview §2; A3 §2.6):

```rust
impl SearchSiteConfig {
    pub fn nstep_8point(stride: i32) -> Self {
        let mut ss_mv  = [Mv::default(); SS_COUNT];
        let mut ss_os  = [0_isize; SS_COUNT];
        let mut idx = 0_usize;
        let mut len  = MAX_FIRST_STEP;
        while len > 0 {
            for &(dr, dc) in &[
                (-len,   0), ( len,    0),
                (   0, -len), (   0,  len),
                (-len,-len), (-len,  len),
                ( len,-len), ( len,  len),
            ] {
                let m = Mv { row: dr as i16, col: dc as i16 };
                ss_mv[idx] = m;
                ss_os[idx] = (m.row as isize) * (stride as isize) + (m.col as isize);
                idx += 1;
            }
            len /= 2;
        }
        SearchSiteConfig {
            ss_mv, ss_os,
            searches_per_step: 8,
            total_steps: (idx as u32) / 8,
        }
    }
}
```

The 4-site 2-step variant (`vp9_init_dsmotion_compensation`) is not
referenced in v1 — rt uses 8-site NSTEP only (A3 §2.6).

### 5.13 MV-cost arithmetic

Three flavours, all preserved per A3 §4.11:

```rust
#[inline]
fn mv_cost(diff: Mv, joint: &[i32; 4], comp_cost: [&[i32; MV_VALS as usize]; 2]) -> i32 {
    debug_assert!((diff.row as i32) >= -MV_MAX && (diff.row as i32) < MV_MAX);
    debug_assert!((diff.col as i32) >= -MV_MAX && (diff.col as i32) < MV_MAX);
    joint[get_mv_joint(diff) as usize]
        + comp_cost[0][diff.row as usize]   // offset-by-MV_MAX view
        + comp_cost[1][diff.col as usize]
}

#[inline]
fn mvsad_err_cost(mv: Mv, ref_mv: Mv, costs: &MeCostsRef, sad_per_bit: i32) -> u32 {
    let diff = Mv { row: mv.row - ref_mv.row, col: mv.col - ref_mv.col };
    round_power_of_two_u(
        (mv_cost(diff, costs.nmvjointsadcost, costs.mvsadcost) as u32) * (sad_per_bit as u32),
        VP9_PROB_COST_SHIFT,
    )
}

#[inline]
fn mv_err_cost(mv: Mv, ref_mv: Mv, costs: &MeCostsRef, error_per_bit: i32) -> i32 {
    let diff = Mv { row: mv.row - ref_mv.row, col: mv.col - ref_mv.col };
    let shift = RDDIV_BITS + VP9_PROB_COST_SHIFT - RD_EPB_SHIFT + PIXEL_TRANSFORM_ERROR_SCALE;
    round64_power_of_two(
        (mv_cost(diff, costs.nmvjointcost, costs.mvcost) as i64) * (error_per_bit as i64),
        shift,
    ) as i32
}

#[inline]
fn vp9_mv_bit_cost(mv: Mv, ref_mv: Mv, costs: &MeCostsRef, weight: i32) -> i32 {
    let diff = Mv { row: mv.row - ref_mv.row, col: mv.col - ref_mv.col };
    round_power_of_two(
        mv_cost(diff, costs.nmvjointcost, costs.mvcost) * weight,
        7,
    )
}
```

Constants (A3 §5.8):

```rust
pub const VP9_PROB_COST_SHIFT: u32 = 9;       // vp9_cost.h:24
pub const PIXEL_TRANSFORM_ERROR_SCALE: u32 = 4; // vp9_mcomp.c:86
pub const RDDIV_BITS: u32 = 7;                  // vp9/encoder/vp9_rd.h
pub const RD_EPB_SHIFT: u32 = 6;                // vp9/encoder/vp9_rd.h
pub const MV_COST_WEIGHT: i32 = 108;            // vp9_rd.h:38
pub const MV_COST_WEIGHT_SUB: i32 = 120;        // vp9_rd.h:39
```

The exact integer arithmetic — `round_power_of_two` vs
`round64_power_of_two`, `i64` intermediate widths — is byte-load-bearing
per D7 (mirror libvpx integer widths exactly). These three helpers live
in `rvp9-core::util` shared with the rest of the encoder; we don't fork
them here.

### 5.14 `reference_masking` (L-06)

`vp9_pickmode.c:2194-2216`. Per A3 §5.6. Computes the skip mask for
non-ZERO modes based on `pred_mv_sad` ratios.

**Threshold: `pred_mv_sad[ref] > (pred_mv_sad[other] << 1)`** — i.e.
2× hard-coded. **Decision (L-06 RESOLVED):** we freeze at 2×. Rationale:

- The threshold is hot-path-critical; making it adaptive complicates
  the cost-model loop and gates byte parity behind a tuning knob.
- Downstream forks have tried adaptive variants (A3 §7 #7) without
  documented BD-rate wins.
- If a future preset wants different sensitivity, exposing a
  `f32` factor field on `MeSpeedFeatures` is cheap and additive — no
  shape change.

Algorithm:

```rust
pub fn reference_skip_mask(
    pred_mv_sad:              &[u32; MAX_REF_FRAMES as usize],
    ref_frame_flags:          RefFrameFlags,
    this_mv_is_zero_and_last: bool,    // = (this_mv == 0 && ref == LAST)
    usable_ref_frame:         RefFrameId,
    force_skip_low_temp_var:  bool,
    is_src_frame_alt_ref:     bool,
    this_ref:                 RefFrameId,
    this_mv_zero_and_altref:  bool,
) -> u8 {
    if this_mv_is_zero_and_last { return 0; }

    let mut mask: u8 = 0;
    if (usable_ref_frame as i8) < (RefFrameId::Altref as i8) {
        // Below-altref masking: compare against the "other" ref.
        if !force_skip_low_temp_var && (usable_ref_frame as i8) > (RefFrameId::Last as i8) {
            let other = if this_ref == RefFrameId::Last { RefFrameId::Golden }
                        else { RefFrameId::Last };
            if ref_frame_flags.contains(other) {
                if pred_mv_sad[this_ref as usize]
                    > pred_mv_sad[other as usize] << 1
                {
                    mask |= 1 << (this_ref as u8);
                }
            }
        }
    } else if !is_src_frame_alt_ref && !this_mv_zero_and_altref {
        let r1 = if this_ref == RefFrameId::Golden { RefFrameId::Last } else { RefFrameId::Golden };
        let r2 = if this_ref == RefFrameId::Altref { RefFrameId::Last } else { RefFrameId::Altref };
        let p = pred_mv_sad[this_ref as usize];
        if p > pred_mv_sad[r1 as usize] << 1
           || p > pred_mv_sad[r2 as usize] << 1
        {
            mask |= 1 << (this_ref as u8);
        }
    }
    mask
}
```

The `<< 1` shifts are unsigned (`u32` operands). No saturation needed
because `pred_mv_sad` values are bounded by the block area × 255, well
below `u32::MAX >> 1`.

### 5.15 `vp9_NEWMV_diff_bias` (L-43)

`vp9_pickmode.c:1298-1355`. A2 §4.8. Applied after `model_rd_for_sb_y[_large]`
but **before** `block_yrd`, and the bias modifies `this_rdc.rdcost`.
Call site at `vp9_pickmode.c:1029-1031` (the consumer in
`vp9_pick_inter_mode`).

```rust
/// `vp9_NEWMV_diff_bias`. Modifies `rdcost` to penalise NEWMV vectors
/// that diverge from above/left neighbour MVs, and to favour
/// near-zero LAST_FRAME MV on noisy content.
///
/// Ordering invariant (L-43): MUST run AFTER `encode_breakout_test`
/// (which only reads `var, sse_y` from the model) but BEFORE
/// `block_yrd` consumes `this_rdc.rdcost` for the
/// "is-this-mode-best?" comparison. Both invariants are preserved by
/// the call-site order in LLD-mode-decision.
pub fn newmv_diff_bias(
    rdcost:           &mut i64,
    this_mode:        InterMode,
    mv:               Mv,                    // 1/8-pel units
    above_mi:         Option<&ModeInfo>,
    left_mi:          Option<&ModeInfo>,
    bsize:            BlockSize,
    ref_frame:        RefFrameId,
    noise_level:      NoiseLevel,
    noise_enabled:    bool,
    lowvar_highsumdiff: bool,
    is_skin:          bool,
);
```

Algorithm verbatim (A2 §4.8, byte-parity required per D8):

```text
if this_mode == NEWMV:
    av = average_of_valid_neighbour_mvs(above_mi, left_mi)   // (0,0) if none
    row_diff = av.row - mv.row
    col_diff = av.col - mv.col
    if |row_diff| > 48 || |col_diff| > 48:
        if bsize > BLOCK_32X32: rdcost <<= 1
        else:                   rdcost = 3 * rdcost >> 1     // *1.5

if noise_enabled && noise_level >= Medium && bsize >= BLOCK_32X32
   && ref_frame == LAST && |mv.row| < 8 && |mv.col| < 8:
    rdcost = 7 * (rdcost >> 3)                                // *0.875
elif lowvar_highsumdiff && !is_skin && bsize >= BLOCK_16X16
     && ref_frame == LAST && |mv.row| < 16 && |mv.col| < 16:
    rdcost = 7 * (rdcost >> 3)
```

Magic numbers `48 / 8 / 16` are hard-coded (A2 §7 Q6). The 1/8-pel
threshold `< 8` means "within 1 integer pel"; `< 16` means "within 2
integer pels". The `48` is in 1/8-pel = 6 integer pels of divergence
from neighbour average.

### 5.16 Speed-feature constants (cpu-used 6/7/8)

Per A3 §5.5. Locked values for v1:

| Field | cpu6 | cpu7 | cpu8 |
|---|---|---|---|
| `search_method` | `Nstep` | `FastDiamond` | `FastDiamond` |
| `fullpel_search_step_param` | 6 | 10 | 10 |
| `subpel_search_method` | `Tree` | `Tree` | `Tree` |
| `subpel_force_stop` | `EighthPel` | `EighthPel` | `EighthPel` (`HalfPel` for screen-content tune — out of v1) |
| `subpel_search_level` | 1 | 1 | 1 |
| `use_downsampled_sad` | true | true | true |
| `reference_masking` | true | true | true |
| `limit_newmv_early_exit` | true | false (SVC>0) | false |
| `bias_golden` | true | true | true |
| `use_accurate_subpel_search` | `Use2Taps` | `Use2Taps` | `Use2Taps` |

`enable_adaptive_subpel_force_stop` (A3 §5.9) is a speed-9 feature —
out of v1 scope.

The SVC override at speed 7 (`subpel_search_method =
TreePrunedEvenmore` for non-reference SVC frames, `vp9_speed_features.c:715`)
is gated off in v1 by L-05 (only `Tree` is implemented). When SVC
non-ref frames are encoded at cpu-used 7, we use `Tree` and accept the
BD-rate delta. This is a sub-divergence: SVC LLD will document it
separately. We log it here in §7.

---

## 6. SIMD dispatch

This module **consumes** SIMD kernels owned by HLD-simd; it does not
add new SIMD-specific kernels of its own. The integration points:

### 6.1 Kernels consumed (HLD-simd §3.x)

| Kernel | HLD-simd § | Use site in §5 | Function signature |
|---|---|---|---|
| `vpx_sad{w}x{h}` | §3.1 | §5.3 `sdf` (scalar fallback for OOB sites) | `SadFn` |
| `vpx_sad_skip_{w}x{h}` | §3.2 | §5.2 `use_downsampled_sad` path (`sdsf`) | `SadFn` |
| `vpx_sad{w}x{h}x4d` | §3.4 | §5.3 `sdx4df` — **the hot kernel** | `SadX4dFn` |
| `vpx_sad_skip_{w}x{h}x4d` | §3.5 | §5.2 `sdsx4df` downsampled path | `SadX4dFn` |
| `vpx_variance{w}x{h}` | §3.6 | §5.7 `vf` sub-pel center error; §5.8 `calc_int_cost_list`; `vp9_get_mvpred_var` | `VarianceFn` |
| `vpx_sub_pixel_variance{w}x{h}` | §3.7 | §5.7 `svf` — every sub-pel iteration | `SubPixelVarianceFn` |
| `vpx_int_pro_row` | §3.23 | §5.6 step 1 | `IntProRowFn` |
| `vpx_int_pro_col` | §3.23 | §5.6 step 2 | `IntProColFn` |
| `vpx_vector_var` | §3.23 | §5.6 step 4 | `VectorVarFn` |

Kernels we do **not** consume in the non-RD path (still allocated in
the BFP slot for symmetry):

| Kernel | HLD-simd § | Why not consumed |
|---|---|---|
| `vpx_sad{w}x{h}_avg` (`sdaf`) | §3.3 | `vp9_refining_search_8p_c` only — L-08 stubbed |
| `vpx_sub_pixel_avg_variance{w}x{h}` (`svaf`) | §3.8 | Compound NEWMV sub-pel — L-08 stubbed |
| `vpx_comp_avg_pred` | §3.10 | `setup_center_error` compound branch — dead at our config |

### 6.2 ME-specific VP9 RTCD entry

`vp9_diamond_search_sad` (`vp9/common/vp9_rtcd_defs.pl:176-177`) — the
single VP9-specific ME kernel that has its own RTCD slot. HLD-simd
§3.23 commits to **scalar-only in v1**:

> we ship **scalar only** in v1 (divergence ledger L-RTCD-03 — relies
> on AVX2 `sdx4df` for the SAD batches; NEON hand-tuned variant from
> libvpx omitted because the gain over the scalar+sdx4df-neon
> combination is small. Re-evaluate in Stage 5).

This LLD respects that decision: `diamond_search_sad` (§5.3) is a
plain Rust function compiled to native code; it calls into the SIMD
kernels via `BlockFnPtrs.sdx4df` and `.sdf` only.

The NEON-specific `vp9_diamond_search_sad_neon`
(`vp9/encoder/arm/neon/vp9_diamond_search_sad_neon.c`) is NOT ported
in v1. We accept the perf delta — Stage 5 to re-evaluate.

### 6.3 Test strategy

This LLD owns no SIMD kernels; differential tests for the consumed
kernels are owned by HLD-simd §3.x. ME-specific differential testing
lives in §9 of this LLD.

---

## 7. Deliberate divergences from libvpx

Each entry below feeds `notes/divergence-ledger.md` with the canonical
format (the entries marked L-05 / L-06 / L-08 are the mandatory ones
called out by Arthur's dispatch prompt). Byte-parity impact is "none
in matched configs" unless otherwise stated.

### 7.1 `vp9_refining_search_8p_c` not ported (compound NEWMV) — L-08

**Where:** `rvp9-encoder::motion_est::search` — no compound-pred joint
refinement. **libvpx:** `vp9_refining_search_8p_c`
(`vp9/encoder/vp9_mcomp.c:2825-2873`).
**Libvpx behaviour:** iterative 8-point joint refinement for compound
NEWMV during RD-loop mode decision (good/best presets).
**Rvp9 behaviour:** function does not exist.
**Why:** rt non-RD path restricts compound to ZEROMV pairs
(`vp9_pickmode.c:2064-2070`) — compound NEWMV is dead code at our
config. **Future:** re-port if compound NEWMV becomes reachable.
Decision: **L-08 RESOLVED**.

### 7.2 `accurate_sub_pel_search` stubbed (only USE_2_TAPS) — L-05

**Where:** `rvp9-encoder::motion_est::subpel`. **libvpx:**
`accurate_sub_pel_search` branch in `vp9_mcomp.c:629-720`, dispatched
from `vp9_mcomp.c:756-763` and parallel sites in all four tree
variants.
**Libvpx behaviour:** when `use_accurate_subpel_search ≠ USE_2_TAPS`,
builds an 8-tap (or 4-tap, or sharp 8-tap) filtered sub-pel prediction
and computes variance via convolution + `vf` rather than the fused
`svf` bilinear kernel.
**Rvp9 behaviour:** only the fused `svf` path. The
`AccurateSubPelSearch` enum exists for surface parity; non-`Use2Taps`
variants `debug_assert!(false, "L-05 stubbed")` and silently fall
through to `Use2Taps` in release.
**Why:** `USE_2_TAPS` is the unconditional value at cpu-used ≥ 4
(`vp9_speed_features.c:928` + rt cascade never overrides). Accurate
variants are reachable only from good/best presets (cpu-used 0-3,
out of D1 scope). Decision: **L-05 RESOLVED**.

### 7.3 `vp9_find_best_sub_pixel_tree_pruned[_more,_evenmore]` stubbed — L-05

**Where:** `rvp9-encoder::motion_est::subpel`. **libvpx:**
`_pruned` (`vp9_mcomp.c:515-597`), `_pruned_more`
(`vp9_mcomp.c:453-513`), `_pruned_evenmore` (`vp9_mcomp.c:387-451`).
**Libvpx behaviour:** dispatches to a pruned tree based on
`sf.mv.subpel_search_method`; pruned variants use `cost_list[5]`
(§5.8) to skip half-pel samples.
**Rvp9 behaviour:** dispatcher always picks `SUBPEL_TREE`. `cost_list`
still populated (§5.8) so dead variants are re-enableable.
**Why:** at cpu-used 6-8 the rt setter selects `SUBPEL_TREE` for
non-SVC frames; `SUBPEL_TREE_PRUNED_EVENMORE` only fires on SVC
non-reference frames at cpu7 (`vp9_speed_features.c:715`).
**Byte-parity impact:** SVC cpu-used 7 non-reference frames diverge
from libvpx at the sub-pel step.
**Compression-efficiency impact:** estimated ≤0.1% BD-rate hit on SVC
cpu7 non-ref; insignificant on 720p30 / 1080p30 matched-rate. To be
confirmed in Stage 4d.
**Future:** port `_pruned_evenmore` in Stage 4d if BD-rate gap > 0.3%
on the SVC corpus. Decision: **L-05 sub-clause**.

### 7.4 `vp9_skip_sub_pixel_tree` stubbed — L-05

**Where:** `rvp9-encoder::motion_est::subpel`. **libvpx:**
`vp9_mcomp.c:354-385`.
**Libvpx behaviour:** when `subpel_force_stop == FULL_PEL`, skip
sub-pel and emit integer-pel MV.
**Rvp9 behaviour:** `subpel_force_stop = FullPel` triggers
`unreachable_speed_feature`; debug-panic, release no-op.
**Why:** `FULL_PEL` is selected only at cpu-used 8 with screen-content
tune (`vp9_speed_features.c:769`); screen-content tune is out of v1
scope. **Future:** wire when screen-content tune is enabled.
Decision: **L-05 sub-clause**.

### 7.5 Exhaustive mesh search stubbed

**Where:** no `exhaustive_mesh_search` or `full_pixel_exhaustive`.
**libvpx:** `vp9_mcomp.c:1637-1721, 2609-2662`. Dead at our config
because `sf.exhaustive_searches_thresh = INT_MAX` in rt
(`vp9_speed_features.c:462`); the dispatcher condition for
`run_exhaustive_search` (`vp9_mcomp.c:2924-2935`) never fires.
**Future:** re-port if a quality preset enables it. No L-number — the
rt setter makes it unreachable.

### 7.6 `vp9_diamond_search_sad` NEON specialisation dropped — L-RTCD-03 (HLD-simd)

**Where:** `rvp9-encoder::motion_est::search::diamond_search_sad` is
scalar Rust on every arch. **libvpx:**
`vp9/encoder/arm/neon/vp9_diamond_search_sad_neon.c`.
**Libvpx behaviour:** hand-tuned NEON re-implementation fusing the
cost-vs-bound check with the SAD batch.
**Rvp9 behaviour:** plain Rust driver that calls into the NEON
`sdx4df` kernel.
**Why:** HLD-simd §3.23 — performance gap is small relative to the
maintenance cost of a parallel ME driver. Stage 5 to re-evaluate.

### 7.7 `vp9_init_dsmotion_compensation` (4-site init) not ported

**Where:** `SearchSiteConfig` exposes only `nstep_8point`. **libvpx:**
`vp9_mcomp.c:98-114`. Dead at rt — `cpi->ss_cfg` is fed by
`vp9_init3smotion_compensation` (8-site) exclusively.
**Future:** add if cpu-used ≤ 3 preset is added. No L-number.

### 7.8 `reference_masking` threshold frozen at 2× — L-06

**Where:** `rvp9-encoder::motion_est::reference_mask`. **libvpx:**
`vp9/encoder/vp9_pickmode.c:2194-2216`.
**Libvpx behaviour:** hard-codes `pred_mv_sad[ref] > pred_mv_sad[other]
<< 1` (2×).
**Rvp9 behaviour:** identical 2× threshold; not exposed as tunable.
**Why:** no documented BD-rate improvement from adaptive variants
(A3 §7 #7). Keeping it constant simplifies the cost-model loop.
**Byte-parity impact:** none. **Future:** Stage 5 may expose
`MeSpeedFeatures::reference_mask_factor: u8` (default 2) as an
additive change. Decision: **L-06 RESOLVED**.

---

## 8. Open questions

### 8.1 `vp9_int_pro_motion_estimation` under VBR-RT

**Status:** open. Owner: rate-control LLD + this LLD.

Currently gated on `oxcf.rc_mode == VPX_CBR` per A3 §3.1.3
(`vp9_pickmode.c:1586-1592`). D-13 RESOLVED commits us to VBR-RT
in v1. Question: do we extend `int_pro_motion_estimation` to VBR-RT,
or keep it CBR-only?

**Recommendation:** keep CBR-only in v1 (mirror libvpx). VBR-RT
without altref is a small enough scope (PLAN.md §2) that adding the
int-pro fast path is a perf optimisation, not a correctness gate. If
Stage 4a perf benchmarks show int-pro-less VBR-RT misses the
1080p60 budget, re-evaluate.

This question is not blocking for Stage 3 (decoder); it surfaces in
Stage 4a (encoder MVP).

### 8.2 `cost_list[5]` population when SUBPEL_TREE is the only consumer

**Status:** decision needed during implementation. Owner: this LLD.

We always populate `cost_list[5]` (§5.8) but SUBPEL_TREE (the only
sub-pel variant we ship) ignores it. Performance: 4 extra variance
calls per ME with no consumer.

**Question:** is the cost-list construction worth keeping?

**Recommendation:** keep, for three reasons:
1. The pruned variants are an L-05 stub; re-enabling them later is
   single-line change. Without cost_list they'd silently degrade.
2. Differential tests (§9) verify the cost-list layout against
   libvpx-C output; dropping it removes a useful oracle.
3. The 4 extra variance calls are ~1% of ME wall-time on the
   reference Stage 0 benchmarks.

This is a documentation issue rather than a code-shape one. **No
blocker.**

### 8.3 `auto_mv_step_size` adaptive step (rt off by default)

**Status:** confirmed off, no implementation. Owner: this LLD.

`sf.mv.auto_mv_step_size = 0` for all rt paths (`vp9_speed_features.c:935`).
A3 §7 #11 notes the field exists but is dead at our config. We do not
implement `vp9_init_search_range` (the step-size adapter).

**No blocker** — we ship `auto_mv_step_size` as an enum unit type
that always means "off".

### 8.4 SVC non-ref cpu-used 7 sub-pel divergence

**Status:** accepted divergence (§7.3). Owner: SVC LLD (cross-link).

At cpu-used 7 with SVC non-reference frames, libvpx switches
`subpel_search_method` to `SUBPEL_TREE_PRUNED_EVENMORE`
(`vp9_speed_features.c:715`). We always use `SUBPEL_TREE`. The
SVC LLD must document the impact in its own divergence section and
in the SVC corpus BD-rate report; Stage 4d gates the decision to
port `_evenmore`.

**Blocker for Stage 4d sign-off** if BD-rate gap exceeds 0.3% on the
SVC corpus.

### 8.5 `pred_mv` 3rd-candidate condition (A3 §7 #12)

**Status:** parametrised. Owner: mode-decision LLD.

`vp9_mv_pred` uses the 3rd candidate `block.pred_mv[ref]` only if
`bsize < x->max_partition_size`. For the var-based partition path
`max_partition_size == BLOCK_64X64`, so the 3rd candidate is active
for every block size except full SB.

We expose `use_third_candidate: bool` on the call site, computed in
mode-decision from the partition state. **No blocker.**

---

## 9. Test strategy

### 9.1 Unit tests (in-module)

- `Mv::INVALID == { row: -32768, col: -32768 }` equals libvpx's
  `0x80008000` (`vp9_mv.h:22`).
- `set_mv_search_range` against a hand-table of 8 (mv, mv_limits)
  pairs covering corners and centre of the UMV window.
- `set_subpel_mv_search_range` invariants: output always in
  `[MV_LOW+1, MV_UPP-1]`.
- `nstep_8point(stride)` produces `ss_count == 88`,
  `total_steps == 11`, and the first step's 8 offsets are
  `(±1024, 0)`, `(0, ±1024)`, `(±1024, ±1024)`.
- `mvsad_err_cost(mv, mv, costs, sadpb) == 0` (diff is zero).
- `reference_skip_mask` truth-table for the 16 combinations of
  `(usable_ref_frame, this_ref, force_skip_low_temp_var,
  is_src_frame_alt_ref)` × hand-picked SAD ratios across/at-2× / below 2×.

### 9.2 Property tests (`proptest`)

- For random `Mv` × random `MvLimits` × `RefPlane` bounded to a fixed
  test buffer: `vp9_full_pixel_search(NSTEP)` returns an MV inside
  `mv_limits` regardless of input.
- For random buffer × random NEAREST/NEAR/pred_mv: `mv_pred`'s output
  `pred_mv_sad[ref]` ≤ each of the 3 candidates' SAD (i.e. it returns
  the actual minimum).
- For random buffer pairs: `find_best_sub_pixel_tree`'s output
  variance is ≤ the variance at the integer-pel input MV (sub-pel
  cannot get worse).
- For random `cost_list[5]`: the dummy `get_cost_surf_min` (§5.7 of
  A3) — even though we don't ship the pruned variants, we ship
  `cost_list` and the quadratic-fit helper; property: for parabolic
  cost surfaces it should return a value in `[-1, 1]` per dimension.

### 9.3 Differential tests vs libvpx-C

The byte-parity oracle. Per PLAN.md D8, byte parity is a diagnostic
tool, not a release gate — but for ME it is achievable and a strong
signal.

Per-function differential, against libvpx 1.14.1 (`12f3a2a`):

| rvp9 fn | libvpx fn | Comparison |
|---|---|---|
| `diamond_search_sad` | `vp9_diamond_search_sad_c` (force the C path; do not link the NEON specialisation) | Output `(best_mv, var, num00)` bit-equal |
| `full_pixel_diamond` | `full_pixel_diamond` | Output `(best_mv, var)` and final `cost_list[5]` bit-equal |
| `refining_search_sad` | `vp9_refining_search_sad` | Output `(best_mv, var)` bit-equal |
| `find_best_sub_pixel_tree` | `vp9_find_best_sub_pixel_tree` | Output `(best_mv, var, sse)` bit-equal |
| `int_pro_motion_estimation` | `vp9_int_pro_motion_estimation` | Output `(mv, sad)` bit-equal; also test the early-exit path |
| `mv_pred` | `vp9_mv_pred` | Output `mv_best_ref_index, pred_mv_sad, max_mv_context` bit-equal |
| `find_mv_refs_rt` | `mv_refs_rt` | Output `ref_mvs[ref][0..1], const_motion, mode_context[ref]` bit-equal |
| `find_best_ref_mvs` | `vp9_find_best_ref_mvs` | Output `nearest, near` bit-equal |
| `reference_skip_mask` | (inlined in `vp9_pickmode.c:2194-2216`) | Manually re-ported test fixture: for synthetic `pred_mv_sad` arrays, mask bit-equal |
| `newmv_diff_bias` | `vp9_NEWMV_diff_bias` | Output `rdcost` bit-equal |
| `mv_err_cost`, `mvsad_err_cost`, `vp9_mv_bit_cost` | identical | Output bit-equal across `MV_VALS²` input space (sampled) |

Test harness: `tests/differential/me/` invokes libvpx symbols via FFI
into a stripped-down build of `vp9/encoder/vp9_mcomp.o`. Inputs are
shared `[u8; ...]` buffers (Y plane only). The libvpx `MACROBLOCK`
construction is hidden behind a thin `me_harness_libvpx.c` shim that
populates only the fields ME reads.

### 9.4 Conformance vectors (cross-product)

Not applicable directly — conformance vectors are decoder-side. The
indirect signal is: every encoded bitstream from `rvp9-encoder`
decodes cleanly under `vpxdec`, and the per-frame MD5 matches libvpx
when ME and mode-decision are both byte-parity. Once Stage 4a is
running and the differential bitstream harness is operational
(`tests/differential/diff-bitstream.rs`), any ME divergence shows up
as a first-difference byte position; we use that as the regression
oracle.

### 9.5 Performance tests

- ME-only microbench: `bench/me_microbench.rs` (Stage 4a). Reports
  ms per ME-call for each `(bsize, search_method, sub_pel_variant)`
  combination. CI gates against ±5% regression.
- Whole-frame ME profile: 720p30 / 1080p30 / 1080p60 wall-time
  decomposition; ME should account for < 30% of frame budget at
  cpu-used 7 on the Stage 0 reference machine.

### 9.6 SIMD regression

This LLD owns no SIMD kernels (§6), but the consumed kernels' tests
in HLD-simd §3.x are gated for ME-specific edge cases:

- `vpx_sad{w}x{h}x4d`: the offset-alignment test (`refs[0]` aligned,
  `refs[1..3]` 1/2/3 byte-misaligned) is critical because
  `diamond_search_sad` builds the 4 refs via `best_address + ss_os[i]`
  which can land on any 4-byte alignment.
- `vpx_sub_pixel_variance{w}x{h}`: all 64 `(x_offset, y_offset)`
  combinations must be exercised because `find_best_sub_pixel_tree`
  samples them all over the course of its 3-iteration descent.

---

## Summary

**Line count:** ≈ 1780 lines. **Key §7 divergences (feeding divergence-ledger):**
L-05 (accurate sub-pel variants stubbed: the four pruned variants,
USE_2_TAPS-only, FULL_PEL stub, plus SVC cpu7 non-ref sub-clause),
L-08 (`vp9_refining_search_8p_c` compound NEWMV stubbed), L-06
(reference-masking threshold frozen at 2×), L-RTCD-03 cross-link
(`vp9_diamond_search_sad` NEON specialisation dropped per HLD-simd
§3.23), exhaustive-mesh stub, 4-site `init_dsmotion_compensation` not
ported. **§8 blocking questions:** none for Stage 3; §8.4 (SVC cpu7
non-ref sub-pel divergence) potentially blocks Stage 4d if SVC BD-rate
gap exceeds 0.3%.
