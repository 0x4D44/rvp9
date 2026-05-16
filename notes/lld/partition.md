# LLD: `partition` — variance-tree SB partition decision

**Status:** draft v0.1.0, Stage 2 LLD.
**Owner:** LLD-partition (Stage 2).
**Date:** 2026-05-16.
**Inputs read (in order):**
1. `PLAN.md` — locked scope: D-01 RESOLVED (8-bit only); D-05 RESOLVED
   (AQ mode 3 only); D-13 RESOLVED (CBR + VBR-RT, NO altref);
   R-06 RESOLVED (EXT_RATECTRL in v1).
2. `notes/vp9-analysis/_stage2-dispatch-prep.md` — LLD template,
   `### partition.md` focus paragraph.
3. `notes/vp9-analysis/02-encoder-core.md` — primary specialist A2.
   §2 (variance tree), §3.10 (flowchart), §4.1 (vbp thresholds), §4.7
   (`set_vt_partitioning`), §4.15–§4.16 (low-temp-var), §5.1 / §5.14
   (verbatim tables), §6 (SIMD). **A2 §2.9 was patched 2026-05-16 to
   clarify `variance_low[25]` indices [9..=24] are Morton/Z-order at
   the 2×2 sub-quadrant level.**
4. `notes/vp9-analysis/03-motion-estimation.md` — secondary, §4.7
   (`vp9_int_pro_motion_estimation`).
5. `notes/vp9-analysis/00-overview.md` — §3.5 (per-SB state), §4.7 /
   §4.9 (LLD-partition scope).
6. `notes/vp9-analysis/decisions-needed.md` — **L-38** (variance-tree
   rep), **L-39** (`variance_low[25]` layout), **L-44**
   (`set_low_temp_var` 32×32 threshold), **L-45** (`scale_part_thresh_sumdiff`
   speed-6 fall-through), **L-52** (`set_vt_partitioning`
   `block_height == block_width` invariant), **L-53** (partition at
   frame edges), **L-54** (`vbp_thresholds` location).
7. All four HLDs (`00-architecture`, `01-threading`, `02-simd-dispatch`,
   `04-svc`).

**Scope guardrails:**
- 8-bit only. No `u16` paths. libvpx `#if CONFIG_VP9_HIGHBITDEPTH`
  branches in `fill_variance_8x8avg`/`fill_variance_4x4avg` are cited
  for traceability but not ported.
- No altref. `lag_in_frames > 0` rejected at construction (divergence
  ledger §"Alt-ref frames"); the altref-coupled wedge at
  `vp9_speed_features.c:600` ("VBR+lag+alt_ref_gf_group →
  `VAR_BASED_PARTITION`") is unreachable in our scope.
- AQ: only cyclic-refresh (mode 3). The CR-boosted threshold
  re-compute at `vp9_encodeframe.c:1377-1383` IS in scope.

---

## 1. Purpose

The `partition` module owns the per-superblock (SB, 64×64 pixels)
partition-tree decision for the real-time encoder at cpu-used 6/7/8.
For each SB it picks a partition tree over `{64×64, 64×32, 32×64,
32×32, 32×16, 16×32, 16×16, 16×8, 8×16, 8×8, 8×4, 4×8, 4×4}`, writes
that decision into `MODE_INFO::sb_type` cells in the SB's 8×8 grid,
and produces two side-channel signals consumed by `mode_decision`:

1. `MacroBlock::variance_low[25]` — 25 byte-sized flags marking
   sub-blocks whose temporal variance is low enough that mode
   decision may skip golden/NEWMV/intra modes (read via
   `get_force_skip_low_temp_var` in mode_decision LLD; libvpx
   `vp9_pickmode.c:1441-1486`).
2. `MacroBlock::content_state_sb`, `skip_low_source_sad`,
   `lowvar_highsumdiff`, `sb_use_mv_part`, `sb_{mv,col,row}_part` —
   per-SB flags set during the partition pre-passes.

This LLD implements the `VAR_BASED_PARTITION` path (default at
cpu-used 6/7; A2 §3.2) plus the threshold setup, content-state
classification, and supporting `REFERENCE_PARTITION` / `FIXED_PARTITION`
selectors that share data with it. The recursive RD-style
`nonrd_pick_partition` (A2 §3.5) is **not** owned here — it lives in
`mode_decision` because its hot work is `nonrd_pick_sb_modes`. This
LLD only exposes the dispatch enum that picks which path runs.

Bit-exactness contract: byte parity with libvpx under matched
cpu-used + content state + qindex inputs. Variance arithmetic is
fully integer (no float). The only non-determinism risk is SIMD
vs. scalar in `vpx_avg_*` / `vpx_minmax_8x8` / `vpx_variance*`, all of
which are owned by HLD-simd §3.6 / §3.24.

---

## 2. libvpx origin

Primary file: `vp9/encoder/vp9_encodeframe.c`. Lookup tables share
with `vp9/encoder/vp9_pickmode.c`.

| libvpx symbol | File:line | Rust owner |
|---|---|---|
| `set_block_size` | `vp9_encodeframe.c:335-342` | `partition::sb_grid::set_block_size` |
| `Var`, `partition_variance`, `v4x4`..`v64x64` | `vp9_encodeframe.c:344-389` | `partition::tree::*` |
| `variance_node`, `tree_to_node` | `vp9_encodeframe.c:386-438` | replaced by `VarianceTreeNode<'a>` enum (§4.2) |
| `fill_variance`, `get_variance`, `sum_2_variances` | `vp9_encodeframe.c:440-470` | `Var::fill` / `Var::finalize` / `Var::sum_2` |
| `fill_variance_tree` | `vp9_encodeframe.c:460-470` | `V{N}::fold` per level |
| `set_vt_partitioning` | `vp9_encodeframe.c:472-547` | `partition::vt::set_vt_partitioning` |
| `scale_part_thresh_sumdiff` | `vp9_encodeframe.c:549-567` | `partition::thresholds::scale_part_thresh_sumdiff` |
| `set_vbp_thresholds` | `vp9_encodeframe.c:569-635` | `partition::thresholds::set_vbp_thresholds` |
| `vp9_set_variance_partition_thresholds` | `vp9_encodeframe.c:637-684` | `partition::thresholds::set_variance_partition_thresholds` |
| `fill_variance_4x4avg` (8-bit branch only) | `vp9_encodeframe.c:714-748` (HBD 730-738 skipped, D-01) | `partition::sampler::fill_variance_4x4avg` |
| `fill_variance_8x8avg` (8-bit branch only) | `vp9_encodeframe.c:750-784` (HBD 766-774 skipped, D-01) | `partition::sampler::fill_variance_8x8avg` |
| `set_low_temp_var_flag` | `vp9_encodeframe.c:829-893` | `partition::vt::set_low_temp_var_flag` |
| `chroma_check` | `vp9_encodeframe.c:1173-1207` | `partition::chroma::chroma_check` |
| `avg_source_sad` | `vp9_encodeframe.c:1209-1257` | `partition::content_state::avg_source_sad` |
| `choose_partitioning` | `vp9_encodeframe.c:1261-1766` | `partition::choose_partitioning` |
| `set_partial_b64x64_partition` | `vp9_encodeframe.c:2344-2359` | `partition::sb_grid::set_partial_b64x64_partition` |
| `set_fixed_partitioning` | `vp9_encodeframe.c:2366-2395` | `partition::sb_grid::set_fixed_partitioning` |
| `coord_lookup[16]` | `vp9_encodeframe.c:2397-2415` | `partition::tables::COORD_LOOKUP` |
| `pos_shift_16x16[4][4]` | `vp9_pickmode.c:55-57` | `partition::tables::POS_SHIFT_16X16` |
| `VP9_VAR_OFFS[64]` | `vp9_encodeframe.c:70-83` | `partition::tables::VP9_VAR_OFFS` |

Stage 1 specialist anchors: A2 §2.5–§2.9, §3.10, §4.1, §4.7,
§4.15–§4.16, §5.1, §5.14, §6.

---

## 3. Public API

All items `pub(crate)` in `rvp9_encoder::partition`. Nothing escapes
the encoder crate.

### 3.1 Top-level driver

```rust
/// Choose the partition tree for one 64×64 superblock and write the
/// per-8×8 `sb_type` cells. Updates `MacroBlock` side channels
/// (`variance_low`, `content_state_sb`, `skip_low_source_sad`,
/// `lowvar_highsumdiff`, `sb_use_mv_part`, `sb_mv{col,row}_part`).
///
/// **Preconditions:**
///   * `(mi_row, mi_col)` is a multiple of `MI_BLOCK_SIZE = 8`.
///   * Frame loop has called `set_offsets` + `set_segment_index`.
///   * For inter frames: LAST ref buffer is in the pool.
///   * For SVC top-resolution layer with `svc_use_lowres_part`: the
///     lower-layer partition snapshot is populated.
///
/// **Postconditions:**
///   * Every in-frame 8×8 MODE_INFO cell in the SB has `sb_type`
///     set to a valid `BlockSize`.
///   * Out-of-frame cells (partial SB at right/bottom edge) are
///     left at the frame-allocator default (`BlockSize::B4x4 = 0`).
///     L-53 discussion in §5.1.
///   * `mb.variance_low[..]` is fully written (0 = no flag, 1 = set).
///   * `mb.color_sensitivity[..]` set by the chroma post-pass.
///
/// **Panics:** none on legal input. `debug_assert!` checks for
/// internal invariants (L-52: block_height == block_width).
///
/// **Unsafety:** safe wrapper. Internal SIMD pointer math is
/// encapsulated in `simd::dispatch` calls.
pub(crate) fn choose_partitioning(
    sb: &mut SbContext<'_>,
    cm: &Common,
    rc: &RateControl,
    sf: &SpeedFeaturesSnapshot,
    bfp: &Bfp,
    simd: &SimdDispatch,
    coords: SbCoords,
) -> Result<(), EncodeError>;
```

The libvpx `int` return value (always `0` — `vp9_encodeframe.c:1766`)
is replaced by `Result<()>` for forward-compat (§7.1).

### 3.2 Per-frame threshold setup

```rust
/// Compute `vbp_thresholds[0..=3]` plus auxiliary scalars
/// (`vbp_threshold_sad`, `vbp_threshold_copy`, `vbp_threshold_minmax`,
/// `vbp_bsize_min`) from base qindex, noise, denoiser, speed,
/// resolution. Mirrors `vp9_set_variance_partition_thresholds`
/// (`vp9_encodeframe.c:637-684`). Main thread only, once per frame.
pub(crate) fn set_variance_partition_thresholds(
    out: &mut VbpFrameThresholds,
    cm: &Common,
    rc: &RateControl,
    sf: &SpeedFeaturesSnapshot,
    noise: NoiseLevel,
    denoiser: Option<DenoiserLevel>,
    q: u8,
);
```

`VbpFrameThresholds` lives on `Encoder::vbp` (L-54 resolution: per
frame, shared `&` across workers, no `UnsafeCell` — HLD-threading §5.3
"per-frame transient, read-only by workers").

### 3.3 Per-SB content classification

```rust
/// Run SAD + variance of (cur source, last source) on the 64×64,
/// classify into `ContentStateSb`. Sets `mb.content_state_sb` and
/// `mb.zero_temp_sad_source`. Returns the 64×64 SAD.
/// Mirrors `avg_source_sad` (`vp9_encodeframe.c:1209-1257`).
pub(crate) fn avg_source_sad(
    mb: &mut MacroBlock,
    cur_src: &Plane<u8>,
    last_src: &Plane<u8>,
    sb_offset: usize,
    content_state_sb_fd: Option<&mut [u8]>,
    is_screen: bool,
    rc_mode: RateCtlMode,
    bfp: &Bfp,
) -> u32;
```

### 3.4 `variance_low[25]` read accessor (for `mode_decision`)

The read-side lookup of `variance_low` lives in `mode_decision`
(co-located with the call site at `vp9_pickmode.c:2208`), but the
*table and wrapper* are owned here so the layout has a single source
of truth (L-39):

```rust
/// Read-only view onto `MacroBlock::variance_low[25]` with typed
/// query semantics. The `query` implementation lives in
/// `mode_decision::skip_rules`; the wrapper struct and the
/// `POS_SHIFT_16X16` table live here.
#[derive(Copy, Clone)]
pub(crate) struct VarianceLowMask<'a> {
    raw: &'a [u8; 25],
}

impl<'a> VarianceLowMask<'a> {
    pub(crate) fn new(raw: &'a [u8; 25]) -> Self { Self { raw } }
    /// Returns the flag for one mode-decision block. Dispatches on
    /// `sb_type` per `get_force_skip_low_temp_var`
    /// (`vp9_pickmode.c:1441-1486`). `sub_row`, `sub_col` are
    /// `mi_row & 0x7`, `mi_col & 0x7` (8-cell offsets within SB).
    pub(crate) fn query(self, sb_type: BlockSize,
                        sub_row: u8, sub_col: u8) -> u8;
}
```

### 3.5 Partition-search dispatch enum

```rust
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum PartitionSearchType {
    /// cpu-used 6/7 default. `choose_partitioning` then
    /// `nonrd_use_partition` (do-recon at leaves; no RD compare).
    VarBased,
    /// cpu-used 5 default. `choose_partitioning` then
    /// `nonrd_select_partition` (refines via `nonrd_pick_partition`).
    Reference,
    /// cpu-used 8 + ≤352×288 non-key. `get_estimated_pred` +
    /// `nonrd_pick_partition` (ML-driven).
    MlBased,
    /// Seg-skip path. `set_fixed_partitioning`.
    Fixed,
    /// `SOURCE_VAR_BASED_PARTITION` — unreachable in v1 (requires
    /// auto_alt_ref). Kept for exhaustive `match`. `#[doc(hidden)]`.
    #[doc(hidden)]
    SourceVarBased,
}

/// Pick partition strategy per SB. Mirrors switch at
/// `vp9_encodeframe.c:5627-5690`.
pub(crate) fn select_partition_search(
    mb: &MacroBlock, cm: &Common, sf: &SpeedFeaturesSnapshot,
    rc: &RateControl, intra_only: bool,
) -> PartitionSearchType;
```

### 3.6 Partition-walk entry points (re-exported from `mode_decision`)

After `choose_partitioning` writes `sb_type` cells, the walk that
calls `nonrd_pick_sb_modes` + `encode_b_rt` at each leaf is owned by
the mode-decision LLD. Re-exported here for traceability:

```rust
pub(crate) use crate::mode_decision::partition_walk::nonrd_use_partition;
pub(crate) use crate::mode_decision::partition_walk::nonrd_select_partition;
```

---

## 4. Types

### 4.1 `SbContext<'frame>`

```rust
/// One-shot per-SB writable bundle. Built by the SB-row driver
/// before each SB, dropped after. Lifetime ties all borrows to the
/// worker's frame scope (HLD-threading §5.5).
pub(crate) struct SbContext<'frame> {
    /// Worker-owned MacroBlock (HLD-threading §5.4).
    pub mb: &'frame mut MacroBlock,
    /// Disjoint slice of `mi_grid_visible` covering this tile's
    /// columns. Granted via `slice_for_tile` (HLD-threading §6.2).
    pub mi_grid: TileMiGrid<'frame>,
    /// Per-tile scratch for the variance tree (reused across SBs).
    pub scratch: &'frame mut PartitionScratch,
    /// Optional 4×4-downsample scratch (libvpx `vt2`). Lazily
    /// allocated on first need within the tile (§7.2).
    pub scratch_4x4: &'frame mut Option<V16x16Array>,
    /// Read-only frame-level VBP thresholds (L-54).
    pub frame_vbp: &'frame VbpFrameThresholds,
    pub noise: NoiseEstimateSnapshot,
    pub denoiser: DenoiserSnapshot,
    /// Optional per-SB high-content streak buffer.
    pub content_state_sb_fd: Option<&'frame mut [u8]>,
}

/// Per-tile, per-frame scratch.
pub(crate) struct PartitionScratch {
    pub vt: V64x64,
    pub force_split: [bool; 21],
    pub avg_16x16: [i32; 4],
    pub maxvar_16x16: [i32; 4],
    pub minvar_16x16: [i32; 4],
    pub variance4x4downsample: [bool; 16],
}
```

### 4.2 Variance tree (L-38 resolution)

**L-38 decision: mirror libvpx's nested-struct layout
(`V64x64 → V32x32[4] → V16x16[4] → V8x8[4]`, ~5 KB).** Alternative —
flat `[Var; 256]` with hand-rolled indexing — was rejected: it breaks
the natural recursion in `set_vt_partitioning` for no measurable
benefit (the tree fold is 64 scalar adds; SIMD lives at the leaves
via `vpx_avg_8x8` / `vpx_minmax_8x8`). Per A2 §7 Q1.

Storage lives in `PartitionScratch` (heap, per-tile, reused) rather
than stack, to avoid hot-path zeroing on tile workers with tight
stacks.

```rust
/// Mirrors libvpx `Var` (`vp9_encodeframe.c:344-353`). 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct Var {
    pub sum_square_error: u32,   // u32 per A2 §2.6 cite.
    pub sum_error: i32,
    pub log2_count: i32,
    pub variance: i32,           // populated by `Var::finalize`.
}

#[derive(Copy, Clone, Default)]
pub(crate) struct PartitionVariance {
    pub none: Var,
    pub horz: [Var; 2],
    pub vert: [Var; 2],
}

// Each level: `part: PartitionVariance` (none/horz/vert), plus
// `split: [child; 4]` (TL, TR, BL, BR — raster).
#[derive(Copy, Clone, Default)] pub(crate) struct V4x4   { pub part: PartitionVariance, pub split: [Var;   4] }
#[derive(Copy, Clone, Default)] pub(crate) struct V8x8   { pub part: PartitionVariance, pub split: [V4x4;  4] }
#[derive(Copy, Clone, Default)] pub(crate) struct V16x16 { pub part: PartitionVariance, pub split: [V8x8;  4] }
#[derive(Copy, Clone, Default)] pub(crate) struct V32x32 { pub part: PartitionVariance, pub split: [V16x16;4] }
#[derive(Copy, Clone, Default)] pub(crate) struct V64x64 { pub part: PartitionVariance, pub split: [V32x32;4] }

/// 4×4-downsample scratch (libvpx `vt2`); see §7.2.
pub(crate) type V16x16Array = Box<[V16x16; 16]>;
```

The libvpx `variance_node` + `tree_to_node` (`:386-438`) abstraction
is replaced by a typed enum view (§7.5):

```rust
pub(crate) enum VarianceTreeNode<'a> {
    L64(&'a mut V64x64),
    L32(&'a mut V32x32),
    L16(&'a mut V16x16),
    L8(&'a mut V8x8),
}

impl<'a> VarianceTreeNode<'a> {
    pub fn part(&self) -> &PartitionVariance { ... }
    pub fn part_mut(&mut self) -> &mut PartitionVariance { ... }
}
```

`Var::finalize` mirrors `get_variance` (`vp9_encodeframe.c:446-452`):

```rust
impl Var {
    pub(crate) fn finalize(&mut self) {
        // libvpx: (256 * (sse - ((i64)sum*sum) >> log2_count)) >> log2_count
        let sq = ((self.sum_error as i64).wrapping_mul(self.sum_error as i64)
                  >> self.log2_count) as u32;
        self.variance = (256u32.wrapping_mul(
                            self.sum_square_error.wrapping_sub(sq))
                         >> self.log2_count) as i32;
    }
    pub(crate) fn sum_2(a: &Var, b: &Var, out: &mut Var) {
        debug_assert_eq!(a.log2_count, b.log2_count);
        out.sum_square_error =
            a.sum_square_error.wrapping_add(b.sum_square_error);
        out.sum_error = a.sum_error.wrapping_add(b.sum_error);
        out.log2_count = a.log2_count + 1;
        out.variance = 0;
    }
}
```

`wrapping_*` per D7 (HLD-arch §6) — libvpx `uint32_t`/`int32_t`
arithmetic wraps; explicit at every site to keep debug-build behaviour
matching release.

Per-level `fold` methods replace `fill_variance_tree(BLOCK_*)`:

```rust
impl V64x64 {
    pub(crate) fn fold(&mut self) {
        Var::sum_2(&self.split[0].part.none, &self.split[1].part.none,
                   &mut self.part.horz[0]);
        Var::sum_2(&self.split[2].part.none, &self.split[3].part.none,
                   &mut self.part.horz[1]);
        Var::sum_2(&self.split[0].part.none, &self.split[2].part.none,
                   &mut self.part.vert[0]);
        Var::sum_2(&self.split[1].part.none, &self.split[3].part.none,
                   &mut self.part.vert[1]);
        Var::sum_2(&self.part.vert[0], &self.part.vert[1],
                   &mut self.part.none);
    }
}
// V32x32, V16x16, V8x8 follow the same pattern.
```

### 4.3 `force_split[21]`

```rust
/// Per-SB descent flags. 21 entries:
///   [0]      — 64×64
///   [1..=4]  — four 32×32 quadrants (raster: TL, TR, BL, BR)
///   [5..=20] — sixteen 16×16, indexed `5 + i*4 + j` for
///              (32×32 quadrant i, 16×16 sub-position j) (both raster)
/// Citation: `vp9_encodeframe.c:1399-1400`.
pub(crate) type ForceSplit = [bool; 21];
```

`bool` instead of libvpx `int` per §7.9.

### 4.4 `MacroBlock::variance_low[25]` layout (L-39 resolution)

**L-39 decision: 25-byte array (libvpx-matched), Morton-order at
indices [9..=24].**

```rust
// rvp9_encoder::block::MacroBlock (HLD-arch §3.7)
pub(crate) struct MacroBlock {
    // ... many other fields ...

    /// Per-sub-block low-temporal-variance flags. Packed:
    ///   [0]      — 64×64
    ///   [1..=2]  — 64×32 (top, bottom)
    ///   [3..=4]  — 32×64 (left, right)
    ///   [5..=8]  — 32×32 quadrants (TL, TR, BL, BR — raster)
    ///   [9..=24] — 16×16, indexed via `POS_SHIFT_16X16[i][j]`
    ///              where `i = (mi_row & 0x7) >> 1` and
    ///              `j = (mi_col & 0x7) >> 1`. Z-order at the 2×2
    ///              sub-quadrant scale, row-major within each 2×2.
    /// libvpx: `vp9_block.h:202` `uint8_t variance_low[25]`.
    pub variance_low: [u8; 25],
}
```

**Why `[u8; 25]`** (not `[bool; 25]` or a 25-bit bitfield): libvpx
stores `uint8_t variance_low[25]` and read sites walk it as bytes
(`vp9_pickmode.c:1453-1483`). Packing to 4 bytes introduces
shift-pattern divergence risk; `bool` is byte-sized in Rust anyway.

**Layout verification.** From `POS_SHIFT_16X16` (A2 §5.1):

```
       j=0  j=1  j=2  j=3
i=0:    9   10   13   14
i=1:   11   12   15   16
i=2:   17   18   21   22
i=3:   19   20   23   24
```

Each 2×2 sub-quadrant of `(i, j)` maps to a contiguous 4-run: TL →
`{9..=12}`, TR → `{13..=16}`, BL → `{17..=20}`, BR → `{21..=24}`. So
the indexing is **Z-order on the 2×2-of-2×2 grid, row-major within
each 2×2** — A2 §2.9 patched wording (async A7 #17 fix; "row-major"
was the earlier-incorrect description).

**Write side cross-check.** `set_low_temp_var_flag` (line 886) writes
`x->variance_low[(i << 2) + j + 9] = 1` where `i ∈ 0..=3` is the
32×32 quadrant (raster) and `j ∈ 0..=3` is the 16×16 sub-index
within (raster). `(i=0,j=0)→9`, `(i=1,j=0)→13`, `(i=2,j=0)→17` — all
match the table. The asymmetry — **write by raster, read by Morton** —
is precisely the indexing contract we must preserve.

Rust write accessors:

```rust
impl MacroBlock {
    pub(crate) fn set_variance_low_64x64(&mut self)         { self.variance_low[0] = 1; }
    pub(crate) fn set_variance_low_64x32(&mut self, h: u8)  {
        debug_assert!(h < 2); self.variance_low[1 + h as usize] = 1;
    }
    pub(crate) fn set_variance_low_32x64(&mut self, h: u8)  {
        debug_assert!(h < 2); self.variance_low[3 + h as usize] = 1;
    }
    pub(crate) fn set_variance_low_32x32(&mut self, q: u8)  {
        debug_assert!(q < 4); self.variance_low[5 + q as usize] = 1;
    }
    pub(crate) fn set_variance_low_16x16(&mut self, q32: u8, s16: u8) {
        debug_assert!(q32 < 4 && s16 < 4);
        // libvpx: variance_low[(i << 2) + j + 9] = 1
        self.variance_low[9 + ((q32 as usize) << 2) + s16 as usize] = 1;
    }
}
```

### 4.5 `POS_SHIFT_16X16`, `COORD_LOOKUP`, `VP9_VAR_OFFS`

```rust
/// vp9/encoder/vp9_pickmode.c:55-57. Read side of variance_low[]
/// indices [9..=24]. Morton at the 2×2 sub-quadrant scale (L-39).
pub(crate) const POS_SHIFT_16X16: [[u8; 4]; 4] = [
    [9,  10, 13, 14],
    [11, 12, 15, 16],
    [17, 18, 21, 22],
    [19, 20, 23, 24],
];

#[derive(Copy, Clone)]
pub(crate) struct SubCoord { pub row: u8, pub col: u8 }

/// vp9/encoder/vp9_encodeframe.c:2397-2415. Maps (32×32 idx ∈ 0..=3,
/// 16×16 sub-idx ∈ 0..=3) to (row, col) of the 16×16 in 8×8-cell
/// coordinates within the 64×64 SB.
///   index 0..=3   — 32×32 quadrant 0 (TL): (0,0)(0,2)(2,0)(2,2)
///   index 4..=7   — 32×32 quadrant 1 (TR): (0,4)(0,6)(2,4)(2,6)
///   index 8..=11  — 32×32 quadrant 2 (BL): (4,0)(4,2)(6,0)(6,2)
///   index 12..=15 — 32×32 quadrant 3 (BR): (4,4)(4,6)(6,4)(6,6)
/// In words: row/col walk (0,0)(0,2)(2,0)(2,2) within a 4×4-cell
/// quadrant — Z-order at 2-cell stride. Each quadrant's 16×16 sub-idx
/// is a Morton order of the 4 sub-blocks (matching `POS_SHIFT_16X16`'s
/// 2×2-grouping at one level higher).
pub(crate) const COORD_LOOKUP: [SubCoord; 16] = [
    SubCoord{row:0,col:0}, SubCoord{row:0,col:2}, SubCoord{row:2,col:0}, SubCoord{row:2,col:2},
    SubCoord{row:0,col:4}, SubCoord{row:0,col:6}, SubCoord{row:2,col:4}, SubCoord{row:2,col:6},
    SubCoord{row:4,col:0}, SubCoord{row:4,col:2}, SubCoord{row:6,col:0}, SubCoord{row:6,col:2},
    SubCoord{row:4,col:4}, SubCoord{row:4,col:6}, SubCoord{row:6,col:4}, SubCoord{row:6,col:6},
];

/// vp9/encoder/vp9_encodeframe.c:70-83. Key-frame intra ref baseline
/// (all 128). Indexed only as `&[u8; 64]` source for `vpx_avg_*`.
/// HBD variants `VP9_HIGH_VAR_OFFS_{10,12}` (lines 87-109) NOT ported
/// per D-01.
pub(crate) const VP9_VAR_OFFS: [u8; 64] = [128; 64];
```

### 4.6 `VbpFrameThresholds` (L-54 resolution)

**L-54 decision: per-frame state on the encoder god-struct, shared
`&` across workers. No `UnsafeCell`. Not per-tile, not per-thread.**

```rust
/// Set once per frame on the main thread; read by every tile worker
/// during `choose_partitioning`. Workers borrow as
/// `&VbpFrameThresholds` (HLD-threading §5.3 read-only worker
/// transient).
#[derive(Copy, Clone, Debug)]
pub(crate) struct VbpFrameThresholds {
    /// `thresholds[0..=3]` = (64×64, 32×32, 16×16, 8×8).
    pub thresholds: [i64; 4],
    /// `vbp_threshold_sad` — CR+y_sad shortcut to 64×64
    /// (`vp9_encodeframe.c:1467-1483`).
    pub threshold_sad: i64,
    /// `vbp_threshold_copy` — partition-copy gate at speed 7+.
    pub threshold_copy: i64,
    /// `vbp_threshold_minmax` = `15 + (q >> 3)`.
    pub threshold_minmax: i32,
    /// `vbp_bsize_min` — `B8x8` (key) or `B16x16` (inter), the
    /// smallest non-leaf bsize the tree descends to.
    pub bsize_min: BlockSize,
}
```

Per L-54 / A2 §7 Q19: libvpx stores these on `VP9_COMP`; workers
read directly. We mirror. ~6 cache lines, read-only during the frame.

**Per-SB CR override.** At `vp9_encodeframe.c:1377-1383`, CR-boosted
SBs re-call `set_vbp_thresholds` into a **stack-local** `thresholds[4]`
(line 1289-1290 constructs the local from `cpi->vbp_thresholds[0..3]`).
We mirror: `PartitionScratch` carries `thresholds: [i64; 4]`
initialised from `SbContext::frame_vbp` at the top of every
`choose_partitioning` call; CR override mutates the local copy only
(never `Encoder::vbp` — would be a race).

### 4.7 `ContentStateSb`, `SbCoords`

```rust
/// Mirrors libvpx `CONTENT_STATE_SB` (`vp9_encoder.h:137-145`).
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum ContentStateSb {
    Invalid            = 0,
    LowSadLowSumdiff   = 1,
    LowSadHighSumdiff  = 2,
    HighSadLowSumdiff  = 3,
    HighSadHighSumdiff = 4,
    LowVarHighSumdiff  = 5,
    VeryHighSad        = 6,
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct SbCoords {
    pub mi_row: u32,
    pub mi_col: u32,
    /// `(mi_stride >> 3) * (mi_row >> 3) + (mi_col >> 3)`. Used to
    /// index `prev_partition[]`, `content_state_sb_fd[]`,
    /// `copied_frame_cnt[]`.
    pub sb_offset: usize,
}
```

### 4.8 Marker-trait summary

| Type | `Send` | `Sync` | `Copy` | Notes |
|---|---|---|---|---|
| `Var`, `PartitionVariance`, `V4x4`..`V64x64` | yes | yes | yes | POD |
| `V16x16Array` | yes | yes | no | `Box`; never cloned |
| `PartitionScratch` | yes | yes | no | size + non-Copy by design (§7.4) |
| `VbpFrameThresholds` | yes | yes | yes | shared `&` across workers |
| `ContentStateSb`, `PartitionSearchType`, `SbCoords` | yes | yes | yes | POD/enum |
| `VarianceLowMask<'_>` | yes | yes | yes | `&[u8; 25]` wrapper |
| `SbContext<'_>` | no | no | no | holds `&mut` borrows |

---

## 5. Internal algorithm

### 5.1 Frame-edge cropping (L-53 resolution)

**L-53 decision: mirror libvpx's edge handling exactly.** Three
mechanisms work together:

1. **`set_block_size` no-op guard** (`vp9_encodeframe.c:338`): out-of-frame
   writes are silently dropped. Rust:
   ```rust
   pub(crate) fn set_block_size(grid: &mut TileMiGrid<'_>, cm: &Common,
                                 mi_row: u32, mi_col: u32, bsize: BlockSize) {
       if cm.mi_cols > mi_col && cm.mi_rows > mi_row {
           grid.mi_at_mut(mi_row, mi_col).sb_type = bsize;
       }
   }
   ```

2. **`set_vt_partitioning` must-fit guards** (`vp9_encodeframe.c:493-494,
   510-511, 518, 531`): return 0 ("descend further") if the right or
   bottom half of the candidate block would fall outside the frame.

3. **Top-level descent override** (`vp9_encodeframe.c:1701-1703`):
   `if (mi_col + 8 > cm.mi_cols || mi_row + 8 > cm.mi_rows ||
        !set_vt_partitioning(L64, ...))` → drop into 32×32 descent.
   The `||` short-circuit avoids building the 64×64 tree node for a
   partial SB; we mirror for bit-parity and the perf benefit.

**Out-of-frame cells stay at `BlockSize::B4x4 = 0`** (the frame
allocator's zero-init default — libvpx `vpx_calloc`-zeroes `cm->mi`
each allocation cycle per A1 §2.8). No divergence.

### 5.2 `set_vt_partitioning` invariant (L-52 resolution)

**L-52 decision: `debug_assert!(block_height == block_width)`. The
function is only ever called with square bsizes
(B64x64/B32x32/B16x16/B8x8).** Citation: `vp9_encodeframe.c:482`
`assert(block_height == block_width);`. Sub-block writes (32×64,
64×32, etc.) happen via `set_block_size`-with-`subsize` (lines 525,
538), never via re-entering `set_vt_partitioning` with a non-square
bsize.

Body (mirrors `vp9_encodeframe.c:472-547`):

```rust
pub(crate) fn set_vt_partitioning(
    cm: &Common, grid: &mut TileMiGrid<'_>,
    node: VarianceTreeNode<'_>, bsize: BlockSize,
    mi_row: u32, mi_col: u32,
    threshold: i64, bsize_min: BlockSize, force_split: bool,
) -> bool {
    let bw = num_8x8_blocks_wide(bsize);
    let bh = num_8x8_blocks_high(bsize);
    debug_assert_eq!(bh, bw, "non-square bsize {:?}", bsize);
    if force_split { return false; }
    let is_key = cm.frame_is_intra_only();
    let fits = |bw: u32, bh: u32| mi_col + bw/2 < cm.mi_cols
                                    && mi_row + bh/2 < cm.mi_rows;

    // Leaf.
    if bsize == bsize_min {
        if is_key { node.part_mut().none.finalize(); }
        if fits(bw, bh)
            && (node.part().none.variance as i64) < threshold {
            set_block_size(grid, cm, mi_row, mi_col, bsize);
            return true;
        }
        return false;
    }
    debug_assert!((bsize as u8) > (bsize_min as u8));

    // Non-leaf.
    if is_key { node.part_mut().none.finalize(); }
    // Key frame: above 32×32 OR very high variance → descend.
    if is_key && ((bsize as u8) > (BlockSize::B32x32 as u8)
                  || (node.part().none.variance as i64) > (threshold << 4))
    { return false; }
    if fits(bw, bh) && (node.part().none.variance as i64) < threshold {
        set_block_size(grid, cm, mi_row, mi_col, bsize);
        return true;
    }
    // Vertical split (two side-by-side bsize/2).
    if mi_row + bh/2 < cm.mi_rows {
        let sub = get_subsize(bsize, Partition::Vert);
        node.part_mut().vert[0].finalize();
        node.part_mut().vert[1].finalize();
        if (node.part().vert[0].variance as i64) < threshold
            && (node.part().vert[1].variance as i64) < threshold
            && get_plane_block_size(sub, PlaneType::Chroma).is_valid() {
            set_block_size(grid, cm, mi_row, mi_col, sub);
            set_block_size(grid, cm, mi_row, mi_col + bw/2, sub);
            return true;
        }
    }
    // Horizontal split.
    if mi_col + bw/2 < cm.mi_cols {
        let sub = get_subsize(bsize, Partition::Horz);
        node.part_mut().horz[0].finalize();
        node.part_mut().horz[1].finalize();
        if (node.part().horz[0].variance as i64) < threshold
            && (node.part().horz[1].variance as i64) < threshold
            && get_plane_block_size(sub, PlaneType::Chroma).is_valid() {
            set_block_size(grid, cm, mi_row, mi_col, sub);
            set_block_size(grid, cm, mi_row + bh/2, mi_col, sub);
            return true;
        }
    }
    false  // Reject; caller must descend to four-way split.
}
```

The `get_plane_block_size(sub, PlaneType::Chroma).is_valid()` check
(libvpx `get_plane_block_size(subsize, &xd->plane[1]) < BLOCK_INVALID`)
guards against invalid chroma partitions on horz/vert split (4:2:0
subsampling). Mirror exactly. `finalize` is in-place — once called,
subsequent reads by `set_low_temp_var_flag` see the finalised value.

### 5.3 `set_vbp_thresholds` (mirrors `vp9_encodeframe.c:569-635`)

```rust
pub(crate) fn set_vbp_thresholds(
    out: &mut [i64; 4],
    cm: &Common, rc: &RateControl, sf: &SpeedFeaturesSnapshot,
    noise: NoiseLevel, denoiser: Option<DenoiserLevel>,
    q: u8, content_state: ContentStateSb,
) {
    let is_key = cm.frame_is_intra_only();
    let mult: i64 = if is_key { 20 } else { sf.variance_part_thresh_mult as i64 };
    let mut tb: i64 = mult * (cm.y_dequant[q as usize][1] as i64);

    if is_key {
        out[0] = tb;
        out[1] = tb >> 2;
        out[2] = tb >> 2;
        out[3] = tb << 2;
        return;
    }
    // Inter frames.
    if cm.width >= 640 && cm.height >= 480 && noise.enabled() {
        match noise.level() {
            NoiseLevel::High   => tb = 3 * tb,
            NoiseLevel::Medium => tb <<= 1,
            l if l < NoiseLevel::Low => tb = (7 * tb) >> 3,
            _ => {}
        }
    }
    // Denoiser branch vs scale_part_thresh_sumdiff branch.
    if let Some(level) = denoiser.filter(|_| sf.speed > 5) {
        tb = vp9_scale_part_thresh(tb, level, content_state,
                                   cm.svc_temporal_layer_id());
    } else {
        tb = scale_part_thresh_sumdiff(tb, sf.speed, cm.width, cm.height,
                                       content_state);
    }
    out[0] = tb;
    out[2] = tb << (sf.speed as u32);
    if cm.width >= 1280 && cm.height >= 720 && sf.speed < 7 {
        out[2] <<= 1;
    }
    if cm.width <= 352 && cm.height <= 288 {
        out[0] = tb >> 3;
        out[1] = tb >> 1;
        out[2] = tb << 3;
        if rc.avg_frame_qindex[FrameType::Inter as usize] > 220 {
            out[2] <<= 2;
        } else if rc.avg_frame_qindex[FrameType::Inter as usize] > 200 {
            out[2] <<= 1;
        }
    } else if cm.width < 1280 && cm.height < 720 {
        out[1] = (5 * tb) >> 2;
    } else if cm.width < 1920 && cm.height < 1080 {
        out[1] = tb << 1;
    } else {
        out[1] = (5 * tb) >> 1;
    }
    if sf.disable_16x16part_nonkey { out[2] = i64::MAX; }
    // NOTE: `out[3]` deliberately NOT touched in inter branch — see §7.8.
}
```

### 5.4 `scale_part_thresh_sumdiff` at speed 6 (L-45 resolution)

**L-45 decision: confirmed intentional — at speed 6 the function
falls through both arms and returns `threshold_base` unchanged.
Mirror exactly. Doc-comment explicitly notes the fall-through so
future readers don't "fix" it.**

Citation: `vp9_encodeframe.c:549-567`. There is no `else if (speed == 6)`
arm. The content-state scaling (`{kLowSadLowSumdiff,
kHighSadLowSumdiff, kLowVarHighSumdiff}` → multiply by 5/4) applies
only at speed 7 and speed 8. At speed 6 the threshold scaling is
driven entirely by other inputs (noise level, denoiser, resolution).

```rust
/// libvpx: `vp9_encodeframe.c:549-567`. **Speed 6 deliberately falls
/// through both branches and returns `threshold_base`** — see L-45.
pub(crate) fn scale_part_thresh_sumdiff(
    tb: i64, speed: u8, width: u32, height: u32,
    content_state: ContentStateSb,
) -> i64 {
    if speed >= 8 {
        if width <= 640 && height <= 480 { return (5 * tb) >> 2; }
        if matches!(content_state,
                    ContentStateSb::LowSadLowSumdiff
                  | ContentStateSb::HighSadLowSumdiff
                  | ContentStateSb::LowVarHighSumdiff) {
            return (5 * tb) >> 2;
        }
    } else if speed == 7 {
        if matches!(content_state,
                    ContentStateSb::LowSadLowSumdiff
                  | ContentStateSb::HighSadLowSumdiff
                  | ContentStateSb::LowVarHighSumdiff) {
            return (5 * tb) >> 2;
        }
    }
    tb  // speed 6 (and speeds 0..5) reach here unchanged.
}
```

### 5.5 `set_low_temp_var_flag` (L-44 resolution)

**L-44 decision: mirror the libvpx wedge exactly. The 32×32 threshold
expression depends on `sf.short_circuit_low_temp_var`:**

| `short_circuit_low_temp_var` | 32×32 threshold | Active at cpu-used |
|---|---|---|
| `1` | `(5 * thresholds[1]) >> 3` | 6 (CBR && !screen); 7 (inherited) |
| `2` | `thresholds[1] >> 1` | 8 (CBR && !screen && 720p+ noise≥medium) |
| `3` | `(5 * thresholds[1]) >> 3` | 8 (CBR && !screen, default low-noise) |

Sources: A2 §5.13 (speed-feature dispatch slices); libvpx
`vp9_speed_features.c:687, 771-776`. The mode-decision LLD reads
`variance_low[]` and applies different skip rules for
`short_circuit_low_temp_var ∈ {1, 2, 3}`; we only mirror the write
bit pattern here.

```rust
pub(crate) fn set_low_temp_var_flag(
    sb: &mut SbContext<'_>, cm: &Common, sf: &SpeedFeaturesSnapshot,
    vt: &V64x64, thresholds: &[i64; 4],
    ref_frame_partition: RefFrame,
    mi_row: u32, mi_col: u32,
) {
    let mv_thr: i32 = if cm.width > 640 { 8 } else { 4 };
    let mv = sb.mb.mi_current().mv[0].as_mv();
    let gate = ref_frame_partition == RefFrame::Last
        && (sf.short_circuit_low_temp_var == 1
            || (mv.col.abs() < mv_thr && mv.row.abs() < mv_thr));
    if !gate { return; }

    match sb.mb.mi_current().sb_type {
        BlockSize::B64x64 => {
            if (vt.part.none.variance as i64) < (thresholds[0] >> 1) {
                sb.mb.set_variance_low_64x64();
            }
        }
        BlockSize::B64x32 => for i in 0..2 {
            if (vt.part.horz[i].variance as i64) < (thresholds[0] >> 2) {
                sb.mb.set_variance_low_64x32(i as u8);
            }
        }
        BlockSize::B32x64 => for i in 0..2 {
            if (vt.part.vert[i].variance as i64) < (thresholds[0] >> 2) {
                sb.mb.set_variance_low_32x64(i as u8);
            }
        }
        _ => {
            // Per-32×32 quadrant walk (libvpx "default" arm).
            const IDX: [(u32, u32); 4] = [(0,0), (0,4), (4,0), (4,4)];
            for i in 0..4 {
                let (dr, dc) = IDX[i];
                if cm.mi_cols <= mi_col + dc
                    || cm.mi_rows <= mi_row + dr { continue; }
                let this = sb.mi_grid.mi_at(mi_row + dr, mi_col + dc);
                if this.sb_type == BlockSize::B32x32 {
                    // L-44 wedge: t32x32 depends on short_circuit value.
                    let t32: i64 = if sf.short_circuit_low_temp_var == 1
                                   || sf.short_circuit_low_temp_var == 3 {
                        (5 * thresholds[1]) >> 3
                    } else {
                        thresholds[1] >> 1
                    };
                    if (vt.split[i].part.none.variance as i64) < t32 {
                        sb.mb.set_variance_low_32x32(i as u8);
                    }
                } else if sf.short_circuit_low_temp_var >= 2
                    && matches!(this.sb_type,
                                BlockSize::B16x16
                              | BlockSize::B32x16
                              | BlockSize::B16x32) {
                    for j in 0..4 {
                        if (vt.split[i].split[j].part.none.variance as i64)
                            < (thresholds[2] >> 8) {
                            sb.mb.set_variance_low_16x16(i as u8, j as u8);
                        }
                    }
                }
            }
        }
    }
}
```

### 5.6 `choose_partitioning` outline

Mirrors `vp9_encodeframe.c:1261-1766` and A2 §3.10's flowchart. Steps
keyed to libvpx line numbers:

1. **Locals + per-SB thresholds.** Initialise `force_split=[false;21]`,
   `avg_16x16`, `maxvar_16x16`, `minvar_16x16`, `variance4x4downsample`.
   Snapshot `thresholds[4]` from `sb.frame_vbp.thresholds` (1289-1290).
2. **`scene_change` / `force_64_split` / `is_key` flags** (1291-1310).
3. **Reset** `mb.variance_low.fill(0)` (1339).
4. **`avg_source_sad`-driven shortcuts** (1341-1374): read
   `mb.content_state_sb`, set `skip_low_source_sad` /
   `lowvar_highsumdiff`; call `scale_partitioning_svc` (SVC LLD) or
   `copy_partitioning` (`partition::copy`); both may return early.
5. **CR re-threshold** (1377-1383): if `aq_mode == CR && seg.enabled
   && segment_id_boosted`, call `set_vbp_thresholds` into the **local**
   `thresholds[4]` using the boosted q.
6. **Screen + base-layer scene-change** (1386-1388): if screen &&
   spatial_layer==0 && force_64_split → `thresholds[1] = (3*thresholds[1])>>2`.
7. **4×4-avg gate**: `threshold_4x4avg = (speed<8) ? thresholds[1]<<1 :
   i64::MAX` (1391).
8. **Edge crop for variance fill** (1393-1394): `pixels_wide = 64 +
   (mb_to_right_edge >> 3 if negative)`; same for `pixels_high`.
9. **Inter prediction pass** (1407-1525): determine `bsize_pred`,
   compute `y_sad_g` (SAD vs. golden, speed<8 only), compute `y_sad`
   (via `vp9_int_pro_motion_estimation` for non-speed-8 paths, sets
   `sb_use_mv_part` / `sb_mv{col,row}_part`); pick LAST vs. GOLDEN;
   `vp9_build_inter_predictors_sb`; CR + low-y_sad shortcut → set
   64×64 + `variance_low[0]=1` and return; `partition-copy` shortcut
   on low `y_sad_last` → return. (KEY frames bypass this and set
   `d = VP9_VAR_OFFS.as_ptr(), dp = 0`.)
10. **16×16-avg fill** over `(i, j) ∈ 0..4 × 0..4` (1573-1612):
    `fill_variance_8x8avg` → 4 leaf Vars per 16×16; `fold()` +
    `finalize()`. If `var16 > thresholds[2]` raise `force_split` at
    16×16, parent 32×32, and 64×64. Otherwise if `compute_minmax_variance
    && var16 > thresholds[1] && !CR_boosted`: run `compute_minmax_8x8`,
    compare against `threshold_minmax` (doubled for `VeryHighSad`).
    If `is_key || (low_res && var16 > threshold_4x4avg)`: enable
    `variance4x4downsample`; lazy-alloc `vt2`; `fill_variance_4x4avg`.
    Accumulate `avg_16x16` / `maxvar_16x16` / `minvar_16x16` per 32×32.
11. **32×32 combine** (1655-1697): per `i ∈ 0..4`, if any sub used
    4×4 downsample re-fold via `vt2`; `vt.split[i].fold()`. If
    `!force_split[i+1]`: check `var32 > thresholds[1]`, the
    `var32>thresholds[1]>>1 && var32>avg16>>1` skew, and the
    `noise<kLow && height<=360 && (max-min)>thresholds[1]>>1 &&
    max>thresholds[1]` skew. Update `max/min_var_32x32`.
12. **64×64 combine** (1681-1697): if `!force_split[0]`: `vt.fold();
    vt.part.none.finalize()`. Two noise-conditioned skew tests gate
    `force_split[0]`.
13. **Top-level descent** (1701-1748, the L-53 wedge):
    `sb_too_small = mi_col+8 > cm.mi_cols || mi_row+8 > cm.mi_rows`.
    If `sb_too_small || !set_vt_partitioning(L64, ...)`: 3-level
    nested descent into 32×32 → 16×16 → 8×8 → (4×4 if `use_4x4_partition`).
    Each level picks `vtemp` from `vt2` if `variance4x4downsample[i*4+j]`,
    else from `vt.split[i].split[j]`.
14. **Post-passes** (1750-1763): `update_prev_partition` if
    `copy_partition_flag`; `update_partition_svc` if SVC second-from-top
    layer; `set_low_temp_var_flag` if `short_circuit_low_temp_var > 0`;
    `chroma_check`.

Returns `Ok(())`.

Subtleties:

- `cyclic_refresh_segment_id_boosted` / `_base` are read-only
  predicates imported from the `aq` LLD.
- `compute_minmax_8x8` lives in `partition::sampler` (§8.1 #1).
- VBR-lag + altref / `use_altref_onepass` paths (libvpx 1432-1457)
  are unreachable per D-13. We always take the `int_pro` branch for
  inter-frame `y_sad`.
- `scale_partitioning_svc`, `update_partition_svc` belong in `svc`
  LLD; `copy_partitioning`, `update_prev_partition` belong in
  `partition::copy`.

### 5.7 `fill_variance_8x8avg` (8-bit only)

```rust
pub(crate) fn fill_variance_8x8avg(
    s: *const u8, sp: i32, d: *const u8, dp: i32,
    x16: u32, y16: u32, vst: &mut V16x16,
    pixels_wide: i32, pixels_high: i32, is_key: bool,
    simd: &SimdDispatch,
) {
    for k in 0..4 {
        let x8 = (x16 + ((k & 1) << 3)) as i32;
        let y8 = (y16 + ((k >> 1) << 3)) as i32;
        let mut sse: u32 = 0;
        let mut sum: i32 = 0;
        if x8 < pixels_wide && y8 < pixels_high {
            // SAFETY: x8/y8 bounds-checked; strides match libvpx.
            let s_avg = unsafe {
                (simd.vpx_avg_8x8)(
                    s.offset(y8 as isize * sp as isize + x8 as isize), sp)
            } as i32;
            let d_avg = if is_key { 128 } else {
                unsafe {
                    (simd.vpx_avg_8x8)(
                        d.offset(y8 as isize * dp as isize + x8 as isize), dp)
                } as i32
            };
            sum = s_avg - d_avg;
            sse = (sum as u32).wrapping_mul(sum as u32);
        }
        vst.split[k as usize].part.none = Var {
            sum_square_error: sse, sum_error: sum,
            log2_count: 0, variance: 0,
        };
    }
}
```

`fill_variance_4x4avg` is structurally identical with
`simd.vpx_avg_4x4` and `v4x4.part.none` at each of four 4×4 cells of
an 8×8 sub-block. HBD branches at `vp9_encodeframe.c:730-738, 766-774`
are **not** ported per D-01.

`sum * sum` is `i32 * i32`. `|sum|` is at most 255 for an 8×8
average → overflow impossible — but `wrapping_mul` keeps the bit
pattern stable across debug/release.

---

## 6. SIMD dispatch

This module consumes (does not own) kernels from `rvp9_core::simd`.
All are dispatched through `SimdDispatch` (HLD-simd §4) or `Bfp`
(HLD-simd §4.3).

| Kernel | Use site | HLD-simd § | ISAs in libvpx |
|---|---|---|---|
| `vpx_avg_8x8` | `fill_variance_8x8avg` | §3.24 | scalar, sse2, neon |
| `vpx_avg_4x4` | `fill_variance_4x4avg` | §3.24 | scalar, sse2, neon |
| `vpx_minmax_8x8` | `compute_minmax_8x8` | §3.24 | scalar, sse2, neon |
| `vpx_variance{64x64, 32x32, 16x16, 8x8}` | indirectly via `Bfp.vf` in `block_variance` (consumed for the y_sad shortcut) | §3.6 | scalar, sse2, avx2, neon |
| `vpx_sad{64x64, 32x32, 16x16}` | `y_sad_g`, `y_sad_last`, `avg_source_sad` | §3.3 | scalar, sse2, avx2, neon |
| `vpx_int_pro_row`, `vpx_int_pro_col`, `vpx_vector_var` | via `motion_estimation::int_pro` | §3.10 | scalar, sse2, neon |

**No SIMD kernels owned by this module.** Differential tests for each
kernel are owned by HLD-simd. Our integration test (§9.4) re-runs
`fill_variance_8x8avg` with `SimdDispatch::scalar_only()` vs.
`SimdDispatch::with_caps(detected)` and asserts byte-identical
output.

---

## 7. Deliberate divergences from libvpx

Each entry is a draft for `notes/divergence-ledger.md`. Format per
the ledger's `§Format` block.

### 7.1 `choose_partitioning` returns `Result<()>` instead of `int (always 0)`

**Where:** `partition::choose_partitioning`. libvpx: `vp9_encodeframe.c:1261`.
**What libvpx does:** declares `int` return; every path returns `0`.
**What rvp9 does:** `Result<(), EncodeError>`; no v1 path returns `Err`.
**Why:** HLD-arch §4 — `Result`-typed fallibility end-to-end.
Forward-compat for future SVC partition-copy validation.
**Byte-parity impact:** none. **Compression-efficiency impact:** none.
**Future plan:** first real `Err` site lands when `svc.md` introduces
low-res partition-copy validation.

### 7.2 `vt2` (4×4-downsample scratch) hoisted to per-tile, lazy-alloc

**Where:** `SbContext::scratch_4x4: &'frame mut Option<V16x16Array>`.
libvpx: `vp9_encodeframe.c:1556` (`vpx_calloc`) + 1764 (`vpx_free`).
**What libvpx does:** ~5 KB alloc + zero per SB whenever any 16×16
needs 4×4 downsample.
**What rvp9 does:** lazy `Box<[V16x16; 16]>` per tile worker, reused
across SBs. Write-before-read inside `fill_variance_4x4avg`, no zero.
**Why:** removes ~50 KB/s alloc/free traffic at 1080p30 speed 8 key
frames.
**Byte-parity impact:** none. **Compression-efficiency impact:** none.
**Future plan:** pre-allocate in `WorkerPool::new` if first-SB
`Box::new` shows in profiles.

### 7.3 `set_block_size` is `debug_assert!`-guarded for non-edge callers

**Where:** `partition::sb_grid::set_block_size`. libvpx: `vp9_encodeframe.c:338`.
**What libvpx does:** silently no-ops on out-of-frame writes.
**What rvp9 does:** safe wrapper still no-ops (libvpx match);
`debug_assert!` in non-`choose_partitioning` callers that they
bounds-check upstream.
**Why:** trap unexpected OOB in debug builds. Release matches libvpx.
**Byte-parity impact:** none. **Compression-efficiency impact:** none.
**Future plan:** keep.

### 7.4 `PartitionScratch` is not `Copy`

**Where:** `partition::PartitionScratch`.
**What libvpx does:** stack `v64x64 vt` + `int force_split[21]` inside
`choose_partitioning`.
**What rvp9 does:** owned by `ThreadData`, never cloned. No `Copy`.
**Why:** ~5 KB `Copy` is a foot-gun; no code path needs cloning.
**Byte-parity impact:** none. **Compression-efficiency impact:** none.
**Future plan:** keep.

### 7.5 Drop `variance_node` + `tree_to_node` indirection

**Where:** `partition::tree::VarianceTreeNode<'a>`. libvpx:
`vp9_encodeframe.c:386-438`.
**What libvpx does:** `variance_node` (pointer-of-pointer struct) +
`switch(bsize)` populator on every `set_vt_partitioning` call.
**What rvp9 does:** typed enum carrying a `&mut` to the right level.
**Why:** Rust enums lower to the same jump table; pointer-setup
redundant. Lifetime makes the borrow scope explicit.
**Byte-parity impact:** none. **Compression-efficiency impact:** none.
**Future plan:** keep. C used the pattern only for lack of tagged
unions.

### 7.6 Explicit `Var::finalize` instead of always-eager `get_variance`

**Where:** `partition::tree::Var::finalize`. libvpx: `vp9_encodeframe.c:446-452`.
**What libvpx does:** consumers call `get_variance` redundantly; no
side effect (output overwritten with the same value each time).
**What rvp9 does:** mutating method; readers must finalise before
reading `variance`. No automatic refresh.
**Why:** explicit lifecycle. HLD-arch "no hidden side effects".
**Byte-parity impact:** none if every reader finalises first. §9.3
property test guards. **Compression-efficiency impact:** none.
**Future plan:** keep.

### 7.7 Speed-6 fall-through in `scale_part_thresh_sumdiff` documented (L-45)

**Where:** `partition::thresholds::scale_part_thresh_sumdiff`. libvpx:
`vp9_encodeframe.c:549-567`.
**What libvpx does:** at speed 6, no arm matches → returns
`threshold_base` unchanged.
**What rvp9 does:** mirrors. Doc-comment explicitly notes the
fall-through.
**Why:** L-45 audit confirmed intentional. Doc-comment prevents
future "fix this missing arm" regressions.
**Byte-parity impact:** none. **Compression-efficiency impact:** none.
**Future plan:** keep with prominent doc-comment.

### 7.8 `vbp_thresholds[3]` cross-frame retention (inter-frame branch)

**Where:** `partition::thresholds::set_vbp_thresholds`. libvpx:
`vp9_encodeframe.c:614-635`.
**What libvpx does:** inter frames don't touch `thresholds[3]`; it
retains the last key-frame value (`threshold_base << 2`).
**What rvp9 does:** mirror. `VbpFrameThresholds::thresholds[3]` is
initialised at construction and re-written only on key frames.
**Why:** real cross-frame dependency. Diverging would break the
`use_4x4_partition && key` path's `thresholds[3]` reads (line 1733).
**Byte-parity impact:** would diverge on multi-frame sequences with
key frames. §9.5 tests. **Compression-efficiency impact:** none.
**Future plan:** keep. Document the cross-frame dependency on the
field's doc comment.

### 7.9 `force_split: [bool; 21]` instead of `int[21]`

**Where:** `partition::ForceSplit`.
**What libvpx does:** `int force_split[21]` (84 bytes).
**What rvp9 does:** `[bool; 21]` (21 bytes); entries only ever 0/1.
**Why:** trivial type narrowing.
**Byte-parity impact:** none. **Compression-efficiency impact:** none.
**Future plan:** keep.

### 7.10 `variance_low: [u8; 25]` Morton-order traceability (L-39)

**Where:** `MacroBlock::variance_low: [u8; 25]`. libvpx: `vp9_block.h:202`.
**What libvpx does:** identical type + layout. **(Not a divergence —
listed for L-39 traceability.)** L-39 considered (and rejected)
packing to 25 bits.
**What rvp9 does:** identical.
**Byte-parity impact:** none. **Compression-efficiency impact:** none.
**Future plan:** keep.

---

## 8. Open questions

### 8.1 Non-blocking — for downstream LLD review

1. **`compute_minmax_8x8` location.** Currently planned to live in
   `partition::sampler` alongside `fill_variance_*`. Alternative:
   inline into `fill_variance_8x8avg` to avoid a second pass over the
   same pixels. **Recommendation:** keep separate for v1 to match
   libvpx structure; revisit at Stage 5 profile.

2. **`copy_partitioning` / `update_prev_partition` owner.** Plan:
   `partition::copy::*`. They touch `cpi->prev_partition` and
   `cpi->copied_frame_cnt` (HLD-threading §5.2 — per-SB-disjoint
   worker writes via `UnsafeCell<[T]>` with tile-disjoint slices).
   **Recommendation:** accept `partition::copy::*`.

3. **`scale_partitioning_svc` / `update_partition_svc` owner.** Belong
   in `svc.md`; called from `choose_partitioning` at lines 1357 and
   1755. Plan: `pub(crate)` cross-module re-exports. **Recommendation:**
   confirm at SVC LLD review.

4. **`chroma_check` placement.** Owner here (`partition::chroma`)
   matches libvpx (`vp9_encodeframe.c:1173`). The MD LLD reads
   `color_sensitivity` but doesn't write it. Settled.

5. **Test-vector source for the variance tree.** No conformance
   oracle (encoder-only); differential against libvpx is the sole
   ground truth. §9.4 / §9.6 cover this.

### 8.2 Blocking — Arthur or HLD-threading owner ask

**None.** All Stage 1 / Stage 2 questions in this module's scope
(L-38, L-39, L-44, L-45, L-52, L-53, L-54) are resolved in the LLD
body above.

The only external assumption is **HLD-threading §5.2's `UnsafeCell<[T]>`
tile-disjoint model** for `prev_partition`, `prev_variance_low`,
`prev_segment_id`, `copied_frame_cnt`, `content_state_sb_fd` — already
settled there; no new ask.

---

## 9. Test strategy

### 9.1 Unit tests — verbatim tables

`POS_SHIFT_16X16` and `COORD_LOOKUP` are byte-copied from libvpx; a
transcription typo is the smallest possible regression. Two test
shapes:

- `pos_shift_16x16_verbatim` — `assert_eq!(POS_SHIFT_16X16, [[9,10,13,14],
  [11,12,15,16], [17,18,21,22], [19,20,23,24]])`.
- `pos_shift_16x16_is_morton_per_2x2_quadrant` — for each 2×2
  sub-quadrant of `(i, j)`, collect the four indices into a
  `BTreeSet` and assert they equal one of `{9..=12}`, `{13..=16}`,
  `{17..=20}`, `{21..=24}` (the four contiguous Morton runs).
- `coord_lookup_verbatim` — pairwise `(row, col)` equality against the
  expected 16-entry list from A2 §5.14.

### 9.2 Unit tests — `Var` arithmetic

- `var_finalize_hand_computed`: drive `Var { sse=12345, sum=-300,
  log2_count=6 }`, assert `variance == 43756` (hand-computed:
  `(256 * (12345 - ((-300)² >> 6))) >> 6`).
- `sum_2_log2_count_invariant`: assert that `Var::sum_2` increments
  `log2_count` by exactly 1 and adds `sse` / `sum` fields.
- `wrapping_arithmetic_matches_uint32` (negative case): drive a
  pathological large-magnitude `sum_error` and verify the
  `wrapping_mul` semantics match libvpx's silent `uint32_t` wrap.

### 9.3 Property tests

- `finalize_is_idempotent`: for `sse`/`sum`/`log2_count` over their
  legal ranges, `v.finalize()` twice yields the same `variance`.

### 9.4 Differential SIMD-vs-scalar — `fill_variance_8x8avg`

Replay with `SimdDispatch::scalar_only()` vs.
`SimdDispatch::with_caps(detected)` on random 16×16 `(s, d)` patches.
Assert the four resulting `Var` records are byte-identical. HLD-simd
§3.24 owns the kernel-level diff; this LLD owns the integration at
the `(s_avg - d_avg)` arithmetic site.

### 9.5 Cross-frame test — `thresholds[3]` retention

Run `set_variance_partition_thresholds` for a key frame followed by
an inter frame with different `q`. Assert
`vbp.thresholds[3]` is unchanged by the inter call.

### 9.6 Differential against libvpx — partition tree byte-exactness

Patch libvpx with `third_party/libvpx/patches/partition_trace.patch`
that dumps per-frame `partition.bin` (flat `BlockSize` codes per 8×8
MI cell). Build rvp9 with the same dump. `tests/differential/partition.rs`
encodes a corpus clip with both encoders at matched config (cpu-used,
target_bitrate, content state) and asserts byte-identical traces.
On mismatch the harness reports `(frame_idx, sb_idx, mi_row, mi_col)`.
**Test gate:** 0 divergences on {640×360, 720p, 1080p} ×
{synth-gradient, real-content} clips at cpu-used 6/7/8.

### 9.7 Differential against libvpx — `variance_low[25]` bit-exact

Same instrumentation; per-frame trace of every `variance_low[25]`
array per SB. Same corpus and gate.

### 9.8 Edge-case tests — frame edges (L-53)

- `partial_sb_at_right_edge_descends_below_64x64`: configure a 72×64
  frame; the second SB column (mi_col=8) is partial. Assert every
  leaf's `sb_type` has `max_pixel_width() <= 8` in that strip.
- `out_of_frame_cells_remain_default`: assert cells beyond
  `cm.mi_cols`/`cm.mi_rows` keep `BlockSize::B4x4 = 0` (the
  `set_block_size` no-op guard's documented behaviour).

### 9.9 Determinism — `set_low_temp_var_flag` L-44 wedge

Two tests target the threshold-expression wedge:

- `speed6_uses_5_over_8_threshold` (`short_circuit_low_temp_var = 1`):
  set `vt.split[i].part.none.variance = ((5 * thresholds[1]) >> 3) - 1`;
  assert `variance_low[5 + i] == 1`.
- `speed8_noise_uses_half_threshold` (`short_circuit_low_temp_var = 2`):
  set the variance below `thresholds[1] >> 1`; assert the same flag.

### 9.10 Miri + coverage targets

Miri runs the deterministic unit tests (no threading) and catches OOB
in `s.offset(...)` if `pixels_wide` / `pixels_high` get miscomputed.

Coverage gates:
- 100 % branch coverage of `set_vt_partitioning` (7 branches).
- 100 % branch coverage of `scale_part_thresh_sumdiff` including the
  speed-6 fall-through (§7.7 / L-45).
- 100 % branch coverage of `set_low_temp_var_flag` (every sb_type arm
  + the L-44 wedge).
- ≥ 90 % line coverage of `choose_partitioning`; SVC + copy-partition
  branches are gated by speed-7+/SVC-enabled and covered by
  integration tests rather than unit.

---

**Line-count summary:** ~1350 lines.
**Key §7 divergence-ledger entries (10):** §7.1 `choose_partitioning`
`Result` return; §7.2 `vt2` per-tile lazy scratch; §7.3 `set_block_size`
debug-assert for non-edge callers; §7.4 `PartitionScratch` non-Copy;
§7.5 drop `variance_node` indirection; §7.6 explicit `Var::finalize`;
§7.7 speed-6 fall-through documented (L-45); §7.8 `thresholds[3]`
cross-frame retention; §7.9 `force_split: [bool; 21]`; §7.10
`variance_low: [u8; 25]` Morton-order traceability (L-39).
**§8 blocking questions:** none. All L-38/39/44/45/52/53/54 resolved
in the body.
