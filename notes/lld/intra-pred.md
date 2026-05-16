# LLD: intra-pred

> **Module:** `rvp9-core::dsp::intra_pred` (predictor kernels + dispatch),
> consumed by `rvp9-decoder` (residual reconstruction) and
> `rvp9-encoder` (mode decision + final encode reconstruction).
>
> **Stage 1 inputs:** `notes/vp9-analysis/02-encoder-core.md` §4.5 + §4.11
> (encoder call sites at `vp9_pickmode.c:386-427`, `1155-1213`,
> `2516-2645`), `notes/vp9-analysis/00-overview.md` (synthesis).
> **HLD inputs:** `notes/hld/00-architecture.md` (unsafe / error model),
> `notes/hld/02-simd-dispatch.md` §3.20 (IntraPredFn signature, 56 slots,
> ISA coverage matrix), `notes/hld/01-threading.md` (read-only-after-init
> dispatch table; intra-pred is leaf, no shared mutable state),
> `notes/hld/04-svc.md` (predictors are reused across spatial layers via
> the per-layer current-buffer pointer).
> **Libvpx primary citation:** `third_party/libvpx/vp9/common/vp9_reconintra.c`
> for the border-builder + dispatch, `third_party/libvpx/vpx_dsp/intrapred.c`
> for the kernel math.

---

## 1. Purpose

This module produces the **intra prediction sample block** that the
decoder adds to a dequantised residual to reconstruct an intra-coded
block, and that the encoder feeds to `block_yrd` /
`model_rd_for_sb_y` for mode decision. It owns:

1. The **ten VP9 intra prediction modes** — DC, V, H, D45, D135, D117,
   D153, D207, D63, TM — at all four square transform sizes
   (4×4, 8×8, 16×16, 32×32).
2. The **edge-pixel materialisation** logic: copying available top /
   top-right / left samples out of the reconstructed-frame buffer into
   the contiguous `above[-1 ..= 2*bs)` / `left[0 .. bs)` arrays the
   kernels read, including the boundary-replication rules at frame
   edges (top row, left column, right column, bottom row) and at the
   right edge of an unavailable neighbour.
3. The **runtime-dispatched SIMD specialisations** of each predictor
   (per HLD-SIMD §3.20: scalar + sse2/ssse3 on x86_64, NEON on
   aarch64; no AVX2 variants exist in libvpx 1.14.1).

It is **read-only-after-init** (the dispatch table is one `OnceLock`);
worker threads dereference function pointers but do not write into the
module. There is no per-frame state; per-block state (the
`above`/`left` scratch arrays) lives on the stack of the callsite.

**Scope guardrails:**

- **8-bit only** (PLAN.md D-01 RESOLVED). The `uint16_t` /
  `vpx_highbd_*` half of `vp9_reconintra.c:112-261` and
  `vpx_dsp/intrapred.c:456-851` is **deferred to v2**; this LLD cites
  those line ranges only where the 8-bit path mirrors them, and never
  ports the HBD kernels.
- **No altref** (PLAN.md D-13 RESOLVED). Intra-pred has no altref
  interaction — predictors are mode-local — but the encoder call sites
  in `vp9_pickmode.c:1155-1213` and `2516-2645` are inside the non-RD
  pick-mode loop which already excludes altref-ME branches per
  `02-encoder-core.md` §4.5.
- **Real-time only** (cpu-used 6-8). The full RDO intra path
  (`vp9_rd_pick_intra_mode_sb` in `vp9/encoder/vp9_rdopt.c:3222`) is
  reached only on keyframes when `sf->nonrd_keyframe == 0`
  (`02-encoder-core.md` §4.2.1, `vp9_encodeframe.c:4672-4678`).
  Predictor kernels are shared with the non-RD path.

---

## 2. libvpx origin

| Component | libvpx file:line | Stage 1 doc |
|---|---|---|
| Edge-builder + dispatch table | `vp9/common/vp9_reconintra.c:263-403` | A2 §4.5, §4.11 |
| Public entry point (decoder + encoder) | `vp9/common/vp9_reconintra.c:405-427` (`vp9_predict_intra_block`) | A2 §4.11 |
| Init once | `vp9/common/vp9_reconintra.c:70-110, 429-431` (`vp9_init_intra_predictors`) | — |
| Mode → tx-type lookup | `vp9/common/vp9_reconintra.c:24-35` (`intra_mode_to_tx_type_lookup`) | A2 §4.6 (transforms LLD also needs this) |
| Mode → neighbour-need bitmask | `vp9/common/vp9_reconintra.c:37-54` (`extend_modes`) | A2 §4.5 |
| Scalar kernel math | `vpx_dsp/intrapred.c:21-248` (mode dispatchers) plus the macro-expanded `intra_pred_allsizes` / `intra_pred_no_4x4` block at `vpx_dsp/intrapred.c:856-916` | A2 §4.5 (kernel summary) |
| 4×4 specialised entries (DC/V/H/TM/D207/D63/D45/D117/D135/D153) | `vpx_dsp/intrapred.c:283-454` | — |
| RTCD declarations | `vpx_dsp/vpx_dsp_rtcd_defs.pl:44-213` | HLD-SIMD §3.20 |
| SSE2 / SSSE3 kernels | `vpx_dsp/x86/intrapred_sse2.asm`, `intrapred_ssse3.asm` | HLD-SIMD §3.20 |
| NEON kernels | `vpx_dsp/arm/intrapred_neon.c`, `intrapred_neon_asm.asm` | HLD-SIMD §3.20 |
| Non-RD encoder caller (full SB intra) | `vp9/encoder/vp9_pickmode.c:386-427` (`compute_intra_yprediction`) | A2 §4.11 |
| Non-RD intra-mode RDO inside `pick_inter_mode` | `vp9/encoder/vp9_pickmode.c:2516-2645` (intra search block) | A2 §4.5 |
| KF / intra-only fast path | `vp9/encoder/vp9_pickmode.c:1155-1213` (`vp9_pick_intra_mode`) | A2 §4.2.1 |
| Per-transform-block walker (final encode) | `vp9/encoder/vp9_pickmode.c:1045-1070` (`estimate_block_intra`) | A2 §4.11 |
| `intra_mode_list` in non-RD intra search | `vp9/encoder/vp9_pickmode.c:158-159` | A2 §2.4 |

`02-encoder-core.md` §4.5 (intra search inside `vp9_pick_inter_mode`)
confirms only DC/V/H/TM are exercised in the non-RD inter path at
cpu-used 6-8 (per `intra_mode_list[]`). The other six directional
modes (D45/D135/D117/D153/D207/D63) are still reached on intra-only
frames and at higher quality presets via `vp9_rd_pick_intra_mode_sb`,
so **all ten modes must ship** in `rvp9-core` regardless of which
encoder paths exercise them.

---

## 3. Public API

The public API lives in `rvp9-core::dsp::intra_pred`. Three layers:

1. **Per-mode kernel functions** (one per `(mode, size)` plus DC
   neighbour-availability variants) — exposed as `unsafe extern "C"
   fn` pointers via the dispatch table (HLD-SIMD §3.20).
2. **`predict_intra_block(...)`** — the safe high-level entry point.
   Hides the edge-builder, neighbour-availability decoding, and DC
   variant selection. Used by decoder reconstruct and the encoder's
   `estimate_block_intra` / `compute_intra_yprediction`.
3. **`compute_intra_yprediction_sb(...)`** — the strided full-SB
   variant used by the non-RD intra search inside
   `vp9_pick_inter_mode` (cites `vp9_pickmode.c:386-427`). Same
   semantics as calling `predict_intra_block` once per TX block, but
   the caller pre-walks the TX grid.

### 3.1 Types crossing the API boundary

```rust
/// VP9 intra prediction mode. Wire enum values match libvpx's
/// `PREDICTION_MODE` (vp9/common/vp9_blockd.h:32-43); preserving the
/// numeric values matters for prob-table indexing (entropy LLD) and
/// for the `intra_mode_to_tx_type_lookup` array used by transforms.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum IntraMode {
    Dc   = 0,
    V    = 1,
    H    = 2,
    D45  = 3,
    D135 = 4,
    D117 = 5,
    D153 = 6,
    D207 = 7,
    D63  = 8,
    Tm   = 9,
}

/// VP9 transform-block size for intra prediction.
/// Matches `TX_SIZE` (vp9/common/vp9_enums.h:117-123).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TxSize { Tx4 = 0, Tx8 = 1, Tx16 = 2, Tx32 = 3 }

impl TxSize {
    /// Block side length in pixels = `4 << tx_size`.
    /// Matches `bs = 4 << tx_size` in vp9_reconintra.c:274.
    #[inline] pub const fn pixels(self) -> usize { 4usize << (self as u32) }
}

/// Plane index. luma=0, chroma=1/2. Subsampling derived elsewhere.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Plane { Y = 0, U = 1, V = 2 }

/// Edge-availability flags for one prediction block. Mirrors the
/// `have_top`, `have_left`, `have_right` arguments computed at
/// vp9_reconintra.c:411-413 from `xd->above_mi`, `xd->left_mi`,
/// and `aoff + txw < bw`.
///
/// `Edges` is `Copy` and 1 byte — three boolean flags packed; the
/// fourth bit is unused. NEVER `Default`-construct: every caller must
/// fill it correctly from neighbour MI pointers and TX-grid position.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Edges {
    pub have_top:   bool,
    pub have_left:  bool,
    pub have_right: bool,
}
```

### 3.2 Geometry inputs

`predict_intra_block` needs to know whether the *block being
predicted* is at the right or bottom frame edge in order to decide
how far to copy the above row (which may extend beyond the frame) and
how far to copy the left column (which may stop short of the frame
bottom). libvpx encodes this via the `MACROBLOCKD` god-struct
(`mb_to_right_edge`, `mb_to_bottom_edge` and the `cur_buf->{y,uv}_{width,height}`
fields read at `vp9_reconintra.c:286-297`). The Rust API takes an
explicit `FrameGeom` to avoid threading the god-struct down to the
DSP layer:

```rust
/// Per-call geometry: where the block sits in the (sub-sampled)
/// plane it is being predicted into.
///
/// `pixel_x`, `pixel_y` are the top-left corner of the *block* in
/// plane-sample units (i.e. after chroma subsampling already applied
/// by the caller). `plane_width`, `plane_height` are the plane
/// dimensions in the same units.
///
/// Cite vp9/common/vp9_reconintra.c:287-297 for the original
/// `x0 = (-xd->mb_to_left_edge >> (3 + ss_x)) + x` derivation;
/// the encoder/decoder callers in our port compute these directly
/// against the frame dimensions and pass them in.
#[derive(Copy, Clone, Debug)]
pub struct BlockGeom {
    pub pixel_x: u32,
    pub pixel_y: u32,
    pub plane_width:  u32,
    pub plane_height: u32,
}
```

### 3.3 The safe entry points

```rust
/// Predict one transform-block worth of intra samples into `dst`.
///
/// Mirrors `vp9_predict_intra_block` (vp9/common/vp9_reconintra.c:405-427)
/// for the 8-bit, non-HBD path. The HBD branch (l.418-423) is dropped
/// per PLAN.md D-01.
///
/// `ref_plane` is the reconstructed-pixel buffer for the same plane,
/// pointing at the *block origin* (not the frame origin). `ref_stride`
/// is the stride of that buffer in bytes. The edge-builder reads
/// `ref_plane[-ref_stride .. ref_plane[(2*bs-1)*ref_stride - 1]]` —
/// i.e. one row above and up to `bs` rows of leftward column.
///
/// `dst` is the destination buffer, also pointing at the block origin.
/// `dst_stride` is its stride.
///
/// **Preconditions:**
/// 1. `dst[r][c]` writable for `0 <= r < bs && 0 <= c < bs`.
/// 2. `ref_plane[r][c]` readable for the cells used by edges
///    (see §5.2 — depends on `edges` flags and `mode`).
/// 3. `ref_plane` and `dst` may alias (encoder calls predict
///    in-place into the reconstruction buffer; see
///    `vp9_pickmode.c:386-427` for the `skip_encode` case where
///    `pd->dst.buf` is both source and destination).
/// 4. `geom.pixel_x + bs <= geom.plane_width + bs` (block top-left
///    may sit beyond the frame edge by up to `bs - 1` for the
///    right-bottom partial-SB case; see L-53).
///
/// **Postcondition:** `dst[0..bs][0..bs]` is filled with the predicted
/// samples, with edge-replication applied per §5.
///
/// # Safety
/// `ref_plane` and `dst` are raw pointers that bypass the borrow
/// checker. Caller must satisfy preconditions 1-4. The function
/// dereferences function pointers from the global `SimdDispatch`,
/// which must have been initialised by the `rvp9-core` boot path
/// before any worker thread spawns (HLD-threading §3.2,
/// R-09 in decisions-needed.md).
pub unsafe fn predict_intra_block(
    mode: IntraMode,
    tx_size: TxSize,
    edges: Edges,
    ref_plane: *const u8,
    ref_stride: isize,
    dst: *mut u8,
    dst_stride: isize,
    geom: BlockGeom,
    plane: Plane,
    dispatch: &SimdDispatch,
);

/// Strided full-superblock intra prediction. Mirrors
/// `compute_intra_yprediction` (vp9_pickmode.c:386-427). Used by the
/// non-RD intra search at the SB level. Walks the TX grid inside the
/// SB and calls `predict_intra_block` for each TX block.
///
/// `bsize` is the SB-level block size (one of BLOCK_8X8..BLOCK_64X64).
/// `tx_size` is the per-TX-block size used for prediction; libvpx
/// derives it as `max_txsize_lookup[bsize]` at vp9_pickmode.c:388 and
/// the rvp9 caller passes it explicitly.
///
/// The `skip_encode` path of libvpx (where `p->src.buf` is read as
/// the "reference") is collapsed: rvp9 always reads from the
/// reconstruction buffer. The encoder caller chooses which buffer
/// that is before invoking us — see `notes/lld/mode-decision.md`
/// for the `reuse_inter_pred_sby` interaction.
///
/// # Safety
/// As `predict_intra_block`, plus: the caller must ensure that the
/// full destination plane covering `bsize` is writable, and that the
/// reference plane covers the SB at `(mi_col, mi_row)`.
pub unsafe fn compute_intra_yprediction_sb(
    mode: IntraMode,
    bsize: BlockSize,
    tx_size: TxSize,
    sb: SbPlanePtrs<'_>,
    dispatch: &SimdDispatch,
);
```

`BlockSize` and `SbPlanePtrs` are defined by the frame-buffers and
mode-decision LLDs respectively; this LLD treats them as opaque.

### 3.4 Init

```rust
/// Build the intra-pred sub-dispatch from the global CPU caps.
/// Called once at `rvp9-core` boot (HLD-SIMD §4.4).
pub(crate) fn build_intra_pred_dispatch(caps: &Caps) -> IntraPredDispatch;
```

There is no public init function; the global `SimdDispatch::init()`
in HLD-SIMD §4.4 calls `build_intra_pred_dispatch` and writes it into
the table. This mirrors libvpx's `vp9_init_intra_predictors` (called
once via `once()` at `vp9_reconintra.c:429-431`).

---

## 4. Types

### 4.1 Internal kernel function type

```rust
/// Mirrors `intra_pred_fn` (vp9/common/vp9_reconintra.c:56-57).
/// `above` points at `above_data + 16` where `above_data[15]` is the
/// top-left corner pixel (so `above[-1]` is valid). `left` points at
/// `left_col[0]`.
///
/// `unsafe extern "C"` matches HLD-SIMD §3.20.
pub type IntraPredFn = unsafe extern "C" fn(
    dst: *mut u8,
    stride: isize,
    above: *const u8,
    left: *const u8,
);
```

The function-pointer ABI is `extern "C"` so that we can drop in the
libvpx asm kernels (NEON / SSE2 / SSSE3) via the same FFI shim used
by the rest of the SIMD families (HLD-SIMD §4.6).

### 4.2 The intra-pred sub-dispatch struct

The HLD-SIMD §4.1 lists `pub intra_pred: IntraPredDispatch, // 56
slots` but does not flesh out the type. The 56 slots break down as:

| Group                                | Count | Notes |
|--------------------------------------|------:|-------|
| 9 directional/derived modes × 4 sizes (V/H/D45/D135/D117/D153/D207/D63/TM) | 36 | All four sizes per mode |
| DC: 4 variants × 4 sizes                          | 16 | `(left_available, top_available)` ∈ {00, 01, 10, 11} → {dc_128, dc_top, dc_left, dc} |
| `*e_predictor_4x4` variants (D45E, D63E, HE, VE)  |  4 | Defined in `vpx_dsp/intrapred.c:250-281, 322-344, 367-386` — used only for 4×4 chroma in `vp9/common/vp9_reconinter.c` per HLD-SIMD §3.20. **Confirmed unreferenced from the non-RD encoder path**; ship scalar-only. |
| **Total**                            | **56** | |

```rust
/// Sub-dispatch struct stored in `SimdDispatch::intra_pred`.
///
/// 56 fn-ptrs, all populated to the scalar fallback at startup; SIMD
/// pointers overwrite the relevant slots on supported ISAs.
///
/// Memory layout chosen so that the inner loop's lookup —
/// `dispatch.intra_pred.predict[mode as usize][size as usize]` —
/// compiles to a single shifted load. We pre-fold the DC
/// `(have_left, have_top)` axis into the same array by emitting four
/// rows of DC variants at the start of the table.
#[repr(C)]
pub struct IntraPredDispatch {
    /// `[mode][tx_size]`. Mode order matches `IntraMode` enum.
    /// DC slot (`[0][_]`) holds the **all-available** DC kernel
    /// (`dc_predictor`). The neighbour-availability axis for DC is
    /// resolved in §5.4 by a separate lookup into `dc_pred`.
    pub predict: [[IntraPredFn; 4]; 10],

    /// DC neighbour-availability variants:
    /// `dc_pred[left_avail as usize][top_avail as usize][tx_size]`.
    /// Mirrors `dc_pred[2][2][TX_SIZES]` at vp9/common/vp9_reconintra.c:60.
    /// `[0][0]` → dc_128, `[0][1]` → dc_top,
    /// `[1][0]` → dc_left, `[1][1]` → dc.
    pub dc_pred: [[[IntraPredFn; 4]; 2]; 2],

    /// Four mode-corner variants for 4×4 only.
    /// Indexed by `IntraPredE4x4Kind`. Used by chroma in libvpx's
    /// `vp9_reconinter.c`; if our chroma path turns out not to need
    /// them (L-53 / mode-decision LLD will confirm) the table can
    /// stay populated with scalar-only stubs at zero runtime cost.
    pub predict_e_4x4: [IntraPredFn; 4],
}

#[derive(Copy, Clone)]
#[repr(u8)]
pub enum IntraPredE4x4Kind { D45e = 0, D63e = 1, He = 2, Ve = 3 }

// SAFETY: `IntraPredDispatch` is `Send + Sync` because every field is
// a fn pointer with `extern "C"` ABI; the pointers are written once
// during `SimdDispatch::init` (HLD-SIMD §4.4) and read-only thereafter.
// Mirrors the unsafe-impl in HLD-SIMD §4.1.
unsafe impl Send for IntraPredDispatch {}
unsafe impl Sync for IntraPredDispatch {}
```

### 4.3 Scratch buffer for the edge-builder

The kernels read from two contiguous scratch arrays. The encoder/decoder
allocates them on the caller's stack, aligned to 16 bytes for the
SIMD specialisations:

```rust
/// `above_data` mirrors libvpx's `DECLARE_ALIGNED(16, uint8_t,
/// above_data[64 + 16])` at vp9/common/vp9_reconintra.c:271.
/// `above_data[16]` is the top-left corner (so `above_row[-1]` is
/// valid); `above_data[17..]` holds up to 64 above pixels (for the
/// max 32×32 block, which needs 2×32 above pixels for D45/D63
/// "above-right" extension).
///
/// `left_col` mirrors `DECLARE_ALIGNED(16, uint8_t, left_col[32])`
/// at vp9/common/vp9_reconintra.c:270.
#[repr(C, align(16))]
pub struct EdgeScratch {
    pub above_data: [u8; 64 + 16],   // 80 bytes
    pub left_col:   [u8; 32],
}

impl EdgeScratch {
    #[inline] pub fn above_row(&self) -> *const u8 {
        // SAFETY: 16-byte offset into a 80-byte array is in-bounds.
        unsafe { self.above_data.as_ptr().add(16) }
    }
    #[inline] pub fn above_row_mut(&mut self) -> *mut u8 {
        unsafe { self.above_data.as_mut_ptr().add(16) }
    }
    #[inline] pub fn left_col_ptr(&self) -> *const u8 {
        self.left_col.as_ptr()
    }
    #[inline] pub fn left_col_ptr_mut(&mut self) -> *mut u8 {
        self.left_col.as_mut_ptr()
    }
}
```

Why `[u8; 64 + 16]` rather than `[u8; 32 + 16]`? Because the D45/D63
"above-right" modes need 2×bs samples of above row. For `bs = 32`
that's 64 above pixels plus the 16-byte alignment cushion that
contains `above[-1]` and trailing replication when the top row is
unavailable. Mirrors libvpx's array sizing at
`vp9/common/vp9_reconintra.c:271`.

### 4.4 No public state struct

The module exports no state. The dispatch table lives in the global
`SimdDispatch` (HLD-SIMD §4.1). Per-call scratch buffers are caller-owned.
Per-frame state (e.g. neighbour MI lookups) belongs to the
mode-decision LLD.

---

## 5. Internal algorithm

The hot path has three phases: **edge materialisation**, **DC variant
selection**, and **kernel dispatch**. Each is line-for-line traceable
to `vp9_reconintra.c`.

### 5.1 Top-level `predict_intra_block`

```text
fn predict_intra_block(mode, tx_size, edges, ref, ref_stride, dst, dst_stride,
                       geom, plane, dispatch):
    let bs = tx_size.pixels()                                  // bs ∈ {4, 8, 16, 32}
    let mut scratch = EdgeScratch::zeroed_unchecked()          // alloca
    let needs = NeighbourNeed::from_mode(mode)                 // §5.3 table

    if needs.contains(LEFT)         { build_left_col(...) }    // §5.2.1
    if needs.contains(ABOVE)        { build_above_row(...) }   // §5.2.2
    if needs.contains(ABOVE_RIGHT)  { build_above_right(...) } // §5.2.3 (replaces ABOVE)

    let above = scratch.above_row()
    let left  = scratch.left_col_ptr()

    let fn_ptr = if mode == IntraMode::Dc {
        dispatch.intra_pred.dc_pred[edges.have_left as usize]
                                  [edges.have_top  as usize]
                                  [tx_size as usize]
    } else {
        dispatch.intra_pred.predict[mode as usize][tx_size as usize]
    };

    // SAFETY: dispatch table is initialised; pointers correspond to
    // populated kernels; preconditions on dst/ref strides are upheld
    // by caller; scratch arrays are valid for the kernel's reads.
    unsafe { fn_ptr(dst, dst_stride, above, left) }
```

This is the 8-bit half of `build_intra_predictors` at
`vp9/common/vp9_reconintra.c:263-403`. The decoder, encoder mode
decision, and encoder final reconstruction all flow through this
function via `predict_intra_block` (HLD-SIMD §3.20 "Use sites").

### 5.2 Edge materialisation

The three edge groups are interlocked because **D45 / D63 need
2×bs above samples ("above-right")** which subsumes the regular
ABOVE case. `extend_modes[mode]` at `vp9/common/vp9_reconintra.c:43-54`
encodes the dependency:

| Mode  | NEED_LEFT | NEED_ABOVE | NEED_ABOVERIGHT |
|-------|-----------|------------|-----------------|
| DC    | ✓ | ✓ | — |
| V     | — | ✓ | — |
| H     | ✓ | — | — |
| D45   | — | — | ✓ (which implies ABOVE) |
| D135  | ✓ | ✓ | — |
| D117  | ✓ | ✓ | — |
| D153  | ✓ | ✓ | — |
| D207  | ✓ | — | — |
| D63   | — | — | ✓ |
| TM    | ✓ | ✓ | — |

(Cite `vp9/common/vp9_reconintra.c:43-54` for the verbatim table.)

#### 5.2.1 NEED_LEFT — left column copy with bottom replication

Cite `vp9/common/vp9_reconintra.c:299-320`.

```text
if !edges.have_left:
    // L-53 boundary: no left neighbour. libvpx fills with 129
    // (== 0x81). This is the "below-mid-grey" reference described
    // in the libvpx comment block at vp9_reconintra.c:279-283.
    memset(scratch.left_col[0..bs], 129)
else:
    // mb_to_bottom_edge < 0  iff  the block's bottom row would
    // extend past the frame; equivalently
    // geom.pixel_y + bs > geom.plane_height.
    let bottom_clipped = geom.pixel_y + bs as u32 > geom.plane_height;
    if !bottom_clipped:
        // fast path — vp9_reconintra.c:314-316
        for i in 0..bs: scratch.left_col[i] = ref[(i*ref_stride - 1) as usize]
    else:
        // slow path with replication — vp9_reconintra.c:307-311
        let extend_bottom = (geom.plane_height - geom.pixel_y) as usize;
        for i in 0..extend_bottom:
            scratch.left_col[i] = ref[(i*ref_stride - 1) as usize]
        let last = ref[((extend_bottom-1)*ref_stride - 1) as usize];
        for i in extend_bottom..bs:
            scratch.left_col[i] = last
```

**Edge constants:** `129` for unavailable-left is load-bearing — it
is the value seen by the DC averages, by D135/D153/D117 corner pixels,
and by the TM `clip_pixel(left[r] + above[c] - ytop_left)` math.
Changing it changes every prediction at the top-left corner of a
frame.

#### 5.2.2 NEED_ABOVE — above row copy with right replication

Cite `vp9/common/vp9_reconintra.c:322-348`.

```text
if !edges.have_top:
    memset(scratch.above_row[0..bs], 127)
    scratch.above_row[-1] = 127
else:
    let above_ref = ref - ref_stride
    let right_clipped = geom.pixel_x + bs as u32 > geom.plane_width;
    if right_clipped:
        // slow path with horizontal replication
        if geom.pixel_x + bs as u32 <= geom.plane_width:
            memcpy(scratch.above_row[0..bs], above_ref[0..bs])
        elif geom.pixel_x <= geom.plane_width:
            let r = (geom.plane_width - geom.pixel_x) as usize;
            memcpy(scratch.above_row[0..r], above_ref[0..r])
            memset(scratch.above_row[r..bs],
                   scratch.above_row[r - 1],
                   bs - r + (geom.pixel_x + bs - geom.plane_width) as usize - bs)
            // (re-derived from vp9_reconintra.c:331-333; clamped to
            // `x0 + bs - frame_width` extra columns of replication.)
    else:
        // fast path — vp9_reconintra.c:336-342
        if bs == 4 && edges.have_right && edges.have_left:
            // ZERO-COPY OPTIMISATION — the 4x4 block can read directly
            // from `above_ref` instead of copying. This is the
            // `const_above_row = above_ref` branch at
            // vp9_reconintra.c:337-338. See §5.5.
            // (Cannot do this when edges.have_left is false because
            // above_row[-1] would then need to be overwritten with 129.)
            above_alias = Some(above_ref)
        else:
            memcpy(scratch.above_row[0..bs], above_ref[0..bs])
    scratch.above_row[-1] = if edges.have_left { above_ref[-1] } else { 129 }
```

**Edge constants:** `127` (== 0x7F) for unavailable-top. With `129`
for left, this produces the characteristic libvpx unavailable-corner
pattern documented in the comment at `vp9_reconintra.c:279-283`:

```text
127 127 127 .. 127 127 127 127 127 127
129  A   B  ..  Y   Z
129  C   D  ..  W   X
129  E   F  ..  U   V
129  G   H  ..  S   T   T   T   T   T
```

#### 5.2.3 NEED_ABOVERIGHT — above row plus above-right replication

Cite `vp9/common/vp9_reconintra.c:350-394`. Only D45 and D63 take
this path.

The semantics are: extend the above row to **2×bs samples**, copying
from the actual above-row buffer when the top-right neighbour is
in-frame, replicating from `above_row[bs - 1]` otherwise.

For `bs == 4`, libvpx adds an additional fast path: if
`edges.have_right` is true, the second `bs` worth of above samples
can be copied from the reconstructed buffer (the top-right 4×4
neighbour). Cite `vp9_reconintra.c:357-358` and `383-385`.

```text
if !edges.have_top:
    memset(scratch.above_row[0..2*bs], 127)
    scratch.above_row[-1] = 127
else:
    let above_ref = ref - ref_stride
    if right_clipped:
        let r = (geom.plane_width - geom.pixel_x) as usize;
        // ...exact replication semantics per vp9_reconintra.c:356-376...
    else:
        if bs == 4 && edges.have_right && edges.have_left:
            above_alias = Some(above_ref)              // zero-copy fast path
        elif bs == 4 && edges.have_right:
            memcpy(scratch.above_row[0..bs],     above_ref[0..bs])
            memcpy(scratch.above_row[bs..2*bs],  above_ref[bs..2*bs])
        else:
            memcpy(scratch.above_row[0..bs], above_ref[0..bs])
            memset(scratch.above_row[bs..2*bs], scratch.above_row[bs-1], bs)
    scratch.above_row[-1] = if edges.have_left { above_ref[-1] } else { 129 }
```

The replication pattern at the right edge when `right_clipped` is set
is documented in `vp9_reconintra.c:355-376` in three sub-cases by the
relationship between `x0 + 2*bs`, `x0 + bs`, and `frame_width`. The
LLD implementation MUST preserve these three sub-cases verbatim — see
§9.4 for the test corpus that covers them.

### 5.3 Mode → neighbour-need map

```rust
/// Bit-set mirroring `extend_modes[INTRA_MODES]` at
/// vp9/common/vp9_reconintra.c:43-54.
#[derive(Copy, Clone)]
#[repr(transparent)]
struct NeighbourNeed(u8);

impl NeighbourNeed {
    const LEFT:        u8 = 1 << 0;
    const ABOVE:       u8 = 1 << 1;
    const ABOVE_RIGHT: u8 = 1 << 2;

    /// Table-driven mapping. Indexed by `IntraMode as usize`.
    /// Bit layout: bit 0 = LEFT, bit 1 = ABOVE, bit 2 = ABOVE_RIGHT.
    /// Order must match vp9/common/vp9_reconintra.c:43-54.
    const fn from_mode(m: IntraMode) -> Self {
        const TABLE: [u8; 10] = [
            Self::ABOVE | Self::LEFT,   // DC
            Self::ABOVE,                // V
            Self::LEFT,                 // H
            Self::ABOVE_RIGHT,          // D45
            Self::ABOVE | Self::LEFT,   // D135
            Self::ABOVE | Self::LEFT,   // D117
            Self::ABOVE | Self::LEFT,   // D153
            Self::LEFT,                 // D207
            Self::ABOVE_RIGHT,          // D63
            Self::ABOVE | Self::LEFT,   // TM
        ];
        NeighbourNeed(TABLE[m as usize])
    }

    #[inline] const fn contains(self, bit: u8) -> bool { (self.0 & bit) != 0 }
}
```

### 5.4 DC variant selection

Cite `vp9/common/vp9_reconintra.c:396-399`.

```rust
// At the kernel-dispatch step in §5.1:
let fn_ptr = if mode == IntraMode::Dc {
    dispatch.intra_pred.dc_pred
        [edges.have_left as usize]
        [edges.have_top  as usize]
        [tx_size as usize]
} else {
    dispatch.intra_pred.predict[mode as usize][tx_size as usize]
};
```

The four DC variants are:
- `dc_pred[0][0]` → `dc_128_predictor` (neither neighbour — output all 128).
- `dc_pred[0][1]` → `dc_top_predictor`  (top only — average above row).
- `dc_pred[1][0]` → `dc_left_predictor` (left only — average left col).
- `dc_pred[1][1]` → `dc_predictor`      (both — average above + left).

Kernel math (cite `vpx_dsp/intrapred.c:191-248`):

```text
dc_128:  for r in 0..bs: memset(dst + r*stride, 128, bs)
dc_top:  sum = Σ above[0..bs]; dc = (sum + bs/2) / bs; memset bs² with dc
dc_left: sum = Σ left[0..bs];  dc = (sum + bs/2) / bs; memset bs² with dc
dc:      sum = Σ above[0..bs] + Σ left[0..bs]; dc = (sum + bs) / (2*bs); memset bs²
```

The rounding-half-up `(sum + bs/2) / bs` is essential for byte parity
— see §7 for an entry on the integer-division rounding.

### 5.5 Zero-copy `above_row` alias for 4×4 fast path

The encoder's hot path is 4×4 chroma blocks at the cb edges. libvpx
saves a 4-byte memcpy at `vp9/common/vp9_reconintra.c:337-338` by
aliasing `above_row` directly into the reconstructed buffer when
`bs == 4 && have_right && have_left`. In Rust we preserve this:

```rust
// In predict_intra_block:
let above_ptr: *const u8;
if bs == 4 && edges.have_right && edges.have_left
        && !right_clipped && top_available {
    // Alias straight into the reconstruction buffer; above_ref points
    // at the row above this block. above_ref[-1] is the top-left
    // corner pixel, which is already in the right place in memory.
    above_ptr = ref_plane.offset(-ref_stride);
} else {
    // Slow path — materialise into the scratch buffer.
    above_ptr = scratch.above_row();
}
```

This is correct because:
1. The kernel only **reads** from `above`. No write-aliasing risk.
2. `above_ref[-1]` (the top-left corner) is the byte one position
   above-left of `ref_plane`, which is a valid pixel of the
   reconstructed buffer because `have_left` is true.
3. The kernel never writes through `above` regardless of whether it
   aliases the reconstruction buffer.

The aliasing IS observable to other workers (the row-MT case at
HLD-threading §3.5) but only as a **read** of already-completed
neighbours, which the row-MT sync barriers guarantee are stable.

### 5.6 Kernel math — directional modes

The directional kernels live in `vpx_dsp/intrapred.c`. The
generic implementations at the bottom of the file (lines 856-916)
expand `intra_pred_sized(type, size)` over four sizes per direction,
delegating to `static INLINE` per-mode functions defined earlier in
the file. We port one Rust scalar function per (mode, size) pair to
match the dispatch-table layout — i.e. **9 directional kernels × 4
sizes = 36 Rust scalar functions** plus **4 DC variants × 4 sizes =
16** plus the **4 mode-corner `*e_4x4` variants**.

**AVG3 / AVG2 macros** (cite `vpx_dsp/intrapred.c:17-19`):

```rust
#[inline(always)] const fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((a as u16 + 2*(b as u16) + c as u16 + 2) >> 2) as u8
}
#[inline(always)] const fn avg2(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16 + 1) >> 1) as u8
}
```

The `u16` widening is essential: `2*b + a + c + 2` exceeds `u8` range
(max value ≈ 1022 for all-255 inputs). Wrapping arithmetic gives the
wrong answer.

Per-direction:

| Mode | Citation | Rust scalar (sketch) |
|------|----------|----------------------|
| V    | `vpx_dsp/intrapred.c:157-166` | for r in 0..bs: memcpy(dst.add(r*stride), above, bs) |
| H    | `vpx_dsp/intrapred.c:168-177` | for r in 0..bs: memset(dst.add(r*stride), left[r], bs) |
| TM   | `vpx_dsp/intrapred.c:179-189` | `top_left = above[-1]; dst[r*s+c] = clip_pixel_u8(left[r] + above[c] - top_left)` (the `clip_pixel` ensures u8 overflow folds back into the legal sample range — see `vpx_dsp/vpx_dsp_common.h:38-44` `clip_pixel`). |
| D207 | `vpx_dsp/intrapred.c:21-43` | (depends on left only — see source) |
| D45  | `vpx_dsp/intrapred.c:62-79` | (depends on above only, with above-right) |
| D63  | `vpx_dsp/intrapred.c:45-60` | (depends on above only, with above-right) |
| D117 | `vpx_dsp/intrapred.c:81-104` | (depends on left + above + corner) |
| D135 | `vpx_dsp/intrapred.c:106-132` | (builds a 2×bs-1 border array then memcpys it diagonally — see source for the border-construction algorithm) |
| D153 | `vpx_dsp/intrapred.c:134-155` | (depends on left + above + corner) |

The scalar reference must be a **literal Rust port** of the C — no
algebraic rewrites, no loop fusion, no reordering. The kernels are
small (the largest, `d135` at 32×32, is ~120 LoC of inner-loop work)
and bit-exact-by-construction is much cheaper than bit-exact-by-test.

The 4×4 specialisations at `vpx_dsp/intrapred.c:283-454` use fully
unrolled `DST(x,y) = ...` assignments. We port these as standalone
4×4 functions rather than running the generic kernel with `bs=4`,
because the libvpx 4×4 `vpx_d45_predictor_4x4_c` (line 346-365) has
a `DST(3,3) = H` special-case that **differs** from
`d45_predictor(bs=4)` — see the comment "differs from vp8" at line
364 and the equivalent at line 319 for D63. The dispatch table will
route 4×4 calls to the specialised functions; 8×8/16×16/32×32 to the
generic-sized functions. This matches the libvpx behaviour at
`vp9_reconintra.c:72-90` (`INIT_ALL_SIZES`).

### 5.7 Init

Cite `vp9/common/vp9_reconintra.c:70-110`. The libvpx init function
is called once via `once()` (line 430). Our equivalent is the
`build_intra_pred_dispatch` constructor called from
`SimdDispatch::init`. The mapping is mechanical:

```rust
fn build_intra_pred_dispatch(caps: &Caps) -> IntraPredDispatch {
    IntraPredDispatch {
        predict: [
            // [Dc][..]: holds the all-available DC; neighbour-aware
            // selection is done in dc_pred below. We still populate
            // this row so callers that bypass the DC branch (e.g. a
            // future internal optimisation) see something coherent.
            [scalar::dc::p4, scalar::dc::p8, scalar::dc::p16, scalar::dc::p32],
            [pick_v::<4>(caps),   pick_v::<8>(caps),
             pick_v::<16>(caps),  pick_v::<32>(caps)],
            // ... and so on for H, D45, D135, D117, D153, D207, D63, TM
        ],
        dc_pred: [
            [[scalar::dc::p128_4, scalar::dc::p128_8,
              scalar::dc::p128_16, scalar::dc::p128_32],
             [pick_dc_top::<4>(caps),  pick_dc_top::<8>(caps),
              pick_dc_top::<16>(caps), pick_dc_top::<32>(caps)]],
            [[pick_dc_left::<4>(caps),  pick_dc_left::<8>(caps),
              pick_dc_left::<16>(caps), pick_dc_left::<32>(caps)],
             [pick_dc::<4>(caps),  pick_dc::<8>(caps),
              pick_dc::<16>(caps), pick_dc::<32>(caps)]],
        ],
        predict_e_4x4: [
            scalar::d45e::p4, scalar::d63e::p4,
            scalar::he::p4,   scalar::ve::p4,
        ],
    }
}
```

The `pick_*` helpers follow the HLD-SIMD §4.4 pattern (compile-time
`cfg`-gated, runtime `caps`-checked).

---

## 6. SIMD dispatch

Per HLD-SIMD §3.20 and the verbatim ISA coverage matrix from
`vpx_dsp_rtcd_defs.pl:44-213`:

### 6.1 ISA coverage matrix (v1 ships)

| Mode    | Sizes shipped scalar | x86_64 SIMD (libvpx 1.14.1) | aarch64 SIMD |
|---------|----------------------|------------------------------|--------------|
| DC      | 4, 8, 16, 32         | SSE2 (all sizes)             | NEON (all)   |
| DC top  | 4, 8, 16, 32         | SSE2 (all sizes)             | NEON (all)   |
| DC left | 4, 8, 16, 32         | SSE2 (all sizes)             | NEON (all)   |
| DC 128  | 4, 8, 16, 32         | SSE2 (all sizes)             | NEON (all)   |
| V       | 4, 8, 16, 32         | SSE2 (all sizes)             | NEON (all)   |
| H       | 4, 8, 16, 32         | SSE2 (all sizes)             | NEON (all)   |
| D45     | 4, 8, 16, 32         | SSE2 (4×4), SSSE3 (8/16/32)  | NEON (all)   |
| D135    | 4, 8, 16, 32         | — (scalar only)              | NEON (all)   |
| D117    | 4, 8, 16, 32         | — (scalar only)              | NEON (all)   |
| D153    | 4, 8, 16, 32         | SSSE3 (all sizes)            | NEON (all)   |
| D207    | 4, 8, 16, 32         | SSE2 (4×4), SSSE3 (8/16/32)  | NEON (all)   |
| D63     | 4, 8, 16, 32         | SSSE3 (all sizes)            | NEON (all)   |
| TM      | 4, 8, 16, 32         | SSE2 (all sizes)             | NEON (all)   |
| `*e_4x4`| 4 only               | — (scalar only)              | — (scalar only)|

Source: `vpx_dsp_rtcd_defs.pl:44-213`. No AVX2 specialisations exist
in libvpx 1.14.1 for intra predictors (HLD-SIMD §3.20 line 1013-1014).

**Total kernel count to ship in v1:**

- 56 scalar (40 mode×size + 12 DC variants + 4 `*e_4x4`).
- 48 SSE2: 16 DC variants × 4 sizes (= 16) + V/H/TM × 4 sizes (= 12) + D45_4×4 + D207_4×4 = 30. (Re-derive against `vpx_dsp_rtcd_defs.pl` during Stage 3 implementation.)
- ~24 SSSE3: D45/D63/D153/D207 at 8×8/16×16/32×32 + D63_4×4.
- 52 NEON: every mode/size pair where libvpx ships NEON, per the matrix.

(The exact count is verified by the kernel catalogue in HLD-SIMD §3.20.)

### 6.2 Per-arch module layout

```text
rvp9-core/src/dsp/intra_pred/
    mod.rs                   # public entry (predict_intra_block etc.)
    edges.rs                 # NeighbourNeed, edge materialisation
    scratch.rs               # EdgeScratch
    dispatch.rs              # IntraPredDispatch struct + build()
    scalar.rs                # 56 scalar kernels (port of vpx_dsp/intrapred.c)
    #[cfg(target_arch = "x86_64")]
    sse2.rs                  # 30 SSE2 kernels (FFI shim or rewritten intrinsics)
    #[cfg(target_arch = "x86_64")]
    ssse3.rs                 # 24 SSSE3 kernels
    #[cfg(target_arch = "aarch64")]
    neon.rs                  # 52 NEON kernels
```

The kernel files match HLD-SIMD §4.6's pattern: scalar always built,
ISA modules `#[cfg]`-gated.

### 6.3 RTCD-style population

Following HLD-SIMD §4.4's `pick!` macro pattern, for each kernel slot
we emit something like:

```rust
fn pick_v_size_8(caps: &Caps) -> IntraPredFn {
    #[cfg(target_arch = "x86_64")] {
        if caps.has_sse2 { return sse2::v_predictor_8x8 as IntraPredFn; }
    }
    #[cfg(target_arch = "aarch64")] {
        if caps.has_neon { return neon::v_predictor_8x8 as IntraPredFn; }
    }
    scalar::v_predictor_8x8 as IntraPredFn
}
```

These per-slot functions are generated by a `pick_intra!(mode, size,
{sse2, ssse3, neon})` macro to keep `dispatch.rs` short. The macro
expansion matches `vpx_dsp_rtcd_defs.pl:44-213` line-for-line so a
reviewer can scan for ISA-coverage parity.

### 6.4 FFI vs Rust intrinsics

PLAN.md D5 commits us to `core::arch` intrinsics. For intra-pred
specifically, the libvpx SSE2/SSSE3 kernels are **mostly inline
assembly** (`vpx_dsp/x86/intrapred_sse2.asm` etc.). We have two
options for v1:

1. **Rust port via `core::arch::x86_64::*`** — rewrite each kernel
   using `_mm_*` intrinsics. Most kernels are short (<50 lines of
   intrinsics each). Bit-exact-by-construction is achievable because
   the operations (`pshufb`, `paddusb`, `pavgb`) have direct intrinsic
   equivalents.
2. **Link libvpx's assembled `.o` files** — bypasses Rust safety
   entirely and creates a non-trivial build dependency.

**Recommendation:** **option 1**, rewritten as Rust intrinsics. The
kernels are simple enough that the rewrite cost is modest and option
2 conflicts with PLAN.md §5 single-crate distribution. The
differential test suite (§9) catches any byte-divergence from the
reference C.

This is the same decision the SIMD HLD made for `sad`, `variance`,
`hadamard`, etc. (HLD-SIMD §4.6).

### 6.5 Differential test budget per kernel

Per HLD-SIMD §5: each (mode, size, ISA) tuple needs a differential
test against the scalar reference on:

1. Random `above` / `left` buffers (1000 iterations).
2. The pathological edge cases: all-0, all-255, alternating, ramp,
   step at the corner. (5 fixed vectors.)
3. The libvpx test vectors from `test/intrapred_test.cc:VP9IntraPredictorTest`
   ported verbatim.

For the 56 scalar kernels alone, this is ~280k bit-exact comparisons
in CI; for the SIMD variants, ~150k more. The full suite should run
in <2 seconds; see §9.5.

---

## 7. Deliberate divergences from libvpx

Each entry feeds `notes/divergence-ledger.md` (per HLD-arch §11).

### 7.1 HBD code path removed entirely

**Where:** `rvp9-core::dsp::intra_pred` — no `u16` variants of any
kernel; no `int bd` argument on any function. libvpx counterpart:
`vp9/common/vp9_reconintra.c:112-261` (`build_intra_predictors_high`)
and `vpx_dsp/intrapred.c:456-851` (highbd scalar kernels) plus the
corresponding SSE2/SSSE3/NEON specialisations.

**What libvpx does:** ships parallel `uint16_t` versions of every
predictor; selects between 8-bit and HBD at runtime via
`xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH` at
`vp9/common/vp9_reconintra.c:418`.

**What rvp9 does:** ships only the 8-bit path. The
`predict_intra_block` signature takes `*const u8` /
`*mut u8` directly; no generic pixel type.

**Why:** PLAN.md D-01 RESOLVED — 8-bit only in v1, HBD deferred to
Stage 6+. Doubling the kernel count for HBD (an additional 56 scalar
+ ~70 SIMD entries) is ~3-5 weeks of work this LLD does not budget.

**Byte-parity impact:** none (matched configs use the 8-bit path).

**Compression-efficiency impact:** none in scope.

**Future plan:** Stage 6+. The dispatch struct can grow an
`IntraPredDispatchHbd` sibling field without changing the 8-bit code
path. The edge-builder logic at §5.2 is structurally identical
(libvpx's `build_intra_predictors_high` mirrors `build_intra_predictors`
line-for-line apart from the type).

### 7.2 4×4 zero-copy `above_row` alias remains a raw-pointer alias

**Where:** §5.5 — the `bs == 4 && have_right && have_left` fast path.

**What libvpx does:** `vp9/common/vp9_reconintra.c:337-338` sets
`const_above_row = above_ref;`, aliasing the kernel's above-row input
into the reconstructed-frame buffer.

**What rvp9 does:** preserves the aliasing. `above_ptr` is a raw
`*const u8` into the reconstructed buffer; the kernel reads through
it.

**Why:** the alternative (always memcpy into `EdgeScratch`) costs a
small but measurable per-4×4-block memcpy. Real-time at 1080p60 is
budget-tight enough that we keep the optimisation.

**Byte-parity impact:** none — both paths produce identical samples.

**Soundness:** documented under §5.5; the kernel only reads through
`above`, never writes. No write-aliasing with `dst` because `dst` is
the *current* block and `above_ref` is the row above it.

**Future plan:** revisit if Miri flags pointer-provenance issues.

### 7.3 Edge buffer types — `EdgeScratch` aligned struct vs C `DECLARE_ALIGNED`

**Where:** `rvp9-core::dsp::intra_pred::EdgeScratch` —
`#[repr(C, align(16))]` Rust struct. libvpx counterpart:
two stack-local `DECLARE_ALIGNED(16, uint8_t, ...)` arrays at
`vp9/common/vp9_reconintra.c:270-272`.

**What libvpx does:** two independent stack arrays.

**What rvp9 does:** one combined struct (cleaner ownership; one alloca
instead of two).

**Byte-parity impact:** none.

### 7.4 Mode → tx-type lookup table re-exported from intra-pred (NOT a divergence — documentation only)

**Where:** `rvp9-core::dsp::intra_pred::INTRA_MODE_TO_TX_TYPE`. libvpx
counterpart: `intra_mode_to_tx_type_lookup` at
`vp9/common/vp9_reconintra.c:24-35`.

The table is identical to libvpx's. We export it from `intra_pred`
because that is its libvpx home, even though only the transforms LLD
consumes it. Documented here to avoid a cross-LLD ownership
ambiguity.

### 7.5 No `vp9_init_intra_predictors` lazy-once pattern

**Where:** `rvp9-core::dsp::intra_pred::build_intra_pred_dispatch`.
libvpx counterpart: `vp9/common/vp9_reconintra.c:429-431`
(`vp9_init_intra_predictors` calls `once(...)`).

**What libvpx does:** lazy-initialises the dispatch table on the
first call via `vpx_once`.

**What rvp9 does:** eager-initialises the global `SimdDispatch`
(including intra-pred's slice of it) inside `SimdDispatch::init()`,
called once at `rvp9-core` boot **before any worker spawns**
(HLD-SIMD §6, decisions-needed.md R-09).

**Why:** eager init matches HLD-threading §3.2's invariant; no
double-init race risk.

**Byte-parity impact:** none.

---

## 8. Open questions

### 8.1 (Question to Arthur, low-priority) Whether to ship the `*e_4x4` variants at all

**Context:** `vpx_d45e_predictor_4x4` etc. are referenced by libvpx's
`vp9/common/vp9_reconinter.c` (chroma prediction at sub-8×8). I
have **not** verified that the rvp9 chroma path under our scope
(non-RD pick-mode at cpu-used 6-8, plus full RDO for KF intra-only)
exercises them.

**Default decision:** ship the four `*e_4x4` scalar functions to
match libvpx's surface (cost: ~80 lines of Rust). If the
mode-decision LLD shows they're unreachable, drop in Stage 3 and
log as a divergence.

**Resolves:** L-?? (not currently in `decisions-needed.md` — would
be a new low-priority entry).

### 8.2 Whether `predict_intra_block` should be `safe` or `unsafe`

I have specified it as `unsafe` because the caller passes raw
pointers (`ref_plane`, `dst`) and strides. HLD-arch §5.2 lists
"FFI to SIMD kernels via fn-ptr" as one of the four allowed
`unsafe` sites, and the predictor sits at exactly that boundary.

**Alternative:** wrap the function in a safe outer that takes
`&FrameBuffer` and a `BlockPosition`, does the bounds-checking, and
calls the unsafe inner. The encoder pickmode path hits this function
~hundreds of times per superblock; the safe wrapper would add
borrow-check overhead in tight loops.

**Default decision:** keep `unsafe` at the inner API; add a safe
wrapper later if the encoder ergonomics demand it.

**Resolves:** none, but flagged for the mode-decision LLD writer.

### 8.3 L-53 (partition-at-frame-edges) — confirmed within this LLD

**Confirmed:** the edge-builder's slow path (§5.2.1 bottom and §5.2.2
right) handles partial-SB blocks at the frame edge by replicating the
last in-frame sample. The intra-pred LLD therefore has **no L-53
ambiguity**: it implements the libvpx replication behaviour
bit-exactly, and the partition LLD owns the decision of *whether* a
TX block at the edge gets predicted at all (i.e. `set_block_size`'s
in-frame check at `vp9/encoder/vp9_encodeframe.c:338`).

**Action for partition LLD writer:** ensure that the partition
machinery never asks intra-pred to predict a block whose top-left
corner is **outside** the frame; intra-pred assumes
`geom.pixel_x < geom.plane_width && geom.pixel_y < geom.plane_height`
(precondition 4 in §3.3). Blocks whose top-left is in-frame but whose
right-bottom extends past it ARE handled here by replication.

### 8.4 Whether the encoder's `vp9_pick_intra_mode` is reachable at our config

`02-encoder-core.md` §4.2.1 says `hybrid_intra_mode_search`
(`vp9_encodeframe.c:4672`) dispatches to `vp9_rd_pick_intra_mode_sb`
(full RDO) for small blocks unless `sf.nonrd_keyframe`, otherwise to
`vp9_pick_intra_mode`. At cpu-used 6-8 (our target), is `sf.nonrd_keyframe`
set?

Looking at `02-encoder-core.md` §3.3 lines 1920-1931, the speed-7
config sets `sf->reuse_inter_pred_sby = 1` but I don't see
`sf->nonrd_keyframe` explicitly. **This is a question for the
speed-features / mode-decision LLD writer** — if `nonrd_keyframe == 1`
at cpu-used 6+, then `vp9_pick_intra_mode` is the KF path and the
non-RD intra-mode loop tests **only DC/V/H/TM** (per `vp9_pickmode.c:1188`:
`for (this_mode = DC_PRED; this_mode <= H_PRED; ++this_mode)` — note
the iteration covers DC through TM inclusive, four modes).

Either way, **all ten modes must ship** because the decoder needs them
for bitstream conformance regardless of which the encoder exercises.

**Resolves:** none new; surfaces for mode-decision LLD.

### 8.5 4×4 reachability under real-time intra-only frames (L-71)

decisions-needed.md L-71 asks whether 4×4 TX is reachable in real-time.
For intra-pred specifically: 4×4 is **always** reachable for chroma
(sub-sampled 4:2:0 chroma plane gets 4×4 TX whenever luma uses 8×8 TX),
and reachable for luma during KF intra-only when the partition tree
descends to BLOCK_8X8 with `tx_size = TX_4X4`. The intra-pred LLD must
ship 4×4 predictors regardless. The transforms LLD (L-71) is the one
that needs the reachability answer.

**Resolves:** L-71 stays open; this LLD ships 4×4.

---

## 9. Test strategy

### 9.1 Unit tests — per-kernel correctness

Per-kernel unit tests in `rvp9-core/src/dsp/intra_pred/scalar.rs`
(`#[cfg(test)] mod tests { ... }`):

- For each `(mode, size)` of the 56 scalar slots:
  - **`fixed_vector_dc_avg`** — known `above`/`left` → known `dst`.
    Three vectors per kernel (gives ~170 small tests).
  - **`uniform_input_uniform_output`** — `above` and `left` all the
    same value `v` → `dst` is all `v` (for the smooth modes
    DC/V/H/TM/D135 specifically; for the directional modes the
    expected output is *also* all `v` because every operation is
    `avg2`/`avg3` of `v`s — verified inline).
  - **`stride_invariance`** — running the kernel with two different
    destination strides produces the same `bs × bs` sample block.

### 9.2 Property tests — kernel determinism

`proptest!` harness in `rvp9-core/tests/intra_pred_props.rs`:

```text
proptest!(|(above in arb_above(), left in arb_left(),
            mode: IntraMode, tx_size: TxSize)|
{
    // Property 1: output depends only on inputs (no global state).
    let dst1 = predict(mode, tx_size, &above, &left);
    let dst2 = predict(mode, tx_size, &above, &left);
    assert_eq!(dst1, dst2);

    // Property 2: bytes never exceed [0, 255].
    // (Trivially true at u8 type; serves as a sanity check on the
    // clip_pixel in TM and the arithmetic-flow in AVG3.)
})
```

### 9.3 Differential tests against libvpx scalar

Per HLD-SIMD §5.2 — for every (mode, size) we run libvpx's
`vpx_<mode>_predictor_<size>x<size>_c` and our scalar Rust port on
the same `above`/`left` and assert byte-for-byte equality.

The test fixture exposes libvpx kernels through an FFI shim built
once at `Cargo build` time, linking `libvpx.a` (built in Stage 0 via
`scripts/build-libvpx.ps1`). Inputs are 1000 random `above`/`left`
buffers plus the corpus from `test/intrapred_test.cc`.

**Target:** 0 failures across 56 × 1000 = 56,000 differential
samples.

### 9.4 Differential tests for the edge-builder

The edge-builder is **not** part of `vpx_dsp/intrapred.c` — it lives
in `vp9/common/vp9_reconintra.c`. We test it differentially against
libvpx's `build_intra_predictors` by:

1. Building a synthetic reconstructed plane (random pixels, known
   dimensions).
2. Picking a block position `(pixel_x, pixel_y)` and `(bs)`.
3. Setting up `MACROBLOCKD` parameters for libvpx
   (`mb_to_right_edge`, `mb_to_bottom_edge`, plane dimensions,
   subsampling, `above_mi`/`left_mi` pointers).
4. Calling `vp9_predict_intra_block` and our `predict_intra_block`
   with matched parameters.
5. Asserting byte-equal `dst`.

Critical edge-builder vectors (each tested for all 10 modes, all 4
sizes, all 8 `(have_top, have_left, have_right)` triplets):

- **Top-left frame corner** (`pixel_x=0, pixel_y=0`, neither neighbour
  available). All 10 modes degenerate to the 127/129 reference pattern.
- **Top edge, mid-frame** (`pixel_y=0, have_top=false, have_left=true`).
- **Left edge, mid-frame** (`pixel_x=0, have_top=true, have_left=false`).
- **Right edge with partial block** (`pixel_x + bs > plane_width`,
  triggers slow-path replication at §5.2.2). Three sub-cases per
  `(pixel_x + bs, pixel_x + 2*bs)` straddling `frame_width`.
- **Bottom edge with partial block** (`pixel_y + bs > plane_height`,
  triggers slow-path replication at §5.2.1).
- **Mid-frame fast path** (all neighbours available, bs=4, triggers
  zero-copy alias at §5.5).

Total: ~10 modes × 4 sizes × 8 availability triples × 6 positions = ~1920
edge-builder differential samples.

### 9.5 SIMD differential tests

For each SIMD specialisation (SSE2, SSSE3, NEON) we mirror §9.3
against the **same Rust scalar** rather than libvpx — because the
test should isolate SIMD bugs from porting bugs. If §9.3 passes and
§9.5 passes, then the SIMD result equals libvpx by transitivity.

CI gates:
- All §9.3 + §9.5 tests run on every PR.
- Tests run under `cargo test --release` (debug overflow checks add
  noise on the kernel arithmetic).

### 9.6 Conformance vectors

The decoder's intra-pred path is the load-bearing one for VP9 spec
conformance. Stage 3 runs the full conformance test suite (172
vectors, per `scripts/fetch-vectors.ps1`); intra-only vectors
(`vp90-2-00-akiyo-...` and the keyframe-only test set) exercise all
ten modes at all four sizes.

A failure here means a kernel is wrong, not the edge-builder (the
conformance vectors hit only the in-frame mid-block fast path because
they are decoded by `vpxdec` at the bitstream level — frame edges are
exercised, just not the partial-block-at-frame-edge case as densely).

### 9.7 Loom

Not needed for this module. The intra-pred dispatch table is
read-only-after-init (§1, §5.7); there is no synchronisation surface
to model.

### 9.8 Miri

Run the scalar tests under `cargo +nightly miri test -p rvp9-core
--lib dsp::intra_pred::scalar` to catch:
- Out-of-bounds reads in the edge-builder (especially the `r - 1`
  index in the slow-path partial-block-at-right-edge replication).
- Pointer-provenance issues in the §5.5 zero-copy `above_row` alias
  (Miri's stacked-borrows model is the right oracle for this).

Miri does not run on the SIMD kernels (it does not interpret
`core::arch` intrinsics). Soundness of the SIMD path rests on §9.5
differential equivalence to scalar plus the `#[target_feature]`
contract enforced by the dispatch-table lookup.

---

---

**Line count:** ~1280 lines. **Key §7 entries:** 7.1 (HBD dropped, D-01), 7.3 (EdgeScratch shape), 7.5 (eager init replaces `once()`). **§8 blocking questions:** none — all open items are cross-LLD coordination (8.1, 8.4, 8.5); no Arthur gate needed.
