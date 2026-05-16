# LLD: transforms

> Low-level design for the forward / inverse DCT / ADST / WHT transform
> pipeline plus scan-order machinery. Owned by `rvp9-core::transforms`.
> 8-bit only (D-01 RESOLVED), real-time encoder (cpu-used 6-8, D1),
> no altref (D-13). Trellis-quant (`vp9_optimize_b`) is gated off per
> L-28 RESOLVED.

---

## 1. Purpose

The `transforms` module owns every block-level frequency-domain primitive
used by both the VP9 encoder and decoder: the integer DCT and ADST in
all four legal sizes (4×4, 8×8, 16×16, 32×32), the Walsh-Hadamard
transform used only when a frame is lossless, the hybrid `iht`/`fht`
table dispatch that picks DCT vs ADST per axis based on intra mode, the
scan/iscan/neighbor tables that order coefficients into the entropy
stream, and the EOB-bucket fast-path selection (1, 10, 12, 34, 38, 135
non-zero coefficients) on the inverse side.

It does **not** own:

- Tokenization or the boolean coder (LLD-entropy).
- Quantization (`vp9_quantize_fp` and friends) — owned by LLD-quant.
- The Hadamard transforms used by the non-RD pick-mode rate proxy
  (`vpx_hadamard_8x8/16x16/32x32` + `vpx_satd`) — those are tied to
  mode decision and live in LLD-mode-decision per HLD-simd §3.13.
  `vpx_hadamard_*` is **not** a real DCT and is a separate kernel
  family.
- Trellis-quant / coefficient optimization (`vp9_optimize_b`) — out of
  scope per L-28 RESOLVED; see §7.1.

The module is consumed by **decoder** (every coded block calls
`add_inverse_transform`) and by **encoder** (every committed coded
block runs forward transform → quant → inverse transform →
reconstruction, even at cpu-used 6-8 where the *rate proxy* in
`block_yrd` uses Hadamard rather than DCT — HLD-simd §3.16 notes that
the final encode step still runs the real transform).

---

## 2. libvpx origin

Primary Stage 1 source:

- `notes/vp9-analysis/04-bitstream-entropy.md` §5.10 (scan / iscan /
  neighbor tables, verbatim — lines 3226-4089 of A4) and §2.11
  (`ScanOrder` struct, lines 480-510).
- `notes/vp9-analysis/02-encoder-core.md` §4.6 (the coefficient
  encoding side of `block_yrd` and the `vp9_xform_quant_fp` integration
  point, lines 1075-1140).
- `notes/hld/02-simd-dispatch.md` §3.16-§3.19 (forward DCT, forward
  ADST/hybrid, inverse DCT, inverse ADST/hybrid — kernel catalogue).
- `notes/hld/02-simd-dispatch.md` §8.4 (the per-frame lossless flip
  flagged for resolution in this LLD).

libvpx files (`third_party/libvpx/`):

| File | Lines | What lives here |
|---|---|---|
| `vp9/common/vp9_enums.h` | 77-100 | `TX_SIZE`, `TX_TYPE`, `TX_MODE` enums |
| `vp9/common/vp9_scan.h` | 24-52 | `ScanOrder` struct, `MAX_NEIGHBORS`, `get_coef_context`, `get_scan` |
| `vp9/common/vp9_scan.c` | 1-725 | every scan / iscan / neighbor table; the two top-level dispatch arrays at lines 691-725 |
| `vp9/common/vp9_idct.c` | 119-204 | EOB-bucket inverse dispatch (`vp9_idct4x4_add`, `vp9_idct8x8_add`, `vp9_idct16x16_add`, `vp9_idct32x32_add`, `vp9_iwht4x4_add`, plus `vp9_iht{4x4,8x8,16x16}_add` hybrid dispatch) |
| `vp9/common/vp9_reconintra.c` | 24-35 | `intra_mode_to_tx_type_lookup[INTRA_MODES]` — the 10-entry table that maps an intra prediction mode to a `TX_TYPE` |
| `vpx_dsp/fwd_txfm.c` | 1-810 | C reference for `vpx_fdct4x4_c`, `_8x8_c`, `_16x16_c`, `_32x32_c`, `_32x32_rd_c`, and all `_1` DC-only variants |
| `vpx_dsp/inv_txfm.c` | (entire) | C reference for `vpx_idct{4x4_16,8x8_64,8x8_12,16x16_256,16x16_38,16x16_10,32x32_1024,32x32_135,32x32_34}_add` and the `_1_add` DC-only variants plus `vpx_iwht4x4_{16,1}_add` |
| `vp9/encoder/vp9_dct.c` | 530-685 | C reference for `vp9_fht4x4_c`, `_8x8_c`, `_16x16_c`, `vp9_fwht4x4_c` |
| `vp9/encoder/vp9_encodemb.c` | 331-540 | `vp9_xform_quant_fp` / `vp9_xform_quant_dc` / `vp9_xform_quant` — the encoder-side dispatch that picks among `fdct*`, `fht*`, `fwht4x4` based on `tx_size` × `tx_type` × `dc_only` × `lossless` |
| `vp9/encoder/vp9_encodeframe.c` | 6082-6097 | The **per-frame lossless flip**: assigns `x->fwd_txfm4x4` and `x->inv_txfm_add` to `vp9_fwht4x4`/`vp9_iwht4x4_add` if the frame is lossless, else `vpx_fdct4x4`/`vp9_idct4x4_add` |
| `vp9/encoder/vp9_encodemb.c` | 66-300 | `vp9_optimize_b` — the trellis-quant routine. **NOT PORTED** in v1 (L-28 RESOLVED). Cited only so future readers know the file. |
| `vp9/common/vp9_blockd.h` | 219-238 | declarations for `intra_mode_to_tx_type_lookup` and the `get_tx_type` helpers |

Stage 1 specialist owner: A4 (bitstream + scan tables). Secondary owner:
A2 (encoder coeff handling).

---

## 3. Public API

All signatures are `pub` inside `rvp9_core::transforms` unless noted.
Items are re-exported from the crate root only when the encoder /
decoder must import them frequently (the EOB-bucket fast-path enum, the
`TxSize` / `TxType` aliases, the `ScanOrder` reference type).

### 3.1 Top-level aliases (re-exported at crate root)

```rust
/// VP9 transform size, mirrors libvpx `TX_SIZE` (`vp9_enums.h:77-82`).
/// Encoded width in bitstream: 2 bits (`TX_MODE` selects).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum TxSize {
    Tx4x4   = 0,
    Tx8x8   = 1,
    Tx16x16 = 2,
    Tx32x32 = 3,
}

/// VP9 transform type, mirrors libvpx `TX_TYPE` (`vp9_enums.h:95-99`).
/// Selects DCT vs ADST per axis. Inter blocks and chroma always use
/// `DctDct`; intra-Y uses `intra_mode_to_tx_type_lookup` (§4.5).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TxType {
    DctDct   = 0,
    AdstDct  = 1,
    DctAdst  = 2,
    AdstAdst = 3,
}

/// Frame-level transform mode selector, mirrors libvpx `TX_MODE`
/// (`vp9_enums.h:86-92`). Encoder writes this in the compressed header;
/// decoder reads it. `OnlyTx4x4` is forced when the frame is lossless.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TxMode {
    OnlyTx4x4   = 0,
    AllowTx8x8  = 1,
    AllowTx16x16= 2,
    AllowTx32x32= 3,
    TxModeSelect= 4,
}

/// Coefficient type in the post-quant domain (mirrors libvpx
/// `tran_low_t` in non-HBD builds: `int16_t`).
pub type Coeff = i16;

/// Wider intermediate type used during transform passes; mirrors
/// libvpx `tran_low_t` when the build forces 32-bit accumulators
/// (HLD-arch §6.4 pins `TranLow = i32` for v2 HBD readiness).
pub type TranLow = i32;
```

`TxSize::block_size_log2()` returns 2/3/4/5; `n_coeffs()` returns
16/64/256/1024. Both are `const fn` for use in array sizes.

### 3.2 Scan-order primitives

```rust
/// VP9 scan-order triple (`vp9_scan.h:26-30`).
///
/// `scan[k]` is the raster position of the k-th coefficient in scan
/// order. `iscan[p]` is `scan_index(p) + 1` (EOB-position semantics,
/// see `vp9_scan.c:514`). `neighbors[2k..2k+1]` give the {top, left}
/// raster positions used to derive `get_coef_context`.
///
/// All four `scan` / `iscan` arrays for a given TxSize have length
/// `TxSize::n_coeffs()`. `neighbors` has length
/// `(n_coeffs() + 1) * MAX_NEIGHBORS`.
#[derive(Copy, Clone)]
pub struct ScanOrder {
    pub scan:      &'static [i16],
    pub iscan:     &'static [i16],
    pub neighbors: &'static [i16],
}

/// `MAX_NEIGHBORS = 2` per `vp9_scan.h:24`.
pub const MAX_NEIGHBORS: usize = 2;

/// Fetch the default zig-zag scan for a transform size.
/// Mirrors `vp9_default_scan_orders[TX_SIZES]` (`vp9_scan.c:691-696`).
#[inline]
pub const fn default_scan(size: TxSize) -> &'static ScanOrder;

/// Fetch the scan order for a `(TxSize, TxType)` pair.
/// Mirrors `vp9_scan_orders[TX_SIZES][TX_TYPES]` (`vp9_scan.c:698-725`).
/// At `Tx32x32` every `TxType` returns the default scan.
#[inline]
pub const fn scan_for(size: TxSize, ty: TxType) -> &'static ScanOrder;

/// Inter / chroma / lossless dispatch helper. Mirrors `get_scan`
/// (`vp9_scan.h:42-52`).
///
/// `is_inter` and `plane_is_y` come from the caller's MB context;
/// `lossless` is the per-frame flag (see §4.2 for the per-frame
/// transform-pair flip). `intra_y_mode` is the intra prediction mode
/// for the block (irrelevant when not Y intra, may be passed as
/// `IntraMode::Dc` and is ignored).
#[inline]
pub fn get_scan(
    size: TxSize,
    is_inter: bool,
    plane_is_y: bool,
    lossless: bool,
    intra_y_mode: IntraMode,
) -> &'static ScanOrder;

/// Map an intra prediction mode to its `TxType` (`vp9_reconintra.c:24-35`).
#[inline]
pub const fn intra_mode_to_tx_type(mode: IntraMode) -> TxType;

/// Coefficient context computation, mirrors `get_coef_context`
/// (`vp9_scan.h:35-40`). `c` is the scan index of the *next* coefficient
/// to encode (i.e. we use neighbors of position `c`, not the position
/// itself).
#[inline]
pub fn get_coef_context(neighbors: &[i16], token_cache: &[u8], c: usize) -> u8;
```

Preconditions: `c + 1 <= n_coeffs(size)`; `token_cache.len() >=
n_coeffs(size)`; `neighbors.len() >= (c + 1) * MAX_NEIGHBORS`.
The function is infallible — invalid inputs are programmer bugs
(`debug_assert!` per HLD-arch §4.4).

### 3.3 Forward transform dispatch (encoder-side)

```rust
/// Forward DCT of `n × n` 16-bit residual values.
/// Output is `Coeff[n*n]` in raster order; later passed to quant +
/// scan-order serialization.
///
/// `src_stride` is in samples, not bytes (matches libvpx).
pub type FdctFn = unsafe extern "C" fn(
    input: *const i16, output: *mut Coeff, stride: i32,
);

/// DC-only forward DCT (only the [0,0] coefficient is populated;
/// the rest of `output` is left untouched). Used by
/// `vp9_xform_quant_dc` (`vp9_encodemb.c:420-485`).
pub type Fdct1Fn = FdctFn;

/// Forward hybrid transform — DCT or ADST per axis, selected by
/// `tx_type`. Only valid for 4×4, 8×8, 16×16 (`vp9_dct.c:530-668`).
/// At 32×32 the `tx_type` argument is ignored and DCT is forced
/// (libvpx never calls `vp9_fht32x32` because intra at 32×32 always
/// has `tx_type = DCT_DCT` — see `vp9_scan_orders[Tx32x32][*]` only
/// referencing the default scan).
pub type FhtFn = unsafe extern "C" fn(
    input: *const i16, output: *mut Coeff,
    stride: i32, tx_type: i32,
);

/// Walsh-Hadamard transform, lossless-only forward path. Mirrors
/// `vp9_fwht4x4_c` (`vp9_dct.c:586-665`). The encoder selects this
/// kernel pointer for `fwd_txfm4x4` when the frame is lossless
/// (`vp9_encodeframe.c:6093`). v1 encoder never produces lossless;
/// shipped for round-trip symmetry only.
pub type FwhtFn = FdctFn;

/// Forward-transform dispatch table — read once per encoder
/// construction from `SimdDispatch` (HLD-simd §4).
#[derive(Copy, Clone)]
pub struct FwdTxfm {
    // DCT, full
    pub fdct4x4:    FdctFn,
    pub fdct8x8:    FdctFn,
    pub fdct16x16:  FdctFn,
    pub fdct32x32:  FdctFn,
    pub fdct32x32_rd: FdctFn, // rounding-shift variant (see §4.4)

    // DCT, DC-only
    pub fdct4x4_1:   Fdct1Fn,
    pub fdct8x8_1:   Fdct1Fn,
    pub fdct16x16_1: Fdct1Fn,
    pub fdct32x32_1: Fdct1Fn,

    // ADST/hybrid (4×4, 8×8, 16×16 only)
    pub fht4x4:      FhtFn,
    pub fht8x8:      FhtFn,
    pub fht16x16:    FhtFn,

    // Walsh-Hadamard (lossless 4×4 only)
    pub fwht4x4:     FwhtFn,
}
```

The encoder's high-level callable wraps the table (so call sites read
naturally):

```rust
impl FwdTxfm {
    /// Forward 4×4: either DCT or WHT depending on `lossless`. Mirrors
    /// the assignment at `vp9_encodeframe.c:6093` (`x->fwd_txfm4x4`).
    #[inline]
    pub unsafe fn fwd_txfm4x4(&self, lossless: bool)
        -> FdctFn /* picks fwht4x4 or fdct4x4 */;

    /// Forward at any size × any type, dispatching to the right kernel.
    /// Mirrors `vp9_xform_quant_fp` flow (`vp9_encodemb.c:349-417`).
    #[inline]
    pub unsafe fn forward(
        &self,
        size: TxSize, ty: TxType,
        src_diff: *const i16, coeff_out: *mut Coeff,
        src_stride: i32,
        rd_transform: bool,   // selects `_rd` 32×32 variant (§4.4)
        lossless: bool,       // forces WHT for 4×4 if true
    );

    /// DC-only forward (output[0] is the sum × scale). Mirrors
    /// `vp9_xform_quant_dc` (`vp9_encodemb.c:420-485`). For 4×4
    /// `lossless` selects `vp9_fwht4x4` (NB libvpx `vp9_xform_quant_dc`
    /// at 4×4 also routes through `fwd_txfm4x4` so the DC-only variant
    /// here uses the regular `fdct4x4_1` for non-lossless and falls
    /// back to the full `fwht4x4` for lossless; we mirror exactly.)
    #[inline]
    pub unsafe fn forward_dc(
        &self,
        size: TxSize,
        src_diff: *const i16, coeff_out: *mut Coeff,
        src_stride: i32,
        lossless: bool,
    );
}
```

`unsafe` because all signatures cross a raw-pointer boundary into a
function pointer indirected through `SimdDispatch`. Preconditions:
`src_diff` is `n` rows valid for the given stride; `coeff_out` has
`n_coeffs(size)` capacity. CPU-feature precondition is satisfied by
construction — `SimdDispatch` populated the table only with kernels
whose ISA is detected.

### 3.4 Inverse transform dispatch (decoder + encoder reconstruction)

```rust
/// EOB-bucketed inverse DCT add. The kernel **adds** the reconstructed
/// pixel deltas to `dest` (this is `vpx_idct*_add` semantics from
/// `vpx_dsp_rtcd_defs.pl:620-652`).
///
/// `stride` is samples. `input` is `Coeff[n_coeffs(size)]` in raster
/// order. `dest` is the in-place reconstruction buffer (the predicted
/// values are already there; the inverse transform produces the
/// residual that gets added).
pub type IdctAddFn = unsafe extern "C" fn(
    input: *const Coeff, dest: *mut Pixel, stride: i32,
);

/// EOB-bucketed inverse hybrid (DCT/ADST) add. Same shape, plus
/// `tx_type`. Mirrors `vp9_iht*_add` family (`vp9_idct.c:182-204`).
pub type IhtAddFn = unsafe extern "C" fn(
    input: *const Coeff, dest: *mut Pixel, stride: i32, tx_type: i32,
);

/// EOB bucket for the inverse-DCT fast paths, mirrors the thresholds
/// in `vp9_idct.c:135-179`. Numeric values match libvpx's filename
/// suffixes (1, 10, 12, 34, 38, 135) for ease of cross-reference.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum EobBucket {
    /// DC-only (eob == 1).
    Dc,
    /// 4×4: full (eob > 1). 8×8: 12 (eob ∈ [2,12]). 16×16: 10
    /// (eob ∈ [2,10]). 32×32: 34 (eob ∈ [2,34]).
    Tiny,
    /// 8×8: full (eob > 12). 16×16: 38 (eob ∈ [11,38]). 32×32:
    /// 135 (eob ∈ [35,135]).
    Small,
    /// 16×16: full (eob > 38). 32×32: 1024 (eob > 135).
    Full,
}

impl EobBucket {
    /// Map `eob` to the appropriate bucket given the transform size.
    /// Mirrors `vp9_idct{4x4,8x8,16x16,32x32}_add` selector logic
    /// (`vp9_idct.c:119-179`).
    #[inline]
    pub const fn for_size(size: TxSize, eob: u16) -> EobBucket;
}

/// Inverse-transform dispatch table (mirrors HLD-simd §3.18-§3.19).
#[derive(Copy, Clone)]
pub struct InvTxfm {
    // 4×4 IDCT: eob ≤ 1 → _1_add, else _16_add. (`vp9_idct.c:119-125`)
    pub idct4x4_16_add:   IdctAddFn,
    pub idct4x4_1_add:    IdctAddFn,

    // 8×8 IDCT, three buckets (`vp9_idct.c:135-149`)
    pub idct8x8_64_add:   IdctAddFn,
    pub idct8x8_12_add:   IdctAddFn,
    pub idct8x8_1_add:    IdctAddFn,

    // 16×16 IDCT, four buckets (`vp9_idct.c:151-164`)
    pub idct16x16_256_add: IdctAddFn,
    pub idct16x16_38_add:  IdctAddFn,
    pub idct16x16_10_add:  IdctAddFn,
    pub idct16x16_1_add:   IdctAddFn,

    // 32×32 IDCT, four buckets (`vp9_idct.c:166-179`)
    pub idct32x32_1024_add: IdctAddFn,
    pub idct32x32_135_add:  IdctAddFn,
    pub idct32x32_34_add:   IdctAddFn,
    pub idct32x32_1_add:    IdctAddFn,

    // Walsh-Hadamard inverse (lossless-only). `eob > 1` → `_16_add`,
    // else `_1_add`. (`vp9_idct.c:127-133`).
    pub iwht4x4_16_add:    IdctAddFn,
    pub iwht4x4_1_add:     IdctAddFn,

    // Hybrid IDCT/IADST (TX_TYPE != DCT_DCT). 4×4, 8×8, 16×16 only.
    // The wrapper at `vp9_idct.c:182-204` falls back to `vp9_idct*_add`
    // when tx_type == DCT_DCT, so the caller MUST check tx_type first.
    pub iht4x4_16_add:     IhtAddFn,
    pub iht8x8_64_add:     IhtAddFn,
    pub iht16x16_256_add:  IhtAddFn,
}

impl InvTxfm {
    /// Dispatch the 4×4 inverse, honoring lossless. Mirrors the
    /// per-frame `x->inv_txfm_add` flip at `vp9_encodeframe.c:6095`.
    /// `lossless` selects `iwht*` over `idct4x4*`.
    #[inline]
    pub unsafe fn inv4x4(&self, lossless: bool, eob: u16) -> IdctAddFn;

    /// Dispatch the full inverse path for any (size, tx_type, eob,
    /// lossless). Mirrors `vp9_iht*_add` / `vp9_idct*_add` flow in
    /// `vp9_idct.c:119-204`. Caller passes the already-dequantized
    /// `input` coefficients and the in-place reconstruction buffer.
    ///
    /// SAFETY contract per HLD-arch §5.5 Template B applies.
    #[inline]
    pub unsafe fn inverse(
        &self,
        size: TxSize, ty: TxType,
        eob: u16, lossless: bool,
        input: *const Coeff, dest: *mut Pixel, stride: i32,
    );
}
```

Preconditions: `input` has `n_coeffs(size)` valid coefficients; `dest`
is `n × stride`-valid for write; `eob <= n_coeffs(size)`.

### 3.5 Lossless-flip helper

```rust
/// Per-frame transform-pair pinning. Mirrors `vp9_encodeframe.c:6082-6097`
/// which assigns `x->fwd_txfm4x4` and `x->inv_txfm_add` based on
/// `xd->lossless`.
///
/// In v1 the **encoder** can construct only the non-lossless variant
/// (per D-01 and §7.2). The **decoder** constructs whichever variant
/// the bitstream demands so conformance vectors with lossless frames
/// pass.
pub struct PerFrameTransforms {
    /// 4×4 forward (DCT or WHT). Encoder-only.
    pub fwd_txfm4x4: FdctFn,
    /// 4×4 inverse (IDCT or IWHT). Decoder + encoder reconstruction.
    pub inv_txfm_add: IdctAddFn,
}

impl PerFrameTransforms {
    /// Build the per-frame pair. `lossless` = (base_qindex == 0 &&
    /// every delta_q is 0 && segmentation off OR identity per-segment)
    /// — see `vp9_encodeframe.c:6082-6085` for the libvpx predicate.
    pub fn select(disp: &SimdDispatch, lossless: bool) -> Self;
}
```

### 3.6 Lossless predicate

```rust
/// Predicate mirroring `vp9_encodeframe.c:6082-6085`:
///   xd->lossless = (cm->base_qindex == 0 &&
///                   cm->y_dc_delta_q == 0 &&
///                   cm->uv_dc_delta_q == 0 &&
///                   cm->uv_ac_delta_q == 0);
/// Note this is the *frame-level* predicate; segmentation can still
/// override per-segment via `seg.q_index_delta` but a frame is "lossless"
/// only if the predicate above holds AND no segment shifts away from
/// q=0. The decoder applies the same predicate per block via `xd->lossless`.
pub fn is_frame_lossless(
    base_qindex: u8,
    y_dc_delta_q: i8, uv_dc_delta_q: i8, uv_ac_delta_q: i8,
) -> bool;
```

---

## 4. Types

### 4.1 `TxSize`, `TxType`, `TxMode`, `EobBucket`, `IntraMode`

Already defined in §3. Layout notes:

- `TxSize` is `#[repr(u8)]` to match libvpx's `typedef uint8_t TX_SIZE`
  (`vp9_enums.h:77`). The discriminants `0..3` are part of the
  bitstream encoding for `tx_size` (probability-coded against
  `cm->fc->tx_probs[*]` — see LLD-entropy §5.8). We rely on
  `repr(u8)` plus the discriminant values for round-trip with the
  bitstream.
- `TxType` is `#[repr(u8)]` with values `0..3` matching `DCT_DCT`
  / `ADST_DCT` / `DCT_ADST` / `ADST_ADST`. The bitstream never carries
  `tx_type` — it is always derived from the intra prediction mode via
  §4.5. We keep the enum public because every kernel signature carries
  it.
- `TxMode` is `#[repr(u8)]` `0..4`. `OnlyTx4x4 = 0` is the value the
  decoder must read when a frame is lossless (`vp9_encodeframe.c:4659-4661`:
  `if (xd->lossless) return ONLY_4X4`). The encoder writes `cm->tx_mode`
  to the compressed header per LLD-bitstream §3.
- `EobBucket` is a Rust-only convenience for the dispatch helper; not
  bitstream-visible. The variant names are deliberately abstract
  (`Dc`, `Tiny`, `Small`, `Full`) so the same enum works across sizes;
  the const `for_size` mapping pins it to libvpx's exact thresholds.

`Copy + Clone + Send + Sync` are derived on all of `TxSize`, `TxType`,
`TxMode`, `EobBucket`, `IntraMode`. They are POD and pass through
worker threads freely.

### 4.2 `ScanOrder`

```rust
#[derive(Copy, Clone)]
pub struct ScanOrder {
    pub scan:      &'static [i16],
    pub iscan:     &'static [i16],
    pub neighbors: &'static [i16],
}
```

`Copy + Clone + Send + Sync` (all `'static` references — no interior
mutability). All instances of `ScanOrder` live in static arrays inside
this module; the dispatch helpers return `&'static ScanOrder`. No
allocation.

The underlying tables are 16-byte aligned in libvpx
(`DECLARE_ALIGNED(16, ..., ...)` — `vp9_scan.c:3254`). In Rust we get
this via `#[repr(align(16))]` on a wrapper struct that contains the
table, then dereference to a `&'static [i16]`. **Open question O-1
in §8** — confirm whether 16-byte alignment is load-bearing for SIMD
kernels that consume the scan tables directly (libvpx alignment is
defensive; investigate).

### 4.3 Table storage

Every scan / iscan / neighbor table lives in this module as:

```rust
#[repr(align(16))]
struct Align16<T: ?Sized>(T);

static DEFAULT_SCAN_4X4: Align16<[i16; 16]> =
    Align16([0, 4, 1, 5, 8, 2, 12, 9, 3, 6, 13, 10, 7, 14, 11, 15]);
// ... 22 more tables (see §5 catalogue)
```

The dispatch arrays (`vp9_default_scan_orders` and `vp9_scan_orders`)
become:

```rust
pub static DEFAULT_SCAN_ORDERS: [ScanOrder; 4] = [
    ScanOrder { scan: &DEFAULT_SCAN_4X4.0,    iscan: &DEFAULT_ISCAN_4X4.0,
                neighbors: &DEFAULT_SCAN_4X4_NEIGHBORS.0 },
    ScanOrder { scan: &DEFAULT_SCAN_8X8.0,    iscan: &DEFAULT_ISCAN_8X8.0,
                neighbors: &DEFAULT_SCAN_8X8_NEIGHBORS.0 },
    ScanOrder { scan: &DEFAULT_SCAN_16X16.0,  iscan: &DEFAULT_ISCAN_16X16.0,
                neighbors: &DEFAULT_SCAN_16X16_NEIGHBORS.0 },
    ScanOrder { scan: &DEFAULT_SCAN_32X32.0,  iscan: &DEFAULT_ISCAN_32X32.0,
                neighbors: &DEFAULT_SCAN_32X32_NEIGHBORS.0 },
];

pub static SCAN_ORDERS: [[ScanOrder; 4]; 4] = [
    [ /* Tx4x4 */
        ScanOrder { /* DCT_DCT  */ … default … },
        ScanOrder { /* ADST_DCT */ scan: &ROW_SCAN_4X4.0, iscan: &ROW_ISCAN_4X4.0, neighbors: &ROW_SCAN_4X4_NEIGHBORS.0 },
        ScanOrder { /* DCT_ADST */ scan: &COL_SCAN_4X4.0, iscan: &COL_ISCAN_4X4.0, neighbors: &COL_SCAN_4X4_NEIGHBORS.0 },
        ScanOrder { /* ADST_ADST*/ … default … },
    ],
    [ /* Tx8x8 */ … parallel … ],
    [ /* Tx16x16 */ … parallel … ],
    [ /* Tx32x32 */
        // all four tx_type slots are the default 32×32 scan
        // (vp9_scan.c:709-723)
        … default × 4 …
    ],
];
```

This matches `vp9_scan_orders[TX_SIZES][TX_TYPES]` exactly
(`vp9_scan.c:698-725`).

### 4.4 `FwdTxfm` and `InvTxfm` dispatch tables

Already defined in §3.3 and §3.4. Both are `Copy + Clone + Send + Sync`
because every field is an `unsafe extern "C" fn` pointer (function
pointers are `Send + Sync` in Rust; the kernels they point to are
read-only code).

Storage: each table is owned by `SimdDispatch` (HLD-simd §4) and lives
in a `OnceLock<SimdDispatch>` initialized at encoder/decoder
construction time. Workers receive a `&'static FwdTxfm` /
`&'static InvTxfm` by reference.

### 4.5 `intra_mode_to_tx_type_lookup`

```rust
// Verbatim from third_party/libvpx/vp9/common/vp9_reconintra.c:24-35.
pub const INTRA_MODE_TO_TX_TYPE_LOOKUP: [TxType; INTRA_MODES] = [
    TxType::DctDct,    // DC      = 0
    TxType::AdstDct,   // V       = 1
    TxType::DctAdst,   // H       = 2
    TxType::DctDct,    // D45     = 3
    TxType::AdstAdst,  // D135    = 4
    TxType::AdstDct,   // D117    = 5
    TxType::DctAdst,   // D153    = 6
    TxType::DctAdst,   // D207    = 7
    TxType::AdstDct,   // D63     = 8
    TxType::AdstAdst,  // TM      = 9
];
```

`INTRA_MODES = 10`. This is a *common* table — both encoder and decoder
need it to derive `tx_type` from a Y intra mode (`vp9_scan.h:50`,
`vp9_decodeframe.c:325,350,380`). It lives in this module rather than
in `intra_pred` because every consumer is in transforms / entropy
context: scan selection (`get_scan`), `vp9_iht*_add` dispatch
(`vp9_idct.c:182`), and the forward-transform side of `vp9_xform_quant_*`.

### 4.6 `PerFrameTransforms`

Already defined in §3.5. `Copy` because it holds two function pointers.
Built once per frame on the main thread before tile workers spawn; each
worker receives a `PerFrameTransforms` value by copy (cheap).

### 4.7 No interior mutability

Every type in this module is **immutable after construction**. There
are no `UnsafeCell` fields, no atomic counters, no `RefCell`. The
worker-thread aliasing contract (HLD-arch §5.7) is therefore trivial
for transforms: workers receive shared references to read-only data.

---

## 5. Internal algorithm

### 5.1 Scan-order catalogue

Stage 1 A4 §5.10 transcribes every table verbatim. The Rust port
copies each table byte-for-byte. The complete catalogue (24 tables):

| Table | Length | A4 location | libvpx file:line |
|---|---|---|---|
| `default_scan_4x4` | 16 | A4 §5.10.1 / line 3253 | `vp9_scan.c:9-12` |
| `col_scan_4x4` | 16 | A4 §5.10.1 / 3261 | `vp9_scan.c:14-17` |
| `row_scan_4x4` | 16 | A4 §5.10.1 / 3269 | `vp9_scan.c:19-22` |
| `default_scan_8x8` | 64 | A4 §5.10.2 / 3279 | `vp9_scan.c:24-30` |
| `col_scan_8x8` | 64 | A4 §5.10.2 / 3290 | `vp9_scan.c:32-38` |
| `row_scan_8x8` | 64 | A4 §5.10.2 / 3301 | `vp9_scan.c:40-46` |
| `default_scan_16x16` | 256 | A4 §5.10.3 / 3314 | `vp9_scan.c:48-66` |
| `col_scan_16x16` | 256 | A4 §5.10.3 / 3339 | `vp9_scan.c:68-86` |
| `row_scan_16x16` | 256 | A4 §5.10.3 / 3364 | `vp9_scan.c:88-106` |
| `default_scan_32x32` | 1024 | A4 §5.10.4 / 3391 | `vp9_scan.c:108-186` |
| `default_scan_4x4_neighbors` | 34 (17×2) | A4 §5.10.5 / 3480 | `vp9_scan.c:188-191` |
| `col_scan_4x4_neighbors` | 34 | A4 §5.10.5 / 3489 | `vp9_scan.c:193-196` |
| `row_scan_4x4_neighbors` | 34 | A4 §5.10.5 / 3498 | `vp9_scan.c:198-201` |
| `default_scan_8x8_neighbors` | 130 (65×2) | A4 §5.10.5 / 3535 | `vp9_scan.c:218-225` |
| `col_scan_8x8_neighbors` | 130 | A4 §5.10.5 / 3507 | `vp9_scan.c:203-210` |
| `row_scan_8x8_neighbors` | 130 | A4 §5.10.5 / 3521 | `vp9_scan.c:212-218` |
| `default_scan_16x16_neighbors` | 514 (257×2) | A4 §5.10.5 / 3633 | `vp9_scan.c:289-322` |
| `col_scan_16x16_neighbors` | 514 | A4 §5.10.5 / 3549 | `vp9_scan.c:226-258` |
| `row_scan_16x16_neighbors` | 514 | A4 §5.10.5 / 3591 | `vp9_scan.c:260-289` |
| `default_scan_32x32_neighbors` | 2050 (1025×2) | A4 §5.10.5 / 3675 | `vp9_scan.c:323-449` |
| `vp9_default_iscan_4x4` | 16 | A4 §5.10.6 / 3835 | `vp9_scan.c:451-454` |
| `vp9_col_iscan_4x4` | 16 | A4 §5.10.6 / 3843 | `vp9_scan.c:456-459` |
| `vp9_row_iscan_4x4` | 16 | A4 §5.10.6 / 3851 | `vp9_scan.c:461-464` |
| `vp9_default_iscan_8x8` | 64 | A4 §5.10.6 / 3881 | `vp9_scan.c:481-486` |
| `vp9_col_iscan_8x8` | 64 | A4 §5.10.6 / 3859 | `vp9_scan.c:466-472` |
| `vp9_row_iscan_8x8` | 64 | A4 §5.10.6 / 3870 | `vp9_scan.c:474-480` |
| `vp9_default_iscan_16x16` | 256 | A4 §5.10.6 / 3940 | `vp9_scan.c:506-525` |
| `vp9_col_iscan_16x16` | 256 | A4 §5.10.6 / 3892 | `vp9_scan.c:488-506` |
| `vp9_row_iscan_16x16` | 256 | A4 §5.10.6 / 3915 | `vp9_scan.c:527-545` |
| `vp9_default_iscan_32x32` | 1024 | A4 §5.10.6 / 3965 | `vp9_scan.c:547-630` |

Note iscan tables have **scan-index + 1** semantics (`vp9_scan.c:514`,
A4 §5.10.6): the value at position `p` is the EOB position if the last
nonzero coefficient lives at raster position `p`. This is exploited by
the EOB-bucketed inverse dispatch (§5.4).

Neighbor tables have length `(n + 1) * MAX_NEIGHBORS` rather than
`n * MAX_NEIGHBORS`. The trailing pair (positions `n` and `n+1`) is a
sentinel — A4 §2.11 line 503: "Length is `(N + 1) * MAX_NEIGHBORS`
where `N` is the number of coefficients."

**32×32 hybrid scan**: the dispatch array at `vp9_scan.c:707-723` shows
that **all four `tx_type` slots for 32×32 reference the default 32×32
scan**. This is because intra prediction at 32×32 is gated to
`tx_type = DCT_DCT` by the partition / tx-size logic (no 32×32 ADST
exists in VP9). The Rust dispatch table mirrors this: `SCAN_ORDERS[3][0]
== SCAN_ORDERS[3][1] == SCAN_ORDERS[3][2] == SCAN_ORDERS[3][3]`.

### 5.2 `get_scan` selector logic

Mirrors `vp9_scan.h:42-52`:

```rust
pub fn get_scan(
    size: TxSize,
    is_inter: bool,
    plane_is_y: bool,
    lossless: bool,
    intra_y_mode: IntraMode,
) -> &'static ScanOrder {
    // The libvpx predicate: use the default scan when the block is
    // inter, OR when the plane is chroma, OR when the frame is lossless.
    if is_inter || !plane_is_y || lossless {
        &DEFAULT_SCAN_ORDERS[size as usize]
    } else {
        // Intra-Y, non-lossless: pick the per-tx_type scan keyed on the
        // intra mode's derived tx_type.
        let ty = INTRA_MODE_TO_TX_TYPE_LOOKUP[intra_y_mode as usize];
        &SCAN_ORDERS[size as usize][ty as usize]
    }
}
```

The lossless branch matters: even for Y intra at a lossless frame, the
default scan is used. This pairs with the per-frame `tx_mode =
ONLY_4X4` and `xd->lossless = 1` forcing every transform to be 4×4
with WHT instead of DCT/ADST (and there is no ADST WHT).

### 5.3 Forward-transform flow (encoder)

Mirrors `vp9_xform_quant_fp` (`vp9_encodemb.c:349-417` for the 8-bit
branch). At cpu-used 6-8 the encoder reaches this from the
non-RD-loop committing-the-mode step (after `block_yrd` has selected
a mode using the Hadamard rate proxy — A2 §4.6 / `vp9_pickmode.c`).

```text
forward(size, tx_type, src_diff, coeff_out, src_stride,
        rd_transform, lossless):
    match size:
        Tx32x32:
            // libvpx never invokes fht at 32×32 (see §5.1 note); ignore
            // tx_type. `rd_transform` selects the lower-precision
            // `vpx_fdct32x32_rd` for cpu-used 6-8 quality tier
            // (vp9_encodemb.c:332-336). When `use_lp32x32fdct=1` we
            // use `_rd`.
            if rd_transform:
                fdct32x32_rd(src_diff, coeff_out, src_stride)
            else:
                fdct32x32(src_diff, coeff_out, src_stride)
        Tx16x16:
            if tx_type == DctDct:
                fdct16x16(src_diff, coeff_out, src_stride)
            else:
                fht16x16(src_diff, coeff_out, src_stride, tx_type)
        Tx8x8:
            if tx_type == DctDct:
                fdct8x8(src_diff, coeff_out, src_stride)
            else:
                fht8x8(src_diff, coeff_out, src_stride, tx_type)
        Tx4x4:
            if lossless:
                fwht4x4(src_diff, coeff_out, src_stride)   // tx_type ignored
            elif tx_type == DctDct:
                fdct4x4(src_diff, coeff_out, src_stride)
            else:
                fht4x4(src_diff, coeff_out, src_stride, tx_type)
```

The `rd_transform` flag corresponds to libvpx `x->use_lp32x32fdct`
which is set by speed feature `sf->use_quant_fp = !is_keyframe` (A2
§5.12 line 1905). At cpu-used 6-8 this is `1` on inter frames, `0` on
keyframes — so the lower-precision `_rd` variant is the inter-frame
default.

**Why we keep both**: the keyframe path runs `vpx_fdct32x32_c` (full
precision) and the inter path runs `vpx_fdct32x32_rd_c` (rounding
shift). They produce different coefficients; without the split, KF and
inter would diverge from libvpx. See `vpx_dsp/fwd_txfm.c:735-770`.

### 5.4 Inverse-transform flow (decoder + encoder reconstruction)

Mirrors `vp9_iht*_add` and `vp9_idct*_add` (`vp9_idct.c:119-204`):

```text
inverse(size, tx_type, eob, lossless, input, dest, stride):
    bucket := EobBucket::for_size(size, eob)
    match size:
        Tx4x4:
            if lossless:
                // tx_type is irrelevant under lossless (forced DCT_DCT
                // by spec; the WHT replaces DCT entirely).
                if bucket == Dc:
                    iwht4x4_1_add(input, dest, stride)
                else:
                    iwht4x4_16_add(input, dest, stride)
            elif tx_type == DctDct:
                if bucket == Dc:
                    idct4x4_1_add(input, dest, stride)
                else:
                    idct4x4_16_add(input, dest, stride)
            else:
                // Hybrid 4×4: no DC fast path in libvpx (always _16_add).
                // See vp9_idct.c:182-188.
                iht4x4_16_add(input, dest, stride, tx_type)
        Tx8x8:
            if tx_type == DctDct:
                match bucket:
                    Dc    => idct8x8_1_add(input, dest, stride)
                    Tiny  => idct8x8_12_add(input, dest, stride)   // eob<=12
                    _     => idct8x8_64_add(input, dest, stride)
            else:
                iht8x8_64_add(input, dest, stride, tx_type)
        Tx16x16:
            if tx_type == DctDct:
                match bucket:
                    Dc    => idct16x16_1_add(input, dest, stride)
                    Tiny  => idct16x16_10_add(input, dest, stride)  // eob<=10
                    Small => idct16x16_38_add(input, dest, stride)  // eob<=38
                    Full  => idct16x16_256_add(input, dest, stride)
            else:
                iht16x16_256_add(input, dest, stride, tx_type)
        Tx32x32:
            // tx_type is forced DCT_DCT at 32×32.
            match bucket:
                Dc    => idct32x32_1_add(input, dest, stride)
                Tiny  => idct32x32_34_add(input, dest, stride)    // eob<=34
                Small => idct32x32_135_add(input, dest, stride)   // eob<=135
                Full  => idct32x32_1024_add(input, dest, stride)
```

Concrete thresholds with origin:

| Size  | `Dc` | `Tiny` (≤) | `Small` (≤) | `Full` (≤) | Origin |
|-------|------|------------|-------------|------------|---|
| 4×4   | eob==1 | eob >1, full (no separate Tiny path) | — | — | `vp9_idct.c:119-125` |
| 8×8   | eob==1 | eob ≤ 12 | eob > 12 | — | `vp9_idct.c:135-149` |
| 16×16 | eob==1 | eob ≤ 10 | eob ≤ 38 | eob > 38 | `vp9_idct.c:151-164` |
| 32×32 | eob==1 | eob ≤ 34 | eob ≤ 135 | eob > 135 | `vp9_idct.c:166-179` |

The `EobBucket::for_size` helper centralises this dispatch:

```rust
pub const fn for_size(size: TxSize, eob: u16) -> EobBucket {
    match size {
        TxSize::Tx4x4   => if eob == 1 { EobBucket::Dc } else { EobBucket::Tiny },
        TxSize::Tx8x8   => if eob == 1 { EobBucket::Dc }
                           else if eob <= 12 { EobBucket::Tiny }
                           else { EobBucket::Full },
        TxSize::Tx16x16 => if eob == 1 { EobBucket::Dc }
                           else if eob <= 10 { EobBucket::Tiny }
                           else if eob <= 38 { EobBucket::Small }
                           else { EobBucket::Full },
        TxSize::Tx32x32 => if eob == 1 { EobBucket::Dc }
                           else if eob <= 34 { EobBucket::Tiny }
                           else if eob <= 135 { EobBucket::Small }
                           else { EobBucket::Full },
    }
}
```

Note: at 8×8 the libvpx selector at `vp9_idct.c:135-149` is **two-way**
(no Small bucket); the helper returns `Full` for everything above the
Tiny threshold so the dispatch table sees a clean match. Similarly,
4×4 only has Dc + Tiny (libvpx switches on `eob > 1` vs `eob == 1` at
line 121).

### 5.5 Per-frame lossless flip

Source: `vp9_encodeframe.c:6082-6097`.

```c
xd->lossless = cm->base_qindex == 0 && cm->y_dc_delta_q == 0 &&
               cm->uv_dc_delta_q == 0 && cm->uv_ac_delta_q == 0;
...
x->fwd_txfm4x4 = xd->lossless ? vp9_fwht4x4 : vpx_fdct4x4;
x->inv_txfm_add = xd->lossless ? vp9_iwht4x4_add : vp9_idct4x4_add;
if (xd->lossless) x->optimize = 0;
cm->tx_mode = select_tx_mode(cpi, xd);  // returns ONLY_4X4 if lossless
```

Three things to mirror:

1. **Predicate.** `is_frame_lossless()` (§3.6) returns the same boolean.
2. **Forward 4×4 pair flip.** `PerFrameTransforms::fwd_txfm4x4` holds
   either `fwd.fwht4x4` (lossless) or `fwd.fdct4x4` (lossy).
3. **Inverse 4×4 pair flip.** `PerFrameTransforms::inv_txfm_add` holds
   either `inv.iwht4x4_16_add` (lossless) or `inv.idct4x4_16_add`
   (lossy). The DC-only fast path is selected per-block by EOB
   (`inv4x4()` returns `_1_add` when `eob == 1`).

The third libvpx side-effect (`x->optimize = 0`) is the trellis-quant
gate. In our port trellis-quant is dropped entirely (L-28, §7.1) so
this assignment has no Rust mirror.

The fourth side-effect (`cm->tx_mode = ONLY_4X4` if lossless) is
implemented in the encoder's frame-init path; this LLD exposes the
predicate but the assignment lives in LLD-bitstream / encoder
frame-init.

**Encoder scope reminder**: v1 encoder never produces lossless. The
`is_frame_lossless()` predicate is fed by the encoder's frame-init
code but always evaluates `false` because the encoder's `base_qindex`
range is gated above zero by the rate controller (LLD-rate-control).
The decoder, however, must honor lossless frames in conformance
vectors. See §7.2.

### 5.6 Block-level encode flow (cross-reference)

End-to-end at cpu-used 6-8 (combining transforms with quant + entropy):

```text
for each transform block of (size, tx_type) in the picked partition:
    // 1. Compute residual (LLD-mode-decision)
    vpx_subtract_block(rows, cols, src_diff, src, pred)

    // 2. Forward transform (this LLD)
    transforms.fwd.forward(size, tx_type, src_diff, coeff,
                           src_stride, rd_transform=use_lp32x32fdct,
                           lossless=false)
    // (Encoder v1 never lossless; the param is constant false.)

    // 3. Quantize (LLD-quant)
    quant.fp(size, coeff, n_coeffs(size), &mb_plane,
             qcoeff, dqcoeff, dequant, &mut eob,
             &scan_for(size, tx_type))

    // 4. Trellis-quant gate (L-28 RESOLVED — NO-OP in v1)

    // 5. Tokenize (LLD-entropy): walks qcoeff in scan order using
    //    the same ScanOrder retrieved at step 3.

    // 6. Inverse transform for reconstruction (this LLD)
    if eob > 0:
        transforms.inv.inverse(size, tx_type, eob, lossless=false,
                               dqcoeff, dest, dest_stride)
```

The `ScanOrder` retrieved at step 3 is the **same** triple used at step
5 — they must reference the identical static struct, not a copy. This
is why `scan_for` returns `&'static ScanOrder` rather than a value.

### 5.7 Block-level decode flow (cross-reference)

```text
for each transform block parsed from the bitstream:
    // 1. Determine (size, tx_type, eob, lossless) from frame headers
    //    + per-block context. (LLD-bitstream + LLD-entropy.)
    let scan = get_scan(size, is_inter, plane_is_y, lossless, intra_y_mode);

    // 2. Detokenize: walk scan.scan in order, write to dqcoeff using
    //    scan.neighbors for context.  (LLD-entropy.)

    // 3. Dequantize (LLD-quant: dequant from dq_tables × qcoeff).

    // 4. Inverse transform (this LLD).
    if eob > 0:
        transforms.inv.inverse(size, tx_type, eob, lossless,
                               dqcoeff, dest, dest_stride)
```

### 5.8 Numerical contract — fixed-point shifts and rounding

The C reference (`vpx_dsp/fwd_txfm.c`) is integer throughout with
`ROUND_POWER_OF_TWO` rounding shifts. HLD-arch §6.5 standardises the
Rust spelling: `transforms::common::round_power_of_two(v, n)` is the
canonical helper, wraps `(v.wrapping_add(1 << (n - 1)) >> n)`. Every
multiply by `cospi_*_64` and `fdct_round_shift` callsite uses it.

The transform constants live in
`third_party/libvpx/vpx_dsp/txfm_common.h` (`cospi_*_64`,
`sinpi_*_9`, `cospi_*_64_minus_1`, `dct_const_bits = 14`,
`dct_const_round_shift = 1 << (dct_const_bits - 1)`). The Rust port
copies them verbatim into `rvp9_core::transforms::common`:

```rust
pub const DCT_CONST_BITS: i32 = 14;
pub const DCT_CONST_ROUNDING: i32 = 1 << (DCT_CONST_BITS - 1);

pub const COSPI_1_64: i32  = 16364;
pub const COSPI_2_64: i32  = 16305;
// ... (33 cosine constants)
pub const SINPI_1_9: i32   = 5283;
pub const SINPI_2_9: i32   = 9929;
pub const SINPI_3_9: i32   = 13377;
pub const SINPI_4_9: i32   = 15212;
```

(Full list: `txfm_common.h:21-71`. The Rust file is a 1:1 transcription.)

### 5.9 Forward DCT scaling — pre-multiply by 16 / 4

The 4×4 forward DCT pre-multiplies inputs by 16 on the first pass
(`vpx_fdct.c:36-39`) and adds 1 to the [0,0] sample if it's non-zero
(`vpx_fdct.c:40-42`). The 8×8 pre-multiplies by 4 (`vpx_fdct.c:106-109`).
The 16×16 and 32×32 have their own scaling sequences. These are
**not** divergences — they are bit-exact integer DCT scaling per the
VP9 spec §8.7. The Rust port keeps every wrapping arithmetic site
explicit per HLD-arch §6.2 (`wrapping_mul`, `wrapping_add`).

The `fdct4x4_c +1 on in[0,0]` quirk: this is a libvpx workaround for
SIMD vector layout (`vpx_dsp/fwd_txfm.c:36-42`). Rust scalar mirrors it
exactly. SIMD implementations don't need the workaround (they handle
the layout differently in vectorised loads) — but the **scalar
reference** keeps the workaround so the differential test passes
against the libvpx scalar. See §9.

---

## 6. SIMD dispatch

### 6.1 Kernels owned by this module

Per HLD-simd §3.16-§3.19 (forward DCT, forward ADST, inverse DCT,
inverse ADST), this module exposes 14 forward kernels and 17 inverse
kernels through `FwdTxfm` / `InvTxfm`. The ISA matrix is reproduced
here for self-containment; HLD-simd is the authority and version
that updates when ISAs change.

**Forward DCT** (`vpx_dsp_rtcd_defs.pl:533-612`, 8-bit branch
lines 587-612):

| Kernel | scalar | sse2 | ssse3 | avx2 | neon |
|---|---|---|---|---|---|
| `vpx_fdct4x4` | x | x |  |  | x |
| `vpx_fdct4x4_1` | x | x |  |  | x |
| `vpx_fdct8x8` | x | x | x* |  | x |
| `vpx_fdct8x8_1` | x | x |  |  | x |
| `vpx_fdct16x16` | x | x |  | x | x |
| `vpx_fdct16x16_1` | x | x |  |  | x |
| `vpx_fdct32x32` | x | x |  | x | x |
| `vpx_fdct32x32_rd` | x | x |  | x | x |
| `vpx_fdct32x32_1` | x | x |  |  | x |

(`*` ssse3 only on x86_64.)

**Forward ADST / WHT** (`vp9/common/vp9_rtcd_defs.pl:151-164`):

| Kernel | scalar | sse2 | neon |
|---|---|---|---|
| `vp9_fht4x4` | x | x | x |
| `vp9_fht8x8` | x | x | x |
| `vp9_fht16x16` | x | x | x |
| `vp9_fwht4x4` | x | x |  |

**Inverse DCT + WHT** (`vpx_dsp_rtcd_defs.pl:620-652`, 8-bit branch):

| Kernel | scalar | sse2 | ssse3 | avx2 | neon |
|---|---|---|---|---|---|
| `vpx_idct4x4_16_add` | x | x |  |  | x |
| `vpx_idct4x4_1_add` | x | x |  |  | x |
| `vpx_idct8x8_64_add` | x | x |  |  | x |
| `vpx_idct8x8_12_add` | x | x | x |  | x |
| `vpx_idct8x8_1_add` | x | x |  |  | x |
| `vpx_idct16x16_256_add` | x | x |  | x | x |
| `vpx_idct16x16_38_add` | x | x |  |  | x |
| `vpx_idct16x16_10_add` | x | x |  |  | x |
| `vpx_idct16x16_1_add` | x | x |  |  | x |
| `vpx_idct32x32_1024_add` | x | x |  | x | x |
| `vpx_idct32x32_135_add` | x | x | x | x | x |
| `vpx_idct32x32_34_add` | x | x | x |  | x |
| `vpx_idct32x32_1_add` | x | x |  |  | x |
| `vpx_iwht4x4_16_add` | x | x |  |  |  |
| `vpx_iwht4x4_1_add` | x |  |  |  |  |

**Inverse ADST/Hybrid** (`vp9/common/vp9_rtcd_defs.pl:63-67` + the
SIMD variants under `vp9/common/x86/vp9_idct_intrin_sse2.c` and
`vp9/common/arm/neon/`):

| Kernel | scalar | sse2 | neon |
|---|---|---|---|
| `vp9_iht4x4_16_add` | x | x | x |
| `vp9_iht8x8_64_add` | x | x | x |
| `vp9_iht16x16_256_add` | x | x | x |

### 6.2 RTCD signature

Per HLD-simd §4 the dispatch table is built once at process start
inside `SimdDispatch::detect()`. The two transform sub-tables are
public fields:

```rust
pub struct SimdDispatch {
    // ... other kernels (HLD-simd §4) ...
    pub fwd: transforms::FwdTxfm,
    pub inv: transforms::InvTxfm,
    // ... (intra, loopfilter, quant, etc.)
}
```

Workers borrow `&'static SimdDispatch` (from a `OnceLock`) and call
`disp.inv.inverse(...)` directly.

### 6.3 Promotion rules

HLD-simd §3.18 promotes **ssse3 to mandatory** for the
`idct8x8_12 / idct32x32_135 / idct32x32_34` kernels — the SSE2 paths
exist but are markedly slower and SSSE3 is universal on any CPU shipped
since 2007. This is an explicit divergence from libvpx's library
construction (where the SSSE3 variants are optional); see §7.4.

### 6.4 Test strategy (this module)

The SIMD-vs-scalar differential plan from HLD-simd §6 applies:

1. **Round-trip**: for every `(size, tx_type)` pair, generate a random
   `i16[n × n]` residual, run forward, then inverse, verify the result
   matches the original to within the spec-defined rounding error.
   `for tx_type in [DctDct, AdstDct, DctAdst, AdstAdst]`,
   `for size in [Tx4x4, Tx8x8, Tx16x16]`. Plus `(Tx32x32, DctDct)`.
2. **Cross-SIMD differential**: for every kernel, every active ISA's
   output must match the scalar reference bit-exactly. Property test:
   1024 random `i16[n × n]` inputs per ISA pair, assert byte-equal
   output. HLD-simd §6 lists this as the canonical pattern.
3. **EOB-bucket boundary tests** (inverse only): for each
   `(size, bucket)` pair, construct an input whose EOB sits exactly at
   the bucket threshold (1 / 10 / 12 / 34 / 38 / 135) and verify the
   correct dispatcher selects the right kernel and produces the same
   result as the next bucket up.
4. **Lossless flip tests**: build a frame with `base_qindex = 0`, run
   it through `is_frame_lossless()`, verify `PerFrameTransforms`
   selects `fwht4x4` / `iwht4x4_*`. Round-trip a 4×4 WHT through
   `fwht4x4` then `iwht4x4_16_add` and assert pixel-exact recovery
   (WHT is integer-invertible by spec).
5. **Conformance vectors** (Stage 3 gate): every test vector with a
   lossless frame, a 32×32 intra block, a hybrid (ADST) intra block,
   and EOB at each bucket boundary. The conformance harness owns this.
6. **Differential against libvpx scalar reference**: link
   `third_party/libvpx`'s C kernels into the test harness, run
   side-by-side over random inputs, byte-equal assertion.

---

## 7. Deliberate divergences from libvpx

Each entry maps to a divergence-ledger row (the ledger already has
templates; these feed `notes/divergence-ledger.md` per HLD-arch §1.5).

### 7.1 Trellis-quant (`vp9_optimize_b`) not ported — L-28 RESOLVED

**Where:** `rvp9-core::transforms` does not expose any "optimize
coefficients" entry point; `rvp9-encoder::quant` does not call one.

**libvpx:** `vp9_optimize_b()` in `vp9/encoder/vp9_encodemb.c:66-300`
plus its callsites at lines 632, 884, 900, 917, 935, 963, 975, 987,
1003. Runs a trellis-quant optimization over each coded block,
reading `scan`, `neighbors`, `token_costs[tx_size][...]`, and choosing
between "keep current value" and "reduce by 1" per scan position
greedily.

**Why drop:** at cpu-used 6-8 libvpx sets `sf->optimize_coefficients =
0` in `set_rt_speed_feature_framesize_independent` (A6 §5.12 line 1514
analysis). Every `enable_trellis_opt` flag derives from this; the
function body is never executed at our speed setting. Drop it entirely.

**Byte-parity impact:** none in matched configs (`--cpu-used 7
--rt` is our reference).

**Future plan:** if a quality preset (cpu-used ≤ 4) is added in
v2, this becomes a new LLD entry. Status: ledger entry exists
(2026-05-16 — Trellis-quantization). This LLD does **not** add a new
ledger row — it confirms the existing one applies.

### 7.2 Lossless encoder path not built; decoder honors lossless

**Where:** `rvp9-encoder::*` cannot produce a lossless frame —
the encoder's rate controller (LLD-rate-control) refuses `base_qindex
== 0`. `PerFrameTransforms::select(_, lossless: true)` therefore is
only ever invoked from the **decoder** path. The forward WHT
(`vp9_fwht4x4`) is shipped in `FwdTxfm` for completeness + future
proofing but its only test consumer is the round-trip differential.

**libvpx:** `vp9_encodeframe.c:6093-6097` flips both forward and inverse
based on `xd->lossless`. The full lossless encode flow runs at the
slowest quality presets (cpu-used 0, plus the lossless config flag).

**Why split:** v1 encoder does not target lossless (no real-time use
case). The decoder must still honor lossless frames in conformance
vectors.

**Byte-parity impact:** none in matched configs. A lossless conformance
vector exercises the WHT inverse path only; the WHT forward path is
exercised by tests, not by production encode.

**Future plan:** if a lossless encoder preset is ever added, the
forward WHT is already wired in `FwdTxfm` — only the rate-controller
gate needs lifting. **New ledger entry needed.**

### 7.3 Hybrid scan for 32×32: dispatch redundancy preserved

**Where:** `SCAN_ORDERS[Tx32x32][*]` repeats the same `ScanOrder`
struct four times (one per `TxType`), even though libvpx semantically
never reaches the non-default slots.

**libvpx:** `vp9_scan.c:707-723` does the same — four redundant entries
in the dispatch array. We mirror exactly to keep the indexing pattern
uniform; the alternative (returning the default scan from `scan_for`
when `size == Tx32x32` and skipping the entry storage) would diverge
from libvpx's data layout. Memory cost is 3 × `sizeof(ScanOrder) = 72
bytes` per ISA. Trivial.

**Byte-parity impact:** none.

**Future plan:** none.

**Ledger:** no new entry — this is byte-for-byte parity with libvpx.

### 7.4 SSSE3 promoted to required for three inverse-DCT kernels

**Where:** `SimdDispatch::detect()` (HLD-simd §3.18) refuses to start
on an x86_64 host without SSSE3 when those three kernels would
otherwise fall back to the slower SSE2 variants.

**libvpx:** allows SSE2-only x86 to run; the SSSE3 paths are optional.
SSE2 fallbacks for `idct8x8_12 / idct32x32_135 / idct32x32_34` exist
and work correctly.

**Why diverge:** SSSE3 is universal on hardware shipped since 2007; the
SSE2 fallback is markedly slower (~30% on Skylake-class) and harms our
realtime gate. HLD-simd §3.18 owns this decision.

**Byte-parity impact:** none — the SSSE3 and SSE2 paths produce
bit-exact output (verified in §6.4 differential tests).

**Future plan:** revisit only if a pre-Nehalem (pre-2008) target
becomes interesting. None planned.

**Ledger:** **new entry** to be added at Stage 2 close — "SSSE3
mandatory for three inverse-DCT kernels on x86_64".

### 7.5 EOB-bucket dispatch centralised in `EobBucket::for_size`

**Where:** `rvp9-core::transforms::EobBucket::for_size` replaces the
inline `if/else` cascades in `vp9_idct.c:119-179`.

**libvpx:** four separate functions (`vp9_idct4x4_add`,
`_8x8_add`, `_16x16_add`, `_32x32_add`) each with its own
`if eob == 1 { ... } else if eob <= N { ... }` ladder.

**Why centralise:** preserves byte-identical dispatch decisions while
making the test surface cleaner (`EobBucket::for_size` is unit-testable
in isolation). The dispatch tables in §3.4 still hold one entry per
EOB bucket per size, exactly mirroring libvpx's per-size kernel set.

**Byte-parity impact:** none — same kernel selected for the same `(size,
eob)`.

**Future plan:** none.

**Ledger:** no new entry — pure refactoring with provable equivalence.

### 7.6 Scan-order tables: alignment via `#[repr(align(16))]`

**Where:** the wrapper struct `Align16<T>` in §4.3.

**libvpx:** `DECLARE_ALIGNED(16, static const int16_t, ...)` in
`vp9_scan.c` lines 3254 etc. (compiler attribute).

**Why mirror:** unverified whether 16-byte alignment is load-bearing
for any SIMD kernel that reads scan tables directly. libvpx's
alignment is defensive; we mirror to avoid a guessing game. See §8 O-1.

**Byte-parity impact:** none.

**Ledger:** no new entry — implementation detail.

### 7.7 32×32 hybrid (`vp9_fht32x32` / `vp9_iht32x32`) absent

**Where:** `FwdTxfm` has no `fht32x32` field; `InvTxfm` has no
`iht32x32_*_add` field.

**libvpx:** also absent. 32×32 hybrid is not part of the VP9 spec; the
scan-table dispatch (`vp9_scan_orders[3][*]`) forces every 32×32 block
to use the default DCT-DCT scan, which means `tx_type` at 32×32 is
unobservable in the bitstream and the encoder/decoder always run the
DCT.

**Why document:** future readers may wonder why the table is 14
forward / 17 inverse rather than 16 / 20. This is the answer.

**Byte-parity impact:** none.

**Ledger:** no new entry — spec-mandated.

### 7.8 Forward 32×32 _rd vs full split

**Where:** `FwdTxfm.fdct32x32` and `FwdTxfm.fdct32x32_rd` are two
separate fields; `forward()` selects by `rd_transform` parameter.

**libvpx:** `vp9_encodemb.c:331-337` (`static INLINE void
fdct32x32(int rd_transform, ...)`) does the same — selects based on
`x->use_lp32x32fdct`.

**Why mirror:** the two kernels produce different coefficients
(`vpx_fdct32x32_rd_c` uses an extra rounding shift). Byte parity at
cpu-used 6-8 requires the same selection (KF runs full precision,
inter runs `_rd`).

**Byte-parity impact:** none in matched configs.

**Ledger:** no new entry — byte-for-byte parity.

---

## 8. Open questions

### O-1. Scan-table alignment requirement

`DECLARE_ALIGNED(16, ...)` in `vp9_scan.c` may or may not be
load-bearing for SIMD kernels that read scan tables. The kernels we ship
read scan tables only inside `vp9_quantize_fp_*` (LLD-quant); the
question is whether those SIMD impls assume 16-byte alignment via
`movdqa` rather than `movdqu`. **Action:** grep
`vpx_dsp/x86/quantize_sse2.c` and `_avx2.c` for `_mm_load_si128`-style
calls on the `scan` argument. Defer resolution to LLD-quant author
(this LLD's owner cannot block their work).

### O-2. `fdct32x32_rd` vs `fdct32x32` for keyframes — verify

A2 §5.12 line 1905 has `sf->use_quant_fp = !is_keyframe`, which implies
KF uses the full-precision `fdct32x32` and inter uses `fdct32x32_rd`.
Confirm by tracing `x->use_lp32x32fdct` assignment from `cpi->sf` to
`x` in `vp9_encodeframe.c`. **Action:** add a Stage 3 unit test that
encodes a 1-frame KF clip and asserts the captured kernel pointer is
`fdct32x32` (not `_rd`).

### O-3. Hybrid 32×32 unreachability — assert at encode time?

Per §5.1 and §7.7 32×32 hybrid never happens. Should we `debug_assert!`
in `forward()` that `size == Tx32x32 → tx_type == DctDct`, or silently
ignore? **Recommendation:** `debug_assert!` — surfaces a programmer
error in the caller (mode-decision should never propose a non-DCT_DCT
32×32 block). Pending Arthur sign-off.

### O-4. `intra_y_mode` parameter when irrelevant

`get_scan` takes `intra_y_mode: IntraMode` even when `is_inter ||
!plane_is_y || lossless` (where the parameter is unused). Two options:
take an `Option<IntraMode>` (Rust-idiomatic), or take a plain
`IntraMode` with a documented "ignored when X" rule (libvpx-style).
**Recommendation:** plain `IntraMode` — the call site cost of
`Some(mode)` everywhere is higher than the readability gain.

### O-5. `EobBucket` for 4×4 — 2-way or 4-way?

At 4×4 libvpx has only two paths (DC vs full); the `Tiny` variant is
already used for "anything non-DC". Should `EobBucket::for_size` at
4×4 return `Tiny` for `eob > 1`, leaving `Small` and `Full` unused?
Or should we collapse to a 2-variant enum for 4×4? **Recommendation:**
keep `Tiny` for 4×4 — matches the table structure in §3.4 and
keeps `EobBucket` size-independent. The compiler will eliminate the
unreachable branches.

### O-6. `cospi_*_64` constants location

Should `cospi_*_64` constants live in `transforms::common` (this
module) or `transforms::scalar::dct_consts`? **Recommendation:**
`common` — the constants are not impl-specific.

### O-7. SIMD targeting `vpx_iwht4x4_16_add`

HLD-simd §3.18 lists `vpx_iwht4x4_16_add` as scalar + sse2 only.
`vpx_iwht4x4_1_add` as scalar only. Are these reachable enough at
real-time to bother with SSE2? Real-time encoder doesn't produce
lossless, but the decoder must support lossless conformance vectors,
and there are exactly two such vectors in our Stage 0 corpus. **Action:**
keep SSE2 + scalar for `_16_add`; scalar-only for `_1_add`. No new
work.

---

## 9. Test strategy

### 9.1 Unit tests

Per-module:

- **`scan_order_lengths`** — every scan / iscan table has the expected
  length (16/64/256/1024); every neighbor table has `(n+1) * 2`.
- **`iscan_is_scan_plus_one`** — for every `(size, tx_type)`,
  `iscan[scan[k]] == k + 1` for all `k in 0..n_coeffs()`. This is the
  invariant from `vp9_scan.c:514` (A4 §5.10.6).
- **`scan_dispatch_default_chroma`** — `get_scan(size, false, false,
  false, _)` returns `&DEFAULT_SCAN_ORDERS[size]`. (Chroma always uses
  default.)
- **`scan_dispatch_inter`** — `get_scan(size, true, true, false,
  IntraMode::AnyVariant)` returns `&DEFAULT_SCAN_ORDERS[size]`. (Inter
  always uses default.)
- **`scan_dispatch_lossless`** — `get_scan(size, false, true, true,
  any_mode)` returns `&DEFAULT_SCAN_ORDERS[size]`. (Lossless forces
  default.)
- **`scan_dispatch_intra_y`** — every `(size, intra_mode)` pair returns
  the scan corresponding to `INTRA_MODE_TO_TX_TYPE_LOOKUP[mode]`.
- **`intra_to_tx_type`** — `INTRA_MODE_TO_TX_TYPE_LOOKUP` matches the
  10 entries in `vp9_reconintra.c:24-35` byte-for-byte.
- **`eob_bucket_thresholds`** — for each `(size, eob)` in the boundary
  set `{1, 2, 10, 11, 12, 13, 34, 35, 38, 39, 135, 136}`,
  `EobBucket::for_size` returns the correct variant.
- **`per_frame_lossless_predicate`** — `is_frame_lossless(0, 0, 0, 0)
  == true`; any one component non-zero → `false`.

### 9.2 Differential tests against libvpx scalar

Per HLD-simd §6, each kernel has a 1024-iteration property test that
generates random `i16` inputs (Gaussian-weighted clusters near zero to
match real residual distributions) and asserts byte-equal output vs
the libvpx C reference (linked via `cc::Build` in `[dev-dependencies]`):

- **Forward DCT** (4×4 / 8×8 / 16×16 / 32×32 / 32×32_rd / four DC-only
  variants): random `i16[n × n]` → assert `our_fdct(x) ==
  libvpx_vpx_fdct(x)`.
- **Forward FHT** (4×4 / 8×8 / 16×16, four `tx_type` values): same.
- **Forward FWHT** (4×4): same.
- **Inverse DCT** (every EOB-bucket variant): random `i16[n × n]`
  coefficient + random `u8[n × stride]` dest → assert
  `our_idct_add(x, dest, stride) == libvpx_vpx_idct_add(x, dest,
  stride)`. The `_add` semantics mean the test compares the post-add
  pixel buffer, not the kernel's internal output.
- **Inverse FHT** (4×4 / 8×8 / 16×16, four `tx_type` values): same.
- **Inverse FWHT** (4×4 / `_1` variant): same.

### 9.3 Differential tests across ISAs

For every kernel with multiple ISA impls (e.g. `vpx_fdct16x16` has
scalar, sse2, avx2, neon), pick the scalar reference as ground truth
and assert every other ISA produces bit-equal output on the same
randomised inputs. ISA-specific dispatch is gated by
`is_x86_feature_detected!` / `aarch64_feature_detected!`; skip-on-no-
support. HLD-simd §6 owns the runner.

### 9.4 Round-trip tests

For every `(size, tx_type)`:

- Generate random `Pixel[n × n]` `src` and another `Pixel[n × n]`
  `pred`.
- Compute `src_diff = src - pred` (32-bit math, clamp to `i16`).
- Forward: `coeff = forward(size, tx_type, src_diff, ...)`.
- Build a "dequantized" buffer that exactly equals `coeff` (skip
  quant, since we're testing the transform pipeline in isolation).
- Inverse: `dest = pred; inverse(size, tx_type, eob, false, coeff,
  dest, stride)`.
- Assert `dest == src` to within the spec-allowed rounding tolerance
  (see VP9 spec §8.7). For DCT the tolerance is integer; the transform
  is bit-invertible.

For `(Tx4x4, lossless=true)`:

- Same flow with `lossless=true` (selects WHT). Assert **pixel-exact**
  recovery (WHT is exactly invertible by spec).

### 9.5 EOB-bucket boundary tests

For each `(size, bucket_threshold)` pair in §5.4:

- Construct a coefficient buffer with exactly `threshold` nonzero
  entries placed in scan order.
- Compute `eob = threshold + 1` (matches the libvpx convention).
- Call `inverse(size, DctDct, eob, false, coeff, dest, stride)`.
- Verify the dispatcher selected the correct kernel by checking the
  output against the next-bucket-up kernel (which should produce
  identical output since both are valid for this EOB).

This catches off-by-one bugs in `EobBucket::for_size` and verifies the
contract that "smaller bucket producing the same result as larger
bucket is correct" — the smaller bucket just runs faster.

### 9.6 Conformance vectors (Stage 3 gate)

The 14 conformance vectors in `tests/conformance/vectors/` cover:

- 32×32 intra blocks (DCT_DCT only by spec)
- 16×16 hybrid intra (ADST × DCT permutations)
- 8×8 / 4×4 lossy and lossless variants
- Frames with EOB at each bucket boundary (covered statistically by
  test content)

The Stage 3 exit criterion ("100% pass on official VP9 conformance
test vectors") implicitly verifies this LLD's inverse path. Any
divergence between our inverse output and the libvpx reference shows
up as pixel-level mismatch in the per-frame MD5 comparison (Stage 3
adds per-frame MD5 parsing per the Stage 0 deferred items).

### 9.7 Lossless flip end-to-end

A decoder-side conformance vector with `base_qindex = 0` exercises:

- `is_frame_lossless()` returns `true`.
- `PerFrameTransforms::select()` picks `iwht4x4_*_add`.
- Every 4×4 block in the frame routes through the WHT inverse.
- The output matches the spec-defined lossless pixel value.

If this fails, either (a) the predicate is wrong, (b) the table flip
is wrong, or (c) the WHT kernel is wrong. The differential tests in
§9.2 catch (c); the unit test in §9.1 catches (a); the integration test
in §9.7 itself catches (b).

### 9.8 Miri

Pure-Rust scalar paths in `rvp9-core::transforms` carry no `unsafe`
beyond the `extern "C" fn` calls into SIMD wrappers. Miri can run
every test in this LLD's test suite *with SIMD dispatch disabled* (a
build feature `force-scalar-dispatch`) and verify aliasing /
provenance correctness on the scan-table reads and the dispatch helper.

The transform module sits in `rvp9-core` which is forbidden-`unsafe`
per HLD-arch §5.3 (the SIMD intrinsics live in `rvp9_core::simd::*`
sub-modules, not in `transforms` itself). So Miri-cleanness is a hard
gate. Stage 5 PLAN.md §6 task 5 owns the workspace-wide Miri runs.

### 9.9 Performance regression CI

Per HLD-simd §7, bench gates per kernel. For this module:

- `bench::fdct_per_size` — measure ns/call for each forward kernel on
  a synthetic 1080p frame's worth of blocks.
- `bench::idct_per_bucket` — measure ns/call for each inverse kernel
  at each EOB bucket; the fast paths must be faster than the slow
  paths.
- `bench::round_trip_64x64sb` — measure end-to-end forward+quant+
  inverse for a whole 64×64 superblock to catch dispatch overhead.

A 5% regression on any of these triggers CI failure. HLD-simd §7 owns
the harness; this LLD declares which kernels live in this module's
bench surface.

---

## 10. Summary of decisions tied to this LLD

- D-01 RESOLVED: 8-bit only. No `vpx_highbd_*` kernels. The
  `intra_mode_to_tx_type` table and every kernel signature uses
  `u8` / `i16` / `i32`. v2 expansion is type-parametric and
  contained at the `Pixel` / `Coeff` / `TranLow` aliases (HLD-arch §7).
- D-13 RESOLVED: no altref. Doesn't directly touch this LLD — transforms
  are altref-agnostic — but the absence of altref means the encoder's
  per-frame `PerFrameTransforms` is built per frame with no lookahead
  side-effects.
- L-28 RESOLVED: trellis-quant skipped. `vp9_optimize_b` not ported.
  See §7.1.
- L-71: 4×4 reachability at real-time. Confirmed: 4×4 is reachable in
  intra-only paths (keyframes) and through `tx_mode = TxModeSelect` at
  intra sub-8×8 partitions per A2 §4.10 calculate_tx_size. The 4×4
  forward and inverse kernels are shipped. `notes/vp9-analysis/decisions-needed.md`
  line 615 confirms: "A2 §4 confirms non-RD path uses `TX_16X16` max
  (`tx_size_search_method = USE_TX_8X8` per A6 §5.12). But A8 notes:
  KF intra-only blocks can hit TX_4X4 through `tx_mode = TX_MODE_SELECT`
  and the per-block tx-size decision in `select_tx_mode`. Stage 2
  LLD-transforms confirms (likely keeps the kernel; the SIMD cost is
  small)."

  **Confirmed: 4×4 transforms are in v1 scope.** All four forward 4×4
  kernels (`fdct4x4`, `fdct4x4_1`, `fht4x4`, `fwht4x4`) and all six
  inverse 4×4 kernels (`idct4x4_16_add`, `idct4x4_1_add`,
  `iwht4x4_16_add`, `iwht4x4_1_add`, `iht4x4_16_add`) ship with full
  ISA coverage per §6.1.
- L-71 follow-up (resolution): close `L-71` in `decisions-needed.md` as
  RESOLVED via this LLD §5.5 + §10.

---

_End of LLD-transforms._
