# LLD: quant

Module: `rvp9_core::quant`.

Stage 1 anchors:
- A5 (rate control) §2, §3.3, §4.1, §4.2, §5.1-§5.3, §5.4, §7 Q-RC-4 /
  Q-RC-16 / Q-RC-30 — quant tables, qindex search, qindex→qstep, and the
  `vp9_compute_qdelta_by_rate` rate-ratio inverse.
- A4 (bitstream / entropy) §3.2 (uncompressed-header `encode_quantization`),
  §5.1 (`QINDEX_BITS`).
- A1 §2.8 (`vp9_set_quantizer` slot in the encode lifecycle, step 25 of
  the per-frame call chain).
- A2 §2.6, §6.9 (RD path → quantize_b; non-RD path → quantize_fp).
- HLD references: `notes/hld/00-architecture.md` §2.1 (module exists in
  `rvp9-core`), §6 (integer-width policy + `wrapping_*`), §7.7 (8-bit-only
  quant gating); `notes/hld/02-simd-dispatch.md` §3.15 (quantize_fp /
  quantize_b RTCD signatures, line 784-822).
- decisions-needed.md L-25, L-26.

---

## 1. Purpose

The `quant` module owns every quantizer-related primitive shared between
the encoder and decoder:

1. The two **8-bit qindex→quantizer lookup tables** (`dc_qlookup`,
   `ac_qlookup`, 256 entries each, `int16_t` per libvpx).
2. The **qindex resolution** function `vp9_get_qindex(seg, segment_id,
   base_qindex)` that merges base qindex with per-segment delta or
   absolute overrides from the segmentation feature
   `SEG_LVL_ALT_Q = 0`.
3. The encoder-only `QUANTS` table-of-tables (4096 `int16_t`s per chroma
   plane × Y/UV × quantization mode) generated at frame init via
   `invert_quant` and friends, plus per-plane dequant lookups.
4. The forward `vp9_quantize_fp` / `vp9_quantize_fp_32x32` and
   `vpx_quantize_b` / `vpx_quantize_b_32x32` kernels themselves (scalar
   reference; SIMD dispatch lives in `simd::dispatch` per HLD-simd
   §3.15).
5. The qindex→qstep float bridge `vp9_convert_qindex_to_q` and the two
   `O(QINDEX_RANGE)` inverse search routines that rate control depends
   on: `vp9_compute_qdelta` and `vp9_compute_qdelta_by_rate`. **L-26 is
   resolved here**: match libvpx's linear loop exactly and test
   exhaustively over all 256 qindex values × frame types × a curated
   ratio grid.

The module is split into three sub-files mirroring libvpx layout:

- `rvp9_core::quant::common` — the lookup tables and resolution helpers
  (corresponds to `vp9/common/vp9_quant_common.{c,h}`).
- `rvp9_core::quant::tables` — encoder-side `Quants` table generation
  (corresponds to `vp9/encoder/vp9_quantize.{c,h}`).
- `rvp9_core::quant::kernels` — the scalar reference kernels and the
  `simd::dispatch` slot accessors (corresponds to
  `vp9/encoder/vp9_quantize.c` plus `vpx_dsp/quantize.c`).

The float bridge + `vp9_compute_qdelta` / `vp9_compute_qdelta_by_rate`
live in `rvp9_core::quant::common` (not `rvp9-encoder::rate_control`)
because the decoder also needs `vp9_convert_qindex_to_q` for the few
spec-defined places it parses quant info (none touch qdelta_by_rate, but
keeping the family colocated avoids future drift).

Scope guardrails per PLAN.md and the dispatch prompt:

- **8-bit only.** `dc_qlookup_10`, `dc_qlookup_12`, `ac_qlookup_10`,
  `ac_qlookup_12`, `vp9_highbd_quantize_fp*`, `vpx_highbd_quantize_b*`
  are out of scope. `vp9_dc_quant` / `vp9_ac_quant` lose their `bit_depth`
  parameter at the Rust level (the alias is `BitDepth::Eight` by D-01
  RESOLVED; we drop the runtime switch).
- **No altref.** No effect on this module — altref does not introduce
  any quant table.
- **No trellis quant (L-28 ledgered).** `vp9_optimize_b` is not ported;
  irrelevant to this LLD.

---

## 2. libvpx origin

| libvpx file | Lines | Maps to |
|---|---|---|
| `vp9/common/vp9_quant_common.h` | 21-30 | `MINQ`, `MAXQ`, `QINDEX_RANGE`, `QINDEX_BITS` constants; `vp9_dc_quant`, `vp9_ac_quant`, `vp9_get_qindex` signatures |
| `vp9/common/vp9_quant_common.c` | 15-35 | `dc_qlookup[256]` (8-bit) — verbatim transcribed into `rvp9_core::quant::common::DC_QLOOKUP` |
| `vp9/common/vp9_quant_common.c` | 89-110 | `ac_qlookup[256]` (8-bit) — verbatim into `AC_QLOOKUP` |
| `vp9/common/vp9_quant_common.c` | 38-86, 113-161 | HBD lookup tables — **excluded** (D-01 RESOLVED); citation kept for traceability |
| `vp9/common/vp9_quant_common.c` | 164-178 | `vp9_dc_quant(qindex, delta, bit_depth)` → `dc_qlookup[clamp(qindex+delta, 0, MAXQ)]` |
| `vp9/common/vp9_quant_common.c` | 180-194 | `vp9_ac_quant(qindex, delta, bit_depth)` |
| `vp9/common/vp9_quant_common.c` | 196-206 | `vp9_get_qindex(seg, segment_id, base_qindex)` — segment override logic |
| `vp9/common/vp9_seg_common.h` | 20-26, 35, 42-55, 57-78 | Segmentation feature interface (`SEG_LVL_ALT_Q`, `SEGMENT_DELTADATA`, `SEGMENT_ABSDATA`, `segfeature_active`, `get_segdata`) — surfaced via a thin Rust trait in `rvp9_core::quant::common` so `vp9_get_qindex` does not depend on the full bitstream header type |
| `vp9/encoder/vp9_quantize.h` | 21-38 | `QUANTS` struct (8 `int16_t[QINDEX_RANGE][8]` table-of-tables) |
| `vp9/encoder/vp9_quantize.c` | 25-55 | `vp9_quantize_fp_c` scalar kernel |
| `vp9/encoder/vp9_quantize.c` | 91-122 | `vp9_quantize_fp_32x32_c` scalar kernel |
| `vp9/encoder/vp9_quantize.c` | 158-166 | `invert_quant` helper (reciprocal-multiply prep) |
| `vp9/encoder/vp9_quantize.c` | 168-182 | `get_qzbin_factor` |
| `vp9/encoder/vp9_quantize.c` | 184-244 | `vp9_init_quantizer` — fills the 8 sub-tables of `QUANTS` for all 256 qindex values |
| `vp9/encoder/vp9_quantize.c` | 246-285 | `vp9_init_plane_quantizers` — per-superblock plane wiring including segment delta resolution |
| `vp9/encoder/vp9_quantize.c` | 287-289 | `vp9_frame_init_quantizer` |
| `vp9/encoder/vp9_quantize.c` | 291-303 | `vp9_set_quantizer` — main entry point used by rate control |
| `vp9/encoder/vp9_quantize.c` | 305-326 | `vp9_quantizer_to_qindex` / `vp9_qindex_to_quantizer` (1-pass external `--cq-level` mapping; CLI-facing only) |
| `vp9/encoder/vp9_encoder.h` | 732-733 | `cpi->y_dequant[QINDEX_RANGE][8]`, `cpi->uv_dequant[QINDEX_RANGE][8]` — the dequant tables consumed by `vp9_init_plane_quantizers` |
| `vpx_dsp/quantize.c` | 118-165 | `vpx_quantize_b_c` scalar |
| `vpx_dsp/quantize.c` | 216-275 | `vpx_quantize_b_32x32_c` scalar |
| `vp9/encoder/vp9_ratectrl.c` | 159-172 | `vp9_convert_qindex_to_q` (qindex→qstep float) |
| `vp9/encoder/vp9_ratectrl.c` | 174-183 | `vp9_convert_q_to_qindex` (inverse, smallest qindex with `qstep ≥ q_val`) |
| `vp9/encoder/vp9_ratectrl.c` | 185-196 | `vp9_rc_bits_per_mb` — needed by `vp9_compute_qdelta_by_rate` |
| `vp9/encoder/vp9_ratectrl.c` | 2541-2560 | `vp9_compute_qdelta(rc, qstart, qtarget, bit_depth)` |
| `vp9/encoder/vp9_ratectrl.c` | 2562-2584 | `vp9_compute_qdelta_by_rate(rc, frame_type, qindex, rate_target_ratio, bit_depth)` — **L-26** anchor |
| `vp9/encoder/vp9_ratectrl.c` | 689-735 | `get_rate_correction_factor` / `set_rate_correction_factor` — show the `rcf_mult[frame_size_selector]` scale-on-read / unscale-on-write (**L-25** anchor; the value lives in `rate_control` but the off-by-one trap shape is best characterised here next to the qindex→qstep family it feeds) |
| `vp9/encoder/vp9_ratectrl.h` | 79-86 | `rcf_mult[FRAME_SCALE_STEPS] = { 1.0, 2.0 }` (load-bearing for L-25) |
| `vp9/encoder/vp9_bitstream.c` | 741-757 | `write_delta_q` / `encode_quantization` — wire format for `base_qindex` + the three deltas; consumed by the bitstream LLD but the value source is here |
| `vp9/common/vp9_rtcd_defs.pl` | 134-138 | `vp9_quantize_fp` / `_32x32` RTCD entries |
| `vpx_dsp/vpx_dsp_rtcd_defs.pl` | 725-729 | `vpx_quantize_b` / `_32x32` RTCD entries |

Two libvpx files this module **does not** port:

- `vp9/encoder/vp9_encodemb.c` — trellis-quant (`vp9_optimize_b`). L-28
  RESOLVED in the divergence ledger: at cpu-used 6-8
  `sf->optimize_coefficients = 0`, the path is dead.
- The HBD branches of every file above (see scope guardrails).

---

## 3. Public API

All items are `pub` from `rvp9_core::quant` unless noted. Signatures
respect HLD §6 integer-width policy: every libvpx `int` is `i32`,
`int16_t` is `i16`, `uint8_t` is `u8`. No `unsafe` on the safe-Rust
surface; the kernel functions exposed through `simd::dispatch` are
`unsafe extern "C" fn` per HLD-simd §3.15.

### 3.1 Constants

```rust
/// vp9_quant_common.h:21
pub const MINQ: u8 = 0;
/// vp9_quant_common.h:22
pub const MAXQ: u8 = 255;
/// vp9_quant_common.h:23
pub const QINDEX_RANGE: usize = 256;  // MAXQ - MINQ + 1
/// vp9_quant_common.h:24 — number of bits used to serialize base_qindex
/// in the uncompressed header (encode_quantization, vp9_bitstream.c:753).
pub const QINDEX_BITS: u32 = 8;
```

### 3.2 Tables (re-exported `pub const`)

```rust
/// vp9_quant_common.c:15-35. AC and DC lookups are i16, not u16:
/// vp9_quantize_fp's multiplications need signed widening to i32 before
/// the >>16 shift.
pub const DC_QLOOKUP: [i16; QINDEX_RANGE] = [ /* see §5.1 */ ];

/// vp9_quant_common.c:89-110.
pub const AC_QLOOKUP: [i16; QINDEX_RANGE] = [ /* see §5.2 */ ];

/// vp9_quantize.c:307-313. Maps the user-facing 0-63 `--cq-level`
/// quantizer to the internal 0-255 qindex. Only used by the CLI /
/// encoder config; the encoder loop only sees qindex.
pub const QUANTIZER_TO_QINDEX: [i32; 64] = [ /* see §5.3 */ ];
```

### 3.3 Quant table lookups

```rust
/// vp9_quant_common.c:164-178. 8-bit only per D-01 RESOLVED. Clamps
/// `qindex + delta` to `[0, MAXQ]` matching libvpx.
#[inline]
pub fn vp9_dc_quant(qindex: i32, delta: i32) -> i16;

/// vp9_quant_common.c:180-194. 8-bit only.
#[inline]
pub fn vp9_ac_quant(qindex: i32, delta: i32) -> i16;
```

### 3.4 Segment qindex resolution

Trait-mediated to avoid pulling the full bitstream header into
`rvp9_core::quant`. Implemented by `bitstream::header::Segmentation`,
`encoder::aq::CyclicRefreshSegmentation`, and a `Disabled` no-op.

```rust
/// vp9_seg_common.h:57-78 — segmentation feature accessors.
pub trait SegmentationView {
    fn segfeature_active(&self, segment_id: u8, feature: SegFeature) -> bool;
    fn segdata(&self, segment_id: u8, feature: SegFeature) -> i16;
    fn abs_delta(&self) -> bool;  // SEGMENT_ABSDATA vs SEGMENT_DELTADATA
}

/// vp9_seg_common.h:34-40.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SegFeature { AltQ = 0, AltLf = 1, RefFrame = 2, Skip = 3 }

/// vp9_quant_common.c:196-206. Resolves base + segment override.
#[inline]
pub fn vp9_get_qindex<S: SegmentationView>(
    seg: &S,
    segment_id: u8,
    base_qindex: i32,
) -> i32;
```

### 3.5 qindex → qstep float bridge

```rust
/// vp9_ratectrl.c:159-172. Returns `ac_qlookup[qindex] / 4.0` (8-bit
/// scale). f64 because rate-control downstream is f64. **L-26**:
/// every q-from-qindex site must call this — no inlined formula.
#[inline]
pub fn vp9_convert_qindex_to_q(qindex: i32) -> f64;

/// vp9_ratectrl.c:174-183. Inverse: smallest qindex with `qstep >=
/// q_val`. Returns MAXQ if none satisfies. O(QINDEX_RANGE).
#[inline]
pub fn vp9_convert_q_to_qindex(q_val: f64) -> i32;
```

### 3.6 Rate-control inverse functions (L-26)

```rust
/// vp9_ratectrl.c:185-196. Pure: (frame_type, qindex, correction).
/// Lives here because `vp9_compute_qdelta_by_rate` (below) calls it
/// with correction = 1.0, and we keep the family colocated.
/// Asserts `correction_factor ∈ [MIN_BPB_FACTOR, MAX_BPB_FACTOR]`.
pub fn vp9_rc_bits_per_mb(
    frame_type: FrameType,
    qindex: i32,
    correction_factor: f64,
) -> i32;

/// vp9_ratectrl.c:2541-2560. Returns target_index - start_index where
/// each is the smallest qindex with `qstep >= qstart` (resp.
/// `qtarget`) inside `[best_quality, worst_quality)`. Bounds passed
/// explicitly (no global state). See §5.7 for the off-by-one
/// behaviour when no i satisfies.
pub fn vp9_compute_qdelta(
    best_quality: i32,
    worst_quality: i32,
    qstart: f64,
    qtarget: f64,
) -> i32;

/// vp9_ratectrl.c:2562-2584. **L-26 RESOLVED:** match libvpx
/// exactly. See §5.8 for the algorithm + casting semantics.
pub fn vp9_compute_qdelta_by_rate(
    best_quality: i32,
    worst_quality: i32,
    frame_type: FrameType,
    qindex: i32,
    rate_target_ratio: f64,
) -> i32;
```

`FrameType` re-exports from `rvp9_core::bitstream::header`. The
bits-per-mb formula only branches on `KeyFrame` vs everything else
(`vp9_ratectrl.c:188` — enumerator = `KEY ? 2_700_000 : 1_800_000`).

### 3.7 External-quantizer mapping (encoder CLI)

```rust
/// vp9_quantize.c:315-317. 0-63 user-facing → 0-255 internal.
pub fn vp9_quantizer_to_qindex(quantizer: u8) -> i32;

/// vp9_quantize.c:319-326. Inverse: smallest u8 quantizer such that
/// `quantizer_to_qindex[q] >= qindex`. Returns 63 if no match.
pub fn vp9_qindex_to_quantizer(qindex: i32) -> u8;
```

### 3.8 Encoder-side `Quants` (table-of-tables for SIMD)

The eight `int16_t[QINDEX_RANGE][8]` sub-tables that the SIMD quant
kernels read (`vp9_quantize.h:21-38`). Allocated once per encoder,
refilled by `init_quantizer` (§3.8) whenever any delta_q changes.
16-byte alignment is mandatory for SIMD lane loads (`#[repr(C,
align(16))]`). Heap-allocated inside `EncoderContext`; total ~32 KiB.

```rust
#[repr(C, align(16))]
pub struct Quants {
    pub y_quant:        [[i16; 8]; QINDEX_RANGE],
    pub y_quant_shift:  [[i16; 8]; QINDEX_RANGE],
    pub y_zbin:         [[i16; 8]; QINDEX_RANGE],
    pub y_round:        [[i16; 8]; QINDEX_RANGE],
    pub y_quant_fp:     [[i16; 8]; QINDEX_RANGE],
    pub y_round_fp:     [[i16; 8]; QINDEX_RANGE],
    pub uv_quant:       [[i16; 8]; QINDEX_RANGE],
    pub uv_quant_shift: [[i16; 8]; QINDEX_RANGE],
    pub uv_zbin:        [[i16; 8]; QINDEX_RANGE],
    pub uv_round:       [[i16; 8]; QINDEX_RANGE],
    pub uv_quant_fp:    [[i16; 8]; QINDEX_RANGE],
    pub uv_round_fp:    [[i16; 8]; QINDEX_RANGE],
}

/// Sister tables to `Quants` (`vp9_encoder.h:732-733`). libvpx puts
/// these on VP9_COMP directly; we bundle to colocate the same
/// lifecycle.
#[repr(C, align(16))]
pub struct Dequants {
    pub y_dequant:  [[i16; 8]; QINDEX_RANGE],
    pub uv_dequant: [[i16; 8]; QINDEX_RANGE],
}

impl Quants  { pub fn new_zeroed() -> Box<Self>; }
impl Dequants { pub fn new_zeroed() -> Box<Self>; }

/// Delta-q values from the bitstream header. VP9 spec §6.2.7: each
/// is a signed 4-bit value (−15..=+15, sign-magnitude wire form).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub struct QuantDeltas {
    pub y_dc_delta_q:  i8,  // vp9_set_quantizer:296
    pub uv_dc_delta_q: i8,  // vp9_set_quantizer:297
    pub uv_ac_delta_q: i8,  // vp9_set_quantizer:298
}

/// vp9_quantize.c:184-244. Fills both tables for all 256 qindex
/// values from `(deltas, sharpness)`. 8-bit path; `bit_depth` param
/// dropped (D-01). 2048 inner iterations; called whenever delta_q or
/// sharpness changes (typically once at encoder init). Mandatory
/// rebuild — `base_qindex` is not consumed here, but any delta
/// change invalidates every row.
pub fn init_quantizer(
    quants: &mut Quants,
    dequants: &mut Dequants,
    deltas: QuantDeltas,
    sharpness: u8,
);
```

### 3.9 Per-block plane wiring

```rust
/// vp9_quantize.c:246-285. Computes per-plane quant arrays by
/// selecting the row of `Quants` at the resolved (segment-adjusted)
/// qindex. Returns the resolved qindex; the caller stores it as
/// `MacroblockState.q_index` (consumed downstream by ME and
/// mode-decision). The view borrows from `Quants` / `Dequants`;
/// 16-byte alignment of the underlying tables guarantees aligned
/// SIMD loads.
pub fn init_plane_quantizers<'q, S: SegmentationView>(
    quants: &'q Quants,
    dequants: &'q Dequants,
    seg: &S,
    segment_id: u8,
    base_qindex: i32,
) -> (i32 /* qindex */, MacroblockPlaneQuant<'q>);

/// vp9_block.h:30-46 quant fields. `'q` ties to the parent `Quants` /
/// `Dequants`.
pub struct MacroblockPlaneQuant<'q> {
    pub zbin:        &'q [i16; 8],
    pub round:       &'q [i16; 8],
    pub quant:       &'q [i16; 8],
    pub quant_shift: &'q [i16; 8],
    pub dequant:     &'q [i16; 8],
    pub round_fp:    &'q [i16; 8],
    pub quant_fp:    &'q [i16; 8],
    /// vp9_block.h:45. Squared-zbin for hot-path comparison.
    pub quant_thred: [i64; 2],
}
```

### 3.10 Scalar quant kernels (and SIMD dispatch slots)

Two exposure routes: (1) scalar callable `kernels::*_scalar` for the
differential harness and no-SIMD builds; (2) function-pointer slots
on `rvp9_core::simd::SimdDispatch` (HLD-simd §3.15 line 1333-1336).
Encoder code goes through dispatch, not the scalar function name.

Signatures match libvpx ABI bit-for-bit (HLD-simd §3.15 re-quote):

```rust
/// vp9_rtcd_defs.pl:134. Non-RD encoder path (A2 §6.9).
pub type QuantizeFpFn = unsafe extern "C" fn(
    coeff: *const i16,
    n_coeffs: isize,
    mb_plane: *const MacroblockPlane,
    qcoeff: *mut i16,
    dqcoeff: *mut i16,
    dequant: *const i16,
    eob: *mut u16,
    scan_order: *const ScanOrder,
);

/// vpx_dsp_rtcd_defs.pl:725. RD-loop variant + decoder dequant.
pub type QuantizeBFn = QuantizeFpFn;

/// vpx_dsp_rtcd_defs.pl:728. n_coeffs implicit = 1024.
pub type QuantizeB32x32Fn = unsafe extern "C" fn(
    coeff: *const i16,
    mb_plane: *const MacroblockPlane,
    qcoeff: *mut i16,
    dqcoeff: *mut i16,
    dequant: *const i16,
    eob: *mut u16,
    scan_order: *const ScanOrder,
);
```

`MacroblockPlane` and `ScanOrder` are FFI POD structs owned by
`rvp9_core::simd` (HLD-simd §3.15 line 813); re-exported here.

---

## 4. Types

### 4.1 `BitDepth` is *not* a field in v1

D-01 RESOLVED. There is no `BitDepth` enum on the quant module signature
— every function is implicitly 8-bit. The only public reference to bit
depth in our quant API is the **absence** of the libvpx
`vpx_bit_depth_t bit_depth` parameter. The HBD wrappers (`switch
(bit_depth) { ... }` in `vp9_dc_quant` etc.) are dropped.

### 4.2 `FrameType`

Re-exported from `rvp9_core::bitstream::header`. The quant module only
distinguishes `KeyFrame` vs other (for `vp9_rc_bits_per_mb`'s
`enumerator`). Definition lives in the bitstream LLD; this LLD just
documents the usage.

### 4.3 `SegmentationView` trait

Defined in §3.4 above. Trait rather than a concrete `&Segmentation`
struct because:

1. Decoder, encoder bitstream parse, and cyclic-refresh AQ all want to
   call `vp9_get_qindex` but each carries the segmentation differently
   (the encoder updates the table mid-frame via
   `vp9_cyclic_refresh_setup`; the decoder reads it once per frame).
2. Trait dispatch is statically resolved at compile time at every
   call site (`<S: SegmentationView>`); zero-cost.

### 4.4 `MIN_BPB_FACTOR` / `MAX_BPB_FACTOR`

`vp9_ratectrl.c:44-45`:

```rust
pub const MIN_BPB_FACTOR: f64 = 0.005;
pub const MAX_BPB_FACTOR: f64 = 50.0;
```

These are used by `vp9_rc_bits_per_mb` as the assertion bounds on
`correction_factor`. They also bound the cleanup in `set_rate_correction_factor`
(`vp9_ratectrl.c:719`), which is owned by the rate-control LLD — but
the constants belong here, next to the formula they bound.

### 4.5 `FRAME_SCALE_STEPS` / `rcf_mult` (L-25 fragment)

`vp9_ratectrl.h:79-86` declares:

```c
#define FRAME_SCALE_STEPS 2
static const int    frame_scale_factor[FRAME_SCALE_STEPS] = { 16, 24 };
static const double rate_thresh_mult[FRAME_SCALE_STEPS]   = { 1.0, 2.0 };
static const double rcf_mult[FRAME_SCALE_STEPS]           = { 1.0, 2.0 };
```

Rate control owns the use sites; this LLD only re-exports the constants
so `rate_control.md` can pull them without re-declaring:

```rust
pub const FRAME_SCALE_STEPS: usize = 2;
pub const FRAME_SCALE_FACTOR: [i32; FRAME_SCALE_STEPS] = [16, 24];
pub const RATE_THRESH_MULT:   [f64; FRAME_SCALE_STEPS] = [1.0, 2.0];
pub const RCF_MULT:           [f64; FRAME_SCALE_STEPS] = [1.0, 2.0];
```

The off-by-one trap **lives in rate_control** (`get_rate_correction_factor`
at line 708 multiplies by `rcf_mult`, `set_rate_correction_factor` at
line 717 divides). The trap is: if you forget to scale-on-read OR
unscale-on-write, the stored `rate_correction_factors[KF_STD]` ends up
permanently 2× off after a single resize event. See §7 entry "L-25".

---

## 5. Internal algorithm

### 5.1 `DC_QLOOKUP[256]` — verbatim from `vp9_quant_common.c:15-35`

```text
  4,    8,    8,    9,   10,   11,   12,   12,   13,   14,   15,   16,   17,
 18,   19,   19,   20,   21,   22,   23,   24,   25,   26,   26,   27,   28,
 29,   30,   31,   32,   32,   33,   34,   35,   36,   37,   38,   38,   39,
 40,   41,   42,   43,   43,   44,   45,   46,   47,   48,   48,   49,   50,
 51,   52,   53,   53,   54,   55,   56,   57,   57,   58,   59,   60,   61,
 62,   62,   63,   64,   65,   66,   66,   67,   68,   69,   70,   70,   71,
 72,   73,   74,   74,   75,   76,   77,   78,   78,   79,   80,   81,   81,
 82,   83,   84,   85,   85,   87,   88,   90,   92,   93,   95,   96,   98,
 99,  101,  102,  104,  105,  107,  108,  110,  111,  113,  114,  116,  117,
118,  120,  121,  123,  125,  127,  129,  131,  134,  136,  138,  140,  142,
144,  146,  148,  150,  152,  154,  156,  158,  161,  164,  166,  169,  172,
174,  177,  180,  182,  185,  187,  190,  192,  195,  199,  202,  205,  208,
211,  214,  217,  220,  223,  226,  230,  233,  237,  240,  243,  247,  250,
253,  257,  261,  265,  269,  272,  276,  280,  284,  288,  292,  296,  300,
304,  309,  313,  317,  322,  326,  330,  335,  340,  344,  349,  354,  359,
364,  369,  374,  379,  384,  389,  395,  400,  406,  411,  417,  423,  429,
435,  441,  447,  454,  461,  467,  475,  482,  489,  497,  505,  513,  522,
530,  539,  549,  559,  569,  579,  590,  602,  614,  626,  640,  654,  668,
684,  700,  717,  736,  755,  775,  796,  819,  843,  869,  896,  925,  955,
988, 1022, 1058, 1098, 1139, 1184, 1232, 1282, 1336,
```

### 5.2 `AC_QLOOKUP[256]` — verbatim from `vp9_quant_common.c:89-110`

```text
  4,    8,    9,   10,   11,   12,   13,   14,   15,   16,   17,   18,   19,
 20,   21,   22,   23,   24,   25,   26,   27,   28,   29,   30,   31,   32,
 33,   34,   35,   36,   37,   38,   39,   40,   41,   42,   43,   44,   45,
 46,   47,   48,   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,
 59,   60,   61,   62,   63,   64,   65,   66,   67,   68,   69,   70,   71,
 72,   73,   74,   75,   76,   77,   78,   79,   80,   81,   82,   83,   84,
 85,   86,   87,   88,   89,   90,   91,   92,   93,   94,   95,   96,   97,
 98,   99,  100,  101,  102,  104,  106,  108,  110,  112,  114,  116,  118,
120,  122,  124,  126,  128,  130,  132,  134,  136,  138,  140,  142,  144,
146,  148,  150,  152,  155,  158,  161,  164,  167,  170,  173,  176,  179,
182,  185,  188,  191,  194,  197,  200,  203,  207,  211,  215,  219,  223,
227,  231,  235,  239,  243,  247,  251,  255,  260,  265,  270,  275,  280,
285,  290,  295,  300,  305,  311,  317,  323,  329,  335,  341,  347,  353,
359,  366,  373,  380,  387,  394,  401,  408,  416,  424,  432,  440,  448,
456,  465,  474,  483,  492,  501,  510,  520,  530,  540,  550,  560,  571,
582,  593,  604,  615,  627,  639,  651,  663,  676,  689,  702,  715,  729,
743,  757,  771,  786,  801,  816,  832,  848,  864,  881,  898,  915,  933,
951,  969,  988, 1007, 1026, 1046, 1066, 1087, 1108, 1129, 1151, 1173, 1196,
1219, 1243, 1267, 1292, 1317, 1343, 1369, 1396, 1423, 1451, 1479, 1508, 1537,
1567, 1597, 1628, 1660, 1692, 1725, 1759, 1793, 1828,
```

These will be transcribed into `rvp9_core::quant::common` as a single
`const fn`-friendly `[i16; 256]` literal each. Test §9.1 below
asserts the entire content matches the libvpx C arrays via a regenerable
fixture (run once from `third_party/libvpx`, diff against ours).

### 5.3 `QUANTIZER_TO_QINDEX[64]` — `vp9_quantize.c:307-313`

```text
  0,   4,   8,  12,  16,  20,  24,  28,  32,  36,  40,  44,  48,  52,  56,  60,
 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112, 116, 120, 124,
128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188,
192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 249, 255,
```

Note the irregular last two entries (`244, 249, 255` instead of `244,
248, 252`). This is intentional in libvpx; verify in tests.

### 5.4 `vp9_dc_quant` / `vp9_ac_quant`

`vp9_quant_common.c:164-194` post-HBD-removal: simple
`(DC|AC)_QLOOKUP[clamp(qindex + delta, 0, MAXQ)]`. Inputs are bounded
`qindex ∈ [0,255]`, `delta ∈ [-128,128]` (segment data is `i16` but
practical range fits `i8`); use `wrapping_add` per HLD §6.2.

### 5.5 `vp9_get_qindex` — segment Q application

`vp9_quant_common.c:196-206`:

```text
if segfeature_active(seg, segment_id, SEG_LVL_ALT_Q):
    data       = segdata(seg, segment_id, SEG_LVL_ALT_Q)   // i16
    seg_qindex = abs_delta ? data : base_qindex + data
    return clamp(seg_qindex, 0, MAXQ)
else:
    return base_qindex
```

Two consumers: the **decoder** (every transform block,
`vp9_decodemb.c`) and the **encoder** `init_plane_quantizers`
(`vp9_quantize.c:251`, once per superblock). AQ-3 cyclic refresh
(`vp9_aq_cyclicrefresh.c:649-672`) writes `qindex_delta[i]` into
`feature_data[i][SEG_LVL_ALT_Q]` for segments 1 and 2 with
`abs_delta = false`; segment 0 has `SEG_LVL_ALT_Q` disabled and
returns `base_qindex` unchanged. We still implement the absolute path
for decoder spec-compliance with arbitrary upstream encoders.

### 5.6 Floating-point bridge `vp9_convert_qindex_to_q`

`vp9_ratectrl.c:159-172`. 8-bit only: returns `vp9_ac_quant(qindex,
0) as f64 / 4.0`. The `/4.0` is the historical VP8 quantizer scale.
Determinism: libvpx emits `vpx_clear_system_state()` (x87 reset on
x86); we are SSE2-only on x86_64 and don't target i686, so no
equivalent is needed.

### 5.7 `vp9_compute_qdelta` (L-26 sibling)

`vp9_ratectrl.c:2541-2560`:

```text
start_index  = worst_quality   // sentinel
target_index = worst_quality
for i in best_quality .. worst_quality:        // exclusive
    start_index = i
    if vp9_convert_qindex_to_q(i) >= qstart: break
for i in best_quality .. worst_quality:
    target_index = i
    if vp9_convert_qindex_to_q(i) >= qtarget: break
return target_index - start_index
```

Off-by-one details (must match libvpx exactly):

1. The C loop is exclusive at `worst_quality`. If no i satisfies the
   condition, `start_index = worst_quality - 1` from the final
   in-loop assignment — **NOT** the `worst_quality` sentinel
   (`vp9_ratectrl.c:2548-2551`). Rust `for i in best..worst { start =
   i; if ... break }` reproduces this.
2. If `best_quality == worst_quality`, the loop runs zero times and
   both indices remain at `worst_quality`. libvpx initialises both
   to `rc->worst_quality` at line 2543-2544; mirror exactly.

### 5.8 `vp9_compute_qdelta_by_rate` — L-26 RESOLVED

`vp9_ratectrl.c:2562-2584`:

```text
vp9_compute_qdelta_by_rate(best_quality, worst_quality, frame_type,
                           qindex, rate_target_ratio):
    base_bits_per_mb   = vp9_rc_bits_per_mb(frame_type, qindex, 1.0)
    target_bits_per_mb = (int)(rate_target_ratio * base_bits_per_mb)
    target_index = worst_quality       // sentinel
    for i in best_quality .. worst_quality:
        if vp9_rc_bits_per_mb(frame_type, i, 1.0) <= target_bits_per_mb:
            target_index = i
            break
    return target_index - qindex
```

**Mandatory matching points (L-26):**

- The cast `(int)(rate_target_ratio * base_bits_per_mb)` truncates
  toward zero. Rust `as i32` on `f64` does the same per IEEE-754
  conversion. Verify with a test (§9.2).
- The loop runs `i ∈ [best, worst)` (exclusive). If no i satisfies the
  inequality, `target_index = worst_quality` from the initialiser; the
  return is `worst_quality - qindex` (which can be negative if
  `qindex > worst_quality`, but the caller should never pass such a
  qindex — assert in debug).
- The `1.0` correction factor is a **hard-coded ratio of 1.0**; we
  must not substitute the live rate-correction-factor here.
- `vp9_rc_bits_per_mb` itself: the C formula is

  ```text
  q   = ac_qlookup[qindex] / 4.0
  enu = (frame_type == KEY) ? 2700000 : 1800000
  enu += (int)(enu * q) >> 12          // line 194
  return (int)(enu * correction_factor / q)
  ```

  The `(int)(enu * q) >> 12` is the trap: libvpx truncates the
  intermediate to `int` *before* the shift, then arithmetic-shifts
  right by 12. We mirror with `((enu as f64 * q) as i32) >> 12`. The
  outer `(int)(...)` cast on the return is also a truncation; we use
  `as i32`. (See exhaustive coverage test §9.2.)

### 5.9 `init_quantizer` — encoder table fill

`vp9_quantize.c:184-244`. The 8-bit-only loop body, per qindex
`q ∈ 0..256`:

```text
qzbin_factor     = (q == 0) ? 64 : (vp9_dc_quant(q, 0) < 148) ? 84 : 80
qrounding_factor = (q == 0) ? 64 : 48
sharpness_adj    = 16 * (7 - sharpness) / 7
if sharpness > 0 && q > 0:
    qzbin_factor     = 64 + sharpness_adj
    qrounding_factor = 64 - sharpness_adj

for i in 0..2:    // 0=DC, 1=AC
    qrnd_fp = (i == 0) ? 48 : 42
    if q == 0: qrnd_fp = 64
    if sharpness > 0: qrnd_fp = 64 - sharpness_adj

    // Y plane
    quant = (i == 0) ? vp9_dc_quant(q, deltas.y_dc_delta_q)
                     : vp9_ac_quant(q, 0)
    (q_inv, q_shift) = invert_quant(quant)
    y_quant       [q][i] = q_inv
    y_quant_shift [q][i] = q_shift
    y_quant_fp    [q][i] = (1 << 16) / quant
    y_round_fp    [q][i] = (qrnd_fp * quant) >> 7
    y_zbin        [q][i] = round_power_of_two(qzbin_factor * quant, 7)
    y_round       [q][i] = (qrounding_factor * quant) >> 7
    y_dequant     [q][i] = quant

    // UV plane: mirror with uv_dc_delta_q / uv_ac_delta_q
    ...

// Broadcast i=1 (AC) values to slots 2..8 (vp9_quantize.c:226-242).
// SIMD lanes 2..7 must hold the same AC values for correct register
// broadcasts in vpx_quantize_b_sse2 etc.
for i in 2..8:
    {y,uv}_*[q][i] = {y,uv}_*[q][1]
    {y,uv}_dequant[q][i] = {y,uv}_dequant[q][1]
```

`invert_quant(d) → (quant, shift)` is the Newton reciprocal prep for
`vpx_quantize_b` (`vp9_quantize.c:158-166`):

```text
l = floor(log2(d))                 // (loop: while t > 1: t >>= 1; l += 1)
m = 1 + (1 << (16 + l)) / d
quant = (i16)(m - (1 << 16))       // truncating cast wraps; mirror with `as i16`
shift = 1 << (16 - l)
```

Used in `vpx_quantize_b_c:154-157` as `((tmp * quant) >> 16 + tmp) *
quant_shift >> 16`. Slot 0 vs slot 1 = DC vs AC; SIMD selects via
`(rc != 0)`. Slots 2..7 duplication is **mandatory** for SIMD lane
broadcasts.

### 5.10 `init_plane_quantizers` — per-block plane wiring

`vp9_quantize.c:246-285`. Plain table indexing after `vp9_get_qindex`:

```text
qindex = vp9_get_qindex(seg, segment_id, base_qindex)
mp_y = MacroblockPlaneQuant {
    quant:       &quants.y_quant[qindex],
    quant_fp:    &quants.y_quant_fp[qindex],
    round_fp:    &quants.y_round_fp[qindex],
    quant_shift: &quants.y_quant_shift[qindex],
    zbin:        &quants.y_zbin[qindex],
    round:       &quants.y_round[qindex],
    dequant:     &dequants.y_dequant[qindex],
    quant_thred: [
        (zbin[0] as i64).pow(2),
        (zbin[1] as i64).pow(2),
    ],
}
// UV mirrors (planes 1, 2 → mp_uv).
```

`set_error_per_bit(x, rdmult)` + `vp9_initialize_me_consts(...)`
happen outside this LLD (mode-decision / motion-estimation own those
side effects).

### 5.11 `vp9_set_quantizer` — entry point from rate control

`vp9_quantize.c:291-303`:

```text
common.base_qindex   = q
common.y_dc_delta_q  = 0
common.uv_dc_delta_q = 0
common.uv_ac_delta_q = 0
if cpi.oxcf.delta_q_uv != 0:
    common.uv_dc_delta_q = cpi.oxcf.delta_q_uv
    common.uv_ac_delta_q = cpi.oxcf.delta_q_uv
    init_quantizer(...)              // refill
```

Resets all three delta_q fields to zero each call; restores from
config + refills `Quants` only if `delta_q_uv != 0`. Call sites in
libvpx: `vp9_encoder.c:4172, 4201, 4262, 4651` — once per frame in
the main path plus twice on re-encode-overshoot. The rate_control
LLD owns the call sites; this LLD owns the body.

```rust
/// vp9_quantize.c:291-303. Mutates `header.base_qindex` and the
/// three delta_q fields. Calls `init_quantizer` if `config_delta_q_uv
/// != 0`.
pub fn set_quantizer(
    header: &mut UncompressedHeader,
    quants: &mut Quants,
    dequants: &mut Dequants,
    config_delta_q_uv: i8,
    sharpness: u8,
    new_qindex: i32,
);
```

`UncompressedHeader` is re-exported from `rvp9_core::bitstream::header`;
the relevant fields are `base_qindex: u8`, `y_dc_delta_q: i8`,
`uv_dc_delta_q: i8`, `uv_ac_delta_q: i8`.

### 5.12 Scalar quant kernels — `quantize_fp` family

`vp9_quantize.c:25-55`:

```text
zero qcoeff[0..n_coeffs], dqcoeff[0..n_coeffs]
eob_idx = -1
for i in 0..n_coeffs:
    rc    = scan[i]
    c     = coeff[rc] as i32
    sign  = c >> 31                  // arith shift, 0 or -1
    abs_c = (c ^ sign) - sign
    sel   = (rc != 0) as usize
    tmp   = clamp(abs_c + round_fp[sel], i16::MIN, i16::MAX)
    tmp   = (tmp * quant_fp[sel] as i32) >> 16
    qcoeff[rc]  = ((tmp ^ sign) - sign) as i16
    dqcoeff[rc] = qcoeff[rc] * dequant[sel] as i16
    if tmp != 0: eob_idx = i
eob = (eob_idx + 1) as u16
```

The 32×32 variant (`vp9_quantize_fp_32x32_c`,
`vp9_quantize.c:91-122`) differs:
1. Pre-skips if `abs_coeff < dequant[sel] >> 2` (line 111).
2. Uses `ROUND_POWER_OF_TWO(round[sel], 1)` (line 112).
3. Final shift `>> 15` not `>> 16` (line 114).
4. `dqcoeff = qcoeff * dequant / 2` (line 116) — `i32` widening
   needed to avoid `i16` sign-truncation on intermediates.

The 32×32 path is rarely hit in non-RD (`block_yrd` uses Hadamard,
not DCT-32) but is reachable for the final-commit forward DCT before
bitstream pack. Both ship.

`vpx_quantize_b_c` (`vpx_dsp/quantize.c:118-165`) is the
decoder-dequant + RD-loop variant. Differences from `_fp`:
- Pre-scan decrements `non_zero_count` from the scan tail while
  coefficients fall below `zbin` (lines 135-143).
- Two-stage Newton multiply (line 154-157).
- Uses `zbin[sel]` as deadzone; sub-zbin coefficients are zeroed.

The two kernels are **not interchangeable**. The encoder picks via
`sf->use_quant_fp` (always 1 at cpu-used 6-8 per A2); the decoder
always uses `_b`'s inverse path.

### 5.13 Where the SIMD lives

This LLD does **not** implement SIMD. Per HLD-arch §2.2, the SIMD
kernels live in `rvp9_core::simd::{x86_64, aarch64, scalar}`. The
quant module re-exports the FFI types, exposes the function-pointer
slots through `simd::dispatch()`, and ships the scalar reference at
`rvp9_core::simd::scalar::quant`. SIMD differential testing is
owned by HLD-simd §3.15 and LLD-simd-dispatch.

---

## 6. SIMD dispatch

Per HLD-simd §3.15 line 784-822, this module owns four scalar reference
implementations and four dispatch slots:

| Kernel | ISAs shipped (8-bit only) | libvpx scalar source |
|---|---|---|
| `vp9_quantize_fp` | scalar, sse2, ssse3, avx2, neon | `vp9_quantize.c:25-55` |
| `vp9_quantize_fp_32x32` | scalar, ssse3, avx2, neon | `vp9_quantize.c:91-122` |
| `vpx_quantize_b` | scalar, sse2, ssse3, avx, avx2, neon | `vpx_dsp/quantize.c:118-165` |
| `vpx_quantize_b_32x32` | scalar, ssse3, avx, avx2, neon | `vpx_dsp/quantize.c:216-275` |

For v1: scalar + sse2 (or ssse3 where libvpx omits sse2) + avx2 + neon.

**Differential test plan** (mirrors HLD-simd §3.15 line 815-822):

1. Random `i16` coefficient inputs from a Gaussian distribution (real
   coefficients cluster near zero).
2. Edge: all-zero coefficients → `eob == 0`, all qcoeff/dqcoeff zero.
3. Edge: DC-only → `eob == 1`.
4. Edge: deadzone boundary — coefficient exactly equal to `zbin[0]` and
   `zbin[1]` (probes `vpx_quantize_b`'s pre-scan).
5. **EOB tracking** — the SIMD variant uses horizontal reduction
   (`_mm_minpos_epu16` on x86, `vmaxv` on NEON). This is historically
   the most error-prone part of the kernel. Stage 4a SIMD bring-up
   must exercise EOB at every position.
6. Cross-ISA differential: for each kernel × {sse2, ssse3, avx2, neon},
   the output (qcoeff, dqcoeff, eob) must match scalar bit-exactly.

Mandatory FFI invariants:

- All input/output pointers must be 16-byte aligned. `Quants` is
  `#[repr(C, align(16))]` to enforce on the dequant side; the
  encoder-side `coeff` / `qcoeff` / `dqcoeff` buffers come from the
  transform LLD's superblock layout, which has the same alignment.
- `n_coeffs` is exactly `1 << (2 * log2_block_size)` per the calling
  convention — never zero.
- `scan_order.scan[..n_coeffs]` is a permutation of `0..n_coeffs`. Our
  transforms LLD owns the scan tables (verbatim from `vp9_scan.c`); we
  pass them as `&'static ScanOrder` references obtained from
  `rvp9_core::transforms::scan::get_scan(tx_size, tx_type)`.

---

## 7. Deliberate divergences from libvpx

Entries below feed `notes/divergence-ledger.md`.

### 7.1 — Drop `vpx_bit_depth_t` parameter from quant API

**Where:** `rvp9_core::quant::common::vp9_dc_quant` /
`vp9_ac_quant` / `vp9_convert_qindex_to_q`. libvpx counterpart:
`vp9/common/vp9_quant_common.c:164-194`,
`vp9/encoder/vp9_ratectrl.c:159-172`.

**What libvpx does:** runtime switch on `bit_depth` (`VPX_BITS_8` /
`VPX_BITS_10` / `VPX_BITS_12`) selects between `dc_qlookup`,
`dc_qlookup_10`, `dc_qlookup_12` (resp. AC).

**What rvp9 does:** the function signature has no `bit_depth`
parameter; the 8-bit path is unconditional.

**Why:** D-01 RESOLVED — 8-bit only in v1. The HBD tables and switch
exist only to support Profile 2/3 which are out of scope. Keeping the
runtime branch would be dead code with non-trivial measurement cost in
the qindex search loops (`vp9_compute_qdelta_by_rate` is called per
frame; the inner `vp9_convert_qindex_to_q` is on the hot path).

**Byte-parity impact:** none for our Profile 0 scope.

**Compression-efficiency impact:** none within Profile 0.

**Future plan:** Stage 6+ re-introduces the parameter as a generic
or const-generic on `Pixel`. Existing call sites that pass no bit
depth will need to pass `BitDepth::Eight`. Decision: D-01.

### 7.2 — `QUANTS` / `Dequants` are split structs, not merged

**Where:** `rvp9_core::quant::tables::Quants` +
`rvp9_core::quant::tables::Dequants`. libvpx counterpart: `QUANTS`
struct in `vp9_quantize.h:21-38` (only `y/uv_quant*` and `y/uv_zbin`
and `y/uv_round*`) plus `y_dequant` / `uv_dequant` floating on
`VP9_COMP` directly (`vp9_encoder.h:732-733`).

**What libvpx does:** the 8 quant sub-tables live in `cpi->quants`;
the 2 dequant sub-tables live on `cpi` directly as siblings.

**What rvp9 does:** bundle the 2 dequant sub-tables into a sibling
struct `Dequants`. Both are allocated together (`Box<(Quants,
Dequants)>` in the encoder) and refilled in the same `init_quantizer`
call.

**Why:** the libvpx split is historical (Quants was added later to
align some int16 arrays; dequants pre-existed on VP9_COMP). Both are
table-of-tables of the same shape and lifecycle. Bundling improves
cache locality for `init_plane_quantizers` and gives `init_quantizer`
a single output parameter.

**Byte-parity impact:** none (memory layout is internal).

**Compression-efficiency impact:** none.

**Future plan:** none. Decision: this LLD.

### 7.3 — `vp9_init_plane_quantizers` returns lifetime-bound borrowed view, not raw pointers on a macroblock_plane

**Where:** `rvp9_core::quant::tables::init_plane_quantizers`.
libvpx counterpart: `vp9/encoder/vp9_quantize.c:246-285` writes raw
pointers into `x->plane[i]`.

**What libvpx does:** mutates `MACROBLOCK.plane[0..3].{quant,
quant_fp, round_fp, quant_shift, zbin, round}` and `MACROBLOCKD.plane.dequant`
in place, storing raw pointers into the `cpi->quants` tables. These
pointers are then read by SIMD kernels via `mb_plane->quant` etc.

**What rvp9 does:** returns a `MacroblockPlaneQuant<'q>` per plane
(see §3.9) where `'q` is the lifetime of the `Quants` / `Dequants`
references. The encoder stores these by value in its `MacroblockState`.
The SIMD kernels still read raw pointers; we convert at the FFI
boundary via `(view.quant.as_ptr(), ...)`.

**Why:** Rust's borrow checker forbids the storage-of-raw-pointer
pattern across function boundaries without `unsafe`; the borrowed view
encodes the lifetime in the type system and lets the safe-Rust caller
keep the pointers without surfacing `unsafe`. The cost is one extra
struct field (the `'q` borrow), zero runtime overhead.

**Byte-parity impact:** none.

**Compression-efficiency impact:** none.

**Future plan:** if FFI overhead is a real problem (it should not be,
because the pointer conversion is a single ASM instruction), consider
inlining the kernel call wrapper. Decision: this LLD.

### 7.4 — `vp9_compute_qdelta_by_rate` parameters explicit (no `RATE_CONTROL` god-struct)

**Where:** `rvp9_core::quant::common::vp9_compute_qdelta_by_rate`.
libvpx counterpart: `vp9/encoder/vp9_ratectrl.c:2562-2584` takes
`const RATE_CONTROL *rc` and reads `rc->best_quality`,
`rc->worst_quality`.

**What libvpx does:** the function reaches into the `RATE_CONTROL`
struct for the qindex bounds.

**What rvp9 does:** the function takes `best_quality: i32`,
`worst_quality: i32` as explicit parameters; no reference to
`RateControl` type at all. Same for `vp9_compute_qdelta`.

**Why:** the function is a pure inverse-search; passing the bounds
explicitly removes a transitive dependency on the encoder rate-control
type and makes the function callable from anywhere (including unit
tests with synthetic bounds). The cost is two extra arguments at every
call site (already only ~10 call sites total across the codebase).

**Byte-parity impact:** none.

**Compression-efficiency impact:** none.

**Future plan:** none. Decision: L-26.

### 7.5 — `vp9_quantizer_to_qindex` / `vp9_qindex_to_quantizer` only on the encoder API

**Where:** functions live in `rvp9_core::quant::common` but are only
re-exported from `rvp9_encoder` (not from `rvp9_decoder`). libvpx
counterpart: `vp9/encoder/vp9_quantize.c:315-326`.

**What libvpx does:** the functions are declared on the encoder side
only.

**What rvp9 does:** functions live in core to keep the quant family
colocated, but are only re-exported from the encoder crate. The
decoder never refers to user-facing 0-63 quantizers; it deals only in
qindex (0-255).

**Why:** colocation aids maintenance; re-export gating preserves API
hygiene.

**Byte-parity impact:** none.

**Compression-efficiency impact:** none.

**Future plan:** none. Decision: this LLD.

### 7.6 — L-25 trap is documented here but enforced in rate-control

**Where:** `rate_control.md` §X (TBD). Quant LLD declares the
constants `RCF_MULT` and `FRAME_SCALE_FACTOR` and notes the trap.

**What libvpx does:** `get_rate_correction_factor`
(`vp9_ratectrl.c:689-710`) multiplies by `rcf_mult[frame_size_selector]`
on read; `set_rate_correction_factor` (`vp9_ratectrl.c:712-735`)
divides by the same on write. The stored value is therefore
*resolution-independent*; on a resize event the multiplier changes
and the same stored value yields a different output rcf.

**What rvp9 does:** mirror the scale-on-read / unscale-on-write
exactly. The trap: a single forgotten scale or unscale leaks a 2×
factor permanently into `rate_correction_factors[level]`.

**Why:** mirror libvpx behaviour on resolution change exactly. The
property test (§9.5) asserts that `set_rate_correction_factor(x);
get_rate_correction_factor() == x` for all (x, frame_size_selector)
pairs.

**Byte-parity impact:** none if implemented correctly. Critical
divergence if implemented wrong.

**Compression-efficiency impact:** none if correct; up to several
BD-rate percent if wrong (the rcf walks freely outside its intended
range).

**Future plan:** none. Decision: L-25.

---

## 8. Open questions

None are blocking. The recommendations below are notes for Stage 3-4a
implementers; they are local optimisations or FFI confirmations, not
architectural questions.

- **`init_quantizer` fast-path** — libvpx rebuilds all 256 × 8 entries
  every time `delta_q_uv != 0` is set, even if values are unchanged.
  Rec: ship the full rebuild in v1; memoize in Stage 4a if profiling
  flags it.
- **`quant_thred` storage** — keep inside `MacroblockPlaneQuant`,
  recompute per call (2 multiplies). Precomputing into `Quants`
  costs 8 KiB for no measurable benefit.
- **EOB type** — `u16`, matching libvpx's `uint16_t *eob_ptr`. Value
  range [0, 1024]; fits.
- **`MacroblockState.q_index` ownership** — encoder LLDs own the
  storage; this LLD returns the i32 from `init_plane_quantizers` and
  the caller writes it into its own state.
- **`wrapping_*` discipline in `vp9_rc_bits_per_mb`** — `enumerator
  * q` at KEY's max q is 2.7M × 457 ≈ 1.23 × 10^9, close to `i32::MAX`.
  Use `wrapping_mul` + debug `assert!` per HLD §6.2. Mirror libvpx
  exactly if wrap fires.
- **`init_quantizer`'s `qzbin_factor * quant`** — max 100 × 1828 =
  183K, fits comfortably in `i32`. Use plain `*` per HLD §6.2
  table-size carve-out.

---

## 9. Test strategy

### 9.1 Verbatim-table verification

For each of `DC_QLOOKUP`, `AC_QLOOKUP`, `QUANTIZER_TO_QINDEX`:

```rust
#[test]
fn dc_qlookup_matches_libvpx_verbatim() {
    // 256 hex-coded values matching vp9_quant_common.c:15-35,
    // pasted in a fixture file. Bytes are not strictly necessary
    // (the values are textual), but the fixture is line-for-line.
    assert_eq!(&DC_QLOOKUP[..], &EXPECTED_DC_QLOOKUP[..]);
}
```

These tests catch transcription errors at compile time + run time.

### 9.2 `vp9_compute_qdelta_by_rate` — L-26 exhaustive

```rust
#[test]
fn compute_qdelta_by_rate_exhaustive() {
    // Cover every (qindex, frame_type, rate_target_ratio) cell from
    // Stage 1 §3.3 callers. Anchor outputs against libvpx C reference
    // captured into a fixture file by running libvpx with the same
    // inputs.
    for qindex in 0..256 {
        for &frame_type in &[FrameType::KeyFrame, FrameType::InterFrame] {
            for &ratio in &[0.5, 0.75, 1.0, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0] {
                let got = vp9_compute_qdelta_by_rate(
                    /* best */ 0, /* worst */ 255,
                    frame_type, qindex, ratio,
                );
                let expected = libvpx_oracle_compute_qdelta_by_rate(...);
                assert_eq!(got, expected, "qi={qindex} ft={frame_type:?} r={ratio}");
            }
        }
    }
}
```

The oracle is generated once by compiling a small C harness against
the `third_party/libvpx` sources; the output is committed under
`tests/fixtures/quant/compute_qdelta_by_rate.bin`.

### 9.3 `vp9_compute_qdelta` — exhaustive symmetric

Same shape as 9.2, parameterised on `(qstart, qtarget)` pairs from a
representative grid (each f64 derived from a qindex via
`vp9_convert_qindex_to_q`).

### 9.4 `vp9_get_qindex` — segment override

Property test: output equals the libvpx formula in §5.5. Edge cases:

- Delta mode, `base + delta < 0` → clamped to 0.
- Delta mode, `base + delta > MAXQ` → clamped to 255.
- Abs mode ignores `base_qindex`.
- Segment 0 with `SEG_LVL_ALT_Q` disabled → returns `base_qindex`
  (the AQ-3 normal case).

### 9.5 `rate_correction_factors[]` resolution-change round-trip — L-25

Test lives in `rate_control.md` but flagged from here: for every
`(stored_rcf, frame_size_selector)` pair, the sequence "write rcf ×,
switch selector, read, switch back, read" must return × both times.
No leak across selector flips.

### 9.6 SIMD differential (delegated to HLD-simd / LLD-simd-dispatch)

Quant LLD contributes: scalar reference impls in
`rvp9_core::simd::scalar::quant::{quantize_fp, quantize_fp_32x32,
quantize_b, quantize_b_32x32}`; Gaussian-distributed `i16` input
generator + edge cases from §6; regenerable libvpx fixture
(harness-compiled scalar kernels dumping qcoeff/dqcoeff/eob).

### 9.7 Round-trip through `init_quantizer` → `quantize_fp` → dequant

For a random coefficient array and a qindex sweep over `{0, 32, 64,
96, 128, 160, 192, 224, 248}`, encode + dequantise through rvp9 and
libvpx in parallel. Outputs must match bit-exactly. This is the
single highest-signal integration test for the quant subsystem.

### 9.8 `init_quantizer` byte-exact comparison

Run `init_quantizer` on rvp9 and `vp9_init_quantizer` on libvpx for
the same `(deltas, sharpness)`. All 8 sub-tables × 256 × 8 entries
must be byte-identical. Catches bugs in `invert_quant` or any
per-q factor.

### 9.9 Conformance-suite delegation

Stage 3 PLAN exit: 100% pass on official VP9 conformance vectors.
Every transform block decode runs `vp9_dc_quant` / `vp9_ac_quant` /
`vp9_get_qindex`, so conformance is itself a strong test — but a
downstream one, not owned here.

### 9.10 Negative test — HBD inputs rejected

Encoder constructor and decoder header-parse both refuse `profile != 0`.
Test that the quant module is never reached on HBD inputs, defending
the 8-bit-only assumption.

---

**Summary:** Key §7 entries for the divergence ledger: 7.1 (drop
`vpx_bit_depth_t`), 7.2 (`Quants` / `Dequants` split), 7.3
(`init_plane_quantizers` lifetime-bound view), 7.4
(`vp9_compute_qdelta_by_rate` explicit-parameter calling convention),
7.5 (`vp9_quantizer_to_qindex` re-export hygiene), 7.6 (L-25
cross-link). §8 blocking questions: none.
