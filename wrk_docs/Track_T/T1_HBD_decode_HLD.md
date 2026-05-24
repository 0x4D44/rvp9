# T-1 — VP9 Profile 2 (HBD 10-bit 4:2:0) decode: HLD

**Status:** SCOPE / HLD only — no implementation. Authored Track T (2026-05-24)
as the T-1 deliverable. Sibling of G7.1 (Profile 1 chroma).

**One-line:** Add a parallel `u16` pixel pipeline so the decoder can produce
bit-exact reconstructions for VP9 **Profile 2** (10-bit 4:2:0), then prepare
a follow-up for 12-bit and Profile 3.

---

## 1. Background & scope

VP9 profile matrix (recap from G7.1 §1):

| Profile | Bit depth | Chroma | Status today (rvp9) |
|---------|-----------|--------|---------------------|
| 0       | 8         | 4:2:0  | 201/201 + closed gate (G7.1 closed) |
| 1       | 8         | 4:2:2 / 4:4:0 / 4:4:4 | 2 vectors passing; full G7.1c done |
| **2**   | **10 / 12** | **4:2:0** | rejected (`UnsupportedProfile`) — **this track** |
| 3       | 10 / 12   | 4:2:2 / 4:4:0 / 4:4:4 | rejected; out of scope |

**In scope (T-1):** Profile 2 **10-bit only**, 4:2:0 chroma, scalar-first, bit-exact
against `vpxdec` for the six 10-bit conformance vectors. Lossless (WHT) HBD path
included only because the lossless flag is bitstream-orthogonal to bit depth — if
the gate already accepts a lossless HBD vector we must not silently mis-decode.

**Out of scope (T-1):**
- 12-bit (Profile 2 still — phase 2 of this track).
- Profile 3 (HBD + non-4:2:0 chroma — phase 3 of this track, gated on T-1 +
  G7.1's chroma infrastructure both being live).
- SIMD HBD kernels — all HBD work lands as scalar reference; SIMD is a
  separate optimisation track.
- Encoder HBD (the encoder hardcodes Profile 0 — `crates/rvp9-encoder/src/lib.rs:526`
  and the per-`BitDepth::Eight` sites in `frame_loop.rs:1544,1794`,
  `lib.rs:683,873,4711`).

**Why this is multi-week, not one PR:** every pixel-touching kernel needs an HBD
sibling. The 8-bit dispatch table (`crates/rvp9-core/src/dsp/dispatch.rs:1980-2487`)
has ~30 slots all typed `fn(..., &mut [u8], ...)`. Doubling that for `&mut [u16]` is
mechanical for some kernels (intra DC/V/H, idct adds) but load-bearing for others
(loopfilter, convolve8). Honest scope: T-1 is **2-3 weeks** with overnight passes;
the first slice this HLD specifies is **one overnight pass** that produces
something differential-testable.

---

## 2. Current state

### 2.1 The gate
`crates/rvp9-decoder/src/lib.rs:1180-1185` no longer rejects Profile 1 (G7.1
closed it), but Profile 2/3 (HBD) still surface as `DecodeError::UnsupportedProfile`
through the decode path; the residual policy lives in:
- `crates/rvp9-decoder/src/lib.rs:1177-1179` (comment: "HBD profiles still will
  reject when they're wired") — no explicit `match` arm there; today the rejection
  fires at the first kernel call that assumes 8-bit data.
- We will need to **add** an explicit gate at `decode_frame` analogous to the
  pre-G7.1 chroma gate, and lift it phase-by-phase.

### 2.2 What already works (parses but doesn't decode)
- **Bit-depth parse:** `crates/rvp9-core/src/bitstream/header.rs:486-502`
  (`read_bitdepth_colorspace_sampling`) already reads
  `bit_depth_one_more_than_8` and populates `ColorConfig::bit_depth ∈
  {Eight, Ten, Twelve}`. **No bitstream-parser change is required** to thread
  bit-depth through Profile 2.
- **Profile decoding:** `read_profile_raw` at
  `crates/rvp9-core/src/bitstream/header.rs:449-460` already accepts profile
  2 (no change). `Profile::from_u8` at
  `crates/rvp9-core/src/bitstream/frame.rs:53-61` already maps 2 →
  `Profile2`.
- **BitDepth enum:** `crates/rvp9-core/src/bitstream/frame.rs:89-98` —
  `Eight / Ten / Twelve` already in place with `#[repr(u8)]` numeric values
  matching libvpx `vpx_bit_depth_t`.
- **Header threading to decoder:** `UncompressedHeader::bit_depth` field
  (`crates/rvp9-core/src/bitstream/header.rs:75`) is already populated.
- **Quant LUTs (8-bit) exist** at `crates/rvp9-core/src/quant/tables.rs`
  (`DC_QLOOKUP`, `AC_QLOOKUP`). **HBD LUTs (`dc_qlookup_10`, `ac_qlookup_10`,
  `dc_qlookup_12`, `ac_qlookup_12`) are missing** — see libvpx
  `third_party/libvpx/vp9/common/vp9_quant_common.c:38-160` for the verbatim
  768 i16 values to transcribe.

### 2.3 Where 8-bit is hardcoded (the work)

| # | Site | File:line | Nature |
|---|------|-----------|--------|
| A | `pub type Pixel = u8` | `crates/rvp9-core/src/lib.rs:49` | The keystone alias; doc says "8-bit only in v1 (D-01)" |
| B | Frame buffer `Box<[u8]>` | `crates/rvp9-core/src/frame.rs:154-187` | Allocates 1 byte per sample; HBD needs `Box<[u16]>` or a typed view over `Box<[u8]>` |
| C | `PlaneRef { data: &[u8] }` / `PlaneMut` / `BorderedPlaneRef` | `crates/rvp9-core/src/frame.rs:1076-1145` | Plane accessors return `&[u8]` |
| D | Inverse-transform `_add` kernels | `crates/rvp9-core/src/transforms/inverse.rs:85-…` (29 lossless/iwht hits + every `*_add` site, ~30 fns) | All take `dest: &mut [u8]` and call `clip_pixel_add(u8, i64) -> u8` (`constants.rs:194`) |
| E | `clip_pixel_add` | `crates/rvp9-core/src/transforms/constants.rs:187-200` | 8-bit saturation `[0, 255]` literal |
| F | All 10 intra predictors | `crates/rvp9-core/src/intra_pred.rs` (~46 `u8` sites; 2442 LOC) | `above: &[u8]`, `left: &[u8]`, `dst: &mut [u8]` |
| G | All 12 loop-filter kernels | `crates/rvp9-core/src/loopfilter/apply.rs` + per-kernel files | `&mut [u8]` strides + 8-bit-implicit `mblim / lim / hev_thr` thresholds |
| H | 8-tap convolve / convolve_avg / copy | `crates/rvp9-core/src/inter/filter.rs:32-…` + `dsp::SimdDispatch::convolve8*` (`dispatch.rs:2390-2487`) | Same: `&[u8]` / `&mut [u8]` |
| I | `Yv12Buffer::extend_borders` / `extend_plane` | `crates/rvp9-core/src/frame.rs:1011-1060` | Operates on `&mut [u8]`; byte-fill semantics |
| J | Decoder reconstruct drivers | `crates/rvp9-decoder/src/reconstruct.rs:46-49`, `reconstruct_inter.rs` | Imports `idct*_add` / `iwht*_add` directly; calls dispatch slots typed `&mut [u8]` |
| K | `SimdDispatch` slot signatures | `crates/rvp9-core/src/dsp/dispatch.rs:1980-2487` | ~30 kernel slots all `fn(..., &mut [u8], ...)` |
| L | Dequant tables (lacking HBD) | `crates/rvp9-core/src/quant/tables.rs` | Only 8-bit `DC_QLOOKUP` / `AC_QLOOKUP`; no `_10` / `_12` siblings |
| M | `dc_quant` / `ac_quant` signature | `crates/rvp9-core/src/quant/mod.rs:49-62` | No `bit_depth` parameter (libvpx `vp9_dc_quant` takes `vpx_bit_depth_t`) |
| N | MD5 visible-plane oracle | `crates/rvp9-harness/src/...` (caller: `tests/conformance.rs:188`) | `md5_of_yv12_visible` Y-then-U-then-V; assumes byte planes — HBD vectors hash 2 bytes per sample (little-endian per libvpx convention) |

**LOC delta estimate (per module):**
- Module A/B/C (pixel-type / frame): ~250 LOC of new typed view code + ~80 LOC
  changes to `Yv12Buffer` (add `BitDepth` field, byte-multiplier on alloc).
- Module D (inv-txfm HBD kernels): **~1,200 LOC of new scalar HBD kernels**
  (~12 functions × ~100 LOC mirroring libvpx `inv_txfm.c:1454-2599`).
- Module E (clip_pixel_add / wraplow): ~30 LOC for `highbd_*` siblings.
- Module F (intra HBD): **~600 LOC** (all 10 modes × DC fallbacks; libvpx
  `vpx_dsp/intrapred.c` HBD section provides the literal port).
- Module G (loopfilter HBD): **~800 LOC** for all 12 edge kernels; libvpx
  `vpx_dsp/loopfilter.c` HBD path mirrors closely.
- Module H (convolve HBD): ~400 LOC.
- Module I (border-extend HBD): ~80 LOC (mostly stride-multiplier book-keeping).
- Module J (decoder dispatch wiring): ~150 LOC (per call site, route to
  8 vs 16 dispatch based on `BitDepth`).
- Module K (`SimdDispatch` HBD slot pairs): ~600 LOC of slot duplication.
- Module L (HBD quant tables): ~300 LOC (768 i16 literals + 4 LUT defs).
- Module M (`dc_quant` / `ac_quant` bit_depth param): ~30 LOC + ~12 call-site
  updates.
- Module N (harness HBD MD5): ~80 LOC of `md5_of_yv12_hbd_visible`.

**Total: ~4,500 LOC** of net code addition for 10-bit-only, before SIMD. 12-bit
adds ~50 LOC (extra LUT entries + an extra `BitDepth::Twelve` arm in clamps and
threshold tables).

---

## 3. Design decisions

### 3.1 Scope — 10-bit only first
**Recommendation:** Land **10-bit 4:2:0 only** in the first sub-track.
Add 12-bit immediately after (it's mostly a table swap + a clamp-bound
change), and treat Profile 3 (HBD + non-4:2:0) as a third sub-track gated on
both T-1 and G7.1 being live.

**Rationale:**
- 10-bit alone unlocks the six `vp92-2-20-10bit-yuv420.webm` family vectors
  and represents almost all real-world VP9 HBD content on Google Play /
  YouTube delivery. 12-bit is academic for VP9 in practice.
- 12-bit reuses **every kernel** added for 10-bit (the `bd` parameter is
  threaded through; only `highbd_clip_pixel_add` and `HIGHBD_WRAPLOW`'s
  clamp range differ). Adding 12-bit as a 1-day follow-up after 10-bit
  conforms keeps the first-pass surface small.
- Profile 3 is multiplicative — it needs the HBD pipeline **and** the
  non-4:2:0 chroma plumbing that G7.1 already shipped. Bundling them would
  expand the first-pass cone and slow bisection of HBD-only bugs.

### 3.2 Pixel-width abstraction — pick: **(c) parallel `u8` and `u16` code paths, dispatched at frame boundary**

Three options considered:

**(a) Generic `Pixel: PixelTrait` with monomorphization.** The "Rust-idiomatic"
choice: replace `Pixel = u8` with a `trait Pixel { type Wide; const BITS: u32; fn
clip(x: i32) -> Self; ... }`, monomorphize every kernel for `u8` and `u16`. Code
ends up clean and the compiler handles dispatch.

  Trade-off: every `pub fn` in `rvp9-core` becomes a `pub fn<P: Pixel>`,
  cascading generics through ~50 files. The intra predictors and idct kernels
  cross-call (`idct4` reads `dest`, calls `clip_pixel_add`) — every helper
  becomes generic. Compile time doubles (we monomorphize for both pixel widths
  in every TU). And the `SimdDispatch` table can't hold a generic kernel as a
  `fn(...)` pointer: you'd end up with two parallel `SimdDispatch<P>`
  structs, eating the win.

**(b) Runtime polymorphism via a single dispatch table with `*mut u8` + bit-depth.**
Keep `SimdDispatch` slots typed `fn(*mut u8, usize, /* bit_depth */ u8)` and let
the kernel branch internally. Mirrors libvpx's `int bd` parameter.

  Trade-off: unsafe at every site (we hand-cast `&mut [u16]` to `*mut u8`),
  loses Rust's slice-bounds checks. Every hot kernel pays a runtime branch.
  Libvpx tolerates this because C is unsafe by default; rvp9 has chosen
  typed views and we'd be regressing the safety model just to share signatures.

**(c) Parallel `u8` and `u16` code paths, dispatch at the frame boundary.**
Add an `HbdSimdDispatch` (or extend `SimdDispatch` with `u16`-typed slot
fields named `idct4x4_add_hbd`, `intra_v_pred_hbd`, etc.). The decoder
inspects `frame.header.bit_depth` once at the top of `decode_frame` and
selects which code path to drive through `tile.rs` for the whole frame.
`Yv12Buffer` carries a `bit_depth: BitDepth` field and exposes
`y_plane_u16() -> PlaneRef<u16>` accessors only when `bit_depth != Eight`
(panics otherwise).

  Trade-off: per-kernel implementation work is the largest of the three
  (every kernel has two source files). But: it's the **only** option that
  preserves rvp9's existing safety story, keeps the 8-bit fast path
  untouched (no perf regression for the 201 conformance vectors that ship
  today), and gives the cleanest differential-test surface (the 8-bit
  kernel is the oracle for the 16-bit kernel via downshift-and-compare).

  **Decision: (c).** Generics are tempting; the dispatch model and the
  load-bearing test discipline (8-bit kernels are the spec) push us to
  parallel code paths. Adopt the libvpx-mirroring naming convention
  (`*_hbd` suffix in rvp9 = `vpx_highbd_*` in libvpx) so cross-reference
  stays one-to-one.

### 3.3 Frame buffer layout
Add `bit_depth: BitDepth` to `Yv12Buffer` (`crates/rvp9-core/src/frame.rs:161`).
Keep `buffer_alloc: Box<[u8]>` but multiply allocation by `bytes_per_pixel = if
bit_depth == Eight { 1 } else { 2 }`. Stride accounting (`y_stride`,
`uv_stride`) stays in **bytes** (libvpx convention — `YV12_BUFFER_CONFIG::y_stride`
is bytes too). Add typed accessors:

```rust
pub fn y_plane_u16(&self) -> PlaneRef<'_, u16> { /* requires bit_depth != Eight */ }
```

Plane data in the `u16` views is little-endian (matches libvpx's
`uint16_t *dest` and Windows/x86 default; portable: explicitly use
`bytemuck::cast_slice` for the type pun).

### 3.4 Dispatch table extension
Extend `SimdDispatch` (`crates/rvp9-core/src/dsp/dispatch.rs:1980`) with a
**second** parallel kernel pointer per slot, suffixed `_hbd`. Examples:

```rust
pub(crate) idct4x4_add: fn(&[i32; 16], &mut [u8], usize),
pub(crate) idct4x4_add_hbd: fn(&[i32; 16], &mut [u16], usize, u32 /* bd */),
```

The decoder driver picks the suffix at frame boundary based on `bit_depth`. SIMD
slots default to the scalar reference for HBD (we do not port AVX2 HBD kernels
in T-1).

### 3.5 Lossless interaction
The `iwht*_add` family (`crates/rvp9-core/src/transforms/inverse.rs`, 29 hits)
also needs HBD siblings because `lossless == true` can coexist with
`bit_depth == Ten`. Libvpx ships `vpx_highbd_iwht4x4_*` (see
`third_party/libvpx/vpx_dsp/inv_txfm.c` for the C source). Treat as a small
appendix to module D's port list.

---

## 4. Bitstream parser changes

Almost none. The current `parse_uncompressed_header`
(`crates/rvp9-core/src/bitstream/header.rs`) already:
- accepts `Profile::Profile2` from `read_profile_raw` (header.rs:449-460),
- reads `bit_depth_one_more_than_8` for Profile 2/3 (header.rs:486-502),
- forces 4:2:0 chroma for Profile 2 (`read_bitdepth_colorspace_sampling`
  header.rs:512-522).

What *needs* to change:
1. **Writer-side `UnsupportedProfile` gate.** `write_uncompressed_header`
   (`header.rs:594-603`) currently rejects any non-Profile-0 header. The
   decoder doesn't go through this path (decode doesn't write), so it's
   strictly an encoder concern — leave it for the encoder track and don't
   touch in T-1.
2. **Plumb `BitDepth` into the decoder driver.** Today `UncompressedHeader::
   bit_depth` is populated but the decoder ignores it. Add a single read:
   ```
   match h.bit_depth {
       BitDepth::Eight  => self.decode_*_8bit(frame),
       BitDepth::Ten    => self.decode_*_hbd(frame, 10),
       BitDepth::Twelve => Err(DecodeError::UnsupportedProfile { profile: 2 }), // until phase 2
   }
   ```
   at `lib.rs:1180-1185`. Sized at ~10 LOC for the gate + dispatch shim.
3. **Quant init takes bit-depth.** Mirror libvpx `setup_segmentation_dequant`
   (`vp9_decodeframe.c:1427-1448` — calls `vp9_dc_quant(qindex,
   y_dc_delta_q, cm->bit_depth)`). Today our quant init pipeline calls
   `dc_quant(qindex, delta)` without a bit-depth parameter (`quant/mod.rs:49`).
   Add a `bit_depth: BitDepth` parameter to both `dc_quant` and `ac_quant`,
   dispatch on it to pick `DC_QLOOKUP` / `DC_QLOOKUP_10` / `DC_QLOOKUP_12`.
   ~20 LOC + ~12 call-site updates.
4. **Header → buffer plumbing.** When the decoder allocates a fresh
   `Yv12Buffer` (today's call sites pass `chroma`/`border`/`align`), pass
   `bit_depth` as well so the buffer doubles its allocation for HBD.

---

## 5. IDCT / iADST design

### 5.1 Which kernels need HBD variants
Every `_add` kernel in `crates/rvp9-core/src/transforms/inverse.rs`:

- `idct4x4_16_add`, `idct4x4_1_add`
- `idct8x8_64_add`, `idct8x8_1_add`, `idct8x8_12_add`
- `idct16x16_256_add`, `idct16x16_1_add`, `idct16x16_10_add`, `idct16x16_38_add`
- `idct32x32_1024_add`, `idct32x32_1_add`, `idct32x32_34_add`, `idct32x32_135_add`
- `iht4x4_16_add`, `iht8x8_64_add`, `iht16x16_256_add`
- `iwht4x4_16_add`, `iwht4x4_1_add` (lossless)

Plus the 1D primitives `idct4 / idct8 / idct16 / idct32 / iadst4 / iadst8 /
iadst16` need HBD siblings (libvpx: `vpx_highbd_idct4_c` etc. at
`third_party/libvpx/vpx_dsp/inv_txfm.c:1351-…`).

**Total: ~12 add-kernels + 7 1D-primitives = 19 new functions.**

### 5.2 Scalar first, SIMD never (in T-1)
All HBD kernels land as **scalar** ports of libvpx's C reference. No SIMD HBD
in this track. The 8-bit AVX2/SSE kernels remain the production fast path for
Profile 0/1; HBD users (rare) accept the scalar perf hit.

### 5.3 What the scalar HBD kernel looks like
Libvpx `vpx_highbd_idct4x4_16_add_c` (`vpx_dsp/inv_txfm.c:1454-1477`) is the
shape we mirror. Key delta vs the 8-bit kernel:
1. `dest: *uint16_t` → Rust `&mut [u16]`.
2. `vpx_highbd_idct4_c(input, outptr, bd)` instead of `idct4_c` — the 1D core
   takes a `bd` parameter that drives the modular-truncation clamp
   (`HIGHBD_WRAPLOW(x, bd) = ((x << (24 - bd)) >> (24 - bd))` —
   `vpx_dsp/inv_txfm.h:81`) and the `highbd_clip_pixel_add` saturation bound
   (`(1 << bd) - 1` — `vpx_dsp/inv_txfm.h:110-117`).
3. The `_1_add` (DC-only) fast paths additionally pass `bd` through to
   `highbd_clip_pixel_add` for the broadcast loop.

The differential test discipline: for `bit_depth = 10`, the HBD kernel's
output must equal the 8-bit kernel's output **left-shifted by 2** when fed an
input scaled by `<< 2` and a `dest` zeroed (modulo the modular-truncation
clamp, which only fires on out-of-range inputs that conformance streams
never produce). This gives a per-kernel sanity check that's faster than
running a full frame through the harness.

### 5.4 `wraplow` / `clip_pixel_add` HBD siblings
Add to `crates/rvp9-core/src/transforms/constants.rs`:
- `pub fn highbd_wraplow(x: i32, bd: u32) -> i32` mirroring
  `HIGHBD_WRAPLOW` macro.
- `pub fn highbd_clip_pixel_add(dest: u16, trans: i64, bd: u32) -> u16`
  mirroring `highbd_clip_pixel_add` (clamps to `(1 << bd) - 1`).

These are ~10 LOC each and don't interact with the existing `wraplow` /
`clip_pixel_add` — keep the 8-bit fast path 100% unchanged.

---

## 6. Test corpus

### 6.1 Official VP9 HBD vectors
libvpx ships **six 10-bit and six 12-bit** Profile 2/3 vectors in
`third_party/libvpx/test/test-data.sha1:686-701`. The Profile 2 (10-bit
4:2:0) ones we want for T-1:

```
vp92-2-20-10bit-yuv420.webm                f77673b566f686853adefe0c578ad251b7241281
vp92-2-20-10bit-yuv420.webm.md5            abdedfaddacbbe1a15ac7a54e86360f03629fb7a
```

For phase 2 (12-bit Profile 2):

```
vp92-2-20-12bit-yuv420.webm                0c2c355a1b17b28537c5a3b19997c8783b69f1af
vp92-2-20-12bit-yuv420.webm.md5            afb2c2798703e039189b0a15c8ac5685aa51d33f
```

For Profile 3 (out of scope until phase 3): `vp93-2-20-{10,12}bit-yuv{422,440,444}.webm`
(8 files, all present in upstream).

**Hosting:** same WebM bucket the existing vectors come from
(`https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx`).
Extend `scripts/fetch-vectors.sh` (the `VECTORS=(...)` array — currently 15
entries seeded; Stage 3 plan was already to grow it).

### 6.2 Harness extension
`tests/conformance/` already has the directory structure
(`vectors/` 28 entries — 14 vectors × `.webm` + `.md5`). Adding HBD vectors
is a one-line addition to `fetch-vectors.sh` plus a tweak to
`crates/rvp9-harness/src/...` (the visible-MD5 oracle):

`md5_of_yv12_visible` (caller: `tests/conformance.rs:188`) currently hashes the
Y, U, V visible plane bytes in order. The HBD `.md5` files in upstream were
generated against **little-endian uint16_t** plane bytes (libvpx convention
— see `vpx_dsp_rtcd_defs.pl` HBD section and `vpxdec`'s YUV writer).
Implementation: add `md5_of_yv12_hbd_visible(buf)` that pads each 10-bit
sample into the low 10 bits of a 16-bit word (high 6 bits zero), then hashes
the byte representation of those `u16`s little-endian.

Conformance counter expansion (`crates/rvp9-harness/tests/conformance.rs`):
- Add a third gating bucket: `total_hbd_gated` for the transitional phase
  where HBD vectors are in the corpus but T-1 phases A-B haven't finished.
  Mirror the `total_gated` plumbing already in place for Profile 1 (G7.0
  history at `conformance.rs:101-117`).
- Once T-1 phase D lands, the gate flips off and the counter goes to zero;
  the conformance floor rises from 203 (today: 201 + 2 Profile-1) to
  **209** (+ 6 Profile-2-10bit vectors).

### 6.3 Self-encoded vectors
`vpxenc.exe --profile=2 --bit-depth=10 --i420` produces self-encoded HBD;
extend `scripts/encode-corpus.ps1` only if the official six vectors don't
cover a needed code path (resize / multi-tile / show_existing_frame in HBD).
Lesson from G7.1: trust the official corpus first; self-encoded vectors
become useful only when corner coverage gaps appear.

### 6.4 Bit-exact gate
For each vector, compare rvp9's decoded `u16` planes byte-for-byte against
`vpxdec --noblit -o decoded.yuv` output (already the methodology for
Profile 0). The `.md5` files in upstream were produced exactly this way, so
a passing MD5 implies bit-exact pixels. No new tooling needed beyond the
HBD MD5 oracle in §6.2.

---

## 7. Phasing

### 7.1 First-slice deliverable (one overnight pass; ~6-8 hours)

**T-1 phase 0: HBD scaffolding + first scalar kernel.** Land the *minimum*
that makes a single 10-bit IDCT block bit-exact-verifiable in isolation,
plus the dispatch plumbing to wire it later:

1. **HBD quant tables** (`crates/rvp9-core/src/quant/tables.rs`): transcribe
   `dc_qlookup_10`, `ac_qlookup_10` from libvpx
   `vp9/common/vp9_quant_common.c:38-160` (verbatim 512 i16 values). Add
   `_12` siblings while we're touching the file — they're cheap and avoid a
   second visit. **~200 LOC.**
2. **Quant API takes bit-depth** (`crates/rvp9-core/src/quant/mod.rs:49-62`):
   add `bit_depth: BitDepth` to `dc_quant` / `ac_quant`. Default-fill all
   ~12 existing call sites with `BitDepth::Eight`. **~30 LOC + call-site
   sweep.**
3. **HBD constants** (`crates/rvp9-core/src/transforms/constants.rs`): add
   `highbd_wraplow`, `highbd_clip_pixel_add`. **~30 LOC.**
4. **One HBD IDCT kernel.** Port `vpx_highbd_idct4x4_16_add_c` to
   `crates/rvp9-core/src/transforms/inverse.rs` as
   `highbd_idct4x4_16_add(input: &[i32; 16], dest: &mut [u16], stride:
   usize, bd: u32)`. Mirror libvpx `vpx_dsp/inv_txfm.c:1454-1477` line by
   line. **~80 LOC, including the `highbd_idct4` 1D helper.**
5. **Differential unit test.** In the same file as the kernel: build a
   random `[i32; 16]` input + zero `[u16; 64]` dest, run the new HBD kernel
   at `bd = 10`, run the 8-bit `idct4x4_16_add` with the input downshifted
   by 2 and a zero `[u8; 64]` dest, assert the 8-bit output upshifted by 2
   equals the HBD output (modulo the < 1-LSB rounding tolerance — exact
   equality for clean inputs, tolerance documented in the test). Plus one
   golden vector with hand-computed expected values from a reference run.
   **~60 LOC.**
6. **Dispatch slot stub.** Add `pub(crate) highbd_idct4x4_add: fn(&[i32;
   16], &mut [u16], usize, u32)` to `SimdDispatch`, with a `pub fn
   highbd_idct4x4_add(&self) -> ...` accessor. Populate with the scalar
   ref. **No call sites yet** — they wire in phase 1. **~30 LOC.**
7. **Bit-depth gate** (`crates/rvp9-decoder/src/lib.rs:1180-1186`): add an
   explicit match on `frame.header.bit_depth`. Today `BitDepth::Eight`
   falls through; `Ten`/`Twelve` return `DecodeError::UnsupportedProfile {
   profile: frame.header.profile as u8 }`. This **doesn't change conformance
   numbers** (already failing) but makes the rejection deliberate and
   future phases bisectable. **~10 LOC.**

After the overnight pass: 201 conformance still passes, one new unit test
+ one differential test prove the HBD scalar machinery works in isolation,
all phase-1 dependencies (quant API, dispatch slot, gate) are in place.
**Total LOC added: ~440.** This is the only first-slice goal.

### 7.2 Subsequent slices (each its own overnight pass)

- **T-1 phase 1: HBD intra path end-to-end (scalar).** Add 10 HBD intra
  kernels + `Yv12Buffer` bit-depth field + `u16` plane accessors + decoder
  intra reconstruction shim. Differential-test against `vpxdec` on a
  single-keyframe HBD test stream. Cut the gate down to "only the
  inter/loopfilter paths reject for HBD".
- **T-1 phase 2: HBD inverse-transform full set.** Port the remaining 11
  HBD `_add` kernels + 6 HBD 1D primitives. Intra keyframes now decode
  fully bit-exact.
- **T-1 phase 3: HBD loop filter (long pole).** Port 12 HBD edge kernels.
  See G7.1 §3.4 for analogous "long pole" pattern.
- **T-1 phase 4: HBD inter (convolve8 family + scale).** Port 11
  `convolve8*` HBD variants + `extend_borders` HBD. Inter frames now decode.
- **T-1 phase 5: Lift HBD gate + add corpus.** Add the six 10-bit vectors,
  extend MD5 oracle, lift the `BitDepth::Ten` arm of the gate. **Floor
  rises from 201 → 207.**
- **T-1 phase 6: 12-bit (Profile 2 12-bit).** Adds `BitDepth::Twelve` to
  every threading site; floor rises to **209**.
- **T-1 phase 7: Profile 3 (HBD + non-4:2:0).** Strictly gated on G7.1's
  full Profile-1 chroma plumbing being live (which it is) + T-1 6-bit done.

Each phase is independently committable; Profile 0/1 stays at 203/203
throughout.

---

## 8. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|-----------|
| R1 | **Modular-truncation behavior diverges from libvpx.** Libvpx `HIGHBD_WRAPLOW` is a sign-preserving left-shift / arithmetic-right-shift dance (`vpx_dsp/inv_txfm.h:81`). Rust signed shifts are well-defined here, but the `(24 - bd)` calculation can shift by 12 / 14 — must keep as `i32`, not `u32`. | Medium | High (silent corruption) | Transcribe macro to a single Rust `const fn` with unit tests at the `bd ∈ {8, 10, 12}` × representative input matrix. Compare against `vpxdec --noblit` byte-for-byte on the first overnight pass. |
| R2 | **MD5 oracle endianness wrong.** HBD `.md5` files in upstream were generated against little-endian `uint16_t` planes. On big-endian CI hosts (none today, but cross-build matters) the byte order flips and every vector mismatches. | Low | Medium | Hash the explicitly little-endian byte representation (`u16::to_le_bytes`) rather than `bytemuck::cast_slice` (which is host-endian). Add a comment in `md5_of_yv12_hbd_visible` citing this rationale. |
| R3 | **Loop-filter HBD threshold scaling.** Libvpx scales `mblim` / `lim` / `hev_thr` by `<< (bd - 8)` inside the HBD kernel (`vpx_dsp/loopfilter.c` HBD path). Skipping this means filters fire on the wrong edges → near-miss bit-exact failures. | High | Medium | Make HBD threshold-scaling explicit in the kernel signature (pass `bd`, scale at the kernel entry). Code-review checkpoint: every HBD lpf kernel must read `bd` and scale; add a dispatch-table sanity test that an HBD slot rejects `bd == 8`. |
| R4 | **Lossless × HBD interaction.** A lossless stream at `bit_depth == 10` is bitstream-legal but contrived. If we forget to add HBD iwht kernels, a single such vector would hard-crash with the slot signature mismatch. | Low | Medium | Include `highbd_iwht*_add` in T-1 phase 2 (the inverse-transform full set). Document in `reconstruct.rs:31-33` that the lossless-HBD path requires phase-2 kernels. Until phase 2, the bit-depth gate at the top of `decode_frame` rejects all HBD so the issue cannot fire. |
| R5 | **Self-encoded oracle isn't trustworthy in HBD.** G7.1 §6 noted that self-encoded ledgers became trustworthy only after the M1 frame-parallel adaptation fix. HBD self-encode uses different quant tables — there's no a-priori reason to trust freshly-encoded HBD vectors as oracles. | Medium | Low (testing-velocity) | Use the **official six 10-bit vectors as the only oracle** for phases 1-5. Treat self-encoded HBD as a corner-coverage tool only after phase 5 conforms. |

---

## 9. Validation strategy

### 9.1 Bit-exact gate
Per HBD vector: compare rvp9's reconstructed `u16` visible planes (Y → U → V,
little-endian byte representation) MD5 against the upstream `.md5` file. The
upstream `.md5` was produced by `vpxdec --noblit` on the same vector
(`third_party/libvpx/test/test_vectors.cc` cites the methodology). Match =
bit-exact.

### 9.2 Per-kernel differential
For each HBD `_add` kernel as it lands:
1. Random `[i32; N]` input (uniform in `[-2^14, 2^14]` for `bd = 10`).
2. Run the 8-bit kernel with input `>> 2` and a `[u8; ...]` dest.
3. Run the HBD kernel at `bd = 10` with input and a `[u16; ...]` dest.
4. Assert `(dest_8bit as u16) << 2 == dest_hbd` for every pixel (clean
   inputs satisfy this exactly; document tolerance for clamp-triggering
   inputs).

This catches modular-truncation bugs (R1) and saturation-bound bugs early,
before they leak into a full-frame MD5.

### 9.3 Conformance corpus expansion
Phased increments, all in `crates/rvp9-harness/tests/conformance.rs`:
- **Phase 0-4:** corpus unchanged at 14 vectors × 1 file each + 2 Profile-1
  vectors. Floor stays at 203. HBD changes don't appear in conformance.
- **Phase 5:** add `vp92-2-20-10bit-yuv420.webm` family (1 vector + 1
  variation per fetch, ~6 vectors). Floor rises to **209**.
- **Phase 6:** add `vp92-2-20-12bit-yuv420.webm` family (~6 vectors).
  Floor rises to **215**.
- **Phase 7:** add `vp93-2-20-*bit-yuv{422,440,444}.webm` (12 vectors).
  Floor rises to **227**.

### 9.4 Regression gates
Every phase must keep Profile 0 (201) and Profile 1 (2) green. Mechanism is
already in place: the bit-depth gate at the top of `decode_frame` falls
through to the 8-bit path for `BitDepth::Eight`, so the 8-bit dispatch slots
are physically unable to be called with `&mut [u16]`. This is the structural
reason for picking option (c) in §3.2.

---

## 10. Recommendation

Schedule **T-1 phase 0 (the overnight pass in §7.1)** first to validate the
dispatch model and the modular-truncation port. If phase 0's differential
test fails for `bd = 10`, the whole HBD design needs revisiting (R1 fired).
If it passes, the remaining 6-7 phases are mechanical kernel ports following
the same template.

Do **not** bundle T-1 with G7.1 follow-up encoder work or with SIMD HBD
ports. Both expand the cone; both can wait for scalar HBD to be conforming
first.
