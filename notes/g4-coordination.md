# G4.x SIMD rollout — coordination note

**Status:** G4.1 (idct8x8 AVX2 pilot) landed on `waves-c-hardening`.
G4.2 / G4.3 / G4.4 depth-scoped rollout (2026-05-21) extended the
dispatch shape with one slot per remaining decoder kernel family;
five of the six HLD §G4 families now have a `SimdDispatch` slot
(entropy decoder excluded by design — `BoolReader` is sequential).
Three of those slots carry a real AVX2 implementation
(`idct8x8_add`, `idct4x4_add`, `dc_predictor_4x4`); the other three
(`lpf_vertical_4`, `plane_dequant_pair`, `extend_frame_borders`)
are scalar-only and document their deferral rationale in their
`SimdDispatch` field doc.

Subsequent G4.x SIMD work and the encoder-side forward-transform
SIMD work MUST follow the per-kernel field-naming convention and
safe-wrapper pattern established here.

## G4.2 / G4.3 / G4.4 depth-scoped rollout (2026-05-21)

Per-family deliverables — see `crates/rvp9-core/src/dsp/dispatch.rs`
docstring on `SimdDispatch` for the canonical slot table.

| Family            | Slot                   | AVX2 impl | Decoder wired | Rationale for non-AVX2 deferral |
|-------------------|------------------------|-----------|---------------|---------------------------------|
| inverse 4×4       | `idct4x4_add`          | yes       | yes (`reconstruct.rs` + `reconstruct_inter.rs`) | n/a |
| intra-pred DC 4×4 | `dc_predictor_4x4`     | yes       | no (slot only — `intra_predict` doesn't yet thread `&SimdDispatch`; follow-up to extend the dispatch arg through `tile.rs:1122`) | n/a |
| loopfilter v4     | `lpf_vertical_4`       | no — scalar slot | no | libvpx SSE2 LF gathers 8 transverse columns via load-and-transpose; non-trivial port deferred |
| quant dequant     | `plane_dequant_pair`   | no — scalar slot | no | architectural: dequant is fused into the bool-coder per-token loop in `decode_coefs` (see `entropy/detokenize.rs:415`). No batchable kernel exists; family slot reserves naming for a future entropy-decoder rewrite |
| frame-buffers     | `extend_frame_borders` | no — scalar slot | no | inner work is `slice::fill` + `slice::copy_within`; platform `memset`/`memmove` already vectorise. Manual SIMD here wouldn't move the needle |

## SimdDispatch field naming convention

Per-kernel slots on `crates/rvp9-core/src/dsp/dispatch.rs::SimdDispatch`
use the schema:

```
{kernel}_{variant}
```

- `kernel` = the libvpx kernel family name (e.g. `idct8x8`, `idct16x16`,
  `intra_predict_v`, `lpf_horizontal_16`, `quantize_b`).
- `variant` = the operation suffix (e.g. `add` for the dst-add IT kernels;
  empty for in-place / non-add kernels).

Examples:

| G4 stage           | Slot name              | Scalar reference                                      |
|--------------------|------------------------|-------------------------------------------------------|
| G4.1               | `idct8x8_add`          | `rvp9_core::transforms::inverse::idct8x8_64_add`      |
| G4.2 (2026-05-21)  | `idct4x4_add`          | `rvp9_core::transforms::inverse::idct4x4_16_add`      |
| G4.3a (2026-05-21) | `dc_predictor_4x4`     | `rvp9_core::intra_pred::dc_predictor` (via scalar adapter in `dispatch.rs`) |
| G4.3b (2026-05-21) | `lpf_vertical_4`       | `rvp9_core::loopfilter::lpf_vertical_4` (scalar slot only) |
| G4.4a (2026-05-21) | `plane_dequant_pair`   | `rvp9_core::quant::plane_dequant_pair` (scalar slot only) |
| G4.4b (2026-05-21) | `extend_frame_borders` | `rvp9_core::frame::extend_frame_borders` (scalar slot only) |
| (future)           | `idct16x16_add`        | `rvp9_core::transforms::inverse::idct16x16_256_add`   |
| (future)           | `idct32x32_add`        | `rvp9_core::transforms::inverse::idct32x32_1024_add`  |
| (future)           | `iht4x4_add`           | `rvp9_core::transforms::inverse::iht4x4_16_add`       |
| (future)           | `iht8x8_add`           | `rvp9_core::transforms::inverse::iht8x8_64_add`       |
| (future)           | `iht16x16_add`         | `rvp9_core::transforms::inverse::iht16x16_256_add`    |

The encoder-side forward DCT rollout (Stage 4b / 5a) MUST mirror this
schema for its forward-transform slots:

| Encoder stage | Slot name      | Scalar reference                              |
|---------------|----------------|-----------------------------------------------|
| (forward 4)   | `fdct4x4`      | `rvp9_core::transforms::forward::fdct4x4`     |
| (forward 8)   | `fdct8x8`      | `rvp9_core::transforms::forward::fdct8x8`     |
| (forward 16)  | `fdct16x16`    | `rvp9_core::transforms::forward::fdct16x16`   |
| (forward 32)  | `fdct32x32`    | `rvp9_core::transforms::forward::fdct32x32`   |
| (hybrid 4)    | `fht4x4`       | `rvp9_core::transforms::forward::fht4x4`      |
| (hybrid 8)    | `fht8x8`       | `rvp9_core::transforms::forward::fht8x8`      |
| (hybrid 16)   | `fht16x16`     | `rvp9_core::transforms::forward::fht16x16`    |

(Pick whichever scalar reference matches the dispatch signature; add
`_add` only when the kernel writes through a dst+residual add pattern.)

## Safe-wrapper convention

Per HLD V4 §G4 RS3-M1: `#[target_feature(enable = "<isa>")]` requires
`unsafe fn` on stable Rust. A safe `fn` pointer cannot be coerced from
`unsafe fn`. Every per-arch kernel file therefore exposes BOTH:

1. `unsafe fn <kernel>_<isa>_inner(...)` — the `#[target_feature]`
   inner with the actual intrinsic calls. **Marked `unsafe fn`.**
2. `fn <kernel>_<isa>(...)` — the safe wrapper, one `unsafe { ... }`
   call to the inner. **SAFETY comment names the OnceLock + caps
   invariant** (per `crate::dsp::dispatch` module doc).

`SimdDispatch::build` installs the safe wrapper into the slot, not the
inner. The G11 `clippy::undocumented_unsafe_blocks = deny` lint enforces
the SAFETY comment.

See `crates/rvp9-core/src/dsp/x86_64/idct8x8_avx2.rs` for the
reference implementation.

## Dispatch threading through the decoder

Per HLD V4 DA3-M6: production code holds `Arc<SimdDispatch>`, NOT
`&'static SimdDispatch`. The Decoder constructs one Arc at
`Decoder::with_threads` (via `rvp9_core::dsp::init()`) and threads
`&SimdDispatch` borrows down through:

- `decode_tile_intra` / `decode_tile_inter`
- `decode_partition` / `decode_inter_partition`
- `decode_block` / `decode_inter_block`
- `reconstruct_intra_planes`
- `reconstruct_intra_block` / `reconstruct_inter_block`
- `inverse_transform_add` / `inverse_transform_add_dct`

The per-TX dispatch site calls `dispatch.idct8x8_add()(coefs, dest, stride)`
— no per-call `OnceLock::get`. G5 (tile threading) inherits this
pattern; `WorkerPool::new(Arc<SimdDispatch>)` will take an Arc clone
and pass it through the per-tile scope.

## Source-side aligned-load discipline (HLD V6 §G4 M6-6)

Every `unsafe fn` in `crates/rvp9-core/src/dsp/x86_64/*.rs` MUST carry
a grep-able sentinel comment of the form:

```
// G4: loads from `dst` use _mm256_loadu_si256 (unaligned)
```

(or `_mm_loadu_si128` / `_mm_loadl_epi64` for the SSE2 / SSSE3 tiers).
The `scripts/check-simd-sentinels.{sh,ps1}` scripts enforce this for
the G4.1 pilot. The earlier CI workflow follow-up is superseded for release
qualification: [PLAN.md Stage 5](../PLAN.md) requires local-first gates and
prohibits hosted workflows. No workflow integration remains pending.

## Proptest oracle

Per-kernel scalar-vs-SIMD differential tests live in the same file as
the safe wrapper (or in a `#[cfg(test)] mod tests` co-located there).
The five canonical input strategies are:

1. `uniform_strategy` — random i32 coefficients in `[-32768, 32767]`.
2. `sparse_strategy` — 1..=4 non-zero coefficients at random positions.
3. `eob_bucket_strategy` — K non-zero coefficients with K drawn from
   the per-TX-size dispatcher-seam set:
   - `Tx4x4`:    K ∈ {1, 2, 16}            (G4.2 idct4x4 uses this)
   - `Tx8x8`:    K ∈ {1, 2, 12, 13, 64}    (G4.1 idct8x8 uses this)
   - `Tx16x16`:  K ∈ {1, 2, 10, 11, 38, 256}
   - `Tx32x32`:  K ∈ {1, 2, 34, 135, 1024}
4. `high_magnitude_strategy` — coefficients near ±i16::MAX
   (saturation paths).
5. `pathological_strategy` — all-equal / all-min / all-max /
   alternating sign.

Plus directed fixtures:
- All-zero input.
- Single-coefficient impulse sweep across every position.
- Misaligned `dst` pointer at byte offsets `{0, 1, 3, 7, 15}` mod 32
  (exercises unaligned-load paths).

See `crates/rvp9-core/src/dsp/x86_64/idct8x8_avx2.rs` `mod tests` for
the reference implementation.

## Encoder coordination (G4.2 wiring scope)

The G4.1 pilot threads `&SimdDispatch` through the decoder reconstruct
path only. The encoder's RD inner loop in `mode_decision` still calls
the scalar idct families directly. Three call sites carry a
`TODO(G4.2)` and MUST be routed through `SimdDispatch::idct8x8_add()`
when G4.2 wires `&SimdDispatch` (or `Arc<SimdDispatch>`) through the
mode-decision call tree:

| File                                              | Line  | Caller                              |
|---------------------------------------------------|-------|-------------------------------------|
| `crates/rvp9-core/src/mode_decision/inter.rs`     | ~684  | inter RD recon, `TxSize::Tx8x8` arm |
| `crates/rvp9-core/src/mode_decision/intra.rs`     | ~428  | intra RD recon arm 1, `Tx8x8`       |
| `crates/rvp9-core/src/mode_decision/intra.rs`     | ~1067 | intra RD recon arm 2, `Tx8x8`       |

These sites bypass the dispatch deliberately for the G4.1 scope: the
"shape locked" claim in the G4.1 ledger entry covers the DECODER only.
Threading dispatch through `mode_decision` is G4.2 encoder coordination
work — out of scope for the pilot but in-scope for the next slot.

## Contract for G5 (worker threading)

Per HLD V4 §G4 DA3-M6, G5 tile workers will capture
`Arc<SimdDispatch>` (NOT `&'static SimdDispatch`). The G4.1 pilot
threads `&SimdDispatch` borrows; the decoder holds the `Arc` at the
top-level frame loop and passes borrows down. G5 worker closures
diverge: they must own an `Arc<SimdDispatch>` clone (one atomic per
spawn, NOT per per-block call).

**Pattern at the spawn boundary:**

```rust
// At the top of the per-frame decode, we have the frame-owned Arc.
let dispatch: Arc<SimdDispatch> = rvp9_core::dsp::init();

// Spawn N workers. Each gets ITS OWN clone — one atomic increment per
// worker, amortised across thousands of per-block dispatch calls.
for tile in tiles {
    let dispatch_worker = Arc::clone(&dispatch); // one atomic per worker
    pool.spawn(move || {
        // Inside the worker, the per-block hot loop borrows from the
        // Arc — NO Arc::clone in here.
        for block in tile.blocks() {
            // Call sites taking `&SimdDispatch` deref-borrow:
            decode_block(&*dispatch_worker, block, /* … */);
        }
    });
}
```

**Anti-pattern (do NOT do this):**

```rust
// WRONG — this would do one atomic increment per per-block call.
for block in tile.blocks() {
    let d = Arc::clone(&dispatch); // BAD: hot-loop atomic
    decode_block(&d, block, /* … */);
}
```

The borrow signature `&SimdDispatch` was a deliberate G4.1 choice
because it composes uniformly with both the `Arc::deref` pattern (G5
workers) and the borrow-from-frame-state pattern (single-threaded
decoder path today). G4.2+ call sites SHOULD keep `&SimdDispatch` in
their inner signatures; only the worker-spawn boundary owns the `Arc`.

## G4 rollout review (2026-05-21) — follow-ups

### Wave C follow-ups

- **G4.3.1 — `intra_predict` dispatch threading.** The G4.3a slot
  `dc_predictor_4x4` is shape-locked and bit-exact (proptest +
  directed fixtures pass) but the kernel is not yet executable in
  production — the `intra_predict` API in
  `crates/rvp9-core/src/intra_pred.rs` doesn't thread
  `&SimdDispatch`, and the decoder's only caller is at
  `crates/rvp9-decoder/src/tile.rs:1122`. The threading entry point
  needs to pass `dispatch: &SimdDispatch` through `intra_predict`,
  then dispatch on `(mode, side)` to either the SimdDispatch slot or
  the scalar fallback.

  This is the difference between "shape locked + bit-exact" (current
  state, HLD G4 exit literal reading) and "executable in production"
  (intent reading). Filed as **G4.3.1** for a follow-up wave.

### Per-slot caps tiering follow-ups

The current `Caps` struct exposes a single `avx2: bool` flag — there
is no separate model for SSE2, SSE4.1, SSSE3 etc. Two G4 AVX2 slots
land kernels that the SCALAR-vs-SIMD differential tests confirm are
bit-exact but use intrinsic sets narrower than AVX2:

| Slot                | Actual ISA tier used | Caps gate today | Action when finer caps land |
|---------------------|----------------------|-----------------|-----------------------------|
| `dc_predictor_4x4`  | SSE2 (`_mm_sad_epu8`, `_mm_cvtsi128_si32`) | `caps.avx2` | retag → `caps.sse2` (carries TODO at `dc_pred_4x4_avx2.rs:64-67`) |

When the `Caps` struct grows finer-grained tier flags (G4-tiering
sub-thread), each slot's `SimdDispatch::build` selector should be
revisited to widen host coverage. No spec-level work required — the
kernels are already bit-exact and the dispatch shape is per-slot.

### Documented coverage gaps

- **idct4x4 ADST hybrids bypass SIMD.** The `Tx4x4` dispatch site in
  `crates/rvp9-decoder/src/reconstruct.rs` is gated on
  `tx_type == TxType::DctDct`; ADST hybrid types (`DCT_ADST`,
  `ADST_DCT`, `ADST_ADST`) still call scalar
  `iht4x4_16_add`. Adding SIMD coverage for the hybrid arms requires
  separate dispatch slots (one each for the three hybrid combinations,
  or one shared "iht4x4_add" slot that dispatches internally on
  `tx_type`) — **filed as G4.3a follow-up**. The inter path
  (`reconstruct_inter.rs`) is unaffected: inter is DCT-DCT-only by
  spec, so it has no hybrid arm to cover.

### Deferral rationale references

The `plane_dequant_pair` slot ships scalar-only because dequant is
fused into the bool-coder per-token loop in
`crates/rvp9-core/src/entropy/detokenize.rs:415` — there is no
batchable kernel a SIMD dispatch could target. Cross-check: libvpx
ships SIMD dequant only in the encoder (`vpx_quantize_b`), not the
decoder — confirmed via libvpx review during G4.4a slot landing.
