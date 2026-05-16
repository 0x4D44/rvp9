# HLD-SIMD — Runtime SIMD dispatch (RTCD equivalent)

**Author:** HLD-SIMD writer (Stage 2).
**Status:** Draft. Resolves PLAN.md Q2 (decisions-needed.md D-02).
**Date:** 2026-05-16.
**Inputs read:**
- `PLAN.md` (D5 — `core::arch` intrinsics + scalar fallback; D-01 RESOLVED — 8-bit only).
- `notes/vp9-analysis/00-overview.md` §1, §3, §4, §7.
- Stage 1 specialists §6 sections in `01-architecture.md`, `02-encoder-core.md`,
  `03-motion-estimation.md`, `04-bitstream-entropy.md`, `05-rate-control.md`,
  `06-threading-svc.md`.
- `notes/vp9-analysis/decisions-needed.md` (D-02, R-09 RTCD-init-before-worker-spawn).
- Direct lookup of `third_party/libvpx/vpx_dsp/vpx_dsp_rtcd_defs.pl`,
  `vp9/common/vp9_rtcd_defs.pl`, and `vpx_scale/vpx_scale_rtcd.pl` for kernel completeness.
- Peer HLDs: `notes/hld/00-architecture.md` and `notes/hld/01-threading.md`
  not present at time of writing — assumptions noted in §8 Open questions.

**Scope of this document:**

1. Resolve Q2 (hand-written tables vs build.rs generator).
2. Catalogue every SIMD kernel libvpx ships at 8-bit (Profile 0).
3. Specify the runtime feature detection mechanism, the one-time-init invariant,
   the dispatch-table layout, the scalar-fallback policy, and the per-kernel
   testing strategy.
4. Define the performance regression CI gate.
5. State the 8-bit-only simplification explicitly so v2 HBD work has a clear
   expansion vector.

**Out of scope:**

- HBD (10/12-bit) kernel doubling — deferred to Stage 6+ per D-01.
- The cross-tile threading model — `notes/hld/01-threading.md` owns
  Send/Sync and worker spawn lifecycle. We only state the invariant
  "RTCD-init-before-worker-spawn" here and require threading-HLD to enforce it.
- Per-LLD implementation details — each kernel's actual SIMD body
  belongs to `notes/lld/transforms.md`, `notes/lld/intra-pred.md`,
  `notes/lld/loopfilter.md`, `notes/lld/motion-estimation.md`,
  `notes/lld/quant.md`, `notes/lld/denoiser.md`. This doc only specifies
  *signature*, *ISA targets*, and *test contract*.

---

## 0. TL;DR

- **Q2 resolution:** hand-written Rust dispatch tables. Reason: kernel count
  is bounded and stable (≈ 240 distinct entries at 8-bit), `vpx_dsp_rtcd_defs.pl`
  syntax is non-trivial to parse from a build.rs (it embeds Perl conditionals
  and feature gates), and a hand-written table reads as a single grep-able
  source of truth.
- **Detection:** `is_x86_feature_detected!` for x86_64 (sse2 / ssse3 / sse4_1 /
  avx2), `is_aarch64_feature_detected!` for aarch64 (neon — but mandatory on
  every aarch64 target we ship, so the `is_…` check is a guard against
  unsupported targets and a documentation hook rather than a real branch).
- **Init invariant (R-09):** `rvp9_init_rtcd()` runs exactly once at
  `Encoder::new` / `Decoder::new` *before* worker spawn; workers always read
  through a fully-populated `&'static SimdDispatch`. Enforced via `OnceLock`.
- **Tables:** one `SimdDispatch` struct holding `Option<fn(...)>` pointers
  per kernel. The fallback (`None`) reads the scalar implementation from the
  same module, so a missing SIMD pointer is impossible to observe.
- **Test contract:** every kernel has (a) a scalar reference always present
  even on SIMD-enabled builds, (b) a per-ISA proptest that drives random
  inputs through both impls and asserts byte-exact equality, (c) a fixed
  edge-case suite (all-zero, all-max, alternating, stride mismatch, odd
  width, boundary alignment).
- **CI gates:** SIMD-vs-scalar differential tests on every PR. Bench gate
  per ISA per kernel — flag if median time deviates more than 5% from the
  previous commit on the same hardware class.

**Kernel count summary (8-bit only):**

| Family                       | Distinct kernels |
|------------------------------|------------------|
| SAD / SAD-skip / SAD-avg / SADx4D / SAD-skip-x4D | 65 |
| Variance / get-NxN-var / MSE / get4x4sse_cs / get_mb_ss / vpx_sse | 22 |
| Sub-pixel variance / sub-pixel avg variance      | 26 |
| Compound average prediction (`vpx_comp_avg_pred`) | 1 |
| Subtract / sum-of-squares                        | 2  |
| Hadamard 8x8 / 16x16 / 32x32 + SATD             | 4  |
| Quant (`vp9_quantize_fp` + `_32x32`, `vpx_quantize_b` + `_32x32`) | 4 |
| Block-error (`vp9_block_error_fp`) + (`vp9_block_error`) | 2 |
| Forward DCT (4×4, 8×8, 16×16, 32×32, 32×32_rd) + DC variants | 10 |
| Forward ADST/Hybrid (`vp9_fht4x4`, `_8x8`, `_16x16`, `vp9_fwht4x4`) | 4 |
| Inverse DCT (4×4, 8×8 [64,12,1], 16×16 [256,38,10,1], 32×32 [1024,135,34,1], iwht4x4 [16,1]) | 14 |
| Inverse ADST/Hybrid (`vp9_iht4x4_16_add`, `_8x8_64`, `_16x16_256`) | 3 |
| Intra predictors (10 modes × 4 sizes = 40 entries; 4 mode-corner variants `*e`) | 44 |
| Convolve 8-tap (8 entries) + scaled (6 entries) + copy + avg | 16 |
| Loop filter (vert16 / vert16_dual / vert8 / vert8_dual / vert4 / vert4_dual + 6 horiz) | 12 |
| ME-specific (`vpx_int_pro_row`, `_col`, `vpx_vector_var`, `vp9_diamond_search_sad`) | 4 |
| `vpx_avg_8x8`, `vpx_avg_4x4`, `vpx_minmax_8x8`                          | 3 |
| Denoiser (`vp9_denoiser_filter`)                  | 1 |
| Temporal filter (`vp9_apply_temporal_filter`)     | 0 (cpu-used 6-8 unreachable; gate to scalar — `notes/lld/lookahead.md`) |
| Scale-and-extend frame (`vp9_scale_and_extend_frame`) — SVC + dynamic resize | 1 |
| `vp9_filter_by_weight8x8` / `16x16` (post-proc, optional)               | 2 |
| `vpx_plane_add_noise` (optional post-proc; not v1 scope)                | 0 |
| Frame-border extend / copy_y                      | 3 |
| **Total in-scope for v1**                         | **≈ 243** |

For comparison: libvpx's complete RTCD surface at 8-bit + HBD is roughly
470 kernels (the v2 expansion vector — §9).

---

## 1. Q2 — hand-written tables vs build.rs generator

### 1.1 The two options

**Option A — hand-written Rust dispatch tables.** One Rust module per kernel
family (`rvp9-core/src/dsp/variance.rs`, `rvp9-core/src/dsp/sad.rs`, etc.)
declares the scalar `fn`, declares each ISA-specific `unsafe fn`, and
populates a `SimdDispatch` struct field-by-field at boot.

**Option B — build.rs generator parsing `vpx_dsp_rtcd_defs.pl`.** A build
script reads the libvpx Perl files at compile time, emits Rust modules
declaring the function signatures and stub bodies, and the human implements
the per-ISA bodies in matching files under `rvp9-core/src/dsp/<isa>/`.

### 1.2 Costs and benefits

| Dimension                   | Option A (hand-written)            | Option B (build.rs generator)    |
|-----------------------------|------------------------------------|----------------------------------|
| Initial setup               | ~1 day to scaffold all families    | ~3-5 days to parse Perl + emit Rust |
| Adding a new kernel         | Two-line addition to one `.rs`     | Edit Perl file or maintain parallel input |
| Catching libvpx upstream drift | Manual sync (requires diff Perl)  | Automatic — regen on every build  |
| Build-time complexity       | Zero (pure cargo)                  | build.rs runs Perl-equivalent parser + emits ~2000 lines |
| Compile-time error quality  | Native rustc spans on the source file the human edits | Errors point at generated code; require source-map back to Perl |
| IDE support (rust-analyzer) | Native — signatures visible in source | Stub Rust visible; generated dispatch invisible until first build |
| Debuggability               | Step into the dispatch directly    | Step through generated `out_dir` files |
| Cross-platform portability  | Pure Rust                          | Requires Perl-like parser (or our own minimal parser) |
| Risk of subtle drift        | We are the single source of truth — drift requires conscious edit | Easy to forget to upstream-sync; risk of silent change |
| Maintenance burden in v1    | Low (240 entries, stable)          | Medium — parser must keep up with Perl edge cases |
| Maintenance burden in v2    | Grows linearly with kernel count   | Sub-linear if generator handles edge cases — but only if those edge cases are stable |

### 1.3 Why `vpx_dsp_rtcd_defs.pl` is awkward to parse

The Perl file is not a declarative table. It contains:

- Plain `add_proto` / `specialize` pairs (the easy 80%).
- `if (vpx_config("CONFIG_VP9_TEMPORAL_DENOISING") eq "yes")` conditional
  blocks gating whole kernels (denoiser, `vp9_apply_temporal_filter`).
- `$vpx_highbd_fdct4x4_1_neon=vpx_fdct4x4_1_neon;` symbol-aliasing for
  highbd path that re-uses the 8-bit kernel.
- `"$ssse3_x86_64"` string-interpolated ISA tags (the SSSE3-only-on-x86_64
  bind for `vpx_hadamard_8x8`).
- Branching on `bitdepth` for the `tran_low_t` type — 8-bit uses `int16_t`,
  HBD uses `int32_t`. Two parallel `add_proto` blocks per kernel.
- A `forward_decls` Perl subroutine emitting struct forward-declarations
  consumed by the prototypes (`struct macroblock`, `struct
  search_site_config`).

A toy parser that handles 80% of the file is easy. A parser that handles
100% of it is roughly the same work as just transcribing the table by
hand once. And the human cost of being wrong on the 20% is high: every
mistaken alias produces a silent perf regression that may not surface
until benchmarking.

### 1.4 Decision

**Option A — hand-written dispatch tables.**

Rationale:

1. The kernel surface in v1 is bounded (≈ 243 entries) and stable. libvpx
   1.14.1 is the pinned upstream; we are not chasing a moving target.
2. The dispatch-table struct is itself the contract — the moment a
   kernel ships, its scalar implementation, its scalar test, and its
   `SimdDispatch` slot all land together. There is no "we got the slot
   but not the impl" failure mode.
3. Q2 in `decisions-needed.md` already recommends Option A (D-02 default).
4. We control the divergence ledger. Any libvpx-relative omission must be
   conscious; we do not want a build-script silently mirroring upstream
   when we have a documented reason to deviate (cf. denoiser SSE2-vs-C
   parity TODO at `vp9/encoder/vp9_denoiser.c:65-67`, A7 challenge #6).
5. If in Stage 5+ we find ourselves writing >20 SIMD kernels per month
   on a sustained basis, we revisit. The cost of switching from A to B is
   one parser-write away; the cost of switching B → A is throwing away
   the parser. A is the lower-regret choice.

**Divergence ledger entry:** L-RTCD-01 — "rvp9 uses hand-written Rust
dispatch tables in `rvp9-core/src/dsp/dispatch.rs` instead of a Perl-RTCD-
generator. Source of truth is the table itself; sync against
`vpx_dsp_rtcd_defs.pl@v1.14.1` is verified by the kernel-catalogue
checklist in `notes/hld/02-simd-dispatch.md` §3."

---

## 2. Runtime feature detection

### 2.1 The detection primitive

x86_64:

```rust
// stable Rust since 1.27.
#[cfg(target_arch = "x86_64")]
fn detect_x86() -> X86Caps {
    X86Caps {
        sse2:   is_x86_feature_detected!("sse2"),
        ssse3:  is_x86_feature_detected!("ssse3"),
        sse4_1: is_x86_feature_detected!("sse4.1"),
        avx2:   is_x86_feature_detected!("avx2"),
    }
}
```

Aarch64:

```rust
#[cfg(target_arch = "aarch64")]
fn detect_aarch64() -> AArch64Caps {
    AArch64Caps {
        neon: std::arch::is_aarch64_feature_detected!("neon"),  // typically always true
        dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),  // ARMv8.4+
    }
}
```

We do **not** ship NEON-dotprod kernels in v1 (the gain over plain NEON for
SAD-class kernels is real but only on the newest hardware; the engineering
cost of duplicating every SAD/variance/sub-variance for dotprod is high
relative to the win on our target deployment hardware). v2 expansion vector.

Other targets (wasm32, riscv64, ppc64le, etc.) fall back to scalar — see §5.

### 2.2 The init invariant (R-09)

> **Invariant R-09:** the SIMD dispatch table must be fully populated
> before any worker thread is spawned. No worker thread ever observes a
> partially-initialised table; the table is `&'static` and immutable after
> init.

libvpx achieves this implicitly via `vp9_initialize_enc()` calling
`vp9_rtcd()`, `vpx_dsp_rtcd()`, `vpx_scale_rtcd()` (architecture §6.1 /
`vp9_encoder.c:929-942`) before the first encode call. The encoder is
single-threaded between `vpx_codec_enc_init()` and `vpx_codec_encode()`.

In rvp9 we enforce R-09 with `OnceLock`:

```rust
// rvp9-core/src/dsp/dispatch.rs
use std::sync::OnceLock;

static DISPATCH: OnceLock<SimdDispatch> = OnceLock::new();

pub fn init() -> &'static SimdDispatch {
    DISPATCH.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        let caps = detect_x86();
        #[cfg(target_arch = "aarch64")]
        let caps = detect_aarch64();
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let caps = ();
        SimdDispatch::build(&caps)
    })
}

#[inline]
pub fn get() -> &'static SimdDispatch {
    // Fast path: get_or_init under contention is once-then-cheap.
    init()
}
```

Both `Encoder::new()` and `Decoder::new()` call `dispatch::init()` as
their first action. Worker spawning (HLD-threading §X) happens later in
the same constructor; by the time a worker thread runs, `DISPATCH` is
already populated.

**No `unsafe`** is required for the dispatch itself: `OnceLock<T>` is
`Sync` for `T: Send + Sync`, and `&'static T` is freely shareable. The
unsafe lives one level deeper, inside each per-ISA kernel body (which
must be marked `unsafe fn` because it requires the target feature).

### 2.3 What happens if a worker reads before init?

`OnceLock::get_or_init` returning `&'static T` is a memory barrier; once a
worker successfully reads any kernel pointer it has Acquire-ordered seen
every other field of `SimdDispatch`. There is no torn-read window.

If `init()` is called concurrently by two threads (e.g. two encoders
spinning up at the same moment), `OnceLock` linearises them: one wins,
populates the table, the loser gets the same `&'static SimdDispatch`. The
construction is idempotent (CPU caps don't change at runtime), so
double-construction is benign even if it could race — but `OnceLock`
forbids that race anyway.

### 2.4 Compile-time vs runtime detection

There are two orthogonal axes:

1. **Has the build enabled this ISA?** Controlled by `target-cpu` /
   `target-features` flags to rustc. If the user builds with
   `-C target-feature=-avx2`, the AVX2 intrinsics module may not even
   compile (`#[cfg(target_feature = "avx2")]`).
2. **Does the runtime CPU support this ISA?** Controlled by
   `is_x86_feature_detected!`. A binary built on a Haswell box may run on
   a Sandy Bridge box.

In practice we ship one binary that supports all four x86_64 ISA levels.
This means:

- All four kernel modules (`sse2_kernels.rs`, `ssse3_kernels.rs`,
  `sse4_1_kernels.rs`, `avx2_kernels.rs`) compile unconditionally because
  they use `#[target_feature(enable = "...")]` attributes on each `fn`
  rather than `#[cfg]` gates. The `unsafe` body is only allowed to run
  when the runtime check has confirmed the feature.
- The dispatch table picks the best of {scalar, sse2, ssse3, sse4_1, avx2}
  per kernel.

For embedded / size-sensitive builds the user can override with a Cargo
feature `rvp9-core/scalar-only`, which strips the SIMD modules and forces
every dispatch slot to the scalar fn. See §5.

---

## 3. Kernel catalogue

This is the master list. Every kernel libvpx ships at 8-bit that is reached
under our cpu-used 6-8 / REALTIME mode is enumerated here. Kernels that
libvpx ships at 8-bit but that are **unreachable** in our config (cpu-used
≤ 5, two-pass, altref) are flagged explicitly as **out of v1**.

For each kernel: the Rust function signature, the libvpx scalar reference
file:line, the ISAs we ship in v1, and the testing strategy. ISA codes:

- **scalar** — always present.
- **sse2** — x86_64 baseline (mandatory on every x86_64 since 2003).
- **ssse3** — x86_64, ~2007 hardware (Penryn).
- **sse4_1** — x86_64, ~2008 (Nehalem). Required only by `vpx_sse`,
  `vp9_apply_temporal_filter` (skipped in v1).
- **avx2** — x86_64, ~2013 (Haswell).
- **neon** — aarch64 (mandatory feature on every aarch64 target).

ISAs we **do not** ship in v1: ssse3-x86_64-only conditional (we'll ship
the sse2 form even where libvpx specialises for ssse3_x86_64; the perf
delta is negligible compared to the avx2 alternative), avx512 (only
`vpx_sad64x64x4d` specialises; the avx2 form is within 5% on Skylake-class
silicon and we save engineering cost), neon_dotprod, neon_i8mm, msa, mmi,
vsx, lsx, dspr2 (non-target architectures).

**Block sizes** in VP9 follow `BLOCK_4X4 … BLOCK_64X64` and are
enumerated in `vp9/common/vp9_enums.h:31-45`:

```
BLOCK_4X4   BLOCK_4X8   BLOCK_8X4
BLOCK_8X8   BLOCK_8X16  BLOCK_16X8
BLOCK_16X16 BLOCK_16X32 BLOCK_32X16
BLOCK_32X32 BLOCK_32X64 BLOCK_64X32
BLOCK_64X64
```

13 block sizes. Where a kernel family covers all 13, we list it as
"× 13 sizes" rather than enumerating each.

### 3.1 SAD family — `vpx_sad{w}x{h}` (motion estimation, partition copy, scene-detect)

**Citations:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:753-790`. Specialist A3 §6.1.

**Rust signature:**

```rust
pub type SadFn = unsafe extern "C" fn(
    src: *const u8, src_stride: i32,
    refp: *const u8, ref_stride: i32,
) -> u32;
```

For each of 13 block sizes (4x4, 4x8, 8x4, 8x8, 8x16, 16x8, 16x16, 16x32,
32x16, 32x32, 32x64, 64x32, 64x64), one fn per ISA:

| Block size | scalar | sse2 | avx2 | neon |
|-----------:|:------:|:----:|:----:|:----:|
| 4×4    | yes | yes | —   | yes |
| 4×8    | yes | yes | —   | yes |
| 8×4    | yes | yes | —   | yes |
| 8×8    | yes | yes | —   | yes |
| 8×16   | yes | yes | —   | yes |
| 16×8   | yes | yes | —   | yes |
| 16×16  | yes | yes | —   | yes |
| 16×32  | yes | yes | —   | yes |
| 32×16  | yes | yes | yes | yes |
| 32×32  | yes | yes | yes | yes |
| 32×64  | yes | yes | yes | yes |
| 64×32  | yes | yes | yes | yes |
| 64×64  | yes | yes | yes | yes |

**Scalar reference:** `vpx_dsp/sad.c` (`SADMXN` macro at line 60).

**Use sites:** `cpi->fn_ptr[bsize].sdf` consumed by `vp9_int_pro_motion_estimation`
(A3 §4.7), `vp9_diamond_search_sad_c` (A3 §4.6), `vp9_pick_intra_mode`'s
sub-pel pre-step, `nonrd_pick_inter_mode_sb` ref-pruning (`vp9_pickmode.c`),
`is_superblock_flat_static` (A5).

**Differential test plan (Tier-1):**

- Proptest: random `[u8; 64*64+padding]` source and reference buffers, random
  strides in `[width, 4*width]`, run scalar + every available ISA, assert all
  return values bit-equal (`u32 == u32`).
- Edge-case suite: all-zero, all-max (255), src == ref (zero SAD), 1-pel
  difference, alternating columns (high-freq spatial pattern),
  src_stride != ref_stride.
- 10 000 iterations per size per ISA per CI run.

### 3.2 SAD-skip — `vpx_sad_skip_{w}x{h}` (even-rows-only SAD)

**Citations:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:792-829`. Specialist A3 §6.2.

**Rust signature:** identical to `SadFn`.

Same 13 sizes. Reads only even rows of source and reference, returning a
SAD scaled to the full block (libvpx multiplies by 2 internally). ISAs:

| Block size | scalar | sse2 | avx2 | neon |
|-----------:|:------:|:----:|:----:|:----:|
| 4×4   | yes | —   | —   | yes |
| 4×8   | yes | yes | —   | yes |
| 8×4   | yes | —   | —   | yes |
| 8×8   | yes | yes | —   | yes |
| 8×16  | yes | yes | —   | yes |
| 16×8  | yes | yes | —   | yes |
| 16×16 | yes | yes | —   | yes |
| 16×32 | yes | yes | —   | yes |
| 32×16 | yes | yes | yes | yes |
| 32×32 | yes | yes | yes | yes |
| 32×64 | yes | yes | yes | yes |
| 64×32 | yes | yes | yes | yes |
| 64×64 | yes | yes | yes | yes |

**Scalar reference:** `vpx_dsp/sad.c:103-117` (`SAD_SKIP_MXN` macro).

**Use sites:** `cpi->fn_ptr[bsize].sdsf` consumed by
`full_pixel_diamond` when `use_downsampled_sad` is on (A3 §4.6,
`vp9_mcomp.c:2589-2598`).

**Differential test plan:** as §3.1.

### 3.3 SAD-avg — `vpx_sad{w}x{h}_avg` (compound-pred SAD)

**Citations:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:891-928`. Specialist A3 §6.3.

**Rust signature:**

```rust
pub type SadAvgFn = unsafe extern "C" fn(
    src: *const u8, src_stride: i32,
    refp: *const u8, ref_stride: i32,
    second_pred: *const u8,
) -> u32;
```

ISAs by size: same coverage as base SAD (sse2 on all, avx2 on 32×16 +).

**Use sites:** `cpi->fn_ptr[bsize].sdaf` — only `vp9_refining_search_8p_c`
during compound NEWMV. **Not reached** in non-RD path (A3 §6.3 confirms).
We must still ship the kernel because the BFP table is populated
uniformly; the worker just never calls `.sdaf` at cpu-used 6-8. Scalar
suffices in v1 for unreachable code; SIMD optional. **Decision:** ship
scalar only in v1 for this row; mark with divergence ledger entry
L-RTCD-02 ("compound-NEWMV scalar only, BD-rate unaffected because path
unreachable at cpu-used 6-8"). Verified by Stage 4a smoke run.

**Differential test plan:** scalar proptest only (no SIMD impl), assert
implementation invariants (sum of |src - (pred+ref+1)/2|).

### 3.4 SAD x4D — `vpx_sad{w}x{h}x4d` (4 references, single source)

**Citations:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:933-970`. Specialist A3 §6.4.
**The single most performance-critical SIMD entry point in ME.**

**Rust signature:**

```rust
pub type SadX4dFn = unsafe extern "C" fn(
    src: *const u8, src_stride: i32,
    ref_array: *const *const u8, ref_stride: i32,
    sad_array: *mut u32,
);
```

13 sizes. ISA coverage:

| Block size | scalar | sse2 | avx2 | neon |
|-----------:|:------:|:----:|:----:|:----:|
| 4×4   | yes | yes | —   | yes |
| 4×8   | yes | yes | —   | yes |
| 8×4   | yes | yes | —   | yes |
| 8×8   | yes | yes | —   | yes |
| 8×16  | yes | yes | —   | yes |
| 16×8  | yes | yes | —   | yes |
| 16×16 | yes | yes | —   | yes |
| 16×32 | yes | yes | —   | yes |
| 32×16 | yes | yes | —   | yes |
| 32×32 | yes | yes | yes | yes |
| 32×64 | yes | yes | —   | yes |
| 64×32 | yes | yes | —   | yes |
| 64×64 | yes | yes | yes | yes |

(libvpx also has an `avx512`-specialised `vpx_sad64x64x4d`; we ship avx2
only — the win is single-digit percent and avx2 is universally available.)

**Scalar reference:** `vpx_dsp/sad.c:178-194` (`SADMXNX4D` macro).

**Use sites:** `cpi->fn_ptr[bsize].sdx4df` consumed by
`vp9_diamond_search_sad_c` (A3 §4.6) — issues 4 SADs per inner-loop
iteration (8 search sites = 2 batches). Also used by
`vp9_full_pixel_search` (A3 §4.5).

**Differential test plan:** as §3.1 plus 4-reference contention check
(all 4 refs identical → all 4 sads identical), reference offset alignment
test (refs[0] aligned, refs[1] +1 misaligned, refs[2] +2 misaligned,
refs[3] +3 misaligned to stress unaligned load paths).

### 3.5 SAD-skip x4D — `vpx_sad_skip_{w}x{h}x4d`

**Citations:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:972-1009`. Specialist A3 §6.5.

**Rust signature:** same as `SadX4dFn`.

ISA coverage trimmed: scalar everywhere; sse2 + neon on most; avx2 only on
32x16/32x32/32x64/64x32/64x64.

**Use sites:** `cpi->fn_ptr[bsize].sdsx4df` — diamond search when
`use_downsampled_sad` and even-vs-odd-rows discrepancy ≤ 10%.

**Differential test plan:** as §3.4.

### 3.6 Variance — `vpx_variance{w}x{h}`

**Citations:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:1262-1299`. Specialist A2 §6.1,
A3 §6.6.

**Rust signature:**

```rust
pub type VarianceFn = unsafe extern "C" fn(
    src: *const u8, src_stride: i32,
    refp: *const u8, ref_stride: i32,
    sse_out: *mut u32,
) -> u32;  // returns variance; *sse_out gets the SSE
```

13 sizes. ISA coverage: sse2 + avx2 + neon on all sizes ≥ 8×4; avx2
absent on 4×8 / 4×4.

**Scalar reference:** `vpx_dsp/variance.c:160-178` (`VAR` macro).

**Use sites:** `cpi->fn_ptr[bsize].vf` — sub-pel center error
(`setup_center_error` in `vp9_mcomp.c`), `vp9_get_mvpred_var` final
re-evaluation, `compute_minmax` in partition decision (A2 §3.3),
AQ-variance (A5 — but mode 1/2 dropped per D-05, so AQ usage is via
cyclic refresh segment-level variance only), `vp9_int_pro_motion_estimation`
candidate scoring.

**The exact rounding semantics matter for byte parity.** Libvpx's macro:

```c
*sse_var = sum_squared;
return *sse_var - (uint32_t)(((int64_t)sum * sum) >> (n_log2));
```

where `n_log2 = log2(w*h)`. Reproduce exactly; do not "simplify."

**Differential test plan:**

- Proptest: random buffers, all 13 sizes, all ISAs; assert var equal and
  sse equal.
- Edge: src == ref → variance == 0, sse == 0.
- Edge: src all-zero, ref all-128 → variance == 0, sse == w*h*128*128.
- Edge: src all-0, ref alternating 0/255 → known variance value.
- Strides: src_stride == w (tight), src_stride == 2*w (loose), mixed.

### 3.7 Sub-pixel variance — `vpx_sub_pixel_variance{w}x{h}`

**Citations:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:1334-1371`. Specialist A2 §6.12,
A3 §6.7. **Per-iteration core of every sub-pel-tree variant.**

**Rust signature:**

```rust
pub type SubPixelVarianceFn = unsafe extern "C" fn(
    src: *const u8, src_stride: i32,
    x_offset: i32, y_offset: i32,
    refp: *const u8, ref_stride: i32,
    sse_out: *mut u32,
) -> u32;
```

`x_offset, y_offset ∈ [0..7]` are the eighth-pel phases. At our default
`use_accurate_subpel_search = USE_2_TAPS`, the implementation applies
**bilinear** interpolation; the fused kernel does interp + variance in one pass.

13 sizes. ISA coverage:

| Block size | scalar | sse2 | ssse3 | avx2 | neon |
|-----------:|:------:|:----:|:-----:|:----:|:----:|
| 4×4   | yes | yes | yes | —   | yes |
| 4×8   | yes | yes | yes | —   | yes |
| 8×4   | yes | yes | yes | —   | yes |
| 8×8   | yes | yes | yes | —   | yes |
| 8×16  | yes | yes | yes | —   | yes |
| 16×8  | yes | yes | yes | —   | yes |
| 16×16 | yes | yes | yes | —   | yes |
| 16×32 | yes | yes | yes | —   | yes |
| 32×16 | yes | yes | yes | —   | yes |
| 32×32 | yes | yes | yes | yes | yes |
| 32×64 | yes | yes | yes | —   | yes |
| 64×32 | yes | yes | yes | —   | yes |
| 64×64 | yes | yes | yes | yes | yes |

**Scalar reference:** `vpx_dsp/variance.c:240-282` (`SUBPIX_VAR` macro).

**Use sites:** `cpi->fn_ptr[bsize].svf` — every iteration of every sub-pel
refinement variant (A3 §5: `vp9_find_best_sub_pixel_tree`,
`_pruned`, `_pruned_more`, `_pruned_evenmore`).

**Differential test plan:**

- All 64 (x_offset, y_offset) combinations × 13 sizes × every ISA.
- Edge: x_offset == 0 && y_offset == 0 → reduces to integer variance
  (assert equality with `vpx_variance_…`).
- Property: variance is invariant under (x↔y, w↔h, src↔ref) swaps when
  inputs are symmetric.

### 3.8 Sub-pixel-avg variance — `vpx_sub_pixel_avg_variance{w}x{h}`

**Citations:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:1373-1410`. Specialist A3 §6.8.

**Rust signature:**

```rust
pub type SubPixelAvgVarianceFn = unsafe extern "C" fn(
    src: *const u8, src_stride: i32,
    x_offset: i32, y_offset: i32,
    refp: *const u8, ref_stride: i32,
    sse_out: *mut u32,
    second_pred: *const u8,
) -> u32;
```

13 sizes. ISA coverage same as sub-pixel variance with sse2 + ssse3
on all and avx2 only on 64×64/32×32.

**Use sites:** `cpi->fn_ptr[bsize].svaf` — **not reached** in non-RD path
beyond compound-ZEROMV (sub-pel refinement skipped for ZEROMV per A3 §3).
Ship scalar only in v1; document in divergence ledger L-RTCD-02 alongside
SAD-avg.

**Differential test plan:** scalar proptest only.

### 3.9 Variance helpers — `vpx_get8x8var`, `vpx_get16x16var`, `vpx_mse*`, `vpx_get4x4sse_cs`, `vpx_get_mb_ss`, `vpx_sse`

**Citations:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:1304-1325`, `vpx_sse` at line 747.

| Kernel | Signature | ISAs |
|---|---|---|
| `vpx_get8x8var` | `(src, ss, ref, rs, *sse, *sum)` | scalar, sse2, neon |
| `vpx_get16x16var` | same | scalar, sse2, avx2, neon |
| `vpx_mse16x16` | `(src, ss, ref, rs, *sse) -> u32` | scalar, sse2, avx2, neon |
| `vpx_mse16x8`, `vpx_mse8x16`, `vpx_mse8x8` | same | scalar, sse2, neon |
| `vpx_get_mb_ss` | `(*const i16) -> u32` (sums squares of 256 elements) | scalar, sse2 |
| `vpx_get4x4sse_cs` | `(src, ss, ref, rs) -> u32` (SSE of two 4x4 byte arrays) | scalar, neon |
| `vpx_sse` | `(src, ss, ref, rs, w, h) -> i64` | scalar, sse4_1, avx2, neon, neon_dotprod |

**Use sites:** `vpx_get8x8var` — `block_variance` (A2 §6.2,
`vp9_pickmode.c:465-470`); `vpx_get16x16var` — partition heuristics;
`vpx_mse*` — `vp9_pick_filter_level` (deblock strength search,
`vp9_picklpf.c`); `vpx_sse` — temporal filter post-MC error (cpu-used 6-8
unreachable; scalar suffices); `vpx_get4x4sse_cs` — chroma sampling step.

**Differential test plan:** as §3.6.

### 3.10 `vpx_comp_avg_pred` — compound average prediction

**Citation:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:1328-1329`.

```rust
pub type CompAvgPredFn = unsafe extern "C" fn(
    comp_pred: *mut u8, pred: *const u8,
    width: i32, height: i32,
    refp: *const u8, ref_stride: i32,
);
```

ISAs: scalar, sse2, avx2, neon.

**Use sites:** `setup_center_error` (A3 sub-pel) when second_pred != NULL;
`accurate_sub_pel_search` (off in v1). Required because `setup_center_error`
calls it unconditionally inside the compound branch even if the rest of
sub-pel is skipped.

**Differential test plan:** trivially the byte-wise average of two
sources; differential test over random width × height × stride.

### 3.11 `vpx_subtract_block` — residual computation

**Citation:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:744-745`. A2 §6.10.

```rust
pub type SubtractBlockFn = unsafe extern "C" fn(
    rows: i32, cols: i32,
    diff: *mut i16, diff_stride: isize,
    src: *const u8, src_stride: isize,
    pred: *const u8, pred_stride: isize,
);
```

ISAs: scalar, sse2, avx2, neon.

**Use sites:** `block_yrd` (A2 §4.6) — produces `p->src_diff` consumed by
Hadamard + quantise. Hot path; every inter mode candidate hits this once
per partition.

**Differential test plan:** all 13 block sizes (rows / cols from
`BLOCK_4X4` … `BLOCK_64X64`); diff_stride and src_stride independent.

### 3.12 `vpx_sum_squares_2d_i16` — SSE of a 2D int16 block

**Citation:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:1011-1012`.

```rust
pub type SumSquares2dI16Fn = unsafe extern "C" fn(
    src: *const i16, stride: i32, size: i32,
) -> u64;
```

ISAs: scalar, sse2, neon.

**Use sites:** noise estimation `vp9_update_noise_estimate` (A5).

**Differential test plan:** scalar reference + sse2/neon diff.

### 3.13 Hadamard transforms + SATD

**Citations:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:845-879`. A2 §6.6/6.7, A3 §6.11.

For 8-bit, `tran_low_t = int16_t`.

```rust
pub type HadamardFn = unsafe extern "C" fn(
    src_diff: *const i16, src_stride: isize,
    coeff: *mut i16,
);

pub type SatdFn = unsafe extern "C" fn(
    coeff: *const i16, length: i32,
) -> i32;
```

| Kernel | ISAs (8-bit) |
|---|---|
| `vpx_hadamard_8x8` | scalar, sse2, neon (and ssse3-only-on-x86_64 in libvpx; we ship sse2 + avx2 from 16×16) |
| `vpx_hadamard_16x16` | scalar, sse2, avx2, neon |
| `vpx_hadamard_32x32` | scalar, sse2, avx2, neon |
| `vpx_satd` | scalar, sse2, avx2, neon |

**Use sites:** `block_yrd` (A2 §4.6 / `vp9_pickmode.c`) — every inter mode
candidate.

**Differential test plan:** Hadamard is a unitary integer transform with
known coefficients; build a fixed test vector from libvpx test suite
(`test/hadamard_test.cc:vp9HadamardTest`). Random fuzzing too.

### 3.14 `vp9_block_error_fp` / `vp9_block_error`

**Citation:** `vp9/common/vp9_rtcd_defs.pl:129-132`. A2 §6.8.

```rust
pub type BlockErrorFpFn = unsafe extern "C" fn(
    coeff: *const i16, dqcoeff: *const i16, block_size: i32,
) -> i64;

pub type BlockErrorFn = unsafe extern "C" fn(
    coeff: *const i16, dqcoeff: *const i16,
    block_size: isize, ssz: *mut i64,
) -> i64;
```

`_fp` ISAs: scalar, sse2, avx2, neon.
non-fp ISAs: scalar (and `*const tran_low_t` in libvpx — but at 8-bit
that's int16_t). Used only in RD path (cpu-used ≤ 4); ship scalar only.

**Differential test plan:** scalar + SIMD diff over random int16 arrays;
edge case coeff == dqcoeff → returns 0.

### 3.15 Quantisation

**Citations:** `vp9/common/vp9_rtcd_defs.pl:134-138`,
`vpx_dsp/vpx_dsp_rtcd_defs.pl:725-729`. A2 §6.9.

```rust
pub type QuantizeFpFn = unsafe extern "C" fn(
    coeff: *const i16, n_coeffs: isize,
    mb_plane: *const MacroblockPlane,
    qcoeff: *mut i16, dqcoeff: *mut i16,
    dequant: *const i16, eob: *mut u16,
    scan_order: *const ScanOrder,
);

pub type QuantizeBFn = QuantizeFpFn;  // same shape, different semantics
```

| Kernel | ISAs (8-bit) |
|---|---|
| `vp9_quantize_fp` | scalar, sse2, ssse3, avx2, neon |
| `vp9_quantize_fp_32x32` | scalar, ssse3, avx2, neon |
| `vpx_quantize_b` | scalar, sse2, ssse3, avx, avx2, neon |
| `vpx_quantize_b_32x32` | scalar, ssse3, avx, avx2, neon |

For v1 we ship: scalar, sse2 (where libvpx has it; otherwise `ssse3` as
the x86_64 baseline — we promote ssse3 to a required ISA for these two
32x32 variants), avx2, neon. The `_b` variants are used by the
**decoder**; `_fp` by the **encoder**.

`MacroblockPlane` and `ScanOrder` are stable POD structs; mirror exactly.

**Differential test plan:**

- Random coefficient arrays from a Gaussian-weighted distribution
  (coefficients cluster near zero in real video).
- Edge: all coefficients == 0 → eob == 0, qcoeff all zero.
- Edge: DC-only coefficient → eob == 1.
- Specifically test **EOB tracking** — the SIMD impl uses horizontal
  reduction (`_mm_minpos_epu16`-style) and is the most error-prone part.

### 3.16 Forward transforms — DCT

**Citations:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:533-612` (the file has
parallel sections for `CONFIG_VP9_HIGHBITDEPTH = no` and `= yes`; we read
the 8-bit branch at line 587-612).

```rust
pub type FdctFn = unsafe extern "C" fn(
    input: *const i16, output: *mut i16, stride: i32,
);

pub type Fdct1Fn = FdctFn;  // DC-only, returns one coefficient
```

| Kernel | ISAs (8-bit) |
|---|---|
| `vpx_fdct4x4` | scalar, sse2, neon |
| `vpx_fdct4x4_1` | scalar, sse2, neon |
| `vpx_fdct8x8` | scalar, sse2, neon (and ssse3-only-on-x86_64) |
| `vpx_fdct8x8_1` | scalar, sse2, neon |
| `vpx_fdct16x16` | scalar, sse2, avx2, neon |
| `vpx_fdct16x16_1` | scalar, sse2, neon |
| `vpx_fdct32x32` | scalar, sse2, avx2, neon |
| `vpx_fdct32x32_rd` | scalar, sse2, avx2, neon |
| `vpx_fdct32x32_1` | scalar, sse2, neon |

For v1 we ship scalar + sse2 + (avx2 where listed) + neon.

**Use sites:** `block_yrd` for the non-RD path uses Hadamard (§3.13), not
DCT. The forward DCT is used by the **RD path** (cpu-used ≤ 4), which we
don't reach. But the **encoder still needs forward DCT** for the
encode-final-residual step when committing the chosen mode — even at
cpu-used 7, the picked partition is encoded with a real forward DCT and
quantised. So all sizes are reachable.

**Differential test plan:**

- Inverse-round-trip: `idct(fdct(x))` should match `x` to within rounding
  error described in the VP9 spec §8.7.
- Fixed test vectors from libvpx `test/dct_test.cc:Trans4x4DCT`.
- DC-only inputs (only the [0,0] coefficient nonzero).

### 3.17 Forward transforms — ADST/Hybrid + Walsh-Hadamard

**Citation:** `vp9/common/vp9_rtcd_defs.pl:151-164`.

```rust
pub type FhtFn = unsafe extern "C" fn(
    input: *const i16, output: *mut i16,
    stride: i32, tx_type: i32,
);

pub type FwhtFn = FdctFn;
```

| Kernel | ISAs |
|---|---|
| `vp9_fht4x4` | scalar, sse2, neon |
| `vp9_fht8x8` | scalar, sse2, neon |
| `vp9_fht16x16` | scalar, sse2, neon |
| `vp9_fwht4x4` | scalar, sse2 |

**Use sites:** intra blocks with `tx_type != DCT_DCT` (ADST_DCT,
DCT_ADST, ADST_ADST). The non-RD path **does** reach this — intra mode
decision at the SB boundary uses ADST for sloped predictors. `vp9_fwht4x4`
is lossless-mode only (D-01 RESOLVED — lossless is in scope only for
conformance; encoder doesn't produce lossless in v1).

**Differential test plan:** as §3.16, varying tx_type ∈ {0,1,2,3} for fht.

### 3.18 Inverse transforms — DCT

**Citation:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:620-652`.

```rust
pub type IdctFn = unsafe extern "C" fn(
    input: *const i16, dest: *mut u8, stride: i32,
);
```

| Kernel | ISAs (8-bit) |
|---|---|
| `vpx_idct4x4_16_add` | scalar, sse2, neon |
| `vpx_idct4x4_1_add` | scalar, sse2, neon |
| `vpx_idct8x8_64_add` | scalar, sse2, neon |
| `vpx_idct8x8_12_add` | scalar, sse2, ssse3, neon |
| `vpx_idct8x8_1_add` | scalar, sse2, neon |
| `vpx_idct16x16_256_add` | scalar, sse2, avx2, neon |
| `vpx_idct16x16_38_add` | scalar, sse2, neon |
| `vpx_idct16x16_10_add` | scalar, sse2, neon |
| `vpx_idct16x16_1_add` | scalar, sse2, neon |
| `vpx_idct32x32_1024_add` | scalar, sse2, avx2, neon |
| `vpx_idct32x32_135_add` | scalar, sse2, ssse3, avx2, neon |
| `vpx_idct32x32_34_add` | scalar, sse2, ssse3, neon |
| `vpx_idct32x32_1_add` | scalar, sse2, neon |
| `vpx_iwht4x4_16_add` | scalar, sse2 |
| `vpx_iwht4x4_1_add` | scalar |

For v1 we ship the listed set, with **ssse3 promoted to mandatory** for
the `idct8x8_12 / idct32x32_135 / idct32x32_34` cases (the SSE2 fallback
exists but is markedly slower and we cannot skip it since SSSE3 is
available on all current x86_64 hardware).

The 1, 12, 38, 135, 10, 34 numeric suffixes are **EOB-based fast paths**:
the dispatcher picks the smallest one whose EOB upper bound covers the
input, saving work on sparse coefficients.

**Use sites:** decoder + encoder reconstruction loop. Hot path.

**Differential test plan:**

- Inverse-round-trip with forward DCT (§3.16).
- Conformance test vectors from `test/dct_test.cc:Trans32x32`.
- EOB boundary tests: feed exactly N=10, 12, 34, 38, 135 nonzero coefficients
  and verify the fast paths produce correct output.

### 3.19 Inverse transforms — ADST/Hybrid

**Citation:** `vp9/common/vp9_rtcd_defs.pl:63-67`. (No `specialize` lines for
`vp9_iht*_add` in the encoder rtcd; the ISA-specific versions are
declared in `vpx_dsp_rtcd_defs.pl` and reused. Let me check.) — From a
fresh grep of vpx_dsp: confirmed they ship `vp9_iht{4x4_16,8x8_64,16x16_256}_add`
under `vp9/common/x86/vp9_idct_intrin_sse2.c` and `vp9/common/arm/neon/`.
ISAs: scalar, sse2, neon.

```rust
pub type IhtFn = unsafe extern "C" fn(
    input: *const i16, dest: *mut u8, stride: i32, tx_type: i32,
);
```

**Use sites:** decoder + intra reconstruction for non-DCT_DCT tx_type.

**Differential test plan:** as §3.18, with tx_type ∈ {0,1,2,3}.

### 3.20 Intra predictors

**Citations:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:44-213`.

10 intra modes × 4 sizes (4x4, 8x8, 16x16, 32x32) = 40 base entries, plus
4 `*e_predictor_4x4` mode-corner variants (D45E, D63E, HE, VE) used only
for 4x4 chroma. Total: **44 distinct kernels.**

```rust
pub type IntraPredFn = unsafe extern "C" fn(
    dst: *mut u8, stride: isize,
    above: *const u8, left: *const u8,
);
```

The 10 modes and their VP9 enum values:

| Mode | Symbol  | Description                       |
|------|---------|-----------------------------------|
| 0    | DC      | Average of top + left             |
| 1    | V       | Vertical (replicate top row)      |
| 2    | H       | Horizontal (replicate left col)   |
| 3    | D45     | Diagonal 45° down-right           |
| 4    | D135    | Diagonal 135° down-left           |
| 5    | D117    | 117° (between V and D135)         |
| 6    | D153    | 153° (between H and D135)         |
| 7    | D207    | 207° (below H, towards D45)       |
| 8    | D63     | 63° (between V and D45)           |
| 9    | TM      | True Motion (top + left - top_left) |

Plus the three DC variants (`dc_top`, `dc_left`, `dc_128`) used when the
neighbor is unavailable (= 3 × 4 = 12 additional entries; libvpx folds
these into the same module but they are distinct dispatch slots). Net:
**56 dispatch slots for intra prediction.**

Per-mode ISA coverage condensed (from `vpx_dsp_rtcd_defs.pl:44-213`):

| Mode    | scalar | sse2 / ssse3 (4x4-32x32)  | neon (all) |
|---------|:------:|:-------------------------:|:----------:|
| DC      | yes    | sse2 (all sizes)          | yes        |
| DC top  | yes    | sse2 (all sizes)          | yes        |
| DC left | yes    | sse2 (all sizes)          | yes        |
| DC 128  | yes    | sse2 (all sizes)          | yes        |
| V       | yes    | sse2 (all sizes)          | yes        |
| H       | yes    | sse2 (all sizes)          | yes        |
| D45     | yes    | sse2 (4x4); ssse3 (8x8+)  | yes        |
| D135    | yes    | —                         | yes        |
| D117    | yes    | —                         | yes        |
| D153    | yes    | ssse3 (all sizes)         | yes        |
| D207    | yes    | sse2 (4x4); ssse3 (8x8+)  | yes        |
| D63     | yes    | ssse3 (all sizes)         | yes        |
| TM      | yes    | sse2 (all sizes)          | yes        |
| `*e`    | yes    | —                         | —          |

For v1 we ship scalar + sse2/ssse3 (per libvpx coverage) + neon. No avx2
intra predictors in libvpx 1.14.1; we don't add any.

**Use sites:** intra mode-decision (`pick_intra_mode`, A2 §4) and final
encode reconstruction (`vp9_build_intra_predictors`).

**Differential test plan:**

- Fixed test vectors from `test/intrapred_test.cc:VP9IntraPredictorTest`.
- Random above/left buffer + all 10 modes + all 4 sizes + all available
  ISAs.
- Edge: above and left both all-128 → DC mode outputs all-128.
- Stride invariance: dst stride independent of block size.

### 3.21 Convolve — 8-tap subpel filters

**Citations:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:378-413`.

```rust
pub type ConvolveFn = unsafe extern "C" fn(
    src: *const u8, src_stride: isize,
    dst: *mut u8, dst_stride: isize,
    filter: *const InterpKernel,
    x0_q4: i32, x_step_q4: i32,
    y0_q4: i32, y_step_q4: i32,
    w: i32, h: i32,
);
```

| Kernel | ISAs |
|---|---|
| `vpx_convolve_copy` | scalar, sse2, neon |
| `vpx_convolve_avg`  | scalar, sse2, neon |
| `vpx_convolve8`     | scalar, sse2, ssse3, avx2, neon |
| `vpx_convolve8_horiz`     | scalar, sse2, ssse3, avx2, neon |
| `vpx_convolve8_vert`      | scalar, sse2, ssse3, avx2, neon |
| `vpx_convolve8_avg`       | scalar, sse2, ssse3, avx2, neon |
| `vpx_convolve8_avg_horiz` | scalar, sse2, ssse3, avx2, neon |
| `vpx_convolve8_avg_vert`  | scalar, sse2, ssse3, avx2, neon |
| `vpx_scaled_2d`   | scalar, ssse3, neon |
| `vpx_scaled_horiz` | scalar (only) |
| `vpx_scaled_vert`  | scalar (only) |
| `vpx_scaled_avg_2d` | scalar (only) |
| `vpx_scaled_avg_horiz` | scalar (only) |
| `vpx_scaled_avg_vert` | scalar (only) |

For v1 we ship scalar + ssse3 (promoted to required for 8-tap path) +
avx2 + neon. SSE2-only paths skipped — ssse3 is universally available on
target hardware and `vpx_convolve8_sse2` is purely a fallback for pre-2008
silicon we don't support.

`InterpKernel = [i16; 8]`. There are 4 InterpKernel sets in VP9:
`vp9_eighttap`, `vp9_eighttap_smooth`, `vp9_eighttap_sharp`,
`vp9_bilinear` (`vp9/common/vp9_filter.c`). The convolve kernels are
filter-agnostic; the filter index is selected by the caller via the
`InterpFilter` enum.

**Use sites:** `vp9_build_inter_predictors_sby` / `_sbuv`
(`vp9/common/vp9_reconinter.c`) — every inter mode + final encode
reconstruction. **The biggest single SIMD surface the non-RD path uses**
(A2 §6.13).

**Differential test plan:**

- Fixed test vectors from `test/convolve_test.cc:ConvolveTest`.
- Random src + each of the 4 InterpKernels + all 8 phase offsets +
  varied (w, h).
- Edge: phase 0 (no filter applied; equivalent to copy).
- Boundary: source at top-left corner of buffer (forces edge replication).

### 3.22 Loop filter — `vpx_lpf_*`

**Citations:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:449-483`.

```rust
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

| Kernel | ISAs (8-bit) |
|---|---|
| `vpx_lpf_vertical_16`      | scalar, sse2, neon |
| `vpx_lpf_vertical_16_dual` | scalar, sse2, neon |
| `vpx_lpf_vertical_8`       | scalar, sse2, neon |
| `vpx_lpf_vertical_8_dual`  | scalar, sse2, neon |
| `vpx_lpf_vertical_4`       | scalar, sse2, neon |
| `vpx_lpf_vertical_4_dual`  | scalar, sse2, neon |
| `vpx_lpf_horizontal_16`      | scalar, sse2, avx2, neon |
| `vpx_lpf_horizontal_16_dual` | scalar, sse2, avx2, neon |
| `vpx_lpf_horizontal_8`       | scalar, sse2, neon |
| `vpx_lpf_horizontal_8_dual`  | scalar, sse2, neon |
| `vpx_lpf_horizontal_4`       | scalar, sse2, neon |
| `vpx_lpf_horizontal_4_dual`  | scalar, sse2, neon |

For v1: scalar + sse2 + (avx2 where listed) + neon.

**Use sites:** `vp9_loop_filter_frame` post-encode, also at decode (the
loop filter is **mandatory** at decode; the bitstream specifies whether
to skip per-frame, but our reference decoder must always be able to apply
it). A6 §6 RTCD.

**Differential test plan:**

- Fixed test vectors from `test/lpf_test.cc:Loop8Test6Param`.
- Random pixel buffer + random blimit/limit/thresh values → bit-exact
  output.
- Edge: all pixels equal → no filtering.
- Edge: input lies exactly on the filter boundary thresholds.

### 3.23 ME-specific kernels — integral projection + diamond search

**Citations:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:882-888`, `vp9/common/vp9_rtcd_defs.pl:176-177`.

```rust
pub type IntProRowFn = unsafe extern "C" fn(
    hbuf: *mut i16,         // [i16; 16]
    refp: *const u8, ref_stride: i32, height: i32,
);

pub type IntProColFn = unsafe extern "C" fn(
    refp: *const u8, width: i32,
) -> i16;

pub type VectorVarFn = unsafe extern "C" fn(
    refp: *const i16, src: *const i16, bwl: i32,
) -> i32;
```

| Kernel | ISAs |
|---|---|
| `vpx_int_pro_row` | scalar, sse2, neon |
| `vpx_int_pro_col` | scalar, sse2, neon |
| `vpx_vector_var`  | scalar, sse2, neon |
| `vp9_diamond_search_sad` | scalar, neon (libvpx ships only the NEON specialisation; we drop it — A3 §6.12 notes the AVX2 sdx4df path renders it unnecessary on x86) |

**Use sites:** `vp9_int_pro_motion_estimation` (`vp9/encoder/vp9_mcomp.c`),
A3 §4.7. Hot path.

**Differential test plan:**

- Specific: random `*ref` buffer; integral projection rows/cols should
  satisfy `sum(int_pro_row) == sum_of_all_pixels` and
  `sum(int_pro_col) == sum_of_all_pixels`.
- `vector_var`: dot product of `(ref - mean) * (src - mean)`-style; verify
  against scalar.
- `vp9_diamond_search_sad`: we ship **scalar only** in v1 (divergence
  ledger L-RTCD-03 — relies on AVX2 `sdx4df` for the SAD batches; NEON
  hand-tuned variant from libvpx omitted because the gain over the
  scalar+sdx4df-neon combination is small. Re-evaluate in Stage 5).

### 3.24 Partition-decision helpers — `vpx_avg_8x8`, `vpx_avg_4x4`, `vpx_minmax_8x8`

**Citations:** `vpx_dsp/vpx_dsp_rtcd_defs.pl:835-842`. A2 §6.4/6.5.

```rust
pub type Avg8x8Fn = unsafe extern "C" fn(src: *const u8, stride: i32) -> u32;
pub type MinMax8x8Fn = unsafe extern "C" fn(
    s: *const u8, p: i32, d: *const u8, dp: i32,
    min_out: *mut i32, max_out: *mut i32,
);
```

ISAs: scalar, sse2, neon for all three.

**Use sites:** `choose_partitioning` (`vp9_encodeframe.c`) — variance-based
partition decision in the non-RD path. A2 §3.

**Differential test plan:** trivial reductions; differential over random
buffers.

### 3.25 Denoiser — `vp9_denoiser_filter`

**Citation:** `vp9/common/vp9_rtcd_defs.pl:122-126` (gated by
`CONFIG_VP9_TEMPORAL_DENOISING`). A5 §6.

```rust
pub type DenoiserFilterFn = unsafe extern "C" fn(
    sig: *const u8, sig_stride: i32,
    mc_avg: *const u8, mc_avg_stride: i32,
    avg: *mut u8, avg_stride: i32,
    increase_denoising: i32,
    bs: i32,    // BLOCK_SIZE enum value
    motion_magnitude: i32,
) -> i32;       // 0 = COPY_BLOCK, 1 = FILTER_BLOCK
```

ISAs: scalar, sse2, neon.

**Use sites:** `vp9_denoiser_denoise` (`vp9_denoiser.c`), per-block, in
the tile worker.

**Caveat:** A7 challenge #6 / A5 §6 — libvpx has an open TODO at
`vp9/encoder/vp9_denoiser.c:65-67` warning that `total_adj` may not be
bit-exact across SSE2 and C when `increase_denoising == 1`. We must match
**one** of libvpx-C or libvpx-SSE2 and pick which; differential test will
then permit a documented epsilon in the `increase_denoising=1` case
between our two impls. **Decision:** match libvpx-C as the source of
truth in v1 (it's the spec-of-record; libvpx-SSE2 is documented as a
known mild divergence in `vp9_denoiser.c:65-67`). Divergence ledger
L-RTCD-04 records this.

**Differential test plan:** as §3.6 + the `increase_denoising` two-path
case.

### 3.26 Scale-and-extend frame — `vp9_scale_and_extend_frame`

**Citation:** `vp9/common/vp9_rtcd_defs.pl:223-224`.

```rust
pub type ScaleAndExtendFn = unsafe extern "C" fn(
    src: *const YV12BufferConfig, dst: *mut YV12BufferConfig,
    filter_type: i32, phase_scaler: i32,
);
```

ISAs: scalar, ssse3, neon.

**Use sites:** SVC inter-layer prediction + dynamic resize. **Not in
Stage 4a MVP** (no SVC); ship scalar + ssse3 starting at Stage 4d (temporal
SVC). Listed here so the dispatch table allocates the slot.

### 3.27 Post-proc / filter-by-weight — `vp9_filter_by_weight8x8` / `16x16`

**Citation:** `vp9/common/vp9_rtcd_defs.pl:52-56`.

```rust
pub type FilterByWeightFn = unsafe extern "C" fn(
    src: *const u8, src_stride: i32,
    dst: *mut u8, dst_stride: i32,
    src_weight: i32,
);
```

ISAs: scalar, sse2.

**Use sites:** Decoder post-processing (deblock/visualisation). **Not on
the encoder hot path.** Ship scalar in v1; decoder LLD owns SIMD when it
gets there.

### 3.28 Frame border / extend / copy_y — `vpx_scale` family

**Citation:** `vpx_scale/vpx_scale_rtcd.pl`.

```rust
pub type ExtendBordersFn = unsafe extern "C" fn(ybf: *mut YV12BufferConfig);
pub type CopyYFn = unsafe extern "C" fn(src: *const YV12BufferConfig, dst: *mut YV12BufferConfig);
```

| Kernel | ISAs |
|---|---|
| `vp8_yv12_extend_frame_borders` | scalar only in vpx_scale_rtcd.pl (sse2 variant lives in vpx_scale source as a normal function, **not** RTCD-dispatched in libvpx) |
| `vp8_yv12_copy_frame` | scalar |
| `vpx_yv12_copy_y` | scalar |

Three scalar-only kernels; trivial port. No SIMD slot.

### 3.29 Kernels explicitly NOT in v1 scope

For traceability — kernels libvpx ships at 8-bit that we deliberately
don't implement in v1:

| Kernel | Reason |
|---|---|
| `vp9_apply_temporal_filter` (SSE4.1, NEON) | Only used by altref path; D-13 RESOLVED no altref in v1 |
| All HBD kernels (`vpx_highbd_*`) | D-01 RESOLVED 8-bit only in v1 |
| `vpx_plane_add_noise` | Decoder post-proc only; opt-in only |
| `vpx_post_proc_down_and_*` family | Decoder post-proc |
| `vpx_avg_8x8`-AVX2 / `vpx_avg_4x4`-AVX2 | Not specialised in libvpx |
| `vp9_diamond_search_sad_neon` | A3 §6.12 — scalar+sdx4df-neon competitive |
| MIPS MSA / mmi / VSX / LSX / DSPR2 / NEON dotprod / NEON i8mm / AVX-512 specialisations | Non-target architectures or marginal-win extensions |

These are tabulated as v2 (or never) expansion vectors; the dispatch table
struct does NOT have slots for any of them.

---

## 4. Dispatch-table layout

### 4.1 Top-level struct

```rust
// rvp9-core/src/dsp/dispatch.rs
//
// One struct, one OnceLock, one init function. All scalar fn pointers are
// always populated; SIMD fn pointers are Some(...) only when both the
// build supports the ISA AND the runtime CPU supports it.

#[repr(C)]
pub struct SimdDispatch {
    pub sad: SadDispatch,
    pub sad_skip: SadDispatch,
    pub sad_avg: SadAvgDispatch,
    pub sad_x4d: SadX4dDispatch,
    pub sad_skip_x4d: SadX4dDispatch,
    pub variance: VarianceDispatch,
    pub sub_pixel_variance: SubPixelVarianceDispatch,
    pub sub_pixel_avg_variance: SubPixelAvgVarianceDispatch,
    pub get8x8var: VarSumFn,
    pub get16x16var: VarSumFn,
    pub mse: MseDispatch,
    pub comp_avg_pred: CompAvgPredFn,
    pub subtract_block: SubtractBlockFn,
    pub sum_squares_2d_i16: SumSquares2dI16Fn,
    pub sse: SseFn,
    pub get4x4sse_cs: VarianceFn,    // signature reuse
    pub get_mb_ss: GetMbSsFn,
    pub hadamard_8x8: HadamardFn,
    pub hadamard_16x16: HadamardFn,
    pub hadamard_32x32: HadamardFn,
    pub satd: SatdFn,
    pub block_error_fp: BlockErrorFpFn,
    pub block_error: BlockErrorFn,
    pub quantize_fp: QuantizeFpFn,
    pub quantize_fp_32x32: QuantizeFpFn,
    pub quantize_b: QuantizeBFn,
    pub quantize_b_32x32: QuantizeBFn,
    pub fdct: FdctDispatch,            // (4,8,16,32) + (4_1,8_1,16_1,32_1) + 32_rd
    pub fht: FhtDispatch,              // (4,8,16) + fwht4x4
    pub idct: IdctDispatch,            // 14 variants
    pub iht: IhtDispatch,              // 3 variants
    pub intra_pred: IntraPredDispatch, // 56 slots
    pub convolve: ConvolveDispatch,    // 8-tap (8) + scaled (6) + copy + avg
    pub lpf: LpfDispatch,              // 12 entries
    pub int_pro_row: IntProRowFn,
    pub int_pro_col: IntProColFn,
    pub vector_var: VectorVarFn,
    pub diamond_search_sad: DiamondSearchSadFn,
    pub avg_8x8: Avg8x8Fn,
    pub avg_4x4: Avg8x8Fn,
    pub minmax_8x8: MinMax8x8Fn,
    pub denoiser_filter: DenoiserFilterFn,
    pub scale_and_extend_frame: ScaleAndExtendFn,
    pub filter_by_weight_8x8: FilterByWeightFn,
    pub filter_by_weight_16x16: FilterByWeightFn,
    pub yv12_extend_borders: ExtendBordersFn,
    pub yv12_copy_frame: CopyYFn,
    pub yv12_copy_y: CopyYFn,
}

unsafe impl Send for SimdDispatch {}
unsafe impl Sync for SimdDispatch {}
```

`Send` and `Sync` are safe because every field is a `fn` pointer — once
populated, it never changes; the `fn` itself is reentrant
(`unsafe extern "C" fn` from libvpx semantics).

### 4.2 Per-family sub-table

Block-size-indexed families use one struct per family with a field per size:

```rust
#[repr(C)]
pub struct SadDispatch {
    pub b4x4:   SadFn,
    pub b4x8:   SadFn,
    pub b8x4:   SadFn,
    pub b8x8:   SadFn,
    pub b8x16:  SadFn,
    pub b16x8:  SadFn,
    pub b16x16: SadFn,
    pub b16x32: SadFn,
    pub b32x16: SadFn,
    pub b32x32: SadFn,
    pub b32x64: SadFn,
    pub b64x32: SadFn,
    pub b64x64: SadFn,
}

impl SadDispatch {
    #[inline]
    pub fn get(&self, bsize: BlockSize) -> SadFn {
        match bsize {
            BlockSize::B4x4   => self.b4x4,
            BlockSize::B4x8   => self.b4x8,
            BlockSize::B8x4   => self.b8x4,
            BlockSize::B8x8   => self.b8x8,
            BlockSize::B8x16  => self.b8x16,
            BlockSize::B16x8  => self.b16x8,
            BlockSize::B16x16 => self.b16x16,
            BlockSize::B16x32 => self.b16x32,
            BlockSize::B32x16 => self.b32x16,
            BlockSize::B32x32 => self.b32x32,
            BlockSize::B32x64 => self.b32x64,
            BlockSize::B64x32 => self.b64x32,
            BlockSize::B64x64 => self.b64x64,
        }
    }
}
```

The exhaustive `match` is jump-table-compiled by rustc; the worker hot
loop sees a single load from the dispatch table.

### 4.3 BFP table — encoder-internal mirror of libvpx's `cpi->fn_ptr[BLOCK_SIZES]`

For the encoder we precompute the BFP table (architecture A1 §5,
`vp9_encoder.c:2664-2724`) once per encoder lifetime, copying pointers
out of `SimdDispatch`:

```rust
// rvp9-encoder/src/bfp.rs
pub struct BlockFnPtrs {
    pub sdf: SadFn,
    pub sdsf: SadFn,
    pub sdaf: SadAvgFn,
    pub vf: VarianceFn,
    pub svf: SubPixelVarianceFn,
    pub svaf: SubPixelAvgVarianceFn,
    pub sdx4df: SadX4dFn,
    pub sdsx4df: SadX4dFn,
}

pub struct Bfp {
    table: [BlockFnPtrs; 13],
}

impl Bfp {
    pub fn new(d: &SimdDispatch) -> Self {
        let mk = |b: BlockSize| BlockFnPtrs {
            sdf:    d.sad.get(b),
            sdsf:   d.sad_skip.get(b),
            sdaf:   d.sad_avg.get(b),
            vf:     d.variance.get(b),
            svf:    d.sub_pixel_variance.get(b),
            svaf:   d.sub_pixel_avg_variance.get(b),
            sdx4df: d.sad_x4d.get(b),
            sdsx4df: d.sad_skip_x4d.get(b),
        };
        Bfp {
            table: [
                mk(BlockSize::B4x4),  mk(BlockSize::B4x8),
                mk(BlockSize::B8x4),  mk(BlockSize::B8x8),
                mk(BlockSize::B8x16), mk(BlockSize::B16x8),
                mk(BlockSize::B16x16),mk(BlockSize::B16x32),
                mk(BlockSize::B32x16),mk(BlockSize::B32x32),
                mk(BlockSize::B32x64),mk(BlockSize::B64x32),
                mk(BlockSize::B64x64),
            ],
        }
    }
    #[inline]
    pub fn get(&self, b: BlockSize) -> &BlockFnPtrs { &self.table[b as usize] }
}
```

The worker reads through `&Bfp` (owned by the encoder, immutable after
init, shared `&` across workers). This matches the libvpx access pattern
exactly — `cpi->fn_ptr[bsize].vf(...)`.

### 4.4 Population — example for SAD

```rust
// rvp9-core/src/dsp/dispatch.rs

impl SimdDispatch {
    fn build(caps: &Caps) -> Self {
        SimdDispatch {
            sad: SadDispatch::build(caps),
            // ...
        }
    }
}

impl SadDispatch {
    fn build(caps: &Caps) -> Self {
        macro_rules! pick {
            ($size:ident) => {{
                #[cfg(target_arch = "x86_64")] {
                    if caps.avx2 && has_avx2_sad::$size() {
                        sad::avx2::$size as SadFn
                    } else if caps.sse2 {
                        sad::sse2::$size as SadFn
                    } else {
                        sad::scalar::$size as SadFn
                    }
                }
                #[cfg(target_arch = "aarch64")] {
                    if caps.neon {
                        sad::neon::$size as SadFn
                    } else {
                        sad::scalar::$size as SadFn
                    }
                }
                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))] {
                    sad::scalar::$size as SadFn
                }
            }}
        }

        SadDispatch {
            b4x4:   pick!(sad4x4),
            b4x8:   pick!(sad4x8),
            // ... etc.
        }
    }
}

// has_avx2_sad::<size>() returns true only for the sizes where libvpx
// ships an AVX2 variant; this is a compile-time const fn.
```

The cosmetics matter: it must read clean enough that a reviewer can verify
*by eye* that every size has the right ISAs. We deliberately keep each
family in its own file so the per-family table fits on one screen.

### 4.5 Scalar reference module

Every kernel has a scalar implementation in a `scalar/` submodule that is
**always compiled** regardless of target arch or features:

```rust
// rvp9-core/src/dsp/sad/scalar.rs

#[inline(never)]
pub fn sad4x4(src: *const u8, ss: i32, refp: *const u8, rs: i32) -> u32 {
    // Bit-exact port of vpx_sad4x4_c from vpx_dsp/sad.c:60-85.
    // SAFETY: caller guarantees `src` and `refp` are valid for reads of
    // 4×4 bytes at strides `ss` and `rs` respectively.
    let mut sad = 0u32;
    for j in 0..4 {
        for i in 0..4 {
            let s = unsafe { *src.add((j * ss + i) as usize) } as i32;
            let r = unsafe { *refp.add((j * rs + i) as usize) } as i32;
            sad += (s - r).unsigned_abs();
        }
    }
    sad
}
```

The scalar impl is `#[inline(never)]` to ensure it has its own symbol —
the differential test wants a real function pointer to call, not an
inlined copy.

### 4.6 ISA-specialised module

```rust
// rvp9-core/src/dsp/sad/avx2.rs

use core::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub unsafe fn sad32x32(src: *const u8, ss: i32, refp: *const u8, rs: i32) -> u32 {
    // SAFETY: caller guarantees AVX2 is supported (checked by dispatch
    // table at init) and that pointers are valid for reads of 32×32 bytes
    // at the given strides.
    let mut acc = _mm256_setzero_si256();
    for j in 0..32 {
        let s = _mm256_loadu_si256(src.add((j * ss) as usize) as *const __m256i);
        let r = _mm256_loadu_si256(refp.add((j * rs) as usize) as *const __m256i);
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(s, r));
    }
    // horizontal sum of the four u64 lanes
    let s0 = _mm256_extracti128_si256::<0>(acc);
    let s1 = _mm256_extracti128_si256::<1>(acc);
    let s = _mm_add_epi64(s0, s1);
    let s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
    _mm_cvtsi128_si64(s) as u32
}
```

Note: the `unsafe fn` is required because `#[target_feature(enable = "avx2")]`
forces the caller-callee target-feature contract. The dispatch table holds
the function pointer with the matching signature (`unsafe extern "C" fn`)
and the caller marks the call site `unsafe`.

---

## 5. Scalar-fallback policy

### 5.1 Three roles of the scalar reference

For every kernel listed in §3, the scalar implementation must exist and
must serve three distinct purposes:

1. **Build-without-target-features fallback.** `cargo build --target
   x86_64-unknown-linux-gnu -C target-feature=-avx2,-sse4.1,-ssse3` must
   produce a working encoder that uses scalar everywhere SSE2 is also
   unavailable. (SSE2 is unconditionally available on x86_64 by spec.)
2. **Differential testing oracle.** The scalar impl is the "trusted by
   construction" reference. Every SIMD impl is tested by feeding the same
   input through both and asserting bit-exact output equality.
3. **Cross-platform correctness.** Targets we don't ship SIMD for —
   wasm32, riscv64, ppc64le, mips64, etc. — fall through to scalar
   everywhere and produce correct (if slow) output.

### 5.2 The contract

> **Contract — Scalar Reference:** For every SIMD kernel in the dispatch
> table there exists a scalar function with the same signature and
> semantics. The scalar function is bit-exactly equivalent to libvpx's
> `*_c` reference for that kernel (the same `.c` file the SIMD specialised
> from). The scalar function is always compiled and always reachable via
> the same dispatch path (`SimdDispatch` field with `pick!` falling
> through to `scalar::*`).

### 5.3 The "scalar-only" Cargo feature

```toml
# rvp9-core/Cargo.toml
[features]
default = []
scalar-only = []  # Disable all SIMD modules; force scalar dispatch.
```

```rust
// rvp9-core/src/dsp/sad/mod.rs

pub mod scalar;
#[cfg(all(not(feature = "scalar-only"), target_arch = "x86_64"))]
pub mod sse2;
#[cfg(all(not(feature = "scalar-only"), target_arch = "x86_64"))]
pub mod avx2;
#[cfg(all(not(feature = "scalar-only"), target_arch = "aarch64"))]
pub mod neon;
```

With `scalar-only`, the SIMD modules are not compiled; the dispatch
table's `pick!` macros short-circuit to scalar; the binary is smaller and
the build is faster (useful for `miri` runs and for `wasm32` targets that
don't have target_feature variants).

### 5.4 What does "bit-exact" mean?

For all integer kernels (SAD, variance, transforms in the integer
domain, intra-pred, loop-filter, quant, Hadamard, SATD, etc.) bit-exact
means literally: for every well-defined input, the scalar and SIMD impls
return the same bit pattern in every output byte.

For the **forward DCT**, libvpx uses lifting and shift-rounding that is
identical across C and SIMD by construction; we mirror exactly.

For the **inverse DCT**, the same applies: integer math throughout, no
floating point.

There are no float kernels in the SIMD dispatch table for VP9. (The
encoder has float code in the **rate controller** and **partition
heuristics**, but those are not RTCD-dispatched.)

### 5.5 Edge case: the denoiser `increase_denoising=1` known divergence

Per §3.25 and A7 challenge #6, libvpx itself has a known SSE2-vs-C
discrepancy in `vp9_denoiser_filter` when `increase_denoising=1`. Our
scalar impl matches libvpx-C (the spec-of-record). Our SSE2 impl will
match libvpx-SSE2 (so we get matching output if a user opts into SIMD).
The differential test for `vp9_denoiser_filter` allows a documented epsilon
(≤ 1 pixel) only when `increase_denoising=1`; with `increase_denoising=0`
the diff is bit-exact. Divergence ledger L-RTCD-04 records this.

### 5.6 What if a SIMD impl diverges from scalar?

The differential test fails and the PR cannot land. There is no opt-out;
if a SIMD impl is wrong, we either fix the SIMD impl, fix the scalar impl
(if the scalar was wrong and libvpx-C also was the bug), or document the
divergence in the divergence ledger and add a per-kernel epsilon to the
test (which requires Arthur sign-off).

---

## 6. Testing strategy

### 6.1 The test pyramid

Three layers, each running on every CI build:

1. **Unit tests** — scalar vs SIMD differential, per kernel, ≥ 10 000
   random inputs per ISA per kernel.
2. **Integration tests** — feed a small encode through the full encoder
   with `scalar-only`, then with SIMD on; assert the produced bitstream
   bytes are identical.
3. **Conformance tests** — VP9 official test vectors decoded through our
   decoder; both scalar and SIMD configs must pass.

### 6.2 Unit-test framework

Per-kernel differential tests live in `rvp9-core/src/dsp/<family>/tests.rs`.

```rust
// rvp9-core/src/dsp/sad/tests.rs

use proptest::prelude::*;
use super::{scalar, sse2, avx2};

fn assert_sad_match(
    f_scalar: SadFn, f_simd: SadFn,
    src: &[u8], ss: i32, refp: &[u8], rs: i32,
) {
    let s = unsafe { f_scalar(src.as_ptr(), ss, refp.as_ptr(), rs) };
    let v = unsafe { f_simd(src.as_ptr(), ss, refp.as_ptr(), rs) };
    assert_eq!(s, v,
        "scalar={} simd={} src_stride={} ref_stride={}", s, v, ss, rs);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn sad4x4_sse2_matches_scalar(
        src in prop::collection::vec(any::<u8>(), 16..64),
        refp in prop::collection::vec(any::<u8>(), 16..64),
        ss in 4i32..16,
        rs in 4i32..16,
    ) {
        if cfg!(target_arch = "x86_64") && is_x86_feature_detected!("sse2") {
            assert_sad_match(scalar::sad4x4, sse2::sad4x4, &src, ss, &refp, rs);
        }
    }

    #[test]
    fn sad4x4_avx2_matches_scalar(
        // No AVX2 sad4x4 in libvpx; skip.
    ) { /* ... */ }

    // ... one #[test] per (kernel, ISA) pair.
}

// Edge cases as standalone #[test] functions:
#[test]
fn sad4x4_all_zero() {
    let src = [0u8; 16];
    let refp = [0u8; 16];
    let s = unsafe { scalar::sad4x4(src.as_ptr(), 4, refp.as_ptr(), 4) };
    assert_eq!(s, 0);
}

#[test]
fn sad4x4_all_max() {
    let src = [0u8; 16];
    let refp = [255u8; 16];
    let s = unsafe { scalar::sad4x4(src.as_ptr(), 4, refp.as_ptr(), 4) };
    assert_eq!(s, 4 * 4 * 255);
}
```

### 6.3 Test-case multiplier per kernel

For each kernel, the unit-test count is:

```
test_cases_per_kernel =
    edge_cases (≈ 5-10)
  + proptest_cases (10 000 default; bumped to 100 000 nightly)
  + libvpx_test_vectors (≈ 50, ported from `test/`)
```

Total per kernel per ISA per CI run: ~10 100 cases. With ~240 kernels × ~3
ISAs each, that's ~7 200 000 scalar-vs-SIMD comparisons per CI run.
Anticipated wall-time on a single mid-range x86_64 CI host: 3-5 minutes
total for the entire SIMD differential suite. Acceptable.

### 6.4 Stride / alignment matrix

For pointer-stride kernels, the proptest strategy generates:

- Tight stride (stride == w).
- Loose stride (stride ∈ [w, 4*w]).
- Source stride != Reference stride.
- Source pointer alignment ∈ {0, 1, 2, 4, 8, 16, 32 bytes} mod 32.

This is critical for AVX2: unaligned loads have different latency than
aligned, and a buggy SIMD impl that assumes alignment will fail only on
specific alignments.

### 6.5 Integration test — full-encode bitstream equivalence

```rust
// tests/differential/scalar_vs_simd.rs

use rvp9_encoder::{Encoder, EncoderConfig};

#[test]
fn scalar_and_simd_produce_identical_bitstream() {
    // Encode a short clip with scalar dispatch.
    let cfg = EncoderConfig {
        cpu_used: 7,
        bitrate: 1_000_000,
        ..Default::default()
    };
    let bitstream_scalar = {
        // Force scalar via env var that the dispatch table reads.
        std::env::set_var("RVP9_FORCE_SCALAR", "1");
        let mut enc = Encoder::new(cfg).unwrap();
        encode_test_clip_to_bytes(&mut enc)
    };
    let bitstream_simd = {
        std::env::remove_var("RVP9_FORCE_SCALAR");
        let mut enc = Encoder::new(cfg).unwrap();
        encode_test_clip_to_bytes(&mut enc)
    };
    assert_eq!(bitstream_scalar, bitstream_simd,
        "scalar and SIMD must produce identical bitstreams");
}
```

This is the gate that catches "SIMD impl looks bit-exact per unit but
diverges in some path no unit test exercises."

### 6.6 Per-kernel test summary table

For brevity, this is the test-strategy summary per kernel family.
"Differential" means scalar-vs-SIMD bit-exact; "epsilon" denotes a known
small allowed deviation with divergence-ledger entry.

| Kernel family       | Diff strategy | Edge cases             | Notes |
|---------------------|---------------|------------------------|-------|
| SAD / SAD_skip / SADx4D / SAD_skip_x4D | bit-exact | zero, max, equal-buffers, stride mismatch | hot path |
| SAD_avg / sub-pixel-avg | scalar-only (path unreachable) | as above | L-RTCD-02 |
| Variance / sub-pixel-variance | bit-exact | zero, max, src==ref, all phases | rounding sensitive |
| Hadamard / SATD | bit-exact + inverse round-trip | impulse, DC-only | tx unitarity |
| Quantize fp / b | bit-exact | all-zero coeffs, DC-only, EOB boundaries | EOB reduction tricky |
| Forward DCT (4-32) | bit-exact + inverse round-trip | impulse | spec-defined |
| Inverse DCT (4-32) | bit-exact + forward round-trip | EOB-bucket boundaries (1, 10, 12, 34, 38, 135) | fast-path correctness |
| Forward ADST (fht) | bit-exact + inverse | tx_type 0..3 | |
| Inverse ADST (iht) | bit-exact + forward | tx_type 0..3 | |
| Intra predictors | bit-exact | flat input → flat output, DC variants | 10 modes × 4 sizes × 4 dc variants |
| Convolve (copy/avg/8-tap) | bit-exact | phase 0 → copy, w/h corner cases | 4 filter sets |
| Loop filter | bit-exact | already-flat input, threshold boundaries | hot path |
| Int_pro_row / col / vector_var | bit-exact | + sum-invariant | |
| Diamond search SAD | scalar-only | (NEON-only in libvpx; we drop SIMD) | L-RTCD-03 |
| `vpx_avg_*` / `vpx_minmax_8x8` | bit-exact | uniform input | |
| Denoiser filter | bit-exact for inc=0; epsilon ≤ 1 for inc=1 | block-size matrix | L-RTCD-04 |
| Scale-and-extend frame | bit-exact | (Stage 4d+) | |
| Filter-by-weight | bit-exact | (decoder only; scalar enough) | |
| YV12 extend / copy | bit-exact / scalar-only | (no SIMD slot) | |

### 6.7 Fuzz testing

`cargo fuzz` targets — one per major kernel family, feeding arbitrary
(size, stride, content) tuples — run nightly. Goal: discover any path
that triggers UB (out-of-bounds, signed overflow). The fuzz harnesses
share the differential assertion macro from §6.2.

### 6.8 Sanitizers

- **Miri** runs the entire scalar test suite. SIMD modules are excluded
  (Miri does not support intrinsics).
- **AddressSanitizer** runs the SIMD test suite (Linux only, with
  `RUSTFLAGS="-Z sanitizer=address" cargo +nightly test`).
- **UBSan** runs the SIMD test suite. Critical: catches signed-overflow
  in DCT/IDCT and quantise paths.

These are nightly CI, not per-PR (too slow). Per-PR runs are just the
differential tests.

---

## 7. Performance regression CI

### 7.1 Goal

Catch performance regressions per kernel **before** they merge. The
target is: "no kernel may slow down by more than 5% on the reference
hardware without an explicit ledger entry justifying the regression."

### 7.2 Mechanism

Criterion benchmark per kernel family:

```rust
// rvp9-bench/benches/sad.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_sad(c: &mut Criterion) {
    let d = rvp9_core::dsp::dispatch::get();
    let src = vec![0u8; 64*64];
    let refp = vec![0u8; 64*64];
    let mut grp = c.benchmark_group("sad");
    for size in &[BlockSize::B8x8, BlockSize::B16x16, BlockSize::B32x32, BlockSize::B64x64] {
        grp.bench_with_input(
            BenchmarkId::new("dispatched", format!("{:?}", size)),
            size,
            |b, &s| {
                let f = d.sad.get(s);
                b.iter(|| unsafe {
                    black_box(f(src.as_ptr(), 64, refp.as_ptr(), 64))
                });
            },
        );
    }
    grp.finish();
}

criterion_group!(benches, bench_sad);
criterion_main!(benches);
```

CI runs Criterion in compare mode against the previous commit on the
same hardware class:

```bash
cargo bench --bench sad -- --baseline previous-master --strict
```

`--strict` fails the build if a benchmark's median time exceeds the
baseline by more than the threshold (default 5%, configurable per family
with `criterion.toml`).

### 7.3 What we benchmark

- One benchmark per `(kernel, size, ISA)` tuple, dispatched through the
  real `SimdDispatch`. That's ~240 × ~3 = 720 benchmarks.
- Wall-time per benchmark: ~3-10 seconds. Full bench run: ~30-60 minutes
  per ISA.

This is too slow for per-PR. We run:

- **Per-PR:** a fast subset — 1 representative size per family,
  dispatched-only (no scalar). ~30 benchmarks, ~5 minutes.
- **Nightly:** the full bench suite, with comparison against the prior
  night's baseline. Results posted to `bench/results/`.
- **Per-release:** full bench against libvpx baseline at matched config,
  on the dev machine + on a CI Linux box (different microarchitecture).

### 7.4 Regression policy

If a per-PR bench fails by > 5%:

1. The PR author either fixes the regression or
2. Adds a divergence-ledger entry justifying the slowdown (e.g. "we
   chose simpler code that's 6% slower because it eliminates an
   unsafe block").

If a nightly bench fails by > 5%, the daily commit-bisect process
identifies the responsible commit and opens an issue.

### 7.5 Hardware variation

CI runs on multiple x86_64 microarchitectures (we target at minimum:
Sandy Bridge as the SSE2 floor, Haswell as the AVX2 floor, Skylake as the
AVX2 "modern" target). Each runs the same suite; the bench gate is
per-microarchitecture so that a Haswell win on an AVX2 path doesn't mask
a Skylake regression.

Aarch64 CI runs on a single Apple Silicon / Ampere class machine. NEON
gates are softer (10% threshold) because aarch64 CI hosts are noisier.

---

## 8. Open questions

These are surfaced to Arthur for either resolution or explicit
deferral. Items with **\[needs HLD-arch\]** or **\[needs HLD-threading\]**
are dependencies on other in-flight HLDs and assume the noted defaults
unless the peer HLD overrules.

### 8.1 Init sequencing relative to encoder constructor — \[needs HLD-arch\]

**Assumed default:** `Encoder::new()` and `Decoder::new()` each call
`rvp9_core::dsp::dispatch::init()` as their first action. The encoder /
decoder constructor returns only after `OnceLock` is populated; from
that point any worker spawned by the encoder/decoder may safely read.

**Risk:** if `HLD-arch` chooses a different lifecycle (e.g. `Encoder` is
just a struct, with a separate `EncoderRuntime::start()` that spawns
workers), the init point shifts. We need confirmation that the chosen
lifecycle guarantees R-09 (init-before-worker-spawn).

### 8.2 Worker access pattern to dispatch table — \[needs HLD-threading\]

**Assumed default:** workers receive `&'static SimdDispatch` (via
`dispatch::get()`) plus `&Bfp` (via `Arc<Encoder>` shared state). Both
are immutable for the worker's lifetime.

**Risk:** if `HLD-threading` opts for thread-local function-pointer
caches (e.g. to avoid pointer chasing in the inner loop), our table
design is unaffected — the worker just memcpys the pointers it cares
about into its `ThreadData` once. But the locality of access pattern
should be confirmed.

### 8.3 Should we pre-bind kernel pointers in `cpi->find_fractional_mv_step`-style? — \[needs HLD-motion-estimation\]

A1 §6.3 notes per-frame function-pointer wiring:
`cpi->find_fractional_mv_step` and `cpi->diamond_search_sad` are set
per-frame by speed features. These are second-level dispatch on top of
the RTCD-level table. The HLD-motion-estimation should adopt the same
pattern: `EncoderFrame::find_fractional_mv_step: SubPixelSearchFn`,
re-bound each frame from `cpi->sf`.

This HLD does **not** populate those. They live one layer up in the
encoder state machine.

### 8.4 fwd_txfm4x4 / inv_txfm_add lossless flip — \[needs HLD-transforms\]

A1 §6.3: libvpx flips `cpi->td.mb.fwd_txfm4x4` and `inv_txfm_add` between
lossless and lossy at `vp9_encodeframe.c:6086-6095`. Lossless is not v1
scope as an encoder mode (D-13), but our **decoder must support lossless
frames** for conformance vector coverage.

Our `SimdDispatch` includes both `iwht4x4_*` variants in the inverse-DCT
slot. The per-frame "flip" is just "pick `vpx_iwht4x4_16_add` vs
`vpx_idct4x4_16_add` based on the lossless bit." That logic lives in the
decoder LLD, not here.

### 8.5 SIMD coverage gaps in unreachable paths — \[needs Stage 4a smoke\]

We are deliberately shipping scalar-only for: SAD_avg, SubPixelAvgVar,
Diamond_search_sad, Block_error (non-fp), Apply_temporal_filter,
Filter_by_weight, Scale_and_extend_frame (until Stage 4d).

**Risk:** Stage 4a's first end-to-end encode may surface a call path
through one of these that we believed was unreachable. The integration
test in §6.5 catches this: if the scalar dispatch is hit on the encoder
hot path, the bench gate (§7) will reveal the slowdown immediately and
we'll need to either (a) realise the path is reachable after all and
ship SIMD, or (b) understand why our model said it was unreachable but
the encoder dispatch reached it (probably a speed-feature corner case).

### 8.6 Re-dispatch flag — A4 §6 thin coverage

A4's §6 RTCD section is two paragraphs: "the entropy coder and bitstream
paths are entirely scalar." This is correct (entropy is a serial
arithmetic loop), but the section is thin. **No re-dispatch needed** —
A4 owns the bitstream surface and confirmed no SIMD; the boolean coder
will be hand-tuned scalar Rust (see LLD-bitstream).

Likewise A5 §6 covers only `vp9_denoiser_filter` plus a list of
"depends on" kernels (SAD via `cpi->fn_ptr`, `vpx_convolve_copy`,
`vp9_build_inter_predictors_sby`, `vp9_compute_skin_block`). The
denoiser-side SIMD is fully captured in this doc §3.25. No re-dispatch.

A6 §6 is thin: 4 entries (`vp9_accumulate_frame_counts`,
loop-filter row jobs, temporal_filter_*, downscaling for SVC).
`vp9_accumulate_frame_counts` is scalar; loop-filter is captured in
§3.22; temporal_filter is out of v1 (§3.29); SVC downscaling is captured
in §3.26. **No re-dispatch needed.**

A1 §6 covers init-mechanism only (§2 of this doc); the actual kernels
live in §3.

A2 §6 is thorough — 13 sub-sections covering variance, get8x8var, SAD,
avg, minmax, hadamard, satd, block_error_fp, quantize_fp, subtract_block,
int_pro, sub_pixel_variance, inter-pred / convolve. **All captured.**

A3 §6 is thorough — 12 sub-sections covering all SAD variants, variance
variants, comp_avg_pred, integral projection, hadamard / SATD, and
diamond_search_sad. **All captured.**

**No specialist §6 needs re-dispatch.**

### 8.7 `vpx_sse` ISA target — sse4_1

`vpx_sse` is the only kernel that requires sse4_1 (not just sse2). The
sse4_1 baseline (Nehalem, 2008) is comfortably below our minimum target,
and `vpx_sse` is reached only from `vp9_temporal_filter_*` which is out
of scope (§3.29). Ship scalar-only for v1.

### 8.8 Should we ship a build feature for "x86_64 baseline = avx2"?

A consumer running on Haswell+ hardware could opt for a binary that
short-circuits the `is_x86_feature_detected!` calls (saving a branch
per dispatch). This is a v2 micro-optimisation — defer.

### 8.9 CPUID caching across multiple encoder instances

`is_x86_feature_detected!` itself caches its result internally (after the
first call). Multiple encoder instances calling `dispatch::init()`
concurrently each pay the OnceLock init cost once for the whole process;
subsequent calls are a single Acquire load. **No additional caching
needed.**

### 8.10 Aarch64 NEON-dotprod opt-in?

Per §2.1 we don't ship dotprod in v1. The decision is reversible — adding
a column to the dispatch tables is a mechanical edit. Defer to a v1.x
patch release once we have aarch64 production traffic data showing the
gain justifies the maintenance burden.

---

## 9. v2 expansion vectors (HBD path)

**8-bit-only simplification:** PLAN.md D-01 RESOLVED 2026-05-16 to 8-bit
only in v1. This means our SIMD scope is **roughly halved** compared to
libvpx's full surface. Concretely:

| Surface area              | libvpx (8-bit + HBD) | rvp9 v1 (8-bit only) |
|---------------------------|-----------------------:|-----------------------:|
| Distinct RTCD kernels     | ~470 (`grep specialize vpx_dsp_rtcd_defs.pl` × 2 for HBD parallels) | ~243 |
| Scalar implementations    | ~470 | ~243 |
| sse2 specialisations      | ~250 | ~125 |
| avx2 specialisations      | ~120 | ~60 |
| neon specialisations      | ~280 | ~140 |
| Per-kernel scalar diff tests | ~470 | ~243 |
| Per-kernel SIMD diff tests | ~1,200 | ~620 |

(Counts approximate — exact tally is whatever the kernel catalogue in §3
ends up at after specialist sign-off.)

### 9.1 The v2 HBD path

When v2 adds 10/12-bit support, the work is:

1. Duplicate each kernel signature with `*const u16` instead of `*const
   u8` (for source/destination pixels). Coefficients move from `i16` to
   `i32`.
2. Duplicate each kernel implementation. The scalar bodies are mostly
   identical-with-different-widths; the SIMD bodies need rewriting (epu8
   → epu16, etc.).
3. Add new dispatch-table fields per HBD kernel:
   `pub sad_highbd: SadHighbdDispatch`, etc.
4. Encoder lifecycle gains a `bit_depth: BitDepth` enum at construction;
   the encoder picks `&dispatch.sad` or `&dispatch.sad_highbd` once at
   construction, then passes the chosen sub-table to workers.
5. Differential tests duplicate (10-bit, 12-bit).

**Estimated effort to add HBD on top of the v1 architecture:** ~4-6 weeks
of focused work. The architecture in §3 / §4 of this doc is designed so
that adding the `*_highbd` fields is a mechanical extension that does
**not** require redesigning the dispatch table.

### 9.2 Other v2 expansion items

- **AVX-512:** add `avx512f` column to detection caps; ship `vpx_sad64x64x4d_avx512`
  (the only kernel libvpx specialises). ~1 week.
- **NEON dotprod / i8mm:** add `dotprod` / `i8mm` columns; specialise the
  SAD / variance / dot-product-heavy kernels. ~2 weeks for SAD; more for
  variance.
- **WASM SIMD128:** add a `wasm32` arch path; specialise the most-hot
  kernels (SAD x4d, idct/fdct). Useful for browser-embedded use cases.
  ~3 weeks.
- **PPC VSX / MIPS MSA / LoongArch LSX:** libvpx has these; we never
  will unless a paid customer asks. Document only.

---

## 10. Summary

This HLD establishes:

1. **Q2 RESOLVED:** hand-written Rust dispatch tables (§1).
2. **Catalogue:** ~243 distinct 8-bit kernels across 28 families (§3).
3. **Init invariant R-09:** `OnceLock<SimdDispatch>` populated by
   constructor before worker spawn (§2.2).
4. **Dispatch layout:** field-per-kernel structs, sub-tables for
   block-size families, encoder-internal `Bfp` mirror of libvpx's
   `cpi->fn_ptr[]` (§4).
5. **Scalar fallback:** always compiled, always reachable, always the
   test oracle (§5).
6. **Testing:** scalar-vs-SIMD diff (proptest + edge + fixed vectors),
   full-encode bitstream equivalence integration test, conformance
   tests with both scalar and SIMD configs (§6).
7. **Bench gate:** Criterion-based per-kernel-per-ISA regression
   detection at 5% threshold (§7).
8. **8-bit simplification:** scope ~halved relative to libvpx's full
   8-bit + HBD surface; clean v2 expansion vector for HBD (§9).

Cross-LLD ownership of the kernels:

- `notes/lld/transforms.md` — fdct, idct, fht, iht, fwht, iwht.
- `notes/lld/intra-pred.md` — all 56 intra-pred slots.
- `notes/lld/loopfilter.md` — all 12 lpf slots.
- `notes/lld/quant.md` — quantize_fp/_32x32, quantize_b/_32x32.
- `notes/lld/motion-estimation.md` — SAD family, sub-pixel variance,
  int_pro, vector_var, diamond_search_sad, comp_avg_pred.
- `notes/lld/mode-decision.md` — variance, hadamard, satd, block_error_fp,
  subtract_block, avg_8x8, avg_4x4, minmax_8x8.
- `notes/lld/denoiser.md` — denoiser_filter.
- `notes/lld/svc.md` — scale_and_extend_frame.
- `notes/lld/frame-buffers.md` — yv12_extend_borders, yv12_copy_frame,
  yv12_copy_y.

Each LLD references the relevant §3 sub-section here for the
function-pointer signature, the dispatch-table slot, and the
test-coverage contract.

---

*End of HLD-SIMD.*
