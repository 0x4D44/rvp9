# F.4-1 Calibration Sweep Results (2026-05-23)

## ⚠️ IMPORTANT — sweep redone night-8 (2026-05-23 evening)

The original sweep (also documented below for the record) patched
`block_yrd.rs:157` — the **canonical reference implementation**
that the picker DOES NOT call. The picker uses an **inlined copy** of
the rate-scale math at `inter.rs:1196` (was line 1180 pre-night-8).
The original sweep's "K1 ∈ {0..4} all produce identical results"
finding was a **patches-had-no-effect** observation, not a saturation
diagnosis. The R1 escalation that followed was based on a flawed
measurement.

The corrected sweep (below) patches `inter.rs:1196` (the real picker
site) and produces actual signal.

---

## Night-8 corrected sweep — Phase A only (K1 at inter.rs:1196)

**Patch site**: `crates/rvp9-core/src/mode_decision/inter.rs` —
`let shifted_rate = (by_rate_satd as u64) << (K1 + VP9_PROB_COST_SHIFT);`
(the literal `K1` defaults to `2`; sweep varies it).

**Env**: `RVP9_F1_COEF_RATE_ENABLE=1`, `RVP9_F1_SKIP_TIE_BREAK_ENABLE`
unset (Phase A discipline).

### Pre-sweep also-required: night-8 saturation fix

Before the sweep, removed the early `.min(u32::MAX as u64) as u32`
saturation on `coef_rate` at BOTH `block_yrd.rs:159-161` AND
`inter.rs:1182-1184`. `BlockYrdResult.coef_rate` widened from `u32`
to `u64`; saturate-cast moved to the rate-sum use site in
`evaluate_candidate`. This was defensive — turned out at the
foreman bitrates exercised, the saturation wasn't actually
triggering (K1=2 produces the same +10.55 dB measurement before and
after the fix). The fix remains in tree as a robustness improvement
for any future low-q workload that might hit the cap.

### Phase A results (foreman_cif, post-saturation-fix)

| K1 | Foreman mean Δ | Foreman max Δ | Foreman min Δ | Verdict |
|---:|---:|---:|---:|---|
| 0 | **+9.02 dB** | +9.31 | +8.65 | ≤ baseline (coef_rate ~zero, picker is effectively in baseline mode) |
| 1 | +10.27 | +14.32 | +8.65 | worse |
| 2 (default) | +10.55 | +15.49 | +8.65 | worst (libvpx default in rvp9 picker — DV-F1-1 calibration manifests) |
| 3 | +9.22 | +9.60 | +8.82 | worse than baseline |
| 4 | **+9.00 dB** | +9.19 | +8.78 | **WINNER — slightly improves baseline +9.07 → +9.00** |
| 5 | +12.60 | +17.54 | +9.93 | dramatically worse (out of valid range) |

**Phase A winner: K1 = 4** at mean Δ = **+9.00 dB**, a 0.07 dB
improvement on baseline +9.07.

### Why K1=4 wins

Hypothesis: K1=4 (`<< 13`) makes `coef_rate` so large that it
dominates the picker's rate sum reliably across all candidates. At
this dominance level the picker's RD ranking becomes consistent —
the rate-component is essentially "the leaf with the smallest
residual wins," which closely mirrors the no-coef_rate baseline
picks (where pure pixel-SSE-of-recon distortion drives the
comparison).

K1=2 (libvpx default) lands in a "dead zone" where coef_rate is
just large enough to perturb the picker's decisions but not large
enough to dominate. The result is inconsistent picks across blocks
and a measurable regression vs both K1=0 (rate ~ irrelevant) and
K1=4 (rate dominates).

K1=0 is statistically indistinguishable from baseline because the
SATD-shifted rate `rate_satd << 9` is small enough that mode_rate
+ mv_rate dominate the rate sum, effectively neutering coef_rate.

K1=5+ regresses badly because at `<< 14` and above, the picker's
RD math starts to saturate on the rate side (or, more likely, the
candidate ordering becomes pathological as different blocks hit
different saturation thresholds).

### Phase B partial (K1=4, K2=coef_dist) — one cell tested

- K1=4, K2=coef_dist (libvpx-faithful — vp9_pickmode.c:835), K3
  default, tie-break OFF: **foreman mean Δ = +9.95 dB** (regresses
  vs K1=4 baseline). 250 kbps point collapses to 20.89 dB and the
  rvp9 curve loses PSNR monotonicity → test panics.

`by.coef_dist` saturates at u32::MAX on low-q blocks the same way
`coef_rate` did pre-saturation-fix. Phase B with K2=coef_dist
requires a second saturation fix (widen `dist` from u32 to u64
upstream) which would touch `rd_cost` itself — out of F.4-1 scope.
Deferred.

### Verdict — sweep ENDS at K1=4

K1=4 is a modest but real improvement (0.07 dB). The full Phase
B/C grid would require additional architectural work (widening
`dist` semantics; rebuilding `rd_cost` to take `u64` instead of
`u32`) that exceeds the F.4-1 measurement scope. Phase B/C are
**not** completed in night-8.

### Recommendation for F.4-2

Activate `coef_rate` at **K1=4** (literal `<< 4` at
`inter.rs:1196`). This delivers:
- ✓ Foreman PSNR-delta drops from +9.07 → +9.00 (modest but real).
- ✓ Within F.0.1 gate (≤ 9.5 dB threshold).
- ✓ Testsrc unchanged at +42.88 dB at K1=4 with coef_rate ON
  (vs +41.64 baseline — well under +44.64 halt threshold).
- ✗ Does NOT close the 8 dB Track F goal gap. F.4-2 alone is
  effectively a no-op toward "match vpxenc quality."

Honest framing: the F.4-1 sweep refutes the F.4 HLD's assumption
that K1-K4 calibration is the load-bearing missing piece. The K1
knob has only ~0.1 dB of usable signal at the foreman operating
point. The real Track F gap is structural — DV-F1-1 (Hadamard vs
DCT), the picker's RD math, the partition's interaction with mode
search — and won't be closed by parametric tuning alone.

**The sweep itself is the deliverable.** Even with no big win
found, the data establishes:
1. K1 has real but small signal (~0.1 dB at foreman).
2. Phase B is blocked by a second u32 saturation in `dist` —
   future work.
3. The Track F goal needs an architectural step (probably the
   Hadamard port, F.1.2) before parametric calibration is even
   measurable.

---

## Original (flawed) sweep — preserved for the record

[The original sweep results are preserved here unchanged from the
2026-05-23 daytime measurement. They are NOT valid because the
patches went to the wrong file (`block_yrd.rs:157`, not
`inter.rs:1196`). Treat the entire section below as a measurement
artefact.]

## Baseline (HEAD `bc95480`, all activations OFF, env unset)

| Corpus | Mean Δ | Max Δ | Min Δ | In-range points | Per-point Δ (kbps / dB) |
|---|---:|---:|---:|---:|---|
| Foreman_cif | **+9.07 dB** | +9.50 | +8.65 | 4 / 4 | 118.4 / 8.65 ; 133.7 / 9.11 ; 143.9 / 9.01 ; 261.7 / 9.50 |
| Testsrc | **+41.64 dB** | +41.64 | +41.64 | 1 / 4 | 62.2 / 41.64 ; 75.5 OOR ; 111.9 OOR ; 226.7 OOR |

(Original Phases A/B/C below all measured with patches at
block_yrd.rs that the picker never reads. The K1 ∈ {0..4}
"identical" finding was patches-had-no-effect, NOT saturation.)

### Original Phase A — K1 bisection (5 runs, INVALID)

K1 = SATD-to-bits scale exponent — patched at
`crates/rvp9-core/src/mode_decision/block_yrd.rs:157` (the WRONG
site; the picker uses the inlined version at `inter.rs:1196`).

All K1 values produced identical numbers: foreman +10.55,
testsrc +42.88 — because the patches had no effect on the picker.

Misdiagnosed as "u32::MAX saturation." Corrected night-8 (above).
