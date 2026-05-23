# E15 — BD-rate gate matched-bitrate-PSNR fallback

**Scope: harness/test only — zero encoder/decoder logic touched.**

After Track E (E1–E14), rvp9's PSNR curve (24–27 dB on foreman_cif at
the four operating points 25/50/100/250 kbps) sits entirely below
libvpx's (28–37 dB), so the Bjøntegaard cubic integral has no PSNR
overlap and `bdrate()` returns NaN. The gate had nothing useful to
report. Track F (`block_yrd` / `find_mv_refs` / motion-aware partition)
needs a number to drive down.

## What changed

1. `crates/rvp9-harness/src/bdrate.rs` — added
   `pub fn psnr_at_bitrate(curve, target_kbps) -> Option<f64>`. Piecewise
   linear interpolation in `ln(bitrate)` (same domain the Bjøntegaard
   cubic fit uses). Returns `None` when `target_kbps` is outside the
   curve's bitrate range — the fallback never extrapolates. 4 new unit
   tests cover endpoints, log-midpoint, out-of-range, and unsorted
   input. `cargo test -p rvp9-harness --lib bdrate` → 15/15 green.

2. `crates/rvp9-encoder/tests/bdrate_gate.rs` — `run_stage4a_gate` now
   has three branches:
   - `bd` finite → existing PASS/FAIL report (unchanged path).
   - `bd` NaN AND libvpx curve present → **fallback**: per rvp9 point,
     interpolate libvpx PSNR at that bitrate, report per-point Δ +
     `mean/max/min`. Skip out-of-range rvp9 points (no extrapolation).
   - `bd` NaN AND no libvpx curve → existing DEFERRED message.

Fires automatically when BD-rate is undefined; no env knob needed.

## Foreman fallback numbers (this run)

```
  rvp9 118.4 kbps / 24.01 dB  vs  libvpx-interp 32.65 dB  →  Δ = +8.65 dB
  rvp9 133.7 kbps / 24.10 dB  vs  libvpx-interp 33.21 dB  →  Δ = +9.11 dB
  rvp9 143.9 kbps / 24.53 dB  vs  libvpx-interp 33.54 dB  →  Δ = +9.01 dB
  rvp9 261.7 kbps / 26.96 dB  vs  libvpx-interp 36.46 dB  →  Δ = +9.50 dB
  Summary: mean Δ = +9.07 dB, max Δ = +9.50 dB, min Δ = +8.65 dB
```

Consistent with E9 / E13 matched-Q oracle finding of ~9 dB prediction
quality deficit. **Mean Δ = +9.07 dB is the metric for Track F to
drive down.** Gate condition for "fallback acceptable" is intentionally
OPEN — Track F target TBD.

## Testsrc behaviour

The well-formed BD-rate path is unchanged (verified by code inspection
of the `!bd.is_nan()` branch). On this run testsrc's BD-rate happened
to also be undefined (rvp9 produced HIGHER bitrates than libvpx on the
synthetic content, so the bitrate ranges don't overlap either way) —
the fallback correctly handled three out-of-range points and reported
the single in-range Δ.

## Validation

- `cargo test -p rvp9-harness --lib bdrate --release` → 15/15 pass.
- `cargo test -p rvp9-encoder --release` lib → 94/94 pass (cross-decode
  byte-exact gates intact — no encoder change, by construction).
- `cargo test -p rvp9-encoder --release --test bdrate_gate
  bdrate_gate_stage4a_realcontent -- --ignored --nocapture` → fallback
  numbers above; test passes.
- Clippy clean on `bdrate.rs` and `bdrate_gate.rs` (pre-existing
  unrelated clippy errors in `rvp9-core/dsp/x86_64/tm_predictor_avx2.rs`
  and `rvp9-core/loopfilter/apply.rs` on main were not touched per the
  hard constraint).
