# Track T — T-3 baseline ledger: `BoolReader::read` ns/call

**Slice scope:** measurement only. Per the T-3 HLD §8 framing (carried in the
implementation brief, the HLD file itself was not present in this worktree at
authoring time and is referenced through the brief), this slice consciously
does NOT modify `BoolReader::read`, `fill`, or `fill_slow`. It establishes
the noise-floor baseline against which any future Phase-2 attempt (BMI2 /
inline-asm renormaliser) must clear before shipping.

If the future attempt cannot beat this baseline by a margin that is large
relative to the run-to-run dispersion documented below, the HLD's contrarian
recommendation to defer the change stands.

## Hardware anchor

- **CPU:** Intel(R) Core(TM) Ultra 9 285HX (Arrow Lake-HX)
- **OS / shell:** Windows 11 Pro 26200, PowerShell 7
- **Toolchain:** workspace `rust-toolchain.toml`, `cargo build --release`
  (release + debuginfo per workspace default)
- **Detected `X86Caps`** (from the microbench's first stdout line):

  ```
  X86Caps { sse2=true, ssse3=true, sse4_1=true, avx2=true, bmi2=true }
  ```

  All five capability bits are present, so a future BMI2-using variant would
  in principle dispatch on this host. (The BMI2 cap is plumbed in this same
  slice — see `crates/rvp9-core/src/dsp/caps.rs:36-58` — but no kernel
  currently selects on it.)

## Methodology

Driven by `crates/rvp9-bench/src/bin/dsp-microbench.rs` under the
`bool_reader` subcommand. Reproduce:

```
cargo run -p rvp9-bench --release --bin dsp-microbench -- bool_reader
```

- **Source corpus:** 24 Mbit of deterministic `(bit, prob)` pairs generated
  by two seeded LCG streams (`bit_seed = 0xC0FFEE01`,
  `prob_seed = 0xDEADBEEF`). Probabilities are drawn from a 100-slot fixed
  table containing 50 × p=128, 30 × p=32, 20 × p=224 — mirroring the
  coefficient-decode prob distribution noted in the brief.
- **Encoded stream:** the `(bit, prob)` pairs are pushed through
  `BoolWriter`. The resulting encoded slice is ~2.55 MB (we over-shoot the
  1 MB target so the decoder never crosses EOF mid-batch — that would invoke
  the cold `fill_slow` path and skew the headline).
- **Per batch:** fresh `BoolReader::new(encoded)` (so the cache state is
  consistent across batches), read 1,000,000 bits using the matching prob
  sequence, time only the read loop.
- **Per run:** 7 batches, take the `min` ns/call (most stable estimator for
  a CPU-bound microbench — least perturbed by scheduler interrupts /
  frequency dips).
- **Headline:** median of 5 independent runs; range = min/max of the 5
  per-run minima.
- All loads pass through `black_box(p)` so the optimiser cannot hoist the
  prob lookup or specialise the call.

## Results (5 independent runs)

| Run | Min ns/call (per-run best across 7 batches) |
| --- | -------------------------------------------- |
| 1   | 4.33                                         |
| 2   | 4.47                                         |
| 3   | 4.45                                         |
| 4   | 4.34                                         |
| 5   | 4.46                                         |

- **Median (headline):** **4.45 ns/call**
- **Min across runs:** 4.33 ns/call
- **Max across runs:** 4.47 ns/call
- **Cross-run spread:** 0.14 ns (~3.1% of the median)
- **Per-batch absolute min observed:** 4.01 ns (run 1)
- **Per-batch absolute max observed:** 4.55 ns (run 3)

The intra-run min/max envelope (4.01 → 4.55) bounds what a single
optimistic point reading can swing — about ±6% off the cross-run median.
Any Phase-2 attempt should clear this ~6% noise band before its delta is
considered evidence rather than chance.

## Throughput context

At 4.45 ns/call, `BoolReader::read` decodes roughly **225 Mbits/s**
single-thread on this host (1 bit per `read` call). That is the per-thread
ceiling for symbol decode — every coefficient, every mode bit, every MV
component runs through this function.

Per-frame attribution is left to wall-clock benches (`decode-throughput`,
`decode-headtohead`). The brief allows omitting that attribution here for
clarity; the microbench is the relevant oracle for any `read`-only change.

## Notes / caveats

- **This is a relative ruler, not a statistical benchmark.** No
  confidence intervals; the 5×7 sample design just bounds dispersion well
  enough to spot a real win.
- **Same host required for any comparison.** This baseline is meaningful
  only on the Arrow Lake-HX dev box above. A future-day comparison run on
  the same host can use this ledger directly; comparisons across hosts
  must re-baseline.
- **Background load matters.** Numbers above were taken on an idle dev
  box; concurrent worker agents in other worktrees can move the median by
  >10%. Re-baseline before pre/post comparing.
- **No `read` modification was made.** The hot loop is byte-identical to
  the post-R.8 + post-Track-S Phase-1 state. If a future agent does
  modify `read` / `fill`, they must re-baseline (the ledger entry should
  cite the new headline + commit SHA).

## Decision criterion for Phase 2

A `BoolReader::read` change should ship only if **all** of these hold:

1. Median ns/call on the same host drops by ≥10% (clear of the ~6%
   intra-run noise band with ≥4% safety margin).
2. The improvement holds across at least 5 independent re-runs (same
   protocol as above).
3. Conformance (`cargo test -p rvp9-harness --test conformance --release`)
   stays at 201/201.
4. `decode-throughput` on the canonical fixture (testsrc_1080p30 t=4)
   shows a directionally consistent improvement at p50 wall-clock —
   even if the microbench wins, a flat or negative wall-clock indicates
   the speedup is below the higher-level noise floor and therefore not
   worth the maintenance cost.

If 1-2 hold but 4 fails, **do not ship** (per the HLD's stance that
bench-noise floor is structural at this level of the decoder).

## Companion change in this slice

- `crates/rvp9-core/src/dsp/caps.rs:36-58` — `X86Caps.bmi2` field added
  (detected via `std::is_x86_feature_detected!("bmi2")`). No dispatch
  consumer yet.
- `crates/rvp9-core/src/dsp/caps.rs:115-130` — `x86_bmi2_field_is_stable`
  sanity test (asserts detection is deterministic; does not assert
  presence, since pre-Haswell x86_64 lacks BMI2).
- `crates/rvp9-bench/src/bin/dsp-microbench.rs` — new `bool_reader`
  subcommand. Default-no-arg behaviour (the existing DSP suite) is
  preserved.
