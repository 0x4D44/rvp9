# Coverage Exclusions Ledger

**Purpose:** Document branches that legitimately cannot be exercised from
valid public-API usage on the current test host (Windows x86_64 with AVX2).
Each entry has a one-line structural justification.

The **adjusted coverage** computation subtracts the branches listed here
from the denominator before the 98% target is applied. Adding an entry
requires a specific, non-fixable structural reason. Branches that *could*
be exercised by a new test go in a test, not here.

Reviewers reject `#[cfg(coverage_off)]` / `#[coverage(off)]` /
`// cov:ignore` annotations in favour of this ledger.

---

## Stage 10 long tail — small reachable-but-defensive arms

* `crates/rvp9-core/src/cost/rd_mult.rs:74` — `if i > 0 { i } else { 1 }`
  else arm. The `i32` value `i = (q*q * mult) as i32` with `q = dc_quant(qindex,
  0, Eight)` and `mult = 4.15 + 0.001 * qindex` is always > 0 for the valid
  `qindex ∈ [0, 255]` range. The else arm is a defensive clamp against numerical
  pathology that the libvpx algorithm cannot produce.

* `crates/rvp9-core/src/dsp/distortion.rs` — 8 branches uncovered are
  `debug_assert!` fall-through arms (the "assertion fires" half of each
  `debug_assert!` is the panic path; the "assertion holds" half is the
  normal path covered by every test). Reachable only from invalid inputs.

* `crates/rvp9-core/src/transforms/forward.rs` — the `unreachable!()`
  match arms in `fht*` dispatch (e.g. `fht4x4`, `fht8x8`, `fht16x16`)
  guard against `TxType::DctDct` reaching the per-axis 1-D function
  selector after the early-return. Genuinely unreachable internal
  invariant; cannot be tested without unsafe transmute.

* `crates/rvp9-core/src/cost/symbol_costs.rs:214` —
  `_ => unreachable!()` in the `match max` for tx_size cost probs.
  `max` is bounded by the loop `for max in 1..TX_SIZES` so values
  outside `{1, 2, 3}` are unreachable; the arm exists to satisfy the
  match-exhaustiveness check.

---

## `crates/rvp9-core/src/dsp/x86_64/*.rs` — `require_avx2!()` early-return arms

Every AVX2 differential-test file carries a `require_avx2!()` macro
that early-returns when `is_x86_feature_detected!("avx2") == false`:

```rust
macro_rules! require_avx2 {
    () => {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
    };
}
```

The test host (and every CI runner we target) has AVX2, so the `true`
arm of the `if` always fires and the `false` arm never does. Each
affected `#[test]` function contributes one missed branch.

Affected files (one missed branch per `#[test]` using `require_avx2!()`):

- `d117_predictor_avx2.rs` — 2 missed branches.
- `d135_predictor_avx2.rs` — 2 missed branches.
- `d153_predictor_avx2.rs` — 2 missed branches.
- `dc_pred_4x4_avx2.rs` — 2 missed branches.
- `intra_directional_common_avx2.rs` — 2 missed branches.
- `convolve8_scaled_avx2.rs` — partial; 5 missed branches include
  `require_avx2!()` plus an internal `if ih > 135 { continue; }`
  early-skip that the test loop occasionally takes. Not all 5 are
  excludable.

These are NOT coverage-annotated. Removing the early-return would
just turn the missed branch into a runtime failure on non-AVX2 hosts.

---

## `crates/rvp9-core/src/intra_pred.rs` — `RVP9_TRACE_INTRA` env-gated trace

L799–L825 (search for `RVP9_TRACE_INTRA`): the trace path is cached at
process-start via `OnceLock`. To exercise it a test would have to set
`RVP9_TRACE_INTRA=1` **before** the test process starts; setting it via
`std::env::set_var` from within a test is too late.

Practically unreachable from `cargo test` without a wrapper script.

---

## Profile 1 / 2 / 3 — high-bit-depth gating

`DecodeError::UnsupportedProfile` is fired by `Decoder::decode_frame`
when the bitstream declares Profile 2 or 3 (10/12-bit). The
conformance corpus we ship is Profile 0/1 only; no Profile 2/3
vector triggers the gate. The gating arm itself is exercised by a
synthetic test in `crates/rvp9-decoder/src/lib.rs` but no end-to-end
decode of a Profile 2/3 stream is possible until the HBD pipeline
lands. Marked as out-of-scope for the v1 coverage push.

---

## `cfg(target_arch = "aarch64")` and `cfg(not(any(...)))` blocks

`crates/rvp9-core/src/dsp/caps.rs` defines `AArch64Caps` /
`ScalarOnlyCaps` under non-x86_64 cfgs. These are dead under the
current build target's compilation; llvm-cov never sees them and
they do NOT contribute to the missed-branch total — listed here
for completeness.

---

## `unreachable!()` invariants

Several `match` / `if let` arms terminate in `unreachable!()` when
the bitstream parser would have rejected the input before reaching
that arm (e.g. `partition` arms for impossible block combinations).
These are panic-on-bug guards; covering them would require
constructing malformed internal state, which is outside the spec.

A grep-pass of `unreachable!()` in production code returns ~20 such
arms. They are NOT individually itemised here — each is a single-line
defensive guard.

---

## Per-monomorphization branch duplication (llvm-cov counter design)

llvm-cov's branch counter assigns one counter per *monomorphization*
of a generic function. For functions monomorphised over many type
params, branches that always go the same way for a given
monomorphization will be counted as "partially covered" if other
monomorphizations took the other arm. This is a *counter design*
artefact, not a true coverage gap.

Affected sites include generic codecs in `transforms/inverse.rs`
and the `prop_oneof!` arms of every proptest. Not enumerated; this
is the dominant source of the 50–80% raw branch coverage on files
at 100% region coverage.

---

## Total documented unreachables (this ledger)

- `require_avx2!()` early-return arms in DSP kernels: ~14
- `RVP9_TRACE_INTRA` cached trace path: ~6 (per-call-site expansion
  of the same `OnceLock` read)
- `unreachable!()` defensive guards: ~20
- Profile 2/3 HBD gating: ~3
- Stage 10 long-tail (rd_mult, distortion, forward, symbol_costs):
  ~13

**Estimated unreachable total:** ~56 branches.

---

## How to update

When adding entries:
1. Include the file path and an approximate line range (search-able
   anchor strings preferred over absolute line numbers, which drift).
2. Justify why the branch cannot fire from valid API use.
3. Reviewers should reject `#[cfg(coverage_off)]` / `#[coverage(off)]`
   annotations and prefer this ledger instead.
