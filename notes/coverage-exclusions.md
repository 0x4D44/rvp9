# Coverage Exclusions Ledger

**Purpose:** Document branches that legitimately cannot be exercised from
valid public-API usage. Each entry needs a one-line justification.

## Stage 10 long-tail

* `crates/rvp9-core/src/cost/rd_mult.rs:74` — `if i > 0 { i } else { 1 }`
  else arm. The `i32` value `i = (q*q * mult) as i32` with `q = dc_quant(qindex,
  0, Eight)` and `mult = 4.15 + 0.001 * qindex` is always > 0 for the valid
  `qindex ∈ [0, 255]` range. The else arm is a defensive clamp against numerical
  pathology that the libvpx algorithm cannot produce.

* `crates/rvp9-core/src/dsp/distortion.rs` — 8 branches uncovered are
  `debug_assert!` fall-through arms (the "assertion fires" half of each
  `debug_assert!` is the panic path; the "assertion holds" half is the
  normal path covered by every test). Reachable only from invalid inputs;
  not exercising these is by design.

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

## How to update

When adding entries:
1. Include the absolute file path and line range.
2. Justify why the branch cannot fire from valid API use.
3. Reviewers should reject `#[cfg(coverage_off)]` or `#[coverage(off)]`
   annotations and prefer this ledger instead.
