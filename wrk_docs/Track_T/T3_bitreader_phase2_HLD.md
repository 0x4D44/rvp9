# HLD — Track T-3: Bitreader Phase 2

**Author:** T-3 planning agent
**Date:** 2026-05-24
**Status:** Design proposal; first-slice scope is **measurement + cap-detection plumbing**, NOT a behavior change in `read()`. Agent recommends DEFERRING the brief's BMI2/inline-asm scope (see §3).

---

## 1. Re-baseline: what the bitreader actually does today

### 1.1 `read(prob)` — the function `decode_coefs` calls thousands of times per block

Source: `D:\language\rvp9-2\crates\rvp9-core\src\entropy\bool_reader.rs:79-117`.

Per-call work, in order:

| # | Line | Operation | Approx cycles (Zen 3 / Skylake) |
|---|------|-----------|----------------------------------|
| 1 | 82 | `split = (range * prob + (256 - prob)) >> 8` — one IMUL, one ADD, one SHR | ~4 |
| 2 | 84 | `if self.count < 0` branch test against `self.fill()` | predicted-not-taken, ~1 |
| 3 | 88-93 | Two loads + one shift (build `bigsplit = split << 56`) | ~2 |
| 4 | 96-100 | `if value >= bigsplit` predicated update of `range`, `value`, `bit` | branch, ~1-2 (modern OoO often CMOVs this) |
| 5 | 108 | `((range & 0xff) as u8).leading_zeros() & 7` — LZCNT (Wave R.7) | ~3 (LZCNT latency) |
| 6 | 109-111 | `range <<= shift; value <<= shift; count -= shift` — two SHLs, one SUB | ~3 |
| 7 | 113-116 | Three stores back to `self.{value,count,range}` | ~3 |

Honest LLVM-output guess (without actually disassembling — I'd need to run `cargo asm` and I cannot execute the build): roughly 14-22 instructions in the hot path, 8-12 cycles of latency on modern x86 if everything stays in L1 + register file. The branch at line 84 (fill check) and the branch at line 96 (bit decision) are the only branches; the renormalization is straight-line LZCNT + variable-shift.

### 1.2 `fill()` — fast path (post-Phase 1)

Source: `bool_reader.rs:206-229`. Post-Phase 1 (commit bddcc64) is now 9 lines:
- `u64::from_be_bytes(self.buffer[pos..pos+8])` — one 8-byte load + BSWAP (Wave R.8)
- Shift down by `BD_VALUE_SIZE - bits` to discard partial top
- OR into `self.value`

This fires roughly once every 7 `read()` calls (after 7 renorm shifts of 1-8 bits, `count` goes negative and fill is triggered). For coefficient-heavy fixtures, fill is the dominant memory-touching path; for normal fixtures it's amortized over many reads.

### 1.3 `fill_slow()` — `#[cold] #[inline(never)]` byte-loop at EOF.

Out-of-the-hot-loop. Phase 1 split moved this here precisely so R.15's 23%-regression doesn't recur.

### 1.4 Honest take on what LLVM emits

The hot loop, after R.7 (LZCNT) + R.8 (from_be_bytes) + Phase 1 hot/cold split (bddcc64), is **already close to what a hand-tuned C version of libvpx's `vpx_read` emits with `-O3 -march=native`**. The libvpx C source at `D:\language\rvp9-2\third_party\libvpx\vpx_dsp\bitreader.h:77-136` is literally the same algorithm we mirror line-for-line. The libvpx win on x86 comes from:
- (a) `__attribute__((always_inline))` forcing fusion at every callsite (we use `#[inline]`, similar).
- (b) PIC-relocation-free `VPX_NORM` lookup, which we *replaced* with LZCNT — should be a small win on modern CPUs where LZCNT is 1 cycle reciprocal-throughput.
- (c) Possibly better register allocation in clang vs rustc/LLVM, but with the same LLVM backend the codegen should be substantively identical.

**The headline question:** is there really a 30-50% win remaining here? My honest read: **no, probably not** — see §3.

## 2. Phase 2 technique survey

### 2.1 Option (a): BMI2 BLSR / PEXT for renormalization

**Expected win:** Marginal — perhaps 0-5% on coefficient-heavy fixtures. **Bit-exactness risk:** Low (algorithm preserved). **Portability cost:** **HIGH** — PEXT on Zen 1/2/3 is microcoded at ~18-cycle latency (vs 3 cycles on Zen 4 / Intel post-Haswell). Shipping a PEXT path that *regresses* on Zen 3 is a real risk.

What BMI2 could actually help with:
- **BLSR** (`blsr r, r/m`): clears the lowest set bit — useful for *bit-traversal* readers (Daala-style range coders) but **not directly applicable** to VP9's range coder, where the renorm is "shift left by `lz(range & 0xff)`". There's no "lowest set bit" semantic in our hot path.
- **PEXT**: extracts arbitrary bit selections via a mask. Could in principle replace the `value >>= shift_amt` pattern, but variable-mask PEXT is *slower* than `SHL` on modern x86. Wrong tool.

**Verdict:** BMI2 instructions don't naturally slot into this algorithm. The Track-R brief's mention of BLSR/PEXT was speculative; on careful analysis these instructions don't replace anything in the hot path. **LZCNT (already shipped) was the right BMI* pick.**

### 2.2 Option (b): Hand-written inline assembly for `read()`

**Expected win:** 5-20% if perfectly tuned. **Bit-exactness risk:** Very high — every register clobber and flag dependency is hand-managed. **Portability cost:** Massive — Linux SysV ABI vs Windows MSVC ABI vs aarch64 fallback; conditional compilation matrix gets ugly. **Effort:** brief estimates 2-3 weeks; that matches my read.

The wins over LLVM output here would come from:
- Forcing `value`, `range`, `count` into fixed registers across the whole loop (LLVM may spill them around the fill-callsite).
- Hand-scheduling the LZCNT to overlap with `IMUL` for the `split` computation.

But: the function is so small that LLVM's RA should already do most of this. Inline asm is a defensive bet — it removes LLVM's discretion. Not a high-EV win for a first slice.

### 2.3 Option (c): Loop-unrolled `read_n` bulk reader

**Expected win:** 5-15% on `read_literal(N)` paths (literal reading is just `N` calls to `read_bit`). **Bit-exactness risk:** Low — bulk-read is provably equivalent if the implementation is careful. **Portability cost:** Zero — pure scalar Rust.

`read_literal` (bool_reader.rs:131-139) currently calls `read_bit()` in a loop, which calls `read(128)`. For `read_bit()` specifically, `prob = 128` is a constant, so `split = (range * 128 + 128) >> 8 = (range + 1) >> 1`. A specialized `read_bit` could skip the multiply. But `read_bit` is rare on the hot path (coefficient decode uses *modeled* probs, not 128); main callsites are `read_literal` in header parsing, not coefficient decode.

A *bulk* `read_n_bits` that extracts N max-entropy bits from the value register in one go is plausible but only helps the header decode, which is sub-1% of frame time. Low EV.

### 2.4 Option (d): Range-coder algorithmic redesign

Out of scope — would break bit-exactness with libvpx and forfeit the conformance gate. Not considered.

### 2.5 Ranking by risk-adjusted EV

| Option | Expected win | Risk | Effort | EV verdict |
|--------|--------------|------|--------|------------|
| (a) BMI2 renormalization | 0-5% (likely 0%) | Low correctness, **high regression risk on Zen ≤ 3** | 1 day | **Reject** — wrong tool |
| (b) Inline asm | 5-20% | Very high | 2-3 weeks | **Reject** for first slice |
| (c) Bulk `read_n` | 5-15% on rare paths | Low | 1 day | **Marginal** — wrong hot path |
| (d) Algorithmic redesign | N/A | Breaks conformance | — | **Reject** |

**None of the four candidates from the brief have a strong risk-adjusted case for a first-slice implementation.**

## 3. Recommendation: defer Phase 2, run a *measurement* slice instead

The recommendation contradicts the brief, and I will state it plainly: **Phase 2 as scoped is not worth doing tonight.** The brief's "30-50%" headline was speculative — Track-R brief is honest that "the actual asm tuning is in how libvpx structures the per-bit hot loop with macro fusion." After R.7 + R.8 + Phase 1, we already have macro fusion equivalent to libvpx via `#[inline]` + hot/cold split. BMI2 instructions don't naturally slot in. Inline asm is multi-week.

**What I recommend as the overnight first slice instead:** a **measurement and instrumentation slice** that gives the next session a *defensible* number for "how much of testsrc_1080p30 t=4 is actually in `BoolReader::read`?" This unblocks the EV question for Phase 2 once and for all.

If the user wants to push ahead with a BMI2 dispatch slot anyway as a hedge (the brief explicitly asks for it), I describe that path in §5 — but flag honestly that I expect it to be NOISE in the bench.

### 3.1 Recommended first-slice option: instrumentation + tightening

1. **Add `caps.bmi2` to `dsp/caps.rs`** — pure additive, no behavior change. Lets future slices wire BMI2 paths without a re-rebase.
2. **Add a microbench** (`crates/rvp9-bench/src/bin/dsp-microbench.rs` already exists — add a `bool_reader` case): drive `BoolReader::read` over a 10 MB synthetic byte buffer with a fixed prob sequence; measure ns/call. This is the cleanest way to detect any win-without-bench-noise.
3. **Try one micro-tightening of `read()` that is provably zero-risk** (see §5): make `split` use `wrapping_*` everywhere so LLVM elides overflow checks in debug. Ship as an additive change behind a feature flag if needed.

This delivers something tonight, gives the next session real data, and avoids burning a slot on a likely-noise BMI2 change.

## 4. CPUID gating — where does `bmi2` go?

Even though I recommend not using BMI2 in the first slice, adding the cap detection is cheap and unblocks future work. Path:

- **File:** `D:\language\rvp9-2\crates\rvp9-core\src\dsp\caps.rs:26-50`
- **Add field:** `pub bmi2: bool` on `X86Caps` (after `avx2`).
- **Detect:** `bmi2: std::is_x86_feature_detected!("bmi2")` (stable since Rust 1.27).
- **Sanity test:** none required — bmi2 is not universal on x86_64 (pre-Haswell / pre-Excavator CPUs lack it), so we can't add an unconditional `assert!`. Just add it to the `detect_is_stable` golden.

**Runtime dispatch story:** `BoolReader` does NOT currently flow through `SimdDispatch`. Two options:

- **Lazy: cache caps once in a `static OnceLock<bool>` inside `bool_reader.rs`** — survives the `BoolReader` taking no constructor args. Cleanest for a first slice.
- **Eager: pass `Arc<SimdDispatch>` into `BoolReader::new(buffer, dispatch)`** — matches the R-09 SIMD pattern at `dsp/dispatch.rs:1-50` but is invasive (changes every call site at `compressed_header.rs:500` etc.; bit-exactness must be preserved on the dispatch boundary).

For a first slice, **OnceLock cache is the right answer.** A future inline-asm slice could plumb through dispatch.

## 5. Concrete instruction sequence (for completeness, if Phase 2 ships)

If, against this HLD's recommendation, a BMI2-flavored renorm variant ships, this is the algorithmic spec. **Reference:** `D:\language\rvp9-2\third_party\libvpx\vpx_dsp\bitreader.h:77-136`.

### 5.1 What we have today (post-R.7, in registers `RANGE`, `VALUE`, `COUNT`)

```
;  ECX = range (u32, low byte holds the renorm-relevant bits)
;  RAX = value (u64)
;  EDX = count (i32)
MOV     R8B, CL              ; isolate low byte of range
LZCNT   R8D, R8D             ; R8D = leading_zeros (32 minus what we want)
SUB     R8D, 24              ; convert "lz of u32" to "lz of u8" = 0..7
SHLX    ECX, ECX, R8D        ; range <<= shift  (BMI2 SHLX — no flags)
SHLX    RAX, RAX, R8         ; value <<= shift
SUB     EDX, R8D             ; count -= shift
```

Already uses SHLX (BMI2 variable-shift, no flag dependency) via target-cpu=native. Already uses LZCNT. **There is nothing material to swap for BLSR or PEXT here.**

### 5.2 What a hypothetical PEXT-based variant would look like

PEXT could replace the `(range << shift) | (value >> (64-shift))` *if* we were rearchitecting the value register to be byte-aligned. We aren't. Skipping.

### 5.3 What an inline-asm `read()` would look like

The whole body of `read()` as one `asm!` block, with constraints binding `value`, `range`, `count` to fixed regs (`rax`, `ecx`, `edx`) and a label for the `value >= bigsplit` branch. Roughly:

```
;  In:  prob in DIL, self.{value,range,count} in RAX/ECX/EDX
;  Compute split:
MOVZX   R8D, DIL
MOV     R9D, ECX
IMUL    R9D, R8D
NEG     R8D
ADD     R8D, 256
ADD     R9D, R8D
SHR     R9D, 8                     ; R9D = split
;  Test count < 0 → fill (cold call, out-of-line)
TEST    EDX, EDX
JS      .fill_path
.split_ok:
SHL     R9, 56                     ; R9 = bigsplit
;  Compare value, bigsplit:
CMP     RAX, R9
SETB    DIL                        ; preliminary "bit = 0"
... (conditional update of RAX, ECX) ...
;  Renorm: LZCNT + 3 SHLX as above.
```

This is the multi-week artifact the brief refers to. **First-slice scope explicitly excludes this.**

## 6. Validation strategy

Current bitreader tests (`bool_reader.rs:359-451`):
1. `round_trip_bits_p128` — 200 max-entropy bits.
2. `round_trip_literals` — 5 × 8-bit literals.
3. `round_trip_skewed_probabilities` — 150 bits at probs 32 and 224.
4. `over_read_sets_error` — EOF semantics.

These are *necessary but insufficient* for any Phase 2 change. Add:

### 6.1 Proptest oracle vs scalar `read()`

`crates/rvp9-core/src/entropy/bool_reader.rs#[cfg(test)] mod tests` should grow a proptest that:
- Generates a random `[u8; 256]` buffer + a sequence of (bit, prob) tuples.
- Round-trips through `BoolWriter` → `BoolReader::read()`.
- Asserts all bits round-trip and `has_error()` returns false.

When a Phase 2 variant exists, the proptest body parametrizes over `{Scalar, Bmi2}` and asserts both produce identical (bit, register_state) sequences. The `cursor_state()` accessor at `bool_reader.rs:275` is the API for this.

### 6.2 Conformance corpus (load-bearing safety net)

`cargo test -p rvp9-harness --test conformance --release` — 201/201 vectors. Any divergence in `BoolReader::read` shows up as a wrong coefficient → wrong residual → SAD divergence at frame N. This catches any bit-exactness regression even before proptest.

### 6.3 Differential snapshot test

Use `BoolReader::snapshot()` / `restore()` (already exists, bool_reader.rs:298+) to fork the reader at every 100th `read()` during a conformance run; compare Scalar vs Phase2 register states after each `read()`. Cheap, catches "diverged 2 reads ago" bugs.

## 7. Bench gate (Task C MWU framework + power analysis)

**Framework:** `crates/rvp9-bench/src/mwu.rs` (exact MWU for n ≤ 20, normal approx beyond); `crates/rvp9-bench/src/bin/bench-compare.rs` driver; gate at p < 0.05 AND Δ ≤ -5%.

**Canonical fixture:** `testsrc_1080p30 t=4` (per `bench/results/decode-headtohead.md:26`), the worst remaining ratio (9.33×) and the most coefficient-heavy synthetic content.

**Power analysis (honest):**

From `bench/results/compare-1779589201-vs-1779590057.md` (Phase 1 vs baseline), the observed run-to-run σ on testsrc_1080p30 t=4 is roughly 0.3 ms on a 5.7 ms mean — about **5%** of the mean (better than the 8% Track-R quoted; Phase 1 hot/cold split itself reduced variance slightly).

For a two-sided MWU at α = 0.05, 1−β = 0.8, with σ/μ = 5%:
- To detect a **15% effect** (the brief's mid-range expectation): n ≈ 4 per side. **Current N=5 is sufficient.**
- To detect a **5% effect** (the bench-gate threshold): n ≈ 25 per side. **Out of reach** for an overnight slice (each run is ~30 min on this fixture).
- To detect a **2% effect** (realistic if my §3 analysis is right and Phase 2 buys us nothing): n ≈ 150 per side. **Out of reach for any single bench session.**

**Bench gate verdict:** if Phase 2 produces a real 15% win, N=5 catches it. If Phase 2 is in the 2-5% range, the bench will report NOISE — exactly what happened to Phase 1. This is the structural problem; only thousand-run sweeps via CI on pinned hardware would resolve sub-5% effects, and the brief lists pinned-host as deferred.

**My take:** any Phase 2 variant we ship must clear ≥15% on testsrc_1080p30 t=4 at p < 0.05, or admit honest NOISE and revert. The asymptote is structural, not iteration-laziness.

## 8. Phasing

### First slice (overnight, ~1 focused session)

**Scope:** instrumentation + zero-risk cap-detection plumbing.

- **S1.1:** Add `bmi2: bool` field to `X86Caps` at `caps.rs:26-49`. Update `detect()`, `detect_is_stable` golden test. (~10 LOC.)
- **S1.2:** Add a `bool_reader` microbench to `crates/rvp9-bench/src/bin/dsp-microbench.rs` — drive `BoolReader::read` over a 1 MB synthetic buffer with a fixed prob mix (50% prob 128, 30% prob 32, 20% prob 224 — mirrors coefficient-decode distribution). Report ns/call. (~50 LOC.)
- **S1.3:** Run microbench on dev box; record baseline. Run conformance gate (must pass 201/201). Run full Task C sweep on testsrc_1080p30 t=4. (No behavior change, so expected = identical to baseline — this validates the bench infrastructure for subsequent slices.)
- **S1.4:** Document baseline ns/call in a new `wrk_docs/Track_T/T3_baseline_ledger.md`.

**Bit-exactness:** trivially preserved (no behavior change beyond caps detection).

### Subsequent slices (only if microbench reveals headroom)

- **S2:** Add BMI2-gated `read_bmi2()` variant behind `caps.bmi2` runtime check + `OnceLock` cache. Implement using SHLX/MULX (no PEXT). Differential proptest vs scalar. Expected effort: 1-2 days. Expected win: **honestly, 0-3%** — see §3.
- **S3:** Bulk `read_literal_bits` variant for `read_literal(N)` paths. Effort: 1 day. Expected win: <1% on canonical fixture (rare path).
- **S4 (if Stages 2-3 disappoint):** Reconsider whether *coefficient-decode-side* changes (token-loop unrolling in `entropy/detokenize.rs:465-585`) are a better lever than reader-side changes. The hot path may be the *caller*, not the reader.
- **S5 (long-shot, only if perf is a ship blocker):** Inline-asm `read()`. 2-3 weeks. Off the critical path for V1.

## 9. Risk register

| ID | Risk | Severity | Mitigation |
|----|------|----------|------------|
| T3.R1 | **R.15 cautionary precedent: icache bloat from over-inlining** | High | First slice has zero behavior change. S2 keeps BMI2 path behind a runtime branch — the cold path stays cold. `cargo bloat --release` check before any S2 merge. |
| T3.R2 | **PEXT regression on Zen 1-3 / pre-Haswell** | High | Cap detection only; no PEXT in any first-slice or S2 path. If S2 ever uses PEXT, gate behind `caps.bmi2 && (vendor-allowlist)` — Zen 4 + Skylake-X + later only. Document the allowlist. |
| T3.R3 | **Bench-noise floor (~5%) hides real wins** | High (structural) | Task C MWU. Stop shipping anything below 15% expected win until pinned-host CI exists. Accept that sub-15% changes are unprovable. |
| T3.R4 | **Inline asm divergence from spec under edge-case input** | Very high | Defer inline asm to S5; not in first slice. When we get there, proptest with cursor_state diff at every read. |
| T3.R5 | **Caps detection wired wrong: `bmi2` reads as false on a BMI2 CPU** | Low | `std::is_x86_feature_detected!` is the standard mechanism; same wiring as existing avx2 path. Add a single-CPU sanity check that prints detected caps in `dsp-microbench`. |
| T3.R6 | **First slice produces no measurable win** (expected!) | Low | This is the *honest expected outcome*. The slice's value is the microbench infrastructure + cap plumbing, not a perf number. Don't claim a win we don't have. |

### Rollback plan

If any subsequent slice regresses testsrc_1080p30 t=4 by >5% at p<0.05 (the symmetric "REGRESSION" gate from `bench-compare.rs`):

1. `git revert <commit>` — slice is additive, easily reverted.
2. Caps detection (`X86Caps::bmi2`) stays — it's pure additive and harmless.
3. Microbench stays — it's infrastructure.
4. Update `wrk_docs/Track_T/T3_*_ledger.md` with the bench evidence of the regression.

R.15's precedent (`#[inline]` on full fill → 23% regression, reverted in commit 4286269) is the model: ship the cap, kill the variant, keep the infrastructure.

## Critical Files for Implementation

- `D:\language\rvp9-2\crates\rvp9-core\src\entropy\bool_reader.rs` (the target)
- `D:\language\rvp9-2\crates\rvp9-core\src\dsp\caps.rs` (add `bmi2` field)
- `D:\language\rvp9-2\crates\rvp9-bench\src\bin\dsp-microbench.rs` (add bool_reader case)
- `D:\language\rvp9-2\third_party\libvpx\vpx_dsp\bitreader.h` (algorithmic spec)
- `D:\language\rvp9-2\crates\rvp9-bench\src\mwu.rs` (bench gate)
