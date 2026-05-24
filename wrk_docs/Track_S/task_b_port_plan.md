# Task B — Port libvpx bitreader hot loop (Phase 1: hot/cold split)

**Date:** 2026-05-24
**Author:** Track-S team lead
**Status:** Design done, awaiting Task A bench-gate resolution before dispatching impl.

## Scope decision

Per the scoping report (Explore agent, 2026-05-24), recommended first-step is **hot/cold split alone**. Defer BMI2 unless this step underdelivers.

Rationale:
- R.15 (commit `4286269`) failed naïve `#[inline]` on `fill()` because the 50-LOC body bloated the per-bit hot loop's icache footprint by 23%.
- A correct hot/cold split lets the **fast path** inline into `read()` (small body, hot path) while the **slow path** (EOF byte-loop) stays out-of-line via `#[cold] #[inline(never)]`.
- Expected: 15-30% on coefficient-heavy fixtures (testsrc). Lower than the brief's 30-50% claim because we're not doing BMI2 yet, but it's the safer first move.
- BMI2 is a follow-up Task (B.2) if this delivers a measurable win and the gap is still material.

## The refactor (single function split)

Current `BoolReader::fill` (bool_reader.rs:195-235) has a natural if/else: fast path top, EOF slow path bottom. Split is:

```rust
#[inline]
pub fn fill(&mut self) {
    let bytes_left = self.buffer.len() - self.pos;
    let bits_left = (bytes_left as i32) * 8;
    let shift = BD_VALUE_SIZE - 8 - (self.count + 8);

    if bits_left > BD_VALUE_SIZE {
        // FAST PATH — inline-friendly, ~9 lines.
        let bits = (shift & !0x7) + 8;
        let bytes: [u8; 8] = self.buffer[self.pos..self.pos + 8]
            .try_into()
            .expect("bits_left > 64 guarantees 8 bytes available");
        let mut nv = u64::from_be_bytes(bytes);
        nv >>= BD_VALUE_SIZE - bits;
        self.count += bits;
        self.pos += (bits >> 3) as usize;
        self.value |= nv << (shift & 0x7);
    } else {
        self.fill_slow(bits_left, shift);
    }
}

#[cold]
#[inline(never)]
fn fill_slow(&mut self, bits_left: i32, mut shift: i32) {
    // SLOW PATH — EOF byte-loop, ~16 lines. Out-of-line.
    let bits_over = shift + 8 - bits_left;
    let mut loop_end = 0;
    if bits_over >= 0 {
        self.count += LOTS_OF_BITS;
        loop_end = bits_over;
    }
    if bits_over < 0 || bits_left > 0 {
        while shift >= loop_end {
            self.count += 8;
            self.value |= u64::from(self.buffer[self.pos]) << shift;
            self.pos += 1;
            shift -= 8;
            if self.pos >= self.buffer.len() {
                break;
            }
        }
    }
}
```

Key choices:
- `#[inline]` (not `#[inline(always)]`) on `fill` — explicit hint, but lets LLVM decide. Body is now ~9 lines so inlining is profitable.
- `#[cold] #[inline(never)]` on `fill_slow` — explicit "don't inline, it's the rare path".
- Pass `bits_left` and `shift` to `fill_slow` to avoid recomputation across the call boundary.

## Validation

- **Bit-exactness:** Conformance corpus 201/201; harness gates frame_parallel + tile_threading + profile1 all pass. Bool reader's existing 4 round-trip tests + the proptest in `tests/integration/...` if any.
- **No new clippy/test failures.**
- **Bench gate** (Task C MWU framework): testsrc_1080p30 t=4 ≥ 5% improvement at p < 0.05.

## Risks (per scoping)

1. **Split splits state wrong.** If the refactored `fill_slow` mutates `self.value` or `self.count` differently from the original, conformance breaks. Mitigation: literal cut-and-paste of the slow-path body; only the entry args change. Conformance 201/201 catches any bit-exact drift.
2. **`#[inline]` on fill regresses (same as R.15).** Possible if LLVM still chooses to inline the WHOLE fill including the call to fill_slow. Mitigation: `#[cold] #[inline(never)]` on fill_slow forces out-of-line — the call instruction stays small. If still regresses, drop `#[inline]` on fill and rely on LLVM's default heuristic.
3. **BD_VALUE_SIZE arithmetic gotcha.** The slow-path code computes `bits_over = shift + 8 - bits_left`. Type-check that the moved parameters preserve i32 width and signedness.
4. **No proptest exists for BoolReader.** Existing tests are 4 round-trip tests on bool_reader.rs:335-427. The real safety net is conformance corpus + the harness regressions. **Don't add new tests for this work** — proptest infrastructure would be its own task.

## What's OUT of scope for Task B Phase 1

- BMI2 intrinsics (BLSR / PEXT). Defer to Phase 2 if Phase 1 delivers.
- Adding `bmi2` flag to `dsp/caps.rs`. Same.
- Tweaking `read()` body (LZCNT already shipped in R.7; `from_be_bytes` in R.8).
- Adding proptest oracle. Separate task.

## Estimated effort

~30 LOC net change in `bool_reader.rs`. 1-2 small agent turns. Bench validation: ~30 min wall clock (baseline + candidate sweeps via Task C harness).
