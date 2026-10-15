# RVP9-BUG-CRU-00002 — fdct8x8 SSE2 extreme-residual tests are vacuous: every extreme case takes the scalar fallback

- **State:** Fixed
- **Priority:** Should
- **Severity:** Medium
- **Area:** core/transforms forward_sse2
- **Raised:** 2026-08-23T17:09:45Z
- **Discovery source:** Agent
- **Owner:** -
- **Owner role:** -
- **Owner run:** -
- **Owner host:** -
- **Owner branch:** -
- **Owner base:** -
- **Owner fingerprint:** -
- **Owner since:** -
- **Owner until:** -
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-23T17:09:45Z, raised via `deltic bugs new` model=claude-opus-5@high) -> Fixed (2026-10-15T23:35:11Z, deltic:auto role=fix run=fix-20261015T232621Z-604f2624 branch=task/bug-RVP9-BUG-CRU-00002-run-fix-20261015T232621Z-604f2624 code=485d2e829f25e5d6221c3cc0a0e54b8a9b567005 gate=manual)

## Observation

`fdct8x8_sse2_matches_scalar_valid_residuals`
(`crates/rvp9-core/src/transforms/forward_sse2.rs:295-348`) has four groups of cases. The
three "extreme" groups — constant residual (`:318-325`), axis-alternating extremes
(`:327-337`), and per-index impulse (`:339-347`) — all drive `pass1_is_safe`
(`crates/rvp9-core/src/transforms/forward_sse2.rs:102-110`) to `false`, so
`fdct8x8_inner` returns early at `:127-129` and `fdct8x8` calls `super::fdct8x8_scalar`
at `:154-156`. The test then asserts `vector == scalar` where `vector` **is** the scalar
output. Those assertions cannot fail regardless of what `fdct8x8_inner` computes.

Derivation (closed form, no execution needed). The guard rejects any first-pass output
outside `[-4096, 4095]`. For a constant residual of magnitude `v`, inputs are `4v` after the
`_mm_slli_epi16(_, 2)` at `:117`, so `x0 + x1` at `:85` is `32v` and
`output[0] = round(32v * COSPI_16_64 / 2^14) = round(22.62 * v)`. The guard therefore fires
whenever `|v| > 181`:
- constant `±255` → `|output[0]| = 5771 > 4095` → fallback;
- axis-alternating `±255`, rows alternating → `output[7] = -5228` → fallback;
- axis-alternating `±255`, columns alternating → `|output[0]| = 5771` → fallback;
- impulse (63 lanes at `-255`, one at `+255`) → every non-impulse column is constant
  `-255` → `|output[0]| = 5770` → fallback.

**Expected:** the extreme cases exercise the SSE2 kernel at the boundary of its valid range.
**Actual:** they exercise only the scalar fallback wiring. The SSE2 arithmetic is covered
solely by the 512 random cases at `:296-317`, whose residuals are uniform in `[-255, 255]`;
their per-column means are near zero, so they sit far from the `±4095` guard boundary and
only cross it by chance at roughly 3.5 sigma.

Consequence: `fdct8x8_inner`'s behaviour just below the guard — where `x0 + x1` reaches
`8 * 4095 = 32760`, within 7 of `i16::MAX` — is untested. That is precisely where an i16
wrap would first appear, and where the saturating `_mm_packs_epi32` in `mul_round16`
(`:23`) diverges from the scalar oracle's truncating `wraplow`
(`crates/rvp9-core/src/transforms/constants.rs:183-185`). The guard bound itself is correct
(verified: `8 * 4095 = 32760 < 32767`, while a symmetric `±4096` bound would overflow at
`8 * 4096 = 32768`) — the defect is that nothing tests the code the bound protects.

Independently found by two reviewers in the 2026-08-23 review pass.

**Suggested fix:** add cases engineered to land just inside the guard — e.g. a constant
residual of `±180` (`|output[0]| = 4072`) and a column-mean pattern tuned so the largest
first-pass output is in `4000..=4095` — and assert the SSE2 path actually ran (expose
`pass1_is_safe`'s verdict under `#[cfg(test)]`, or assert `fdct8x8_inner` returned `true`)
so the case cannot silently degrade into a scalar-vs-scalar comparison again. Add a matching
just-outside case that asserts the fallback *is* taken.

## Fix

<unfixed — raised only>

## Notes
