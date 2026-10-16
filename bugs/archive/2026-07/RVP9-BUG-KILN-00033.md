# RVP9-BUG-KILN-00033 - Conformance binary can exit successfully after skipped errors

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** harness/conformance-cli
- **Raised:** 2026-07-07
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-harness/` at `cfabd55`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T171626Z-p21311-n210115000-c1 branch=task/bug-RVP9-BUG-KILN-00033-run-fix-20260730T171626Z-p21311-n210115000-c1 code=a44fbd78933b2414138665059c857c7c597dd4f1 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5); residual split to RVP9-BUG-KILN-00063)

## Observation

The standalone conformance binary can exit 0 without proving all MD5-ledger frames decoded correctly.

`crates/rvp9-harness/src/bin/conformance.rs:111` handles `parse_packet` errors by incrementing `skipped`, advancing `shown_idx` at line `127`, and continuing. Non-`Unimplemented` decode errors are also counted as skipped at lines `198` and `209`. Skips only fail when `--strict` is set at line `319`.

The binary prints the expected MD5 row count at `crates/rvp9-harness/src/bin/conformance.rs:300`, but `run_one_vector` returns success at line `217` without checking that the decoded shown-frame count equals `md5_entries.len()`.

The integration test is narrower but still has the same parse-error gap: `crates/rvp9-harness/tests/conformance.rs:155` treats `parse_packet` errors as skipped frames. It does hard-fail non-`Unimplemented` decode errors at lines `225` and `435`.

## Fix

Open. Make parse errors and non-`Unimplemented` decode errors fatal in the conformance binary unless an explicit non-gating mode is requested, and assert shown-frame accounting against the MD5 ledger length. Mirror the parse-error accounting in the integration test so parse failures cannot disappear into skipped counts.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 harness review. Existing `RVP9-BUG-KILN-00032` covers `real_world.rs` and decoder conformance coverage losing oracle data; it does not cover the standalone harness conformance binary's exit contract.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

A real exit-contract change, not a log line. Parse errors and non-`Unimplemented` decode errors
now `return Err(...)` by default (`crates/rvp9-harness/src/bin/conformance.rs:118`, `:209`,
`:235`); `Unimplemented` alone still counts as a skip; `main` converts any `Err` into
`vector_failures += 1` and `vector_failures > 0` forces `exit(1)`. The shown-frame-vs-ledger
check fires even under `--non-gating`.
`tests::malformed_packet_is_fatal_by_default`, `::non_gating_mode_can_skip_a_malformed_packet`
and `::missing_shown_frame_is_fatal_even_in_non_gating_mode` all observed passing, as is the
hardened `conformance_runner_walks_all_vectors` -- the latter with real conformance vectors
fetched for this verification, so it decoded actual streams rather than short-circuiting.
Residual split to RVP9-BUG-KILN-00063: `tests/conformance.rs:66-71` still silently returns green
when the vectors dir is missing; the sibling commit added the `rvp9_conformance_vectors` cfg to
this very crate but did not apply it to this file.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
