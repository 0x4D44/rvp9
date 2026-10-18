# RVP9-BUG-FLU-00055 — Public API rustdoc emits broken-link and private-item warnings

- **State:** Closed
- **Priority:** Could
- **Severity:** Low
- **Area:** docs/public-api
- **Raised:** 2026-10-17T21:51:55Z
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
- **State history:** Open (2026-10-17T21:51:55Z, raised via `deltic bugs new --land`) -> Fixed (2026-10-18T20:24:37Z, deltic:auto role=fix run=fix-20261018T201724Z-881b3f01 branch=task/bug-RVP9-BUG-FLU-00055-run-fix-20261018T201724Z-881b3f01 code=768edb747ac3ca1edb936c1d820cd04543a3518d gate=manual) -> Closed (2026-10-18T21:49:57Z, deltic:manual role=verify run=verify-20261018T214434Z-cc7cff10 verified_fix_run=fix-20261018T201724Z-881b3f01 verdict=close)

## Observation

`cargo doc --workspace --no-deps --locked --offline` on macOS arm64 at commit `e5bb4335742bcf4327d2b64a4fcd2f4c90486268` (rustc/cargo 1.95.0) exited 0 and generated workspace HTML, but rustdoc emitted 143 warning diagnostics: 140 from public library docs (`rvp9-core` 79, `rvp9-io` 2, `rvp9-decoder` 14, `rvp9-encoder` 45) and 3 from binary docs. Examples include public docs linking to private items and unresolved targets such as `update_mv_probs`, `write_tx_size`, and `SimdDispatch::build`; the binary warnings include an unresolved `build_encoder_config_for_test` link, an unclosed `<n>` tag, and an unresolved `Decoder::with_threads(--threads)` link. Expected: public API links resolve to documented accessible targets, and literal text is not parsed as malformed HTML. Actual: the command succeeds while emitting these warnings. No source changes were made. Raw output and command identity are recorded at `wrk_docs/evidence/2026-10-15-release-readiness/recovered-validation-20261017/postfix-main-qualification/macos-current-source-rustdoc-20261017/rustdoc.raw.txt` (SHA-256 `4bc08659d1481bf7bb009d021be8261ee52b5fa308e010733a9122e5577372d4`).

Evidence fingerprint: `manual:v1:manual-v1-workspace-rustdoc-doc-link-warnings`


## Fix

Corrected broken intra-doc links throughout the public API documentation. Links now target documented public items; references to private implementation details and literal syntax use inline code or corrected Markdown. No API or runtime behavior changed.

Validation: the baseline `cargo doc --workspace --no-deps --locked --offline` emitted 143 rustdoc warning diagnostics. After the documentation fixes, the same command completed with zero warnings. `cargo fmt --all --check` and `git diff --check` passed, and a source diff audit confirmed all 290 changed Rust source lines are comments.

### Verification summary (2026-10-18T21:49:57Z, deltic:auto run=verify-20261018T214434Z-cc7cff10 verified_fix_run=fix-20261018T201724Z-881b3f01 verdict=close)

Verifier note: Manual verify: Independent verification at 5b4b3f62: exact original cargo doc --workspace --no-deps --locked --offline emits zero warnings. Reversing only 768edb7 documentation hunks reproduces 143 rustdoc warnings, matching original observation; exit zero alone was not treated as green. Restored every byte and same command emits zero warnings again. Audited public link targets, private references as code and literal HTML syntax. cargo fmt --all --check and diff check pass.

## Notes
