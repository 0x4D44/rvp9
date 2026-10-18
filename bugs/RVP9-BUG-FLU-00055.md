# RVP9-BUG-FLU-00055 — Public API rustdoc emits broken-link and private-item warnings

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** docs/public-api
- **Raised:** 2026-10-17T21:51:55Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261018T201724Z-881b3f01
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLU-00055-run-fix-20261018T201724Z-881b3f01
- **Owner base:** b7f74ed5574ffcf10946771a5b6a10b19315574d
- **Owner fingerprint:** -
- **Owner since:** 2026-10-18T20:17:24Z
- **Owner until:** 2026-10-19T00:17:24Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-10-17T21:51:55Z, raised via `deltic bugs new --land`)

## Observation

`cargo doc --workspace --no-deps --locked --offline` on macOS arm64 at commit `e5bb4335742bcf4327d2b64a4fcd2f4c90486268` (rustc/cargo 1.95.0) exited 0 and generated workspace HTML, but rustdoc emitted 143 warning diagnostics: 140 from public library docs (`rvp9-core` 79, `rvp9-io` 2, `rvp9-decoder` 14, `rvp9-encoder` 45) and 3 from binary docs. Examples include public docs linking to private items and unresolved targets such as `update_mv_probs`, `write_tx_size`, and `SimdDispatch::build`; the binary warnings include an unresolved `build_encoder_config_for_test` link, an unclosed `<n>` tag, and an unresolved `Decoder::with_threads(--threads)` link. Expected: public API links resolve to documented accessible targets, and literal text is not parsed as malformed HTML. Actual: the command succeeds while emitting these warnings. No source changes were made. Raw output and command identity are recorded at `wrk_docs/evidence/2026-10-15-release-readiness/recovered-validation-20261017/postfix-main-qualification/macos-current-source-rustdoc-20261017/rustdoc.raw.txt` (SHA-256 `4bc08659d1481bf7bb009d021be8261ee52b5fa308e010733a9122e5577372d4`).

Evidence fingerprint: `manual:v1:manual-v1-workspace-rustdoc-doc-link-warnings`


## Fix

<unfixed — raised only>

## Notes
