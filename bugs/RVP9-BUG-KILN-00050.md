# RVP9-BUG-KILN-00050 - rvp9enc marks parse-only flags as supported no-ops

- **State:** Closed
- **Priority:** Should
- **Severity:** Medium
- **Area:** cli/rvp9enc option contract
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-cli/` at `8d66f33`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T200415Z-p68292-n478425000-c1 branch=task/bug-RVP9-BUG-KILN-00050-run-fix-20260730T200415Z-p68292-n478425000-c1 code=5a671ce1f2252f055c13732166155f11bdce6352 gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

The shared parser contract says `apply: Some(handler)` means a flag is wired and that `apply: None` is the "recognised but not implemented yet" path at `crates/rvp9-cli/src/args.rs:8`-`12`. Help output derives `[supported]` directly from the same field at `crates/rvp9-cli/src/args.rs:174`-`181`.

`rvp9enc` violates that contract for parse-only flags. `EncOpts` explicitly says fields below the W0.1 divider are "parsed and stored" for later waves at `crates/rvp9-cli/src/bin/rvp9enc.rs:67`-`75`. Examples accepted via `apply: Some` include `--threads` at `crates/rvp9-cli/src/bin/rvp9enc.rs:510`-`516`, `--tile-columns` and `--tile-rows` at `crates/rvp9-cli/src/bin/rvp9enc.rs:520`-`536`, `--profile` at `crates/rvp9-cli/src/bin/rvp9enc.rs:460`-`466`, `--bit-depth` at `crates/rvp9-cli/src/bin/rvp9enc.rs:720`-`726`, `--color-space` at `crates/rvp9-cli/src/bin/rvp9enc.rs:886`-`892`, `--row-mt` at `crates/rvp9-cli/src/bin/rvp9enc.rs:936`-`942`, and `--resize-*` at `crates/rvp9-cli/src/bin/rvp9enc.rs:946`-`972`.

The production config builder starts at `crates/rvp9-cli/src/bin/rvp9enc.rs:1516` and does not consume those fields; it hardcodes 8-bit config at `crates/rvp9-cli/src/bin/rvp9enc.rs:1545`-`1548`, while the packet headers still hardcode `Profile0`, `WireBitDepth::Eight`, `Bt601`, and `Yuv420` at `crates/rvp9-encoder/src/frame_loop.rs:4416`-`4424` and `crates/rvp9-encoder/src/frame_loop.rs:4854`-`4864`. Source search found those stored options used only by parser-storage tests, not runtime behavior.

Expected: a `[supported]` flag changes behavior or fails loudly when the requested mode is unsupported. Actual: users can request non-default profiles, bit depths, tiling, row-mt, resize, and related modes and still get default behavior with no warning.

## Fix

Move unplumbed flags back to `apply: None`, reject non-default values during config building, or fully plumb them into encoder config/header/runtime behavior. Add behavior tests for each supported flag class rather than storage-only tests.

## Notes

`RVP9-BUG-KILN-00015` covers the lower-level `EncoderConfig::bit_depth` acceptance path. This bug is the CLI support-contract violation where parse-only values are advertised and accepted before reaching the encoder.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Every flag the observation named is now `apply: None` -- `--profile`, `--threads`,
`--tile-columns`, `--tile-rows`, `--bit-depth`, `--color-space`, `--row-mt`, `--resize-*` and
about 25 more -- with their `EncOpts` fields and parse helpers deleted (-606 lines).
The converse was checked too, which is where a near-miss would hide: every flag left as
`apply: Some` is genuinely consumed by the config builder (noise_sensitivity, temporal_layers,
auto_alt_ref, enable_tpl, lag_in_frames, cpu_used, deadline, min_q/max_q, cq_level all traced to
their use sites). The "apply means wired" contract now holds in both directions.
`enc_parse_only_flags_reject_and_are_tagged_not_yet` observed passing -- 39 flags, asserting
non-zero exit plus "not supported yet" on stderr AND that the `--help` line carries `[not yet]`
rather than `[supported]`, parsed from real `--help` stdout of the spawned binary.
Residual noted, not blocking: this removes the false CLAIM rather than plumbing the features;
users still cannot set profile/bit-depth/tiling. RVP9-BUG-KILN-00015 tracks the encoder-level
bit_depth item separately.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
