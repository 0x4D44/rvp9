# RVP9-BUG-KILN-00050 - rvp9enc marks parse-only flags as supported no-ops

- **State:** Open
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-cli/` at `8d66f33`)

## Observation

The shared parser contract says `apply: Some(handler)` means a flag is wired and that `apply: None` is the "recognised but not implemented yet" path at `crates/rvp9-cli/src/args.rs:8`-`12`. Help output derives `[supported]` directly from the same field at `crates/rvp9-cli/src/args.rs:174`-`181`.

`rvp9enc` violates that contract for parse-only flags. `EncOpts` explicitly says fields below the W0.1 divider are "parsed and stored" for later waves at `crates/rvp9-cli/src/bin/rvp9enc.rs:67`-`75`. Examples accepted via `apply: Some` include `--threads` at `crates/rvp9-cli/src/bin/rvp9enc.rs:510`-`516`, `--tile-columns` and `--tile-rows` at `crates/rvp9-cli/src/bin/rvp9enc.rs:520`-`536`, `--profile` at `crates/rvp9-cli/src/bin/rvp9enc.rs:460`-`466`, `--bit-depth` at `crates/rvp9-cli/src/bin/rvp9enc.rs:720`-`726`, `--color-space` at `crates/rvp9-cli/src/bin/rvp9enc.rs:886`-`892`, `--row-mt` at `crates/rvp9-cli/src/bin/rvp9enc.rs:936`-`942`, and `--resize-*` at `crates/rvp9-cli/src/bin/rvp9enc.rs:946`-`972`.

The production config builder starts at `crates/rvp9-cli/src/bin/rvp9enc.rs:1516` and does not consume those fields; it hardcodes 8-bit config at `crates/rvp9-cli/src/bin/rvp9enc.rs:1545`-`1548`, while the packet headers still hardcode `Profile0`, `WireBitDepth::Eight`, `Bt601`, and `Yuv420` at `crates/rvp9-encoder/src/frame_loop.rs:4416`-`4424` and `crates/rvp9-encoder/src/frame_loop.rs:4854`-`4864`. Source search found those stored options used only by parser-storage tests, not runtime behavior.

Expected: a `[supported]` flag changes behavior or fails loudly when the requested mode is unsupported. Actual: users can request non-default profiles, bit depths, tiling, row-mt, resize, and related modes and still get default behavior with no warning.

## Fix

Move unplumbed flags back to `apply: None`, reject non-default values during config building, or fully plumb them into encoder config/header/runtime behavior. Add behavior tests for each supported flag class rather than storage-only tests.

## Notes

`RVP9-BUG-KILN-00015` covers the lower-level `EncoderConfig::bit_depth` acceptance path. This bug is the CLI support-contract violation where parse-only values are advertised and accepted before reaching the encoder.
