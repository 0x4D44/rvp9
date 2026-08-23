# RVP9-BUG-CRU-00005 — write_loopfilter and write_delta_q silently truncate out-of-range fields instead of returning Err

- **State:** Open
- **Priority:** Could
- **Severity:** Low
- **Area:** core/bitstream sub_parsers
- **Raised:** 2026-08-23T17:09:50Z
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
- **State history:** Open (2026-08-23T17:09:50Z, raised via `deltic bugs new` model=claude-opus-5@high)

## Observation

Commit `bdb22b494` ("fix(core): reject invalid tile header geometry") replaced a release-mode
no-op `debug_assert!` in `write_tile_info` with real validation returning
`BitstreamError::InvalidTileInfo` (`crates/rvp9-core/src/bitstream/sub_parsers.rs:574-590`).
Two immediate siblings in the same file still have the original defect.

**`write_loopfilter` (`crates/rvp9-core/src/bitstream/sub_parsers.rs:135-140`).**

```rust
wb.write_literal(u32::from(lf.filter_level), 6);
wb.write_literal(u32::from(lf.sharpness_level), 3);
```

`LoopFilterParams.filter_level` and `.sharpness_level` are public `u8` fields with no
invariant (`sub_parsers.rs:71-75`). VP9 allows `filter_level` in `0..=63` and
`sharpness_level` in `0..=7`. `RawBitWriter::write_literal`
(`crates/rvp9-core/src/bitstream/raw.rs:98-105`) writes the low `bits` bits and does not
signal truncation.

Repro: set `filter_level = 64`, call `write_loopfilter`, then `parse_loopfilter` on the
result. `64 & 0x3F == 0`, so the round-trip returns `filter_level = 0` and the function
returns `Ok(())`. The emitted header is structurally valid and silently describes a different
loop filter than the caller asked for.

**`write_delta_q` (`crates/rvp9-core/src/bitstream/sub_parsers.rs:226-232`).** Same class:
`write_inv_signed_literal(i32::from(delta_q), 4)` writes a 4-bit magnitude plus a sign bit,
valid range `-15..=15`, but `delta_q` is an `i8` reached through the public
`write_quantization` (`sub_parsers.rs:237-245`) via public `QuantizationParams` fields.
`y_dc_delta_q = 100` writes `100 & 0xF == 4` and round-trips as `4`. `delta_q = -128`
round-trips as `0`, because `unsigned_abs() == 128` truncates to `0` while the sign bit is
still set.

**Expected:** a public bitstream writer given a field value it cannot represent returns
`Err`, as `write_tile_info` now does.
**Actual:** it silently truncates and returns `Ok(())`.

Severity is Low because `rvp9-encoder` itself never supplies out-of-range values — these are
reachable only by an external consumer of the published `rvp9-core` crate driving the writers
directly, which is exactly the audience `write_tile_info` was just hardened for.

Checked and found already correct, so not part of this bug: `QuantizationParams.base_qindex`
(`u8` into `QINDEX_BITS = 8`, always fits) and `RawBitWriter::write_literal` /
`write_bit` themselves (documented clamp at `raw.rs:96-101`, buffer-overflow poison flag at
`raw.rs:66-75`).

**Suggested fix:** validate in `write_loopfilter` and `write_delta_q` before any bit is
written, returning a new `BitstreamError` variant (or reusing the shape of `InvalidTileInfo`),
and mirror the existing regression test `write_tile_info_rejects_invalid_geometry_before_writing`
(`sub_parsers.rs:1084-1105`, with its helper at `:1065-1082`) — including its `assert_eq!(buf, original)` check that no bits
were emitted.

## Fix

<unfixed — raised only>

## Notes
