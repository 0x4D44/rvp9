# libvpx decode_coefs oracle

Independent C harness that runs libvpx's `decode_coefs` (and
`vp9_iht4x4_16_add_c`) on captured rvp9 inputs, so we can diff against
rvp9's production output to localize bugs that fool internal-only
differential testing.

Built during **Stage 3 stretch M.8** to break the M.6 deadlock —
where rvp9's production `decode_coefs` and our own "verbatim libvpx
ref port" produced the SAME wrong output on real conformance
bitstreams. Both implementations had a shared transcription bug that
N=100 random differential testing could not catch.

The oracle was the smoking gun: **rvp9's `iht_4_dispatch` had the rows
and cols swapped** for `AdstDct` / `DctAdst` because libvpx's
`transform_2d { transform_1d cols, rows }` struct puts `cols` FIRST
(brace-initializer order), not the natural English reading order of
"rows then cols". M.3's reference port made the same mistake. Fix:
swap the two asymmetric entries in `iht_{4,8,16}_dispatch` —
see commit message.

## Files

- `oracle.c` — the harness. Copies libvpx's `decode_coefs` and
  `read_bool` / `read_coeff` helpers verbatim (BSD-licensed) into the
  same translation unit so the static `decode_coefs` is callable from
  our wrapper. Constructs a minimal `MACROBLOCKD` + `vpx_reader` from
  the captured snapshot, then calls `decode_coefs` followed by
  `vp9_iht4x4_16_add_c` and prints both outputs.
- `snapshot.h` — captured reader pre-state + bitstream bytes + coef
  probs for the failing block of `vp90-2-02-size-32x32.webm` frame 1
  sub-1.
- `build.sh` — one-line gcc invocation.
- `README.md` — this file.

## Build

Requires:

- A working `gcc` (MSYS2 / MinGW-w64 on Windows; system gcc on Linux).
- A pre-built libvpx static archive `libvpx.a` + the `vpx_config.h`
  it was configured against. The default location is
  `D:/language/libvpx-stock/`, which is the Stage 0 in-tree build
  of libvpx v1.14.1.

```bash
bash build.sh
```

Override the libvpx tree with `LIBVPX_DIR=/some/other/libvpx`. The
build links against `$LIBVPX_DIR/libvpx.a` for all the static-data
tables (`vpx_norm`, `vp9_pareto8_full`, `vp9_cat*_prob`,
`vp9_coefband_trans_*`, `vp9_scan_orders`) and the `vpx_reader_fill` /
`idct4_c` / `iadst4_c` / `vp9_iht4x4_16_add_c` functions.

## Run

```text
$ ./oracle.exe
=== Stage 3 stretch M.8 oracle ===
Snapshot: vp90-2-02-size-32x32.webm frame 1 sub-1 (Tx4x4, intra,
  Y plane, DCT_ADST → col_scan_4x4, initial_pt=1, dq=(177,231))
Pre-state: range=200 value=0x77688b3a02cb0000 count=42 pos=15

libvpx decode_coefs:
  eob      = 16
  dq_coefs = [177, 0, 0, 0, -693, -231, 0, 0, 462, 231, 0, 0,
              231, 231, 231, -231]

rvp9 production:
  eob      = 16
  dq_coefs = [177, 0, 0, 0, -693, -231, 0, 0, 462, 231, 0, 0,
              231, 231, 231, -231]

=== MATCH ===

libvpx vp9_iht4x4_16_add_c(DctAdst, dq_coefs above, dest=128):
  residual (dest - 128) =
    [+2, +2, -14, -3]
    [-42, -58, -13, -28]
    [+17, +26, -8, +21]
    [+33, +49, +60, +38]
```

`oracle.exe` exits 0 iff libvpx's `decode_coefs` output matches the
hard-coded rvp9 production output for the snapshot. (The iht4x4
residual is informational — diff against rvp9 lives in the Rust
sibling at `crates/rvp9-harness/src/bin/decode_coefs_oracle_diff.rs`.)

## Rust diff sibling

```bash
cargo run --release --bin decode-coefs-oracle-diff
```

Replays the same snapshot through rvp9's production `decode_coefs` and
`iht4x4_16_add`, then diffs against the libvpx oracle's output
(hard-coded in the binary as `LIBVPX_DQ_COEFS` /
`LIBVPX_IHT4X4_RESIDUAL`). Both reports must be `MATCH` for the
binary to exit 0.

## Re-capturing a new snapshot

The trace dump that produced `snapshot.h` is gated by
`RVP9_DEBUG_REFCOEFS=trace` in
`crates/rvp9-core/src/entropy/detokenize.rs`. To capture a different
block, edit the filter (currently
`tx_size=Tx4x4 && initial_pt=1 && dequant=(177, 231)`), then run:

```bash
RVP9_DEBUG_REFCOEFS=trace cargo run --release --bin conformance \
    -- --vectors-dir tests/conformance/vectors --limit N 2>trace.log
```

The trace prints, per matching call:

```
RVP9_TRACE_BYTES decode_coefs(...) buffer_len=N pre_pos=P
  pre_range=R pre_value=0xV pre_count=C
RVP9_TRACE_BYTES_HEX <N hex pairs>
RVP9_TRACE_PROBS coef_probs[Tx4x4][PLY][REF]= <108 hex pairs>
```

Convert those into a new `snapshot.h` (the Python snippets in the
journal entry for M.8 show the formatting). Re-run `build.sh` and
`./oracle.exe` to see what libvpx decodes for the new snapshot.
