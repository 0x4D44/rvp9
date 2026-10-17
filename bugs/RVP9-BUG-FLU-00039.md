# RVP9-BUG-FLU-00039 — AArch64 width-four I8MM convolution reads beyond accepted tight source slice

- **State:** Open
- **Priority:** Must
- **Severity:** High
- **Area:** core/dsp/convolve8-neon
- **Raised:** 2026-10-15T22:39:05Z
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
- **State history:** Open (2026-10-15T22:39:05Z, raised via `deltic bugs new` model=gpt-6.1-sol@medium)

## Observation

On Apple M5 Max/aarch64-apple-darwin at candidate code 1a84f60a24e8d82e8e6454743fbb80521280df0e, native AddressSanitizer library testing reports `ERROR: AddressSanitizer: heap-buffer-overflow`, `READ of size 4` in existing test dsp::aarch64::convolve8_neon::tests::single_axis_matches_scalar_all_filters_and_shapes. Full native four-library command exits 101/SIGABRT; a second focused command reproduces the same report: CARGO_TARGET_DIR="$PWD/.scratch/asan-target" RUSTFLAGS="-Zsanitizer=address -Ctarget-cpu=native" RUSTDOCFLAGS="-Zsanitizer=address" cargo +nightly test -Zbuild-std --target aarch64-apple-darwin --locked -p rvp9-core --lib dsp::aarch64::convolve8_neon::tests::single_axis_matches_scalar_all_filters_and_shapes -- --exact --nocapture --test-threads=1. Installed nightly rustc bd08c9e71 (2026-06-25), LLVM22.1.7; no source/test changes. Both stacks reach load_four_repeated at convolve8_neon.rs:52, convolve8_horiz_i8mm_4_inner at :319, safe wrapper convolve8_horiz_neon at :654, test at :909. The first report is a 4-byte read crossing the end of a 56-byte allocated source. Source-confirmed root: validate_horiz accepts the minimal w+7=11 bytes for a width4 source row, but I8MM loads8 bytes at row start plus4 at row+8, requiring12 bytes; its safety comment incorrectly asserts those requirements agree. The unchanged test supplies the documented tight source span, not an intentionally malformed buffer. Expected: the safe wrapper stays within its input slice on all accepted geometries. Actual: the I8MM width4 path reads beyond the slice on the final row. Padded production callers may mask the read; no universal production reachability claim is made. Raw full/focused sanitizer reports are preserved under wrk_docs/evidence/2026-10-15-release-readiness. No fix or suppression performed; separate bug owner must address it before native sanitizer acceptance.

## Fix

<unfixed — raised only>

## Notes

### Additional qualification evidence (2026-10-16)

The separately run native encoder ASan library also aborts in existing `frame_loop::tests::arf_decoder_fc_bank_one_stays_synced_after_non_show_frame`. Its stack reaches `build_inter_prediction_from_tight_plane -> convolve8_neon -> convolve8_horiz_neon`. Source/disassembly review identifies the same four-byte width4I8MM load at row+8. The reference allocation is4096bytes. This test uses CPU7CBR with lag5 and ALTREF; do not extrapolate it to lag-zero production reachability. Raw `macos-asan-dependent-libraries.log` is retained with the original reports.

Repair-scope note from source review: wider I8MM/dotprod horizontal paths load16bytes per eight outputs while validation may guarantee only15 on the last group of widths divisible by8. This wider mismatch has not been separately reproduced at runtime; inspect it rather than fixing only the observed width4 case. No fix or suppression was made in this task.


### Additional source audit evidence (2026-10-17)

The exact-candidate static SIMD audit at source `fb5c0c917038db6c640b3dbc7c2b84b567233333` confirmed the wider mismatch noted above. `validate_horiz` accepts `w + 7` bytes, but the i8mm/dotprod eight-output kernels load 16 bytes at the final vector offset `x = w - 8`, requiring `w + 8`. For `w=8,h=4,src_stride=15,src.len()=60`, the final load reads byte 60, one past the slice. The width-four loader likewise requires 12 bytes while validation requires 11. Both paths are behind runtime feature checks and reachable through safe dispatch getters. This is a source-only confirmation of the wider repair scope, not a second ledger issue or an additional runtime reproduction.
