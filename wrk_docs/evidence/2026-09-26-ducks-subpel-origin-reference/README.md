# Ducks full-tree subpixel origin/reference census

The opt-in diagnostic at `find_best_sub_pixel_tree` reconciled with the prior
independent Ducks census: **554,931** actual entries. Every entry came from the
final inter picker; the A4 speculative picker and unknown origin contributed
zero. Turning the counter on did not change the IVF bytes.

| Block | LAST | GOLDEN | Total |
|---|---:|---:|---:|
| 8×8 | 15,643 | 7,397 | 23,040 |
| 16×16 | 347,933 | 182,969 | 530,902 |
| 32×32 | 908 | 81 | 989 |
| All | 364,484 | 190,447 | 554,931 |

All other block/reference buckets were zero. The complete deterministic CLI
report is in `ducks-on.csv` (UTF-8 conversion of Crucible's PowerShell capture).

## Reproduction identity

- Source: unintegrated diagnostic commit `37254c15`, applied as four
  hash-verified source files to a clean copy of production code commit
  `48cab42d` (code-equivalent to integrated docs-only head `3edb8497`).
- Crucible: AMD Ryzen 9 9950X3D, x86-64. Isolated source and target under
  `C:\Users\marti\AppData\Local\Temp\deltic-scratch\ducks-origin-census-5411c114`.
- Built release `rvp9enc.exe` SHA-256:
  `18d228ee9ae38af04529e428b4fc78089e6241f88ea87d6f4ad48c1c9bf436f1`.
- Pinned 60-frame Ducks Y4M SHA-256:
  `6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7`.
- Common arguments: `--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr
  --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0
  --lag-in-frames=0 --tune-content=default --aq-mode=0
  --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf`.
  The only difference was `RVP9_SUBPEL_ORIGIN_CENSUS=0` versus `=1`.
- Both output IVFs have the canonical SHA-256
  `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`.
- Mac AArch64: rustfmt, 20 subpixel tests, encoder test build, CLI build,
  deliberate-red counter assertion, and a five-frame counter-on/off exact IVF
  comparison passed. Crucible x86-64: release CLI build and focused counter
  test passed.

## Interpretation and next discriminator

The earlier pinned libvpx census inferred 349,401 16×16 full-tree searches
from 3,494,010 fractional variance calls divided by ten, versus RVP9's
530,902 actual entries. RVP9's 347,933 LAST entries nearly equal libvpx's
**all-reference** estimate, but this does **not** prove GOLDEN caused the
181,501-entry difference: libvpx's LAST/GOLDEN split is unknown. If its GOLDEN
count is `g`, the implied LAST excess is `g - 1,468` and GOLDEN excess is
`182,969 - g`, assuming the two references exhaust the comparable scope.

Count libvpx actual full-tree entries by block/reference and reconcile to its
existing 370,507-entry total before attributing the excess. Its CBR temporal
GOLDEN path in `vp9_pickmode.c:search_new_mv` also has integer-projection SAD
rejection gates before fractional search; count branch admissions and rejection
reasons separately. Those policy gates may explain a count difference, but
copying them into RVP9 would not preserve exact output and is not an approved
production optimization.
