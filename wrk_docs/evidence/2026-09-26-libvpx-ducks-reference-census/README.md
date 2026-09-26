# Direct libvpx Ducks subpixel reference census

An opt-in counter at the pinned libvpx `vp9_find_best_sub_pixel_tree` entry
records its current block shape and primary reference. The x86-64 Ducks encode
remained byte-identical to the pinned libvpx baseline and reproduced the
independent **370,507** full-tree-entry count.

| Block | libvpx LAST | libvpx GOLDEN | RVP9 LAST | RVP9 GOLDEN |
|---|---:|---:|---:|---:|
| 8×8 | 18,950 | 0 | 15,643 | 7,397 |
| 16×16 | 334,895 | 14,506 | 347,933 | 182,969 |
| 32×32 | 2,059 | 97 | 908 | 81 |
| All | 355,904 | 14,603 | 364,484 | 190,447 |

At 16×16, RVP9 performs **181,501** more entries: +13,038 LAST and
**+168,463 GOLDEN**. GOLDEN therefore accounts for **92.82%** of the
16×16 search-count gap. Across all shapes, RVP9 performs 184,424 more
entries: +8,580 LAST and +175,844 GOLDEN. These are direct reference-count
differences, not proof that changing either encoder's reference policy would
preserve output or quality.

## Identity and checks

- Pinned libvpx source commit: `12f3a2ac603e8f10742105519e0cd03c3b8f71dd`.
  The scratch copy carried the prior opt-in SVF and entry census plus the
  reference/shape extension in `libvpx-reference-census.patch.txt`. To apply,
  replace `|_` with one space and strip the remaining leading `|` column;
  the resulting patch passes `patch --dry-run` against the pinned source.
- Crucible scratch build: `C:\Users\marti\AppData\Local\Temp\deltic-scratch\ducks-origin-census-5411c114\libvpx-ref-build\vpxenc.exe`.
  Built with `-O3 -fomit-frame-pointer -DRVP9_DIAG_LIBVPX_SVF_CENSUS=1`
  and the pinned libvpx VP9 x86-64 MinGW configure flags.
- Diagnostic `vpxenc.exe` SHA-256:
  `4443751fae97b630894ea6e4bd81495991c312a294620699a44ff3be982cb521`.
- Pinned Ducks Y4M SHA-256:
  `6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7`.
- Command arguments: `--codec=vp9 --passes=1 --rt --cpu-used=6
  --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1
  --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0
  --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf`, with
  `RVP9_DIAG_LIBVPX_SVF_CENSUS=1`.
- Output IVF SHA-256 matched pinned baseline:
  `b24234d8db46c22e03585f1973b27c2e62315f0464da392a2ecd2769d2124fa8`.
- All five nonzero `LIBVPX_SUBPEL_REF_SHAPE` rows sum to 370,507, and the
  existing `LIBVPX_SUBPEL_SEARCHES,total` reporter also prints 370,507.
  Block codes 3/6/9 mean 8×8/16×16/32×32; reference codes 1/2 mean
  LAST/GOLDEN, per pinned libvpx headers. The full captured stderr is in
  `ducks.stderr.txt`.

## Next discriminator

The large GOLDEN difference is localized but not causally explained. First
measure RVP9 GOLDEN full-tree CPU share with exact-IP profiling: search count
is not process time, so the 168,463 calls are not a CPU-saving ceiling. Then
count admissions, first rejection reasons and survivors in pinned libvpx's
CBR temporal-GOLDEN branch at `vp9_pickmode.c:search_new_mv`. Its
integer-projection SAD gates can reject before fractional search; reconcile
survivors to the 14,506 16×16 GOLDEN entries. Do not port that policy as an
exact-output optimization without a differential oracle.
