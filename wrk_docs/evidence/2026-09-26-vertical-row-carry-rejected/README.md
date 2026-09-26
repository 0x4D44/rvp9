# AVX2 vertical row carry: exact but slower

The bounded prototype in `row-carry.patch` reused each widened source row as the next top row only in the 16-wide vertical-only sub-pixel pass. It left all filter taps, rounding, stores, other widths, and the two-pass path unchanged. The source SHA-256 was `9fb424d43cc45753b417a4ac03f1bf04189e043a2c31a24f8e0d40939da80db4`. It was **not** integrated.

On Crucible, the new direct-temp scalar differential covered 8/16/32-row blocks, strides 17/23, all eight vertical phases, and an exact final-row source footprint. Deliberately swapping the top and bottom inputs to `bil_round` made it fail at height 8, stride 17, phase 0; restoring them passed. All six focused x86 AVX2 sub-pixel tests then passed. On AArch64, portable compile, five focused scalar sub-pixel tests, format, and diff checks passed; x86 cross-target test compilation also passed.

One 15-pair Crucible screen used the pinned 60-frame Ducks, Tree, and Town clips at CPU6/CBR1260, one thread, no tiles, AQ0, noise0, lag0, and KF60. Control was clean integrated code commit `48cab42d` (source-equivalent to `91da9803`, whose intervening commit was docs-only), EXE SHA-256 `c4b24b2c5b16add984a2dd6d32e5a7080037290e58b254aff838a0dcf26d63c1`. Candidate EXE SHA-256 was `44daa6bc0ba98259d4d1ddabae29d08a242baf8f5c7a91559c96e51961b797f2`. Both arms had one warmup per clip, then alternating-order paired encodes. All **96** encodes exited zero and matched their expected IVF SHA-256. `screen01/identity.csv`, `raw.csv`, and `summary.csv` contain every result; the reported ratios were independently recomputed from raw CPU seconds.

| Candidate/control process-CPU ratio | Ducks | Tree | Town | Equal-clip overall |
| --- | ---: | ---: | ---: | ---: |
| Screen 01, 15 pairs | 1.019256 | 1.013603 | 1.027028 | **1.019947** |

The candidate used about **2.0% more process CPU** overall, with every clip slower. A second timing screen is not needed to reject this bounded variant. Its saved loads and widens did not translate into whole-encode speed, and the change may have affected code layout or loop control; this screen does not isolate why. No new pinned-libvpx comparison was run. The latest integrated parity result remains 1.134647× equal-clip process CPU.
