# One-axis AVX2 half-pel eligibility census

This is an output-neutral opportunity count, not a speed measurement. Diagnostic commit `260d5f42` on `task/20260926-TSK-HUM-measure-one-axis-halfpel-subpel-calls` counts completed AVX2 sub-pixel variance calls with exactly one fractional axis, by axis, phase, and block shape. It reports only under `RVP9_DIAG_SUBPEL_PHASE=1` after a successful `rvp9enc` encode. The diagnostic is not integrated.

The hot one-axis AVX2 loops in the hash-matched integrated executable (`c0db2195fa6d2ab722fbfbd9828c4c6541942f42080e65231ea861714f8540b56`) still widen bytes and multiply bilinear taps at RVAs around `0x6a999` and `0x6aef6`; they do not use byte averaging there. At phase 4, the taps are `[64,64]`, making the exact output `(a+b+1)>>1`, which `_mm_avg_epu8` computes. The prior exact-IP profile attributed 67/1,611 Tree and 56/1,457 Town process samples to all one-axis sub-pixel work; these are gross upper bounds, not predicted half-pel savings.

Crucible built the three-file diagnostic copy in an isolated worktree. Release binary SHA-256: `6335ad30b45081acec4285ac8d52ba8faa5be6b00ad604bc1fd6e1d009bbb0ab`. The runner used pinned 60-frame Ducks, Tree, and Town Y4Ms at CPU6/CBR1260/30fps, one thread, no tiles, AQ0, noise0, lag0, KF60. `manifest.csv` records the input and IVF output hashes; all three encodes exited zero and reproduced the pinned integrated IVF hashes. The raw per-shape/phase reports are in each clip's `stderr.txt` file. The synthetic counter test deliberately failed when the expected x/8×8/phase-4 count was changed from 2 to 1, then passed after restoration; native and x86 CLI checks and formatting also passed.

| Clip | Horizontal calls | Horizontal phase 4 | Vertical calls | Vertical phase 4 | Phase 4 share of all one-axis calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ducks | 1,707,301 | 1,109,859 | 2,154,386 | 1,109,862 | 57.48% |
| Tree | 359,722 | 248,946 | 351,586 | 248,948 | 70.00% |
| Town | 356,096 | 240,900 | 347,298 | 240,900 | 68.50% |

Only phases 2, 4, and 6 occurred in this pinned search. Ducks used 8×8, 16×16, and 32×32 one-axis shapes; Tree and Town also used larger shapes. The high phase-4 call share justifies a bounded one-axis byte-average prototype, with exact temp and `(var,sse)` differentials and paired CPU gating. It does **not** establish a speed gain or change the current pinned RVP9/libvpx parity result.
