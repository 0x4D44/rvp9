# One-axis half-pel byte average: exact, but slower

The x86 prototype in `02663fbc` used rounded byte averaging for phase-4 horizontal and vertical sub-pixel prediction. All other phases and the variance reducer remained unchanged. The bounded follow-up `1f2842e7` restored the existing source loader's AVX2 target-feature contract to test whether the first variant's wider code-generation change caused its regression. Neither prototype is integrated.

Crucible ran the pinned 60-frame 1080p Ducks, Tree, and Town inputs with CPU6, CBR 1260 kb/s, one thread, no tiles, AQ0, noise0, lag0, and KF60. Each screen used a warmup for each arm and clip, then 15 alternating-order pairs per clip. The CSV files preserve identities, every process-CPU and wall-time observation, and paired CPU ratios. Candidate and control outputs matched the pinned IVF SHA-256 for every encode. Lower candidate/control process-CPU ratio is better.

| Variant | Ducks | Tree | Town | Decision |
| --- | ---: | ---: | ---: | --- |
| A, direct byte average | 1.067775 | 1.017987 | 1.012974 | Reject: all three slower. |
| B, original AVX2 loader contract | 1.031196 | 1.005251 | 0.996096 | Reject: material Ducks regression; Town's 0.4% gain is too small to offset it. |

The integrated control binary SHA-256 was `c0db2195fa6d2ab722fbfbd9828c4c6541942f42080e65231ea861714f8540b56`. Variant A was `2acfffb40bf2ccdbb603e36783d883767a54c1fba004088e3a73f91b6e13efff`; variant B was `245e2f71d8bd0468486baa367cc76432d0e3322d304ef2c1897eb40eabec672a`. The copied `halfpel-screen.ps1` records the exact command-line protocol and expected input/output hashes. Crucible lacked `deltic quiet`; CPU rather than wall time is the decision metric. Some wall runs stalled under unrelated machine activity, but all raw observations remain in the CSVs.

The new exact-width, round-up, and scalar `(variance, sse)` differentials passed on x86 Crucible (7/7 AVX2 module tests); the round-up test first failed under a deliberate truncation mutation. Native and x86 compile checks, formatting, and diff checks also passed. Exactness is established, but the paired screens do not justify shipping this optimization. No libvpx parity comparison was run for this rejected prototype.
