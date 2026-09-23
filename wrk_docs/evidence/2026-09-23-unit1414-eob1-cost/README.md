# Unit 1414 — inter TX selector EOB mix and EOB1 cost candidate

The diagnostic starts from integrated `3976a03a`. Its temporary source change
is gated by `runtime-profiling` and `RVP9_TX_SELECT_EOB_HIST`; the ordinary
build path is unchanged. The local and Crucible diagnostic `lib.rs` blobs
matched (`29dfa9d49b4fcdf2f7baf626aad01d96923c5800`). The Crucible
feature-enabled release build passed. One 60-frame 1920x1080@30 Ducks CPU6,
CBR 1260 kbps, one-thread encode produced the same IVF SHA-256 as the exact
integrated control: `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`.

The first keyframe does not run the inter selector. Across the remaining 59
frames, the post-quantization EOB histogram was:

| TX size | EOB 0 | EOB 1 | EOB 2 | EOB >=3 |
| --- | ---: | ---: | ---: | ---: |
| 4 | 3,970,270 | 311,398 | 3,894 | 321 |
| 8 | 1,410,690 | 283,861 | 32,245 | 19,003 |
| 16 | 240,425 | 105,420 | 25,517 | 68,242 |
| 32 | 0 | 0 | 0 | 0 |

In total, EOB 1 was 700,679 of 6,471,286 selector quantized blocks (10.83%).
EOB 0 was 5,621,385 (86.87%) and already has an early cost return. The
raw per-frame output is in `ducks-eob-hist.txt` (SHA-256
`d56a14ed4bfa9f9d60c86da1f8dd713f02662aaad3bf060be17160b01b28b7f6`).
The temporary instrumentation is preserved as `selector-eob-hist.patch.gz`
(compressed SHA-256 `727111c8d6be9244828b0152ed15c2921598d9f882d67795a45e4ed3679e7419`;
uncompressed patch SHA-256 `63d5ea78ad9c71d04a7ae989ccf68a00057606bb565ee8e3e95cce56439b689d`).
Both local and Crucible source trees were restored before the candidate build.
This is frequency evidence, not performance evidence; the selector callsite's
full directional ceiling was only 3.09% of raw encoder samples in Unit 1412.

The EOB1 candidate's focused differential covered all four TX sizes, token
categories and signs, both plane/ref selectors, and all three coefficient
contexts. A deliberately wrong EOB band made it fail (`2869 != 4227`), and
the restored test passed on AArch64 and x86 release. The 60-frame candidate
Ducks IVF hash also matched the integrated control. The candidate release
encoder SHA-256 was
`347409ce75d70d7ac088fe598f61f244cb70ad5f3c005b03535cb0c28a41e66b`.

Two independent, full Crucible A/B screens used the same three pinned 1080p
60-frame clips and exact integrated control binary (`adf2da0757c6d9a9f1f37f028159f476ddd5c3055629eb254b2490634935d677`).
Each had one warmup, five matched pairs, four clip-order rotations, and no
host-load ceiling. All candidate/control IVF hashes matched per clip and were
stable across runs. The raw rows, summaries, and full manifests are preserved
as `screen-a-*` and `screen-b-*`.

| Screen | Ducks paired | Tree paired | Town paired | Corpus paired |
| --- | ---: | ---: | ---: | ---: |
| A | 0.967252 | 1.010507 | 0.989075 | 0.988787 |
| B | 0.941518 | 1.001265 | 1.000217 | 0.980597 |

Ducks won 13/20 and 16/20 matched pairs. Tree won 9/20 and 10/20; Town
won 11/20 and 10/20. Both corpus paired medians favour the candidate by
1.1–1.9%, but the benefit is concentrated in Ducks. Screen A sampled 15.14
to 23.97 busy logical cores out of 32 during clip rotations; B sampled 9.37
to 22.07. Individual pair ratios include multi-fold stalls in both directions
(A 0.131–11.506, B 0.288–4.042), so the clip-level effect remains uncertain.
The paired corpus result, not the more favourable separate median, is the
retention measure. The candidate patch is `eob1-candidate.patch.gz`
(compressed SHA-256 `234ced19f302e8ba43e7d9856a6902f5da9dde9503c13eb1dd65d9c40867e83c`;
uncompressed patch SHA-256 `079f0176d0db14c08bc19283e5f3da2592550f2a73823719313e72c5244e510b`).

A fresh pinned-libvpx screen used the same clips and operating point, with
one warmup, three pairs, and two rotations. Its candidate/libvpx paired
ratios were 2.536224 (Ducks), 1.177385 (Tree), and 0.985861 (Town), giving
1.433202 for the equal-clip corpus. The harness exited 1 because that is
above its 1.05 limit; it still wrote complete `libvpx-*` raw rows, summary,
and manifest, with stable hashes within each arm. Sampled load ranged from
5.54 to 13.55 busy logical cores of 32 during clip rotations. Three Ducks
candidate timings were 6.87, 9.30, and 18.37 seconds, versus the other three
at 2.45–2.81 seconds; the largest reference Ducks timing was 2.98 seconds.
Those asymmetric stalls make this direct run a poor trend estimate against
Unit 1410's 1.227734. It does show that parity remains unproven and the
1.05 target is not met. The two exact-binary candidate/control screens above
are the evidence for retaining this local EOB1 change.
