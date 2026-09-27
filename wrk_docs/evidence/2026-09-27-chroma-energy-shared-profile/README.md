# Chroma-energy code/profile crossover

This experiment separates the proposed shortcut from the generated profile used to compile it. The integrated-code control is c717a24a; the candidate is cf9282faf on Crucible and the same encoder source as local rebased commit 54f27e9a. The candidate encoder source SHA-256 is f26ab81338156b0710fc8db8d8f392617c029133417482e05e9504ff27ecc98a. Both code versions were compiled with each of two existing native Crucible profiles and the same release flags, -Ctarget-cpu=native and -Cprofile-use. The candidate-generated profile SHA-256 is bc6254a6fb9b4f9bb001580479e949dbf25ae7165cc9ddc520ad4ab586e41d61; the control-generated profile is 82542a8a0ff8cc8ad5291dddb46254abd505eb41e9566f54a700b5584c5c1f16.

| Shared profile | Control EXE SHA-256 | Candidate EXE SHA-256 |
| --- | --- | --- |
| Candidate | d93c5d35d867e0ea61d5a1ec66e1e213bebc1bad98a8190fc93612c037060526 | e04818b196ac97c5b080513802d57f55826ff0daa2e8854d2adc55578c3d271c |
| Control | ce0b5bc008ab9700abacad99ea3445691e1a3ca8b5f114b9ce682e15d27730f2 | 93120058f371a36a0b94321b796eaed363e6e5f030c95d06056c66877a79a022 |

The [candidate-profile runner](screen-candidate-profile.ps1) and [control-profile runner](screen-control-profile.ps1) pin the binaries, canonical inputs, and expected 60-frame IVF hashes. Each screen warms both arms, alternates arm and clip order, and records process CPU. At the longer runs' start, Crucible had no compiler, linker, or encoder process and about 9.6% host CPU load. No quiet lease was active. Process affinity remained at the host default. A separate build storm began only after the second screen finished.

| Candidate/control CPU, geometric mean | Ducks | Tree | Town | Equal-clip |
| --- | ---: | ---: | ---: | ---: |
| Candidate profile, 5 pairs/clip | 0.885366 | 0.975444 | 0.976520 | 0.944790 |
| Candidate profile, 15 pairs/clip | 0.878699 | 0.960369 | 0.967851 | **0.934750** |
| Control profile, 5 pairs/clip | 0.938633 | 1.012202 | 0.996641 | 0.981975 |
| Control profile, 15 pairs/clip | 0.951630 | 1.014334 | 1.024012 | **0.996135** |

The identity, raw, and summary CSVs for each row are named candidate/control and candidate/control-15. The long runs each have 90 timed encodes plus six warmups; all 96/96 per run exited zero and matched their canonical output hashes. The 15 per-clip paired process-CPU ratios are retained in each summary CSV; no pair was excluded. Cross-profile compilation reported an LLVM function-control-flow hash mismatch for the changed chroma-proof function, so that function's mismatched profile counts were discarded. This is expected for the changed function, but it limits how fully the shared-profile builds can isolate code from PGO behavior.

**Interpretation:** the shortcut consistently helps Ducks, but its equal-clip gain falls from 6.5% with the candidate profile to roughly 0.4% with the control profile, while Tree and Town regress under the latter. Profile sensitivity is a real risk, not by itself a reason to discard the exact code: a subsequent [independently trained full-PGO pair](../2026-09-27-chroma-energy-independent-pgo/README.md) again saved 1.7% equal-clip CPU. That second natural-build gain meets the retention gate; integration and an integrated-trunk direct libvpx check follow separately. The earlier candidate/libvpx direct result was 1.065957×, above the ≤1.05× target.
