# Source-identical PGO profile replay

Crucible's second production-PGO build was [2.4613% slower](../2026-09-27-integrated-pgo-repeat/README.md) than the first, although shipped source and the 28-row training recipe were unchanged. This experiment isolates the profile input to the final build and tests whether multithread training can yield different counters with exact encoded output.

The unchanged second-build instrumented executable encoded the same 60-frame Ducks input three times with the first production training-row arguments (`--cpu-used=6 --end-usage=vbr --target-bitrate=2500 --threads=4 --tile-columns=2`). All three IVFs had the same SHA-256 `89b52e0f4da12f3d6d26f65a8b159211804bed3e0bed948357617c3e82f67466`, but all three `llvm-profdata show --counts --all-functions` dumps and merged row profiles had different hashes. With only `--threads=1` changed, three runs again produced identical IVFs to one another (not to the four-thread output), and their counter dumps and profiles were byte-identical. The [six-run hashes](row-repro-hashes.csv) retain both controls. This establishes profile-count nondeterminism in at least one multithread training row; it does not show that every row varies or that single-thread training is a suitable production replacement.

To isolate the build input, the same `cef7895fce53d40315946069324e8ff58324d902` source tree and Crucible toolchain were recompiled with the *first* merged profile, SHA-256 `3f02af95fbeca067d6d94b233a353681bce4ed25c76930a94f0564895ac10335`, instead of the second profile, SHA-256 `19f90b04c93ac4ce6f6301d6b72b4be89447e3f37ddfd41adc377ee19616259f`. The replay executable SHA-256 was `2d97dc5d9c2ab738e9303b2bd10607de3fbe0f00e3e72d1de30857a084a31c4c`. The [paired screen](old-profile-replay-raw.csv) compared this replay against the second build over 15 pairs per clip, with alternating arms and clip order, one warmup per arm/clip, exact canonical IVF hashes, and no excluded rows.

| Clip | Old-profile replay / second build process CPU |
| --- | ---: |
| Ducks | 0.970701 |
| Tree | 0.981581 |
| Town | 0.976684 |
| Equal-clip geometric mean | **0.976312** |

The replay restores almost all of the second build's CPU loss (second/first was `1.024613`, and replay/second was `0.976312`). Thus the profile input is a strong causal explanation for this particular executable difference. These screens alone do not prove that four-thread scheduling is the only profile cause, nor that either profile will be reproducibly fastest. The first integrated artifact's direct libvpx comparison remains **1.054068×**; the replay was not measured directly against libvpx. Keep the ≤1.05× goal open.
