# Chroma-energy certificate: production-PGO measurements

The bounded early zero-chroma candidate is committed locally as 891ba61c and mirrored on Crucible as cf9282faf. Both copies of crates/rvp9-encoder/src/lib.rs have SHA-256 f26ab81338156b0710fc8db8d8f392617c029133417482e05e9504ff27ecc98a. The candidate uses the checked, 8-bit DctDct fast-quantizer energy bound in the [design note](../../2026.09.27%20-%20HLD%20-%20chroma%20zero-proof%20energy%20certificate%20hypothesis.md); every miss retains the existing transform proof. The new old-path differential test was deliberately made red and restored. Three focused encoder tests and the all-target encoder check passed on AArch64 and native Crucible x86.

Crucible built the candidate with the unchanged 28-row, 60-frame production-PGO recipe: [manifest](build-manifest.txt), [build wrapper](build-pgo-energy.sh), and [paired runner](screen-energy.ps1). The candidate EXE SHA-256 is e04818b196ac97c5b080513802d57f55826ff0daa2e8854d2adc55578c3d271c; the integrated-code control EXE is ce0b5bc008ab9700abacad99ea3445691e1a3ca8b5f114b9ce682e15d27730f2. The pinned libvpx EXE is 88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d. The runner checks binary and input hashes, warms both arms, alternates arm and clip order, measures process CPU, and requires each 60-frame IVF to match its canonical hash. All 330 timed encodes below exited zero and matched.

| Candidate/control CPU, geometric mean | Ducks | Tree | Town | Equal-clip |
| --- | ---: | ---: | ---: | ---: |
| 5-pair PGO screen 1 | 0.919349 | 0.970144 | 0.986844 | 0.958345 |
| 10-pair PGO screen 2 | 0.919123 | 1.001760 | 1.001769 | 0.973423 |
| 10-pair PGO screen 3 | 0.914826 | 0.979409 | 1.010553 | 0.967432 |
| 15-pair candidate / libvpx | 1.195407 | 1.040519 | 0.973762 | **1.065957** |
| 15-pair integrated-code control / libvpx | 1.302019 | 1.049559 | 0.981034 | **1.102646** |

Each row has its own identity, raw, and summary CSV here. The fresh candidate direct result remains above the ≤1.05× parity target. More importantly, the fresh control build measures 1.102646× libvpx while an [earlier PGO build of the same encoder source](../2026-09-27-early-dual-tx-chroma-proof/README.md) measured 1.061826×. The encoder source diff from f78e7d89 to c717a24a is empty. This ~3.8% equal-clip PGO build spread is larger than the candidate's 2.7–4.2% apparent paired savings. The fresh control was built and measured separately, not concurrently with the candidate/libvpx run, so subtracting their direct ratios would also be unsound.

**Decision:** keep the candidate code out of origin/main for now. The screens establish exact output and an advantage over this specific fresh PGO control binary, but not a repeatable code-caused gain over the previously measured integrated build. A useful next experiment is to compile both source trees against one shared native profile and pair those binaries, then independently rebuild the PGO arms to bound profile variability. The older integrated PGO executable has been removed with its former worktree, so a direct binary-to-binary check against it was unavailable. The conditional mathematical proof and source-level x86/AArch64 tests stand separately from this performance uncertainty.
