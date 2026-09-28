# NSTEP initial-variance skip: exact, no demonstrated CPU gain

The integrated production-PGO image remains at **1.057638× pinned libvpx process CPU** at equal clip weight, just above the ≤1.05× goal ([60-pair direct comparison](../2026-09-27-integrated-pgo-direct-repeat/README.md)). Neither NSTEP candidate below is integrated. All ratios here compare a candidate with that same integrated RVP9 image, not directly with libvpx; lower is better.

`full_pixel_diamond` eagerly computes the initial full-pixel variance score after the first diamond search. A diagnostic build counted calls and whether any subsequent distinct candidate compared that score. Over the canonical 60-frame encodes:

| Clip | Initial-score entries | Compared | Unused | Direct early returns |
| --- | ---: | ---: | ---: | ---: |
| Ducks | 629,530 | 468,941 | 160,589 | 92,425 |
| Tree | 131,265 | 46,568 | 84,697 | 61,057 |
| Town | 125,089 | 20,239 | 104,850 | 80,668 |

The diagnostic branch is local commit `b95296be` (preceded by `74d0c635`). Gate-on and gate-off native x86 encodes matched canonical IVFs; AArch64 and native x86 focused tests passed. Counts repeated exactly across two runs. The diagnostic code is not in production.

The first candidate deferred the score through `Option<u32>` until a distinct candidate needed it. Its local code commit is `1c736702`, and its production-PGO EXE SHA-256 was `ecd54d111c8a951a59decc88efc03dba34ea73446f3a6e0277d74ea4f2af008c`. All 126 screen encodes matched their canonical IVF hashes. It was slower on two independent paired Crucible screens:

| Candidate/control CPU | Ducks | Tree | Town | Equal-clip |
| --- | ---: | ---: | ---: | ---: |
| 5 pairs/clip | 1.051472 | 1.009810 | 1.016794 | 1.025865 |
| 15 pairs/clip | 1.056900 | 1.012756 | 1.000804 | 1.023205 |

The narrower second candidate skips the initial score only when the first diamond search already makes both restart and refinement impossible. Its local code commit is `e281fd3a`; its 28-row production-PGO EXE SHA-256 was `d2a36272264f9128a9e4626d09378228c5abff564a0289b923c799527f74b2ff` (31 raw profiles). The focused regression failed with the guard removed, then passed; 22 full-pixel tests passed on AArch64 and native x86. Every screen output matched its canonical IVF hash.

| Candidate/control CPU | Ducks | Tree | Town | Conditions |
| --- | ---: | ---: | ---: | --- |
| 5 pairs/clip | 1.059216 | 0.832520 | 0.927493 | Severe isolated process-CPU spikes; not usable for gain claims |
| 15 pairs/clip, first | 1.042387 | 0.973732 | 1.013785 | Residual contention, but Ducks slower |
| 15 pairs/clip, second | 0.979803 | 1.042682 | 1.017455 | Build storm tripled process CPU for several rows; unreliable |
| 15 pairs/clip, third | 1.038115 | 0.987930 | 0.994607 | Further build storm; Ducks again slower |

The paired measurements do not establish a gain, and the two less-contended 15-pair screens both put Ducks about 4% behind control. Do not integrate the early-return guard on this evidence. This directory retains `raw.csv`, `summary.csv`, and `identity.csv` from each screen as `early-screen-0{1,2,3,4}-*` and `lazy-screen-0{1,2}-*`. The bulky output IVFs remain only in the corresponding `.scratch/nstep-early-screen-*` and `.scratch/nstep-lazy-screen-*` Crucible worktrees; those paths are not durable repo artifacts.
