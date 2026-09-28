# Laplacian model clamp census: no-go

The exact integrated production-PGO encoder remains **1.057638×** pinned libvpx equal-clip process CPU, above the ≤1.05× target. This is an output-neutral operation census, **not** a new performance comparison.

`model_rd_from_var_lapndz` divides a wrapped `u64` numerator by nonzero variance, then clamps the quotient to `MAX_XSQ_Q10`. An exact pre-divide comparison against `MAX_XSQ_Q10 * var` could skip the divide on clamped calls. The product fits `u64`; the comparison must use the post-wrap numerator to preserve current results. Before prototyping the branch, an opt-in diagnostic counted actual nonzero calls and clamped quotients without changing the division or encode decisions.

| Canonical CPU6/CBR1260 clip, 60 frames | Nonzero calls | Clamped | Clamp rate |
| --- | ---: | ---: | ---: |
| Ducks | 2,713,714 | 139,337 | 5.13% |
| Tree | 843,053 | 100,724 | 11.95% |
| Town | 1,033,842 | 80,550 | 7.79% |

The raw [census](census.csv) records the input, output and native x86 diagnostic EXE SHA-256 values, exit codes, and unsaturated counts. Every IVF matched its canonical integrated hash. The diagnostic EXE SHA-256 was `e6945ef99a7ec1b0e36559d38322fef11c4a6a63979ba72e8a1b36eb1c6154f5`. The opt-in source is local commit `3483bfd7` on `task/20260928-TSK-HUM-count-laplacian-model-saturation-opportu`, also applied to Crucible's isolated task worktree; it is **not integrated**. The Crucible worktree retains the run script, raw stderr and IVFs in `.scratch/lapndz-saturation-60f/`.

The counter-reconciliation test was deliberately made red and restored green on AArch64, then passed independently on AArch64 and native x86. The AArch64 CLI built, and a tiny gate-off/on encode produced identical IVF SHA-256 values. The native x86 release CLI built; all three 1080p encodes exited zero, and `nonzero = clamped + unclamped` held. The counters include all Laplacian model call sites, not only the 16×16 DC/AC calls. A branch/product check on every nonzero call to avoid a divide in only 5.13% of Ducks calls has an unattractive opportunity ratio; no production prototype or paired PGO screen is warranted from this evidence alone.
