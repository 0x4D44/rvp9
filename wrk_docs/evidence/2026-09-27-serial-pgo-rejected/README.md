# Single-thread PGO training is stable but slower

The [profile replay](../2026-09-27-pgo-profile-replay/README.md) established that a four-thread Ducks training row produced three different counter profiles despite identical IVF output; the same row with one thread produced three identical profiles. This bounded recipe experiment changed only `--threads=4` to `--threads=1` in all 28 production-PGO training rows. It retained the same source tree, 60 frames per row, inputs, CPU presets, SVC/AQ/denoising rows, and `--tile-columns=2`. The candidate recipe SHA-256 was `5b36915a611afaf3d119c5f14a40ae0b4875524e543c7e0f502722af3016dd27`; it was used only from Crucible task scratch, not integrated. The [build manifest](manifest.txt) records 31 raw profiles. Its merged profile SHA-256 was `fe93e4bdbb85318a8ef9655b4b86ff91ef1ffab5c14e283244dbc2de6f5cc998` and optimized EXE SHA-256 was `43ba8ebbf007031daf1791e1a2933a5a92297dec5cebe9e87c760f29485d9e21`.

The [paired screen](raw.csv) compared that candidate against the best previously integrated production-PGO EXE, SHA-256 `3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad`, on Crucible. It used 15 process-CPU pairs per clip, alternating arm and clip order, one warmup per arm/clip, the standard 60-frame 1080p CPU6 CBR settings, and no excluded rows. All 96 encodes exited normally and matched their canonical IVF hashes. [Identity](identity.csv), [protocol](protocol.txt), and [summary](summary.csv) are retained beside the raw rows.

| Clip | Single-thread PGO / integrated control CPU |
| --- | ---: |
| Ducks | 1.083633 |
| Tree | 1.033080 |
| Town | 1.026488 |
| Equal-clip geometric mean | **1.047426** |

**Decision:** reject the one-thread training recipe. It stabilizes the representative row's counters but costs 4.74% equal-clip CPU versus the integrated control, including an 8.36% Ducks regression. The task worktree's one-line recipe prototype was restored to the integrated `--threads=4` setting; no shipped change is retained. This does not rule out more careful counter aggregation or training-weight changes, but those need their own paired, repeatable evidence.
