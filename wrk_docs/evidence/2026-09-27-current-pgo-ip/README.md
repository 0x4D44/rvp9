# Current integrated encoder: exact production-PGO IP profile

The control source at `c717a24a` has the same shipped code as current docs-only `origin/main`. Crucible built it with the unchanged 28-row, 60-frame production PGO recipe ([manifest](../2026-09-27-chroma-residual-reuse/control-pgo-manifest.txt)). The exact EXE/PDB SHA-256 values are `ce0b5bc008ab9700abacad99ea3445691e1a3ca8b5f114b9ce682e15d27730f2` and `7dbf06fcda772d00e99403a86ab4811e20d2a96607392e527c18e5cf251c832d`. The [capture wrapper](capture-pgo.ps1) pins them, the integrated HEAD, and the reviewed WPR runner. Its Tree smoke and all three full captures produced canonical IVFs.

| Clip | Canonical IVF SHA-256 | ETL SHA-256 | Direct EXE samples | Nonzero source-line samples |
| --- | --- | --- | ---: | ---: |
| Ducks | `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089` | `9c504c27c9f72f2ddfd1252f450e68d6cd288936f73500b1eb24748665d3e37b` | 16,158 | 15,994 |
| Tree | `5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41` | `cbbcea243616fd5aa00f2b45f3b0fb14b0937d2fb778112e86177333dceeea72` | 9,301 | 9,233 |
| Town | `4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f` | `d125378adea5e0914366cd2a19d47e3d074be0cef197fa451fef44189beabff9` | 7,785 | 7,731 |

The extractor in `../2026-09-25-unit1434-three-clip-ip-attribution/extract-ip.py` checked the PE/PDB GUID and age, DbgHelp's loaded PDB path, and exact image bounds before resolving samples. Each `*-xperf-tracestats.txt` has zero lost buffers and events. The committed capture logs, `*-summary.json`, per-symbol and per-source-line CSVs preserve the checks and aggregation. Raw ETLs and per-IP maps remain at `D:\worktrees\rvp9-2\20260927-TSK-HUM-profile-integrated-early-chroma-parity\.scratch\after-early-chroma-pgo-ip\<clip>\`; their ~107/104/89 MB ETLs are not committed.

Ducks' largest broad symbols are partition emission (5,200 samples), mode picking (1,753), NSTEP motion search (1,447), NEWMV search (1,380), and filter candidate evaluation (1,302). The hottest Ducks source labels are `lib.rs:11766` (early chroma-proof call, 1,794 samples), `inter.rs:3986` (NEWMV, 1,236), `lib.rs:11818` (TX selection, 1,036), and `full_pixel.rs:930` (NSTEP, 1,015). Tree and Town have strong `inter.rs:4627` filter-evaluation labels. These lines own inlined work; none is an operation's removable cost without instruction-level attribution.

This is an attribution profile, **not a paired speed comparison**. The last direct production-PGO parity result remains **1.061826× pinned libvpx equal-clip process CPU**, above the ≤1.05× target. The previous plain-release profile must not be used to rank PGO candidates over this image.
