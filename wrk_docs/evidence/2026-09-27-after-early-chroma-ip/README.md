# Integrated post-chroma plain-release instruction profile

The integrated `c717a24a` native Crucible release build produced canonical Ducks, Tree, and Town 60-frame IVFs in sequential WPR captures. The [capture wrapper](capture-current.ps1) pins the exact worktree, EXE/PDB hashes, and the earlier reviewed capture script. Its Tree smoke run also matched the canonical IVF. The EXE SHA-256 is `b3d75f658512967df0c25ea69f301c7068e715dcf038ca9cbaeef52af44f3d99`; paired PDB SHA-256 is `8d55424adbec941c63ab981840c48d471b2dff75d24968f0350a5ec7bd1c64b3`. The extractor checked the PE/PDB GUID and age and DbgHelp's loaded PDB path.

| Clip | Canonical IVF SHA-256 | ETL SHA-256 | Direct EXE samples | Nonzero source-line samples |
| --- | --- | --- | ---: | ---: |
| Ducks | `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089` | `496ed102c12da2e827b4f7a77c90d4b819a5bbdce15b2aebb178f1e2b7087628` | 19,488 | 19,281 |
| Tree | `5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41` | `eb7b10a715dcaca48c15b46e023ffd1d03023d48c7433e9a11b353c8108d3bab` | 8,435 | 8,397 |
| Town | `4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f` | `275fed372068b404e109020dcdf6f7c51e328303f5e1796e0f9ae38a8f7d0115` | 7,275 | 7,236 |

Each ETL's xperf `tracestats` reports zero lost buffers and events. The raw ETLs, per-IP maps, symbol/line summaries, and extraction reports remain at `D:\worktrees\rvp9-2\20260927-TSK-HUM-profile-integrated-early-chroma-parity\.scratch\after-early-chroma-ip\<clip>\`. The 118/95/93 MB ETLs are not committed. Extraction used `wrk_docs/evidence/2026-09-25-unit1434-three-clip-ip-attribution/extract-ip.py` against the exact EXE/PDB pair.

The hottest Ducks symbols include partition emission (2,048 samples), inter-mode picking (1,573), scalar quantization (1,087), diamond motion search (1,076), SAD4D AVX2 (899), subpel variance AVX2 (884), and the new early chroma proof (798). The chroma proof's residual loop at `crates/rvp9-encoder/src/lib.rs:4953` has 655 Ducks samples. Tree and Town instead have AVX2 subpel variance and SIMD quantization at the top. The partition/mode symbols include inlined work; the counts are not estimates of removable CPU.

This build was **plain release, not production PGO**. Its traces can guide a bounded hypothesis but cannot replace the last paired production-PGO result: **1.061826× pinned libvpx equal-clip process CPU** after the early chroma change, versus the ≤1.05× target. The old subpel-footprint and NSTEP-seed hypotheses were already rejected after paired PGO screens; these samples do not revive them.
