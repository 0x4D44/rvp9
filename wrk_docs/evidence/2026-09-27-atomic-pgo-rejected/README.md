# Atomic-counter PGO training rejected

The [profile replay](../2026-09-27-pgo-profile-replay/README.md) found that the same four-thread Ducks training row produced different counter profiles across exact-output runs. This bounded experiment kept the original 28 rows, `--threads=4`, `--tile-columns=2`, 60-frame inputs, and source tree, but passed `-Cllvm-args=-instrprof-atomic-counter-update-all` to the *instrumented* Rust build only. The optimized build used the normal production flags. The scratch candidate recipe SHA-256 was `e5b561d3f3413d35ddbe38eaf3deaf49fa75f2590abec4ebe3b41f32b14a5ed7`. LLVM calls this option [testing-only](https://llvm.org/doxygen/InstrProfiling_8cpp_source.html); Rust also warns that `-C llvm-args` has no usual CLI stability guarantee in its [codegen options](https://doc.rust-lang.org/rustc/codegen-options/).

The full matrix completed on Crucible and collected 31 raw profiles. The [manifest](manifest.txt) records the build. The merged profile SHA-256 was `7ce5bd213953ca39ab8d86fd94f658fff9e143ad0003d33b80fb4adddfe444d5`; the optimized EXE SHA-256 was `fa166cfd48bc5c16d77ac412c4bf7a65fe852a27d6dd488da81dfcfaf6a7879e`. Three additional runs of the same instrumented Ducks row produced identical IVF hashes but [three different counter-dump and merged-profile hashes](row-hashes.csv). The experiment therefore did **not** establish deterministic training, regardless of which counter updates the compiler made atomic; thread scheduling can change the executed path as well as the counter accuracy.

The [paired screen](raw.csv) compared the candidate against the best integrated production-PGO EXE, SHA-256 `3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad`. It used 15 process-CPU pairs per clip, alternating arm and clip order, one warmup per arm/clip, the standard 60-frame 1080p CPU6 CBR settings, and no excluded rows. All 96 encodes exited normally and matched their canonical IVF hashes. [Identity](identity.csv), [protocol](protocol.txt), and [summary](summary.csv) accompany the raw rows.

| Clip | Atomic-training PGO / integrated control CPU |
| --- | ---: |
| Ducks | 1.101992 |
| Tree | 1.047792 |
| Town | 1.079009 |
| Equal-clip geometric mean | **1.076034** |

**Decision:** reject the atomic-counter option. The optimized encoder costs 7.60% equal-clip CPU versus the integrated control, and the representative row's profile still varies. The one-line recipe prototype was restored; no shipped change is retained. The integrated direct parity result remains **1.054068× libvpx**.
