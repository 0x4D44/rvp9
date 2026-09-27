# Early chroma energy certificate: diagnostic census

The proposed `65*SSE < (T-4)^2` predicate was **observational only in this run**. The exact TX8/TX4 proof still ran on every call. Diagnostic source is local, unintegrated commit `5ead1477` (`crates/rvp9-encoder/src/lib.rs` SHA-256 `90c805fd7ac27fba3cfbcfbc5fad8751265cdb198a8cd8afb28256112624a85c`); the native Crucible release EXE SHA-256 is `f8126a88b91eafa3a59d00a690a130e3ef4913ca06f18778dc69885e4631799b`. It was built without PGO for a **counting** run, not a CPU comparison.

| 60-frame clip | Exact proof calls | Exact EOB0 calls | Predicate hits | Hit rate | Predicate-hit / proof-false |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ducks | 304,058 | 298,591 | 251,653 | 82.76% | 0 |
| Tree | 43,122 | 40,054 | 12,816 | 29.72% | 0 |
| Town | 7,731 | 7,188 | 2,858 | 36.97% | 0 |

Every run emitted 59 `[CHROMA_ENERGY]` frame rows. All three process exits were zero and their IVF SHA-256 values matched the current canonical outputs. [The runner](run-chroma-energy.ps1), [identities](identity.csv), and complete [Ducks](ducks.stderr.txt), [Tree](tree.stderr.txt), and [Town](town.stderr.txt) counters are retained. The first frame is a keyframe and has no inter chroma-proof calls. These counts cover the exact current CPU6 single-thread 60-frame clips, not all legal residuals or all presets.

The later [exact-rational bound check](check-chroma-bound.py) and [design note](../../2026.09.27%20-%20HLD%20-%20chroma%20zero-proof%20energy%20certificate%20hypothesis.md) give a conditional proof for the current 8-bit DctDct fast-quantizer paths, with source-reviewed SIMD equivalence. `python3 wrk_docs/evidence/2026-09-27-chroma-energy-census/check-chroma-bound.py` printed `Tmax 837 Smax 10675`, all intermediates below 32,768, and `ALL CHECKS PASS`. Zero observed false hits remains empirical corroboration, not the proof. A production candidate still requires exact differentials, canonical outputs, and repeatable paired production-PGO CPU gain. The latest direct integrated parity remains 1.061826× pinned libvpx equal-clip process CPU against the ≤1.05× target.
