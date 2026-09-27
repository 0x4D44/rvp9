# Early chroma energy certificate: diagnostic census

The proposed `65*SSE < (T-4)^2` predicate is **not a proven skip condition**. This diagnostic measured it without changing the encoder decision. The exact TX8/TX4 proof still ran on every call. Diagnostic source is local, unintegrated commit `5ead1477` (`crates/rvp9-encoder/src/lib.rs` SHA-256 `90c805fd7ac27fba3cfbcfbc5fad8751265cdb198a8cd8afb28256112624a85c`); the native Crucible release EXE SHA-256 is `f8126a88b91eafa3a59d00a690a130e3ef4913ca06f18778dc69885e4631799b`. It was built without PGO for a **counting** run, not a CPU comparison.

| 60-frame clip | Exact proof calls | Exact EOB0 calls | Predicate hits | Hit rate | Predicate-hit / proof-false |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ducks | 304,058 | 298,591 | 251,653 | 82.76% | 0 |
| Tree | 43,122 | 40,054 | 12,816 | 29.72% | 0 |
| Town | 7,731 | 7,188 | 2,858 | 36.97% | 0 |

Every run emitted 59 `[CHROMA_ENERGY]` frame rows. All three process exits were zero and their IVF SHA-256 values matched the current canonical outputs. [The runner](run-chroma-energy.ps1), [identities](identity.csv), and complete [Ducks](ducks.stderr.txt), [Tree](tree.stderr.txt), and [Town](town.stderr.txt) counters are retained. The first frame is a keyframe and has no inter chroma-proof calls. These counts cover the exact current CPU6 single-thread 60-frame clips, not all legal residuals or all presets.

The scalar quantizer threshold formula was checked against the current `vp9_quantize_fp` operation, but no formal bound yet proves that the repository's integer DCT4/DCT8 and SIMD implementations fit the proposed `65*SSE` margin. Zero observed false hits is only empirical evidence. Do **not** turn the diagnostic predicate into a production shortcut until the full integer-domain bound is established and checked on both x86 and AArch64. A production candidate would still require deliberate-red differentials, canonical outputs, and repeatable paired production-PGO CPU gain. The latest direct integrated parity remains 1.061826× pinned libvpx equal-clip process CPU against the ≤1.05× target.
