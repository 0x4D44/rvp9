# Unit 1424 — TX4 batching prototype, not retained

Arthur approved the bounded prototype in the TX4 batching HLD. It grouped
four 4×4 DCTs in each 8×8 region using four-lane AVX2 or NEON arithmetic,
then quantized and priced each tile separately. The selector kept each
tile's nonzero context, `block_error >> 2`, strict TX-size argmin, and a
nonnegative lower-bound prune after each group. The original scalar walk
remained available as a differential oracle. The exact, uncommitted source
diff is `unit1424-tx4-batch.diff.gz` (SHA-256
`595e91de2ee7fa5bae6db14374ece0cb330cfd47fe2983478dcdfd4434902f2b`).

Focused tests compared all 16 transform coefficients against the scalar
reference on mixed-sign, ±255 constant, DC-only, and high-frequency blocks.
They also compared each tile's quantized coefficients, dequantized
coefficients, EOB, coefficient rate, propagated contexts, distortion, and
signaling-inclusive score at qindex 0 and 255 under fast and zbin
quantization. The selector test compared batched/scalar and pruned/exhaustive
choices across six leaf shapes, seven qindexes, and both quantizers. On the
AArch64 host the batch transform and five encoder TX tests passed. On
Crucible x86 the batch transform and five encoder TX tests passed. A
deliberate `+1` corruption of tile 2's DC coefficient made the tile-level
test fail at qindex 0; removing it restored green. Rustfmt and diff checks
passed. The prototype's x86 AVX2 kernel required a const-immediate shift
fix and a sign-extension fix before those tests passed. Its later NEON
final-pass correction and broader scalar-oracle test did not affect the
x86 release binary used in the timing screens.

Crucible compared candidate SHA-256
`3b311c2bf7436ed9986c809b46febf22e1bb8788991a7391c4544c9b1082559a`
with integrated-control SHA-256
`e56466e554d816282b742f0c3805abb4f812968fd539324cf7e507ce1180528a`.
The native Rust 1.95.0 release builds used `CARGO_PROFILE_RELEASE_DEBUG=2`
and no declared PGO. Three independent screens each used the same three
60-frame 1080p clips, CPU6, CBR 1260 kb/s, one thread, one warmup, seven
pairs, two reversed rotations, and no load ceiling. All candidate IVF hashes
matched control and remained stable within each arm.

| Screen | Candidate/control paired corpus | Separate medians | Notable load |
|---|---:|---:|---|
| A | 0.994901× | 0.989157× | Later windows 25–28 of 32 busy cores |
| B | 0.956996× | 1.091328× | Second-rotation Ducks 22.8 busy cores; multi-second stalls in both arms |
| C | 1.037013× | 1.090495× | Measurement windows 3.4–13.2 busy cores |

The paired corpus result reversed between B and C. B's paired gain also
contradicted its separate medians because long stalls skewed Ducks. A was
near neutral with implausibly opposite clip effects. The 60-second load
averages do not resolve shorter stalls. These screens do **not** establish
a reproducible whole-encode gain, even though output was exact. Under the
HLD's retention rule, the code candidate was rejected; only this evidence
and the archived diff are retained. No new direct libvpx parity number is
claimed from these screens. The integrated build's most recent pinned-libvpx
estimate remains provisional in Unit 1422, and Unit 1423 documents the
unresolved old/current build-performance contradiction.

Each screen's manifest, raw CSV, and summary CSV preserve the exact binary
and input hashes, command line, per-window load, timings, and output hashes.
