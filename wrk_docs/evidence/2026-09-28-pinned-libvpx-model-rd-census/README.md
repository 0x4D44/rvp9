# Pinned libvpx model-RD call census

The Ducks CPU gap is **not explained by RVP9 simply entering the 16×16 luma model more often**. On the same canonical 60-frame input, pinned libvpx entered `model_rd_for_sb_y` 3,014,184 times at 16×16. The separate RVP9 [inner-model diagnostic](../2026-09-28-filter-evaluator-stage-census/README.md) counted 2,858,463 calls at 16×16. Those are source-level entry counts, not equivalent operations or CPU costs: the codecs use different picker paths, model internals and call boundaries. This is a no-go for a *call-frequency-only* explanation, not proof that the model is cheap or that the codecs do equal work.

The isolated Crucible diagnostic used pinned libvpx submodule `12f3a2ac603e8f10742105519e0cd03c3b8f71dd`. A compile-time `RVP9_DIAG_LIBVPX_MODEL_RD_CENSUS` gate added per-shape counters at entry to the two functions in `vp9/encoder/vp9_pickmode.c`, reported with `atexit`; the canonical encode uses one thread. The archived [base64-encoded diagnostic patch](diagnostic.patch.b64) decodes to SHA-256 `99aa61383ef0db88b64f687d50942735d18a6f204d76fc32ad194895751ac1b1`; the built `vpxenc.exe` SHA-256 is `95204dd0850a243d3fd8a171bb58376a5f93f4b16d81b08f7d81c70448937ab1`. Raw stdout/stderr/IVFs remain in `D:\worktrees\rvp9-2\20260928-TSK-HUM-count-pinned-libvpx-model-rd-calls\.scratch\libvpx-model-rd-*` on Crucible. No production source was changed.

| Clip | Small 8×8 | Small 16×16 | Small 32×32 | Large shapes | Total entries |
| --- | ---: | ---: | ---: | --- | ---: |
| Ducks | 274,185 | 3,014,184 | 9,378 | 64×64: 4 | 3,297,751 |
| Tree | 136,182 | 615,184 | 319,137 | 32×64: 32; 64×32: 116; 64×64: 5,412 | 1,076,063 |
| Town | 179,023 | 787,384 | 236,879 | 32×64: 25; 64×32: 73; 64×64: 7,773 | 1,211,157 |

Every non-listed shape counted zero. A repeated Ducks encode produced identical census rows and an identical IVF. All outputs matched the pinned libvpx canonical SHA-256: Ducks `b24234d8db46c22e03585f1973b27c2e62315f0464da392a2ecd2769d2124fa8`, Tree `b930800c5bac04a67aa0a25b3344031c5aebe6f3efe4ba569d354301bf238964`, Town `e05daf970ff4f9888a64b6edc2f4a2cd7151cf0ee2bc0b1de4170cce5f1b24bb`.

This is **not** a new performance screen. The last uninstrumented integrated production-PGO result remains [1.057638×](../2026-09-27-integrated-pgo-direct-repeat/README.md) libvpx equal-clip process CPU, versus the ≤1.05× target. Ducks remains 1.169951×. The next Ducks investigation should compare operation mix or cost at matching decision boundaries, not assume RVP9 makes more model entries.
