# SSE2 chroma energy reducer — exact, slower overall

The scalar 8×8 squared-error loop in `prove_inter_chroma_zero_eob` was a small, exact x86 hotspot. A bounded SSE2 prototype passed focused differentials on AArch64 and x86, including deliberate red mutations, and produced canonical IVF outputs. It nevertheless **increased** paired Crucible process CPU by 2.36% equal-clip. Do not integrate the code on this evidence.

## Why it was tested

The exact integrated production-PGO EXE (SHA-256 `3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad`) has scalar residual arithmetic and per-pixel bounds checks in RVA `0x10d36c–0x10d79d`. Summing the retained exact-IP maps in the [integrated image profile](../2026-09-27-fresh-integrated-pgo-ip/README.md) gives 173/18,127 Ducks, 9/8,195 Tree, and 2/8,629 Town samples. Even deleting this whole range would save at most 0.3625% equal-clip sampled CPU; the bound includes instructions not necessarily removable by SSE2.

A disposable, native Crucible Stage-0 test compared the existing `sse_block_sse2` kernel with scalar SSE over 64 deterministic stride-128 tiles. Exact values matched on 64/64 tiles; a +1 mutation failed. Across eight alternating passes of 1.28 million calls per arm, scalar used 25.565–26.446 ns/tile and SSE2 5.139–5.514 ns/tile. That kernel result was a hypothesis, not a whole-encoder prediction.

## Prototype and correctness

The unintegrated prototype is commit `3562af6b` on branch `task/20260927-TSK-HUM-simd-chroma-energy-certificate-reduction`. It re-exports the existing safe SSE2 reducer through `rvp9_core::dsp::accelerated`, calls it for complete chroma 8×8 tiles on x86-64, and preserves the scalar path on AArch64/Miri. The quantizer bound, geometry, early exit, and old-transform fallback are unchanged.

- AArch64: `cargo test -p rvp9-encoder inter_chroma_energy_ --lib` passed 2/2; `cargo check --locked -p rvp9-encoder --all-targets` and `cargo fmt --all --check` passed. Mutating the new helper result by +1 made its differential fail at pattern 0, 769601 versus 769600; restoration passed.
- Crucible x86-64: the same focused tests passed 2/2, including an independent rerun in the PGO-screen worktree. The same +1 mutation failed at pattern 0, 769601 versus 769600, then restoration passed. Two pre-existing dead-code warnings were in untouched files.
- The candidate and control both matched the canonical IVF SHA-256 for all 60-frame Ducks, Tree, and Town runs, including warmups and 15 measured pairs per clip. The runner checked binary/input hashes and alternated arm and clip order.

## Paired process CPU

Both arms used the frozen integrated production-PGO profile SHA-256 `3f02af95fbeca067d6d94b233a353681bce4ed25c76930a94f0564895ac10335` and native 9950X3D codegen. The candidate was built from the prototype's two exact source files in an isolated Crucible worktree with `cargo build --locked --release -p rvp9-cli --bin rvp9enc`. Its EXE SHA-256 is `62da4151ec4ce5ca13e5edcefc7c676a6b596b2b5f2d1e9523261c4d309cad6e`. Both source-file hashes matched the committed local prototype. Before measurement, Crucible sampled about 6–12% host CPU, within Arthur's relaxed load policy.

| Clip | Candidate/control process CPU, geometric mean | Direction |
| --- | ---: | --- |
| Ducks | 1.027881× | slower |
| Tree | 1.026593× | slower |
| Town | 1.016499× | slower |
| Equal-clip | 1.023645× | slower |

The profile-use build warned that it discarded profile counts for the changed chroma-proof function and one `rvp9_core` filter evaluator after control-flow hash mismatches. That and broad code-layout changes may contribute to the regression; the run does not isolate the kernel's local contribution. The result is still enough to reject this whole-image binary. A fresh-profile retry is not justified by a hotspot whose total sampled share is below 0.4%. The integrated 60-pair RVP9/libvpx result remains [1.057638×](../2026-09-27-integrated-pgo-direct-repeat/README.md) equal-clip against the 1.05× target.

The exact runner, input/binary identities, per-encode CPU and output hashes, and per-clip ratios are retained alongside this note. The runner was copied to and executed from the remote task worktree's `.scratch`; its relative root assumes that location. The bulky IVFs stay in Crucible's isolated task worktree.
