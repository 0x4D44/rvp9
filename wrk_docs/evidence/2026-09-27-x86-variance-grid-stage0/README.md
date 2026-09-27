# x86 8×8 variance-grid fusion: Stage-0 no-go

The [older grid-fusion design](../../2026.05.29%20-%20HLD%20-%208x8%20variance%20grid%20fusion%20V4.md) deferred an x86 implementation pending a small timing probe. The current integrated production-PGO image made this worth rechecking: the whole `model_rd_for_sb_y_large` symbol has 20/18,127 Ducks, 440/8,195 Tree, and 422/8,629 Town direct samples. These are whole-function shares, not the grid's share.

On Crucible (Ryzen 9 9950X3D), the attached throwaway [probe](probe.rs) compared the current indirect per-cell loop against a direct AVX2 8×8-grid loop. It copied the integer-equivalent 8×8 specialization of `variance_avx2.rs:var_sum_sse_inner`, returned the same cell arrays and totals, and checked random and constant-difference inputs before timing. Rust 1.95 compiled it with `--edition=2024 -C opt-level=3 -C target-cpu=native -C debuginfo=1`. Assembly confirmed the direct reducer was inlined into the grid; the current loop retained an indirect call.

Nine alternating runs gave these median nanoseconds per whole grid:

| Grid | Input | Current | Fused | Current/fused |
| --- | --- | ---: | ---: | ---: |
| 32×32 | cache-hot | 69.707 | 65.088 | 1.07096 |
| 32×32 | rotating beyond L3 | 524.272 | 505.660 | 1.03681 |
| 64×64 | cache-hot | 257.401 | 238.012 | 1.08146 |
| 64×64 | rotating beyond L3 | 850.895 | 814.351 | 1.04487 |

The design note's 1.15× cache-hot kill gate was not met. Even if the 1.08146× speedup applied to the *entire* large-model function (it does not), multiplying by its average 3.45665% direct-sample share suggests only ~0.2604 percentage point equal-clip CPU saving. The current 1.057638× libvpx result needs ~0.7222 point to reach 1.05×. This is a heuristic projection, **not a rigorous ceiling**: samples may be attributed outside that symbol, and a standalone cache benchmark is not an encode. It is enough to decline a relatively large production SIMD patch without a stronger operation-level reason. No shipped code changed; no end-to-end candidate was measured.

The cold probe rotated 32,768 inputs (192 MiB at 32×32, 512 MiB at 64×64), beyond the host's L3. It measured 49,152 and 24,576 grids per pass, respectively. The cache-hot probe reused one input. The raw sorted values are in [timings.md](timings.md). The 5,586-line assembly and binary were scratch artifacts, not retained; `rustc --emit=asm` on the attached source reproduces the assembly check.
