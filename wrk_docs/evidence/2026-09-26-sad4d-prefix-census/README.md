# Search-only 16×16 SAD4D four-row early-exit census

The unintegrated, output-neutral diagnostic commit `8aca0210` counted the two NSTEP search callsites whose four 16×16 reference blocks were in-plane. It computed four-row prefix SADs before the unchanged full SAD4D call. A batch could be rejected early only when **all four** prefixes already met or exceeded the entry best: later rows and MV cost are nonnegative, and selection uses strict `<`. The cost-list caller was not instrumented because it needs exact SADs.

| Pinned 60-frame clip | Eligible batches | All four prefixes ≥ best | Hit rate |
| --- | ---: | ---: | ---: |
| Ducks | 6,134,402 | 836 | 0.013628% |
| Tree | 400,826 | 3,962 | 0.988459% |
| Town | 290,317 | 10,647 | 3.667370% |

These rates reject the proposed four-row early exit before a production kernel rewrite. Ducks' exact integrated profile assigned 122/2,505 direct EXE samples (4.87%) to the whole 16×16 AVX2 SAD4D kernel; the part saved on a hit is only a fraction of that kernel. Four lane reductions and a branch on every call would also cost CPU. This is an opportunity count, **not** a new performance or libvpx parity measurement. A later-row checkpoint is a different candidate and is not tested here.

Crucible built the diagnostic from the two source files in `8aca0210`, copied into an isolated task worktree. The source SHA-256 values were `76a8fd963f084ef36359d8798fbce5a98a8b90f4878d228f1e07e174c4776e8e` for `full_pixel.rs` and `11186f85f7d610dc89672e1657b9a0b3fae5c3dc2713f58281635b398b814fb6` for `rvp9enc.rs`. The release executable SHA-256 was `cc36c5079c389f4cdebf8bcfee52b2f50cd002f31d4fb079c5c775c3876b51c4`. The included runner checks each pinned input and 60-frame IVF SHA-256; all three outputs exactly matched the integrated baseline. `raw-stderr/` retains the original CLI counter lines, including PowerShell's native-stderr wrapping. The diagnostic's extra scalar prefix work means its CPU or wall time must not be compared with production.

The focused counter fixture deliberately failed when the four-lane predicate was changed from `all` to `any` (`(2, 2)` versus `(2, 1)`), then passed after restoration. Root reran the focused test and all 21 full-pixel tests on AArch64; core/CLI compile checks, formatting, and diff checks passed. The x86 Crucible focused fixture passed. A local two-frame enabled/disabled run produced identical IVF hashes; the pinned Crucible IVF hashes provide the full-output check. The diagnostic code remains unintegrated and the shipped SAD4D kernel is unchanged.
