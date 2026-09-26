# Shared first-round half-pel cardinals — rejected

Code candidate `c251d0d0` is isolated on the task branch, not integrated. It
shares 33-column and 33-row bilinear strips across the four first-round 32×32
cardinal probes, then computes four separate exact variance/MV scores in the
existing comparison order. Other shapes, phases, borders, malformed inputs,
and non-AVX2 dispatch use the old path. The candidate is distinct from the
rejected Unit 1376 full-width-load edit.

The source files used by Crucible matched the local candidate SHA-256s:
`sub_pixel.rs` `e32d698b166cdbbae2c1dc5a00004478b409291f33f2b4b7ab3003e22e0a6212`;
`subpel_variance_avx2.rs` `d59f9ee890947d8ad4ed463f051635a800b2ec8d2c84ba3f961b6569f479c73e`.
Native release candidate EXE SHA-256:
`77dd28f592ee7931e693ef9f078a6540ee6a24f5d77edc3ea341c7e0ae37e912`.
The current integrated control EXE SHA-256 was
`c4b24b2c5b16add984a2dd6d32e5a7080037290e58b254aff838a0dcf26d63c1`.

Both runs used pinned 60-frame Ducks, Tree, and Town inputs at cpu-used 6,
one thread, CBR 1260 kbps, alternating arm and clip order, one warmup per
clip and arm, and 15 paired process-CPU samples per clip. The runner checked
both EXE and input hashes before launching and checked every IVF against the
canonical output hash. All **192 encodes** completed with exact output.
The captured `identity.csv`, `raw.csv`, and `summary.csv` for each run and
the exact `screen-shared-cardinals.ps1` runner are here. The execution copy
remains at `D:\worktrees\rvp9-2\20260926-TSK-HUM-crucible-profile-after-zero-eob\.scratch\full-lane-prototype\screen-shared-cardinals.ps1`.

| Run | Ducks candidate/control CPU | Tree | Town | Equal-clip geometric mean |
| --- | ---: | ---: | ---: | ---: |
| 15a | 1.012366× | 0.996987× | 1.039122× | 1.016010× |
| 15b | 1.061772× | 1.017514× | 1.022524× | 1.033749× |

Run 15a saw several multi-second wall-time scheduling stalls and large
process-CPU outliers; run 15b was steadier. The candidate does not show a
repeatable whole-encode gain and is slower on the corpus in both screens.
Reject it; do not integrate the code. The Tree 15a 0.3% apparent edge is
below the measurement spread and reversed in 15b. No libvpx parity screen is
warranted for a candidate that loses against integrated RVP9. The last pinned
RVP9/libvpx equal-clip CPU ratio remains 1.134647×, not a new measurement.

Focused correctness checks before screening: AArch64 `cargo test -p rvp9-core
shared_first_round`, the 11 `sub_pixel_search` tests and `cargo fmt --all
--check`; Crucible x86 both `shared_first_round` tests. A temporary +1 variance
mutation made the candidate-0 metric differential fail (`9007873` vs
`9007872`), then the restored candidate passed. The x86 test asserts AVX2
dispatch rather than silently skipping it.
