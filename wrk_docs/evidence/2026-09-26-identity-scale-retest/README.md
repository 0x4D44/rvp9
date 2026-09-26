# Identity-scale tight-plane prediction retest

The candidate is commit `544da81d` on the current-tree task branch. The x86-64
Crucible control is the code-equivalent `48cab42d` release binary (SHA-256
`c4b24b2c5b16add984a2dd6d32e5a7080037290e58b254aff838a0dcf26d63c1`);
the candidate release binary is SHA-256
`a370912808375e022575473d95cae928da3c8e03c8198c1d3fccdafb92c29c02`.
Both used the pinned three 1080p Y4M inputs and canonical IVF output hashes.

`identity-15{a,b,c}` are separate 15-pair, alternating-order screens. Each raw
row includes process CPU, elapsed time and output hash; the scripts refused a
hash mismatch. `identity-smoke` is a one-pair setup check. Crucible had 12–15
other heavy build processes during the screens, with visibly variable timing;
interpret small differences cautiously. The corresponding PowerShell runner is
in the task worktree's `.scratch/screen-identity-scale.ps1`.

The three unpinned corpus geometric candidate/control CPU ratios were 0.983521,
0.979577, and 0.988942. The clip ratios did not agree: Ducks was 0.982707,
1.033477, and 0.998495; Tree was 1.008059, 0.930002, and 0.991056;
Town was 0.960375, 0.977983, and 0.977392. An additional 15-pair screen
pinning both arms to logical CPU 0 was 1.029315 Ducks, 0.991277 Tree, and
0.995588 Town, or 1.005254 corpus. All outputs were exact. A total-CPU
spot check during the campaign read 15–25%, but pinned jobs still suffered
multi-second scheduling stalls. The apparent gain is not repeatable across
affinity conditions, so this candidate is not retained or integrated.

Focused validation: macOS AArch64 `cargo test -p rvp9-core inter::scale --lib`
(21/21), `cargo check -p rvp9-encoder`, and `cargo fmt --all --check` passed;
Crucible x86-64 `cargo test -p rvp9-core inter::scale --lib` passed (21/21).
The new differential was deliberately made red by changing one predicted
output byte, then restored and rerun green.
