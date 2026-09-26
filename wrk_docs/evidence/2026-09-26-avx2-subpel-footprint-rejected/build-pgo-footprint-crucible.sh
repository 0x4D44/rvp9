#!/usr/bin/env bash
set -euo pipefail

repo=/d/worktrees/rvp9-2/20260926-TSK-HUM-prototype-avx2-subpel-footprint-check-ho
training=/d/worktrees/rvp9-2/acceptance-d639c265/tests/corpus
cd "$repo"
test "$(git rev-parse HEAD)" = b1673592ca5aa5a3a709c4a3dcb04e549051ec8d
test "$(git diff --name-only)" = crates/rvp9-core/src/dsp/x86_64/subpel_variance_avx2.rs
test "$(sha256sum crates/rvp9-core/src/dsp/x86_64/subpel_variance_avx2.rs | cut -d' ' -f1)" = c3319a4e26d2cba3e31950b641e033c4157adda12d137f58ff2a0a8402453e06
test ! -e .scratch/encoder-pgo-footprint
shasum() {
    test "$1" = -a && test "$2" = 256
    shift 2
    sha256sum "$@"
}
export -f shasum
bash ./scripts/build-encoder-pgo.sh \
    --build-root .scratch/encoder-pgo-footprint \
    --training-dir "$training" \
    --frames 60
