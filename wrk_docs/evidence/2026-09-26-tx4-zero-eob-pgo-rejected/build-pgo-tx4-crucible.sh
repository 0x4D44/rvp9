#!/usr/bin/env bash
set -euo pipefail

repo=/d/worktrees/rvp9-2/20260926-TSK-HUM-prototype-tx4-zero-eob-avx2-screen
training=/d/worktrees/rvp9-2/acceptance-d639c265/tests/corpus
cd "$repo"
test "$(git rev-parse HEAD)" = 443be7786d203399a46666413cc964b978d1a5a0
test "$(git diff --name-only)" = crates/rvp9-core/src/dsp/x86_64/quantize_fp_avx2.rs
test ! -e .scratch/encoder-pgo-tx4
command -v sha256sum >/dev/null
shasum() {
    test "$1" = -a && test "$2" = 256
    shift 2
    sha256sum "$@"
}
export -f shasum
bash ./scripts/build-encoder-pgo.sh \
    --build-root .scratch/encoder-pgo-tx4 \
    --training-dir "$training" \
    --frames 60
