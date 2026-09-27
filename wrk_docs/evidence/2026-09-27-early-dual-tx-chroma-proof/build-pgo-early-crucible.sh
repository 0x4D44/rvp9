#!/usr/bin/env bash
set -euo pipefail

repo=/d/worktrees/rvp9-2/20260927-TSK-HUM-validate-early-dual-tx-chroma-proof-on-c
training=/d/worktrees/rvp9-2/acceptance-d639c265/tests/corpus
cd "$repo"
test "$(git rev-parse --short HEAD)" = 0cc6405da
git diff --quiet
test ! -e .scratch/encoder-pgo-early
command -v sha256sum >/dev/null
shasum() {
    test "$1" = -a && test "$2" = 256
    shift 2
    sha256sum "$@"
}
export -f shasum
bash ./scripts/build-encoder-pgo.sh \
    --build-root .scratch/encoder-pgo-early \
    --training-dir "$training" \
    --frames 60
