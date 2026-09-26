#!/usr/bin/env bash
set -euo pipefail

repo=/d/worktrees/rvp9-2/20260926-TSK-HUM-prototype-nstep-duplicate-variance-score
training=/d/worktrees/rvp9-2/acceptance-d639c265/tests/corpus
cd "$repo"
test "$(git rev-parse HEAD)" = 8c5df486b3f60f0bdcad0ee88c683370ccf6967f
test "$(git diff --name-only)" = crates/rvp9-core/src/mv_search/full_pixel.rs
test ! -e .scratch/encoder-pgo-nstep
command -v sha256sum >/dev/null
shasum() {
    test "$1" = -a && test "$2" = 256
    shift 2
    sha256sum "$@"
}
export -f shasum
bash ./scripts/build-encoder-pgo.sh \
    --build-root .scratch/encoder-pgo-nstep \
    --training-dir "$training" \
    --frames 60
