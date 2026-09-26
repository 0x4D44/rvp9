#!/usr/bin/env bash
set -euo pipefail

repo=/d/worktrees/rvp9-2/20260926-TSK-HUM-prototype-diagonal-rate-bound-skip
training=/d/worktrees/rvp9-2/acceptance-d639c265/tests/corpus
cd "$repo"
test "$(git rev-parse HEAD)" = 684185fe66b18ea1176ffeac0793f8df5e783afb
test "$(git diff --name-only)" = crates/rvp9-core/src/mv_search/sub_pixel.rs
test "$(git hash-object crates/rvp9-core/src/mv_search/sub_pixel.rs)" = bbc6abe8cba37aed0073ae60b790cfcf207e2412
test ! -e .scratch/encoder-pgo-diagonal-bound
shasum() {
    test "$1" = -a && test "$2" = 256
    shift 2
    sha256sum "$@"
}
export -f shasum
bash ./scripts/build-encoder-pgo.sh \
    --build-root .scratch/encoder-pgo-diagonal-bound \
    --training-dir "$training" \
    --frames 60
