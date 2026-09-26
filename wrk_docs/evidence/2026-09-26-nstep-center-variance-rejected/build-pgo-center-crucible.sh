#!/usr/bin/env bash
set -euo pipefail

repo=/d/worktrees/rvp9-2/20260926-TSK-HUM-reuse-nstep-winner-variance-at-subpixel
training=/d/worktrees/rvp9-2/acceptance-d639c265/tests/corpus
cd "$repo"
test "$(git rev-parse HEAD)" = 02498a1da801f7a3b8a664459a000bd1b9098d91
test "$(git hash-object crates/rvp9-core/src/mv_search/full_pixel.rs)" = d74234250540fe2426186b48e5c151f714104c09
test "$(git hash-object crates/rvp9-core/src/mv_search/sub_pixel.rs)" = 058f838dc6d0eb5012f4fb02e8064be33207af98
test "$(git hash-object crates/rvp9-core/src/mode_decision/inter.rs)" = 491702ba0d540f1a643dc114939310beca6aa74c
test ! -e .scratch/encoder-pgo-center-variance
shasum() {
    test "$1" = -a && test "$2" = 256
    shift 2
    sha256sum "$@"
}
export -f shasum
bash ./scripts/build-encoder-pgo.sh \
    --build-root .scratch/encoder-pgo-center-variance \
    --training-dir "$training" \
    --frames 60
