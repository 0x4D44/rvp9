#!/usr/bin/env bash
set -euo pipefail

repo=/d/worktrees/rvp9-2/20260926-TSK-HUM-prototype-nstep-prepared-initial-seed-re
training=/d/worktrees/rvp9-2/acceptance-d639c265/tests/corpus
cd "$repo"
test "$(git rev-parse HEAD)" = 84d5e0fea20ca2c5033644c03d01d2f2d4bd87b7
test "$(git diff --name-only)" = crates/rvp9-core/src/mv_search/full_pixel.rs
test "$(sha256sum crates/rvp9-core/src/mv_search/full_pixel.rs | cut -d' ' -f1)" = 365373e7d4536296b816e4a4e3371143c933cfb6b5311fce0b58f68eda76cf92
test ! -e .scratch/encoder-pgo-seed
shasum() {
    test "$1" = -a && test "$2" = 256
    shift 2
    sha256sum "$@"
}
export -f shasum
bash ./scripts/build-encoder-pgo.sh \
    --build-root .scratch/encoder-pgo-seed \
    --training-dir "$training" \
    --frames 60
