#!/usr/bin/env bash
set -euo pipefail

repo=/d/worktrees/rvp9-2/20260927-TSK-HUM-prototype-exact-newmv-seed-sad-reuse-cru
training=/d/worktrees/rvp9-2/acceptance-d639c265/tests/corpus
cd "$repo"
test "$(git rev-parse HEAD)" = 46d114c3eea93eb2e875973754998dec326fe63d
test "$(git diff --name-only)" = crates/rvp9-core/src/mode_decision/inter.rs
test ! -e .scratch/encoder-pgo-seed-sad
command -v sha256sum >/dev/null
shasum() {
    test "$1" = -a && test "$2" = 256
    shift 2
    sha256sum "$@"
}
export -f shasum
bash ./scripts/build-encoder-pgo.sh \
    --build-root .scratch/encoder-pgo-seed-sad \
    --training-dir "$training" \
    --frames 60
