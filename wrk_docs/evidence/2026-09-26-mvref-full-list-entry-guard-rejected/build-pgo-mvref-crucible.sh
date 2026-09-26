#!/usr/bin/env bash
set -euo pipefail

repo=/d/worktrees/rvp9-2/20260926-TSK-HUM-prototype-full-list-entry-guard-x86-vali
training=/d/worktrees/rvp9-2/acceptance-d639c265/tests/corpus
cd "$repo"
test "$(git rev-parse HEAD)" = 865e1fd4edf0ed09d26bcf75720d8eed0b19d12f
test "$(git diff --name-only)" = crates/rvp9-core/src/inter/mv_ref.rs
test "$(sha256sum crates/rvp9-core/src/inter/mv_ref.rs | cut -d ' ' -f 1)" = ebf0d2ef70ea0fc4e7fa031dbb4c6d546419a00c15414633a9116ca89e4ba71d
test ! -e .scratch/encoder-pgo-mvref
shasum() {
    test "$1" = -a && test "$2" = 256
    shift 2
    sha256sum "$@"
}
export -f shasum
bash ./scripts/build-encoder-pgo.sh \
    --build-root .scratch/encoder-pgo-mvref \
    --training-dir "$training" \
    --frames 60
