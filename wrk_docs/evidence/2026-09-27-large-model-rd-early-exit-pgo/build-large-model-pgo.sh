#!/usr/bin/env bash
set -euo pipefail

repo=/d/worktrees/rvp9-2/20260927-TSK-HUM-validate-early-large-model-rd-exit-on-cr
training=/d/worktrees/rvp9-2/acceptance-d639c265/tests/corpus
cd "$repo"
test "$(git rev-parse origin/main)" = 9b4a35f95127c737be301e40f12557d6f685f065
test "$(sha256sum crates/rvp9-core/src/mode_decision/model_rd.rs | cut -d ' ' -f 1)" = 4417907c441e6b33ef26d0768a77a0f030d465ddf11df93ec7338731d20c3300
test ! -e .scratch/large-model-pgo
shasum() {
    test "$1" = -a && test "$2" = 256
    shift 2
    sha256sum "$@"
}
export -f shasum
bash ./scripts/build-encoder-pgo.sh \
    --build-root .scratch/large-model-pgo \
    --training-dir "$training" \
    --frames 60
