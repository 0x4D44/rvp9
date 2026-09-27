#!/usr/bin/env bash
set -euo pipefail

repo=/d/worktrees/rvp9-2/20260927-TSK-HUM-prototype-checked-subpel-source-footprin
training=/d/worktrees/rvp9-2/acceptance-d639c265/tests/corpus
cd "$repo"
test "$(git rev-parse HEAD)" = f18a4e61e27d6b5b021d946b7f09da24fa237e2c
test "$(git diff --name-only)" = crates/rvp9-core/src/dsp/x86_64/subpel_variance_avx2.rs
test ! -e .scratch/encoder-pgo-subpel-footprint
command -v sha256sum >/dev/null
shasum() {
    test "$1" = -a && test "$2" = 256
    shift 2
    sha256sum "$@"
}
export -f shasum
bash ./scripts/build-encoder-pgo.sh \
    --build-root .scratch/encoder-pgo-subpel-footprint \
    --training-dir "$training" \
    --frames 60
