#!/usr/bin/env bash
set -euo pipefail

repo=/d/worktrees/rvp9-2/20260926-TSK-HUM-prototype-tx4-vector-dc-source-reuse
training=/d/worktrees/rvp9-2/acceptance-d639c265/tests/corpus
cd "$repo"
test "$(git rev-parse HEAD)" = 5af4801fff11d78017c1fc95738395a1fe3b0978
test "$(git diff --name-only)" = crates/rvp9-core/src/dsp/x86_64/quantize_fp_avx2.rs
test ! -e .scratch/encoder-pgo-dc-source
command -v sha256sum >/dev/null
shasum() {
    test "$1" = -a && test "$2" = 256
    shift 2
    sha256sum "$@"
}
export -f shasum
bash ./scripts/build-encoder-pgo.sh \
    --build-root .scratch/encoder-pgo-dc-source \
    --training-dir "$training" \
    --frames 60
