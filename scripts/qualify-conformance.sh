#!/usr/bin/env bash
# Qualify the current decoder against the complete pinned 8-bit VP9 corpus.
# Every run is fail-closed on exactly 307 vectors and 4,816 displayed frames.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

./scripts/fetch-vectors.sh --jobs "${RVP9_VECTOR_FETCH_JOBS:-8}"
cargo build --release --locked --bin conformance

run_conformance() {
    target/release/conformance \
        --vectors-dir tests/conformance/vectors \
        --official \
        --strict \
        --threads "$1"
}

echo "Qualifying all official vectors with native dispatch at one thread."
run_conformance 1

echo "Qualifying all official vectors with native dispatch at two threads."
run_conformance 2

echo "Qualifying all official vectors with forced-scalar dispatch at one thread."
RVP9_FORCE_SCALAR=1 run_conformance 1

echo "Qualifying all official vectors with forced-scalar dispatch at two threads."
RVP9_FORCE_SCALAR=1 run_conformance 2

echo "Complete conformance gate passed in all four dispatch/thread modes."
