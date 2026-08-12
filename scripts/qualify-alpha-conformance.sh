#!/usr/bin/env bash
# Qualify the v0.2.0-alpha.1 decoder surface against the complete pinned
# official 8-bit VP9 corpus, excluding only the separately tracked spatial-SVC
# stream (RVP9-BUG-FLUX-00022).

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

./scripts/fetch-vectors.sh --jobs "${RVP9_VECTOR_FETCH_JOBS:-8}"
cargo build --release --locked --bin conformance

run_conformance() {
    target/release/conformance \
        --vectors-dir tests/conformance/vectors \
        --strict \
        --alpha-supported
}

echo "Qualifying alpha-supported official vectors with native dispatch."
run_conformance

echo "Qualifying alpha-supported official vectors with scalar dispatch."
RVP9_FORCE_SCALAR=1 run_conformance

echo "Alpha conformance gate passed in native and forced-scalar modes."
