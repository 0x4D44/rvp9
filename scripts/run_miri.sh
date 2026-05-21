#!/usr/bin/env bash
# scripts/run_miri.sh — bash port of scripts/run_miri.ps1.
# See the PowerShell version for the full rationale (why a subset,
# which tests are excluded, what the 2026-05-21 baseline is).
#
# Usage from the repo root:
#   ./scripts/run_miri.sh

set -euo pipefail

MODULES=(
    'transforms::'
    'intra_pred::'
    'quant::'
    'dsp::'
)

echo "Running miri across ${#MODULES[@]} module subsets in rvp9-core..."
for m in "${MODULES[@]}"; do
    echo ""
    echo "--- miri: rvp9-core::$m ---"
    cargo +nightly miri test --package rvp9-core --lib "$m"
done
echo ""
echo "All miri module subsets passed."
