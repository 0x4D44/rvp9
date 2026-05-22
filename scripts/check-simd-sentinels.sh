#!/usr/bin/env bash
# G4.1 — source-side aligned-load discipline check.
#
# In every `*.rs` kernel file under `crates/rvp9-core/src/dsp/x86_64`
# (the AVX2 / SSE2 / SSSE3 / SSE4.1 impls, excluding `mod.rs`), reads
# from the `dst` buffer MUST use UNALIGNED load intrinsics
# (`_mm256_loadu_si256` / `_mm_loadu_si128` / `_mm_loadl_epi64`),
# because `dst` is not guaranteed 16/32-byte aligned and an aligned
# load on an unaligned pointer faults.
#
# Rather than mandate a sentinel comment above every `unsafe fn` (most
# kernels are store-only to `dst` and legitimately never load from it),
# this check detects the actual violation directly: it FAILS if any
# kernel file calls an ALIGNED load intrinsic:
#
#     _mm256_load_si256  _mm_load_si128   _mm_load_si64
#     _mm256_load_ps     _mm_load_ps
#     _mm256_load_pd     _mm_load_pd
#
# The safe unaligned variants (`loadu` / `loadl`) are NOT flagged: the
# pattern anchors on the exact intrinsic name with a word boundary so
# `_mm_load_si128` does not match `_mm_loadu_si128` or `_mm_loadl_epi64`.
#
# Usage:
#   scripts/check-simd-sentinels.sh

set -euo pipefail

DIR="crates/rvp9-core/src/dsp/x86_64"
RC=0

if [ ! -d "$DIR" ]; then
    echo "ERROR: directory $DIR does not exist"
    exit 2
fi

# Aligned-load intrinsics that must never appear in a kernel. The
# boundary is enforced with explicit non-word-character classes
# `(^|[^A-Za-z0-9_])...([^A-Za-z0-9_]|$)` (portable across awk variants
# where `\b` / `\<` are unreliable in regexes), so e.g. `_mm_load_si128`
# matches `_mm_load_si128(` but never `_mm_loadu_si128` /
# `_mm_loadl_epi64`. The pattern is embedded literally in the awk
# program (passing it via `-v` would strip the backslashes).
for f in "$DIR"/*.rs; do
    base="$(basename "$f")"
    if [ "$base" = "mod.rs" ]; then
        continue
    fi
    awk -v file="$f" '
        BEGIN { rc = 0 }
        {
            if (match($0, /(^|[^A-Za-z0-9_])(_mm256_load_si256|_mm_load_si128|_mm256_load_ps|_mm_load_ps|_mm256_load_pd|_mm_load_pd|_mm_load_si64)([^A-Za-z0-9_]|$)/)) {
                m = substr($0, RSTART, RLENGTH)
                # Strip the boundary chars captured on either side so the
                # report shows only the intrinsic name.
                gsub(/^[^A-Za-z0-9_]+|[^A-Za-z0-9_]+$/, "", m)
                printf "ALIGNED load: %s:%d uses %s (use the unaligned variant: loadu/loadl — `dst` is not guaranteed aligned)\n", \
                       file, NR, m
                rc = 1
            }
        }
        END { exit rc }
    ' "$f" || RC=1
done

if [ "$RC" -eq 0 ]; then
    echo "OK: no aligned-load intrinsics under $DIR; all SIMD loads use unaligned (loadu/loadl) variants."
fi

exit "$RC"
