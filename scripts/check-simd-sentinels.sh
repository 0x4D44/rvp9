#!/usr/bin/env bash
# G4.1 — source-side aligned-load discipline check (per-function granularity).
#
# Every `unsafe fn` in `crates/rvp9-core/src/dsp/x86_64/*.rs` (the AVX2 /
# SSE2 / SSSE3 / SSE4.1 kernel impls) MUST use unaligned-load intrinsics
# for reads from `dst`. The convention is documented by a grep-able
# sentinel comment of the form:
#
#     // G4: loads from `dst` use _mm256_loadu_si256 (unaligned)
#
# (or the equivalent for SSE2 / SSSE3 — `_mm_loadu_si128` /
# `_mm_loadl_epi64`).
#
# **Per-function check:** for every `unsafe fn <name>(...)` declaration
# under the kernel directory, this script requires a sentinel comment
# either on the line immediately preceding the `unsafe fn` line, or in
# the contiguous block of comment / attribute lines that precedes it.
# This catches the failure mode where a second `unsafe fn` is appended
# to an existing kernel file but the author forgets to add a sentinel
# above it (the original "one sentinel per file" check would pass
# silently).
#
# Usage:
#   scripts/check-simd-sentinels.sh

set -euo pipefail

DIR="crates/rvp9-core/src/dsp/x86_64"
SENTINEL='G4: loads from `dst` use'
RC=0

if [ ! -d "$DIR" ]; then
    echo "ERROR: directory $DIR does not exist"
    exit 2
fi

# Walk every .rs file in the kernel dir except mod.rs.
for f in "$DIR"/*.rs; do
    base="$(basename "$f")"
    if [ "$base" = "mod.rs" ]; then
        continue
    fi
    # Find every `unsafe fn <name>(` declaration in the file and
    # confirm the contiguous block of comment/attribute lines that
    # precedes it contains the sentinel string. A "block" is any run
    # of lines starting with one of: `//`, `#[`, `#![`, or blank
    # whitespace.
    awk -v file="$f" -v sentinel="$SENTINEL" '
        BEGIN { rc = 0 }
        # Match any "unsafe fn <ident>(" at any indent. Allow leading
        # `pub ` / `pub(crate) ` etc. modifiers.
        /^[[:space:]]*(pub[[:space:]]*(\([^)]*\)[[:space:]]*)?)?unsafe[[:space:]]+fn[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\(/ {
            # Walk backwards through the buffered preceding lines
            # looking for the sentinel inside the contiguous comment /
            # attribute / blank block.
            found = 0
            for (i = NR - 1; i >= 1; i--) {
                line = lines[i]
                # Stop scanning once we hit a non-comment / non-attr /
                # non-blank line (e.g. a previous fn body close brace).
                if (line !~ /^[[:space:]]*$/ &&
                    line !~ /^[[:space:]]*\/\// &&
                    line !~ /^[[:space:]]*#\[/ &&
                    line !~ /^[[:space:]]*#!\[/) {
                    break
                }
                if (index(line, sentinel) > 0) {
                    found = 1
                    break
                }
            }
            if (!found) {
                # Strip leading whitespace for a tidier error message.
                trimmed = $0
                sub(/^[[:space:]]+/, "", trimmed)
                printf "MISSING sentinel: %s:%d before `%s` (expected `// %s ...` in the preceding comment block)\n", \
                       file, NR, trimmed, sentinel
                rc = 1
            }
        }
        { lines[NR] = $0 }
        END { exit rc }
    ' "$f" || RC=1
done

if [ "$RC" -eq 0 ]; then
    echo "OK: every \`unsafe fn\` under $DIR carries a preceding G4 unaligned-load sentinel."
fi

exit "$RC"
