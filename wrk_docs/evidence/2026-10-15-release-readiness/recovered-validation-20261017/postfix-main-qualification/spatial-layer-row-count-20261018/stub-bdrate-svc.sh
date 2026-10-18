#!/usr/bin/env bash
set -euo pipefail
out=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --csv) out="$2"; shift 2 ;;
        *) shift ;;
    esac
done
[ -n "$out" ]
printf 'metric,layer,value\n' > "$out"
for (( layer = 0; layer < SPATIAL_BDRATE_ROWS; layer++ )); do
    printf 'bdrate_pct,%s,0.1\n' "$layer" >> "$out"
done
