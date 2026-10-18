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
printf 'bdrate_pct,0,0.1\n' >> "$out"
printf 'bdrate_pct,1,0.1\n' >> "$out"
exit 73
