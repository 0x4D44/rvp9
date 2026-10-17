#!/bin/sh
set -eu
name=${0##*/}
if [ "$#" -eq 1 ] && [ "$1" = "--help" ]; then
    printf '%s\n' "$name --help" >> "${RVP9_PREFLIGHT_HELP_LOG:?}"
    case "$name" in
        bdrate) flags="--work-dir --reference-input" ;;
        bdrate-svc) flags="--work-dir --svc-kind" ;;
        encode-headtohead) flags="--anchor-kind --temporal-layers --max-load1 --rotations --work-dir" ;;
        realtime) flags="--frame-time-csv --p99-budget-ms --budget-ms --temporal-layers" ;;
        *) flags="--help" ;;
    esac
    printf 'Usage: %s %s\n' "$name" "$flags"
    exit 0
fi
{
    printf '%s' "$name"
    for arg in "$@"; do
        printf '\t%s' "$arg"
    done
    printf '\n'
} >> "${RVP9_PREFLIGHT_NONHELP_LOG:?}"
exit 97
