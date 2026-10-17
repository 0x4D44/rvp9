#!/bin/sh
printf '%s\t' "${0##*/}" >> "$RVP9_STUB_LOG"
printf '%s\n' "$*" >> "$RVP9_STUB_LOG"
if [ "${1-}" = "--help" ]; then
    printf '%s\n' '--work-dir' '--reference-input' '--svc-kind' '--anchor-kind' '--temporal-layers' '--max-load1' '--rotations' '--frame-time-csv' '--p99-budget-ms' '--budget-ms'
    exit 0
fi
printf 'unexpected stub invocation: %s %s\n' "${0##*/}" "$*" >&2
exit 97
