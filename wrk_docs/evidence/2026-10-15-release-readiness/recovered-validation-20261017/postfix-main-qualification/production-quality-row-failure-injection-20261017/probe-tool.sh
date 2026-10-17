#!/usr/bin/env bash
set -u
name=${0##*/}
printf '%s\t' "$name" >> "$RVP9_TEST_LOG"
printf '%s\n' "$*" >> "$RVP9_TEST_LOG"
if [ "${1-}" = "--help" ]; then
    printf '%s\n' '--work-dir' '--reference-input' '--svc-kind' '--anchor-kind' '--temporal-layers' '--max-load1' '--rotations' '--frame-time-csv' '--p99-budget-ms' '--budget-ms'
    exit 0
fi
case "$name" in
    bdrate)
        if [ "${RVP9_TEST_MODE-}" = "bdrate-fail" ]; then
            printf 'injected bdrate child failure: exit 73\n' >&2
            exit 73
        fi
        [ "${RVP9_TEST_MODE-}" = "decoder-mismatch" ] || exit 97
        work=""
        csv=""
        while [ "$#" -gt 0 ]; do
            case "$1" in
                --work-dir) work="$2"; shift 2 ;;
                --csv) csv="$2"; shift 2 ;;
                *) shift ;;
            esac
        done
        [ -n "$work" ] && [ -n "$csv" ] || exit 96
        for rate in 420 630 840 1260 2100 3360; do
            printf 'synthetic test stream %s\n' "$rate" > "$work/test_${rate}k.ivf"
        done
        printf 'bdrate_pct,0.400000\n' > "$csv"
        exit 0
        ;;
    ffmpeg)
        case "${1-}" in
            -version|--version) printf 'synthetic ffmpeg stub\n'; exit 0 ;;
        esac
        last="${@: -1}"
        [ "$last" != "-" ] || exit 0
        out="$last"
        name_out=${out##*/}
        case "$name_out" in
            *_720p30_*) target=720p30; width=1280; height=720; fps=30 ;;
            *_1080p30_*) target=1080p30; width=1920; height=1080; fps=30 ;;
            *_1080p60_*) target=1080p60; width=1920; height=1080; fps=60 ;;
            *) printf 'unexpected ffmpeg output path: %s\n' "$out" >&2; exit 97 ;;
        esac
        template="$RVP9_TEMPLATE_DIR/$target.y4m"
        if [ ! -e "$template" ]; then
            header="YUV4MPEG2 W${width} H${height} F${fps}:1 Ip A1:1 C420jpeg"
            frame_bytes=$(( width * height * 3 / 2 ))
            expected=$(( ${#header} + 1 + 60 * (6 + frame_bytes) ))
            printf '%s\n' "$header" > "$template"
            truncate -s "$expected" "$template"
        fi
        ln "$template" "$out"
        ;;
    rvp9dec) printf 'rvp9-decoded-i420\n' ;;
    vpxdec) printf 'vpx-decoded-i420\n' ;;
    *) printf 'unexpected non-help invocation for %s: %s\n' "$name" "$*" >&2; exit 97 ;;
esac
