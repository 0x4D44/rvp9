#!/usr/bin/env bash

set -euo pipefail

readonly BITRATE_POINTS=6

readonly DECODERS_PER_STREAM=3

readonly EMPTY_SHA256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

readonly DECODE_TIMEOUT=900

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

sha256_file() {
    shasum -a 256 "$1" | awk '{ print $1 }'
}

decode_sha_rvp9() {
    deltic timeout "$DECODE_TIMEOUT" "$rvp9dec" --codec=vp9 --i420 "$1" \
        </dev/null 2>>"$2" |
        shasum -a 256 | awk '{ print $1 }'
}

decode_sha_vpx() {
    deltic timeout "$DECODE_TIMEOUT" "$vpxdec" --codec=vp9 --i420 -o - "$1" \
        </dev/null 2>>"$2" |
        shasum -a 256 | awk '{ print $1 }'
}

cross_check_stream() {
    local curve_id="$1" ivf="$2" kind="$3" log="$4" csv="$5"
    local bytes stream_sha rvp9_sha vpx_sha ffmpeg_status

    [ -s "$ivf" ] || fail "$curve_id: expected stream is missing or empty: $ivf"
    bytes="$(wc -c < "$ivf" | tr -d ' ')"
    stream_sha="$(sha256_file "$ivf")"

    if ! rvp9_sha="$(decode_sha_rvp9 "$ivf" "$log")"; then
        fail "$curve_id: rvp9dec failed on $ivf (log $log)"
    fi
    if ! vpx_sha="$(decode_sha_vpx "$ivf" "$log")"; then
        fail "$curve_id: pinned vpxdec failed on $ivf (log $log)"
    fi
    ffmpeg_status="ok"
    if ! deltic timeout "$DECODE_TIMEOUT" "$ffmpeg_bin" \
        -hide_banner -loglevel error -nostdin -c:v vp9 -i "$ivf" -f null - \
        >>"$log" 2>&1
    then
        ffmpeg_status="failed"
    fi
    cross_run_count=$(( cross_run_count + DECODERS_PER_STREAM ))
    cross_stream_count=$(( cross_stream_count + 1 ))

    local identical="yes" status="PASS"
    [ "$rvp9_sha" = "$vpx_sha" ] || { identical="no"; status="FAIL"; }
    [ "$ffmpeg_status" = "ok" ] || status="FAIL"
    if [ "$rvp9_sha" = "$EMPTY_SHA256" ] || [ "$vpx_sha" = "$EMPTY_SHA256" ]; then
        identical="empty-decode"
        status="FAIL"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$curve_id" "$(basename "$ivf")" "$kind" "$bytes" "$stream_sha" \
        "$rvp9_sha" "$vpx_sha" "$identical" "$ffmpeg_status" "$status" >> "$csv"

    if [ "$status" != "PASS" ]; then
        fail "$curve_id: cross-decoder mismatch on $(basename "$ivf") (rvp9dec=$rvp9_sha vpxdec=$vpx_sha ffmpeg=$ffmpeg_status)"
    fi
}

cross_check_curve() {
    local curve_id="$1" streams="$2" layers="$3" grid="$4" log="$5" csv="$6"
    local rate layer expected actual

    expected=$(( BITRATE_POINTS * (1 + layers) ))
    actual="$(find "$streams" -maxdepth 1 -type f -name 'test_*.ivf' | awk 'END { print NR + 0 }')"
    if [ "$actual" -ne "$expected" ]; then
        fail "$curve_id: found $actual rvp9 test streams, expected $expected"
    fi

    for rate in $(printf '%s' "$grid" | tr ',' ' '); do
        cross_check_stream "$curve_id" "$streams/test_${rate}k_full.ivf" full "$log" "$csv"
        for (( layer = 0; layer < layers; layer++ )); do
            cross_check_stream "$curve_id" "$streams/test_${rate}k_sl${layer}.ivf" \
                "sl$layer" "$log" "$csv"
        done
    done
}

rvp9dec="$1"

vpxdec="$2"

ffmpeg_bin="$3"

streams="$4"

decode_log="$5"

cross_csv="$6"

cross_run_count=0

cross_stream_count=0

mkdir -p "$streams"

printf 'curve_id,stream,kind,bytes,stream_sha256,rvp9dec_sha256,vpxdec_sha256,decoders_identical,ffmpeg_decode,status\n' > "$cross_csv"

: > "$decode_log"

cross_check_curve "synthetic-spatial-2layer" "$streams" 2 "420,630,840,1260,2100,3360" "$decode_log" "$cross_csv"

printf "UNEXPECTED_SUCCESS\n"
