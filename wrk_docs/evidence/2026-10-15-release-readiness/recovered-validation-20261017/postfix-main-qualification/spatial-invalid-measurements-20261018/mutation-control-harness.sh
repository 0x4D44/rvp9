#!/usr/bin/env bash
set -euo pipefail
readonly FRAMES=60
readonly KEYFRAME_INTERVAL=60
readonly AQ_MODE=0
readonly NOISE_SENSITIVITY=0
readonly TEMPORAL_LAYERS=1
readonly INTER_LAYER_PRED=0
readonly DEFAULT_MAX_BDRATE_PCT="0.5"
readonly EXPERIMENTAL_SPATIAL_VBR_MAX_BDRATE_PCT="25.0"
readonly QUALITY_CURVE_TIMEOUT=7200
readonly EXPECTED_QUALITY_CURVES=96

target_spec() {
    case "$1" in
        720p30)  printf '30|1280|720|1|0|1260|33.333|50\n' ;;
        1080p30) printf '30|1920|1080|4|2|2835|33.333|50\n' ;;
        1080p60) printf '60|1920|1080|8|3|5670|16.667|25\n' ;;
        *) printf 'unknown target: %s\n' "$1" >&2; return 1 ;;
    esac
}
target_bitrates() {
    case "$1/$2" in
        720p30/2)  printf '420,630,840,1260,2100,3360\n' ;;
        720p30/3)  printf '420,630,840,1260,2100,3360\n' ;;
        1080p30/2) printf '950,1425,1890,2835,4725,7560\n' ;;
        1080p30/3) printf '945,1428,1890,2835,4725,7560\n' ;;
        1080p60/2) printf '1900,2850,3780,5670,9450,15120\n' ;;
        1080p60/3) printf '1890,2856,3780,5670,9450,15120\n' ;;
        *) printf 'unknown target/layers: %s/%s\n' "$1" "$2" >&2; return 1 ;;
    esac
}
quality_bdrate_limit() {
    local mode="$1"
    if [ "$mode" = "vbr" ]; then
        printf '%s\n' "$EXPERIMENTAL_SPATIAL_VBR_MAX_BDRATE_PCT"
    else
        printf '%s\n' "$DEFAULT_MAX_BDRATE_PCT"
    fi
}
fail() {
    printf '%s\n' "$*" >&2
    exit 1
}
require_number() {
    :
}
number_le() {
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 <= b + 0) }'
}
acceptance_input() {
    printf '%s/%s_%s.y4m\n' "$input_dir" "$1" "$2"
}
run_quality_curve() {
    local mode="$1" target="$2" clip="$3" cpu="$4" layers="$5"
    local fps width height threads tiles speed_kbps p50 p99
    local grid curve_id dir streams input recorded layer value status gate

    IFS='|' read -r fps width height threads tiles speed_kbps p50 p99 \
        <<< "$(target_spec "$target")"
    grid="$(target_bitrates "$target" "$layers")"
    gate="$(quality_bdrate_limit "$mode" "$cpu" "$layers")"
    curve_id="${mode}-${target}-${clip}-cpu${cpu}-s${layers}"
    dir="$quality_dir/$curve_id"
    streams="$dir/streams"
    input="$(acceptance_input "$clip" "$target")"
    mkdir -p "$streams"

    quality_curve_count=$(( quality_curve_count + 1 ))
    printf '[quality %3d/%d] %s\n' \
        "$quality_curve_count" "$EXPECTED_QUALITY_CURVES" "$curve_id"

    if ! deltic timeout "$QUALITY_CURVE_TIMEOUT" "$bdrate_svc" \
        --anchor-encoder "$spatial_anchor" \
        --test-encoder "$test_encoder" \
        --decoder "$vpxdec" \
        --input "$input" \
        --svc-kind spatial \
        --spatial-layers "$layers" \
        --inter-layer-pred "$INTER_LAYER_PRED" \
        --temporal-layers "$TEMPORAL_LAYERS" \
        --cpu-used "$cpu" \
        --end-usage "$mode" \
        --threads "$threads" \
        --tile-columns "$tiles" \
        --aq-mode "$AQ_MODE" \
        --noise-sensitivity "$NOISE_SENSITIVITY" \
        --limit "$FRAMES" \
        --keyframe-interval "$KEYFRAME_INTERVAL" \
        --bitrates "$grid" \
        --max-bdrate-pct "$gate" \
        --work-dir "$streams" \
        --csv "$dir/bdrate.csv" \
        > "$dir/bdrate.log" 2>&1 </dev/null
    then
        fail "$curve_id: bdrate-svc failed or a BD-rate gate did not pass (log $dir/bdrate.log)"
    fi

    # Re-read the per-layer BD-rate values and gate them here as well, so the
    # summary cannot record a pass the numbers do not support.
    recorded=0
    while IFS=',' read -r layer value; do
        require_number "$curve_id SL$layer BD-rate" "$value"
        status="PASS"
        if ! number_le "$value" "$gate"; then
            status="FAIL"
        fi
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,SL%s,%s,%s,%s\n' \
            "$curve_id" "$mode" "$target" "$clip" "$cpu" "$layers" "$threads" \
            "$tiles" "$grid" "$layer" "$value" "$gate" "$status" \
            >> "$quality_dir/quality-summary.csv"
        [ "$status" = "PASS" ] ||
            fail "$curve_id: SL$layer BD-rate $value% exceeds +$gate%"
        recorded=$(( recorded + 1 ))
        quality_gate_count=$(( quality_gate_count + 1 ))
    done < <(awk -F, '$1 == "bdrate_pct" { print $2 "," $3 }' "$dir/bdrate.csv")

    if [ "$recorded" -ne "$layers" ]; then
        fail "$curve_id: recorded $recorded layer gates, expected $layers"
    fi

    cross_check_curve "$curve_id" "$streams" "$layers" "$grid" \
        "$dir/cross-decoder.log" "$quality_dir/cross-decoder.csv"
    discard_curve_evidence "$streams"
}

set -euo pipefail
quality_curve_count=0
quality_gate_count=0
quality_dir="$SPATIAL_CASE_DIR/quality"
input_dir="$SPATIAL_CASE_DIR/inputs"
spatial_anchor="$SPATIAL_CASE_DIR/anchor"
test_encoder="$SPATIAL_CASE_DIR/test-encoder"
vpxdec="$SPATIAL_CASE_DIR/vpxdec"
bdrate_svc="$1"
mkdir -p "$quality_dir"
printf 'curve_id,mode,target,clip,cpu_used,spatial_layers,threads,tile_columns,bitrates_kbps,layer,bdrate_pct,gate_pct,status\n' > "$quality_dir/quality-summary.csv"
cross_check_curve() {
    : > "$SPATIAL_CASE_DIR/cross-check-called"
}
discard_curve_evidence() {
    :
}
run_quality_curve cbr 720p30 in_to_tree 6 2
