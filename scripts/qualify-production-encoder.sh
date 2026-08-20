#!/usr/bin/env bash
# Declare the complete non-spatial v1 production acceptance matrix for the
# rvp9 encoder from one checked-in orchestrator.
#
# Scope: this file is the *matrix definition* plus its two dry oracles. The
# quality, relative-speed, and absolute-latency execution phases are NOT
# implemented in this unit; an ordinary invocation fails loudly and says so.
#
# Spatial SVC is deliberately absent. It is qualified separately and on its
# own experimental terms by scripts/qualify-spatial-svc.sh; no spatial row,
# field, or layer count appears here.
#
# Feature axes (never combined -- one feature per row):
#   baseline     CBR and VBR, CPU6/7/8, AQ0, denoiser0, 1 temporal layer
#   aq3          VBR only, CPU7 only, AQ3, denoiser0, 1 temporal layer
#   denoiser     CBR and VBR, CPU6/7/8, AQ0, denoiser1, 1 temporal layer
#   temporal_t2  CBR and VBR, CPU6/7/8, AQ0, denoiser0, 2 temporal layers
#   temporal_t3  CBR and VBR, CPU6/7/8, AQ0, denoiser0, 3 temporal layers
#
# The denoiser row set covers BOTH rate-control modes on purpose. Stage 4c's
# focused evidence was VBR, while the final v1 contract includes both rate
# control modes; CBR denoiser rows are first-class rather than an optional
# extra.
#
# Exact totals this matrix must produce (asserted by --self-test, and by
# --print-plan against its own iterators):
#
#     200 quality curves        6 bitrate points each
#     129 corpus-aggregate per-layer BD-rate gates
#     344 important-clip per-layer guards
#   2,640 candidate streams     retained for cross-decoder checking
#   7,920 decoder runs          3 decoders per retained stream
#      75 relative-speed corpus groups   feature/mode/cpu/target
#     200 latency rows
#   1,000 measured latency runs  5 measured runs per row
#
# Latency coverage is intentionally broad: every quality configuration is
# measured on every clip of its target, not on one representative clip. A
# feature that only misses its frame budget on the hardest clip is exactly
# the failure a single-clip sample hides, and the marginal cost of a latency
# run is small next to a quality curve.
#
# Gates:
#   quality  arithmetic mean of the clip BD-rates <= +0.5% per layer versus
#            pinned vpxenc for every feature/mode/cpu/target group. No
#            important clip/layer may show an unexplained regression beyond
#            +2.0%.
#   aq3      additionally the existing AQ gain-parity oracle: both AQ gains
#            finite and negative, and |gain_rvp9 - gain_libvpx| <= 0.3
#            percentage points. That oracle is a separate required check,
#            not a second BD-rate curve.
#   speed    <= 1.05x pinned vpxenc wall time, host load1 <= 0.75, one warmup
#            plus five measured pairs over two reversed rotations.
#   latency  median p50 and median p99 inside the per-target frame budgets
#            over five measured runs; AQ3 rows additionally cap max frame
#            time at 66 ms.
#
#   scripts/qualify-production-encoder.sh --print-plan   # dry oracle, no I/O
#   scripts/qualify-production-encoder.sh --self-test    # checks it, no I/O
#
# Bash 3.2 compatible (macOS system bash): no associative arrays, no mapfile,
# no Bash 4 syntax.

set -euo pipefail

# ---------------------------------------------------------------------------
# Fixed acceptance constants
# ---------------------------------------------------------------------------

readonly PLAN_SCHEMA=1
readonly PLAN_VERSION=1
readonly MATRIX_NAME="production-v1-nonspatial"

readonly PINNED_LIBVPX_REVISION="12f3a2ac603e8f10742105519e0cd03c3b8f71dd"

readonly FEATURES="baseline aq3 denoiser temporal_t2 temporal_t3"
readonly MODES="cbr vbr"
readonly TARGETS="720p30 1080p30 1080p60"
readonly CPUS="6 7 8"

readonly FRAMES=60
readonly KEYFRAME_INTERVAL=60
readonly LAG_IN_FRAMES=0
readonly TUNE_CONTENT="default"

readonly MAX_AGGREGATE_BDRATE_PCT="0.5"
readonly MAX_IMPORTANT_CLIP_BDRATE_PCT="2.0"
readonly AQ_GAIN_DELTA_MAX_PP="0.3"
readonly AQ_MAX_FRAME_MS=66
readonly MAX_RATIO="1.05"
readonly MAX_LOAD1="0.75"

readonly SPEED_WARMUPS=1
readonly SPEED_PAIRS=5
readonly SPEED_ROTATIONS=2

readonly LATENCY_WARMUP_RUNS=1
readonly LATENCY_MEASURED_RUNS=5

readonly BITRATE_POINTS=6
readonly DECODERS_PER_STREAM=3

# Expected totals. These are asserted against what the iterators actually
# produce, so any matrix edit fails loudly instead of quietly rescoping the
# release contract.
readonly EXPECTED_CONFIG_ROWS=25
readonly EXPECTED_QUALITY_CURVES=200
readonly EXPECTED_QUALITY_AGGREGATE_GROUPS=75
readonly EXPECTED_QUALITY_AGGREGATE_LAYER_GATES=129
readonly EXPECTED_QUALITY_CLIP_LAYER_GUARDS=344
readonly EXPECTED_CANDIDATE_STREAMS=2640
readonly EXPECTED_DECODER_RUNS=7920
readonly EXPECTED_SPEED_GROUPS=75
readonly EXPECTED_LATENCY_ROWS=200
readonly EXPECTED_LATENCY_MEASURED_RUNS=1000
readonly EXPECTED_REQUIRED_ORACLES=1

# ---------------------------------------------------------------------------
# Matrix tables
#
# Every consumer reads the matrix through these functions and the three row
# iterators below, so there is exactly one matrix definition in the script.
# ---------------------------------------------------------------------------

# fps|width|height|threads|tile_columns|speed_kbps|p50_budget_ms|p99_budget_ms
target_spec() {
    case "$1" in
        720p30)  printf '30|1280|720|1|0|1260|33.333|50\n' ;;
        1080p30) printf '30|1920|1080|4|2|2835|33.333|50\n' ;;
        1080p60) printf '60|1920|1080|8|3|5670|16.667|25\n' ;;
        *) printf 'unknown target: %s\n' "$1" >&2; return 1 ;;
    esac
}

target_clips() {
    case "$1" in
        720p30|1080p30) printf 'in_to_tree ducks_take_off old_town_cross\n' ;;
        1080p60)        printf 'in_to_tree old_town_cross\n' ;;
        *) printf 'unknown target: %s\n' "$1" >&2; return 1 ;;
    esac
}

# Six points per curve, centred on the target's real-time operating bitrate.
target_bitrates() {
    case "$1" in
        720p30)  printf '420,630,840,1260,2100,3360\n' ;;
        1080p30) printf '950,1425,1890,2835,4725,7560\n' ;;
        1080p60) printf '1900,2850,3780,5670,9450,15120\n' ;;
        *) printf 'unknown target: %s\n' "$1" >&2; return 1 ;;
    esac
}

source_clip_file() {
    case "$1" in
        in_to_tree)     printf 'in_to_tree_420_720p50.y4m\n' ;;
        ducks_take_off) printf 'ducks_take_off_420_720p50.y4m\n' ;;
        old_town_cross) printf 'old_town_cross_420_720p50.y4m\n' ;;
        *) printf 'unknown clip: %s\n' "$1" >&2; return 1 ;;
    esac
}

# --- per-feature axes -------------------------------------------------------

feature_modes() {
    case "$1" in
        aq3) printf 'vbr\n' ;;
        baseline|denoiser|temporal_t2|temporal_t3) printf '%s\n' "$MODES" ;;
        *) printf 'unknown feature: %s\n' "$1" >&2; return 1 ;;
    esac
}

feature_cpus() {
    case "$1" in
        aq3) printf '7\n' ;;
        baseline|denoiser|temporal_t2|temporal_t3) printf '%s\n' "$CPUS" ;;
        *) printf 'unknown feature: %s\n' "$1" >&2; return 1 ;;
    esac
}

feature_aq_mode() {
    case "$1" in
        aq3) printf '3\n' ;;
        baseline|denoiser|temporal_t2|temporal_t3) printf '0\n' ;;
        *) printf 'unknown feature: %s\n' "$1" >&2; return 1 ;;
    esac
}

feature_denoiser() {
    case "$1" in
        denoiser) printf '1\n' ;;
        baseline|aq3|temporal_t2|temporal_t3) printf '0\n' ;;
        *) printf 'unknown feature: %s\n' "$1" >&2; return 1 ;;
    esac
}

feature_temporal_layers() {
    case "$1" in
        temporal_t2) printf '2\n' ;;
        temporal_t3) printf '3\n' ;;
        baseline|aq3|denoiser) printf '1\n' ;;
        *) printf 'unknown feature: %s\n' "$1" >&2; return 1 ;;
    esac
}

# Multi-layer rows gate every layer and therefore retain the full stream plus
# one extracted stream per layer. Single-layer rows retain the full stream
# only -- extracting "layer 0" of a single-layer stream is the same bytes.
streams_per_point() {
    if [ "$1" -gt 1 ]; then
        printf '%s\n' "$(( 1 + $1 ))"
    else
        printf '1\n'
    fi
}

# --- row iterators ----------------------------------------------------------

# feature|mode|cpu|aq|denoiser|temporal_layers -- the configuration spine.
emit_configs() {
    local feature mode cpu aq denoiser layers
    for feature in $FEATURES; do
        aq="$(feature_aq_mode "$feature")"
        denoiser="$(feature_denoiser "$feature")"
        layers="$(feature_temporal_layers "$feature")"
        for mode in $(feature_modes "$feature"); do
            for cpu in $(feature_cpus "$feature"); do
                printf '%s|%s|%s|%s|%s|%s\n' \
                    "$feature" "$mode" "$cpu" "$aq" "$denoiser" "$layers"
            done
        done
    done
}

# feature|mode|target|clip|cpu|aq|denoiser|temporal_layers -- the quality and
# latency spine. Clip is innermost so consecutive rows share a corpus group.
emit_clip_rows() {
    local feature mode cpu aq denoiser layers target clip
    while IFS='|' read -r feature mode cpu aq denoiser layers; do
        for target in $TARGETS; do
            for clip in $(target_clips "$target"); do
                printf '%s|%s|%s|%s|%s|%s|%s|%s\n' \
                    "$feature" "$mode" "$target" "$clip" "$cpu" \
                    "$aq" "$denoiser" "$layers"
            done
        done
    done < <(emit_configs)
}

# feature|mode|target|cpu|aq|denoiser|temporal_layers -- the relative-speed
# spine. Clips are grouped into one corpus measurement per group.
emit_speed_groups() {
    local feature mode cpu aq denoiser layers target
    while IFS='|' read -r feature mode cpu aq denoiser layers; do
        for target in $TARGETS; do
            printf '%s|%s|%s|%s|%s|%s|%s\n' \
                "$feature" "$mode" "$target" "$cpu" "$aq" "$denoiser" "$layers"
        done
    done < <(emit_configs)
}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Dry-run oracle
#
# --print-plan must not touch the filesystem, resolve a tool, create a work
# directory, or launch anything. It only reads the tables above.
# ---------------------------------------------------------------------------

print_plan() {
    local curves=0 aggregate_groups=0 aggregate_gates=0 clip_guards=0
    local streams=0 runs=0 groups=0 rows=0 measured=0 oracles=0 configs=0
    local feature mode target clip cpu aq denoiser layers
    local fps width height threads tiles speed_kbps p50 p99
    local grid clips per_point curve_streams curve_runs max_frame

    printf 'plan schema=%s version=%s matrix=%s\n' \
        "$PLAN_SCHEMA" "$PLAN_VERSION" "$MATRIX_NAME"
    printf 'libvpx_revision=%s\n' "$PINNED_LIBVPX_REVISION"
    printf 'frames=%s keyframe_interval=%s lag_in_frames=%s tune_content=%s passes=1 realtime=1\n' \
        "$FRAMES" "$KEYFRAME_INTERVAL" "$LAG_IN_FRAMES" "$TUNE_CONTENT"
    printf 'bitrate_points=%s decoders_per_stream=%s\n' \
        "$BITRATE_POINTS" "$DECODERS_PER_STREAM"
    printf 'max_aggregate_bdrate_pct=%s max_important_clip_bdrate_pct=%s aggregate_method=arithmetic_mean_clip_bdrate_pct aq_gain_delta_max_pp=%s aq_max_frame_ms=%s\n' \
        "$MAX_AGGREGATE_BDRATE_PCT" "$MAX_IMPORTANT_CLIP_BDRATE_PCT" \
        "$AQ_GAIN_DELTA_MAX_PP" "$AQ_MAX_FRAME_MS"
    printf 'max_ratio=%s max_load1=%s speed_warmups=%s speed_pairs=%s speed_rotations=%s\n' \
        "$MAX_RATIO" "$MAX_LOAD1" "$SPEED_WARMUPS" "$SPEED_PAIRS" \
        "$SPEED_ROTATIONS"
    printf 'latency_warmup_runs=%s latency_measured_runs=%s\n' \
        "$LATENCY_WARMUP_RUNS" "$LATENCY_MEASURED_RUNS"
    printf 'spatial_layers=excluded spatial_qualifier=scripts/qualify-spatial-svc.sh\n'

    while IFS='|' read -r feature mode cpu aq denoiser layers; do
        configs=$(( configs + 1 ))
        printf 'config %s %s cpu%s aq=%s denoiser=%s temporal_layers=%s\n' \
            "$feature" "$mode" "$cpu" "$aq" "$denoiser" "$layers"
    done < <(emit_configs)

    while IFS='|' read -r feature mode target clip cpu aq denoiser layers; do
        IFS='|' read -r fps width height threads tiles speed_kbps p50 p99 \
            <<< "$(target_spec "$target")"
        grid="$(target_bitrates "$target")"
        per_point="$(streams_per_point "$layers")"
        curve_streams=$(( BITRATE_POINTS * per_point ))
        curve_runs=$(( curve_streams * DECODERS_PER_STREAM ))
        curves=$(( curves + 1 ))
        clip_guards=$(( clip_guards + layers ))
        streams=$(( streams + curve_streams ))
        runs=$(( runs + curve_runs ))
        printf 'quality %s %s %s %s cpu%s aq=%s denoiser=%s temporal_layers=%s source=%s fps=%s res=%sx%s threads=%s tiles=%s bitrates=%s points=%s layer_guards=%s streams=%s decoder_runs=%s max_clip_bdrate_pct=%s\n' \
            "$feature" "$mode" "$target" "$clip" "$cpu" \
            "$aq" "$denoiser" "$layers" "$(source_clip_file "$clip")" \
            "$fps" "$width" "$height" "$threads" "$tiles" \
            "$grid" "$BITRATE_POINTS" "$layers" "$curve_streams" \
            "$curve_runs" "$MAX_IMPORTANT_CLIP_BDRATE_PCT"
    done < <(emit_clip_rows)

    while IFS='|' read -r feature mode target cpu aq denoiser layers; do
        clips="$(target_clips "$target")"
        clips="${clips// /,}"
        aggregate_groups=$(( aggregate_groups + 1 ))
        aggregate_gates=$(( aggregate_gates + layers ))
        printf 'quality_aggregate %s %s %s cpu%s aq=%s denoiser=%s temporal_layers=%s clips=%s layer_gates=%s method=arithmetic_mean_clip_bdrate_pct max_bdrate_pct=%s\n' \
            "$feature" "$mode" "$target" "$cpu" "$aq" "$denoiser" \
            "$layers" "$clips" "$layers" "$MAX_AGGREGATE_BDRATE_PCT"
    done < <(emit_speed_groups)

    while IFS='|' read -r feature mode target cpu aq denoiser layers; do
        IFS='|' read -r fps width height threads tiles speed_kbps p50 p99 \
            <<< "$(target_spec "$target")"
        clips="$(target_clips "$target")"
        clips="${clips// /,}"
        groups=$(( groups + 1 ))
        printf 'speed %s %s %s cpu%s aq=%s denoiser=%s temporal_layers=%s clips=%s target_kbps=%s fps=%s res=%sx%s threads=%s tiles=%s warmups=%s pairs=%s rotations=%s max_ratio=%s max_load1=%s\n' \
            "$feature" "$mode" "$target" "$cpu" \
            "$aq" "$denoiser" "$layers" "$clips" "$speed_kbps" \
            "$fps" "$width" "$height" "$threads" "$tiles" \
            "$SPEED_WARMUPS" "$SPEED_PAIRS" "$SPEED_ROTATIONS" \
            "$MAX_RATIO" "$MAX_LOAD1"
    done < <(emit_speed_groups)

    while IFS='|' read -r feature mode target clip cpu aq denoiser layers; do
        IFS='|' read -r fps width height threads tiles speed_kbps p50 p99 \
            <<< "$(target_spec "$target")"
        if [ "$aq" -ne 0 ]; then
            max_frame="$AQ_MAX_FRAME_MS"
        else
            max_frame="none"
        fi
        rows=$(( rows + 1 ))
        measured=$(( measured + LATENCY_MEASURED_RUNS ))
        printf 'latency %s %s %s %s cpu%s aq=%s denoiser=%s temporal_layers=%s target_kbps=%s fps=%s res=%sx%s threads=%s tiles=%s warmup_runs=%s measured_runs=%s p50_budget_ms=%s p99_budget_ms=%s max_frame_budget_ms=%s\n' \
            "$feature" "$mode" "$target" "$clip" "$cpu" \
            "$aq" "$denoiser" "$layers" "$speed_kbps" \
            "$fps" "$width" "$height" "$threads" "$tiles" \
            "$LATENCY_WARMUP_RUNS" "$LATENCY_MEASURED_RUNS" \
            "$p50" "$p99" "$max_frame"
    done < <(emit_clip_rows)

    # The AQ gain-parity oracle remains the authoritative Stage 4b Foreman
    # four-curve test. It is a separate required check, not a reinterpretation
    # of the broad corpus rows above.
    oracles=$(( oracles + 1 ))
    printf 'oracle aq_gain_parity feature=aq3 mode=vbr cpu=7 aq=0,3 denoiser=0 temporal_layers=1 source=tests/corpus/foreman_cif.y4m frames=64 crop=256x256+48+16 bitrates_bps=25000,50000,100000,250000 requires=gain_rvp9_finite_negative,gain_libvpx_finite_negative,aq_activation max_gain_delta_pp=%s implementation=crates/rvp9-encoder/tests/quality_gate.rs\n' \
        "$AQ_GAIN_DELTA_MAX_PP"

    if [ "$configs" -ne "$EXPECTED_CONFIG_ROWS" ] ||
        [ "$curves" -ne "$EXPECTED_QUALITY_CURVES" ] ||
        [ "$aggregate_groups" -ne "$EXPECTED_QUALITY_AGGREGATE_GROUPS" ] ||
        [ "$aggregate_gates" -ne "$EXPECTED_QUALITY_AGGREGATE_LAYER_GATES" ] ||
        [ "$clip_guards" -ne "$EXPECTED_QUALITY_CLIP_LAYER_GUARDS" ] ||
        [ "$streams" -ne "$EXPECTED_CANDIDATE_STREAMS" ] ||
        [ "$runs" -ne "$EXPECTED_DECODER_RUNS" ] ||
        [ "$groups" -ne "$EXPECTED_SPEED_GROUPS" ] ||
        [ "$rows" -ne "$EXPECTED_LATENCY_ROWS" ] ||
        [ "$measured" -ne "$EXPECTED_LATENCY_MEASURED_RUNS" ] ||
        [ "$oracles" -ne "$EXPECTED_REQUIRED_ORACLES" ]
    then
        fail "matrix iterators disagree with the declared non-spatial v1 totals"
    fi

    printf 'TOTALS config_rows=%s quality_curves=%s quality_aggregate_groups=%s quality_aggregate_layer_gates=%s quality_clip_layer_guards=%s candidate_streams=%s decoder_runs=%s speed_groups=%s latency_rows=%s latency_measured_runs=%s required_oracles=%s\n' \
        "$configs" "$curves" "$aggregate_groups" "$aggregate_gates" \
        "$clip_guards" "$streams" "$runs" "$groups" "$rows" \
        "$measured" "$oracles"
}

# ---------------------------------------------------------------------------
# Self-test
#
# Creates nothing and launches nothing: it regenerates the plan from the same
# iterators and asserts the totals, the exact target grids and settings, the
# feature-combination guardrails, the absence of spatial rows, and the TOTALS
# line's internal consistency.
# ---------------------------------------------------------------------------

selftest_failures=0

expect_equal() {
    local what="$1" actual="$2" expected="$3"
    if [ "$actual" = "$expected" ]; then
        printf 'ok    %s = %s\n' "$what" "$actual"
    else
        printf 'FAIL  %s = %s, expected %s\n' "$what" "$actual" "$expected"
        selftest_failures=$(( selftest_failures + 1 ))
    fi
}

totals_field() {
    printf '%s\n' "$1" |
        awk -v key="$2" '$1 == "TOTALS" {
            for (i = 2; i <= NF; i++) {
                split($i, kv, "=")
                if (kv[1] == key) { print kv[2] }
            }
        }'
}

plan_count() {
    printf '%s\n' "$1" | awk -v pattern="$2" '$0 ~ pattern { n++ } END { print n + 0 }'
}

# Count plan lines whose kind is $2 and whose field $3 equals $4.
count_by_field() {
    printf '%s\n' "$1" |
        awk -v kind="$2" -v idx="$3" -v want="$4" \
            '$1 == kind && $idx == want { n++ } END { print n + 0 }'
}

self_test() {
    local plan feature mode target cpu

    printf 'qualify-production-encoder self-test\n'
    if ! plan="$(print_plan)"; then
        printf '\nself-test FAILED: the plan could not be generated (see the error above)\n' >&2
        return 1
    fi

    # --- declared totals ---------------------------------------------------
    expect_equal "config_rows" \
        "$(totals_field "$plan" config_rows)" "$EXPECTED_CONFIG_ROWS"
    expect_equal "quality_curves" \
        "$(totals_field "$plan" quality_curves)" "$EXPECTED_QUALITY_CURVES"
    expect_equal "quality_aggregate_groups" \
        "$(totals_field "$plan" quality_aggregate_groups)" "$EXPECTED_QUALITY_AGGREGATE_GROUPS"
    expect_equal "quality_aggregate_layer_gates" \
        "$(totals_field "$plan" quality_aggregate_layer_gates)" "$EXPECTED_QUALITY_AGGREGATE_LAYER_GATES"
    expect_equal "quality_clip_layer_guards" \
        "$(totals_field "$plan" quality_clip_layer_guards)" "$EXPECTED_QUALITY_CLIP_LAYER_GUARDS"
    expect_equal "candidate_streams" \
        "$(totals_field "$plan" candidate_streams)" "$EXPECTED_CANDIDATE_STREAMS"
    expect_equal "decoder_runs" \
        "$(totals_field "$plan" decoder_runs)" "$EXPECTED_DECODER_RUNS"
    expect_equal "speed_groups" \
        "$(totals_field "$plan" speed_groups)" "$EXPECTED_SPEED_GROUPS"
    expect_equal "latency_rows" \
        "$(totals_field "$plan" latency_rows)" "$EXPECTED_LATENCY_ROWS"
    expect_equal "latency_measured_runs" \
        "$(totals_field "$plan" latency_measured_runs)" "$EXPECTED_LATENCY_MEASURED_RUNS"
    expect_equal "required_oracles" \
        "$(totals_field "$plan" required_oracles)" "$EXPECTED_REQUIRED_ORACLES"

    # --- TOTALS consistency: the line must agree with the emitted rows -----
    expect_equal "TOTALS is the final plan line" \
        "$(printf '%s\n' "$plan" | tail -n 1 | awk '{ print $1 }')" "TOTALS"
    expect_equal "exactly one TOTALS line" \
        "$(plan_count "$plan" '^TOTALS ')" 1
    expect_equal "emitted config rows match TOTALS" \
        "$(plan_count "$plan" '^config ')" "$(totals_field "$plan" config_rows)"
    expect_equal "emitted quality rows match TOTALS" \
        "$(plan_count "$plan" '^quality ')" "$(totals_field "$plan" quality_curves)"
    expect_equal "emitted quality aggregate groups match TOTALS" \
        "$(plan_count "$plan" '^quality_aggregate ')" "$(totals_field "$plan" quality_aggregate_groups)"
    expect_equal "emitted speed groups match TOTALS" \
        "$(plan_count "$plan" '^speed ')" "$(totals_field "$plan" speed_groups)"
    expect_equal "emitted latency rows match TOTALS" \
        "$(plan_count "$plan" '^latency ')" "$(totals_field "$plan" latency_rows)"
    expect_equal "emitted oracle rows match TOTALS" \
        "$(plan_count "$plan" '^oracle ')" "$(totals_field "$plan" required_oracles)"
    expect_equal "summed clip layer guards match TOTALS" \
        "$(printf '%s\n' "$plan" | awk '$1 == "quality" { split($17, kv, "="); n += kv[2] } END { print n + 0 }')" \
        "$(totals_field "$plan" quality_clip_layer_guards)"
    expect_equal "summed aggregate layer gates match TOTALS" \
        "$(printf '%s\n' "$plan" | awk '$1 == "quality_aggregate" { split($10, kv, "="); n += kv[2] } END { print n + 0 }')" \
        "$(totals_field "$plan" quality_aggregate_layer_gates)"
    expect_equal "summed streams match TOTALS" \
        "$(printf '%s\n' "$plan" | awk '$1 == "quality" { split($18, kv, "="); n += kv[2] } END { print n + 0 }')" \
        "$(totals_field "$plan" candidate_streams)"
    expect_equal "summed decoder runs match TOTALS" \
        "$(printf '%s\n' "$plan" | awk '$1 == "quality" { split($19, kv, "="); n += kv[2] } END { print n + 0 }')" \
        "$(totals_field "$plan" decoder_runs)"
    expect_equal "decoder runs are 3x streams" \
        "$(printf '%s\n' "$plan" | awk '$1 == "quality" { split($18, a, "="); split($19, b, "="); if (b[2] != 3 * a[2]) n++ } END { print n + 0 }')" \
        0
    expect_equal "latency measured runs match TOTALS" \
        "$(( $(plan_count "$plan" '^latency ') * LATENCY_MEASURED_RUNS ))" \
        "$(totals_field "$plan" latency_measured_runs)"

    # --- exact per-target grids and settings -------------------------------
    #
    # The expected grids, clip counts, and row counts below are written out
    # literally. Deriving them from target_bitrates()/target_clips() would let
    # the matrix agree with itself and pass a corrupted grid unnoticed.
    expect_equal "720p30 clips" "$(target_clips 720p30)" \
        "in_to_tree ducks_take_off old_town_cross"
    expect_equal "1080p30 clips" "$(target_clips 1080p30)" \
        "in_to_tree ducks_take_off old_town_cross"
    expect_equal "1080p60 clips" "$(target_clips 1080p60)" \
        "in_to_tree old_town_cross"

    expect_equal "quality rows for 720p30" \
        "$(count_by_field "$plan" quality 4 720p30)" 75
    expect_equal "quality rows for 1080p30" \
        "$(count_by_field "$plan" quality 4 1080p30)" 75
    expect_equal "quality rows for 1080p60" \
        "$(count_by_field "$plan" quality 4 1080p60)" 50
    expect_equal "latency rows for 720p30" \
        "$(count_by_field "$plan" latency 4 720p30)" 75
    expect_equal "latency rows for 1080p30" \
        "$(count_by_field "$plan" latency 4 1080p30)" 75
    expect_equal "latency rows for 1080p60" \
        "$(count_by_field "$plan" latency 4 1080p60)" 50
    for target in $TARGETS; do
        expect_equal "quality aggregate groups for $target" \
            "$(count_by_field "$plan" quality_aggregate 4 "$target")" "$EXPECTED_CONFIG_ROWS"
        expect_equal "speed groups for $target" \
            "$(count_by_field "$plan" speed 4 "$target")" "$EXPECTED_CONFIG_ROWS"
    done

    expect_equal "720p30 quality rows use grid 420,630,840,1260,2100,3360" \
        "$(printf '%s\n' "$plan" | awk '$1 == "quality" && $4 == "720p30" && $15 == "bitrates=420,630,840,1260,2100,3360" { n++ } END { print n + 0 }')" \
        75
    expect_equal "1080p30 quality rows use grid 950,1425,1890,2835,4725,7560" \
        "$(printf '%s\n' "$plan" | awk '$1 == "quality" && $4 == "1080p30" && $15 == "bitrates=950,1425,1890,2835,4725,7560" { n++ } END { print n + 0 }')" \
        75
    expect_equal "1080p60 quality rows use grid 1900,2850,3780,5670,9450,15120" \
        "$(printf '%s\n' "$plan" | awk '$1 == "quality" && $4 == "1080p60" && $15 == "bitrates=1900,2850,3780,5670,9450,15120" { n++ } END { print n + 0 }')" \
        50
    expect_equal "every quality row sweeps six points" \
        "$(printf '%s\n' "$plan" | awk '$1 == "quality" && $16 == "points=6" { n++ } END { print n + 0 }')" \
        "$EXPECTED_QUALITY_CURVES"
    expect_equal "every bitrate grid holds six comma-separated points" \
        "$(printf '%s\n' "$plan" | awk '$1 == "quality" { n = split($15, p, ","); if (n != 6) bad++ } END { print bad + 0 }')" \
        0

    expect_equal "720p30 quality rows are 30fps 1280x720 1 thread 0 tiles" \
        "$(plan_count "$plan" '^quality [a-z0-9_]+ [a-z]+ 720p30 .* fps=30 res=1280x720 threads=1 tiles=0 ')" \
        "$(count_by_field "$plan" quality 4 720p30)"
    expect_equal "1080p30 quality rows are 30fps 1920x1080 4 threads 2 tiles" \
        "$(plan_count "$plan" '^quality [a-z0-9_]+ [a-z]+ 1080p30 .* fps=30 res=1920x1080 threads=4 tiles=2 ')" \
        "$(count_by_field "$plan" quality 4 1080p30)"
    expect_equal "1080p60 quality rows are 60fps 1920x1080 8 threads 3 tiles" \
        "$(plan_count "$plan" '^quality [a-z0-9_]+ [a-z]+ 1080p60 .* fps=60 res=1920x1080 threads=8 tiles=3 ')" \
        "$(count_by_field "$plan" quality 4 1080p60)"

    expect_equal "720p30 speed groups carry the full three-clip corpus at 1260 kbps" \
        "$(plan_count "$plan" '^speed .* 720p30 .* clips=in_to_tree,ducks_take_off,old_town_cross target_kbps=1260 ')" \
        "$EXPECTED_CONFIG_ROWS"
    expect_equal "1080p30 speed groups carry the full three-clip corpus at 2835 kbps" \
        "$(plan_count "$plan" '^speed .* 1080p30 .* clips=in_to_tree,ducks_take_off,old_town_cross target_kbps=2835 ')" \
        "$EXPECTED_CONFIG_ROWS"
    expect_equal "1080p60 speed groups carry the two-clip corpus at 5670 kbps" \
        "$(plan_count "$plan" '^speed .* 1080p60 .* clips=in_to_tree,old_town_cross target_kbps=5670 ')" \
        "$EXPECTED_CONFIG_ROWS"
    expect_equal "ducks_take_off never appears at 1080p60" \
        "$(plan_count "$plan" ' 1080p60 ducks_take_off ')" 0

    expect_equal "every speed group states the gate" \
        "$(plan_count "$plan" '^speed .* warmups=1 pairs=5 rotations=2 max_ratio=1.05 max_load1=0.75$')" \
        "$EXPECTED_SPEED_GROUPS"

    # --- latency budgets ---------------------------------------------------
    expect_equal "720p30 latency budgets are 33.333/50 ms" \
        "$(plan_count "$plan" '^latency .* 720p30 .* p50_budget_ms=33.333 p99_budget_ms=50 ')" \
        "$(count_by_field "$plan" latency 4 720p30)"
    expect_equal "1080p30 latency budgets are 33.333/50 ms" \
        "$(plan_count "$plan" '^latency .* 1080p30 .* p50_budget_ms=33.333 p99_budget_ms=50 ')" \
        "$(count_by_field "$plan" latency 4 1080p30)"
    expect_equal "1080p60 latency budgets are 16.667/25 ms" \
        "$(plan_count "$plan" '^latency .* 1080p60 .* p50_budget_ms=16.667 p99_budget_ms=25 ')" \
        "$(count_by_field "$plan" latency 4 1080p60)"
    expect_equal "every latency row runs one warmup and five measured runs" \
        "$(plan_count "$plan" '^latency .* warmup_runs=1 measured_runs=5 ')" \
        "$EXPECTED_LATENCY_ROWS"
    expect_equal "latency coverage equals quality coverage, clip for clip" \
        "$(printf '%s\n' "$plan" | awk '$1 == "quality" { print $2, $3, $4, $5, $6 }' | sort | comm -3 - <(printf '%s\n' "$plan" | awk '$1 == "latency" { print $2, $3, $4, $5, $6 }' | sort) | wc -l | tr -d ' ')" \
        0
    expect_equal "only aq3 latency rows carry the 66 ms max-frame cap" \
        "$(plan_count "$plan" 'max_frame_budget_ms=66$')" \
        "$(count_by_field "$plan" latency 2 aq3)"
    expect_equal "non-aq latency rows carry no max-frame cap" \
        "$(plan_count "$plan" 'max_frame_budget_ms=none$')" \
        "$(( EXPECTED_LATENCY_ROWS - $(count_by_field "$plan" latency 2 aq3) ))"

    # --- quality gates -----------------------------------------------------
    expect_equal "every quality row carries the +2.0% important-clip guard" \
        "$(plan_count "$plan" '^quality .* max_clip_bdrate_pct=2.0$')" \
        "$EXPECTED_QUALITY_CURVES"
    expect_equal "no quality row uses another clip ceiling" \
        "$(printf '%s\n' "$plan" | awk '$1 == "quality" && $20 != "max_clip_bdrate_pct=2.0" { n++ } END { print n + 0 }')" \
        0
    expect_equal "every quality aggregate group uses arithmetic mean and +0.5%" \
        "$(plan_count "$plan" '^quality_aggregate .* method=arithmetic_mean_clip_bdrate_pct max_bdrate_pct=0.5$')" \
        "$EXPECTED_QUALITY_AGGREGATE_GROUPS"
    expect_equal "quality aggregate coverage equals speed-group coverage" \
        "$(printf '%s\n' "$plan" | awk '$1 == "quality_aggregate" { print $2, $3, $4, $5 }' | sort | comm -3 - <(printf '%s\n' "$plan" | awk '$1 == "speed" { print $2, $3, $4, $5 }' | sort) | wc -l | tr -d ' ')" \
        0

    # --- AQ gain-parity oracle --------------------------------------------
    expect_equal "the AQ gain-parity oracle is present exactly once" \
        "$(plan_count "$plan" '^oracle aq_gain_parity feature=aq3 mode=vbr cpu=7 aq=0,3 denoiser=0 temporal_layers=1 source=tests/corpus/foreman_cif.y4m frames=64 crop=256x256[+]48[+]16 bitrates_bps=25000,50000,100000,250000 requires=gain_rvp9_finite_negative,gain_libvpx_finite_negative,aq_activation max_gain_delta_pp=0.3 implementation=crates/rvp9-encoder/tests/quality_gate.rs$')" \
        1
    expect_equal "the broad matrix still contains eight aq3 quality rows" \
        "$(count_by_field "$plan" quality 2 aq3)" 8

    # --- feature coverage --------------------------------------------------
    expect_equal "baseline quality rows" "$(count_by_field "$plan" quality 2 baseline)" 48
    expect_equal "aq3 quality rows" "$(count_by_field "$plan" quality 2 aq3)" 8
    expect_equal "denoiser quality rows" "$(count_by_field "$plan" quality 2 denoiser)" 48
    expect_equal "temporal_t2 quality rows" "$(count_by_field "$plan" quality 2 temporal_t2)" 48
    expect_equal "temporal_t3 quality rows" "$(count_by_field "$plan" quality 2 temporal_t3)" 48
    expect_equal "baseline speed groups" "$(count_by_field "$plan" speed 2 baseline)" 18
    expect_equal "aq3 speed groups" "$(count_by_field "$plan" speed 2 aq3)" 3
    expect_equal "denoiser speed groups" "$(count_by_field "$plan" speed 2 denoiser)" 18
    expect_equal "temporal_t2 speed groups" "$(count_by_field "$plan" speed 2 temporal_t2)" 18
    expect_equal "temporal_t3 speed groups" "$(count_by_field "$plan" speed 2 temporal_t3)" 18

    # Baseline, denoiser, and both temporal features must cover CPU6/7/8 and
    # both rate-control modes; a reduced sample is exactly the Stage 4e trap.
    for feature in baseline denoiser temporal_t2 temporal_t3; do
        for mode in $MODES; do
            expect_equal "$feature covers $mode" \
                "$(printf '%s\n' "$plan" | awk -v f="$feature" -v m="$mode" \
                    '$1 == "config" && $2 == f && $3 == m { n++ } END { print n + 0 }')" \
                3
        done
        for cpu in $CPUS; do
            expect_equal "$feature covers cpu$cpu" \
                "$(printf '%s\n' "$plan" | awk -v f="$feature" -v c="cpu$cpu" \
                    '$1 == "config" && $2 == f && $4 == c { n++ } END { print n + 0 }')" \
                2
        done
    done
    expect_equal "denoiser is qualified in both modes, not CBR only" \
        "$(printf '%s\n' "$plan" | awk '$1 == "config" && $2 == "denoiser" { print $3 }' | sort -u | tr '\n' ',')" \
        "cbr,vbr,"

    # --- AQ confinement ----------------------------------------------------
    expect_equal "aq3 configs are VBR/CPU7 only" \
        "$(plan_count "$plan" '^config aq3 vbr cpu7 aq=3 denoiser=0 temporal_layers=1$')" 1
    expect_equal "aq3 has exactly one config row" \
        "$(count_by_field "$plan" config 2 aq3)" 1
    expect_equal "no non-aq3 row enables AQ" \
        "$(printf '%s\n' "$plan" | awk '$1 == "config" && $2 != "aq3" && $5 != "aq=0" { n++ } END { print n + 0 }')" \
        0
    expect_equal "no AQ row is CBR" \
        "$(printf '%s\n' "$plan" | awk '$1 == "config" && $5 == "aq=3" && $3 != "vbr" { n++ } END { print n + 0 }')" \
        0
    expect_equal "no AQ row is off CPU7" \
        "$(printf '%s\n' "$plan" | awk '$1 == "config" && $5 == "aq=3" && $4 != "cpu7" { n++ } END { print n + 0 }')" \
        0

    # --- forbidden feature combinations ------------------------------------
    expect_equal "AQ never combines with the denoiser" \
        "$(printf '%s\n' "$plan" | awk '$1 == "config" && $5 != "aq=0" && $6 != "denoiser=0" { n++ } END { print n + 0 }')" \
        0
    expect_equal "AQ never combines with temporal SVC" \
        "$(printf '%s\n' "$plan" | awk '$1 == "config" && $5 != "aq=0" && $7 != "temporal_layers=1" { n++ } END { print n + 0 }')" \
        0
    expect_equal "the denoiser never combines with temporal SVC" \
        "$(printf '%s\n' "$plan" | awk '$1 == "config" && $6 != "denoiser=0" && $7 != "temporal_layers=1" { n++ } END { print n + 0 }')" \
        0
    expect_equal "every config uses aq=0 or aq=3" \
        "$(printf '%s\n' "$plan" | awk '$1 == "config" && $5 != "aq=0" && $5 != "aq=3" { n++ } END { print n + 0 }')" \
        0
    expect_equal "every config uses 1, 2, or 3 temporal layers" \
        "$(printf '%s\n' "$plan" | awk '$1 == "config" && $7 != "temporal_layers=1" && $7 != "temporal_layers=2" && $7 != "temporal_layers=3" { n++ } END { print n + 0 }')" \
        0

    # --- per-layer retention and gating ------------------------------------
    #
    # Both directions matter: the negative checks catch a row with the wrong
    # gate count, and the positive counts catch a temporal tier that has
    # silently vanished (a T3 feature emitting two layers violates nothing).
    expect_equal "feature_temporal_layers maps temporal_t2 to 2" \
        "$(feature_temporal_layers temporal_t2)" 2
    expect_equal "feature_temporal_layers maps temporal_t3 to 3" \
        "$(feature_temporal_layers temporal_t3)" 3
    expect_equal "single-layer quality rows" \
        "$(count_by_field "$plan" quality 9 temporal_layers=1)" 104
    expect_equal "two-layer quality rows" \
        "$(count_by_field "$plan" quality 9 temporal_layers=2)" 48
    expect_equal "three-layer quality rows" \
        "$(count_by_field "$plan" quality 9 temporal_layers=3)" 48
    expect_equal "quality rows guarding one clip layer" \
        "$(count_by_field "$plan" quality 17 layer_guards=1)" 104
    expect_equal "quality rows guarding two clip layers" \
        "$(count_by_field "$plan" quality 17 layer_guards=2)" 48
    expect_equal "quality rows guarding three clip layers" \
        "$(count_by_field "$plan" quality 17 layer_guards=3)" 48
    expect_equal "quality rows retaining 6 streams" \
        "$(count_by_field "$plan" quality 18 streams=6)" 104
    expect_equal "quality rows retaining 18 streams" \
        "$(count_by_field "$plan" quality 18 streams=18)" 48
    expect_equal "quality rows retaining 24 streams" \
        "$(count_by_field "$plan" quality 18 streams=24)" 48
    expect_equal "single-layer rows guard one clip layer and retain 6 streams" \
        "$(printf '%s\n' "$plan" | awk '$1 == "quality" && $9 == "temporal_layers=1" && ($17 != "layer_guards=1" || $18 != "streams=6") { n++ } END { print n + 0 }')" \
        0
    expect_equal "T2 rows guard two clip layers and retain full+2 per point" \
        "$(printf '%s\n' "$plan" | awk '$1 == "quality" && $9 == "temporal_layers=2" && ($17 != "layer_guards=2" || $18 != "streams=18") { n++ } END { print n + 0 }')" \
        0
    expect_equal "T3 rows guard three clip layers and retain full+3 per point" \
        "$(printf '%s\n' "$plan" | awk '$1 == "quality" && $9 == "temporal_layers=3" && ($17 != "layer_guards=3" || $18 != "streams=24") { n++ } END { print n + 0 }')" \
        0

    expect_equal "single-layer aggregate groups gate one layer" \
        "$(count_by_field "$plan" quality_aggregate 10 layer_gates=1)" 39
    expect_equal "T2 aggregate groups gate two layers" \
        "$(count_by_field "$plan" quality_aggregate 10 layer_gates=2)" 18
    expect_equal "T3 aggregate groups gate three layers" \
        "$(count_by_field "$plan" quality_aggregate 10 layer_gates=3)" 18

    # --- spatial exclusion --------------------------------------------------
    expect_equal "no plan row carries a spatial layer field" \
        "$(plan_count "$plan" 'spatial_layers=[0-9]|spatial_layer=|inter_layer_pred=| s[0-9]+ |bdrate-svc')" 0
    expect_equal "no quality, aggregate, speed, or latency row mentions spatial" \
        "$(printf '%s\n' "$plan" | awk '($1 == "quality" || $1 == "quality_aggregate" || $1 == "speed" || $1 == "latency") && /spatial/ { n++ } END { print n + 0 }')" \
        0
    expect_equal "spatial is declared excluded exactly once, in the header" \
        "$(plan_count "$plan" '^spatial_layers=excluded spatial_qualifier=scripts/qualify-spatial-svc.sh$')" \
        1

    if [ "$selftest_failures" -ne 0 ]; then
        printf '\nself-test FAILED with %s mismatch(es)\n' "$selftest_failures" >&2
        return 1
    fi
    printf '\nself-test passed\n'
}

# ---------------------------------------------------------------------------
# Option parsing
# ---------------------------------------------------------------------------

usage() {
    cat <<'USAGE'
Usage:
  scripts/qualify-production-encoder.sh --print-plan
  scripts/qualify-production-encoder.sh --self-test

  --print-plan   emit the full non-spatial v1 acceptance matrix; no I/O
  --self-test    regenerate and validate the matrix; no I/O

The quality, speed, and latency execution phases are not enabled yet, so a
real run is rejected. Spatial SVC is qualified separately by
scripts/qualify-spatial-svc.sh and is deliberately absent from this matrix.
USAGE
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

main() {
    case "${1-}" in
        --help|-h)
            usage
            return 0
            ;;
        --print-plan)
            [ "$#" -eq 1 ] || fail "--print-plan must be the only argument"
            print_plan
            return 0
            ;;
        --self-test)
            [ "$#" -eq 1 ] || fail "--self-test must be the only argument"
            self_test
            return 0
            ;;
        "")
            usage >&2
            fail "no arguments; try --help"
            ;;
    esac

    usage >&2
    fail "execution phases (quality, speed, latency) are not enabled in this unit: '$1' cannot start a real run. Only --print-plan and --self-test are implemented; the matrix they describe has not been executed by this script."
}

main "$@"
