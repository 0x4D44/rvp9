#!/usr/bin/env bash
# Reproduce the complete Stage 4e spatial-SVC acceptance matrix for CBR *and*
# VBR from one checked-in orchestrator.
#
# The script composes the existing benchmark binaries; it never reimplements
# codec logic and applies the checked-in release gates:
#
#   quality  96 bdrate-svc curves (2 modes x 8 clip/target combinations x
#            CPU6/7/8 x 2/3 spatial layers) -> 240 per-layer BD-rate gates at
#            <= +0.5% versus the pinned vp9_spatial_svc_encoder for CBR and
#            <= +25.0% for the experimental spatial-VBR preview. Each curve's
#            2,016-stream cross-decoder share runs immediately, before the
#            curve's bulky evidence is discarded.
#   speed    36 encode-headtohead corpus groups (2 modes x 3 targets x
#            CPU6/7/8 x 2/3 layers) at <= 1.05x vpxenc.
#   latency  96 realtime rows, one warmup plus five measured runs each, gating
#            the median p50 and median p99 against the frame budgets.
#
# Cross-decoder checking is inseparable from the quality phase: every rvp9 test
# stream must decode to identical I420 SHA-256 under rvp9dec and the pinned
# vpxdec, and must decode successfully under ffmpeg. ffmpeg is a decode-success
# oracle only -- no byte identity is claimed for it.
#
# Disk: quality peaks at roughly one curve's decoded Y4M working set (about
# 2.5 GB for a 1080p60 three-layer curve) because every curve's generated Y4M
# and IVF evidence is deleted as soon as its hashes and results are recorded.
# Caller inputs, binaries, summaries, and logs are never deleted.
#
# The script builds nothing. It validates and hashes the exact binaries it is
# handed, so the acceptance artifact under test cannot change underneath the
# matrix.
#
#   scripts/qualify-spatial-svc.sh --print-plan     # dry-run oracle, no I/O
#   scripts/qualify-spatial-svc.sh --self-test      # checks the plan, no I/O
#   scripts/qualify-spatial-svc.sh --work-dir DIR [--phase all]

set -euo pipefail

# ---------------------------------------------------------------------------
# Fixed acceptance constants
# ---------------------------------------------------------------------------

readonly PINNED_LIBVPX_REVISION="12f3a2ac603e8f10742105519e0cd03c3b8f71dd"

readonly MODES="cbr vbr"
readonly TARGETS="720p30 1080p30 1080p60"
readonly CPUS="6 7 8"
readonly LAYER_COUNTS="2 3"

readonly FRAMES=60
readonly KEYFRAME_INTERVAL=60
readonly AQ_MODE=0
readonly NOISE_SENSITIVITY=0
readonly TEMPORAL_LAYERS=1
readonly INTER_LAYER_PRED=0
readonly LAG_IN_FRAMES=0
readonly TUNE_CONTENT="default"

readonly DEFAULT_MAX_BDRATE_PCT="0.5"
readonly EXPERIMENTAL_SPATIAL_VBR_MAX_BDRATE_PCT="25.0"
readonly MAX_RATIO="1.05"
readonly MAX_LOAD1="0.75"

readonly SPEED_WARMUPS=1
readonly SPEED_PAIRS=5
readonly SPEED_ROTATIONS=2

readonly LATENCY_WARMUP_RUNS=1
readonly LATENCY_MEASURED_RUNS=5

readonly BITRATE_POINTS=6
readonly DECODERS_PER_STREAM=3

# SHA-256 of the empty byte string; a decoder that emits nothing must not pass.
readonly EMPTY_SHA256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# Wall-clock bounds for every child process, in seconds. Sized for the slowest
# 1080p60 three-layer work on a quiet host, with generous headroom.
readonly PREPROCESS_TIMEOUT=1800
readonly QUALITY_CURVE_TIMEOUT=7200
readonly DECODE_TIMEOUT=900
readonly SPEED_GROUP_TIMEOUT=14400
readonly LATENCY_RUN_TIMEOUT=900

# The dry-run oracle's expected totals. These are asserted against the counts
# the row iterators actually produce, so a matrix edit fails loudly.
readonly EXPECTED_QUALITY_CURVES=96
readonly EXPECTED_QUALITY_LAYER_GATES=240
readonly EXPECTED_CROSS_DECODER_STREAMS=2016
readonly EXPECTED_CROSS_DECODER_RUNS=6048
readonly EXPECTED_SPEED_GROUPS=36
readonly EXPECTED_LATENCY_ROWS=96
readonly EXPECTED_LATENCY_MEASURED_RUNS=480

# ---------------------------------------------------------------------------
# Matrix tables
#
# Every phase reads the matrix through these four functions and the two row
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

# Every sweep point is exactly divisible by the spatial weight sum (5 for two
# layers, 21 for three), which bdrate-svc requires for an exact allocation.
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

source_clip_file() {
    case "$1" in
        in_to_tree)     printf 'in_to_tree_420_720p50.y4m\n' ;;
        ducks_take_off) printf 'ducks_take_off_420_720p50.y4m\n' ;;
        old_town_cross) printf 'old_town_cross_420_720p50.y4m\n' ;;
        *) printf 'unknown clip: %s\n' "$1" >&2; return 1 ;;
    esac
}

# Pinned multithreaded libvpx emits several distinct hashes for CPU8 1080p
# spatial encodes. The anchor-side exception is recorded per group; the rvp9
# test arm stays strictly hash-stable in the underlying runner either way.
anchor_hash_policy() {
    if [ "$1" = 8 ] && [ "$2" != "720p30" ]; then
        printf 'allow-instability\n'
    else
        printf 'strict\n'
    fi
}

# Spatial VBR is an experimental preview in v0.2.0. Correctness and speed are
# unchanged, while spatial CBR keeps the production +0.5% quality ceiling.
quality_bdrate_limit() {
    local mode="$1"
    if [ "$mode" = "vbr" ]; then
        printf '%s\n' "$EXPERIMENTAL_SPATIAL_VBR_MAX_BDRATE_PCT"
    else
        printf '%s\n' "$DEFAULT_MAX_BDRATE_PCT"
    fi
}

# mode|target|clip|cpu|layers -- the quality/cross-decoder/latency spine.
# Clip is innermost so consecutive rows share a mode/target/cpu/layer group.
emit_clip_rows() {
    local mode target cpu layers clip
    for mode in $MODES; do
        for target in $TARGETS; do
            for cpu in $CPUS; do
                for layers in $LAYER_COUNTS; do
                    for clip in $(target_clips "$target"); do
                        printf '%s|%s|%s|%s|%s\n' \
                            "$mode" "$target" "$clip" "$cpu" "$layers"
                    done
                done
            done
        done
    done
}

# mode|target|cpu|layers -- the relative-speed spine; one group per corpus.
emit_groups() {
    local mode target cpu layers
    for mode in $MODES; do
        for target in $TARGETS; do
            for cpu in $CPUS; do
                for layers in $LAYER_COUNTS; do
                    printf '%s|%s|%s|%s\n' "$mode" "$target" "$cpu" "$layers"
                done
            done
        done
    done
}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

require_number() {
    if [ -z "$2" ] ||
        ! printf '%s\n' "$2" |
            awk '{ ok = ($0 ~ /^[+-]?([0-9]+(\.[0-9]+)?|\.[0-9]+)$/) } END { exit !ok }'
    then
        fail "$1: expected a decimal number, got '$2'"
    fi
}

# True when $1 <= $2, both already validated as decimal numbers.
number_le() {
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 <= b + 0) }'
}

sha256_file() {
    shasum -a 256 "$1" | awk '{ print $1 }'
}

optional_sha256() {
    if [ -f "$1" ]; then
        sha256_file "$1"
    else
        printf 'unavailable\n'
    fi
}

utc_now() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

median_of_five() {
    printf '%s\n' "$@" | sort -n | awk 'NR == 3 { print }'
}

distinct_count() {
    printf '%s\n' "$@" | sort -u | awk 'END { print NR + 0 }'
}

# ---------------------------------------------------------------------------
# Dry-run oracle
#
# --print-plan must not build, check a path, create a file, preprocess a
# source, or launch an encoder. It only reads the tables above.
# ---------------------------------------------------------------------------

print_plan() {
    local curves=0 gates=0 streams=0 runs=0 groups=0 rows=0 measured=0
    local mode target clip cpu layers
    local fps width height threads tiles speed_kbps p50 p99
    local grid clips curve_streams gate

    printf 'plan schema=1\n'
    printf 'libvpx_revision=%s\n' "$PINNED_LIBVPX_REVISION"
    printf 'frames=%s keyframe_interval=%s aq_mode=%s denoiser=%s temporal_layers=%s\n' \
        "$FRAMES" "$KEYFRAME_INTERVAL" "$AQ_MODE" "$NOISE_SENSITIVITY" "$TEMPORAL_LAYERS"
    printf 'inter_layer_pred=%s lag_in_frames=%s tune_content=%s passes=1 realtime=1\n' \
        "$INTER_LAYER_PRED" "$LAG_IN_FRAMES" "$TUNE_CONTENT"
    printf 'max_bdrate_pct=%s experimental_spatial_vbr_max_bdrate_pct=%s max_ratio=%s max_load1=%s\n' \
        "$DEFAULT_MAX_BDRATE_PCT" \
        "$EXPERIMENTAL_SPATIAL_VBR_MAX_BDRATE_PCT" "$MAX_RATIO" "$MAX_LOAD1"

    while IFS='|' read -r mode target clip cpu layers; do
        IFS='|' read -r fps width height threads tiles speed_kbps p50 p99 \
            <<< "$(target_spec "$target")"
        grid="$(target_bitrates "$target" "$layers")"
        gate="$(quality_bdrate_limit "$mode" "$cpu" "$layers")"
        curve_streams=$(( BITRATE_POINTS * (1 + layers) ))
        curves=$(( curves + 1 ))
        gates=$(( gates + layers ))
        streams=$(( streams + curve_streams ))
        runs=$(( runs + curve_streams * DECODERS_PER_STREAM ))
        printf 'quality %s %s %s cpu%s s%s threads=%s tiles=%s bitrates=%s layer_gates=%s streams=%s decoder_runs=%s max_bdrate_pct=%s\n' \
            "$mode" "$target" "$clip" "$cpu" "$layers" "$threads" "$tiles" \
            "$grid" "$layers" "$curve_streams" \
            "$(( curve_streams * DECODERS_PER_STREAM ))" "$gate"
    done < <(emit_clip_rows)

    while IFS='|' read -r mode target cpu layers; do
        IFS='|' read -r fps width height threads tiles speed_kbps p50 p99 \
            <<< "$(target_spec "$target")"
        clips="$(target_clips "$target" | tr ' ' ',')"
        groups=$(( groups + 1 ))
        printf 'speed %s %s cpu%s s%s clips=%s target_kbps=%s threads=%s tiles=%s warmups=%s pairs=%s rotations=%s max_ratio=%s max_load1=%s anchor_hash=%s\n' \
            "$mode" "$target" "$cpu" "$layers" "$clips" "$speed_kbps" \
            "$threads" "$tiles" "$SPEED_WARMUPS" "$SPEED_PAIRS" \
            "$SPEED_ROTATIONS" "$MAX_RATIO" "$MAX_LOAD1" \
            "$(anchor_hash_policy "$cpu" "$target")"
    done < <(emit_groups)

    while IFS='|' read -r mode target clip cpu layers; do
        IFS='|' read -r fps width height threads tiles speed_kbps p50 p99 \
            <<< "$(target_spec "$target")"
        rows=$(( rows + 1 ))
        measured=$(( measured + LATENCY_MEASURED_RUNS ))
        printf 'latency %s %s %s cpu%s s%s target_kbps=%s threads=%s tiles=%s warmup_runs=%s measured_runs=%s p50_budget_ms=%s p99_budget_ms=%s\n' \
            "$mode" "$target" "$clip" "$cpu" "$layers" "$speed_kbps" \
            "$threads" "$tiles" "$LATENCY_WARMUP_RUNS" "$LATENCY_MEASURED_RUNS" \
            "$p50" "$p99"
    done < <(emit_clip_rows)

    if [ "$curves" -ne "$EXPECTED_QUALITY_CURVES" ] ||
        [ "$gates" -ne "$EXPECTED_QUALITY_LAYER_GATES" ] ||
        [ "$streams" -ne "$EXPECTED_CROSS_DECODER_STREAMS" ] ||
        [ "$runs" -ne "$EXPECTED_CROSS_DECODER_RUNS" ] ||
        [ "$groups" -ne "$EXPECTED_SPEED_GROUPS" ] ||
        [ "$rows" -ne "$EXPECTED_LATENCY_ROWS" ] ||
        [ "$measured" -ne "$EXPECTED_LATENCY_MEASURED_RUNS" ]
    then
        fail "matrix iterators disagree with the declared Stage 4e totals"
    fi

    printf 'quality_curves=%s\n' "$curves"
    printf 'quality_layer_gates=%s\n' "$gates"
    printf 'cross_decoder_streams=%s\n' "$streams"
    printf 'cross_decoder_runs=%s\n' "$runs"
    printf 'speed_groups=%s\n' "$groups"
    printf 'latency_rows=%s\n' "$rows"
    printf 'latency_measured_runs=%s\n' "$measured"
}

# ---------------------------------------------------------------------------
# Self-test
#
# Creates nothing and launches nothing: it re-reads the plan text and asserts
# the totals, representative CBR/VBR rows, the exact per-target S2/S3 bitrate
# grids, and CPU/target/mode coverage.
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

plan_field() {
    printf '%s\n' "$1" | awk -v key="$2" -F= '$1 == key { print $2 }'
}

plan_count() {
    printf '%s\n' "$1" | awk -v pattern="$2" '$0 ~ pattern { n++ } END { print n + 0 }'
}

self_test() {
    local plan mode target clip cpu layers grid expected count

    printf 'qualify-spatial-svc self-test\n'
    if ! plan="$(print_plan)"; then
        printf '\nself-test FAILED: the plan could not be generated (see the error above)\n' >&2
        return 1
    fi

    expect_equal "quality_curves" \
        "$(plan_field "$plan" quality_curves)" "$EXPECTED_QUALITY_CURVES"
    expect_equal "quality_layer_gates" \
        "$(plan_field "$plan" quality_layer_gates)" "$EXPECTED_QUALITY_LAYER_GATES"
    expect_equal "cross_decoder_streams" \
        "$(plan_field "$plan" cross_decoder_streams)" "$EXPECTED_CROSS_DECODER_STREAMS"
    expect_equal "cross_decoder_runs" \
        "$(plan_field "$plan" cross_decoder_runs)" "$EXPECTED_CROSS_DECODER_RUNS"
    expect_equal "speed_groups" \
        "$(plan_field "$plan" speed_groups)" "$EXPECTED_SPEED_GROUPS"
    expect_equal "latency_rows" \
        "$(plan_field "$plan" latency_rows)" "$EXPECTED_LATENCY_ROWS"
    expect_equal "latency_measured_runs" \
        "$(plan_field "$plan" latency_measured_runs)" "$EXPECTED_LATENCY_MEASURED_RUNS"

    # The totals must terminate the plan so a caller can assert on `tail -7`.
    expect_equal "totals terminate the plan" \
        "$(printf '%s\n' "$plan" | tail -n 7 | head -n 1 | cut -d= -f1)" \
        "quality_curves"

    # Representative rows: one CBR, one VBR, at opposite corners.
    expect_equal "representative cbr quality row" \
        "$(plan_count "$plan" '^quality cbr 720p30 in_to_tree cpu6 s2 threads=1 tiles=0 bitrates=420,630,840,1260,2100,3360 layer_gates=2 streams=18 decoder_runs=54 max_bdrate_pct=0.5$')" \
        1
    expect_equal "representative vbr quality row" \
        "$(plan_count "$plan" '^quality vbr 1080p60 old_town_cross cpu8 s3 threads=8 tiles=3 bitrates=1890,2856,3780,5670,9450,15120 layer_gates=3 streams=24 decoder_runs=72 max_bdrate_pct=25.0$')" \
        1
    expect_equal "experimental spatial VBR quality rows use +25.0%" \
        "$(plan_count "$plan" '^quality vbr .* max_bdrate_pct=25.0$')" \
        48
    expect_equal "spatial CBR quality rows use +0.5%" \
        "$(plan_count "$plan" '^quality .* max_bdrate_pct=0.5$')" \
        48
    expect_equal "representative vbr speed row" \
        "$(plan_count "$plan" '^speed vbr 1080p30 cpu8 s2 clips=in_to_tree,ducks_take_off,old_town_cross target_kbps=2835 threads=4 tiles=2 warmups=1 pairs=5 rotations=2 max_ratio=1.05 max_load1=0.75 anchor_hash=allow-instability$')" \
        1
    expect_equal "representative cbr latency row" \
        "$(plan_count "$plan" '^latency cbr 1080p60 in_to_tree cpu7 s3 target_kbps=5670 threads=8 tiles=3 warmup_runs=1 measured_runs=5 p50_budget_ms=16.667 p99_budget_ms=25$')" \
        1

    # Exact S2/S3 bitrate grids, per target, counted over both modes, all three
    # CPUs, and every clip of that target.
    for target in $TARGETS; do
        for layers in $LAYER_COUNTS; do
            grid="$(target_bitrates "$target" "$layers")"
            expected=$(( 2 * 3 * $(target_clips "$target" | wc -w) ))
            count="$(printf '%s\n' "$plan" |
                awk -v t="$target" -v l="s$layers" -v g="bitrates=$grid" \
                    '$1 == "quality" && $3 == t && $6 == l && $9 == g { n++ } END { print n + 0 }')"
            expect_equal "quality rows for $target s$layers grid $grid" "$count" "$expected"
        done
    done

    # CPU coverage: 96 quality rows split evenly across CPU6/7/8.
    for cpu in $CPUS; do
        expect_equal "quality rows on cpu$cpu" \
            "$(printf '%s\n' "$plan" | awk -v c="cpu$cpu" '$1 == "quality" && $5 == c { n++ } END { print n + 0 }')" \
            32
        expect_equal "speed groups on cpu$cpu" \
            "$(printf '%s\n' "$plan" | awk -v c="cpu$cpu" '$1 == "speed" && $4 == c { n++ } END { print n + 0 }')" \
            12
        expect_equal "latency rows on cpu$cpu" \
            "$(printf '%s\n' "$plan" | awk -v c="cpu$cpu" '$1 == "latency" && $5 == c { n++ } END { print n + 0 }')" \
            32
    done

    # Target coverage: 2 modes x 3 CPUs x 2 layer counts x clips-per-target.
    for target in $TARGETS; do
        expected=$(( 2 * 3 * 2 * $(target_clips "$target" | wc -w) ))
        expect_equal "quality rows for $target" \
            "$(printf '%s\n' "$plan" | awk -v t="$target" '$1 == "quality" && $3 == t { n++ } END { print n + 0 }')" \
            "$expected"
        expect_equal "latency rows for $target" \
            "$(printf '%s\n' "$plan" | awk -v t="$target" '$1 == "latency" && $3 == t { n++ } END { print n + 0 }')" \
            "$expected"
        expect_equal "speed groups for $target" \
            "$(printf '%s\n' "$plan" | awk -v t="$target" '$1 == "speed" && $3 == t { n++ } END { print n + 0 }')" \
            12
    done

    # Mode coverage: neither mode may be reduced relative to the other.
    for mode in $MODES; do
        expect_equal "quality rows in $mode" \
            "$(printf '%s\n' "$plan" | awk -v m="$mode" '$1 == "quality" && $2 == m { n++ } END { print n + 0 }')" \
            48
        expect_equal "speed groups in $mode" \
            "$(printf '%s\n' "$plan" | awk -v m="$mode" '$1 == "speed" && $2 == m { n++ } END { print n + 0 }')" \
            18
        expect_equal "latency rows in $mode" \
            "$(printf '%s\n' "$plan" | awk -v m="$mode" '$1 == "latency" && $2 == m { n++ } END { print n + 0 }')" \
            48
    done

    # The anchor hash exception is confined to CPU8 1080p rows.
    expect_equal "speed groups allowing anchor hash instability" \
        "$(plan_count "$plan" 'anchor_hash=allow-instability$')" 8
    expect_equal "cpu8 720p30 speed groups stay strict" \
        "$(plan_count "$plan" '^speed [a-z]+ 720p30 cpu8 .*anchor_hash=strict$')" 4

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
  scripts/qualify-spatial-svc.sh --work-dir PATH [--phase PHASE] [path options]
  scripts/qualify-spatial-svc.sh --print-plan
  scripts/qualify-spatial-svc.sh --self-test

  --phase {quality,speed,latency,all}   default: all
                                        cross-decoder is part of `quality`
  --work-dir PATH                       required for any real run

Path options (repo-relative defaults, all validated before a real run):
  --source-dir PATH          canonical 720p50 Y4M originals
  --test-encoder PATH        production PGO rvp9enc
  --spatial-anchor PATH      pinned vp9_spatial_svc_encoder
  --bdrate-svc PATH          rvp9-bench bdrate-svc
  --encode-headtohead PATH   rvp9-bench encode-headtohead
  --realtime PATH            rvp9-bench realtime
  --rvp9dec PATH             rvp9 decoder
  --vpxdec PATH              pinned libvpx vpxdec
  --ffmpeg PATH              ffmpeg executable or name on PATH
  --libvpx-source PATH       pinned libvpx checkout
USAGE
}

phase="all"
work_dir=""
opt_source_dir=""
opt_test_encoder=""
opt_spatial_anchor=""
opt_bdrate_svc=""
opt_headtohead=""
opt_realtime=""
opt_rvp9dec=""
opt_vpxdec=""
opt_ffmpeg=""
opt_libvpx_source=""

parse_args() {
    local flag
    while [ "$#" -gt 0 ]; do
        flag="$1"
        case "$flag" in
            --help|-h) usage; exit 0 ;;
            --*)
                [ "$#" -ge 2 ] || fail "$flag requires a value"
                ;;
            *) usage >&2; fail "unknown argument: $flag" ;;
        esac
        case "$flag" in
            --phase)             phase="$2" ;;
            --work-dir)          work_dir="$2" ;;
            --source-dir)        opt_source_dir="$2" ;;
            --test-encoder)      opt_test_encoder="$2" ;;
            --spatial-anchor)    opt_spatial_anchor="$2" ;;
            --bdrate-svc)        opt_bdrate_svc="$2" ;;
            --encode-headtohead) opt_headtohead="$2" ;;
            --realtime)          opt_realtime="$2" ;;
            --rvp9dec)           opt_rvp9dec="$2" ;;
            --vpxdec)            opt_vpxdec="$2" ;;
            --ffmpeg)            opt_ffmpeg="$2" ;;
            --libvpx-source)     opt_libvpx_source="$2" ;;
            *) usage >&2; fail "unknown argument: $flag" ;;
        esac
        shift 2
    done
    case "$phase" in
        quality|speed|latency|all) ;;
        *) fail "--phase must be one of quality, speed, latency, all" ;;
    esac
    if [ -z "$work_dir" ]; then
        usage >&2
        fail "--work-dir is required for a real run"
    fi
}

# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

repo_root=""
host_triple=""
exe_suffix=""
git_commit=""
libvpx_revision=""
source_dir=""
test_encoder=""
spatial_anchor=""
bdrate_svc=""
headtohead=""
realtime_bin=""
rvp9dec=""
vpxdec=""
ffmpeg_bin=""
libvpx_source=""
input_dir=""
quality_dir=""
speed_dir=""
latency_dir=""
manifest=""

default_path() {
    if [ -n "$1" ]; then
        printf '%s\n' "$1"
    else
        printf '%s\n' "$2"
    fi
}

require_executable() {
    if [ ! -f "$2" ] || [ ! -x "$2" ]; then
        fail "$1 is not an executable file: $2"
    fi
}

# Caller-supplied paths are relative to the caller's directory, not the repo
# root the script changes into a moment later.
absolute_path() {
    case "$1" in
        ''|/*) printf '%s\n' "$1" ;;
        *)     printf '%s/%s\n' "$PWD" "$1" ;;
    esac
}

# --ffmpeg alone may be a bare name looked up through PATH.
absolute_program() {
    case "$1" in
        */*) absolute_path "$1" ;;
        *)   printf '%s\n' "$1" ;;
    esac
}

resolve_environment() {
    work_dir="$(absolute_path "$work_dir")"
    opt_source_dir="$(absolute_path "$opt_source_dir")"
    opt_libvpx_source="$(absolute_path "$opt_libvpx_source")"
    opt_test_encoder="$(absolute_path "$opt_test_encoder")"
    opt_spatial_anchor="$(absolute_path "$opt_spatial_anchor")"
    opt_bdrate_svc="$(absolute_path "$opt_bdrate_svc")"
    opt_headtohead="$(absolute_path "$opt_headtohead")"
    opt_realtime="$(absolute_path "$opt_realtime")"
    opt_rvp9dec="$(absolute_path "$opt_rvp9dec")"
    opt_vpxdec="$(absolute_path "$opt_vpxdec")"
    opt_ffmpeg="$(absolute_program "$opt_ffmpeg")"

    for tool in awk basename find git head shasum sort tail tr uname wc; do
        command -v "$tool" >/dev/null 2>&1 || fail "missing required tool: $tool"
    done
    command -v deltic >/dev/null 2>&1 ||
        fail "missing deltic; every child process must run under 'deltic timeout'"
    command -v rustc >/dev/null 2>&1 || fail "missing required tool: rustc"

    repo_root="$(git rev-parse --show-toplevel)"
    cd "$repo_root"
    git_commit="$(git rev-parse HEAD)"
    host_triple="$(rustc -vV | awk '/^host: / { print $2 }')"
    case "$host_triple" in
        *-windows-*) exe_suffix=".exe" ;;
        *) exe_suffix="" ;;
    esac

    source_dir="$(default_path "$opt_source_dir" "$repo_root/tests/corpus")"
    test_encoder="$(default_path "$opt_test_encoder" \
        "$repo_root/target/encoder-pgo/optimized/release/rvp9enc$exe_suffix")"
    spatial_anchor="$(default_path "$opt_spatial_anchor" \
        "$repo_root/build/libvpx/examples/vp9_spatial_svc_encoder$exe_suffix")"
    bdrate_svc="$(default_path "$opt_bdrate_svc" \
        "$repo_root/target/release/bdrate-svc$exe_suffix")"
    headtohead="$(default_path "$opt_headtohead" \
        "$repo_root/target/release/encode-headtohead$exe_suffix")"
    realtime_bin="$(default_path "$opt_realtime" \
        "$repo_root/target/release/realtime$exe_suffix")"
    rvp9dec="$(default_path "$opt_rvp9dec" \
        "$repo_root/target/release/rvp9dec$exe_suffix")"
    vpxdec="$(default_path "$opt_vpxdec" "$repo_root/build/libvpx/vpxdec$exe_suffix")"
    ffmpeg_bin="$(default_path "$opt_ffmpeg" "ffmpeg")"
    libvpx_source="$(default_path "$opt_libvpx_source" "$repo_root/third_party/libvpx")"

    require_executable "--test-encoder" "$test_encoder"
    require_executable "--spatial-anchor" "$spatial_anchor"
    require_executable "--bdrate-svc" "$bdrate_svc"
    require_executable "--encode-headtohead" "$headtohead"
    require_executable "--realtime" "$realtime_bin"
    require_executable "--rvp9dec" "$rvp9dec"
    require_executable "--vpxdec" "$vpxdec"
    command -v "$ffmpeg_bin" >/dev/null 2>&1 ||
        fail "missing ffmpeg; install it or pass --ffmpeg"
    ffmpeg_bin="$(command -v "$ffmpeg_bin")"
    require_executable "--ffmpeg" "$ffmpeg_bin"

    # The pinned checkout must be a standalone repository at the exact
    # revision. Without the containment check an uninitialized submodule
    # silently resolves to this superproject and reports the wrong revision.
    [ -d "$libvpx_source" ] || fail "missing libvpx source: $libvpx_source"
    local libvpx_root
    libvpx_root="$(git -C "$libvpx_source" rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -z "$libvpx_root" ] ||
        [ "$(cd "$libvpx_root" && pwd -P)" != "$(cd "$libvpx_source" && pwd -P)" ]
    then
        fail "libvpx source is not an initialized standalone checkout: $libvpx_source"
    fi
    libvpx_revision="$(git -C "$libvpx_source" rev-parse HEAD)"
    if [ "$libvpx_revision" != "$PINNED_LIBVPX_REVISION" ]; then
        fail "wrong libvpx revision: $libvpx_revision (expected $PINNED_LIBVPX_REVISION)"
    fi

    local clip
    for clip in in_to_tree ducks_take_off old_town_cross; do
        if [ ! -s "$source_dir/$(source_clip_file "$clip")" ]; then
            fail "missing canonical source: $source_dir/$(source_clip_file "$clip")
run scripts/fetch-corpus.sh or pass --source-dir"
        fi
    done

    mkdir -p "$work_dir"
    work_dir="$(cd "$work_dir" && pwd -P)"
    input_dir="$work_dir/inputs"
    quality_dir="$work_dir/quality"
    speed_dir="$work_dir/speed"
    latency_dir="$work_dir/latency"
    manifest="$work_dir/manifest.txt"
}

require_empty_phase_dir() {
    if [ -e "$1" ]; then
        [ -d "$1" ] || fail "phase path exists and is not a directory: $1"
        if [ -n "$(ls -A "$1" 2>/dev/null)" ]; then
            fail "refusing to overwrite non-empty phase directory: $1"
        fi
    fi
    mkdir -p "$1"
}

# ---------------------------------------------------------------------------
# Host load
# ---------------------------------------------------------------------------

windows_load1() {
    # Windows has no Unix load average. Measure a full minute of aggregate
    # processor time and normalize it to busy logical cores, preserving the
    # existing 0.75 ceiling's "less than one busy core" meaning.
    command -v powershell.exe >/dev/null 2>&1 || return 1
    powershell.exe -NoProfile -NonInteractive -Command \
        '$ErrorActionPreference = "Stop"; $values = @((Get-Counter -Counter "\Processor(_Total)\% Processor Time" -SampleInterval 5 -MaxSamples 13).CounterSamples | Select-Object -Skip 1 -ExpandProperty CookedValue); if ($values.Count -ne 12) { exit 2 }; $meanPercent = ($values | Measure-Object -Average).Average; $busyCores = $meanPercent * [Environment]::ProcessorCount / 100.0; [Console]::Out.WriteLine($busyCores.ToString("0.######", [Globalization.CultureInfo]::InvariantCulture))' |
        tr -d '\r'
    return ${PIPESTATUS[0]}
}

read_load1() {
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) windows_load1; return $? ;;
    esac
    if [ -r /proc/loadavg ]; then
        awk '{ print $1 }' /proc/loadavg
        return 0
    fi
    if command -v sysctl >/dev/null 2>&1; then
        sysctl -n vm.loadavg 2>/dev/null | tr -d '{}' | awk '{ print $1 }'
        return 0
    fi
    return 1
}

load_metric_name() {
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) printf 'windows_60s_mean_busy_logical_cores\n'; return ;;
    esac
    if [ -r /proc/loadavg ] || command -v sysctl >/dev/null 2>&1; then
        printf 'unix_one_minute_load_average\n'
    else
        printf 'unavailable\n'
    fi
}

enforce_load() {
    local point="$1" load
    load="$(read_load1 || true)"
    if [ -z "$load" ]; then
        fail "cannot read host one-minute load at '$point'; the latency gate fails closed"
    fi
    require_number "one-minute load at '$point'" "$load"
    printf 'load_observation %s=%s\n' "$point" "$load" >> "$manifest"
    if ! number_le "$load" "$MAX_LOAD1"; then
        fail "host one-minute load $load exceeds $MAX_LOAD1 at '$point'"
    fi
    printf '  host load at %s: %s (limit %s)\n' "$point" "$load" "$MAX_LOAD1"
}

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

manifest_kv() {
    printf '%s=%s\n' "$1" "$2" >> "$manifest"
}

manifest_artifact() {
    manifest_kv "$1_path" "$2"
    manifest_kv "$1_sha256" "$(sha256_file "$2")"
}

write_manifest_header() {
    {
        printf 'run_begin schema=1\n'
        printf 'script=scripts/qualify-spatial-svc.sh\n'
    } >> "$manifest"
    manifest_kv phase "$phase"
    manifest_kv started_utc "$(utc_now)"
    manifest_kv git_commit "$git_commit"
    manifest_kv git_status_clean \
        "$(if [ -z "$(git status --porcelain)" ]; then printf 'yes'; else printf 'no'; fi)"
    manifest_kv libvpx_commit "$libvpx_revision"
    manifest_kv host "$(uname -a)"
    manifest_kv host_triple "$host_triple"
    manifest_kv host_load_metric "$(load_metric_name)"
    manifest_kv rustc "$(rustc --version)"
    manifest_kv ffmpeg \
        "$(deltic timeout 60 "$ffmpeg_bin" -version </dev/null 2>/dev/null | head -n 1)"
    manifest_kv work_dir "$work_dir"

    manifest_artifact qualification_script \
        "$repo_root/scripts/qualify-spatial-svc.sh"
    manifest_artifact test_encoder "$test_encoder"
    manifest_artifact spatial_anchor "$spatial_anchor"
    manifest_artifact bdrate_svc "$bdrate_svc"
    manifest_artifact encode_headtohead "$headtohead"
    manifest_artifact realtime "$realtime_bin"
    manifest_artifact rvp9dec "$rvp9dec"
    manifest_artifact vpxdec "$vpxdec"
    manifest_artifact ffmpeg_binary "$ffmpeg_bin"

    local clip
    for clip in in_to_tree ducks_take_off old_town_cross; do
        manifest_artifact "source_$clip" "$source_dir/$(source_clip_file "$clip")"
    done

    printf 'matrix_begin\n' >> "$manifest"
    print_plan >> "$manifest"
    printf 'matrix_end\n' >> "$manifest"
}

# ---------------------------------------------------------------------------
# Acceptance inputs
# ---------------------------------------------------------------------------

acceptance_input() {
    printf '%s/%s_%s.y4m\n' "$input_dir" "$1" "$2"
}

prepare_inputs() {
    local target clip src out filter
    local fps width height threads tiles speed_kbps p50 p99

    mkdir -p "$input_dir"
    printf 'Preparing acceptance inputs in %s\n' "$input_dir"
    for target in $TARGETS; do
        IFS='|' read -r fps width height threads tiles speed_kbps p50 p99 \
            <<< "$(target_spec "$target")"
        if [ "$target" = "720p30" ]; then
            filter="fps=$fps"
        else
            filter="scale=$width:$height:flags=bicubic,fps=$fps"
        fi
        for clip in $(target_clips "$target"); do
            src="$source_dir/$(source_clip_file "$clip")"
            out="$(acceptance_input "$clip" "$target")"
            if [ -s "$out" ]; then
                continue
            fi
            printf '  %s\n' "$out"
            deltic timeout "$PREPROCESS_TIMEOUT" "$ffmpeg_bin" \
                -hide_banner -loglevel error -nostdin -y -i "$src" \
                -vf "$filter" -frames:v "$FRAMES" -pix_fmt yuv420p "$out"
            [ -s "$out" ] || fail "ffmpeg produced no output: $out"
        done
    done
}

record_acceptance_inputs() {
    local target clip input
    for target in $TARGETS; do
        for clip in $(target_clips "$target"); do
            input="$(acceptance_input "$clip" "$target")"
            [ -s "$input" ] || fail "acceptance input is missing or empty: $input"
            manifest_artifact "input_${clip}_${target}" "$input"
        done
    done
}

# ---------------------------------------------------------------------------
# Quality phase, with its inseparable cross-decoder checks
# ---------------------------------------------------------------------------

quality_curve_count=0
quality_gate_count=0
cross_stream_count=0
cross_run_count=0

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

# One rvp9 test stream: rvp9dec and the pinned vpxdec must agree byte-for-byte
# on I420, and ffmpeg must decode it. ffmpeg is a decode-success oracle only.
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

# Enumerate the exact expected stream set rather than globbing, so a missing or
# surplus artifact is itself a failure.
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

# Delete only this curve's generated Y4M and IVF evidence, never a caller
# input, binary, summary, or log. bdrate-svc owns the directory exclusively --
# it refuses to start unless it is empty.
discard_curve_evidence() {
    local artifact
    for artifact in "$1"/*.ivf "$1"/*.y4m; do
        if [ -f "$artifact" ]; then
            rm -f "$artifact"
        fi
    done
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

run_quality() {
    local mode target clip cpu layers

    require_empty_phase_dir "$quality_dir"
    printf 'curve_id,mode,target,clip,cpu_used,spatial_layers,threads,tile_columns,bitrates_kbps,layer,bdrate_pct,gate_pct,status\n' \
        > "$quality_dir/quality-summary.csv"
    printf 'curve_id,stream,kind,bytes,stream_sha256,rvp9dec_sha256,vpxdec_sha256,decoders_identical,ffmpeg_decode,status\n' \
        > "$quality_dir/cross-decoder.csv"

    while IFS='|' read -r mode target clip cpu layers; do
        run_quality_curve "$mode" "$target" "$clip" "$cpu" "$layers"
    done < <(emit_clip_rows)

    [ "$quality_curve_count" -eq "$EXPECTED_QUALITY_CURVES" ] ||
        fail "ran $quality_curve_count quality curves, expected $EXPECTED_QUALITY_CURVES"
    [ "$quality_gate_count" -eq "$EXPECTED_QUALITY_LAYER_GATES" ] ||
        fail "recorded $quality_gate_count layer gates, expected $EXPECTED_QUALITY_LAYER_GATES"
    [ "$cross_stream_count" -eq "$EXPECTED_CROSS_DECODER_STREAMS" ] ||
        fail "checked $cross_stream_count streams, expected $EXPECTED_CROSS_DECODER_STREAMS"
    [ "$cross_run_count" -eq "$EXPECTED_CROSS_DECODER_RUNS" ] ||
        fail "ran $cross_run_count decoder runs, expected $EXPECTED_CROSS_DECODER_RUNS"

    manifest_kv quality_curves "$quality_curve_count"
    manifest_kv quality_layer_gates "$quality_gate_count"
    manifest_kv cross_decoder_streams "$cross_stream_count"
    manifest_kv cross_decoder_runs "$cross_run_count"
    manifest_kv quality_summary_sha256 "$(sha256_file "$quality_dir/quality-summary.csv")"
    manifest_kv cross_decoder_summary_sha256 "$(sha256_file "$quality_dir/cross-decoder.csv")"
    printf 'Quality phase passed: %s curves, %s layer gates, %s streams, %s decoder runs.\n' \
        "$quality_curve_count" "$quality_gate_count" "$cross_stream_count" "$cross_run_count"
}

# ---------------------------------------------------------------------------
# Relative-speed phase
# ---------------------------------------------------------------------------

speed_group_count=0
speed_failure_count=0

run_speed_group() {
    local mode="$1" target="$2" cpu="$3" layers="$4"
    local fps width height threads tiles speed_kbps p50 p99
    local group_id dir run_dir policy clip clip_list separate paired status
    local -a clip_args
    local -a hash_args

    IFS='|' read -r fps width height threads tiles speed_kbps p50 p99 \
        <<< "$(target_spec "$target")"
    group_id="${mode}-${target}-cpu${cpu}-s${layers}"
    dir="$speed_dir/$group_id"
    run_dir="$dir/run"
    policy="$(anchor_hash_policy "$cpu" "$target")"
    mkdir -p "$dir"

    clip_args=()
    clip_list=""
    for clip in $(target_clips "$target"); do
        clip_args+=(--clip "$clip=$(acceptance_input "$clip" "$target")")
        if [ -z "$clip_list" ]; then clip_list="$clip"; else clip_list="$clip_list,$clip"; fi
    done

    hash_args=()
    if [ "$policy" = "allow-instability" ]; then
        hash_args+=(--allow-anchor-hash-instability)
    fi

    speed_group_count=$(( speed_group_count + 1 ))
    printf '[speed %2d/%d] %s (%s clips, anchor hash %s)\n' \
        "$speed_group_count" "$EXPECTED_SPEED_GROUPS" "$group_id" "$clip_list" "$policy"

    status="PASS"
    if ! deltic timeout "$SPEED_GROUP_TIMEOUT" "$headtohead" \
        --anchor-encoder "$spatial_anchor" --anchor-kind spatial-svc \
        --anchor-revision "$libvpx_revision" \
        --test-encoder "$test_encoder" --test-kind vpxenc \
        --test-revision "$git_commit" \
        "${clip_args[@]}" \
        --work-dir "$run_dir" \
        --cpu-used "$cpu" --target-bitrate "$speed_kbps" --end-usage "$mode" \
        --fps "$fps/1" --threads "$threads" --tile-columns "$tiles" \
        --spatial-layers "$layers" --inter-layer-pred "$INTER_LAYER_PRED" \
        --aq-mode "$AQ_MODE" --noise-sensitivity "$NOISE_SENSITIVITY" \
        --limit "$FRAMES" --keyframe-interval "$KEYFRAME_INTERVAL" \
        --warmups "$SPEED_WARMUPS" --pairs "$SPEED_PAIRS" \
        --rotations "$SPEED_ROTATIONS" --max-ratio "$MAX_RATIO" \
        --max-load1 "$MAX_LOAD1" \
        ${hash_args[@]+"${hash_args[@]}"} \
        > "$dir/headtohead.log" 2>&1 </dev/null
    then
        status="FAIL"
    fi

    separate=""
    paired=""
    if [ -f "$run_dir/summary.csv" ]; then
        separate="$(awk -F, '$1 == "corpus" { print $4 }' "$run_dir/summary.csv")"
        paired="$(awk -F, '$1 == "corpus" { print $5 }' "$run_dir/summary.csv")"
    fi
    if [ -z "$separate" ] || [ -z "$paired" ]; then
        separate="${separate:-unavailable}"
        paired="${paired:-unavailable}"
        status="FAIL"
    else
        require_number "$group_id separate ratio" "$separate"
        require_number "$group_id paired ratio" "$paired"
        number_le "$separate" "$MAX_RATIO" || status="FAIL"
        number_le "$paired" "$MAX_RATIO" || status="FAIL"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$group_id" "$mode" "$target" "$cpu" "$layers" "$clip_list" \
        "$speed_kbps" "$threads" "$tiles" "$policy" "$status" \
        "$separate" "$paired" "$MAX_RATIO" \
        "$(optional_sha256 "$run_dir/summary.csv")" \
        "$(optional_sha256 "$run_dir/manifest.txt")" \
        >> "$speed_dir/speed-summary.csv"

    if [ "$status" != "PASS" ]; then
        printf '%s\n' \
            "$group_id: relative-speed gate failed (separate=$separate paired=$paired, log $dir/headtohead.log)" >&2
        return 1
    fi
}

run_speed() {
    local mode target cpu layers

    require_empty_phase_dir "$speed_dir"
    printf 'group_id,mode,target,cpu_used,spatial_layers,clips,target_kbps,threads,tile_columns,anchor_hash_policy,status,corpus_separate_ratio,corpus_paired_ratio,max_ratio,summary_sha256,manifest_sha256\n' \
        > "$speed_dir/speed-summary.csv"

    while IFS='|' read -r mode target cpu layers; do
        if ! run_speed_group "$mode" "$target" "$cpu" "$layers"; then
            speed_failure_count=$(( speed_failure_count + 1 ))
        fi
    done < <(emit_groups)

    [ "$speed_group_count" -eq "$EXPECTED_SPEED_GROUPS" ] ||
        fail "ran $speed_group_count speed groups, expected $EXPECTED_SPEED_GROUPS"

    manifest_kv speed_groups "$speed_group_count"
    manifest_kv speed_failures "$speed_failure_count"
    manifest_kv speed_summary_sha256 "$(sha256_file "$speed_dir/speed-summary.csv")"
    [ "$speed_failure_count" -eq 0 ] ||
        fail "$speed_failure_count of $speed_group_count relative-speed groups failed"
    printf 'Relative-speed phase passed: %s corpus groups at or below %sx.\n' \
        "$speed_group_count" "$MAX_RATIO"
}

# ---------------------------------------------------------------------------
# Absolute-latency phase
# ---------------------------------------------------------------------------

latency_row_count=0
latency_measured_count=0

run_latency_row() {
    local mode="$1" target="$2" clip="$3" cpu="$4" layers="$5"
    local fps width height threads tiles speed_kbps p50_budget p99_budget
    local row_id dir input run role log ivf frametimes
    local p50 p99 failed_runs distinct median_p50 median_p99 status run_status
    local -a p50_samples
    local -a p99_samples
    local -a hashes

    IFS='|' read -r fps width height threads tiles speed_kbps p50_budget p99_budget \
        <<< "$(target_spec "$target")"
    row_id="${mode}-${target}-${clip}-cpu${cpu}-s${layers}"
    dir="$latency_dir/$row_id"
    input="$(acceptance_input "$clip" "$target")"
    mkdir -p "$dir"

    latency_row_count=$(( latency_row_count + 1 ))
    printf '[latency %3d/%d] %s\n' \
        "$latency_row_count" "$EXPECTED_LATENCY_ROWS" "$row_id"

    p50_samples=()
    p99_samples=()
    hashes=()
    failed_runs=0

    for (( run = 0; run <= LATENCY_MEASURED_RUNS; run++ )); do
        if [ "$run" -eq 0 ]; then role="warmup"; else role="measured"; fi
        log="$dir/run$run.log"
        ivf="$dir/run$run.ivf"
        frametimes="$dir/run$run.frame-times.csv"

        run_status="PASS"
        if ! deltic timeout "$LATENCY_RUN_TIMEOUT" "$realtime_bin" \
            --encoder "$test_encoder" --input "$input" \
            --cpu-used "$cpu" --end-usage "$mode" \
            --target-bitrate "$speed_kbps" \
            --threads "$threads" --tile-columns "$tiles" \
            --spatial-layers "$layers" --inter-layer-pred "$INTER_LAYER_PRED" \
            --aq-mode "$AQ_MODE" --noise-sensitivity "$NOISE_SENSITIVITY" \
            --limit "$FRAMES" --kf-max-dist "$KEYFRAME_INTERVAL" \
            --budget-ms "$p50_budget" --p99-budget-ms "$p99_budget" \
            --frame-time-csv "$frametimes" --output "$ivf" \
            > "$log" 2>&1 </dev/null
        then
            run_status="FAIL"
        fi

        p50="$(awk '/^encode p50:/ { print $3 }' "$log")"
        p99="$(awk '/^encode p99:/ { print $3 }' "$log")"
        if [ -z "$p50" ] || [ -z "$p99" ]; then
            run_status="FAIL"
            p50="${p50:-unavailable}"
            p99="${p99:-unavailable}"
        else
            require_number "$row_id run$run p50" "$p50"
            require_number "$row_id run$run p99" "$p99"
            if [ "$run" -gt 0 ] && [ "$run_status" = "PASS" ]; then
                p50_samples+=("$p50")
                p99_samples+=("$p99")
            fi
        fi

        if [ -s "$ivf" ]; then
            hashes+=("$(sha256_file "$ivf")")
        else
            hashes+=("missing")
            run_status="FAIL"
        fi
        if [ "$run_status" != "PASS" ]; then
            failed_runs=$(( failed_runs + 1 ))
        fi
        [ "$run" -eq 0 ] || latency_measured_count=$(( latency_measured_count + 1 ))

        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$row_id" "$mode" "$target" "$clip" "$cpu" "$layers" "$run" \
            "$role" "$p50" "$p99" "${hashes[$run]}" "$run_status" \
            >> "$latency_dir/latency-raw.csv"
    done

    # A row that lost any measured run cannot produce a median; it records an
    # explicit FAIL rather than a median over a short sample.
    status="PASS"
    distinct="$(distinct_count "${hashes[@]}")"
    if [ "${#p50_samples[@]}" -ne "$LATENCY_MEASURED_RUNS" ]; then
        median_p50="unavailable"
        median_p99="unavailable"
        status="FAIL"
    else
        median_p50="$(median_of_five "${p50_samples[@]}")"
        median_p99="$(median_of_five "${p99_samples[@]}")"
        number_le "$median_p50" "$p50_budget" || status="FAIL"
        number_le "$median_p99" "$p99_budget" || status="FAIL"
    fi
    [ "$failed_runs" -eq 0 ] || status="FAIL"
    [ "$distinct" -eq 1 ] || status="FAIL"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$row_id" "$mode" "$target" "$clip" "$cpu" "$layers" "$threads" \
        "$tiles" "$speed_kbps" "$median_p50" "$p50_budget" "$median_p99" \
        "$p99_budget" "$failed_runs" "$distinct" "$status" \
        >> "$latency_dir/latency-summary.csv"

    [ "$status" = "PASS" ] ||
        fail "$row_id: latency gate failed (median p50 $median_p50/$p50_budget ms, median p99 $median_p99/$p99_budget ms, failed runs $failed_runs, distinct IVF hashes $distinct)"
}

run_latency() {
    local mode target clip cpu layers group previous_group=""

    require_empty_phase_dir "$latency_dir"
    printf 'row_id,mode,target,clip,cpu_used,spatial_layers,run_index,role,p50_ms,p99_ms,ivf_sha256,status\n' \
        > "$latency_dir/latency-raw.csv"
    printf 'row_id,mode,target,clip,cpu_used,spatial_layers,threads,tile_columns,target_kbps,median_p50_ms,p50_budget_ms,median_p99_ms,p99_budget_ms,failed_runs,distinct_ivf_hashes,status\n' \
        > "$latency_dir/latency-summary.csv"

    # The speed runner does its own load checks; the latency phase needs this
    # explicit one because `realtime` has none.
    enforce_load "latency-start"
    while IFS='|' read -r mode target clip cpu layers; do
        group="${mode}-${target}-cpu${cpu}-s${layers}"
        if [ "$group" != "$previous_group" ]; then
            enforce_load "latency-group:$group"
            previous_group="$group"
        fi
        run_latency_row "$mode" "$target" "$clip" "$cpu" "$layers"
    done < <(emit_clip_rows)
    enforce_load "latency-complete"

    [ "$latency_row_count" -eq "$EXPECTED_LATENCY_ROWS" ] ||
        fail "ran $latency_row_count latency rows, expected $EXPECTED_LATENCY_ROWS"
    [ "$latency_measured_count" -eq "$EXPECTED_LATENCY_MEASURED_RUNS" ] ||
        fail "ran $latency_measured_count measured runs, expected $EXPECTED_LATENCY_MEASURED_RUNS"

    manifest_kv latency_rows "$latency_row_count"
    manifest_kv latency_measured_runs "$latency_measured_count"
    manifest_kv latency_raw_sha256 "$(sha256_file "$latency_dir/latency-raw.csv")"
    manifest_kv latency_summary_sha256 "$(sha256_file "$latency_dir/latency-summary.csv")"
    printf 'Absolute-latency phase passed: %s rows, %s measured runs.\n' \
        "$latency_row_count" "$latency_measured_count"
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

main() {
    case "${1-}" in
        --print-plan)
            [ "$#" -eq 1 ] || fail "--print-plan must be the only argument"
            if print_plan; then return 0; else return 1; fi
            ;;
        --self-test)
            [ "$#" -eq 1 ] || fail "--self-test must be the only argument"
            if self_test; then return 0; else return 1; fi
            ;;
        "")
            usage >&2
            fail "no arguments; try --help"
            ;;
    esac

    parse_args "$@"
    resolve_environment

    case "$phase" in
        quality) require_empty_phase_dir "$quality_dir" ;;
        speed)   require_empty_phase_dir "$speed_dir" ;;
        latency) require_empty_phase_dir "$latency_dir" ;;
        all)
            require_empty_phase_dir "$quality_dir"
            require_empty_phase_dir "$speed_dir"
            require_empty_phase_dir "$latency_dir"
            ;;
    esac

    write_manifest_header
    prepare_inputs
    record_acceptance_inputs

    case "$phase" in
        quality) run_quality ;;
        speed)   run_speed ;;
        latency) run_latency ;;
        all)     run_quality; run_speed; run_latency ;;
    esac

    manifest_kv ended_utc "$(utc_now)"
    printf 'run_end status=pass\n' >> "$manifest"
    printf '\nStage 4e spatial-SVC acceptance phase "%s" passed.\nEvidence: %s\n' \
        "$phase" "$manifest"
}

main "$@"
