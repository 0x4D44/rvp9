#!/usr/bin/env bash
# Declare the complete non-spatial v1 production acceptance matrix for the
# rvp9 encoder from one checked-in orchestrator.
#
# Scope: this file is the *matrix definition*, its two dry oracles, and the
# resumable QUALITY execution phase. The relative-speed and absolute-latency
# execution phases are NOT implemented in this unit; asking for them (or for
# `--phase all`) fails loudly and says so.
#
# A normal real run is fresh. --work-dir must be outside the repository, its
# parent must exist, and the directory itself must be absent or empty. An
# explicit --resume reopens only a run whose sealed fingerprint matches the
# exact committed tree, matrix, tools, corpus, and prepared inputs. Rows publish
# atomically after their retained artifacts and result fragments are sealed.
# Partial attempts are never deleted or reused. Reused rows are decoded again;
# a receipt is not permission to trust an old decoder observation. Receipts are
# checksums for a trusted local evidence tree, not signatures against a
# malicious editor; the sealed quality measurement itself is replayed.
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
#            finite, |gain_rvp9 - gain_libvpx| <= 0.3 percentage
#            points, and explicit evidence that cyclic refresh actually
#            engaged. That oracle is a separate required check, not a second
#            BD-rate curve.
#
#            The real-content oracle reports the decoded count of non-base
#            segmentation cells and active frames. Both counts must be
#            non-zero, so an AQ-labelled no-op cannot pass by omission.
#   speed    <= 1.05x pinned vpxenc wall time, host load1 <= 0.75, one warmup
#            plus five measured pairs over two reversed rotations.
#   latency  median p50 and median p99 inside the per-target frame budgets
#            over five measured runs; AQ3 rows additionally cap max frame
#            time at 66 ms.
#
# Cross-decoder checking is inseparable from the quality phase: every retained
# rvp9 candidate stream must decode to an identical, non-empty I420 SHA-256
# under rvp9dec and the pinned vpxdec, and must decode successfully under
# ffmpeg. ffmpeg is a decode-success oracle only -- no byte identity is claimed
# for it. A decoder failure or mismatch aborts the phase immediately; a quality
# number that misses its ceiling does not, so the evidence stays complete.
#
# Disk: the phase deletes a completed row's generated Y4M as soon as that row's
# hashes are recorded, so successful progress adds only retained IVF/CSV/logs.
# An interrupted attempt deliberately remains untouched and may retain roughly
# 2.5 GB (1080p60 single-layer) or 4.5 GB (three-layer); its exact hidden attempt
# directory is evidence to inspect or move aside after the campaign.
#
# The script builds nothing. It validates and hashes the exact binaries it is
# handed, and it requires the PGO manifest that produced the encoder under test
# so the acceptance artifact cannot silently drift from the source tree.
#
# A real run must name its phase explicitly. There is no default phase: a
# release qualification states which contract it is measuring, and an operator
# who omits it gets a refusal rather than a silently narrowed run.
#
#   scripts/qualify-production-encoder.sh --print-plan   # dry oracle, no I/O
#   scripts/qualify-production-encoder.sh --self-test    # checks it, no I/O
#   scripts/qualify-production-encoder.sh --work-dir DIR --phase quality \
#       --pgo-manifest PATH [path options]
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

# Receipt schema for resumable quality rows. Bump whenever the receipt or
# artifact-fingerprint contract changes.
readonly QUALITY_RECEIPT_SCHEMA=1

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
readonly EXPECTED_PREPARED_INPUTS=16
readonly EXPECTED_SPEED_GROUPS=75
readonly EXPECTED_LATENCY_ROWS=200
readonly EXPECTED_LATENCY_MEASURED_RUNS=1000
readonly EXPECTED_REQUIRED_ORACLES=1

# ---------------------------------------------------------------------------
# Execution constants
#
# None of these change the matrix or a gate; they describe how the quality
# phase runs the matrix it was already given.
# ---------------------------------------------------------------------------

# SHA-256 of the empty byte string; a decoder that emits nothing must not pass.
readonly EMPTY_SHA256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# Deterministic denoiser exercise. Applied after the scale/fps preprocessing so
# the noisy input is pixel-aligned with the clean PSNR reference. The seed is
# fixed, so the same clean input always yields the same noisy input.
readonly NOISE_FILTER="noise=all_seed=48917:alls=12:allf=t+u"

# Wall-clock bounds for every child process, in seconds. Sized for the slowest
# 1080p60 three-layer work on a quiet host, with generous headroom.
readonly TOOL_VERSION_TIMEOUT=60
readonly PREPROCESS_TIMEOUT=1800
readonly QUALITY_ROW_TIMEOUT=7200
readonly DECODE_TIMEOUT=900
readonly AQ_ORACLE_TIMEOUT=7200

# The authoritative Stage 4b AQ gain-parity oracle, named exactly once so the
# executor and the self-test cannot drift apart.
readonly AQ_ORACLE_PACKAGE="rvp9-encoder"
readonly AQ_ORACLE_TEST_TARGET="quality_gate"
readonly AQ_ORACLE_TEST_NAME="bdrate_gate_stage4b_aq_realcontent"
readonly AQ_ORACLE_CORPUS_ENV="RVP9_CORPUS_DIR"
readonly AQ_ORACLE_VPXENC_ENV="RVP9_VPXENC"
readonly AQ_ORACLE_VPXDEC_ENV="RVP9_VPXDEC"
readonly AQ_ORACLE_GATE_LINE="Stage 4b literal gate (|delta| <= 0.3 %): PASS"

# The oracle resolves its clip through RVP9_CORPUS_DIR, so this file must exist
# under --source-dir. Without it the test prints "[skip] foreman_cif.y4m not
# found" and returns success, which is exactly the pass-by-omission a release
# gate must never accept -- so the run validates and hashes the clip up front.
readonly AQ_ORACLE_SOURCE_FILE="foreman_cif.y4m"

# Exact canonical source identities used by every accepted run. Basenames
# alone are not evidence: a truncated or substituted Y4M can still encode and
# yield finite curves.
corpus_expected_sha256() {
    case "$1" in
        in_to_tree_420_720p50.y4m)
            printf '81166517a3054c789923535b5da23fbbb9bbd7fd4c28dac83918ac351e121e74\n'
            ;;
        ducks_take_off_420_720p50.y4m)
            printf '0e09aa923ce72e2f6d1c978c3c1cc8dd9588d2ede819081f040213e361970efc\n'
            ;;
        old_town_cross_420_720p50.y4m)
            printf '0fbc2ec0c552ad693528b2e49409c7449880b9230055a7696346473d1aaf4123\n'
            ;;
        foreman_cif.y4m)
            printf 'b0e3c35eff9fe1da91521a02eb9f0e3ef4bc5aec341ecef1567cd87f1711267b\n'
            ;;
        *) return 1 ;;
    esac
}

corpus_expected_header() {
    case "$1" in
        in_to_tree_420_720p50.y4m|ducks_take_off_420_720p50.y4m|old_town_cross_420_720p50.y4m)
            printf 'YUV4MPEG2 W1280 H720 F50:1 Ip A1:1\n'
            ;;
        foreman_cif.y4m)
            printf 'YUV4MPEG2 W352 H288 F30000:1001 Ip A128:117\n'
            ;;
        *) return 1 ;;
    esac
}

# A finite negative AQ gain on both sides shows AQ changed the rate-distortion
# outcome; it does not show that cyclic refresh actually engaged in the encoder
# under test. Those are different claims, and only the second one is what
# "aq3 is qualified" means.
#
# crates/rvp9-encoder/tests/quality_gate.rs decodes every rvp9 point in the
# exact VBR/CPU7/Foreman curve. It counts decoded mi-cells with a non-base
# segment id and the frames containing them, then emits this exact shape:
#
#     AQ activation: boosted_mi_cells=N active_frames=M
#
# Both N and M must be positive. Missing, duplicate, malformed, or zero-valued
# evidence fails closed.
readonly AQ_ORACLE_ACTIVATION_PREFIX="AQ activation:"

# The PGO manifest lines a real run reads. The binary line is what ties the
# manifest to the file actually under test; without it the provenance chain
# proves a commit but not which binary came out of it.
readonly PGO_MANIFEST_COMMIT_KEY="git_commit"
readonly PGO_MANIFEST_BINARY_KEY="optimized_binary"
readonly PGO_MANIFEST_BINARY_SHA_KEY="optimized_binary_sha256"

# Encoder-shipped paths. The PGO binary was built from one source commit; if
# any of these moved between that commit and HEAD, the binary under test is not
# the binary this tree describes. Test, doc, and bench-script changes after the
# PGO source commit are deliberately allowed.
readonly PGO_GUARDED_PATHS="crates/rvp9-core/src crates/rvp9-core/Cargo.toml \
crates/rvp9-encoder/src crates/rvp9-encoder/Cargo.toml \
crates/rvp9-cli/src crates/rvp9-cli/Cargo.toml \
crates/rvp9-io/src crates/rvp9-io/Cargo.toml \
Cargo.toml Cargo.lock .cargo/config.toml"

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

# manifest_key|filename -- every corpus file a real run must find non-empty
# under --source-dir, and the manifest key its SHA-256 is recorded under.
# Validation and the evidence manifest both read this one list, so a required
# input cannot be checked without being hashed or hashed without being checked.
# The three 720p50 originals feed the broad matrix; foreman_cif feeds the AQ
# gain-parity oracle.
required_corpus_inputs() {
    local clip
    for clip in in_to_tree ducks_take_off old_town_cross; do
        printf 'source_%s|%s\n' "$clip" "$(source_clip_file "$clip")"
    done
    printf 'source_foreman_cif|%s\n' "$AQ_ORACLE_SOURCE_FILE"
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

# --- row command policy -----------------------------------------------------
#
# One place decides, per feature, which runner measures a row, which anchor it
# is measured against, which preprocessed input both encoders consume, and
# which input the PSNR reference comes from. The executor dispatches through
# these functions and the self-test asserts on them, so a command-policy
# assertion is an assertion about what actually runs.

row_runner_kind() {
    case "$1" in
        baseline|aq3|denoiser)   printf 'bdrate\n' ;;
        temporal_t2|temporal_t3) printf 'bdrate-svc\n' ;;
        *) printf 'unknown feature: %s\n' "$1" >&2; return 1 ;;
    esac
}

# The pinned temporal adapter exists because vpxenc cannot configure VP9
# temporal SVC on its command line. It is the anchor for T2/T3 and for nothing
# else; every single-layer row is anchored on pinned vpxenc itself.
row_anchor_kind() {
    case "$1" in
        baseline|aq3|denoiser)   printf 'vpxenc\n' ;;
        temporal_t2|temporal_t3) printf 'temporal-adapter\n' ;;
        *) printf 'unknown feature: %s\n' "$1" >&2; return 1 ;;
    esac
}

# Both encoders always consume the same input. Only the denoiser rows encode
# the derived noisy input; everything else encodes the clean one.
row_input_kind() {
    case "$1" in
        denoiser)                         printf 'noisy\n' ;;
        baseline|aq3|temporal_t2|temporal_t3) printf 'clean\n' ;;
        *) printf 'unknown feature: %s\n' "$1" >&2; return 1 ;;
    esac
}

# PSNR is always measured against the clean source, including on the denoiser
# rows -- measuring a denoiser against its own noisy input would reward
# preserving the noise.
row_reference_kind() {
    case "$1" in
        baseline|aq3|denoiser|temporal_t2|temporal_t3) printf 'clean\n' ;;
        *) printf 'unknown feature: %s\n' "$1" >&2; return 1 ;;
    esac
}

# The exact basenames of the rvp9 candidate streams a row must retain, in the
# order they are checked. Single-layer rows come from bdrate, which names its
# retained IVF `<role>_<bitrate>k.ivf`; temporal rows come from bdrate-svc,
# which additionally writes one extracted operating point per layer.
row_stream_names() {
    local feature="$1" grid="$2" layers rate layer
    layers="$(feature_temporal_layers "$feature")"
    for rate in $(printf '%s' "$grid" | tr ',' ' '); do
        if [ "$layers" -gt 1 ]; then
            printf 'test_%sk_full.ivf\n' "$rate"
            layer=0
            while [ "$layer" -lt "$layers" ]; do
                printf 'test_%sk_tl%s.ivf\n' "$rate" "$layer"
                layer=$(( layer + 1 ))
            done
        else
            printf 'test_%sk.ivf\n' "$rate"
        fi
    done
}

# Which execution phases this unit implements. Speed and latency (and the `all`
# fan-out that includes them) are declared, not executed.
phase_supported() {
    case "$1" in
        quality)          printf 'implemented\n' ;;
        speed|latency|all) printf 'not-implemented\n' ;;
        *)                printf 'unknown\n' ;;
    esac
}

# --- external command policy ------------------------------------------------
#
# Every external command a real phase invokes, audited against the script body
# rather than assumed. Anything not listed here is either a bash builtin
# (printf, read, cd, pwd, command, test) or a caller-supplied binary validated
# separately by path (--test-encoder, --vpxenc, --temporal-anchor, --bdrate,
# --bdrate-svc, --rvp9dec, --vpxdec, --ffmpeg).
#
# Deliberately absent: sort, comm, uniq, and tail, which only --self-test
# invokes, and ls, which nothing invokes any more. Requiring a command a phase
# never runs would turn a healthy host into a spurious refusal. cat is listed
# because usage() invokes it, even though every usage path precedes this check.
#
# --print-plan invokes nothing here -- it is printf and shell arithmetic only,
# and stays runnable with an empty PATH.
phase_required_commands() {
    case "$1" in
        quality)
            printf 'awk basename cargo cat cmp date deltic dirname env find git grep head mkdir mv rmdir rustc shasum tr uname unlink wc\n'
            ;;
        speed|latency|all)
            printf 'phase has no runner, so no command set: %s\n' "$1" >&2
            return 1
            ;;
        *) printf 'unknown phase: %s\n' "$1" >&2; return 1 ;;
    esac
}

# Why each required command is needed, so the list is an audit rather than a
# copied incantation. The self-test asserts every listed command has a reason.
command_purpose() {
    case "$1" in
        awk)      printf 'every numeric comparison, CSV aggregate, and log parse\n' ;;
        basename) printf 'naming a retained candidate stream in the cross-decoder CSV\n' ;;
        cargo)    printf 'the AQ gain-parity oracle runs through "cargo test"\n' ;;
        cat)      printf 'printing usage and replaying sealed row fragments\n' ;;
        cmp)      printf 'proving fresh decoder observations match a sealed row\n' ;;
        date)     printf 'the manifest run start and end timestamps\n' ;;
        deltic)   printf 'every child process runs under "deltic timeout"\n' ;;
        dirname)  printf 'canonicalising the PGO manifest binary path\n' ;;
        env)      printf 'passing the corpus and vpxenc overrides to the oracle\n' ;;
        find)     printf 'counting retained streams, the work-dir emptiness check, per-row Y4M discard\n' ;;
        git)      printf 'HEAD, the PGO source commit, and the pinned libvpx revision\n' ;;
        grep)     printf 'the oracle log skip, gate, and result checks\n' ;;
        head)     printf 'the recorded ffmpeg version line\n' ;;
        mkdir)    printf 'creating the work, input, and per-row directories\n' ;;
        mv)       printf 'atomically publishing row receipts and complete result directories\n' ;;
        rmdir)    printf 'releasing the exact invocation lock directory\n' ;;
        rustc)    printf 'the host triple and the recorded toolchain version\n' ;;
        shasum)   printf 'every artifact SHA-256 and both decoded-output hashes\n' ;;
        tr)       printf 'flattening lists into manifest and CSV fields\n' ;;
        uname)    printf 'the recorded host description\n' ;;
        unlink)   printf 'releasing the exact invocation lock owner file\n' ;;
        wc)       printf 'stream byte counts, clip counts, and git status entries\n' ;;
        *)        printf '\n' ;;
    esac
}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

is_number() {
    [ -n "$1" ] &&
        printf '%s\n' "$1" |
            awk '{ ok = ($0 ~ /^[+-]?([0-9]+(\.[0-9]+)?|\.[0-9]+)$/) } END { exit !ok }'
}

require_number() {
    is_number "$2" || fail "$1: expected a decimal number, got '$2'"
}

# True when $1 <= $2, both already validated as decimal numbers.
number_le() {
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 <= b + 0) }'
}

# True when abs($1) <= $2, both already validated as decimal numbers.
number_abs_le() {
    awk -v a="$1" -v b="$2" 'BEGIN { if (a < 0) a = -a; exit !(a <= b) }'
}

sha256_file() {
    shasum -a 256 "$1" | awk '{ print $1 }'
}

utc_now() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

# The value of the first `<key>=<value>` line, reading the manifest on stdin.
# Everything after the first `=` is the value, so a path containing `=` survives
# intact. Reading stdin rather than a path keeps this testable without a file.
manifest_field() {
    awk -v key="$1" '
        index($0, key "=") == 1 {
            print substr($0, length(key) + 2)
            exit
        }'
}

# An absolute path with the parent directory's symlinks resolved. macOS ships
# no realpath and no `readlink -f`, so this is the portable form. It compares
# two spellings of the same file; it deliberately does not resolve a symlinked
# leaf, because the binary under test is the leaf the caller named.
canonical_file() {
    local dir base resolved
    dir="$(dirname "$1")"
    base="$(basename "$1")"
    [ -d "$dir" ] || return 1
    resolved="$(cd "$dir" && pwd -P)" || return 1
    case "$resolved" in
        */) printf '%s%s\n' "$resolved" "$base" ;;
        *)  printf '%s/%s\n' "$resolved" "$base" ;;
    esac
}

# Canonical spelling for a fresh output directory. Its parent must already
# exist so `..` and symlink spellings cannot hide a path inside the source
# worktree before the runner performs its containment check.
canonical_output_dir() {
    local parent base resolved
    if [ -e "$1" ]; then
        [ -d "$1" ] || fail "--work-dir exists and is not a directory: $1"
        (cd "$1" && pwd -P) || fail "--work-dir cannot be resolved: $1"
        return
    fi
    parent="$(dirname "$1")"
    base="$(basename "$1")"
    [ -d "$parent" ] ||
        fail "--work-dir parent does not exist: $parent"
    resolved="$(cd "$parent" && pwd -P)" ||
        fail "--work-dir parent cannot be resolved: $parent"
    case "$resolved" in
        */) printf '%s%s\n' "$resolved" "$base" ;;
        *)  printf '%s/%s\n' "$resolved" "$base" ;;
    esac
}

# Whether a directory is safe to use as a fresh run root. Absent is fine,
# empty is fine, anything else is refused -- nothing is ever deleted to make
# room. An unreadable or unlistable directory is refused too: "I could not
# look" must never read as "there was nothing there".
require_absent_or_empty_dir() {
    local what="$1" path="$2" entries
    [ -e "$path" ] || return 0
    [ -d "$path" ] || fail "$what exists and is not a directory: $path"
    [ -r "$path" ] || fail "$what is not readable: $path"
    entries="$(find "$path" -mindepth 1 -maxdepth 1)" ||
        fail "$what could not be listed, so its contents are unknown: $path"
    [ -z "$entries" ] ||
        fail "refusing to reuse a non-empty $what: $path
normal mode runs fresh and will not delete anything to make room. Point $what
at a new or empty directory, or pass --resume for a sealed matching run."
}

# Classify the cyclic-refresh activation evidence in an oracle log read on
# stdin. Only one exact report with two positive counters is a pass.
aq_activation_status() {
    awk -v prefix="$1" '
        index($0, prefix) == 1 {
            seen++
            value = substr($0, length(prefix) + 1)
        }
        END {
            if (seen > 1) { print "ambiguous"; exit }
            if (seen != 1) { print "not-reported"; exit }
            gsub(/^[ \t]+/, "", value)
            gsub(/[ \t]+$/, "", value)
            if (value !~ /^boosted_mi_cells=[0-9]+ active_frames=[0-9]+$/) {
                print "malformed"
                exit
            }
            split(value, fields, " ")
            split(fields[1], boosted, "=")
            split(fields[2], frames, "=")
            if (boosted[2] + 0 == 0 || frames[2] + 0 == 0) {
                print "absent"
            } else {
                print "present"
            }
        }'
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
    printf 'oracle aq_gain_parity feature=aq3 mode=vbr cpu=7 aq=0,3 denoiser=0 temporal_layers=1 source=tests/corpus/foreman_cif.y4m frames=64 crop=256x256+48+16 bitrates_bps=25000,50000,100000,250000 requires=gain_rvp9_finite,gain_libvpx_finite,aq_activation max_gain_delta_pp=%s implementation=crates/rvp9-encoder/tests/quality_gate.rs\n' \
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
# Touches no path and starts no encoder, decoder, or build. It regenerates the
# plan from the same iterators and asserts the totals, the exact target grids
# and settings, the feature-combination guardrails, the absence of spatial rows,
# the TOTALS line's internal consistency, and -- by driving the real functions
# rather than restating them -- the CLI refusals, the corpus and PGO manifest
# contracts, the external-command policy, and the AQ activation classifier.
#
# Unlike --print-plan it does use text tools (awk, sort, comm, uniq, tr, wc,
# grep, tail, cat, sed), so it is not PATH-free; "no I/O" means it writes nothing
# and reads no file under test.
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

# `refused` when refusal $1 begins with the literal prefix $2, else
# `not-refused`. A helper rather than an inline `case`, so the assertions stay
# one command substitution deep.
refusal_starts_with() {
    case "$1" in
        "$2"*) printf 'refused\n' ;;
        *)     printf 'not-refused\n' ;;
    esac
}

# Count plan lines whose kind is $2 and whose field $3 equals $4.
count_by_field() {
    printf '%s\n' "$1" |
        awk -v kind="$2" -v idx="$3" -v want="$4" \
            '$1 == kind && $idx == want { n++ } END { print n + 0 }'
}

self_test() {
    local plan feature mode target clip cpu aq denoiser layers phase_name tool
    local prepared_inputs=0 kind

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
    for target in $TARGETS; do
        for clip in $(target_clips "$target"); do
            for kind in clean noisy; do
                prepared_inputs=$(( prepared_inputs + 1 ))
            done
        done
    done
    expect_equal "prepared inputs follow the target/clip matrix" \
        "$prepared_inputs" "$EXPECTED_PREPARED_INPUTS"

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
        "$(plan_count "$plan" '^oracle aq_gain_parity feature=aq3 mode=vbr cpu=7 aq=0,3 denoiser=0 temporal_layers=1 source=tests/corpus/foreman_cif.y4m frames=64 crop=256x256[+]48[+]16 bitrates_bps=25000,50000,100000,250000 requires=gain_rvp9_finite,gain_libvpx_finite,aq_activation max_gain_delta_pp=0.3 implementation=crates/rvp9-encoder/tests/quality_gate.rs$')" \
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

    # --- row command policy -------------------------------------------------
    #
    # Which runner, which anchor, and which preprocessed input each feature
    # uses. The executor dispatches on exactly these functions, so a change
    # that re-points a feature at the wrong runner fails here.
    for feature in baseline aq3 denoiser; do
        expect_equal "$feature rows are measured by bdrate" \
            "$(row_runner_kind "$feature")" "bdrate"
        expect_equal "$feature rows are anchored on pinned vpxenc" \
            "$(row_anchor_kind "$feature")" "vpxenc"
    done
    for feature in temporal_t2 temporal_t3; do
        expect_equal "$feature rows are measured by bdrate-svc" \
            "$(row_runner_kind "$feature")" "bdrate-svc"
        expect_equal "$feature rows are anchored on the pinned temporal adapter" \
            "$(row_anchor_kind "$feature")" "temporal-adapter"
    done
    expect_equal "no single-layer feature uses bdrate-svc" \
        "$(for feature in $FEATURES; do
              if [ "$(feature_temporal_layers "$feature")" -eq 1 ] &&
                  [ "$(row_runner_kind "$feature")" != "bdrate" ]; then printf 'x'; fi
          done | wc -c | tr -d ' ')" 0
    expect_equal "no multi-layer feature uses bdrate" \
        "$(for feature in $FEATURES; do
              if [ "$(feature_temporal_layers "$feature")" -gt 1 ] &&
                  [ "$(row_runner_kind "$feature")" != "bdrate-svc" ]; then printf 'x'; fi
          done | wc -c | tr -d ' ')" 0
    expect_equal "the temporal adapter anchors exactly the two temporal features" \
        "$(for feature in $FEATURES; do
              if [ "$(row_anchor_kind "$feature")" = "temporal-adapter" ]; then
                  printf '%s,' "$feature"
              fi
          done)" \
        "temporal_t2,temporal_t3,"

    # --- clean versus noisy inputs ------------------------------------------
    expect_equal "only the denoiser rows encode the noisy input" \
        "$(for feature in $FEATURES; do
              if [ "$(row_input_kind "$feature")" = "noisy" ]; then printf '%s,' "$feature"; fi
          done)" \
        "denoiser,"
    expect_equal "the denoiser rows encode noisy and reference clean" \
        "$(row_input_kind denoiser)/$(row_reference_kind denoiser)" "noisy/clean"
    expect_equal "every feature measures PSNR against the clean source" \
        "$(for feature in $FEATURES; do
              if [ "$(row_reference_kind "$feature")" != "clean" ]; then printf 'x'; fi
          done | wc -c | tr -d ' ')" 0
    expect_equal "every non-denoiser feature encodes the clean input" \
        "$(for feature in baseline aq3 temporal_t2 temporal_t3; do
              printf '%s' "$(row_input_kind "$feature")"
          done)" \
        "cleancleancleanclean"
    expect_equal "the noisy input is derived with the pinned deterministic seed" \
        "$NOISE_FILTER" "noise=all_seed=48917:alls=12:allf=t+u"

    # --- exact retained stream shape per row --------------------------------
    #
    # Written out literally. Deriving the names from the same helper the
    # executor uses would let the helper agree with itself.
    expect_equal "a single-layer 720p30 row retains six test_RATEk.ivf" \
        "$(row_stream_names baseline "$(target_bitrates 720p30)" | tr '\n' ' ')" \
        "test_420k.ivf test_630k.ivf test_840k.ivf test_1260k.ivf test_2100k.ivf test_3360k.ivf "
    expect_equal "an aq3 row retains the same single-layer shape" \
        "$(row_stream_names aq3 "$(target_bitrates 1080p30)" | tr '\n' ' ')" \
        "test_950k.ivf test_1425k.ivf test_1890k.ivf test_2835k.ivf test_4725k.ivf test_7560k.ivf "
    expect_equal "a denoiser row retains the same single-layer shape" \
        "$(row_stream_names denoiser "$(target_bitrates 1080p60)" | tr '\n' ' ')" \
        "test_1900k.ivf test_2850k.ivf test_3780k.ivf test_5670k.ivf test_9450k.ivf test_15120k.ivf "
    expect_equal "a T2 point retains the full stream plus TL0 and TL1" \
        "$(row_stream_names temporal_t2 420 | tr '\n' ' ')" \
        "test_420k_full.ivf test_420k_tl0.ivf test_420k_tl1.ivf "
    expect_equal "a T3 point retains the full stream plus TL0, TL1, and TL2" \
        "$(row_stream_names temporal_t3 420 | tr '\n' ' ')" \
        "test_420k_full.ivf test_420k_tl0.ivf test_420k_tl1.ivf test_420k_tl2.ivf "
    expect_equal "a complete unordered T3 layer-id set is valid" \
        "$(layer_id_set_status 3 '2 0 1')" "valid"
    expect_equal "a duplicate temporal layer id is invalid" \
        "$(layer_id_set_status 3 '0 1 1')" "invalid"
    expect_equal "a missing temporal layer id is invalid" \
        "$(layer_id_set_status 3 '0 2')" "invalid"
    expect_equal "an out-of-range temporal layer id is invalid" \
        "$(layer_id_set_status 3 '0 1 3')" "invalid"
    expect_equal "a T3 1080p60 row retains 24 streams" \
        "$(row_stream_names temporal_t3 "$(target_bitrates 1080p60)" | awk 'END { print NR + 0 }')" \
        24
    expect_equal "no expected stream name is an anchor artifact" \
        "$(for feature in $FEATURES; do
              row_stream_names "$feature" "$(target_bitrates 720p30)"
          done | grep -c -v '^test_' | tr -d ' ')" 0
    expect_equal "every expected stream name is unique inside its row" \
        "$(for feature in $FEATURES; do
              row_stream_names "$feature" "$(target_bitrates 720p30)" | sort |
                  uniq -d
          done | awk 'END { print NR + 0 }')" 0

    # The enumerated names must reconstruct the declared stream total exactly,
    # independently of the plan's own `streams=` arithmetic.
    expect_equal "enumerated stream names sum to the declared candidate total" \
        "$(while IFS='|' read -r feature mode target clip cpu aq denoiser layers; do
              row_stream_names "$feature" "$(target_bitrates "$target")"
          done < <(emit_clip_rows) | awk 'END { print NR + 0 }')" \
        "$EXPECTED_CANDIDATE_STREAMS"

    # --- AQ oracle command policy -------------------------------------------
    expect_equal "the AQ oracle names the Foreman real-content test" \
        "$AQ_ORACLE_TEST_NAME" "bdrate_gate_stage4b_aq_realcontent"
    expect_equal "the AQ oracle runs in the rvp9-encoder quality_gate target" \
        "$AQ_ORACLE_PACKAGE/$AQ_ORACLE_TEST_TARGET" "rvp9-encoder/quality_gate"
    expect_equal "the AQ oracle overrides the corpus, vpxenc, and vpxdec by env" \
        "$AQ_ORACLE_CORPUS_ENV,$AQ_ORACLE_VPXENC_ENV,$AQ_ORACLE_VPXDEC_ENV" \
        "RVP9_CORPUS_DIR,RVP9_VPXENC,RVP9_VPXDEC"
    expect_equal "the AQ oracle requires the literal 0.3 pp gate line" \
        "$AQ_ORACLE_GATE_LINE" "Stage 4b literal gate (|delta| <= 0.3 %): PASS"

    # --- phase availability -------------------------------------------------
    expect_equal "the quality phase is implemented" \
        "$(phase_supported quality)" "implemented"
    for phase_name in speed latency all; do
        expect_equal "the $phase_name phase is refused" \
            "$(phase_supported "$phase_name")" "not-implemented"
    done
    expect_equal "an unknown phase is neither implemented nor merely unbuilt" \
        "$(phase_supported nonsense)" "unknown"
    expect_equal "exactly one phase is runnable in this unit" \
        "$(for phase_name in quality speed latency all; do
              if [ "$(phase_supported "$phase_name")" = "implemented" ]; then printf 'x'; fi
          done | wc -c | tr -d ' ')" 1

    # --- a real run must name its phase --------------------------------------
    #
    # These drive the real parser in a subshell, so they assert what an
    # operator actually gets, not a restatement of the code. parse_args touches
    # no path, which is why /w and /m need not exist.
    expect_equal "a real run assumes no phase" "${phase:-<none>}" "<none>"
    expect_equal "a real run refuses a missing --phase" \
        "$(parse_args_error --work-dir /w --pgo-manifest /m)" \
        "--phase is required for a real run: no phase is assumed. Pass --phase quality; speed, latency, and all are declared by --print-plan but have no runner here."
    expect_equal "a real run accepts an explicit --phase quality" \
        "$(parse_args_error --phase quality --work-dir /w --pgo-manifest /m)" \
        "accepted"
    expect_equal "a real run accepts explicit resume" \
        "$(parse_args_error --resume --phase quality --work-dir /w --pgo-manifest /m)" \
        "accepted"
    expect_equal "resume is a valueless flag in any position" \
        "$(parse_args_error --phase quality --work-dir /w --resume --pgo-manifest /m)" \
        "accepted"
    expect_equal "duplicate resume is refused" \
        "$(parse_args_error --resume --phase quality --resume --work-dir /w --pgo-manifest /m)" \
        "--resume may be passed only once"
    expect_equal "a real run refuses an unknown --phase" \
        "$(parse_args_error --phase nonsense --work-dir /w --pgo-manifest /m)" \
        "--phase must be one of quality, speed, latency, all"
    for phase_name in speed latency all; do
        expect_equal "a real run refuses --phase $phase_name by name" \
            "$(refusal_starts_with \
                  "$(parse_args_error --phase "$phase_name" --work-dir /w --pgo-manifest /m)" \
                  "--phase $phase_name is not implemented in this unit")" \
            "refused"
    done
    expect_equal "--phase still requires a value" \
        "$(parse_args_error --phase)" "--phase requires a value"
    expect_equal "a real run still requires --work-dir" \
        "$(parse_args_error --phase quality --pgo-manifest /m)" \
        "--work-dir is required for a real run"
    expect_equal "a real run still requires --pgo-manifest" \
        "$(refusal_starts_with \
              "$(parse_args_error --phase quality --work-dir /w)" \
              "--pgo-manifest is required for a real run")" \
        "refused"

    # --- fresh and resumed work directories ----------------------------------
    #
    # The emptiness check itself needs a filesystem, so what is asserted here is
    # the contract the CLI states. Resume is explicit and receipt-bound; normal
    # mode remains fresh and never deletes what it finds.
    expect_equal "usage keeps normal mode absent or empty" \
        "$(plan_count "$(usage)" 'absent or empty unless --resume is explicit')" 1
    expect_equal "usage keeps the work directory outside the repository" \
        "$(plan_count "$(usage)" 'outside the repo, with an existing parent')" 1
    expect_equal "usage states resume requires a sealed matching run" \
        "$(plan_count "$(usage)" 'reopen this exact sealed run')" 1
    expect_equal "usage states resumed rows are decoded again" \
        "$(plan_count "$(usage)" 'cross-decoded again before reuse')" 1
    expect_equal "usage states --phase is required and never assumed" \
        "$(plan_count "$(usage)" 'required; no phase is assumed')" 1
    expect_equal "usage states the PGO manifest binds the encoder under test" \
        "$(plan_count "$(usage)" 'optimized_binary= and optimized_binary_sha256= must')" 1

    # --- required corpus inputs ----------------------------------------------
    #
    # Written out literally: deriving the list from the same function the run
    # validates and hashes with would let it agree with itself.
    expect_equal "the AQ oracle clip is foreman_cif.y4m" \
        "$AQ_ORACLE_SOURCE_FILE" "foreman_cif.y4m"
    expect_equal "the required corpus inputs are the three originals plus the AQ oracle clip" \
        "$(required_corpus_inputs | tr '\n' ' ')" \
        "source_in_to_tree|in_to_tree_420_720p50.y4m source_ducks_take_off|ducks_take_off_420_720p50.y4m source_old_town_cross|old_town_cross_420_720p50.y4m source_foreman_cif|foreman_cif.y4m "
    expect_equal "the canonical source SHA-256 allowlist is exact" \
        "$(for corpus_file in in_to_tree_420_720p50.y4m ducks_take_off_420_720p50.y4m old_town_cross_420_720p50.y4m foreman_cif.y4m; do corpus_expected_sha256 "$corpus_file"; done | tr '\n' ' ')" \
        "81166517a3054c789923535b5da23fbbb9bbd7fd4c28dac83918ac351e121e74 0e09aa923ce72e2f6d1c978c3c1cc8dd9588d2ede819081f040213e361970efc 0fbc2ec0c552ad693528b2e49409c7449880b9230055a7696346473d1aaf4123 b0e3c35eff9fe1da91521a02eb9f0e3ef4bc5aec341ecef1567cd87f1711267b "
    expect_equal "the canonical source headers are exact" \
        "$(for corpus_file in in_to_tree_420_720p50.y4m ducks_take_off_420_720p50.y4m old_town_cross_420_720p50.y4m foreman_cif.y4m; do corpus_expected_header "$corpus_file"; done | tr '\n' '|')" \
        "YUV4MPEG2 W1280 H720 F50:1 Ip A1:1|YUV4MPEG2 W1280 H720 F50:1 Ip A1:1|YUV4MPEG2 W1280 H720 F50:1 Ip A1:1|YUV4MPEG2 W352 H288 F30000:1001 Ip A128:117|"
    expect_equal "every required corpus input has its own manifest key" \
        "$(required_corpus_inputs | awk -F'|' '{ print $1 }' | sort | uniq -d |
              awk 'END { print NR + 0 }')" 0
    expect_equal "the AQ oracle clip is a required corpus input" \
        "$(required_corpus_inputs | awk -F'|' -v f="$AQ_ORACLE_SOURCE_FILE" \
              '$2 == f { n++ } END { print n + 0 }')" 1
    expect_equal "the plan's oracle row names the clip the run validates" \
        "$(plan_count "$plan" "source=tests/corpus/$AQ_ORACLE_SOURCE_FILE ")" 1

    # --- PGO manifest binding ------------------------------------------------
    expect_equal "the PGO manifest keys bind source, binary path, and binary hash" \
        "$PGO_MANIFEST_COMMIT_KEY,$PGO_MANIFEST_BINARY_KEY,$PGO_MANIFEST_BINARY_SHA_KEY" \
        "git_commit,optimized_binary,optimized_binary_sha256"
    expect_equal "the manifest parser reads the optimized binary path" \
        "$(printf 'git_commit=abc123\noptimized_binary=/pgo/optimized/release/rvp9enc\n' |
              manifest_field "$PGO_MANIFEST_BINARY_KEY")" \
        "/pgo/optimized/release/rvp9enc"
    expect_equal "the manifest parser reads the source commit" \
        "$(printf 'git_commit=abc123\noptimized_binary=/pgo/rvp9enc\n' |
              manifest_field "$PGO_MANIFEST_COMMIT_KEY")" \
        "abc123"
    expect_equal "the manifest parser reads the optimized binary hash" \
        "$(printf 'optimized_binary_sha256=0123456789abcdef\n' |
              manifest_field "$PGO_MANIFEST_BINARY_SHA_KEY")" \
        "0123456789abcdef"
    expect_equal "the manifest parser keeps an = inside a value" \
        "$(printf 'optimized_binary=/o=pt/rvp9enc\n' |
              manifest_field "$PGO_MANIFEST_BINARY_KEY")" \
        "/o=pt/rvp9enc"
    expect_equal "the manifest parser reports nothing for a missing key" \
        "[$(printf 'git_commit=abc123\n' | manifest_field "$PGO_MANIFEST_BINARY_KEY")]" \
        "[]"
    expect_equal "the manifest parser does not match a key as a suffix" \
        "[$(printf 'pgo_optimized_binary=/wrong\n' |
              manifest_field "$PGO_MANIFEST_BINARY_KEY")]" \
        "[]"
    expect_equal "the manifest parser takes the first value, not the last" \
        "$(printf 'optimized_binary=/first\noptimized_binary=/second\n' |
              manifest_field "$PGO_MANIFEST_BINARY_KEY")" \
        "/first"

    # --- resumable row receipts ---------------------------------------------
    local receipt_fixture
    receipt_fixture="$(printf 'receipt_schema=1\nrun_fingerprint_sha256=run123\nrow_id=row-a\nartifact_sha256=artifact456\ncompleted_utc=now\n')"
    expect_equal "a receipt binds one run, row, and artifact tree" \
        "$(printf '%s\n' "$receipt_fixture" |
              receipt_consistency_status run123 row-a artifact456)" "valid"
    expect_equal "a duplicate receipt binding is ambiguous" \
        "$(printf '%s\nrun_fingerprint_sha256=run123\n' "$receipt_fixture" |
              receipt_consistency_status run123 row-a artifact456)" "ambiguous"
    expect_equal "a receipt from another run is rejected" \
        "$(printf '%s\n' "$receipt_fixture" |
              receipt_consistency_status other row-a artifact456)" "wrong-run"
    expect_equal "a receipt for another row is rejected" \
        "$(printf '%s\n' "$receipt_fixture" |
              receipt_consistency_status run123 row-b artifact456)" "wrong-row"
    expect_equal "a receipt for changed artifacts is rejected" \
        "$(printf '%s\n' "$receipt_fixture" |
              receipt_consistency_status run123 row-a changed)" "wrong-artifact"
    expect_equal "an old receipt schema is rejected" \
        "$(printf '%s\n' "$receipt_fixture" | sed 's/receipt_schema=1/receipt_schema=0/' |
              receipt_consistency_status run123 row-a artifact456)" "wrong-schema"

    # --- external command policy ---------------------------------------------
    expect_equal "the quality phase declares the commands it invokes" \
        "$(phase_required_commands quality)" \
        "awk basename cargo cat cmp date deltic dirname env find git grep head mkdir mv rmdir rustc shasum tr uname unlink wc"
    expect_equal "every required command declares why it is needed" \
        "$(for tool in $(phase_required_commands quality); do
              if [ -z "$(command_purpose "$tool")" ]; then printf 'x'; fi
          done | wc -c | tr -d ' ')" 0
    expect_equal "an uninvoked command declares no purpose" \
        "[$(command_purpose sort)]" "[]"
    expect_equal "a phase with no runner declares no command set" \
        "$(for phase_name in speed latency all nonsense; do
              if phase_required_commands "$phase_name" >/dev/null 2>&1; then printf 'x'; fi
          done | wc -c | tr -d ' ')" 0

    # --- AQ cyclic-refresh activation ----------------------------------------
    #
    # Required, and fail-closed: only one exact report with two positive
    # counters is a pass.
    expect_equal "the activation report prefix is the declared contract" \
        "$AQ_ORACLE_ACTIVATION_PREFIX" "AQ activation:"
    expect_equal "the plan declares AQ activation a required oracle input" \
        "$(plan_count "$plan" 'requires=gain_rvp9_finite,gain_libvpx_finite,aq_activation ')" \
        1
    expect_equal "the AQ delta gate accepts either sign at the boundary" \
        "$(for delta in -0.3 0.3; do number_abs_le "$delta" 0.3 && printf y; done)" \
        "yy"
    expect_equal "the AQ delta gate rejects either sign beyond the boundary" \
        "$(for delta in -0.301 0.301; do number_abs_le "$delta" 0.3 || printf n; done)" \
        "nn"
    expect_equal "a log with no activation line is not-reported" \
        "$(printf 'Curve 1: rvp9 AqMode::Off\nStage 4b literal gate (|delta| <= 0.3 %%): PASS\n' |
              aq_activation_status "$AQ_ORACLE_ACTIVATION_PREFIX")" \
        "not-reported"
    expect_equal "a measured activation report is present" \
        "$(printf 'AQ activation: boosted_mi_cells=812 active_frames=16\n' |
              aq_activation_status "$AQ_ORACLE_ACTIVATION_PREFIX")" \
        "present"
    expect_equal "an empty activation report is malformed" \
        "$(printf 'AQ activation:    \n' |
              aq_activation_status "$AQ_ORACLE_ACTIVATION_PREFIX")" \
        "malformed"
    expect_equal "zero boosted cells are absent" \
        "$(printf 'AQ activation: boosted_mi_cells=0 active_frames=16\n' |
              aq_activation_status "$AQ_ORACLE_ACTIVATION_PREFIX")" \
        "absent"
    expect_equal "zero active frames are absent" \
        "$(printf 'AQ activation: boosted_mi_cells=812 active_frames=0\n' |
              aq_activation_status "$AQ_ORACLE_ACTIVATION_PREFIX")" \
        "absent"
    expect_equal "an informal activation report is malformed" \
        "$(printf 'AQ activation: 812 blocks over 16 refresh cycles\n' |
              aq_activation_status "$AQ_ORACLE_ACTIVATION_PREFIX")" \
        "malformed"
    expect_equal "a none activation report is malformed" \
        "$(printf 'AQ activation: NONE\n' |
              aq_activation_status "$AQ_ORACLE_ACTIVATION_PREFIX")" \
        "malformed"
    expect_equal "two activation reports are ambiguous, never a pass" \
        "$(printf 'AQ activation: boosted_mi_cells=812 active_frames=16\nAQ activation: boosted_mi_cells=9 active_frames=1\n' |
              aq_activation_status "$AQ_ORACLE_ACTIVATION_PREFIX")" \
        "ambiguous"
    expect_equal "an activation mention that is not the report line is not-reported" \
        "$(printf 'note: AQ activation: 812 blocks\n' |
              aq_activation_status "$AQ_ORACLE_ACTIVATION_PREFIX")" \
        "not-reported"

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
  scripts/qualify-production-encoder.sh --work-dir PATH --phase quality \
      --pgo-manifest PATH [path options]
  scripts/qualify-production-encoder.sh --print-plan
  scripts/qualify-production-encoder.sh --self-test
  scripts/qualify-production-encoder.sh --self-test-resume

  --print-plan   emit the full non-spatial v1 acceptance matrix; no I/O
  --self-test    regenerate and validate the matrix; no I/O
  --self-test-resume
                 exercise receipt publication, reuse counters, tamper refusal,
                 attempt cleanup bounds, and the invocation lock in OS temp

Real-run options:
  --work-dir PATH        required; outside the repo, with an existing parent;
                         absent or empty unless --resume is explicit
  --resume               reopen this exact sealed run; completed rows are
                         validated and cross-decoded again before reuse
  --phase PHASE          required; no phase is assumed. quality executes;
                         speed, latency, and all are declared by the matrix
                         but not implemented here.
  --pgo-manifest PATH    required; the manifest scripts/build-encoder-pgo.sh
                         wrote beside the rvp9enc under test. Its
                         optimized_binary= and optimized_binary_sha256= must
                         match the --test-encoder.

Path options (repo-relative defaults, all validated before a real run):
  --source-dir PATH      canonical 720p50 Y4M originals
  --test-encoder PATH    production PGO rvp9enc
  --vpxenc PATH          pinned libvpx vpxenc (single-layer anchor)
  --temporal-anchor PATH pinned build/libvpx/vpx_temporal_svc_reference
  --bdrate PATH          rvp9-bench bdrate (release)
  --bdrate-svc PATH      rvp9-bench bdrate-svc (release)
  --rvp9dec PATH         rvp9 decoder (release)
  --vpxdec PATH          pinned libvpx vpxdec
  --ffmpeg PATH          ffmpeg executable or name on PATH
  --libvpx-source PATH   pinned libvpx checkout

A normal real run is fresh. --resume accepts only a sealed matching run,
leaves partial attempts untouched, and never reuses an unsealed row. Spatial
SVC is qualified separately by scripts/qualify-spatial-svc.sh and is
deliberately absent from this matrix.
USAGE
}

phase=""
work_dir=""
resume="no"
opt_pgo_manifest=""
opt_source_dir=""
opt_test_encoder=""
opt_vpxenc=""
opt_temporal_anchor=""
opt_bdrate=""
opt_bdrate_svc=""
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
            --resume)
                [ "$resume" = "no" ] || fail "--resume may be passed only once"
                resume="yes"
                shift
                continue
                ;;
            --*) [ "$#" -ge 2 ] || fail "$flag requires a value" ;;
            *) usage >&2; fail "unknown argument: $flag" ;;
        esac
        case "$flag" in
            --phase)           phase="$2" ;;
            --work-dir)        work_dir="$2" ;;
            --pgo-manifest)    opt_pgo_manifest="$2" ;;
            --source-dir)      opt_source_dir="$2" ;;
            --test-encoder)    opt_test_encoder="$2" ;;
            --vpxenc)          opt_vpxenc="$2" ;;
            --temporal-anchor) opt_temporal_anchor="$2" ;;
            --bdrate)          opt_bdrate="$2" ;;
            --bdrate-svc)      opt_bdrate_svc="$2" ;;
            --rvp9dec)         opt_rvp9dec="$2" ;;
            --vpxdec)          opt_vpxdec="$2" ;;
            --ffmpeg)          opt_ffmpeg="$2" ;;
            --libvpx-source)   opt_libvpx_source="$2" ;;
            *) usage >&2; fail "unknown argument: $flag" ;;
        esac
        shift 2
    done

    # A real run states its phase. Defaulting to quality would let an operator
    # who meant "all" walk away believing the whole contract was measured.
    if [ -z "$phase" ]; then
        usage >&2
        fail "--phase is required for a real run: no phase is assumed. Pass --phase quality; speed, latency, and all are declared by --print-plan but have no runner here."
    fi

    # Reject an unimplemented phase before anything is resolved, created, or
    # launched, so the refusal is the same with or without a toolchain.
    case "$(phase_supported "$phase")" in
        implemented) ;;
        not-implemented)
            fail "--phase $phase is not implemented in this unit: only the quality phase executes. The relative-speed and absolute-latency phases (and the 'all' fan-out that includes them) are declared by --print-plan but have no runner here."
            ;;
        *) fail "--phase must be one of quality, speed, latency, all" ;;
    esac
    if [ -z "$work_dir" ]; then
        usage >&2
        fail "--work-dir is required for a real run"
    fi
    if [ -z "$opt_pgo_manifest" ]; then
        usage >&2
        fail "--pgo-manifest is required for a real run: the qualification records which source commit produced the encoder under test"
    fi
}

# The self-test's window into the parser: run it in a subshell and report its
# refusal, or `accepted`. parse_args only assigns globals and calls fail(), so
# this resolves no path, launches nothing, and writes nothing; the subshell
# keeps the parser's assignments out of the caller. The refusal is the last
# stderr line, because some paths print the usage text first.
parse_args_error() {
    local captured
    if captured="$( ( parse_args "$@" ) 2>&1 1>/dev/null )"; then
        printf 'accepted\n'
    else
        printf '%s\n' "$captured" | tail -n 1
    fi
}

# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

repo_root=""
host_triple=""
exe_suffix=""
git_commit=""
git_status_clean=""
libvpx_revision=""
pgo_manifest=""
pgo_manifest_sha256=""
pgo_source_commit=""
pgo_optimized_binary=""
pgo_optimized_binary_sha256=""
source_dir=""
test_encoder=""
vpxenc=""
temporal_anchor=""
bdrate_bin=""
bdrate_svc_bin=""
rvp9dec=""
vpxdec=""
ffmpeg_bin=""
libvpx_source=""
libvpx_config_h=""
libvpx_config_mk=""
input_dir=""
quality_dir=""
quality_rows_dir=""
quality_summary=""
quality_aggregate=""
cross_decoder_csv=""
manifest=""
run_fingerprint=""
quality_output_dir=""
quality_recheck_dir=""
quality_lock_dir=""
quality_lock_owner=""
quality_lock_acquired="no"
active_quality_attempt=""

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

require_help_flag() {
    local what="$1" binary="$2" flag="$3" help
    help="$(deltic timeout "$TOOL_VERSION_TIMEOUT" "$binary" --help </dev/null 2>&1)" ||
        fail "$what failed its --help capability probe: $binary"
    printf '%s\n' "$help" | grep -Fq -- "$flag" ||
        fail "$what is stale or incompatible; --help does not advertise $flag: $binary"
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

# The PGO binary was compiled from one source commit into one output file.
# Record both, prove the commit exists here, prove the recorded output file is
# the very binary handed to --test-encoder, and prove that no encoder-shipped
# path moved between that commit and HEAD -- otherwise the binary under test is
# not the tree being qualified.
validate_pgo_manifest() {
    local path drifted="" recorded recorded_sha recorded_canonical test_canonical actual_sha

    [ -f "$pgo_manifest" ] || fail "--pgo-manifest is not a file: $pgo_manifest"
    pgo_manifest_sha256="$(sha256_file "$pgo_manifest")"
    pgo_source_commit="$(manifest_field "$PGO_MANIFEST_COMMIT_KEY" < "$pgo_manifest")"
    [ -n "$pgo_source_commit" ] ||
        fail "--pgo-manifest records no '$PGO_MANIFEST_COMMIT_KEY=' line: $pgo_manifest"
    git cat-file -e "${pgo_source_commit}^{commit}" 2>/dev/null ||
        fail "the PGO manifest's source commit is not in this repository: $pgo_source_commit"
    pgo_source_commit="$(git rev-parse "${pgo_source_commit}^{commit}")"

    # scripts/build-encoder-pgo.sh records the profile-guided binary it built.
    # Without this, a manifest from one build could be presented alongside an
    # unrelated rvp9enc and the whole provenance chain would still pass.
    recorded="$(manifest_field "$PGO_MANIFEST_BINARY_KEY" < "$pgo_manifest")"
    [ -n "$recorded" ] ||
        fail "--pgo-manifest records no '$PGO_MANIFEST_BINARY_KEY=' line: $pgo_manifest
this manifest predates the binary record, or the PGO build did not finish"
    pgo_optimized_binary="$recorded"
    recorded_canonical="$(canonical_file "$recorded")" ||
        fail "the PGO manifest's $PGO_MANIFEST_BINARY_KEY does not resolve to an existing directory: $recorded"
    test_canonical="$(canonical_file "$test_encoder")" ||
        fail "--test-encoder does not resolve to an existing directory: $test_encoder"
    if [ "$recorded_canonical" != "$test_canonical" ]; then
        fail "the PGO manifest describes a different binary than --test-encoder:
  manifest $PGO_MANIFEST_BINARY_KEY = $recorded_canonical
  --test-encoder                    = $test_canonical
qualify the binary the manifest was written for, or pass the matching manifest"
    fi

    recorded_sha="$(manifest_field "$PGO_MANIFEST_BINARY_SHA_KEY" < "$pgo_manifest")"
    case "$recorded_sha" in
        ''|*[!0-9a-f]*)
            fail "--pgo-manifest records no valid '$PGO_MANIFEST_BINARY_SHA_KEY=' value: $pgo_manifest
rebuild the PGO encoder with the current scripts/build-encoder-pgo.sh"
            ;;
    esac
    [ "${#recorded_sha}" -eq 64 ] ||
        fail "--pgo-manifest has a malformed $PGO_MANIFEST_BINARY_SHA_KEY: $recorded_sha"
    actual_sha="$(sha256_file "$test_encoder")"
    [ "$recorded_sha" = "$actual_sha" ] ||
        fail "the PGO encoder hash does not match its build manifest:
  manifest $PGO_MANIFEST_BINARY_SHA_KEY = $recorded_sha
  actual --test-encoder SHA-256        = $actual_sha
rebuild or restore the exact PGO binary before qualification"
    pgo_optimized_binary_sha256="$recorded_sha"

    for path in $PGO_GUARDED_PATHS; do
        if ! git diff --quiet "$pgo_source_commit" HEAD -- "$path"; then
            drifted="$drifted $path"
        fi
    done
    if [ -n "$drifted" ]; then
        fail "encoder-shipped paths changed between the PGO source commit $pgo_source_commit and HEAD $git_commit:$drifted
rebuild the PGO encoder (scripts/build-encoder-pgo.sh) or qualify the commit it was built from"
    fi
}

resolve_environment() {
    local tool required_commands corpus_key corpus_file libvpx_root
    local expected_sha actual_sha expected_header actual_header

    work_dir="$(absolute_path "$work_dir")"
    opt_pgo_manifest="$(absolute_path "$opt_pgo_manifest")"
    opt_source_dir="$(absolute_path "$opt_source_dir")"
    opt_libvpx_source="$(absolute_path "$opt_libvpx_source")"
    opt_test_encoder="$(absolute_path "$opt_test_encoder")"
    opt_vpxenc="$(absolute_path "$opt_vpxenc")"
    opt_temporal_anchor="$(absolute_path "$opt_temporal_anchor")"
    opt_bdrate="$(absolute_path "$opt_bdrate")"
    opt_bdrate_svc="$(absolute_path "$opt_bdrate_svc")"
    opt_rvp9dec="$(absolute_path "$opt_rvp9dec")"
    opt_vpxdec="$(absolute_path "$opt_vpxdec")"
    opt_ffmpeg="$(absolute_program "$opt_ffmpeg")"

    # Everything the selected phase will actually invoke, checked before any
    # work begins, each refusal naming why the command is needed. Resolving the
    # list first means an undeclared phase fails loudly instead of silently
    # checking nothing.
    required_commands="$(phase_required_commands "$phase")" ||
        fail "no external command set is declared for the $phase phase"
    for tool in $required_commands; do
        command -v "$tool" >/dev/null 2>&1 ||
            fail "missing command required by the $phase phase: $tool ($(command_purpose "$tool"))"
    done

    work_dir="$(canonical_output_dir "$work_dir")"

    if [ "$resume" = "yes" ]; then
        [ -d "$work_dir" ] ||
            fail "--resume requires an existing --work-dir: $work_dir"
    else
        # A normal real run is fresh. Refuse a work directory that already
        # holds anything before a single file is written into it.
        require_absent_or_empty_dir "--work-dir" "$work_dir"
    fi

    repo_root="$(git rev-parse --show-toplevel)"
    cd "$repo_root"
    case "$work_dir" in
        "$repo_root"|"$repo_root"/*)
            fail "--work-dir must be outside the repository worktree: $work_dir"
            ;;
    esac
    git_commit="$(git rev-parse HEAD)"
    if [ -z "$(git status --porcelain)" ]; then
        git_status_clean="yes"
    else
        git_status_clean="no"
        fail "the qualification worktree is dirty; commit the exact runner and test code before collecting release evidence"
    fi
    host_triple="$(rustc -vV | awk '/^host: / { print $2 }')"
    case "$host_triple" in
        *-windows-*) exe_suffix=".exe" ;;
        *) exe_suffix="" ;;
    esac

    pgo_manifest="$opt_pgo_manifest"
    source_dir="$(default_path "$opt_source_dir" "$repo_root/tests/corpus")"
    test_encoder="$(default_path "$opt_test_encoder" \
        "$repo_root/target/encoder-pgo/optimized/release/rvp9enc$exe_suffix")"
    vpxenc="$(default_path "$opt_vpxenc" "$repo_root/build/libvpx/vpxenc$exe_suffix")"
    temporal_anchor="$(default_path "$opt_temporal_anchor" \
        "$repo_root/build/libvpx/vpx_temporal_svc_reference$exe_suffix")"
    bdrate_bin="$(default_path "$opt_bdrate" "$repo_root/target/release/bdrate$exe_suffix")"
    bdrate_svc_bin="$(default_path "$opt_bdrate_svc" \
        "$repo_root/target/release/bdrate-svc$exe_suffix")"
    rvp9dec="$(default_path "$opt_rvp9dec" "$repo_root/target/release/rvp9dec$exe_suffix")"
    vpxdec="$(default_path "$opt_vpxdec" "$repo_root/build/libvpx/vpxdec$exe_suffix")"
    ffmpeg_bin="$(default_path "$opt_ffmpeg" "ffmpeg")"
    libvpx_source="$(default_path "$opt_libvpx_source" "$repo_root/third_party/libvpx")"

    require_executable "--test-encoder" "$test_encoder"
    require_executable "--vpxenc" "$vpxenc"
    require_executable "--temporal-anchor" "$temporal_anchor"
    require_executable "--bdrate" "$bdrate_bin"
    require_executable "--bdrate-svc" "$bdrate_svc_bin"
    require_executable "--rvp9dec" "$rvp9dec"
    require_executable "--vpxdec" "$vpxdec"
    command -v "$ffmpeg_bin" >/dev/null 2>&1 ||
        fail "missing ffmpeg; install it or pass --ffmpeg"
    ffmpeg_bin="$(command -v "$ffmpeg_bin")"
    require_executable "--ffmpeg" "$ffmpeg_bin"
    require_help_flag "--bdrate" "$bdrate_bin" "--work-dir"
    require_help_flag "--bdrate" "$bdrate_bin" "--reference-input"
    require_help_flag "--bdrate-svc" "$bdrate_svc_bin" "--work-dir"
    require_help_flag "--bdrate-svc" "$bdrate_svc_bin" "--svc-kind"

    validate_pgo_manifest

    # The pinned checkout must be a standalone repository at the exact
    # revision. Without the containment check an uninitialized submodule
    # silently resolves to this superproject and reports the wrong revision.
    [ -d "$libvpx_source" ] || fail "missing libvpx source: $libvpx_source"
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
    if ! git -C "$libvpx_source" diff --quiet ||
        ! git -C "$libvpx_source" diff --cached --quiet
    then
        fail "libvpx source is dirty; the reference must be the exact pinned revision: $libvpx_source"
    fi

    libvpx_config_h="$(dirname "$vpxenc")/vpx_config.h"
    libvpx_config_mk="$(dirname "$vpxenc")/config.mk"
    [ -s "$libvpx_config_h" ] ||
        fail "missing libvpx build configuration beside vpxenc: $libvpx_config_h"
    [ -s "$libvpx_config_mk" ] ||
        fail "missing libvpx build configuration beside vpxenc: $libvpx_config_mk"
    grep -q '^#define CONFIG_VP9_ENCODER 1$' "$libvpx_config_h" ||
        fail "libvpx reference was built without the VP9 encoder: $libvpx_config_h"
    grep -q '^#define CONFIG_VP9_DECODER 1$' "$libvpx_config_h" ||
        fail "libvpx reference was built without the VP9 decoder: $libvpx_config_h"
    grep -q '^#define CONFIG_VP9_TEMPORAL_DENOISING 1$' "$libvpx_config_h" ||
        fail "libvpx reference was built without VP9 temporal denoising: $libvpx_config_h"

    # Every required corpus input, including the AQ oracle's foreman_cif clip.
    # The oracle resolves that clip through RVP9_CORPUS_DIR and merely prints
    # "[skip]" when it is absent, so an unchecked corpus turns the release's
    # one AQ oracle into a silent no-op.
    while IFS='|' read -r corpus_key corpus_file; do
        if [ ! -s "$source_dir/$corpus_file" ]; then
            fail "missing or empty corpus input ($corpus_key): $source_dir/$corpus_file
run scripts/fetch-corpus.sh or pass --source-dir"
        fi
        expected_sha="$(corpus_expected_sha256 "$corpus_file")" ||
            fail "no canonical SHA-256 is declared for corpus input: $corpus_file"
        actual_sha="$(sha256_file "$source_dir/$corpus_file")"
        [ "$actual_sha" = "$expected_sha" ] ||
            fail "wrong corpus input SHA-256 for $corpus_file: $actual_sha (expected $expected_sha)"
        expected_header="$(corpus_expected_header "$corpus_file")" ||
            fail "no canonical Y4M header is declared for corpus input: $corpus_file"
        actual_header="$(head -n 1 "$source_dir/$corpus_file")"
        [ "$actual_header" = "$expected_header" ] ||
            fail "wrong corpus Y4M header for $corpus_file: '$actual_header' (expected '$expected_header')"
    done < <(required_corpus_inputs)

    mkdir -p "$work_dir"
    work_dir="$(cd "$work_dir" && pwd -P)"
    input_dir="$work_dir/inputs"
    quality_dir="$work_dir/quality"
    quality_rows_dir="$quality_dir/rows"
    quality_summary="$quality_dir/quality-summary.csv"
    quality_aggregate="$quality_dir/quality-aggregate.csv"
    cross_decoder_csv="$quality_dir/cross-decoder.csv"
    manifest="$work_dir/manifest.txt"
}

# A fixed atomic mkdir prevents two fresh/resumed invocations from sharing a
# row tree or manifest. A killed process may leave this small lock directory;
# the next invocation fails with the recorded owner rather than guessing that
# it is stale and risking a PID-reuse or cross-host race.
release_quality_lock() {
    local recorded=""
    if [ -n "$active_quality_attempt" ] && [ -d "$active_quality_attempt" ]; then
        printf 'interrupted quality attempt retained for inspection: %s\n' \
            "$active_quality_attempt" >&2
    fi
    [ "$quality_lock_acquired" = "yes" ] || return 0
    if [ -f "$quality_lock_dir/owner" ]; then
        recorded="$(cat "$quality_lock_dir/owner")"
        [ "$recorded" = "$quality_lock_owner" ] || {
            printf 'refusing to release a quality lock now owned by %s\n' "$recorded" >&2
            return 1
        }
        unlink "$quality_lock_dir/owner"
    fi
    rmdir "$quality_lock_dir"
    quality_lock_acquired="no"
}

acquire_quality_lock() {
    local recorded="unrecorded"
    quality_lock_dir="$work_dir/.quality-run.lock"
    if ! mkdir "$quality_lock_dir" 2>/dev/null; then
        if [ -f "$quality_lock_dir/owner" ]; then
            recorded="$(cat "$quality_lock_dir/owner")"
        fi
        fail "another quality invocation owns --work-dir $work_dir: $recorded
if that process was killed, verify the recorded owner is gone and move the exact lock directory aside before --resume"
    fi
    quality_lock_owner="$(uname -n)|pid=$$|started=$(utc_now)|commit=$git_commit"
    quality_lock_acquired="yes"
    printf '%s\n' "$quality_lock_owner" > "$quality_lock_dir/owner"
}

install_quality_lock_traps() {
    trap release_quality_lock EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
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
    local corpus_key corpus_file
    {
        printf 'run_begin schema=%s\n' "$PLAN_SCHEMA"
        printf 'script=scripts/qualify-production-encoder.sh\n'
    } >> "$manifest"
    manifest_kv matrix "$MATRIX_NAME"
    manifest_kv phase "$phase"
    manifest_kv started_utc "$(utc_now)"
    manifest_kv git_commit "$git_commit"
    manifest_kv git_status_clean "$git_status_clean"
    manifest_kv git_status_entries "$(git status --porcelain | wc -l | tr -d ' ')"
    manifest_kv libvpx_commit "$libvpx_revision"
    manifest_kv host "$(uname -a)"
    manifest_kv host_triple "$host_triple"
    manifest_kv rustc \
        "$(deltic timeout "$TOOL_VERSION_TIMEOUT" rustc --version </dev/null)"
    manifest_kv cargo \
        "$(deltic timeout "$TOOL_VERSION_TIMEOUT" cargo --version </dev/null)"
    manifest_kv ffmpeg \
        "$(deltic timeout "$TOOL_VERSION_TIMEOUT" "$ffmpeg_bin" -version </dev/null 2>/dev/null | head -n 1)"
    manifest_kv work_dir "$work_dir"

    # The PGO provenance chain: which manifest, which source commit, and the
    # proof that no encoder-shipped path moved since.
    manifest_kv pgo_manifest_path "$pgo_manifest"
    manifest_kv pgo_manifest_sha256 "$pgo_manifest_sha256"
    manifest_kv pgo_source_commit "$pgo_source_commit"
    manifest_kv pgo_optimized_binary "$pgo_optimized_binary"
    manifest_kv pgo_optimized_binary_sha256 "$pgo_optimized_binary_sha256"
    manifest_kv pgo_optimized_binary_is_test_encoder "yes"
    manifest_kv pgo_guarded_paths "$(printf '%s' "$PGO_GUARDED_PATHS" | tr -s ' \n' ';')"
    manifest_kv pgo_guarded_paths_unchanged_since_source_commit "yes"

    manifest_artifact qualification_script \
        "$repo_root/scripts/qualify-production-encoder.sh"
    manifest_artifact test_encoder "$test_encoder"
    manifest_artifact vpxenc "$vpxenc"
    manifest_artifact temporal_anchor "$temporal_anchor"
    manifest_artifact libvpx_config_h "$libvpx_config_h"
    manifest_artifact libvpx_config_mk "$libvpx_config_mk"
    manifest_artifact bdrate "$bdrate_bin"
    manifest_artifact bdrate_svc "$bdrate_svc_bin"
    manifest_artifact rvp9dec "$rvp9dec"
    manifest_artifact vpxdec "$vpxdec"
    manifest_artifact ffmpeg_binary "$ffmpeg_bin"

    # Every validated corpus input, hashed from the same list the validation
    # walked -- the AQ oracle's foreman_cif clip included.
    while IFS='|' read -r corpus_key corpus_file; do
        manifest_artifact "$corpus_key" "$source_dir/$corpus_file"
    done < <(required_corpus_inputs)

    printf 'matrix_begin\n' >> "$manifest"
    print_plan >> "$manifest"
    printf 'matrix_end\n' >> "$manifest"
}

# ---------------------------------------------------------------------------
# Acceptance inputs
#
# Eight clean 60-frame inputs -- one per target/clip pair -- scaled and
# frame-rate converted exactly as scripts/qualify-spatial-svc.sh does, plus one
# deterministic noisy derivative of each for the denoiser rows. The noise is
# applied AFTER the scale/fps conversion so the noisy input stays pixel-aligned
# with the clean PSNR reference.
# ---------------------------------------------------------------------------

acceptance_input() {
    printf '%s/%s_%s_%s.y4m\n' "$input_dir" "$1" "$2" "$3"
}

validate_prepared_y4m() {
    local target="$1" path="$2" fps width height threads tiles speed_kbps p50 p99
    local header actual_bytes frame_bytes expected_bytes
    IFS='|' read -r fps width height threads tiles speed_kbps p50 p99 \
        <<< "$(target_spec "$target")"
    [ -s "$path" ] || fail "prepared acceptance input is missing or empty: $path"
    header="$(head -n 1 "$path")"
    case " $header " in
        " YUV4MPEG2 W${width} H${height} F${fps}:1 Ip "*" C420"*) ;;
        *) fail "prepared acceptance input has the wrong Y4M shape for $target: '$header' ($path)" ;;
    esac
    frame_bytes=$(( width * height * 3 / 2 ))
    expected_bytes=$(( ${#header} + 1 + FRAMES * (6 + frame_bytes) ))
    actual_bytes="$(wc -c < "$path" | tr -d ' ')"
    [ "$actual_bytes" -eq "$expected_bytes" ] ||
        fail "prepared acceptance input is not exactly $FRAMES untagged I420 frames: $path has $actual_bytes bytes, expected $expected_bytes"
}

prepared_input_count=0

prepare_inputs() {
    local target clip src clean noisy filter
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
            clean="$(acceptance_input "$clip" "$target" clean)"
            noisy="$(acceptance_input "$clip" "$target" noisy)"
            if [ ! -s "$clean" ]; then
                printf '  %s\n' "$clean"
                deltic timeout "$PREPROCESS_TIMEOUT" "$ffmpeg_bin" \
                    -hide_banner -loglevel error -nostdin -y -i "$src" \
                    -vf "$filter" -frames:v "$FRAMES" -pix_fmt yuv420p "$clean" \
                    </dev/null
                [ -s "$clean" ] || fail "ffmpeg produced no output: $clean"
            fi
            validate_prepared_y4m "$target" "$clean"
            if [ ! -s "$noisy" ]; then
                printf '  %s\n' "$noisy"
                deltic timeout "$PREPROCESS_TIMEOUT" "$ffmpeg_bin" \
                    -hide_banner -loglevel error -nostdin -y -i "$clean" \
                    -vf "$NOISE_FILTER" -frames:v "$FRAMES" -pix_fmt yuv420p "$noisy" \
                    </dev/null
                [ -s "$noisy" ] || fail "ffmpeg produced no output: $noisy"
            fi
            validate_prepared_y4m "$target" "$noisy"
            prepared_input_count=$(( prepared_input_count + 2 ))
        done
    done
}

record_acceptance_inputs() {
    local target clip kind input
    for target in $TARGETS; do
        for clip in $(target_clips "$target"); do
            for kind in clean noisy; do
                input="$(acceptance_input "$clip" "$target" "$kind")"
                [ -s "$input" ] || fail "acceptance input is missing or empty: $input"
                manifest_artifact "input_${kind}_${clip}_${target}" "$input"
            done
        done
    done
    manifest_kv acceptance_inputs "$prepared_input_count"
}

# Stable identity of everything that may affect a quality row. The printed
# plan binds every matrix row and gate; the remaining lines bind the committed
# runner, encoder/reference tools, corpus, generated inputs, and host. Paths
# are included as well as hashes because helper binaries may resolve adjacent
# files at runtime.
run_fingerprint_payload() {
    local target clip kind input corpus_key corpus_file path

    printf 'receipt_schema=%s\n' "$QUALITY_RECEIPT_SCHEMA"
    printf 'phase=%s\n' "$phase"
    printf 'work_dir=%s\n' "$work_dir"
    printf 'git_commit=%s\n' "$git_commit"
    printf 'host_triple=%s\n' "$host_triple"
    printf 'host=%s\n' "$(uname -a)"
    printf 'plan_sha256=%s\n' "$(print_plan | shasum -a 256 | awk '{ print $1 }')"
    printf 'noise_filter=%s\n' "$NOISE_FILTER"
    printf 'pgo_manifest=%s|%s\n' "$pgo_manifest" "$pgo_manifest_sha256"
    printf 'pgo_source_commit=%s\n' "$pgo_source_commit"
    printf 'libvpx_revision=%s\n' "$libvpx_revision"

    for path in \
        "$repo_root/scripts/qualify-production-encoder.sh" \
        "$test_encoder" "$vpxenc" "$temporal_anchor" \
        "$libvpx_config_h" "$libvpx_config_mk" \
        "$bdrate_bin" "$bdrate_svc_bin" "$rvp9dec" "$vpxdec" \
        "$ffmpeg_bin"
    do
        printf 'artifact=%s|%s\n' "$path" "$(sha256_file "$path")"
    done

    while IFS='|' read -r corpus_key corpus_file; do
        path="$source_dir/$corpus_file"
        printf 'corpus=%s|%s|%s\n' "$corpus_key" "$path" "$(sha256_file "$path")"
    done < <(required_corpus_inputs)

    for target in $TARGETS; do
        for clip in $(target_clips "$target"); do
            for kind in clean noisy; do
                input="$(acceptance_input "$clip" "$target" "$kind")"
                printf 'input=%s|%s\n' "$input" "$(sha256_file "$input")"
            done
        done
    done
}

current_run_fingerprint() {
    run_fingerprint_payload | shasum -a 256 | awk '{ print $1 }'
}

manifest_key_count() {
    awk -v key="$1" 'index($0, key "=") == 1 { n++ } END { print n + 0 }' "$2"
}

# Pure-text receipt oracle. Exactly one copy of every binding key must agree;
# a duplicate is ambiguity, not a later value overriding an earlier one.
receipt_consistency_status() {
    local expected_run="$1" expected_row="$2" expected_artifact="$3"
    awk -v schema="$QUALITY_RECEIPT_SCHEMA" -v run="$expected_run" \
        -v row="$expected_row" -v artifact="$expected_artifact" '
        index($0, "receipt_schema=") == 1 {
            schema_n++; schema_v = substr($0, 16)
        }
        index($0, "run_fingerprint_sha256=") == 1 {
            run_n++; run_v = substr($0, 24)
        }
        index($0, "row_id=") == 1 {
            row_n++; row_v = substr($0, 8)
        }
        index($0, "artifact_sha256=") == 1 {
            artifact_n++; artifact_v = substr($0, 17)
        }
        END {
            if (schema_n != 1 || run_n != 1 || row_n != 1 || artifact_n != 1) {
                print "ambiguous"; exit
            }
            if (schema_v != schema) { print "wrong-schema"; exit }
            if (run_v != run) { print "wrong-run"; exit }
            if (row_v != row) { print "wrong-row"; exit }
            if (artifact_v != artifact) { print "wrong-artifact"; exit }
            print "valid"
        }'
}

seal_fresh_run() {
    [ "$prepared_input_count" -eq "$EXPECTED_PREPARED_INPUTS" ] ||
        fail "prepared $prepared_input_count acceptance inputs before sealing, expected $EXPECTED_PREPARED_INPUTS"
    run_fingerprint="$(current_run_fingerprint)"
    manifest_kv run_fingerprint_sha256 "$run_fingerprint"
    manifest_kv acceptance_inputs_sealed "yes"
}

# Resume never repairs or regenerates an input. Every prepared input must still
# have the exact shape and hash recorded before the original row campaign
# started. A run interrupted during preparation is cheap to restart fresh and
# deliberately cannot be resumed.
validate_resume_inputs() {
    local target clip kind input key recorded actual
    prepared_input_count=0
    for target in $TARGETS; do
        for clip in $(target_clips "$target"); do
            for kind in clean noisy; do
                input="$(acceptance_input "$clip" "$target" "$kind")"
                validate_prepared_y4m "$target" "$input"
                key="input_${kind}_${clip}_${target}_sha256"
                [ "$(manifest_key_count "$key" "$manifest")" -eq 1 ] ||
                    fail "--resume requires exactly one $key record in $manifest"
                recorded="$(manifest_field "$key" < "$manifest")"
                actual="$(sha256_file "$input")"
                [ "$actual" = "$recorded" ] ||
                    fail "--resume input hash mismatch for $input: $actual (sealed $recorded)"
                prepared_input_count=$(( prepared_input_count + 1 ))
            done
        done
    done
}

validate_resume_contract() {
    local recorded
    [ -f "$manifest" ] ||
        fail "--resume found no sealed manifest: $manifest"
    [ "$(manifest_key_count acceptance_inputs_sealed "$manifest")" -eq 1 ] &&
        [ "$(manifest_field acceptance_inputs_sealed < "$manifest")" = "yes" ] ||
        fail "--resume requires one acceptance_inputs_sealed=yes record in $manifest"
    [ "$(manifest_key_count run_fingerprint_sha256 "$manifest")" -eq 1 ] ||
        fail "--resume requires exactly one run_fingerprint_sha256 record in $manifest"
    [ "$(grep -c '^run_end ' "$manifest" || true)" -eq 0 ] ||
        fail "--resume refuses a run that already has a run_end record: $manifest"
    [ ! -e "$quality_dir/completed" ] ||
        fail "--resume found a published quality result without run_end; finalization is ambiguous: $quality_dir/completed"

    validate_resume_inputs
    recorded="$(manifest_field run_fingerprint_sha256 < "$manifest")"
    run_fingerprint="$(current_run_fingerprint)"
    [ "$run_fingerprint" = "$recorded" ] ||
        fail "--resume fingerprint mismatch: current $run_fingerprint, sealed $recorded
use the exact committed tree, tools, corpus, inputs, host, and work directory that started this run"
    manifest_kv resumed_utc "$(utc_now)"
}

# ---------------------------------------------------------------------------
# Quality phase, with its inseparable cross-decoder checks
# ---------------------------------------------------------------------------

quality_row_count=0
reused_row_count=0
quality_clip_gate_count=0
quality_regression_count=0
aggregate_group_count=0
aggregate_gate_count=0
aggregate_failure_count=0
cross_stream_count=0
cross_run_count=0
aq_oracle_count=0
aq_oracle_failure_count=0
aq_result_status=""
aq_result_exit_code=""
aq_result_gain_rvp9=""
aq_result_gain_libvpx=""
aq_result_delta=""
aq_result_activation=""
aq_result_log_sha256=""
aq_result_summary_sha256=""

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

# Which operating point a retained stream name denotes. Derived from the name
# the runner actually wrote, so an unexpected shape cannot be silently binned.
stream_kind() {
    local name="$1" layer
    case "$name" in
        *_full.ivf)
            printf 'full\n'
            ;;
        *_tl[0-9].ivf)
            layer="${name##*_tl}"
            printf 'tl%s\n' "${layer%.ivf}"
            ;;
        test_*k.ivf)
            printf 'single\n'
            ;;
        *)
            printf 'unknown stream shape: %s\n' "$name" >&2
            return 1
            ;;
    esac
}

# One rvp9 candidate stream: rvp9dec and the pinned vpxdec must agree
# byte-for-byte on non-empty I420, and ffmpeg must decode it. ffmpeg is a
# decode-success oracle only. A failure here aborts the phase immediately --
# a corrupt bitstream invalidates every number measured from it.
cross_check_stream() {
    local row_id="$1" ivf="$2" log="$3"
    local name bytes stream_sha rvp9_sha vpx_sha ffmpeg_status kind
    local identical="yes" status="PASS"

    name="$(basename "$ivf")"
    kind="$(stream_kind "$name")" ||
        fail "$row_id: unrecognised retained stream name: $name"
    [ -s "$ivf" ] || fail "$row_id: expected candidate stream is missing or empty: $ivf"
    bytes="$(wc -c < "$ivf" | tr -d ' ')"
    stream_sha="$(sha256_file "$ivf")"

    if ! rvp9_sha="$(decode_sha_rvp9 "$ivf" "$log")"; then
        fail "$row_id: rvp9dec failed on $name (log $log)"
    fi
    if ! vpx_sha="$(decode_sha_vpx "$ivf" "$log")"; then
        fail "$row_id: pinned vpxdec failed on $name (log $log)"
    fi
    ffmpeg_status="ok"
    if ! deltic timeout "$DECODE_TIMEOUT" "$ffmpeg_bin" \
        -hide_banner -loglevel error -nostdin -c:v vp9 -i "$ivf" -f null - \
        >>"$log" 2>&1 </dev/null
    then
        ffmpeg_status="failed"
    fi
    cross_stream_count=$(( cross_stream_count + 1 ))
    cross_run_count=$(( cross_run_count + DECODERS_PER_STREAM ))

    [ "$rvp9_sha" = "$vpx_sha" ] || { identical="no"; status="FAIL"; }
    [ "$ffmpeg_status" = "ok" ] || status="FAIL"
    if [ "$rvp9_sha" = "$EMPTY_SHA256" ] || [ "$vpx_sha" = "$EMPTY_SHA256" ]; then
        identical="empty-decode"
        status="FAIL"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$row_id" "$name" "$kind" "$bytes" "$stream_sha" \
        "$rvp9_sha" "$vpx_sha" "$identical" "$ffmpeg_status" "$status" \
        >> "$cross_decoder_csv"

    if [ "$status" != "PASS" ]; then
        fail "$row_id: cross-decoder mismatch on $name (rvp9dec=$rvp9_sha vpxdec=$vpx_sha ffmpeg=$ffmpeg_status)"
    fi
}

# Check the exact enumerated stream set. The glob count is a surplus guard, not
# the oracle: the oracle is the name list row_stream_names() declares.
cross_check_row() {
    local row_id="$1" feature="$2" streams="$3" grid="$4" log="$5"
    local name expected actual

    expected="$(row_stream_names "$feature" "$grid" | awk 'END { print NR + 0 }')"
    actual="$(find "$streams" -maxdepth 1 -type f -name 'test_*.ivf' | awk 'END { print NR + 0 }')"
    if [ "$actual" -ne "$expected" ]; then
        fail "$row_id: found $actual retained rvp9 candidate streams, expected $expected"
    fi

    while IFS= read -r name; do
        cross_check_stream "$row_id" "$streams/$name" "$log"
    done < <(row_stream_names "$feature" "$grid")
}

# Delete only this row's generated Y4M, and only from the row's own streams
# directory inside the work dir. Retained IVF, CSV, logs, and the caller's
# inputs are untouched.
row_attempt_path_status() {
    local streams="$1" resolved rows_resolved attempt_dir attempt_name
    [ -d "$streams" ] || { printf 'missing\n'; return; }
    resolved="$(cd "$streams" && pwd -P)"
    rows_resolved="$(cd "$quality_rows_dir" && pwd -P)"
    attempt_dir="$(dirname "$resolved")"
    attempt_name="$(basename "$attempt_dir")"
    if [ "$(dirname "$attempt_dir")" != "$rows_resolved" ]; then
        printf 'not-direct\n'
        return
    fi
    case "$attempt_name" in
        .*.attempt.*) printf 'valid\n' ;;
        *) printf 'not-attempt\n' ;;
    esac
}

discard_row_y4m() {
    local streams="$1" status
    status="$(row_attempt_path_status "$streams")"
    [ "$status" = "valid" ] ||
        fail "refusing to delete generated Y4M from an invalid row path ($status): $streams"
    local resolved
    resolved="$(cd "$streams" && pwd -P)"
    find "$resolved" -maxdepth 1 -type f -name '*.y4m' -delete
}

# Per-layer BD-rate percentages as `layer,value` lines. bdrate reports a single
# curve; bdrate-svc reports one per temporal operating point.
row_layer_bdrates() {
    local runner="$1" csv="$2"
    if [ "$runner" = "bdrate" ]; then
        awk -F, '$1 == "bdrate_pct" { print "0," $2 }' "$csv"
    else
        awk -F, '$1 == "bdrate_pct" { print $2 "," $3 }' "$csv"
    fi
}

row_bdrate_lines() {
    awk -F, '$1 == "bdrate_pct" { n++ } END { print n + 0 }' "$1"
}

# Validate an unordered whitespace-separated layer-id set. Exactly one of
# every id in 0..expected-1 must be present; duplicates, gaps, and out-of-range
# ids are all invalid.
layer_id_set_status() {
    local expected="$1" ids="$2" id seen=" " count=0 want
    for id in $ids; do
        case "$id" in
            ''|*[!0-9]*) printf 'invalid\n'; return ;;
        esac
        [ "$id" -lt "$expected" ] || { printf 'invalid\n'; return; }
        case "$seen" in
            *" $id "*) printf 'invalid\n'; return ;;
        esac
        seen="$seen$id "
        count=$(( count + 1 ))
    done
    [ "$count" -eq "$expected" ] || { printf 'invalid\n'; return; }
    want=0
    while [ "$want" -lt "$expected" ]; do
        case "$seen" in
            *" $want "*) ;;
            *) printf 'invalid\n'; return ;;
        esac
        want=$(( want + 1 ))
    done
    printf 'valid\n'
}

# Hash every retained artifact whose contents justify reusing a completed row.
# The ordered stream-name oracle makes missing, surplus, or renamed IVF files
# visible even if a receipt itself is present.
row_artifact_fingerprint() {
    local feature="$1" grid="$2" dir="$3" name path
    for name in bdrate.csv runner.log quality-fragment.csv \
        cross-decoder-fragment.csv cross-decoder.log
    do
        path="$dir/$name"
        [ -f "$path" ] || return 1
        printf '%s|%s|%s\n' "$name" "$(wc -c < "$path" | tr -d ' ')" \
            "$(sha256_file "$path")"
    done
    while IFS= read -r name; do
        path="$dir/streams/$name"
        [ -f "$path" ] || return 1
        printf 'streams/%s|%s|%s\n' "$name" \
            "$(wc -c < "$path" | tr -d ' ')" "$(sha256_file "$path")"
    done < <(row_stream_names "$feature" "$grid")
}

row_artifact_sha256() {
    row_artifact_fingerprint "$@" | shasum -a 256 | awk '{ print $1 }'
}

validate_quality_fragment() {
    local fragment="$1" row_id="$2" feature="$3" mode="$4" target="$5"
    local clip="$6" cpu="$7" aq="$8" denoiser="$9" layers="${10}"
    local threads="${11}" tiles="${12}" grid="${13}" runner="${14}"
    local anchor_kind="${15}" layer value status expected_status layer_ids=""
    local count=0 failures=0
    local got_row got_feature got_mode got_target got_clip got_cpu got_aq
    local got_denoiser got_layers got_threads got_tiles got_grid got_runner
    local got_anchor got_input got_reference got_gate extra

    [ -f "$fragment" ] || fail "$row_id: missing quality fragment: $fragment"
    while IFS=',' read -r got_row got_feature got_mode got_target got_clip \
        got_cpu got_aq got_denoiser got_layers got_threads got_tiles \
        got_grid got_runner got_anchor got_input got_reference layer value \
        got_gate status extra
    do
        [ -z "$extra" ] || fail "$row_id: quality fragment has surplus columns"
        [ "$got_row|$got_feature|$got_mode|$got_target|$got_clip" = \
          "$row_id|$feature|$mode|$target|$clip" ] ||
            fail "$row_id: quality fragment identity mismatch"
        [ "$got_cpu|$got_aq|$got_denoiser|$got_layers|$got_threads|$got_tiles" = \
          "$cpu|$aq|$denoiser|$layers|$threads|$tiles" ] ||
            fail "$row_id: quality fragment configuration mismatch"
        [ "$got_grid|$got_runner|$got_anchor" = \
          "$(printf '%s' "$grid" | tr ',' ';')|$runner|$anchor_kind" ] ||
            fail "$row_id: quality fragment runner mismatch"
        [ "$got_input|$got_reference|$got_gate" = \
          "$(row_input_kind "$feature")|$(row_reference_kind "$feature")|$MAX_IMPORTANT_CLIP_BDRATE_PCT" ] ||
            fail "$row_id: quality fragment input or gate mismatch"
        require_number "$row_id TL$layer BD-rate" "$value"
        expected_status="PASS"
        if ! number_le "$value" "$MAX_IMPORTANT_CLIP_BDRATE_PCT"; then
            expected_status="FAIL"
            failures=$(( failures + 1 ))
            printf '  QUALITY MISS %s TL%s: %s%% exceeds +%s%% (reused row)\n' \
                "$row_id" "$layer" "$value" "$MAX_IMPORTANT_CLIP_BDRATE_PCT" >&2
        fi
        [ "$status" = "$expected_status" ] ||
            fail "$row_id: quality fragment status $status disagrees with BD-rate $value"
        layer_ids="$layer_ids $layer"
        count=$(( count + 1 ))
    done < "$fragment"

    [ "$count" -eq "$layers" ] ||
        fail "$row_id: quality fragment has $count layer rows, expected $layers"
    [ "$(layer_id_set_status "$layers" "$layer_ids")" = "valid" ] ||
        fail "$row_id: quality fragment has invalid layer ids '$layer_ids'"
    [ "$(awk -F, '{ print $17 "," $18 }' "$fragment")" = \
      "$(row_layer_bdrates "$runner" "$quality_rows_dir/$row_id/bdrate.csv")" ] ||
        fail "$row_id: quality fragment disagrees with its retained BD-rate CSV"

    quality_clip_gate_count=$(( quality_clip_gate_count + count ))
    quality_regression_count=$(( quality_regression_count + failures ))
}

publish_row_receipt() {
    local row_id="$1" feature="$2" grid="$3" dir="$4" receipt artifact_sha
    receipt="$dir/receipt"
    artifact_sha="$(row_artifact_sha256 "$feature" "$grid" "$dir")" ||
        fail "$row_id: cannot fingerprint the completed row artifacts"
    {
        printf 'receipt_schema=%s\n' "$QUALITY_RECEIPT_SCHEMA"
        printf 'run_fingerprint_sha256=%s\n' "$run_fingerprint"
        printf 'row_id=%s\n' "$row_id"
        printf 'artifact_sha256=%s\n' "$artifact_sha"
        printf 'completed_utc=%s\n' "$(utc_now)"
    } > "$dir/receipt.partial"
    mv "$dir/receipt.partial" "$receipt"
}

completed_row_receipt_status() {
    local expected_run="$1" row_id="$2" feature="$3" grid="$4" dir="$5"
    local artifact_sha
    [ -f "$dir/receipt" ] || { printf 'missing\n'; return; }
    if ! artifact_sha="$(row_artifact_sha256 "$feature" "$grid" "$dir")"; then
        printf 'incomplete\n'
        return
    fi
    receipt_consistency_status "$expected_run" "$row_id" "$artifact_sha" \
        < "$dir/receipt"
}

decoder_recheck_status() {
    if cmp -s "$1" "$2"; then
        printf 'match\n'
    else
        printf 'mismatch\n'
    fi
}

reuse_quality_row() {
    local feature="$1" mode="$2" target="$3" clip="$4" cpu="$5"
    local aq="$6" denoiser="$7" layers="$8"
    local fps width height threads tiles speed_kbps p50 p99 grid row_id dir
    local receipt_status summary_out cross_out recheck log
    local actual_streams expected_streams

    IFS='|' read -r fps width height threads tiles speed_kbps p50 p99 \
        <<< "$(target_spec "$target")"
    grid="$(target_bitrates "$target")"
    row_id="${feature}-${mode}-${target}-${clip}-cpu${cpu}"
    dir="$quality_rows_dir/$row_id"
    receipt_status="$(completed_row_receipt_status \
        "$run_fingerprint" "$row_id" "$feature" "$grid" "$dir")"
    [ "$receipt_status" = "valid" ] ||
        fail "$row_id: completed row receipt is not reusable ($receipt_status)"
    [ -z "$(find "$dir/streams" -maxdepth 1 -type f -name '*.y4m' -print)" ] ||
        fail "$row_id: completed row still contains generated Y4M output"
    expected_streams="$(row_stream_names "$feature" "$grid" | awk 'END { print NR + 0 }')"
    actual_streams="$(find "$dir/streams" -maxdepth 1 -type f -name 'test_*.ivf' | awk 'END { print NR + 0 }')"
    [ "$actual_streams" -eq "$expected_streams" ] ||
        fail "$row_id: completed row has $actual_streams streams, expected $expected_streams"

    validate_quality_fragment "$dir/quality-fragment.csv" "$row_id" \
        "$feature" "$mode" "$target" "$clip" "$cpu" "$aq" "$denoiser" \
        "$layers" "$threads" "$tiles" "$grid" "$(row_runner_kind "$feature")" \
        "$(row_anchor_kind "$feature")"

    summary_out="$quality_summary"
    cross_out="$cross_decoder_csv"
    recheck="$quality_recheck_dir/$row_id.csv"
    log="$quality_recheck_dir/$row_id.log"
    [ ! -e "$recheck" ] && [ ! -e "$log" ] ||
        fail "$row_id: recheck output path already exists"
    : > "$recheck"
    local cross_decoder_csv="$recheck"
    cross_check_row "$row_id" "$feature" "$dir/streams" "$grid" "$log"
    [ "$(decoder_recheck_status "$recheck" "$dir/cross-decoder-fragment.csv")" = "match" ] ||
        fail "$row_id: fresh decoder observations disagree with the sealed row"

    cat "$dir/quality-fragment.csv" >> "$summary_out"
    cat "$dir/cross-decoder-fragment.csv" >> "$cross_out"
    quality_row_count=$(( quality_row_count + 1 ))
    reused_row_count=$(( reused_row_count + 1 ))
    printf '[quality %3d/%d] %s (reused; all candidate streams re-decoded)\n' \
        "$quality_row_count" "$EXPECTED_QUALITY_CURVES" "$row_id"
    return 0
}

run_quality_row() {
    local feature="$1" mode="$2" target="$3" clip="$4" cpu="$5"
    local aq="$6" denoiser="$7" layers="$8"
    local fps width height threads tiles speed_kbps p50 p99
    local grid row_id final_dir attempt streams runner anchor_kind encode_input reference_input
    local summary_out="$quality_summary" cross_out="$cross_decoder_csv"
    local rc=0 recorded=0 layer value status layer_ids=""

    IFS='|' read -r fps width height threads tiles speed_kbps p50 p99 \
        <<< "$(target_spec "$target")"
    grid="$(target_bitrates "$target")"
    row_id="${feature}-${mode}-${target}-${clip}-cpu${cpu}"
    final_dir="$quality_rows_dir/$row_id"
    if [ -e "$final_dir" ]; then
        [ -f "$final_dir/receipt" ] ||
            fail "$row_id: row directory exists without a receipt: $final_dir"
        # Call directly, not as an `if` condition: Bash disables `errexit` for
        # the full dynamic subtree of a conditional function call.
        reuse_quality_row "$feature" "$mode" "$target" "$clip" "$cpu" \
            "$aq" "$denoiser" "$layers"
        return 0
    fi
    attempt="$quality_rows_dir/.${row_id}.attempt.$(date -u '+%Y%m%dT%H%M%SZ')-$$"
    [ ! -e "$attempt" ] || fail "$row_id: attempt path already exists: $attempt"
    mkdir "$attempt"
    active_quality_attempt="$attempt"
    streams="$attempt/streams"
    mkdir "$streams"

    # Called functions see these dynamically scoped paths, so a row writes
    # only private fragments until its receipt and directory publish.
    local quality_summary="$attempt/quality-fragment.csv"
    local cross_decoder_csv="$attempt/cross-decoder-fragment.csv"
    : > "$quality_summary"
    : > "$cross_decoder_csv"

    runner="$(row_runner_kind "$feature")"
    anchor_kind="$(row_anchor_kind "$feature")"
    encode_input="$(acceptance_input "$clip" "$target" "$(row_input_kind "$feature")")"
    reference_input="$(acceptance_input "$clip" "$target" "$(row_reference_kind "$feature")")"
    [ -s "$encode_input" ] || fail "$row_id: encode input is missing: $encode_input"
    [ -s "$reference_input" ] || fail "$row_id: PSNR reference is missing: $reference_input"

    quality_row_count=$(( quality_row_count + 1 ))
    printf '[quality %3d/%d] %s (%s, %s anchor, %s input)\n' \
        "$quality_row_count" "$EXPECTED_QUALITY_CURVES" "$row_id" \
        "$runner" "$anchor_kind" "$(row_input_kind "$feature")"

    if [ "$runner" = "bdrate" ]; then
        deltic timeout "$QUALITY_ROW_TIMEOUT" "$bdrate_bin" \
            --anchor-encoder "$vpxenc" \
            --test-encoder "$test_encoder" \
            --decoder "$vpxdec" \
            --input "$encode_input" \
            --reference-input "$reference_input" \
            --cpu-used "$cpu" \
            --bitrates "$grid" \
            --end-usage "$mode" \
            --fps "$fps/1" \
            --threads "$threads" \
            --tile-columns "$tiles" \
            --lag-in-frames "$LAG_IN_FRAMES" \
            --tune-content "$TUNE_CONTENT" \
            --aq-mode "$aq" \
            --noise-sensitivity "$denoiser" \
            --limit "$FRAMES" \
            --anchor-arg "--kf-max-dist=$KEYFRAME_INTERVAL" \
            --test-arg "--kf-max-dist=$KEYFRAME_INTERVAL" \
            --work-dir "$streams" \
            --csv "$attempt/bdrate.csv" \
            > "$attempt/runner.log" 2>&1 </dev/null || rc=$?
        # bdrate gates nothing itself, so any non-zero exit is a harness or
        # measurement failure, never a recorded quality miss.
        [ "$rc" -eq 0 ] ||
            fail "$row_id: bdrate failed (exit $rc, log $attempt/runner.log)"
    else
        deltic timeout "$QUALITY_ROW_TIMEOUT" "$bdrate_svc_bin" \
            --anchor-encoder "$temporal_anchor" \
            --test-encoder "$test_encoder" \
            --decoder "$vpxdec" \
            --input "$encode_input" \
            --reference-input "$reference_input" \
            --svc-kind temporal \
            --temporal-layers "$layers" \
            --cpu-used "$cpu" \
            --end-usage "$mode" \
            --threads "$threads" \
            --tile-columns "$tiles" \
            --aq-mode "$aq" \
            --noise-sensitivity "$denoiser" \
            --limit "$FRAMES" \
            --keyframe-interval "$KEYFRAME_INTERVAL" \
            --bitrates "$grid" \
            --max-bdrate-pct "$MAX_IMPORTANT_CLIP_BDRATE_PCT" \
            --work-dir "$streams" \
            --csv "$attempt/bdrate.csv" \
            > "$attempt/runner.log" 2>&1 </dev/null || rc=$?
        # bdrate-svc writes its CSV before returning its own gate verdict, so a
        # non-zero exit with a complete CSV is a quality miss to record, while
        # a short or absent CSV is a real failure.
        if [ "$rc" -ne 0 ]; then
            if [ ! -s "$attempt/bdrate.csv" ] ||
                [ "$(row_bdrate_lines "$attempt/bdrate.csv")" -ne "$layers" ]
            then
                fail "$row_id: bdrate-svc failed before completing its measurement (exit $rc, log $attempt/runner.log)"
            fi
        fi
    fi

    [ -s "$attempt/bdrate.csv" ] || fail "$row_id: runner wrote no CSV: $attempt/bdrate.csv"

    while IFS=',' read -r layer value; do
        require_number "$row_id TL$layer BD-rate" "$value"
        status="PASS"
        if ! number_le "$value" "$MAX_IMPORTANT_CLIP_BDRATE_PCT"; then
            status="FAIL"
            quality_regression_count=$(( quality_regression_count + 1 ))
            printf '  QUALITY MISS %s TL%s: %s%% exceeds +%s%%\n' \
                "$row_id" "$layer" "$value" "$MAX_IMPORTANT_CLIP_BDRATE_PCT" >&2
        fi
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$row_id" "$feature" "$mode" "$target" "$clip" "$cpu" "$aq" \
            "$denoiser" "$layers" "$threads" "$tiles" \
            "$(printf '%s' "$grid" | tr ',' ';')" "$runner" "$anchor_kind" \
            "$(row_input_kind "$feature")" "$(row_reference_kind "$feature")" \
            "$layer" "$value" "$MAX_IMPORTANT_CLIP_BDRATE_PCT" "$status" \
            >> "$quality_summary"
        layer_ids="$layer_ids $layer"
        recorded=$(( recorded + 1 ))
        quality_clip_gate_count=$(( quality_clip_gate_count + 1 ))
    done < <(row_layer_bdrates "$runner" "$attempt/bdrate.csv")

    [ "$recorded" -eq "$layers" ] ||
        fail "$row_id: recorded $recorded layer BD-rates, expected $layers"
    [ "$(layer_id_set_status "$layers" "$layer_ids")" = "valid" ] ||
        fail "$row_id: runner reported an invalid layer-id set '$layer_ids' (expected exactly 0..$(( layers - 1 )))"

    cross_check_row "$row_id" "$feature" "$streams" "$grid" "$attempt/cross-decoder.log"
    discard_row_y4m "$streams"
    publish_row_receipt "$row_id" "$feature" "$grid" "$attempt"
    mv "$attempt" "$final_dir"
    [ -f "$final_dir/receipt" ] &&
        [ ! -e "$final_dir/$(basename "$attempt")" ] ||
        fail "$row_id: row publication nested instead of replacing the absent final path"
    active_quality_attempt=""
    cat "$final_dir/quality-fragment.csv" >> "$summary_out"
    cat "$final_dir/cross-decoder-fragment.csv" >> "$cross_out"
}

# Equal-weight arithmetic mean of the per-clip BD-rate percentages, one gate per
# feature/mode/cpu/target/layer group.
compute_quality_aggregates() {
    local feature mode target cpu aq denoiser layers
    local group_id clips clip_count layer count mean status

    while IFS='|' read -r feature mode target cpu aq denoiser layers; do
        aggregate_group_count=$(( aggregate_group_count + 1 ))
        clips="$(target_clips "$target")"
        clip_count="$(printf '%s\n' "$clips" | wc -w | tr -d ' ')"
        layer=0
        while [ "$layer" -lt "$layers" ]; do
            group_id="${feature}-${mode}-${target}-cpu${cpu}-tl${layer}"
            count="$(awk -F, -v f="$feature" -v m="$mode" -v t="$target" \
                -v c="$cpu" -v l="$layer" \
                'NR > 1 && $2 == f && $3 == m && $4 == t && $6 == c && $17 == l { n++ } END { print n + 0 }' \
                "$quality_summary")"
            [ "$count" -eq "$clip_count" ] ||
                fail "$group_id: aggregated $count clip BD-rates, expected $clip_count"
            mean="$(awk -F, -v f="$feature" -v m="$mode" -v t="$target" \
                -v c="$cpu" -v l="$layer" \
                'NR > 1 && $2 == f && $3 == m && $4 == t && $6 == c && $17 == l { s += $18; n++ }
                 END { if (n > 0) printf "%.6f\n", s / n }' \
                "$quality_summary")"
            require_number "$group_id mean BD-rate" "$mean"
            status="PASS"
            if ! number_le "$mean" "$MAX_AGGREGATE_BDRATE_PCT"; then
                status="FAIL"
                aggregate_failure_count=$(( aggregate_failure_count + 1 ))
                printf '  AGGREGATE MISS %s: mean %s%% exceeds +%s%%\n' \
                    "$group_id" "$mean" "$MAX_AGGREGATE_BDRATE_PCT" >&2
            fi
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "$group_id" "$feature" "$mode" "$target" "$cpu" "$aq" \
                "$denoiser" "$layers" "$(printf '%s' "$clips" | tr ' ' ';')" \
                "$clip_count" "$layer" "$mean" "$MAX_AGGREGATE_BDRATE_PCT" \
                "$status" \
                >> "$quality_aggregate"
            aggregate_gate_count=$(( aggregate_gate_count + 1 ))
            layer=$(( layer + 1 ))
        done
    done < <(emit_speed_groups)
}

# ---------------------------------------------------------------------------
# AQ gain-parity oracle
#
# The authoritative Stage 4b Foreman four-curve test, run once after the corpus
# curves. It is a separate required check, not a reinterpretation of the broad
# aq3 rows: those measure rvp9-vs-libvpx BD-rate, this measures whether AQ
# earns the same gain in both encoders.
# ---------------------------------------------------------------------------

# Count log lines exactly equal to a literal. It always exits 0: `grep -c`
# exits 1 when it counts nothing, which under `set -e` would abort the run
# before the count could be turned into a reason.
count_lines_exact() {
    awk -v want="$2" '$0 == want { n++ } END { print n + 0 }' "$1"
}

# The trailing `: <value> %` of the last line matching a pattern. The pattern
# must not contain a backslash escape: `awk -v` expands escapes inside the
# assignment, so a `\(` reaches the regex engine as an unbalanced `(`.
oracle_metric() {
    awk -v pat="$2" '$0 ~ pat { line = $0 }
        END {
            if (line != "") {
                n = split(line, parts, ":")
                value = parts[n]
                gsub(/[ %]/, "", value)
                sub(/^\+/, "", value)
                print value
            }
        }' "$1"
}

run_aq_oracle() {
    local log="$quality_output_dir/aq-oracle.log"
    local summary="$quality_output_dir/aq-oracle-summary.csv"
    local rc=0 status="PASS" gate_lines gain_rvp9 gain_libvpx delta
    local activation_status
    local -a reasons

    aq_oracle_count=$(( aq_oracle_count + 1 ))
    printf '[oracle %d/%d] %s\n' \
        "$aq_oracle_count" "$EXPECTED_REQUIRED_ORACLES" "$AQ_ORACLE_TEST_NAME"

    env "$AQ_ORACLE_CORPUS_ENV=$source_dir" "$AQ_ORACLE_VPXENC_ENV=$vpxenc" \
        "$AQ_ORACLE_VPXDEC_ENV=$vpxdec" \
        deltic timeout "$AQ_ORACLE_TIMEOUT" \
        cargo test --release -p "$AQ_ORACLE_PACKAGE" \
        --test "$AQ_ORACLE_TEST_TARGET" "$AQ_ORACLE_TEST_NAME" \
        -- --exact --ignored --nocapture --test-threads=1 \
        > "$log" 2>&1 </dev/null || rc=$?

    reasons=()
    [ "$rc" -eq 0 ] || reasons+=("cargo test exited $rc")
    grep -q '^test result: ok\.' "$log" ||
        reasons+=("the log carries no 'test result: ok.' line")

    # An unmeasurable or skipped run is a failure, not a pass by omission.
    if grep -E 'NOT MEASURABLE|\[skip\]|\[libvpx skipped\]|libvpx-stock not available|Curves 3/4 skipped' \
        "$log" >/dev/null
    then
        reasons+=("the log reports a skipped or unmeasurable curve")
    fi

    gain_rvp9="$(oracle_metric "$log" 'rvp9 +AQ gain :')"
    gain_libvpx="$(oracle_metric "$log" 'libvpx +AQ gain :')"
    delta="$(oracle_metric "$log" '^Delta .*gain_rvp9')"

    # Both encoders must report finite gains before their parity delta means
    # anything. PLAN.md's Stage 4b exit is parity within 0.3 percentage point;
    # the pinned encoder itself shows a positive AQ penalty on this slice.
    if ! is_number "$gain_rvp9"; then
        reasons+=("rvp9 AQ gain is not a finite number: '${gain_rvp9:-absent}'")
        gain_rvp9="unavailable"
    fi
    if ! is_number "$gain_libvpx"; then
        reasons+=("libvpx AQ gain is not a finite number: '${gain_libvpx:-absent}'")
        gain_libvpx="unavailable"
    fi
    if ! is_number "$delta"; then
        reasons+=("the log reports no numeric gain delta: '${delta:-absent}'")
        delta="unavailable"
    elif ! number_abs_le "$delta" "$AQ_GAIN_DELTA_MAX_PP"; then
        reasons+=("AQ gain delta $delta exceeds $AQ_GAIN_DELTA_MAX_PP percentage point")
    fi

    # Cyclic-refresh activation is REQUIRED, and its absence is a failure --
    # never a pass by omission. A negative AQ gain says AQ moved the numbers;
    # only an activation report says cyclic refresh actually engaged.
    #
    activation_status="$(aq_activation_status "$AQ_ORACLE_ACTIVATION_PREFIX" < "$log")"
    case "$activation_status" in
        present) ;;
        absent)
            reasons+=("the log reports '$AQ_ORACLE_ACTIVATION_PREFIX' with no activation")
            ;;
        ambiguous)
            reasons+=("the log reports '$AQ_ORACLE_ACTIVATION_PREFIX' more than once, so activation is ambiguous")
            ;;
        malformed)
            reasons+=("the log's '$AQ_ORACLE_ACTIVATION_PREFIX' line is malformed; expected positive boosted_mi_cells and active_frames counters")
            ;;
        *)
            reasons+=("the log carries no '$AQ_ORACLE_ACTIVATION_PREFIX' line, so cyclic-refresh activation is unproven")
            ;;
    esac

    gate_lines="$(count_lines_exact "$log" "$AQ_ORACLE_GATE_LINE")"
    [ "$gate_lines" -eq 1 ] ||
        reasons+=("the literal <=0.3 pp gate line is not present exactly once (found $gate_lines)")

    if [ "${#reasons[@]}" -ne 0 ]; then
        status="FAIL"
        aq_oracle_failure_count=$(( aq_oracle_failure_count + 1 ))
        printf '  AQ ORACLE FAILED (%s): %s\n' "$log" \
            "$(printf '%s; ' ${reasons[@]+"${reasons[@]}"})" >&2
    fi

    printf 'oracle,test,gain_rvp9_pct,gain_libvpx_pct,delta_pp,max_delta_pp,activation,exit_code,status\n' \
        > "$summary"
    printf 'aq_gain_parity,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$AQ_ORACLE_TEST_NAME" "$gain_rvp9" "$gain_libvpx" "$delta" \
        "$AQ_GAIN_DELTA_MAX_PP" "$activation_status" "$rc" "$status" \
        >> "$summary"

    # Keep these in memory until every phase total is proven and the complete
    # result directory publishes. An interrupted invocation must not append a
    # stale first copy of a manifest key that a later resume would duplicate.
    aq_result_status="$status"
    aq_result_exit_code="$rc"
    aq_result_gain_rvp9="$gain_rvp9"
    aq_result_gain_libvpx="$gain_libvpx"
    aq_result_delta="$delta"
    aq_result_activation="$activation_status"
    aq_result_log_sha256="$(sha256_file "$log")"
    aq_result_summary_sha256="$(sha256_file "$summary")"
    printf '  rvp9 AQ gain %s%%, libvpx AQ gain %s%%, delta %s pp, activation %s: %s\n' \
        "$gain_rvp9" "$gain_libvpx" "$delta" "$activation_status" "$status"
}

run_quality() {
    local feature mode target clip cpu aq denoiser layers completed_dir invocation_id

    completed_dir="$quality_dir/completed"
    [ ! -e "$completed_dir" ] ||
        fail "quality result is already published: $completed_dir"
    mkdir -p "$quality_rows_dir"
    invocation_id="$(date -u '+%Y%m%dT%H%M%SZ')-$$"
    quality_output_dir="$quality_dir/.result-attempt-$invocation_id"
    quality_recheck_dir="$quality_output_dir/decoder-rechecks"
    [ ! -e "$quality_output_dir" ] ||
        fail "quality result attempt already exists: $quality_output_dir"
    mkdir "$quality_output_dir"
    mkdir "$quality_recheck_dir"
    quality_summary="$quality_output_dir/quality-summary.csv"
    quality_aggregate="$quality_output_dir/quality-aggregate.csv"
    cross_decoder_csv="$quality_output_dir/cross-decoder.csv"

    printf 'row_id,feature,mode,target,clip,cpu_used,aq_mode,noise_sensitivity,temporal_layers,threads,tile_columns,bitrates_kbps,runner,anchor,input_kind,reference_kind,layer,bdrate_pct,gate_pct,status\n' \
        > "$quality_summary"
    printf 'group_id,feature,mode,target,cpu_used,aq_mode,noise_sensitivity,temporal_layers,clips,clip_count,layer,mean_bdrate_pct,gate_pct,status\n' \
        > "$quality_aggregate"
    printf 'row_id,stream,kind,bytes,stream_sha256,rvp9dec_sha256,vpxdec_sha256,decoders_identical,ffmpeg_decode,status\n' \
        > "$cross_decoder_csv"

    while IFS='|' read -r feature mode target clip cpu aq denoiser layers; do
        run_quality_row "$feature" "$mode" "$target" "$clip" "$cpu" \
            "$aq" "$denoiser" "$layers"
    done < <(emit_clip_rows)

    compute_quality_aggregates
    run_aq_oracle

    # Exact totals. A phase that measured the wrong amount of the matrix has
    # not qualified the release, whatever its numbers say.
    [ "$quality_row_count" -eq "$EXPECTED_QUALITY_CURVES" ] ||
        fail "ran $quality_row_count quality curves, expected $EXPECTED_QUALITY_CURVES"
    [ "$quality_clip_gate_count" -eq "$EXPECTED_QUALITY_CLIP_LAYER_GUARDS" ] ||
        fail "recorded $quality_clip_gate_count clip/layer checks, expected $EXPECTED_QUALITY_CLIP_LAYER_GUARDS"
    [ "$aggregate_group_count" -eq "$EXPECTED_QUALITY_AGGREGATE_GROUPS" ] ||
        fail "aggregated $aggregate_group_count groups, expected $EXPECTED_QUALITY_AGGREGATE_GROUPS"
    [ "$aggregate_gate_count" -eq "$EXPECTED_QUALITY_AGGREGATE_LAYER_GATES" ] ||
        fail "recorded $aggregate_gate_count aggregate layer gates, expected $EXPECTED_QUALITY_AGGREGATE_LAYER_GATES"
    [ "$cross_stream_count" -eq "$EXPECTED_CANDIDATE_STREAMS" ] ||
        fail "checked $cross_stream_count candidate streams, expected $EXPECTED_CANDIDATE_STREAMS"
    [ "$cross_run_count" -eq "$EXPECTED_DECODER_RUNS" ] ||
        fail "ran $cross_run_count decoder runs, expected $EXPECTED_DECODER_RUNS"
    [ "$aq_oracle_count" -eq "$EXPECTED_REQUIRED_ORACLES" ] ||
        fail "ran $aq_oracle_count required oracles, expected $EXPECTED_REQUIRED_ORACLES"

    # One directory rename publishes the complete result set. Interrupted
    # invocation directories remain private evidence and are never overwritten.
    mv "$quality_output_dir" "$completed_dir"
    [ -f "$completed_dir/quality-summary.csv" ] &&
        [ ! -e "$completed_dir/$(basename "$quality_output_dir")" ] ||
        fail "quality result publication nested instead of replacing the absent completed path"
    quality_output_dir="$completed_dir"
    quality_recheck_dir="$completed_dir/decoder-rechecks"
    quality_summary="$completed_dir/quality-summary.csv"
    quality_aggregate="$completed_dir/quality-aggregate.csv"
    cross_decoder_csv="$completed_dir/cross-decoder.csv"

    manifest_kv quality_curves "$quality_row_count"
    manifest_kv quality_rows_reused "$reused_row_count"
    manifest_kv quality_rows_encoded_this_invocation "$(( quality_row_count - reused_row_count ))"
    manifest_kv quality_clip_layer_checks "$quality_clip_gate_count"
    manifest_kv quality_clip_layer_failures "$quality_regression_count"
    manifest_kv quality_aggregate_groups "$aggregate_group_count"
    manifest_kv quality_aggregate_layer_gates "$aggregate_gate_count"
    manifest_kv quality_aggregate_failures "$aggregate_failure_count"
    manifest_kv candidate_streams "$cross_stream_count"
    manifest_kv decoder_runs "$cross_run_count"
    manifest_kv required_oracles "$aq_oracle_count"
    manifest_kv aq_oracle_status "$aq_result_status"
    manifest_kv aq_oracle_exit_code "$aq_result_exit_code"
    manifest_kv aq_oracle_gain_rvp9_pct "$aq_result_gain_rvp9"
    manifest_kv aq_oracle_gain_libvpx_pct "$aq_result_gain_libvpx"
    manifest_kv aq_oracle_delta_pp "$aq_result_delta"
    manifest_kv aq_oracle_activation "$aq_result_activation"
    manifest_kv aq_oracle_log_sha256 "$aq_result_log_sha256"
    manifest_kv aq_oracle_summary_sha256 "$aq_result_summary_sha256"
    manifest_kv quality_summary_sha256 "$(sha256_file "$quality_summary")"
    manifest_kv quality_aggregate_sha256 "$(sha256_file "$quality_aggregate")"
    manifest_kv cross_decoder_summary_sha256 "$(sha256_file "$cross_decoder_csv")"

    printf '\nQuality phase qualified %s curves (%s reused with fresh decoder checks), %s clip/layer checks, %s aggregate groups (%s layer gates), %s candidate streams, %s decoder runs, %s oracle.\n' \
        "$quality_row_count" "$reused_row_count" \
        "$quality_clip_gate_count" \
        "$aggregate_group_count" "$aggregate_gate_count" \
        "$cross_stream_count" "$cross_run_count" "$aq_oracle_count"

    if [ "$quality_regression_count" -ne 0 ] ||
        [ "$aggregate_failure_count" -ne 0 ] ||
        [ "$aq_oracle_failure_count" -ne 0 ]
    then
        manifest_kv quality_verdict "fail"
        fail "quality phase FAILED: $quality_regression_count clip/layer regression(s) beyond +$MAX_IMPORTANT_CLIP_BDRATE_PCT%, $aggregate_failure_count aggregate group(s) beyond +$MAX_AGGREGATE_BDRATE_PCT%, $aq_oracle_failure_count oracle failure(s). Evidence: $manifest"
    fi
    manifest_kv quality_verdict "pass"
    printf 'Quality phase passed.\n'
}

# Focused filesystem exercise for the resume boundary. It uses only synthetic
# text/IVF placeholders and a private mktemp directory; no encoder, decoder,
# corpus, repository artifact, or caller path is touched.
resume_self_test() {
    local test_root row_id row_dir rate artifact_sha changed_sha attempt_dir
    local lock_work temp_parent bytes stream_sha
    [ -n "${TMPDIR-}" ] || fail "--self-test-resume requires TMPDIR"
    temp_parent="${TMPDIR%/}"
    test_root="$(mktemp -d "$temp_parent/rvp9-qualify-resume.XXXXXX")" ||
        fail "could not create resume self-test directory under TMPDIR"

    cleanup_resume_self_test() {
        local resolved
        [ -d "$test_root" ] || return 0
        resolved="$(cd "$test_root" && pwd -P)"
        case "$resolved" in
            "$(cd "$temp_parent" && pwd -P)"/rvp9-qualify-resume.*) ;;
            *) printf 'refusing to clean unexpected resume self-test path: %s\n' "$resolved" >&2; return 1 ;;
        esac
        find "$resolved" -depth -type f -exec unlink {} \;
        find "$resolved" -depth -type d -exec rmdir {} \;
    }
    trap cleanup_resume_self_test EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM

    # The fixed mkdir lock is acquired and released without touching any
    # caller-owned directory.
    lock_work="$test_root/lock-work"
    mkdir "$lock_work"
    work_dir="$lock_work"
    git_commit="resume-self-test"
    acquire_quality_lock
    [ -s "$quality_lock_dir/owner" ] || fail "resume self-test: lock has no owner"
    if mkdir "$quality_lock_dir" 2>/dev/null; then
        fail "resume self-test: a second invocation acquired the same lock"
    fi
    release_quality_lock

    row_id="baseline-cbr-720p30-in_to_tree-cpu6"
    quality_rows_dir="$test_root/rows"
    row_dir="$quality_rows_dir/$row_id"
    mkdir -p "$row_dir/streams"
    run_fingerprint="run123"
    printf 'bdrate_pct,0.400000\n' > "$row_dir/bdrate.csv"
    printf 'runner evidence\n' > "$row_dir/runner.log"
    printf '%s,baseline,cbr,720p30,in_to_tree,6,0,0,1,1,0,420;630;840;1260;2100;3360,bdrate,vpxenc,clean,clean,0,0.400000,2.0,PASS\n' \
        "$row_id" > "$row_dir/quality-fragment.csv"
    : > "$row_dir/cross-decoder-fragment.csv"
    printf 'decoder evidence\n' > "$row_dir/cross-decoder.log"
    for rate in 420 630 840 1260 2100 3360; do
        printf 'ivf-%s' "$rate" > "$row_dir/streams/test_${rate}k.ivf"
        bytes="$(wc -c < "$row_dir/streams/test_${rate}k.ivf" | tr -d ' ')"
        stream_sha="$(sha256_file "$row_dir/streams/test_${rate}k.ivf")"
        printf '%s,test_%sk.ivf,single,%s,%s,decodehash,decodehash,yes,ok,PASS\n' \
            "$row_id" "$rate" "$bytes" "$stream_sha" \
            >> "$row_dir/cross-decoder-fragment.csv"
    done
    artifact_sha="$(row_artifact_sha256 baseline '420,630,840,1260,2100,3360' "$row_dir")" ||
        fail "resume self-test: cannot fingerprint fixture"
    publish_row_receipt "$row_id" baseline \
        '420,630,840,1260,2100,3360' "$row_dir"
    [ "$(receipt_consistency_status run123 "$row_id" "$artifact_sha" < "$row_dir/receipt")" = "valid" ] ||
        fail "resume self-test: published receipt is invalid"

    # Replace only the expensive decoder operation with a deterministic
    # observation; reuse_quality_row still executes its real receipt, artifact,
    # fragment, counter, and byte-comparison paths.
    cross_check_row() {
        local reused_id="$1" streams="$3" rate ivf observed_bytes observed_sha
        for rate in 420 630 840 1260 2100 3360; do
            ivf="$streams/test_${rate}k.ivf"
            observed_bytes="$(wc -c < "$ivf" | tr -d ' ')"
            observed_sha="$(sha256_file "$ivf")"
            printf '%s,test_%sk.ivf,single,%s,%s,decodehash,decodehash,yes,ok,PASS\n' \
                "$reused_id" "$rate" "$observed_bytes" "$observed_sha" \
                >> "$cross_decoder_csv"
            cross_stream_count=$(( cross_stream_count + 1 ))
            cross_run_count=$(( cross_run_count + 3 ))
        done
    }
    quality_summary="$test_root/replayed-quality.csv"
    cross_decoder_csv="$test_root/replayed-cross.csv"
    quality_recheck_dir="$test_root/rechecks"
    mkdir "$quality_recheck_dir"
    : > "$quality_summary"
    : > "$cross_decoder_csv"
    quality_row_count=0
    reused_row_count=0
    quality_clip_gate_count=0
    quality_regression_count=0
    cross_stream_count=0
    cross_run_count=0
    reuse_quality_row baseline cbr 720p30 in_to_tree 6 0 0 1
    [ "$quality_row_count|$reused_row_count|$quality_clip_gate_count|$quality_regression_count|$cross_stream_count|$cross_run_count" = \
      "1|1|1|0|6|18" ] ||
        fail "resume self-test: reused counters diverged"
    printf 'deliberate mismatch\n' > "$test_root/decoder-mismatch.csv"
    [ "$(decoder_recheck_status "$test_root/decoder-mismatch.csv" \
          "$row_dir/cross-decoder-fragment.csv")" = "mismatch" ] ||
        fail "resume self-test: decoder mismatch classifier passed unequal evidence"

    printf 'changed' >> "$row_dir/streams/test_420k.ivf"
    changed_sha="$(row_artifact_sha256 baseline '420,630,840,1260,2100,3360' "$row_dir")"
    [ "$changed_sha" != "$artifact_sha" ] &&
        [ "$(completed_row_receipt_status run123 "$row_id" baseline \
              '420,630,840,1260,2100,3360' "$row_dir")" = "wrong-artifact" ] ||
        fail "resume self-test: changed stream did not invalidate receipt"

    attempt_dir="$quality_rows_dir/.guard-test.attempt.now-$$"
    mkdir -p "$attempt_dir/streams"
    mkdir -p "$quality_rows_dir/not-an-attempt/streams"
    mkdir -p "$quality_rows_dir/nested/child/streams"
    [ "$(row_attempt_path_status "$quality_rows_dir/not-an-attempt/streams")" = "not-attempt" ] ||
        fail "resume self-test: non-attempt direct row passed deletion guard"
    [ "$(row_attempt_path_status "$quality_rows_dir/nested/child/streams")" = "not-direct" ] ||
        fail "resume self-test: nested row passed deletion guard"
    printf 'generated' > "$attempt_dir/streams/test.y4m"
    discard_row_y4m "$attempt_dir/streams"
    [ ! -e "$attempt_dir/streams/test.y4m" ] ||
        fail "resume self-test: bounded Y4M discard left its exact fixture"

    cleanup_resume_self_test
    trap - EXIT HUP INT TERM
    printf 'resume filesystem self-test passed\n'
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
        --self-test-resume)
            [ "$#" -eq 1 ] || fail "--self-test-resume must be the only argument"
            resume_self_test
            return 0
            ;;
        "")
            usage >&2
            fail "no arguments; try --help"
            ;;
    esac

    parse_args "$@"
    resolve_environment
    acquire_quality_lock
    install_quality_lock_traps

    if [ "$resume" = "yes" ]; then
        validate_resume_contract
    else
        # Preprocessing is bounded and cheap next to the row campaign. Seal
        # only after every prepared input exists, so an interrupted prepare is
        # restarted in a new directory instead of being mistaken for evidence.
        prepare_inputs
        write_manifest_header
        record_acceptance_inputs
        seal_fresh_run
    fi
    run_quality

    manifest_kv ended_utc "$(utc_now)"
    printf 'run_end status=pass\n' >> "$manifest"
    printf '\nNon-spatial v1 acceptance phase "%s" passed.\nEvidence: %s\n' \
        "$phase" "$manifest"
}

main "$@"
