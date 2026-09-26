#!/usr/bin/env bash
# Build a host-trained, profile-guided rvp9enc release binary.
#
# The fixed training matrix covers the production realtime surface rather than
# one benchmark row: cpu-used 6/7/8, native 720p and scaled 1080p, VBR, CBR
# with cyclic-refresh AQ, denoising, temporal/spatial SVC, and bitrate extremes.
# Generated profiles are compiler-, source-, and host-specific. They stay under
# target/ and must be regenerated on the host that will run the encoder.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/build-encoder-pgo.sh [options]

Options:
  --build-root DIR       Generated build/profile root
                         (default: target/encoder-pgo)
  --training-dir DIR     Directory containing the canonical 720p Y4M corpus
                         (default: tests/corpus)
  --frames N             Frames per training row (default: 60)
  --print-training-plan  Print the fixed 28-row matrix and exit
  --help                 Show this help

Environment overrides:
  RVP9_PGO_FFMPEG          ffmpeg executable
  RVP9_PGO_LLVM_PROFDATA   llvm-profdata executable

Output:
  <build-root>/optimized/release/rvp9enc[.exe]
EOF
}

repo_root="$(git rev-parse --show-toplevel)"
build_root="$repo_root/target/encoder-pgo"
training_dir="$repo_root/tests/corpus"
frames=60
print_training_plan=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --build-root)
            [ "$#" -ge 2 ] || { echo "--build-root requires a value" >&2; exit 2; }
            build_root="$2"
            shift 2
            ;;
        --training-dir)
            [ "$#" -ge 2 ] || { echo "--training-dir requires a value" >&2; exit 2; }
            training_dir="$2"
            shift 2
            ;;
        --frames)
            [ "$#" -ge 2 ] || { echo "--frames requires a value" >&2; exit 2; }
            frames="$2"
            shift 2
            ;;
        --print-training-plan)
            print_training_plan=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$frames" in
    ''|*[!0-9]*) echo "--frames must be a positive integer" >&2; exit 2 ;;
esac
if [ "$frames" -lt 1 ]; then
    echo "--frames must be a positive integer" >&2
    exit 2
fi

case "$build_root" in
    /*) ;;
    *) build_root="$repo_root/$build_root" ;;
esac
case "$training_dir" in
    /*) ;;
    *) training_dir="$repo_root/$training_dir" ;;
esac

# label|input-key|cpu-used|bitrate-kbps|end-usage|extra arguments
training_rows=(
    "1080-in-tree-cpu6|1080-in-tree|6|2500|vbr|"
    "1080-ducks-cpu6|1080-ducks|6|2500|vbr|"
    "1080-old-town-cpu6|1080-old-town|6|2500|vbr|"
    "1080-in-tree-cpu6-cbr-t1|1080-in-tree|6|1500|cbr|"
    "1080-ducks-cpu6-cbr-t1|1080-ducks|6|1500|cbr|"
    "1080-old-town-cpu6-cbr-t1|1080-old-town|6|1500|cbr|"
    "1080-in-tree-cpu7|1080-in-tree|7|2500|vbr|"
    "1080-ducks-cpu7|1080-ducks|7|2500|vbr|"
    "1080-old-town-cpu7|1080-old-town|7|2500|vbr|"
    "1080-in-tree-cpu8|1080-in-tree|8|2500|vbr|"
    "1080-ducks-cpu8|1080-ducks|8|2500|vbr|"
    "1080-old-town-cpu8|1080-old-town|8|2500|vbr|"
    "720-in-tree-cpu7|720-in-tree|7|1500|vbr|"
    "720-ducks-cpu7|720-ducks|7|1500|vbr|"
    "720-old-town-cpu7|720-old-town|7|1500|vbr|"
    "1080-in-tree-cbr-aq3|1080-in-tree|7|2500|cbr|--aq-mode=3"
    "1080-ducks-cbr-aq3|1080-ducks|7|2500|cbr|--aq-mode=3"
    "1080-old-town-cbr-aq3|1080-old-town|7|2500|cbr|--aq-mode=3"
    "1080-in-tree-denoise|1080-in-tree|7|2500|vbr|--noise-sensitivity=1"
    "1080-ducks-denoise|1080-ducks|7|2500|vbr|--noise-sensitivity=1"
    "1080-old-town-denoise|1080-old-town|7|2500|vbr|--noise-sensitivity=1"
    "1080-old-town-low-rate|1080-old-town|7|500|vbr|"
    "1080-old-town-high-rate|1080-old-town|7|6000|vbr|"
    "1080-old-town-temporal2|1080-old-town|7|2500|vbr|--temporal-layers=2"
    "1080-old-town-temporal3|1080-old-town|7|2500|vbr|--temporal-layers=3"
    "1080-old-town-spatial2-cpu6|1080-old-town|6|2835|cbr|--spatial-layers=2 --inter-layer-pred=0"
    "1080-old-town-spatial3-cpu6|1080-old-town|6|2835|cbr|--spatial-layers=3 --inter-layer-pred=0"
    "1080-old-town-spatial2-cpu7|1080-old-town|7|2835|cbr|--spatial-layers=2 --inter-layer-pred=0"
    "1080-old-town-spatial3-cpu7|1080-old-town|7|2835|cbr|--spatial-layers=3 --inter-layer-pred=0"
    "1080-old-town-spatial2-cpu8|1080-old-town|8|2835|cbr|--spatial-layers=2 --inter-layer-pred=0"
    "1080-old-town-spatial3-cpu8|1080-old-town|8|2835|cbr|--spatial-layers=3 --inter-layer-pred=0"
)

if [ "${#training_rows[@]}" -ne 31 ]; then
    echo "internal error: encoder PGO training matrix must contain 31 rows" >&2
    exit 1
fi

if [ "$print_training_plan" -eq 1 ]; then
    printf '%s\n' 'label|input|cpu-used|bitrate-kbps|end-usage|extra-arguments'
    printf '%s\n' "${training_rows[@]}"
    exit 0
fi

for tool in awk cargo git rustc sed shasum; do
    command -v "$tool" >/dev/null 2>&1 || { echo "missing required tool: $tool" >&2; exit 1; }
done

ffmpeg_bin="${RVP9_PGO_FFMPEG:-ffmpeg}"
command -v "$ffmpeg_bin" >/dev/null 2>&1 || {
    echo "missing ffmpeg; install it or set RVP9_PGO_FFMPEG" >&2
    exit 1
}

host_triple="$(rustc -vV | sed -n 's/^host: //p')"
rust_sysroot="$(rustc --print sysroot)"
exe_suffix=""
case "$host_triple" in
    *-windows-*) exe_suffix=".exe" ;;
esac

llvm_profdata="${RVP9_PGO_LLVM_PROFDATA:-}"
if [ -z "$llvm_profdata" ]; then
    bundled_profdata="$rust_sysroot/lib/rustlib/$host_triple/bin/llvm-profdata$exe_suffix"
    if [ -x "$bundled_profdata" ]; then
        llvm_profdata="$bundled_profdata"
    elif command -v llvm-profdata >/dev/null 2>&1; then
        llvm_profdata="$(command -v llvm-profdata)"
    elif command -v xcrun >/dev/null 2>&1 && xcrun --find llvm-profdata >/dev/null 2>&1; then
        llvm_profdata="$(xcrun --find llvm-profdata)"
    else
        echo "missing llvm-profdata; install llvm-tools-preview or set RVP9_PGO_LLVM_PROFDATA" >&2
        exit 1
    fi
fi
if [ ! -x "$llvm_profdata" ]; then
    echo "llvm-profdata is not executable: $llvm_profdata" >&2
    exit 1
fi

source_in_tree="$training_dir/in_to_tree_420_720p50.y4m"
source_ducks="$training_dir/ducks_take_off_420_720p50.y4m"
source_old_town="$training_dir/old_town_cross_420_720p50.y4m"
for input in "$source_in_tree" "$source_ducks" "$source_old_town"; do
    if [ ! -s "$input" ]; then
        echo "missing canonical training input: $input" >&2
        echo "run scripts/fetch-corpus.sh first" >&2
        exit 1
    fi
done

inputs_dir="$build_root/inputs"
instrumented_target="$build_root/instrumented"
optimized_target="$build_root/optimized"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
run_root="$build_root/runs/$run_id"
profile_dir="$run_root/profiles"
manifest="$run_root/manifest.txt"
training_output="$run_root/training.ivf"
mkdir -p "$inputs_dir" "$profile_dir" "$instrumented_target" "$optimized_target"

input_1080_in_tree="$inputs_dir/in_to_tree_420_1080p30_${frames}f.y4m"
input_1080_ducks="$inputs_dir/ducks_take_off_420_1080p30_${frames}f.y4m"
input_1080_old_town="$inputs_dir/old_town_cross_420_1080p30_${frames}f.y4m"

scale_input() {
    local source="$1"
    local output="$2"
    "$ffmpeg_bin" -hide_banner -loglevel error -y -i "$source" \
        -vf 'scale=1920:1080:flags=bicubic,fps=30' \
        -frames:v "$frames" -pix_fmt yuv420p "$output"
}

echo "[encoder PGO 1/4] Preparing representative training inputs..."
scale_input "$source_in_tree" "$input_1080_in_tree"
scale_input "$source_ducks" "$input_1080_ducks"
scale_input "$source_old_town" "$input_1080_old_town"

{
    echo "rvp9 encoder PGO training manifest"
    echo "git_commit=$(git -C "$repo_root" rev-parse HEAD)"
    echo "git_describe=$(git -C "$repo_root" describe --always --dirty)"
    echo "host_triple=$host_triple"
    echo "rustc=$(rustc --version)"
    echo "cargo=$(cargo --version)"
    echo "llvm_profdata=$($llvm_profdata --version | head -n 1)"
    echo "ffmpeg=$($ffmpeg_bin -version | head -n 1)"
    echo "frames_per_row=$frames"
    echo "training_rows=${#training_rows[@]}"
    echo "rustflags=-Ctarget-cpu=native -Cprofile-use=<generated-profdata>"
    echo "training_plan_begin"
    printf '%s\n' "${training_rows[@]}"
    echo "training_plan_end"
} > "$manifest"

echo "[encoder PGO 2/4] Building instrumented rvp9enc..."
profile_generate_flags="-Ctarget-cpu=native"
profile_generate_flags+=$'\x1f'
profile_generate_flags+="-Cprofile-generate=$profile_dir"
CARGO_TARGET_DIR="$instrumented_target" \
    CARGO_ENCODED_RUSTFLAGS="$profile_generate_flags" \
    cargo build --release -p rvp9-cli --bin rvp9enc

instrumented_bin="$instrumented_target/release/rvp9enc$exe_suffix"
if [ ! -x "$instrumented_bin" ]; then
    echo "expected instrumented binary at $instrumented_bin" >&2
    exit 1
fi

resolve_input() {
    case "$1" in
        720-in-tree) printf '%s\n' "$source_in_tree" ;;
        720-ducks) printf '%s\n' "$source_ducks" ;;
        720-old-town) printf '%s\n' "$source_old_town" ;;
        1080-in-tree) printf '%s\n' "$input_1080_in_tree" ;;
        1080-ducks) printf '%s\n' "$input_1080_ducks" ;;
        1080-old-town) printf '%s\n' "$input_1080_old_town" ;;
        *) echo "unknown training input key: $1" >&2; return 1 ;;
    esac
}

echo "[encoder PGO 3/4] Training ${#training_rows[@]} production rows..."
row_number=0
for row in "${training_rows[@]}"; do
    IFS='|' read -r label input_key cpu_used bitrate end_usage extra <<< "$row"
    input="$(resolve_input "$input_key")"
    threads=4
    tile_columns=2
    case "$label" in
        *-cpu6-cbr-t1) threads=1; tile_columns=0 ;;
    esac
    extra_args=()
    if [ -n "$extra" ]; then
        # The matrix is fixed above; whitespace separates its trusted CLI args.
        read -r -a extra_args <<< "$extra"
    fi
    encoder_args=(
        --codec=vp9 --ivf --passes=1 --rt
        --cpu-used="$cpu_used" --target-bitrate="$bitrate"
        --end-usage="$end_usage" --fps=30/1 --threads="$threads"
        --tile-columns="$tile_columns" --lag-in-frames=0 --tune-content=default
        --limit="$frames" --quiet
    )
    if [ "${#extra_args[@]}" -gt 0 ]; then
        encoder_args+=("${extra_args[@]}")
    fi
    row_number=$((row_number + 1))
    printf '  [%02d/%02d] %s\n' "$row_number" "${#training_rows[@]}" "$label"
    LLVM_PROFILE_FILE="$profile_dir/%m-%p.profraw" \
        "$instrumented_bin" "${encoder_args[@]}" \
        -o "$training_output" "$input"
done

shopt -s nullglob
raw_profiles=("$profile_dir"/*.profraw)
shopt -u nullglob
if [ "${#raw_profiles[@]}" -lt "${#training_rows[@]}" ]; then
    echo "expected at least ${#training_rows[@]} raw profiles, found ${#raw_profiles[@]}" >&2
    exit 1
fi

profdata="$run_root/encoder.profdata"
"$llvm_profdata" merge -o "$profdata" "${raw_profiles[@]}"
if [ ! -s "$profdata" ]; then
    echo "llvm-profdata did not produce $profdata" >&2
    exit 1
fi

echo "[encoder PGO 4/4] Building profile-guided rvp9enc..."
profile_use_flags="-Ctarget-cpu=native"
profile_use_flags+=$'\x1f'
profile_use_flags+="-Cprofile-use=$profdata"
CARGO_TARGET_DIR="$optimized_target" \
    CARGO_ENCODED_RUSTFLAGS="$profile_use_flags" \
    cargo build --release -p rvp9-cli --bin rvp9enc

optimized_bin="$optimized_target/release/rvp9enc$exe_suffix"
if [ ! -x "$optimized_bin" ]; then
    echo "expected optimized binary at $optimized_bin" >&2
    exit 1
fi

{
    echo "profdata=$profdata"
    echo "raw_profile_count=${#raw_profiles[@]}"
    echo "optimized_binary=$optimized_bin"
    echo "optimized_binary_sha256=$(shasum -a 256 "$optimized_bin" | awk '{ print $1 }')"
} >> "$manifest"

echo "[encoder PGO done] $optimized_bin"
echo "Training manifest: $manifest"
echo "Profiles are host/source/compiler specific; regenerate after any of them changes."
