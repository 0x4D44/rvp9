#!/usr/bin/env bash
# Qualify the v0.2.0-alpha.3 encoder surface across the supported real-time
# configuration matrix. Every produced stream must decode byte-exactly with
# rvp9dec, pinned libvpx vpxdec, and ffmpeg.

set -euo pipefail

readonly PINNED_LIBVPX_REVISION="12f3a2ac603e8f10742105519e0cd03c3b8f71dd"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

frames="${RVP9_ALPHA_FRAMES:-60}"
source_dir="${RVP9_ALPHA_SOURCE_DIR:-$repo_root/tests/corpus}"
vpxdec="${RVP9_ALPHA_VPXDEC:-$repo_root/build/libvpx/vpxdec}"
libvpx_source="${RVP9_ALPHA_LIBVPX_SOURCE:-$repo_root/third_party/libvpx}"
ffmpeg_bin="${RVP9_ALPHA_FFMPEG:-ffmpeg}"

case "$frames" in
    ''|*[!0-9]*) echo "RVP9_ALPHA_FRAMES must be a positive integer" >&2; exit 2 ;;
esac
if [ "$frames" -lt 1 ]; then
    echo "RVP9_ALPHA_FRAMES must be a positive integer" >&2
    exit 2
fi

for tool in cargo cmp git rustc shasum stat; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "missing required tool: $tool" >&2
        exit 1
    }
done
command -v "$ffmpeg_bin" >/dev/null 2>&1 || {
    echo "missing ffmpeg; install it or set RVP9_ALPHA_FFMPEG" >&2
    exit 1
}
if [ ! -x "$vpxdec" ]; then
    echo "missing pinned vpxdec: $vpxdec" >&2
    echo "build it with scripts/build-libvpx.sh or set RVP9_ALPHA_VPXDEC" >&2
    exit 1
fi

libvpx_root="$(git -C "$libvpx_source" rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$libvpx_root" ] || [ "$(cd "$libvpx_root" && pwd -P)" != "$(cd "$libvpx_source" && pwd -P)" ]; then
    echo "libvpx source is not an initialized standalone checkout: $libvpx_source" >&2
    echo "initialize third_party/libvpx or set RVP9_ALPHA_LIBVPX_SOURCE" >&2
    exit 1
fi
libvpx_revision="$(git -C "$libvpx_source" rev-parse HEAD)"
if [ "$libvpx_revision" != "$PINNED_LIBVPX_REVISION" ]; then
    echo "wrong libvpx revision: $libvpx_revision" >&2
    echo "expected: $PINNED_LIBVPX_REVISION" >&2
    exit 1
fi

for clip in \
    in_to_tree_420_720p50.y4m \
    ducks_take_off_420_720p50.y4m \
    old_town_cross_420_720p50.y4m
do
    if [ ! -s "$source_dir/$clip" ]; then
        echo "missing canonical source: $source_dir/$clip" >&2
        echo "run scripts/fetch-corpus.sh or set RVP9_ALPHA_SOURCE_DIR" >&2
        exit 1
    fi
done

host_triple="$(rustc -vV | sed -n 's/^host: //p')"
exe_suffix=""
case "$host_triple" in
    *-windows-*) exe_suffix=".exe" ;;
esac

work_dir="${RVP9_ALPHA_WORK_DIR:-}"
if [ -z "$work_dir" ]; then
    work_dir="$(mktemp -d "${TMPDIR:-/tmp}/rvp9-alpha-encoder.XXXXXX")"
else
    mkdir -p "$work_dir"
    work_dir="$(cd "$work_dir" && pwd -P)"
fi
manifest="$work_dir/qualification.txt"

cargo build --release --locked -p rvp9-cli --bins
rvp9enc="$repo_root/target/release/rvp9enc$exe_suffix"
rvp9dec="$repo_root/target/release/rvp9dec$exe_suffix"
for binary in "$rvp9enc" "$rvp9dec"; do
    if [ ! -x "$binary" ]; then
        echo "expected release binary: $binary" >&2
        exit 1
    fi
done

source_for_key() {
    case "$1" in
        720-in-tree) echo "$source_dir/in_to_tree_420_720p50.y4m" ;;
        720-ducks) echo "$source_dir/ducks_take_off_420_720p50.y4m" ;;
        720-old-town) echo "$source_dir/old_town_cross_420_720p50.y4m" ;;
        1080p30-in-tree) echo "$work_dir/in_to_tree_1080p30.y4m" ;;
        1080p30-ducks) echo "$work_dir/ducks_take_off_1080p30.y4m" ;;
        1080p30-old-town) echo "$work_dir/old_town_cross_1080p30.y4m" ;;
        1080p60-in-tree) echo "$work_dir/in_to_tree_1080p60.y4m" ;;
        1080p60-old-town) echo "$work_dir/old_town_cross_1080p60.y4m" ;;
        *) echo "unknown input key: $1" >&2; return 1 ;;
    esac
}

prepare_1080_input() {
    source="$1"
    fps="$2"
    output="$3"
    if [ -s "$output" ]; then
        return
    fi
    "$ffmpeg_bin" -hide_banner -loglevel error -y -i "$source" \
        -vf "scale=1920:1080:flags=bicubic,fps=$fps" \
        -frames:v "$frames" -pix_fmt yuv420p "$output"
}

echo "Preparing 1080p qualification inputs in $work_dir"
prepare_1080_input "$source_dir/in_to_tree_420_720p50.y4m" 30 \
    "$(source_for_key 1080p30-in-tree)"
prepare_1080_input "$source_dir/ducks_take_off_420_720p50.y4m" 30 \
    "$(source_for_key 1080p30-ducks)"
prepare_1080_input "$source_dir/old_town_cross_420_720p50.y4m" 30 \
    "$(source_for_key 1080p30-old-town)"
prepare_1080_input "$source_dir/in_to_tree_420_720p50.y4m" 60 \
    "$(source_for_key 1080p60-in-tree)"
prepare_1080_input "$source_dir/old_town_cross_420_720p50.y4m" 60 \
    "$(source_for_key 1080p60-old-town)"

# label|input-key|fps|cpu|mode|aq|noise|layers|bitrate|threads|tile-columns
rows=(
    "base-cpu6-vbr-in-tree|720-in-tree|30/1|6|vbr|0|0|1|1500|1|0"
    "base-cpu6-cbr-in-tree|720-in-tree|30/1|6|cbr|0|0|1|1500|1|0"
    "base-cpu7-vbr-ducks|720-ducks|30/1|7|vbr|0|0|1|1500|1|0"
    "base-cpu7-cbr-ducks|720-ducks|30/1|7|cbr|0|0|1|1500|1|0"
    "base-cpu8-vbr-old-town|720-old-town|30/1|8|vbr|0|0|1|1500|1|0"
    "base-cpu8-cbr-old-town|720-old-town|30/1|8|cbr|0|0|1|1500|1|0"
    "resolution-1080p30-vbr-in-tree|1080p30-in-tree|30/1|7|vbr|0|0|1|2500|4|2"
    "resolution-1080p30-cbr-ducks|1080p30-ducks|30/1|7|cbr|0|0|1|2500|4|2"
    "resolution-1080p60-vbr-old-town|1080p60-old-town|60/1|7|vbr|0|0|1|4000|8|2"
    "resolution-1080p60-cbr-in-tree|1080p60-in-tree|60/1|7|cbr|0|0|1|4000|8|2"
    "aq3-720p-vbr-in-tree|720-in-tree|30/1|7|vbr|3|0|1|1500|1|0"
    "aq3-720p-cbr-ducks|720-ducks|30/1|7|cbr|3|0|1|1500|1|0"
    "aq3-1080p-vbr-old-town|1080p30-old-town|30/1|7|vbr|3|0|1|2500|4|2"
    "aq3-1080p-cbr-in-tree|1080p30-in-tree|30/1|7|cbr|3|0|1|2500|4|2"
    "denoise-720p-vbr-ducks|720-ducks|30/1|7|vbr|0|1|1|1500|1|0"
    "denoise-720p-cbr-old-town|720-old-town|30/1|7|cbr|0|1|1|1500|1|0"
    "denoise-1080p-vbr-in-tree|1080p30-in-tree|30/1|7|vbr|0|1|1|2500|4|2"
    "denoise-1080p-cbr-ducks|1080p30-ducks|30/1|7|cbr|0|1|1|2500|4|2"
    "temporal2-cpu6-vbr-in-tree|720-in-tree|30/1|6|vbr|0|0|2|1500|1|0"
    "temporal2-cpu6-cbr-in-tree|720-in-tree|30/1|6|cbr|0|0|2|1500|1|0"
    "temporal2-cpu7-vbr-ducks|720-ducks|30/1|7|vbr|0|0|2|1500|1|0"
    "temporal2-cpu7-cbr-ducks|720-ducks|30/1|7|cbr|0|0|2|1500|1|0"
    "temporal2-cpu8-vbr-old-town|720-old-town|30/1|8|vbr|0|0|2|1500|1|0"
    "temporal2-cpu8-cbr-old-town|720-old-town|30/1|8|cbr|0|0|2|1500|1|0"
    "temporal3-cpu6-vbr-in-tree|720-in-tree|30/1|6|vbr|0|0|3|1500|1|0"
    "temporal3-cpu6-cbr-in-tree|720-in-tree|30/1|6|cbr|0|0|3|1500|1|0"
    "temporal3-cpu7-vbr-ducks|720-ducks|30/1|7|vbr|0|0|3|1500|1|0"
    "temporal3-cpu7-cbr-ducks|720-ducks|30/1|7|cbr|0|0|3|1500|1|0"
    "temporal3-cpu8-vbr-old-town|720-old-town|30/1|8|vbr|0|0|3|1500|1|0"
    "temporal3-cpu8-cbr-old-town|720-old-town|30/1|8|cbr|0|0|3|1500|1|0"
    "temporal2-1080p30-vbr-old-town|1080p30-old-town|30/1|7|vbr|0|0|2|2500|4|2"
    "temporal3-1080p30-vbr-in-tree|1080p30-in-tree|30/1|7|vbr|0|0|3|2500|4|2"
)

if [ "${#rows[@]}" -ne 32 ]; then
    echo "internal error: alpha encoder matrix must contain 32 rows" >&2
    exit 1
fi

{
    echo "rvp9 v0.2.0-alpha.3 encoder qualification"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "host_triple=$host_triple"
    echo "rustc=$(rustc --version)"
    echo "cargo=$(cargo --version)"
    echo "ffmpeg=$($ffmpeg_bin -version | head -n 1)"
    echo "libvpx_revision=$libvpx_revision"
    echo "vpxdec_sha256=$(shasum -a 256 "$vpxdec" | awk '{print $1}')"
    echo "frames_per_row=$frames"
    echo "rows=${#rows[@]}"
    for clip in \
        in_to_tree_420_720p50.y4m \
        ducks_take_off_420_720p50.y4m \
        old_town_cross_420_720p50.y4m
    do
        echo "source_sha256[$clip]=$(shasum -a 256 "$source_dir/$clip" | awk '{print $1}')"
    done
    echo "results_begin"
} > "$manifest"

run_row() {
    row_number="$1"
    row="$2"
    IFS='|' read -r label input_key fps cpu mode aq noise layers bitrate threads tiles <<< "$row"
    input="$(source_for_key "$input_key")"
    ivf="$work_dir/$label.ivf"
    rvp_yuv="$work_dir/$label.rvp9.yuv"
    vpx_yuv="$work_dir/$label.vpx.yuv"
    ffmpeg_yuv="$work_dir/$label.ffmpeg.yuv"

    case "$input_key" in
        720-*) width=1280; height=720 ;;
        1080p*) width=1920; height=1080 ;;
        *) echo "unknown dimensions for input key: $input_key" >&2; return 1 ;;
    esac
    expected_bytes=$((width * height * 3 / 2 * frames))

    printf '[%02d/%02d] %s\n' "$row_number" "${#rows[@]}" "$label"
    "$rvp9enc" --codec=vp9 --passes=1 --rt \
        --cpu-used="$cpu" --end-usage="$mode" \
        --target-bitrate="$bitrate" --fps="$fps" \
        --threads="$threads" --tile-columns="$tiles" \
        --lag-in-frames=0 --tune-content=default \
        --aq-mode="$aq" --noise-sensitivity="$noise" \
        --temporal-layers="$layers" --kf-max-dist=60 \
        --limit="$frames" --ivf --quiet -o "$ivf" "$input"

    "$rvp9dec" --codec=vp9 --i420 -o "$rvp_yuv" "$ivf"
    "$vpxdec" --codec=vp9 --i420 -o "$vpx_yuv" "$ivf"
    "$ffmpeg_bin" -hide_banner -loglevel error -y -c:v vp9 \
        -i "$ivf" -pix_fmt yuv420p -f rawvideo "$ffmpeg_yuv"

    for decoded in "$rvp_yuv" "$vpx_yuv" "$ffmpeg_yuv"; do
        actual_bytes="$(stat -f '%z' "$decoded" 2>/dev/null || stat -c '%s' "$decoded")"
        if [ "$actual_bytes" -ne "$expected_bytes" ]; then
            echo "$label: decoded byte count $actual_bytes, expected $expected_bytes: $decoded" >&2
            return 1
        fi
    done
    cmp -s "$rvp_yuv" "$vpx_yuv" || {
        echo "$label: rvp9dec and vpxdec outputs differ" >&2
        return 1
    }
    cmp -s "$rvp_yuv" "$ffmpeg_yuv" || {
        echo "$label: rvp9dec and ffmpeg outputs differ" >&2
        return 1
    }

    stream_sha="$(shasum -a 256 "$ivf" | awk '{print $1}')"
    decoded_sha="$(shasum -a 256 "$rvp_yuv" | awk '{print $1}')"
    echo "$label|PASS|ivf_sha256=$stream_sha|decoded_sha256=$decoded_sha|bytes=$expected_bytes" \
        >> "$manifest"

    rm -f "$ivf" "$rvp_yuv" "$vpx_yuv" "$ffmpeg_yuv"
}

row_number=0
for row in "${rows[@]}"; do
    row_number=$((row_number + 1))
    run_row "$row_number" "$row"
done

echo "results_end" >> "$manifest"
rm -f \
    "$(source_for_key 1080p30-in-tree)" \
    "$(source_for_key 1080p30-ducks)" \
    "$(source_for_key 1080p30-old-town)" \
    "$(source_for_key 1080p60-in-tree)" \
    "$(source_for_key 1080p60-old-town)"
echo "Alpha encoder qualification passed: ${#rows[@]}/${#rows[@]} streams."
echo "Evidence: $manifest"
