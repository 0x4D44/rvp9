#!/usr/bin/env bash
set -euo pipefail

root=/d/worktrees/rvp9-2/20260928-TSK-HUM-count-final-emission-tx4-x86
exe="$root/target/release/rvp9enc.exe"
inputs=/d/worktrees/rvp9-2/acceptance-d639c265/target/encoder-pgo/inputs
output="$root/.scratch/tx4-census"
mkdir -p "$output"

for clip in ducks tree town; do
    case "$clip" in
        ducks)
            name=ducks_take_off_420_1080p30_60f.y4m
            input_sha=6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7
            expected=8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089
            ;;
        tree)
            name=in_to_tree_420_1080p30_60f.y4m
            input_sha=ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791
            expected=5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41
            ;;
        town)
            name=old_town_cross_420_1080p30_60f.y4m
            input_sha=b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909
            expected=4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f
            ;;
    esac
    actual_input_sha=$(sha256sum "$inputs/$name" | cut -d' ' -f1)
    if [[ "$actual_input_sha" != "$input_sha" ]]; then
        echo "input SHA mismatch: $clip $actual_input_sha" >&2
        exit 1
    fi
    RVP9_EMIT_PROFILE=1 "$exe" --codec=vp9 --passes=1 --rt \
        --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 \
        --threads=1 --tile-columns=0 --lag-in-frames=0 \
        --tune-content=default --aq-mode=0 --noise-sensitivity=0 \
        --limit=60 --kf-max-dist=60 --quiet --ivf \
        -o "$output/$clip.ivf" "$inputs/$name" \
        >"$output/$clip.stdout.txt" 2>"$output/$clip.stderr.txt"
    actual=$(sha256sum "$output/$clip.ivf" | cut -d' ' -f1)
    printf '%s input=%s output=%s expected=%s\n' "$clip" "$input_sha" "$actual" "$expected"
    if [[ "$actual" != "$expected" ]]; then
        exit 1
    fi
done
