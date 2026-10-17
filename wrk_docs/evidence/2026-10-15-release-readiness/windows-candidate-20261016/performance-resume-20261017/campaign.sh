set -uo pipefail
exec </dev/null
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
export PATH="/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN=1.95.0 CARGO_BUILD_JOBS=4
root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6
logs="$root/performance-resume-logs"
owner=rvp9-resume-speed474-20261017
trap 'rc=$?; deltic quiet end --owner "$owner" > "$logs/quiet-end.log" 2>&1; exit "$rc"' EXIT
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
[ -f "$root/production-speed/manifest.txt" ] || exit 93
printf '%s\n' 'deltic timeout 86400 bash -c '"'"'shasum() { shift 2; command sha256sum "$@"; }; stat() { if [ "$1" = -f ] && [ "$2" = %z ]; then command stat -c %s "$3"; else command stat "$@"; fi; }; source scripts/qualify-production-encoder.sh --resume --phase speed --work-dir /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/production-speed --pgo-manifest /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/pgo/runs/20261017T052417Z-1120/manifest.txt --test-encoder /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/pgo/optimized/release/rvp9enc.exe --source-dir /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/tests/corpus --libvpx-source /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/third_party/libvpx --bdrate /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target/release/bdrate.exe --bdrate-svc /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target/release/bdrate-svc.exe --encode-headtohead /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target/release/encode-headtohead.exe --realtime /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target/release/realtime.exe --rvp9dec /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target/release/rvp9dec.exe --vpxenc /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/build/libvpx/vpxenc.exe --vpxdec /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/build/libvpx/vpxdec.exe --temporal-anchor /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/build/libvpx/vpx_temporal_svc_reference.exe'"'"'' > "$logs/campaign.command.txt"
date -u +%FT%TZ > "$logs/started.txt"
deltic quiet start --reason rvp9-resume-speed474 --for 20 --wait --max 900 --owner "$owner" > "$logs/quiet-start.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$logs/quiet-start.exit.txt"
[ "$rc" = 0 ] || exit "$rc"
deltic quiet status --json > "$logs/quiet-grant.json"
grep -q '"heavy_processes": 0' "$logs/quiet-grant.json" || { printf 'refusing: heavy processes still running after grant\n' > "$logs/campaign.log"; exit 94; }
deltic timeout 86400 bash -c 'shasum() { shift 2; command sha256sum "$@"; }; stat() { if [ "$1" = -f ] && [ "$2" = %z ]; then command stat -c %s "$3"; else command stat "$@"; fi; }; source scripts/qualify-production-encoder.sh --resume --phase speed --work-dir /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/production-speed --pgo-manifest /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/pgo/runs/20261017T052417Z-1120/manifest.txt --test-encoder /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/pgo/optimized/release/rvp9enc.exe --source-dir /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/tests/corpus --libvpx-source /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/third_party/libvpx --bdrate /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target/release/bdrate.exe --bdrate-svc /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target/release/bdrate-svc.exe --encode-headtohead /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target/release/encode-headtohead.exe --realtime /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target/release/realtime.exe --rvp9dec /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target/release/rvp9dec.exe --vpxenc /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/build/libvpx/vpxenc.exe --vpxdec /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/build/libvpx/vpxdec.exe --temporal-anchor /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/build/libvpx/vpx_temporal_svc_reference.exe' > "$logs/campaign.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$logs/campaign.exit.txt"
date -u +%FT%TZ > "$logs/ended.txt"
{ git rev-parse HEAD; git status --porcelain; } > "$logs/final-source.txt"
printf 'canonical_speed_terminal_exit=%s\n' "$rc"
exit "$rc"
