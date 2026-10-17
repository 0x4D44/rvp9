set -uo pipefail
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
export PATH="/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN=1.95.0 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target
root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
printf '%s\n' 'bash -c '"'"'export RVP9_ALPHA_WORK_DIR=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/alpha32 RVP9_ALPHA_VPXDEC=/d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/build/libvpx/vpxdec.exe; shasum() { shift 2; command sha256sum "$@"; }; stat() { if [ "$1" = -f ] && [ "$2" = %z ]; then command stat -c %s "$3"; else command stat "$@"; fi; }; source scripts/qualify-alpha-encoder.sh'"'"'' > "$root/logs/alpha32.command.txt"
printf 'START alpha32 bash_pid=%s utc=%s\n' "$$" "$(date -u +%FT%TZ)"
deltic timeout 14400 bash -c 'export RVP9_ALPHA_WORK_DIR=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/alpha32 RVP9_ALPHA_VPXDEC=/d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/build/libvpx/vpxdec.exe; shasum() { shift 2; command sha256sum "$@"; }; stat() { if [ "$1" = -f ] && [ "$2" = %z ]; then command stat -c %s "$3"; else command stat "$@"; fi; }; source scripts/qualify-alpha-encoder.sh' </dev/null > "$root/logs/alpha32.log" 2>&1
s=$?
printf '%s\n' "$s" > "$root/logs/alpha32.exit.txt"
printf 'END alpha32 exit=%s utc=%s\n' "$s" "$(date -u +%FT%TZ)"
exit "$s"
