set -uo pipefail
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
export PATH="/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN=1.95.0 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target
root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
printf '%s\n' 'bash -c '"'"'[ ! -e /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/pgo ] || exit 93; shasum() { shift 2; command sha256sum "$@"; }; source scripts/build-encoder-pgo.sh --training-dir /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/tests/corpus --build-root /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/pgo'"'"'' > "$root/logs/pgo-build.command.txt"
printf 'START pgo-build bash_pid=%s utc=%s\n' "$$" "$(date -u +%FT%TZ)"
deltic timeout 14400 bash -c '[ ! -e /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/pgo ] || exit 93; shasum() { shift 2; command sha256sum "$@"; }; source scripts/build-encoder-pgo.sh --training-dir /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/tests/corpus --build-root /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/pgo' </dev/null > "$root/logs/pgo-build.log" 2>&1
s=$?
printf '%s\n' "$s" > "$root/logs/pgo-build.exit.txt"
printf 'END pgo-build exit=%s utc=%s\n' "$s" "$(date -u +%FT%TZ)"
exit "$s"
