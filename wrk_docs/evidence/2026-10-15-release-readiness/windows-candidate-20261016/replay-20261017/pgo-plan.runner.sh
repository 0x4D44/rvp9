set -uo pipefail
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
export PATH="/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN=1.95.0 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target
root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
printf '%s\n' 'bash scripts/build-encoder-pgo.sh --print-training-plan' > "$root/logs/pgo-plan.command.txt"
printf 'START pgo-plan bash_pid=%s utc=%s\n' "$$" "$(date -u +%FT%TZ)"
deltic timeout 60 bash scripts/build-encoder-pgo.sh --print-training-plan </dev/null > "$root/logs/pgo-plan.log" 2>&1
s=$?
printf '%s\n' "$s" > "$root/logs/pgo-plan.exit.txt"
printf 'END pgo-plan exit=%s utc=%s\n' "$s" "$(date -u +%FT%TZ)"
exit "$s"
