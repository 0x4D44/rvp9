set -uo pipefail
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
export PATH="/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN=1.95.0 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target
root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
printf '%s\n' 'cargo build --release --locked -p rvp9-harness --bin conformance -p rvp9-cli --bin rvp9enc --bin rvp9dec -p rvp9-bench --bin bdrate --bin bdrate-svc --bin encode-headtohead --bin realtime' > "$root/logs/build.command.txt"
printf 'START build bash_pid=%s utc=%s\n' "$$" "$(date -u +%FT%TZ)"
deltic timeout 1800 cargo build --release --locked -p rvp9-harness --bin conformance -p rvp9-cli --bin rvp9enc --bin rvp9dec -p rvp9-bench --bin bdrate --bin bdrate-svc --bin encode-headtohead --bin realtime </dev/null > "$root/logs/build.log" 2>&1
s=$?
printf '%s\n' "$s" > "$root/logs/build.exit.txt"
printf 'END build exit=%s utc=%s\n' "$s" "$(date -u +%FT%TZ)"
exit "$s"
