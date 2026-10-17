set -uo pipefail
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
export PATH="/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN=1.95.0 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target
root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
printf '%s\n' 'cargo test --locked -p rvp9-encoder --lib hf_laplacian_rows_match_scalar_for_steps_spans_and_tight_tails -- --nocapture' > "$root/logs/laplacian-rows.command.txt"
printf 'START laplacian-rows bash_pid=%s utc=%s\n' "$$" "$(date -u +%FT%TZ)"
deltic timeout 1800 cargo test --locked -p rvp9-encoder --lib hf_laplacian_rows_match_scalar_for_steps_spans_and_tight_tails -- --nocapture </dev/null > "$root/logs/laplacian-rows.log" 2>&1
s=$?
printf '%s\n' "$s" > "$root/logs/laplacian-rows.exit.txt"
printf 'END laplacian-rows exit=%s utc=%s\n' "$s" "$(date -u +%FT%TZ)"
exit "$s"
