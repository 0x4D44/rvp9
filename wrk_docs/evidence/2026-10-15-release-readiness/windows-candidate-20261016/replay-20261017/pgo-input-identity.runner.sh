set -uo pipefail
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
export PATH="/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN=1.95.0 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target
root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
printf '%s\n' 'bash -c '"'"'rustc -vV; cargo -vV; sha256sum scripts/build-encoder-pgo.sh tests/corpus/in_to_tree_420_720p50.y4m tests/corpus/ducks_take_off_420_720p50.y4m tests/corpus/old_town_cross_420_720p50.y4m; "$(rustc --print sysroot)/lib/rustlib/x86_64-pc-windows-msvc/bin/llvm-profdata.exe" --version'"'"'' > "$root/logs/pgo-input-identity.command.txt"
printf 'START pgo-input-identity bash_pid=%s utc=%s\n' "$$" "$(date -u +%FT%TZ)"
deltic timeout 300 bash -c 'rustc -vV; cargo -vV; sha256sum scripts/build-encoder-pgo.sh tests/corpus/in_to_tree_420_720p50.y4m tests/corpus/ducks_take_off_420_720p50.y4m tests/corpus/old_town_cross_420_720p50.y4m; "$(rustc --print sysroot)/lib/rustlib/x86_64-pc-windows-msvc/bin/llvm-profdata.exe" --version' </dev/null > "$root/logs/pgo-input-identity.log" 2>&1
s=$?
printf '%s\n' "$s" > "$root/logs/pgo-input-identity.exit.txt"
printf 'END pgo-input-identity exit=%s utc=%s\n' "$s" "$(date -u +%FT%TZ)"
exit "$s"
