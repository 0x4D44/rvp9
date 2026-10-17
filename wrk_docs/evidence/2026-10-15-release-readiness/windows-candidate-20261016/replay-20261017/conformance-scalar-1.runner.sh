set -uo pipefail
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
export PATH="/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN=1.95.0 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target
root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
printf '%s\n' 'env RVP9_FORCE_SCALAR=1 /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target/release/conformance.exe --vectors-dir tests/conformance/vectors --official --strict --threads 1' > "$root/logs/conformance-scalar-1.command.txt"
printf 'START conformance-scalar-1 bash_pid=%s utc=%s\n' "$$" "$(date -u +%FT%TZ)"
deltic timeout 1800 env RVP9_FORCE_SCALAR=1 /c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target/release/conformance.exe --vectors-dir tests/conformance/vectors --official --strict --threads 1 </dev/null > "$root/logs/conformance-scalar-1.log" 2>&1
s=$?
printf '%s\n' "$s" > "$root/logs/conformance-scalar-1.exit.txt"
printf 'END conformance-scalar-1 exit=%s utc=%s\n' "$s" "$(date -u +%FT%TZ)"
exit "$s"
