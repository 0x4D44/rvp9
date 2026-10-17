set -uo pipefail
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
export PATH="/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN=1.95.0 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target
root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
printf '%s\n' 'bash -c '"'"'date -u; uname -a; git rev-parse HEAD; git status --porcelain; rustc -vV; cargo -vV; echo CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS; echo CARGO_TARGET_DIR=$CARGO_TARGET_DIR; git submodule status; sha256sum build/libvpx/vpxenc.exe build/libvpx/vpxdec.exe build/libvpx/vpx_temporal_svc_reference.exe; find tests/conformance/vectors -maxdepth 1 -type f | wc -l'"'"'' > "$root/logs/identity.command.txt"
printf 'START identity bash_pid=%s utc=%s\n' "$$" "$(date -u +%FT%TZ)"
deltic timeout 60 bash -c 'date -u; uname -a; git rev-parse HEAD; git status --porcelain; rustc -vV; cargo -vV; echo CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS; echo CARGO_TARGET_DIR=$CARGO_TARGET_DIR; git submodule status; sha256sum build/libvpx/vpxenc.exe build/libvpx/vpxdec.exe build/libvpx/vpx_temporal_svc_reference.exe; find tests/conformance/vectors -maxdepth 1 -type f | wc -l' </dev/null > "$root/logs/identity.log" 2>&1
s=$?
printf '%s\n' "$s" > "$root/logs/identity.exit.txt"
printf 'END identity exit=%s utc=%s\n' "$s" "$(date -u +%FT%TZ)"
exit "$s"
