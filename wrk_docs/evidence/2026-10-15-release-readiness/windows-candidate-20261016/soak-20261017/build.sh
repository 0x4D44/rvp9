set -uo pipefail
exec </dev/null
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-soak474-d407b661
export PATH="/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
export CARGO_TARGET_DIR='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target' CARGO_BUILD_JOBS=4 CARGO_NET_OFFLINE=true
unset RVP9_FORCE_SCALAR
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
{ date -u +%FT%TZ; git rev-parse HEAD; git status --porcelain; rustc +1.95.0 -vV; cargo +1.95.0 -vV; echo CARGO_TARGET_DIR="$CARGO_TARGET_DIR" CARGO_BUILD_JOBS=4; sha256sum crates/rvp9-bench/src/bin/soak.rs crates/rvp9-harness/src/y4m.rs build/libvpx/vpxdec.exe; } > "$root/build-identity.log" 2>&1
deltic timeout 900 cargo +1.95.0 build --release --locked --offline -p rvp9-bench --bin soak > "$root/build.log" 2>&1
rc=$?; echo "$rc" > "$root/build.exit.txt"; echo "soak_build_exit=$rc"
if [ "$rc" = 0 ]; then sha256sum "$CARGO_TARGET_DIR/release/soak.exe" > "$root/soak-binary.sha256"; fi
exit "$rc"
