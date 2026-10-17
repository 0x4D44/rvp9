set -uo pipefail
exec </dev/null
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-asan474-a78b6878
export PATH="/c/Program Files/Microsoft Visual Studio/18/Enterprise/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64:/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS=4 CARGO_NET_OFFLINE=true
unset RVP9_FORCE_SCALAR
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
{ date -u +%FT%TZ; git rev-parse HEAD; git status --porcelain; rustc +nightly -vV; cargo +nightly -vV; echo msvc_runtime_directory='C:/Program Files/Microsoft Visual Studio/18/Enterprise/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64'; sha256sum "$root/preflight.rs"; } > "$root/toolchain.log" 2>&1
printf '%s\n' 'rustc +nightly --target x86_64-pc-windows-msvc -Zsanitizer=address -Copt-level=1 -Cdebuginfo=1 -Ctarget-cpu=native -Clinker=rust-lld --crate-name native_asan_preflight preflight.rs -o preflight.exe' > "$root/compile.command.txt"
deltic timeout 120 rustc +nightly --target x86_64-pc-windows-msvc -Zsanitizer=address -Copt-level=1 -Cdebuginfo=1 -Ctarget-cpu=native -Clinker=rust-lld --crate-name native_asan_preflight "$root/preflight.rs" -o "$root/preflight.exe" > "$root/compile.log" 2>&1
rc=$?; echo "$rc" > "$root/compile.exit.txt"
echo "native_asan_compile_exit=$rc"
[ "$rc" = 0 ] || exit "$rc"
deltic timeout 30 "$root/preflight.exe" > "$root/run.log" 2>&1
rc=$?; echo "$rc" > "$root/run.exit.txt"
echo "native_asan_run_exit=$rc"
{ sha256sum "$root/preflight.rs" "$root/preflight.exe"; git rev-parse HEAD; git status --porcelain; } > "$root/artifacts.txt"
exit "$rc"
