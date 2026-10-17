set -uo pipefail
exec </dev/null
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-soak474-d407b661
export PATH="/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
mkdir -p "$root/transient"
export TEMP='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-soak474-d407b661/transient' TMP='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-soak474-d407b661/transient'
export RVP9_CORPUS_DIR='D:/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/tests/corpus'
unset RVP9_FORCE_SCALAR
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
{ date -u +%FT%TZ; git rev-parse HEAD; git status --porcelain; echo TEMP="$TEMP" TMP="$TMP" RVP9_CORPUS_DIR="$RVP9_CORPUS_DIR"; echo 'testsrc_720p30.y4m: absent, no replacement'; echo 'sources: Foreman CIF + explicit built-in SyntheticGradient256x256'; sha256sum tests/corpus/foreman_cif.y4m crates/rvp9-bench/src/bin/soak.rs build/libvpx/vpxdec.exe 'C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target/release/soak.exe'; } > "$root/run-identity.log" 2>&1
printf '%s\n' 'deltic timeout 180 soak.exe --duration-secs 60 --frames-per-iter 8 --report-every-secs 10 --cross-decode --vpxdec D:/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/build/libvpx/vpxdec.exe' > "$root/run.command.txt"
deltic timeout 180 'C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target/release/soak.exe' --duration-secs 60 --frames-per-iter 8 --report-every-secs 10 --cross-decode --vpxdec 'D:/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi/build/libvpx/vpxdec.exe' > "$root/run.log" 2>&1
rc=$?; echo "$rc" > "$root/run.exit.txt"; echo "native_soak_exit=$rc"
{ date -u +%FT%TZ; git rev-parse HEAD; git status --porcelain; find "$root/transient" -maxdepth 2 -type f; sha256sum "$root/build.log" "$root/build-identity.log" "$root/run.log" "$root/run-identity.log" "$root/run.exit.txt"; } > "$root/terminal.log" 2>&1
exit "$rc"
