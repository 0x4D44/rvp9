set -uo pipefail
exec </dev/null
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-packagegraph474-3255285b
export PATH="/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
export CARGO_TARGET_DIR='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-packagegraph474-3255285b/cargo-target' CARGO_BUILD_JOBS=4 CARGO_NET_OFFLINE=true RUSTUP_TOOLCHAIN=1.95.0
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
[ ! -e "$CARGO_TARGET_DIR" ] || exit 93
{ date -u +%FT%TZ; git rev-parse HEAD; git status --porcelain; rustc -vV; cargo -vV; uname -a; printf 'CARGO_TARGET_DIR=%s\nCARGO_BUILD_JOBS=%s\nCARGO_NET_OFFLINE=%s\nRUSTUP_TOOLCHAIN=%s\n' "$CARGO_TARGET_DIR" "$CARGO_BUILD_JOBS" "$CARGO_NET_OFFLINE" "$RUSTUP_TOOLCHAIN"; sha256sum Cargo.toml Cargo.lock .cargo/config.toml crates/rvp9-{core,io,decoder,encoder,cli}/Cargo.toml; printf 'repo_config_begin\n'; cat .cargo/config.toml; printf 'repo_config_end\n'; } > "$root/identity.log" 2>&1
printf '%s\n' 'deltic timeout 1800 cargo +1.95.0 package --offline --locked -vv -p rvp9-core -p rvp9-io -p rvp9-decoder -p rvp9-encoder -p rvp9-cli' > "$root/package.command.txt"
deltic timeout 1800 cargo +1.95.0 package --offline --locked -vv -p rvp9-core -p rvp9-io -p rvp9-decoder -p rvp9-encoder -p rvp9-cli > "$root/package.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$root/package.exit.txt"; printf 'native_package_exit=%s\n' "$rc"
{ date -u +%FT%TZ; git rev-parse HEAD; git status --porcelain; find "$CARGO_TARGET_DIR/package" -maxdepth 5 -type f -path '*/tmp-registry/*' -print; } > "$root/terminal.log" 2>&1
for archive in "$CARGO_TARGET_DIR/package/"*.crate; do
 [ -f "$archive" ] || continue
 name=$(basename "$archive" .crate)
 sha256sum "$archive" >> "$root/archive-sha256.txt"
 wc -c "$archive" >> "$root/archive-bytes.txt"
 tar -tzf "$archive" > "$root/$name.filelist.txt"
 for member in Cargo.toml Cargo.toml.orig Cargo.lock .cargo_vcs_info.json; do
  tar -xOf "$archive" "$name/$member" > "$root/$name.$member.txt" 2> "$root/$name.$member.stderr.txt"
 done
done
exit "$rc"
