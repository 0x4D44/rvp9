set -euo pipefail
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
shasum() { shift 2; command sha256sum "$@"; }
file_expected="$(command sha256sum rust-toolchain.toml)"
file_actual="$(shasum -a 256 rust-toolchain.toml)"
[ "$file_actual" = "$file_expected" ]
printf 'file_adapter=PASS\nfile_hash=%s\n' "$file_actual"
stdin_actual="$(printf abc | shasum -a 256)"
[ "$stdin_actual" = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  -' ]
printf 'stdin_adapter=PASS\nstdin_hash=%s\n' "$stdin_actual"
