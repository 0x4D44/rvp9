set -euo pipefail
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
stat() { if [ "$1" = -f ] && [ "$2" = %z ]; then command stat -c %s "$3"; else command stat "$@"; fi; }
for f in rust-toolchain.toml tests/corpus/in_to_tree_420_720p50.y4m tests/corpus/ducks_take_off_420_720p50.y4m tests/corpus/old_town_cross_420_720p50.y4m; do
 actual="$(stat -f %z "$f")"
 expected="$(wc -c < "$f" | tr -d ' ')"
 [[ "$actual" =~ ^[0-9]+$ ]]
 [ "$actual" -eq "$expected" ]
 printf '%s|gnu_stat_bytes=%s|independent_wc_bytes=%s|numeric=PASS\n' "$f" "$actual" "$expected"
done
