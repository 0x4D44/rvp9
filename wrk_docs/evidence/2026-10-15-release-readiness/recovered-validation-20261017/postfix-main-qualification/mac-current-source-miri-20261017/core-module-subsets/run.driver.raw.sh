#!/usr/bin/env bash
set -u
scratch='/private/var/folders/bl/xncqcmfj4cj3b8z7j0sy78rr0000gn/T/deltic-scratch/rvp9-core-miri-subsets-20261017-e10cc2d3'
repo='/Users/md/worktrees/rvp9/20261017-TSK-HUM-requalify-post-fix-candidate'
{
  printf 'source='
  git -C "$repo" rev-parse HEAD
  printf '\n--- rustc ---\n'
  rustc +nightly -Vv
  printf '\n--- cargo ---\n'
  cargo +nightly -V
  printf '\n--- miri ---\n'
  cargo +nightly miri --version
  printf '\n--- host ---\n'
  uname -a
} > "$scratch/context.raw.txt" 2>&1
overall=0
for module in transforms intra_pred quant dsp; do
  printf 'RUN_START=%s\n' "$module" >> "$scratch/summary.txt"
  nice -n 15 deltic timeout 900 env CARGO_TARGET_DIR="$scratch/target" CARGO_BUILD_JOBS=1 cargo +nightly miri test --locked --offline --package rvp9-core --lib "${module}::" -- --test-threads=1 </dev/null > "$scratch/${module}.raw.txt" 2>&1
  code=$?
  printf 'RUN_EXIT=%s:%s\n' "$module" "$code" >> "$scratch/summary.txt"
  if [ "$code" -ne 0 ]; then overall=1; fi
done
printf 'OVERALL_EXIT=%s\n' "$overall" >> "$scratch/summary.txt"
exit "$overall"
