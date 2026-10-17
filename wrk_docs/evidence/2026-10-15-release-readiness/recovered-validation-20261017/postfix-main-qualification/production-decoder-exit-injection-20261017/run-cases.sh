#!/usr/bin/env bash
set -u
root='/private/var/folders/bl/xncqcmfj4cj3b8z7j0sy78rr0000gn/T/deltic-scratch/rvp9-decoder-exit-injection-3b1d9d1a'
script='/Users/md/worktrees/rvp9/20261017-TSK-HUM-requalify-post-fix-candidate/scripts/qualify-production-encoder.sh'
run_case() {
  local mode="$1" label="$2" work="$root/work-$2" output="$root/$2.raw.txt" invocations="$root/$2-invocations.raw.txt" status
  set +e
  env PATH="$root/tools/bin:$root/tools:$PATH" \
    RVP9_TEST_MODE="$mode" \
    RVP9_TEST_LOG="$invocations" \
    RVP9_FAKE_CORPUS="$root/corpus" \
    RVP9_SHA_CACHE="$root/sha.cache" \
    RVP9_TEMPLATE_DIR="$root/tools/templates" \
    bash "$script" --work-dir "$work" --phase quality \
      --pgo-manifest "$root/pgo.manifest" --source-dir "$root/corpus" \
      --test-encoder "$root/tools/rvp9enc" --vpxenc "$root/tools/vpxenc" \
      --temporal-anchor "$root/tools/temporal-anchor" --bdrate "$root/tools/bdrate" \
      --bdrate-svc "$root/tools/bdrate-svc" --encode-headtohead "$root/tools/encode-headtohead" \
      --realtime "$root/tools/realtime" --rvp9dec "$root/tools/rvp9dec" \
      --vpxdec "$root/tools/vpxdec" --ffmpeg "$root/tools/ffmpeg" \
      >"$output" 2>&1 </dev/null
  status=$?
  set -e
  printf '%s\n' "$status" > "$root/$label.exit.txt"
  printf '%s exit=%s work=%s\n' "$label" "$status" "$work"
}
set -e
run_case decoder-child-exit decoder-child-exit
run_case decoder-empty-output decoder-empty-output
