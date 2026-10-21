#!/usr/bin/env bash
# Focused portable-artifact provenance checks; all fixtures stay in .scratch.
set -euo pipefail
repo="$(git rev-parse --show-toplevel)"
set -- --profile beta-cbr
source "${BETA_QUALIFIER_SCRIPT:-$repo/scripts/qualify-production-encoder.sh}"
mkdir -p "$repo/.scratch"
fixture="$(mktemp -d "$repo/.scratch/beta-profile.XXXXXX")"
cleanup() {
    rm -f "$fixture/BUILD-METADATA.txt" "$fixture/rvp9enc" "$fixture/rvp9dec" "$fixture/other" "$fixture/probe.log"
    rmdir "$fixture"
}
trap cleanup EXIT
printf 'portable fixture\n' > "$fixture/rvp9enc"
printf 'different binary\n' > "$fixture/other"
printf 'decoder fixture\n' > "$fixture/rvp9dec"
opt_artifact_manifest="$fixture/BUILD-METADATA.txt"
test_encoder="$fixture/rvp9enc"
rvp9dec="$fixture/rvp9dec"
source_commit="$(git rev-parse HEAD)"
encoder_sha="$(sha256_file "$test_encoder")"
decoder_sha="$(sha256_file "$rvp9dec")"
write_fixture() {
    printf 'build_kind=%s\nsource_commit=%s\nrvp9enc_path=%s\nrvp9enc_sha256=%s\n' \
        "$1" "$2" "$3" "$4" > "$opt_artifact_manifest"
    printf 'rvp9dec_path=rvp9dec\nrvp9dec_sha256=%s\n' "$decoder_sha" >> "$opt_artifact_manifest"
}
reject_fixture() {
    if (validate_artifact_manifest) > "$fixture/probe.log" 2>&1; then
        printf 'FAIL: accepted %s\n' "$1" >&2; exit 1
    fi
    printf 'ok rejected %s\n' "$1"
}
write_fixture portable-release "$source_commit" rvp9enc "$encoder_sha"
validate_artifact_manifest
[ "$pgo_optimized_binary" = "$test_encoder" ]
printf 'ok relative artifact path and exact SHA accepted\n'
printf 'source_commit=%s\n' "$source_commit" >> "$opt_artifact_manifest"
reject_fixture 'duplicate source commit'
write_fixture portable-release "$source_commit" rvp9enc "$encoder_sha"
rvp9dec="$fixture/other"
reject_fixture 'different decoder path'
rvp9dec="$fixture/rvp9dec"
printf 'mutated decoder\n' >> "$rvp9dec"
reject_fixture 'wrong decoder SHA'
printf 'decoder fixture\n' > "$rvp9dec"
write_fixture portable-release "$source_commit" other "$encoder_sha"
reject_fixture 'different binary path'
write_fixture portable-release "$source_commit" rvp9enc "$(printf '%064d' 0)"
reject_fixture 'wrong binary SHA'
write_fixture portable-release "$source_commit" rvp9enc abc
reject_fixture 'short SHA'
write_fixture portable-release HEAD rvp9enc "$encoder_sha"
reject_fixture 'moving source ref'
write_fixture portable-release "$(printf '%040d' 0)" rvp9enc "$encoder_sha"
reject_fixture 'unknown source commit'
write_fixture pgo "$source_commit" rvp9enc "$encoder_sha"
reject_fixture 'wrong build kind'
write_fixture portable-release "$source_commit" rvp9enc "$encoder_sha"
printf 'changed binary\n' >> "$test_encoder"
reject_fixture 'binary changed after manifest'
# A source-drift test must not edit the real repository or manufacture commits.
# Stub only the git diff observation; every other identity check stays real.
printf 'portable fixture\n' > "$test_encoder"
git() {
    if [ "${1-}" = diff ]; then return 1; fi
    command git "$@"
}
reject_fixture 'encoder source drift'
printf 'beta portable artifact checks passed\n'
