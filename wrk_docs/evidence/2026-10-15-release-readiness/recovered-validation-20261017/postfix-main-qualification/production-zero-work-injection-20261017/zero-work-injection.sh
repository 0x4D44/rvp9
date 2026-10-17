#!/usr/bin/env bash
set -euo pipefail

script_path=${1:?usage: zero-work-injection.sh SCRIPT_PATH SCRATCH_ROOT}
scratch_root=${2:?usage: zero-work-injection.sh SCRIPT_PATH SCRATCH_ROOT}
mkdir -p "$scratch_root/current" "$scratch_root/mutated"

current_functions="$scratch_root/current/functions.sh"
mutated_functions="$scratch_root/mutated/functions.sh"
runner="$scratch_root/phase-runner.sh"

# Source the exact checked-in function bodies without invoking the CLI main.
sed '$d' "$script_path" > "$current_functions"
# Negative control: remove only the exact-total guards in the three phase functions.
sed -e '3896,3909d' -e '4164,4165d' -e '4431,4434d' "$script_path" | sed '$d' > "$mutated_functions"

cat > "$runner" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail
requested_phase=${1:?}
case_root=${2:?}
functions_file=${3:?}
source "$functions_file"
phase="$requested_phase"

# Zero-row iterator and no-op host/child boundaries: no codec or oracle process runs.
emit_clip_rows() { :; }
emit_speed_groups() { :; }
enforce_load() { :; }
compute_quality_aggregates() { :; }
run_aq_oracle() {
    aq_oracle_count=1
    aq_oracle_failure_count=0
    aq_result_status=PASS
    aq_result_exit_code=0
    aq_result_gain_rvp9=0
    aq_result_gain_libvpx=0
    aq_result_delta=0
    aq_result_activation=present
    aq_result_log_sha256=stub
    aq_result_summary_sha256=stub
}

mkdir -p "$case_root"
manifest="$case_root/manifest.txt"
: > "$manifest"
case "$phase" in
    quality)
        quality_dir="$case_root/results"
        quality_rows_dir="$quality_dir/rows"
        run_quality
        ;;
    speed)
        speed_dir="$case_root/results"
        run_speed
        ;;
    latency)
        latency_dir="$case_root/results"
        run_latency
        ;;
    *)
        fail "invalid phase: $phase"
        ;;
esac
RUNNER

assertion_log="$scratch_root/assertions.raw.txt"
: > "$assertion_log"
for phase in quality speed latency; do
    case "$phase" in
        quality) expected='ran 0 quality curves, expected 200' ;;
        speed) expected='ran 0 relative-speed groups, expected 75' ;;
        latency) expected='ran 0 latency rows, expected 200' ;;
    esac
    case_root="$scratch_root/current/$phase"
    mkdir -p "$case_root"
    if bash "$runner" "$phase" "$case_root" "$current_functions" > "$case_root/phase.raw.txt" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    [ "$rc" -eq 1 ] || { printf 'FAIL: current %s exited %s, expected refusal\n' "$phase" "$rc"; exit 1; }
    [ ! -e "$case_root/results/completed" ] || { printf 'FAIL: current %s published completed output\n' "$phase"; exit 1; }
    grep -F "$expected" "$case_root/phase.raw.txt" >/dev/null || { printf 'FAIL: current %s lacked expected row-count refusal\n' "$phase"; exit 1; }
    printf 'CURRENT %s: exit=1, zero rows refused, completed=no\n' "$phase" >> "$assertion_log"
done

for phase in quality speed latency; do
    case_root="$scratch_root/mutated/$phase"
    mkdir -p "$case_root"
    if bash "$runner" "$phase" "$case_root" "$mutated_functions" > "$case_root/phase.raw.txt" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    [ "$rc" -eq 0 ] || { printf 'FAIL: negative-control %s exited %s, expected empty-phase publication\n' "$phase" "$rc"; exit 1; }
    [ -d "$case_root/results/completed" ] || { printf 'FAIL: negative-control %s did not expose the zero-work acceptance\n' "$phase"; exit 1; }
    case "$phase" in
        quality) summary="$case_root/results/completed/quality-summary.csv" ;;
        speed) summary="$case_root/results/completed/speed-summary.csv" ;;
        latency) summary="$case_root/results/completed/latency-summary.csv" ;;
    esac
    [ "$(wc -l < "$summary" | tr -d ' ')" -eq 1 ] || { printf 'FAIL: negative-control %s output was not header-only\n' "$phase"; exit 1; }
    printf 'NEGATIVE CONTROL %s: exit=0, header-only completed output published\n' "$phase" >> "$assertion_log"
done

cat "$assertion_log"
