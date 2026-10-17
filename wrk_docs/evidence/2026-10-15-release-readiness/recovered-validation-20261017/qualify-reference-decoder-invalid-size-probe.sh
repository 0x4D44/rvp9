set -euo pipefail
label=diagnostic; decoded=unused; actual_bytes=invalid; expected_bytes=147456
check_size() {
if [ "$actual_bytes" -ne "$expected_bytes" ]; then
            echo "$label: decoded byte count $actual_bytes, expected $expected_bytes: $decoded" >&2
            return 1
        fi
printf "ACCEPTED_INVALID_SIZE\n"
}
check_size
