#!/usr/bin/env bash
set -u
real=/usr/bin/shasum
if [ "$#" -lt 3 ] || [ "$1" != "-a" ] || [ "$2" != "256" ] || [ "$3" = "-" ]; then
    exec "$real" "$@"
fi
file="$3"
case "$file" in
    "$RVP9_FAKE_CORPUS/in_to_tree_420_720p50.y4m") digest=81166517a3054c789923535b5da23fbbb9bbd7fd4c28dac83918ac351e121e74 ;;
    "$RVP9_FAKE_CORPUS/ducks_take_off_420_720p50.y4m") digest=0e09aa923ce72e2f6d1c978c3c1cc8dd9588d2ede819081f040213e361970efc ;;
    "$RVP9_FAKE_CORPUS/old_town_cross_420_720p50.y4m") digest=0fbc2ec0c552ad693528b2e49409c7449880b9230055a7696346473d1aaf4123 ;;
    "$RVP9_FAKE_CORPUS/foreman_cif.y4m") digest=b0e3c35eff9fe1da91521a02eb9f0e3ef4bc5aec341ecef1567cd87f1711267b ;;
    *)
        key="$(stat -f '%d:%i:%z:%m' "$file")" || exec "$real" "$@"
        digest="$(awk -v key="$key" '$1 == key { value = $2 } END { print value }' "$RVP9_SHA_CACHE" 2>/dev/null)"
        if [ -z "$digest" ]; then
            value="$("$real" -a 256 "$file")" || exit $?
            digest="${value%% *}"
            printf '%s %s\n' "$key" "$digest" >> "$RVP9_SHA_CACHE"
        fi
        ;;
esac
printf '%s  %s\n' "$digest" "$file"
