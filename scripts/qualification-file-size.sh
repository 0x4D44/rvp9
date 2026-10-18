#!/usr/bin/env bash
# Return a canonical decimal file size across BSD and GNU stat implementations.

qualification_file_size_is_decimal() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

qualification_file_size_canonical_decimal() {
    local value="$1" leading_zeroes
    qualification_file_size_is_decimal "$value" || return 1

    leading_zeroes=${value%%[!0]*}
    value=${value#"$leading_zeroes"}
    if [ -z "$value" ]; then
        value=0
    fi
    printf '%s\n' "$value"
}

qualification_file_size() {
    local path="$1" size

    if size="$(stat -f '%z' "$path" 2>/dev/null)" &&
        qualification_file_size_is_decimal "$size"; then
        qualification_file_size_canonical_decimal "$size"
        return
    fi
    if size="$(stat -c '%s' "$path" 2>/dev/null)" &&
        qualification_file_size_is_decimal "$size"; then
        qualification_file_size_canonical_decimal "$size"
        return
    fi

    printf 'could not determine numeric decoded file size: %s\n' "$path" >&2
    return 1
}
