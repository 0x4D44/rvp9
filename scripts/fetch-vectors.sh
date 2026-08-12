#!/usr/bin/env bash
# Download every manifest-backed 8-bit VP9 conformance vector from the pinned
# libvpx revision, together with its decoded-output MD5 ledger.

set -euo pipefail

EXPECTED_LIBVPX_COMMIT="12f3a2ac603e8f10742105519e0cd03c3b8f71dd"
EXPECTED_VECTOR_COUNT=307
BASE_URL="https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx"

usage() {
    cat <<'EOF'
Usage: scripts/fetch-vectors.sh [--list] [--jobs N]

  --list    Print the pinned 8-bit VP9 vector names without downloading.
  --jobs N  Download and verify up to N files concurrently (default: 8).
EOF
}

list_only=0
jobs=8
while [ "$#" -gt 0 ]; do
    case "$1" in
        --list)
            list_only=1
            shift
            ;;
        --jobs)
            if [ "$#" -lt 2 ]; then
                echo "--jobs requires a positive integer" >&2
                exit 2
            fi
            jobs="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "--jobs requires a positive integer, got: $jobs" >&2
    exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
libvpx_dir="$repo_root/third_party/libvpx"
manifest="$libvpx_dir/test/test-data.sha1"
dest="$repo_root/tests/conformance/vectors"

if [ ! -f "$manifest" ]; then
    echo "missing pinned libvpx test manifest: $manifest" >&2
    echo "run: git submodule update --init third_party/libvpx" >&2
    exit 1
fi

actual_commit="$(git -C "$libvpx_dir" rev-parse HEAD)"
if [ "$actual_commit" != "$EXPECTED_LIBVPX_COMMIT" ]; then
    echo "libvpx pin mismatch: expected $EXPECTED_LIBVPX_COMMIT, got $actual_commit" >&2
    exit 1
fi

file_table="$(mktemp "${TMPDIR:-/tmp}/rvp9-vectors.XXXXXX")"
cleanup() {
    rm -f "$file_table"
}
trap cleanup EXIT INT TERM

# The pinned manifest also contains VP8, malformed-input, high-bit-depth, and
# encoder-only media. Select every valid Profile 0/1 VP9 stream that has a
# sibling decoded-output MD5 ledger. Emit both files with their pinned SHA-1.
awk '
    {
        hash = $1
        name = $2
        sub(/^\*/, "", name)
        hashes[name] = hash
    }
    END {
        for (ledger in hashes) {
            if (ledger !~ /^vp9[01].*\.(ivf|webm)\.md5$/) {
                continue
            }
            vector = ledger
            sub(/\.md5$/, "", vector)
            if (vector in hashes) {
                print hashes[vector], vector
                print hashes[ledger], ledger
            }
        }
    }
' "$manifest" | LC_ALL=C sort -k2,2 > "$file_table"

vector_count="$(awk '$2 !~ /\.md5$/ { count++ } END { print count + 0 }' "$file_table")"
if [ "$vector_count" -ne "$EXPECTED_VECTOR_COUNT" ]; then
    echo "pinned vector-set drift: expected $EXPECTED_VECTOR_COUNT, found $vector_count" >&2
    exit 1
fi

if [ "$list_only" -eq 1 ]; then
    awk '$2 !~ /\.md5$/ { print $2 }' "$file_table"
    exit 0
fi

if command -v shasum >/dev/null 2>&1; then
    sha1_file() {
        shasum -a 1 "$1" | awk '{ print $1 }'
    }
elif command -v sha1sum >/dev/null 2>&1; then
    sha1_file() {
        sha1sum "$1" | awk '{ print $1 }'
    }
else
    echo "need shasum or sha1sum to verify conformance downloads" >&2
    exit 1
fi

mkdir -p "$dest"

download_one() {
    local expected="$1"
    local name="$2"
    local target="$dest/$name"
    local actual
    local partial

    if [ -f "$target" ]; then
        actual="$(sha1_file "$target")"
        if [ "$actual" = "$expected" ]; then
            printf '  verified %s\n' "$name"
            return 0
        fi
        echo "checksum mismatch for existing $target" >&2
        echo "expected $expected, got $actual" >&2
        return 1
    fi

    partial="$(mktemp "$dest/.${name}.part.XXXXXX")"
    if ! curl -fsSL --retry 3 -o "$partial" "$BASE_URL/$name"; then
        rm -f "$partial"
        echo "download failed: $name" >&2
        return 1
    fi
    actual="$(sha1_file "$partial")"
    if [ "$actual" != "$expected" ]; then
        rm -f "$partial"
        echo "download checksum mismatch for $name" >&2
        echo "expected $expected, got $actual" >&2
        return 1
    fi
    mv "$partial" "$target"
    printf '  fetched  %s\n' "$name"
}

echo "Fetching $vector_count pinned 8-bit VP9 vectors and their MD5 ledgers."
echo "libvpx: $actual_commit"
echo "destination: $dest"

pids=()
failed=0
while read -r expected name; do
    download_one "$expected" "$name" &
    pids+=("$!")
    if [ "${#pids[@]}" -ge "$jobs" ]; then
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failed=1
            fi
        done
        pids=()
    fi
done < "$file_table"
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    echo "one or more conformance files failed download or verification" >&2
    exit 1
fi

echo "Verified $vector_count vectors and $vector_count MD5 ledgers."
