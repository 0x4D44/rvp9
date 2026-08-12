#!/usr/bin/env bash
# Build a portable macOS Apple Silicon alpha archive from integrated main.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

target_triple="aarch64-apple-darwin"
workspace_version="$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -n 1)"
case "$workspace_version" in
    *-alpha.*) ;;
    *) echo "workspace version is not an alpha: $workspace_version" >&2; exit 1 ;;
esac

head_revision="$(git rev-parse HEAD)"
origin_main="$(git rev-parse refs/remotes/origin/main 2>/dev/null || true)"
if [ -z "$origin_main" ] || [ "$head_revision" != "$origin_main" ]; then
    echo "package only from integrated origin/main" >&2
    echo "HEAD=$head_revision origin/main=${origin_main:-missing}" >&2
    exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
    echo "package only from a clean worktree" >&2
    git status --short >&2
    exit 1
fi

host_triple="$(rustc -vV | sed -n 's/^host: //p')"
if [ "$host_triple" != "$target_triple" ]; then
    echo "this alpha artifact is qualified only on $target_triple; host is $host_triple" >&2
    exit 1
fi

output_dir="${RVP9_ALPHA_DIST_DIR:-$repo_root/dist}"
case "$output_dir" in
    /*) ;;
    *) output_dir="$repo_root/$output_dir" ;;
esac
archive_base="rvp9-$workspace_version-$target_triple"
stage_dir="$output_dir/$archive_base"
archive_path="$output_dir/$archive_base.tar.gz"
checksum_path="$archive_path.sha256"
for path in "$stage_dir" "$archive_path" "$checksum_path"; do
    if [ -e "$path" ]; then
        echo "refusing to overwrite existing release output: $path" >&2
        exit 1
    fi
done

mkdir -p "$stage_dir"
CARGO_TARGET_DIR="$repo_root/target/alpha-portable" \
    CARGO_ENCODED_RUSTFLAGS="" \
    RUSTFLAGS="" \
    cargo build --target "$target_triple" --release --locked \
        -p rvp9-cli --bins

binary_dir="$repo_root/target/alpha-portable/$target_triple/release"
cp "$binary_dir/rvp9enc" "$binary_dir/rvp9dec" "$stage_dir/"
cp README.md RELEASE_NOTES.md LICENSE PATENTS ATTRIBUTION.md "$stage_dir/"

libvpx_revision="$(git ls-tree HEAD third_party/libvpx | awk '{print $3}')"
{
    echo "rvp9_version=$workspace_version"
    echo "git_commit=$head_revision"
    echo "libvpx_revision=$libvpx_revision"
    echo "rustc=$(rustc --version)"
    echo "cargo=$(cargo --version)"
    echo "target=$target_triple"
    echo "rustflags=<empty; target baseline>"
    echo "cargo_profile=release"
    echo "cargo_command=cargo build --target $target_triple --release --locked -p rvp9-cli --bins"
} > "$stage_dir/BUILD-METADATA.txt"

tar -C "$output_dir" -czf "$archive_path" "$archive_base"
(
    cd "$output_dir"
    shasum -a 256 "$archive_base.tar.gz" > "$archive_base.tar.gz.sha256"
)

echo "Created $archive_path"
cat "$checksum_path"
