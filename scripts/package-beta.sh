#!/usr/bin/env bash
# Build portable beta archives from clean integrated main on each native host.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
workspace_version="$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -n 1)"
if [[ ! "$workspace_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+-beta\.[0-9]+$ ]]; then
    echo "workspace version is not a beta: $workspace_version" >&2
    exit 1
fi
head_revision="$(git rev-parse HEAD)"
origin_main="$(git rev-parse refs/remotes/origin/main 2>/dev/null || true)"
if [ "$head_revision" != "$origin_main" ]; then
    echo "package only from integrated origin/main" >&2
    exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
    echo "package only from a clean worktree" >&2
    exit 1
fi

host_triple="$(rustc -vV | sed -n 's/^host: //p')"
exe_suffix=""
case "$(uname -s):$(uname -m):$host_triple" in
    Darwin:arm64:aarch64-apple-darwin) ;;
    MINGW*:x86_64:x86_64-pc-windows-msvc|MSYS*:x86_64:x86_64-pc-windows-msvc)
        exe_suffix=".exe" ;;
    *) echo "unsupported native beta packaging host: $(uname -s) $(uname -m) $host_triple" >&2; exit 1 ;;
esac
target_triple="$host_triple"

absolute_path() {
    local path="$1"
    if [ -n "$exe_suffix" ]; then
        path="$(cygpath -u "$path")"
    fi
    case "$path" in
        /*) printf '%s\n' "$path" ;;
        *) printf '%s/%s\n' "$repo_root" "$path" ;;
    esac
}
sha256() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        sha256sum "$1" | awk '{print $1}'
    fi
}

output_dir="$(absolute_path "${RVP9_BETA_DIST_DIR:-$repo_root/dist}")"
target_dir="$(absolute_path "${RVP9_BETA_TARGET_DIR:-${CARGO_TARGET_DIR:-$repo_root/target/beta-portable}}")"
cargo_target_dir="$target_dir"
if [ -n "$exe_suffix" ]; then
    cargo_target_dir="$(cygpath -m "$target_dir")"
fi
archive_base="rvp9-$workspace_version-$target_triple"
stage_dir="$output_dir/$archive_base"
archive_path="$output_dir/$archive_base.tar.gz"
checksum_path="$archive_path.sha256"
for path in "$stage_dir" "$archive_path" "$checksum_path"; do
    if [ -e "$path" ] || [ -L "$path" ]; then
        echo "refusing to overwrite existing release output: $path" >&2
        exit 1
    fi
done

# An explicit target plus empty flags prevents workspace target-cpu=native
# and inherited PGO flags from entering this portable build.
CARGO_TARGET_DIR="$cargo_target_dir" CARGO_ENCODED_RUSTFLAGS="" RUSTFLAGS="" \
    deltic timeout 1800 cargo build --target "$target_triple" --release --locked \
        -p rvp9-cli --bins </dev/null

# Check before creating distribution files, which may be untracked outputs.
if [ "$(git rev-parse HEAD)" != "$head_revision" ] || \
   [ "$(git rev-parse refs/remotes/origin/main)" != "$head_revision" ] || \
   [ -n "$(git status --porcelain)" ]; then
    echo "source changed during beta packaging; refusing archive" >&2
    exit 1
fi
mkdir -p "$output_dir"
# mkdir without -p claims the staging name and refuses a concurrent packager.
mkdir "$stage_dir"
binary_dir="$target_dir/$target_triple/release"
for binary in rvp9enc rvp9dec; do
    cp "$binary_dir/$binary$exe_suffix" "$stage_dir/"
    actual_version="$("$stage_dir/$binary$exe_suffix" --version)"
    if [ "$actual_version" != "$binary $workspace_version" ]; then
        echo "unexpected packaged binary version: $actual_version" >&2
        exit 1
    fi
done
cp README.md RELEASE_NOTES.md LICENSE PATENTS ATTRIBUTION.md "$stage_dir/"

libvpx_revision="$(git ls-tree HEAD third_party/libvpx | awk '{print $3}')"
{
    echo "rvp9_version=$workspace_version"
    echo "source_commit=$head_revision"
    echo "git_commit=$head_revision"
    echo "libvpx_revision=$libvpx_revision"
    echo "rustc=$(rustc --version)"
    echo "cargo=$(cargo --version)"
    echo "target=$target_triple"
    echo "build_kind=portable-release"
    echo "rustflags=<empty; target baseline>"
    echo "cargo_encoded_rustflags=<empty>"
    echo "cargo_profile=release"
    echo "cargo_command=cargo build --target $target_triple --release --locked -p rvp9-cli --bins"
    for binary in rvp9enc rvp9dec; do
        echo "${binary}_path=$binary$exe_suffix"
        echo "${binary}_sha256=$(sha256 "$stage_dir/$binary$exe_suffix")"
    done
} > "$stage_dir/BUILD-METADATA.txt"

tar -C "$output_dir" -czf "$archive_path" "$archive_base"
printf '%s  %s\n' "$(sha256 "$archive_path")" "$archive_base.tar.gz" > "$checksum_path"
echo "Created $archive_path"
echo "Artifact manifest: $stage_dir/BUILD-METADATA.txt"
cat "$checksum_path"
