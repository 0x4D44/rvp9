set -euo pipefail
exec </dev/null
root='/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-packagegraph474-3255285b'
{ date -u +%FT%TZ; tar --version | head -1; } > "$root/archive-inspection.identity.txt"
for archive in "$root/cargo-target/package/"*.crate; do
 name=$(basename "$archive" .crate)
 tar -tzf "$archive" > "$root/$name.inspected-filelist.txt"
 for member in Cargo.toml Cargo.toml.orig Cargo.lock .cargo_vcs_info.json; do
  tar -xOf "$archive" "$name/$member" > "$root/$name.inspected-$member.txt"
 done
done
find "$root/cargo-target/package/tmp-registry" -maxdepth 6 -type f -print > "$root/tmp-registry-filelist.txt"
find "$root/cargo-target/package/tmp-registry" -maxdepth 6 -type f ! -name '*.crate' -exec sha256sum '{}' ';' > "$root/tmp-registry-record-sha256.txt"
printf 'archive_inspection_exit=0\n'
