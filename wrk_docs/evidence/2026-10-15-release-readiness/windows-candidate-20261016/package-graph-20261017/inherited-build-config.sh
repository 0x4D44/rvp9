set -euo pipefail
r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-packagegraph474-3255285b'
{ date -u +%FT%TZ; for name in RUSTFLAGS RUSTDOCFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER; do printf '%s=%s\n' "$name" "${!name-}"; done; for file in /c/Users/marti/.cargo/config /c/Users/marti/.cargo/config.toml; do if [ -f "$file" ]; then sha256sum "$file"; sed -n '/^\[build\]/p;/^\[target\.x86_64-pc-windows-msvc\]/p;/^[[:space:]]*linker[[:space:]]*=/p;/^[[:space:]]*rustflags[[:space:]]*=/p' "$file"; fi; done; } > "$r/inherited-build-config.log"
