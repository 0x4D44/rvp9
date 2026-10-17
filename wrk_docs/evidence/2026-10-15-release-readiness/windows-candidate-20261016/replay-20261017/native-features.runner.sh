set -uo pipefail
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
export PATH="/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN=1.95.0 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target
root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
printf '%s\n' 'bash -c '"'"'root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6
cat > "$root/native-features.rs" <<'"'"'"'"'"'"'"'"'RS'"'"'"'"'"'"'"'"'
fn main() { println!("avx2={} sse2={} ssse3={} sse41={} bmi2={} fma={}", std::is_x86_feature_detected!("avx2"), std::is_x86_feature_detected!("sse2"), std::is_x86_feature_detected!("ssse3"), std::is_x86_feature_detected!("sse4.1"), std::is_x86_feature_detected!("bmi2"), std::is_x86_feature_detected!("fma")); }
RS
rustc --crate-name replay_features "$root/native-features.rs" -o "$root/native-features.exe" && "$root/native-features.exe"'"'"'' > "$root/logs/native-features.command.txt"
printf 'START native-features bash_pid=%s utc=%s\n' "$$" "$(date -u +%FT%TZ)"
deltic timeout 60 bash -c 'root=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6
cat > "$root/native-features.rs" <<'"'"'RS'"'"'
fn main() { println!("avx2={} sse2={} ssse3={} sse41={} bmi2={} fma={}", std::is_x86_feature_detected!("avx2"), std::is_x86_feature_detected!("sse2"), std::is_x86_feature_detected!("ssse3"), std::is_x86_feature_detected!("sse4.1"), std::is_x86_feature_detected!("bmi2"), std::is_x86_feature_detected!("fma")); }
RS
rustc --crate-name replay_features "$root/native-features.rs" -o "$root/native-features.exe" && "$root/native-features.exe"' </dev/null > "$root/logs/native-features.log" 2>&1
s=$?
printf '%s\n' "$s" > "$root/logs/native-features.exit.txt"
printf 'END native-features exit=%s utc=%s\n' "$s" "$(date -u +%FT%TZ)"
exit "$s"
