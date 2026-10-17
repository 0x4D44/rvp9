import pathlib,json,subprocess
p=pathlib.Path(__file__).parent;root=json.loads(p.joinpath('root.json').read_text());r=root['posix'];n=root['native']
source='''fn main() {
    let source: Vec<u8> = (0..4096).map(|i| (i & 255) as u8).collect();
    let mut data = std::hint::black_box(source);
    for (index, value) in data.iter_mut().enumerate() {
        *value = value.wrapping_add(index as u8);
    }
    let sum: u64 = data.iter().map(|&value| u64::from(value)).sum();
    assert_eq!(data.len(), 4096);
    assert_eq!(sum, 520192);
    println!("native_asan_preflight_ok arch={} pointer_bits={} avx2={} bytes={} checksum={}", std::env::consts::ARCH, usize::BITS, std::is_x86_feature_detected!("avx2"), data.len(), sum);
}
'''
p.joinpath('preflight.rs').write_text(source)
subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15',str(p/'preflight.rs'),'marti@crucible:'+n+'/preflight.rs'],check=True,stdin=subprocess.DEVNULL,timeout=30)
script=f'''set -uo pipefail
exec </dev/null
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
root={r}
export PATH="/c/Program Files/Microsoft Visual Studio/18/Enterprise/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64:/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS=4 CARGO_NET_OFFLINE=true
unset RVP9_FORCE_SCALAR
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
{{ date -u +%FT%TZ; git rev-parse HEAD; git status --porcelain; rustc +nightly-2026-04-12 -vV; cargo +nightly-2026-04-12 -vV; echo msvc_runtime_directory='C:/Program Files/Microsoft Visual Studio/18/Enterprise/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64'; sha256sum "$root/preflight.rs"; }} > "$root/toolchain.log" 2>&1
printf '%s\\n' 'rustc +nightly-2026-04-12 --target x86_64-pc-windows-msvc -Zsanitizer=address -Copt-level=1 -Cdebuginfo=1 -Ctarget-cpu=native -Clinker=rust-lld --crate-name native_asan_preflight preflight.rs -o preflight.exe' > "$root/compile.command.txt"
deltic timeout 120 rustc +nightly-2026-04-12 --target x86_64-pc-windows-msvc -Zsanitizer=address -Copt-level=1 -Cdebuginfo=1 -Ctarget-cpu=native -Clinker=rust-lld --crate-name native_asan_preflight "$root/preflight.rs" -o "$root/preflight.exe" > "$root/compile.log" 2>&1
rc=$?; echo "$rc" > "$root/compile.exit.txt"
echo "native_asan_compile_exit=$rc"
[ "$rc" = 0 ] || exit "$rc"
deltic timeout 30 "$root/preflight.exe" > "$root/run.log" 2>&1
rc=$?; echo "$rc" > "$root/run.exit.txt"
echo "native_asan_run_exit=$rc"
{{ sha256sum "$root/preflight.rs" "$root/preflight.exe"; git rev-parse HEAD; git status --porcelain; }} > "$root/artifacts.txt"
exit "$rc"
'''
p.joinpath('preflight.sh').write_text(script)
subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15',str(p/'preflight.sh'),'marti@crucible:'+n+'/preflight.sh'],check=True,stdin=subprocess.DEVNULL,timeout=30)
launch=f'''$ErrorActionPreference='Stop'
$r='{n}'
$p=Start-Process 'C:/Program Files/Git/bin/bash.exe' -ArgumentList @('--noprofile','--norc',"$r/preflight.sh") -PassThru -NoNewWindow
$p.Id | Set-Content "$r/runner.pid"
"native_asan_root_pid=$($p.Id)"
if(!$p.WaitForExit(180000)) {{ taskkill /PID $p.Id /T /F; throw 'native ASan preflight bound reached' }}
$p.Refresh(); "native_asan_launcher_completed=$($p.HasExited)"
'''
p.joinpath('launch.ps1').write_text(launch)
