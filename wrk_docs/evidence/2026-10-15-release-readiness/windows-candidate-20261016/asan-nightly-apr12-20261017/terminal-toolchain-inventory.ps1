$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-asanapr12-474-b98479dc'
"utc=$([DateTime]::UtcNow.ToString('o'))"
"root48564_present=$($null -ne (Get-Process -Id 48564 -ErrorAction SilentlyContinue))"
"rustc_count=$([Diagnostics.Process]::GetProcessesByName('rustc').Count)"
"preflight_count=$([Diagnostics.Process]::GetProcessesByName('preflight').Count)"
foreach($date in @('2026-04-12','2026-04-08')) {
 $t="C:/Users/marti/.rustup/toolchains/nightly-$date-x86_64-pc-windows-msvc"
 "toolchain_identity_begin=$date"
 & "$t/bin/rustc.exe" -vV
 $a="$t/lib/rustlib/x86_64-pc-windows-msvc/lib/librustc-nightly_rt.asan.a"
 "expected_asan_archive=$a"
 "expected_asan_archive_exists=$(Test-Path $a)"
 if(Test-Path $a) { $f=Get-Item $a; "archive_bytes=$($f.Length) sha256=$((Get-FileHash $a -Algorithm SHA256).Hash.ToLower())" }
}
$dll='C:/Program Files/Microsoft Visual Studio/18/Enterprise/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64/clang_rt.asan_dynamic-x86_64.dll'
"configured_runtime=$dll"
"configured_runtime_sha256=$((Get-FileHash $dll -Algorithm SHA256).Hash.ToLower())"
"run_exit_exists=$(Test-Path "$r/run.exit.txt")"
"preflight_exe_exists=$(Test-Path "$r/preflight.exe")"
foreach($name in @('preflight.rs','toolchain.log','compile.command.txt','compile.log','compile.exit.txt','runner.pid')) {
 $f=Get-Item "$r/$name"; $h=Get-FileHash $f.FullName -Algorithm SHA256
 "$name|bytes=$($f.Length)|sha256=$($h.Hash.ToLower())"
}
$task='D:/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi'
"head=$(git -C $task rev-parse HEAD)"
"status=$(git -C $task status --porcelain)"
