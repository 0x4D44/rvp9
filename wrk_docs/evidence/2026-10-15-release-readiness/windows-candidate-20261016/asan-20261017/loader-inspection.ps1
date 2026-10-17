$ErrorActionPreference='Continue'
$ProgressPreference='SilentlyContinue'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-asan474-a78b6878'
Get-Date -Format o
Get-Process -Id 37376 -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,StartTime | Format-List
$dump='C:/Program Files/Microsoft Visual Studio/18/Enterprise/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64/dumpbin.exe'
$runtime='C:/Program Files/Microsoft Visual Studio/18/Enterprise/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64/clang_rt.asan_dynamic-x86_64.dll'
"dumpbin_exists=$(Test-Path $dump)"
foreach($f in @("$r/preflight.rs","$r/preflight.exe",$runtime,'C:/Users/marti/.rustup/toolchains/nightly-x86_64-pc-windows-msvc/lib/rustlib/x86_64-pc-windows-msvc/lib/librustc-nightly_rt.asan.a')) {
 $i=Get-Item $f
 "$($i.FullName)|bytes=$($i.Length)|sha256=$((Get-FileHash $f -Algorithm SHA256).Hash.ToLower())"
}
& $dump /IMPORTS "$r/preflight.exe"
"imports_dump_exit=$LASTEXITCODE"
& $dump /EXPORTS $runtime
"runtime_exports_dump_exit=$LASTEXITCODE"
"candidate_soak_exists=$(Test-Path 'C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target/release/soak.exe')"
